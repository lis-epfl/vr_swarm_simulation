#!/usr/bin/env python3
"""
Batch Mesh Comparison for Swarm Experiments

Processes all swarm point clouds in Assets/FinalPointClouds/ and compares them
against the ground truth mesh, saving results to a CSV file.
"""

import os
import sys
import csv
import re
from pathlib import Path
import numpy as np

# Suppress print statements from mesh_comparison
import io
import contextlib

try:
    import open3d as o3d
except ImportError:
    print("Error: Open3D is required. Install with: pip install open3d")
    sys.exit(1)

try:
    import trimesh
    HAS_TRIMESH = True
except ImportError:
    HAS_TRIMESH = False


class SilentMeshComparator:
    """Silent version of MeshComparator - no print statements."""
    
    def __init__(self, ground_truth_path: str):
        self.gt_path = Path(ground_truth_path)
        self.normalization_scale = None
        self.gt_mesh = None
        self.gt_pcd = None
        self.gt_kdtree = None
        self._load_ground_truth()
        
    def _load_ground_truth(self):
        """Load ground truth mesh and convert to point cloud."""
        if self.gt_path.suffix.lower() == '.fbx' and HAS_TRIMESH:
            trimesh_mesh = trimesh.load(str(self.gt_path), force='mesh')
            vertices = np.asarray(trimesh_mesh.vertices)
            triangles = np.asarray(trimesh_mesh.faces)
            
            self.gt_mesh = o3d.geometry.TriangleMesh()
            self.gt_mesh.vertices = o3d.utility.Vector3dVector(vertices)
            self.gt_mesh.triangles = o3d.utility.Vector3iVector(triangles)
            
            if hasattr(trimesh_mesh.visual, 'vertex_colors'):
                colors = np.asarray(trimesh_mesh.visual.vertex_colors)[:, :3] / 255.0
                self.gt_mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
        else:
            self.gt_mesh = o3d.io.read_triangle_mesh(str(self.gt_path))
        
        if not self.gt_mesh.has_vertex_normals():
            self.gt_mesh.compute_vertex_normals()
        
        num_samples = max(100000, len(self.gt_mesh.vertices) * 10)
        self.gt_pcd = self.gt_mesh.sample_points_uniformly(number_of_points=num_samples)
        self.gt_kdtree = o3d.geometry.KDTreeFlann(self.gt_pcd)
    
    def compute_mesh_visibility(self, reconstructed_pcd: o3d.geometry.PointCloud, 
                                threshold: float = 0.1) -> dict:
        """Compute mesh visibility (coverage) metric."""
        adjusted_threshold = threshold
        if self.normalization_scale is not None:
            adjusted_threshold = threshold / self.normalization_scale
        
        recon_kdtree = o3d.geometry.KDTreeFlann(reconstructed_pcd)
        gt_points = np.asarray(self.gt_pcd.points)
        covered_points = 0
        distances = []
        
        for gt_point in gt_points:
            [k, idx, dist_sq] = recon_kdtree.search_knn_vector_3d(gt_point, 1)
            distance = np.sqrt(dist_sq[0])
            distances.append(distance)
            
            if distance < adjusted_threshold:
                covered_points += 1
        
        coverage_ratio = covered_points / len(gt_points)
        
        return {
            'coverage_percentage': coverage_ratio * 100,
            'covered_points': covered_points,
            'total_points': len(gt_points),
            'threshold': threshold,
            'adjusted_threshold': adjusted_threshold
        }
    
    def compute_mesh_overlap(self, reconstructed_pcd: o3d.geometry.PointCloud, threshold: float = 0.1) -> dict:
        """Compute mesh overlap (accuracy) metric."""
        recon_points = np.asarray(reconstructed_pcd.points)
        distances_to_gt = []
        
        for recon_point in recon_points:
            [k, idx, dist_sq] = self.gt_kdtree.search_knn_vector_3d(recon_point, 1)
            distance = np.sqrt(dist_sq[0])
            distances_to_gt.append(distance)
        
        distances_to_gt = np.array(distances_to_gt)
        
        # Compute distances from GT → recon for Chamfer
        recon_kdtree = o3d.geometry.KDTreeFlann(reconstructed_pcd)
        gt_points = np.asarray(self.gt_pcd.points)
        distances_from_gt = []
        
        for gt_point in gt_points:
            [k, idx, dist_sq] = recon_kdtree.search_knn_vector_3d(gt_point, 1)
            distance = np.sqrt(dist_sq[0])
            distances_from_gt.append(distance)
        
        distances_from_gt = np.array(distances_from_gt)
        
        chamfer_distance = (np.mean(distances_to_gt) + np.mean(distances_from_gt)) / 2
        
        # Accuracy percentage
        adjusted_threshold = threshold
        if self.normalization_scale is not None:
            adjusted_threshold = threshold / self.normalization_scale
        
        accurate_points = np.sum(distances_to_gt < adjusted_threshold)
        accuracy_percentage = (accurate_points / len(distances_to_gt)) * 100
        
        return {
            'mean_distance_to_gt': np.mean(distances_to_gt),
            'chamfer_distance': chamfer_distance,
            'accuracy_percentage': accuracy_percentage,
            'threshold': threshold,
            'adjusted_threshold': adjusted_threshold,
        }
    
    def compare(self, reconstructed_path: str, coverage_threshold: float = 0.1) -> dict:
        """Run comparison pipeline."""
        recon_pcd = o3d.io.read_point_cloud(reconstructed_path)
        original_size = len(recon_pcd.points)
        
        # Voxelize FIRST to remove duplicate/overlapping points (before alignment)
        voxel_size = 0.05  # 5cm voxels (same as swarm_pointcloud_builder.py)
        recon_pcd = recon_pcd.voxel_down_sample(voxel_size)
        voxelized_size = len(recon_pcd.points)
        
        # Apply X-flip to GT mesh (Unity coordinate system)
        flip_transform = np.eye(4)
        flip_transform[0, 0] = -1
        self.gt_mesh.transform(flip_transform)
        
        # Scale GT to half size (GT is 2x too big)
        self.gt_mesh.scale(0.5, center=[0, 0, 0])
        
        # Normalize GT mesh
        gt_center = self.gt_mesh.get_center()
        # self.gt_mesh.translate(-gt_center)  # Commented out - not needed
        gt_max_bound = np.max(self.gt_mesh.get_max_bound() - self.gt_mesh.get_min_bound())
        normalization_scale_factor = 1.0 / gt_max_bound
        self.gt_mesh.scale(normalization_scale_factor, center=[0, 0, 0])
        
        # Rebuild GT point cloud and KDTree after all transformations
        num_samples = max(100000, len(self.gt_mesh.vertices) * 10)
        self.gt_pcd = self.gt_mesh.sample_points_uniformly(number_of_points=num_samples)
        self.gt_kdtree = o3d.geometry.KDTreeFlann(self.gt_pcd)
        
        # Normalize reconstruction (use same scale factor as GT)
        recon_center = recon_pcd.get_center()
        # recon_pcd.translate(-recon_center)  # Commented out - not needed
        recon_pcd.scale(normalization_scale_factor, center=[0, 0, 0])
        
        self.normalization_scale = gt_max_bound
        
        # Compute metrics on voxelized point clouds
        visibility = self.compute_mesh_visibility(recon_pcd, coverage_threshold)
        overlap = self.compute_mesh_overlap(recon_pcd, coverage_threshold)
        
        return {
            'visibility': visibility,
            'overlap': overlap,
            'original_size': original_size,
            'voxelized_size': voxelized_size
        }


def parse_swarm_filename(filename: str) -> tuple:
    """
    Parse swarm filename to extract drone count, interval, and convergence time.
    Expected formats: 
      - swarm_raw_3_drones_3_interval.ply (without time)
      - swarm_raw_4_drones_3_interval_80.54s.ply (with convergence time)
    
    Returns:
        (num_drones, interval, convergence_time) or (None, None, None) if parsing fails
        convergence_time will be None if not present in filename
    """
    # Pattern with time: swarm_raw_{num}_drones_{num}_interval_{time}s.ply
    match = re.match(r'swarm_raw_(\d+)_drones_(\d+)_interval_([\d.]+)s\.ply', filename)
    if match:
        return int(match.group(1)), int(match.group(2)), float(match.group(3))
    
    # Pattern without time: swarm_raw_{num}_drones_{num}_interval.ply
    match = re.match(r'swarm_raw_(\d+)_drones_(\d+)_interval\.ply', filename)
    if match:
        return int(match.group(1)), int(match.group(2)), None
    
    return None, None, None


def main():
    # Paths
    final_pc_folder = Path("Assets/FinalPointClouds")
    gt_path = "Assets/Comparisons/House_with_Texture3.obj"  # House1 ground truth
    output_csv = "comparison_swarm_house1.csv"
    
    if not final_pc_folder.exists():
        print(f"Error: Folder not found: {final_pc_folder}")
        return 1
    
    if not Path(gt_path).exists():
        print(f"Error: Ground truth not found: {gt_path}")
        return 1
    
    # Find all swarm PLY files
    ply_files = sorted([f for f in final_pc_folder.glob("swarm_raw_*.ply")])
    
    if not ply_files:
        print(f"No swarm PLY files found in {final_pc_folder}")
        return 1
    
    print(f"Found {len(ply_files)} swarm point clouds to process")
    print(f"Ground truth: {gt_path}")
    print(f"Output: {output_csv}")
    print(f"\nProcessing and writing results incrementally...\n")
    
    # Initialize CSV file with headers
    fieldnames = [
        'filename',
        'drones', 
        'interval',
        'convergence_time',
        'num_points',
        'mesh_visibility', 
        'mesh_overlap', 
        'chamfer_distance', 
        'mean_distance_to_gt', 
        'threshold',
        'adjusted_threshold'
    ]
    
    with open(output_csv, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
    
    # Prepare results list for summary
    results = []

    # Time tracking
    import time
    start_time = time.time()
    
    for i, ply_file in enumerate(ply_files, 1):
        filename = ply_file.name
        num_drones, interval, convergence_time = parse_swarm_filename(filename)
        
        if num_drones is None:
            print(f"  [{i}/{len(ply_files)}] Skipping {filename} (cannot parse)")
            continue
        
        print(f"  [{i}/{len(ply_files)}] Processing {filename}... ", end='', flush=True)
        
        try:
            # Load point cloud to check size
            test_pcd = o3d.io.read_point_cloud(str(ply_file))
            num_points = len(test_pcd.points)
            
            # Create fresh comparator for each file
            comparator = SilentMeshComparator(gt_path)
            
            # Run comparison with threshold 0.1m
            comparison_results = comparator.compare(str(ply_file), coverage_threshold=0.1)
            
            # Extract metrics
            voxelized_pts = comparison_results['voxelized_size']
            row = {
                'filename': filename,
                'drones': num_drones,
                'interval': interval,
                'convergence_time': convergence_time if convergence_time is not None else '',
                'num_points': num_points,
                'mesh_visibility': comparison_results['visibility']['coverage_percentage'],
                'mesh_overlap': comparison_results['overlap']['accuracy_percentage'],
                'chamfer_distance': comparison_results['overlap']['chamfer_distance'],
                'mean_distance_to_gt': comparison_results['overlap']['mean_distance_to_gt'],
                'threshold': comparison_results['overlap']['threshold'],
                'adjusted_threshold': comparison_results['overlap']['adjusted_threshold']
            }
            
            # Write to CSV immediately
            with open(output_csv, 'a', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writerow(row)
            
            results.append(row)
            time_str = f" time={convergence_time:.2f}s" if convergence_time is not None else ""
            print(f"[OK] pts={num_points:,}->{voxelized_pts:,} vis={row['mesh_visibility']:.1f}% overlap={row['mesh_overlap']:.1f}%{time_str}")
            
        except Exception as e:
            print(f"[ERROR] {e}")
            continue
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"\nTotal processing time: {elapsed_time:.2f} seconds for {len(ply_files)} point clouds, {elapsed_time/len(ply_files):.2f} seconds per point cloud on average.\n")

    if not results:
        print("\nNo results to save")
        return 1
    
    # Print summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"Total point clouds processed: {len(results)}")
    print(f"\nMesh Visibility (Coverage) Range:")
    vis_values = [r['mesh_visibility'] for r in results]
    print(f"  Min: {min(vis_values):.2f}%")
    print(f"  Max: {max(vis_values):.2f}%")
    print(f"  Avg: {np.mean(vis_values):.2f}%")
    print(f"\nMesh Overlap (Accuracy) Range:")
    overlap_values = [r['mesh_overlap'] for r in results]
    print(f"  Min: {min(overlap_values):.2f}%")
    print(f"  Max: {max(overlap_values):.2f}%")
    print(f"  Avg: {np.mean(overlap_values):.2f}%")
    
    # Summary by interval
    print(f"\nAverage Metrics by Interval:")
    intervals = sorted(set(r['interval'] for r in results))
    for interval in intervals:
        interval_results = [r for r in results if r['interval'] == interval]
        avg_vis = np.mean([r['mesh_visibility'] for r in interval_results])
        avg_overlap = np.mean([r['mesh_overlap'] for r in interval_results])
        avg_time = np.mean([r['convergence_time'] for r in interval_results if r['convergence_time'] != ''])
        print(f"  Interval {interval}s: vis={avg_vis:.1f}% overlap={avg_overlap:.1f}% avg_time={avg_time:.2f}s ({len(interval_results)} files)")
    
    print(f"{'='*70}\n")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
