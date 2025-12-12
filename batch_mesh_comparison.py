#!/usr/bin/env python3
"""
Batch Mesh Comparison for NBV Experiments

Processes all point clouds in Assets/FinalPointClouds/ and compares them
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
    
    def compute_mesh_overlap(self, reconstructed_pcd: o3d.geometry.PointCloud) -> dict:
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
        accuracy_threshold = 0.1
        if self.normalization_scale is not None:
            accuracy_threshold = 0.1 / self.normalization_scale
        
        accurate_points = np.sum(distances_to_gt < accuracy_threshold)
        accuracy_percentage = (accurate_points / len(distances_to_gt)) * 100
        
        return {
            'mean_distance_to_gt': np.mean(distances_to_gt),
            'chamfer_distance': chamfer_distance,
            'accuracy_percentage': accuracy_percentage,
            'accuracy_threshold': accuracy_threshold,
        }
    
    def compare(self, reconstructed_path: str, coverage_threshold: float = 0.1) -> dict:
        """Run comparison pipeline."""
        recon_pcd = o3d.io.read_point_cloud(reconstructed_path)
        
        # Apply X-flip to GT mesh (Unity coordinate system)
        flip_transform = np.eye(4)
        flip_transform[0, 0] = -1
        self.gt_mesh.transform(flip_transform)
        
        # Normalize GT mesh
        gt_center = self.gt_mesh.get_center()
        self.gt_mesh.translate(-gt_center)
        gt_max_bound = np.max(self.gt_mesh.get_max_bound() - self.gt_mesh.get_min_bound())
        self.gt_mesh.scale(1.0 / gt_max_bound, center=[0, 0, 0])
        
        # Rebuild GT point cloud and KDTree
        num_samples = max(100000, len(self.gt_mesh.vertices) * 10)
        self.gt_pcd = self.gt_mesh.sample_points_uniformly(number_of_points=num_samples)
        self.gt_kdtree = o3d.geometry.KDTreeFlann(self.gt_pcd)
        
        # Normalize reconstruction
        recon_center = recon_pcd.get_center()
        recon_pcd.translate(-recon_center)
        recon_max_bound = np.max(recon_pcd.get_max_bound() - recon_pcd.get_min_bound())
        recon_pcd.scale(1.0 / recon_max_bound, center=[0, 0, 0])
        
        self.normalization_scale = min(gt_max_bound, recon_max_bound)
        
        # Multi-scale ICP alignment
        gt_down = self.gt_pcd.voxel_down_sample(voxel_size=0.05)
        recon_down = recon_pcd.voxel_down_sample(voxel_size=0.05)
        
        gt_down.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
        recon_down.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
        
        # Coarse alignment
        coarse_result = o3d.pipelines.registration.registration_icp(
            recon_down, gt_down, 0.2, np.eye(4),
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=100)
        )
        recon_pcd.transform(coarse_result.transformation)
        
        # Fine alignment
        fine_result = o3d.pipelines.registration.registration_icp(
            recon_pcd, self.gt_pcd, 0.05, np.eye(4),
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=200)
        )
        recon_pcd.transform(fine_result.transformation)
        
        # Compute metrics
        visibility = self.compute_mesh_visibility(recon_pcd, coverage_threshold)
        overlap = self.compute_mesh_overlap(recon_pcd)
        
        return {
            'visibility': visibility,
            'overlap': overlap
        }


def parse_filename(filename: str) -> tuple:
    """
    Parse filename to extract drone count and iteration count.
    Expected format: raw_3_drones_1_iterations.ply
    
    Returns:
        (num_drones, num_iterations) or (None, None) if parsing fails
    """
    # Pattern: raw_{num}_drones_{num}_iterations.ply
    match = re.match(r'raw_(\d+)_drones_(\d+)_iterations\.ply', filename)
    if match:
        return int(match.group(1)), int(match.group(2))
    return None, None


def main():
    # Paths
    final_pc_folder = Path("Assets/FinalPointClouds")
    gt_path = "Assets/Comparisons/House_with_Texture3.obj"
    output_csv = "mesh_comparison_results.csv"
    
    if not final_pc_folder.exists():
        print(f"Error: Folder not found: {final_pc_folder}")
        return 1
    
    if not Path(gt_path).exists():
        print(f"Error: Ground truth not found: {gt_path}")
        return 1
    
    # Find all raw PLY files
    ply_files = sorted([f for f in final_pc_folder.glob("raw_*.ply")])
    
    if not ply_files:
        print(f"No PLY files found in {final_pc_folder}")
        return 1
    
    print(f"Found {len(ply_files)} point clouds to process")
    print(f"Ground truth: {gt_path}")
    print(f"Output: {output_csv}")
    print(f"\nProcessing...")
    
    # Prepare CSV
    results = []
    
    for i, ply_file in enumerate(ply_files, 1):
        filename = ply_file.name
        num_drones, num_iterations = parse_filename(filename)
        
        if num_drones is None:
            print(f"  [{i}/{len(ply_files)}] Skipping {filename} (cannot parse)")
            continue
        
        print(f"  [{i}/{len(ply_files)}] Processing {filename}... ", end='', flush=True)
        
        try:
            # Create fresh comparator for each file
            comparator = SilentMeshComparator(gt_path)
            
            # Run comparison with threshold 0.1m
            comparison_results = comparator.compare(str(ply_file), coverage_threshold=0.1)
            
            # Extract metrics
            row = {
                'filename': filename,
                'drones': num_drones,
                'iterations': num_iterations,
                'mesh_visibility': comparison_results['visibility']['coverage_percentage'],
                'mesh_overlap': comparison_results['overlap']['accuracy_percentage'],
                'chamfer_distance': comparison_results['overlap']['chamfer_distance'],
                'mean_distance_to_gt': comparison_results['overlap']['mean_distance_to_gt'],
                'accuracy_threshold': comparison_results['overlap']['accuracy_threshold']
            }
            
            results.append(row)
            print(f"✓")
            
        except Exception as e:
            print(f"✗ Error: {e}")
            continue
    
    if not results:
        print("\nNo results to save")
        return 1
    
    # Write CSV
    print(f"\nWriting results to {output_csv}...")
    
    with open(output_csv, 'w', newline='') as csvfile:
        fieldnames = [
            'filename', 
            'drones', 
            'iterations', 
            'mesh_visibility', 
            'mesh_accuracy', 
            'chamfer_distance', 
            'mean_distance_to_gt', 
            'accuracy_threshold'
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for row in results:
            writer.writerow(row)
    
    print(f"✓ Saved {len(results)} results to {output_csv}")
    
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
    print(f"\nMesh Accuracy (Overlap) Range:")
    acc_values = [r['mesh_accuracy'] for r in results]
    print(f"  Min: {min(acc_values):.2f}%")
    print(f"  Max: {max(acc_values):.2f}%")
    print(f"  Avg: {np.mean(acc_values):.2f}%")
    print(f"{'='*70}\n")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
