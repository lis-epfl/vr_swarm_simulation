#!/usr/bin/env python3
"""
mesh_comparison.py - Mesh Quality Evaluation

Compares reconstructed point clouds against ground truth mesh.
Computes:
1. Mesh Visibility (Coverage): How much of GT mesh is covered
2. Mesh Overlap (Accuracy): How close reconstruction is to GT

Usage:
    python mesh_comparison.py <reconstructed.ply> [--ground-truth <mesh.fbx>] [--visualize]
"""
import copy

"""USAGE:
# Basic usage
python mesh_comparison.py pointcloud_raw_frame0001_20251211_120000.ply

# With visualization
python mesh_comparison.py pointcloud_raw_frame0001_20251211_120000.ply --visualize

# Custom ground truth and threshold
python mesh_comparison.py my_reconstruction.ply --ground-truth ../../Houses/MyHouse.fbx --threshold 0.15

# Save results to JSON
python mesh_comparison.py my_reconstruction.ply --output results.json"""

import os
import sys
import argparse
import numpy as np
from pathlib import Path

try:
    import open3d as o3d
    HAS_OPEN3D = True
except ImportError:
    print("Error: Open3D is required. Install with: pip install open3d")
    sys.exit(1)

# Try to import trimesh for FBX loading
try:
    import trimesh
    HAS_TRIMESH = True
except ImportError:
    print("Warning: trimesh not found. FBX loading may be limited.")
    print("Install with: pip install trimesh")
    HAS_TRIMESH = False


class MeshComparator:
    """Compares reconstructed point cloud with ground truth mesh."""
    
    def __init__(self, ground_truth_path: str, visualization: bool = False, normalize: bool = True, align: bool = True):
        """
        Initialize comparator with ground truth mesh.
        
        Args:
            ground_truth_path: Path to ground truth mesh (FBX, OBJ, PLY, etc.)
            visualization: Whether to show visualizations
            normalize: Whether to normalize both meshes to same scale
            align: Whether to align reconstruction to GT using ICP
        """
        self.gt_path = Path(ground_truth_path)
        self.visualization = visualization
        self.normalize = normalize
        self.align = align
        self.normalization_scale = None  # Store scale factor for threshold adjustment
        self.gt_mesh = None
        self.gt_pcd = None
        self.gt_kdtree = None
        
        # Load ground truth
        print(f"\n{'='*60}")
        print("Loading Ground Truth Mesh")
        print(f"{'='*60}")
        self._load_ground_truth()
        
    def _load_ground_truth(self):
        """Load ground truth mesh and convert to point cloud."""
        if not self.gt_path.exists():
            raise FileNotFoundError(f"Ground truth not found: {self.gt_path}")
        
        print(f"Loading: {self.gt_path}")
        
        # Try loading with Open3D first
        try:
            if self.gt_path.suffix.lower() == '.fbx' and HAS_TRIMESH:
                # Use trimesh for FBX, then convert to Open3D
                print("  Using trimesh for FBX loading...")
                trimesh_mesh = trimesh.load(str(self.gt_path), force='mesh')
                
                # Convert to Open3D mesh
                vertices = np.asarray(trimesh_mesh.vertices)
                triangles = np.asarray(trimesh_mesh.faces)
                
                self.gt_mesh = o3d.geometry.TriangleMesh()
                self.gt_mesh.vertices = o3d.utility.Vector3dVector(vertices)
                self.gt_mesh.triangles = o3d.utility.Vector3iVector(triangles)
                
                # Try to get vertex colors if available
                if hasattr(trimesh_mesh.visual, 'vertex_colors'):
                    colors = np.asarray(trimesh_mesh.visual.vertex_colors)[:, :3] / 255.0
                    self.gt_mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
            else:
                # Use Open3D directly
                self.gt_mesh = o3d.io.read_triangle_mesh(str(self.gt_path))
            
            if not self.gt_mesh.has_vertices():
                raise ValueError("Loaded mesh has no vertices")
            
            print(f"  ✓ Loaded mesh: {len(self.gt_mesh.vertices)} vertices, {len(self.gt_mesh.triangles)} triangles")
            
            # Compute normals if not present
            if not self.gt_mesh.has_vertex_normals():
                self.gt_mesh.compute_vertex_normals()
            
            # Sample points from mesh surface for comparison
            # Use high density sampling for accurate coverage measurement
            num_samples = max(100000, len(self.gt_mesh.vertices) * 10)
            print(f"  Sampling {num_samples} points from mesh surface...")
            self.gt_pcd = self.gt_mesh.sample_points_uniformly(number_of_points=num_samples)
            print(f"  ✓ Sampled {len(self.gt_pcd.points)} points")
            
            # Build KD-tree for fast nearest neighbor search
            self.gt_kdtree = o3d.geometry.KDTreeFlann(self.gt_pcd)
            
        except Exception as e:
            print(f"Error loading ground truth: {e}")
            raise
    
    def compute_mesh_visibility(self, reconstructed_pcd: o3d.geometry.PointCloud, 
                                threshold: float = 0.1) -> dict:
        """
        Compute mesh visibility (coverage) metric.
        
        Measures what percentage of the ground truth mesh surface is covered
        by the reconstruction.
        
        Args:
            reconstructed_pcd: Reconstructed point cloud
            threshold: Distance threshold (meters) to consider a point "covered"
        
        Returns:
            Dictionary with visibility metrics
        """
        print(f"\n{'='*60}")
        print("Computing Mesh Visibility (Coverage)")
        print(f"{'='*60}")
        
        # Adjust threshold if normalized
        adjusted_threshold = threshold
        if self.normalization_scale is not None:
            adjusted_threshold = threshold / self.normalization_scale
            print(f"Original threshold: {threshold}m")
            print(f"Adjusted threshold (normalized): {adjusted_threshold:.6f}")
        else:
            print(f"Distance threshold: {threshold}m")
        
        # Build KD-tree for reconstructed points
        recon_kdtree = o3d.geometry.KDTreeFlann(reconstructed_pcd)
        
        gt_points = np.asarray(self.gt_pcd.points)
        covered_points = 0
        distances = []
        
        print("Checking coverage for each ground truth point...")
        for i, gt_point in enumerate(gt_points):
            if i % 1000000 == 0:
                print(f"  Progress: {i}/{len(gt_points)} ({100*i/len(gt_points):.1f}%)")
            
            # Find nearest reconstructed point
            [k, idx, dist_sq] = recon_kdtree.search_knn_vector_3d(gt_point, 1)
            distance = np.sqrt(dist_sq[0])
            distances.append(distance)
            
            if distance < adjusted_threshold:
                covered_points += 1
        
        coverage_ratio = covered_points / len(gt_points)
        avg_distance = np.mean(distances)
        median_distance = np.median(distances)
        
        results = {
            'coverage_ratio': coverage_ratio,
            'coverage_percentage': coverage_ratio * 100,
            'covered_points': covered_points,
            'total_points': len(gt_points),
            'avg_distance_to_recon': avg_distance,
            'median_distance_to_recon': median_distance,
            'threshold': threshold,
            'adjusted_threshold': adjusted_threshold
        }
        
        print(f"\n{'='*60}")
        print("Visibility Results:")
        print(f"{'='*60}")
        print(f"  Coverage:           {results['coverage_percentage']:.2f}%")
        print(f"  Covered points:     {results['covered_points']:,} / {results['total_points']:,}")
        print(f"  Avg distance to GT: {results['avg_distance_to_recon']:.4f}m")
        print(f"  Median distance:    {results['median_distance_to_recon']:.4f}m")
        
        return results
    
    def compute_mesh_overlap(self, reconstructed_pcd: o3d.geometry.PointCloud) -> dict:
        """
        Compute mesh overlap (accuracy) metric.
        
        Measures how accurately the reconstruction matches the ground truth
        by computing distances from reconstructed points to GT surface.
        
        Args:
            reconstructed_pcd: Reconstructed point cloud
        
        Returns:
            Dictionary with overlap/accuracy metrics
        """
        print(f"\n{'='*60}")
        print("Computing Mesh Overlap (Accuracy)")
        print(f"{'='*60}")
        
        recon_points = np.asarray(reconstructed_pcd.points)
        distances_to_gt = []
        
        print("Computing distances from reconstruction to ground truth...")
        for i, recon_point in enumerate(recon_points):
            if i % 100000 == 0:
                print(f"  Progress: {i}/{len(recon_points)} ({100*i/len(recon_points):.1f}%)")
            
            # Find nearest GT point
            [k, idx, dist_sq] = self.gt_kdtree.search_knn_vector_3d(recon_point, 1)
            distance = np.sqrt(dist_sq[0])
            distances_to_gt.append(distance)
        
        distances_to_gt = np.array(distances_to_gt)
        
        # Compute Chamfer distance (bidirectional)
        # Already have distances from recon → GT
        # Now compute distances from GT → recon
        recon_kdtree = o3d.geometry.KDTreeFlann(reconstructed_pcd)
        gt_points = np.asarray(self.gt_pcd.points)
        distances_from_gt = []
        
        print("Computing distances from ground truth to reconstruction...")
        for i, gt_point in enumerate(gt_points):
            if i % 10000 == 0:
                print(f"  Progress: {i}/{len(gt_points)} ({100*i/len(gt_points):.1f}%)")
            
            [k, idx, dist_sq] = recon_kdtree.search_knn_vector_3d(gt_point, 1)
            distance = np.sqrt(dist_sq[0])
            distances_from_gt.append(distance)
        
        distances_from_gt = np.array(distances_from_gt)
        
        # Get reconstruction size for percentage calculations
        recon_bbox = reconstructed_pcd.get_axis_aligned_bounding_box()
        recon_extent = recon_bbox.get_extent()
        recon_diagonal = np.linalg.norm(recon_extent)
        
        # Compute metrics
        chamfer_distance = (np.mean(distances_to_gt) + np.mean(distances_from_gt)) / 2
        hausdorff_distance = max(np.max(distances_to_gt), np.max(distances_from_gt))
        
        # Calculate accuracy as percentage of points within threshold
        # Use the same adjusted threshold as coverage
        accuracy_threshold = 0.1  # Default
        if self.normalization_scale is not None:
            accuracy_threshold = 0.1 / self.normalization_scale
        
        accurate_points = np.sum(distances_to_gt < accuracy_threshold)
        accuracy_percentage = (accurate_points / len(distances_to_gt)) * 100
        
        results = {
            'mean_distance_to_gt': np.mean(distances_to_gt),
            'median_distance_to_gt': np.median(distances_to_gt),
            'std_distance_to_gt': np.std(distances_to_gt),
            'max_distance_to_gt': np.max(distances_to_gt),
            'mean_distance_from_gt': np.mean(distances_from_gt),
            'chamfer_distance': chamfer_distance,
            'hausdorff_distance': hausdorff_distance,
            'percentile_90': np.percentile(distances_to_gt, 90),
            'percentile_95': np.percentile(distances_to_gt, 95),
            'recon_diagonal': recon_diagonal,
            # Accuracy as percentage
            'accuracy_percentage': accuracy_percentage,
            'accurate_points': accurate_points,
            'accuracy_threshold': accuracy_threshold,
            # Percentages relative to reconstruction size (for reference)
            'mean_distance_pct': (np.mean(distances_to_gt) / recon_diagonal) * 100,
            'chamfer_distance_pct': (chamfer_distance / recon_diagonal) * 100,
            'hausdorff_distance_pct': (hausdorff_distance / recon_diagonal) * 100,
        }
        
        print(f"\n{'='*60}")
        print("Overlap/Accuracy Results:")
        print(f"{'='*60}")
        print(f"  Accuracy (points within threshold): {results['accuracy_percentage']:.2f}%")
        print(f"    ({results['accurate_points']:,} / {len(distances_to_gt):,} points within {accuracy_threshold:.6f} of GT)")
        print(f"\n  Distance Metrics:")
        print(f"    Reconstruction diagonal: {recon_diagonal:.4f}m")
        print(f"    Mean distance (recon→GT):   {results['mean_distance_to_gt']:.4f}m ({results['mean_distance_pct']:.2f}% of size)")
        print(f"    Median distance (recon→GT): {results['median_distance_to_gt']:.4f}m")
        print(f"    Std deviation:              {results['std_distance_to_gt']:.4f}m")
        print(f"    Chamfer distance:           {results['chamfer_distance']:.4f}m ({results['chamfer_distance_pct']:.2f}% of size)")
        print(f"    Hausdorff distance:         {results['hausdorff_distance']:.4f}m ({results['hausdorff_distance_pct']:.2f}% of size)")
        print(f"    90th percentile:            {results['percentile_90']:.4f}m")
        print(f"    95th percentile:            {results['percentile_95']:.4f}m")
        
        return results
    
    def visualize_comparison(self, reconstructed_pcd: o3d.geometry.PointCloud,
                            visibility_results: dict = None):
        """
        Visualize ground truth mesh with unseen areas highlighted.
        
        Args:
            reconstructed_pcd: Reconstructed point cloud
            visibility_results: Visibility results for identifying unseen points
        """
        if not self.visualization:
            return
        
        print(f"\n{'='*60}")
        print("Visualization")
        print(f"{'='*60}")
        
        # Color ground truth mesh (gray)
        self.gt_mesh.paint_uniform_color([0.7, 0.7, 0.7])
        
        # Find unseen GT points (not covered by reconstruction)
        if visibility_results:
            threshold = visibility_results['adjusted_threshold']
            recon_kdtree = o3d.geometry.KDTreeFlann(reconstructed_pcd)
            gt_points = np.asarray(self.gt_pcd.points)
            
            unseen_points = []
            print(f"  Identifying unseen areas (threshold: {threshold:.6f})...")
            
            for i, gt_point in enumerate(gt_points):
                if i % 100000 == 0 and i > 0:
                    print(f"    Progress: {i}/{len(gt_points)} ({100*i/len(gt_points):.1f}%)")
                
                [k, idx, dist_sq] = recon_kdtree.search_knn_vector_3d(gt_point, 1)
                distance = np.sqrt(dist_sq[0])
                
                if distance >= threshold:  # Not covered
                    unseen_points.append(gt_point)
            
            # Create point cloud for unseen areas
            unseen_pcd = o3d.geometry.PointCloud()
            unseen_pcd.points = o3d.utility.Vector3dVector(np.array(unseen_points))
            unseen_pcd.paint_uniform_color([0.8, 0.2, 0.8])  # Purple
            
            print(f"  Unseen points: {len(unseen_points):,} / {len(gt_points):,} ({100*len(unseen_points)/len(gt_points):.1f}%)")
            print("\nShowing ground truth (gray) and unseen areas (purple)...")
            print("  Close the window to continue")
            
            o3d.visualization.draw_geometries(
                [self.gt_mesh, unseen_pcd],
                window_name="Ground Truth (gray) - Unseen Areas (purple)",
                width=1280,
                height=720
            )
        else:
            print("  No visibility results available, showing GT only")
            o3d.visualization.draw_geometries(
                [self.gt_mesh],
                window_name="Ground Truth",
                width=1280,
                height=720
            )
    
    def compare(self, reconstructed_path: str, coverage_threshold: float = 0.1) -> dict:
        """
        Run full comparison pipeline.
        
        Args:
            reconstructed_path: Path to reconstructed point cloud
            coverage_threshold: Distance threshold for coverage metric
        
        Returns:
            Dictionary with all metrics
        """
        print(f"\n{'='*60}")
        print("Loading Reconstructed Point Cloud")
        print(f"{'='*60}")
        print(f"Loading: {reconstructed_path}")
        
        recon_pcd = o3d.io.read_point_cloud(reconstructed_path)
        print(f"  ✓ Loaded {len(recon_pcd.points)} points")
        
        # Normalize if requested
        if self.normalize:
            print(f"\n{'='*60}")
            print("Normalizing meshes to same scale...")
            print(f"{'='*60}")
            
            # Get bounding boxes before normalization
            gt_bbox_orig = self.gt_mesh.get_axis_aligned_bounding_box()
            recon_bbox_orig = recon_pcd.get_axis_aligned_bounding_box()
            gt_extent_orig = gt_bbox_orig.get_extent()
            recon_extent_orig = recon_bbox_orig.get_extent()
            
            print(f"Original GT size: [{gt_extent_orig[0]:.2f}, {gt_extent_orig[1]:.2f}, {gt_extent_orig[2]:.2f}]")
            print(f"Original recon size: [{recon_extent_orig[0]:.2f}, {recon_extent_orig[1]:.2f}, {recon_extent_orig[2]:.2f}]")
            
            # Apply X-flip to GT mesh first (Unity coordinate system)
            flip_transform = np.eye(4)
            flip_transform[0, 0] = -1
            self.gt_mesh.transform(flip_transform)
            print(f"  Applied X-flip to GT mesh")
            
            # Normalize GT mesh: center + scale to unit sphere
            gt_center = self.gt_mesh.get_center()
            self.gt_mesh.translate(-gt_center)
            gt_max_bound = np.max(self.gt_mesh.get_max_bound() - self.gt_mesh.get_min_bound())
            self.gt_mesh.scale(1.0 / gt_max_bound, center=[0, 0, 0])
            
            # Rebuild GT point cloud and KDTree after normalization
            num_samples = max(100000, len(self.gt_mesh.vertices) * 10)
            self.gt_pcd = self.gt_mesh.sample_points_uniformly(number_of_points=num_samples)
            self.gt_kdtree = o3d.geometry.KDTreeFlann(self.gt_pcd)
            
            # Normalize reconstruction: center + scale to unit sphere
            recon_center = recon_pcd.get_center()
            recon_pcd.translate(-recon_center)
            recon_max_bound = np.max(recon_pcd.get_max_bound() - recon_pcd.get_min_bound())
            recon_pcd.scale(1.0 / recon_max_bound, center=[0, 0, 0])
            
            # Store the larger of the two original scales for threshold adjustment
            self.normalization_scale = min(gt_max_bound, recon_max_bound)
            print(f"  ✓ Both meshes normalized to unit scale (original scale: {self.normalization_scale:.2f}m)")
        
        # Advanced alignment after normalization
        if self.align:
            print(f"\n{'='*60}")
            print("Multi-scale ICP alignment...")
            print(f"{'='*60}")
            
            # Downsample for coarse alignment
            gt_down = self.gt_pcd.voxel_down_sample(voxel_size=0.05)
            recon_down = recon_pcd.voxel_down_sample(voxel_size=0.05)
            
            # Estimate normals
            gt_down.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
            recon_down.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
            
            # Coarse alignment with larger threshold
            print(f"  Coarse alignment...")
            coarse_result = o3d.pipelines.registration.registration_icp(
                recon_down, gt_down, 0.2, np.eye(4),
                o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=100)
            )
            print(f"    Fitness: {coarse_result.fitness:.4f}, RMSE: {coarse_result.inlier_rmse:.4f}")
            
            # Apply coarse transformation
            recon_pcd.transform(coarse_result.transformation)
            
            # Fine alignment with smaller threshold
            print(f"  Fine alignment...")
            fine_result = o3d.pipelines.registration.registration_icp(
                recon_pcd, self.gt_pcd, 0.05, np.eye(4),
                o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=200)
            )
            print(f"    Fitness: {fine_result.fitness:.4f}, RMSE: {fine_result.inlier_rmse:.4f}")
            
            # Apply fine transformation
            recon_pcd.transform(fine_result.transformation)
            
            # Verify alignment
            gt_center = self.gt_pcd.get_center()
            recon_center = recon_pcd.get_center()
            center_dist = np.linalg.norm(gt_center - recon_center)
            
            print(f"\n  Alignment verification:")
            print(f"    GT center: [{gt_center[0]:.6f}, {gt_center[1]:.6f}, {gt_center[2]:.6f}]")
            print(f"    Recon center: [{recon_center[0]:.6f}, {recon_center[1]:.6f}, {recon_center[2]:.6f}]")
            print(f"    Center distance: {center_dist:.6f}")
            
            if center_dist < 0.01:
                print(f"  ✓ Centers aligned within tolerance")
            else:
                print(f"  ⚠ Centers have {center_dist:.6f} offset (may indicate partial reconstruction)")
            
            print(f"  ✓ Multi-scale ICP complete")
        
        # Check bounding boxes for scale/alignment issues
        recon_bbox = recon_pcd.get_axis_aligned_bounding_box()
        gt_bbox = self.gt_mesh.get_axis_aligned_bounding_box()
        
        recon_extent = recon_bbox.get_extent()
        gt_extent = gt_bbox.get_extent()
        recon_center = recon_bbox.get_center()
        gt_center = gt_bbox.get_center()
        
        print(f"\n{'='*60}")
        print("Bounding Box Analysis:")
        print(f"{'='*60}")
        print(f"Ground Truth:")
        print(f"  Center: [{gt_center[0]:.2f}, {gt_center[1]:.2f}, {gt_center[2]:.2f}]")
        print(f"  Size:   [{gt_extent[0]:.2f}, {gt_extent[1]:.2f}, {gt_extent[2]:.2f}]")
        print(f"\nReconstruction:")
        print(f"  Center: [{recon_center[0]:.2f}, {recon_center[1]:.2f}, {recon_center[2]:.2f}]")
        print(f"  Size:   [{recon_extent[0]:.2f}, {recon_extent[1]:.2f}, {recon_extent[2]:.2f}]")
        print(f"\nCenter offset: {np.linalg.norm(recon_center - gt_center):.2f}m")
        print(f"Scale ratio (avg): {np.mean(recon_extent / gt_extent):.2f}x")
        
        # Warning if significant mismatch
        if np.linalg.norm(recon_center - gt_center) > 1.0:
            print(f"\n⚠ WARNING: Centers are {np.linalg.norm(recon_center - gt_center):.2f}m apart!")
            print("  The meshes may need alignment before comparison.")
        
        scale_ratio = np.mean(recon_extent / gt_extent)
        if scale_ratio < 0.5 or scale_ratio > 2.0:
            print(f"\n⚠ WARNING: Scale mismatch detected (ratio: {scale_ratio:.2f}x)!")
            print("  The reconstruction may need scaling before comparison.")
        
        # Compute metrics
        visibility = self.compute_mesh_visibility(recon_pcd, coverage_threshold)
        overlap = self.compute_mesh_overlap(recon_pcd)
        
        # Combine results
        results = {
            'reconstruction_file': reconstructed_path,
            'ground_truth_file': str(self.gt_path),
            'num_reconstructed_points': len(recon_pcd.points),
            'num_gt_vertices': len(self.gt_mesh.vertices),
            'num_gt_triangles': len(self.gt_mesh.triangles),
            'visibility': visibility,
            'overlap': overlap
        }
        
        # Visualize if requested
        if self.visualization:
            self.visualize_comparison(recon_pcd, visibility)
        
        return results


def save_results(results: dict, output_path: str):
    """Save comparison results to JSON file."""
    import json
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Compare reconstructed point cloud with ground truth mesh"
    )
    parser.add_argument(
        "reconstructed",
        type=str,
        help="Path to reconstructed point cloud (.ply)"
    )
    parser.add_argument(
        "--ground-truth",
        type=str,
        default="../../Comparisons/House_with_Texture3.obj",
        help="Path to ground truth mesh (default: ../../Comparisons/House_with_Texture3.obj)"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.1,
        help="Coverage threshold in meters (default: 0.1)"
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        default=True,
        help="Normalize both meshes to same scale before comparison (default: True)"
    )
    parser.add_argument(
        "--no-normalize",
        dest="normalize",
        action="store_false",
        help="Disable normalization"
    )
    parser.add_argument(
        "--align",
        action="store_true",
        default=True,
        help="Align reconstruction to ground truth using ICP (default: True)"
    )
    parser.add_argument(
        "--no-align",
        dest="align",
        action="store_false",
        help="Disable ICP alignment"
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Show visualization"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON file for results (optional)"
    )
    
    args = parser.parse_args()
    
    # Create comparator
    comparator = MeshComparator(args.ground_truth, visualization=args.visualize, normalize=args.normalize, align=args.align)
    
    # Run comparison
    results = comparator.compare(args.reconstructed, coverage_threshold=args.threshold)
    
    # Save results if output specified
    if args.output:
        save_results(results, args.output)
    
    # Print summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Reconstruction:  {args.reconstructed}")
    print(f"Ground Truth:    {args.ground_truth}")
    print(f"Accuracy (mean distance to GT): {results['overlap']['mean_distance_to_gt']:.4f}m")
    print(f"Chamfer Dist:    {results['overlap']['chamfer_distance']:.4f}m")

    print(f"Coverage (Mesh Visibility):        {results['visibility']['coverage_percentage']:.2f}%")
    print(f"Mesh Visibility: The percentage of GT mesh surface captured.")
    print(f"Acuracy (Mesh Overlap): {results['overlap']['accuracy_percentage']:.2f}%")
    print(f"Mesh Overlap: How much of the points are within the adjusted threshold (scaled for normalization) {results['overlap']['accuracy_threshold']:.4f}m.")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
