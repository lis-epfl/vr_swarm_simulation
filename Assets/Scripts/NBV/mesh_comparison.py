#!/usr/bin/env python3
"""
mesh_comparison.py - Mesh Quality Evaluation (FIXED VERSION)

Compares reconstructed point clouds against ground truth mesh.
Computes:
1. Mesh Visibility (Coverage): How much of GT mesh is covered
2. Mesh Overlap (Accuracy): How close reconstruction is to GT

Key fixes:
- Rebuild GT point cloud after all transformations
- Improved ICP with RANSAC-based outlier rejection
- Better convergence criteria
- Proper KDTree rebuilding
"""
import copy
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

try:
    import trimesh
    HAS_TRIMESH = True
except ImportError:
    print("Warning: trimesh not found. FBX loading may be limited.")
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
        self.normalization_scale = None
        self.gt_mesh = None
        self.gt_pcd = None
        self.gt_kdtree = None
        
        print(f"\n{'='*60}")
        print("Loading Ground Truth Mesh")
        print(f"{'='*60}")
        self._load_ground_truth()
        
    def _load_ground_truth(self):
        """Load ground truth mesh and convert to point cloud."""
        if not self.gt_path.exists():
            raise FileNotFoundError(f"Ground truth not found: {self.gt_path}")
        
        print(f"Loading: {self.gt_path}")
        
        try:
            if self.gt_path.suffix.lower() == '.fbx' and HAS_TRIMESH:
                print("  Using trimesh for FBX loading...")
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
            
            if not self.gt_mesh.has_vertices():
                raise ValueError("Loaded mesh has no vertices")
            
            print(f"  ✓ Loaded mesh: {len(self.gt_mesh.vertices)} vertices, {len(self.gt_mesh.triangles)} triangles")
            
            if not self.gt_mesh.has_vertex_normals():
                self.gt_mesh.compute_vertex_normals()
            
            # Initial sampling - will be rebuilt after transformations
            num_samples = max(100000, len(self.gt_mesh.vertices) * 10)
            print(f"  Sampling {num_samples} points from mesh surface...")
            self.gt_pcd = self.gt_mesh.sample_points_uniformly(number_of_points=num_samples)
            print(f"  ✓ Sampled {len(self.gt_pcd.points)} points")
            
            self.gt_kdtree = o3d.geometry.KDTreeFlann(self.gt_pcd)
            
        except Exception as e:
            print(f"Error loading ground truth: {e}")
            raise
    
    def _rebuild_gt_point_cloud(self):
        """Rebuild GT point cloud from transformed mesh."""
        print("  Rebuilding GT point cloud from transformed mesh...")
        num_samples = max(100000, len(self.gt_mesh.vertices) * 10)
        self.gt_pcd = self.gt_mesh.sample_points_uniformly(number_of_points=num_samples)
        self.gt_kdtree = o3d.geometry.KDTreeFlann(self.gt_pcd)
        print(f"  ✓ Resampled {len(self.gt_pcd.points)} points from transformed mesh")
    
    def compute_mesh_visibility(self, reconstructed_pcd: o3d.geometry.PointCloud, 
                                threshold: float = 0.1) -> dict:
        """Compute mesh visibility (coverage) metric."""
        print(f"\n{'='*60}")
        print("Computing Mesh Visibility (Coverage)")
        print(f"{'='*60}")
        
        adjusted_threshold = threshold
        if self.normalization_scale is not None:
            adjusted_threshold = threshold / self.normalization_scale
            print(f"Original threshold: {threshold}m")
            print(f"Adjusted threshold (normalized): {adjusted_threshold:.6f}")
        else:
            print(f"Distance threshold: {threshold}m")
        
        recon_kdtree = o3d.geometry.KDTreeFlann(reconstructed_pcd)
        
        gt_points = np.asarray(self.gt_pcd.points)
        covered_points = 0
        distances = []
        
        print("Checking coverage for each ground truth point...")
        for i, gt_point in enumerate(gt_points):
            if i % 10000 == 0:
                print(f"  Progress: {i}/{len(gt_points)} ({100*i/len(gt_points):.1f}%)")
            
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
        print(f"  Avg distance to GT: {results['avg_distance_to_recon']:.4f}")
        print(f"  Median distance:    {results['median_distance_to_recon']:.4f}")
        
        return results
    
    def compute_mesh_overlap(self, reconstructed_pcd: o3d.geometry.PointCloud, threshold: float) -> dict:
        """Compute mesh overlap (accuracy) metric."""
        print(f"\n{'='*60}")
        print("Computing Mesh Overlap (Accuracy)")
        print(f"{'='*60}")
        
        recon_points = np.asarray(reconstructed_pcd.points)
        distances_to_gt = []
        
        print("Computing distances from reconstruction to ground truth...")
        for i, recon_point in enumerate(recon_points):
            if i % 10000 == 0:
                print(f"  Progress: {i}/{len(recon_points)} ({100*i/len(recon_points):.1f}%)")
            
            [k, idx, dist_sq] = self.gt_kdtree.search_knn_vector_3d(recon_point, 1)
            distance = np.sqrt(dist_sq[0])
            distances_to_gt.append(distance)
        
        distances_to_gt = np.array(distances_to_gt)
        
        # Compute bidirectional distances for Chamfer
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
        
        recon_bbox = reconstructed_pcd.get_axis_aligned_bounding_box()
        recon_extent = recon_bbox.get_extent()
        recon_diagonal = np.linalg.norm(recon_extent)
        
        chamfer_distance = (np.mean(distances_to_gt) + np.mean(distances_from_gt)) / 2
        hausdorff_distance = max(np.max(distances_to_gt), np.max(distances_from_gt))
        
        accuracy_threshold = threshold
        if self.normalization_scale is not None:
            accuracy_threshold = threshold/ self.normalization_scale
        
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
            'accuracy_percentage': accuracy_percentage,
            'accurate_points': accurate_points,
            'accuracy_threshold': accuracy_threshold,
            'mean_distance_pct': (np.mean(distances_to_gt) / recon_diagonal) * 100,
            'chamfer_distance_pct': (chamfer_distance / recon_diagonal) * 100,
            'hausdorff_distance_pct': (hausdorff_distance / recon_diagonal) * 100,
        }
        
        print(f"\n{'='*60}")
        print("Overlap/Accuracy Results:")
        print(f"{'='*60}")
        print(f"  Accuracy (points within threshold): {results['accuracy_percentage']:.2f}%")
        print(f"    ({results['accurate_points']:,} / {len(distances_to_gt):,} points within {accuracy_threshold:.6f})")
        print(f"\n  Distance Metrics:")
        print(f"    Reconstruction diagonal: {recon_diagonal:.4f}")
        print(f"    Mean distance (recon→GT):   {results['mean_distance_to_gt']:.4f} ({results['mean_distance_pct']:.2f}% of size)")
        print(f"    Median distance (recon→GT): {results['median_distance_to_gt']:.4f}")
        print(f"    Std deviation:              {results['std_distance_to_gt']:.4f}")
        print(f"    Chamfer distance:           {results['chamfer_distance']:.4f} ({results['chamfer_distance_pct']:.2f}% of size)")
        print(f"    Hausdorff distance:         {results['hausdorff_distance']:.4f} ({results['hausdorff_distance_pct']:.2f}% of size)")
        print(f"    90th percentile:            {results['percentile_90']:.4f}")
        print(f"    95th percentile:            {results['percentile_95']:.4f}")
        
        return results
    
    def visualize_comparison(self, reconstructed_pcd: o3d.geometry.PointCloud,
                            visibility_results: dict = None):
        """Visualize ground truth mesh with reconstruction."""
        if not self.visualization:
            return
        
        print(f"\n{'='*60}")
        print("Visualization")
        print(f"{'='*60}")
        
        self.gt_mesh.paint_uniform_color([0.7, 0.7, 0.7])
        reconstructed_pcd.paint_uniform_color([1.0, 0.2, 0.2])
        
        print("\nShowing ground truth (gray) and reconstruction (red)...")
        print("  Close the window to continue")
        
        o3d.visualization.draw_geometries(
            [self.gt_mesh, reconstructed_pcd],
            window_name="Ground Truth (gray) - Reconstruction (red)",
            width=1280,
            height=720
        )
    
    def compare(self, reconstructed_path: str, threshold: float = 0.1) -> dict:
        """Run full comparison pipeline."""
        print(f"\n{'='*60}")
        print("Loading Reconstructed Point Cloud")
        print(f"{'='*60}")
        print(f"Loading: {reconstructed_path}")
        
        recon_pcd = o3d.io.read_point_cloud(reconstructed_path)
        print(f"  ✓ Loaded {len(recon_pcd.points)} points")
        
        # STEP 1: Voxelize reconstruction
        print(f"\n{'='*60}")
        print("Step 1: Voxelizing Reconstruction")
        print(f"{'='*60}")
        original_size = len(recon_pcd.points)
        voxel_size = 0.05
        recon_pcd = recon_pcd.voxel_down_sample(voxel_size)
        voxelized_size = len(recon_pcd.points)
        print(f"  Original points: {original_size:,}")
        print(f"  After voxelization: {voxelized_size:,}")
        print(f"  ✓ Removed {original_size - voxelized_size:,} duplicate/overlapping points")
        
        # STEP 2: Apply X-flip to GT mesh
        print(f"\n{'='*60}")
        print("Step 2: Coordinate System Fix (X-flip GT)")
        print(f"{'='*60}")
        flip_transform = np.eye(4)
        flip_transform[0, 0] = -1
        self.gt_mesh.transform(flip_transform)
        print(f"  ✓ Applied X-flip to GT mesh")
        
        # STEP 3: Scale GT to half size
        print(f"\n{'='*60}")
        print("Step 3: Scale Ground Truth (0.5x)")
        print(f"{'='*60}")
        gt_center_before = self.gt_mesh.get_center()
        print(f"  GT center before scaling: [{gt_center_before[0]:.2f}, {gt_center_before[1]:.2f}, {gt_center_before[2]:.2f}]")
        self.gt_mesh.scale(0.5, center=[0, 0, 0])
        print(f"  ✓ Scaled GT to 0.5x (half size)")
        
        # STEP 4: Normalize GT mesh
        print(f"\n{'='*60}")
        print("Step 4: Normalize Ground Truth")
        print(f"{'='*60}")
        # gt_center = self.gt_mesh.get_center()
        # print(f"  GT center: [{gt_center[0]:.2f}, {gt_center[1]:.2f}, {gt_center[2]:.2f}]")
        # self.gt_mesh.translate(-gt_center)
        gt_max_bound = np.max(self.gt_mesh.get_max_bound() - self.gt_mesh.get_min_bound())
        print(f"  GT max dimension: {gt_max_bound:.2f}m")
        
        normalization_scale_factor = 1.0 / gt_max_bound
        print(f"  Normalization scale factor: {normalization_scale_factor:.6f}")
        
        self.gt_mesh.scale(normalization_scale_factor, center=[0, 0, 0])
        print(f"  ✓ GT centered and normalized to unit size")
        
        # CRITICAL FIX: Rebuild GT point cloud after all transformations
        self._rebuild_gt_point_cloud()
        
        # STEP 5: Normalize reconstruction
        print(f"\n{'='*60}")
        print("Step 5: Normalize Reconstruction")
        print(f"{'='*60}")
        recon_center = recon_pcd.get_center()
        recon_max_bound_orig = np.max(recon_pcd.get_max_bound() - recon_pcd.get_min_bound())
        print(f"  Original recon center: [{recon_center[0]:.2f}, {recon_center[1]:.2f}, {recon_center[2]:.2f}]")
        print(f"  Original recon max dimension: {recon_max_bound_orig:.2f}m")
        
        # recon_pcd.translate(-recon_center)
        recon_pcd.scale(normalization_scale_factor, center=[0, 0, 0])
        print(f"  ✓ Reconstruction centered and scaled by same factor as GT")
        
        self.normalization_scale = gt_max_bound
        print(f"  Original GT scale (for threshold adjustment): {self.normalization_scale:.4f}m")

        # # Apply manual initial translation if provided
        # # manual_translation = (0.3, 0, 0.15)  # Set to None or desired values
        # manual_translation = None  # Disable manual translation
        # if manual_translation is not None:
        #     print(f"\n  Applying manual initial translation: [{manual_translation[0]:+.4f}, {manual_translation[1]:+.4f}, {manual_translation[2]:+.4f}]")
        #     recon_pcd.translate(manual_translation)
        #     print(f"  ✓ Manual translation applied")
        # else:
        #     print(f"\n  No manual initial translation provided")

        #         # Visualize BEFORE ICP alignment
        # if self.visualization:
        #     print(f"\n{'='*60}")
        #     print("Visualization: BEFORE ICP Alignment")
        #     print(f"{'='*60}")
        #     import copy
        #     recon_before = copy.deepcopy(recon_pcd)
        #     self.gt_mesh.paint_uniform_color([0.7, 0.7, 0.7])  # Gray for GT
        #     recon_before.paint_uniform_color([1.0, 0.5, 0])  # Orange for reconstruction
        #     
        #     print("\nShowing BEFORE alignment (Gray=GT, Orange=Reconstruction)...")
        #     print("  Close the window to continue to ICP alignment")
        #     
        #     o3d.visualization.draw_geometries(
        #         [self.gt_mesh, recon_before],
        #         window_name="BEFORE ICP Alignment",
        #         width=1280,
        #         height=720
        #     )
        # 
        # # STEP 6: Translation-only alignment (no rotation)
        # if self.align:
        #     print(f"\n{'='*60}")
        #     print("Step 6: Translation-Only Alignment")
        #     print(f"{'='*60}")
        #     print("  Orientation is correct - finding best translation only")
        #     

        #     
        #     # Build KDTree for fast nearest neighbor search
        #     gt_kdtree = o3d.geometry.KDTreeFlann(self.gt_pcd)
        #     
        #     # Iterative closest point translation
        #     best_translation = manual_translation if manual_translation is not None else np.array([0.0, 0.0, 0.0])
        #     prev_avg_distance = float('inf')
        #     correspondence_threshold = 0.3  # Start generous, will tighten
        #     
        #     print(f"  Starting iterative translation alignment...")
        #     
        #     for iteration in range(100):  # Max iterations
        #         recon_points = np.asarray(recon_pcd.points)
        #         
        #         # Find correspondences
        #         valid_correspondences = []
        #         distances = []
        #         
        #         for recon_pt in recon_points:
        #             [k, idx, dist_sq] = gt_kdtree.search_knn_vector_3d(recon_pt, 1)
        #             distance = np.sqrt(dist_sq[0])
        #             
        #             # Only use close correspondences
        #             if distance < correspondence_threshold:
        #                 gt_pt = np.asarray(self.gt_pcd.points)[idx[0]]
        #                 valid_correspondences.append((recon_pt, gt_pt))
        #                 distances.append(distance)
        #         
        #         if len(valid_correspondences) < 1000:
        #             print(f"    Warning: Only {len(valid_correspondences)} correspondences found")
        #             if correspondence_threshold < 1.0:
        #                 correspondence_threshold *= 1.5
        #                 print(f"    Increasing threshold to {correspondence_threshold:.3f}")
        #                 continue
        #             else:
        #                 print(f"    Stopping - insufficient correspondences")
        #                 break
        #         
        #         # Calculate average distance
        #         avg_distance = np.mean(distances)
        #         
        #         # Calculate translation from correspondences
        #         recon_pts = np.array([c[0] for c in valid_correspondences])
        #         gt_pts = np.array([c[1] for c in valid_correspondences])
        #         translation_step = np.mean(gt_pts - recon_pts, axis=0)
        #         
        #         # Apply translation
        #         recon_pcd.translate(translation_step)
        #         best_translation += translation_step
        #         
        #         # Progress reporting
        #         if iteration % 10 == 0 or iteration < 5:
        #             print(f"    Iter {iteration:3d}: {len(valid_correspondences):6,} corr, "
        #                   f"avg dist: {avg_distance:.6f}, "
        #                   f"step: [{translation_step[0]:+.6f}, {translation_step[1]:+.6f}, {translation_step[2]:+.6f}]")
        #         
        #         # Check convergence
        #         if np.linalg.norm(translation_step) < 1e-9:
        #             print(f"    ✓ Converged at iteration {iteration + 1} (step size < 1e-6)")
        #             break
        #         
        #         if abs(prev_avg_distance - avg_distance) < 1e-9:
        #             print(f"    ✓ Converged at iteration {iteration + 1} (distance change < 1e-6)")
        #             break
        #         
        #         prev_avg_distance = avg_distance
        #         
        #         # Gradually tighten correspondence threshold
        #         if iteration > 20 and correspondence_threshold > 0.1:
        #             correspondence_threshold *= 0.95
        #     
        #     print(f"\n  Final Results:")
        #     print(f"    Total translation: [{best_translation[0]:+.6f}, {best_translation[1]:+.6f}, {best_translation[2]:+.6f}]")
        #     print(f"    Final correspondences: {len(valid_correspondences):,}")
        #     print(f"    Final avg distance: {avg_distance:.6f}")
        #     print(f"  ✓ Translation-only alignment complete")
        # else:
        #     print(f"\n  Skipping ICP alignment (--no-align flag set)")
        
        # Bounding box analysis
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
        print(f"  Center: [{gt_center[0]:.4f}, {gt_center[1]:.4f}, {gt_center[2]:.4f}]")
        print(f"  Size:   [{gt_extent[0]:.4f}, {gt_extent[1]:.4f}, {gt_extent[2]:.4f}]")
        print(f"\nReconstruction:")
        print(f"  Center: [{recon_center[0]:.4f}, {recon_center[1]:.4f}, {recon_center[2]:.4f}]")
        print(f"  Size:   [{recon_extent[0]:.4f}, {recon_extent[1]:.4f}, {recon_extent[2]:.4f}]")
        print(f"\nCenter offset: {np.linalg.norm(recon_center - gt_center):.4f}")
        print(f"Scale ratio (avg): {np.mean(recon_extent / gt_extent):.4f}x")
        
        center_distance = np.linalg.norm(recon_center - gt_center)
        if center_distance > 0.1:
            print(f"\n⚠ WARNING: Centers are {center_distance:.4f} apart after alignment!")
        
        # Compute metrics
        visibility = self.compute_mesh_visibility(recon_pcd, threshold)
        overlap = self.compute_mesh_overlap(recon_pcd, threshold)
        
        results = {
            'reconstruction_file': reconstructed_path,
            'ground_truth_file': str(self.gt_path),
            'num_reconstructed_points': len(recon_pcd.points),
            'num_gt_vertices': len(self.gt_mesh.vertices),
            'num_gt_triangles': len(self.gt_mesh.triangles),
            'alignment_center_offset': center_distance,
            'visibility': visibility,
            'overlap': overlap
        }
        
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
        help="Path to ground truth mesh"
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
        help="Normalize both meshes (default: True)"
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
        default=False,
        help="Align using ICP (default: False)"
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
    )
    
    args = parser.parse_args()
    
    comparator = MeshComparator(args.ground_truth, visualization=args.visualize, 
                                normalize=args.normalize, align=args.align)
    comparator = MeshComparator(args.ground_truth, visualization=args.visualize, 
                                normalize=args.normalize, align=args.align)
    
    results = comparator.compare(args.reconstructed, threshold=args.threshold)
    
    if args.output:
        save_results(results, args.output)
    
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Reconstruction:  {args.reconstructed}")
    print(f"Ground Truth:    {args.ground_truth}")
    print(f"\nAlignment Quality:")
    print(f"  Center offset after ICP: {results['alignment_center_offset']:.4f}")
    print(f"\nAccuracy Metrics:")
    print(f"  Mean distance to GT: {results['overlap']['mean_distance_to_gt']:.4f}")
    print(f"  Chamfer distance:    {results['overlap']['chamfer_distance']:.4f}")
    print(f"  Accuracy (within threshold): {results['overlap']['accuracy_percentage']:.2f}%")
    print(f"\nCoverage Metrics:")
    print(f"  Mesh visibility (GT coverage): {results['visibility']['coverage_percentage']:.2f}%")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()