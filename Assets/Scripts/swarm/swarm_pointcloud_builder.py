#!/usr/bin/env python3
"""
swarm_pointcloud_builder.py - Olfati-Saber Swarm Point Cloud Builder

Processes captured RGB and depth images from Unity SwarmImageCapture.cs
Performs segmentation, creates point clouds, and incrementally merges them.

Usage:
    python swarm_pointcloud_builder.py [--output-dir OUTPUT_DIR] [--poll-interval SECONDS]
"""

import os
import time
import json
import struct
import numpy as np
import cv2
import torch
import argparse
from typing import List, Optional, Tuple
from scipy.spatial.transform import Rotation
from dataclasses import dataclass
from pathlib import Path

# Check and create D:/ drive directories if they don't exist
D_DRIVE_BASE = r"D:\advaith\unity-run-files"
if os.path.exists("D:\\"):
    os.makedirs(os.path.join(D_DRIVE_BASE, "ProcessedImages", "SwarmPointClouds"), exist_ok=True)
    os.makedirs(os.path.join(D_DRIVE_BASE, "ProcessedImages", "SwarmCapture"), exist_ok=True)
    print(f"D:/ drive directories ready at {D_DRIVE_BASE}")

# import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")

try:
    import open3d as o3d
    HAS_OPEN3D = True
except ImportError:
    HAS_OPEN3D = False
    print("Warning: Open3D not available. Point cloud operations will be limited.")

from segment_anything import sam_model_registry, SamAutomaticMaskGenerator


@dataclass
class CameraIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int


@dataclass
class DronePose:
    position: np.ndarray
    quaternion: np.ndarray


@dataclass
class DroneData:
    drone_id: int
    rgb_image: np.ndarray
    depth_image: np.ndarray
    pose: DronePose


class SwarmPointCloudBuilder:
    """Processes swarm capture data and builds incremental point clouds."""
    
    MIN_DISTANCE_FROM_DRONE = 1.1  # meters (filter out drone mesh)
    TARGET_DOWNSAMPLE_SIZE = 8000  # Target number of points for downsampled cloud
    
    def __init__(self, 
                 capture_dir: str = r"D:\advaith\unity-run-files\ProcessedImages\SwarmCapture",
                 output_dir: str = r"D:\advaith\unity-run-files\ProcessedImages\SwarmPointClouds",
                 sam_model_type: str = "vit_h",
                 min_building_area: int = 1000,
                 capture_interval: int = 1):
        """
        Initialize the point cloud builder.
        
        Args:
            capture_dir: Directory where Unity saves captures
            output_dir: Directory to save point clouds
            sam_model_type: SAM model type (vit_b, vit_l, vit_h)
            min_building_area: Minimum area for segmentation masks
            capture_interval: Process every Nth capture (1=all, 2=every 2nd, 3=every 3rd, etc.)
        """
        self.capture_dir = Path(capture_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.sam_model_type = sam_model_type
        self.sam_model_paths = {
            "vit_b": "../../segmentAnything/sam_vit_b_01ec64.pth",
            "vit_l": "../../segmentAnything/sam_vit_l_0b3195.pth",
            "vit_h": "../../segmentAnything/sam_vit_h_4b8939.pth",
        }
        self.min_building_area = min_building_area
        self.capture_interval = capture_interval
        
        self.mask_generator = None
        self.device = None
        self.processed_captures = set()  # Track processed capture folders
        
        # Accumulated point cloud state
        self.accumulated_points = None
        self.accumulated_colors = None
        
        # CSV logging
        self.csv_file = None
        self.csv_writer = None
        self.initialize_csv_logging()
    
    def initialize_csv_logging(self):
        """Initialize CSV file for drone position logging."""
        import csv
        from datetime import datetime
        
        try:
            # Create output directory in D:/ drive
            csv_dir = r"D:\advaith\unity-run-files"
            os.makedirs(csv_dir, exist_ok=True)
            
            # Generate timestamped CSV filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            csv_filename = f"swarm_positions_{timestamp}.csv"
            csv_path = os.path.join(csv_dir, csv_filename)
            
            # Open CSV file for writing
            self.csv_file = open(csv_path, 'w', newline='')
            self.csv_writer = csv.writer(self.csv_file)
            
            # Write header
            self.csv_writer.writerow(['capture_time', 'drone_id', 'pos_x', 'pos_y', 'pos_z', 'quat_x', 'quat_y', 'quat_z', 'quat_w'])
            self.csv_file.flush()
            
            print(f"[CSV] Logging initialized: {csv_path}")
        except Exception as e:
            print(f"[CSV] Warning: Failed to initialize CSV logging: {e}")
            self.csv_writer = None
    
    def log_drone_positions(self, capture_name: str, drone_data_list: List[DroneData]):
        """Log all drone positions for this capture to CSV."""
        if self.csv_writer is None:
            return
        
        try:
            # Extract timestamp from capture name (capture_YYYYMMDD_HHMMSS)
            capture_time = capture_name.replace('capture_', '')
            
            # Log each drone
            for drone_data in drone_data_list:
                pos = drone_data.pose.position
                quat = drone_data.pose.quaternion
                
                self.csv_writer.writerow([
                    capture_time,
                    f"Drone {drone_data.drone_id}",
                    f"{pos[0]:.6f}",
                    f"{pos[1]:.6f}",
                    f"{pos[2]:.6f}",
                    f"{quat[0]:.6f}",
                    f"{quat[1]:.6f}",
                    f"{quat[2]:.6f}",
                    f"{quat[3]:.6f}"
                ])
            
            # Flush to ensure data is written immediately
            self.csv_file.flush()
        except Exception as e:
            print(f"[CSV] Warning: Failed to log positions: {e}")
        
    def initialize(self):
        """Initialize SAM model."""
        # Segmentation disabled - using depth values only
        # print("Initializing Segment Anything model...")
        # 
        # # Try to use GPU (CUDA only - MPS has compatibility issues with SAM)
        # if torch.cuda.is_available():
        #     self.device = torch.device('cuda')
        #     print(f"Using device: cuda (NVIDIA GPU)")
        # else:
        #     self.device = torch.device('cpu')
        #     print(f"Using device: cpu (WARNING: This will be slow!)")
        # 
        # checkpoint_path = self.sam_model_paths[self.sam_model_type]
        # if not os.path.exists(checkpoint_path):
        #     print(f"Error: SAM checkpoint not found at {checkpoint_path}")
        #     return False
        # 
        # sam = sam_model_registry[self.sam_model_type](checkpoint=checkpoint_path)
        # sam.to(self.device)
        # 
        # # Optimized SAM parameters for faster processing
        # self.mask_generator = SamAutomaticMaskGenerator(
        #     sam,
        #     points_per_side=8,  # Reduced from 16 for 4x speedup
        #     pred_iou_thresh=0.86,  # Slightly lower for speed
        #     stability_score_thresh=0.92,  # Slightly lower for speed
        #     crop_n_layers=0,
        #     crop_n_points_downscale_factor=1,
        #     min_mask_region_area=5000,  # Increased to filter small segments
        # )
        # 
        # print("[OK] SAM model initialized")
        print("[OK] Segmentation disabled - using depth values only")
        return True
    
    def find_new_captures(self) -> List[Path]:
        """Find new capture folders that haven't been processed yet."""
        if not self.capture_dir.exists():
            return []
        
        new_captures = []
        for folder in sorted(self.capture_dir.iterdir()):
            if folder.is_dir() and folder.name.startswith("capture_"):
                if folder.name not in self.processed_captures:
                    # Check if capture is complete (has metadata.json)
                    metadata_path = folder / "metadata.json"
                    if not metadata_path.exists():
                        continue
                    
                    # Verify all expected drone files exist
                    try:
                        with open(metadata_path, 'r') as f:
                            metadata = json.load(f)
                        drone_count = metadata.get('drone_count', 0)
                        
                        # Check if all required files exist for each drone
                        all_files_exist = True
                        for i in range(drone_count):
                            depth_file = folder / f"drone_{i}_depth.raw"
                            pose_file = folder / f"drone_{i}_pose.json"
                            
                            if not depth_file.exists() or not pose_file.exists():
                                all_files_exist = False
                                break
                        
                        if all_files_exist:
                            new_captures.append(folder)
                    except:
                        # If we can't read metadata or it's malformed, skip for now
                        pass
        
        # Filter by capture interval (use index-based subsampling)
        if self.capture_interval > 1:
            filtered_captures = []
            for idx, capture in enumerate(new_captures):
                if idx % self.capture_interval == 0:
                    filtered_captures.append(capture)
            print(f"[Interval Filter] Processing every {self.capture_interval}th capture: {len(filtered_captures)}/{len(new_captures)} captures")
            return filtered_captures
        
        return new_captures
    
    def load_capture_data(self, capture_folder: Path) -> Tuple[List[DroneData], CameraIntrinsics]:
        """Load all data from a capture folder."""
        print(f"\nLoading capture: {capture_folder.name}")
        
        # Load metadata
        with open(capture_folder / "metadata.json", 'r') as f:
            metadata = json.load(f)
        
        drone_count = metadata['drone_count']
        intrinsics_data = metadata['camera_intrinsics']
        intrinsics = CameraIntrinsics(
            fx=intrinsics_data['fx'],
            fy=intrinsics_data['fy'],
            cx=intrinsics_data['cx'],
            cy=intrinsics_data['cy'],
            width=metadata['image_width'],
            height=metadata['image_height']
        )
        
        print(f"  Drone count: {drone_count}")
        print(f"  Resolution: {intrinsics.width}x{intrinsics.height}")
        print(f"  Intrinsics: fx={intrinsics.fx:.1f}, fy={intrinsics.fy:.1f}")
        
        # Load data for each drone
        drone_data_list = []
        for i in range(drone_count):
            # RGB not needed - using depth only
            # rgb_path = capture_folder / f"drone_{i}_rgb.png"
            # rgb_image = cv2.imread(str(rgb_path))
            # rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
            
            # Load Depth (raw float32 binary)
            depth_path = capture_folder / f"drone_{i}_depth.raw"
            with open(depth_path, 'rb') as f:
                depth_bytes = f.read()
            depth_image = np.frombuffer(depth_bytes, dtype=np.float32)
            depth_image = depth_image.reshape((intrinsics.height, intrinsics.width))
            
            # Create dummy RGB for compatibility (we'll generate colors from depth)
            rgb_image = np.zeros((intrinsics.height, intrinsics.width, 3), dtype=np.uint8)
            
            # Load Pose
            pose_path = capture_folder / f"drone_{i}_pose.json"
            with open(pose_path, 'r') as f:
                pose_data = json.load(f)
            
            position = np.array([
                pose_data['position']['x'],
                pose_data['position']['y'],
                pose_data['position']['z']
            ])
            quaternion = np.array([
                pose_data['quaternion']['x'],
                pose_data['quaternion']['y'],
                pose_data['quaternion']['z'],
                pose_data['quaternion']['w']
            ])
            
            pose = DronePose(position, quaternion)
            drone_data = DroneData(i, rgb_image, depth_image, pose)
            drone_data_list.append(drone_data)
            
            print(f"  Drone {i}: Depth {depth_image.shape}, Pos {position}")
        
        return drone_data_list, intrinsics
    
    def segment_rgb_image(self, rgb_image: np.ndarray) -> Optional[np.ndarray]:
        """Segment RGB image using SAM to find building/object masks."""
        masks = self.mask_generator.generate(rgb_image)
        
        if len(masks) == 0:
            return None
        
        height, width = rgb_image.shape[:2]
        center_x, center_y = width // 2, height // 2
        
        # Find segments at image center with sufficient area
        center_segments = []
        for mask_info in masks:
            mask = mask_info['segmentation']
            area = mask_info['area']
            
            if (0 <= center_y < mask.shape[0] and 
                0 <= center_x < mask.shape[1] and
                mask[center_y, center_x] and 
                area >= self.min_building_area):
                center_segments.append(mask_info)
        
        if not center_segments:
            return None
        
        # Select best segment based on stability and area
        best_segment = max(center_segments,
                          key=lambda x: x['stability_score'] * min(x['area']/10000, 10))
        
        binary_mask = best_segment['segmentation'].astype(np.float32)
        return binary_mask
    
    def depth_to_point_cloud(self, 
                            depth_image: np.ndarray, 
                            segmentation_mask: Optional[np.ndarray],
                            rgb_image: np.ndarray,
                            intrinsics: CameraIntrinsics) -> Tuple[np.ndarray, np.ndarray]:
        """Convert depth image to 3D point cloud in camera frame."""
        height, width = depth_image.shape
        
        # Apply segmentation mask if available
        if segmentation_mask is not None:
            depth_masked = depth_image * segmentation_mask
        else:
            depth_masked = depth_image
        
        # Create pixel coordinate grids
        v, u = np.mgrid[0:height, 0:width]
        
        # Filter valid depth values
        valid_mask = (depth_masked > 0) & ~np.isnan(depth_masked)
        u_valid = u[valid_mask]
        v_valid = v[valid_mask]
        depth_valid = depth_masked[valid_mask]
        
        # Unproject to 3D camera coordinates
        X_cam = (u_valid - intrinsics.cx) * depth_valid / intrinsics.fx
        Y_cam = (v_valid - intrinsics.cy) * depth_valid / intrinsics.fy
        Z_cam = depth_valid
        
        points = np.stack([X_cam, Y_cam, Z_cam], axis=-1)
        
        # Generate colors from depth (grayscale based on distance)
        # Normalize depth to 0-1 range for visualization
        depth_normalized = np.clip((depth_valid - depth_valid.min()) / (depth_valid.max() - depth_valid.min() + 1e-6), 0, 1)
        colors = np.stack([depth_normalized, depth_normalized, depth_normalized], axis=-1)
        
        return points, colors
    
    def transform_to_global_frame(self, 
                                  points_local: np.ndarray, 
                                  pose: DronePose) -> np.ndarray:
        """Transform points from camera frame to global frame."""
        if points_local.shape[0] == 0:
            return points_local
        
        # Create transformation matrix from pose
        rotation = Rotation.from_quat(pose.quaternion)
        transform_matrix = np.eye(4)
        transform_matrix[:3, :3] = rotation.as_matrix()
        transform_matrix[:3, 3] = pose.position
        
        # Transform points
        points_local_hom = np.hstack((points_local, np.ones((points_local.shape[0], 1))))
        points_global_hom = (transform_matrix @ points_local_hom.T).T
        points_global = points_global_hom[:, :3]
        
        return points_global
    
    def fuse_point_clouds(self, 
                         drone_data_list: List[DroneData],
                         intrinsics: CameraIntrinsics) -> Tuple[np.ndarray, np.ndarray]:
        """Fuse point clouds from all drones into a single global point cloud."""
        all_points = []
        all_colors = []
        
        for drone_data in drone_data_list:
            print(f"  Processing drone {drone_data.drone_id}...")
            
            # Segmentation disabled - use depth values only
            # mask = self.segment_rgb_image(drone_data.rgb_image)
            # if mask is None:
            #     print(f"    No valid segmentation found")
            #     continue
            
            # Convert depth to point cloud (pass None for mask to use all depth values)
            points_local, colors = self.depth_to_point_cloud(
                drone_data.depth_image,
                None,  # No segmentation mask - use all depth values
                drone_data.rgb_image,
                intrinsics
            )
            
            # Skip if no points generated
            if len(points_local) == 0:
                print(f"    No valid points generated")
                continue
            
            # Filter out points close to drone (likely drone mesh)
            mask_dist = np.abs(points_local[:, 2]) > self.MIN_DISTANCE_FROM_DRONE
            filtered_points = points_local[mask_dist]
            filtered_colors = colors[mask_dist]
            
            print(f"    Generated {len(filtered_points)} points")
            
            # Transform to global frame
            points_global = self.transform_to_global_frame(filtered_points, drone_data.pose)
            
            all_points.append(points_global)
            all_colors.append(filtered_colors)
        
        if len(all_points) == 0:
            return np.array([]), np.array([])
        
        # Combine all point clouds
        global_points = np.vstack(all_points)
        global_colors = np.vstack(all_colors)
        
        print(f"  Total fused points: {len(global_points)}")
        
        return global_points, global_colors
    
    def merge_with_accumulated(self, 
                               new_points: np.ndarray, 
                               new_colors: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Merge new point cloud with accumulated point cloud."""
        if self.accumulated_points is None or len(self.accumulated_points) == 0:
            # First capture, just store it
            print("  First capture - initializing accumulated cloud")
            return new_points, new_colors
        
        # Merge with previous
        print(f"  Merging with accumulated cloud ({len(self.accumulated_points)} points)")
        merged_points = np.vstack([self.accumulated_points, new_points])
        merged_colors = np.vstack([self.accumulated_colors, new_colors])
        
        print(f"  Merged cloud size: {len(merged_points)} points")
        
        # Voxelize to remove duplicates
        if HAS_OPEN3D:
            print("  Voxelizing to remove duplicates...")
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(merged_points)
            pcd.colors = o3d.utility.Vector3dVector(merged_colors)
            
            # Voxel size based on point cloud density
            voxel_size = 0.05  # 5cm voxels
            pcd_voxelized = pcd.voxel_down_sample(voxel_size)
            
            merged_points = np.asarray(pcd_voxelized.points)
            merged_colors = np.asarray(pcd_voxelized.colors)
            
            print(f"  After voxelization: {len(merged_points)} points")
        
        return merged_points, merged_colors
    
    def downsample_point_cloud(self, 
                              points: np.ndarray, 
                              colors: np.ndarray,
                              target_size: int = TARGET_DOWNSAMPLE_SIZE) -> Tuple[np.ndarray, np.ndarray]:
        """Downsample point cloud to target size with uniform density."""
        if len(points) <= target_size:
            return points, colors
        
        if not HAS_OPEN3D:
            # Fallback: random sampling
            indices = np.random.choice(len(points), target_size, replace=False)
            return points[indices], colors[indices]
        
        # Use Open3D for better downsampling
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        
        # Use farthest point sampling for uniform distribution
        ratio = target_size / len(points)
        pcd_downsampled = pcd.random_down_sample(ratio)
        
        # Adjust if not exactly target size
        current_size = len(pcd_downsampled.points)
        if current_size < target_size:
            # Add more points
            remaining = target_size - current_size
            all_indices = set(range(len(points)))
            sampled_indices = set(tuple(pt) for pt in np.asarray(pcd_downsampled.points))
            # This is approximate, but good enough
            pcd_downsampled = pcd.random_down_sample(ratio * 1.1)
        
        down_points = np.asarray(pcd_downsampled.points)
        down_colors = np.asarray(pcd_downsampled.colors)
        
        # Final trim if needed
        if len(down_points) > target_size:
            down_points = down_points[:target_size]
            down_colors = down_colors[:target_size]
        
        return down_points, down_colors
    
    def save_point_cloud(self, 
                        points: np.ndarray, 
                        colors: np.ndarray,
                        filename: str):
        """Save point cloud to PLY file."""
        filepath = self.output_dir / filename
        
        if HAS_OPEN3D:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            pcd.colors = o3d.utility.Vector3dVector(colors)
            o3d.io.write_point_cloud(str(filepath), pcd)
        else:
            # Fallback: write ASCII PLY
            with open(filepath, 'w') as f:
                f.write("ply\n")
                f.write("format ascii 1.0\n")
                f.write(f"element vertex {len(points)}\n")
                f.write("property float x\n")
                f.write("property float y\n")
                f.write("property float z\n")
                f.write("property uchar red\n")
                f.write("property uchar green\n")
                f.write("property uchar blue\n")
                f.write("end_header\n")
                for i in range(len(points)):
                    x, y, z = points[i]
                    r, g, b = (colors[i] * 255).astype(np.uint8)
                    f.write(f"{x} {y} {z} {r} {g} {b}\n")
        
        print(f"  Saved: {filepath} ({len(points)} points)")
    
    def process_capture(self, capture_folder: Path):
        """Process a single capture folder."""
        # Check if already processed (safety check)
        if capture_folder.name in self.processed_captures:
            print(f"⏭ Skipping {capture_folder.name} - already processed")
            return
        
        # Mark as processing immediately to prevent duplicate processing
        self.processed_captures.add(capture_folder.name)
        
        print(f"\n{'='*60}")
        print(f"Processing: {capture_folder.name}")
        print(f"{'='*60}")
        
        try:
            # Load data
            drone_data_list, intrinsics = self.load_capture_data(capture_folder)
            
            # Log drone positions to CSV
            self.log_drone_positions(capture_folder.name, drone_data_list)
            
            # Fuse point clouds from all drones
            print("\nFusing point clouds...")
            new_points, new_colors = self.fuse_point_clouds(drone_data_list, intrinsics)
            
            if len(new_points) == 0:
                print("[WARNING] No points generated from this capture")
                return
            
            # Merge with accumulated point cloud
            print("\nMerging with accumulated cloud...")
            merged_points, merged_colors = self.merge_with_accumulated(new_points, new_colors)
            
            # Update accumulated state
            self.accumulated_points = merged_points
            self.accumulated_colors = merged_colors
            
            # Save raw merged point cloud
            timestamp = capture_folder.name.replace("capture_", "")
            raw_filename = f"pointcloud_raw_{timestamp}.ply"
            print(f"\nSaving raw point cloud...")
            self.save_point_cloud(merged_points, merged_colors, raw_filename)
            
            # Downsampling disabled - not used in pipeline
            # print(f"\nDownsampling to {self.TARGET_DOWNSAMPLE_SIZE} points...")
            # down_points, down_colors = self.downsample_point_cloud(
            #     merged_points, merged_colors, self.TARGET_DOWNSAMPLE_SIZE
            # )
            # 
            # down_filename = f"pointcloud_downsampled_{timestamp}.ply"
            # print(f"Saving downsampled point cloud...")
            # self.save_point_cloud(down_points, down_colors, down_filename)
            
            print(f"\n[OK] Processing complete!")
            print(f"  Raw: {len(merged_points)} points")
            # print(f"  Downsampled: {len(down_points)} points")
            
        except Exception as e:
            print(f"\n[ERROR] Error processing {capture_folder.name}: {e}")
            print(f"   This capture will be skipped.")
            # Keep it marked as processed to avoid infinite retry loop
            import traceback
            traceback.print_exc()
    
    def run(self, poll_interval: float = 2.0):
        """Main processing loop - polls for new captures."""
        print(f"\n{'='*60}")
        print("Swarm Point Cloud Builder - Running")
        print(f"{'='*60}")
        print(f"Capture directory: {self.capture_dir}")
        print(f"Output directory: {self.output_dir}")
        print(f"Poll interval: {poll_interval}s")
        print(f"\nWaiting for captures from Unity...")
        print("(Press Ctrl+C to stop, or Unity will signal when done)\n")
        
        # Path to done file that Unity writes when experiment ends
        done_file = os.path.join(os.path.dirname(self.capture_dir), "../../swarm_done.txt")
        done_file = os.path.normpath(done_file)
        
        try:
            while True:
                # Check if Unity signaled completion
                if os.path.exists(done_file):
                    # Read completion info
                    try:
                        with open(done_file, 'r') as f:
                            content = f.read().strip()
                        
                        if content.startswith('done,'):
                            completion_time = float(content.split(',')[1])
                            print(f"\n[OK] Swarm converged in {completion_time:.2f}s - shutting down...")
                        elif content.startswith('timeout,'):
                            timeout_val = float(content.split(',')[1])
                            print(f"\n[WARNING] Timeout reached ({timeout_val:.0f}s) - shutting down...")
                        else:
                            print("\n[OK] Unity experiment completed - shutting down...")
                    except:
                        print("\n[OK] Unity experiment completed - shutting down...")
                    
                    print(f"Processed {len(self.processed_captures)} captures total")
                    # Don't delete done file - let launcher read it first
                    
                    # Close CSV file
                    if self.csv_file:
                        self.csv_file.close()
                        print("[CSV] Logging closed")
                    
                    break
                
                # Find new captures
                new_captures = self.find_new_captures()
                
                # Process each new capture
                for capture_folder in new_captures:
                    self.process_capture(capture_folder)
                
                # Wait before next poll
                time.sleep(poll_interval)
                
        except KeyboardInterrupt:
            print("\n\nStopping...")
            print(f"Processed {len(self.processed_captures)} captures total")
            
            # Close CSV file
            if self.csv_file:
                self.csv_file.close()
                print("[CSV] Logging closed")
    
    def process_all_captures(self):
        """Batch process all existing captures (no polling, process once and exit)."""
        print(f"\n=== Batch Processing Mode (Interval={self.capture_interval}) ===")
        print(f"Capture directory: {self.capture_dir}")
        print(f"Output directory: {self.output_dir}")
        
        # Find all captures to process
        all_captures = self.find_new_captures()
        
        if not all_captures:
            print("No captures found to process.")
            return
        
        print(f"Found {len(all_captures)} captures to process\n")
        
        # Process each capture
        for capture_folder in all_captures:
            self.process_capture(capture_folder)
        
        # Close CSV file
        if self.csv_file:
            self.csv_file.close()
            print("[CSV] Logging closed")
        
        print(f"\n=== Batch Processing Complete ===")
        print(f"Processed {len(self.processed_captures)} captures total")


def main():
    parser = argparse.ArgumentParser(description="Swarm Point Cloud Builder")
    parser.add_argument("--capture-dir", type=str, 
                       default="../../ProcessedImages/SwarmCapture",
                       help="Directory where Unity saves captures")
    parser.add_argument("--output-dir", type=str,
                       default="../../ProcessedImages/SwarmPointClouds", 
                       help="Directory to save point clouds")
    parser.add_argument("--poll-interval", type=float, default=2.0,
                       help="Polling interval in seconds (ignored in batch mode)")
    parser.add_argument("--sam-model", type=str, default="vit_h",
                       choices=["vit_b", "vit_l", "vit_h"],
                       help="SAM model type")
    parser.add_argument("--capture-interval", type=int, default=1,
                       help="Process every Nth capture (1=all, 2=every 2nd, 3=every 3rd)")
    parser.add_argument("--batch-mode", action="store_true",
                       help="Process all existing captures once and exit (no polling)")
    
    args = parser.parse_args()
    
    # Create builder
    builder = SwarmPointCloudBuilder(
        capture_dir=args.capture_dir,
        output_dir=args.output_dir,
        sam_model_type=args.sam_model,
        capture_interval=args.capture_interval
    )
    
    # Initialize
    if not builder.initialize():
        print("Failed to initialize. Exiting.")
        return
    
    # Run in batch mode or polling mode
    if args.batch_mode:
        builder.process_all_captures()
    else:
        builder.run(poll_interval=args.poll_interval)


if __name__ == "__main__":
    main()
