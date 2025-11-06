"""
NBVPointCloudProcessor.py - MAP-NBV Implementation for Drone Swarm 3D Reconstruction

This script implements the complete MAP-NBV pipeline:
1. Reads RGB + Depth images from shared memory (Unity)
2. Segments RGB with SAM (Segment Anything Model)
3. Masks depth with segmentation
4. Converts to 3D point clouds using pinhole camera model
5. Transforms to global frame using drone poses
6. Fuses into unified global point cloud
7. Saves point cloud visualization (.ply file)
8. Sends random movement commands back to Unity (for now)

Usage:
    python NBVPointCloudProcessor.py

Architecture:
- Reads from 3 shared memory maps (images+depth, intrinsics, poses)
- Writes to 1 shared memory map (commands)
- Generates point cloud files in ProcessedImages folder
- Uses GPU-accelerated SAM for segmentation
"""

import numpy as np
import cv2
import mmap
import struct
import time
import os
from dataclasses import dataclass
from typing import List, Tuple, Optional
import random

# SAM imports
import torch
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

# Point cloud handling
try:
    import open3d as o3d
    HAS_OPEN3D = True
except ImportError:
    print("⚠️  Open3D not installed. Install with: pip install open3d")
    print("   Point clouds will be saved as PLY text files (can be viewed in MeshLab/CloudCompare)")
    HAS_OPEN3D = False

@dataclass
class CameraIntrinsics:
    """Camera intrinsic parameters"""
    fx: float  # Focal length x (pixels)
    fy: float  # Focal length y (pixels)
    cx: float  # Principal point x (pixels)
    cy: float  # Principal point y (pixels)
    width: int
    height: int

@dataclass
class DronePose:
    """Drone pose in world coordinates"""
    position: np.ndarray  # (3,) array: [x, y, z]
    quaternion: np.ndarray  # (4,) array: [x, y, z, w]

@dataclass
class DroneData:
    """Complete data for one drone"""
    drone_id: int
    rgb_image: np.ndarray  # (H, W, 3)
    depth_image: np.ndarray  # (H, W)
    pose: DronePose

class NBVPointCloudProcessor:
    """Main processor for MAP-NBV point cloud reconstruction"""
    
    def __init__(self,
                 processing_interval: float = 3.0,
                 debug: bool = True):
        """
        Initialize the point cloud processor
        
        Args:
            processing_interval: How often to process (seconds)
            debug: Enable debug logging
        """
        self.processing_interval = processing_interval
        self.debug = debug
        
        # Shared memory names
        self.image_memory_name = "NBVImageDepthMemory"
        self.intrinsics_memory_name = "NBVCameraIntrinsics"
        self.pose_memory_name = "NBVDronePoses"
        self.command_memory_name = "NBVCommandMemory"
        
        # Output settings
        self.output_folder = "../../ProcessedImages/PointClouds"
        os.makedirs(self.output_folder, exist_ok=True)
        
        # SAM settings
        self.sam_model_path = "../../segmentAnything/sam_vit_b_01ec64.pth"
        self.min_building_area = 1000  # pixels (reduced from 7500 for smaller images)
        
        # Memory objects
        self.image_mmf: Optional[mmap.mmap] = None
        self.intrinsics_mmf: Optional[mmap.mmap] = None
        self.pose_mmf: Optional[mmap.mmap] = None
        self.command_mmf: Optional[mmap.mmap] = None
        
        # SAM model
        self.mask_generator: Optional[SamAutomaticMaskGenerator] = None
        self.device = None
        
        # Processing state
        self.running = False
        self.last_processed_time = 0
        self.frames_processed = 0
        
        # Camera intrinsics (loaded from Unity)
        self.intrinsics: Optional[CameraIntrinsics] = None
        
    def initialize(self) -> bool:
        """Initialize shared memory and SAM model"""
        try:
            print("🚀 Initializing NBVPointCloudProcessor...")
            
            # Load SAM model
            self.load_sam_model()
            
            # Initialize shared memory (we'll determine sizes dynamically)
            print("✅ NBVPointCloudProcessor initialized successfully")
            print(f"   Processing interval: {self.processing_interval}s")
            print(f"   Output folder: {self.output_folder}")
            print(f"   Device: {self.device}")
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to initialize: {e}")
            return False
    
    def load_sam_model(self):
        """Load SAM model onto GPU"""
        print("🔧 Loading SAM model...")
        start_time = time.time()
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load model
        sam = sam_model_registry["vit_b"](checkpoint=self.sam_model_path)
        sam.to(self.device)
        
        # Create mask generator
        self.mask_generator = SamAutomaticMaskGenerator(
            sam,
            points_per_side=16,
            pred_iou_thresh=0.90,
            stability_score_thresh=0.96,
            crop_n_layers=0,
            crop_n_points_downscale_factor=1,
            min_mask_region_area=2000,
        )
        
        load_time = time.time() - start_time
        print(f"✅ SAM model loaded in {load_time:.2f}s on {self.device}")
    
    def read_camera_intrinsics(self) -> Optional[CameraIntrinsics]:
        """Read camera intrinsics from shared memory"""
        try:
            # Open shared memory for intrinsics
            intrinsics_size = 24  # 6 floats
            self.intrinsics_mmf = mmap.mmap(-1, intrinsics_size, self.intrinsics_memory_name)
            
            # Read intrinsics
            self.intrinsics_mmf.seek(0)
            data = self.intrinsics_mmf.read(24)
            fx, fy, cx, cy, width, height = struct.unpack('ffffff', data)
            
            intrinsics = CameraIntrinsics(fx, fy, cx, cy, int(width), int(height))
            
            if self.debug:
                print(f"📷 Camera Intrinsics:")
                print(f"   fx={fx:.2f}, fy={fy:.2f}")
                print(f"   cx={cx:.2f}, cy={cy:.2f}")
                print(f"   Resolution: {int(width)}x{int(height)}")
            
            return intrinsics
            
        except Exception as e:
            if self.debug:
                print(f"⚠️  Failed to read intrinsics: {e}")
            return None
    
    def read_drone_data_from_memory(self) -> Optional[List[DroneData]]:
        """
        Read all drone data from shared memory
        Returns list of DroneData objects
        """
        try:
            # First, open a small memory map to read header
            header_size = 16  # flag + drone_count + width + height
            temp_mmf = mmap.mmap(-1, header_size, self.image_memory_name)
            
            # Read header
            temp_mmf.seek(0)
            flag = struct.unpack('i', temp_mmf.read(4))[0]
            
            if flag != 2:  # Not ready
                temp_mmf.close()
                return None
            
            drone_count = struct.unpack('i', temp_mmf.read(4))[0]
            width = struct.unpack('i', temp_mmf.read(4))[0]
            height = struct.unpack('i', temp_mmf.read(4))[0]
            temp_mmf.close()
            
            # Calculate total memory size needed
            rgb_size = width * height * 3
            depth_size = width * height * 4  # float32
            per_drone_size = rgb_size + depth_size
            total_size = 16 + (drone_count * per_drone_size)
            
            # Now open full memory map
            self.image_mmf = mmap.mmap(-1, total_size, self.image_memory_name)
            
            # Read all drone data
            drone_data_list = []
            
            for drone_id in range(drone_count):
                # Calculate offsets
                rgb_offset = 16 + (drone_id * per_drone_size)
                depth_offset = rgb_offset + rgb_size
                
                # Read RGB
                self.image_mmf.seek(rgb_offset)
                rgb_bytes = self.image_mmf.read(rgb_size)
                rgb_image = np.frombuffer(rgb_bytes, dtype=np.uint8).reshape((height, width, 3))
                rgb_image = cv2.flip(rgb_image, 0)  # Unity coordinate flip
                
                # Read Depth
                self.image_mmf.seek(depth_offset)
                depth_bytes = self.image_mmf.read(depth_size)
                depth_image = np.frombuffer(depth_bytes, dtype=np.float32).reshape((height, width))
                depth_image = cv2.flip(depth_image, 0)  # Unity coordinate flip
                
                # Debug depth data on first read
                if self.debug and drone_id == 0 and len(drone_data_list) == 0:
                    print(f"   Depth check (Drone 0): min={depth_image.min():.3f}, max={depth_image.max():.3f}, mean={depth_image.mean():.3f}")
                
                # Read pose (from separate memory)
                pose = self.read_drone_pose(drone_id, drone_count)
                
                drone_data = DroneData(
                    drone_id=drone_id,
                    rgb_image=rgb_image,
                    depth_image=depth_image,
                    pose=pose
                )
                
                drone_data_list.append(drone_data)
            
            if self.debug:
                print(f"📥 Read data for {len(drone_data_list)} drones")
                print(f"   Image size: {width}x{height}")
            
            # IMPORTANT: Reset flag to 0 to tell Unity we're done reading
            self.image_mmf.seek(0)
            
            self.image_mmf.write(struct.pack('i', 0))
            
            return drone_data_list
            
        except Exception as e:
            if self.debug:
                print(f"⚠️  Error reading drone data: {e}")
            return None
    
    def read_drone_pose(self, drone_id: int, total_drones: int) -> DronePose:
        """Read pose for specific drone"""
        try:
            # Open pose memory if not already open
            pose_size = 4 + (total_drones * 28)  # count + (pos + quat) per drone
            if self.pose_mmf is None:
                self.pose_mmf = mmap.mmap(-1, pose_size, self.pose_memory_name)
            
            # Calculate offset for this drone
            offset = 4 + (drone_id * 28)
            
            # Read position (12 bytes)
            self.pose_mmf.seek(offset)
            pos_data = self.pose_mmf.read(12)
            px, py, pz = struct.unpack('fff', pos_data)
            
            # Read quaternion (16 bytes)
            quat_data = self.pose_mmf.read(16)
            qx, qy, qz, qw = struct.unpack('ffff', quat_data)
            
            return DronePose(
                position=np.array([px, py, pz]),
                quaternion=np.array([qx, qy, qz, qw])
            )
            
        except Exception as e:
            if self.debug:
                print(f"⚠️  Failed to read pose for drone {drone_id}: {e}")
            # Return identity pose
            return DronePose(
                position=np.zeros(3),
                quaternion=np.array([0, 0, 0, 1])
            )
    
    def segment_rgb_image(self, rgb_image: np.ndarray, drone_id: int = 0) -> Optional[np.ndarray]:
        """
        Segment RGB image with SAM and return binary mask
        Returns mask of largest center building segment
        """
        try:
            # Save debug image to see what camera sees
            if self.debug and self.frames_processed < 3:  # Only save first 3 frames
                debug_path = os.path.join(self.output_folder, f"debug_drone{drone_id}_frame{self.frames_processed:04d}.png")
                cv2.imwrite(debug_path, cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR))
            
            # Generate all segments
            masks = self.mask_generator.generate(rgb_image)
            
            if len(masks) == 0:
                if self.debug:
                    print(f"      ⚠️  SAM found 0 segments in image")
                return None
            
            if self.debug:
                print(f"      🔍 SAM found {len(masks)} total segments")
            
            # Find center building (same logic as SegAny_parallel.py)
            height, width = rgb_image.shape[:2]
            center_x, center_y = width // 2, height // 2
            
            # Find segments containing center point with minimum area
            center_segments = []
            for mask_info in masks:
                mask = mask_info['segmentation']
                area = mask_info['area']
                
                # Check if center point is in mask and meets area requirement
                if (0 <= center_y < mask.shape[0] and 0 <= center_x < mask.shape[1] and
                    mask[center_y, center_x] and area >= self.min_building_area):
                    center_segments.append(mask_info)
            
            if not center_segments:
                if self.debug:
                    print(f"      ⚠️  No segments contain center point or meet area threshold (min: {self.min_building_area} pixels)")
                return None
            
            # Get best segment
            best_segment = max(center_segments,
                             key=lambda x: x['stability_score'] * min(x['area']/10000, 10))
            
            binary_mask = best_segment['segmentation'].astype(np.float32)
            
            return binary_mask
            
        except Exception as e:
            if self.debug:
                print(f"⚠️  Segmentation failed: {e}")
            return None
    
    def depth_to_point_cloud(self, 
                            depth_image: np.ndarray,
                            segmentation_mask: Optional[np.ndarray],
                            rgb_image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert segmented depth to 3D point cloud using pinhole camera model
        
        Returns:
            points: (N, 3) array of 3D points in camera frame
            colors: (N, 3) array of RGB colors (0-1 range)
        """
        if self.intrinsics is None:
            raise ValueError("Camera intrinsics not loaded!")
        
        height, width = depth_image.shape
        
        # Apply segmentation mask if provided
        if segmentation_mask is not None:
            depth_masked = depth_image * segmentation_mask
        else:
            depth_masked = depth_image
        
        # Create pixel coordinate grids
        v, u = np.mgrid[0:height, 0:width]
        
        # Get valid depth points (non-zero and not NaN)
        valid_mask = (depth_masked > 0) & ~np.isnan(depth_masked)
        
        # Extract valid pixels
        u_valid = u[valid_mask]
        v_valid = v[valid_mask]
        depth_valid = depth_masked[valid_mask]
        
        # Pinhole camera model: X = (u - cx) * Z / fx
        X_cam = (u_valid - self.intrinsics.cx) * depth_valid / self.intrinsics.fx
        Y_cam = (v_valid - self.intrinsics.cy) * depth_valid / self.intrinsics.fy
        Z_cam = depth_valid
        
        # Stack into point cloud
        points = np.stack([X_cam, Y_cam, Z_cam], axis=-1)
        
        # Get corresponding colors from RGB image
        rgb_valid = rgb_image[valid_mask]
        colors = rgb_valid.astype(np.float32) / 255.0  # Normalize to 0-1
        
        return points, colors
    
    def quaternion_to_rotation_matrix(self, q: np.ndarray) -> np.ndarray:
        """Convert quaternion to 3x3 rotation matrix"""
        qx, qy, qz, qw = q
        
        R = np.array([
            [1 - 2*(qy**2 + qz**2),     2*(qx*qy - qw*qz),     2*(qx*qz + qw*qy)],
            [    2*(qx*qy + qw*qz), 1 - 2*(qx**2 + qz**2),     2*(qy*qz - qw*qx)],
            [    2*(qx*qz - qw*qy),     2*(qy*qz + qw*qx), 1 - 2*(qx**2 + qy**2)]
        ])
        
        return R
    
    def transform_to_global_frame(self, 
                                 points_local: np.ndarray,
                                 pose: DronePose) -> np.ndarray:
        """
        Transform points from camera frame to global world frame
        
        Args:
            points_local: (N, 3) points in camera frame
            pose: Drone pose in world
            
        Returns:
            points_global: (N, 3) points in world frame
        """
        # Create rotation matrix from quaternion
        R = self.quaternion_to_rotation_matrix(pose.quaternion)
        
        # Create 4x4 transformation matrix
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = pose.position
        
        # Convert points to homogeneous coordinates
        N = points_local.shape[0]
        points_homo = np.hstack([points_local, np.ones((N, 1))])
        
        # Apply transformation
        points_global_homo = (T @ points_homo.T).T
        points_global = points_global_homo[:, :3]
        
        return points_global
    
    def fuse_point_clouds(self, 
                         drone_data_list: List[DroneData]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Process all drones and fuse into global point cloud
        
        Returns:
            global_points: (N, 3) fused point cloud
            global_colors: (N, 3) corresponding colors
        """
        all_points = []
        all_colors = []
        
        for drone_data in drone_data_list:
            if self.debug:
                print(f"\n🔄 Processing Drone {drone_data.drone_id}...")
            
            # Segment RGB
            segment_start = time.time()
            mask = self.segment_rgb_image(drone_data.rgb_image, drone_data.drone_id)
            segment_time = time.time() - segment_start
            
            if mask is None:
                if self.debug:
                    print(f"   ❌ No building segment found ({segment_time:.2f}s)")
                continue
            
            if self.debug:
                building_area = np.sum(mask)
                print(f"   ✅ Segmented building: {building_area:,.0f} pixels ({segment_time:.2f}s)")
            
            # Convert to 3D point cloud
            pc_start = time.time()
            points_local, colors = self.depth_to_point_cloud(
                drone_data.depth_image,
                mask,
                drone_data.rgb_image
            )
            pc_time = time.time() - pc_start
            
            if self.debug:
                print(f"   🎯 Generated {len(points_local):,} 3D points ({pc_time:.2f}s)")
                if len(points_local) == 0:
                    # Debug depth data
                    depth_min = np.min(drone_data.depth_image)
                    depth_max = np.max(drone_data.depth_image)
                    depth_mean = np.mean(drone_data.depth_image)
                    valid_depth = np.sum((drone_data.depth_image > 0) & ~np.isnan(drone_data.depth_image))
                    print(f"      ⚠️  Depth data issue:")
                    print(f"         Min: {depth_min:.3f}, Max: {depth_max:.3f}, Mean: {depth_mean:.3f}")
                    print(f"         Valid pixels: {valid_depth}/{drone_data.depth_image.size}")
                    print(f"         Masked pixels: {np.sum(mask):.0f}")
            
            # Transform to global frame
            transform_start = time.time()
            points_global = self.transform_to_global_frame(points_local, drone_data.pose)
            transform_time = time.time() - transform_start
            
            if self.debug:
                print(f"   🌍 Transformed to global frame ({transform_time:.2f}s)")
            
            all_points.append(points_global)
            all_colors.append(colors)
        
        if len(all_points) == 0:
            if self.debug:
                print(f"\n⚠️  No point clouds generated (all drones had 0 points)")
            return np.array([]), np.array([])
        
        # Concatenate all point clouds
        global_points = np.vstack(all_points)
        global_colors = np.vstack(all_colors)
        
        if self.debug:
            print(f"\n📊 Global Point Cloud:")
            print(f"   Total points: {len(global_points):,}")
            if len(global_points) > 0:
                print(f"   Bounds: X=[{global_points[:,0].min():.2f}, {global_points[:,0].max():.2f}]")
                print(f"           Y=[{global_points[:,1].min():.2f}, {global_points[:,1].max():.2f}]")
                print(f"           Z=[{global_points[:,2].min():.2f}, {global_points[:,2].max():.2f}]")
        
        return global_points, global_colors
    
    def save_point_cloud(self, 
                        points: np.ndarray,
                        colors: np.ndarray,
                        frame_id: int):
        """Save point cloud to PLY file"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"pointcloud_frame{frame_id:04d}_{timestamp}.ply"
        filepath = os.path.join(self.output_folder, filename)
        
        try:
            if HAS_OPEN3D:
                # Use Open3D (better performance, binary format)
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(points)
                pcd.colors = o3d.utility.Vector3dVector(colors)
                o3d.io.write_point_cloud(filepath, pcd)
            else:
                # Fallback: Write ASCII PLY
                self.save_ply_ascii(filepath, points, colors)
            
            if self.debug:
                print(f"💾 Saved point cloud: {filename}")
                print(f"   Location: {filepath}")
            
        except Exception as e:
            print(f"❌ Failed to save point cloud: {e}")
    
    def save_ply_ascii(self, filepath: str, points: np.ndarray, colors: np.ndarray):
        """Save point cloud as ASCII PLY (fallback without Open3D)"""
        with open(filepath, 'w') as f:
            # Write header
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
            
            # Write points
            for i in range(len(points)):
                x, y, z = points[i]
                r, g, b = (colors[i] * 255).astype(np.uint8)
                f.write(f"{x} {y} {z} {r} {g} {b}\n")
    
    def send_random_commands(self, drone_count: int) -> bool:
        """Send random movement commands to Unity (placeholder for NBV planning)"""
        try:
            # Open command memory
            command_size = 4 + (drone_count * 12)  # flag + commands
            if self.command_mmf is None:
                self.command_mmf = mmap.mmap(-1, command_size, self.command_memory_name)
            
            # Check if Unity is ready
            self.command_mmf.seek(0)
            flag = struct.unpack('i', self.command_mmf.read(4))[0]
            
            if flag != 0:
                return False  # Unity still processing
            
            # Generate random commands
            commands = []
            for drone_id in range(drone_count):
                movement_scale = (drone_id + 1) * 5.0
                x = random.uniform(-2.0, 2.0) * movement_scale
                y = random.uniform(-2.0, 2.0) * movement_scale
                z = random.uniform(-2.0, 2.0) * movement_scale
                commands.append((x, y, z))
            
            # Write commands
            self.command_mmf.seek(4)
            for x, y, z in commands:
                self.command_mmf.write(struct.pack('fff', x, y, z))
            
            # Set flag
            self.command_mmf.seek(0)
            self.command_mmf.write(struct.pack('i', 2))
            
            if self.debug:
                print(f"📤 Sent {drone_count} random commands to Unity")
            
            return True
            
        except Exception as e:
            if self.debug:
                print(f"⚠️  Failed to send commands: {e}")
            return False
    
    def processing_loop(self):
        """Main processing loop"""
        self.running = True
        
        print(f"🚀 Point Cloud Processing started (interval: {self.processing_interval}s)")
        
        while self.running:
            try:
                current_time = time.time()
                
                # Process at specified interval
                if current_time - self.last_processed_time < self.processing_interval:
                    time.sleep(0.1)
                    continue
                
                print(f"\n{'='*60}")
                print(f"🔄 Processing cycle {self.frames_processed + 1} at {current_time:.1f}")
                print(f"{'='*60}")
                
                cycle_start = time.time()
                
                # Read camera intrinsics (once)
                if self.intrinsics is None:
                    self.intrinsics = self.read_camera_intrinsics()
                    if self.intrinsics is None:
                        print("⏳ Waiting for camera intrinsics from Unity...")
                        time.sleep(1.0)
                        continue
                
                # Read all drone data
                drone_data_list = self.read_drone_data_from_memory()
                
                if drone_data_list is None or len(drone_data_list) == 0:
                    print("⏳ Waiting for drone data from Unity...")
                    time.sleep(1.0)
                    continue
                
                # Process and fuse point clouds
                fusion_start = time.time()
                global_points, global_colors = self.fuse_point_clouds(drone_data_list)
                fusion_time = time.time() - fusion_start
                
                if len(global_points) > 0:
                    print(f"\n✅ Point cloud fusion completed in {fusion_time:.2f}s")
                    
                    # Save point cloud
                    self.save_point_cloud(global_points, global_colors, self.frames_processed)
                    
                    # Send random commands
                    self.send_random_commands(len(drone_data_list))
                    
                    self.frames_processed += 1
                else:
                    print("❌ No valid point cloud generated")
                
                cycle_time = time.time() - cycle_start
                print(f"\n⏱️  Total cycle time: {cycle_time:.2f}s")
                
                self.last_processed_time = current_time
                
            except KeyboardInterrupt:
                raise
            except Exception as e:
                print(f"⚠️  Error in processing loop: {e}")
                import traceback
                traceback.print_exc()
                time.sleep(1.0)
    
    def start(self):
        """Start processing"""
        if not self.initialize():
            return False
        
        try:
            self.processing_loop()
        except KeyboardInterrupt:
            print("\n⏹️  Stopping processor...")
            self.stop()
        
        return True
    
    def stop(self):
        """Stop processing and cleanup"""
        self.running = False
        self.cleanup()
    
    def cleanup(self):
        """Clean up resources"""
        if self.image_mmf:
            self.image_mmf.close()
        if self.intrinsics_mmf:
            self.intrinsics_mmf.close()
        if self.pose_mmf:
            self.pose_mmf.close()
        if self.command_mmf:
            self.command_mmf.close()
        
        print("🧹 NBVPointCloudProcessor cleaned up")

def main():
    """Main function"""
    print("="*60)
    print("🎮 NBV Point Cloud Processor - MAP-NBV Implementation")
    print("="*60)
    
    processor = NBVPointCloudProcessor(
        processing_interval=3.0,
        debug=True
    )
    
    print("\n📝 Make sure:")
    print("   1. Unity is running with updated NBVImageCapture")
    print("   2. Drones have RGB + Depth cameras configured")
    print("   3. Shared memory is properly set up")
    print()
    
    processor.start()

if __name__ == "__main__":
    main()
