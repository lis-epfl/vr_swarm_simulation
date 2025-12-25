import os
import time
import mmap
import struct
import numpy as np
import cv2
import torch
import sys
from typing import List, Optional, Tuple
from scipy.spatial.transform import Rotation
from dataclasses import dataclass
import ctypes
from ctypes import wintypes
import merge_point_clouds

# Check and create D:/ drive directories if they don't exist
D_DRIVE_BASE = r"D:\advaith\unity-run-files"
if os.path.exists("D:\\"):
    os.makedirs(os.path.join(D_DRIVE_BASE, "FinalPointClouds_NBV"), exist_ok=True)
    os.makedirs(os.path.join(D_DRIVE_BASE, "ProcessedImages", "PointClouds"), exist_ok=True)
    print(f"✓ D:/ drive directories ready at {D_DRIVE_BASE}")

try:
    import open3d as o3d
    HAS_OPEN3D = True
except ImportError:
    HAS_OPEN3D = False
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

#import candidate_positions
# sys.path.append("/Users/advaithsriram/pointr-nbv/scripts") #MAC
sys.path.append(r"C:/Users/sriram/PoinTr/scripts") #WINDOWS

from candidate_positions import main as candidate_positions_main
# sys.path.remove("/Users/advaithsriram/pointr-nbv/scripts") #MAC
sys.path.remove(r"C:/Users/sriram/PoinTr/scripts") #WINDOWS

DOWN_SAMPLE_SIZE = 2000

VISUALIZE_BOOL = False

# Windows API for shared memory
kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)

FILE_MAP_ALL_ACCESS = 0xF001F
INVALID_HANDLE_VALUE = -1

OpenFileMapping = kernel32.OpenFileMappingW
OpenFileMapping.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.LPCWSTR]
OpenFileMapping.restype = wintypes.HANDLE

MapViewOfFile = kernel32.MapViewOfFile
MapViewOfFile.argtypes = [wintypes.HANDLE, wintypes.DWORD, wintypes.DWORD, wintypes.DWORD, ctypes.c_size_t]
MapViewOfFile.restype = wintypes.LPVOID

UnmapViewOfFile = kernel32.UnmapViewOfFile
UnmapViewOfFile.argtypes = [wintypes.LPCVOID]
UnmapViewOfFile.restype = wintypes.BOOL

CloseHandle = kernel32.CloseHandle
CloseHandle.argtypes = [wintypes.HANDLE]
CloseHandle.restype = wintypes.BOOL

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

class MAP_NBV_Trial:
    MIN_DISTANCE_FROM_DRONE = 1.1  # meters
    timestamp = None

    def send_nbv_commands(self, selected_nbvs, flag=1):
        """Send selected NBV positions to Unity via shared memory using Windows API.
        
        Args:
            selected_nbvs: List of (position, euler) tuples for each drone
            flag: Command flag (1 = new NBV commands)
        
        Returns:
            True if commands were sent successfully, False otherwise
        """
        try:
            drone_count = len(selected_nbvs)
            # Unity creates memory with size for 10 drones (124 bytes)
            command_size = 4 + (10 * 12)  # flag + (10 drones × 12 bytes per command)
            
            # OPEN existing shared memory created by Unity using Windows API
            if self.command_mmf is None:
                try:
                    # Open the existing shared memory (created by Unity)
                    handle = OpenFileMapping(FILE_MAP_ALL_ACCESS, False, self.command_memory_name)
                    
                    if not handle or handle == INVALID_HANDLE_VALUE:
                        error = ctypes.get_last_error()
                        print(f"✗ Failed to open command shared memory '{self.command_memory_name}' (error {error})")
                        print(f"   Unity may not have created it yet. Make sure Unity is running first!")
                        return False
                    
                    # Map the memory into our process
                    ptr = MapViewOfFile(handle, FILE_MAP_ALL_ACCESS, 0, 0, command_size)
                    
                    if not ptr:
                        error = ctypes.get_last_error()
                        CloseHandle(handle)
                        print(f"✗ Failed to map command shared memory view (error {error})")
                        return False
                    
                    # Store handle and pointer for later use
                    self.command_handle = handle
                    self.command_ptr = ptr
                    self.command_size = command_size
                    
                    print(f"✓ Opened existing command shared memory: {self.command_memory_name} ({command_size} bytes)")
                    print(f"   Handle: {handle}, Pointer: {ptr}")
                    
                except Exception as e:
                    print(f"✗ Exception opening command shared memory: {e}")
                    import traceback
                    traceback.print_exc()
                    return False
            
            # Read current flag from shared memory
            flag_bytes = ctypes.string_at(self.command_ptr, 4)
            current_flag = struct.unpack('i', flag_bytes)[0]
            
            if current_flag != 0:
                print(f"Unity not ready (flag={current_flag}), waiting...")
                return False
            
            # Write NBV positions for each drone
            offset = 4  # Start after flag
            for idx, (pos, euler) in enumerate(selected_nbvs):
                # Pack position as 3 floats
                pos_bytes = struct.pack('fff', pos[0], pos[1], pos[2])
                # Write to shared memory
                ctypes.memmove(self.command_ptr + offset, pos_bytes, 12)
                offset += 12
                print(f"  Drone {idx}: NBV target = ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})")
            
            # Fill remaining drone slots with zeros (if less than 10 drones)
            for idx in range(drone_count, 10):
                zero_bytes = struct.pack('fff', 0.0, 0.0, 0.0)
                ctypes.memmove(self.command_ptr + offset, zero_bytes, 12)
                offset += 12
            
            # Set flag to signal new commands (flag = 1)
            flag_bytes = struct.pack('i', flag)
            ctypes.memmove(self.command_ptr, flag_bytes, 4)
            
            # DEBUG: Read back what we just wrote
            verify_bytes = ctypes.string_at(self.command_ptr, 16)
            print(f"   DEBUG: Memory after write: {verify_bytes.hex()}")
            
            print(f"✓ Sent NBV commands (flag={flag}) for {drone_count} drones")
            return True
            
        except Exception as e:
            print(f"✗ Failed to send NBV commands: {e}")
            import traceback
            traceback.print_exc()
            return False

    def quaternion_to_euler(self, q):
        """Convert quaternion (x, y, z, w) to Euler angles (roll, pitch, yaw) in radians."""
        rotation = Rotation.from_quat(q)
        roll, pitch, yaw = rotation.as_euler('xyz', degrees=False)
        return [roll, pitch, yaw]
    
    def run_pointr_inference(self, downsampled_file):
        """Run PoinTr inference on the downsampled PLY file using subprocess."""
        import subprocess
        pointr_dir = "C:/Users/sriram/PoinTr" # WINDOWS
        inference_script = os.path.join(pointr_dir, "tools/inference.py")
        config_path = os.path.join(pointr_dir, "cfgs/ShapeNet55_models/PoinTr.yaml")
        checkpoint_path = os.path.join(pointr_dir, "models/checkpoint55.pth")
        abs_downsampled_file = os.path.abspath(downsampled_file)
        abs_out_pc_root = "C:/Users/sriram/vr_swarm_simulation/Assets/ProcessedImages/PointClouds"
        os.makedirs(abs_out_pc_root, exist_ok=True)
        cmd = [
            "python", # WINDOWS
            inference_script,
            config_path,
            checkpoint_path,
            "--pc", abs_downsampled_file,
            "--out_pc_root", abs_out_pc_root,
            "--save_ply"
        ]
        print(f"Running inference on {abs_downsampled_file} ...")
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=pointr_dir)
        print("PoinTr STDOUT:", result.stdout)
        print("PoinTr STDERR:", result.stderr)


    def random_downsample(self, points, colors, size=DOWN_SAMPLE_SIZE, seed=42):
        np.random.seed(seed)
        if len(points) <= size:
            return points, colors
        if not HAS_OPEN3D:
            return points, colors
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        ratio = size / len(points)
        downsampled = pcd.random_down_sample(ratio)
        down_points = np.asarray(downsampled.points)
        down_colors = np.asarray(downsampled.colors)
        return down_points, down_colors
    
    def __init__(self, processing_interval=3.0):
        self.processing_interval = processing_interval
        self.image_memory_name = "NBVImageDepthMemory"
        self.intrinsics_memory_name = "NBVCameraIntrinsics"
        self.pose_memory_name = "NBVDronePoses"
        # self.command_memory_name = "NBVCommandMemory"
        self.command_memory_name = "NBVCommandSharedMemory"
        
        # Use D:/ drive for output to save space on C:/
        self.output_folder = r"D:\advaith\unity-run-files\ProcessedImages\PointClouds"
        os.makedirs(self.output_folder, exist_ok=True)
        
        print(f"  Output folder: {os.path.abspath(self.output_folder)}")
        
        self.sam_model_type = "vit_h"
        
        # Try relative paths first, fallback to absolute
        relative_sam = {
            "vit_b": "../../segmentAnything/sam_vit_b_01ec64.pth",
            "vit_l": "../../segmentAnything/sam_vit_l_0b3195.pth",
            "vit_h": "../../segmentAnything/sam_vit_h_4b8939.pth",
        }
        absolute_sam = {
            "vit_b": r"C:\Users\sriram\vr_swarm_simulation\Assets\segmentAnything\sam_vit_b_01ec64.pth",
            "vit_l": r"C:\Users\sriram\vr_swarm_simulation\Assets\segmentAnything\sam_vit_l_0b3195.pth",
            "vit_h": r"C:\Users\sriram\vr_swarm_simulation\Assets\segmentAnything\sam_vit_h_4b8939.pth",
        }
        
        # Check if relative paths work
        if os.path.exists(relative_sam[self.sam_model_type]):
            self.sam_model_paths = relative_sam
        else:
            self.sam_model_paths = absolute_sam
        self.min_building_area = 1000
        self.image_mmf = None
        self.intrinsics_mmf = None
        self.pose_mmf = None
        self.command_mmf = None  # Legacy mmap (not used anymore)
        # Windows API shared memory for commands
        self.command_handle = None
        self.command_ptr = None
        self.command_size = 0
        self.mask_generator = None
        self.device = None
        self.intrinsics = None
        self.running = False
        self.last_processed_time = 0
        self.frames_processed = 0
        self.accumulated_points = None  # Store accumulated raw point cloud
        self.accumulated_colors = None
        self.previous_pcd = None  # Track accumulated point cloud
        
        # Read max iterations from config file if available
        self.max_iterations = self.read_max_iterations_from_config()
        

    def read_max_iterations_from_config(self):
        """Read max iterations from nbv_config.json."""
        # Try multiple paths
        possible_paths = [
            "nbv_config.json",
            "../../nbv_config.json",
            "../../../nbv_config.json",
            r"C:\Users\sriram\vr_swarm_simulation\nbv_config.json"
        ]
        
        for config_path in possible_paths:
            if os.path.exists(config_path):
                try:
                    import json
                    with open(config_path, 'r') as f:
                        config = json.load(f)
                        max_iter = config.get('maxIterations', 5)
                        print(f"✓ Max iterations from config: {max_iter} (from {config_path})")
                        return max_iter
                except Exception as e:
                    print(f"  Warning: Could not read config at {config_path}: {e}")
        
        print(f"  Warning: Config file not found, using default max_iterations=5")
        return 5  # Default

    def initialize(self):
        # Segmentation disabled - using depth values only
        # self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # checkpoint_path = self.sam_model_paths[self.sam_model_type]
        # sam = sam_model_registry[self.sam_model_type](checkpoint=checkpoint_path)
        # sam.to(self.device)
        # self.mask_generator = SamAutomaticMaskGenerator(
        #     sam,
        #     points_per_side=16,
        #     pred_iou_thresh=0.90,
        #     stability_score_thresh=0.96,
        #     crop_n_layers=0,
        #     crop_n_points_downscale_factor=1,
        #     min_mask_region_area=2000,
        # )
        print("  ✓ Segmentation disabled - using depth values only")
        return True

    def read_camera_intrinsics(self):
        intrinsics_size = 24
        self.intrinsics_mmf = mmap.mmap(-1, intrinsics_size, self.intrinsics_memory_name)
        self.intrinsics_mmf.seek(0)
        data = self.intrinsics_mmf.read(24)
        fx, fy, cx, cy, width, height = struct.unpack('ffffff', data)
        return CameraIntrinsics(fx, fy, cx, cy, int(width), int(height))

    def read_drone_data_from_memory(self):
        header_size = 16
        temp_mmf = mmap.mmap(-1, header_size, self.image_memory_name)
        temp_mmf.seek(0)
        flag = struct.unpack('i', temp_mmf.read(4))[0]
        if flag != 2:
            temp_mmf.close()
            return None
        drone_count = struct.unpack('i', temp_mmf.read(4))[0]
        width = struct.unpack('i', temp_mmf.read(4))[0]
        height = struct.unpack('i', temp_mmf.read(4))[0]
        temp_mmf.close()
        rgb_size = width * height * 3
        depth_size = width * height * 4
        per_drone_size = rgb_size + depth_size
        total_size = 16 + (drone_count * per_drone_size)
        self.image_mmf = mmap.mmap(-1, total_size, self.image_memory_name)
        drone_data_list = []
        for drone_id in range(drone_count):
            rgb_offset = 16 + (drone_id * per_drone_size)
            depth_offset = rgb_offset + rgb_size
            self.image_mmf.seek(rgb_offset)
            rgb_bytes = self.image_mmf.read(rgb_size)
            rgb_image = np.frombuffer(rgb_bytes, dtype=np.uint8).reshape((height, width, 3))
            rgb_image = cv2.flip(rgb_image, 0)
            self.image_mmf.seek(depth_offset)
            depth_bytes = self.image_mmf.read(depth_size)
            depth_image = np.frombuffer(depth_bytes, dtype=np.float32).reshape((height, width))
            depth_image = cv2.flip(depth_image, 0)
            pose = self.read_drone_pose(drone_id, drone_count)
            drone_data = DroneData(drone_id, rgb_image, depth_image, pose)
            drone_data_list.append(drone_data)
        self.image_mmf.seek(0)
        self.image_mmf.write(struct.pack('i', 0))
        return drone_data_list

    def read_drone_pose(self, drone_id, total_drones):
        pose_size = 4 + (total_drones * 28)
        if self.pose_mmf is None:
            self.pose_mmf = mmap.mmap(-1, pose_size, self.pose_memory_name)
        offset = 4 + (drone_id * 28)
        self.pose_mmf.seek(offset)
        pos_data = self.pose_mmf.read(12)
        px, py, pz = struct.unpack('fff', pos_data)
        quat_data = self.pose_mmf.read(16)
        qx, qy, qz, qw = struct.unpack('ffff', quat_data)
        return DronePose(np.array([px, py, pz]), np.array([qx, qy, qz, qw]))

    def segment_rgb_image(self, rgb_image):
        masks = self.mask_generator.generate(rgb_image)
        if len(masks) == 0:
            return None
        
        height, width = rgb_image.shape[:2]
        center_x, center_y = width // 2, height // 2
        
        # Define multiple sample points across a wider region (not just center)
        # This helps catch buildings that are off to the side
        sample_points = [
            (center_x, center_y),  # Center
            (center_x - width//6, center_y),  # Left
            (center_x + width//6, center_y),  # Right
            (center_x - width//4, center_y),  # Further left
            (center_x + width//4, center_y),  # Further right
            (center_x, center_y - height//8),  # Above center
            (center_x, center_y + height//8),  # Below center
            (center_x - width//8, center_y - height//8),  # Upper left
            (center_x + width//8, center_y - height//8),  # Upper right
        ]
        
        candidate_segments = []
        for mask_info in masks:
            mask = mask_info['segmentation']
            area = mask_info['area']
            
            # Skip small segments
            if area < self.min_building_area:
                continue
            
            # Filter out sky - more aggressive detection
            y_coords = np.where(mask)[0]
            x_coords = np.where(mask)[1]
            
            if len(y_coords) > 0:
                mean_y = np.mean(y_coords)
                max_y = np.max(y_coords)
                min_y = np.min(y_coords)
                y_span = max_y - min_y
                
                # Sky characteristics:
                # 1. Predominantly in upper portion (mean < 30% of height)
                # 2. OR extends to the very top edge (min_y < 10 pixels)
                # 3. AND spans a large vertical area (indicates it's not just a small object)
                is_in_upper_region = mean_y < height * 0.3
                touches_top_edge = min_y < 10
                large_vertical_span = y_span > height * 0.4
                
                if (is_in_upper_region and large_vertical_span) or (touches_top_edge and large_vertical_span):
                    continue
            
            # Filter out ground (typically in lower portion, large horizontal area)
            if len(y_coords) > 0 and len(x_coords) > 0:
                mean_y_ground = np.mean(y_coords)
                min_y_ground = np.min(y_coords)
                x_spread = np.max(x_coords) - np.min(x_coords)
                y_spread = np.max(y_coords) - np.min(y_coords)
                
                # Ground characteristics:
                # 1. In bottom half with very wide horizontal spread
                # 2. OR very flat (wide but not tall)
                is_in_bottom = mean_y_ground > height * 0.6
                spans_wide = x_spread > width * 0.65
                is_flat = y_spread < height * 0.2 and x_spread > width * 0.5
                
                if (is_in_bottom and spans_wide) or is_flat:
                    continue
            
            # Check if any sample point falls within this segment
            hits_sample_point = False
            for sx, sy in sample_points:
                if 0 <= sy < mask.shape[0] and 0 <= sx < mask.shape[1] and mask[sy, sx]:
                    hits_sample_point = True
                    break
            
            # Include segments that:
            # 1. Hit any sample point, OR
            # 2. Are large and stable (building might be off-center), OR
            # 3. Are in the middle vertical band (left-center-right) even if not hitting sample points
            in_middle_band = False
            if len(x_coords) > 0:
                centroid_x = np.mean(x_coords)
                # Middle 70% of image horizontally
                in_middle_band = width * 0.15 < centroid_x < width * 0.85
            
            is_large_stable = area > self.min_building_area * 2 and mask_info['stability_score'] > 0.95
            is_medium_stable_centered = area > self.min_building_area and mask_info['stability_score'] > 0.90 and in_middle_band
            
            if hits_sample_point or is_large_stable or is_medium_stable_centered:
                candidate_segments.append(mask_info)
        
        if not candidate_segments:
            return None
        
        # STEP 1: Filter by size - keep only large segments (likely buildings)
        # Calculate areas and find reasonable threshold
        areas = [seg['area'] for seg in candidate_segments]
        max_area = max(areas)
        
        # Keep segments that are at least 30% of the largest segment
        # This filters out small objects while keeping building-sized segments
        size_threshold = max_area * 0.3
        large_segments = [seg for seg in candidate_segments if seg['area'] >= size_threshold]
        
        if not large_segments:
            # If filtering is too aggressive, fall back to top 5 largest
            large_segments = sorted(candidate_segments, key=lambda x: x['area'], reverse=True)[:5]
        
        # STEP 2: Among large segments, find the one with centroid closest to vertical center
        # Calculate centroids for all large segments
        segments_with_centroids = []
        for seg in large_segments:
            mask = seg['segmentation']
            y_coords, x_coords = np.where(mask)
            
            if len(y_coords) == 0:
                continue
            
            centroid_x = np.mean(x_coords)
            centroid_y = np.mean(y_coords)
            
            # Calculate vertical position (0 = top, 1 = bottom)
            vertical_position = centroid_y / height
            
            # Distance from ideal vertical center (0.5 = perfect middle)
            # Buildings typically have centroids around 0.4-0.6
            vertical_distance = abs(vertical_position - 0.5)
            
            segments_with_centroids.append({
                'segment': seg,
                'centroid_x': centroid_x,
                'centroid_y': centroid_y,
                'vertical_position': vertical_position,
                'vertical_distance': vertical_distance,
                'area': seg['area']
            })
        
        if not segments_with_centroids:
            return None
        
        # Find segment with centroid closest to vertical middle
        best_candidate = min(segments_with_centroids, key=lambda x: x['vertical_distance'])
        best_segment = best_candidate['segment']
        
        # Optional: Log candidates for debugging
        if False:  # Set to True for debugging
            print(f"\nSegment selection (filtered to {len(large_segments)} large segments):")
            sorted_candidates = sorted(segments_with_centroids, key=lambda x: x['vertical_distance'])
            for i, cand in enumerate(sorted_candidates[:3]):
                print(f"  {i+1}. Area: {cand['area']}, "
                      f"Centroid: ({cand['centroid_x']:.0f}, {cand['centroid_y']:.0f}), "
                      f"Vertical pos: {cand['vertical_position']:.2f}, "
                      f"Dist from middle: {cand['vertical_distance']:.2f}")
        
        binary_mask = best_segment['segmentation'].astype(np.float32)
        return binary_mask

    def depth_to_point_cloud(self, depth_image, segmentation_mask, rgb_image):
        if self.intrinsics is None:
            raise ValueError("Camera intrinsics not loaded!")
        height, width = depth_image.shape
        if segmentation_mask is not None:
            depth_masked = depth_image * segmentation_mask
        else:
            depth_masked = depth_image
        v, u = np.mgrid[0:height, 0:width]
        valid_mask = (depth_masked > 0) & ~np.isnan(depth_masked)
        u_valid = u[valid_mask]
        v_valid = v[valid_mask]
        depth_valid = depth_masked[valid_mask]
        X_cam = (u_valid - self.intrinsics.cx) * depth_valid / self.intrinsics.fx
        Y_cam = -(v_valid - self.intrinsics.cy) * depth_valid / self.intrinsics.fy
        Z_cam = depth_valid
        points = np.stack([X_cam, Y_cam, Z_cam], axis=-1)
        rgb_valid = rgb_image[v_valid, u_valid, :]
        colors = rgb_valid.astype(np.float32) / 255.0
        return points, colors

    def transform_to_global_frame(self, points_local, pose):
        if points_local.shape[0] == 0:
            return points_local
        position = pose.position
        quaternion = pose.quaternion
        rotation = Rotation.from_quat(quaternion)
        transform_matrix = np.eye(4)
        transform_matrix[:3, :3] = rotation.as_matrix()
        transform_matrix[:3, 3] = position
        points_local_hom = np.hstack((points_local, np.ones((points_local.shape[0], 1))))
        points_global_hom = (transform_matrix @ points_local_hom.T).T
        points_global = points_global_hom[:, :3]
        return points_global

    def fuse_point_clouds(self, drone_data_list):
        all_points = []
        all_colors = []
        for drone_data in drone_data_list:
            # Segmentation disabled - use depth values only
            # mask = self.segment_rgb_image(drone_data.rgb_image)
            # if mask is None:
            #     continue
            
            # Pass None for mask to use all depth values
            points_local, colors = self.depth_to_point_cloud(
                drone_data.depth_image,
                None,  # No segmentation mask - use all depth values
                drone_data.rgb_image
            )
            
            # Skip if no points generated
            if len(points_local) == 0:
                continue
            
            # Filter out points close to the drone origin (likely drone mesh)
            mask_dist = np.abs(points_local[:, 2]) > self.MIN_DISTANCE_FROM_DRONE
            filtered_points = points_local[mask_dist]
            filtered_colors = colors[mask_dist]
            points_global = self.transform_to_global_frame(filtered_points, drone_data.pose)
            all_points.append(points_global)
            all_colors.append(filtered_colors)
        if len(all_points) == 0:
            return np.array([]), np.array([])
        global_points = np.vstack(all_points)
        global_colors = np.vstack(all_colors)
        return global_points, global_colors

    def save_point_cloud(self, points, colors, frame_id):
        self.timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        # Save RAW accumulated point cloud
        filename_raw = f"pointcloud_raw_frame{frame_id:04d}_{self.timestamp}.ply"
        filepath_raw = os.path.join(self.output_folder, filename_raw)
        # print(f"\nSaving raw accumulated point cloud: {len(points)} points")
        
        if HAS_OPEN3D:
            pcd_raw = o3d.geometry.PointCloud()
            pcd_raw.points = o3d.utility.Vector3dVector(points)
            pcd_raw.colors = o3d.utility.Vector3dVector(colors)
            o3d.io.write_point_cloud(filepath_raw, pcd_raw)
        else:
            with open(filepath_raw, 'w') as f:
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
        
        print(f"✓ Saved frame {frame_id}: {len(points)} points")

        # Downsample the accumulated point cloud for PoinTr processing
        # print(f"\nDownsampling to {DOWN_SAMPLE_SIZE} points for PoinTr...")
        down_points, down_colors = self.random_downsample(points, colors, size=DOWN_SAMPLE_SIZE, seed=42)
        filename_down = f"pointcloud_downsampled_frame{frame_id:04d}_{self.timestamp}.ply"
        filepath_down = os.path.join(self.output_folder, filename_down)

        if HAS_OPEN3D:
            pcd_down = o3d.geometry.PointCloud()
            pcd_down.points = o3d.utility.Vector3dVector(down_points)
            pcd_down.colors = o3d.utility.Vector3dVector(down_colors)
            o3d.io.write_point_cloud(filepath_down, pcd_down)
        else:
            with open(filepath_down, 'w') as f:
                f.write("ply\n")
                f.write("format ascii 1.0\n")
                f.write(f"element vertex {len(down_points)}\n")
                f.write("property float x\n")
                f.write("property float y\n")
                f.write("property float z\n")
                f.write("property uchar red\n")
                f.write("property uchar green\n")
                f.write("property uchar blue\n")
                f.write("end_header\n")
                for i in range(len(down_points)):
                    x, y, z = down_points[i]
                    r, g, b = (down_colors[i] * 255).astype(np.uint8)
                    f.write(f"{x} {y} {z} {r} {g} {b}\n")
        
        # print(f"  Saved downsampled: {filepath_down} ({len(down_points)} points)")

        # Run PoinTr inference on the downsampled file
        self.run_pointr_inference(filepath_down)

    def processing_loop(self):
        self.running = True
        while self.running:
            current_time = time.time()
            if current_time - self.last_processed_time < self.processing_interval:
                time.sleep(0.1)
                continue
            if self.intrinsics is None:
                self.intrinsics = self.read_camera_intrinsics()
                if self.intrinsics is None:
                    time.sleep(1.0)
                    continue
            drone_data_list = self.read_drone_data_from_memory()
            if drone_data_list is None or len(drone_data_list) == 0:
                time.sleep(1.0)
                continue

            # Debug: Print drone positions to verify they're updating
            print(f"\nDrone positions at capture:")
            for idx, drone_data in enumerate(drone_data_list):
                pos = drone_data.pose.position
                print(f"  Drone {idx}: ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})")
            
            # Fuse current drone images into a point cloud
            current_points, current_colors = self.fuse_point_clouds(drone_data_list)

            if len(current_points) > 0:
                # Merge with accumulated point cloud BEFORE downsampling
                if self.accumulated_points is not None:
                    prev_count = len(self.accumulated_points)
                    global_points = np.vstack([self.accumulated_points, current_points])
                    global_colors = np.vstack([self.accumulated_colors, current_colors])
                    print(f"Fusion: {prev_count} (prev) + {len(current_points)} (new) = {len(global_points)} (total)")
                else:
                    print(f"\nFirst capture - initializing accumulated cloud")
                    global_points = current_points
                    global_colors = current_colors
                
                # Update accumulated cloud
                self.accumulated_points = global_points
                self.accumulated_colors = global_colors
                
                # Save both raw and processed versions
                self.save_point_cloud(global_points, global_colors, self.frames_processed)
                
                # NEW: Save to FinalPointClouds after each iteration for progression tracking
                self.save_iteration_milestone(global_points, global_colors, self.frames_processed, len(drone_data_list))
                
                # Store current frame for file paths
                current_frame = self.frames_processed
                self.frames_processed += 1
                
                print(f"\n{'='*60}")
                print(f"Capture {self.frames_processed} completed")
                print(f"{'='*60}")
                
                # Check if max_iterations is 0 (no NBV movements at all)
                if self.max_iterations == 0:
                    print(f"\n{'='*60}")
                    print(f"🛑 MAX ITERATIONS = 0 (no NBV movements)")
                    print(f"{'='*60}")
                    print(f"Total captures: {self.frames_processed}")
                    print(f"Stopping NBV pipeline...")
                    
                    # Cleanup temporary files (milestones already saved)
                    self.cleanup_temporary_files()
                    
                    # Trigger Unity to stop Play mode
                    self.trigger_unity_stop()
                    
                    self.running = False
                    break
                
                # Check if we've already completed max_iterations NBV movements
                # frames_processed - 1 = completed NBV movements so far
                nbv_movements_so_far = self.frames_processed - 1
                if nbv_movements_so_far >= self.max_iterations:
                    print(f"\n{'='*60}")
                    print(f"🛑 MAX NBV MOVEMENTS COMPLETE ({nbv_movements_so_far}/{self.max_iterations})")
                    print(f"{'='*60}")
                    print(f"Total captures: {self.frames_processed}")
                    print(f"Stopping NBV pipeline...")
                    
                    # Cleanup temporary files (milestones already saved)
                    self.cleanup_temporary_files()
                    
                    # Trigger Unity to stop Play mode
                    self.trigger_unity_stop()
                    
                    self.running = False
                    break

                # After inference, run candidate_positions.py main on fine_new.ply
                try:
                    # Get the downsampled file name (used for inference) - use CURRENT frame number
                    filename_down = f"pointcloud_downsampled_frame{current_frame:04d}_{self.timestamp}.ply"
                    downsampled_folder = os.path.splitext(filename_down)[0]  # Remove .ply extension
                    # fine_ply_path = os.path.join(self.output_folder, downsampled_folder, "fine.ply")
                    fine_ply_path = os.path.join(self.output_folder, downsampled_folder, "fine_new_unexplored.ply")
                    drone_positions = []
                    for drone_data in drone_data_list:
                        pos = drone_data.pose.position.tolist()
                        quat = drone_data.pose.quaternion
                        euler = self.quaternion_to_euler(quat)
                        drone_positions.append((pos, euler))

                    selected_nbvs = candidate_positions_main(
                        fine_ply_path,
                        visualize=VISUALIZE_BOOL,
                        drone_count=len(drone_data_list),
                        drone_positions=drone_positions
                    )
                    
                    print(f"\n{'='*60}")
                    print(f"NBV Planning Complete - Moving drones to NBV positions")
                    print(f"{'='*60}")
                    
                    # Send NBV commands to Unity
                    if not self.send_nbv_commands(selected_nbvs):
                        print("⚠ Failed to send NBV commands, skipping wait")
                    else:
                        # Wait until all drones are within vicinity of their NBV poses
                        print("\nWaiting for drones to reach NBV positions...")
                        vicinity_threshold = 3.5  # meters (increased to account for height offset)
                        max_wait_time = 45.0  # seconds
                        start_time = time.time()
                        check_interval = 1.0  # seconds
                        
                        all_in_vicinity = False
                        while not all_in_vicinity and (time.time() - start_time) < max_wait_time:
                            # Read latest drone positions from shared memory
                            current_drone_data = self.read_drone_data_from_memory()
                            
                            if current_drone_data is None or len(current_drone_data) != len(selected_nbvs):
                                time.sleep(check_interval)
                                continue
                            
                            # Check distance for each drone
                            all_in_vicinity = True
                            for idx, (nbv_pos, _) in enumerate(selected_nbvs):
                                drone_pos = current_drone_data[idx].pose.position
                                dist = np.linalg.norm(np.array(nbv_pos) - drone_pos)
                                
                                if dist > vicinity_threshold:
                                    all_in_vicinity = False
                                    # print(f"  Drone {idx}: {dist:.2f}m from target (target: {nbv_pos}, current: {drone_pos})")
                                    break
                            
                            if not all_in_vicinity:
                                time.sleep(check_interval)
                        
                        elapsed_time = time.time() - start_time
                        if all_in_vicinity:
                            print(f"✓ All drones reached NBV positions in {elapsed_time:.1f}s!")
                        else:
                            print(f"⚠ Timeout after {elapsed_time:.1f}s - drones did not reach NBV positions")
                    
                    print(f"{'='*60}\n")
                    
                    # Update last processed time after NBV movement completes
                    self.last_processed_time = current_time
                        
                except Exception as e:
                    print(f"Error running candidate_positions.py: {e}")
            else:
                # No points captured - update time and continue
                self.last_processed_time = current_time

    def save_iteration_milestone(self, points, colors, iteration_num, num_drones):
        """Save point cloud to FinalPointClouds after each iteration for progression tracking."""
        import shutil
        
        # Create FinalPointClouds_NBV directory on D:/ drive if it doesn't exist
        final_folder = r"D:\advaith\unity-run-files\FinalPointClouds_NBV"
        os.makedirs(final_folder, exist_ok=True)
        
        # Save with iteration number in filename
        milestone_name = f"NBV_raw_{num_drones}_drones_iteration_{iteration_num}.ply"
        milestone_path = os.path.join(final_folder, milestone_name)
        
        try:
            if HAS_OPEN3D:
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(points)
                pcd.colors = o3d.utility.Vector3dVector(colors)
                o3d.io.write_point_cloud(milestone_path, pcd)
            else:
                with open(milestone_path, 'w') as f:
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
            
            print(f"  ✓ Saved iteration {iteration_num} milestone: {milestone_name} ({len(points)} points)")
        except Exception as e:
            print(f"  ⚠ Warning: Could not save iteration milestone: {e}")
    
    def cleanup_temporary_files(self):
        """Delete temporary files in ProcessedImages/PointClouds folder."""
        # Try to import send2trash for safe deletion
        try:
            from send2trash import send2trash
            use_recycle_bin = True
        except ImportError:
            print("  Warning: send2trash not installed, files will be deleted permanently")
            print("  Install with: pip install send2trash")
            use_recycle_bin = False
        
        print(f"\n{'='*60}")
        print(f"CLEANING UP TEMPORARY FILES")
        print(f"{'='*60}")
        
        deleted_files = 0
        deleted_folders = 0
        
        try:
            # Only delete files/folders in the output_folder
            if use_recycle_bin:
                for item in os.listdir(self.output_folder):
                    item_path = os.path.join(self.output_folder, item)
                    try:
                        # Send to recycle bin (safer)
                        send2trash(item_path)
                        if os.path.isfile(item_path):
                            deleted_files += 1
                        else:
                            deleted_folders += 1
                    except Exception as e:
                        print(f"  Warning: Could not move {item} to recycle bin: {e}")
            else:
                print(f"  ⚠ Skipping cleanup - send2trash not installed")
            
            if use_recycle_bin:
                print(f"  ✓ Moved {deleted_files} files and {deleted_folders} folders to Recycle Bin")
                print(f"  ✓ Cleanup complete")
            print(f"{'='*60}\n")
        except Exception as e:
            print(f"  ⚠ Error during cleanup: {e}")

    def save_final_point_clouds_and_cleanup(self, num_drones, num_iterations):
        """Save final point clouds to FinalPointClouds folder and cleanup temporary files."""
        import shutil
        
        # Try to import send2trash for safe deletion
        try:
            from send2trash import send2trash
            use_recycle_bin = True
        except ImportError:
            print("  Warning: send2trash not installed, files will be deleted permanently")
            print("  Install with: pip install send2trash")
            use_recycle_bin = False
        
        # Create FinalPointClouds_NBV directory on D:/ drive if it doesn't exist
        final_folder = r"D:\advaith\unity-run-files\FinalPointClouds_NBV"
        os.makedirs(final_folder, exist_ok=True)
        
        # Find the most recent raw and downsampled point clouds
        try:
            files = os.listdir(self.output_folder)
            raw_files = sorted([f for f in files if f.startswith('pointcloud_raw_')])
            down_files = sorted([f for f in files if f.startswith('pointcloud_downsampled_')])
            
            if raw_files and down_files:
                latest_raw = os.path.join(self.output_folder, raw_files[-1])
                # latest_down = os.path.join(self.output_folder, down_files[-1])
                
                # Copy to FinalPointClouds with new names
                raw_final_name = f"NBV_raw_{num_drones}_drones_{num_iterations}_iterations.ply"
                # down_final_name = f"NBV_downsampled_{num_drones}_drones_{num_iterations}_iterations.ply"
                
                raw_final_path = os.path.join(final_folder, raw_final_name)
                # down_final_path = os.path.join(final_folder, down_final_name)
                
                shutil.copy2(latest_raw, raw_final_path)
                # shutil.copy2(latest_down, down_final_path)
                
                print(f"\n{'='*60}")
                print(f"FINAL POINT CLOUDS SAVED")
                print(f"{'='*60}")
                print(f"  Raw: {raw_final_name}")
                # print(f"  Downsampled: {down_final_name}")
                print(f"  Location: {final_folder}")
                
                # Cleanup: Delete all files in ProcessedImages/PointClouds EXCEPT the final ones we just copied
                print(f"\n{'='*60}")
                print(f"CLEANING UP TEMPORARY FILES")
                print(f"{'='*60}")
                
                deleted_files = 0
                deleted_folders = 0
                
                # Only delete files/folders in the output_folder
                if use_recycle_bin:
                    for item in os.listdir(self.output_folder):
                        item_path = os.path.join(self.output_folder, item)
                        try:
                            # Send to recycle bin (safer)
                            send2trash(item_path)
                            if os.path.isfile(item_path):
                                deleted_files += 1
                            else:
                                deleted_folders += 1
                        except Exception as e:
                            print(f"  Warning: Could not move {item} to recycle bin: {e}")
                else:
                    print(f"  ⚠ Skipping cleanup - send2trash not installed")
                
                if use_recycle_bin:
                    print(f"  ✓ Moved {deleted_files} files and {deleted_folders} folders to Recycle Bin")
                    print(f"  ✓ Cleanup complete")
                print(f"{'='*60}\n")
                
            else:
                print(f"  ⚠ Warning: Could not find final point clouds to save")
                
        except Exception as e:
            print(f"  ⚠ Error saving final point clouds: {e}")
            import traceback
            traceback.print_exc()
    
    def trigger_unity_stop(self):
        """Write trigger file to stop Unity Play mode."""
        # Unity looks in project root (Assets/../nbv_play_trigger.txt)
        trigger_file = r"C:\Users\sriram\vr_swarm_simulation\nbv_play_trigger.txt"
        
        try:
            with open(trigger_file, 'w') as f:
                f.write("stop")
            print(f"  ✓ Unity stop trigger sent: {trigger_file}")
            print(f"  Unity should stop Play mode within a few seconds...")
        except Exception as e:
            print(f"  ⚠ Failed to write Unity stop trigger: {e}")

    def start(self):
        if not self.initialize():
            return False
        try:
            self.processing_loop()
        except KeyboardInterrupt:
            self.stop()
        return True

    def stop(self):
        self.running = False
        self.cleanup()

    def cleanup(self):
        if self.image_mmf:
            self.image_mmf.close()
        if self.intrinsics_mmf:
            self.intrinsics_mmf.close()
        if self.pose_mmf:
            self.pose_mmf.close()
        # Cleanup Windows API shared memory
        if self.command_ptr:
            UnmapViewOfFile(self.command_ptr)
            self.command_ptr = None
        if self.command_handle:
            CloseHandle(self.command_handle)
            self.command_handle = None
        print("Cleaned up MAP_NBV_Trial")

def main():
    print(" NBV Point Cloud Processor - MAP-NBV Implementation")
    print("="*60)
    processor = MAP_NBV_Trial(processing_interval=3.0)
    print("\n Make sure:")
    print("   1. Unity is running with updated NBVImageCapture")
    print("   2. Drones have RGB + Depth cameras configured")
    print("   3. Shared memory is properly set up")
    print()
    
    processor.start()

if __name__ == "__main__":
    main()