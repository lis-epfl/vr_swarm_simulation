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
try:
    import open3d as o3d
    HAS_OPEN3D = True
except ImportError:
    HAS_OPEN3D = False
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

#import candidate_positions
sys.path.append("/Users/advaithsriram/pointr-nbv/scripts")
from candidate_positions import main as candidate_positions_main
sys.path.remove("/Users/advaithsriram/pointr-nbv/scripts")

DOWN_SAMPLE_SIZE = 2000

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

    def send_nbv_commands(self, selected_nbvs):
        """Send selected NBV positions to Unity via shared memory."""
        try:
            drone_count = len(selected_nbvs)
            command_size = 4 + (drone_count * 12)  # flag + commands
            
            if self.command_mmf is None:
                self.command_mmf = mmap.mmap(-1, command_size, self.command_memory_name)
            
            self.command_mmf.seek(0)
            flag = struct.unpack('i', self.command_mmf.read(4))[0]
            
            # Check if Unity is ready
            if flag != 0:
                return False
            
            # Write commands
            self.command_mmf.seek(4)
            for pos, euler in selected_nbvs:
                # Send x, y, z from pos
                self.command_mmf.write(struct.pack('fff', pos[0], pos[1], pos[2]))
                # CHECK, WHEN THE DRONES MOVE, IF THE YAW IS CORRECT
                # DO THEY NEED ORIENTATION?
            
            # Set flag to 2 (commands ready)
            self.command_mmf.seek(0)
            self.command_mmf.write(struct.pack('i', 2))
            print(f"Sent NBV commands: {selected_nbvs}")
            return True
        except Exception as e:
            print(f"Failed to send NBV commands: {e}")
            return False

    def quaternion_to_euler(self, q):
        """Convert quaternion (x, y, z, w) to Euler angles (roll, pitch, yaw) in radians."""
        rotation = Rotation.from_quat(q)
        roll, pitch, yaw = rotation.as_euler('xyz', degrees=False)
        return [roll, pitch, yaw]
    
    def run_pointr_inference(self, downsampled_file):
        """Run PoinTr inference on the downsampled PLY file using subprocess."""
        import subprocess
        pointr_dir = "/Users/advaith/pointr-nbv"
        inference_script = os.path.join(pointr_dir, "tools/inference.py")
        config_path = os.path.join(pointr_dir, "cfgs/ShapeNet55_models/PoinTr.yaml")
        checkpoint_path = os.path.join(pointr_dir, "models/checkpoint55.pth")
        cmd = [
            "python3",
            inference_script,
            config_path,
            checkpoint_path,
            "--pc", downsampled_file,
            "--out_pc_root", "../../ProcessedImages/PointClouds",
            "--save_ply"
        ]
        print(f"Running inference on {downsampled_file} ...")
        result = subprocess.run(cmd, capture_output=True, text=True)
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
        self.command_memory_name = "NBVCommandMemory"
        self.output_folder = "../../ProcessedImages/PointClouds"
        os.makedirs(self.output_folder, exist_ok=True)
        self.sam_model_type = "vit_h"
        self.sam_model_paths = {
            "vit_b": "../../segmentAnything/sam_vit_b_01ec64.pth",
            "vit_l": "../../segmentAnything/sam_vit_l_0b3195.pth",
            "vit_h": "../../segmentAnything/sam_vit_h_4b8939.pth",
        }
        self.min_building_area = 1000
        self.image_mmf = None
        self.intrinsics_mmf = None
        self.pose_mmf = None
        self.command_mmf = None
        self.mask_generator = None
        self.device = None
        self.intrinsics = None
        self.running = False
        self.last_processed_time = 0
        self.frames_processed = 0
        

    def initialize(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        checkpoint_path = self.sam_model_paths[self.sam_model_type]
        sam = sam_model_registry[self.sam_model_type](checkpoint=checkpoint_path)
        sam.to(self.device)
        self.mask_generator = SamAutomaticMaskGenerator(
            sam,
            points_per_side=16,
            pred_iou_thresh=0.90,
            stability_score_thresh=0.96,
            crop_n_layers=0,
            crop_n_points_downscale_factor=1,
            min_mask_region_area=2000,
        )
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
        center_segments = []
        for mask_info in masks:
            mask = mask_info['segmentation']
            area = mask_info['area']
            if (0 <= center_y < mask.shape[0] and 0 <= center_x < mask.shape[1] and
                mask[center_y, center_x] and area >= self.min_building_area):
                center_segments.append(mask_info)
        if not center_segments:
            return None
        best_segment = max(center_segments,
                         key=lambda x: x['stability_score'] * min(x['area']/10000, 10))
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
        MIN_DISTANCE_FROM_DRONE = 0.2
        all_points = []
        all_colors = []
        for drone_data in drone_data_list:
            mask = self.segment_rgb_image(drone_data.rgb_image)
            if mask is None:
                continue
            points_local, colors = self.depth_to_point_cloud(
                drone_data.depth_image,
                mask,
                drone_data.rgb_image
            )
            mask_dist = np.abs(points_local[:, 2]) > MIN_DISTANCE_FROM_DRONE
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
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename_ply = f"pointcloud_frame{frame_id:04d}_{timestamp}.ply"
        filepath_ply = os.path.join(self.output_folder, filename_ply)
        # Save full point cloud first
        if HAS_OPEN3D:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            pcd.colors = o3d.utility.Vector3dVector(colors)
            o3d.io.write_point_cloud(filepath_ply, pcd)
        else:
            with open(filepath_ply, 'w') as f:
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

        # Downsample and save downsampled point cloud
        down_points, down_colors = self.random_downsample(points, colors, size=DOWN_SAMPLE_SIZE, seed=42)
        filename_down = f"pointcloud_downsampled_frame{frame_id:04d}_{timestamp}.ply"
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

            global_points, global_colors = self.fuse_point_clouds(drone_data_list)


            if len(global_points) > 0:
                self.save_point_cloud(global_points, global_colors, self.frames_processed)

                # After inference, run candidate_positions.py main on fine.ply
                try:
                    fine_ply_path = os.path.join(self.output_folder, "fine.ply")
                    drone_positions = []
                    for drone_data in drone_data_list:
                        pos = drone_data.pose.position.tolist()
                        quat = drone_data.pose.quaternion
                        euler = self.quaternion_to_euler(quat)
                        drone_positions.append((pos, euler))

                    selected_nbvs = candidate_positions_main(
                        fine_ply_path,
                        visualize=False,
                        drone_count=len(drone_data_list),
                        drone_positions=drone_positions
                    )
                    
                    self.send_nbv_commands(selected_nbvs)
                except Exception as e:
                    print(f"Error running candidate_positions.py: {e}")
                self.frames_processed += 1
            self.last_processed_time = current_time


            

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
