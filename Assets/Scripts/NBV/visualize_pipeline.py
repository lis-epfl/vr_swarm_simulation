"""
visualize_pipeline.py - Visualize the complete NBV processing pipeline

This script reads the latest point cloud and displays:
- For each drone: RGB image, Segmented RGB, Depth image, Segmented Depth
- Combined 3D point cloud visualization

Usage:
    py -3.10 visualize_pipeline.py [optional_ply_file]
"""

import numpy as np
import cv2
import open3d as o3d
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import struct
import mmap

# Import from the main processor
from NBVPointCloudProcessor import NBVPointCloudProcessor, DroneData


class PipelineVisualizer:
    def __init__(self):
        self.processor = NBVPointCloudProcessor(debug=True)
        self.output_dir = Path("Assets/ProcessedImages")
        self.pointcloud_dir = self.output_dir / "PointClouds"
        
        # Segmentation disabled - using depth values only
        # print("🔄 Initializing SAM model...")
        # self.processor.initialize()
        # print("✅ SAM model ready")
        print("✅ Segmentation disabled - using depth values only")
        
    def find_latest_pointcloud(self):
        """Find the most recent point cloud file"""
        if not self.pointcloud_dir.exists():
            return None
        
        ply_files = list(self.pointcloud_dir.glob("pointcloud_frame*.ply"))
        if not ply_files:
            return None
        
        # Sort by modification time
        ply_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        return ply_files[0]
    
    def read_current_data(self):
        """Read the current data from shared memory"""
        try:
            return self.processor.read_drone_data_from_memory()
        except Exception as e:
            print(f"Error reading data from memory: {e}")
            return None
    
    def visualize_drone_data(self, drone_data_list):
        """Create a matplotlib figure showing all drone data"""
        if not drone_data_list:
            print("No drone data to visualize")
            return
        
        num_drones = len(drone_data_list)
        
        # Simplified: 2 columns (RGB, Depth) × N rows (drones) - no segmentation
        fig, axes = plt.subplots(num_drones, 2, figsize=(12, 4 * num_drones))
        
        # Handle single drone case
        if num_drones == 1:
            axes = axes.reshape(1, -1)
        
        for drone_id, drone_data in enumerate(drone_data_list):
            print(f"\nProcessing Drone {drone_id}...")
            
            # Get RGB image
            rgb_image = drone_data.rgb_image
            depth_image = drone_data.depth_image
            
            # Segmentation disabled - skip SAM processing
            # print(f"  Running SAM segmentation...")
            # all_masks = self.processor.mask_generator.generate(rgb_image)
            
            # Simplified visualization - no segmentation
            
            # Plot RGB
            axes[drone_id, 0].imshow(cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB))
            axes[drone_id, 0].set_title(f'Drone {drone_id}: RGB Image')
            axes[drone_id, 0].axis('off')
            
            # Plot Depth
            depth_display = axes[drone_id, 1].imshow(depth_image, cmap='turbo', vmin=0, vmax=50)
            valid_pixels = np.sum((depth_image > 0) & ~np.isnan(depth_image))
            axes[drone_id, 1].set_title(f'Drone {drone_id}: Depth Image\n(min={depth_image[depth_image>0].min():.2f}m, max={depth_image.max():.2f}m, {valid_pixels:,} valid pixels)')
            axes[drone_id, 1].axis('off')
            plt.colorbar(depth_display, ax=axes[drone_id, 1], fraction=0.046, pad=0.04)
        
        plt.tight_layout()
        
        # Save figure
        output_path = self.output_dir / "pipeline_visualization.png"
        self.output_dir.mkdir(parents=True, exist_ok=True)  # Create directory if needed
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\n✅ Saved visualization to: {output_path}")
        
        plt.show()
    
    def visualize_pointcloud(self, ply_file):
        """Visualize point cloud using Open3D"""
        if not Path(ply_file).exists():
            print(f"Point cloud file not found: {ply_file}")
            return
        
        print(f"\n📊 Loading point cloud: {ply_file}")
        pcd = o3d.io.read_point_cloud(str(ply_file))
        
        num_points = len(pcd.points)
        print(f"   Points: {num_points}")
        
        if num_points == 0:
            print("⚠️  Point cloud is empty!")
            return
        
        # Print bounds
        bbox = pcd.get_axis_aligned_bounding_box()
        print(f"   Bounds: {bbox.get_min_bound()} to {bbox.get_max_bound()}")
        
        # Visualize
        print("\n🔍 Opening 3D viewer...")
        print("   Controls:")
        print("   - Left mouse: Rotate")
        print("   - Right mouse: Pan")
        print("   - Scroll: Zoom")
        print("   - Press 'H' for help")
        
        o3d.visualization.draw_geometries(
            [pcd],
            window_name="NBV Point Cloud",
            width=1280,
            height=720,
            left=50,
            top=50,
            point_show_normal=False
        )
    
    def run(self, specific_ply=None):
        """Run the complete visualization pipeline"""
        print("=" * 70)
        print("NBV Pipeline Visualizer")
        print("=" * 70)
        
        # 1. Read current data from shared memory
        print("\n📥 Reading data from shared memory...")
        drone_data_list = self.read_current_data()
        
        if drone_data_list:
            print(f"✅ Successfully read data for {len(drone_data_list)} drones")
            
            # 2. Visualize 2D pipeline (RGB, Segmentation, Depth)
            print("\n🎨 Creating 2D visualization...")
            self.visualize_drone_data(drone_data_list)
        else:
            print("⚠️  No data available in shared memory")
            print("   Make sure Unity is running and has captured at least one frame!")
        
        # 3. Visualize 3D point cloud
        if specific_ply:
            ply_file = Path(specific_ply)
        else:
            ply_file = self.find_latest_pointcloud()
        
        if ply_file and ply_file.exists():
            self.visualize_pointcloud(ply_file)
        else:
            print("\n⚠️  No point cloud files found")
            print(f"   Looking in: {self.pointcloud_dir}")


def main():
    """Main entry point"""
    visualizer = PipelineVisualizer()
    
    # Check if specific PLY file was provided
    specific_ply = sys.argv[1] if len(sys.argv) > 1 else None
    
    if specific_ply:
        print(f"Using specified point cloud: {specific_ply}")
    
    visualizer.run(specific_ply)


if __name__ == "__main__":
    main()
