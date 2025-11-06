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
        
        # Initialize SAM model
        print("🔄 Initializing SAM model...")
        self.processor.initialize()
        print("✅ SAM model ready")
        
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
        
        # Create figure with subplots: 4 columns (RGB, Segmented RGB, Depth, Segmented Depth) × N rows (drones)
        fig, axes = plt.subplots(num_drones, 4, figsize=(16, 4 * num_drones))
        
        # Handle single drone case
        if num_drones == 1:
            axes = axes.reshape(1, -1)
        
        for drone_id, drone_data in enumerate(drone_data_list):
            print(f"\nProcessing Drone {drone_id}...")
            
            # Get RGB image
            rgb_image = drone_data.rgb_image
            depth_image = drone_data.depth_image
            
            # Process with SAM to get ALL building masks (not just the best one)
            print(f"  Running SAM segmentation...")
            
            # Run SAM directly to get all masks
            all_masks = self.processor.mask_generator.generate(rgb_image)
            
            print(f"      🔍 SAM found {len(all_masks)} total segments")
            
            # Filter for building masks (center point + min area)
            height, width = rgb_image.shape[:2]
            center_x, center_y = width // 2, height // 2
            
            building_masks = []
            for mask_info in all_masks:
                mask = mask_info['segmentation']
                area = mask_info['area']
                
                # Check if center point is in mask and meets area requirement
                if (0 <= center_y < mask.shape[0] and 0 <= center_x < mask.shape[1] and
                    mask[center_y, center_x] and area >= self.processor.min_building_area):
                    building_masks.append(mask)
            
            print(f"  Found {len(building_masks)} building segments (center + area filter)")
            
            # Create segmented RGB visualization
            segmented_rgb = rgb_image.copy()
            if len(building_masks) > 0:
                # Overlay masks with different colors
                colors = plt.cm.rainbow(np.linspace(0, 1, len(building_masks)))
                for mask, color in zip(building_masks, colors):
                    # Ensure mask is boolean and matches image shape
                    mask_bool = mask.astype(bool) if not mask.dtype == bool else mask
                    
                    # Check if mask shape matches image shape, transpose if needed
                    if mask_bool.shape != rgb_image.shape[:2]:
                        print(f"    ⚠️  Mask shape {mask_bool.shape} doesn't match image {rgb_image.shape[:2]}, transposing...")
                        mask_bool = mask_bool.T
                    
                    color_rgb = (np.array(color[:3]) * 255).astype(np.uint8)
                    segmented_rgb[mask_bool] = segmented_rgb[mask_bool] * 0.5 + color_rgb * 0.5
            
            # Create segmented depth (combined mask)
            segmented_depth = depth_image.copy()
            if len(building_masks) > 0:
                combined_mask = np.zeros(rgb_image.shape[:2], dtype=bool)
                for mask in building_masks:
                    mask_bool = mask.astype(bool) if not mask.dtype == bool else mask
                    
                    # Check if mask shape matches image shape, transpose if needed
                    if mask_bool.shape != depth_image.shape:
                        mask_bool = mask_bool.T
                    
                    combined_mask |= mask_bool
                # Zero out non-building areas
                segmented_depth[~combined_mask] = 0
            
            # Plot RGB
            axes[drone_id, 0].imshow(cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB))
            axes[drone_id, 0].set_title(f'Drone {drone_id}: RGB Image')
            axes[drone_id, 0].axis('off')
            
            # Plot Segmented RGB
            axes[drone_id, 1].imshow(cv2.cvtColor(segmented_rgb.astype(np.uint8), cv2.COLOR_BGR2RGB))
            axes[drone_id, 1].set_title(f'Drone {drone_id}: Segmented RGB ({len(building_masks)} buildings)')
            axes[drone_id, 1].axis('off')
            
            # Plot Depth
            depth_display = axes[drone_id, 2].imshow(depth_image, cmap='turbo', vmin=0, vmax=50)
            axes[drone_id, 2].set_title(f'Drone {drone_id}: Depth Image\n(min={depth_image.min():.2f}m, max={depth_image.max():.2f}m)')
            axes[drone_id, 2].axis('off')
            plt.colorbar(depth_display, ax=axes[drone_id, 2], fraction=0.046, pad=0.04)
            
            # Plot Segmented Depth
            seg_depth_display = axes[drone_id, 3].imshow(segmented_depth, cmap='turbo', vmin=0, vmax=50)
            valid_pixels = np.sum(segmented_depth > 0)
            axes[drone_id, 3].set_title(f'Drone {drone_id}: Segmented Depth\n({valid_pixels} valid pixels)')
            axes[drone_id, 3].axis('off')
            plt.colorbar(seg_depth_display, ax=axes[drone_id, 3], fraction=0.046, pad=0.04)
        
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
