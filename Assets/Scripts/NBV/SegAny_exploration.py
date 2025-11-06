"""
SegAny Exploration - Standalone Python Script
GPU-accelerated SAM processing for drone swarm images

This script processes 9 drone images from the same timestamp using GPU acceleration.
Generates building segmentation masks and saves visualization results.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import time
import random
import re
from pathlib import Path
import torch
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

# Global configuration
NUM_DRONES = 9
DRONE_IMAGES_FOLDER = "../../DroneImages"
PROCESSED_IMAGES_FOLDER = "../../ProcessedImages"
SAM_MODEL_PATH = "../../segmentAnything/sam_vit_b_01ec64.pth"
MIN_BUILDING_AREA = 7500

def setup_output_directory():
    """Create ProcessedImages directory if it doesn't exist"""
    if not os.path.exists(PROCESSED_IMAGES_FOLDER):
        os.makedirs(PROCESSED_IMAGES_FOLDER)
        print(f"📁 Created directory: {PROCESSED_IMAGES_FOLDER}")
    else:
        print(f"📁 Using existing directory: {PROCESSED_IMAGES_FOLDER}")

def parse_drone_filename(filename):
    """
    Parse drone filename to extract drone_id, date, and time
    Format: drone_<id>_<date>_<time>.png
    Example: drone_0_20251016_144226.png
    Returns: (drone_id, date, time) or None if parsing fails
    """
    pattern = r'drone_(\d+)_(\d{8})_(\d{6})\.png'
    match = re.match(pattern, filename)
    if match:
        drone_id = int(match.group(1))
        date = match.group(2)
        time_str = match.group(3)
        return drone_id, date, time_str
    return None

def find_drone_images_for_timestamp(images_folder, target_date, target_time):
    """
    Find all drone images for a specific date and time
    Returns list of (drone_id, filepath) tuples
    """
    pattern = os.path.join(images_folder, f"drone_*_{target_date}_{target_time}.png")
    matching_files = glob.glob(pattern)
    
    drone_images = []
    for filepath in matching_files:
        filename = os.path.basename(filepath)
        parsed = parse_drone_filename(filename)
        if parsed:
            drone_id, date, time_str = parsed
            drone_images.append((drone_id, filepath))
    
    # Sort by drone_id
    drone_images.sort(key=lambda x: x[0])
    return drone_images

def get_random_timestamp_images(images_folder, num_drones):
    """
    Select a random timestamp and get images for all drones at that time
    Returns list of (drone_id, filepath) tuples
    """
    # Find all drone_0 images (reference drone)
    drone_0_pattern = os.path.join(images_folder, "drone_0_*.png")
    drone_0_files = glob.glob(drone_0_pattern)
    
    if not drone_0_files:
        print(f"❌ No drone_0 images found in {images_folder}")
        return []
    
    # Pick a random drone_0 image
    random_file = random.choice(drone_0_files)
    filename = os.path.basename(random_file)
    
    # Parse the timestamp
    parsed = parse_drone_filename(filename)
    if not parsed:
        print(f"❌ Could not parse filename: {filename}")
        return []
    
    drone_id, date, time_str = parsed
    print(f"🎯 Selected timestamp: {date}_{time_str}")
    print(f"📸 Reference image: {filename}")
    
    # Find all drone images for this timestamp
    drone_images = find_drone_images_for_timestamp(images_folder, date, time_str)
    
    print(f"🔍 Found {len(drone_images)} drone images for this timestamp:")
    for drone_id, filepath in drone_images:
        print(f"   Drone {drone_id}: {os.path.basename(filepath)}")
    
    return drone_images

def load_sam_model():
    """Load SAM model onto GPU and return mask generator"""
    print("🚀 Loading SAM model...")
    model_start_time = time.time()
    
    # Force CUDA device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 Using device: {device}")
    
    if not torch.cuda.is_available():
        print("⚠️  CUDA not available, using CPU (will be slower)")
    
    # Load model on GPU
    sam = sam_model_registry["vit_b"](checkpoint=SAM_MODEL_PATH)
    sam.to(device)
    
    # Create mask generator with GPU acceleration
    # mask_generator = SamAutomaticMaskGenerator(
    #     sam,
    #     points_per_side=32,
    #     pred_iou_thresh=0.88,
    #     stability_score_thresh=0.95,
    #     crop_n_layers=1,
    #     crop_n_points_downscale_factor=2,
    #     min_mask_region_area=1000,
    # )
    # Your current optimized settings:
    mask_generator = SamAutomaticMaskGenerator(
        sam,
        points_per_side=16,          # ← Reduce from 32 to 16 (big impact!)
        pred_iou_thresh=0.90,        # ← Higher = fewer masks
        stability_score_thresh=0.96, # ← Higher = more selective  
        crop_n_layers=0,             # ← Disable cropping (major speedup!)
        crop_n_points_downscale_factor=1,
        min_mask_region_area=2000,   # ← Larger minimum
    )

    model_end_time = time.time()
    model_load_time = model_end_time - model_start_time
    
    print(f"✅ SAM model loaded in {model_load_time:.2f} seconds")
    return mask_generator, device, model_load_time

def point_in_mask(mask, x, y):
    """Check if point (x,y) is inside the segmentation mask"""
    if 0 <= y < mask.shape[0] and 0 <= x < mask.shape[1]:
        return mask[y, x]
    return False

def resize_for_processing(image_rgb, target_size=512):
    """Resize image for faster processing, then scale results back"""
    h, w = image_rgb.shape[:2]
    scale = target_size / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    
    resized = cv2.resize(image_rgb, (new_w, new_h))
    return resized, scale


def process_drone_image(image_path, mask_generator, drone_id):
    """
    Process a single drone image and return results
    Returns: (success, processing_time, results_dict)
    """
    start_time = time.time()
    
    try:
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            return False, 0, {"error": f"Failed to load {image_path}"}
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width, _ = image.shape

        #resize for faster processing
        # resized, scale = resize_for_processing(image_rgb, target_size=512)
        
        # Generate masks
        masks = mask_generator.generate(image_rgb)
        # masks = mask_generator.generate(resized)
        
        # Find center building segment
        center_x = width // 2
        center_y = height // 2
        
        # Find best center segment
        center_segments = []
        for j, mask_info in enumerate(masks):
            mask = mask_info['segmentation']
            area = mask_info['area']
            
            if point_in_mask(mask, center_x, center_y) and area >= MIN_BUILDING_AREA:
                center_segments.append((j, mask_info))
        
        processing_time = time.time() - start_time
        
        if center_segments:
            # Get best segment
            best_segment = max(center_segments, 
                             key=lambda x: x[1]['stability_score'] * min(x[1]['area']/10000, 10))
            
            segment_idx, segment_info = best_segment
            center_mask = segment_info['segmentation']
            
            # Calculate building center
            building_pixels = np.where(center_mask)
            building_center_y = int(np.mean(building_pixels[0]))
            building_center_x = int(np.mean(building_pixels[1]))
            
            results = {
                "success": True,
                "image_rgb": image_rgb,
                "mask": center_mask,
                "area": segment_info['area'],
                "stability": segment_info['stability_score'],
                "center": (building_center_x, building_center_y),
                "image_center": (center_x, center_y),
                "offset": (building_center_x - center_x, building_center_y - center_y),
                "dimensions": (width, height),
                "total_segments": len(masks)
            }
            
            return True, processing_time, results
        else:
            return False, processing_time, {
                "success": False,
                "image_rgb": image_rgb,
                "image_center": (center_x, center_y),
                "dimensions": (width, height),
                "total_segments": len(masks),
                "error": "No suitable building segment found"
            }
            
    except Exception as e:
        processing_time = time.time() - start_time
        return False, processing_time, {"error": str(e)}

def save_visualization(drone_id, results, output_folder, timestamp):
    """Save visualization for a single drone"""
    filename = f"drone_{drone_id}_{timestamp}_processed.png"
    output_path = os.path.join(output_folder, filename)
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Original image
    axes[0].imshow(results["image_rgb"])
    center_x, center_y = results["image_center"]
    axes[0].plot(center_x, center_y, 'r+', markersize=15, markeredgewidth=2, label='Image Center')
    
    if results["success"]:
        building_center_x, building_center_y = results["center"]
        axes[0].plot(building_center_x, building_center_y, 'b+', markersize=15, markeredgewidth=2, label='Building Center')
        axes[0].set_title(f"Drone {drone_id} - Original Image\nBuilding Area: {results['area']:,} px", fontsize=12)
    else:
        axes[0].set_title(f"Drone {drone_id} - Original Image\nNo Building Detected", fontsize=12)
    
    axes[0].legend()
    axes[0].axis('off')
    
    # Segmentation mask or all segments
    if results["success"]:
        axes[1].imshow(results["mask"], cmap='gray')
        axes[1].set_title(f"Building Mask\nOffset: {results['offset']}", fontsize=12)
    else:
        axes[1].imshow(results["image_rgb"])
        axes[1].plot(center_x, center_y, 'r+', markersize=15, markeredgewidth=2)
        axes[1].set_title(f"No Building Found\n{results['total_segments']} segments detected", fontsize=12)
    
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()  # Close to free memory
    
    print(f"💾 Saved: {filename}")

def main():
    """Main processing function"""
    print("="*60)
    print("🚁 DRONE SWARM SAM PROCESSING")
    print("="*60)
    
    # Setup
    setup_output_directory()
    
    # Get random timestamp images
    drone_images = get_random_timestamp_images(DRONE_IMAGES_FOLDER, NUM_DRONES)
    
    if not drone_images:
        print("❌ No drone images found for processing")
        return
    
    if len(drone_images) != NUM_DRONES:
        print(f"⚠️  Expected {NUM_DRONES} drones, found {len(drone_images)}")
    
    # Extract timestamp for file naming
    first_file = os.path.basename(drone_images[0][1])
    parsed = parse_drone_filename(first_file)
    timestamp = f"{parsed[1]}_{parsed[2]}" if parsed else "unknown"
    
    # Load SAM model
    mask_generator, device, model_load_time = load_sam_model()
    
    # Process all drone images
    print(f"\n🔄 Processing {len(drone_images)} drone images...")
    total_start_time = time.time()
    
    processing_times = []
    successful_detections = 0
    
    for drone_id, image_path in drone_images:
        print(f"\n📸 Processing Drone {drone_id}: {os.path.basename(image_path)}")
        
        success, processing_time, results = process_drone_image(image_path, mask_generator, drone_id)
        processing_times.append(processing_time)
        
        if success:
            successful_detections += 1
            print(f"✅ Building detected!")
            print(f"   Area: {results['area']:,} pixels")
            print(f"   Offset: {results['offset']}")
        else:
            print(f"❌ {results.get('error', 'No building found')}")
        
        print(f"   Processing time: {processing_time:.2f} seconds")
        
        # Save visualization
        save_visualization(drone_id, results, PROCESSED_IMAGES_FOLDER, timestamp)
    
    total_end_time = time.time()
    total_processing_time = total_end_time - total_start_time
    
    # Print final statistics
    print("\n" + "="*60)
    print("📊 PROCESSING RESULTS")
    print("="*60)
    print(f"🤖 SAM model load time: {model_load_time:.2f} seconds")
    print(f"⏱️  Total processing time: {total_processing_time:.2f} seconds")
    print(f"🖼️  Images processed: {len(processing_times)}")
    print(f"✅ Successful detections: {successful_detections}/{len(processing_times)}")
    print(f"📈 Success rate: {(successful_detections/len(processing_times))*100:.1f}%")
    
    if processing_times:
        avg_time = sum(processing_times) / len(processing_times)
        min_time = min(processing_times)
        max_time = max(processing_times)
        
        print(f"\n📊 Per-Image Processing Times:")
        for i, (drone_id, _) in enumerate(drone_images):
            print(f"   Drone {drone_id}: {processing_times[i]:.2f} seconds")
        
        print(f"\n📈 Statistics:")
        print(f"   Average: {avg_time:.2f} seconds per image")
        print(f"   Fastest: {min_time:.2f} seconds")
        print(f"   Slowest: {max_time:.2f} seconds")
        print(f"   Throughput: {1/avg_time:.2f} images/second")
        
        if torch.cuda.is_available():
            print(f"   GPU: {torch.cuda.get_device_name(0)}")
    
    print(f"\n💾 Results saved to: {PROCESSED_IMAGES_FOLDER}")
    print("="*60)

if __name__ == "__main__":
    main()