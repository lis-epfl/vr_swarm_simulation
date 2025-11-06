import sys
import os
import cv2
import numpy as np
from pathlib import Path
import json

def detect_objects_in_images(image_folder):
    """
    Process all images in the folder and detect bounding boxes
    """
    image_folder = Path(image_folder)
    results = []
    
    print(f"Processing images in: {image_folder}")
    
    for image_file in sorted(image_folder.glob("*.png")):
        print(f"Processing: {image_file.name}")
        
        # Load image
        image = cv2.imread(str(image_file))
        if image is None:
            print(f"Could not load image: {image_file}")
            continue
            
        # Your object detection logic here
        bounding_boxes = detect_objects(image)
        
        result = {
            'image': image_file.name,
            'drone_id': extract_drone_id(image_file.name),
            'bounding_boxes': bounding_boxes,
            'image_size': {'width': image.shape[1], 'height': image.shape[0]}
        }
        
        results.append(result)
        
        # Save annotated image
        annotated_image = draw_bounding_boxes(image, bounding_boxes)
        annotated_path = image_folder / f"annotated_{image_file.name}"
        cv2.imwrite(str(annotated_path), annotated_image)
    
    # Save results to JSON
    results_path = image_folder / "detection_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    return results

def extract_drone_id(filename):
    """Extract drone ID from filename like 'drone_0_capture_...'"""
    try:
        return int(filename.split('_')[1])
    except:
        return -1

def detect_objects(image):
    """
    Replace this with your actual object detection algorithm
    Returns list of bounding boxes in format: [(x, y, width, height), ...]
    """
    # Convert to different color spaces for better detection
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Method 1: Color-based detection (example for detecting colorful objects)
    # Adjust these HSV ranges based on your target object colors
    lower_bound = np.array([0, 50, 50])
    upper_bound = np.array([180, 255, 255])
    mask = cv2.inRange(hsv, lower_bound, upper_bound)
    
    # Method 2: Edge detection
    edges = cv2.Canny(gray, 50, 150)
    
    # Combine masks
    combined_mask = cv2.bitwise_or(mask, edges)
    
    # Morphological operations to clean up
    kernel = np.ones((5, 5), np.uint8)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
    
    # Find contours
    contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    bounding_boxes = []
    for contour in contours:
        # Filter small objects
        area = cv2.contourArea(contour)
        if area > 500:  # Minimum area threshold
            x, y, w, h = cv2.boundingRect(contour)
            
            # Filter by aspect ratio and size
            aspect_ratio = w / h
            if 0.2 < aspect_ratio < 5.0 and w > 30 and h > 30:
                bounding_boxes.append({
                    'x': int(x),
                    'y': int(y), 
                    'width': int(w),
                    'height': int(h),
                    'area': int(area),
                    'confidence': min(area / 10000, 1.0)  # Simple confidence score
                })
    
    return bounding_boxes

def draw_bounding_boxes(image, bounding_boxes):
    """Draw bounding boxes on the image"""
    annotated = image.copy()
    
    for i, bbox in enumerate(bounding_boxes):
        x, y, w, h = bbox['x'], bbox['y'], bbox['width'], bbox['height']
        
        # Draw rectangle
        cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Draw label
        label = f"Object_{i} ({bbox.get('confidence', 0):.2f})"
        cv2.putText(annotated, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    return annotated

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python bounding_box_detection.py <image_folder>")
        sys.exit(1)
    
    image_folder = sys.argv[1]
    
    if not os.path.exists(image_folder):
        print(f"Error: Image folder does not exist: {image_folder}")
        sys.exit(1)
    
    print(f"Starting object detection on images in: {image_folder}")
    results = detect_objects_in_images(image_folder)
    
    # Print summary
    print(f"\n=== DETECTION SUMMARY ===")
    print(f"Processed {len(results)} images")
    
    for result in results:
        drone_id = result['drone_id']
        num_objects = len(result['bounding_boxes'])
        print(f"Drone {drone_id}: {num_objects} objects detected")
        
        for j, bbox in enumerate(result['bounding_boxes']):
            print(f"  Object {j}: ({bbox['x']}, {bbox['y']}, {bbox['width']}, {bbox['height']}) confidence: {bbox['confidence']:.2f}")
    
    print(f"\nResults saved to: {os.path.join(image_folder, 'detection_results.json')}")
    print("Annotated images saved with 'annotated_' prefix")