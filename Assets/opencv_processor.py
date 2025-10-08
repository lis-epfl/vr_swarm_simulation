import cv2
import numpy as np
import sys
import os
from pathlib import Path
import json
import datetime

def process_drone_images(batch_file_path):
    """
    Process drone images with OpenCV functions.
    
    Args:
        batch_file_path: Path to text file containing list of image paths
    """
    
    # Read image paths from batch file
    with open(batch_file_path, 'r') as f:
        image_paths = [line.strip() for line in f.readlines()]
    
    if not image_paths:
        print("No image paths found in batch file")
        return
    
    # Create output directory
    output_dir = Path(batch_file_path).parent / "processed"
    output_dir.mkdir(exist_ok=True)
    
    # Process each image (simplified - no metadata saving for now)
    processed_count = 0
    
    for i, image_path in enumerate(image_paths):
        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                print(f"Failed to load image: {image_path}")
                continue
            
            # Extract drone index from filename
            filename = Path(image_path).stem
            drone_index = extract_drone_index(filename)
            
            # Perform OpenCV operations (simplified)
            processed_image = perform_basic_opencv_analysis(image, drone_index)
            
            # Save processed image
            output_path = output_dir / f"processed_{filename}.jpg"
            cv2.imwrite(str(output_path), processed_image)
            
            processed_count += 1
            print(f"Processed drone {drone_index} image successfully")
            
        except Exception as e:
            print(f"Error processing {image_path}: {str(e)}")
    
    print(f"✅ Processing complete! {processed_count}/{len(image_paths)} images processed successfully")
    return processed_count

def extract_drone_index(filename):
    """Extract drone index from filename like 'drone_0_20241008_123456'"""
    try:
        parts = filename.split('_')
        if len(parts) >= 2 and parts[0] == 'drone':
            return int(parts[1])
    except:
        pass
    return -1

def perform_basic_opencv_analysis(image, drone_index):
    """
    Simplified OpenCV processing - basic edge detection and feature highlighting.
    Customize this function with your specific computer vision tasks.
    """
    
    # Create a copy for processing
    processed_image = image.copy()
    
    # 1. Convert to grayscale for analysis
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 2. Edge Detection
    edges = cv2.Canny(gray, 50, 150)
    
    # 3. Find contours (objects)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter significant contours and draw them
    significant_contours = [c for c in contours if cv2.contourArea(c) > 100]
    cv2.drawContours(processed_image, significant_contours, -1, (0, 255, 0), 2)
    
    # 4. Add simple overlay info
    overlay_text = f"Drone {drone_index} | Objects: {len(significant_contours)}"
    cv2.putText(processed_image, overlay_text, (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    return processed_image

# COMMENTED OUT - DETAILED ANALYSIS (uncomment if you want full analysis later)
"""
def perform_opencv_analysis(image, drone_index):
    """
    Perform various OpenCV operations on the drone image.
    Customize this function with your specific computer vision tasks.
    """
    
    # Get image dimensions
    height, width, channels = image.shape
    
    # Create a copy for processing
    processed_image = image.copy()
    
    # 1. Edge Detection
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    
    # 2. Object Detection (simple contour detection)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter significant contours
    significant_contours = [c for c in contours if cv2.contourArea(c) > 100]
    
    # Draw contours on processed image
    cv2.drawContours(processed_image, significant_contours, -1, (0, 255, 0), 2)
    
    # 3. Feature Detection (SIFT/ORB)
    try:
        orb = cv2.ORB_create()
        keypoints, descriptors = orb.detectAndCompute(gray, None)
        cv2.drawKeypoints(processed_image, keypoints, processed_image, color=(255, 0, 0))
    except:
        keypoints = []
        descriptors = None
    
    # 4. Color Analysis
    color_analysis = analyze_dominant_colors(image)
    
    # 5. Calculate image statistics
    mean_intensity = np.mean(gray)
    std_intensity = np.std(gray)
    
    # 6. Detect potential obstacles/objects
    obstacle_detection = detect_obstacles(gray, significant_contours)
    
    # 7. Calculate coverage metrics (for mapping/exploration)
    coverage_metrics = calculate_coverage_metrics(gray, edges)
    
    # Compile analysis results
    analysis = {
        'drone_index': drone_index,
        'image_size': {'width': width, 'height': height},
        'edge_pixels': int(np.sum(edges > 0)),
        'num_contours': len(significant_contours),
        'num_keypoints': len(keypoints),
        'mean_intensity': float(mean_intensity),
        'std_intensity': float(std_intensity),
        'dominant_colors': color_analysis,
        'obstacles': obstacle_detection,
        'coverage': coverage_metrics,
        'summary': f"Found {len(significant_contours)} objects, {len(keypoints)} features"
    }
    
    # Add informational text to processed image
    add_analysis_overlay(processed_image, analysis)
    
    return {
        'processed_image': processed_image,
        'analysis': analysis,
        'edges': edges,
        'keypoints': keypoints
    }

def analyze_dominant_colors(image, k=3):
    """Analyze dominant colors in the image using K-means clustering"""
    try:
        # Reshape image to be a list of pixels
        data = image.reshape((-1, 3))
        data = np.float32(data)
        
        # Apply K-means
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
        _, labels, centers = cv2.kmeans(data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        
        # Convert back to uint8
        centers = np.uint8(centers)
        
        return {
            'dominant_colors': centers.tolist(),
            'num_clusters': k
        }
    except:
        return {'dominant_colors': [], 'num_clusters': 0}

def detect_obstacles(gray_image, contours):
    """Detect potential obstacles based on contour analysis"""
    obstacles = []
    
    for i, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if area < 500:  # Filter small objects
            continue
            
        # Get bounding rectangle
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / h
        
        # Calculate contour properties
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 0
        
        # Classify potential obstacle type
        obstacle_type = classify_obstacle(area, aspect_ratio, solidity)
        
        obstacle = {
            'id': i,
            'center': (int(x + w/2), int(y + h/2)),
            'area': float(area),
            'bounding_box': [int(x), int(y), int(w), int(h)],
            'aspect_ratio': float(aspect_ratio),
            'solidity': float(solidity),
            'type': obstacle_type
        }
        obstacles.append(obstacle)
    
    return obstacles

def classify_obstacle(area, aspect_ratio, solidity):
    """Simple obstacle classification based on geometric properties"""
    if area > 5000:
        if 0.8 < aspect_ratio < 1.2 and solidity > 0.8:
            return "building"
        elif aspect_ratio > 2.0:
            return "wall"
        else:
            return "large_structure"
    elif area > 1000:
        if solidity > 0.7:
            return "vehicle"
        else:
            return "vegetation"
    else:
        return "small_object"

def calculate_coverage_metrics(gray_image, edges):
    """Calculate metrics useful for exploration/mapping"""
    
    height, width = gray_image.shape
    total_pixels = height * width
    
    # Edge density (information content)
    edge_density = np.sum(edges > 0) / total_pixels
    
    # Texture analysis using variance
    texture_variance = cv2.Laplacian(gray_image, cv2.CV_64F).var()
    
    # Contrast measure
    contrast = gray_image.std()
    
    # Information entropy (measure of information content)
    hist = cv2.calcHist([gray_image], [0], None, [256], [0, 256])
    hist = hist / total_pixels  # Normalize
    entropy = -np.sum(hist * np.log2(hist + 1e-7))  # Add small value to avoid log(0)
    
    return {
        'edge_density': float(edge_density),
        'texture_variance': float(texture_variance),
        'contrast': float(contrast),
        'entropy': float(entropy),
        'exploration_value': float(edge_density * entropy)  # Combined metric
    }

def add_analysis_overlay(image, analysis):
    """Add analysis information as text overlay on the image"""
    
    # Set up text properties
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    color = (255, 255, 255)  # White text
    thickness = 1
    
    # Background for text
    overlay = image.copy()
    cv2.rectangle(overlay, (10, 10), (300, 100), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, image, 0.3, 0, image)
    
    # Add text information
    y_offset = 25
    texts = [
        f"Drone: {analysis['drone_index']}",
        f"Objects: {analysis['num_contours']}",
        f"Features: {analysis['num_keypoints']}",
        f"Exploration Value: {analysis['coverage']['exploration_value']:.2f}"
    ]
    
    for i, text in enumerate(texts):
        y_pos = y_offset + i * 20
        cv2.putText(image, text, (15, y_pos), font, font_scale, color, thickness)

def main():
    if len(sys.argv) != 2:
        print("Usage: python opencv_processor.py <batch_file_path>")
        sys.exit(1)
    
    batch_file_path = sys.argv[1]
    
    if not os.path.exists(batch_file_path):
        print(f"Batch file not found: {batch_file_path}")
        sys.exit(1)
    
    try:
        results = process_drone_images(batch_file_path)
        print(f"Successfully processed {results} images")
    except Exception as e:
        print(f"Error processing images: {str(e)}")
        sys.exit(1)
"""
# END OF COMMENTED SECTION - All detailed analysis functions are commented out above

if __name__ == "__main__":
    main()