# Adding Depth Cameras to Your Drones - Step-by-Step Guide

## Overview
This guide shows you how to add depth perception to your drone swarm, giving each drone both RGB color images AND depth information.

## What You'll Get
- **RGB Camera**: Normal color images (what you already have)
- **Depth Camera**: Distance information for every pixel (NEW!)
- **Integrated System**: Both working together seamlessly

---

## Step 1: Add DroneDepthCamera Script to Your Drones

### In Unity Editor:

1. **Open your scene** with the drone prefab/swarm

2. **Select one drone** in the Hierarchy

3. **Find the drone's Camera component** (it should already have one for RGB)

4. **Add the DroneDepthCamera script**:
   - In the Inspector, click "Add Component"
   - Search for "DroneDepthCamera"
   - Click to add it

5. **Configure the script**:
   - **RGB Camera**: Drag your drone's existing Camera into this field
   - **Depth Width**: 640 (match your RGB width)
   - **Depth Height**: 480 (match your RGB height)
   - **Max Depth Distance**: 50 (meters - adjust based on your scene)
   - **Min Depth Distance**: 0.1 (meters)
   - **Show Debug Visualization**: Check this to see depth in real-time!

6. **Apply to all drones**:
   - If using a prefab: The changes will apply to all instances
   - If individual drones: Repeat steps 2-5 for each drone

---

## Step 2: Test the Depth Camera

### Quick Test:

1. **Enable debug visualization**:
   - Check "Show Debug Visualization" in DroneDepthCamera component

2. **Play the scene**

3. **Look for depth visualization**:
   - You should see a grayscale image in the top-right corner
   - **Brighter = Closer objects**
   - **Darker = Farther objects**

### Troubleshooting:

**Problem**: No depth visualization appears
- **Solution**: Make sure "RGB Camera" field is assigned
- **Solution**: Check that the drone camera is active

**Problem**: Depth image is all black
- **Solution**: Increase "Max Depth Distance"
- **Solution**: Check that objects are in camera view

**Problem**: Depth image is all white
- **Solution**: Decrease "Min Depth Distance"
- **Solution**: Objects might be too close

---

## Step 3: Access Depth Data in Your Code

### Option A: Get Depth as Texture (for sharing with Python)

```csharp
// In NBVImageCapture.cs or your capture script
DroneDepthCamera depthCam = drone.GetComponent<DroneDepthCamera>();
Texture2D depthTexture = depthCam.CaptureDepth();
byte[] depthBytes = depthTexture.GetRawTextureData();
```

### Option B: Get Depth as Meters (for calculations)

```csharp
// Get actual depth values in meters
DroneDepthCamera depthCam = drone.GetComponent<DroneDepthCamera>();
float[] depthMeters = depthCam.GetDepthDataMeters();

// Access specific pixel depth
int pixelX = 320; // center x
int pixelY = 240; // center y
int pixelIndex = pixelY * 640 + pixelX;
float distanceAtCenter = depthMeters[pixelIndex]; // meters
```

### Option C: Get Visualization (for debugging)

```csharp
// Get grayscale depth visualization
Texture2D depthViz = depthCam.GetDepthVisualization();
```

---

## Step 4: Understand Depth Data Format

### Depth Texture Format:
- **Type**: RFloat (32-bit floating point per pixel)
- **Values**: 0.0 to 1.0 (normalized depth)
- **Conversion**: `actualMeters = minDepth + (normalizedValue * (maxDepth - minDepth))`

### Depth Array Format:
- **Size**: width × height floats
- **Order**: Row-major (left-to-right, top-to-bottom)
- **Units**: Meters
- **Access**: `depth[y * width + x]` = distance at pixel (x, y)

---

## Step 5: Integration Examples

### Example 1: Check if path is clear

```csharp
DroneDepthCamera depthCam = GetComponent<DroneDepthCamera>();
float[] depths = depthCam.GetDepthDataMeters();

// Check center 10x10 pixels
int centerX = 640 / 2;
int centerY = 480 / 2;
float minDistance = float.MaxValue;

for (int y = centerY - 5; y < centerY + 5; y++)
{
    for (int x = centerX - 5; x < centerX + 5; x++)
    {
        int index = y * 640 + x;
        if (depths[index] < minDistance)
            minDistance = depths[index];
    }
}

if (minDistance < 5.0f)
{
    Debug.Log("Obstacle detected at " + minDistance + " meters!");
}
```

### Example 2: Find closest object

```csharp
float[] depths = depthCam.GetDepthDataMeters();
float closestDepth = float.MaxValue;
int closestX = 0, closestY = 0;

for (int y = 0; y < 480; y++)
{
    for (int x = 0; x < 640; x++)
    {
        int index = y * 640 + x;
        if (depths[index] < closestDepth)
        {
            closestDepth = depths[index];
            closestX = x;
            closestY = y;
        }
    }
}

Debug.Log($"Closest object at pixel ({closestX}, {closestY}), distance: {closestDepth}m");
```

### Example 3: Create depth map visualization

```csharp
Texture2D depthViz = depthCam.GetDepthVisualization();
// Save or display this texture
```

---

## Step 6: Performance Tips

### Optimization:

1. **Match RGB resolution**: Use same width/height for depth and RGB
2. **Limit capture frequency**: Don't capture every frame unless needed
3. **Reuse textures**: The script already reuses textures for efficiency
4. **Adjust max depth**: Lower values = better depth precision

### Recommended Settings by Use Case:

**Obstacle Avoidance**:
- Max Depth: 10-20m
- Resolution: 320x240 (faster)
- Frequency: Every frame

**Environment Mapping**:
- Max Depth: 50-100m
- Resolution: 640x480
- Frequency: 0.5-1 second intervals

**Object Detection**:
- Max Depth: 20-30m
- Resolution: 640x480
- Frequency: Every 2-3 seconds

---

## Common Issues & Solutions

### Issue: Depth looks wrong or inverted
**Solution**: Depth camera might be using different coordinate system. The script handles flipping automatically, but check if you need to invert the visualization.

### Issue: Depth values are all the same
**Solution**: All objects might be at max depth distance. Increase `maxDepthDistance` or move closer to objects.

### Issue: Performance is slow
**Solution**: 
- Reduce depth resolution
- Increase capture interval
- Capture depth less frequently than RGB

### Issue: Depth doesn't match RGB
**Solution**: 
- Ensure both cameras have same FOV
- Check that depth camera position matches RGB camera
- Verify both render the same layers (culling mask)

---

## Next Steps

Now that you have depth cameras working:

1. **Integrate with NBVImageCapture.cs**: Capture both RGB and depth together
2. **Send to Python**: Share depth data with NBVImageProcessor.py
3. **Use for navigation**: Implement obstacle avoidance with depth data
4. **Combine with SAM**: Use depth to understand 3D building structure

---

## Questions or Issues?

If something doesn't work:
1. Check that "RGB Camera" field is assigned
2. Enable "Show Debug Visualization" to see if depth is being captured
3. Check Unity Console for error messages
4. Verify camera is active and rendering

Good luck! 🚁✨
