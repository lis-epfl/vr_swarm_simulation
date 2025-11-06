# MAP-NBV Architecture Documentation

## Overview
Implementation of MAP-NBV (Multi-Agent Planning via Next-Best-View) for drone swarm 3D reconstruction.

---

## System Architecture

```
Unity (C#)                          Python
┌─────────────────┐                ┌──────────────────────┐
│ Drone Swarm     │                │ NBVPointCloud        │
│                 │                │ Processor.py         │
│ ┌─────────────┐ │                │                      │
│ │ RGB Camera  │ ├─────RGB────────→│ SAM Segmentation   │
│ │ Depth Camera│ ├────Depth───────→│ Depth Masking      │
│ │ Transform   │ ├────Pose────────→│ Point Cloud Gen    │
│ │ Intrinsics  │ ├──Intrinsics───→│ Global Fusion      │
│ └─────────────┘ │                │                      │
│                 │◄───Commands─────┤ (Random for now)    │
└─────────────────┘                └──────────────────────┘
                                             │
                                             ▼
                                   ┌──────────────────┐
                                   │ Point Cloud File │
                                   │ (.ply, .pcd)     │
                                   └──────────────────┘
```

---

## Shared Memory Layout

### Memory Map 1: Image Data (Unity → Python)
**Name**: `NBVImageDepthMemory`

```
Offset  | Size              | Description
--------|-------------------|----------------------------------
0       | 4 bytes           | Flag (0=ready, 1=writing, 2=data ready)
4       | 4 bytes           | Drone count (dynamic from Unity)
8       | 4 bytes           | Image width
12      | 4 bytes           | Image height
16      | drone_count × (RGB_size + Depth_size)

For each drone i:
  RGB_offset = 16 + i × (RGB_size + Depth_size)
  RGB_size = width × height × 3 (bytes)
  
  Depth_offset = RGB_offset + RGB_size
  Depth_size = width × height × 4 (float32 bytes)
```

**Example** (3 drones, 640×480 images):
- RGB per drone: 640 × 480 × 3 = 921,600 bytes
- Depth per drone: 640 × 480 × 4 = 1,228,800 bytes
- Total per drone: 2,150,400 bytes
- Total memory: 16 + (3 × 2,150,400) = 6,451,216 bytes

### Memory Map 2: Camera Intrinsics (Unity → Python)
**Name**: `NBVCameraIntrinsics`

```
Offset  | Size      | Description
--------|-----------|----------------------------------
0       | 4 bytes   | fx (focal length x in pixels)
4       | 4 bytes   | fy (focal length y in pixels)
8       | 4 bytes   | cx (principal point x)
12      | 4 bytes   | cy (principal point y)
16      | 4 bytes   | width (image width)
20      | 4 bytes   | height (image height)
Total: 24 bytes
```

### Memory Map 3: Drone Poses (Unity → Python)
**Name**: `NBVDronePoses`

```
Offset  | Size              | Description
--------|-------------------|----------------------------------
0       | 4 bytes           | Drone count
4       | drone_count × 28 bytes

For each drone i:
  Pose_offset = 4 + i × 28
  
  Position (12 bytes):
    +0:  float x (Unity position)
    +4:  float y
    +8:  float z
  
  Rotation (16 bytes):
    +12: float qx (Unity quaternion)
    +16: float qy
    +20: float qz
    +24: float qw
```

### Memory Map 4: Commands (Python → Unity)
**Name**: `NBVCommandMemory` (already exists)

```
Current layout (per-drone commands):
[flag: 4 bytes]
[drone0_command: 12 bytes (x,y,z)]
[drone1_command: 12 bytes (x,y,z)]
...
```

---

## Coordinate Systems

### Unity Coordinate System
- **Left-handed**: X=right, Y=up, Z=forward
- **Camera**: Looks down +Z axis
- **World Origin**: Scene center

### Point Cloud Coordinate System
- **Right-handed**: X=right, Y=forward, Z=up
- **Standard for robotics**: ROS convention

### Conversion (Unity → Point Cloud)
```python
# Unity: (x_u, y_u, z_u)
# Point Cloud: (x_pc, y_pc, z_pc)

x_pc = x_u      # X stays same (right)
y_pc = z_u      # Y = Unity's Z (forward)
z_pc = y_u      # Z = Unity's Y (up)
```

---

## Camera Intrinsics Extraction

### From Unity Camera
```csharp
Camera cam = drone.GetComponent<Camera>();
float fovVertical = cam.fieldOfView; // degrees
float aspect = cam.aspect;

// Calculate focal lengths
float fovVerticalRad = fovVertical * Mathf.Deg2Rad;
float fovHorizontalRad = 2 * Mathf.Atan(Mathf.Tan(fovVerticalRad / 2) * aspect);

int height = imageHeight;
int width = imageWidth;

// Intrinsics
float fy = (height / 2.0f) / Mathf.Tan(fovVerticalRad / 2.0f);
float fx = (width / 2.0f) / Mathf.Tan(fovHorizontalRad / 2.0f);
float cx = width / 2.0f;
float cy = height / 2.0f;
```

### Intrinsic Matrix K
```
K = [fx  0   cx]
    [0   fy  cy]
    [0   0   1 ]
```

---

## Pinhole Camera Model

### Depth to 3D Point Conversion

For pixel (u, v) with depth D:

```python
# In camera coordinates
X_cam = (u - cx) * D / fx
Y_cam = (v - cy) * D / fy
Z_cam = D

# Point in camera frame
P_cam = [X_cam, Y_cam, Z_cam]
```

### Vectorized Implementation (NumPy)
```python
# Create meshgrid of pixel coordinates
v, u = np.mgrid[0:height, 0:width]

# Apply to all pixels at once
X_cam = (u - cx) * depth / fx
Y_cam = (v - cy) * depth / fy
Z_cam = depth

# Stack into point cloud (N×3 array)
points_cam = np.stack([X_cam, Y_cam, Z_cam], axis=-1)
```

---

## Point Cloud Fusion

### Local to Global Transformation

For drone at position `p = (px, py, pz)` and rotation `q = (qx, qy, qz, qw)`:

```python
# 1. Convert quaternion to rotation matrix
R = quaternion_to_rotation_matrix(q)

# 2. Create transformation matrix
T = [[R[0,0], R[0,1], R[0,2], px],
     [R[1,0], R[1,1], R[1,2], py],
     [R[2,0], R[2,1], R[2,2], pz],
     [0,      0,      0,      1 ]]

# 3. Transform points
# For each point P_local = (x, y, z, 1)
P_global = T @ P_local

# Vectorized for all points:
points_local_homo = np.hstack([points_local, np.ones((N, 1))])
points_global_homo = (T @ points_local_homo.T).T
points_global = points_global_homo[:, :3]
```

### Fusion Strategy
- **Concatenate**: Simply combine all point clouds
- **Filter**: Remove outliers and duplicates
- **Downsample**: Use voxel grid for efficiency

---

## Data Flow

### Unity Side (C#)
1. **Capture**:
   - RGB: `Camera.Render()` → `Texture2D.GetPixels()`
   - Depth: `DroneDepthCamera.CaptureDepth()`
   - Pose: `transform.position`, `transform.rotation`

2. **Write to Memory**:
   - Set flag = 1 (writing)
   - Write drone count
   - Write image dimensions
   - Write RGB + Depth for each drone
   - Write camera intrinsics
   - Write drone poses
   - Set flag = 2 (ready)

3. **Read Commands**:
   - Check command flag
   - Read per-drone commands
   - Apply to NBV system
   - Reset flag

### Python Side
1. **Read from Memory**:
   - Wait for flag = 2
   - Read drone count (dynamic)
   - Read dimensions
   - Read RGB + Depth for each drone
   - Read intrinsics
   - Read poses

2. **Process**:
   - For each drone:
     * Segment RGB with SAM
     * Apply mask to depth
     * Convert to 3D points
     * Transform to global frame
   - Fuse all point clouds
   
3. **Output**:
   - Save point cloud file (.ply)
   - Generate random commands (for now)
   - Write commands to memory

---

## Point Cloud File Formats

### PLY Format (Recommended)
```
ply
format ascii 1.0
element vertex <N>
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
<x1> <y1> <z1> <r1> <g1> <b1>
<x2> <y2> <z2> <r2> <g2> <b2>
...
```

### PCD Format (Alternative)
```
VERSION 0.7
FIELDS x y z rgb
SIZE 4 4 4 4
TYPE F F F U
COUNT 1 1 1 1
WIDTH <N>
HEIGHT 1
POINTS <N>
DATA ascii
<x1> <y1> <z1> <rgb1>
...
```

---

## Visualization Tools

### Python Libraries
- **Open3D**: Advanced visualization, filtering
- **Matplotlib (3D)**: Quick plotting
- **PyVista**: Interactive visualization

### Example Open3D Visualization
```python
import open3d as o3d

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points_xyz)
pcd.colors = o3d.utility.Vector3dVector(colors_rgb)

o3d.visualization.draw_geometries([pcd])
```

---

## Performance Considerations

### Memory Usage (3 drones, 640×480)
- RGB data: 2.6 MB
- Depth data: 3.5 MB
- Poses: 84 bytes
- Intrinsics: 24 bytes
- **Total**: ~6.1 MB per frame

### Processing Time Estimates
- SAM segmentation: 0.4s per drone
- Point cloud generation: 0.05s per drone
- Fusion: 0.1s
- **Total**: ~1.5s for 3 drones

### Optimization Strategies
- **Parallel SAM**: Process drones simultaneously
- **Downsample depth**: Reduce resolution for faster processing
- **Incremental fusion**: Update global cloud instead of rebuilding
- **GPU acceleration**: Use CUDA for point cloud operations

---

## Error Handling

### Unity Side
- Check if depth camera initialized
- Validate image dimensions
- Handle missing drones gracefully

### Python Side
- Verify memory maps exist
- Validate data integrity
- Handle segmentation failures
- Catch transformation errors

---

## Testing Strategy

### Unit Tests
1. Camera intrinsics extraction
2. Pinhole model conversion
3. Coordinate system transformation
4. Point cloud merging

### Integration Tests
1. Unity → Python data transfer
2. Full pipeline with one drone
3. Multi-drone fusion
4. Command feedback loop

### Validation
1. Compare with ground truth
2. Verify coordinate transformations
3. Check point cloud alignment
4. Measure reconstruction accuracy

---

## Future Enhancements

### Phase 1 (Current)
- ✅ RGB + Depth capture
- ✅ SAM segmentation
- ✅ Point cloud generation
- ✅ Global fusion
- ✅ Visualization

### Phase 2 (Next)
- Uncertainty estimation
- Coverage analysis
- NBV planning algorithm
- Intelligent movement commands

### Phase 3 (Advanced)
- Real-time reconstruction
- Incremental mapping
- Multi-resolution point clouds
- Collision avoidance integration

---

## References

- MAP-NBV Paper: Multi-Agent Planning via Next-Best-View
- Open3D Documentation: http://www.open3d.org/
- Pinhole Camera Model: Multiple View Geometry (Hartley & Zisserman)
- Unity Camera API: Unity Documentation

---

## Contact & Support

For issues or questions:
1. Check Unity Console for C# errors
2. Check Python terminal for processing errors
3. Verify memory map sizes match
4. Enable debug visualization to validate data flow
