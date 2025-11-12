# Camera Pose Transformation Fix

## Problem
The fused global point clouds were not accurately representing the scene because all drone positions were showing as (0, 0, 0). This caused all point clouds to merge at the origin with no proper spatial separation.

## Root Cause
The original code was storing the `DroneBase` GameObject transform, which doesn't move (it stays at origin). The actual drone movement happens in the `DroneParent` child GameObject.

## Solution Architecture

### Transformation Pipeline
To get the correct camera pose for 3D reconstruction, we now follow this pipeline:

1. **Get DroneParent Position & Rotation**: The DroneParent contains the actual drone's world position and orientation
2. **Apply Camera Pitch Angle**: Apply pitch rotation (around local X-axis) to account for camera angle
3. **Apply Camera Offset**: Add camera offset from DroneParent center (0, 0, 0.4299991)

This matches the real-world drone setup where we know:
- The drone's position and orientation (from GPS/IMU)
- The camera's pitch angle relative to drone
- The camera's physical offset from drone center

### Unity Changes (NBVImageCapture.cs)

#### 1. Added Camera Configuration Fields
```csharp
[Header("Camera Pose Calculation")]
[SerializeField] private float cameraPitchAngle = 0.0f; // Camera pitch in degrees
[SerializeField] private Vector3 cameraOffset = new Vector3(0f, 0f, 0.4299991f); // Camera offset
```

#### 2. Added swarmManager Reference
```csharp
private swarmManager swarmManagerScript; // Get camera pitch dynamically
```

#### 3. Updated FindDroneCameras()
**Before:**
```csharp
droneTransforms.Add(drone.transform); // Wrong - DroneBase doesn't move!
```

**After:**
```csharp
Transform droneParent = drone.transform.Find("DroneParent");
if (droneParent == null) { /* error */ }
droneTransforms.Add(droneParent); // Correct - DroneParent has actual position
```

#### 4. Added FindSwarmManager()
```csharp
private void FindSwarmManager()
{
    swarmManagerScript = FindObjectOfType<swarmManager>();
    if (swarmManagerScript != null)
    {
        cameraPitchAngle = swarmManagerScript.GetCameraPitch();
        Debug.Log($"Camera pitch: {cameraPitchAngle}°");
    }
}
```

#### 5. Updated WriteDronePoses()
**Before:**
```csharp
Vector3 pos = droneTransform.position; // DroneBase position (always 0,0,0)
Quaternion rot = droneTransform.rotation;
Marshal.Copy(new float[] { pos.x, pos.y, pos.z }, ...);
```

**After:**
```csharp
// Step 1: Get DroneParent position and rotation
Vector3 dronePos = droneParentTransform.position;
Quaternion droneRot = droneParentTransform.rotation;

// Step 2: Apply camera pitch angle
Quaternion pitchRotation = Quaternion.Euler(cameraPitchAngle, 0f, 0f);
Quaternion cameraRot = droneRot * pitchRotation;

// Step 3: Apply camera offset
Vector3 cameraPos = dronePos + droneRot * cameraOffset;

// Write camera pose to shared memory
Marshal.Copy(new float[] { cameraPos.x, cameraPos.y, cameraPos.z }, ...);
Marshal.Copy(new float[] { cameraRot.x, cameraRot.y, cameraRot.z, cameraRot.w }, ...);
```

### Python Changes (NBVPointCloudProcessor.py)

#### Updated read_drone_pose() Debug Output
```python
if self.debug and drone_id == 0:
    print(f"   📍 Drone {drone_id} Camera Pose: pos=({px:.2f}, {py:.2f}, {pz:.2f}), quat=({qx:.3f}, {qy:.3f}, {qz:.3f}, {qw:.3f})")
```

**No changes needed to transformation logic** - since Unity now sends the correct camera poses, the existing `transform_to_global_frame()` method works correctly!

## Testing

### 1. Start Unity
Make sure the scene is running with multiple drones

### 2. Check Unity Console
You should see debug output like:
```
Drone 0 Camera Pose:
  DroneParent: pos=(15.2, 10.5, 37.8), rot=(0, 45, 0)
  Camera Pitch: -30°
  Camera Offset: (0, 0, 0.43)
  Final Camera: pos=(15.2, 10.5, 38.23), rot=(330, 45, 0)
```

### 3. Start Python Processor
```powershell
py -3.10 Assets\Scripts\NBV\NBVPointCloudProcessor.py
```

### 4. Check Python Console
You should now see **non-zero positions**:
```
📥 Read data for 3 drones
   📍 Drone 0 Camera Pose: pos=(15.20, 10.50, 38.23), quat=(0.259, 0.259, -0.683, 0.683)
   
🔄 Processing Drone 0...
   ✅ Segmented building: 42,189 pixels
   🎯 Generated 35,421 3D points
      Local: X=[-2.5,2.5] Y=[-1.8,1.8] Z=[1.0,25.3]
      Global: X=[12.7,17.7] Y=[8.7,12.3] Z=[13.3,63.5]
      Pose: pos=[15.2 10.5 38.23], quat=[0.259 0.259 -0.683 0.683]
```

### 5. Verify Point Clouds
- Open generated `.ply` files in MeshLab/CloudCompare
- Point clouds from different drones should now be **spatially separated**
- Buildings should align properly in 3D space
- Colors should match RGB images

## Expected Results

✅ **Before Fix**: All drones at (0, 0, 0) → overlapping point clouds  
✅ **After Fix**: Each drone at correct position → properly fused point clouds

✅ **Before**: Single merged blob at origin  
✅ **After**: Accurate 3D reconstruction with proper multi-view fusion

## Configuration

If you need to adjust the camera offset or pitch:

1. **Camera Offset**: Edit in Unity Inspector (NBVImageCapture component)
   - Default: `(0, 0, 0.4299991)` - distance from DroneParent to FPV camera
   
2. **Camera Pitch**: Controlled by swarmManager
   - Automatically read from `swarmManager.cameraPitch`
   - Can also hardcode in NBVImageCapture Inspector if needed

## Notes for Real Drone Deployment

This architecture matches real-world drone setup:
- **Drone Position**: From GPS/IMU (DroneParent)
- **Camera Pitch**: From gimbal encoder or fixed mounting angle
- **Camera Offset**: Measured physical offset from drone center

No changes needed when transitioning from simulation to real drones!
