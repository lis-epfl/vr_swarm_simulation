using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using UnityEngine;

/// <summary>
/// NBVImageCapture.cs - MAP-NBV Implementation: RGB + Depth + Pose Capture with Shared Memory
/// 
/// Architecture:
/// 1. Captures RGB images from drone FPV cameras at configurable frequency
/// 2. Captures Depth images from drone depth cameras (DroneDepthCamera)
/// 3. Captures drone poses (position + rotation quaternion)
/// 4. Writes all data to shared memory for Python point cloud processing
/// 5. Reads position commands from shared memory
/// 6. Sends position updates back to NBV.cs for drone control
/// 
/// Memory Layout (NEW - MAP-NBV):
/// - ImageDepthMemory: [flag][droneCount][width][height][RGB_0][Depth_0][RGB_1][Depth_1]...
/// - DronePoseMemory: [droneCount][pos_0][quat_0][pos_1][quat_1]...
/// - CommandMemory: [flag][cmd_0][cmd_1]... (per-drone commands)
/// 
/// Note: Camera intrinsics handled by separate CameraIntrinsics.cs component
/// </summary>
public class NBVImageCapture : MonoBehaviour
{
    [Header("Image Capture Settings")]
    [SerializeField] private int imageWidth = 300;
    [SerializeField] private int imageHeight = 300;
    [SerializeField] private float captureInterval = 0.5f; // Capture every 0.5 seconds
    [SerializeField] private float droneRefreshInterval = 2.0f; // Check for new drones every 2 seconds
    
    [Header("Shared Memory Settings")]
    [SerializeField] private int maxDroneCount = 10;
    [SerializeField] private string imageMemoryName = "NBVImageDepthMemory"; // NEW: Combined RGB + Depth
    [SerializeField] private string poseMemoryName = "NBVDronePoses"; // NEW: Drone poses
    [SerializeField] private string commandMemoryName = "NBVCommandMemory";
    
    [Header("Per-Drone Command System")]
    [SerializeField] private bool usePerDroneCommands = true; // Enable individual drone commands
    
    [Header("Camera Pose Calculation")]
    [SerializeField] private float cameraPitchAngle = 0.0f; // Camera pitch in degrees (from swarmManager)
    [SerializeField] private Vector3 cameraOffset = new Vector3(0f, 0f, 0.4299991f); // Camera offset from DroneParent
    
    [Header("Integration Mode")]
    [SerializeField] private bool enableCommandReading = true; // Enable reading commands from Python
    
    [Header("Debug")]
    [SerializeField] private bool enableDebugLogging = true; // Enable for debugging
    
    // Shared Memory Management
    private IntPtr imageFileMap;
    private IntPtr imagePtr;
    private IntPtr poseFileMap; // NEW: Drone poses
    private IntPtr posePtr;
    private IntPtr commandFileMap;
    private IntPtr commandPtr;
    
    // Memory layout constants
    private const uint FILE_MAP_ALL_ACCESS = 0xF001F;
    private const uint PAGE_READWRITE = 0x04;
    private const int FlagPosition = 0;
    private const int ImageCountPosition = 4;
    private const int ImageDataPosition = 8;
    private const int CommandFlagPosition = 0;
    private const int CommandDataPosition = 4;
    
    // Calculated sizes
    private int imageSize;
    private int totalImageMemorySize;
    private int commandDataSize = 12; // 3 floats (x, y, z)
    private int totalCommandMemorySize;
    
    // Image capture
    private List<Camera> droneCameras;
    private List<DroneDepthCamera> droneDepthCameras; // NEW: Depth cameras
    private List<Transform> droneTransforms; // NEW: Drone transforms for poses
    private RenderTexture reusableTexture;
    private Texture2D captureTexture;
    private byte[] imageBuffer;
    private byte[] depthBuffer; // NEW: Depth data buffer
    private float nextCaptureTime;
    private float nextDroneRefreshTime;
    private bool isFullyInitialized = false;
    
    // NBV Integration
    private NBV nbvScript;
    private SwarmManager swarmManagerScript;
    private Vector3 currentPositionCommand = Vector3.zero;
    
    // Windows API imports
    [DllImport("kernel32.dll", SetLastError = true, CharSet = CharSet.Auto)]
    private static extern IntPtr CreateFileMapping(IntPtr hFile, IntPtr lpAttributes, uint protect, uint maxSizeHigh, uint maxSizeLow, string name);
    
    [DllImport("kernel32.dll", SetLastError = true)]
    private static extern IntPtr MapViewOfFile(IntPtr hFileMappingObject, uint desiredAccess, uint fileOffsetHigh, uint fileOffsetLow, UIntPtr numberOfBytesToMap);
    
    [DllImport("kernel32.dll", SetLastError = true)]
    private static extern bool UnmapViewOfFile(IntPtr lpBaseAddress);
    
    [DllImport("kernel32.dll", SetLastError = true)]
    private static extern bool CloseHandle(IntPtr hObject);

    void Start()
    {
        InitializeMemoryLayout();
        CreateSharedMemory();
        InitializeImageCapture();
        FindNBVScript();
        FindSwarmManager();
        FindDroneCameras();
        
        nextCaptureTime = Time.time + captureInterval;
        nextDroneRefreshTime = Time.time + droneRefreshInterval;
        
        if (enableDebugLogging)
            Debug.Log($"NBVImageCapture initialized with {droneCameras?.Count ?? 0} drone cameras");
            
        isFullyInitialized = true;
    }

    void Update()
    {
        // Wait for full initialization to complete
        if (!isFullyInitialized)
        {
            return;
        }
        
        // Safety check: ensure droneCameras is initialized
        if (droneCameras == null)
        {
            if (enableDebugLogging)
                Debug.LogWarning("NBVImageCapture: droneCameras is null, initializing...");
            FindDroneCameras();
            return;
        }

        // Periodically refresh drone list to catch newly spawned drones
        if (Time.time >= nextDroneRefreshTime)
        {
            int previousCount = droneCameras?.Count ?? 0;
            FindDroneCameras();
            
            if (droneCameras != null && droneCameras.Count != previousCount && enableDebugLogging)
            {
                Debug.Log($"NBVImageCapture: Drone count changed from {previousCount} to {droneCameras.Count}");
            }
            
            nextDroneRefreshTime = Time.time + droneRefreshInterval;
        }
        
        // Capture images at specified interval
        if (Time.time >= nextCaptureTime && droneCameras != null && droneCameras.Count > 0)
        {
            if (enableDebugLogging)
                Debug.Log($"[NBVImageCapture] Triggering capture at time {Time.time}");
                
            CaptureAndSendImages();
            nextCaptureTime = Time.time + captureInterval;
        }
        
        // Read position commands from Python (only if enabled)
        if (enableCommandReading)
        {
            ReadPositionCommands();
        }
    }

    #region Memory Management
    
    private void InitializeMemoryLayout()
    {
        // RGB: 3 bytes per pixel, Depth: 4 bytes per pixel (float32)
        int rgbSize = imageWidth * imageHeight * 3;
        int depthSize = imageWidth * imageHeight * 4;
        imageSize = rgbSize + depthSize; // Combined RGB + Depth per drone
        
        // ImageDepthMemory: [flag(4)][droneCount(4)][width(4)][height(4)][RGB_0+Depth_0][RGB_1+Depth_1]...
        totalImageMemorySize = 16 + (maxDroneCount * imageSize);
        
        // Per-drone commands
        totalCommandMemorySize = CommandDataPosition + (maxDroneCount * commandDataSize);
        
        if (enableDebugLogging)
        {
            Debug.Log($"MAP-NBV Memory Layout:");
            Debug.Log($"  Resolution: {imageWidth}x{imageHeight}");
            Debug.Log($"  RGB size per drone: {rgbSize} bytes");
            Debug.Log($"  Depth size per drone: {depthSize} bytes");
            Debug.Log($"  Total per drone: {imageSize} bytes");
            Debug.Log($"  ImageDepth memory: {totalImageMemorySize} bytes");
            Debug.Log($"  Command memory: {totalCommandMemorySize} bytes");
            Debug.Log($"  Max drones: {maxDroneCount}");
        }
    }
    
    private void CreateSharedMemory()
    {
        // Create image+depth shared memory
        imageFileMap = CreateFileMapping(new IntPtr(-1), IntPtr.Zero, PAGE_READWRITE, 0, (uint)totalImageMemorySize, imageMemoryName);
        if (imageFileMap == IntPtr.Zero)
        {
            Debug.LogError($"Failed to create image+depth shared memory: {imageMemoryName}");
            return;
        }
        
        imagePtr = MapViewOfFile(imageFileMap, FILE_MAP_ALL_ACCESS, 0, 0, UIntPtr.Zero);
        if (imagePtr == IntPtr.Zero)
        {
            Debug.LogError("Failed to map image+depth memory view");
            CloseHandle(imageFileMap);
            return;
        }
        
        // Create pose shared memory (4 bytes count + 28 bytes per drone)
        int poseMemorySize = 4 + (maxDroneCount * 28);
        poseFileMap = CreateFileMapping(new IntPtr(-1), IntPtr.Zero, PAGE_READWRITE, 0, (uint)poseMemorySize, poseMemoryName);
        if (poseFileMap == IntPtr.Zero)
        {
            Debug.LogError($"Failed to create pose shared memory: {poseMemoryName}");
            return;
        }
        
        posePtr = MapViewOfFile(poseFileMap, FILE_MAP_ALL_ACCESS, 0, 0, UIntPtr.Zero);
        if (posePtr == IntPtr.Zero)
        {
            Debug.LogError("Failed to map pose memory view");
            CloseHandle(poseFileMap);
            return;
        }
        
        // Create command shared memory
        if (enableCommandReading)
        {
            commandFileMap = CreateFileMapping(new IntPtr(-1), IntPtr.Zero, PAGE_READWRITE, 0, (uint)totalCommandMemorySize, commandMemoryName);
            if (commandFileMap == IntPtr.Zero)
            {
                Debug.LogError($"Failed to create command shared memory: {commandMemoryName}");
                return;
            }
            
            commandPtr = MapViewOfFile(commandFileMap, FILE_MAP_ALL_ACCESS, 0, 0, UIntPtr.Zero);
            if (commandPtr == IntPtr.Zero)
            {
                Debug.LogError("Failed to map command memory view");
                CloseHandle(commandFileMap);
                return;
            }
            
            // Initialize command flag
            Marshal.WriteInt32(commandPtr, CommandFlagPosition, 0);
        }
        
        // Initialize memory flags
        Marshal.WriteInt32(imagePtr, FlagPosition, 0);
        Marshal.WriteInt32(posePtr, 0, 0); // Write initial drone count
        
        if (enableDebugLogging)
            Debug.Log($"MAP-NBV shared memory created successfully");
    }
    
    private void DestroySharedMemory()
    {
        if (imagePtr != IntPtr.Zero)
        {
            UnmapViewOfFile(imagePtr);
            imagePtr = IntPtr.Zero;
        }
        
        if (imageFileMap != IntPtr.Zero)
        {
            CloseHandle(imageFileMap);
            imageFileMap = IntPtr.Zero;
        }
        
        if (posePtr != IntPtr.Zero)
        {
            UnmapViewOfFile(posePtr);
            posePtr = IntPtr.Zero;
        }
        
        if (poseFileMap != IntPtr.Zero)
        {
            CloseHandle(poseFileMap);
            poseFileMap = IntPtr.Zero;
        }
        
        if (commandPtr != IntPtr.Zero)
        {
            UnmapViewOfFile(commandPtr);
            commandPtr = IntPtr.Zero;
        }
        
        if (commandFileMap != IntPtr.Zero)
        {
            CloseHandle(commandFileMap);
            commandFileMap = IntPtr.Zero;
        }
    }
    
    #endregion
    
    #region Image Capture
    
    private void InitializeImageCapture()
    {
        reusableTexture = new RenderTexture(imageWidth, imageHeight, 24);
        captureTexture = new Texture2D(imageWidth, imageHeight, TextureFormat.RGB24, false);
        
        int rgbSize = imageWidth * imageHeight * 3;
        int depthSize = imageWidth * imageHeight * 4;
        
        imageBuffer = new byte[maxDroneCount * rgbSize];
        depthBuffer = new byte[maxDroneCount * depthSize];
        
        if (enableDebugLogging)
            Debug.Log($"MAP-NBV capture initialized - RGB buffer: {imageBuffer.Length}, Depth buffer: {depthBuffer.Length}");
    }
    
    private void FindDroneCameras()
    {
        droneCameras = new List<Camera>();
        droneDepthCameras = new List<DroneDepthCamera>();
        droneTransforms = new List<Transform>();
        
        GameObject[] drones = GameObject.FindGameObjectsWithTag("DroneBase");
        
        foreach (GameObject drone in drones)
        {
            // Find FPV camera
            Camera fpvCamera = drone.transform.Find("FPV")?.GetComponent<Camera>();
            if (fpvCamera == null)
                continue;
            
            // Find depth camera component
            DroneDepthCamera depthCamera = fpvCamera.GetComponent<DroneDepthCamera>();
            if (depthCamera == null)
            {
                if (enableDebugLogging)
                    Debug.LogWarning($"Drone {drone.name} missing DroneDepthCamera component!");
                continue;
            }
            
            // Force depth camera to match our capture resolution
            depthCamera.SetResolution(imageWidth, imageHeight);
            
            droneCameras.Add(fpvCamera);
            droneDepthCameras.Add(depthCamera);
            // Store FPV camera transform directly - Unity handles all parent transforms!
            droneTransforms.Add(fpvCamera.transform);
            
            if (droneCameras.Count >= maxDroneCount)
                break;
        }
        
        if (enableDebugLogging)
            Debug.Log($"Found {droneCameras.Count} drones with RGB+Depth cameras");
    }
    
    private void CaptureAndSendImages()
    {
        if (!isFullyInitialized || imagePtr == IntPtr.Zero || posePtr == IntPtr.Zero || 
            droneCameras == null || droneCameras.Count == 0 || 
            droneDepthCameras == null || droneDepthCameras.Count == 0)
        {
            if (enableDebugLogging)
            {
                string missingComponents = "";
                if (!isFullyInitialized) missingComponents += "not-initialized ";
                if (imagePtr == IntPtr.Zero) missingComponents += "imagePtr ";
                if (posePtr == IntPtr.Zero) missingComponents += "posePtr ";
                if (droneCameras == null || droneCameras.Count == 0) missingComponents += "no-RGB-cameras ";
                if (droneDepthCameras == null || droneDepthCameras.Count == 0) missingComponents += "no-depth-cameras ";
                Debug.LogWarning($"MAP-NBV: Cannot capture - missing: {missingComponents}");
            }
            return;
        }
        
        // Check if Python is ready to receive (flag == 0)
        int flag = Marshal.ReadInt32(imagePtr, FlagPosition);
        if (flag != 0)
        {
            if (enableDebugLogging)
                Debug.Log($"[NBVImageCapture] Python not ready (flag={flag}), skipping capture");
            return; // Python is still processing previous data
        }
        
        if (enableDebugLogging)
            Debug.Log($"[NBVImageCapture] Python ready (flag={flag}), starting capture...");
        
        // Set flag to indicate we're writing (flag = 1)
        Marshal.WriteInt32(imagePtr, FlagPosition, 1);
        
        try
        {
            int droneCount = droneCameras.Count;
            int rgbSize = imageWidth * imageHeight * 3;
            int depthSize = imageWidth * imageHeight * 4;
            
            // Write header: [flag(4)][droneCount(4)][width(4)][height(4)]
            Marshal.WriteInt32(imagePtr, 4, droneCount);
            Marshal.WriteInt32(imagePtr, 8, imageWidth);
            Marshal.WriteInt32(imagePtr, 12, imageHeight);
            
            // Capture and write each drone's RGB + Depth
            for (int i = 0; i < droneCount; i++)
            {
                if (enableDebugLogging)
                    Debug.Log($"Capturing data for drone {i}...");
                
                // Capture RGB
                byte[] rgbData = CaptureCameraImage(droneCameras[i]);
                if (rgbData != null && rgbData.Length == rgbSize)
                {
                    int rgbOffset = 16 + (i * (rgbSize + depthSize));
                    Marshal.Copy(rgbData, 0, IntPtr.Add(imagePtr, rgbOffset), rgbSize);
                    
                    if (enableDebugLogging)
                        Debug.Log($"  ✓ RGB captured: {rgbData.Length} bytes");
                }
                else if (enableDebugLogging)
                {
                    Debug.LogWarning($"Failed to capture RGB from drone {i} (data: {(rgbData == null ? "null" : rgbData.Length.ToString())} bytes, expected: {rgbSize})");
                }
                
                // Capture Depth
                if (enableDebugLogging)
                    Debug.Log($"  Calling CaptureDepthData for drone {i}...");
                    
                byte[] depthData = CaptureDepthData(droneDepthCameras[i]);
                
                if (enableDebugLogging)
                    Debug.Log($"  CaptureDepthData returned: {(depthData == null ? "null" : depthData.Length + " bytes")}");
                
                if (depthData != null && depthData.Length == depthSize)
                {
                    int depthOffset = 16 + (i * (rgbSize + depthSize)) + rgbSize;
                    Marshal.Copy(depthData, 0, IntPtr.Add(imagePtr, depthOffset), depthSize);
                    
                    if (enableDebugLogging)
                        Debug.Log($"  ✓ Depth captured: {depthData.Length} bytes");
                }
                else if (enableDebugLogging)
                {
                    Debug.LogWarning($"Failed to capture depth from drone {i} (data: {(depthData == null ? "null" : depthData.Length.ToString())} bytes, expected: {depthSize})");
                }
            }
            
            // Write drone poses to separate memory
            WriteDronePoses(droneCount);
            
            // Set flag to indicate data ready (flag = 2)
            Marshal.WriteInt32(imagePtr, FlagPosition, 2);
            
            if (enableDebugLogging)
                Debug.Log($"Sent RGB+Depth+Pose data for {droneCount} drones");
        }
        catch (Exception e)
        {
            Debug.LogError($"Error capturing data: {e.Message}");
            // Reset flag on error
            Marshal.WriteInt32(imagePtr, FlagPosition, 0);
        }
    }
    
    private byte[] CaptureCameraImage(Camera camera)
    {
        if (camera == null || reusableTexture == null || captureTexture == null)
        {
            if (enableDebugLogging)
                Debug.LogWarning("Cannot capture RGB - camera or texture is null");
            return null;
        }
        
        try
        {
            RenderTexture previousRT = camera.targetTexture;
            camera.targetTexture = reusableTexture;
            RenderTexture.active = reusableTexture;
            
            camera.Render();
            captureTexture.ReadPixels(new Rect(0, 0, imageWidth, imageHeight), 0, 0, false);
            captureTexture.Apply(false);
            
            byte[] imageBytes = captureTexture.GetRawTextureData();
            
            camera.targetTexture = previousRT;
            RenderTexture.active = null;
            
            return imageBytes;
        }
        catch (Exception e)
        {
            Debug.LogError($"Failed to capture RGB: {e.Message}");
            return null;
        }
    }
    
    private byte[] CaptureDepthData(DroneDepthCamera depthCamera)
    {
        if (depthCamera == null)
        {
            if (enableDebugLogging)
                Debug.LogWarning("Cannot capture depth - depthCamera is null");
            return null;
        }
        
        try
        {
            // Get depth data as byte array (float32 format)
            byte[] depthBytes = depthCamera.GetDepthDataBytes();
            
            if (depthBytes == null && enableDebugLogging)
            {
                Debug.LogWarning("DroneDepthCamera.GetDepthDataBytes() returned null!");
            }
            else if (enableDebugLogging && depthBytes != null)
            {
                // Debug first few bytes to see if there's actual data
                if (depthBytes.Length >= 16)
                {
                    float testValue = BitConverter.ToSingle(depthBytes, 0);
                    Debug.Log($"Depth data captured: {depthBytes.Length} bytes, first value: {testValue}");
                }
            }
            
            return depthBytes;
        }
        catch (Exception e)
        {
            Debug.LogError($"Failed to capture depth: {e.Message}");
            return null;
        }
    }
    
    private void WriteDronePoses(int droneCount)
    {
        if (posePtr == IntPtr.Zero || droneTransforms == null || droneTransforms.Count != droneCount)
        {
            if (enableDebugLogging)
                Debug.LogWarning("Cannot write poses - invalid state");
            return;
        }
        
        try
        {
            // Write drone count
            Marshal.WriteInt32(posePtr, 0, droneCount);
            
            // Send FPV camera's actual world transform (Unity handles all parent transforms automatically)
            for (int i = 0; i < droneCount; i++)
            {
                Transform fpvTransform = droneTransforms[i]; // FPV camera transform
                
                Vector3 cameraPos = fpvTransform.position;
                Quaternion cameraRot = fpvTransform.rotation;

                // // Adjust for coordinate system differences (Unity to Python)
                // cameraPos.z = -cameraPos.z;
                // cameraRot = new Quaternion(-cameraRot.x, -cameraRot.y, cameraRot.z, cameraRot.w);
                
                // Debug for first drone
                if (i == 0 && enableDebugLogging)
                {
                    Debug.Log($"Drone 0 FPV Camera (Unity Transform):");
                    Debug.Log($"  Position: {cameraPos}");
                    Debug.Log($"  Rotation: {cameraRot.eulerAngles}");
                }
                
                int offset = 4 + (i * 28);
                
                // Write camera position and rotation
                Marshal.Copy(new float[] { cameraPos.x, cameraPos.y, cameraPos.z }, 0, IntPtr.Add(posePtr, offset), 3);
                Marshal.Copy(new float[] { cameraRot.x, cameraRot.y, cameraRot.z, cameraRot.w }, 0, IntPtr.Add(posePtr, offset + 12), 4);
            }
        }
        catch (Exception e)
        {
            Debug.LogError($"Failed to write poses: {e.Message}");
        }
    }
    
    #endregion
    
    #region Position Commands
    
    private void FindNBVScript()
    {
        // Try to find NBV script automatically
        nbvScript = FindObjectOfType<NBV>();
        
        if (nbvScript == null)
        {
            Debug.LogWarning("NBV script not found. Position commands will not be applied.");
        }
        else if (enableDebugLogging)
        {
            Debug.Log("NBV script found successfully");
        }
    }
    
    private void FindSwarmManager()
    {
        // Try to find SwarmManager script to get camera pitch
        swarmManagerScript = FindObjectOfType<SwarmManager>();
        
        if (swarmManagerScript == null)
        {
            Debug.LogWarning("SwarmManager not found. Using hardcoded camera pitch angle.");
        }
        else 
        {
            // Update camera pitch from SwarmManager
            cameraPitchAngle = swarmManagerScript.GetCameraPitch();
            if (enableDebugLogging)
            {
                Debug.Log($"SwarmManager found - Camera pitch: {cameraPitchAngle}°");
            }
        }
    }
    
    private void ReadPositionCommands()
    {
        if (commandPtr == IntPtr.Zero || droneCameras == null || droneCameras.Count == 0)
            return;
        
        // Check if Python has written new commands (flag == 2)
        int flag = Marshal.ReadInt32(commandPtr, CommandFlagPosition);
        if (flag != 2)
            return; // No new commands available
        
        // Set flag to indicate we're reading (flag = 1)
        Marshal.WriteInt32(commandPtr, CommandFlagPosition, 1);
        
        try
        {
            int droneCount = droneCameras.Count;
            
            // Read per-drone commands
            for (int i = 0; i < droneCount; i++)
            {
                int offset = CommandDataPosition + (i * commandDataSize);
                
                byte[] commandBytes = new byte[commandDataSize];
                Marshal.Copy(IntPtr.Add(commandPtr, offset), commandBytes, 0, commandDataSize);
                
                float x = BitConverter.ToSingle(commandBytes, 0);
                float y = BitConverter.ToSingle(commandBytes, 4);
                float z = BitConverter.ToSingle(commandBytes, 8);
                
                Vector3 command = new Vector3(x, y, z);
                
                // Apply command to specific drone
                ApplyPositionCommandToDrone(i, command);
            }
            
            if (enableDebugLogging)
                Debug.Log($"Received {droneCount} position commands from Python");
        }
        catch (Exception e)
        {
            Debug.LogError($"Failed to read commands: {e.Message}");
        }
        finally
        {
            // Reset flag to indicate we're done reading (flag = 0)
            Marshal.WriteInt32(commandPtr, CommandFlagPosition, 0);
        }
    }
    
    private void ApplyPositionCommand(Vector3 positionCommand)
    {
        if (nbvScript != null)
        {
            // For now, just log the command. You can modify this to integrate with your NBV logic
            if (enableDebugLogging)
                Debug.Log($"Applying position command to NBV: {positionCommand}");
            
            // Example integration: You might want to set a target position or add an offset
            // nbvScript.SetTargetPosition(positionCommand);
            // or
            // nbvScript.AddPositionOffset(positionCommand);
        }
    }
    
    private void ApplyPositionCommandToDrone(int droneIndex, Vector3 command)
    {
        // TODO: Integrate with NBV.cs to apply per-drone commands
        // For now, just log for debugging
        if (enableDebugLogging && Vector3.Distance(command, Vector3.zero) > 0.01f)
        {
            Debug.Log($"Drone {droneIndex} command: {command}");
        }
        
        // Future integration example:
        // if (nbvScript != null)
        // {
        //     nbvScript.ApplyCommandToDrone(droneIndex, command);
        // }
    }
    
    #endregion
    
    #region Public Interface
    
    /// <summary>
    /// Get the current position command received from Python processing
    /// </summary>
    public Vector3 GetCurrentPositionCommand()
    {
        return currentPositionCommand;
    }
    
    /// <summary>
    /// Check if shared memory is properly initialized
    /// </summary>
    public bool IsInitialized()
    {
        bool imageReady = imagePtr != IntPtr.Zero;
        bool commandReady = !enableCommandReading || commandPtr != IntPtr.Zero;
        return imageReady && commandReady;
    }
    
    /// <summary>
    /// Get the number of drone cameras being captured
    /// </summary>
    public int GetDroneCameraCount()
    {
        return droneCameras?.Count ?? 0;
    }
    
    /// <summary>
    /// Force refresh of drone cameras (useful if drones are added/removed dynamically)
    /// </summary>
    public void RefreshDroneCameras()
    {
        FindDroneCameras();
    }
    
    #endregion
    
    #region Unity Lifecycle
    
    void OnValidate()
    {
        // Ensure reasonable values
        imageWidth = Mathf.Clamp(imageWidth, 64, 2048);
        imageHeight = Mathf.Clamp(imageHeight, 64, 2048);
        captureInterval = Mathf.Clamp(captureInterval, 0.1f, 5.0f);
        maxDroneCount = Mathf.Clamp(maxDroneCount, 1, 20);
    }
    
    void OnDestroy()
    {
        DestroySharedMemory();
    }
    
    void OnApplicationQuit()
    {
        DestroySharedMemory();
        if (enableDebugLogging)
            Debug.Log("NBVImageCapture: Shared memory cleaned up on application quit");
    }
    
    #endregion
}