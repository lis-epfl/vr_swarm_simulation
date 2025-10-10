using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using UnityEngine;

/// <summary>
/// NBVImageCapture.cs - Real-time drone image capture with shared memory communication
/// 
/// Architecture:
/// 1. Captures images from drone FPV cameras at configurable frequency
/// 2. Writes image data to shared memory for Python processing
/// 3. Reads position commands from shared memory 
/// 4. Sends position updates back to NBV.cs for drone control
/// 
/// Memory Layout:
/// - ImageSharedMemory: [flag][imageCount][imageData...]
/// - CommandSharedMemory: [flag][commandData...]
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
    [SerializeField] private string imageMemoryName = "NBVImageSharedMemory";
    [SerializeField] private string commandMemoryName = "NBVCommandSharedMemory";
    
    [Header("Integration Mode")]
    [SerializeField] private bool enableCommandReading = false; // DISABLED: NBV.cs now handles vision commands directly
    
    [Header("Debug")]
    [SerializeField] private bool enableDebugLogging = true; // Enable for debugging
    
    // Shared Memory Management
    private IntPtr imageFileMap;
    private IntPtr imagePtr;
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
    private RenderTexture reusableTexture;
    private Texture2D captureTexture;
    private byte[] imageBuffer;
    private float nextCaptureTime;
    private float nextDroneRefreshTime;
    private bool isFullyInitialized = false;
    
    // NBV Integration
    private NBV nbvScript;
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
        imageSize = imageWidth * imageHeight * 3; // RGB format
        totalImageMemorySize = ImageDataPosition + (maxDroneCount * imageSize);
        totalCommandMemorySize = CommandDataPosition + commandDataSize;
        
        if (enableDebugLogging)
        {
            Debug.Log($"Memory Layout:");
            Debug.Log($"  Image size per drone: {imageSize} bytes");
            Debug.Log($"  Total image memory: {totalImageMemorySize} bytes");
            Debug.Log($"  Command memory: {totalCommandMemorySize} bytes");
        }
    }
    
    private void CreateSharedMemory()
    {
        // Create image shared memory
        imageFileMap = CreateFileMapping(new IntPtr(-1), IntPtr.Zero, PAGE_READWRITE, 0, (uint)totalImageMemorySize, imageMemoryName);
        if (imageFileMap == IntPtr.Zero)
        {
            Debug.LogError($"Failed to create image shared memory: {imageMemoryName}");
            return;
        }
        
        imagePtr = MapViewOfFile(imageFileMap, FILE_MAP_ALL_ACCESS, 0, 0, UIntPtr.Zero);
        if (imagePtr == IntPtr.Zero)
        {
            Debug.LogError("Failed to map image memory view");
            CloseHandle(imageFileMap);
            return;
        }
        
        // Create command shared memory (only if enabled)
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
        }
        else
        {
            if (enableDebugLogging)
                Debug.Log("NBVImageCapture: Command reading disabled - NBV.cs handles vision commands directly");
        }
        
        // Initialize memory flags
        Marshal.WriteInt32(imagePtr, FlagPosition, 0);
        Marshal.WriteInt32(commandPtr, CommandFlagPosition, 0);
        
        if (enableDebugLogging)
            Debug.Log("Shared memory created successfully");
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
        imageBuffer = new byte[maxDroneCount * imageSize];
        
        if (enableDebugLogging)
            Debug.Log($"NBVImageCapture: Image capture initialized - buffer size: {imageBuffer.Length}, image size: {imageSize}");
    }
    
    private void FindDroneCameras()
    {
        droneCameras = new List<Camera>();
        
        GameObject[] drones = GameObject.FindGameObjectsWithTag("DroneBase");
        
        foreach (GameObject drone in drones)
        {
            Camera fpvCamera = drone.transform.Find("FPV")?.GetComponent<Camera>();
            if (fpvCamera != null)
            {
                droneCameras.Add(fpvCamera);
                if (droneCameras.Count >= maxDroneCount)
                    break;
            }
        }
        
        if (enableDebugLogging)
            Debug.Log($"Found {droneCameras.Count} drone cameras");
    }
    
    private void CaptureAndSendImages()
    {
        if (!isFullyInitialized || imagePtr == IntPtr.Zero || droneCameras == null || droneCameras.Count == 0 || imageBuffer == null)
        {
            if (enableDebugLogging)
            {
                string missingComponents = "";
                if (!isFullyInitialized) missingComponents += "not-initialized ";
                if (imagePtr == IntPtr.Zero) missingComponents += "imagePtr ";
                if (droneCameras == null) missingComponents += "droneCameras ";
                if (droneCameras?.Count == 0) missingComponents += "no-cameras ";
                if (imageBuffer == null) missingComponents += "imageBuffer ";
                Debug.LogWarning($"NBVImageCapture: Cannot capture images - missing: {missingComponents}");
            }
            return;
        }
        
        // Check if Python is ready to receive (flag == 0)
        int flag = Marshal.ReadInt32(imagePtr, FlagPosition);
        if (flag != 0)
            return; // Python is still processing previous images
        
        // Set flag to indicate we're writing
        Marshal.WriteInt32(imagePtr, FlagPosition, 1);
        
        try
        {
            // Write image count
            Marshal.WriteInt32(imagePtr, ImageCountPosition, droneCameras.Count);
            
            // Capture and write each drone image
            for (int i = 0; i < droneCameras.Count; i++)
            {
                byte[] imageData = CaptureCameraImage(droneCameras[i]);
                if (imageData != null && imageData.Length == imageSize)
                {
                    Array.Copy(imageData, 0, imageBuffer, i * imageSize, imageData.Length);
                }
                else if (enableDebugLogging)
                {
                    Debug.LogWarning($"Failed to capture image from drone {i}");
                }
            }
            
            // Write all image data to shared memory (only if buffer is valid)
            if (imageBuffer != null && imageBuffer.Length >= droneCameras.Count * imageSize)
            {
                Marshal.Copy(imageBuffer, 0, IntPtr.Add(imagePtr, ImageDataPosition), droneCameras.Count * imageSize);
                
                if (enableDebugLogging)
                    Debug.Log($"Sent {droneCameras.Count} drone images to shared memory");
            }
            else if (enableDebugLogging)
            {
                Debug.LogError("Image buffer is invalid - cannot send to shared memory");
            }
        }
        finally
        {
            // Reset flag to indicate we're done writing
            Marshal.WriteInt32(imagePtr, FlagPosition, 0);
        }
    }
    
    private byte[] CaptureCameraImage(Camera camera)
    {
        if (camera == null || reusableTexture == null || captureTexture == null)
        {
            if (enableDebugLogging)
                Debug.LogWarning("NBVImageCapture: Cannot capture image - camera or texture is null");
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
            Debug.LogError($"Failed to capture camera image: {e.Message}");
            return null;
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
    
    private void ReadPositionCommands()
    {
        if (commandPtr == IntPtr.Zero)
            return;
        
        // Check if Python has written new commands (flag == 0)
        int flag = Marshal.ReadInt32(commandPtr, CommandFlagPosition);
        if (flag != 0)
            return; // No new commands available
        
        // Set flag to indicate we're reading
        Marshal.WriteInt32(commandPtr, CommandFlagPosition, 1);
        
        try
        {
            // Read position command (3 floats: x, y, z)
            byte[] commandBytes = new byte[commandDataSize];
            Marshal.Copy(IntPtr.Add(commandPtr, CommandDataPosition), commandBytes, 0, commandDataSize);
            
            float x = BitConverter.ToSingle(commandBytes, 0);
            float y = BitConverter.ToSingle(commandBytes, 4);
            float z = BitConverter.ToSingle(commandBytes, 8);
            
            Vector3 newCommand = new Vector3(x, y, z);
            
            // Apply position command if it's different from current
            if (Vector3.Distance(newCommand, currentPositionCommand) > 0.01f)
            {
                currentPositionCommand = newCommand;
                ApplyPositionCommand(newCommand);
                
                if (enableDebugLogging)
                    Debug.Log($"Received position command: {newCommand}");
            }
        }
        finally
        {
            // Reset flag to indicate we're done reading
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
        imageWidth = Mathf.Clamp(imageWidth, 64, 1024);
        imageHeight = Mathf.Clamp(imageHeight, 64, 1024);
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