using System;
using System.IO.MemoryMappedFiles;
using UnityEngine;

/// <summary>
/// Extracts camera intrinsic parameters from Unity Camera and shares them via shared memory
/// 
/// Computes focal lengths (fx, fy) and principal point (cx, cy) from Unity's camera
/// field of view and sensor size, then writes to shared memory for Python consumption.
/// 
/// Memory Layout (24 bytes total):
///   Offset 0:  float fx (focal length in pixels, x-axis)
///   Offset 4:  float fy (focal length in pixels, y-axis)
///   Offset 8:  float cx (principal point x, typically width/2)
///   Offset 12: float cy (principal point y, typically height/2)
///   Offset 16: float width (image width in pixels)
///   Offset 20: float height (image height in pixels)
///   
/// Attach to: Any GameObject with a Camera component (typically the drone camera)
/// </summary>
public class CameraIntrinsics : MonoBehaviour
{
    [Header("Camera Reference")]
    [Tooltip("The camera to extract intrinsics from. If null, uses GetComponent<Camera>()")]
    public Camera targetCamera;

    [Header("Shared Memory Settings")]
    [Tooltip("Name of the shared memory map for intrinsics")]
    public string memoryName = "NBVCameraIntrinsics";
    
    [Tooltip("Size of the shared memory in bytes (24 bytes for 6 floats)")]
    private const int MEMORY_SIZE = 24;

    [Header("Debug")]
    [Tooltip("Enable debug logging")]
    public bool debugMode = false;

    // Computed intrinsics
    private float fx; // Focal length x (pixels)
    private float fy; // Focal length y (pixels)
    private float cx; // Principal point x (pixels)
    private float cy; // Principal point y (pixels)
    private int width;
    private int height;

    // Shared memory
    private MemoryMappedFile mmf;
    private MemoryMappedViewAccessor accessor;
    private bool isInitialized = false;

    void Start()
    {
        // Get camera reference
        if (targetCamera == null)
        {
            targetCamera = GetComponent<Camera>();
        }

        if (targetCamera == null)
        {
            Debug.LogError("[CameraIntrinsics] No camera found! Please assign targetCamera.");
            enabled = false;
            return;
        }

        // Initialize shared memory
        InitializeMemory();

        // Compute and write intrinsics
        ComputeIntrinsics();
        WriteToMemory();

        Debug.Log($"[CameraIntrinsics] Initialized successfully for camera: {targetCamera.name}");
    }

    void InitializeMemory()
    {
        try
        {
            mmf = MemoryMappedFile.CreateOrOpen(memoryName, MEMORY_SIZE);
            accessor = mmf.CreateViewAccessor(0, MEMORY_SIZE);
            isInitialized = true;

            if (debugMode)
            {
                Debug.Log($"[CameraIntrinsics] Shared memory created: {memoryName} ({MEMORY_SIZE} bytes)");
            }
        }
        catch (Exception e)
        {
            Debug.LogError($"[CameraIntrinsics] Failed to initialize shared memory: {e.Message}");
            isInitialized = false;
        }
    }

    void ComputeIntrinsics()
    {
        // Get image resolution
        width = targetCamera.pixelWidth;
        height = targetCamera.pixelHeight;

        // Get Unity's projection matrix
        Matrix4x4 P = targetCamera.projectionMatrix;

        // See: https://stackoverflow.com/questions/6652253/getting-the-projection-matrix-parameters-back
        // Unity's projection matrix is in NDC, so we need to extract fx, fy, cx, cy
        // fx = P[0,0] * width / 2
        // fy = P[1,1] * height / 2
        // cx = (1 - P[0,2]) * width / 2
        // cy = (1 - P[1,2]) * height / 2
        fx = P[0,0] * width / 2.0f;
        fy = P[1,1] * height / 2.0f;
        cx = (1.0f - P[0,2]) * width / 2.0f;
        cy = (1.0f - P[1,2]) * height / 2.0f;

        if (debugMode)
        {
            Debug.Log($"[CameraIntrinsics] Computed from projectionMatrix:");
            Debug.Log($"  Resolution: {width} x {height}");
            Debug.Log($"  fx: {fx:F2}, fy: {fy:F2}");
            Debug.Log($"  cx: {cx:F2}, cy: {cy:F2}");
            Debug.Log($"  ProjectionMatrix: \n{P}");
        }
    }

    void WriteToMemory()
    {
        if (!isInitialized)
        {
            Debug.LogWarning("[CameraIntrinsics] Memory not initialized, cannot write intrinsics");
            return;
        }

        try
        {
            int offset = 0;

            // Write fx
            accessor.Write(offset, fx);
            offset += 4;

            // Write fy
            accessor.Write(offset, fy);
            offset += 4;

            // Write cx
            accessor.Write(offset, cx);
            offset += 4;

            // Write cy
            accessor.Write(offset, cy);
            offset += 4;

            // Write width
            accessor.Write(offset, (float)width);
            offset += 4;

            // Write height
            accessor.Write(offset, (float)height);

            if (debugMode)
            {
                Debug.Log("[CameraIntrinsics] Successfully wrote intrinsics to shared memory");
            }
        }
        catch (Exception e)
        {
            Debug.LogError($"[CameraIntrinsics] Failed to write to memory: {e.Message}");
        }
    }

    void Update()
    {
        // Update intrinsics if camera resolution or FOV changes
        if (targetCamera != null && isInitialized)
        {
            int currentWidth = targetCamera.pixelWidth;
            int currentHeight = targetCamera.pixelHeight;

            // Check if resolution changed
            if (currentWidth != width || currentHeight != height)
            {
                if (debugMode)
                {
                    Debug.Log($"[CameraIntrinsics] Resolution changed: {width}x{height} -> {currentWidth}x{currentHeight}");
                }

                ComputeIntrinsics();
                WriteToMemory();
            }
        }
    }

    void OnDestroy()
    {
        CleanupMemory();
    }

    void OnApplicationQuit()
    {
        CleanupMemory();
    }

    void CleanupMemory()
    {
        if (accessor != null)
        {
            accessor.Dispose();
            accessor = null;
        }

        if (mmf != null)
        {
            mmf.Dispose();
            mmf = null;
        }

        isInitialized = false;

        if (debugMode)
        {
            Debug.Log("[CameraIntrinsics] Shared memory cleaned up");
        }
    }

    // Public method to get intrinsics programmatically
    public void GetIntrinsics(out float outFx, out float outFy, out float outCx, out float outCy, out int outWidth, out int outHeight)
    {
        outFx = fx;
        outFy = fy;
        outCx = cx;
        outCy = cy;
        outWidth = width;
        outHeight = height;
    }
}
