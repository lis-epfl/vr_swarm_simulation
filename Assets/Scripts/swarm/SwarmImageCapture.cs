using System;
using System.Collections.Generic;
using System.IO;
using UnityEngine;

/// <summary>
/// SwarmImageCapture.cs - Olfati-Saber Swarm Image Capture System
/// 
/// Captures RGB and depth images from all drones in the swarm at regular intervals.
/// Saves data to disk for Python processing (macOS compatible, no shared memory).
/// Only active when swarm mode is set to OLFATI_SABER.
/// 
/// Output Structure:
/// Assets/ProcessedImages/SwarmCapture/
/// ├── capture_YYYYMMDD_HHMMSS/
/// │   ├── metadata.json
/// │   ├── drone_0_rgb.png
/// │   ├── drone_0_depth.raw
/// │   ├── drone_0_pose.json
/// │   └── ...
/// </summary>
public class SwarmImageCapture : MonoBehaviour
{
    [Header("Capture Settings")]
    [SerializeField] private float captureInterval = 5.0f; // Capture every 5.0 seconds
    [SerializeField] private int imageWidth = 1280;
    [SerializeField] private int imageHeight = 720;
    
    [Header("Output Settings")]
    [SerializeField] private string outputFolder = "Assets/ProcessedImages/SwarmCapture";
    
    [Header("Swarm Mode Check")]
    [SerializeField] private bool onlyRunInOlfatiSaberMode = true;
    
    [Header("Debug")]
    [SerializeField] private bool enableDebugLogging = true;
    
    // Internal state
    private List<Camera> droneCameras;
    private List<DroneDepthCamera> droneDepthCameras;
    private List<Transform> droneTransforms;
    private SwarmManager swarmManager;
    private CameraIntrinsics cameraIntrinsics;
    
    // Capture resources
    private RenderTexture reusableTexture;
    private Texture2D captureTexture;
    private float nextCaptureTime;
    private bool isInitialized = false;
    private int captureCount = 0;
    
    void Start()
    {
        // Find SwarmManager
        swarmManager = FindObjectOfType<SwarmManager>();
        if (swarmManager == null)
        {
            Debug.LogWarning("[SwarmImageCapture] SwarmManager not found. Capture will run regardless of mode.");
        }
        
        // Initialize capture resources
        InitializeCaptureResources();
        
        // Find drone cameras
        FindDroneCameras();
        
        // Create output directory
        if (!Directory.Exists(outputFolder))
        {
            Directory.CreateDirectory(outputFolder);
            if (enableDebugLogging)
                Debug.Log($"[SwarmImageCapture] Created output folder: {outputFolder}");
        }
        
        nextCaptureTime = Time.time + captureInterval;
        isInitialized = true;
        
        if (enableDebugLogging)
            Debug.Log($"[SwarmImageCapture] Initialized with {droneCameras?.Count ?? 0} drones");
    }
    
    void Update()
    {
        if (!isInitialized)
            return;
        
        // Check if we should capture (only in Olfati-Saber mode if enabled)
        if (onlyRunInOlfatiSaberMode && swarmManager != null)
        {
            if (swarmManager.swarmAlgorithm != SwarmManager.SwarmAlgorithm.OLFATI_SABER)
            {
                return; // Not in Olfati-Saber mode, skip capture
            }
        }
        
        // Check if it's time to capture
        if (Time.time >= nextCaptureTime)
        {
            // Refresh drone list in case new drones spawned
            FindDroneCameras();
            
            if (droneCameras != null && droneCameras.Count > 0)
            {
                CaptureAndSaveData();
                nextCaptureTime = Time.time + captureInterval;
            }
            else if (enableDebugLogging)
            {
                Debug.LogWarning("[SwarmImageCapture] No drones found to capture");
            }
        }
    }
    
    void InitializeCaptureResources()
    {
        reusableTexture = new RenderTexture(imageWidth, imageHeight, 24);
        captureTexture = new Texture2D(imageWidth, imageHeight, TextureFormat.RGB24, false);
        
        if (enableDebugLogging)
            Debug.Log($"[SwarmImageCapture] Initialized capture resources ({imageWidth}x{imageHeight})");
    }
    
    void FindDroneCameras()
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
                    Debug.LogWarning($"[SwarmImageCapture] Drone {drone.name} missing DroneDepthCamera component!");
                continue;
            }
            
            // Force depth camera to match our capture resolution
            depthCamera.SetResolution(imageWidth, imageHeight);
            
            // Get camera intrinsics (only need to do this once)
            if (cameraIntrinsics == null)
            {
                cameraIntrinsics = fpvCamera.GetComponent<CameraIntrinsics>();
            }
            
            droneCameras.Add(fpvCamera);
            droneDepthCameras.Add(depthCamera);
            droneTransforms.Add(fpvCamera.transform);
        }
        
        if (enableDebugLogging && droneCameras.Count > 0)
            Debug.Log($"[SwarmImageCapture] Found {droneCameras.Count} drones with RGB+Depth cameras");
    }
    
    void CaptureAndSaveData()
    {
        string timestamp = DateTime.Now.ToString("yyyyMMdd_HHmmss");
        string captureFolder = Path.Combine(outputFolder, $"capture_{timestamp}");
        
        // Create capture folder
        Directory.CreateDirectory(captureFolder);
        
        if (enableDebugLogging)
            Debug.Log($"[SwarmImageCapture] Starting capture {captureCount} at {timestamp}");
        
        // Save metadata (camera intrinsics + drone count)
        SaveMetadata(captureFolder);
        
        // Capture and save data for each drone
        for (int i = 0; i < droneCameras.Count; i++)
        {
            try
            {
                // Capture RGB
                byte[] rgbData = CaptureCameraImage(droneCameras[i]);
                if (rgbData != null)
                {
                    string rgbPath = Path.Combine(captureFolder, $"drone_{i}_rgb.png");
                    SaveRGBImage(rgbData, rgbPath);
                }
                
                // Capture Depth
                byte[] depthData = CaptureDepthData(droneDepthCameras[i]);
                if (depthData != null)
                {
                    string depthPath = Path.Combine(captureFolder, $"drone_{i}_depth.raw");
                    File.WriteAllBytes(depthPath, depthData);
                }
                
                // Save Pose
                SaveDronePose(droneTransforms[i], Path.Combine(captureFolder, $"drone_{i}_pose.json"));
                
                if (enableDebugLogging)
                    Debug.Log($"[SwarmImageCapture]   Drone {i}: RGB, Depth, Pose saved");
            }
            catch (Exception e)
            {
                Debug.LogError($"[SwarmImageCapture] Error capturing drone {i}: {e.Message}");
            }
        }
        
        captureCount++;
        
        if (enableDebugLogging)
            Debug.Log($"[SwarmImageCapture] Capture complete: {captureFolder}");
    }
    
    void SaveMetadata(string captureFolder)
    {
        // Get camera intrinsics
        float fx = 0, fy = 0, cx = 0, cy = 0;
        int width = imageWidth, height = imageHeight;
        
        if (cameraIntrinsics != null)
        {
            cameraIntrinsics.GetIntrinsics(out fx, out fy, out cx, out cy, out width, out height);
        }
        else
        {
            // Fallback: compute from first camera
            if (droneCameras.Count > 0)
            {
                Camera cam = droneCameras[0];
                Matrix4x4 P = cam.projectionMatrix;
                fx = P[0, 0] * width / 2.0f;
                fy = P[1, 1] * height / 2.0f;
                cx = (1.0f - P[0, 2]) * width / 2.0f;
                cy = (1.0f - P[1, 2]) * height / 2.0f;
            }
        }
        
        // Create metadata JSON
        string json = $@"{{
    ""drone_count"": {droneCameras.Count},
    ""timestamp"": ""{DateTime.Now:yyyy-MM-dd HH:mm:ss}"",
    ""image_width"": {imageWidth},
    ""image_height"": {imageHeight},
    ""camera_intrinsics"": {{
        ""fx"": {fx},
        ""fy"": {fy},
        ""cx"": {cx},
        ""cy"": {cy}
    }}
}}";
        
        string metadataPath = Path.Combine(captureFolder, "metadata.json");
        File.WriteAllText(metadataPath, json);
    }
    
    void SaveDronePose(Transform droneTransform, string filePath)
    {
        Vector3 pos = droneTransform.position;
        Quaternion rot = droneTransform.rotation;
        
        string json = $@"{{
    ""position"": {{
        ""x"": {pos.x},
        ""y"": {pos.y},
        ""z"": {pos.z}
    }},
    ""quaternion"": {{
        ""x"": {rot.x},
        ""y"": {rot.y},
        ""z"": {rot.z},
        ""w"": {rot.w}
    }}
}}";
        
        File.WriteAllText(filePath, json);
    }
    
    byte[] CaptureCameraImage(Camera camera)
    {
        if (camera == null || reusableTexture == null || captureTexture == null)
            return null;
        
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
            Debug.LogError($"[SwarmImageCapture] Failed to capture RGB: {e.Message}");
            return null;
        }
    }
    
    void SaveRGBImage(byte[] rgbData, string filePath)
    {
        // Convert raw RGB data to PNG
        Texture2D tempTexture = new Texture2D(imageWidth, imageHeight, TextureFormat.RGB24, false);
        tempTexture.LoadRawTextureData(rgbData);
        tempTexture.Apply();
        
        byte[] pngData = tempTexture.EncodeToPNG();
        File.WriteAllBytes(filePath, pngData);
        
        Destroy(tempTexture);
    }
    
    byte[] CaptureDepthData(DroneDepthCamera depthCamera)
    {
        if (depthCamera == null)
            return null;
        
        try
        {
            // Get depth data as byte array (float32 format)
            byte[] depthBytes = depthCamera.GetDepthDataBytes();
            return depthBytes;
        }
        catch (Exception e)
        {
            Debug.LogError($"[SwarmImageCapture] Failed to capture depth: {e.Message}");
            return null;
        }
    }
    
    void OnDestroy()
    {
        if (reusableTexture != null)
        {
            reusableTexture.Release();
            Destroy(reusableTexture);
        }
        
        if (captureTexture != null)
        {
            Destroy(captureTexture);
        }
    }
    
    void OnApplicationQuit()
    {
        OnDestroy();
    }
    
    // Public API
    public int GetDroneCount()
    {
        return droneCameras?.Count ?? 0;
    }
    
    public int GetCaptureCount()
    {
        return captureCount;
    }
}
