using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.IO;
using System;

/// <summary>
/// Continuously captures images from drone FPV cameras and saves them for OpenCV processing.
/// Captures images continuously but only saves/processes them at specified intervals.
/// </summary>
public class ContinuousImageCapture : MonoBehaviour
{
    [Header("Capture Settings")]
    [SerializeField] private NBV nbvScript; // Reference to NBV script (auto-found at runtime)
    
    // Public property to access nbvScript (for NBVScriptFinder and other scripts)
    public NBV NBVScript 
    { 
        get { return nbvScript; } 
        set { nbvScript = value; } 
    }
    [Header("Runtime Info")]
    [SerializeField] private bool nbvScriptFound = false;
    [SerializeField] private string nbvScriptLocation = "Searching...";
    [Header("Capture Configuration")]
    public float captureInterval = 2.0f; // Time between saves (in seconds)
    [Range(1, 20)]
    public int maxDronesPerCapture = 3; // Limit number of drones to capture per cycle
    public bool rotateThroughDrones = true; // If true, cycles through different drones each capture
    public int imageWidth = 640;
    public int imageHeight = 480;
    public string saveDirectory = "DroneImages"; // Folder relative to Assets
    
    [Header("Continuous Capture")]
    public bool enableContinuousCapture = true;
    public bool saveAllDroneImages = false; // Changed to false to reduce images
    
    [Header("OpenCV Integration")]
    public bool callOpenCVProcessing = false; // DISABLED for testing - just capture images
    public string pythonScriptPath = "opencv_processor.py"; // Path to your OpenCV Python script
    
    [Header("Debug")]
    public bool debugLogging = false;
    
    // Private variables
    private float nextCaptureTime = 0f;
    private int currentDroneOffset = 0; // For rotating through drones
    private List<Camera> droneCameras = new List<Camera>();
    private List<RenderTexture> renderTextures = new List<RenderTexture>();
    private List<Texture2D> captureTextures = new List<Texture2D>();
    private string fullSaveDirectory;
    
    // Image data storage for continuous access
    public List<byte[]> latestImageData = new List<byte[]>();
    public List<string> latestImagePaths = new List<string>();
    
    void Start()
    {
        // Setup save directory
        fullSaveDirectory = Path.Combine(Application.dataPath, saveDirectory);
        if (!Directory.Exists(fullSaveDirectory))
        {
            Directory.CreateDirectory(fullSaveDirectory);
            Debug.Log($"Created directory: {fullSaveDirectory}");
        }
        
        // Wait a bit for GameManager to initialize, then find NBV script
        StartCoroutine(InitializeWithDelay());
        
        if (debugLogging)
        {
            Debug.Log($"ContinuousImageCapture initialized. Capture interval: {captureInterval}s");
        }
    }
    
    IEnumerator InitializeWithDelay()
    {
        // Wait a few frames for GameManager/SwarmManager to set up NBV script
        yield return new WaitForSeconds(1f);
        
        // Try to find NBV script
        if (!FindNBVScript())
        {
            // If not found immediately, keep trying every second for up to 10 seconds
            for (int attempts = 0; attempts < 10; attempts++)
            {
                yield return new WaitForSeconds(1f);
                if (FindNBVScript())
                {
                    break;
                }
                Debug.LogWarning($"Still searching for NBV script... (attempt {attempts + 1}/10)");
            }
        }
        
        if (nbvScript != null)
        {
            // Initialize capture system once NBV is found
            InitializeCameraSystem();
            
            // Set initial capture time
            nextCaptureTime = Time.time + captureInterval;
            
            Debug.Log($"✅ ContinuousImageCapture successfully initialized with NBV script from {nbvScriptLocation}");
        }
        else
        {
            Debug.LogError("❌ Could not find NBV script after 10 seconds. Please check your GameManager setup.");
        }
    }
    
    bool FindNBVScript()
    {
        // Search for NBV script in the scene
        NBV[] nbvScripts = FindObjectsOfType<NBV>();
        
        if (nbvScripts.Length > 0)
        {
            nbvScript = nbvScripts[0];
            nbvScriptFound = true;
            nbvScriptLocation = nbvScript.gameObject.name;
            
            if (debugLogging)
            {
                Debug.Log($"Found NBV script on GameObject: {nbvScriptLocation}");
                Debug.Log($"Swarm count: {(nbvScript.swarm != null ? nbvScript.swarm.Count : 0)}");
            }
            
            return true;
        }
        
        nbvScriptFound = false;
        nbvScriptLocation = "Not found";
        return false;
    }
    
    void Update()
    {
        // Try to find NBV script if we don't have it yet
        if (nbvScript == null)
        {
            FindNBVScript();
            return;
        }
        
        if (!enableContinuousCapture)
            return;
            
        // Update camera list if needed (in case drones are added/removed)
        if (Time.time % 5f < 0.1f) // Update every 5 seconds
        {
            UpdateCameraList();
        }
        
        // Capture and save images at specified intervals
        if (Time.time >= nextCaptureTime)
        {
            StartCoroutine(CaptureAndProcessImages());
            nextCaptureTime = Time.time + captureInterval;
        }
    }
    
    void InitializeCameraSystem()
    {
        if (nbvScript == null)
        {
            Debug.LogError("NBV script reference is null!");
            return;
        }
        
        UpdateCameraList();
        CreateRenderTextures();
    }
    
    void UpdateCameraList()
    {
        if (nbvScript == null || nbvScript.swarm == null)
        {
            if (debugLogging)
            {
                Debug.LogWarning("NBV script or swarm list is null");
            }
            return;
        }
        
        droneCameras.Clear();
        
        // Find all drone cameras from the swarm
        foreach (GameObject drone in nbvScript.swarm)
        {
            if (drone == null) continue;
            
            Camera fpvCamera = drone.transform.Find("FPV")?.GetComponent<Camera>();
            if (fpvCamera != null)
            {
                droneCameras.Add(fpvCamera);
            }
        }
        
        if (debugLogging)
        {
            Debug.Log($"Found {droneCameras.Count} drone cameras from {nbvScript.swarm.Count} drones");
        }
        
        // Recreate render textures if camera count changed
        if (renderTextures.Count != droneCameras.Count)
        {
            CreateRenderTextures();
        }
    }
    
    void CreateRenderTextures()
    {
        // Clean up existing textures
        foreach (RenderTexture rt in renderTextures)
        {
            if (rt != null) rt.Release();
        }
        foreach (Texture2D tex in captureTextures)
        {
            if (tex != null) DestroyImmediate(tex);
        }
        
        renderTextures.Clear();
        captureTextures.Clear();
        latestImageData.Clear();
        latestImagePaths.Clear();
        
        // Create new textures for each camera
        for (int i = 0; i < droneCameras.Count; i++)
        {
            RenderTexture rt = new RenderTexture(imageWidth, imageHeight, 24);
            Texture2D tex = new Texture2D(imageWidth, imageHeight, TextureFormat.RGB24, false);
            
            renderTextures.Add(rt);
            captureTextures.Add(tex);
            latestImageData.Add(null);
            latestImagePaths.Add("");
        }
    }
    
    IEnumerator CaptureAndProcessImages()
    {
        if (droneCameras.Count == 0)
        {
            if (debugLogging) Debug.Log("No drone cameras found for capture");
            yield break;
        }
        
        List<string> capturedPaths = new List<string>();
        string timestamp = DateTime.Now.ToString("yyyyMMdd_HHmmss");
        
        // Determine which drones to capture from
        List<int> dronesToCapture = GetDronesToCapture();
        
        if (debugLogging)
        {
            Debug.Log($"Capturing from {dronesToCapture.Count} drones: [{string.Join(", ", dronesToCapture)}]");
        }
        
        // Capture images from selected cameras
        foreach (int droneIndex in dronesToCapture)
        {
            if (droneIndex >= droneCameras.Count) continue;
            
            Camera camera = droneCameras[droneIndex];
            if (camera == null) continue;
            
            // Capture the image
            byte[] imageData = CaptureImageFromCamera(camera, droneIndex);
            
            if (imageData != null)
            {
                // Save to file
                string filename = $"drone_{droneIndex}_{timestamp}.png";
                string fullPath = Path.Combine(fullSaveDirectory, filename);
                
                try
                {
                    File.WriteAllBytes(fullPath, imageData);
                    capturedPaths.Add(fullPath);
                    
                    // Store latest data for external access
                    if (droneIndex < latestImageData.Count)
                    {
                        latestImageData[droneIndex] = imageData;
                        latestImagePaths[droneIndex] = fullPath;
                    }
                    
                    if (debugLogging)
                    {
                        Debug.Log($"Captured image from drone {droneIndex}: {filename}");
                    }
                }
                catch (Exception e)
                {
                    Debug.LogError($"Failed to save image from drone {droneIndex}: {e.Message}");
                }
            }
            
            yield return null; // Wait a frame between captures
        }
        
        // DISABLED: OpenCV processing for testing - no batch files needed
        // Call OpenCV processing if enabled
        // if (callOpenCVProcessing && capturedPaths.Count > 0)
        // {
        //     ProcessImagesWithOpenCV(capturedPaths, timestamp);
        // }
        
        Debug.Log($"✅ Capture cycle completed. {capturedPaths.Count} images captured from {dronesToCapture.Count} selected drones.");
        Debug.Log($"📁 Images saved to: {fullSaveDirectory}");
    }
    
    List<int> GetDronesToCapture()
    {
        List<int> dronesToCapture = new List<int>();
        
        if (saveAllDroneImages)
        {
            // Capture from all drones (original behavior)
            for (int i = 0; i < droneCameras.Count; i++)
            {
                if (ShouldCaptureFromDrone(i))
                {
                    dronesToCapture.Add(i);
                }
            }
        }
        else
        {
            // Limit number of drones per capture
            int availableDrones = droneCameras.Count;
            int dronesToSelect = Mathf.Min(maxDronesPerCapture, availableDrones);
            
            if (rotateThroughDrones)
            {
                // Rotate through different drones each capture cycle
                for (int i = 0; i < dronesToSelect; i++)
                {
                    int droneIndex = (currentDroneOffset + i) % availableDrones;
                    if (ShouldCaptureFromDrone(droneIndex))
                    {
                        dronesToCapture.Add(droneIndex);
                    }
                }
                
                // Update offset for next cycle
                currentDroneOffset = (currentDroneOffset + dronesToSelect) % availableDrones;
            }
            else
            {
                // Just capture from the first few drones that meet criteria
                for (int i = 0; i < availableDrones && dronesToCapture.Count < maxDronesPerCapture; i++)
                {
                    if (ShouldCaptureFromDrone(i))
                    {
                        dronesToCapture.Add(i);
                    }
                }
            }
        }
        
        return dronesToCapture;
    }
    
    byte[] CaptureImageFromCamera(Camera camera, int cameraIndex)
    {
        if (cameraIndex >= renderTextures.Count || cameraIndex >= captureTextures.Count)
            return null;
            
        RenderTexture currentRT = RenderTexture.active;
        RenderTexture previousTarget = camera.targetTexture;
        
        try
        {
            // Set up camera to render to our texture
            camera.targetTexture = renderTextures[cameraIndex];
            RenderTexture.active = renderTextures[cameraIndex];
            
            // Render the camera
            camera.Render();
            
            // Read pixels from render texture
            captureTextures[cameraIndex].ReadPixels(new Rect(0, 0, imageWidth, imageHeight), 0, 0, false);
            captureTextures[cameraIndex].Apply(false);
            
            // Encode to PNG
            byte[] imageData = captureTextures[cameraIndex].EncodeToPNG();
            
            return imageData;
        }
        catch (Exception e)
        {
            Debug.LogError($"Error capturing image from camera {cameraIndex}: {e.Message}");
            return null;
        }
        finally
        {
            // Restore previous settings
            camera.targetTexture = previousTarget;
            RenderTexture.active = currentRT;
        }
    }
    
    bool ShouldCaptureFromDrone(int droneIndex)
    {
        // You can customize this logic based on your needs
        // For example, only capture from drones that are "active" or within certain bounds
        
        if (droneIndex >= nbvScript.swarm.Count)
            return false;
            
        // Example: Only capture if drone is within a certain distance from center
        Vector3 dronePos = nbvScript.swarm[droneIndex].transform.position;
        float distanceFromCenter = Vector3.Distance(dronePos, nbvScript.centerPoint);
        
        return distanceFromCenter <= nbvScript.radius * 2f; // Capture if within 2x radius
    }
    
    void ProcessImagesWithOpenCV(List<string> imagePaths, string timestamp)
    {
        if (imagePaths.Count == 0) return;
        
        try
        {
            // Create a batch file with all image paths
            string batchFilePath = Path.Combine(fullSaveDirectory, $"batch_{timestamp}.txt");
            File.WriteAllLines(batchFilePath, imagePaths);
            
            // Call Python script
            System.Diagnostics.Process pythonProcess = new System.Diagnostics.Process();
            pythonProcess.StartInfo.FileName = "python"; // or "python3"
            pythonProcess.StartInfo.Arguments = $"\"{pythonScriptPath}\" \"{batchFilePath}\"";
            pythonProcess.StartInfo.UseShellExecute = false;
            pythonProcess.StartInfo.RedirectStandardOutput = true;
            pythonProcess.StartInfo.RedirectStandardError = true;
            pythonProcess.StartInfo.CreateNoWindow = true;
            pythonProcess.StartInfo.WorkingDirectory = Application.dataPath;
            
            pythonProcess.Start();
            
            // Don't wait for completion to avoid blocking
            if (debugLogging)
            {
                Debug.Log($"Started OpenCV processing with {imagePaths.Count} images");
            }
        }
        catch (Exception e)
        {
            Debug.LogError($"Failed to start OpenCV processing: {e.Message}");
        }
    }
    
    // Public methods for external access
    public byte[] GetLatestImageData(int droneIndex)
    {
        if (droneIndex >= 0 && droneIndex < latestImageData.Count)
            return latestImageData[droneIndex];
        return null;
    }
    
    public string GetLatestImagePath(int droneIndex)
    {
        if (droneIndex >= 0 && droneIndex < latestImagePaths.Count)
            return latestImagePaths[droneIndex];
        return "";
    }
    
    public int GetDroneCameraCount()
    {
        return droneCameras.Count;
    }
    
    // Manual capture trigger
    [ContextMenu("Capture Images Now")]
    public void CaptureImagesNow()
    {
        if (Application.isPlaying)
        {
            StartCoroutine(CaptureAndProcessImages());
        }
    }
    
    // Manual NBV search trigger
    [ContextMenu("Search for NBV Script")]
    public void SearchForNBVScript()
    {
        if (FindNBVScript())
        {
            Debug.Log($"✅ Found NBV script on {nbvScriptLocation}");
            if (Application.isPlaying)
            {
                InitializeCameraSystem();
            }
        }
        else
        {
            Debug.LogWarning("❌ NBV script not found. Make sure GameManager has initialized the swarm.");
        }
    }
    
    void OnDestroy()
    {
        // Clean up resources
        foreach (RenderTexture rt in renderTextures)
        {
            if (rt != null) rt.Release();
        }
        foreach (Texture2D tex in captureTextures)
        {
            if (tex != null) DestroyImmediate(tex);
        }
    }
}