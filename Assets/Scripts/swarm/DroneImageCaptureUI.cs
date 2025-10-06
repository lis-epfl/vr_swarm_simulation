using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;
using System.IO;

public class DroneImageCaptureUI : MonoBehaviour
{
    [Header("UI References")]
    public Button captureButton;
    public Text statusText;
    
    [Header("Capture Settings")]
    public NBV nbvScript; // Drag your NBV script here
    public string captureFolder = "DroneCaptures";
    public int imageWidth = 1920;
    public int imageHeight = 1080;
    
    void Start()
    {
        if (captureButton != null)
        {
            captureButton.onClick.AddListener(OnCaptureButtonPressed);
        }
        
        if (statusText != null)
        {
            statusText.text = "Ready to capture";
        }
    }
    
    void OnCaptureButtonPressed()
    {
        if (nbvScript != null)
        {
            statusText.text = "Capturing images...";
            StartCoroutine(CaptureAllDroneImages());
        }
    }
    
    IEnumerator CaptureAllDroneImages()
    {
        Debug.Log("Starting image capture for all drones...");
        
        // Create capture folder if it doesn't exist
        string folderPath = Path.Combine(Application.dataPath, captureFolder);
        if (!Directory.Exists(folderPath))
        {
            Directory.CreateDirectory(folderPath);
        }

        // Capture image from each drone
        for (int i = 0; i < nbvScript.swarm.Count; i++)
        {
            GameObject drone = nbvScript.swarm[i];
            Camera droneCamera = drone.transform.Find("FPV")?.GetComponent<Camera>();
            
            if (droneCamera != null)
            {
                string imagePath = Path.Combine(folderPath, $"drone_{i}_capture_{System.DateTime.Now:yyyyMMdd_HHmmss}.png");
                CaptureImage(droneCamera, imagePath, i);
                
                // Wait a frame between captures
                yield return null;
            }
            else
            {
                Debug.LogWarning($"No FPV camera found on drone {i}");
            }
        }

        Debug.Log($"Image capture complete! Images saved to: {folderPath}");
        
        // Update status
        statusText.text = "Capture complete!";
        yield return new WaitForSeconds(2f);
        statusText.text = "Ready to capture";
        
        // Optional: Call Python script
        CallPythonBoundingBoxScript(folderPath);
    }

    void CaptureImage(Camera camera, string filePath, int droneIndex)
    {
        // Create a render texture
        RenderTexture renderTexture = new RenderTexture(imageWidth, imageHeight, 24);
        RenderTexture currentRT = RenderTexture.active;
        
        // Set camera to render to our texture
        camera.targetTexture = renderTexture;
        camera.Render();
        
        // Read pixels from render texture
        RenderTexture.active = renderTexture;
        Texture2D image = new Texture2D(imageWidth, imageHeight, TextureFormat.RGB24, false);
        image.ReadPixels(new Rect(0, 0, imageWidth, imageHeight), 0, 0);
        image.Apply();
        
        // Save to file
        byte[] imageBytes = image.EncodeToPNG();
        File.WriteAllBytes(filePath, imageBytes);
        
        // Cleanup
        camera.targetTexture = null;
        RenderTexture.active = currentRT;
        DestroyImmediate(renderTexture);
        DestroyImmediate(image);
        
        Debug.Log($"Captured image from drone {droneIndex}: {filePath}");
    }

    // Optional: Call Python script for bounding box detection
    void CallPythonBoundingBoxScript(string imageFolderPath)
    {
        try
        {
            System.Diagnostics.Process pythonProcess = new System.Diagnostics.Process();
            pythonProcess.StartInfo.FileName = "python3"; // or "python" depending on your system
            pythonProcess.StartInfo.Arguments = $"bounding_box_detection.py \"{imageFolderPath}\"";
            pythonProcess.StartInfo.UseShellExecute = false;
            pythonProcess.StartInfo.RedirectStandardOutput = true;
            pythonProcess.StartInfo.CreateNoWindow = true;
            pythonProcess.StartInfo.WorkingDirectory = Application.dataPath; // Set working directory
            
            pythonProcess.Start();
            string output = pythonProcess.StandardOutput.ReadToEnd();
            pythonProcess.WaitForExit();
            
            Debug.Log($"Python script output: {output}");
        }
        catch (System.Exception e)
        {
            Debug.LogError($"Failed to run Python script: {e.Message}");
        }
    }
}