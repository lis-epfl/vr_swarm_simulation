using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.IO.MemoryMappedFiles;
using System.Runtime.InteropServices;

[RequireComponent(typeof(MeshFilter), typeof(MeshRenderer))]
public class PyUniSharing : MonoBehaviour
{

    private MemoryMappedFile batchMMF, processedMMF;
    private MemoryMappedViewAccessor batchAccessor, processedAccessor;
    private const int batchFlagPosition = 0;
    public const int imageCount = 16; 
    public const int imageWidth = 225, imageHeight = 225;
    private const int imageSize = imageWidth * imageHeight * 3;
    private const int batchDataPosition = 4; 
    private const int totalBatchSize = batchDataPosition + imageCount * imageSize;
    
    private const int processedFlagPosition = 0;
    private const int processedDataPosition = 4;
    private const int processedImageSize = 225 * 225 * 3;
    private const int totalProcessedSize = processedDataPosition + processedImageSize;
    public List<Camera> camerasToCapture;

    private float sendInterval = 0.02f;
    private float readInterval = 0.02f;
    private float nextSendTime, nextReceiveTime = 0f;


    // Parameters for screen in front of the pilot
    public float radius = 5f;
    public float angleRange = 90f;
    public int segments = 20;
    public float height = 3f;
    private Material curvedScreenMaterial;
    public Texture2D panoTexture;
    public bool resize_dimension = false;

    // Record time instance to debug

    private float lastWritingTime = 0.0f;
    private float lastReadingTime = 0.0f;

    // Start is called before the first frame update
    void Start()
    {
        // Two memories, one for writing new incoming images and one to read processed image
        batchMMF = MemoryMappedFile.CreateOrOpen("BatchSharedMemory", totalBatchSize);
        processedMMF = MemoryMappedFile.CreateOrOpen("ProcessedImageSharedMemory", totalProcessedSize);
        batchAccessor = batchMMF.CreateViewAccessor();
        processedAccessor = processedMMF.CreateViewAccessor();

        // FInd the FPV cameras
        FindCameras();

        // Create the cylindrical view
        GenerateCurvedScreen();
        
        // Get the material attached to the MeshRenderer
        curvedScreenMaterial = GetComponent<MeshRenderer>().material;
        nextSendTime = Time.time;

    }

    // Update is called once per frame
    // flagPosition = 0 : Nobody is ready/writing
    // flagPosition = 1 : Unity/Python is writing/reading
    // Warning: Python should flip the images vertically
    void Update()
    {
        
        if (resize_dimension)
        {
            GenerateCurvedScreen();
        }

        if (camerasToCapture.Count == 0)
        {
                // Debug.LogWarning("No cameras found to capture images.");
                FindCameras();
        }
        else if (Time.time >= nextSendTime && batchAccessor.ReadInt32(batchFlagPosition) == 0){
            
            batchAccessor.Write(batchFlagPosition, 1);

            // Debug.Log($"Time taken from last writing: {Time.time-lastWritingTime} seconds");
            lastWritingTime = Time.time;

            // Create and write batch of images
            // Write a batch of images (simulated data)
            for (int i = 0; i < camerasToCapture.Count && i < imageCount; i++)
            {
                byte[] imageBytes = CaptureCameraImage(camerasToCapture[i], imageWidth, imageHeight);
                
                if (imageBytes != null)
                {
                    batchAccessor.WriteArray(batchDataPosition + i * imageSize, imageBytes, 0, imageBytes.Length);
                }
            }
            // Python and Unity can read now
            batchAccessor.Write(batchFlagPosition, 0);

            // Debug.Log($"Time to write a batch: {Time.time-lastWritingTime} seconds");
            nextSendTime += sendInterval;
        }

        if (Time.time >= nextReceiveTime && processedAccessor.ReadInt32(processedFlagPosition) == 0){
            
            processedAccessor.Write(processedFlagPosition, 1);

            // Debug.Log($"Time taken from last reading: {Time.time-lastReadingTime} seconds");
            lastReadingTime = Time.time;

            byte[] processedImageBytes;
            
            processedImageBytes = ReceiveProcessedImage();

            processedAccessor.Write(processedFlagPosition, 0);

            SetPanoramaImage(processedImageBytes);
            
            // Debug.Log($"Time to read an image: {Time.time-lastReadingTime} seconds");
            

            // Python and Unity can read now
            nextReceiveTime += readInterval;
        }
        
    }

    // Capture the image from the camera and return it as a byte array (RGB format)
    private byte[] CaptureCameraImage(Camera camera, int width, int height)
    {
        // Create a RenderTexture to capture the camera's view, without altering the simulation camera's RenderTexture
        RenderTexture previousRT = camera.targetTexture;
        RenderTexture renderTexture = new RenderTexture(width, height, 24);
        camera.targetTexture = renderTexture;
        RenderTexture.active = renderTexture;
        camera.Render();

        
        Texture2D image = new Texture2D(width, height, TextureFormat.RGB24, false);
        image.ReadPixels(new Rect(0, 0, width, height), 0, 0);
        image.Apply();

        // Convert the Texture2D to a byte array (RGB)
        byte[] imageBytes = image.GetRawTextureData();

        // Clean up
        camera.targetTexture = previousRT;
        RenderTexture.active = null;
        Destroy(renderTexture);
        Destroy(image);

        return imageBytes;
    }

    private void FindCameras()
    {
        camerasToCapture = new List<Camera>();

        // Find all GameObjects in the scene with the tag "DroneBase"
        GameObject[] drones = GameObject.FindGameObjectsWithTag("DroneBase"); // Use "DroneBase" tag
        
        Debug.Log($"Drones found: {drones.Length}"); // Log number of drones found
        
        foreach (GameObject drone in drones)
        {
            Debug.Log($"Checking drone: {drone.name}");
            Camera camera = drone.transform.Find("FPV")?.GetComponent<Camera>();
            if (camera != null)
            {
                camerasToCapture.Add(camera);
                Debug.Log($"Found camera: {camera.name} in {drone.name}");
            }
            else
            {
                Debug.LogWarning($"No camera found in {drone.name}");
            }
        }

        Debug.Log($"Total cameras found: {camerasToCapture.Count}");
    }

    byte[] ReceiveProcessedImage()
    {
        

        byte[] processedImageBytes = new byte[processedImageSize];
        processedAccessor.ReadArray(processedDataPosition, processedImageBytes, 0, processedImageBytes.Length);
        
        return processedImageBytes;
    }

    void OnDestroy()
    {
        batchAccessor.Dispose();
        processedAccessor.Dispose();
        batchMMF.Dispose();
        processedMMF.Dispose();
    }

    private void GenerateCurvedScreen()
    {
        MeshFilter meshFilter = GetComponent<MeshFilter>();
        Mesh mesh = new Mesh();

        int vertCount = (segments + 1) * 2;
        Vector3[] vertices = new Vector3[vertCount];
        Vector2[] uvs = new Vector2[vertCount];
        int[] triangles = new int[segments * 6];// Each segment makes 2 triangles -> 6 parameters

        float angleStep = angleRange / segments;
        float halfHeight = height / 2f;

        for (int i = 0; i <= segments; i++)
        {
            float angle = Mathf.Deg2Rad * (-angleRange / 2 + i * angleStep);
            float x = Mathf.Sin(angle) * radius;
            float z = Mathf.Cos(angle) * radius;

            // Bottom vertices
            vertices[i * 2] = new Vector3(x, -halfHeight, z);
            uvs[i * 2] = new Vector2(i / (float)segments, 0);

            // Top vertices
            vertices[i * 2 + 1] = new Vector3(x, halfHeight, z);
            uvs[i * 2 + 1] = new Vector2(i / (float)segments, 1);

            // Create triangles
            if (i < segments)
            {
                int triangleOffset = i * 6;
                triangles[triangleOffset] = i * 2;
                triangles[triangleOffset + 1] = (i * 2) + 1;
                triangles[triangleOffset + 2] = (i * 2) + 2;

                triangles[triangleOffset + 3] = (i * 2) + 2;
                triangles[triangleOffset + 4] = (i * 2) + 1;
                triangles[triangleOffset + 5] = (i * 2) + 3;
            }
        }

        mesh.vertices = vertices;
        mesh.triangles = triangles;
        mesh.uv = uvs;
        mesh.RecalculateNormals();
        meshFilter.mesh = mesh;
    }

    // Function to load the panorama part from byte[]
    public void SetPanoramaImage(byte[] partPanorama)
    {
        // Convert the byte array to Texture2D
        // Texture2D panoramaTexture = LoadTextureFromBytes(partPanorama);
        Texture2D panoramaTexture = LoadRawRGBTexture(partPanorama, imageWidth, imageHeight);

        // Assign the texture to the material's main texture
        curvedScreenMaterial.mainTexture = panoramaTexture;

        panoTexture = panoramaTexture;
    }   
    
    // Utility function to convert byte[] to Texture2D
    public Texture2D LoadTextureFromBytes(byte[] imageData)
    {
        Texture2D texture = new Texture2D(2, 2);  // Placeholder dimensions; it will resize after LoadImage
        bool isLoaded = texture.LoadImage(imageData);

        return texture;
    }

    public Texture2D LoadRawRGBTexture(byte[] imageData, int width, int height)
    {
        // Check that the byte array length matches the expected size
        if (imageData.Length != width * height * 3)
        {
            Debug.LogError($"Raw image data size does not match expected size for {width}x{height} texture.");
            return null;
        }

        // Create a new Texture2D
        Texture2D texture = new Texture2D(width, height, TextureFormat.RGB24, false);

        // Set pixel data manually
        Color32[] pixels = new Color32[width * height];

        for (int i = 0; i < pixels.Length; i++)
        {
            int byteIndex = i * 3;
            byte r = imageData[byteIndex];
            byte g = imageData[byteIndex + 1];
            byte b = imageData[byteIndex + 2];

            pixels[i] = new Color32(r, g, b, 255); // Set RGB and Alpha = 255
        }

        // Apply the pixels to the texture
        texture.SetPixels32(pixels);
        texture.Apply();

        return texture;
    }

    public void TestSurface(Texture2D panoTexture)
    {
        // Assign the texture to the material's main texture
        curvedScreenMaterial.mainTexture = panoTexture;
    }
}
