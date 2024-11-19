using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using UnityEngine;

[RequireComponent(typeof(MeshFilter), typeof(MeshRenderer))]
public class PyUniSharingFast : MonoBehaviour
{   
    [SerializeField]
    private string batchMapName = "BatchSharedMemory";

    [SerializeField]
    private string processedMapName = "ProcessedImageSharedMemory";

    [SerializeField]
    private int batchImageWidth = 300;

    [SerializeField]
    private int batchImageHeight = 300;

    [SerializeField]
    private int processedImageWidth = 600;

    [SerializeField]
    private int processedImageHeight = 400;

    [SerializeField]
    private float sendInterval = 0.05f;

    [SerializeField]
    private float readInterval = 0.05f;

    private int batchImageCount = 2;
    private int batchImageSize;
    private int boolListSize;
    private int batchDataPosition;
    private int totalBatchSize;

    private int processedImageSize;
    private int totalProcessedSize;

    private IntPtr batchFileMap;
    private IntPtr batchPtr;
    private IntPtr processedFileMap;
    private IntPtr processedPtr;

    public List<Camera> camerasToCapture;
    public List<bool> camerasToStitch;

    private RenderTexture reusableTexture;
    private Texture2D image;
    private byte[] batchImageBuffer;
    private float nextSendTime, nextReceiveTime = 0f;
    private Color32[] pixels;

    [DllImport("kernel32.dll", SetLastError = true, CharSet = CharSet.Auto)]
    private static extern IntPtr CreateFileMapping(IntPtr hFile, IntPtr lpFileMappingAttributes, uint flProtect, uint dwMaximumSizeHigh, uint dwMaximumSizeLow, string lpName);

    [DllImport("kernel32.dll", SetLastError = true)]
    private static extern IntPtr MapViewOfFile(IntPtr hFileMappingObject, uint dwDesiredAccess, uint dwFileOffsetHigh, uint dwFileOffsetLow, UIntPtr dwNumberOfBytesToMap);

    [DllImport("kernel32.dll", SetLastError = true)]
    private static extern bool UnmapViewOfFile(IntPtr lpBaseAddress);

    [DllImport("kernel32.dll", SetLastError = true)]
    private static extern bool CloseHandle(IntPtr hObject);

    // Constant values

    private const uint FILE_MAP_ALL_ACCESS = 0xF001F;
    private const uint PAGE_READWRITE = 0x04;
    
    private const int FlagPosition = 0;
    private const int camerasToStitchPosition = 1;
    private const int processedDataPosition = 4;


    // Parameters for screen in front of the pilot
    public float radius = 5f;
    public float angleRange = 90f;
    public int segments = 20;
    public float height = 3f;
    private Material curvedScreenMaterial;
    private Texture2D panoTexture;
    public bool resize_dimension = false;

    // Other timing values to check the number of camera in the batch
    private float cameraUpdateInterval = 5f; // Time interval to update cameras
    private float nextCameraUpdateTime = 0f; // Next time to update cameras

    void Start()
    {
        CalculateMemorySizes();
        CreateMemoryMaps();

        FindCameras();
        GenerateCurvedScreen();
        curvedScreenMaterial = GetComponent<MeshRenderer>().material;

        reusableTexture = new RenderTexture(batchImageWidth, batchImageHeight, 24);
        image = new Texture2D(batchImageWidth, batchImageHeight, TextureFormat.RGB24, false);
        batchImageBuffer = new byte[batchImageCount * batchImageSize];
        panoTexture = new Texture2D(processedImageWidth, processedImageHeight, TextureFormat.RGB24, false);
        pixels = new Color32[processedImageWidth * processedImageHeight];
        nextSendTime = Time.time;
    }

    void Update()
    {

        if (Time.time >= nextCameraUpdateTime)
        {
            UpdateCameras();
            nextCameraUpdateTime = Time.time + cameraUpdateInterval; // Schedule the next update
        }

        if (resize_dimension)
        {
            GenerateCurvedScreen();
        } 

        if (camerasToCapture.Count == 0)
        {
            FindCameras();
            return;
        }
       

        else if (Time.time >= nextSendTime && Marshal.ReadByte(batchPtr, FlagPosition) == 0)
        {
            Marshal.WriteByte(batchPtr, FlagPosition, 1);

            for (int i = 0; i < boolListSize; i++)
            {
                byte value = (byte)(i < camerasToStitch.Count && camerasToStitch[i] ? 1 : 0);
                Marshal.WriteByte(batchPtr, camerasToStitchPosition + i, value);
            }

            for (int i = 0; i < camerasToCapture.Count && i < batchImageCount; i++)
            {
                if (i < camerasToStitch.Count && camerasToStitch[i])
                {
                    byte[] imageBytes = CaptureCameraImage(camerasToCapture[i]);
                    if (imageBytes != null)
                    {
                        Array.Copy(imageBytes, 0, batchImageBuffer, i * batchImageSize, imageBytes.Length);
                    }
                }
            }

            Marshal.Copy(batchImageBuffer, 0, IntPtr.Add(batchPtr, batchDataPosition), batchImageBuffer.Length);
            Marshal.WriteByte(batchPtr, FlagPosition, 0);
            nextSendTime += sendInterval;
        }

        if (Time.time >= nextReceiveTime && Marshal.ReadInt32(processedPtr, FlagPosition) == 0)
        {
            Marshal.WriteInt32(processedPtr, FlagPosition, 1);

            byte[] processedImageBytes = ReceiveProcessedImage();
            Marshal.WriteInt32(processedPtr, FlagPosition, 0);
            SetPanoramaImage(processedImageBytes);

            nextReceiveTime += readInterval;
        }
    }

    private byte[] CaptureCameraImage(Camera camera)
    {
        RenderTexture previousRT = camera.targetTexture;
        camera.targetTexture = reusableTexture;
        RenderTexture.active = reusableTexture;

        camera.Render();
        image.ReadPixels(new Rect(0, 0, batchImageWidth, batchImageHeight), 0, 0, false);
        image.Apply(false);

        byte[] imageBytes = image.GetRawTextureData();
        camera.targetTexture = previousRT;
        RenderTexture.active = null;

        return imageBytes;
    }

    byte[] ReceiveProcessedImage()
    {
        byte[] processedImageBytes = new byte[processedImageSize];
        Marshal.Copy(IntPtr.Add(processedPtr, processedDataPosition), processedImageBytes, 0, processedImageBytes.Length);
        return processedImageBytes;
    }

    private void GenerateCurvedScreen()
    {
        MeshFilter meshFilter = GetComponent<MeshFilter>();
        Mesh mesh = new Mesh();
        int vertCount = (segments + 1) * 2;
        Vector3[] vertices = new Vector3[vertCount];
        Vector2[] uvs = new Vector2[vertCount];
        int[] triangles = new int[segments * 6];

        float angleStep = angleRange / segments;
        float halfHeight = height / 2f;

        for (int i = 0; i <= segments; i++)
        {
            float angle = Mathf.Deg2Rad * (-angleRange / 2 + i * angleStep);
            float x = Mathf.Sin(angle) * radius;
            float z = Mathf.Cos(angle) * radius;

            vertices[i * 2] = new Vector3(x, -halfHeight, z);
            uvs[i * 2] = new Vector2(i / (float)segments, 0);
            vertices[i * 2 + 1] = new Vector3(x, halfHeight, z);
            uvs[i * 2 + 1] = new Vector2(i / (float)segments, 1);

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

    public void SetPanoramaImage(byte[] partPanorama)
    {
        // panoTexture = LoadRawRGBTexture(partPanorama);
        LoadRawRGBTexture(partPanorama);
        curvedScreenMaterial.mainTexture = panoTexture;
    }

    public void LoadRawRGBTexture(byte[] imageData)
    {
        // panoTexture = new Texture2D(processedImageWidth, processedImageHeight, TextureFormat.RGB24, false);
        // Color32[] pixels = new Color32[processedImageWidth * processedImageHeight];

        for (int i = 0; i < pixels.Length; i++)
        {
            int byteIndex = i * 3;
            pixels[i] = new Color32(imageData[byteIndex], imageData[byteIndex + 1], imageData[byteIndex + 2], 255);
        }

        panoTexture.SetPixels32(pixels);
        panoTexture.Apply();
        // return panoTexture;
    }

    private void FindCameras()
    {
        camerasToCapture = new List<Camera>();

        // Find all GameObjects in the scene with the tag "DroneBase"
        GameObject[] drones = GameObject.FindGameObjectsWithTag("DroneBase"); // Use "DroneBase" tag
        
        // Debug.Log($"Drones found: {drones.Length}"); // Log number of drones found
        
        foreach (GameObject drone in drones)
        {
            // Debug.Log($"Checking drone: {drone.name}");
            Camera camera = drone.transform.Find("FPV")?.GetComponent<Camera>();
            // AttitudeControl attitudeScript = drone.transform.Find(droneParent).GetComponent<AttitudeControl>();
            // bool estimate = attitudeScript.boundaryEstimate

            if (camera != null)
            {
                camerasToCapture.Add(camera);
                // Debug.Log($"Found camera: {camera.name} in {drone.name}");
            }
            // else
            // {
            //     Debug.LogWarning($"No camera found in {drone.name}");
            // }
        }

        // Debug.Log($"Total cameras found: {camerasToCapture.Count}");
    }

    private void UpdateCameraToStitch()
    {
        camerasToStitch = new List<bool>();

        // Find all GameObjects in the scene with the tag "DroneBase"
        GameObject[] drones = GameObject.FindGameObjectsWithTag("DroneBase"); // Use "DroneBase" tag
                
        foreach (GameObject drone in drones)
        {
            AttitudeControl attitudeScript = drone.transform.Find("DroneParent").GetComponent<AttitudeControl>();

            if (attitudeScript != null)
            {
                bool estimate = attitudeScript.boundaryEstimate;
                camerasToStitch.Add(estimate);
            }
            else
            {
                Debug.LogWarning($"No estimate found in {drone.name}");
            }
        }

        // Debug.Log($"Total estimate found: {camerasToStitch.Count}");
    }

    private void CalculateMemorySizes()
    {
        // Calculate memory sizes based on configurable parameters
        batchImageSize = batchImageWidth * batchImageHeight * 3;
        boolListSize = batchImageCount;
        batchDataPosition = boolListSize + camerasToStitchPosition;
        totalBatchSize = batchDataPosition + batchImageCount * batchImageSize;

        processedImageSize = processedImageWidth * processedImageHeight * 3;
        totalProcessedSize = processedDataPosition + processedImageSize; 
    }

    private void CreateMemoryMaps()
    {
        // Destroy any existing memory maps before recreating
        DestroyMemoryMaps();

        // Create memory-mapped files
        batchFileMap = CreateFileMapping(new IntPtr(-1), IntPtr.Zero, PAGE_READWRITE, 0, (uint)totalBatchSize , batchMapName);
        processedFileMap = CreateFileMapping(new IntPtr(-1), IntPtr.Zero, PAGE_READWRITE, 0, (uint)totalProcessedSize, processedMapName);

        if (batchFileMap == IntPtr.Zero || processedFileMap == IntPtr.Zero)
        {
            Debug.LogError("Unable to create memory-mapped files.");
            return;
        }

        batchPtr = MapViewOfFile(batchFileMap, FILE_MAP_ALL_ACCESS, 0, 0, (UIntPtr)totalBatchSize);
        processedPtr = MapViewOfFile(processedFileMap, FILE_MAP_ALL_ACCESS, 0, 0, (UIntPtr)totalProcessedSize);

        if (batchPtr == IntPtr.Zero || processedPtr == IntPtr.Zero)
        {
            Debug.LogError("Unable to map view of file.");
            DestroyMemoryMaps();
        }
    }

    private void DestroyMemoryMaps()
    {
        if (batchPtr != IntPtr.Zero)
        {
            UnmapViewOfFile(batchPtr);
            batchPtr = IntPtr.Zero;
        }
        if (processedPtr != IntPtr.Zero)
        {
            UnmapViewOfFile(processedPtr);
            processedPtr = IntPtr.Zero;
        }
        if (batchFileMap != IntPtr.Zero)
        {
            CloseHandle(batchFileMap);
            batchFileMap = IntPtr.Zero;
        }
        if (processedFileMap != IntPtr.Zero)
        {
            CloseHandle(processedFileMap);
            processedFileMap = IntPtr.Zero;
        }
    }

    private void OnValidate()
    {
        // Recalculate memory sizes
        CalculateMemorySizes();

        // Recreate memory maps to reflect changes
        CreateMemoryMaps();

        // Update reusable resources
        reusableTexture = new RenderTexture(batchImageWidth, batchImageHeight, 24);
        image = new Texture2D(batchImageWidth, batchImageHeight, TextureFormat.RGB24, false);
        batchImageBuffer = new byte[batchImageCount * batchImageSize];
        boolListSize = batchImageCount;

        panoTexture = new Texture2D(processedImageWidth, processedImageHeight, TextureFormat.RGB24, false);
        pixels = new Color32[processedImageWidth * processedImageHeight];

        Debug.Log("Parameters updated: Memory sizes and resources recalculated.");
    }

    private void ValidateTextures()
    {
        if (reusableTexture == null || reusableTexture.width != batchImageWidth || reusableTexture.height != batchImageHeight)
        {
            reusableTexture?.Release();
            reusableTexture = new RenderTexture(batchImageWidth, batchImageHeight, 24);
        }

        if (image == null || image.width != batchImageWidth || image.height != batchImageHeight)
        {
            Destroy(image);
            image = new Texture2D(batchImageWidth, batchImageHeight, TextureFormat.RGB24, false);
        }
    }

    private void UpdateCameras()
    {
        FindCameras();

        int newBatchImageCount = camerasToCapture.Count;
        if (newBatchImageCount != batchImageCount)
        {
            batchImageCount = newBatchImageCount;

            int previousTotalBatchSize = totalBatchSize;
            CalculateMemorySizes();

            if (totalBatchSize != previousTotalBatchSize)
            {
                CreateMemoryMaps();
            }

            batchImageBuffer = new byte[batchImageCount * batchImageSize];
            UpdateCameraToStitch();
            ValidateTextures(); // Ensure textures are updated
            Debug.Log($"Updated cameras. Found {batchImageCount} cameras.");
        }
    }

    void OnDestroy()
    {
        UnmapViewOfFile(batchPtr);
        UnmapViewOfFile(processedPtr);
        CloseHandle(batchFileMap);
        CloseHandle(processedFileMap);
        DestroyMemoryMaps();
    }
}
