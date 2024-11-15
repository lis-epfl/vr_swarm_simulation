using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using UnityEngine;

[RequireComponent(typeof(MeshFilter), typeof(MeshRenderer))]
public class PyUniSharingFast : MonoBehaviour
{
    // WinAPI memory-mapped file parameters
    private const string BatchMapName = "BatchSharedMemory";
    private const string ProcessedMapName = "ProcessedImageSharedMemory";

    private IntPtr batchFileMap;
    private IntPtr batchPtr;
    private IntPtr processedFileMap;
    private IntPtr processedPtr;

    private const int batchFlagPosition = 0;
    public const int imageCount = 16;
    public const int imageWidth = 300, imageHeight = 300;
    private const int imageSize = imageWidth * imageHeight * 3;
    private const int boolListSize = imageCount;
    private const int camerasToStitchPosition = 1;
    private const int batchDataPosition = camerasToStitchPosition + boolListSize;
    private const int totalBatchSize = batchDataPosition + imageCount * imageSize;

    private const int processedFlagPosition = 0;
    private const int processedDataPosition = 4;
    public const int processedImageWidth = 600, processedImageHeight = 400;
    private const int processedImageSize = processedImageWidth * processedImageHeight * 3;
    private const int totalProcessedSize = processedDataPosition + processedImageSize;

    public List<Camera> camerasToCapture;
    public List<bool> camerasToStitch;

    private RenderTexture reusableTexture;
    private Texture2D image;
    private byte[] batchImageBuffer;

    private float sendInterval = 0.05f;
    private float readInterval = 0.05f;
    private float nextSendTime, nextReceiveTime = 0f;

    [DllImport("kernel32.dll", SetLastError = true, CharSet = CharSet.Auto)]
    private static extern IntPtr CreateFileMapping(IntPtr hFile, IntPtr lpFileMappingAttributes, uint flProtect, uint dwMaximumSizeHigh, uint dwMaximumSizeLow, string lpName);

    [DllImport("kernel32.dll", SetLastError = true)]
    private static extern IntPtr MapViewOfFile(IntPtr hFileMappingObject, uint dwDesiredAccess, uint dwFileOffsetHigh, uint dwFileOffsetLow, UIntPtr dwNumberOfBytesToMap);

    [DllImport("kernel32.dll", SetLastError = true)]
    private static extern bool UnmapViewOfFile(IntPtr lpBaseAddress);

    [DllImport("kernel32.dll", SetLastError = true)]
    private static extern bool CloseHandle(IntPtr hObject);

    private const uint FILE_MAP_ALL_ACCESS = 0xF001F;
    private const uint PAGE_READWRITE = 0x04;

    // Parameters for screen in front of the pilot
    public float radius = 5f;
    public float angleRange = 90f;
    public int segments = 20;
    public float height = 3f;
    private Material curvedScreenMaterial;
    public Texture2D panoTexture;
    public bool resize_dimension = false;

    void Start()
    {
        // Create batch and processed memory-mapped files
        batchFileMap = CreateFileMapping(new IntPtr(-1), IntPtr.Zero, PAGE_READWRITE, 0, totalBatchSize, BatchMapName);
        processedFileMap = CreateFileMapping(new IntPtr(-1), IntPtr.Zero, PAGE_READWRITE, 0, totalProcessedSize, ProcessedMapName);

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
            CloseHandle(batchFileMap);
            CloseHandle(processedFileMap);
            return;
        }

        FindCameras();
        GenerateCurvedScreen();
        curvedScreenMaterial = GetComponent<MeshRenderer>().material;

        reusableTexture = new RenderTexture(imageWidth, imageHeight, 24);
        image = new Texture2D(imageWidth, imageHeight, TextureFormat.RGB24, false);
        batchImageBuffer = new byte[imageCount * imageSize];
        nextSendTime = Time.time;
    }

    void Update()
    {
        UpdateCameraToStitch();

        if (resize_dimension)
        {
            GenerateCurvedScreen();
        }

        if (camerasToCapture.Count == 0)
        {
            FindCameras();
        }
        else if (Time.time >= nextSendTime && Marshal.ReadByte(batchPtr, batchFlagPosition) == 0)
        {
            Marshal.WriteByte(batchPtr, batchFlagPosition, 1);

            for (int i = 0; i < boolListSize; i++)
            {
                byte value = (byte)(i < camerasToStitch.Count && camerasToStitch[i] ? 1 : 0);
                Marshal.WriteByte(batchPtr, camerasToStitchPosition + i, value);
            }

            for (int i = 0; i < camerasToCapture.Count && i < imageCount; i++)
            {
                if (i < camerasToStitch.Count && camerasToStitch[i])
                {
                    byte[] imageBytes = CaptureCameraImage(camerasToCapture[i]);
                    if (imageBytes != null)
                    {
                        Array.Copy(imageBytes, 0, batchImageBuffer, i * imageSize, imageBytes.Length);
                    }
                }
            }

            Marshal.Copy(batchImageBuffer, 0, IntPtr.Add(batchPtr, batchDataPosition), batchImageBuffer.Length);
            Marshal.WriteByte(batchPtr, batchFlagPosition, 0);
            nextSendTime += sendInterval;
        }

        if (Time.time >= nextReceiveTime && Marshal.ReadInt32(processedPtr, processedFlagPosition) == 0)
        {
            Marshal.WriteInt32(processedPtr, processedFlagPosition, 1);

            byte[] processedImageBytes = ReceiveProcessedImage();
            Marshal.WriteInt32(processedPtr, processedFlagPosition, 0);
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
        image.ReadPixels(new Rect(0, 0, imageWidth, imageHeight), 0, 0, false);
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

    void OnDestroy()
    {
        UnmapViewOfFile(batchPtr);
        UnmapViewOfFile(processedPtr);
        CloseHandle(batchFileMap);
        CloseHandle(processedFileMap);
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
        Texture2D panoramaTexture = LoadRawRGBTexture(partPanorama);
        curvedScreenMaterial.mainTexture = panoramaTexture;
        panoTexture = panoramaTexture;
    }

    public Texture2D LoadRawRGBTexture(byte[] imageData)
    {
        if (imageData.Length != processedImageWidth * processedImageHeight * 3)
        {
            Debug.LogError($"Raw image data size does not match expected size for {processedImageWidth}x{processedImageHeight} texture.");
            return null;
        }

        Texture2D texture = new Texture2D(processedImageWidth, processedImageHeight, TextureFormat.RGB24, false);
        Color32[] pixels = new Color32[processedImageWidth * processedImageHeight];

        for (int i = 0; i < pixels.Length; i++)
        {
            int byteIndex = i * 3;
            pixels[i] = new Color32(imageData[byteIndex], imageData[byteIndex + 1], imageData[byteIndex + 2], 255);
        }

        texture.SetPixels32(pixels);
        texture.Apply();
        return texture;
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
}
