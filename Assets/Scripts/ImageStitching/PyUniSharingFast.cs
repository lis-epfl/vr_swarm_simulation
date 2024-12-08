using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Text;
using UnityEngine;

[RequireComponent(typeof(MeshFilter), typeof(MeshRenderer))]
public class PyUniSharingFast : MonoBehaviour
{   
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

    [SerializeField]
    private stitcherType typeOfStitcher = stitcherType.CLASSIC; // Possible values: classic, UDIS, NIS

    [SerializeField]
    private bool cylindrical = false;

    [SerializeField]
    private matcherType typeOfMatcher = matcherType.BF;

    [SerializeField]
    private bool ransac = false;

    [SerializeField]
    private int checks = 50;

    [SerializeField]
    private float ratio_thresh = 0.7f;

    [SerializeField]
    private float score_threshold = 0.1f;

    [SerializeField]
    private int focal_length = 1000;

    [SerializeField]
    private bool onlyIHN = false; //Maybe for implementation of NIS only with IHN for fast warping

    private string batchMapName = "BatchSharedMemory";
    private int batchImageCount = 0;
    private int batchImageSize = 0;
    private int boolListSize = 0;
    private int batchDataPosition = 0;
    private int totalBatchSize = 0;

    private string processedMapName = "ProcessedImageSharedMemory";
    private int processedImageSize = 0;
    private int totalProcessedSize = 0;

    private string metadataMapName = "MetadataSharedMemory";
    private int metadataSize = 20 + 64 + 1+ 4 + 64 + 1 + 4 + 4*4 + 1; // 20 bytes for ints (5x4 bytes) + 64 bytes for string + 1 byte bool + 64 bytes for string + 1 byte bool +  4 bytes float + 4*4 int and floats + one bool

    private IntPtr batchFileMap;
    private IntPtr batchPtr;
    private IntPtr processedFileMap;
    private IntPtr processedPtr;

    private IntPtr metadataFileMap;
    private IntPtr metadataPtr;

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

    [DllImport("kernel32.dll", SetLastError = true, CharSet = CharSet.Auto)]
    private static extern IntPtr OpenFileMapping(uint dwDesiredAccess, bool bInheritHandle, string lpName);

    public enum stitcherType
    {
        CLASSIC,
        UDIS,
        NIS,
        REWARP
    }

    public enum matcherType
    {
        BF,
        FLANN
    }
    
    private bool hasStarted = false;

    // Constant values
    private const uint FILE_MAP_ALL_ACCESS = 0xF001F;
    private const uint PAGE_READWRITE = 0x04;
    private const int FlagPosition = 0;
    private const int camerasToStitchPosition = 1;
    private const int numFloatByte = 4;
    private const int processedDataPosition = 4;
    private const int maxBatchWidth = 2000;
    private const int maxBatchHeight = 2000;
    private const int maxBatchImageCount = 30;
    private const int maxBatchImageSize = maxBatchWidth*maxBatchHeight * 3;
    private const int maxTotalBatchSize = camerasToStitchPosition+maxBatchImageSize + numFloatByte + maxBatchImageCount;

    private const int maxProcessedWidth = 4000;
    private const int maxProcessedHeight = 4000;
    private const int maxProcessedSize = maxProcessedWidth * maxProcessedHeight * 3;
    private const int maxTotalProcessedSize = processedDataPosition + maxProcessedSize;

    // Parameters for screen in front of the pilot
    public float radius = 5f;
    public float angleRange = 90f;
    public int segments = 20;
    public float height = 3f;
    private Material curvedScreenMaterial;
    private Texture2D panoTexture;
    public bool resize_dimension = false;

    // Other timing values to check the number of camera in the batch
    private float cameraUpdateInterval = 3f; // Time interval to update cameras
    private float nextCameraUpdateTime = 0f; // Next time to update cameras

    void Start()
    {
        metadataFileMap = CreateFileMapping(new IntPtr(-1), IntPtr.Zero, PAGE_READWRITE, 0, (uint)metadataSize, metadataMapName);
        metadataPtr = MapViewOfFile(metadataFileMap, FILE_MAP_ALL_ACCESS, 0, 0, UIntPtr.Zero);

        CalculateMemorySizes();
        CreateMemoryMaps();
        WriteMetadata();

        FindCameras();
        GenerateCurvedScreen();
        curvedScreenMaterial = GetComponent<MeshRenderer>().material;

        reusableTexture = new RenderTexture(batchImageWidth, batchImageHeight, 24);
        image = new Texture2D(batchImageWidth, batchImageHeight, TextureFormat.RGB24, false);
        batchImageBuffer = new byte[batchImageCount * batchImageSize];
        panoTexture = new Texture2D(processedImageWidth, processedImageHeight, TextureFormat.RGB24, false);
        pixels = new Color32[processedImageWidth * processedImageHeight];
        hasStarted = true;

        nextSendTime = Time.time;
    }

    void Update()
    {
        if(batchImageWidth>maxBatchWidth || batchImageHeight>maxBatchHeight || processedImageWidth>maxProcessedWidth || processedImageHeight>maxProcessedHeight)
        {
            Debug.LogError("Problem Dimensions");
            return;
        }

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

            float headAngle = TakeHeadsetAngle();
            // Write float
            byte[] floatBytes = BitConverter.GetBytes(headAngle);
            Marshal.Copy(floatBytes, 0, IntPtr.Add(batchPtr, boolListSize+camerasToStitchPosition), floatBytes.Length);

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

    private float TakeHeadsetAngle()
    {
        float headAngle = 0f;

        return headAngle;
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
            if(camerasToCapture.Count >maxBatchImageCount) break;
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
        if(batchImageCount>maxBatchImageCount)
        {
            batchImageCount = maxBatchImageCount;
            Debug.LogError("Decrease number of drones or increase maxBatchImageCount constant. Value upperbounded at maxBatchImageCount.");
        }

        boolListSize = batchImageCount;

        if(batchImageWidth>maxBatchWidth)
        {
            batchImageWidth = maxBatchWidth;
            Debug.LogError("Decrease dimensions of images or increase maxBatchWidth constant.");
        } 

        if(batchImageHeight>maxBatchHeight)
        {
            batchImageHeight = maxBatchHeight;
            Debug.LogError("Decrease dimensions of images or increase maxBatchHeight constant.");
        }

        batchImageSize=batchImageWidth*batchImageHeight*3;
        batchDataPosition = boolListSize + camerasToStitchPosition + numFloatByte;
        totalBatchSize = batchDataPosition + batchImageCount * batchImageSize;

        if(processedImageWidth>maxProcessedWidth)
        {
            processedImageWidth = maxProcessedWidth;
            Debug.LogError("Decrease dimensions of images or increase maxProcessedWidth constant.");
        } 

        if(processedImageHeight>maxProcessedHeight)
        {
            processedImageHeight = maxProcessedHeight;
            Debug.LogError("Decrease dimensions of images or increase maxProcessedHeight constant.");
        }

        processedImageSize = processedImageWidth * processedImageHeight * 3;
        totalProcessedSize = processedDataPosition + processedImageSize;
        // Debug.Log($"batchImageWidth: {batchImageWidth}, batchImageHeight: {batchImageHeight}, batchImageSize: {batchImageSize}, batchImageCount: {batchImageCount}, boolListSize: {boolListSize}, camerasToStitchPosition: {camerasToStitchPosition}, batchImageHeight: {batchDataPosition}");
    }

    private void CreateMemoryMaps()
    {
        // Create memory-mapped files in RAM with new IntPtr(-1) with appropriate name
        batchFileMap = CreateFileMapping(new IntPtr(-1), IntPtr.Zero, PAGE_READWRITE, 0, (uint)maxTotalBatchSize , batchMapName);
        processedFileMap = CreateFileMapping(new IntPtr(-1), IntPtr.Zero, PAGE_READWRITE, 0, (uint)maxTotalProcessedSize, processedMapName);
        

        if (batchFileMap == IntPtr.Zero || processedFileMap == IntPtr.Zero|| metadataFileMap == IntPtr.Zero)
        {
            Debug.LogWarning("Unable to create memory-mapped files.");
            return;
        }

        // Creates a pointer that allows the process to access the memory-mapped file
        batchPtr = MapViewOfFile(batchFileMap, FILE_MAP_ALL_ACCESS, 0, 0, UIntPtr.Zero);
        processedPtr = MapViewOfFile(processedFileMap, FILE_MAP_ALL_ACCESS, 0, 0, UIntPtr.Zero);

        if (batchPtr == IntPtr.Zero)
        {
            int errorCode = Marshal.GetLastWin32Error();
            Debug.LogWarning($"Failed to map view of file. Error Code: {errorCode}");
        }

        if (batchPtr == IntPtr.Zero || processedPtr == IntPtr.Zero|| metadataPtr == IntPtr.Zero)
        {
            Debug.LogWarning($"Unable to map view of file. Total batch Size: {totalBatchSize}, Total processed Size: {totalProcessedSize}");
            DestroyMemoryMaps();
        }
    }

    private void DestroyMemoryMaps()
    {
        
        UnmapViewOfFile(batchPtr);
        batchPtr = IntPtr.Zero;
        
        UnmapViewOfFile(processedPtr);
        processedPtr = IntPtr.Zero;
        
        UnmapViewOfFile(metadataPtr);
        metadataPtr = IntPtr.Zero;
        
        CloseHandle(batchFileMap);
        batchFileMap = IntPtr.Zero;
        
        CloseHandle(processedFileMap);
        processedFileMap = IntPtr.Zero;

        CloseHandle(metadataFileMap);
        metadataFileMap = IntPtr.Zero;
    }

    private void OnValidate()
    {
        if(hasStarted)
        {
            CalculateMemorySizes();
            WriteMetadata();
            // Update reusable resources
            reusableTexture = new RenderTexture(batchImageWidth, batchImageHeight, 24);
            image = new Texture2D(batchImageWidth, batchImageHeight, TextureFormat.RGB24, false);
            batchImageBuffer = new byte[batchImageCount * batchImageSize];
            // boolListSize = batchImageCount;

            panoTexture = new Texture2D(processedImageWidth, processedImageHeight, TextureFormat.RGB24, false);
            pixels = new Color32[processedImageWidth * processedImageHeight];
        }
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
        UpdateCameraToStitch();
        if (newBatchImageCount != batchImageCount)
        {
            batchImageCount = newBatchImageCount;
            CalculateMemorySizes();
            WriteMetadata();
            batchImageBuffer = new byte[batchImageCount * batchImageSize];
            ValidateTextures(); // Ensure textures are updated
            
        }
    }

    private void WriteMetadata()
    {
        if (metadataPtr == IntPtr.Zero)
        {
            Debug.LogError($"Problem with metadata memory.");
            return;
        }

        // Write metadata to the shared memory
        int offset = 0;

        // Write integers
        Marshal.WriteInt32(metadataPtr, offset, batchImageWidth);
        offset += 4;
        Marshal.WriteInt32(metadataPtr, offset, batchImageHeight);
        offset += 4;
        Debug.LogWarning(batchImageCount);
        Marshal.WriteInt32(metadataPtr, offset, batchImageCount);
        offset += 4;
        Marshal.WriteInt32(metadataPtr, offset, processedImageWidth);
        offset += 4;
        Marshal.WriteInt32(metadataPtr, offset, processedImageHeight);
        offset += 4;

        // Write string (up to 64 bytes, zero-padded)

        byte[] stringBytes = Encoding.UTF8.GetBytes(typeOfStitcher.ToString());
        byte[] stringBuffer = new byte[64];
        Array.Copy(stringBytes, stringBuffer, Math.Min(stringBytes.Length, stringBuffer.Length));
        Marshal.Copy(stringBuffer, 0, IntPtr.Add(metadataPtr, offset), stringBuffer.Length);
        offset += 64;

        // Write bool
        Marshal.WriteByte(metadataPtr, offset, (byte)(cylindrical ? 1 : 0));
        offset +=1;

        // Wrtite type of matcher
        byte[] stringBytesMatcher = Encoding.UTF8.GetBytes(typeOfMatcher.ToString());
        byte[] stringBufferMatcher = new byte[64];
        Array.Copy(stringBytesMatcher, stringBufferMatcher, Math.Min(stringBytesMatcher.Length, stringBufferMatcher.Length));
        Marshal.Copy(stringBufferMatcher, 0, IntPtr.Add(metadataPtr, offset), stringBufferMatcher.Length);
        offset += 64;

        // Write RANSAC bool
        Marshal.WriteByte(metadataPtr, offset, (byte)(ransac ? 1 : 0));
        offset +=1;

        // write checks (int)
        Marshal.WriteInt32(metadataPtr, offset, checks);
        offset += 4;
        // Write ratio_thresh (float)
        byte[] ratioThreshBytes = BitConverter.GetBytes(ratio_thresh);
        // Ensure correct endianness
        if (!BitConverter.IsLittleEndian)
        {
            Array.Reverse(ratioThreshBytes);
        }
        Marshal.Copy(ratioThreshBytes, 0, IntPtr.Add(metadataPtr, offset), 4);
        offset += 4;

        // Write score_threshold (float)
        byte[] scoreThresholdBytes = BitConverter.GetBytes(score_threshold);
        // Ensure correct endianness
        if (!BitConverter.IsLittleEndian)
        {
            Array.Reverse(scoreThresholdBytes);
        }
        Marshal.Copy(scoreThresholdBytes, 0, IntPtr.Add(metadataPtr, offset), 4);
        offset += 4;
        // write focal_length (int)
        Marshal.WriteInt32(metadataPtr, offset, focal_length);
        offset += 4;
        // write onlyIHN (bool)
        Marshal.WriteByte(metadataPtr, offset, (byte)(onlyIHN ? 1 : 0));

        // Debug.Log("Metadata written to shared memory.");

        if(hasStarted) return;
        offset +=1;
        // Write integers
        Marshal.WriteInt32(metadataPtr, offset, maxTotalBatchSize);
        offset += 4;
        Marshal.WriteInt32(metadataPtr, offset, maxTotalProcessedSize);
    }

    private void CheckExistingMapping(string mapName)
    {
        IntPtr existingMap = OpenFileMapping(FILE_MAP_ALL_ACCESS, false, mapName);
        if (existingMap != IntPtr.Zero)
        {
            // Debug.LogWarning($"A memory map with the name '{mapName}' already exists. Attempting to clean up.");

            // if (CloseHandle(existingMap))
            // {
            //     Debug.Log($"Successfully closed existing memory map handle for: {mapName}");
            // }
            // else
            // {
            //     Debug.LogError($"Failed to close existing memory map handle for: {mapName}. Error: {Marshal.GetLastWin32Error()}");
            // }

            // Delay to ensure the OS fully releases the resource
            System.Threading.Thread.Sleep(100);

            // Recheck
            IntPtr secondCheck = OpenFileMapping(FILE_MAP_ALL_ACCESS, false, mapName);
            if (secondCheck != IntPtr.Zero)
            {
                Debug.LogError($"Memory map '{mapName}' still exists after closing the handle.");
                CloseHandle(secondCheck);
            }
            // else
            // {
            //     Debug.Log($"No memory map found for '{mapName}' after closing.");
            // }
        }
        // else
        // {
        //     Debug.Log($"No existing memory map found for '{mapName}'.");
        // }
    }

    void OnDestroy()
    {
        DestroyMemoryMaps();
    }
    void OnApplicationQuit()
    {
        DestroyMemoryMaps();
        Debug.Log("Application quitting. Memory maps destroyed.");
    }
}
