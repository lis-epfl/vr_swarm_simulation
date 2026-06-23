using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Text;
using UnityEngine;

[RequireComponent(typeof(MeshFilter), typeof(MeshRenderer))]
public class PyUniSharingFast : MonoBehaviour
{   
    [Header("Feature Flags")]
    [SerializeField]
    [Tooltip("Enable writing images to BlockSharedMemory")]
    private bool enableImageWriting = true;

    [SerializeField]
    [Tooltip("Enable reading panorama from PanoramaSharedMemory")]
    private bool enablePanoramaReading = true;

    [Header("Image Dimensions")]
    [SerializeField]
    [Tooltip("Must be 640 to match image_stream.py / StitcherThreading.py, which read a fixed 640x360 block.")]
    private int blockImageWidth = 640;

    [SerializeField]
    [Tooltip("Must be 360 to match image_stream.py / StitcherThreading.py, which read a fixed 640x360 block.")]
    private int blockImageHeight = 360;

    [SerializeField]
    private int panoramaImageWidth = 600;

    [SerializeField]
    private int panoramaImageHeight = 400;

    [Header("Timing")]
    [SerializeField]
    private float sendInterval = 0.05f;

    [SerializeField]
    private float readInterval = 0.05f;

    [Header("Stitcher Configuration")]
    [SerializeField]
    private stitcherType typeOfStitcher = stitcherType.CLASSIC;

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
    private bool onlyIHN = true;

    [SerializeField]
    private FusionMode typeOfFusion = FusionMode.REFERENCE_BLEND;

    [Header("StabStitch REFERENCE_BLEND Blur")]
    [SerializeField]
    [Tooltip("Gaussian blur kernel size for the reference-image soft mask (must be an odd integer)")]
    private int blurKernelSize = 41;

    [SerializeField]
    [Tooltip("Gaussian blur sigma for the reference-image soft mask feathering width (pixels)")]
    private float blurSigma = 15f;

    [SerializeField]
    [Tooltip("Width in pixels of the edge strip where LINEAR blending is applied to hide seams (REFERENCE_BLEND mode only). Interior of the reference image is left pixel-perfect.")]
    private int borderSize = 60;

    private string blockMapName = "BlockSharedMemory";
    private int blockImageCount = 0;
    private int blockImageSize = 0;   // bytes per drone image (W*H*3)
    private int blockSize = 0;        // per-drone block: header + image
    private int totalBlockSize = 0;   // blockImageCount * blockSize

    private string panoramaMapName = "PanoramaSharedMemory";
    private int panoramaImageSize = 0;
    private int totalPanoramaSize = 0;

    private string metadataMapName = "MetadataSharedMemory";
    private int metadataSize = 20 + 64 + 1 + 4 + 64 + 1 + 4 + 4*4 + 1 + 64 + 4 + 4 + 4; // +8 for blurKernelSize (int) + blurSigma (float), +4 for borderSize (int)

    private IntPtr blockFileMap;
    private IntPtr blockPtr;
    private IntPtr panoramaFileMap;
    private IntPtr panoramaPtr;

    private IntPtr metadataFileMap;
    private IntPtr metadataPtr;

    public List<Camera> camerasToCapture;
    public List<bool> camerasToStitch;

    private RenderTexture reusableTexture;
    private Texture2D image;
    private byte[] blockImageBytes;   // reusable scratch for one converted drone image
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
        REWARP,
        STABSTITCH
    }

    public enum matcherType
    {
        BF,
        FLANN
    }

    public enum FusionMode
    {
        REFERENCE_BLEND,
        REFERENCE,
        EDGE_BLEND,
        LINEAR,
        AVERAGE
    }
    
    private bool hasStarted = false;

    // Constant values
    private const uint FILE_MAP_ALL_ACCESS = 0xF001F;
    private const uint PAGE_READWRITE = 0x04;
    private const int FlagPosition = 0;             // panorama flag (int32) at region start
    private const int panoramaDataPosition = 4;
    // Per-drone block layout (matches image_stream.py / ImageSharing.cs):
    //   int32 flag | int32 droneId | float32 heading | RGB24 image
    private const int blockFlagOffset = 0;
    private const int blockDroneIdOffset = 4;
    private const int blockHeadingOffset = 8;
    private const int blockHeaderSize = 12;
    private const int blockImageDataOffset = blockHeaderSize;
    private const int maxBlockWidth = 2000;
    private const int maxBlockHeight = 2000;
    private const int maxBlockImageCount = 30;
    private const int maxBlockImageSize = maxBlockWidth*maxBlockHeight * 3;
    private const int maxTotalBlockSize = maxBlockImageCount * (blockHeaderSize + maxBlockImageSize);
    private const int maxPanoramaWidth = 4000;
    private const int maxPanoramaHeight = 4000;
    private const int maxPanoramaSize = maxPanoramaWidth * maxPanoramaHeight * 3;
    private const int maxTotalPanoramaSize = panoramaDataPosition + maxPanoramaSize;

    // Parameters for screen in front of the pilot
    public float radius = 5f;
    public float angleRange = 90f;
    public int segments = 20;
    public float height = 3f;
    private Material curvedScreenMaterial;
    private Texture2D panoTexture;
    public bool resize_dimension = false;

    // Other timing values to check the number of camera in the block
    private float cameraUpdateInterval = 3f;
    private float nextCameraUpdateTime = 0f;

    void Start()
    {
        // Always create metadata if either feature is enabled
        if (enableImageWriting || enablePanoramaReading)
        {
            metadataFileMap = CreateFileMapping(new IntPtr(-1), IntPtr.Zero, PAGE_READWRITE, 0, (uint)metadataSize, metadataMapName);
            metadataPtr = MapViewOfFile(metadataFileMap, FILE_MAP_ALL_ACCESS, 0, 0, UIntPtr.Zero);
        }

        // Discover cameras first so the block mapping can be sized to the
        // exact drone count (matches image_stream.py / StitcherThreading.py).
        if (enableImageWriting)
        {
            FindCameras();
            blockImageCount = camerasToCapture.Count;
        }

        CalculateMemorySizes();
        CreateMemoryMaps();

        if (enableImageWriting || enablePanoramaReading)
        {
            WriteMetadata();
        }

        if (enablePanoramaReading)
        {
            GenerateCurvedScreen();
            curvedScreenMaterial = GetComponent<MeshRenderer>().material;
            curvedScreenMaterial.SetFloat("_Glossiness", 0f);
            curvedScreenMaterial.SetColor("_EmissionColor", Color.white);
            curvedScreenMaterial.globalIlluminationFlags = MaterialGlobalIlluminationFlags.BakedEmissive;
            curvedScreenMaterial.EnableKeyword("_EMISSION");
            panoTexture = new Texture2D(panoramaImageWidth, panoramaImageHeight, TextureFormat.RGB24, false);
            pixels = new Color32[panoramaImageWidth * panoramaImageHeight];
        }

        if (enableImageWriting)
        {
            reusableTexture = new RenderTexture(blockImageWidth, blockImageHeight, 24);
            image = new Texture2D(blockImageWidth, blockImageHeight, TextureFormat.RGB24, false);
            blockImageBytes = new byte[blockImageSize];
        }

        hasStarted = true;
        nextSendTime = Time.time;
    }

    void Update()
    {
        if(blockImageWidth>maxBlockWidth || blockImageHeight>maxBlockHeight || panoramaImageWidth>maxPanoramaWidth || panoramaImageHeight>maxPanoramaHeight)
        {
            Debug.LogError("Problem Dimensions");
            return;
        }

        if (enableImageWriting && Time.time >= nextCameraUpdateTime)
        {
            UpdateCameras();
            nextCameraUpdateTime = Time.time + cameraUpdateInterval;
        }

        if (enablePanoramaReading && resize_dimension)
        {
            GenerateCurvedScreen();
        }

        // Handle image writing to BlockSharedMemory (one self-describing block
        // per drone, handshaken independently — matches image_stream.py).
        if (enableImageWriting)
        {
            if (camerasToCapture.Count == 0)
            {
                FindCameras();
            }
            else if (Time.time >= nextSendTime && blockPtr != IntPtr.Zero)
            {
                for (int i = 0; i < camerasToCapture.Count && i < blockImageCount; i++)
                {
                    IntPtr block = IntPtr.Add(blockPtr, i * blockSize);

                    // Skip this drone if the consumer is mid-read on its block.
                    if (Marshal.ReadInt32(block, blockFlagOffset) != 0)
                        continue;

                    // Mark busy while we write the header + image.
                    Marshal.WriteInt32(block, blockFlagOffset, 1);

                    // Header: droneId + this drone's world yaw (heading).
                    Marshal.WriteInt32(block, blockDroneIdOffset, i);
                    float heading = camerasToCapture[i].transform.eulerAngles.y;
                    byte[] headingBytes = BitConverter.GetBytes(heading);
                    if (!BitConverter.IsLittleEndian) Array.Reverse(headingBytes);
                    Marshal.Copy(headingBytes, 0, IntPtr.Add(block, blockHeadingOffset), 4);

                    // Image: Unity RGB (bottom-up) -> BGR (top-down) for the consumer.
                    byte[] imageBytes = CaptureCameraImage(camerasToCapture[i]);
                    if (imageBytes != null && imageBytes.Length == blockImageSize)
                    {
                        ConvertToBlockFormat(imageBytes, blockImageBytes);
                        Marshal.Copy(blockImageBytes, 0, IntPtr.Add(block, blockImageDataOffset), blockImageSize);
                    }

                    // Ready for the consumer.
                    Marshal.WriteInt32(block, blockFlagOffset, 0);
                }

                nextSendTime += sendInterval;
            }
        }

        // Handle panorama reading from PanoramaSharedMemory
        if (enablePanoramaReading)
        {
            if (Time.time >= nextReceiveTime && panoramaPtr != IntPtr.Zero && Marshal.ReadInt32(panoramaPtr, FlagPosition) == 0)
            {
                Marshal.WriteInt32(panoramaPtr, FlagPosition, 1);

                byte[] panoramaImageBytes = ReceivePanoramaImage();
                Marshal.WriteInt32(panoramaPtr, FlagPosition, 0);
                SetPanoramaImage(panoramaImageBytes);

                nextReceiveTime += readInterval;
            }
        }
    }

    private byte[] CaptureCameraImage(Camera camera)
    {
        RenderTexture previousRT = camera.targetTexture;
        camera.targetTexture = reusableTexture;
        RenderTexture.active = reusableTexture;

        camera.Render();
        image.ReadPixels(new Rect(0, 0, blockImageWidth, blockImageHeight), 0, 0, false);
        image.Apply(false);

        byte[] imageBytes = image.GetRawTextureData();
        camera.targetTexture = previousRT;
        RenderTexture.active = null;

        return imageBytes;
    }

    // Converts Unity's raw RGB24 texture data (bottom-left origin, RGB order)
    // into the block format the Python consumers expect: top-left origin, BGR
    // order. This matches the cv2 images written by image_stream.py and the
    // BGR->RGB read performed in ImageSharing.cs.
    private void ConvertToBlockFormat(byte[] src, byte[] dst)
    {
        int rowBytes = blockImageWidth * 3;
        for (int y = 0; y < blockImageHeight; y++)
        {
            int srcRow = y * rowBytes;                              // bottom-up source row
            int dstRow = (blockImageHeight - 1 - y) * rowBytes;     // flipped to top-down
            for (int x = 0; x < blockImageWidth; x++)
            {
                int s = srcRow + x * 3;
                int d = dstRow + x * 3;
                dst[d]     = src[s + 2];  // B
                dst[d + 1] = src[s + 1];  // G
                dst[d + 2] = src[s];      // R
            }
        }
    }

    byte[] ReceivePanoramaImage()
    {
        byte[] panoramaImageBytes = new byte[panoramaImageSize];
        Marshal.Copy(IntPtr.Add(panoramaPtr, panoramaDataPosition), panoramaImageBytes, 0, panoramaImageBytes.Length);
        return panoramaImageBytes;
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
        LoadRawRGBTexture(partPanorama);
        curvedScreenMaterial.mainTexture = panoTexture;
        curvedScreenMaterial.SetTexture("_EmissionMap", panoTexture);
    }

    public void LoadRawRGBTexture(byte[] imageData)
    {
        for (int i = 0; i < pixels.Length; i++)
        {
            int byteIndex = i * 3;
            pixels[i] = new Color32(imageData[byteIndex + 2], imageData[byteIndex + 1], imageData[byteIndex], 255);
        }

        panoTexture.SetPixels32(pixels);
        panoTexture.Apply();
    }

    private void FindCameras()
    {
        camerasToCapture = new List<Camera>();

        GameObject[] drones = GameObject.FindGameObjectsWithTag("DroneBase");
        
        foreach (GameObject drone in drones)
        {
            Camera camera = drone.transform.Find("FPV")?.GetComponent<Camera>();

            if (camera != null)
            {
                camerasToCapture.Add(camera);
            }
            if(camerasToCapture.Count >maxBlockImageCount) break;
        }
    }

    private void UpdateCameraToStitch()
    {
        camerasToStitch = new List<bool>();

        GameObject[] drones = GameObject.FindGameObjectsWithTag("DroneBase");
                
        foreach (GameObject drone in drones)
        {
            AttitudeAlgorithm attitudeScript = drone.transform.Find("DroneParent").GetComponent<AttitudeAlgorithm>();

            if (attitudeScript != null)
            {
                bool estimate = attitudeScript.BoundaryEstimate;
                camerasToStitch.Add(estimate);
            }
            else
            {
                Debug.LogWarning($"No estimate found in {drone.name}");
            }
        }
    }

    private void CalculateMemorySizes()
    {
        if(blockImageCount>maxBlockImageCount)
        {
            blockImageCount = maxBlockImageCount;
            Debug.LogError("Decrease number of drones or increase maxBlockImageCount constant. Value upperbounded at maxBlockImageCount.");
        }

        if(blockImageWidth>maxBlockWidth)
        {
            blockImageWidth = maxBlockWidth;
            Debug.LogError("Decrease dimensions of images or increase maxBlockWidth constant.");
        }

        if(blockImageHeight>maxBlockHeight)
        {
            blockImageHeight = maxBlockHeight;
            Debug.LogError("Decrease dimensions of images or increase maxBlockHeight constant.");
        }

        blockImageSize = blockImageWidth*blockImageHeight*3;
        blockSize = blockHeaderSize + blockImageSize;
        totalBlockSize = blockImageCount * blockSize;

        if(panoramaImageWidth>maxPanoramaWidth)
        {
            panoramaImageWidth = maxPanoramaWidth;
            Debug.LogError("Decrease dimensions of images or increase maxPanoramaWidth constant.");
        } 

        if(panoramaImageHeight>maxPanoramaHeight)
        {
            panoramaImageHeight = maxPanoramaHeight;
            Debug.LogError("Decrease dimensions of images or increase maxPanoramaHeight constant.");
        }

        panoramaImageSize = panoramaImageWidth * panoramaImageHeight * 3;
        totalPanoramaSize = panoramaDataPosition + panoramaImageSize;
    }

    private void CreateMemoryMaps()
    {
        // Only create block memory map if image writing is enabled
        if (enableImageWriting)
        {
            CreateBlockMap();
        }

        // Only create panorama memory map if panorama reading is enabled
        if (enablePanoramaReading)
        {
            panoramaFileMap = CreateFileMapping(new IntPtr(-1), IntPtr.Zero, PAGE_READWRITE, 0, (uint)maxTotalPanoramaSize, panoramaMapName);
            if (panoramaFileMap != IntPtr.Zero)
            {
                panoramaPtr = MapViewOfFile(panoramaFileMap, FILE_MAP_ALL_ACCESS, 0, 0, UIntPtr.Zero);
                if (panoramaPtr == IntPtr.Zero)
                {
                    int errorCode = Marshal.GetLastWin32Error();
                    Debug.LogWarning($"Failed to map view of panorama file. Error Code: {errorCode}");
                }
            }
            else
            {
                Debug.LogWarning("Unable to create panorama memory-mapped file.");
            }
        }
    }

    // Creates (or recreates) the per-drone block mapping, sized exactly to
    // blockImageCount * blockSize so it matches what image_stream.py and
    // StitcherThreading.py allocate. All per-block flags are initialised to 0.
    private void CreateBlockMap()
    {
        DestroyBlockMap();

        if (blockImageCount <= 0 || totalBlockSize <= 0)
            return;

        blockFileMap = CreateFileMapping(new IntPtr(-1), IntPtr.Zero, PAGE_READWRITE, 0, (uint)totalBlockSize, blockMapName);
        if (blockFileMap == IntPtr.Zero)
        {
            Debug.LogWarning("Unable to create block memory-mapped file.");
            return;
        }

        blockPtr = MapViewOfFile(blockFileMap, FILE_MAP_ALL_ACCESS, 0, 0, UIntPtr.Zero);
        if (blockPtr == IntPtr.Zero)
        {
            int errorCode = Marshal.GetLastWin32Error();
            Debug.LogWarning($"Failed to map view of block file. Error Code: {errorCode}");
            return;
        }

        // Initialise every block's flag to 0 (ready for the consumer).
        for (int i = 0; i < blockImageCount; i++)
        {
            Marshal.WriteInt32(blockPtr, i * blockSize + blockFlagOffset, 0);
        }
    }

    private void DestroyBlockMap()
    {
        if (blockPtr != IntPtr.Zero)
        {
            UnmapViewOfFile(blockPtr);
            blockPtr = IntPtr.Zero;
        }
        if (blockFileMap != IntPtr.Zero)
        {
            CloseHandle(blockFileMap);
            blockFileMap = IntPtr.Zero;
        }
    }

    private void DestroyMemoryMaps()
    {
        if (blockPtr != IntPtr.Zero)
        {
            UnmapViewOfFile(blockPtr);
            blockPtr = IntPtr.Zero;
        }
        
        if (panoramaPtr != IntPtr.Zero)
        {
            UnmapViewOfFile(panoramaPtr);
            panoramaPtr = IntPtr.Zero;
        }
        
        if (metadataPtr != IntPtr.Zero)
        {
            UnmapViewOfFile(metadataPtr);
            metadataPtr = IntPtr.Zero;
        }
        
        if (blockFileMap != IntPtr.Zero)
        {
            CloseHandle(blockFileMap);
            blockFileMap = IntPtr.Zero;
        }
        
        if (panoramaFileMap != IntPtr.Zero)
        {
            CloseHandle(panoramaFileMap);
            panoramaFileMap = IntPtr.Zero;
        }

        if (metadataFileMap != IntPtr.Zero)
        {
            CloseHandle(metadataFileMap);
            metadataFileMap = IntPtr.Zero;
        }
    }

    private void OnValidate()
    {
        if(hasStarted)
        {
            CalculateMemorySizes();
            
            if (enableImageWriting || enablePanoramaReading)
            {
                WriteMetadata();
            }

            // Update reusable resources for image writing
            if (enableImageWriting)
            {
                reusableTexture = new RenderTexture(blockImageWidth, blockImageHeight, 24);
                image = new Texture2D(blockImageWidth, blockImageHeight, TextureFormat.RGB24, false);
                blockImageBytes = new byte[blockImageSize];
                CreateBlockMap();
            }

            // Update reusable resources for panorama reading
            if (enablePanoramaReading)
            {
                panoTexture = new Texture2D(panoramaImageWidth, panoramaImageHeight, TextureFormat.RGB24, false);
                pixels = new Color32[panoramaImageWidth * panoramaImageHeight];
            }
        }
    }

    private void ValidateTextures()
    {
        if (!enableImageWriting) return;

        if (reusableTexture == null || reusableTexture.width != blockImageWidth || reusableTexture.height != blockImageHeight)
        {
            reusableTexture?.Release();
            reusableTexture = new RenderTexture(blockImageWidth, blockImageHeight, 24);
        }

        if (image == null || image.width != blockImageWidth || image.height != blockImageHeight)
        {
            Destroy(image);
            image = new Texture2D(blockImageWidth, blockImageHeight, TextureFormat.RGB24, false);
        }
    }

    private void UpdateCameras()
    {
        if (!enableImageWriting) return;

        FindCameras();

        int newblockImageCount = camerasToCapture.Count;
        UpdateCameraToStitch();
        if (newblockImageCount != blockImageCount)
        {
            blockImageCount = newblockImageCount;
            CalculateMemorySizes();
            CreateBlockMap();          // resize the mapping to the new drone count
            WriteMetadata();
            blockImageBytes = new byte[blockImageSize];
            ValidateTextures();
        }
    }

    private void WriteMetadata()
    {
        if (metadataPtr == IntPtr.Zero)
        {
            Debug.LogError($"Problem with metadata memory.");
            return;
        }

        int offset = 0;

        Marshal.WriteInt32(metadataPtr, offset, blockImageWidth);
        offset += 4;
        Marshal.WriteInt32(metadataPtr, offset, blockImageHeight);
        offset += 4;
        Debug.LogWarning(blockImageCount);
        Marshal.WriteInt32(metadataPtr, offset, blockImageCount);
        offset += 4;
        Marshal.WriteInt32(metadataPtr, offset, panoramaImageWidth);
        offset += 4;
        Marshal.WriteInt32(metadataPtr, offset, panoramaImageHeight);
        offset += 4;

        byte[] stringBytes = Encoding.UTF8.GetBytes(typeOfStitcher.ToString());
        byte[] stringBuffer = new byte[64];
        Array.Copy(stringBytes, stringBuffer, Math.Min(stringBytes.Length, stringBuffer.Length));
        Marshal.Copy(stringBuffer, 0, IntPtr.Add(metadataPtr, offset), stringBuffer.Length);
        offset += 64;

        Marshal.WriteByte(metadataPtr, offset, (byte)(cylindrical ? 1 : 0));
        offset +=1;

        byte[] stringBytesMatcher = Encoding.UTF8.GetBytes(typeOfMatcher.ToString());
        byte[] stringBufferMatcher = new byte[64];
        Array.Copy(stringBytesMatcher, stringBufferMatcher, Math.Min(stringBytesMatcher.Length, stringBufferMatcher.Length));
        Marshal.Copy(stringBufferMatcher, 0, IntPtr.Add(metadataPtr, offset), stringBufferMatcher.Length);
        offset += 64;

        Marshal.WriteByte(metadataPtr, offset, (byte)(ransac ? 1 : 0));
        offset +=1;

        Marshal.WriteInt32(metadataPtr, offset, checks);
        offset += 4;

        byte[] ratioThreshBytes = BitConverter.GetBytes(ratio_thresh);
        if (!BitConverter.IsLittleEndian)
        {
            Array.Reverse(ratioThreshBytes);
        }
        Marshal.Copy(ratioThreshBytes, 0, IntPtr.Add(metadataPtr, offset), 4);
        offset += 4;

        byte[] scoreThresholdBytes = BitConverter.GetBytes(score_threshold);
        if (!BitConverter.IsLittleEndian)
        {
            Array.Reverse(scoreThresholdBytes);
        }
        Marshal.Copy(scoreThresholdBytes, 0, IntPtr.Add(metadataPtr, offset), 4);
        offset += 4;

        Marshal.WriteInt32(metadataPtr, offset, focal_length);
        offset += 4;

        Marshal.WriteByte(metadataPtr, offset, (byte)(onlyIHN ? 1 : 0));
        offset += 1;

        byte[] fusionModeBytes = Encoding.UTF8.GetBytes(typeOfFusion.ToString());
        byte[] fusionModeBuffer = new byte[64];
        Array.Copy(fusionModeBytes, fusionModeBuffer, Math.Min(fusionModeBytes.Length, fusionModeBuffer.Length));
        Marshal.Copy(fusionModeBuffer, 0, IntPtr.Add(metadataPtr, offset), fusionModeBuffer.Length);
        offset += 64;

        Marshal.WriteInt32(metadataPtr, offset, blurKernelSize);
        offset += 4;

        byte[] blurSigmaBytes = BitConverter.GetBytes(blurSigma);
        if (!BitConverter.IsLittleEndian) Array.Reverse(blurSigmaBytes);
        Marshal.Copy(blurSigmaBytes, 0, IntPtr.Add(metadataPtr, offset), 4);
        offset += 4;

        Marshal.WriteInt32(metadataPtr, offset, borderSize);

        if(hasStarted) return;
        offset += 64;

        Marshal.WriteInt32(metadataPtr, offset, maxTotalBlockSize);
        offset += 4;
        Marshal.WriteInt32(metadataPtr, offset, maxTotalPanoramaSize);
    }

    private void CheckExistingMapping(string mapName)
    {
        IntPtr existingMap = OpenFileMapping(FILE_MAP_ALL_ACCESS, false, mapName);
        if (existingMap != IntPtr.Zero)
        {
            System.Threading.Thread.Sleep(100);

            IntPtr secondCheck = OpenFileMapping(FILE_MAP_ALL_ACCESS, false, mapName);
            if (secondCheck != IntPtr.Zero)
            {
                Debug.LogError($"Memory map '{mapName}' still exists after closing the handle.");
                CloseHandle(secondCheck);
            }
        }
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