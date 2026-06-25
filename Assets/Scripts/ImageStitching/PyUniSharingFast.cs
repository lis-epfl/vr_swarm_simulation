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

    [Header("StabStitch Panorama Quality Fallback")]
    [SerializeField]
    [Tooltip("When the StabStitch panorama is judged bad (poor alignment / distorted warp), hide it and show the individual drone feeds (via ScreenSpawn) instead.")]
    private bool qualityFallbackEnabled = true;

    [SerializeField]
    [Tooltip("Minimum overlap PSNR (dB) for the panorama to be considered good. Higher = stricter (falls back to feeds more readily).")]
    private float qualityThreshold = 18f;

    [SerializeField]
    [Tooltip("ScreenSpawn style used to display the individual drone feeds while the panorama is in fallback.")]
    private ScreenSpawn.ScreenStyle fallbackScreenStyle = ScreenSpawn.ScreenStyle.OUTER_CIRCLE;

    private string blockMapName = "BlockSharedMemory";
    private int blockImageCount = 0;
    private int blockImageSize = 0;   // bytes per drone image (W*H*3)
    private int blockSize = 0;        // per-drone block: header + image
    private int totalBlockSize = 0;   // blockImageCount * blockSize

    private string panoramaMapName = "PanoramaSharedMemory";
    private int panoramaImageSize = 0;
    private int totalPanoramaSize = 0;

    private string metadataMapName = "MetadataSharedMemory";
    private int metadataSize = 20 + 64 + 1 + 4 + 64 + 1 + 4 + 4*4 + 1 + 64 + 4 + 4 + 4 + 1 + 4 + 4; // +8 blurKernelSize+blurSigma, +4 borderSize, +1 qualityFallbackEnabled (bool), +4 qualityThreshold (float), +4 headYaw (float)

    private IntPtr blockFileMap;
    private IntPtr blockPtr;
    private IntPtr panoramaFileMap;
    private IntPtr panoramaPtr;

    private IntPtr metadataFileMap;
    private IntPtr metadataPtr;

    public List<Camera> camerasToCapture;
    public List<bool> camerasToStitch;
    private List<AttitudeAlgorithm> stitchAttitudes;  // per-camera boundary estimator, index-aligned with camerasToCapture

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
    private const int panoramaQualityPosition = 4;  // packed quality word (int32) — see masks below
    private const int panoramaDataPosition = 8;      // RGB24 panorama data

    // Packed quality word (PanoramaSharedMemory [4:8]), written by Python's
    // write_panorama_memory. bit 0 = panorama good; bits 1-3 = failing-gate
    // reason (only set when bit 0 == 0). Keep in sync with StitcherThreading.py.
    private const int QUALITY_OK_BIT = 1 << 0;       // panorama good (show panorama)
    private const int REASON_CANVAS = 1 << 1;        // degenerate warp canvas size
    private const int REASON_DISTORTION = 1 << 2;    // folded/torn mesh (inter-grid loss)
    private const int REASON_PHOTOMETRIC = 1 << 3;   // overlap PSNR below threshold
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

    // Metadata layout: the pilot heading yaw (float) is appended after
    // qualityThreshold. This carries the integrated body yaw (WriteBodyYaw), not
    // the live HMD direction, so Python selects the same views as SelectStitchCameras.
    // Offset = sizes(20) + stitcher(64) + cylindrical(1) + matcher(64) + ransac(1)
    //          + checks(4) + ratio(4) + score(4) + focal(4) + onlyIHN(1) + fusion(64)
    //          + blurKernel(4) + blurSigma(4) + border(4) + qualityEnabled(1) + qualityThreshold(4)
    private const int metadataHeadYawOffset = 248;
    private const int STITCH_COUNT = 3;  // panorama is always 3 views: left / centre / right

    // Parameters for screen in front of the pilot
    public float radius = 5f;
    public float angleRange = 90f;
    public int segments = 20;
    public float height = 3f;
    private Material curvedScreenMaterial;
    private MeshRenderer panoramaRenderer;
    private Texture2D panoTexture;
    public bool resize_dimension = false;

    // Headset-directed stitching + curved-screen placement
    [Header("Headset Direction")]
    [SerializeField]
    [Tooltip("HMD head transform (OVRCameraRig.centerEyeAnchor). Auto-found from the OVRPlayerController if left empty.")]
    private Transform headTransform;

    [SerializeField]
    [Tooltip("Vertical offset of the curved screen above the Arena centre.")]
    private float screenHeightOffset = 0f;

    [SerializeField]
    [Tooltip("Degrees/second the body yaws at full controller yaw-stick deflection. The panorama " +
             "heading integrates this command and the OVRCameraRig is rotated by the same amount " +
             "to mimic body motion. Head tracking is excluded, so the pilot can look around at the " +
             "side screens without moving the panorama.")]
    private float bodyYawRate = 90f;

    [SerializeField]
    [Tooltip("Rotate the OVRCameraRig by the controller yaw-rate command to mimic body motion. " +
             "Disable to advance the panorama heading only, leaving the rig untouched.")]
    private bool driveCameraRigYaw = true;

    // Body heading that drives the panorama. Seeded once from the head's initial
    // yaw, then advanced only by the controller yaw-rate command (never by head
    // tracking). cameraRigTransform is rotated by the same command so the rig
    // turns with the body while the head still yaws freely relative to it.
    private Transform cameraRigTransform;
    private float bodyYaw;
    private bool bodyYawInitialized = false;

    private GameObject arena;
    private int[] selectedStitchIndices = new int[0];  // camera indices written to the 3 blocks, ordered [left, centre, right]

    [Header("Stitching Debug")]
    [SerializeField]
    [Tooltip("Read-only: the drones currently sent to the stitcher, ordered left / centre / right. Updates during Play.")]
    private List<string> stitchedDrones = new List<string>();
    private string lastStitchedDronesKey;  // change-detection so the list only rebuilds when the selection changes

    // Quality fallback: switch between the panorama screen and ScreenSpawn feeds
    [SerializeField] private ScreenSpawn screenSpawn;
    private bool panoramaDisplayActive = true;

    [Header("Stitched Drone Screens")]
    [SerializeField]
    [Tooltip("Hide the individual ScreenSpawn feeds for the drones currently being stitched into the panorama (they already appear in the panorama). Only applies while the panorama is displayed; during quality-fallback all feeds reappear.")]
    private bool hideStitchedDroneScreens = false;
    private string lastHiddenScreensKey;

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
            blockImageCount = Mathf.Min(STITCH_COUNT, camerasToCapture.Count);
        }

        CalculateMemorySizes();
        CreateMemoryMaps();

        // Resolve the HMD head transform now (the OVRCameraRig is in the scene)
        // so the first metadata write seeds a real head yaw rather than 0.
        FindHeadTransform();

        if (enableImageWriting || enablePanoramaReading)
        {
            WriteMetadata();
        }

        if (enablePanoramaReading)
        {
            GenerateCurvedScreen();
            panoramaRenderer = GetComponent<MeshRenderer>();
            curvedScreenMaterial = panoramaRenderer.material;
            curvedScreenMaterial.SetFloat("_Glossiness", 0f);
            curvedScreenMaterial.SetColor("_EmissionColor", Color.white);
            curvedScreenMaterial.globalIlluminationFlags = MaterialGlobalIlluminationFlags.BakedEmissive;
            curvedScreenMaterial.EnableKeyword("_EMISSION");
            panoTexture = new Texture2D(panoramaImageWidth, panoramaImageHeight, TextureFormat.RGB24, false);
            pixels = new Color32[panoramaImageWidth * panoramaImageHeight];

            // ScreenSpawn drives the per-drone feed fallback when the panorama is bad.
            if (screenSpawn == null)
            {
                screenSpawn = FindObjectOfType<ScreenSpawn>();
            }
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

        // The body heading (not the live head direction) drives which three views
        // are stitched and where the curved panorama screen sits. It starts at the
        // initial head yaw, then only advances with the controller yaw-rate command
        // (which also rotates the camera rig to mimic body motion). The head can
        // still yaw freely via HMD tracking to look at the side screens without
        // moving the panorama. Resolve the rig/head lazily (the rig may be added to
        // the scene later).
        FindHeadTransform();
        UpdateBodyYaw();
        WriteBodyYaw(bodyYaw);

        // Select the centre drone (camera yaw closest to the body yaw) plus the
        // two yaw-neighbours. centreYaw drives the curved-screen orientation so
        // the screen snaps to the new view only when the selection changes.
        float centreYaw = SelectStitchCameras(bodyYaw, out selectedStitchIndices);
        UpdateStitchedDronesDisplay(selectedStitchIndices);
        UpdateStitchedScreenHiding(selectedStitchIndices);

        if (enablePanoramaReading)
        {
            UpdateCurvedScreenPose(centreYaw);
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
                // Write only the head-aligned drones selected this frame into the
                // blocks the Python stitcher reads (slots ordered left/centre/right).
                for (int j = 0; j < selectedStitchIndices.Length && j < blockImageCount; j++)
                {
                    int camIdx = selectedStitchIndices[j];
                    IntPtr block = IntPtr.Add(blockPtr, j * blockSize);

                    // Skip this slot if the consumer is mid-read on its block.
                    if (Marshal.ReadInt32(block, blockFlagOffset) != 0)
                        continue;

                    // Mark busy while we write the header + image.
                    Marshal.WriteInt32(block, blockFlagOffset, 1);

                    // Header: droneId + this drone's world yaw (heading).
                    Marshal.WriteInt32(block, blockDroneIdOffset, camIdx);
                    float heading = camerasToCapture[camIdx].transform.eulerAngles.y;
                    byte[] headingBytes = BitConverter.GetBytes(heading);
                    if (!BitConverter.IsLittleEndian) Array.Reverse(headingBytes);
                    Marshal.Copy(headingBytes, 0, IntPtr.Add(block, blockHeadingOffset), 4);

                    // Image: Unity RGB (bottom-up) -> BGR (top-down) for the consumer.
                    byte[] imageBytes = CaptureCameraImage(camerasToCapture[camIdx]);
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

                int qualityWord = Marshal.ReadInt32(panoramaPtr, panoramaQualityPosition);
                byte[] panoramaImageBytes = ReceivePanoramaImage();
                Marshal.WriteInt32(panoramaPtr, FlagPosition, 0);

                // bit 0 = panorama good; bits 1-3 = failing-gate reason (only
                // meaningful when bit 0 is clear). When the panorama is bad (and
                // fallback is enabled) show the individual drone feeds instead.
                bool qualityOk = (qualityWord & QUALITY_OK_BIT) != 0;
                bool panoramaGood = !qualityFallbackEnabled || qualityOk;
                ApplyQualityFallback(panoramaGood, qualityWord);
                if (panoramaGood)
                {
                    SetPanoramaImage(panoramaImageBytes);
                }

                nextReceiveTime += readInterval;
            }
        }
    }

    // Switch between the stitched panorama screen and the individual drone
    // feeds (ScreenSpawn). Only acts on a transition so it doesn't fight
    // ScreenSpawn's per-frame positioning. Python already debounces the
    // quality verdict (hysteresis), so the flag is stable.
    private void ApplyQualityFallback(bool panoramaGood, int qualityWord)
    {
        if (panoramaGood == panoramaDisplayActive)
        {
            return; // no change
        }
        panoramaDisplayActive = panoramaGood;

        // Log only on the transition so it doesn't spam every frame. When the
        // panorama disappears, report which stitch-quality gate(s) tripped.
        if (!panoramaGood)
        {
            Debug.Log($"[Panorama] hidden — showing individual feeds. Stitch quality bad: {DescribeQualityReason(qualityWord)}.");
        }
        else
        {
            Debug.Log("[Panorama] restored — stitch quality good again.");
        }

        // Show/hide the curved panorama screen.
        if (panoramaRenderer != null)
        {
            panoramaRenderer.enabled = panoramaGood;
        }

        // Show/hide the individual feed screens.
        if (screenSpawn != null)
        {
            screenSpawn.ShowFallbackFeeds(!panoramaGood, fallbackScreenStyle);
        }
    }

    // Decode the failing-gate bits of the packed quality word into a readable
    // reason, so the transition log says *why* the panorama was hidden.
    private string DescribeQualityReason(int qualityWord)
    {
        string reasons = "";
        if ((qualityWord & REASON_CANVAS) != 0)
            reasons += (reasons.Length > 0 ? ", " : "") + "canvas (degenerate warp size)";
        if ((qualityWord & REASON_DISTORTION) != 0)
            reasons += (reasons.Length > 0 ? ", " : "") + "distortion (folded/torn mesh)";
        if ((qualityWord & REASON_PHOTOMETRIC) != 0)
            reasons += (reasons.Length > 0 ? ", " : "") + "photometric (overlap PSNR below threshold)";
        return reasons.Length > 0 ? reasons : "unspecified";
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

    // Resolve the HMD head transform (OVRCameraRig.centerEyeAnchor). Prefers the
    // serialized override; otherwise finds the OVRCameraRig that lives in the
    // scene. Resolved once in Start and cached — the Update guard short-circuits
    // every frame after, so FindObjectOfType only runs until the rig's anchors
    // are ready (then Camera.main as an editor fallback when there's no rig).
    private void FindHeadTransform()
    {
        if (headTransform != null && cameraRigTransform != null) return;

        OVRCameraRig rig = FindObjectOfType<OVRCameraRig>();
        if (rig != null)
        {
            // The rig root is the "body" we rotate; centerEyeAnchor is the HMD-
            // tracked head, which yaws relative to the rig as the pilot looks around.
            if (cameraRigTransform == null) cameraRigTransform = rig.transform;
            if (headTransform == null && rig.centerEyeAnchor != null) headTransform = rig.centerEyeAnchor;
            if (headTransform != null) return;
        }

        // Editor fallback when no OVR rig is present: drive the main camera as both
        // the head and the body so the behaviour is still testable.
        if (Camera.main != null)
        {
            if (headTransform == null) headTransform = Camera.main.transform;
            if (cameraRigTransform == null) cameraRigTransform = Camera.main.transform;
        }
    }

    // Advances the body heading that drives the panorama. Seeded once from the
    // head's initial yaw so the panorama starts where the pilot is first looking,
    // then advanced only by the controller yaw-rate command (the same normalised
    // [-1,1] "yaw" the swarm integrates). Head tracking is deliberately excluded
    // so looking around doesn't move the panorama. The same command rotates the
    // OVRCameraRig to mimic body motion; the head still yaws freely relative to it.
    private void UpdateBodyYaw()
    {
        if (!bodyYawInitialized)
        {
            bodyYaw = headTransform != null ? headTransform.eulerAngles.y : 0f;
            bodyYawInitialized = true;
            return;
        }

        float normYaw = InputManager.Instance != null ? InputManager.Instance.InputStatus["yaw"] : 0f;
        float deltaYaw = normYaw * bodyYawRate * Time.deltaTime;
        if (deltaYaw == 0f) return;

        bodyYaw = Mathf.Repeat(bodyYaw + deltaYaw, 360f);

        if (driveCameraRigYaw && cameraRigTransform != null)
        {
            // Rotate the body about the world vertical at the rig's pivot. The head
            // (centerEyeAnchor) rotates with it but keeps its own HMD-tracked yaw.
            cameraRigTransform.Rotate(0f, deltaYaw, 0f, Space.World);
        }
    }

    // Lazily locate the Arena (same tag ScreenSpawn uses) for screen placement.
    private void FindArena()
    {
        if (arena == null)
        {
            arena = GameObject.FindGameObjectWithTag("Arena");
        }
    }

    // Selects the three drones whose FPV cameras straddle the body yaw: the
    // centre drone (camera yaw closest to bodyYaw) plus the yaw-neighbour on
    // each side. Mirrors the Python selection in StitcherThreading.py
    // (get_drone_order + get_subsets_from_order) so both sides agree on which
    // three views form the panorama. Returns the centre drone's yaw (used to
    // place the curved screen); 'selected' holds the camera indices written to
    // the three blocks, ordered [left, centre, right].
    private float SelectStitchCameras(float bodyYaw, out int[] selected)
    {
        if (camerasToCapture == null || camerasToCapture.Count == 0)
        {
            selected = new int[0];
            return bodyYaw;
        }

        // Candidate set: boundary drones only (convex hull). If fewer than three
        // are on the boundary, fall back to the full swarm so the panorama still forms.
        List<int> candidates = new List<int>(camerasToCapture.Count);
        for (int i = 0; i < camerasToCapture.Count; i++)
        {
            if (IsBoundary(i)) candidates.Add(i);
        }
        if (candidates.Count < 3)
        {
            candidates.Clear();
            for (int i = 0; i < camerasToCapture.Count; i++) candidates.Add(i);
        }

        int n = candidates.Count;

        // Centre = the candidate whose camera yaw is closest to bodyYaw.
        int centreCam = candidates[ClosestInList(candidates, bodyYaw)];

        // Fewer than three candidates total: send what we have.
        if (n < 3)
        {
            selected = candidates.ToArray();
            return camerasToCapture[centreCam].transform.eulerAngles.y;
        }

        // Order candidates by yaw ascending (0..360); take centre + circular neighbours.
        candidates.Sort((a, b) =>
            camerasToCapture[a].transform.eulerAngles.y.CompareTo(
            camerasToCapture[b].transform.eulerAngles.y));

        int centrePos = candidates.IndexOf(centreCam);
        int leftPos = (centrePos - 1 + n) % n;
        int rightPos = (centrePos + 1) % n;

        selected = new int[] { candidates[leftPos], candidates[centrePos], candidates[rightPos] };
        return camerasToCapture[centreCam].transform.eulerAngles.y;
    }

    // True when camera i's drone is on the swarm boundary (convex hull). Read live
    // each frame; Unity's null check covers a missing or destroyed AttitudeAlgorithm.
    private bool IsBoundary(int i)
    {
        return stitchAttitudes != null
            && i < stitchAttitudes.Count
            && stitchAttitudes[i] != null
            && stitchAttitudes[i].BoundaryEstimate;
    }

    // Local position within 'indices' whose camera yaw is closest to 'yaw' using
    // circular distance (matches the wraparound handling in
    // StitcherThreading.get_subsets_from_order).
    private int ClosestInList(List<int> indices, float yaw)
    {
        int best = 0;
        float bestDiff = float.MaxValue;
        for (int k = 0; k < indices.Count; k++)
        {
            float diff = Mathf.Abs(camerasToCapture[indices[k]].transform.eulerAngles.y - yaw);
            if (diff > 180f) diff = 360f - diff;
            if (diff < bestDiff)
            {
                bestDiff = diff;
                best = k;
            }
        }
        return best;
    }

    // Anchors the curved panorama screen at the Arena centre and rotates it so
    // its arc faces the centre drone's direction (mirrors the per-drone screen
    // placement in ScreenSpawn.UpdateRealDroneScreen). The screen therefore
    // snaps to a new direction only when the head turns far enough to change
    // the selected centre view.
    private void UpdateCurvedScreenPose(float centreYaw)
    {
        FindArena();
        if (arena == null) return;

        float radians = -centreYaw * Mathf.Deg2Rad;
        Vector3 dir = new Vector3(Mathf.Cos(radians), 0f, Mathf.Sin(radians));

        transform.position = arena.transform.position + Vector3.up * screenHeightOffset;
        transform.rotation = Quaternion.LookRotation(dir, Vector3.up);
    }

    // Lightweight per-frame write of just the body yaw into the metadata block
    // (WriteMetadata is not called every frame). The Python stitcher reads this
    // to choose the head-facing views, so it must match the yaw passed to
    // SelectStitchCameras (the body heading, not the live HMD direction).
    private void WriteBodyYaw(float yaw)
    {
        if (metadataPtr == IntPtr.Zero) return;

        byte[] yawBytes = BitConverter.GetBytes(yaw);
        if (!BitConverter.IsLittleEndian) Array.Reverse(yawBytes);
        Marshal.Copy(yawBytes, 0, IntPtr.Add(metadataPtr, metadataHeadYawOffset), 4);
    }

    // Reflects the current stitch selection in the Inspector (read-only). Only
    // rebuilds the list when the selection changes so it doesn't allocate every
    // frame. Entries are ordered [left, centre, right].
    private void UpdateStitchedDronesDisplay(int[] selected)
    {
        string key = string.Join(",", selected);
        if (key == lastStitchedDronesKey) return;
        lastStitchedDronesKey = key;

        stitchedDrones.Clear();
        for (int j = 0; j < selected.Length; j++)
        {
            string role = (selected.Length == 3)
                ? (j == 0 ? "L" : j == 1 ? "C" : "R")
                : $"#{j}";
            stitchedDrones.Add($"{role}: {DroneName(selected[j])}");
        }
    }

    // Tell ScreenSpawn which stitched drones' individual feeds to hide. Only hides
    // while the flag is on AND the panorama is actually displayed (panoramaDisplayActive);
    // otherwise pushes an empty set so all feeds show. Change-detected to avoid
    // per-frame allocation.
    private void UpdateStitchedScreenHiding(int[] selected)
    {
        if (screenSpawn == null) screenSpawn = FindObjectOfType<ScreenSpawn>();
        if (screenSpawn == null) return;

        bool hide = hideStitchedDroneScreens && panoramaDisplayActive;
        string key = hide ? string.Join(",", selected) : "off";
        if (key == lastHiddenScreensKey) return;
        lastHiddenScreensKey = key;

        if (!hide)
        {
            screenSpawn.SetStitchedDronesHidden(null);
            return;
        }

        var drones = new List<GameObject>(selected.Length);
        for (int j = 0; j < selected.Length; j++)
        {
            int camIdx = selected[j];
            if (camerasToCapture == null || camIdx < 0 || camIdx >= camerasToCapture.Count) continue;
            Camera cam = camerasToCapture[camIdx];
            if (cam == null) continue;
            Transform parent = cam.transform.parent;   // the "Drone N" GameObject
            if (parent != null) drones.Add(parent.gameObject);
        }
        screenSpawn.SetStitchedDronesHidden(drones);
    }

    // Human-readable name of the drone owning a capture camera (the FPV camera's
    // parent), for the Inspector display.
    private string DroneName(int camIdx)
    {
        if (camerasToCapture == null || camIdx < 0 || camIdx >= camerasToCapture.Count)
            return "?";
        Camera cam = camerasToCapture[camIdx];
        if (cam == null) return "?";
        Transform parent = cam.transform.parent;
        return parent != null ? parent.name : cam.name;
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
        stitchAttitudes = new List<AttitudeAlgorithm>();

        GameObject[] drones = GameObject.FindGameObjectsWithTag("DroneBase");

        foreach (GameObject drone in drones)
        {
            Camera camera = drone.transform.Find("FPV")?.GetComponent<Camera>();

            if (camera != null)
            {
                camerasToCapture.Add(camera);
                // Cache this drone's boundary estimator (aligned with camerasToCapture)
                // so only convex-hull boundary drones are selected for stitching.
                AttitudeAlgorithm attitude = drone.transform.Find("DroneParent")?.GetComponent<AttitudeAlgorithm>();
                stitchAttitudes.Add(attitude);
            }
            if(camerasToCapture.Count >maxBlockImageCount) break;
        }
    }

    // Rebuilds the (serialized) camerasToStitch bool list aligned with
    // camerasToCapture, from the cached boundary estimators. Coarse ~3 s snapshot
    // for the inspector; the live selection uses IsBoundary directly each frame.
    private void UpdateCameraToStitch()
    {
        int count = camerasToCapture != null ? camerasToCapture.Count : 0;
        camerasToStitch = new List<bool>(count);
        for (int i = 0; i < count; i++)
        {
            camerasToStitch.Add(IsBoundary(i));
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

        int newblockImageCount = Mathf.Min(STITCH_COUNT, camerasToCapture.Count);
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
        offset += 4;

        // StabStitch panorama-quality fallback parameters
        Marshal.WriteByte(metadataPtr, offset, (byte)(qualityFallbackEnabled ? 1 : 0));
        offset += 1;

        byte[] qualityThresholdBytes = BitConverter.GetBytes(qualityThreshold);
        if (!BitConverter.IsLittleEndian) Array.Reverse(qualityThresholdBytes);
        Marshal.Copy(qualityThresholdBytes, 0, IntPtr.Add(metadataPtr, offset), 4);
        offset += 4;

        // Body yaw (pilot heading). Also written every frame by WriteBodyYaw at the
        // same offset; seeded here at the head's initial yaw so the start-time full
        // write matches the first body heading (UpdateBodyYaw seeds it identically).
        byte[] headYawBytes = BitConverter.GetBytes(headTransform != null ? headTransform.eulerAngles.y : 0f);
        if (!BitConverter.IsLittleEndian) Array.Reverse(headYawBytes);
        Marshal.Copy(headYawBytes, 0, IntPtr.Add(metadataPtr, offset), 4);
        offset += 4;

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