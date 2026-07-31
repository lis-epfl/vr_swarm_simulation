using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Text;
using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using UnityEngine;
using UnityEngine.Rendering;

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
    [Tooltip("Must be 800 in the DJI scene to match the 800x450 drone feed (image_stream_feed.py / ImageSharing.cs); StitcherThreading.py sizes itself from this via the metadata map.")]
    private int blockImageWidth = 800;

    [SerializeField]
    [Tooltip("Must be 450 in the DJI scene to match the 800x450 drone feed (image_stream_feed.py / ImageSharing.cs); StitcherThreading.py sizes itself from this via the metadata map.")]
    private int blockImageHeight = 450;

    // Read-only access so ScreenSpawn can adopt the same FPV feed resolution
    // (PyUniSharingFast is the single source of truth for the block resolution).
    public int BlockImageWidth => blockImageWidth;
    public int BlockImageHeight => blockImageHeight;

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
    private int metadataSize = 20 + 64 + 1 + 4 + 64 + 1 + 4 + 4*4 + 1 + 64 + 4 + 4 + 4 + 1 + 4 + 4 + 1; // +8 blurKernelSize+blurSigma, +4 borderSize, +1 qualityFallbackEnabled (bool), +4 qualityThreshold (float), +4 headYaw (float), +1 printStitchRate (bool)

    private IntPtr blockFileMap;
    private IntPtr blockPtr;
    private IntPtr panoramaFileMap;
    private IntPtr panoramaPtr;

    private IntPtr metadataFileMap;
    private IntPtr metadataPtr;

    public List<Camera> camerasToCapture;
    public List<bool> camerasToStitch;
    private List<AttitudeAlgorithm> stitchAttitudes;  // per-camera boundary estimator, index-aligned with camerasToCapture
    private List<StateFinder> stitchStates;            // per-camera state (IsAlive), index-aligned with camerasToCapture

    private RenderTexture reusableTexture;
    private Texture2D image;
    private byte[] blockImageBytes;   // reusable scratch for one converted drone image
    private float nextSendTime, nextReceiveTime = 0f;

    // Async capture: block images are read back from the GPU with
    // AsyncGPUReadback (no ReadPixels stall) and converted RGBA->BGR by a Burst
    // job in the completion callback. Each pending entry carries the slot /
    // droneId / heading recorded at request time (1-2 frames before completion).
    private struct PendingReadback
    {
        public bool inUse;
        public int slot;               // block slot [left, centre, right]
        public int droneId;            // camerasToCapture index
        public float heading;          // camera yaw at request time
        public byte[] verifyReference; // sync-captured reference, set only during row-order calibration
    }
    private PendingReadback[] pendingReadbacks;
    private Action<AsyncGPUReadbackRequest>[] pendingCallbacks;  // one cached delegate per pool entry
    private NativeArray<byte> convertedBlock;   // reusable BGR24 output of the conversion job
    // AsyncGPUReadback returns rows in the GPU-native order, which differs per
    // graphics API from the bottom-up order ReadPixels gives. The first readback
    // is compared against a synchronous capture of the same RT contents to pick
    // the flip that reproduces the existing top-down BGR block format exactly.
    private bool readbackFlipRows;
    private bool readbackFlipCalibrated = false;
    private bool readbackFlipCalibrating = false;

    // Row-wise RGBA32 -> BGR24 conversion (optionally flipping row order) into
    // the top-down BGR block layout the Python stitcher consumes.
    [BurstCompile]
    private struct ConvertRgbaToBgrJob : IJobParallelFor
    {
        [ReadOnly, NativeDisableParallelForRestriction] public NativeArray<byte> src;  // RGBA32
        [WriteOnly, NativeDisableParallelForRestriction] public NativeArray<byte> dst; // BGR24
        public int width;
        public int height;
        public bool flipRows;

        public void Execute(int y)
        {
            int srcRow = y * width * 4;
            int dstRow = (flipRows ? height - 1 - y : y) * width * 3;
            for (int x = 0; x < width; x++)
            {
                int s = srcRow + x * 4;
                int d = dstRow + x * 3;
                dst[d]     = src[s + 2];  // B
                dst[d + 1] = src[s + 1];  // G
                dst[d + 2] = src[s];      // R
            }
        }
    }

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
    private const int REASON_NO_OVERLAP = 1 << 4;    // selected cameras' yaw gap exceeds FOV (pre-stitch gate)
    private const int REASON_TOO_FEW_IMAGES = 1 << 5; // fewer than 3 selected feeds (pre-stitch gate)
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
    // qualityThreshold (printStitchRate follows the yaw). This carries the
    // integrated body yaw (WriteBodyYaw), not the live HMD direction, so Python
    // selects the same views as SelectStitchCameras.
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

    [SerializeField]
    [Tooltip("Key that recalibrates the body heading to the current CenterEyeAnchor yaw, so the " +
             "panorama centre and the VR velocity frame re-align with wherever the pilot is looking.")]
    private KeyCode calibrateKey = KeyCode.C;

    // Body heading that drives the panorama. Seeded once from the head's initial
    // yaw, then advanced only by the controller yaw-rate command (never by head
    // tracking). cameraRigTransform is rotated by the same command so the rig
    // turns with the body while the head still yaws freely relative to it.
    private Transform cameraRigTransform;
    private float bodyYaw;
    private bool bodyYawInitialized = false;
    // Body-locked mode (non-hull attitude modes): the pilot's yaw stick spins the
    // drone(s) directly, so the body/rig heading is slaved to the followed drone's
    // ACTUAL heading rather than the raw stick, giving zero relative motion between
    // the headset view and the feed. Tracks the followed camera and its last heading
    // so we advance by the true per-frame delta (and reseed cleanly on a drone swap).
    private Camera bodyLockCamera;
    private float previousDroneYaw;
    private bool bodyLockInitialized = false;
    // Set when the calibrate key is pressed; consumed the same frame once the centre-drone yaw
    // is known, so the recentre snaps the view onto the (discrete) panorama centre.
    private bool calibrationRequested = false;

    // Controller-integrated body heading in degrees (0-360). Exposed so the swarm's VR command
    // frame can rotate velocity commands into the pilot's heading. Mirrors the private bodyYaw.
    public static float BodyYawDegrees { get; private set; }

    // The "Drone N" root of the drone at the centre of the stitch selection (camera yaw closest to
    // the body yaw), i.e. the drone the pilot is looking through. Refreshed every frame whether or
    // not stitching is running, so systems that need "whatever the pilot is facing" can anchor on it
    // — SwarmPlaneController orients the vertical swarming plane from this drone's heading.
    public static Transform CentreStitchDrone { get; private set; }

    private GameObject arena;
    private int[] selectedStitchIndices = new int[0];  // camera indices written to the 3 blocks, ordered [left, centre, right]

    [Header("Stitching Debug")]
    [SerializeField]
    [Tooltip("Read-only: the drones currently sent to the stitcher, ordered left / centre / right. Updates during Play.")]
    private List<string> stitchedDrones = new List<string>();
    private string lastStitchedDronesKey;  // change-detection so the list only rebuilds when the selection changes

    [SerializeField]
    [Tooltip("Print the Python stitch/warp loop rate (Hz) to the console. Disable to declutter the log while reading other per-frame diagnostics (e.g. the StabStitch quality PSNR).")]
    private bool printStitchRate = true;

    // Quality fallback: switch between the panorama screen and ScreenSpawn feeds
    [SerializeField] private ScreenSpawn screenSpawn;
    private bool panoramaDisplayActive = true;

    // Pilot toggle for the panorama, driven by the controller click switch
    // (InputManager "userSwitch": 1 = show panorama, -1 = show individual feeds),
    // the Inspector checkbox below, or togglePanoramaKey. When off, the panorama
    // is hidden and the individual feeds are shown (same display path as the
    // quality fallback); the Python stitcher keeps running the whole time.
    [SerializeField]
    [Tooltip("Show the stitched panorama; unticked shows the individual drone feeds instead. " +
             "Mirrors the controller's click switch when one is connected, but can also be " +
             "toggled directly here or with togglePanoramaKey -- for testing without a controller.")]
    private bool panoramaUserEnabled = true;

    [SerializeField]
    [Tooltip("Key that toggles the panorama on/off (for testing without a controller connected).")]
    private KeyCode togglePanoramaKey = KeyCode.T;

    // Edge-detection for the controller's click switch, so a disconnected controller
    // (InputManager's "userSwitch" resting at its default) doesn't fight the manual
    // toggle above every frame -- only an actual change in the reading takes over.
    private float lastControllerUserSwitch;
    private bool controllerUserSwitchInitialized = false;

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
        // Discovery is unconditional (it only builds a few lists): the centre-drone
        // selection publishes CentreStitchDrone every frame, and consumers such as
        // SwarmPlaneController need it whether or not stitching is running.
        FindCameras();
        if (enableImageWriting)
        {
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
            // The texture object is stable (only its bytes change on each read),
            // so bind it to the curved-screen material once here.
            curvedScreenMaterial.mainTexture = panoTexture;
            curvedScreenMaterial.SetTexture("_EmissionMap", panoTexture);

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
            EnsureConvertedBlockBuffer();

            // Readback pool: enough entries for a couple of in-flight 3-slot
            // batches. One cached delegate per entry so requests never allocate.
            pendingReadbacks = new PendingReadback[8];
            pendingCallbacks = new Action<AsyncGPUReadbackRequest>[pendingReadbacks.Length];
            for (int i = 0; i < pendingCallbacks.Length; i++)
            {
                int idx = i;
                pendingCallbacks[i] = request => OnBlockReadback(idx, request);
            }
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

        if (Time.time >= nextCameraUpdateTime)
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

        // Recalibrate on demand. Seed the body heading from the current head yaw so this frame's
        // centre-drone selection is taken relative to where the pilot is looking; the recentre is
        // finished below (once centreYaw is known) by rotating the view onto that centre drone.
        if (Input.GetKeyDown(calibrateKey))
        {
            SeedBodyYawFromHead();
            calibrationRequested = true;
        }

        UpdateBodyYaw();
        WriteBodyYaw(bodyYaw);

        // Select the centre drone (camera yaw closest to the body yaw) plus the
        // two yaw-neighbours. centreYaw drives the curved-screen orientation so
        // the screen snaps to the new view only when the selection changes.
        float centreYaw = SelectStitchCameras(bodyYaw, out selectedStitchIndices);

        // Finish a pending calibration now that the centre drone is known: recentre the view onto
        // it so the head faces the (snapped) panorama centre and the VR velocity forward matches.
        if (calibrationRequested)
        {
            CalibrateToCentre(centreYaw);
            calibrationRequested = false;
        }

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
                // Queue an async GPU readback for the head-aligned drones selected
                // this frame (slots ordered left/centre/right). The block write to
                // shared memory happens in the completion callback, 1-2 frames
                // later — no ReadPixels stall on the main thread.
                for (int j = 0; j < selectedStitchIndices.Length && j < blockImageCount; j++)
                {
                    RequestBlockCapture(j, selectedStitchIndices[j]);
                }

                // Advance by whole intervals, but never fall more than one interval
                // behind — a long stall must not trigger a burst of catch-up sends.
                nextSendTime = Mathf.Max(nextSendTime + sendInterval, Time.time - sendInterval);
            }
        }

        if (Input.GetKeyDown(togglePanoramaKey))
        {
            panoramaUserEnabled = !panoramaUserEnabled;
        }

        // Mirror the pilot's panorama toggle from the click switch, but only on an
        // actual change in its reading -- not every frame -- so a disconnected
        // controller (userSwitch resting at InputManager's default) doesn't
        // immediately undo the manual toggle/Inspector checkbox above. readController.py
        // latches the spring-loaded switch, so userSwitch is a stable on/off level
        // (1 = show panorama, -1 = show feeds) while a controller is connected.
        if (InputManager.Instance != null)
        {
            float userSwitch = InputManager.Instance.InputStatus["userSwitch"];
            if (!controllerUserSwitchInitialized)
            {
                lastControllerUserSwitch = userSwitch;
                controllerUserSwitchInitialized = true;
            }
            else if (userSwitch != lastControllerUserSwitch)
            {
                panoramaUserEnabled = userSwitch > 0f;
                lastControllerUserSwitch = userSwitch;
            }
        }

        // Handle panorama reading from PanoramaSharedMemory
        if (enablePanoramaReading)
        {
            if (Time.time >= nextReceiveTime && panoramaPtr != IntPtr.Zero && Marshal.ReadInt32(panoramaPtr, FlagPosition) == 0)
            {
                Marshal.WriteInt32(panoramaPtr, FlagPosition, 1);

                int qualityWord = Marshal.ReadInt32(panoramaPtr, panoramaQualityPosition);

                // bit 0 = panorama good; bits 1-3 = failing-gate reason (only
                // meaningful when bit 0 is clear). When the panorama is bad (and
                // fallback is enabled) show the individual drone feeds instead.
                bool qualityOk = (qualityWord & QUALITY_OK_BIT) != 0;
                // The pilot's click-switch toggle overrides quality: if they turned
                // the panorama off, hide it and show the feeds regardless of quality.
                bool panoramaGood = (!qualityFallbackEnabled || qualityOk) && panoramaUserEnabled;
                if (panoramaGood)
                {
                    // Upload straight from the mapped view while the flag is held.
                    // Python writes RGB24 bottom-up (flipped + BGR->RGB on its
                    // side), exactly the layout the texture expects — no managed
                    // copy, no per-pixel conversion.
                    panoTexture.LoadRawTextureData(IntPtr.Add(panoramaPtr, panoramaDataPosition), panoramaImageSize);
                }
                Marshal.WriteInt32(panoramaPtr, FlagPosition, 0);

                ApplyQualityFallback(panoramaGood, qualityWord);
                if (panoramaGood)
                {
                    panoTexture.Apply(false);
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
            string reason = !panoramaUserEnabled
                ? "toggled off by pilot (click switch / keyboard / inspector)"
                : $"stitch quality bad: {DescribeQualityReason(qualityWord)}";
            Debug.Log($"[Panorama] hidden — showing individual feeds. {reason}.");
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
        if ((qualityWord & REASON_NO_OVERLAP) != 0)
            reasons += (reasons.Length > 0 ? ", " : "") + "no overlap (camera yaw gap exceeds FOV)";
        if ((qualityWord & REASON_TOO_FEW_IMAGES) != 0)
            reasons += (reasons.Length > 0 ? ", " : "") + "too few feeds (fewer than 3 selected)";
        return reasons.Length > 0 ? reasons : "unspecified";
    }

    // Renders (if needed) and queues an async GPU readback of one selected
    // camera into the given block slot. The camera's ScreenSpawn feed RT is the
    // capture source (SpawnScreens sizes it to the block resolution), so the
    // camera is never rendered twice: an enabled camera's RT already holds this
    // frame's image; a disabled one (feed screen hidden) is rendered on demand
    // here, at the send rate only.
    private void RequestBlockCapture(int slot, int camIdx)
    {
        if (pendingReadbacks == null) return;  // image writing was off at Start
        if (camIdx < 0 || camIdx >= camerasToCapture.Count) return;
        Camera camera = camerasToCapture[camIdx];
        if (camera == null) return;

        RenderTexture rt = camera.targetTexture;
        if (rt != null && rt.width == blockImageWidth && rt.height == blockImageHeight)
        {
            if (!camera.enabled)
            {
                camera.Render();
            }
        }
        else
        {
            // Fallback (no ScreenSpawn / mismatched RT): render into our own RT.
            // The readback request below snapshots the RT at this point in the
            // GPU command stream, so reusing one RT across cameras is safe.
            RenderTexture previousRT = camera.targetTexture;
            camera.targetTexture = reusableTexture;
            camera.Render();
            camera.targetTexture = previousRT;
            rt = reusableTexture;
        }

        int p = AcquirePendingSlot();
        if (p < 0) return;  // pool exhausted (readbacks piling up) — drop this frame

        pendingReadbacks[p].slot = slot;
        pendingReadbacks[p].droneId = camIdx;
        // Heading is recorded now, matching the image being read back — not at
        // completion, when the drone may have yawed on.
        pendingReadbacks[p].heading = camera.transform.eulerAngles.y;

        // One-time row-order calibration: capture the same RT synchronously so
        // the completion callback can pick the flip that reproduces the exact
        // top-down BGR bytes the old ReadPixels path produced.
        if (!readbackFlipCalibrated && !readbackFlipCalibrating)
        {
            readbackFlipCalibrating = true;
            RenderTexture previousActive = RenderTexture.active;
            RenderTexture.active = rt;
            image.ReadPixels(new Rect(0, 0, blockImageWidth, blockImageHeight), 0, 0, false);
            RenderTexture.active = previousActive;
            byte[] reference = new byte[blockImageSize];
            ConvertToBlockFormat(image.GetRawTextureData(), reference);
            pendingReadbacks[p].verifyReference = reference;
        }

        AsyncGPUReadback.Request(rt, 0, TextureFormat.RGBA32, pendingCallbacks[p]);
    }

    private int AcquirePendingSlot()
    {
        for (int i = 0; i < pendingReadbacks.Length; i++)
        {
            if (!pendingReadbacks[i].inUse)
            {
                pendingReadbacks[i].inUse = true;
                return i;
            }
        }
        return -1;
    }

    // Completion callback (main thread): convert the RGBA readback to the
    // top-down BGR block format with a Burst job and write it into the slot's
    // shared-memory block under the usual flag handshake. Runs 1-2 frames after
    // the request; the handshake semantics are identical to the old synchronous
    // path (a busy consumer just drops this frame).
    private void OnBlockReadback(int p, AsyncGPUReadbackRequest request)
    {
        PendingReadback pending = pendingReadbacks[p];
        pendingReadbacks[p].inUse = false;
        pendingReadbacks[p].verifyReference = null;

        // Teardown/resize safety: drop late readbacks once the maps or buffers
        // are gone or the block resolution changed under this request.
        if (request.hasError || blockPtr == IntPtr.Zero || !convertedBlock.IsCreated)
            return;

        NativeArray<byte> data = request.GetData<byte>();
        if (data.Length < blockImageWidth * blockImageHeight * 4 || convertedBlock.Length != blockImageSize)
            return;

        if (pending.verifyReference != null)
        {
            CalibrateReadbackFlip(data, pending.verifyReference);
        }

        var job = new ConvertRgbaToBgrJob
        {
            src = data,
            dst = convertedBlock,
            width = blockImageWidth,
            height = blockImageHeight,
            flipRows = readbackFlipRows,
        };
        job.Schedule(blockImageHeight, 32).Complete();
        convertedBlock.CopyTo(blockImageBytes);

        IntPtr block = IntPtr.Add(blockPtr, pending.slot * blockSize);

        // Skip this slot if the consumer is mid-read on its block.
        if (Marshal.ReadInt32(block, blockFlagOffset) != 0)
            return;

        // Mark busy while we write the header + image.
        Marshal.WriteInt32(block, blockFlagOffset, 1);

        Marshal.WriteInt32(block, blockDroneIdOffset, pending.droneId);
        byte[] headingBytes = BitConverter.GetBytes(pending.heading);
        if (!BitConverter.IsLittleEndian) Array.Reverse(headingBytes);
        Marshal.Copy(headingBytes, 0, IntPtr.Add(block, blockHeadingOffset), 4);

        Marshal.Copy(blockImageBytes, 0, IntPtr.Add(block, blockImageDataOffset), blockImageSize);

        // Ready for the consumer.
        Marshal.WriteInt32(block, blockFlagOffset, 0);
    }

    // Decides readbackFlipRows by converting the first readback both ways and
    // comparing against the synchronous ReadPixels capture of the same RT
    // contents. The winning flip reproduces the old block bytes exactly, so the
    // bridge format is provably unchanged.
    private void CalibrateReadbackFlip(NativeArray<byte> data, byte[] reference)
    {
        int noFlipMismatches = CountConversionMismatches(data, reference, false);
        int flipMismatches = noFlipMismatches == 0 ? int.MaxValue
                                                   : CountConversionMismatches(data, reference, true);
        readbackFlipRows = flipMismatches < noFlipMismatches;
        readbackFlipCalibrated = true;
        readbackFlipCalibrating = false;

        int winner = Mathf.Min(noFlipMismatches, flipMismatches);
        if (winner == 0 || noFlipMismatches == 0)
        {
            Debug.Log($"[PyUniSharingFast] AsyncGPUReadback row order calibrated: flipRows={readbackFlipRows} (byte-identical to the ReadPixels path).");
        }
        else
        {
            Debug.LogWarning($"[PyUniSharingFast] AsyncGPUReadback calibration found no exact match " +
                             $"(mismatched bytes: noFlip={noFlipMismatches}, flip={flipMismatches}); " +
                             $"using flipRows={readbackFlipRows}. Verify the panorama orientation.");
        }
    }

    // (Re)allocates the persistent conversion buffer to the current block size.
    // Any readback still in flight across a resize is dropped by the length
    // check in OnBlockReadback.
    private void EnsureConvertedBlockBuffer()
    {
        if (convertedBlock.IsCreated && convertedBlock.Length == blockImageSize) return;
        if (convertedBlock.IsCreated) convertedBlock.Dispose();
        convertedBlock = new NativeArray<byte>(blockImageSize, Allocator.Persistent, NativeArrayOptions.UninitializedMemory);
    }

    private int CountConversionMismatches(NativeArray<byte> data, byte[] reference, bool flipRows)
    {
        var job = new ConvertRgbaToBgrJob
        {
            src = data,
            dst = convertedBlock,
            width = blockImageWidth,
            height = blockImageHeight,
            flipRows = flipRows,
        };
        job.Schedule(blockImageHeight, 32).Complete();

        int mismatches = 0;
        for (int i = 0; i < blockImageSize; i++)
        {
            if (convertedBlock[i] != reference[i]) mismatches++;
        }
        return mismatches;
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
            SeedBodyYawFromHead();
            return;
        }

        // Non-hull attitude modes (NONE/SIMPLE) spin the drone(s) with the yaw stick,
        // so slave the body/rig heading to the drone's ACTUAL heading instead of the
        // raw stick. Matching rate constants wouldn't cancel relative motion: the drone
        // lags its command through the yaw filter, inner rate loop and drag.
        if (IsBodyLockedToDrone())
        {
            UpdateBodyYawFromDrone();
            return;
        }

        // Hull modes: the drones hold their heading and the stick steers the view.
        bodyLockInitialized = false; // reseed if we later switch back to a body-locked mode
        float normYaw = InputManager.Instance != null ? InputManager.Instance.InputStatus["yaw"] : 0f;
        float deltaYaw = normYaw * bodyYawRate * Time.deltaTime;
        if (deltaYaw == 0f) return;

        bodyYaw = Mathf.Repeat(bodyYaw + deltaYaw, 360f);
        BodyYawDegrees = bodyYaw;

        if (driveCameraRigYaw && cameraRigTransform != null)
        {
            // Rotate the body about the world vertical at the rig's pivot. The head
            // (centerEyeAnchor) rotates with it but keeps its own HMD-tracked yaw.
            cameraRigTransform.Rotate(0f, deltaYaw, 0f, Space.World);
        }
    }

    // True in the non-hull attitude modes, where the yaw stick spins the drone(s)
    // directly (AttitudeAlgorithm feeds inputYawRate into desiredYawRate). The hull
    // modes instead zero the drone yaw and steer the view by the stick, so they keep
    // the input-integrated path above. Defaults to the input path if no SwarmManager.
    private bool IsBodyLockedToDrone()
    {
        SwarmManager sm = SwarmManager.Instance;
        if (sm == null) return false;
        SwarmManager.AttitudeAlgorithm algo = sm.GetSelectedAttitudeAlgorithm();
        return algo != SwarmManager.AttitudeAlgorithm.LOCAL_CONVEXHULL
            && algo != SwarmManager.AttitudeAlgorithm.GLOBAL_CONVEXHULL;
    }

    // Advance the body heading by the followed drone's true per-frame yaw change and
    // rotate the rig by the same amount, so the pilot turns exactly with the drone
    // (no relative motion between the headset view and the feed). The followed drone
    // is the alive camera whose heading is closest to the current body yaw (the view
    // centre); on a drone swap we reseed rather than emit a phantom jump.
    private void UpdateBodyYawFromDrone()
    {
        Camera lockCam = GetBodyLockCamera();
        if (lockCam == null)
        {
            bodyLockInitialized = false; // nothing to follow this frame; hold heading
            return;
        }

        float droneYaw = lockCam.transform.eulerAngles.y;
        if (!bodyLockInitialized || lockCam != bodyLockCamera)
        {
            bodyLockCamera = lockCam;
            previousDroneYaw = droneYaw;
            bodyLockInitialized = true;
            return;
        }

        float deltaYaw = Mathf.DeltaAngle(previousDroneYaw, droneYaw);
        previousDroneYaw = droneYaw;
        if (deltaYaw == 0f) return;

        bodyYaw = Mathf.Repeat(bodyYaw + deltaYaw, 360f);
        BodyYawDegrees = bodyYaw;

        if (driveCameraRigYaw && cameraRigTransform != null)
        {
            cameraRigTransform.Rotate(0f, deltaYaw, 0f, Space.World);
        }
    }

    // The alive camera whose heading is closest to the current body yaw (the drone the
    // pilot is facing). In NONE every drone yaws together, so for a single drone this is
    // just that drone, and for a swarm it stays locked to the same centre drone.
    private Camera GetBodyLockCamera()
    {
        if (camerasToCapture == null || camerasToCapture.Count == 0) return null;

        int best = -1;
        float bestDiff = float.MaxValue;
        for (int i = 0; i < camerasToCapture.Count; i++)
        {
            if (camerasToCapture[i] == null || !IsAlive(i)) continue;
            float diff = Mathf.Abs(Mathf.DeltaAngle(camerasToCapture[i].transform.eulerAngles.y, bodyYaw));
            if (diff < bestDiff)
            {
                bestDiff = diff;
                best = i;
            }
        }
        return best >= 0 ? camerasToCapture[best] : null;
    }

    // Snap the body heading to the current head (CenterEyeAnchor) world yaw. Used both to
    // seed the heading on the first frame and to recalibrate on demand (the calibrateKey),
    // so the panorama centre and the VR velocity frame re-align with wherever the pilot is
    // currently looking. The rig is deliberately not rotated here — only the reference
    // heading moves; CalibrateToCentre finishes the on-demand recentre.
    private void SeedBodyYawFromHead()
    {
        bodyYaw = headTransform != null ? headTransform.eulerAngles.y : 0f;
        bodyYawInitialized = true;
        BodyYawDegrees = bodyYaw;
    }

    // Finishes an on-demand calibration. The panorama stays snapped to the centre drone's yaw,
    // so to align it with the pilot we recentre the *view*: rotate the OVRCameraRig so the head's
    // forward points at the panorama centre (the screen's local +Z direction `dir`), then set the
    // body heading to centreYaw so the stitch selection and VR velocity frame stay in the camera-yaw
    // frame the rest of the system uses. The head look direction is 90 deg off from
    // world-yaw==centreYaw because of how the curved screen is oriented, so we align the head to
    // `dir` rather than to centreYaw itself. The view is only rotated when the rig is being driven
    // (driveCameraRigYaw); otherwise the heading still aligns without moving the view. Also
    // raises/lowers the rig so the pilot's eyes end up level with the curved screen.
    private void CalibrateToCentre(float centreYaw)
    {
        if (driveCameraRigYaw && cameraRigTransform != null && headTransform != null)
        {
            // Aim the head at the panorama's centre, not at the raw camera yaw. The
            // curved screen's centre column lies along the GameObject's local +Z (see
            // GenerateCurvedScreen: the mid vertex is at (0, y, radius)), and
            // UpdateCurvedScreenPose points that +Z at `dir`. So the pilot faces the
            // panorama centre exactly when the head's forward equals `dir` — which is
            // 90 deg off from world-yaw==centreYaw. Compute `dir` identically to
            // UpdateCurvedScreenPose (same frame, same centreYaw) and rotate the rig
            // by the signed horizontal angle from the current head forward onto it.
            float radians = -centreYaw * Mathf.Deg2Rad;
            Vector3 dir = new Vector3(Mathf.Cos(radians), 0f, Mathf.Sin(radians));

            Vector3 headForward = headTransform.forward;
            headForward.y = 0f;
            float delta = Vector3.SignedAngle(headForward, dir, Vector3.up);
            cameraRigTransform.Rotate(0f, delta, 0f, Space.World);
        }

        // Raise/lower the rig so the pilot's eyes sit level with the curved screen
        // (arena centre + screenHeightOffset, the same height used in
        // UpdateCurvedScreenPose). The head (centerEyeAnchor) tracks vertically
        // relative to the rig root, so shift the rig by the gap between the current
        // head height and the target screen height. A yaw rotation about world-up
        // never changes the head's Y, so reading it after the yaw recentre is safe.
        if (cameraRigTransform != null && headTransform != null)
        {
            FindArena();
            if (arena != null)
            {
                float screenY = arena.transform.position.y + screenHeightOffset;
                float headOffsetFromRig = headTransform.position.y - cameraRigTransform.position.y;
                Vector3 rigPos = cameraRigTransform.position;
                rigPos.y = screenY - headOffsetFromRig;
                cameraRigTransform.position = rigPos;
            }
        }

        bodyYaw = Mathf.Repeat(centreYaw, 360f);
        BodyYawDegrees = bodyYaw;
        bodyYawInitialized = true;
        WriteBodyYaw(bodyYaw);
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
            CentreStitchDrone = null;
            return bodyYaw;
        }

        // Candidate set: alive boundary drones only (convex hull). If fewer than three
        // are on the boundary, fall back to all alive drones so the panorama still forms.
        List<int> candidates = new List<int>(camerasToCapture.Count);
        for (int i = 0; i < camerasToCapture.Count; i++)
        {
            if (IsAlive(i) && IsBoundary(i)) candidates.Add(i);
        }
        if (candidates.Count < 3)
        {
            candidates.Clear();
            for (int i = 0; i < camerasToCapture.Count; i++)
            {
                if (IsAlive(i)) candidates.Add(i);
            }
        }

        int n = candidates.Count;

        // Centre = the candidate whose camera yaw is closest to bodyYaw.
        int centreCam = candidates[ClosestInList(candidates, bodyYaw)];

        // Publish the centre drone (the FPV camera's parent is the "Drone N" root, see DroneName)
        // so other systems can anchor on whatever the pilot is looking at — SwarmPlaneController
        // uses it to orient the vertical swarming plane.
        CentreStitchDrone = camerasToCapture[centreCam].transform.parent;

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

    // True when camera i's drone is still alive (DroneHealthMonitor parks dead drones
    // far below the course). A missing/destroyed StateFinder is treated as alive so a
    // setup gap never blanks the panorama. Read live each frame, like IsBoundary.
    private bool IsAlive(int i)
    {
        return stitchStates == null
            || i >= stitchStates.Count
            || stitchStates[i] == null
            || stitchStates[i].IsAlive;
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

    private void FindCameras()
    {
        camerasToCapture = new List<Camera>();
        stitchAttitudes = new List<AttitudeAlgorithm>();
        stitchStates = new List<StateFinder>();

        GameObject[] drones = GameObject.FindGameObjectsWithTag("DroneBase");

        foreach (GameObject drone in drones)
        {
            Camera camera = drone.transform.Find("FPV")?.GetComponent<Camera>();

            if (camera != null)
            {
                camerasToCapture.Add(camera);
                // Cache this drone's boundary estimator (aligned with camerasToCapture)
                // so only convex-hull boundary drones are selected for stitching.
                Transform droneParent = drone.transform.Find("DroneParent");
                AttitudeAlgorithm attitude = droneParent?.GetComponent<AttitudeAlgorithm>();
                stitchAttitudes.Add(attitude);
                // Cache this drone's state too, so dead drones are excluded from stitching.
                StateFinder state = droneParent?.GetComponent<VelocityControl>()?.State;
                stitchStates.Add(state);
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
                EnsureConvertedBlockBuffer();
                CreateBlockMap();
            }

            // Update reusable resources for panorama reading
            if (enablePanoramaReading)
            {
                panoTexture = new Texture2D(panoramaImageWidth, panoramaImageHeight, TextureFormat.RGB24, false);
                if (curvedScreenMaterial != null)
                {
                    curvedScreenMaterial.mainTexture = panoTexture;
                    curvedScreenMaterial.SetTexture("_EmissionMap", panoTexture);
                }
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
        // Re-discovery keeps CentreStitchDrone live for non-stitching consumers, so it runs
        // regardless; only the shared-memory resize below depends on image writing.
        FindCameras();
        UpdateCameraToStitch();

        if (!enableImageWriting) return;

        int newblockImageCount = Mathf.Min(STITCH_COUNT, camerasToCapture.Count);
        if (newblockImageCount != blockImageCount)
        {
            blockImageCount = newblockImageCount;
            CalculateMemorySizes();
            CreateBlockMap();          // resize the mapping to the new drone count
            WriteMetadata();
            blockImageBytes = new byte[blockImageSize];
            EnsureConvertedBlockBuffer();
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

        // Console-verbosity toggle: gates Python's per-loop stitch/warp rate (Hz) prints.
        Marshal.WriteByte(metadataPtr, offset, (byte)(printStitchRate ? 1 : 0));
        offset += 1;

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
        // Flush in-flight readbacks before unmapping so a late completion
        // callback can never touch a dead pointer, then free the job buffer.
        AsyncGPUReadback.WaitAllRequests();
        DestroyMemoryMaps();
        if (convertedBlock.IsCreated) convertedBlock.Dispose();
    }

    void OnApplicationQuit()
    {
        AsyncGPUReadback.WaitAllRequests();
        DestroyMemoryMaps();
        Debug.Log("Application quitting. Memory maps destroyed.");
    }
}