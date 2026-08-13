using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using UnityEngine;

public class ImageSharing : MonoBehaviour
{
    public ScreenSpawn ScreenSpawn;
    public DroneIndicator DroneIndicator;
    
    // Memory mapping constants and parameters
    private const uint FILE_MAP_ALL_ACCESS = 0xF001F;
    private const uint PAGE_READWRITE = 0x04;
    // What CreateFileMapping returns when the named section already exists at a SMALLER
    // size than the one requested. Windows never resizes a section, so an unequal size
    // between the two producers, or between a producer and a stale Python, is fatal —
    // which is why every section here has a compile-time constant size.
    private const int ERROR_ACCESS_DENIED = 5;
    
    // Block layout for each image (wire v2), mirroring PyUniSharingFast's
    // blockPoseHeaderSize and utils/imageSharingUtil.py's BLOCK_HEADER_BYTES:
    //   int flag | int droneId | float yaw
    //   | float camPos[3] | float camRot[4] xyzw | float captureTime | int poseStatus
    //   | image data (ImageSize bytes)
    //
    // The pose is what the sim's PLANAR stitcher needs and the v1 12-byte header could
    // not carry: a heading alone is one scalar, giving no position and one of three
    // rotation degrees of freedom. It is passed straight through from the feed map to
    // the stitch map -- this component does not compute it, because the pose has to be
    // the pose of *that* frame and only the producer knows which telemetry arrived with
    // the pixels.
    private const int MetadataSize = 48;
    private const int PoseOffset = 12;          // float32 x, y, z
    private const int RotOffset = 24;           // float32 x, y, z, w
    private const int CaptureTimeOffset = 40;   // float32, seconds since producer start
    private const int PoseStatusOffset = 44;    // int32 bitfield
    private const int POSE_VALID = 1 << 0;

    // Processed image dimensions and sizes (RGB24). Must match the producer
    // (image_stream_feed.py width/height) — the block layout is offset-based,
    // so a mismatch misaligns every block in the mapping.
    private const int ImageWidth = 800;
    private const int ImageHeight = 450;
    private const int ImageSize = ImageWidth * ImageHeight * 3; // 3 bytes per pixel
    // Stride of a DroneFeedSharedMemory block. Fitted exactly to the 800x450 feed, because
    // that map's producer (image_stream_feed.py) uses the same arithmetic. NOT the stride of
    // a BlockSharedMemory slot — see StitchSlotStride below, which is larger.
    private const int BlockSize = MetadataSize + ImageSize;

    // Fixed capacity of the feed mapping (must match FEED_MAX_DRONES in the
    // DJI_Swarm repo's utils/imageSharingUtil.py). The mapping is always this many
    // blocks so its size never depends on the fleet size or on which process creates
    // it first. droneId == -1 marks a block that holds no new frame (never written,
    // or already consumed by the read loop); producers rewrite droneId every write.
    private const int MaxFeedBlocks = 10;
    private const int FeedBlocksBytes = MaxFeedBlocks * BlockSize;

    // ---------------------------------------------------------------------------
    // Scene-plane trailer, appended after the blocks. Carries the PLANAR standoff
    // the PC computes (from a facade traced on the GUI map plus the live formation)
    // so planarStandoffMetres stops being typed into the inspector by hand. Getting
    // it wrong is the dominant mosaic error: 30 m typed against a true 34.3 m on the
    // 2026-08-11 MED clips is ~21 px of seam, more than everything else combined.
    //
    // WHY GROWING THIS SECTION IS SAFE. A named Windows section cannot be resized,
    // so asking for a larger one than a peer already created normally fails with
    // ERROR_ACCESS_DENIED. It works here only because Windows compares PAGE-ROUNDED
    // sizes, and the block array (10,800,480 B) rounds up to 10,801,152, leaving 672
    // already-backed bytes. Measured: create at the block size and open at +672
    // succeed in either order; +673 is denied. So a Unity carrying this trailer and
    // a DJI_Swarm predating it interoperate in both start orders, with the trailer
    // simply reading zero. check_wire_layout.py asserts the <= 672 bound, because
    // past it both cross orders become a hard failure and the feed dies.
    //
    // This component is the SOLE consumer; the PC is the sole producer. Unity must
    // never write these bytes -- in particular Start()'s block-init loop stops at
    // MaxFeedBlocks, because the PC may legitimately have written already.
    private const int FeedTrailerOffset = FeedBlocksBytes;
    private const int FeedTrailerBytes = 64;
    private const int FeedSectionBytes = FeedTrailerOffset + FeedTrailerBytes;

    // 'PSO1'. A fresh section is zero-filled, so a non-zero magic is what separates
    // "a PC has written here" from "this value happens to be 0" -- without it an
    // untouched trailer reads as a legal-looking 0.0 m standoff.
    private const int FeedTrailerMagic = 0x50534F31;
    private const int FeedTrailerVersion = 1;

    private const int FeedTrMagicOffset = 0;
    private const int FeedTrVersionOffset = 4;
    private const int FeedTrSeqOffset = 8;
    private const int FeedTrHeartbeatOffset = 12;
    private const int FeedTrStandoffOffset = 16;
    private const int FeedTrStatusOffset = 20;
    private const int FeedTrFacadeIdOffset = 24;
    private const int FeedTrLookOffOffset = 28;
    private const int FeedTrSpreadOffset = 32;
    private const int FeedTrPxPerMOffset = 36;
    private const int FeedTrTiltOffset = 40;
    private const int FeedTrViewCountOffset = 44;
    private const int FeedTrEnd = 48;

    private const int FeedTrStatusLocked = 1;
    private const int FeedTrStatusDwelling = 2;
    private const int FeedTrStatusNoFacade = 4;
    private const int FeedTrStatusNoOrigin = 8;

    // How long the trailer's heartbeat may stand still before the standoff is treated
    // as dead. This is the load-bearing validity test, not a nicety: closing the
    // producer's handle does NOT destroy the section, so every byte it last wrote stays
    // readable forever. Without it, one finished clip_replay would pin its clip's
    // standoff into every later session of the editor.
    private const float FeedStandoffMaxAgeSeconds = 2.0f;
    // Sanity envelope. Nothing plausible is outside it, and a value that is says the
    // producer is confused rather than that the wall is 900 m away.
    private const float FeedStandoffMinM = 0.5f;
    private const float FeedStandoffMaxM = 500.0f;

    // The PC-computed PLANAR scene-plane standoff, for PyUniSharingFast to republish.
    //
    // Static for the same reason PyUniSharingFast.PlanarSelected and BodyYawDegrees are,
    // pointing the other way: the component that owns this map and the component that
    // owns MetadataSharedMemory are not the same one, and in every sim scene one of them
    // does not exist. PyUniSharingFast must NOT map DroneFeedSharedMemory itself — one
    // producer, one consumer, and this component is the consumer.
    public static bool FeedStandoffValid { get; private set; }
    public static float FeedStandoffMetres { get; private set; }
    public static int FeedStandoffFacadeId { get; private set; } = -1;
    public static int FeedStandoffStatus { get; private set; }

    // False when the section was already held at the pre-trailer size (see Start).
    private bool feedTrailerAvailable = true;
    private int lastTrailerHeartbeat = int.MinValue;
    private float lastTrailerBeatTime = -1f;

    private static void ClearFeedStandoff()
    {
        FeedStandoffValid = false;
        FeedStandoffMetres = 0f;
        FeedStandoffFacadeId = -1;
        FeedStandoffStatus = 0;
    }

    // Number of screens/indicators to spawn (the fleet size for this run).
    // All MaxFeedBlocks blocks are polled regardless.
    [SerializeField] private int numImages = 1;
    private int TotalProcessedSize;

    // Memory mapped file name (all-drone feed map written by image_stream_feed.py)
    [SerializeField] private string processedMapName = "DroneFeedSharedMemory";

    [Header("Stitcher Feed")]
    [Tooltip("Re-publish the selected feeds into the stitcher's BlockSharedMemory: the 3 " +
             "body-yaw-selected ones under STABSTITCH (mirroring PyUniSharingFast." +
             "SelectStitchCameras in the sim), every fresh feed under PLANAR.")]
    [SerializeField] private bool enableStitchWriting = true;

    [Tooltip("Frames older than this (seconds) are excluded from stitch selection, so a drone that stops streaming drops out of the panorama.")]
    [SerializeField] private float stitchFrameMaxAge = 1f;

    [Tooltip("Degrees added to every drone's compass heading to align compass north with the Unity/HMD yaw frame. Applied to screens, indicators and stitch selection alike.")]
    [SerializeField] private float headingOffsetDegrees = 0f;

    // The stitcher input map, consumed by StitcherThreading.py. Same contract
    // PyUniSharingFast produces in the sim: a FIXED array of StitchSlotCapacity slots at a
    // FIXED StitchSlotStride, created once and never resized.
    //
    // The three constants below are mirrors of PyUniSharingFast's blockSlotCapacity /
    // maxBlockWidth x maxBlockHeight / blockSlotStride, and tools/check_wire_layout.py
    // asserts they agree. They must, for two independent reasons: this component CREATES
    // the section while PyUniSharingFast DESCRIBES it (only that component writes metadata),
    // and both components request the same named section — Windows opens the existing one
    // rather than resizing, so unequal sizes mean whichever starts second is denied.
    //
    // There is deliberately no slot-count knob here any more. It used to default to 3
    // (STABSTITCH's number) and had to be raised by hand to the fleet size before flying
    // PLANAR — and only took effect on a Play restart, because the section was created in
    // Start. PLANAR now simply fills as many of these slots as there are aircraft with a
    // fresh, posed frame.
    private const string stitchMapName = "BlockSharedMemory";
    private const int StitchSlotCapacity = 24;
    private const int StitchMaxImageWidth = 1280;
    private const int StitchMaxImageHeight = 720;
    private const int StitchSlotStride = MetadataSize + StitchMaxImageWidth * StitchMaxImageHeight * 3;
    private const int StitchSectionBytes = StitchSlotCapacity * StitchSlotStride;

    // A left/centre/right panorama is always exactly three views.
    private const int STITCH_COUNT_LRC = 3;
    private IntPtr stitchFileMap = IntPtr.Zero;
    private IntPtr stitchPtr = IntPtr.Zero;

    // Latest frame per drone id, kept in the block's native format (BGR,
    // top-down) so re-publishing to the stitch map is a straight copy.
    private class CachedFrame
    {
        public byte[] imageBytes;
        public float yaw;
        public float lastUpdateTime;

        // Carried through from the feed block untouched. poseStatus == 0 means the
        // producer had no pose to give (no GPS lock, or a tool like image_replay.py that
        // has no telemetry at all); the planar solve drops such a view and STABSTITCH
        // never looks at it.
        public Vector3 pos;
        public Quaternion rot;
        public float captureTime;
        public int poseStatus;
    }
    private readonly Dictionary<int, CachedFrame> frameCache = new Dictionary<int, CachedFrame>();
    private readonly List<int> stitchCandidates = new List<int>();

    // The drone ids this frame's selection put into the panorama, and change-detection on the
    // set actually pushed to ScreenSpawn. Feeds the "hide the screens of the drones already in
    // the panorama" rule, which the sim gets from PyUniSharingFast.UpdateStitchedScreenHiding —
    // that half resolves the selection to "Drone N" GameObjects, of which this scene has none.
    private readonly List<int> publishedStitchIds = new List<int>();
    private string lastHiddenFeedsKey;

    // Update interval for reading from the memory mapped file
    [SerializeField] private float readInterval = 0.05f;
    private float nextReceiveTime = 0f;

    // Debug logging toggle
    [SerializeField] private bool enableDebugLogging = true;

    // Debug image validation and saving
    [SerializeField] private bool enableImageValidation = true;
    [SerializeField] private bool saveDebugImages = false;
    [SerializeField] private string debugImagePath = "Assets/DebugImages/";

    // Memory mapped file handles
    private IntPtr processedFileMap = IntPtr.Zero;
    private IntPtr processedPtr = IntPtr.Zero;

    // Data structure for holding screen data
    private class ScreenData
    {
        public GameObject screenObject;
        public Material material;
        public Texture2D texture;
        public Color32[] pixels;
    }
    // Dictionary mapping image index to its corresponding screen data
    private Dictionary<int, ScreenData> screens = new Dictionary<int, ScreenData>();

    // Debug tracking
    private int totalReadsAttempted = 0;
    private int successfulReads = 0;
    private int skippedReads = 0;

    // Import Windows API functions
    [DllImport("kernel32.dll", SetLastError = true, CharSet = CharSet.Auto)]
    private static extern IntPtr CreateFileMapping(IntPtr hFile, IntPtr lpFileMappingAttributes,
        uint flProtect, uint dwMaximumSizeHigh, uint dwMaximumSizeLow, string lpName);

    [DllImport("kernel32.dll", SetLastError = true)]
    private static extern IntPtr MapViewOfFile(IntPtr hFileMappingObject, uint dwDesiredAccess,
        uint dwFileOffsetHigh, uint dwFileOffsetLow, UIntPtr dwNumberOfBytesToMap);

    [DllImport("kernel32.dll", SetLastError = true)]
    private static extern bool UnmapViewOfFile(IntPtr lpBaseAddress);

    [DllImport("kernel32.dll", SetLastError = true)]
    private static extern bool CloseHandle(IntPtr hObject);

    void Start()
    {
        if (enableDebugLogging) Debug.Log("[ImageSharing] Starting ImageSharing component...");

        // Statics survive a Play session in the editor (domain reload is configurable),
        // so a sim scene entered after this one would otherwise inherit a dead
        // controller's standoff and use it as if it were live. Cleared here and in
        // OnDestroy, at both ends of this component's life.
        ClearFeedStandoff();

        // The feed mapping always has the full fixed capacity (matches
        // imageSharingUtil.FEED_*) so its size never depends on the fleet size. Note this
        // is the FEED stride, fitted to the 800x450 feed — not StitchSlotStride, which is
        // sized to the block section's larger envelope.
        TotalProcessedSize = FeedSectionBytes;
        if (enableDebugLogging) Debug.Log($"[ImageSharing] Total memory size: {TotalProcessedSize} bytes ({MaxFeedBlocks} blocks x {BlockSize} bytes per block + {FeedTrailerBytes} trailer, {numImages} screens)");

        // Create (or open) the memory-mapped file for the processed images and metadata
        processedFileMap = CreateFileMapping(new IntPtr(-1), IntPtr.Zero, PAGE_READWRITE, 0,
            (uint)TotalProcessedSize, processedMapName);
        if (processedFileMap == IntPtr.Zero && Marshal.GetLastWin32Error() == ERROR_ACCESS_DENIED)
        {
            // A DJI_Swarm predating the scene-plane trailer already holds this section at
            // the smaller size. The blocks are laid out identically, so retry without the
            // trailer: the feeds are worth more than the standoff, and losing them both
            // over a setting that has a perfectly good fallback would be the wrong trade.
            //
            // This should not be reachable — the trailer fits inside the section's 4 KB
            // page rounding, which is why check_wire_layout.py asserts that bound. It is
            // here because "unreachable" and "untested" are the same thing in a path that
            // otherwise kills the feed for the whole session.
            feedTrailerAvailable = false;
            TotalProcessedSize = FeedBlocksBytes;
            processedFileMap = CreateFileMapping(new IntPtr(-1), IntPtr.Zero, PAGE_READWRITE, 0,
                (uint)TotalProcessedSize, processedMapName);
            Debug.LogWarning(
                $"[ImageSharing] {processedMapName} already exists at the pre-trailer size " +
                $"({FeedBlocksBytes} bytes). Feeds are unaffected; the PLANAR standoff falls " +
                $"back to PyUniSharingFast.planarStandoffMetres. Restart the DJI_Swarm " +
                $"producer to get the PC-computed standoff.");
        }
        if (processedFileMap == IntPtr.Zero)
        {
            int error = Marshal.GetLastWin32Error();
            Debug.LogError($"[ImageSharing] Unable to create processed image memory map. Error code: {error}");
            return;
        }
        if (enableDebugLogging) Debug.Log($"[ImageSharing] Memory map '{processedMapName}' created successfully. Handle: {processedFileMap}");

        processedPtr = MapViewOfFile(processedFileMap, FILE_MAP_ALL_ACCESS, 0, 0, (UIntPtr)TotalProcessedSize);
        if (processedPtr == IntPtr.Zero)
        {
            int error = Marshal.GetLastWin32Error();
            Debug.LogError($"[ImageSharing] Unable to map view of processed image memory map. Error code: {error}");
            return;
        }
        if (enableDebugLogging) Debug.Log($"[ImageSharing] Memory view mapped successfully. Pointer: {processedPtr}");

        // Initialize all memory blocks (especially the flags). imageIndex is set
        // to -1 as a "no drone has written here yet" marker so the read loop can
        // skip empty capacity blocks; producers overwrite it on their first write.
        if (enableDebugLogging) Debug.Log("[ImageSharing] Initializing memory blocks...");
        for (int block = 0; block < MaxFeedBlocks; block++)
        {
            IntPtr blockPtr = IntPtr.Add(processedPtr, block * BlockSize);
            // Set flag to 0 (ready)
            Marshal.WriteInt32(blockPtr, 0, 0);
            // Set imageIndex to -1 (unwritten marker)
            Marshal.WriteInt32(blockPtr, 4, -1);
            // Set yaw to 0.0f
            Marshal.WriteInt32(blockPtr, 8, 0);
            if (enableDebugLogging) Debug.Log($"[ImageSharing] Initialized block {block}: flag=0");
        }
        if (enableDebugLogging) Debug.Log("[ImageSharing] Memory initialization complete.");

        // Create (or open) the stitcher's 3-slot input map. StitcherThreading.py
        // consumes it; in the DJI scene this component is its sole producer
        // (PyUniSharingFast must keep enableImageWriting disabled).
        if (enableStitchWriting)
        {
            CreateStitchMap();
        }

        // Get the ScreenSpawn script if it hasn't been set
        if (ScreenSpawn == null)
        {
            if (enableDebugLogging) Debug.Log("[ImageSharing] Getting ScreenSpawn component...");
            ScreenSpawn = GetComponent<ScreenSpawn>();
        }

        ScreenSpawn.numScreens = numImages;

        // Share the staleness threshold, so the screens and the stitch selection agree on what
        // counts as still flying: a drone that drops out leaves the panorama and the layout together.
        ScreenSpawn.SetRealFeedTimeout(stitchFrameMaxAge);

        // Spawn screens. No swarm list means the real-drone path: ScreenSpawn binds one screen per
        // feed index and every screen style then works off the state pushed by UpdateRealDroneFeed.
        ScreenSpawn.SpawnScreens();
        if (enableDebugLogging) Debug.Log($"[ImageSharing] ScreenSpawn component found and spawned {numImages} screens.");

        // Spawn drone heading indicators
        if (DroneIndicator != null)
        {
            DroneIndicator.SpawnIndicators(numImages);
            if (enableDebugLogging) Debug.Log($"[ImageSharing] DroneIndicator spawned {numImages} indicator(s).");
        }

        // Find all screens in the scene
        FindAndSetupScreens();
        
        if (enableDebugLogging) Debug.Log($"[ImageSharing] Initialization complete. Read interval: {readInterval}s");
    }

    // Binds a texture to each screen ScreenSpawn created, keyed by its spawn index.
    //
    // Takes the screens from ScreenSpawn.Screens rather than searching the scene for the "Screen"
    // tag. GameObject.FindGameObjectsWithTag only returns ACTIVE objects, and the layouts hide any
    // screen whose feed has not arrived yet — which, when this runs from Start, is every one of
    // them. The tag search therefore found nothing, and since assigning the texture here is what
    // the per-frame update needs in order to push a feed's heading and position back into
    // ScreenSpawn, nothing ever un-hid them. That deadlock presented exactly as "the DJI scene
    // displays no feeds", and the Update() retry below could never break it either.
    //
    // Falls back to the tag search only when ScreenSpawn has nothing to offer, so a scene that
    // wires the screens up some other way still works.
    private void FindAndSetupScreens()
    {
        if (enableDebugLogging) Debug.Log("[ImageSharing] Finding and setting up screens...");
        screens.Clear();

        IReadOnlyList<GameObject> spawned = ScreenSpawn != null ? ScreenSpawn.Screens : null;
        List<GameObject> screenObjects = new List<GameObject>();
        if (spawned != null && spawned.Count > 0)
        {
            screenObjects.AddRange(spawned);
            if (enableDebugLogging) Debug.Log($"[ImageSharing] Taking {screenObjects.Count} screens from ScreenSpawn");
        }
        else
        {
            screenObjects.AddRange(GameObject.FindGameObjectsWithTag("Screen"));
            if (enableDebugLogging) Debug.Log($"[ImageSharing] Found {screenObjects.Count} GameObjects with 'Screen' tag");
        }

        foreach (GameObject go in screenObjects)
        {
            if (go == null)
            {
                continue;
            }

            int index = ParseIndexFromName(go.name);
            if (enableDebugLogging) Debug.Log($"[ImageSharing] Processing screen '{go.name}' with index {index}");

            if (screens.ContainsKey(index))
            {
                Debug.LogWarning($"[ImageSharing] Multiple screens found with index {index}. Only one will be updated.");
                continue;
            }

            // Create a texture and pixel buffer for this screen
            Texture2D tex = new Texture2D(ImageWidth, ImageHeight, TextureFormat.RGB24, false);
            Color32[] pix = new Color32[ImageWidth * ImageHeight];

            // Set the texture to the screen's material
            MeshRenderer renderer = go.GetComponent<MeshRenderer>();
            if (renderer != null)
            {
                renderer.material.mainTexture = tex;
                renderer.material.SetTexture("_EmissionMap", tex);
                if (enableDebugLogging) Debug.Log($"[ImageSharing] Texture assigned to screen '{go.name}'");
            }
            else
            {
                Debug.LogWarning($"[ImageSharing] GameObject '{go.name}' tagged as 'Screen' does not have a MeshRenderer component.");
            }

            ScreenData data = new ScreenData
            {
                screenObject = go,
                material = (renderer != null) ? renderer.material : null,
                texture = tex,
                pixels = pix
            };

            screens.Add(index, data);
        }
        
        if (enableDebugLogging) Debug.Log($"[ImageSharing] Screen setup complete. {screens.Count} screens ready.");
    }

    // Helper method to extract an integer index from a GameObject's name.
    // Assumes the name is in the format "screen_{i}".
    private int ParseIndexFromName(string name)
    {
        string prefix = "screen_";
        if (name.StartsWith(prefix, StringComparison.OrdinalIgnoreCase))
        {
            string numStr = name.Substring(prefix.Length);
            if (int.TryParse(numStr, out int index))
                return index;
        }
        return 0;
    }

    void Update()
    {
        // (Optionally) refresh screens if needed (for example, if new ones are added at runtime)
        if (screens.Count == 0)
        {
            if (enableDebugLogging) Debug.LogWarning("[ImageSharing] No screens found, attempting to find and setup screens...");
            FindAndSetupScreens();
        }

        if (Time.time >= nextReceiveTime && processedPtr != IntPtr.Zero)
        {
            // Before the block scan and outside its bookkeeping, deliberately: the
            // standoff must keep being reported honestly even while the feed path is
            // producing nothing, which is exactly when someone is looking at it. Same
            // argument WriteHeartbeat uses for sitting ahead of every early return.
            ReadFeedTrailer();

            totalReadsAttempted++;
            bool anyDataRead = false;

            // Loop through each capacity block in the memory mapped file
            for (int block = 0; block < MaxFeedBlocks; block++)
            {
                // Compute the pointer for the current block
                IntPtr blockPtr = IntPtr.Add(processedPtr, block * BlockSize);

                // Check if the block is ready (flag is 0)
                int flag = Marshal.ReadInt32(blockPtr, 0);
                
                if (enableDebugLogging) Debug.Log($"[ImageSharing] Block {block}: flag={flag}, offset={block * BlockSize}");
                
                if (flag == 0)
                {
                    // Set flag to busy (1) so producer knows we're reading it
                    Marshal.WriteInt32(blockPtr, 0, 1);

                    // Read the image index (offset 4) and yaw angle (offset 8)
                    int imageIndex = Marshal.ReadInt32(blockPtr, 4);

                    // droneId == -1 means "no new frame": either never written
                    // (marker from Start) or already consumed below. The producer
                    // rewrites droneId on every write, which clears the marker.
                    if (imageIndex < 0)
                    {
                        Marshal.WriteInt32(blockPtr, 0, 0);
                        continue;
                    }

                    anyDataRead = true;
                    if (enableDebugLogging) Debug.Log($"[ImageSharing] Block {block} is ready (flag=0), reading imageIndex: {imageIndex}");

                    byte[] yawBytes = new byte[4];
                    Marshal.Copy(IntPtr.Add(blockPtr, 8), yawBytes, 0, 4);
                    // Normalise to 0-360 and apply the compass-to-Unity yaw frame
                    // offset (used consistently by screens, indicators and the
                    // stitch selection below).
                    float yaw = Mathf.Repeat(BitConverter.ToSingle(yawBytes, 0) + headingOffsetDegrees, 360f);
                    if (enableDebugLogging) Debug.Log($"[ImageSharing] Read yaw: {yaw}");

                    // Copy image data from shared memory (starting at offset 12)
                    byte[] imageBytes = new byte[ImageSize];
                    Marshal.Copy(IntPtr.Add(blockPtr, MetadataSize), imageBytes, 0, ImageSize);
                    if (enableDebugLogging) Debug.Log($"[ImageSharing] Copied {ImageSize} bytes of image data");

                    // Validate image data
                    bool isValidImage = enableImageValidation ? ValidateImageData(imageBytes, imageIndex) : true;
                    if (!isValidImage)
                    {
                        Debug.LogWarning($"[ImageSharing] Image {imageIndex} failed validation");
                        Marshal.WriteInt32(blockPtr, 4, -1);  // consumed: don't retry this frame
                        Marshal.WriteInt32(blockPtr, 0, 0);
                        continue;
                    }

                    // Save debug image if enabled
                    if (saveDebugImages)
                    {
                        SaveDebugImage(imageBytes, imageIndex, yaw);
                    }

                    // Read the pose out of the same block as the pixels, so the two stay
                    // paired all the way to the stitcher. Deliberately not the
                    // yaw-offset-corrected heading above: headingOffsetDegrees exists to
                    // line the feed SCREENS up with the HMD's yaw frame, while the pose
                    // defines its own frame (+Z = North) that the derived scene plane is
                    // expressed in. Applying that offset to one and not the other would
                    // yaw the whole mosaic off the facade.
                    //
                    // Read unconditionally, not under enableStitchWriting: the screen
                    // layouts need the position too (ScreenSpawn ranks the formation
                    // grids by it), and making that depend on an unrelated stitcher
                    // toggle is the kind of coupling nobody finds from the symptom.
                    Vector3 camPos = new Vector3(ReadFloat(blockPtr, PoseOffset + 0),
                                                 ReadFloat(blockPtr, PoseOffset + 4),
                                                 ReadFloat(blockPtr, PoseOffset + 8));
                    Quaternion camRot = new Quaternion(ReadFloat(blockPtr, RotOffset + 0),
                                                       ReadFloat(blockPtr, RotOffset + 4),
                                                       ReadFloat(blockPtr, RotOffset + 8),
                                                       ReadFloat(blockPtr, RotOffset + 12));
                    float blockCaptureTime = ReadFloat(blockPtr, CaptureTimeOffset);
                    int blockPoseStatus = Marshal.ReadInt32(blockPtr, PoseStatusOffset);

                    // Cache the frame for the stitch re-publish. imageBytes is a
                    // fresh array each read, so keeping the reference is safe.
                    if (enableStitchWriting)
                    {
                        if (!frameCache.TryGetValue(imageIndex, out CachedFrame cached))
                        {
                            cached = new CachedFrame();
                            frameCache[imageIndex] = cached;
                        }
                        cached.imageBytes = imageBytes;
                        cached.yaw = yaw;
                        cached.lastUpdateTime = Time.time;
                        cached.pos = camPos;
                        cached.rot = camRot;
                        cached.captureTime = blockCaptureTime;
                        cached.poseStatus = blockPoseStatus;
                    }

                    // If a screen with the matching index exists, update its texture and orientation
                    if (screens.TryGetValue(imageIndex, out ScreenData screenData))
                    {
                        if (enableDebugLogging) Debug.Log($"[ImageSharing] Updating screen with index {imageIndex}");
                        
                        // Convert raw bytes (RGB24) to Color32 array with vertical flip
                        // SetPixels32 expects bottom-left origin, so we flip the image
                        ConvertAndFlipImage(imageBytes, screenData.pixels);
                        
                        // Update texture
                        screenData.texture.SetPixels32(screenData.pixels);
                        screenData.texture.Apply();
                        if (enableDebugLogging) Debug.Log($"[ImageSharing] Texture updated for screen {imageIndex}");

                        // Hand the layout this drone's heading and position. ScreenSpawn places
                        // the screen itself, in whatever style is configured — the feeds go
                        // through the same styles as simulated ones, which is why this pushes
                        // state rather than a position.
                        ScreenSpawn.UpdateRealDroneFeed(
                            imageIndex, yaw, LayoutPosition(camPos),
                            (blockPoseStatus & POSE_VALID) != 0);

                        // Update drone heading indicator
                        DroneIndicator?.UpdateYaw(imageIndex, yaw);
                        
                        successfulReads++;
                    }
                    else
                    {
                        Debug.LogWarning($"[ImageSharing] No screen found for image index {imageIndex}");
                    }

                    // Mark the block consumed (droneId = -1) before releasing it,
                    // so we only process genuinely new frames: without this the
                    // last frame of a drone that stopped streaming would be
                    // re-read every cycle and stay "fresh" for stitch selection
                    // forever. The producer's next write restores droneId.
                    Marshal.WriteInt32(blockPtr, 4, -1);
                    Marshal.WriteInt32(blockPtr, 0, 0);
                    if (enableDebugLogging) Debug.Log($"[ImageSharing] Reset flag to 0 (ready for next write)");
                }
                else
                {
                    skippedReads++;
                    if (enableDebugLogging) Debug.Log($"[ImageSharing] Block {block} is busy (flag={flag}), skipping...");
                }
            }
            
            if (!anyDataRead && enableDebugLogging)
            {
                Debug.LogWarning($"[ImageSharing] No data read this cycle. Total attempts: {totalReadsAttempted}, Successful: {successfulReads}, Skipped: {skippedReads}");
            }

            // Re-publish the 3 body-yaw-selected feeds to the stitcher.
            PublishStitchBlocks();

            nextReceiveTime = Time.time + readInterval;
        }
    }

    // The pose position expressed in the same yaw frame as the heading pushed alongside it.
    //
    // headingOffsetDegrees rotates the compass heading into the HMD's yaw frame and is deliberately
    // NOT applied to the pose on the stitcher path (see the read loop): there the pose defines its
    // own +Z = North frame, which the derived scene plane is expressed in, and turning one without
    // the other would yaw the mosaic off the facade. The screen layouts are the one place the two
    // meet — FORMATION_WALL takes its basis from the headings and then projects these positions onto
    // it — so the rotation is applied here, at the push, and nowhere else. A zero offset (the
    // default) leaves the position untouched.
    private Vector3 LayoutPosition(Vector3 camPos)
    {
        if (Mathf.Approximately(headingOffsetDegrees, 0f))
        {
            return camPos;
        }
        return Quaternion.Euler(0f, headingOffsetDegrees, 0f) * camPos;
    }

    // Creates the stitcher's BlockSharedMemory and readies its flags. Called once, from
    // Start. Same layout PyUniSharingFast produces in the sim (wire v2, see MetadataSize)
    // and — crucially — the same size, since both request the same name.
    private void CreateStitchMap()
    {
        stitchFileMap = CreateFileMapping(new IntPtr(-1), IntPtr.Zero, PAGE_READWRITE, 0,
            (uint)StitchSectionBytes, stitchMapName);
        if (stitchFileMap == IntPtr.Zero)
        {
            int errorCode = Marshal.GetLastWin32Error();
            Debug.LogError($"[ImageSharing] Unable to create stitch memory map '{stitchMapName}' " +
                           $"({StitchSectionBytes} bytes). Error code: {errorCode}" +
                           (errorCode == ERROR_ACCESS_DENIED
                               ? " (ACCESS_DENIED: a section of this name already exists at a smaller " +
                                 "size — a stitcher built against a different StitchSlotCapacity or " +
                                 "image envelope is still running. Stop it, then restart Play.)"
                               : ""));
            return;
        }

        stitchPtr = MapViewOfFile(stitchFileMap, FILE_MAP_ALL_ACCESS, 0, 0, (UIntPtr)StitchSectionBytes);
        if (stitchPtr == IntPtr.Zero)
        {
            Debug.LogError($"[ImageSharing] Unable to map view of stitch memory map. Error code: {Marshal.GetLastWin32Error()}");
            return;
        }

        for (int slot = 0; slot < StitchSlotCapacity; slot++)
        {
            IntPtr p = IntPtr.Add(stitchPtr, slot * StitchSlotStride);
            Marshal.WriteInt32(p, 0, 0);
            // droneId = -1 at creation, not just when a slot goes unused. A fresh
            // section is zero-filled and 0 is a legal drone id, so a never-written slot
            // would otherwise advertise itself as a ready block from drone 0 carrying an
            // all-zero pose — which is a degenerate quaternion downstream. With a
            // fixed-capacity section most slots stay at this sentinel for the whole
            // session, so this loop is the only thing standing between a small fleet and
            // a section full of phantom drone-0 views.
            Marshal.WriteInt32(p, 4, -1);
            Marshal.WriteInt32(p, PoseStatusOffset, 0);
        }
        if (enableDebugLogging) Debug.Log($"[ImageSharing] Stitch map '{stitchMapName}' ready ({StitchSlotCapacity} slots x {StitchSlotStride} bytes).");
    }

    // Chooses which feeds form the panorama and writes them to the stitch map.
    //
    // Two rules, because the two stitchers want different things:
    //   STABSTITCH  the three fresh feeds straddling the pilot's body yaw, ordered
    //               [left, centre, right]. Mirrors PyUniSharingFast.SelectStitchCameras
    //               and Python's get_subsets_from_order, so both sides of the bridge
    //               agree on which views form the panorama.
    //   PLANAR      every fresh feed, full stop (the section's capacity is the only cap,
    //               and a real fleet never reaches it). A facade wall has all its drones
    //               looking at the same surface, so a yaw-ordered pick of three would throw
    //               away most of the mosaic; and the planar solve does not use ring order
    //               at all.
    //
    // Always followed by the screen-hiding push, including on the early returns below: "nothing
    // was published" has to reach ScreenSpawn as an empty set, or the last selection stays hidden
    // after the feeds stop arriving.
    /// <summary>
    /// Read the PC-computed scene-plane standoff out of the feed map's trailer and
    /// publish it on the statics PyUniSharingFast republishes into metadata.
    ///
    /// Four independent tests have to pass before the value is used, and each one
    /// exists because of a distinct way a shared section lies:
    ///   * magic    — a fresh section is zero-filled, so this separates "a PC wrote
    ///                here" from "the value written happens to be 0".
    ///   * version  — a producer from another revision must be refused, not misread.
    ///   * status   — the producer itself saying it has no facade or no pose origin.
    ///   * heartbeat— the only test that catches a producer that has EXITED. Closing
    ///                its handle leaves every byte readable forever, so without this a
    ///                finished clip_replay pins its clip's standoff into every later
    ///                session.
    /// </summary>
    private void ReadFeedTrailer()
    {
        if (!feedTrailerAvailable || processedPtr == IntPtr.Zero) return;
        IntPtr tr = IntPtr.Add(processedPtr, FeedTrailerOffset);

        if (Marshal.ReadInt32(tr, FeedTrMagicOffset) != FeedTrailerMagic ||
            Marshal.ReadInt32(tr, FeedTrVersionOffset) != FeedTrailerVersion)
        {
            SetFeedStandoff(false, 0f, -1, 0, "no producer has written a scene plane");
            return;
        }

        // Seqlock, matching PyUniSharingFast.WriteDynamicState. The standoff is a lone
        // aligned float32 and cannot tear on its own; the lock is what stops a standoff
        // being paired with a DIFFERENT facade's id in the log line, and it is what the
        // 16 reserved trailer bytes will need the day a plane normal goes in them.
        int standoffBits = 0, status = 0, facadeId = -1, beat = 0;
        bool clean = false;
        for (int attempt = 0; attempt < 3; attempt++)
        {
            int seq = Marshal.ReadInt32(tr, FeedTrSeqOffset);
            if ((seq & 1) != 0) continue;                // writer mid-update
            standoffBits = Marshal.ReadInt32(tr, FeedTrStandoffOffset);
            status = Marshal.ReadInt32(tr, FeedTrStatusOffset);
            facadeId = Marshal.ReadInt32(tr, FeedTrFacadeIdOffset);
            beat = Marshal.ReadInt32(tr, FeedTrHeartbeatOffset);
            if (Marshal.ReadInt32(tr, FeedTrSeqOffset) == seq) { clean = true; break; }
        }
        if (!clean)
        {
            // Take the standoff and the heartbeat anyway and let only the DIAGNOSTICS be
            // stale. A busy writer must never cost the mosaic its plane — dropping the
            // frame here would make the panorama flicker on a race that is, at worst, a
            // wrong facade id in a log line.
            standoffBits = Marshal.ReadInt32(tr, FeedTrStandoffOffset);
            status = Marshal.ReadInt32(tr, FeedTrStatusOffset);
            beat = Marshal.ReadInt32(tr, FeedTrHeartbeatOffset);
        }

        if (beat != lastTrailerHeartbeat)
        {
            lastTrailerHeartbeat = beat;
            lastTrailerBeatTime = Time.time;
        }

        float standoff = BitConverter.ToSingle(BitConverter.GetBytes(standoffBits), 0);
        if ((status & (FeedTrStatusNoFacade | FeedTrStatusNoOrigin)) != 0)
        {
            SetFeedStandoff(false, 0f, -1, status,
                            (status & FeedTrStatusNoOrigin) != 0
                                ? "producer has no pose origin yet"
                                : "producer has no facade to measure against");
            return;
        }
        if (lastTrailerBeatTime < 0f ||
            Time.time - lastTrailerBeatTime > FeedStandoffMaxAgeSeconds)
        {
            SetFeedStandoff(false, 0f, -1, status,
                            $"producer heartbeat stalled for " +
                            $"{Time.time - lastTrailerBeatTime:F1} s");
            return;
        }
        if (float.IsNaN(standoff) || float.IsInfinity(standoff) ||
            standoff < FeedStandoffMinM || standoff > FeedStandoffMaxM)
        {
            SetFeedStandoff(false, 0f, -1, status,
                            $"standoff {standoff:F2} m outside " +
                            $"[{FeedStandoffMinM}, {FeedStandoffMaxM}] m");
            return;
        }
        SetFeedStandoff(true, standoff, facadeId, status, null);
    }

    /// <summary>
    /// Publish the statics, logging only on a transition — never the value.
    ///
    /// The standoff moves by centimetres every frame; a per-frame log would bury the
    /// two things worth seeing, which are that the SOURCE changed (the operator has
    /// silently gone back to the inspector value) and that the FACADE changed (a step
    /// in the published plane, which the plane sweep has to re-acquire from).
    /// The operator's live read-outs are the controller console and the GUI.
    /// </summary>
    private void SetFeedStandoff(bool valid, float metres, int facadeId, int status,
                                 string why)
    {
        if (valid != FeedStandoffValid)
        {
            if (valid)
                Debug.Log($"[ImageSharing] PLANAR standoff now from the PC: " +
                          $"{metres:F2} m (facade {facadeId}). " +
                          $"planarStandoffMetres in the inspector is the fallback.");
            else
                Debug.Log($"[ImageSharing] PLANAR standoff falling back to " +
                          $"PyUniSharingFast.planarStandoffMetres — {why}.");
        }
        else if (valid && facadeId != FeedStandoffFacadeId)
        {
            Debug.Log($"[ImageSharing] PLANAR scene plane switched to facade " +
                      $"{facadeId}: {metres:F2} m (was facade {FeedStandoffFacadeId}). " +
                      $"The plane has stepped; the stitcher's sweep will re-acquire.");
        }
        FeedStandoffValid = valid;
        FeedStandoffMetres = metres;
        FeedStandoffFacadeId = valid ? facadeId : -1;
        FeedStandoffStatus = status;
    }

    private void PublishStitchBlocks()
    {
        publishedStitchIds.Clear();
        WriteStitchBlocks();
        PushStitchedScreenHiding();
    }

    private void WriteStitchBlocks()
    {
        if (!enableStitchWriting || stitchPtr == IntPtr.Zero) return;

        // Candidates: drones with a frame fresher than stitchFrameMaxAge.
        stitchCandidates.Clear();
        foreach (KeyValuePair<int, CachedFrame> kv in frameCache)
        {
            if (Time.time - kv.Value.lastUpdateTime <= stitchFrameMaxAge)
            {
                stitchCandidates.Add(kv.Key);
            }
        }

        bool planar = PyUniSharingFast.PlanarSelected;

        // A planar mosaic is worth showing from two overlapping views (MIN_PLANAR_IMAGES);
        // a left/centre/right panorama needs three (MIN_STITCH_IMAGES). Below that, leave
        // the slots untouched so the stitcher's own gates hide the panorama.
        int minViews = planar ? 2 : STITCH_COUNT_LRC;
        if (stitchCandidates.Count < minViews) return;

        int published;
        if (planar)
        {
            // Sorted by drone id rather than by yaw: under a shared heading the yaws are
            // all nearly equal, so ordering by them is a tie broken by noise and the slot
            // a drone lands in would change every frame. Python keys its debug palette and
            // its per-drone corrections on the id, not the slot, but a stable order still
            // makes the logs readable.
            stitchCandidates.Sort();
            published = Mathf.Min(stitchCandidates.Count, StitchSlotCapacity);
            for (int j = 0; j < published; j++)
            {
                WriteStitchSlot(j, stitchCandidates[j]);
                publishedStitchIds.Add(stitchCandidates[j]);
            }
        }
        else
        {
            // Centre = heading closest to the pilot's body yaw (circular distance).
            float bodyYaw = PyUniSharingFast.BodyYawDegrees;
            int centreId = stitchCandidates[0];
            float bestDiff = float.MaxValue;
            foreach (int id in stitchCandidates)
            {
                float diff = Mathf.Abs(Mathf.DeltaAngle(frameCache[id].yaw, bodyYaw));
                if (diff < bestDiff)
                {
                    bestDiff = diff;
                    centreId = id;
                }
            }

            // Order candidates by heading ascending and take the circular neighbours.
            stitchCandidates.Sort((a, b) => frameCache[a].yaw.CompareTo(frameCache[b].yaw));
            int n = stitchCandidates.Count;
            int centrePos = stitchCandidates.IndexOf(centreId);
            int[] selected =
            {
                stitchCandidates[(centrePos - 1 + n) % n],
                centreId,
                stitchCandidates[(centrePos + 1) % n],
            };

            published = STITCH_COUNT_LRC;
            for (int j = 0; j < published; j++)
            {
                WriteStitchSlot(j, selected[j]);
                publishedStitchIds.Add(selected[j]);
            }
        }

        // Retire the slots this frame's selection did not reach — all the way to the
        // section's capacity, not just to some published count. Without it a drone that
        // drops out leaves its last frame in the mosaic forever: the flag handshake alone
        // cannot distinguish a fresh block from a stale one, which is why droneId == -1 is
        // the marker on both maps.
        for (int j = published; j < StitchSlotCapacity; j++)
        {
            IntPtr slot = IntPtr.Add(stitchPtr, j * StitchSlotStride);
            if (Marshal.ReadInt32(slot, 0) != 0) continue;
            Marshal.WriteInt32(slot, 4, -1);
            Marshal.WriteInt32(slot, PoseStatusOffset, 0);
        }
    }

    // Hide the individual screens of the feeds now composited into the panorama — the real-drone
    // half of PyUniSharingFast's hideStitchedDroneScreens, which reaches those screens only if it
    // comes from here: that component resolves its selection to "Drone N" GameObjects and this
    // scene has none, so its set is always empty and every feed stayed visible.
    //
    // The gate stays where the toggle and the quality fallback live (PyUniSharingFast.
    // HideStitchedFeeds) so both paths hide on exactly the same condition; only the membership is
    // decided here. Change-detected because the set is otherwise identical every read cycle.
    private void PushStitchedScreenHiding()
    {
        if (ScreenSpawn == null) return;

        bool hide = PyUniSharingFast.HideStitchedFeeds && publishedStitchIds.Count > 0;
        string key = hide ? string.Join(",", publishedStitchIds) : "off";
        if (key == lastHiddenFeedsKey) return;
        lastHiddenFeedsKey = key;

        ScreenSpawn.SetStitchedRealFeedsHidden(hide ? publishedStitchIds : null);
    }

    // Copies one cached frame, pose included, into stitch slot j.
    private void WriteStitchSlot(int j, int droneId)
    {
        CachedFrame frame = frameCache[droneId];
        IntPtr slot = IntPtr.Add(stitchPtr, j * StitchSlotStride);

        // Skip this slot if the stitcher is mid-read (same handshake as the sim
        // producer in PyUniSharingFast).
        if (Marshal.ReadInt32(slot, 0) != 0) return;
        Marshal.WriteInt32(slot, 0, 1);

        Marshal.WriteInt32(slot, 4, droneId);
        WriteFloat(slot, 8, frame.yaw);

        // Passed through unchanged from the feed block. This component never computes a
        // pose: it has to be the pose of the frame the pixels came from, and only the
        // producer knows which telemetry sample arrived with them.
        WriteFloat(slot, PoseOffset + 0, frame.pos.x);
        WriteFloat(slot, PoseOffset + 4, frame.pos.y);
        WriteFloat(slot, PoseOffset + 8, frame.pos.z);
        WriteFloat(slot, RotOffset + 0, frame.rot.x);
        WriteFloat(slot, RotOffset + 4, frame.rot.y);
        WriteFloat(slot, RotOffset + 8, frame.rot.z);
        WriteFloat(slot, RotOffset + 12, frame.rot.w);
        WriteFloat(slot, CaptureTimeOffset, frame.captureTime);
        Marshal.WriteInt32(slot, PoseStatusOffset, frame.poseStatus);

        Marshal.Copy(frame.imageBytes, 0, IntPtr.Add(slot, MetadataSize), ImageSize);

        Marshal.WriteInt32(slot, 0, 0);
    }

    // Marshal has no WriteSingle, so floats go through their bytes. Same approach the
    // yaw write has always used here; the 4-byte array per call is a few KB/s at the
    // 20 Hz publish rate and is not worth an unsafe block to avoid.
    private static void WriteFloat(IntPtr basePtr, int offset, float value)
    {
        byte[] bytes = BitConverter.GetBytes(value);
        Marshal.Copy(bytes, 0, IntPtr.Add(basePtr, offset), 4);
    }

    private static readonly byte[] readScratch = new byte[4];

    private static float ReadFloat(IntPtr basePtr, int offset)
    {
        // Reads run on the main thread only (Update), so one shared scratch buffer is
        // safe and keeps the per-frame allocation out of the read loop.
        Marshal.Copy(IntPtr.Add(basePtr, offset), readScratch, 0, 4);
        return BitConverter.ToSingle(readScratch, 0);
    }

    // Efficiently converts RGB24 byte array to Color32 array with vertical flip
    // This is optimized to process row-by-row for better cache performance
    private void ConvertAndFlipImage(byte[] imageBytes, Color32[] pixels)
    {
        int rowBytes = ImageWidth * 3; // Number of bytes per row
        
        // Process each row
        for (int y = 0; y < ImageHeight; y++)
        {
            // Calculate source row (top to bottom) and destination row (bottom to top)
            int srcRowStart = y * rowBytes;
            int destRowStart = (ImageHeight - 1 - y) * ImageWidth;
            
            // Process each pixel in the row
            for (int x = 0; x < ImageWidth; x++)
            {
                int srcIndex = srcRowStart + x * 3;
                int destIndex = destRowStart + x;
                
                // Reversed order because source is BGR and we want RGB
                pixels[destIndex] = new Color32(
                    imageBytes[srcIndex + 2],
                    imageBytes[srcIndex + 1],
                    imageBytes[srcIndex],
                    255
                );
            }
        }
    }

    void OnDestroy()
    {
        if (enableDebugLogging) Debug.Log("[ImageSharing] Cleaning up resources...");

        // The statics outlive this component (see the declarations). Leaving a standoff
        // behind would have the next scene — quite possibly a sim scene with no feed map
        // at all — republish a dead controller's number as if it were live.
        ClearFeedStandoff();

        // Clean up memory mapped file resources
        if (processedPtr != IntPtr.Zero)
        {
            UnmapViewOfFile(processedPtr);
            processedPtr = IntPtr.Zero;
            if (enableDebugLogging) Debug.Log("[ImageSharing] Unmapped view of file");
        }
        if (processedFileMap != IntPtr.Zero)
        {
            CloseHandle(processedFileMap);
            processedFileMap = IntPtr.Zero;
            if (enableDebugLogging) Debug.Log("[ImageSharing] Closed file mapping handle");
        }
        if (stitchPtr != IntPtr.Zero)
        {
            UnmapViewOfFile(stitchPtr);
            stitchPtr = IntPtr.Zero;
        }
        if (stitchFileMap != IntPtr.Zero)
        {
            CloseHandle(stitchFileMap);
            stitchFileMap = IntPtr.Zero;
        }

        if (enableDebugLogging) Debug.Log($"[ImageSharing] Final stats - Total attempts: {totalReadsAttempted}, Successful: {successfulReads}, Skipped: {skippedReads}");
    }

    // Mirrors PyUniSharingFast, which has had both hooks for a while. OnDestroy alone is
    // not enough: on a quit that tears the scene down in an unusual order, a section whose
    // handle is never closed outlives the process only until the OS reclaims it — but the
    // window in between is exactly when a restarted Play tries to create it again.
    void OnApplicationQuit()
    {
        OnDestroy();
    }

    // Validates image data to check if it looks reasonable
    private bool ValidateImageData(byte[] imageBytes, int imageIndex)
    {
        if (imageBytes == null || imageBytes.Length == 0)
        {
            Debug.LogWarning($"[ImageSharing] Image {imageIndex}: NULL or empty image data");
            return false;
        }

        if (imageBytes.Length != ImageSize)
        {
            Debug.LogWarning($"[ImageSharing] Image {imageIndex}: Size mismatch. Expected {ImageSize}, got {imageBytes.Length}");
            return false;
        }

        // Check if all bytes are zero (likely uninitialized or corrupt)
        bool allZero = true;
        for (int i = 0; i < imageBytes.Length; i++)
        {
            if (imageBytes[i] != 0)
            {
                allZero = false;
                break;
            }
        }
        if (allZero)
        {
            Debug.LogWarning($"[ImageSharing] Image {imageIndex}: All bytes are zero (likely uninitialized)");
            return false;
        }

        // Calculate statistics for debug info
        int minVal = 255, maxVal = 0;
        long sum = 0;
        for (int i = 0; i < imageBytes.Length; i++)
        {
            minVal = Mathf.Min(minVal, imageBytes[i]);
            maxVal = Mathf.Max(maxVal, imageBytes[i]);
            sum += imageBytes[i];
        }
        double avgVal = (double)sum / imageBytes.Length;

        if (enableDebugLogging)
        {
            Debug.Log($"[ImageSharing] Image {imageIndex}: VALID - Stats: Min={minVal}, Max={maxVal}, Avg={avgVal:F2}");
        }

        return true;
    }

    // Saves image data to disk as PNG for inspection
    private void SaveDebugImage(byte[] imageBytes, int imageIndex, float yaw)
    {
        if (!saveDebugImages || imageBytes == null) return;

        try
        {
            // Create temporary pixel array and convert with flip
            Color32[] tempPixels = new Color32[ImageWidth * ImageHeight];
            ConvertAndFlipImage(imageBytes, tempPixels);

            // Create temporary texture
            Texture2D tempTexture = new Texture2D(ImageWidth, ImageHeight, TextureFormat.RGB24, false);
            tempTexture.SetPixels32(tempPixels);
            tempTexture.Apply();

            // Encode to PNG
            byte[] pngData = tempTexture.EncodeToPNG();
            Destroy(tempTexture);

            // Create directory if needed
            if (!System.IO.Directory.Exists(debugImagePath))
            {
                System.IO.Directory.CreateDirectory(debugImagePath);
            }

            // Generate filename with timestamp
            string timestamp = System.DateTime.Now.ToString("yyyy-MM-dd_HH-mm-ss-fff");
            string filename = $"{debugImagePath}image_{imageIndex}_yaw_{yaw:F2}_{timestamp}.png";
            
            System.IO.File.WriteAllBytes(filename, pngData);
            if (enableDebugLogging)
            {
                Debug.Log($"[ImageSharing] Saved debug image to: {filename}");
            }
        }
        catch (System.Exception ex)
        {
            Debug.LogError($"[ImageSharing] Failed to save debug image: {ex.Message}");
        }
    }
}