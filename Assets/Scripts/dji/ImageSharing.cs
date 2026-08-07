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

    // Fixed capacity of the feed mapping (must match MAX_DRONES in
    // image_stream_feed.py). The mapping is always this many blocks so its size
    // never depends on the fleet size or on which process creates it first.
    // droneId == -1 marks a block that holds no new frame (never written, or
    // already consumed by the read loop); producers rewrite droneId every write.
    private const int MaxFeedBlocks = 10;

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
        
        // The feed mapping always has the full fixed capacity (matches
        // image_stream_feed.py) so its size never depends on the fleet size. Note this is
        // the FEED stride, fitted to the 800x450 feed — not StitchSlotStride, which is
        // sized to the block section's larger envelope.
        TotalProcessedSize = MaxFeedBlocks * BlockSize;
        if (enableDebugLogging) Debug.Log($"[ImageSharing] Total memory size: {TotalProcessedSize} bytes ({MaxFeedBlocks} blocks x {BlockSize} bytes per block, {numImages} screens)");

        // Create (or open) the memory-mapped file for the processed images and metadata
        processedFileMap = CreateFileMapping(new IntPtr(-1), IntPtr.Zero, PAGE_READWRITE, 0,
            (uint)TotalProcessedSize, processedMapName);
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

        // Spawn screens
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

    // Finds all GameObjects with the tag "Screen" and initializes their textures.
    // Assumes each screen's name is in the format "screen_{i}" where i is an integer.
    private void FindAndSetupScreens()
    {
        if (enableDebugLogging) Debug.Log("[ImageSharing] Finding and setting up screens...");
        screens.Clear();
        GameObject[] screenObjects = GameObject.FindGameObjectsWithTag("Screen");
        if (enableDebugLogging) Debug.Log($"[ImageSharing] Found {screenObjects.Length} GameObjects with 'Screen' tag");
        
        foreach (GameObject go in screenObjects)
        {
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

                        // Read the pose out of the same block as the pixels, so the two
                        // stay paired all the way to the stitcher. Deliberately not the
                        // yaw-offset-corrected heading above: headingOffsetDegrees exists
                        // to line the feed SCREENS up with the HMD's yaw frame, while the
                        // pose defines its own frame (+Z = North) that the derived scene
                        // plane is expressed in. Applying that offset to one and not the
                        // other would yaw the whole mosaic off the facade.
                        cached.pos = new Vector3(ReadFloat(blockPtr, PoseOffset + 0),
                                                 ReadFloat(blockPtr, PoseOffset + 4),
                                                 ReadFloat(blockPtr, PoseOffset + 8));
                        cached.rot = new Quaternion(ReadFloat(blockPtr, RotOffset + 0),
                                                    ReadFloat(blockPtr, RotOffset + 4),
                                                    ReadFloat(blockPtr, RotOffset + 8),
                                                    ReadFloat(blockPtr, RotOffset + 12));
                        cached.captureTime = ReadFloat(blockPtr, CaptureTimeOffset);
                        cached.poseStatus = Marshal.ReadInt32(blockPtr, PoseStatusOffset);
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

                        // Update screen orientation
                        ScreenSpawn.UpdateRealDroneScreen(imageIndex, yaw);

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
    private void PublishStitchBlocks()
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
            for (int j = 0; j < published; j++) WriteStitchSlot(j, stitchCandidates[j]);
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
            for (int j = 0; j < published; j++) WriteStitchSlot(j, selected[j]);
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