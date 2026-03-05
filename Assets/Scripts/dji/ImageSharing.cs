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
    
    // Block layout for each image:
    //   int flag (4 bytes), int imageIndex (4 bytes), float yaw (4 bytes), image data (ImageSize bytes)
    private const int MetadataSize = 12;  // flag (4) + index (4) + yaw (4)

    // Processed image dimensions and sizes (RGB24)
    private const int ImageWidth = 640;
    private const int ImageHeight = 360;
    private const int ImageSize = ImageWidth * ImageHeight * 3; // 3 bytes per pixel
    private int BlockSize = MetadataSize + ImageSize; // size per image block

    // Number of image blocks expected in the shared memory (update as needed)
    [SerializeField] private int numImages = 1;
    private int TotalProcessedSize;

    // Memory mapped file name
    [SerializeField] private string processedMapName = "BlockSharedMemory";

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
        
        // Compute total size of shared memory based on number of images
        TotalProcessedSize = numImages * BlockSize;
        if (enableDebugLogging) Debug.Log($"[ImageSharing] Total memory size: {TotalProcessedSize} bytes ({numImages} images x {BlockSize} bytes per block)");

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

        // Initialize all memory blocks to zero (especially the flags)
        if (enableDebugLogging) Debug.Log("[ImageSharing] Initializing memory blocks to zero...");
        for (int block = 0; block < numImages; block++)
        {
            IntPtr blockPtr = IntPtr.Add(processedPtr, block * BlockSize);
            // Set flag to 0 (ready)
            Marshal.WriteInt32(blockPtr, 0, 0);
            // Set imageIndex to 0
            Marshal.WriteInt32(blockPtr, 4, 0);
            // Set yaw to 0.0f
            Marshal.WriteInt32(blockPtr, 8, 0);
            if (enableDebugLogging) Debug.Log($"[ImageSharing] Initialized block {block}: flag=0");
        }
        if (enableDebugLogging) Debug.Log("[ImageSharing] Memory initialization complete.");

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
            
            // Loop through each image block in the memory mapped file
            for (int block = 0; block < numImages; block++)
            {
                // Compute the pointer for the current block
                IntPtr blockPtr = IntPtr.Add(processedPtr, block * BlockSize);

                // Check if the block is ready (flag is 0)
                int flag = Marshal.ReadInt32(blockPtr, 0);
                
                if (enableDebugLogging) Debug.Log($"[ImageSharing] Block {block}: flag={flag}, offset={block * BlockSize}");
                
                if (flag == 0)
                {
                    anyDataRead = true;
                    if (enableDebugLogging) Debug.Log($"[ImageSharing] Block {block} is ready (flag=0), reading data...");
                    
                    // Set flag to busy (1) so producer knows we're reading it
                    Marshal.WriteInt32(blockPtr, 0, 1);
                    if (enableDebugLogging) Debug.Log($"[ImageSharing] Set flag to 1 (busy)");

                    // Read the image index (offset 4) and yaw angle (offset 8)
                    int imageIndex = Marshal.ReadInt32(blockPtr, 4);
                    if (enableDebugLogging) Debug.Log($"[ImageSharing] Read imageIndex: {imageIndex}");

                    byte[] yawBytes = new byte[4];
                    Marshal.Copy(IntPtr.Add(blockPtr, 8), yawBytes, 0, 4);
                    float yaw = BitConverter.ToSingle(yawBytes, 0);
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
                        Marshal.WriteInt32(blockPtr, 0, 0);
                        continue;
                    }

                    // Save debug image if enabled
                    if (saveDebugImages)
                    {
                        SaveDebugImage(imageBytes, imageIndex, yaw);
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

                    // Reset the flag to 0 so the producer can write a new image block
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
            
            nextReceiveTime = Time.time + readInterval;
        }
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
        
        if (enableDebugLogging) Debug.Log($"[ImageSharing] Final stats - Total attempts: {totalReadsAttempted}, Successful: {successfulReads}, Skipped: {skippedReads}");
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