using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using UnityEngine;

public class ImageSharing : MonoBehaviour
{
    public screenSpawn screenSpawn;
    
    // Memory mapping constants and parameters
    private const uint FILE_MAP_ALL_ACCESS = 0xF001F;
    private const uint PAGE_READWRITE = 0x04;
    
    // Block layout for each image:
    //   int flag (4 bytes), int imageIndex (4 bytes), float yaw (4 bytes), image data (ImageSize bytes)
    private const int MetadataSize = 12;  // flag (4) + index (4) + yaw (4)

    // Processed image dimensions and sizes (RGB24)
    private const int ImageWidth = 1920;
    private const int ImageHeight = 1080;
    private const int ImageSize = ImageWidth * ImageHeight * 3; // 3 bytes per pixel
    private int BlockSize = MetadataSize + ImageSize; // size per image block

    // Number of image blocks expected in the shared memory (update as needed)
    [SerializeField] private int numImages = 1;
    private int TotalProcessedSize;

    // Memory mapped file name
    [SerializeField] private string processedMapName = "ProcessedImageSharedMemory";

    // Update interval for reading from the memory mapped file
    [SerializeField] private float readInterval = 0.05f;
    private float nextReceiveTime = 0f;

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
        // Compute total size of shared memory based on number of images
        TotalProcessedSize = numImages * BlockSize;

        // Create (or open) the memory-mapped file for the processed images and metadata
        processedFileMap = CreateFileMapping(new IntPtr(-1), IntPtr.Zero, PAGE_READWRITE, 0,
            (uint)TotalProcessedSize, processedMapName);
        if (processedFileMap == IntPtr.Zero)
        {
            Debug.LogError("Unable to create processed image memory map.");
            return;
        }

        processedPtr = MapViewOfFile(processedFileMap, FILE_MAP_ALL_ACCESS, 0, 0, (UIntPtr)TotalProcessedSize);
        if (processedPtr == IntPtr.Zero)
        {
            Debug.LogError("Unable to map view of processed image memory map.");
            return;
        }

        // Get the screenSpawn script if it hasn't been set
        if (screenSpawn == null)
        {
            screenSpawn = GetComponent<screenSpawn>();
        }

        // Find all screens in the scene
        FindAndSetupScreens();
    }

    // Finds all GameObjects with the tag "Screen" and initializes their textures.
    // Assumes each screen's name is in the format "screen_{i}" where i is an integer.
    private void FindAndSetupScreens()
    {
        screens.Clear();
        GameObject[] screenObjects = GameObject.FindGameObjectsWithTag("Screen");
        foreach (GameObject go in screenObjects)
        {
            int index = ParseIndexFromName(go.name);
            if (screens.ContainsKey(index))
            {
                Debug.LogWarning($"Multiple screens found with index {index}. Only one will be updated.");
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
            }
            else
            {
                Debug.LogWarning($"GameObject '{go.name}' tagged as 'Screen' does not have a MeshRenderer component.");
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
            FindAndSetupScreens();

        if (Time.time >= nextReceiveTime && processedPtr != IntPtr.Zero)
        {
            // Loop through each image block in the memory mapped file
            for (int block = 0; block < numImages; block++)
            {
                // Compute the pointer for the current block
                IntPtr blockPtr = IntPtr.Add(processedPtr, block * BlockSize);

                // Check if the block is ready (flag is 0)
                int flag = Marshal.ReadInt32(blockPtr, 0);
                if (flag == 0)
                {
                    // Set flag to busy (1) so producer knows we're reading it
                    Marshal.WriteInt32(blockPtr, 0, 1);

                    // Read the image index (offset 4) and yaw angle (offset 8)
                    int imageIndex = Marshal.ReadInt32(blockPtr, 4);

                    byte[] yawBytes = new byte[4];
                    Marshal.Copy(IntPtr.Add(blockPtr, 8), yawBytes, 0, 4);
                    float yaw = BitConverter.ToSingle(yawBytes, 0);

                    // Copy image data from shared memory (starting at offset 12)
                    byte[] imageBytes = new byte[ImageSize];
                    Marshal.Copy(IntPtr.Add(blockPtr, MetadataSize), imageBytes, 0, ImageSize);

                    // If a screen with the matching index exists, update its texture and orientation
                    if (screens.TryGetValue(imageIndex, out ScreenData screenData))
                    {
                        // Convert raw bytes (RGB24) to Color32 array
                        for (int i = 0; i < screenData.pixels.Length; i++)
                        {
                            int byteIndex = i * 3;
                            if (byteIndex + 2 < imageBytes.Length)
                            {
                                screenData.pixels[i] = new Color32(
                                    imageBytes[byteIndex],
                                    imageBytes[byteIndex + 1],
                                    imageBytes[byteIndex + 2],
                                    255);
                            }
                        }
                        // Update texture
                        screenData.texture.SetPixels32(screenData.pixels);
                        screenData.texture.Apply();

                        // Update screen using the
                        screenSpawn.UpdateScreenPosition(imageIndex, yaw);
                    }
                    else
                    {
                        Debug.LogWarning($"No screen found for image index {imageIndex}");
                    }

                    // Reset the flag to 0 so the producer can write a new image block
                    Marshal.WriteInt32(blockPtr, 0, 0);
                }
            }
            nextReceiveTime = Time.time + readInterval;
        }
    }

    void OnDestroy()
    {
        // Clean up memory mapped file resources
        if (processedPtr != IntPtr.Zero)
        {
            UnmapViewOfFile(processedPtr);
            processedPtr = IntPtr.Zero;
        }
        if (processedFileMap != IntPtr.Zero)
        {
            CloseHandle(processedFileMap);
            processedFileMap = IntPtr.Zero;
        }
    }
}
