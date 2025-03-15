using System;
using System.Runtime.InteropServices;
using UnityEngine;

public class ImageSharing : MonoBehaviour
{
    // Memory mapping constants and parameters
    private const uint FILE_MAP_ALL_ACCESS = 0xF001F;
    private const uint PAGE_READWRITE = 0x04;
    private const int FlagPosition = 0;
    private const int ProcessedDataPosition = 4;

    // Processed image dimensions and sizes (RGB24)
    private const int ImageWidth = 640;
    private const int ImageHeight = 360;
    private const int ImageSize = ImageWidth * ImageHeight * 3; // 3 bytes per pixel
    private const int TotalProcessedSize = ProcessedDataPosition + ImageSize;

    // Memory mapped file name
    [SerializeField]
    private string processedMapName = "ProcessedImageSharedMemory";

    // Update interval for reading from the memory mapped file
    [SerializeField]
    private float readInterval = 0.05f;
    private float nextReceiveTime = 0f;

    // Memory mapped file handles
    private IntPtr processedFileMap = IntPtr.Zero;
    private IntPtr processedPtr = IntPtr.Zero;

    // Texture to display the image and pixel buffer
    private Texture2D displayTexture;
    private Color32[] pixels;

    // Screen material reference (found via tag "Screen")
    private Material screenMaterial;
    // Store the Screen GameObject
    private GameObject screenObj;

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
        // Create (or open) the memory-mapped file for the processed image
        processedFileMap = CreateFileMapping(new IntPtr(-1), IntPtr.Zero, PAGE_READWRITE, 0,
            (uint)TotalProcessedSize, processedMapName);
        if (processedFileMap == IntPtr.Zero)
        {
            Debug.LogError("Unable to create processed image memory map.");
            return;
        }

        processedPtr = MapViewOfFile(processedFileMap, FILE_MAP_ALL_ACCESS, 0, 0, UIntPtr.Zero);
        if (processedPtr == IntPtr.Zero)
        {
            Debug.LogError("Unable to map view of processed image memory map.");
            return;
        }

        // Create a Texture2D to hold the processed image
        displayTexture = new Texture2D(ImageWidth, ImageHeight, TextureFormat.RGB24, false);
        pixels = new Color32[ImageWidth * ImageHeight];

        // Attempt to find the screen object initially.
        GetScreen();
    }

    // Searches for a GameObject with the tag "Screen".
    // If found, assigns its MeshRenderer's main texture to displayTexture.
    private void GetScreen()
    {
        screenObj = GameObject.FindGameObjectWithTag("Screen");
        if (screenObj != null)
        {
            MeshRenderer renderer = screenObj.GetComponent<MeshRenderer>();
            if (renderer != null)
            {
                // Set both the main texture and the emission map to the display texture.
                renderer.material.mainTexture = displayTexture;
                renderer.material.SetTexture("_EmissionMap", displayTexture);
                screenMaterial = renderer.material;
            }
            else
            {
                Debug.LogWarning("Found GameObject with tag 'Screen' but it has no MeshRenderer component.");
            }
        }
    }

    void Update()
    {
        // If the screen material is missing, try to retrieve the screen again.
        if (screenMaterial == null)
        {
            GetScreen();
        }

        if (Time.time >= nextReceiveTime && processedPtr != IntPtr.Zero)
        {
            // Check if the image is ready (flag is 0)
            if (Marshal.ReadInt32(processedPtr, FlagPosition) == 0)
            {
                // Set flag to busy
                Marshal.WriteInt32(processedPtr, FlagPosition, 1);

                // Allocate buffer for the image data and copy from shared memory (skip flag bytes)
                byte[] imageBytes = new byte[ImageSize];
                Marshal.Copy(IntPtr.Add(processedPtr, ProcessedDataPosition), imageBytes, 0, ImageSize);

                // Convert raw bytes to Color32 array (assuming RGB24 order)
                for (int i = 0; i < pixels.Length; i++)
                {
                    int byteIndex = i * 3;
                    if (byteIndex + 2 < imageBytes.Length)
                    {
                        pixels[i] = new Color32(imageBytes[byteIndex], imageBytes[byteIndex + 1], imageBytes[byteIndex + 2], 255);
                    }
                }

                // Update the texture with new pixel data
                displayTexture.SetPixels32(pixels);
                displayTexture.Apply();

                // Reset flag to 0 so that the producer can write a new image
                Marshal.WriteInt32(processedPtr, FlagPosition, 0);
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
