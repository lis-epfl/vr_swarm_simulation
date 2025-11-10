using System;
using UnityEngine;

/// <summary>
/// DroneDepthCamera.cs - Captures depth information from drone cameras
/// 
/// This script renders depth as a texture that can be used alongside RGB images.
/// Depth values represent distance from the camera in meters.
/// 
/// Usage:
/// 1. Attach this script to each drone GameObject (alongside the RGB camera)
/// 2. Assign the drone's main camera to the rgbCamera field
/// 3. The script will automatically create and configure a depth camera
/// </summary>
public class DroneDepthCamera : MonoBehaviour
{
    [Header("Camera References")]
    [Tooltip("The main RGB camera on this drone")]
    public Camera rgbCamera;
    
    [Header("Depth Settings")]
    [Tooltip("Resolution for depth texture - Set to 0 to auto-match RGB camera")]
    public int depthWidth = 0; // 0 = auto-match RGB camera
    public int depthHeight = 0; // 0 = auto-match RGB camera
    
    [Tooltip("Maximum depth distance in meters")]
    public float maxDepthDistance = 100.0f;
    
    [Tooltip("Minimum depth distance in meters")]
    public float minDepthDistance = 0.1f;
    
    [Header("Debug")]
    [Tooltip("Enable debug visualization")]
    public bool showDebugVisualization = false;
    
    // Internal components
    private Camera depthCamera;
    private RenderTexture depthRenderTexture;
    private Texture2D depthTexture;
    private Material depthMaterial;
    private bool isInitialized = false;
    
    void Start()
    {
        InitializeDepthCamera();
    }
    
    void OnValidate()
    {
        // Helper to show RGB camera info in inspector
        if (rgbCamera != null && Application.isEditor)
        {
            string info = $"RGB Camera Info:\n";
            if (rgbCamera.targetTexture != null)
            {
                info += $"Resolution: {rgbCamera.targetTexture.width}x{rgbCamera.targetTexture.height}\n";
                info += $"Format: {rgbCamera.targetTexture.format}\n";
            }
            info += $"FOV: {rgbCamera.fieldOfView}°\n";
            info += $"Aspect: {rgbCamera.aspect:F2}";
            
            // This will show in console when you select the object
            if (depthWidth == 0 || depthHeight == 0)
            {
                Debug.Log($"[DroneDepthCamera] {gameObject.name} - Will auto-match RGB camera resolution\n{info}");
            }
        }
    }
    
    void InitializeDepthCamera()
    {
        if (isInitialized) return;
        
        // Validate RGB camera reference
        if (rgbCamera == null)
        {
            rgbCamera = GetComponentInChildren<Camera>();
            if (rgbCamera == null)
            {
                Debug.LogError($"[DroneDepthCamera] No RGB camera found on {gameObject.name}!");
                return;
            }
        }
        
        // Auto-detect RGB camera resolution if depth dimensions are 0
        if (depthWidth == 0 || depthHeight == 0)
        {
            // Get RGB camera's render texture or screen resolution
            if (rgbCamera.targetTexture != null)
            {
                depthWidth = rgbCamera.targetTexture.width;
                depthHeight = rgbCamera.targetTexture.height;
                Debug.Log($"[DroneDepthCamera] Auto-detected resolution from RGB camera target texture: {depthWidth}x{depthHeight}");
            }
            else
            {
                // Fallback to default camera resolution (assumes full screen or standard size)
                // You can adjust these defaults based on your setup
                depthWidth = 640;
                depthHeight = 480;
                Debug.Log($"[DroneDepthCamera] RGB camera has no target texture, using default resolution: {depthWidth}x{depthHeight}");
            }
        }
        
        // Create depth camera GameObject
        GameObject depthCameraObj = new GameObject($"{gameObject.name}_DepthCamera");
        depthCameraObj.transform.SetParent(rgbCamera.transform);
        depthCameraObj.transform.localPosition = Vector3.zero;
        depthCameraObj.transform.localRotation = Quaternion.identity;
        
        // Setup depth camera component
        depthCamera = depthCameraObj.AddComponent<Camera>();
        
        // Copy settings from RGB camera
        depthCamera.fieldOfView = rgbCamera.fieldOfView;
        depthCamera.nearClipPlane = minDepthDistance;
        depthCamera.farClipPlane = maxDepthDistance;
        depthCamera.aspect = rgbCamera.aspect;
        
        // Configure for depth rendering
        depthCamera.enabled = false; // Manual rendering
        depthCamera.clearFlags = CameraClearFlags.SolidColor;
        depthCamera.backgroundColor = Color.white; // Far = white (max depth)
        depthCamera.cullingMask = rgbCamera.cullingMask; // Render same layers as RGB
        depthCamera.depthTextureMode = DepthTextureMode.Depth;
        
        // Create render texture for depth - use ARGBFloat to store actual depth values
        depthRenderTexture = new RenderTexture(depthWidth, depthHeight, 24, RenderTextureFormat.RFloat);
        depthRenderTexture.filterMode = FilterMode.Point;
        depthRenderTexture.Create();
        
        depthCamera.targetTexture = depthRenderTexture;
        
        // Create texture for reading depth data
        depthTexture = new Texture2D(depthWidth, depthHeight, TextureFormat.RFloat, false);
        
        // Create and use a simple depth shader that outputs linear depth directly
        CreateAndApplyDepthShader();
        
        isInitialized = true;
        
        Debug.Log($"[DroneDepthCamera] Initialized depth camera for {gameObject.name}");
        Debug.Log($"   RGB Camera: {rgbCamera.name}");
        Debug.Log($"   Resolution: {depthWidth}x{depthHeight} (matching RGB camera)");
        Debug.Log($"   Depth range: {minDepthDistance}m to {maxDepthDistance}m");
        Debug.Log($"   Field of View: {depthCamera.fieldOfView}° (matching RGB camera)");
    }
    
    void CreateAndApplyDepthShader()
    {
        // Create a shader that outputs linear depth directly
        string shaderCode = @"
            Shader ""Custom/LinearDepth""
            {
                SubShader
                {
                    Tags { ""RenderType""=""Opaque"" }
                    Pass
                    {
                        CGPROGRAM
                        #pragma vertex vert
                        #pragma fragment frag
                        #include ""UnityCG.cginc""

                        struct appdata
                        {
                            float4 vertex : POSITION;
                        };

                        struct v2f
                        {
                            float4 pos : SV_POSITION;
                            float depth : TEXCOORD0;
                        };

                        v2f vert(appdata v)
                        {
                            v2f o;
                            o.pos = UnityObjectToClipPos(v.vertex);
                            // Calculate linear eye-space depth
                            float4 viewPos = mul(UNITY_MATRIX_MV, v.vertex);
                            o.depth = -viewPos.z; // Depth in view space (positive forward)
                            return o;
                        }

                        float frag(v2f i) : SV_Target
                        {
                            // Clamp depth to camera's near/far range (0.1m to 100m)
                            // This prevents invalid values from skybox/infinity
                            return clamp(i.depth, 0.1, 100.0);
                        }
                        ENDCG
                    }
                }
            }
        ";
        
        Shader shader = Shader.Find("Custom/LinearDepth");
        if (shader == null)
        {
            // Try to create shader from code (won't work at runtime, but helps for development)
            Debug.LogWarning("[DroneDepthCamera] Custom/LinearDepth shader not found - trying fallback");
            shader = Shader.Find("Hidden/Internal-DepthNormalsTexture");
        }
        
        if (shader != null)
        {
            depthMaterial = new Material(shader);
            depthCamera.SetReplacementShader(shader, "RenderType");
            Debug.Log($"[DroneDepthCamera] Using depth shader: {shader.name}");
            Debug.Log($"[DroneDepthCamera] Shader isSupported: {shader.isSupported}");
            Debug.Log($"[DroneDepthCamera] Camera cullingMask: {depthCamera.cullingMask}");
        }
        else
        {
            Debug.LogError("[DroneDepthCamera] No depth shader available!");
        }
    }
    
    Shader CreateDepthShader()
    {
        // Deprecated - kept for compatibility
        return Shader.Find("Hidden/Internal-DepthNormalsTexture");
    }
    
    /// <summary>
    /// Force set the depth camera resolution (e.g., to match NBVImageCapture settings)
    /// Call this before capturing if you need a specific resolution
    /// </summary>
    public void SetResolution(int width, int height)
    {
        if (width == depthWidth && height == depthHeight)
            return; // Already at correct resolution
            
        depthWidth = width;
        depthHeight = height;
        
        // Reinitialize with new resolution
        if (isInitialized)
        {
            // Clean up old resources
            if (depthRenderTexture != null)
            {
                depthRenderTexture.Release();
                DestroyImmediate(depthRenderTexture);
            }
            if (depthTexture != null)
            {
                DestroyImmediate(depthTexture);
            }
            
            // Recreate with new resolution
            depthRenderTexture = new RenderTexture(depthWidth, depthHeight, 24, RenderTextureFormat.RFloat);
            depthRenderTexture.filterMode = FilterMode.Point;
            depthRenderTexture.Create();
            depthCamera.targetTexture = depthRenderTexture;
            
            depthTexture = new Texture2D(depthWidth, depthHeight, TextureFormat.RFloat, false);
            
            Debug.Log($"[DroneDepthCamera] Resolution changed to {depthWidth}x{depthHeight}");
        }
    }
    
    /// <summary>
    /// Capture depth image and return as Texture2D
    /// </summary>
    public Texture2D CaptureDepth()
    {
        if (!isInitialized)
        {
            Debug.LogWarning("[DroneDepthCamera] Not initialized!");
            return null;
        }
        
        // Render depth camera
        depthCamera.Render();
        
        // Read depth data from render texture
        RenderTexture.active = depthRenderTexture;
        depthTexture.ReadPixels(new Rect(0, 0, depthWidth, depthHeight), 0, 0);
        depthTexture.Apply();
        RenderTexture.active = null;
        
        return depthTexture;
    }
    
    /// <summary>
    /// Get depth data as raw float array (in meters)
    /// Returns linear depth values from custom depth shader
    /// </summary>
    public float[] GetDepthDataMeters()
    {
        if (!isInitialized)
        {
            Debug.LogWarning("[DroneDepthCamera] Not initialized!");
            return null;
        }
        
        // Capture depth using the depth camera
        Texture2D depthTex = CaptureDepth();
        if (depthTex == null) return null;
        
        // Get pixels - our shader outputs linear depth directly in R channel
        Color[] pixels = depthTex.GetPixels();
        float[] depthMeters = new float[pixels.Length];
        
        for (int i = 0; i < pixels.Length; i++)
        {
            // R channel contains linear depth in meters directly
            depthMeters[i] = pixels[i].r;
        }
        
        return depthMeters;
    }
    
    /// <summary>
    /// Get depth data as byte array (for sharing with Python)
    /// Returns depth in meters as float32 array
    /// </summary>
    public byte[] GetDepthDataBytes()
    {
        // Get depth in meters
        float[] depthMeters = GetDepthDataMeters();
        
        if (depthMeters == null)
        {
            Debug.LogWarning($"[DroneDepthCamera] GetDepthDataMeters returned null for {gameObject.name}");
            return null;
        }
        
        // Debug: Check if we have valid depth values
        float minDepth = float.MaxValue;
        float maxDepth = float.MinValue;
        int validCount = 0;
        
        for (int i = 0; i < depthMeters.Length; i++)
        {
            if (depthMeters[i] > 0 && !float.IsNaN(depthMeters[i]) && !float.IsInfinity(depthMeters[i]))
            {
                validCount++;
                minDepth = Mathf.Min(minDepth, depthMeters[i]);
                maxDepth = Mathf.Max(maxDepth, depthMeters[i]);
            }
        }
        
        Debug.Log($"[DroneDepthCamera] {gameObject.name}: Generated {depthMeters.Length} depth values, {validCount} valid, range [{minDepth:F2}, {maxDepth:F2}]");
        
        // Convert float array to byte array
        byte[] bytes = new byte[depthMeters.Length * 4]; // 4 bytes per float
        Buffer.BlockCopy(depthMeters, 0, bytes, 0, bytes.Length);
        
        return bytes;
    }
    
    /// <summary>
    /// Get depth as visualization texture (grayscale image)
    /// </summary>
    public Texture2D GetDepthVisualization()
    {
        Texture2D depthTex = CaptureDepth();
        if (depthTex == null) return null;
        
        // Create visualization texture
        Texture2D visualization = new Texture2D(depthWidth, depthHeight, TextureFormat.RGB24, false);
        Color[] pixels = depthTex.GetPixels();
        
        // Convert to grayscale visualization
        for (int i = 0; i < pixels.Length; i++)
        {
            float depth = pixels[i].r;
            // Invert for better visualization (closer = brighter)
            float visualDepth = 1.0f - depth;
            pixels[i] = new Color(visualDepth, visualDepth, visualDepth, 1.0f);
        }
        
        visualization.SetPixels(pixels);
        visualization.Apply();
        
        return visualization;
    }
    
    void OnDestroy()
    {
        // Cleanup resources
        if (depthRenderTexture != null)
        {
            depthRenderTexture.Release();
            Destroy(depthRenderTexture);
        }
        
        if (depthTexture != null)
        {
            Destroy(depthTexture);
        }
        
        if (depthMaterial != null)
        {
            Destroy(depthMaterial);
        }
        
        if (depthCamera != null)
        {
            Destroy(depthCamera.gameObject);
        }
    }
    
    void OnGUI()
    {
        // Debug visualization
        if (showDebugVisualization && isInitialized)
        {
            Texture2D viz = GetDepthVisualization();
            if (viz != null)
            {
                // Show in top-right corner
                GUI.DrawTexture(new Rect(Screen.width - 320, 10, 300, 225), viz);
                GUI.Label(new Rect(Screen.width - 320, 240, 300, 20), 
                         $"{gameObject.name} Depth (Range: {minDepthDistance}-{maxDepthDistance}m)");
            }
        }
    }
}
