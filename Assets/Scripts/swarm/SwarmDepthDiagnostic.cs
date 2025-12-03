using UnityEngine;

/// <summary>
/// SwarmDepthDiagnostic.cs - Diagnose depth camera issues for swarm
/// 
/// Attach this to any GameObject to check depth camera setup
/// </summary>
public class SwarmDepthDiagnostic : MonoBehaviour
{
    [Header("Diagnostic Settings")]
    [SerializeField] private bool runDiagnostics = true;
    [SerializeField] private bool continuousCheck = false;
    [SerializeField] private float checkInterval = 2.0f;
    
    private float nextCheckTime;

    void Start()
    {
        if (runDiagnostics)
        {
            Debug.Log("🔍 SwarmDepthDiagnostic: Starting depth camera diagnostics...");
            RunDiagnostics();
            nextCheckTime = Time.time + checkInterval;
        }
    }

    void Update()
    {
        if (continuousCheck && Time.time >= nextCheckTime)
        {
            RunDiagnostics();
            nextCheckTime = Time.time + checkInterval;
        }
    }

    [ContextMenu("Run Diagnostics Now")]
    public void RunDiagnostics()
    {
        Debug.Log("==================== DEPTH CAMERA DIAGNOSTICS ====================");
        
        GameObject[] drones = GameObject.FindGameObjectsWithTag("DroneBase");
        Debug.Log($"🚁 Found {drones.Length} drones with 'DroneBase' tag");
        
        if (drones.Length == 0)
        {
            Debug.LogError("❌ No drones found! Make sure drones are tagged 'DroneBase'");
            return;
        }
        
        int dronesWithDepthCamera = 0;
        int dronesWithWorkingDepth = 0;
        
        foreach (GameObject drone in drones)
        {
            Debug.Log($"\n📍 Checking drone: {drone.name}");
            
            // Find FPV camera
            Transform fpvTransform = drone.transform.Find("FPV");
            if (fpvTransform == null)
            {
                Debug.LogError($"  ❌ No FPV child object found!");
                continue;
            }
            
            Camera fpvCamera = fpvTransform.GetComponent<Camera>();
            if (fpvCamera == null)
            {
                Debug.LogError($"  ❌ FPV has no Camera component!");
                continue;
            }
            
            Debug.Log($"  ✅ FPV camera found");
            
            // Check for DroneDepthCamera component
            DroneDepthCamera depthCamera = fpvCamera.GetComponent<DroneDepthCamera>();
            if (depthCamera == null)
            {
                Debug.LogError($"  ❌ DroneDepthCamera component NOT found on FPV camera!");
                Debug.LogError($"     → Add DroneDepthCamera component to {fpvTransform.name}");
                continue;
            }
            
            dronesWithDepthCamera++;
            Debug.Log($"  ✅ DroneDepthCamera component found");
            
            // Test depth capture
            try
            {
                byte[] depthData = depthCamera.GetDepthDataBytes();
                
                if (depthData == null)
                {
                    Debug.LogError($"  ❌ GetDepthDataBytes() returned null!");
                }
                else
                {
                    Debug.Log($"  ✅ Depth data captured: {depthData.Length} bytes");
                    
                    // Check if depth values are valid
                    int validValues = 0;
                    int totalValues = depthData.Length / 4; // float32 = 4 bytes
                    
                    for (int i = 0; i < depthData.Length; i += 4)
                    {
                        float depth = System.BitConverter.ToSingle(depthData, i);
                        if (depth > 0 && !float.IsNaN(depth) && !float.IsInfinity(depth))
                        {
                            validValues++;
                        }
                    }
                    
                    float validPercent = (validValues / (float)totalValues) * 100f;
                    Debug.Log($"  📊 Valid depth values: {validValues}/{totalValues} ({validPercent:F1}%)");
                    
                    if (validPercent < 1f)
                    {
                        Debug.LogWarning($"  ⚠️ Very few valid depth values! Depth might be blank.");
                        Debug.LogWarning($"     Possible causes:");
                        Debug.LogWarning($"     - Depth shader not working");
                        Debug.LogWarning($"     - Camera not rendering anything");
                        Debug.LogWarning($"     - Culling mask issue");
                    }
                    else
                    {
                        dronesWithWorkingDepth++;
                        Debug.Log($"  ✅ Depth capture appears to be working!");
                    }
                }
            }
            catch (System.Exception e)
            {
                Debug.LogError($"  ❌ Error testing depth capture: {e.Message}");
            }
        }
        
        Debug.Log($"\n📊 Summary:");
        Debug.Log($"  Total drones: {drones.Length}");
        Debug.Log($"  Drones with DroneDepthCamera: {dronesWithDepthCamera}");
        Debug.Log($"  Drones with working depth: {dronesWithWorkingDepth}");
        
        if (dronesWithDepthCamera == 0)
        {
            Debug.LogError("\n❌ CRITICAL: No drones have DroneDepthCamera component!");
            Debug.LogError("   FIX: Add DroneDepthCamera component to each drone's FPV camera");
            Debug.LogError("   1. Select drone in Hierarchy");
            Debug.LogError("   2. Find FPV child object");
            Debug.LogError("   3. In Inspector, click 'Add Component'");
            Debug.LogError("   4. Search for 'DroneDepthCamera' and add it");
        }
        else if (dronesWithWorkingDepth == 0)
        {
            Debug.LogError("\n❌ CRITICAL: Depth cameras exist but aren't capturing valid data!");
            Debug.LogError("   Possible fixes:");
            Debug.LogError("   1. Check if 'Custom/LinearDepth' shader exists");
            Debug.LogError("   2. Verify camera culling mask includes visible objects");
            Debug.LogError("   3. Check if drones are looking at scene geometry");
        }
        
        Debug.Log("=================================================================");
    }
}
