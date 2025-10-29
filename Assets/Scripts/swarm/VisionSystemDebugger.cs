using UnityEngine;

/// <summary>
/// VisionSystemDebugger.cs - Debug helper to diagnose vision system setup
/// 
/// Attach this to any GameObject and check the console for diagnostic information
/// </summary>
public class VisionSystemDebugger : MonoBehaviour
{
    [Header("Debug Controls")]
    [SerializeField] private bool runDiagnostics = true;
    [SerializeField] private float diagnosticInterval = 3.0f;
    
    private float nextDiagnosticTime;

    void Start()
    {
        if (runDiagnostics)
        {
            Debug.Log("🔍 VisionSystemDebugger: Starting diagnostics...");
            RunFullDiagnostics();
            nextDiagnosticTime = Time.time + diagnosticInterval;
        }
    }

    void Update()
    {
        if (runDiagnostics && Time.time >= nextDiagnosticTime)
        {
            RunFullDiagnostics();
            nextDiagnosticTime = Time.time + diagnosticInterval;
        }
    }

    private void RunFullDiagnostics()
    {
        Debug.Log("==================== VISION SYSTEM DIAGNOSTICS ====================");
        
        CheckSwarmManager();
        CheckVisionComponents();
        CheckDroneSetup();
        CheckNBVScript();
        
        Debug.Log("=================================================================");
    }

    private void CheckSwarmManager()
    {
        SwarmManager swarmManager = FindObjectOfType<SwarmManager>();
        if (swarmManager != null)
        {
            Debug.Log($"✅ SwarmManager found: Algorithm = {swarmManager.swarmAlgorithm}");
            if (swarmManager.swarmAlgorithm == SwarmManager.SwarmAlgorithm.NBV)
            {
                Debug.Log("✅ NBV algorithm is selected");
            }
            else
            {
                Debug.LogWarning($"⚠️ SwarmManager algorithm is {swarmManager.swarmAlgorithm}, should be NBV for vision system");
            }
        }
        else
        {
            Debug.LogError("❌ SwarmManager not found in scene");
        }
    }

    private void CheckVisionComponents()
    {
        NBVImageCapture imageCapture = FindObjectOfType<NBVImageCapture>();
        NBVImageIntegration integration = FindObjectOfType<NBVImageIntegration>();
        
        if (imageCapture != null)
        {
            Debug.Log($"✅ NBVImageCapture found: {imageCapture.GetDroneCameraCount()} cameras detected");
            Debug.Log($"   - Initialized: {imageCapture.IsInitialized()}");
            Debug.Log($"   - GameObject: {imageCapture.gameObject.name}");
            Debug.Log($"   - Component enabled: {imageCapture.enabled}");
        }
        else
        {
            Debug.LogError("❌ NBVImageCapture component not found in scene");
            Debug.LogError("   → Make sure to add NBVImageCapture component to VisionManager or another GameObject");
        }
        
        if (integration != null)
        {
            Debug.Log($"✅ NBVImageIntegration found: Vision active = {integration.IsVisionActive()}");
            Debug.Log($"   - Current command: {integration.GetCurrentVisionCommand()}");
            Debug.Log($"   - GameObject: {integration.gameObject.name}");
            Debug.Log($"   - Component enabled: {integration.enabled}");
        }
        else
        {
            Debug.LogError("❌ NBVImageIntegration component not found in scene");
        }
    }

    private void CheckDroneSetup()
    {
        GameObject[] dronesWithTag = GameObject.FindGameObjectsWithTag("DroneBase");
        Debug.Log($"🚁 Found {dronesWithTag.Length} GameObjects with 'DroneBase' tag");
        
        int dronesWithFPV = 0;
        int dronesWithCamera = 0;
        
        foreach (GameObject drone in dronesWithTag)
        {
            Transform fpvTransform = drone.transform.Find("FPV");
            if (fpvTransform != null)
            {
                dronesWithFPV++;
                Camera fpvCamera = fpvTransform.GetComponent<Camera>();
                if (fpvCamera != null)
                {
                    dronesWithCamera++;
                    // Debug.Log($"   - {drone.name}: ✅ Has FPV camera");
                }
                else
                {
                    Debug.LogWarning($"   - {drone.name}: ⚠️ Has FPV transform but no Camera component");
                }
            }
            else
            {
                Debug.LogWarning($"   - {drone.name}: ⚠️ Missing FPV child object");
            }
        }
        
        Debug.Log($"📊 Drone Summary: {dronesWithCamera}/{dronesWithTag.Length} drones have working FPV cameras");
        
        if (dronesWithTag.Length == 0)
        {
            Debug.LogWarning("⚠️ No drones found with 'DroneBase' tag. Make sure drones are spawned and tagged correctly.");
        }
    }

    private void CheckNBVScript()
    {
        NBV[] nbvScripts = FindObjectsOfType<NBV>();
        Debug.Log($"🔍 Found {nbvScripts.Length} NBV script(s) in scene");
        
        if (nbvScripts.Length > 0)
        {
            NBV nbvScript = nbvScripts[0]; // Use the first one for diagnosis
            Debug.Log($"✅ NBV script found: Center point = {nbvScript.centerPoint}");
            Debug.Log($"   - Swarm count: {nbvScript.swarm?.Count ?? 0}");
            Debug.Log($"   - GameObject: {nbvScript.gameObject.name}");
            
            if (nbvScripts.Length > 1)
            {
                Debug.LogWarning($"⚠️ Multiple NBV scripts detected! This may cause conflicts.");
                for (int i = 0; i < nbvScripts.Length; i++)
                {
                    Debug.Log($"   NBV {i}: {nbvScripts[i].gameObject.name}");
                }
            }
            
            // Check if NBV has vision system enabled using reflection
            try
            {
                var enableVisionField = nbvScript.GetType().GetField("enableVisionSystem", 
                    System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);
                if (enableVisionField != null)
                {
                    bool enableVision = (bool)enableVisionField.GetValue(nbvScript);
                    Debug.Log($"   - Vision system enabled: {enableVision}");
                }
            }
            catch (System.Exception e)
            {
                Debug.LogWarning($"   - Could not check vision system status: {e.Message}");
            }
        }
        else
        {
            Debug.LogWarning("⚠️ NBV script not found in scene");
        }
    }

    [ContextMenu("Run Diagnostics Now")]
    public void RunDiagnosticsManually()
    {
        RunFullDiagnostics();
    }
}