using UnityEngine;
using System.IO;
using System.Collections;

[System.Serializable]
public class SwarmExperimentConfig
{
    public int numDrones = 2;
    public int dronesAlongX = 2;
    public int dronesAlongZ = 1;
    public float captureInterval = 2.0f;
}

public class SwarmConfig : MonoBehaviour
{
    private static string configFilePath = Path.Combine(Application.dataPath, "../swarm_config.json");
    
    [Header("Default Configuration")]
    public SwarmExperimentConfig defaultConfig = new SwarmExperimentConfig();
    
    [Header("Current Configuration (Runtime)")]
    public SwarmExperimentConfig currentConfig;
    
    [Header("Swarm Components")]
    public swarmSpawn swarmSpawner;
    public SwarmImageCapture captureImages;
    
    [Header("Script Execution Order")]
    [Tooltip("IMPORTANT: SwarmConfig must execute BEFORE swarmSpawn in Project Settings > Script Execution Order")]
    public bool executionOrderConfigured = false;
    
    void Awake()
    {
        if (!executionOrderConfigured)
        {
            Debug.LogWarning("[SwarmConfig] Script Execution Order may not be configured! Set SwarmConfig to run before Default Time.");
        }
        
        LoadConfiguration();
        ApplyConfiguration();
        
        // Force verify configuration was applied
        if (swarmSpawner != null)
        {
            Debug.Log($"[SwarmConfig] Verification - swarmSpawn now has: {swarmSpawner.dronesAlongX}x{swarmSpawner.dronesAlongZ}");
        }
    }
    
    void LoadConfiguration()
    {
        if (File.Exists(configFilePath))
        {
            try
            {
                string json = File.ReadAllText(configFilePath);
                currentConfig = JsonUtility.FromJson<SwarmExperimentConfig>(json);
                
                // ALWAYS force capture interval to 1Hz for multi-interval processing
                currentConfig.captureInterval = 1.0f;
                
                Debug.Log($"[SwarmConfig] Loaded config from {configFilePath}");
                Debug.Log($"  Drones: {currentConfig.numDrones}");
                Debug.Log($"  Capture interval: {currentConfig.captureInterval}s (forced to 1Hz for multi-interval processing)");
                Debug.Log($"  Auto-stop when all drones within 3.5 units of migration point");
            }
            catch (System.Exception e)
            {
                Debug.LogWarning($"[SwarmConfig] Failed to load config: {e.Message}");
                Debug.LogWarning($"[SwarmConfig] Using default configuration");
                currentConfig = defaultConfig;
                currentConfig.captureInterval = 1.0f;  // Force 1Hz even for defaults
            }
        }
        else
        {
            Debug.Log($"[SwarmConfig] No config file found, using defaults");
            currentConfig = defaultConfig;
            currentConfig.captureInterval = 1.0f;  // Force 1Hz even for defaults
            
            // Create default config file for reference
            SaveConfiguration(currentConfig);
        }
    }
    
    void ApplyConfiguration()
    {
        // Find the swarmSpawn script if not assigned
        if (swarmSpawner == null)
        {
            swarmSpawner = FindObjectOfType<swarmSpawn>();
        }
        
        if (swarmSpawner != null)
        {
            swarmSpawner.dronesAlongX = currentConfig.dronesAlongX;
            swarmSpawner.dronesAlongZ = currentConfig.dronesAlongZ;
            Debug.Log($"[SwarmConfig] Applied drone count to swarmSpawn: {currentConfig.dronesAlongX}x{currentConfig.dronesAlongZ}");
        }
        else
        {
            Debug.LogWarning("[SwarmConfig] swarmSpawn not found in scene!");
        }
        
        // Find the capture images script if not assigned
        if (captureImages == null)
        {
            captureImages = FindObjectOfType<SwarmImageCapture>();
        }
        
        if (captureImages != null)
        {
            captureImages.captureInterval = currentConfig.captureInterval;
            Debug.Log($"[SwarmConfig] Applied capture interval: {currentConfig.captureInterval}s");
        }
        else
        {
            Debug.LogWarning("[SwarmConfig] SwarmImageCapture not found in scene!");
        }
    }
    
    void Start()
    {
        // Start monitoring swarm convergence
        StartCoroutine(MonitorSwarmConvergence());
    }
    
    IEnumerator MonitorSwarmConvergence()
    {
        // Migration point from olfatiSaber.cs
        Vector3 migrationPoint = new Vector3(-30, 15, -30);
        float convergenceThreshold = 3.5f;
        float maxTimeout = 300f; // 5 minutes max
        float checkInterval = 0.5f;
        
        float startTime = Time.time;
        float elapsedTime = 0f;
        
        Debug.Log($"[SwarmConfig] Monitoring swarm convergence to {migrationPoint}");
        Debug.Log($"  Threshold: {convergenceThreshold} units");
        Debug.Log($"  Max timeout: {maxTimeout}s");
        
        // Wait for drones to spawn
        yield return new WaitForSeconds(2.0f);
        
        // Verify drone count
        if (swarmSpawner != null && swarmSpawner.swarm != null)
        {
            Debug.Log($"[SwarmConfig] Actual spawned drones: {swarmSpawner.swarm.Count}");
            Debug.Log($"  Expected: {currentConfig.dronesAlongX}x{currentConfig.dronesAlongZ} = {currentConfig.numDrones}");
        }
        
        while (elapsedTime < maxTimeout)
        {
            elapsedTime = Time.time - startTime;
            
            // Get all drones from swarmSpawner
            if (swarmSpawner != null && swarmSpawner.swarm != null && swarmSpawner.swarm.Count > 0)
            {
                bool allConverged = true;
                float maxDistance = 0f;
                
                // Check each drone's distance to migration point
                int droneIndex = 0;
                foreach (GameObject drone in swarmSpawner.swarm)
                {
                    // Find DroneParent child
                    Transform droneParent = drone.transform.Find("DroneParent");
                    if (droneParent != null)
                    {
                        float distance = Vector3.Distance(droneParent.position, migrationPoint);
                        maxDistance = Mathf.Max(maxDistance, distance);
                        
                        if (distance > convergenceThreshold)
                        {
                            allConverged = false;
                        }
                    }
                    droneIndex++;
                }
                
                // Log progress every 5 seconds with individual drone distances
                if (Mathf.FloorToInt(elapsedTime) % 5 == 0 && elapsedTime > 0.1f)
                {
                    Debug.Log($"[SwarmConfig] t={elapsedTime:F1}s, max distance: {maxDistance:F2} units");
                    
                    // Print each drone's status
                    int debugDroneIndex = 0;
                    foreach (GameObject debugDrone in swarmSpawner.swarm)
                    {
                        Transform debugDroneParent = debugDrone.transform.Find("DroneParent");
                        if (debugDroneParent != null)
                        {
                            float debugDist = Vector3.Distance(debugDroneParent.position, migrationPoint);
                            string status = debugDist <= convergenceThreshold ? "✓" : "...";
                            Debug.Log($"  {status} Drone {debugDroneIndex}: pos=({debugDroneParent.position.x:F1}, {debugDroneParent.position.y:F1}, {debugDroneParent.position.z:F1}), dist={debugDist:F2}");
                        }
                        debugDroneIndex++;
                    }
                }
                
                // Check if all drones converged
                if (allConverged)
                {
                    Debug.Log($"[SwarmConfig] ✓ All drones converged in {elapsedTime:F2}s!");
                    Debug.Log($"  Final max distance: {maxDistance:F2} units");
                    
                    // Write done file with completion time
                    string doneFilePath = Path.Combine(Application.dataPath, "../swarm_done.txt");
                    try
                    {
                        File.WriteAllText(doneFilePath, $"done,{elapsedTime:F2}");
                        Debug.Log($"[SwarmConfig] Wrote completion time to {doneFilePath}");
                    }
                    catch (System.Exception e)
                    {
                        Debug.LogWarning($"[SwarmConfig] Failed to write done file: {e.Message}");
                    }
                    
                    // Wait for Python to detect the file
                    yield return new WaitForSeconds(2.0f);
                    
                    #if UNITY_EDITOR
                    UnityEditor.EditorApplication.isPlaying = false;
                    #else
                    Application.Quit();
                    #endif
                    
                    yield break;
                }
            }
            
            yield return new WaitForSeconds(checkInterval);
        }
        
        // Timeout reached
        Debug.LogWarning($"[SwarmConfig] Timeout reached ({maxTimeout}s) - stopping experiment");
        
        string timeoutDoneFilePath = Path.Combine(Application.dataPath, "../swarm_done.txt");
        try
        {
            File.WriteAllText(timeoutDoneFilePath, $"timeout,{maxTimeout:F2}");
        }
        catch (System.Exception e)
        {
            Debug.LogWarning($"[SwarmConfig] Failed to write timeout file: {e.Message}");
        }
        
        yield return new WaitForSeconds(2.0f);
        
        #if UNITY_EDITOR
        UnityEditor.EditorApplication.isPlaying = false;
        #else
        Application.Quit();
        #endif
    }
    
    public void SaveConfiguration(SwarmExperimentConfig config)
    {
        try
        {
            string json = JsonUtility.ToJson(config, true);
            File.WriteAllText(configFilePath, json);
            Debug.Log($"[SwarmConfig] Saved config to {configFilePath}");
        }
        catch (System.Exception e)
        {
            Debug.LogError($"[SwarmConfig] Failed to save config: {e.Message}");
        }
    }
}
