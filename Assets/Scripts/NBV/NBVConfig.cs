using UnityEngine;
using System.IO;
using System.Collections;

[System.Serializable]
public class NBVExperimentConfig
{
    public int numDrones = 2;
    public int dronesAlongX = 2;
    public int dronesAlongZ = 1;
    public float droneSpeed = 12f;
    public int maxIterations = 5;
}

public class NBVConfig : MonoBehaviour
{
    private static string configFilePath = Path.Combine(Application.dataPath, "../nbv_config.json");
    
    [Header("Default Configuration")]
    public NBVExperimentConfig defaultConfig = new NBVExperimentConfig();
    
    [Header("Current Configuration (Runtime)")]
    public NBVExperimentConfig currentConfig;
    
    void Awake()
    {
        LoadConfiguration();
        ApplyConfiguration();
    }
    
    void LoadConfiguration()
    {
        if (File.Exists(configFilePath))
        {
            try
            {
                string json = File.ReadAllText(configFilePath);
                currentConfig = JsonUtility.FromJson<NBVExperimentConfig>(json);
                Debug.Log($"[NBVConfig] Loaded config from {configFilePath}");
                Debug.Log($"  Drones: {currentConfig.numDrones} ({currentConfig.dronesAlongX}x{currentConfig.dronesAlongZ})");
                Debug.Log($"  Speed: {currentConfig.droneSpeed}");
                Debug.Log($"  Max iterations: {currentConfig.maxIterations}");
            }
            catch (System.Exception e)
            {
                Debug.LogWarning($"[NBVConfig] Failed to load config: {e.Message}");
                Debug.LogWarning($"[NBVConfig] Using default configuration");
                currentConfig = defaultConfig;
            }
        }
        else
        {
            Debug.Log($"[NBVConfig] No config file found, using defaults");
            currentConfig = defaultConfig;
            
            // Create default config file for reference
            SaveConfiguration(currentConfig);
        }
    }
    
    void ApplyConfiguration()
    {
        // Find the swarmSpawn script (the actual spawner)
        var swarmSpawner = FindObjectOfType<swarmSpawn>();
        if (swarmSpawner != null)
        {
            swarmSpawner.dronesAlongX = currentConfig.dronesAlongX;
            swarmSpawner.dronesAlongZ = currentConfig.dronesAlongZ;
            Debug.Log($"[NBVConfig] Applied drone count to swarmSpawn: {currentConfig.dronesAlongX}x{currentConfig.dronesAlongZ}");
        }
        else
        {
            Debug.LogWarning("[NBVConfig] swarmSpawn not found in scene!");
        }
        
        // Note: Drone speed will be applied after drones spawn
        // The olfatiSaber components don't exist yet at Awake time
    }
    
    void Start()
    {
        // Apply speed after drones are spawned
        StartCoroutine(ApplySpeedAfterSpawn());
    }
    
    IEnumerator ApplySpeedAfterSpawn()
    {
        // Wait for drones to spawn
        yield return new WaitForSeconds(0.5f);
        
        var olfatiControllers = FindObjectsOfType<OlfatiSaber>();
        foreach (var controller in olfatiControllers)
        {
            controller.c_migration = currentConfig.droneSpeed;
        }
        
        if (olfatiControllers.Length > 0)
        {
            Debug.Log($"[NBVConfig] Applied drone speed to {olfatiControllers.Length} drones: {currentConfig.droneSpeed}");
        }
    }
    
    public static void SaveConfiguration(NBVExperimentConfig config)
    {
        try
        {
            string json = JsonUtility.ToJson(config, true);
            File.WriteAllText(configFilePath, json);
            Debug.Log($"[NBVConfig] Saved config to {configFilePath}");
        }
        catch (System.Exception e)
        {
            Debug.LogError($"[NBVConfig] Failed to save config: {e.Message}");
        }
    }
    
    public static NBVExperimentConfig GetCurrentConfig()
    {
        if (File.Exists(configFilePath))
        {
            string json = File.ReadAllText(configFilePath);
            return JsonUtility.FromJson<NBVExperimentConfig>(json);
        }
        return new NBVExperimentConfig();
    }
}
