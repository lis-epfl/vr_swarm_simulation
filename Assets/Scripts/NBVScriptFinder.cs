using UnityEngine;

/// <summary>
/// Helper script to diagnose and fix NBV script assignment issues.
/// This script will automatically find and assign the NBV script for you.
/// </summary>
public class NBVScriptFinder : MonoBehaviour
{
    [Header("Auto-Assignment")]
    public ContinuousImageCapture imageCaptureScript;
    
    [Header("Diagnostic Info")]
    [SerializeField] private bool nbvScriptFound = false;
    [SerializeField] private string nbvScriptLocation = "Not found";
    [SerializeField] private int droneCount = 0;
    
    [ContextMenu("Find and Assign NBV Script")]
    public void FindAndAssignNBVScript()
    {
        Debug.Log("=== Searching for NBV Script ===");
        
        // Search for NBV script in the scene
        NBV[] nbvScripts = FindObjectsOfType<NBV>();
        
        if (nbvScripts.Length == 0)
        {
            Debug.LogError("❌ No NBV script found in the scene!");
            Debug.LogError("Please make sure you have:");
            Debug.LogError("1. Added the NBV script to a GameObject in your scene");
            Debug.LogError("2. The GameObject is active");
            
            nbvScriptFound = false;
            nbvScriptLocation = "Not found";
            return;
        }
        
        if (nbvScripts.Length > 1)
        {
            Debug.LogWarning($"⚠️ Found {nbvScripts.Length} NBV scripts. Using the first one.");
        }
        
        NBV selectedNBV = nbvScripts[0];
        nbvScriptFound = true;
        nbvScriptLocation = selectedNBV.gameObject.name;
        droneCount = selectedNBV.swarm != null ? selectedNBV.swarm.Count : 0;
        
        Debug.Log($"✅ Found NBV script on GameObject: {selectedNBV.gameObject.name}");
        Debug.Log($"✅ Drone count in swarm: {droneCount}");
        
        // Auto-assign to ContinuousImageCapture if reference exists
        if (imageCaptureScript == null)
        {
            imageCaptureScript = GetComponent<ContinuousImageCapture>();
        }
        
        if (imageCaptureScript != null)
        {
            imageCaptureScript.NBVScript = selectedNBV;
            Debug.Log("✅ Successfully assigned NBV script to ContinuousImageCapture!");
            
            // Mark the object as dirty for Unity to save the changes
            #if UNITY_EDITOR
            UnityEditor.EditorUtility.SetDirty(imageCaptureScript);
            #endif
        }
        else
        {
            Debug.LogError("❌ ContinuousImageCapture script not found on this GameObject!");
            Debug.LogError("Please make sure this script is on the same GameObject as ContinuousImageCapture.");
        }
    }
    
    [ContextMenu("List All NBV Scripts")]
    public void ListAllNBVScripts()
    {
        Debug.Log("=== All NBV Scripts in Scene ===");
        
        NBV[] nbvScripts = FindObjectsOfType<NBV>();
        
        if (nbvScripts.Length == 0)
        {
            Debug.Log("No NBV scripts found in the scene.");
            return;
        }
        
        for (int i = 0; i < nbvScripts.Length; i++)
        {
            NBV nbv = nbvScripts[i];
            GameObject go = nbv.gameObject;
            int swarmCount = nbv.swarm != null ? nbv.swarm.Count : 0;
            
            Debug.Log($"NBV Script #{i + 1}:");
            Debug.Log($"  GameObject: {go.name}");
            Debug.Log($"  Active: {go.activeInHierarchy}");
            Debug.Log($"  Swarm Count: {swarmCount}");
            Debug.Log($"  Script Enabled: {nbv.enabled}");
            Debug.Log($"  Path: {GetGameObjectPath(go)}");
            Debug.Log("---");
        }
    }
    
    [ContextMenu("Create NBV Script if Missing")]
    public void CreateNBVScriptIfMissing()
    {
        NBV[] nbvScripts = FindObjectsOfType<NBV>();
        
        if (nbvScripts.Length > 0)
        {
            Debug.Log("NBV script already exists in the scene.");
            return;
        }
        
        // Look for SwarmManager first
        SwarmManager swarmManager = FindObjectOfType<SwarmManager>();
        
        GameObject targetObject;
        if (swarmManager != null)
        {
            targetObject = swarmManager.gameObject;
            Debug.Log("Adding NBV script to existing SwarmManager.");
        }
        else
        {
            // Create a new GameObject for NBV
            targetObject = new GameObject("NBV Manager");
            Debug.Log("Created new GameObject 'NBV Manager' for NBV script.");
        }
        
        // Add NBV script
        NBV newNBV = targetObject.AddComponent<NBV>();
        
        // Try to auto-populate the swarm list
        AutoPopulateSwarm(newNBV);
        
        Debug.Log($"✅ NBV script added to {targetObject.name}");
        
        // Try to assign it
        FindAndAssignNBVScript();
    }
    
    void AutoPopulateSwarm(NBV nbvScript)
    {
        // Try to find drones in the scene
        GameObject[] allObjects = FindObjectsOfType<GameObject>();
        
        foreach (GameObject obj in allObjects)
        {
            // Look for objects that might be drones (have "Drone" in name or have FPV camera)
            if (obj.name.ToLower().Contains("drone") || obj.transform.Find("FPV") != null)
            {
                if (nbvScript.swarm == null)
                {
                    nbvScript.swarm = new System.Collections.Generic.List<GameObject>();
                }
                
                if (!nbvScript.swarm.Contains(obj))
                {
                    nbvScript.swarm.Add(obj);
                    Debug.Log($"Added {obj.name} to swarm list");
                }
            }
        }
        
        Debug.Log($"Auto-populated swarm with {nbvScript.swarm?.Count ?? 0} drones");
    }
    
    string GetGameObjectPath(GameObject obj)
    {
        string path = obj.name;
        Transform parent = obj.transform.parent;
        
        while (parent != null)
        {
            path = parent.name + "/" + path;
            parent = parent.parent;
        }
        
        return path;
    }
    
    void Update()
    {
        // Update diagnostic info in inspector
        NBV[] nbvScripts = FindObjectsOfType<NBV>();
        nbvScriptFound = nbvScripts.Length > 0;
        
        if (nbvScriptFound)
        {
            NBV nbv = nbvScripts[0];
            nbvScriptLocation = nbv.gameObject.name;
            droneCount = nbv.swarm != null ? nbv.swarm.Count : 0;
        }
        else
        {
            nbvScriptLocation = "Not found";
            droneCount = 0;
        }
    }
}