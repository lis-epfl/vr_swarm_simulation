using UnityEngine;
using UnityEditor;
using System.IO;

#if UNITY_EDITOR
[InitializeOnLoad]
public class AutoPlayNBV
{
    private static string triggerFilePath = Path.Combine(Application.dataPath, "../nbv_play_trigger.txt");
    private static bool wasPlaying = false;
    
    static AutoPlayNBV()
    {
        // Register update callback
        EditorApplication.update += Update;
    }
    
    private static void Update()
    {
        // Check if trigger file exists
        if (File.Exists(triggerFilePath))
        {
            string command = File.ReadAllText(triggerFilePath).Trim().ToLower();
            
            if (command == "play" && !EditorApplication.isPlaying)
            {
                Debug.Log("[AutoPlayNBV] Trigger detected: Starting Play mode...");
                EditorApplication.isPlaying = true;
                wasPlaying = true;
                
                // Delete trigger file
                File.Delete(triggerFilePath);
            }
            else if (command == "stop" && EditorApplication.isPlaying)
            {
                Debug.Log("[AutoPlayNBV] Trigger detected: Stopping Play mode...");
                EditorApplication.isPlaying = false;
                wasPlaying = false;
                
                // Delete trigger file
                File.Delete(triggerFilePath);
            }
        }
        
        // Auto-stop when NBV completes (optional)
        // Uncomment if you want Unity to stop automatically when Python script finishes
        /*
        if (wasPlaying && !EditorApplication.isPlaying)
        {
            wasPlaying = false;
        }
        */
    }
    
    [MenuItem("NBV/Play Experiment")]
    public static void PlayExperiment()
    {
        Debug.Log("[AutoPlayNBV] Starting Play mode via menu...");
        EditorApplication.isPlaying = true;
    }
    
    [MenuItem("NBV/Stop Experiment")]
    public static void StopExperiment()
    {
        Debug.Log("[AutoPlayNBV] Stopping Play mode via menu...");
        EditorApplication.isPlaying = false;
    }
}
#endif
