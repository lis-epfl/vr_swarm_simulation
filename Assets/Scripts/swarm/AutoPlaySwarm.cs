using UnityEngine;
using UnityEditor;
using System.IO;

#if UNITY_EDITOR
[InitializeOnLoad]
public class AutoPlaySwarm
{
    private static string triggerFilePath = Path.Combine(Application.dataPath, "../swarm_play_trigger.txt");
    private static bool wasPlaying = false;
    
    static AutoPlaySwarm()
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
                Debug.Log("[AutoPlaySwarm] Trigger detected: Starting Play mode...");
                EditorApplication.isPlaying = true;
                wasPlaying = true;
                
                // Delete trigger file
                File.Delete(triggerFilePath);
            }
            else if (command == "stop" && EditorApplication.isPlaying)
            {
                Debug.Log("[AutoPlaySwarm] Trigger detected: Stopping Play mode...");
                EditorApplication.isPlaying = false;
                wasPlaying = false;
                
                // Delete trigger file
                File.Delete(triggerFilePath);
            }
        }
    }
    
    [MenuItem("Swarm/Play Experiment")]
    public static void PlayExperiment()
    {
        Debug.Log("[AutoPlaySwarm] Starting Play mode via menu...");
        EditorApplication.isPlaying = true;
    }
    
    [MenuItem("Swarm/Stop Experiment")]
    public static void StopExperiment()
    {
        Debug.Log("[AutoPlaySwarm] Stopping Play mode via menu...");
        EditorApplication.isPlaying = false;
    }
}
#endif
