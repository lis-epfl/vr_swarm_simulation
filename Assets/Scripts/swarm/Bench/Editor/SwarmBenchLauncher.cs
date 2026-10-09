using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;

// Batchmode entry points for the headless swarm bench (run_bench.ps1 calls these; see README.md).
public static class SwarmBenchLauncher
{
    // Unity.exe -batchmode -projectPath <copy> -executeMethod SwarmBenchLauncher.Run -logFile <log>
    // with SWARM_BENCH_CONFIG (and usually SWARM_BENCH_OUT) set. No -quit: the editor stays up, enters
    // play mode in the config's scene, and SwarmBenchRunner exits when the last flight is written.
    public static void Run()
    {
        SwarmBenchRunner.Config cfg = SwarmBenchRunner.LoadConfig(out string error);
        if (cfg == null)
        {
            SwarmBenchRunner.Fatal(SwarmBenchRunner.ExitConfigError, error);
            return;
        }
        string scene = SwarmBenchRunner.ScenePath(cfg.scene);
        if (AssetDatabase.LoadAssetAtPath<SceneAsset>(scene) == null)
        {
            SwarmBenchRunner.Fatal(SwarmBenchRunner.ExitConfigError, $"scene {scene} does not exist in this project", cfg);
            return;
        }
        EditorSceneManager.OpenScene(scene, OpenSceneMode.Single);
        Debug.Log($"[SwarmBench] opened {scene}, entering play mode");
        EditorApplication.EnterPlaymode();
    }

    // Reaching this at all means the project compiled: a compile error leaves batchmode unable to find
    // the method, which run_bench.ps1 -CompileOnly reports from the log.
    public static void CompileCheck()
    {
        Debug.Log("[SwarmBench] compiled");
        EditorApplication.Exit(0);
    }
}
