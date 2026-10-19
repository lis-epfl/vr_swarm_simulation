using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;

/// <summary>
/// Batchmode entry for <see cref="ProceduralCityProbe"/>: opens ProceduralCityWorld and enters play mode; the probe
/// writes its report to PROCEDURAL_PROBE_OUT and exits. Run on the bench's project copy, never the live project:
/// <c>Unity.exe -batchmode -projectPath &lt;copy&gt; -executeMethod ProceduralCityProbeLauncher.Run</c> with
/// PROCEDURAL_PROBE_OUT and EXPERIMENT_RECORDER_DIR set (and CITY_LAYOUT to pick a layout, PROCEDURAL_PROBE_SCENE to
/// probe another scene with a ProceduralCityBuilder, e.g. Assets/Scenes/VoronoiCityWorld.unity).
/// </summary>
public static class ProceduralCityProbeLauncher
{
    public const string SceneEnv = "PROCEDURAL_PROBE_SCENE";

    public static void Run()
    {
        if (string.IsNullOrEmpty(System.Environment.GetEnvironmentVariable(ProceduralCityProbe.OutEnv)))
        {
            Debug.LogError($"ProceduralCityProbeLauncher: set {ProceduralCityProbe.OutEnv} to the report path.");
            EditorApplication.Exit(2);
            return;
        }
        string scene = System.Environment.GetEnvironmentVariable(SceneEnv);
        if (string.IsNullOrEmpty(scene)) scene = ProceduralCityWorldBuilder.ScenePath;
        EditorSceneManager.OpenScene(scene, OpenSceneMode.Single);
        EditorApplication.EnterPlaymode();
    }
}
