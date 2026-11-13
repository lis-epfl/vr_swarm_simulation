using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;

/// <summary>
/// Batchmode entry for <see cref="PlanarCaptureProbe"/>: opens a scene and enters play mode; the probe records its
/// clips to PLANAR_CAPTURE_OUT and exits. Run on a bench copy of the project, never the live one:
/// <c>Unity.exe -batchmode -projectPath &lt;copy&gt; -executeMethod PlanarCaptureLauncher.Run</c> with
/// PLANAR_CAPTURE_OUT set, and PLANAR_CAPTURE_SCENE to choose the scene (ScenicPatchCityWorld by default).
/// No -nographics: the capture renders.
/// </summary>
public static class PlanarCaptureLauncher
{
    public const string SceneEnv = "PLANAR_CAPTURE_SCENE";

    public static void Run()
    {
        if (string.IsNullOrEmpty(System.Environment.GetEnvironmentVariable(PlanarCaptureProbe.OutEnv)))
        {
            Debug.LogError($"PlanarCaptureLauncher: set {PlanarCaptureProbe.OutEnv} to the output folder.");
            EditorApplication.Exit(2);
            return;
        }
        string scene = System.Environment.GetEnvironmentVariable(SceneEnv);
        if (string.IsNullOrEmpty(scene)) scene = "Assets/Scenes/ScenicPatchCityWorld.unity";
        if (AssetDatabase.LoadAssetAtPath<SceneAsset>(scene) == null)
        {
            Debug.LogError($"PlanarCaptureLauncher: no scene at {scene}.");
            EditorApplication.Exit(2);
            return;
        }
        EditorSceneManager.OpenScene(scene, OpenSceneMode.Single);
        EditorApplication.EnterPlaymode();
    }
}
