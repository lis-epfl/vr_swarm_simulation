using System;
using System.IO;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;

// Batchmode entry point for the headless video shots (VideoShotRunner; Video/README.md):
//   Unity.exe -batchmode -projectPath <copy> -executeMethod VideoShotLauncher.Run -logFile <log>
// with VIDEO_SHOTS_CONFIG set to a shot list. No -quit: the editor enters play mode in the config's scene and
// the runner exits when the last shot is written. Do not pass -nographics: the cameras must render.
public static class VideoShotLauncher
{
    public static void Run()
    {
        string path = Environment.GetEnvironmentVariable(VideoShotRunner.ConfigEnv);
        if (string.IsNullOrEmpty(path) || !File.Exists(path))
        {
            Debug.LogError($"[VideoShots] set {VideoShotRunner.ConfigEnv} to a shot list");
            EditorApplication.Exit(2);
            return;
        }
        var cfg = JsonUtility.FromJson<VideoShotRunner.Config>(File.ReadAllText(path));
        if (AssetDatabase.LoadAssetAtPath<SceneAsset>(cfg.scene) == null)
        {
            Debug.LogError($"[VideoShots] scene {cfg.scene} does not exist");
            EditorApplication.Exit(2);
            return;
        }
        EditorSceneManager.OpenScene(cfg.scene, OpenSceneMode.Single);
        Debug.Log($"[VideoShots] opened {cfg.scene}, entering play mode");
        EditorApplication.EnterPlaymode();
    }
}
