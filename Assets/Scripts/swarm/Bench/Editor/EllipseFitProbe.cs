using System.Globalization;
using System.IO;
using System.Text;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;

// S10 Ellipse Avoidance: every obstacle collider's ellipse as Unity fits it (ObstacleFootprint.GetEllipse), at each
// cap, written to JSON for citysim.swarmellipse's fit check against the replica's (swarm_replica.PolyCity.ellipses).
//
//   Unity.exe -batchmode -quit -projectPath <bench copy> -executeMethod EllipseFitProbe.Run -logFile <log>
//
// with ELLIPSE_PROBE_OUT set to the JSON path (and optionally ELLIPSE_PROBE_SCENE, default BlockShapedPatchCityWorld).
public static class EllipseFitProbe
{
    private static readonly float[] Caps = { 1.0f, 1.25f, 1.5f, 2.0f };

    public static void Run()
    {
        string scene = System.Environment.GetEnvironmentVariable("ELLIPSE_PROBE_SCENE");
        if (string.IsNullOrEmpty(scene))
            scene = "Assets/Scenes/BlockShapedPatchCityWorld.unity";
        string outPath = System.Environment.GetEnvironmentVariable("ELLIPSE_PROBE_OUT");
        if (string.IsNullOrEmpty(outPath))
        {
            Debug.LogError("[EllipseFitProbe] ELLIPSE_PROBE_OUT is not set");
            EditorApplication.Exit(1);
            return;
        }
        EditorSceneManager.OpenScene(scene, OpenSceneMode.Single);
        int layer = LayerMask.NameToLayer("Obstacle");
        var inv = CultureInfo.InvariantCulture;
        var sb = new StringBuilder("[\n");
        int n = 0;
        foreach (Collider col in Object.FindObjectsByType<Collider>(FindObjectsInactive.Exclude, FindObjectsSortMode.None))
        {
            if (col.gameObject.layer != layer)
                continue;
            ObstacleFootprint.Footprint fp = ObstacleFootprint.Get(col);
            ObstacleFootprint.GetEllipse(fp, 1.5f);          // fits the axis and aspect read below
            if (n++ > 0)
                sb.Append(",\n");
            sb.Append($"{{\"name\": \"{col.name}\", \"circle\": [{fp.circleCentre.x.ToString("R", inv)}, " +
                      $"{fp.circleCentre.y.ToString("R", inv)}, {fp.circleRadius.ToString("R", inv)}], " +
                      $"\"aspect\": {fp.mveeAspect.ToString("R", inv)}, \"ellipses\": {{");
            for (int i = 0; i < Caps.Length; i++)
            {
                ObstacleFootprint.Ellipse el = ObstacleFootprint.GetEllipse(fp, Caps[i]);
                if (i > 0)
                    sb.Append(", ");
                sb.Append($"\"{Caps[i].ToString("R", inv)}\": [{el.centre.x.ToString("R", inv)}, " +
                          $"{el.centre.y.ToString("R", inv)}, {el.axis.x.ToString("R", inv)}, " +
                          $"{el.axis.y.ToString("R", inv)}, {el.a.ToString("R", inv)}, {el.b.ToString("R", inv)}, " +
                          $"{(el.circle ? 1 : 0)}]");
            }
            sb.Append("}}");
        }
        sb.Append("\n]\n");
        File.WriteAllText(outPath, sb.ToString());
        Debug.Log($"[EllipseFitProbe] {n} obstacle colliders in {scene} -> {outPath}");
        EditorApplication.Exit(0);
    }
}
