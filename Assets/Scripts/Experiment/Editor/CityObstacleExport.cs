using System.Globalization;
using System.IO;
using System.Text;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;

/// <summary>
/// Writes the open scene's building colliders (layer <c>Obstacle</c>) as world-space oriented boxes, plus
/// each city tile's kerb, to <c>city_obstacles_&lt;scene&gt;.json</c> beside <c>hat_visibility.py</c>, which
/// uses them for its line-of-sight test. Re-run after changing the city (tuners, tile moves, buildings).
/// Batch: <c>-executeMethod CityObstacleExport.RunScaledCityWorld</c>.
/// </summary>
public static class CityObstacleExport
{
    [MenuItem("Tools/Swarm/Export city obstacles")]
    public static void ExportOpenScene() => Export(EditorSceneManager.GetActiveScene().name);

    public static void RunScaledCityWorld()
    {
        EditorSceneManager.OpenScene("Assets/Scenes/ScaledCityWorld.unity");
        Export("ScaledCityWorld");
        EditorApplication.Exit(0);
    }

    static string F(float v) => v.ToString("R", CultureInfo.InvariantCulture);
    static string V(Vector3 v) => $"[{F(v.x)},{F(v.y)},{F(v.z)}]";

    static void Export(string sceneName)
    {
        int layer = LayerMask.NameToLayer("Obstacle");
        var sb = new StringBuilder("{\"boxes\":[\n");
        bool first = true;
        foreach (var c in Object.FindObjectsOfType<Collider>())
        {
            if (c.gameObject.layer != layer || !c.enabled) continue;
            string tile = "";
            for (Transform t = c.transform; t != null; t = t.parent)
                if (t.name.StartsWith("MC_Patch")) { tile = t.name; break; }
            Vector3 center, ax, ay, az;
            if (c is BoxCollider b)
            {
                Transform t = b.transform;
                center = t.TransformPoint(b.center);
                ax = t.TransformVector(new Vector3(b.size.x * 0.5f, 0, 0));
                ay = t.TransformVector(new Vector3(0, b.size.y * 0.5f, 0));
                az = t.TransformVector(new Vector3(0, 0, b.size.z * 0.5f));
            }
            else
            {
                Bounds bb = c.bounds; center = bb.center;
                ax = new Vector3(bb.extents.x, 0, 0); ay = new Vector3(0, bb.extents.y, 0); az = new Vector3(0, 0, bb.extents.z);
            }
            sb.Append(first ? "" : ",\n"); first = false;
            sb.Append($"{{\"name\":\"{c.name}\",\"type\":\"{c.GetType().Name}\",\"tile\":\"{tile}\",\"c\":{V(center)},\"ax\":[{V(ax)},{V(ay)},{V(az)}]}}");
        }
        sb.Append("\n],\"tiles\":[\n");
        first = true;
        foreach (var t in Object.FindObjectsOfType<Transform>())
        {
            if (!t.name.StartsWith("MC_Patch") || t.parent == null || t.parent.name.StartsWith("MC_Patch")) continue;
            Transform kerb = CityTiles.FindKerb(t);
            sb.Append(first ? "" : ",\n"); first = false;
            sb.Append($"{{\"name\":\"{t.name}\",\"pos\":{V(t.position)},\"kerb\":{(kerb ? V(kerb.position) : "null")}}}");
        }
        sb.Append("\n]}\n");
        string path = Path.Combine(Application.dataPath, "Scripts", "Experiment", $"city_obstacles_{sceneName}.json");
        File.WriteAllText(path, sb.ToString());
        Debug.Log($"CityObstacleExport: wrote {path}");
    }
}
