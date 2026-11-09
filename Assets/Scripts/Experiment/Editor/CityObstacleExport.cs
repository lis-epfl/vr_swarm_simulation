using System.Globalization;
using System.IO;
using System.Text;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;

/// <summary>
/// Writes the open scene's building colliders (layer <c>Obstacle</c>) as world-space oriented boxes, plus
/// each city tile's kerb, to <c>city_obstacles_&lt;scene&gt;.json</c> beside <c>analyse.py</c>, which uses
/// them for the hat-visibility line-of-sight test and the turning maps. Re-run after changing the city
/// (tuners, tile moves, buildings).
/// Batch: <c>-executeMethod CityObstacleExport.RunScaledCityWorld</c> (or <c>RunDiamondCityWorld</c>).
/// </summary>
public static class CityObstacleExport
{
    [MenuItem("Tools/Swarm/Export city obstacles")]
    public static void ExportOpenScene() => Export(EditorSceneManager.GetActiveScene().name);

    public static void RunScaledCityWorld() => RunScene("ScaledCityWorld");

    public static void RunDiamondCityWorld() => RunScene("DiamondCityWorld");

    /// <summary>
    /// The round-trip check of a generated layout: opens ProceduralCityWorld, builds its layout in edit mode
    /// (CITY_LAYOUT overrides the scene's), and exports it to city_obstacles_ProceduralCityWorld.json, which
    /// citysim compares box by box with the layout file it came from (`vissim.py roundtrip`).
    /// </summary>
    public static void RunProceduralCityWorld()
    {
        EditorSceneManager.OpenScene("Assets/Scenes/ProceduralCityWorld.unity");
        ProceduralCityBuilder builder = Object.FindFirstObjectByType<ProceduralCityBuilder>(FindObjectsInactive.Include);
        string error = null;
        if (builder == null || !builder.Build(out error))
        {
            Debug.LogError("CityObstacleExport: could not build the procedural city" +
                           (builder == null ? " (no ProceduralCityBuilder in the scene)." : $": {error}"));
            EditorApplication.Exit(1);
            return;
        }
        Export("ProceduralCityWorld");
        Debug.Log($"CityObstacleExport: built and exported layout {builder.Layout.layout.id} from {builder.ResolvedPath}");
        EditorApplication.Exit(0);
    }

    static void RunScene(string sceneName)
    {
        EditorSceneManager.OpenScene($"Assets/Scenes/{sceneName}.unity");
        Export(sceneName);
        EditorApplication.Exit(0);
    }

    /// <summary>
    /// The tile (or diamond plaza) a collider belongs to, named as the "tiles" list below names it: the
    /// <c>MC_Patch</c> node whose parent is not one. Not the first <c>MC_Patch</c> ancestor: every patch prefab
    /// nests a node named after its <i>original</i> tile (<c>MC_Patch_10_Scaled</c> holds <c>MC_Patch_10</c>), so a
    /// scene's second copy of a design, <c>MC_Patch_10 (1)</c>, had its buildings tagged <c>MC_Patch_10</c> and a
    /// goal replacing either tile removed the other tile's buildings in analyse.scene_boxes.
    /// </summary>
    static string TileOf(Transform collider)
    {
        for (Transform t = collider; t != null; t = t.parent)
        {
            if (t.GetComponent<DiamondPlaza>() != null) return t.name;
            if (t.name.StartsWith("MC_Patch") && (t.parent == null || !t.parent.name.StartsWith("MC_Patch"))) return t.name;
        }
        return "";
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
            string tile = TileOf(c.transform);
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
        // The blocks in the diamonds' parks, which GoalPatchReplacer may also replace; see DiamondPlaza.
        foreach (var plaza in Object.FindObjectsOfType<DiamondPlaza>())
        {
            sb.Append(first ? "" : ",\n"); first = false;
            sb.Append($"{{\"name\":\"{plaza.name}\",\"pos\":{V(plaza.transform.position)},\"kerb\":{(plaza.Centre ? V(plaza.Centre.position) : "null")}}}");
        }
        sb.Append("\n]}\n");
        string path = Path.Combine(Application.dataPath, "Scripts", "Experiment", $"city_obstacles_{sceneName}.json");
        File.WriteAllText(path, sb.ToString());
        Debug.Log($"CityObstacleExport: wrote {path}");
    }
}
