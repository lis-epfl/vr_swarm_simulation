using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text;
using UnityEditor;
using UnityEngine;

/// <summary>
/// <c>Tools/Swarm/Export building catalogue</c>: scans the 48 ScaledCity patch prefabs and writes one entry per
/// building design to <see cref="AssetPath"/> (a <see cref="BuildingCatalogue"/>, which the procedural city
/// builder copies buildings from) and to <see cref="JsonPath"/> (the same table for the Python layout generator,
/// <c>Assets/Scripts/Experiment/citysim</c>, which chooses the study design from it).
///
/// <para>A design is a building name minus its instance number (<see cref="ObstacleLayerAuditor"/>'s grouping
/// rule), and its source is the first instance, in patch order, that a copy can be made of: a
/// <c>BoxCollider</c> on <c>Obstacle</c> on an object that is active together with all its parents — the
/// checks <c>DiamondParkBuilder.LoadBuilding</c> applies, because the ScaledCity fork hides a few buildings
/// and a copy of one of those would be neither seen nor avoided. Instances that fail are not counted.</para>
///
/// <para>Batch: <c>-executeMethod BuildingCatalogueExport.Run</c>.</para>
/// </summary>
public static class BuildingCatalogueExport
{
    public const string PatchFolder = "Assets/Prefabs/ScaledCity/Patches";
    public const string AssetPath = "Assets/Prefabs/ScaledCity/BuildingCatalogue.asset";
    public const string JsonPath = "Assets/Scripts/Experiment/building_catalogue.json";

    [MenuItem("Tools/Swarm/Export building catalogue")]
    public static void ExportFromMenu()
    {
        string summary = Export(out string error);
        if (summary == null)
        {
            Debug.LogError("BuildingCatalogueExport: " + error);
        }
    }

    /// <summary>Batchmode entry point: exports and exits (1 on failure).</summary>
    public static void Run()
    {
        string summary = Export(out string error);
        if (summary == null)
        {
            Debug.LogError("BuildingCatalogueExport: " + error);
        }
        EditorApplication.Exit(summary == null ? 1 : 0);
    }

    /// <summary>Writes both files. Returns a one-line summary, or null with <paramref name="error"/> set.</summary>
    public static string Export(out string error)
    {
        string[] guids = AssetDatabase.FindAssets("t:Prefab", new[] { PatchFolder });
        List<string> paths = guids.Select(AssetDatabase.GUIDToAssetPath)
                                  .Where(p => p.EndsWith("_Scaled.prefab"))
                                  .OrderBy(p => p, System.StringComparer.Ordinal)
                                  .ToList();
        if (paths.Count == 0)
        {
            error = $"no *_Scaled.prefab under {PatchFolder}.";
            return null;
        }

        int obstacle = LayerMask.NameToLayer("Obstacle");
        var entries = new Dictionary<string, BuildingCatalogue.Entry>();
        var order = new List<string>();
        int skipped = 0;
        foreach (string path in paths)
        {
            GameObject patch = AssetDatabase.LoadAssetAtPath<GameObject>(path);
            if (patch == null)
            {
                continue;
            }
            foreach (Transform t in patch.GetComponentsInChildren<Transform>(true))
            {
                if (!ObstacleLayerAuditor.IsBuilding(t.name))
                {
                    continue;
                }
                BoxCollider box = t.GetComponent<BoxCollider>();
                if (box == null || t.gameObject.layer != obstacle || !ActiveInPatch(t))
                {
                    skipped++;
                    continue;
                }
                string design = ObstacleLayerAuditor.Family(t.name);
                if (!entries.TryGetValue(design, out BuildingCatalogue.Entry entry))
                {
                    entry = new BuildingCatalogue.Entry
                    {
                        design = design,
                        family = FamilyPrefix(t.name),
                        source = t.gameObject,
                        patch = Path.GetFileNameWithoutExtension(path),
                        colliderSize = box.size,
                        colliderCenter = box.center,
                        lossyScale = t.lossyScale,
                    };
                    entries[design] = entry;
                    order.Add(design);
                }
                entry.count++;
            }
        }
        if (entries.Count == 0)
        {
            error = "no building in the patches passed the checks (BoxCollider on Obstacle, active).";
            return null;
        }
        order.Sort(System.StringComparer.Ordinal);

        BuildingCatalogue catalogue = AssetDatabase.LoadAssetAtPath<BuildingCatalogue>(AssetPath);
        bool created = catalogue == null;
        if (created)
        {
            catalogue = ScriptableObject.CreateInstance<BuildingCatalogue>();
        }
        catalogue.entries = order.Select(d => entries[d]).ToList();
        if (created)
        {
            AssetDatabase.CreateAsset(catalogue, AssetPath);
        }
        else
        {
            EditorUtility.SetDirty(catalogue);
        }
        AssetDatabase.SaveAssets();

        File.WriteAllText(JsonPath, ToJson(catalogue, paths.Count), new UTF8Encoding(false));
        AssetDatabase.ImportAsset(JsonPath);

        error = null;
        string summary = $"{catalogue.entries.Count} designs from {paths.Count} patches " +
                         $"({catalogue.entries.Sum(e => e.count)} buildings, {skipped} skipped) -> {AssetPath}, {JsonPath}";
        Debug.Log("BuildingCatalogueExport: " + summary);
        return summary;
    }

    /// <summary>Active with every parent up to the patch root — what a copy needs to be seen and avoided.</summary>
    private static bool ActiveInPatch(Transform t)
    {
        for (Transform p = t; p != null; p = p.parent)
        {
            if (!p.gameObject.activeSelf)
            {
                return false;
            }
        }
        return true;
    }

    private static readonly string[] Families =
    {
        "BnP_Small_Building_", "BnP_Large_Building_", "BnP_Apartment_", "Skyscraper_",
    };

    private static string FamilyPrefix(string name) => Families.FirstOrDefault(name.StartsWith) ?? "";

    private static string F(float v) => v.ToString("0.####", CultureInfo.InvariantCulture);

    private static string V(Vector3 v) => $"[{F(v.x)},{F(v.y)},{F(v.z)}]";

    /// <summary>
    /// One design per line: w / d / h are the footprint along the building's local X and Y and its height,
    /// in metres at the ScaledCity scale; count is how many instances the 48 patches hold.
    /// </summary>
    private static string ToJson(BuildingCatalogue catalogue, int patches)
    {
        var sb = new StringBuilder();
        sb.Append("{\"source\":\"").Append(PatchFolder).Append("\",\"patches\":").Append(patches)
          .Append(",\"designs\":[\n");
        for (int i = 0; i < catalogue.entries.Count; i++)
        {
            BuildingCatalogue.Entry e = catalogue.entries[i];
            sb.Append(i == 0 ? "" : ",\n")
              .Append($"{{\"design\":\"{e.design}\",\"family\":\"{e.family}\",\"w\":{F(e.Width)},\"d\":{F(e.Depth)}," +
                      $"\"h\":{F(e.Height)},\"count\":{e.count},\"patch\":\"{e.patch}\",\"source\":\"{e.source.name}\"," +
                      $"\"colliderSize\":{V(e.colliderSize)},\"colliderCenter\":{V(e.colliderCenter)}," +
                      $"\"lossyScale\":{V(e.lossyScale)}}}");
        }
        sb.Append("\n]}\n");
        return sb.ToString();
    }
}
