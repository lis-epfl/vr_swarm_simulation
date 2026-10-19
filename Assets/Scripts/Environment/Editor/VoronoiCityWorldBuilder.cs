using System.IO;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;

/// <summary>
/// <c>Tools/Swarm/Build Voronoi city world</c>: makes <c>VoronoiCityWorld.unity</c>, DiamondCityWorld's own buildings on
/// the density study's Voronoi streets. The layout is <c>citygen mixcity</c>'s
/// (<c>layouts/VoronoiCity__&lt;id&gt;.json</c>): every building drawn from DiamondCityWorld's 37 designs in the shares
/// they stand there, at the ScaledCity scale, spread evenly over the blocks, to the city's own coverage over its tile
/// grid by default, with its tile pitch as the block spacing -- the same city in everything but its streets, which no
/// longer run straight for kilometres. The goals are the study's (an occluder ringed by patrolled buildings), built from
/// the same mix so a ring of identical buildings does not give one away. The streets are road
/// (<see cref="ProceduralCityBuilder.StreetStyle.Textured"/>: ParkRoad's asphalt, plain junctions, zebra crossings).
///
/// <para>The scene itself is <see cref="ProceduralCityWorldBuilder"/>'s: DiamondCityWorld with its city taken out and a
/// <see cref="ProceduralCityBuilder"/>, here with the city BAKED in (<see cref="ProceduralCityBuilder.Bake"/>), so it is
/// in the editor without Play; Play adopts the bake and makes only the goals. The recorder, the bench, analyse.py and the
/// replica read it as any generated layout (by the id in the session). Tune the density and see it with
/// <see cref="VoronoiCityWindow"/> (<c>Tools/Swarm/Voronoi city…</c>), which generates a layout and rebakes the open
/// scene. Rebuild the scene after changing DiamondCityWorld, and regenerate after restyling its buildings and
/// re-exporting its city obstacles. Batch: <c>-executeMethod VoronoiCityWorldBuilder.Run</c> (CITY_LAYOUT overrides
/// the layout).</para>
/// </summary>
public static class VoronoiCityWorldBuilder
{
    public const string ScenePath = "Assets/Scenes/VoronoiCityWorld.unity";
    /// <summary>citygen mixcity's file names (citygen.MIXCITY_PREFIX).</summary>
    public const string LayoutPattern = "VoronoiCity__*.json";
    public const string RoadMaterialPath = "Assets/Modular City Pack/Materials/MC_Asphalt_Road Texture_M_01.mat";
    public const string JunctionMaterialPath = "Assets/Modular City Pack/Materials/MC_Asphalt_Cross and Corner.mat";
    public const string VoronoiRoadMaterialPath = "Assets/Materials/VoronoiRoad.mat";
    public const string VoronoiJunctionMaterialPath = "Assets/Materials/VoronoiJunction.mat";
    private const float AsphaltTint = 0.5f;

    [MenuItem("Tools/Swarm/Build Voronoi city world")]
    private static void BuildFromMenu()
    {
        if (!EditorSceneManager.SaveCurrentModifiedScenesIfUserWantsTo()) return;
        if (File.Exists(ScenePath) &&
            !EditorUtility.DisplayDialog("Build Voronoi city world",
                $"Rebuild {ScenePath} from DiamondCityWorld? This replaces the scene, including hand edits.",
                "Rebuild", "Cancel"))
        {
            return;
        }
        string layout = ProceduralCityWorldBuilder.DefaultLayout(LayoutPattern);
        if (!ProceduralCityWorldBuilder.Build(ScenePath, layout, true, out string summary, Configure))
        {
            Debug.LogError("VoronoiCityWorldBuilder: " + summary + NoLayoutHint(layout));
        }
    }

    [MenuItem("Tools/Swarm/Build Voronoi city world", true)]
    private static bool CanBuild() => !EditorApplication.isPlayingOrWillChangePlaymode;

    /// <summary>Batchmode: builds the scene (CITY_LAYOUT, else the first mixcity layout) and exits 0 / 1.</summary>
    public static void Run()
    {
        string layout = System.Environment.GetEnvironmentVariable(ProceduralCityBuilder.LayoutEnv);
        if (string.IsNullOrEmpty(layout)) layout = ProceduralCityWorldBuilder.DefaultLayout(LayoutPattern);
        bool ok = ProceduralCityWorldBuilder.Build(ScenePath, layout, true, out string summary, Configure);
        if (!ok) Debug.LogError("VoronoiCityWorldBuilder: " + summary + NoLayoutHint(layout));
        EditorApplication.Exit(ok ? 0 : 1);
    }

    /// <summary>Road streets: ParkRoad's lane-marked asphalt for the stretches, the pack's crossing texture for the
    /// junctions and crossings, both through this scene's own lighter copies (<see cref="Tinted"/>), which ParkRoad is
    /// moved onto too so the spawn road matches. Without the pack (it is git-ignored) the streets stay flat.</summary>
    private static void Configure(ProceduralCityBuilder builder)
    {
        var packRoad = AssetDatabase.LoadAssetAtPath<Material>(RoadMaterialPath);
        Material road = Tinted(VoronoiRoadMaterialPath, RoadMaterialPath);
        Material junction = Tinted(VoronoiJunctionMaterialPath, JunctionMaterialPath);
        if (road == null || junction == null)
        {
            Debug.LogWarning($"VoronoiCityWorldBuilder: no {RoadMaterialPath} / {JunctionMaterialPath} (the Modular City " +
                             "Pack); the streets stay flat.");
            return;
        }
        builder.streetStyle = ProceduralCityBuilder.StreetStyle.Textured;
        builder.streetMaterial = road;
        builder.junctionMaterial = junction;
        foreach (GameObject root in builder.gameObject.scene.GetRootGameObjects())
        {
            if (root.name != "ParkRoad") continue;
            foreach (MeshRenderer r in root.GetComponentsInChildren<MeshRenderer>(true))
            {
                Material[] mats = r.sharedMaterials;
                for (int i = 0; i < mats.Length; i++)
                {
                    if (mats[i] == packRoad) mats[i] = road;
                }
                r.sharedMaterials = mats;
            }
        }
    }

    /// <summary>A copy of the pack material at <see cref="AsphaltTint"/>, made if missing (so a tint set by hand on it
    /// survives a rebuild). The pack's own materials tint the albedo to 0.196, which takes asphalt to ~5% and the
    /// white paint of the lane marks and crossings to ~15%: from a drone they read as black. The pack's materials are
    /// left alone; ParkRoad and the tile cities use them.</summary>
    private static Material Tinted(string path, string sourcePath)
    {
        var m = AssetDatabase.LoadAssetAtPath<Material>(path);
        if (m != null) return m;
        var src = AssetDatabase.LoadAssetAtPath<Material>(sourcePath);
        if (src == null) return null;
        m = new Material(src) { name = Path.GetFileNameWithoutExtension(path) };
        m.color = new Color(AsphaltTint, AsphaltTint, AsphaltTint, 1f);
        AssetDatabase.CreateAsset(m, path);
        return m;
    }

    private static string NoLayoutHint(string layout) =>
        string.IsNullOrEmpty(layout)
            ? " Generate one first: python -B -m citysim.citygen mixcity --city DiamondCityWorld (in Assets/Scripts/Experiment)."
            : "";
}
