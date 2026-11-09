using System.IO;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;

/// <summary>
/// <c>Tools/Swarm/Build block city world</c>: makes <c>BlockCityWorld.unity</c>, the flyable city of S4 Split and
/// Surround (docs/split_study/README.md): small Voronoi blocks with one building each, where the swarm surrounds or
/// passes round a block to see its streets and reforms down the one the pilot picks. The layout is
/// <c>citysim.blockcity</c>'s variant R (a catalogue building whose avoidance cylinder fits its block, so the force law
/// is today's), copied out of the sweep as <c>layouts/BlockCity__&lt;id&gt;.json</c>; by default p 40 with 18 m streets,
/// the design Phase 1 chose that still has a building in most blocks.
///
/// <para>The scene is <see cref="ProceduralCityWorldBuilder"/>'s, with the city baked in as VoronoiCityWorld's is: the
/// landmarks, the street dressing and the road surfaces, and two things set on its <see cref="SwarmManager"/> because a
/// rebuild from DiamondCityWorld resets them: S4's cohesion cutoff (<see cref="S4CohesionCutoff"/>: at 40 the pull
/// across a block drags drones into the building, at 20 a straggler rejoins slowly) and the altitude ceiling under the
/// 50 m roofs (<see cref="CeilingMetres"/>). The separation the city wants, d* 1.0, is the dial's centre, so the dial
/// is unchanged. The layout has no goals: the scene is for flying the manoeuvres, not for the identify task. Batch:
/// <c>-executeMethod BlockCityWorldBuilder.Run</c> (CITY_LAYOUT overrides the layout).</para>
/// </summary>
public static class BlockCityWorldBuilder
{
    public const string ScenePath = "Assets/Scenes/BlockCityWorld.unity";
    public const string LayoutPattern = "BlockCity__*.json";
    /// <summary>S4's profile: SwarmManager.r0_coh (docs/split_study/README.md section 6.4).</summary>
    public const float S4CohesionCutoff = 30f;
    /// <summary>SwarmManager.maxHeightAboveTerrain: under the city's uniform 50 m roofs, as in VoronoiCityWorld.</summary>
    public const float CeilingMetres = 45f;

    [MenuItem("Tools/Swarm/Build block city world")]
    private static void BuildFromMenu()
    {
        if (!EditorSceneManager.SaveCurrentModifiedScenesIfUserWantsTo()) return;
        if (File.Exists(ScenePath) &&
            !EditorUtility.DisplayDialog("Build block city world",
                $"Rebuild {ScenePath} from DiamondCityWorld? This replaces the scene, including hand edits.",
                "Rebuild", "Cancel"))
        {
            return;
        }
        string layout = ProceduralCityWorldBuilder.DefaultLayout(LayoutPattern);
        if (!ProceduralCityWorldBuilder.Build(ScenePath, layout, true, out string summary, ConfigureS4))
        {
            Debug.LogError("BlockCityWorldBuilder: " + summary + NoLayoutHint(layout));
        }
    }

    [MenuItem("Tools/Swarm/Build block city world", true)]
    private static bool CanBuild() => !EditorApplication.isPlayingOrWillChangePlaymode;

    /// <summary>Batchmode: builds the scene (CITY_LAYOUT, else the first BlockCity__ layout) and exits 0 / 1.</summary>
    public static void Run()
    {
        string layout = System.Environment.GetEnvironmentVariable(ProceduralCityBuilder.LayoutEnv);
        if (string.IsNullOrEmpty(layout)) layout = ProceduralCityWorldBuilder.DefaultLayout(LayoutPattern);
        bool ok = ProceduralCityWorldBuilder.Build(ScenePath, layout, true, out string summary, ConfigureS4);
        if (!ok) Debug.LogError("BlockCityWorldBuilder: " + summary + NoLayoutHint(layout));
        EditorApplication.Exit(ok ? 0 : 1);
    }

    /// <summary>The landmarks first (so a rebuild keeps them), the dressing, VoronoiCityWorld's road surfaces where it
    /// has made them, and S4's two SwarmManager values. <see cref="BlockShapedCityWorldBuilder"/> configures its scene
    /// the same way.</summary>
    public static void ConfigureS4(ProceduralCityBuilder builder)
    {
        if (!LandmarkBuilder.AddTo(builder.gameObject.scene, out string landmarks))
        {
            Debug.LogWarning("BlockCityWorldBuilder: no landmarks: " + landmarks);
        }
        VoronoiCityWorldBuilder.EnsureDressing(builder);
        var road = AssetDatabase.LoadAssetAtPath<Material>(VoronoiCityWorldBuilder.VoronoiRoadMaterialPath);
        var junction = AssetDatabase.LoadAssetAtPath<Material>(VoronoiCityWorldBuilder.VoronoiJunctionMaterialPath);
        if (road != null && junction != null)
        {
            builder.streetStyle = ProceduralCityBuilder.StreetStyle.Textured;
            builder.streetMaterial = road;
            builder.junctionMaterial = junction;
        }
        else
        {
            Debug.LogWarning($"BlockCityWorldBuilder: no {VoronoiCityWorldBuilder.VoronoiRoadMaterialPath} (build " +
                             "VoronoiCityWorld once to make it); the streets stay flat.");
        }
        var manager = Object.FindFirstObjectByType<SwarmManager>(FindObjectsInactive.Include);
        if (manager == null)
        {
            Debug.LogWarning("BlockCityWorldBuilder: the scene has no SwarmManager; S4's profile is not set.");
            return;
        }
        Undo.RecordObject(manager, "S4 profile");
        manager.r0_coh = S4CohesionCutoff;
        manager.maxHeightAboveTerrain = CeilingMetres;
        EditorUtility.SetDirty(manager);
    }

    private static string NoLayoutHint(string layout) =>
        string.IsNullOrEmpty(layout)
            ? " Write one first (in Assets/Scripts/Experiment): python -B -m citysim.blockcity one --variant R --p 40 " +
              "--s 18 --seed 1 --out layouts/BlockCity__bbR_p40_s18_seed1.json"
            : "";
}
