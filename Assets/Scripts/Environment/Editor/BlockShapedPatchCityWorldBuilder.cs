using System;
using System.IO;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;
using UnityEngine.SceneManagement;

/// <summary>
/// <c>Tools/Swarm/Build block-shaped patch city world</c>: makes <c>BlockShapedPatchCityWorld.unity</c>,
/// BlockShapedCityWorld on streets without its long sight lines. The layout is <c>citysim.blockcity</c>'s
/// <c>bbB_p40_s18_patch150_j0.15_sight487_seed1</c>: S4's design (p 40, 18 m streets, the 3.5 m footpath), its block
/// seeds on a hex lattice in patches about 150 m across, each turned at random, then 487 tries of
/// <c>citysim.sightline</c>'s pass, which nudges seeds into the long views that are left. At seed 1: 289 buildings,
/// 5.04 of each building's sides drawn as streets (S4's seeds 4.31), and a median longest view from a street of 253 m
/// with 1% over 400 m (S4's 272 m and 3%; hex's 581 m and 99%).
///
/// <para>Everything else is BlockShapedCityWorld's (<see cref="BlockShapedCityWorldBuilder.Configure"/>):
/// DiamondCityWorld's rig, the landmarks, the dressing with the short-street rules and the extra furniture, the
/// windowed prisms, S4's cohesion cutoff and the 45 m ceiling. The goals and their pedestrians are drawn at every Play
/// from the buildings (<see cref="BlockGoals"/>), as there. The Block-shaped city window changes this scene's design
/// too when it is the open one. Batch: <c>-executeMethod BlockShapedPatchCityWorldBuilder.Run</c> (CITY_LAYOUT names
/// another layout).</para>
///
/// <para>Its streets, and PracticeBlockWorld's, are furnished more densely than BlockShapedCityWorld's
/// (<see cref="MoreProps"/>).</para>
/// </summary>
public static class BlockShapedPatchCityWorldBuilder
{
    public const string ScenePath = "Assets/Scenes/BlockShapedPatchCityWorld.unity";
    /// <summary>python -B -m citysim.blockcity one --variant B --p 40 --s 18 --seed 1 --seeds patch --jitter 0.15
    /// --sight-tries 487 --scene (Assets/Scripts/Experiment; ~20-30 min).</summary>
    public const string Layout = "BlockShaped__bbB_p40_s18_patch150_j0.15_sight487_seed1.json";

    [MenuItem("Tools/Swarm/Build block-shaped patch city world")]
    private static void BuildFromMenu()
    {
        if (!EditorSceneManager.SaveCurrentModifiedScenesIfUserWantsTo()) return;
        if (File.Exists(ScenePath) &&
            !EditorUtility.DisplayDialog("Build block-shaped patch city world",
                $"Rebuild {ScenePath} from DiamondCityWorld? This replaces the scene, including hand edits.",
                "Rebuild", "Cancel"))
        {
            return;
        }
        if (!ProceduralCityWorldBuilder.Build(ScenePath, Layout, true, out string summary, Configure))
        {
            Debug.LogError("BlockShapedPatchCityWorldBuilder: " + summary);
        }
    }

    [MenuItem("Tools/Swarm/Build block-shaped patch city world", true)]
    private static bool CanBuild() => !EditorApplication.isPlayingOrWillChangePlaymode;

    /// <summary>Batchmode: builds the scene (CITY_LAYOUT, else <see cref="Layout"/>) and exits 0 / 1.</summary>
    public static void Run()
    {
        string layout = Environment.GetEnvironmentVariable(ProceduralCityBuilder.LayoutEnv);
        if (string.IsNullOrEmpty(layout)) layout = Layout;
        bool ok = ProceduralCityWorldBuilder.Build(ScenePath, layout, true, out string summary, Configure);
        if (ok) Debug.Log("BlockShapedPatchCityWorldBuilder: " + summary);
        else Debug.LogError("BlockShapedPatchCityWorldBuilder: " + summary);
        EditorApplication.Exit(ok ? 0 : 1);
    }

    /// <summary>BlockShapedCityWorld's configuration, then <see cref="MoreProps"/>.</summary>
    private static void Configure(ProceduralCityBuilder builder)
    {
        BlockShapedCityWorldBuilder.Configure(builder);
        CityDressing d = builder.GetComponent<CityDressing>();
        if (d != null) MoreProps(d);
    }

    /// <summary>Whether <paramref name="scene"/> is dressed with <see cref="MoreProps"/>: this scene and
    /// PracticeBlockWorld, which is built from a copy of it.</summary>
    internal static bool HasMoreProps(Scene scene) =>
        scene.path == ScenePath || scene.path == PracticeBlockWorldBuilder.PracticeScenePath;

    /// <summary>The user asked (2026-11-04) for more trees, bus stops, benches, trash cans, fire hydrants and phones in
    /// this scene and its practice copy, and no more traffic signals or signs. Over BlockShapedCityWorld's short-street
    /// rules and furniture: border-tree rows round more of the blocks, more bus stops, benches (each with its bin) and
    /// hydrants, a bin at the kerb by some crossings, and telephone booths at the kerb. Everything new stands at the kerb
    /// or against the facade, out of the lane a goal's walkers take (1.1-1.5 m off it).</summary>
    internal static void MoreProps(CityDressing d)
    {
        CityDressing.Rules r = d.rules;
        r.borderShare = 0.6f;
        r.borderCornerClearance = 3f;
        r.borderInset = 1.1f;
        r.singleTreeArea = 40f;
        r.busStopShare = 0.35f;
        r.benchShare = 0.6f;
        r.hydrantShare = 0.25f;
        d.extras.binShare = 0.15f;
        d.extras.phoneShare = 0.12f;
    }
}
