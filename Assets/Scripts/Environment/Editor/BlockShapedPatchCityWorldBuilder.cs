using System;
using System.IO;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;

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
        if (!ProceduralCityWorldBuilder.Build(ScenePath, Layout, true, out string summary,
                                              BlockShapedCityWorldBuilder.Configure))
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
        bool ok = ProceduralCityWorldBuilder.Build(ScenePath, layout, true, out string summary,
                                                   BlockShapedCityWorldBuilder.Configure);
        if (ok) Debug.Log("BlockShapedPatchCityWorldBuilder: " + summary);
        else Debug.LogError("BlockShapedPatchCityWorldBuilder: " + summary);
        EditorApplication.Exit(ok ? 0 : 1);
    }
}
