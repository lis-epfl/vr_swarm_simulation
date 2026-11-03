using System.IO;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;
using UnityEngine.SceneManagement;

/// <summary>
/// Builds <c>PracticeBlockWorld</c>, where participants train before BlockShapedPatchCityWorld: the same swarm, rig,
/// display and landmarks, the four pedestrians standing under signs that name their hat (as in PracticeWorld), and a
/// small city of that scene's windowed prisms -- a row of buildings to fly round on the way in, then one goal with its
/// ring of neighbours to practise the search on.
///
/// <para><b>It is BlockShapedPatchCityWorld with a smaller city baked in</b>, copied file and all, for
/// <see cref="PracticeWorldBuilder"/>'s reason and one more: the trial scene's swarm tuning (S6 and S8's d_ref,
/// d_shield, obstacle shape, d_obs, slide and adaptive spread) was written into its <see cref="SwarmManager"/> by
/// <c>swarmtune apply</c> and is set by no builder, so a scene built from DiamondCityWorld would fly the old gains. The
/// copy also carries the landmarks, the rig, the input, screens and panorama, the dressing's rules and furniture, the
/// prisms' look, the textured roads and the ceiling. <b>Rebuild after changing BlockShapedPatchCityWorld</b> (a retune,
/// or adopting S10's ellipse), or the two drift apart; rebuilding discards hand edits to the practice scene.</para>
///
/// <para><b>The city is <see cref="Layout"/></b>, <c>citysim.practicecity</c>'s cut of the held-out seed-2 city of the
/// trial design, so no participant learns a piece of the trial city (seed 1): a goal block, its ring, and the ring's own
/// neighbours, 20 buildings, where the seed-2 city has them, two blocks in from its west edge at x -365 where ParkRoad
/// meets it. The edge row is what the swarm flies round first, then the ring, then the goal, 191 m from the spawn. The
/// goal is not in the layout: <see cref="BlockGoals"/> draws it at Play, as it draws the trial scene's three, with
/// <see cref="GoalsAtPlay"/> 1, and the cut leaves it only one building to draw. So the goal is always the same
/// building, while its hat and target walker change from session to session, as in PracticeWorld.</para>
///
/// <para>The <see cref="ExperimentRecorder"/> goes, as from PracticeWorld, so practice sessions write no runs; the start
/// line and the hat stations are PracticeWorld's (<see cref="PracticeWorldBuilder.AddStartLineAndStations"/>), 30 and
/// 45 m down ParkRoad. Batch: <c>-executeMethod PracticeBlockWorldBuilder.Run</c>.</para>
/// </summary>
public static class PracticeBlockWorldBuilder
{
    public const string PracticeScenePath = "Assets/Scenes/PracticeBlockWorld.unity";
    private const string SourceScenePath = BlockShapedPatchCityWorldBuilder.ScenePath;

    /// <summary>python -B -m citysim.practicecity (Assets/Scripts/Experiment): goal block 28 of
    /// bbB_p40_s18_patch150_j0.15_sight487_seed2.</summary>
    public const string Layout = "PracticeBlock__bbB_p40_s18_patch150_j0.15_sight487_seed2_cut_b28.json";

    /// <summary>The goals drawn at Play: one, the only building the cut leaves BlockGoals.</summary>
    private const int GoalsAtPlay = 1;

    [MenuItem("Tools/Swarm/Build practice block world")]
    private static void BuildFromMenu()
    {
        if (!EditorSceneManager.SaveCurrentModifiedScenesIfUserWantsTo())
        {
            return;
        }
        if (File.Exists(PracticeScenePath) &&
            !EditorUtility.DisplayDialog("Build practice block world",
                $"Rebuild {PracticeScenePath} from {SourceScenePath}? This replaces the practice scene, " +
                "including anything changed in it by hand.", "Rebuild", "Cancel"))
        {
            return;
        }
        Build();
    }

    [MenuItem("Tools/Swarm/Build practice block world", true)]
    private static bool CanBuild()
    {
        return !EditorApplication.isPlayingOrWillChangePlaymode;
    }

    /// <summary>Batchmode: builds the scene and exits 0 / 1.</summary>
    public static void Run()
    {
        EditorApplication.Exit(Build() ? 0 : 1);
    }

    /// <summary>
    /// Rebuilds the practice scene from BlockShapedPatchCityWorld, saves it and leaves it open. On failure it logs why,
    /// puts back whatever practice scene was there before, and returns false. The trial scene itself is never opened.
    /// </summary>
    public static bool Build()
    {
        PracticeWorldBuilder.Parts parts = PracticeWorldBuilder.LoadParts(out string error);
        if (parts == null)
        {
            Debug.LogError("PracticeBlockWorldBuilder: " + error);
            return false;
        }
        if (ProceduralCityBuilder.Resolve(Layout) == null)
        {
            Debug.LogError($"PracticeBlockWorldBuilder: no layout {Layout} under Assets/{ProceduralCityBuilder.LayoutFolder}; " +
                           "draw it with python -B -m citysim.practicecity.");
            return false;
        }

        SceneSetup[] previousSetup = EditorSceneManager.GetSceneManagerSetup();
        byte[] previousScene = File.Exists(PracticeScenePath) ? File.ReadAllBytes(PracticeScenePath) : null;

        // Nothing may hold the practice scene open while its file is replaced underneath it.
        EditorSceneManager.NewScene(NewSceneSetup.EmptyScene, NewSceneMode.Single);
        File.Copy(SourceScenePath, PracticeScenePath, true);
        AssetDatabase.ImportAsset(PracticeScenePath, ImportAssetOptions.ForceUpdate);
        Scene scene = EditorSceneManager.OpenScene(PracticeScenePath, OpenSceneMode.Single);

        string summary;
        try
        {
            summary = Populate(scene, parts, out error);
            if (summary != null && !EditorSceneManager.SaveScene(scene))
            {
                summary = null;
                error = $"could not save {PracticeScenePath}.";
            }
        }
        catch (System.Exception e)
        {
            summary = null;
            error = e.ToString();
        }

        if (summary == null)
        {
            Debug.LogError("PracticeBlockWorldBuilder: " + error + " The practice scene was left as it was.");
            EditorSceneManager.NewScene(NewSceneSetup.EmptyScene, NewSceneMode.Single);
            if (previousScene != null)
            {
                File.WriteAllBytes(PracticeScenePath, previousScene);
                AssetDatabase.ImportAsset(PracticeScenePath, ImportAssetOptions.ForceUpdate);
            }
            else
            {
                AssetDatabase.DeleteAsset(PracticeScenePath);
            }
            // An untitled scene has no path to reopen it from.
            if (previousSetup.Length > 0 &&
                System.Array.TrueForAll(previousSetup, s => !string.IsNullOrEmpty(s.path)))
            {
                EditorSceneManager.RestoreSceneManagerSetup(previousSetup);
            }
            return false;
        }

        Debug.Log($"PracticeBlockWorldBuilder: built {PracticeScenePath} from {SourceScenePath}: {summary}");
        return true;
    }

    /// <summary>
    /// Turns a fresh copy of BlockShapedPatchCityWorld into the practice world. Returns a one-line account of what it
    /// did, or null with <paramref name="error"/> set.
    /// </summary>
    private static string Populate(Scene scene, PracticeWorldBuilder.Parts parts, out string error)
    {
        swarmSpawn spawner = Object.FindFirstObjectByType<swarmSpawn>(FindObjectsInactive.Include);
        ProceduralCityBuilder builder = Object.FindFirstObjectByType<ProceduralCityBuilder>(FindObjectsInactive.Include);
        if (spawner == null || builder == null)
        {
            error = $"{SourceScenePath} has no {(spawner == null ? "swarmSpawn" : "ProceduralCityBuilder")}.";
            return null;
        }

        // The trial city goes and the practice city is baked in its place, dressed by the scene's own rules.
        builder.layoutPath = Layout;
        builder.goalsAtPlay = GoalsAtPlay;
        builder.useFixedGoalSeed = false;
        BlockShapedCityWorldBuilder.EnsureDressing(builder);
        BlockShapedCityWorldBuilder.EnsureStyle(builder);
        if (!builder.Bake(out error))
        {
            return null;
        }
        string goal = OnlyGoal(builder.ResolvedPath, out error);
        if (goal == null)
        {
            return null;
        }

        // Bake has stood the spawn where the layout puts it; the practice area is laid out from there.
        Vector3 spawn = new Vector3(spawner.start_x, spawner.start_y, spawner.start_z);
        float[] own = builder.Layout.layout.spawn;
        if (own == null || own.Length < 3 || Vector2.Distance(new Vector2(own[0], own[2]), new Vector2(spawn.x, spawn.z)) > 1f)
        {
            error = $"the scene's spawn {spawn} is not the layout's; turn the ProceduralCityBuilder's setSpawn on.";
            return null;
        }

        int stripped = 0;
        foreach (ExperimentRecorder recorder in Object.FindObjectsByType<ExperimentRecorder>(FindObjectsInactive.Include,
                                                                                              FindObjectsSortMode.None))
        {
            Object.DestroyImmediate(recorder);
            stripped++;
        }

        string stations = PracticeWorldBuilder.AddStartLineAndStations(spawn, parts);

        EditorSceneManager.MarkSceneDirty(scene);
        return $"baked {builder.Layout.layout.id} ({builder.Layout.boxes.Length} buildings; at Play the goal is {goal}), " +
               $"removed {stripped} recorder, {stations}.";
    }

    /// <summary>
    /// The building BlockGoals draws the goal at, checked to be the only one it can: one goal is drawn, two are not.
    /// Null with <paramref name="error"/> set if the layout leaves it no goal or a choice of goals.
    /// </summary>
    private static string OnlyGoal(string path, out string error)
    {
        // Draw sets roles in the file it is handed, so each draw gets a fresh read.
        ProceduralCityBuilder.LayoutGoal[] one = BlockGoals.Draw(Read(path), new System.Random(0), GoalsAtPlay, out error);
        if (one == null)
        {
            error = $"BlockGoals can draw no goal from {Path.GetFileName(path)}: {error}";
            return null;
        }
        if (BlockGoals.Draw(Read(path), new System.Random(0), GoalsAtPlay + 1, out _) != null)
        {
            error = $"BlockGoals can draw more than one goal from {Path.GetFileName(path)}, so the practice goal would " +
                    "move from session to session; draw the layout again with citysim.practicecity.";
            return null;
        }
        error = null;
        return $"{one[0].centre_box} with a ring of {one[0].n_b}";
    }

    private static ProceduralCityBuilder.LayoutFile Read(string path) =>
        JsonUtility.FromJson<ProceduralCityBuilder.LayoutFile>(File.ReadAllText(path));
}
