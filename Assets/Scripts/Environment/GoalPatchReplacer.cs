using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using UnityEngine;

/// <summary>
/// Randomly replaces <c>n</c> of the <c>MC_Patch_*</c> city tiles under <see cref="cityPack"/>
/// with instances of a goal patch prefab. Runs once at play-mode <see cref="Start"/>, so each
/// play session produces a different city layout; the change is not persisted to the scene asset.
///
/// The 48 tiles are direct children of <c>City_Pack_01</c> named <c>MC_Patch_01</c>…<c>MC_Patch_48</c>.
/// A name-prefix filter cleanly separates them from the other (tree/road/ground) children.
///
/// Set <see cref="mode"/> to <see cref="PlacementMode.Replay"/> to instead reproduce the exact goal
/// layout of a previous run: it reads the goal positions from that run's <c>*_session.json</c> (written
/// by <c>ExperimentRecorder</c>) and replaces the tiles nearest those positions, in the same order — so
/// <c>goalIndex</c> lines up with the original. Since past runs did not use a fixed seed, this is the
/// only way to recover a specific historical layout.
///
/// A goal is laid over the <i>block</i> it replaces, not copied from the tile's transform (see
/// <see cref="GoalPositions"/>), takes over any scenery tied to that tile (see
/// <see cref="CarryTiedScenery"/>), and always gets verge trees, whether or not the tile had any (see
/// <see cref="PlantGoalVerges"/>).
///
/// <para><b>Diamond plazas are candidates too</b> (<see cref="includeDiamondPlazas"/>): the blocks standing in the
/// diamonds' parks, found by their <see cref="DiamondPlaza"/> component because they are deliberately not tiles. A goal
/// replacing one is still placed under <see cref="cityPack"/>, where its kerb is a tile's kerb like any other goal's, so
/// the city stays findable and <see cref="StreetWidthTuner"/> brings it to the city's block scale. It matches the
/// block it replaces in the ways a plaza differs from a tile: its road plate is hidden and it stands at the plaza's
/// raised ground. Like every goal it gets verge trees, which round a plaza stand on the park's lawn. Adjacency still
/// takes its spacing from the tile grid alone.</para>
/// </summary>
public class GoalPatchReplacer : MonoBehaviour
{
    public enum PlacementMode
    {
        Random, // fresh random selection each play
        Replay, // reproduce a previous run's goal positions from its session JSON
    }

    [Tooltip("The City_Pack_01 root whose MC_Patch_* children get replaced. Defaults to this transform if unset.")]
    [SerializeField] private Transform cityPack;

    [Tooltip("The goal patch prefab instantiated in place of each selected MC_Patch tile.")]
    [SerializeField] private GameObject goalPrefab;

    [Tooltip("How many tiles to replace. Clamped to the number of available MC_Patch tiles. (Random mode only.)")]
    [SerializeField] private int replaceCount = 3;

    [Tooltip("Name prefix identifying replaceable tiles among the City_Pack children.")]
    [SerializeField] private string tilePrefix = "MC_Patch";

    [Tooltip("Also offer the blocks standing in the diamonds' parks (DiamondPlaza) as goal candidates. A goal " +
             "replacing one has no road ring, like the plaza itself.")]
    [SerializeField] private bool includeDiamondPlazas = true;

    /// <summary>A road plate's name prefix: the street ring round a tile's block, which a plaza does not have.</summary>
    private const string RoadPlatePrefix = "Road_Structure_";

    [Header("Replay (reproduce a past run)")]
    [Tooltip("Random = a fresh random layout each play. Replay = reproduce the goal positions recorded in a previous run's session JSON.")]
    [SerializeField] private PlacementMode mode = PlacementMode.Random;

    [Tooltip("Which run to replay: the file name/stem of a previous run's session JSON in " +
             "persistentDataPath/experiment (e.g. \"ERIC_t7_Swarm_20260706_232112\", with or without the " +
             "\"_session.json\" suffix). An absolute path is also accepted. Only used when mode = Replay.")]
    [SerializeField] private string replaySessionFile = "";

    [Header("Spacing")]
    [Tooltip("If true, no two goals may occupy adjacent grid tiles. (Random mode only.)")]
    [SerializeField] private bool preventAdjacent = true;

    [Tooltip("If true, diagonal neighbours also count as adjacent (blocks all 8 surrounding tiles); otherwise only the 4 edge neighbours are blocked.")]
    [SerializeField] private bool blockDiagonalNeighbors = true;

    [Header("Randomization")]
    [Tooltip("If true, seed the RNG with 'seed' for a reproducible selection each play.")]
    [SerializeField] private bool useFixedSeed = false;
    [SerializeField] private int seed = 0;

    [Tooltip("If true, Destroy the replaced tile; otherwise just deactivate it (reversible).")]
    [SerializeField] private bool destroyOriginal = true;

    [Header("Verge trees")]
    [Tooltip("Plant verge trees on every goal with the scene's VergeTreePlanter settings, whether or not the tile " +
             "it replaced had any. They are laid out around the goal's own block, so a replaced tile's trees are " +
             "replanted rather than carried over. Off, a goal only has trees if the tile it replaced did.")]
    [SerializeField] private bool plantVergeTrees = true;

    [Tooltip("The planter whose settings the goals' trees use. Empty uses the one in this scene; a scene with " +
             "none plants nothing.")]
    [SerializeField] private VergeTreePlanter vergeTreePlanter;

    // The goal patches instantiated this play session, in placement order. Populated in Start;
    // exposed so experiment tooling (e.g. ExperimentRecorder) can find the goals at runtime
    // without scanning by name. Empty until Start has run.
    private readonly List<GameObject> placedGoals = new List<GameObject>();
    public IReadOnlyList<GameObject> PlacedGoals => placedGoals;

    // Start is called before the first frame update
    private void Start()
    {
        if (cityPack == null)
        {
            cityPack = transform;
        }

        if (goalPrefab == null)
        {
            Debug.LogError("GoalPatchReplacer: goalPrefab is not assigned; no tiles replaced.", this);
            return;
        }

        // Collect the direct MC_Patch children only (not nested geometry inside each tile prefab).
        List<Transform> tiles = new List<Transform>();
        foreach (Transform child in cityPack)
        {
            if (child.name.StartsWith(tilePrefix))
            {
                tiles.Add(child);
            }
        }

        // The diamonds' plazas, beside the tiles. Candidates are the two together; the tiles alone set the spacing.
        Dictionary<Transform, DiamondPlaza> plazas = FindPlazas();
        List<Transform> candidates = new List<Transform>(tiles);
        candidates.AddRange(plazas.Keys);

        if (candidates.Count == 0)
        {
            Debug.LogWarning($"GoalPatchReplacer: no '{tilePrefix}' tiles found under {cityPack.name}; nothing to replace.", this);
            return;
        }

        // Where a goal for each candidate would stand. Selection compares these for every pair of candidates and
        // replay for every recorded goal, so they are worked out once, up front.
        Dictionary<Transform, Vector3> goalPositions = GoalPositions(tiles, plazas);

        // Choose which candidates to replace, then run the shared placement loop below.
        List<Transform> selected = mode == PlacementMode.Replay
            ? SelectReplayTiles(candidates, goalPositions)
            : SelectRandomTiles(candidates, tiles, goalPositions);

        if (selected == null || selected.Count == 0)
        {
            Debug.LogWarning("GoalPatchReplacer: no tiles selected; nothing replaced.", this);
            return;
        }

        int onPlazas = 0;
        foreach (Transform tile in selected)
        {
            GameObject goal = Instantiate(goalPrefab, cityPack);
            goal.name = $"goal_patch ({tile.name})";

            goal.transform.localPosition = goalPositions[tile];
            if (plazas.ContainsKey(tile))
            {
                // Not a child of cityPack, so its placement is carried over through the world frame.
                goal.transform.localRotation = Quaternion.Inverse(cityPack.rotation) * tile.rotation;
                goal.transform.localScale = Vector3.one;
                HideRoadPlate(goal.transform);
                onPlazas++;
            }
            else
            {
                goal.transform.localRotation = tile.localRotation;
                goal.transform.localScale = tile.localScale;
            }

            CarryTiedScenery(tile, goal.transform);

            placedGoals.Add(goal);

            if (destroyOriginal)
            {
                Destroy(tile.gameObject);
            }
            else
            {
                tile.gameObject.SetActive(false);
            }
        }

        Debug.Log($"GoalPatchReplacer: replaced {selected.Count} of {tiles.Count} '{tilePrefix}' tiles and {plazas.Count} " +
                  $"diamond plazas with goals, {onPlazas} of them on plazas ({mode}).", this);

        if (plantVergeTrees)
        {
            StartCoroutine(PlantGoalVerges());
        }
    }

    // ------------------------------------------------------------------ placement

    /// <summary>
    /// Where the goal replacing each tile goes, in cityPack-local space: the goal's block laid over the
    /// tile's block, with Y pinned to ground level so goals sit at y=0 even if the replaced patch had a
    /// non-zero height.
    ///
    /// <para>Aligned on the blocks rather than copied from the tile's transform, because the two are not the
    /// same point in every tile. <c>MC_Patch_32</c> has its whole block baked ~318 units off its own pivot —
    /// an authoring quirk of the pack, inherited by the ScaledCity fork — so its transform sits at the city
    /// centre while its buildings stand at the edge. Copying the transform dropped that goal across the four
    /// middle tiles, and because selection takes the grid pitch from the closest pair of tiles, it also
    /// shrank the pitch to ~64 units, so diagonal neighbours stopped being blocked. For every other tile the
    /// two rules agree to within a fifth of a unit, so replaying an earlier run still matches.</para>
    ///
    /// <para>A block is located by its kerb, the point <see cref="StreetWidthTuner"/> scales the block about.
    /// If the goal prefab or a tile has no kerb, that tile falls back to copying its transform.</para>
    ///
    /// <para>A plaza's goal has its kerb laid over the plaza's centre, and stands at the plaza's own ground rather
    /// than at zero: see <see cref="DiamondPlaza"/>.</para>
    /// </summary>
    private Dictionary<Transform, Vector3> GoalPositions(List<Transform> tiles, Dictionary<Transform, DiamondPlaza> plazas)
    {
        // The goal's block relative to its own root, measured on the asset so no instance is needed.
        Transform goalKerb = CityTiles.FindKerb(goalPrefab.transform);
        Vector3 goalBlockOffset = goalKerb != null
            ? goalPrefab.transform.InverseTransformPoint(goalKerb.position)
            : Vector3.zero;

        Dictionary<Transform, Vector3> positions = new Dictionary<Transform, Vector3>(tiles.Count + plazas.Count);
        foreach (Transform tile in tiles)
        {
            Transform kerb = goalKerb != null ? CityTiles.FindKerb(tile) : null;
            Vector3 pos = kerb != null
                ? cityPack.InverseTransformPoint(kerb.position)
                  - tile.localRotation * Vector3.Scale(tile.localScale, goalBlockOffset)
                : tile.localPosition;
            pos.y = 0f;
            positions[tile] = pos;
        }
        foreach (KeyValuePair<Transform, DiamondPlaza> plaza in plazas)
        {
            Quaternion rotation = Quaternion.Inverse(cityPack.rotation) * plaza.Key.rotation;
            Vector3 pos = cityPack.InverseTransformPoint(plaza.Value.Centre.position) - rotation * goalBlockOffset;
            pos.y = cityPack.InverseTransformPoint(plaza.Value.GroundPoint).y;
            positions[plaza.Key] = pos;
        }
        return positions;
    }

    /// <summary>
    /// The active diamond plazas in the city's scene, by their root, if they are included at all. The city's scene, not
    /// this component's: <see cref="SwarmManager"/> moves the object this sits on to DontDestroyOnLoad in Awake.
    /// </summary>
    private Dictionary<Transform, DiamondPlaza> FindPlazas()
    {
        Dictionary<Transform, DiamondPlaza> plazas = new Dictionary<Transform, DiamondPlaza>();
        if (!includeDiamondPlazas)
        {
            return plazas;
        }
        foreach (DiamondPlaza plaza in FindObjectsByType<DiamondPlaza>(FindObjectsInactive.Exclude, FindObjectsSortMode.None))
        {
            if (plaza.gameObject.scene != cityPack.gameObject.scene)
            {
                continue;
            }
            if (plaza.Centre == null)
            {
                Debug.LogWarning($"GoalPatchReplacer: diamond plaza {plaza.name} has no centre to place a goal by; skipped.", plaza);
                continue;
            }
            plazas[plaza.transform] = plaza;
        }
        return plazas;
    }

    /// <summary>Hide a goal's road ring: it replaces a plaza, which stands in the park's lawn with no street round it.</summary>
    private static void HideRoadPlate(Transform goal)
    {
        foreach (Renderer r in goal.GetComponentsInChildren<Renderer>(true))
        {
            if (r.name.StartsWith(RoadPlatePrefix))
            {
                r.gameObject.SetActive(false);
            }
        }
    }

    /// <summary>
    /// Hand the scenery tied to <paramref name="tile"/> (every <see cref="CityTiles.TiedContainers"/> child)
    /// over to the goal replacing it, keeping its world placement. Untied, the street trees and plates hang off
    /// the city root and survive the replacement; tied, they would be destroyed with the tile, leaving a gap in
    /// the medians and verges exactly where each goal is — a cue visible from the air. A tile that was never
    /// tied has nothing to carry. With <see cref="plantVergeTrees"/> on, the verge is replanted a frame later.
    /// </summary>
    private static void CarryTiedScenery(Transform tile, Transform goal)
    {
        foreach (string containerName in CityTiles.TiedContainers)
        {
            Transform container = tile.Find(containerName);
            if (container != null)
            {
                container.SetParent(goal, true);
            }
        }
    }

    /// <summary>
    /// Plant verge trees on every goal. Left to <see cref="CarryTiedScenery"/>, a goal has trees only when the
    /// tile it replaced was one of the <see cref="VergeTreePlanter"/>'s picks.
    ///
    /// <para>Waits a frame because <see cref="StreetWidthTuner"/> only brings a goal's block to the city's block
    /// scale on its first Update. The planter keeps each spot clear of what already stands on the tile — above
    /// all an interior street's mouth — so planting in Start would measure the block at its authored size and
    /// clear the wrong stretch of verge. A coroutine resumes after every Update, so one frame is enough.</para>
    ///
    /// <para>Planting replaces the verge a goal carried over from its tile: those trees were cleared around the
    /// old block's streets and props, not the goal's. The planter seeds each tile by name and a goal is named
    /// for the tile it replaced, so a replayed run gets the same trees.</para>
    ///
    /// <para>The planter finds the city by the goals it is handed, not by its own scene: in the city scenes it
    /// sits, like this component, on <c>gameManager</c>, which <see cref="SwarmManager"/> moves to DontDestroyOnLoad
    /// in Awake. Before 2026-09-30 it looked in its own scene, found no city, and logged an error instead of
    /// planting, so in every run until then a goal kept only the verge carried over from its tile.</para>
    /// </summary>
    private IEnumerator PlantGoalVerges()
    {
        yield return null;

        VergeTreePlanter planter = vergeTreePlanter != null ? vergeTreePlanter : FindPlanter();
        if (planter == null)
        {
            yield break; // a city with no verge trees has none for a goal to match
        }

        List<Transform> goals = new List<Transform>(placedGoals.Count);
        foreach (GameObject goal in placedGoals)
        {
            if (goal != null)
            {
                goals.Add(goal.transform);
            }
        }
        if (goals.Count > 0)
        {
            planter.Plant(goals);
        }
    }

    /// <summary>
    /// The <see cref="VergeTreePlanter"/> in this component's scene or the city's, or null if neither has one. Both,
    /// because this component's is DontDestroyOnLoad at runtime wherever it sits on <c>gameManager</c>.
    /// </summary>
    private VergeTreePlanter FindPlanter()
    {
        foreach (VergeTreePlanter planter in FindObjectsByType<VergeTreePlanter>(FindObjectsInactive.Include,
                                                                                FindObjectsSortMode.None))
        {
            if (planter.gameObject.scene == gameObject.scene || planter.gameObject.scene == cityPack.gameObject.scene)
            {
                return planter;
            }
        }
        return null;
    }

    // ------------------------------------------------------------------ random selection

    /// <summary>
    /// Randomly pick up to <see cref="replaceCount"/> non-adjacent candidates. <paramref name="gridTiles"/> are the
    /// tiles among them, which alone set the spacing.
    /// </summary>
    private List<Transform> SelectRandomTiles(List<Transform> tiles, List<Transform> gridTiles,
                                              Dictionary<Transform, Vector3> goalPositions)
    {
        if (useFixedSeed)
        {
            UnityEngine.Random.InitState(seed);
        }

        int count = Mathf.Clamp(replaceCount, 0, tiles.Count);
        if (count == 0)
        {
            Debug.LogWarning($"GoalPatchReplacer: nothing to replace (found {tiles.Count} tiles, replaceCount clamped to 0).", this);
            return null;
        }

        // Derive the "adjacent" distance from the actual layout: the closest pair of tiles is one
        // grid pitch apart. Tile numbering is scrambled relative to position, so adjacency must come
        // from world XZ, not tile index. Edge neighbours sit ~1 pitch away, diagonals ~1.41 pitch.
        // Measured between goal positions, i.e. between blocks, for the reason GoalPositions gives.
        // The plazas are left out: they are off the grid, so the closest pair involving one says
        // nothing about the pitch, and a shorter one would stop diagonal tiles counting as adjacent.
        List<Transform> spacing = gridTiles.Count > 1 ? gridTiles : tiles;
        float thresholdSq = 0f;
        if (preventAdjacent && spacing.Count > 1)
        {
            float minSq = float.MaxValue;
            for (int a = 0; a < spacing.Count; a++)
            {
                for (int b = a + 1; b < spacing.Count; b++)
                {
                    float d = SqrDistanceXZ(goalPositions[spacing[a]], goalPositions[spacing[b]]);
                    if (d < minSq)
                    {
                        minSq = d;
                    }
                }
            }
            float mult = blockDiagonalNeighbors ? 1.5f : 1.2f;
            thresholdSq = minSq * mult * mult;
        }

        // Full Fisher-Yates shuffle so the accept/reject scan below sees tiles in random order.
        for (int i = tiles.Count - 1; i > 0; i--)
        {
            int j = UnityEngine.Random.Range(0, i + 1);
            (tiles[i], tiles[j]) = (tiles[j], tiles[i]);
        }

        // Greedily accept tiles that aren't adjacent to an already-chosen one, until we reach 'count'.
        List<Transform> selected = new List<Transform>(count);
        foreach (Transform candidate in tiles)
        {
            if (selected.Count >= count)
            {
                break;
            }

            if (preventAdjacent)
            {
                bool adjacent = false;
                foreach (Transform chosen in selected)
                {
                    if (SqrDistanceXZ(goalPositions[candidate], goalPositions[chosen]) <= thresholdSq)
                    {
                        adjacent = true;
                        break;
                    }
                }
                if (adjacent)
                {
                    continue;
                }
            }

            selected.Add(candidate);
        }

        if (selected.Count < count)
        {
            Debug.LogWarning($"GoalPatchReplacer: could only place {selected.Count} of {count} requested goals without adjacency; the non-adjacency constraint left no more valid tiles.", this);
        }

        return selected;
    }

    // ------------------------------------------------------------------ replay selection

    /// <summary>
    /// Reproduce a previous run: for each goal position recorded in <see cref="replaySessionFile"/>,
    /// pick the tile whose goal would stand nearest that position (in cityPack-local XZ). Preserves the
    /// recorded order so the resulting <see cref="PlacedGoals"/> — and thus each goalIndex — matches the
    /// original run.
    /// </summary>
    private List<Transform> SelectReplayTiles(List<Transform> tiles, Dictionary<Transform, Vector3> goalPositions)
    {
        ReplaySession session = LoadReplaySession(out string resolvedPath);
        if (session == null)
        {
            return null; // LoadReplaySession already logged the reason
        }
        if (session.goals == null || session.goals.Count == 0)
        {
            Debug.LogError($"GoalPatchReplacer: replay session '{resolvedPath}' has no goals.", this);
            return null;
        }

        List<Transform> selected = new List<Transform>(session.goals.Count);
        HashSet<Transform> used = new HashSet<Transform>();

        foreach (ReplayGoal g in session.goals)
        {
            // Recorded positions are world-space; compare in cityPack-local space to match tile.localPosition
            // regardless of where City_Pack sits (goal placement mirrors that local frame).
            Vector3 local = cityPack.InverseTransformPoint(new Vector3(g.goalX, g.goalY, g.goalZ));

            Transform best = null;
            float bestSq = float.MaxValue;
            foreach (Transform t in tiles)
            {
                if (used.Contains(t))
                {
                    continue; // a tile already claimed by an earlier goal
                }
                float d = SqrDistanceXZ(local, goalPositions[t]);
                if (d < bestSq)
                {
                    bestSq = d;
                    best = t;
                }
            }

            if (best == null)
            {
                Debug.LogWarning($"GoalPatchReplacer: no free tile left to match replay goal {g.goalIndex}; skipping.", this);
                continue;
            }

            used.Add(best);
            selected.Add(best);

            // ~1 unit tolerance: an exact replay lands on the tile's goal position. A large gap means the
            // scene layout differs from the recorded run (different city/tile set) — replay is only
            // approximate. A run from before goals were block-aligned that placed one via MC_Patch_32
            // recorded it at the city centre, where no tile's goal stands any more, so it warns here too.
            if (bestSq > 1f)
            {
                Debug.LogWarning(
                    $"GoalPatchReplacer: replay goal {g.goalIndex} at ({g.goalX:F1},{g.goalZ:F1}) matched " +
                    $"'{best.name}' but the nearest tile is {Mathf.Sqrt(bestSq):F2} away — layout may not match the recorded run.", this);
            }
        }

        Debug.Log($"GoalPatchReplacer: replaying {selected.Count} goal(s) from {Path.GetFileName(resolvedPath)}.", this);
        return selected;
    }

    /// <summary>Read and parse the replay session JSON. Returns null (and logs) on any failure.</summary>
    private ReplaySession LoadReplaySession(out string resolvedPath)
    {
        resolvedPath = ResolveReplayPath();
        if (resolvedPath == null)
        {
            Debug.LogError(
                $"GoalPatchReplacer: replay mode is on but session file '{replaySessionFile}' was not found " +
                $"(looked in {Path.Combine(Application.persistentDataPath, "experiment")}).", this);
            return null;
        }

        try
        {
            string json = File.ReadAllText(resolvedPath);
            return JsonUtility.FromJson<ReplaySession>(json);
        }
        catch (Exception e)
        {
            Debug.LogError($"GoalPatchReplacer: failed to read replay session '{resolvedPath}'. {e}", this);
            return null;
        }
    }

    /// <summary>
    /// Resolve <see cref="replaySessionFile"/> to a readable path. Accepts an absolute/relative path as-is,
    /// or a bare run name/stem under persistentDataPath/experiment (with or without the "_session.json" suffix).
    /// Returns null if nothing exists.
    /// </summary>
    private string ResolveReplayPath()
    {
        if (string.IsNullOrWhiteSpace(replaySessionFile))
        {
            return null;
        }

        string name = replaySessionFile.Trim();
        string dir = Path.Combine(Application.persistentDataPath, "experiment");
        string[] candidates =
        {
            name,                                        // absolute or cwd-relative path
            Path.Combine(dir, name),                     // full file name in the experiment dir
            Path.Combine(dir, name + ".json"),
            Path.Combine(dir, name + "_session.json"),   // bare run stem
        };

        foreach (string c in candidates)
        {
            if (File.Exists(c))
            {
                return c;
            }
        }
        return null;
    }

    /// <summary>Squared horizontal (XZ) distance, ignoring Y so tiles at different heights still compare by grid cell.</summary>
    private static float SqrDistanceXZ(Vector3 a, Vector3 b)
    {
        float dx = a.x - b.x;
        float dz = a.z - b.z;
        return dx * dx + dz * dz;
    }

    // ---- session JSON subset (JsonUtility reads the fields it recognises, ignores the rest) ----
    [Serializable]
    private class ReplaySession
    {
        public List<ReplayGoal> goals = new List<ReplayGoal>();
    }

    [Serializable]
    private class ReplayGoal
    {
        public int goalIndex;
        public float goalX;
        public float goalY;
        public float goalZ;
    }
}
