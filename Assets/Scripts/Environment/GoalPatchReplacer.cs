using System;
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
/// <see cref="GoalPositions"/>), and takes over any scenery tied to that tile (see
/// <see cref="CarryTiedScenery"/>).
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

        if (tiles.Count == 0)
        {
            Debug.LogWarning($"GoalPatchReplacer: no '{tilePrefix}' tiles found under {cityPack.name}; nothing to replace.", this);
            return;
        }

        // Where a goal for each tile would stand. Selection compares these for every pair of tiles and replay
        // for every recorded goal, so they are worked out once, up front.
        Dictionary<Transform, Vector3> goalPositions = GoalPositions(tiles);

        // Choose which tiles to replace, then run the shared placement loop below.
        List<Transform> selected = mode == PlacementMode.Replay
            ? SelectReplayTiles(tiles, goalPositions)
            : SelectRandomTiles(tiles, goalPositions);

        if (selected == null || selected.Count == 0)
        {
            Debug.LogWarning("GoalPatchReplacer: no tiles selected; nothing replaced.", this);
            return;
        }

        foreach (Transform tile in selected)
        {
            GameObject goal = Instantiate(goalPrefab, cityPack);
            goal.name = $"goal_patch ({tile.name})";

            goal.transform.localPosition = goalPositions[tile];
            goal.transform.localRotation = tile.localRotation;
            goal.transform.localScale = tile.localScale;

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

        Debug.Log($"GoalPatchReplacer: replaced {selected.Count} of {tiles.Count} '{tilePrefix}' tiles with goals ({mode}).", this);
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
    /// </summary>
    private Dictionary<Transform, Vector3> GoalPositions(List<Transform> tiles)
    {
        // The goal's block relative to its own root, measured on the asset so no instance is needed.
        Transform goalKerb = CityTiles.FindKerb(goalPrefab.transform);
        Vector3 goalBlockOffset = goalKerb != null
            ? goalPrefab.transform.InverseTransformPoint(goalKerb.position)
            : Vector3.zero;

        Dictionary<Transform, Vector3> positions = new Dictionary<Transform, Vector3>(tiles.Count);
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
        return positions;
    }

    /// <summary>
    /// Hand the scenery tied to <paramref name="tile"/> (every <see cref="CityTiles.TiedContainers"/> child)
    /// over to the goal replacing it, keeping its world placement. Untied, the street trees and plates hang off
    /// the city root and survive the replacement; tied, they would be destroyed with the tile, leaving a gap in
    /// the medians and verges exactly where each goal is — a cue visible from the air. A tile that was never
    /// tied has nothing to carry.
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

    // ------------------------------------------------------------------ random selection

    /// <summary>Randomly pick up to <see cref="replaceCount"/> non-adjacent tiles.</summary>
    private List<Transform> SelectRandomTiles(List<Transform> tiles, Dictionary<Transform, Vector3> goalPositions)
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
        float thresholdSq = 0f;
        if (preventAdjacent && tiles.Count > 1)
        {
            float minSq = float.MaxValue;
            for (int a = 0; a < tiles.Count; a++)
            {
                for (int b = a + 1; b < tiles.Count; b++)
                {
                    float d = SqrDistanceXZ(goalPositions[tiles[a]], goalPositions[tiles[b]]);
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
