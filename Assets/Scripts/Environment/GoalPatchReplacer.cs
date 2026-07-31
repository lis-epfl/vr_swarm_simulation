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

        // Choose which tiles to replace, then run the shared placement loop below.
        List<Transform> selected = mode == PlacementMode.Replay
            ? SelectReplayTiles(tiles)
            : SelectRandomTiles(tiles);

        if (selected == null || selected.Count == 0)
        {
            Debug.LogWarning("GoalPatchReplacer: no tiles selected; nothing replaced.", this);
            return;
        }

        foreach (Transform tile in selected)
        {
            GameObject goal = Instantiate(goalPrefab, cityPack);
            goal.name = $"goal_patch ({tile.name})";

            // Copy X/Z (and rotation/scale) from the replaced tile, but pin Y to ground level so
            // goals sit at y=0 even if the replaced patch had a non-zero height.
            Vector3 pos = tile.localPosition;
            pos.y = 0f;
            goal.transform.localPosition = pos;
            goal.transform.localRotation = tile.localRotation;
            goal.transform.localScale = tile.localScale;

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

    // ------------------------------------------------------------------ random selection

    /// <summary>Randomly pick up to <see cref="replaceCount"/> non-adjacent tiles.</summary>
    private List<Transform> SelectRandomTiles(List<Transform> tiles)
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
        float thresholdSq = 0f;
        if (preventAdjacent && tiles.Count > 1)
        {
            float minSq = float.MaxValue;
            for (int a = 0; a < tiles.Count; a++)
            {
                for (int b = a + 1; b < tiles.Count; b++)
                {
                    float d = SqrDistanceXZ(tiles[a].localPosition, tiles[b].localPosition);
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
                    if (SqrDistanceXZ(candidate.localPosition, chosen.localPosition) <= thresholdSq)
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
    /// pick the tile nearest that position (in cityPack-local XZ). Preserves the recorded order so the
    /// resulting <see cref="PlacedGoals"/> — and thus each goalIndex — matches the original run.
    /// </summary>
    private List<Transform> SelectReplayTiles(List<Transform> tiles)
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
                float d = SqrDistanceXZ(local, t.localPosition);
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

            // ~1 unit tolerance: an exact replay lands on the tile centre. A large gap means the scene
            // layout differs from the recorded run (different city/tile set) — replay is only approximate.
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
