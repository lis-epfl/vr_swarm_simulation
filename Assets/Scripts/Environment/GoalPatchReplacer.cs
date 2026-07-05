using System.Collections.Generic;
using UnityEngine;

/// <summary>
/// Randomly replaces <c>n</c> of the <c>MC_Patch_*</c> city tiles under <see cref="cityPack"/>
/// with instances of a goal patch prefab. Runs once at play-mode <see cref="Start"/>, so each
/// play session produces a different city layout; the change is not persisted to the scene asset.
///
/// The 48 tiles are direct children of <c>City_Pack_01</c> named <c>MC_Patch_01</c>…<c>MC_Patch_48</c>.
/// A name-prefix filter cleanly separates them from the other (tree/road/ground) children.
/// </summary>
public class GoalPatchReplacer : MonoBehaviour
{
    [Tooltip("The City_Pack_01 root whose MC_Patch_* children get replaced. Defaults to this transform if unset.")]
    [SerializeField] private Transform cityPack;

    [Tooltip("The goal patch prefab instantiated in place of each selected MC_Patch tile.")]
    [SerializeField] private GameObject goalPrefab;

    [Tooltip("How many tiles to replace. Clamped to the number of available MC_Patch tiles.")]
    [SerializeField] private int replaceCount = 3;

    [Tooltip("Name prefix identifying replaceable tiles among the City_Pack children.")]
    [SerializeField] private string tilePrefix = "MC_Patch";

    [Header("Spacing")]
    [Tooltip("If true, no two goals may occupy adjacent grid tiles.")]
    [SerializeField] private bool preventAdjacent = true;

    [Tooltip("If true, diagonal neighbours also count as adjacent (blocks all 8 surrounding tiles); otherwise only the 4 edge neighbours are blocked.")]
    [SerializeField] private bool blockDiagonalNeighbors = true;

    [Header("Randomization")]
    [Tooltip("If true, seed the RNG with 'seed' for a reproducible selection each play.")]
    [SerializeField] private bool useFixedSeed = false;
    [SerializeField] private int seed = 0;

    [Tooltip("If true, Destroy the replaced tile; otherwise just deactivate it (reversible).")]
    [SerializeField] private bool destroyOriginal = true;

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

        if (useFixedSeed)
        {
            Random.InitState(seed);
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

        int count = Mathf.Clamp(replaceCount, 0, tiles.Count);
        if (count == 0)
        {
            Debug.LogWarning($"GoalPatchReplacer: nothing to replace (found {tiles.Count} tiles, replaceCount clamped to 0).", this);
            return;
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
            int j = Random.Range(0, i + 1);
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

            if (destroyOriginal)
            {
                Destroy(tile.gameObject);
            }
            else
            {
                tile.gameObject.SetActive(false);
            }
        }

        if (selected.Count < count)
        {
            Debug.LogWarning($"GoalPatchReplacer: could only place {selected.Count} of {count} requested goals without adjacency; the non-adjacency constraint left no more valid tiles.", this);
        }

        Debug.Log($"GoalPatchReplacer: replaced {selected.Count} of {tiles.Count} '{tilePrefix}' tiles with goals.", this);
    }

    /// <summary>Squared horizontal (XZ) distance, ignoring Y so tiles at different heights still compare by grid cell.</summary>
    private static float SqrDistanceXZ(Vector3 a, Vector3 b)
    {
        float dx = a.x - b.x;
        float dz = a.z - b.z;
        return dx * dx + dz * dz;
    }
}
