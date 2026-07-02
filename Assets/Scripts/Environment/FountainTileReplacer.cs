using System.Collections.Generic;
using UnityEngine;

/// <summary>
/// Randomly replaces <c>n</c> of the <c>MC_Patch_*</c> city tiles under <see cref="cityPack"/>
/// with instances of a fountain tile prefab. Runs once at play-mode <see cref="Start"/>, so each
/// play session produces a different city layout; the change is not persisted to the scene asset.
///
/// The 48 tiles are direct children of <c>City_Pack_01</c> named <c>MC_Patch_01</c>…<c>MC_Patch_48</c>.
/// A name-prefix filter cleanly separates them from the other (tree/road/ground) children.
/// </summary>
public class FountainTileReplacer : MonoBehaviour
{
    [Tooltip("The City_Pack_01 root whose MC_Patch_* children get replaced. Defaults to this transform if unset.")]
    [SerializeField] private Transform cityPack;

    [Tooltip("The fountain tile prefab instantiated in place of each selected MC_Patch tile.")]
    [SerializeField] private GameObject fountainPrefab;

    [Tooltip("How many tiles to replace. Clamped to the number of available MC_Patch tiles.")]
    [SerializeField] private int replaceCount = 3;

    [Tooltip("Name prefix identifying replaceable tiles among the City_Pack children.")]
    [SerializeField] private string tilePrefix = "MC_Patch";

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

        if (fountainPrefab == null)
        {
            Debug.LogError("FountainTileReplacer: fountainPrefab is not assigned; no tiles replaced.", this);
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
            Debug.LogWarning($"FountainTileReplacer: nothing to replace (found {tiles.Count} tiles, replaceCount clamped to 0).", this);
            return;
        }

        // Partial Fisher-Yates shuffle: the first 'count' entries become a random selection without replacement.
        for (int i = 0; i < count; i++)
        {
            int j = Random.Range(i, tiles.Count);
            (tiles[i], tiles[j]) = (tiles[j], tiles[i]);
        }

        for (int i = 0; i < count; i++)
        {
            Transform tile = tiles[i];

            GameObject fountain = Instantiate(fountainPrefab, cityPack);
            fountain.name = $"fountain_tile ({tile.name})";

            // Copy X/Z (and rotation/scale) from the replaced tile, but pin Y to ground level so
            // fountains sit at y=0 even if the replaced patch had a non-zero height.
            Vector3 pos = tile.localPosition;
            pos.y = 0f;
            fountain.transform.localPosition = pos;
            fountain.transform.localRotation = tile.localRotation;
            fountain.transform.localScale = tile.localScale;

            if (destroyOriginal)
            {
                Destroy(tile.gameObject);
            }
            else
            {
                tile.gameObject.SetActive(false);
            }
        }

        Debug.Log($"FountainTileReplacer: replaced {count} of {tiles.Count} '{tilePrefix}' tiles with fountains.", this);
    }
}
