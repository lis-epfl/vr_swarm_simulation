using System;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;
#if UNITY_EDITOR
using UnityEditor;
using UnityEditor.SceneManagement;
#endif

/// <summary>
/// Plants trees on the grass verge that <see cref="StreetWidthTuner"/> opens up between a block's footpath and the
/// road, on a chosen subset of the city's tiles.
///
/// <para><b>Where the verge is.</b> Shrinking a block pulls its footpath edge in to
/// <c>BlockHalfSpan x blockScale</c> from the tile's centre, while the road ring's inner edge stays at
/// <see cref="CityTiles.BlockHalfSpan"/>. No city geometry covers the band between the two (7.6 units wide at
/// 0.8), so it shows the ground under the city — a grass Terrain in the city scenes. Trees go down
/// the middle of that ring on the chosen sides, at a fixed spacing, stopping <see cref="CornerClearance"/> short of
/// each corner so junctions stay open. A spot is skipped when something already on the tile — a lamp, a bus stop,
/// the mouth of the block's interior street — comes within <see cref="Clearance"/> of it.</para>
///
/// <para><b>They stay on the verge.</b> Each tree hangs under its tile's <see cref="CityTiles.VergeContainer"/>,
/// which the tuner scales by half the block's shrink, keeping the row in the middle of the verge if the block scale
/// is changed later. Because the container belongs to the tile, the trees also move with it
/// (<see cref="CityRowOffsetter"/>) and pass to a goal patch that replaces it (<see cref="GoalPatchReplacer"/>).
/// Planting always starts from a cleared verge, so hand edits to planted trees do not survive a replant.</para>
///
/// <para><b>Scripting.</b> Every inspector setting is also a property, and <see cref="Plant()"/>,
/// <see cref="Plant(IEnumerable{Transform})"/> and <see cref="Clear()"/> are public, so an experiment script can pick
/// tiles and plant. The result is deterministic: the same seed, settings and tile names give the same trees,
/// whatever the hierarchy order and wherever the tiles have been moved. Prefer planting in edit mode: trees planted
/// there are marked Batching Static like the rest of the city, whereas trees planted during Play stay ordinary
/// dynamic objects, one draw call per mesh per eye.</para>
/// </summary>
[DisallowMultipleComponent]
public class VergeTreePlanter : MonoBehaviour
{
    public enum TileChoice
    {
        RandomFraction, // tileFraction of the city's tiles, picked by seed
        Listed,         // exactly listedTiles
        All,
    }

    [Flags]
    public enum Sides
    {
        None = 0,
        North = 1 << 0, // the tile's +Z (forward) side
        East = 1 << 1,  // its +X (right) side
        South = 1 << 2,
        West = 1 << 3,
        All = North | East | South | West,
    }

    private const string k_UndoName = "Plant verge trees";
    private const string k_DefaultTreePath = "Assets/Modular City Pack/Prefabs/Props/MC_Tree.prefab";

    /// <summary>How close to the footpath edge an interior street must end to count as opening onto the verge.</summary>
    private const float k_MouthReach = 2f;

    [Tooltip("Prefabs to plant, one picked at random per tree. Filled with the pack's MC_Tree when the component is " +
             "added: the same trunk and canopy meshes as the city's own street trees.")]
    [SerializeField] private GameObject[] treePrefabs = new GameObject[0];

    [Header("Which tiles")]
    [SerializeField] private TileChoice tileChoice = TileChoice.RandomFraction;

    [Tooltip("RandomFraction only: the share of the city's tiles that get trees.")]
    [Range(0f, 1f)]
    [SerializeField] private float tileFraction = 0.25f;

    [Tooltip("Listed only: the tiles (the MC_Patch roots under the city) that get trees.")]
    [SerializeField] private List<Transform> listedTiles = new List<Transform>();

    [Header("Where on each tile")]
    [SerializeField] private Sides sides = Sides.All;

    [Tooltip("Distance between neighbouring trees along a side, in world units.")]
    [Min(1f)]
    [SerializeField] private float spacing = 9f;

    [Tooltip("How far short of each corner of the block a row of trees stops, so junctions stay open.")]
    [Min(0f)]
    [SerializeField] private float cornerClearance = 8f;

    [Tooltip("A spot is skipped if anything already on the tile (a lamp, a bus stop, an interior street's mouth) " +
             "comes within this distance of it.")]
    [Min(0f)]
    [SerializeField] private float clearance = 1.5f;

    [Header("Variation")]
    [Tooltip("Random shift along the side, as a fraction of the spacing either way.")]
    [Range(0f, 0.5f)]
    [SerializeField] private float alongJitter = 0.15f;

    [Tooltip("Random turn about the vertical, in degrees either way.")]
    [Range(0f, 180f)]
    [SerializeField] private float yawJitter = 180f;

    [Tooltip("Random size change, as a fraction either way.")]
    [Range(0f, 0.5f)]
    [SerializeField] private float scaleJitter = 0.15f;

    [SerializeField] private int seed = 1;

    [Header("Scope")]
    [Tooltip("Optional: the city root to plant in. Empty uses the city in this component's scene.")]
    [SerializeField] private Transform cityRoot;

    [Tooltip("What a tree may stand right beside: the ground a verge is made of. Everything else on a tile is an obstacle.")]
    [SerializeField] private string[] groundNamePrefixes = { "FootPath_", "Footpath_", "Carbs_", "Road_Structure_" };

    [Tooltip("A block's interior streets. Where one meets a side of the block, its mouth is kept clear right across " +
             "the verge rather than only up to the footpath edge.")]
    [SerializeField] private string[] interiorStreetNamePrefixes = { "Roads_Street_" };

    public GameObject[] TreePrefabs
    {
        get => treePrefabs;
        set => treePrefabs = value ?? new GameObject[0];
    }

    public TileChoice Choice
    {
        get => tileChoice;
        set => tileChoice = value;
    }

    public float TileFraction
    {
        get => tileFraction;
        set => tileFraction = Mathf.Clamp01(value);
    }

    /// <summary>The tiles <see cref="TileChoice.Listed"/> plants on; edit the list in place.</summary>
    public List<Transform> ListedTiles => listedTiles;

    public Sides SidesToPlant
    {
        get => sides;
        set => sides = value;
    }

    public float Spacing
    {
        get => spacing;
        set => spacing = Mathf.Max(1f, value);
    }

    public float CornerClearance
    {
        get => cornerClearance;
        set => cornerClearance = Mathf.Max(0f, value);
    }

    public float Clearance
    {
        get => clearance;
        set => clearance = Mathf.Max(0f, value);
    }

    public float AlongJitter
    {
        get => alongJitter;
        set => alongJitter = Mathf.Clamp(value, 0f, 0.5f);
    }

    public float YawJitter
    {
        get => yawJitter;
        set => yawJitter = Mathf.Clamp(value, 0f, 180f);
    }

    public float ScaleJitter
    {
        get => scaleJitter;
        set => scaleJitter = Mathf.Clamp(value, 0f, 0.5f);
    }

    public int Seed
    {
        get => seed;
        set => seed = value;
    }

#if UNITY_EDITOR
    private void Reset()
    {
        GameObject tree = AssetDatabase.LoadAssetAtPath<GameObject>(k_DefaultTreePath);
        if (tree != null)
        {
            treePrefabs = new[] { tree };
        }
    }
#endif

    /// <summary>
    /// Clear every tile's verge, then plant on the tiles <see cref="Choice"/> picks. Returns the number of trees.
    /// </summary>
    [ContextMenu("Plant verge trees")]
    public int Plant()
    {
        CityTiles.City city = FindCity();
        if (city == null || !CanPlant(out float blockScale))
        {
            return 0;
        }

        int undoGroup = BeginUndo();
        foreach (Transform tile in city.Tiles)
        {
            ClearTile(tile);
        }
        List<int> chosen = ChooseTiles(city);
        int planted = PlantTiles(city, chosen, blockScale);
        EndUndo(undoGroup, rescan: true);
        return planted;
    }

    /// <summary>
    /// Plant on exactly <paramref name="tiles"/>, replacing their verge trees and leaving every other tile's alone.
    /// Returns the number of trees.
    /// </summary>
    public int Plant(IEnumerable<Transform> tiles)
    {
        CityTiles.City city = FindCity();
        if (city == null || !CanPlant(out float blockScale))
        {
            return 0;
        }

        int undoGroup = BeginUndo();
        List<int> indices = IndicesOf(city, tiles);
        foreach (int i in indices)
        {
            ClearTile(city.Tiles[i]);
        }
        int planted = PlantTiles(city, indices, blockScale);
        EndUndo(undoGroup, rescan: true);
        return planted;
    }

    /// <summary>Remove the verge trees from every tile. Returns the number of tiles that had any.</summary>
    [ContextMenu("Clear verge trees")]
    public int Clear()
    {
        CityTiles.City city = FindCity();
        return city != null ? Clear(city.Tiles) : 0;
    }

    /// <summary>Remove the verge trees from <paramref name="tiles"/>. Returns the number of tiles that had any.</summary>
    public int Clear(IEnumerable<Transform> tiles)
    {
        int undoGroup = BeginUndo();
        int cleared = 0;
        foreach (Transform tile in tiles)
        {
            if (tile != null && ClearTile(tile))
            {
                cleared++;
            }
        }
        EndUndo(undoGroup, rescan: false); // the tuner skips entries whose object is gone
        return cleared;
    }

    /// <summary>The tiles <see cref="Plant()"/> would plant on with the current settings.</summary>
    public List<Transform> ChosenTiles()
    {
        CityTiles.City city = FindCity();
        return city != null ? ChooseTiles(city).Select(i => city.Tiles[i]).ToList() : new List<Transform>();
    }

    private CityTiles.City FindCity()
    {
        string error;
        CityTiles.City city = cityRoot != null
            ? CityTiles.FindCity(cityRoot, out error)
            : CityTiles.FindCity(gameObject.scene, out error);
        if (city == null)
        {
            Debug.LogError("VergeTreePlanter: " + error, this);
        }
        return city;
    }

    /// <summary>Whether there is anything to plant, and anywhere to plant it: <paramref name="blockScale"/> below 1.</summary>
    private bool CanPlant(out float blockScale)
    {
        blockScale = 1f;
        if (treePrefabs == null || !treePrefabs.Any(p => p != null))
        {
            Debug.LogError("VergeTreePlanter: no tree prefabs assigned. MC_Tree from the Modular City Pack matches the " +
                           "city's own street trees.", this);
            return false;
        }

        StreetWidthTuner tuner = Tuner();
        if (tuner != null)
        {
            blockScale = tuner.AppliedBlockScale;
        }
        if (blockScale >= 1f)
        {
            Debug.LogWarning($"VergeTreePlanter: the block scale is {blockScale:F2}, so the footpath still reaches the " +
                             "road edge and there is no verge to plant on. Lower StreetWidthTuner.blockScale first.", this);
            return false;
        }
        if (tuner != null && !tuner.MovesLooseScenery)
        {
            Debug.LogWarning("VergeTreePlanter: StreetWidthTuner's moveLooseScenery is off, so these trees will not stay " +
                             "on the verge if the block scale is changed later.", this);
        }
        return true;
    }

    private StreetWidthTuner Tuner()
    {
        foreach (StreetWidthTuner tuner in FindObjectsByType<StreetWidthTuner>(FindObjectsInactive.Include,
                                                                              FindObjectsSortMode.None))
        {
            if (tuner.gameObject.scene == gameObject.scene)
            {
                return tuner;
            }
        }
        return null;
    }

    private List<int> ChooseTiles(CityTiles.City city)
    {
        switch (tileChoice)
        {
            case TileChoice.All:
                return Enumerable.Range(0, city.Tiles.Count).ToList();

            case TileChoice.Listed:
                return IndicesOf(city, listedTiles);

            default:
                // Ordered by name before shuffling, so the pick depends on the seed and the tile names alone — not on
                // hierarchy order, nor on where a tile has since been moved.
                List<int> order = Enumerable.Range(0, city.Tiles.Count)
                                            .OrderBy(i => city.Tiles[i].name, StringComparer.Ordinal)
                                            .ToList();
                System.Random rng = new System.Random(seed);
                for (int i = order.Count - 1; i > 0; i--)
                {
                    int j = rng.Next(i + 1);
                    (order[i], order[j]) = (order[j], order[i]);
                }
                return order.Take(Mathf.RoundToInt(tileFraction * order.Count)).ToList();
        }
    }

    private List<int> IndicesOf(CityTiles.City city, IEnumerable<Transform> tiles)
    {
        List<int> indices = new List<int>();
        int strays = 0;
        foreach (Transform tile in tiles.Where(t => t != null).Distinct())
        {
            int i = city.Tiles.IndexOf(tile);
            if (i >= 0)
            {
                indices.Add(i);
            }
            else
            {
                strays++;
            }
        }
        if (strays > 0)
        {
            Debug.LogWarning($"VergeTreePlanter: {strays} of the given objects are not tiles of {city.Root.name} — a tile is " +
                             "a direct child of the city root holding a kerb — and were skipped.", this);
        }
        return indices;
    }

    private int PlantTiles(CityTiles.City city, List<int> indices, float blockScale)
    {
        int planted = 0;
        int skipped = 0;
        foreach (int i in indices)
        {
            planted += PlantTile(city, i, blockScale, ref skipped);
        }
        Debug.Log($"VergeTreePlanter: planted {planted} trees on {indices.Count} tiles at block scale {blockScale:F2} " +
                  $"({skipped} spots left empty for clearance).", this);
        return planted;
    }

    private int PlantTile(CityTiles.City city, int index, float blockScale, ref int skipped)
    {
        Transform tile = city.Tiles[index];
        Vector3 centre = city.Centres[index];

        // The middle of the verge, and how far either way along a side the row may run before the corner.
        float footpathEdge = CityTiles.BlockHalfSpan * blockScale;
        float middle = CityTiles.BlockHalfSpan * CityTiles.ScaleFor(blockScale, CityTiles.BlockShare(CityTiles.VergeContainer));
        float reach = middle - cornerClearance;
        if (reach < 0f)
        {
            return 0;
        }

        Vector3 right = Flat(tile.right);
        Vector3 forward = Flat(tile.forward);
        List<Rect> obstacles = Obstacles(tile, centre, right, forward, footpathEdge);

        // One generator per tile, seeded by its name, so a tile's trees do not depend on which other tiles are planted.
        System.Random rng = new System.Random(unchecked(seed * 486187739 + StableHash(tile.name)));
        int count = Mathf.FloorToInt(2f * reach / spacing) + 1;
        Transform verge = null;
        int planted = 0;

        for (int s = 0; s < 4; s++)
        {
            Sides side = (Sides)(1 << s);
            Vector2 outward = side == Sides.North ? Vector2.up
                            : side == Sides.East ? Vector2.right
                            : side == Sides.South ? Vector2.down
                            : Vector2.left;
            Vector2 along = new Vector2(outward.y, -outward.x);

            for (int k = 0; k < count; k++)
            {
                // Every spot draws its numbers whether or not it is planted, so switching a side off or skipping one
                // spot does not reshuffle the trees everywhere else.
                float jitter = Range(rng, -alongJitter, alongJitter) * spacing;
                float yaw = Range(rng, -yawJitter, yawJitter);
                float size = 1f + Range(rng, -scaleJitter, scaleJitter);
                GameObject prefab = treePrefabs[rng.Next(treePrefabs.Length)];
                if ((sides & side) == 0 || prefab == null)
                {
                    continue;
                }

                float u = Mathf.Clamp((k - (count - 1) / 2f) * spacing + jitter, -reach, reach);
                Vector2 spot = outward * middle + along * u;
                if (Blocked(spot, obstacles))
                {
                    skipped++;
                    continue;
                }

                if (verge == null)
                {
                    verge = CreateVerge(tile);
                }
                PlantTree(prefab, verge, centre + right * spot.x + forward * spot.y, yaw, size);
                planted++;
            }
        }
        return planted;
    }

    /// <summary>
    /// Footprints of everything already on the tile, as rectangles in its frame (x along <paramref name="right"/>, y
    /// along <paramref name="forward"/>, about its centre). An interior street reaching a side of the block is
    /// stretched along its length out to the road edge, so its mouth stays clear across the verge.
    /// </summary>
    private List<Rect> Obstacles(Transform tile, Vector3 centre, Vector3 right, Vector3 forward, float footpathEdge)
    {
        List<Rect> rects = new List<Rect>();
        Transform verge = tile.Find(CityTiles.VergeContainer);
        foreach (Renderer renderer in tile.GetComponentsInChildren<Renderer>())
        {
            if (!renderer.enabled
                || (verge != null && renderer.transform.IsChildOf(verge))
                || HasAnyPrefix(renderer.name, groundNamePrefixes))
            {
                continue;
            }

            Rect rect = TileRect(renderer.bounds, centre, right, forward);
            if (HasAnyPrefix(renderer.name, interiorStreetNamePrefixes))
            {
                float roadEdge = CityTiles.BlockHalfSpan;
                if (rect.height >= rect.width)
                {
                    if (rect.yMax >= footpathEdge - k_MouthReach) { rect.yMax = Mathf.Max(rect.yMax, roadEdge); }
                    if (rect.yMin <= -footpathEdge + k_MouthReach) { rect.yMin = Mathf.Min(rect.yMin, -roadEdge); }
                }
                else
                {
                    if (rect.xMax >= footpathEdge - k_MouthReach) { rect.xMax = Mathf.Max(rect.xMax, roadEdge); }
                    if (rect.xMin <= -footpathEdge + k_MouthReach) { rect.xMin = Mathf.Min(rect.xMin, -roadEdge); }
                }
            }
            rects.Add(rect);
        }
        return rects;
    }

    private bool Blocked(Vector2 spot, List<Rect> obstacles)
    {
        foreach (Rect rect in obstacles)
        {
            float dx = Mathf.Max(rect.xMin - spot.x, 0f, spot.x - rect.xMax);
            float dy = Mathf.Max(rect.yMin - spot.y, 0f, spot.y - rect.yMax);
            if (dx * dx + dy * dy <= clearance * clearance)
            {
                return true;
            }
        }
        return false;
    }

    /// <summary>The horizontal footprint of world-space <paramref name="bounds"/> in the tile's frame.</summary>
    private static Rect TileRect(Bounds bounds, Vector3 centre, Vector3 right, Vector3 forward)
    {
        float xMin = float.MaxValue, xMax = float.MinValue, yMin = float.MaxValue, yMax = float.MinValue;
        for (int corner = 0; corner < 4; corner++)
        {
            Vector3 p = new Vector3((corner & 1) == 0 ? bounds.min.x : bounds.max.x, 0f,
                                    (corner & 2) == 0 ? bounds.min.z : bounds.max.z);
            Vector3 d = p - new Vector3(centre.x, 0f, centre.z);
            float x = Vector3.Dot(d, right);
            float y = Vector3.Dot(d, forward);
            xMin = Mathf.Min(xMin, x);
            xMax = Mathf.Max(xMax, x);
            yMin = Mathf.Min(yMin, y);
            yMax = Mathf.Max(yMax, y);
        }
        return Rect.MinMaxRect(xMin, yMin, xMax, yMax);
    }

    private static Transform CreateVerge(Transform tile)
    {
        GameObject go = new GameObject(CityTiles.VergeContainer);
        go.layer = tile.gameObject.layer;
        Transform verge = go.transform;
        verge.SetParent(tile, false);
#if UNITY_EDITOR
        if (!Application.isPlaying)
        {
            Undo.RegisterCreatedObjectUndo(go, k_UndoName);
        }
#endif
        return verge;
    }

    private static void PlantTree(GameObject prefab, Transform verge, Vector3 position, float yaw, float size)
    {
        GameObject tree = null;
#if UNITY_EDITOR
        if (!Application.isPlaying && PrefabUtility.IsPartOfPrefabAsset(prefab))
        {
            tree = (GameObject)PrefabUtility.InstantiatePrefab(prefab, verge);
        }
#endif
        if (tree == null)
        {
            tree = Instantiate(prefab, verge);
        }

        Transform t = tree.transform;
        t.SetPositionAndRotation(position, Quaternion.AngleAxis(yaw, Vector3.up) * prefab.transform.rotation);
        t.localScale = prefab.transform.localScale * size;

#if UNITY_EDITOR
        if (!Application.isPlaying)
        {
            // Batched with the rest of the city once Play starts. Unbatched, every tree is two draw calls per eye.
            foreach (Transform part in tree.GetComponentsInChildren<Transform>(true))
            {
                GameObjectUtility.SetStaticEditorFlags(part.gameObject,
                    GameObjectUtility.GetStaticEditorFlags(part.gameObject) | StaticEditorFlags.BatchingStatic);
            }
            Undo.RegisterCreatedObjectUndo(tree, k_UndoName);
        }
#endif
    }

    private static bool ClearTile(Transform tile)
    {
        Transform verge = tile.Find(CityTiles.VergeContainer);
        if (verge == null)
        {
            return false;
        }
#if UNITY_EDITOR
        if (!Application.isPlaying)
        {
            Undo.DestroyObjectImmediate(verge.gameObject);
            return true;
        }
#endif
        // Destroy only takes effect at the end of the frame; unparented first, a replant this frame cannot find it.
        verge.SetParent(null, true);
        Destroy(verge.gameObject);
        return true;
    }

    private static int BeginUndo()
    {
#if UNITY_EDITOR
        if (!Application.isPlaying)
        {
            Undo.IncrementCurrentGroup();
            Undo.SetCurrentGroupName(k_UndoName);
            return Undo.GetCurrentGroup();
        }
#endif
        return -1;
    }

    private void EndUndo(int undoGroup, bool rescan)
    {
#if UNITY_EDITOR
        if (!Application.isPlaying)
        {
            Undo.CollapseUndoOperations(undoGroup);
            EditorSceneManager.MarkSceneDirty(gameObject.scene);

            // The tuner only moves what its last scan found, so new trees join its list now, with the verge's share,
            // rather than staying put the next time the block scale changes.
            StreetWidthTuner tuner = rescan ? Tuner() : null;
            if (tuner != null)
            {
                tuner.Rescan();
            }
        }
#endif
    }

    private static Vector3 Flat(Vector3 v)
    {
        v.y = 0f;
        return v.normalized;
    }

    private static float Range(System.Random rng, float min, float max)
    {
        return min + (float)rng.NextDouble() * (max - min);
    }

    /// <summary>FNV-1a. <see cref="string.GetHashCode()"/> is not guaranteed to be the same between runs.</summary>
    private static int StableHash(string text)
    {
        unchecked
        {
            uint hash = 2166136261;
            foreach (char c in text)
            {
                hash = (hash ^ c) * 16777619;
            }
            return (int)hash;
        }
    }

    private static bool HasAnyPrefix(string objectName, string[] prefixes)
    {
        if (prefixes == null)
        {
            return false;
        }
        foreach (string prefix in prefixes)
        {
            if (!string.IsNullOrEmpty(prefix) && objectName.StartsWith(prefix))
            {
                return true;
            }
        }
        return false;
    }
}
