using System.Collections.Generic;
using System.Linq;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;
using UnityEngine.SceneManagement;

/// <summary>
/// Ties the city's loose scenery to the tiles it stands on, so that moving a tile moves everything on it — and
/// puts the street medians the street-width tuning dragged onto a verge back in the middle of the street.
///
/// <para><b>What is loose.</b> The pack's <c>MC_Patch</c> tile prefabs hold only a block — buildings, footpath,
/// kerb, street furniture and one asphalt plate. Everything else in <c>City_Pack_01</c> (and the ScaledCity
/// fork) sits straight under the city root: twelve <c>Green_Belt_Tile</c> groups, whose 96 grass strips and 672
/// trees stand on the lines between tiles, down the middle of every street; a <c>Garden</c>; six stray trees;
/// and 48 more <c>Road_Structure</c> plates, each within a unit of the plate already inside a tile. Move a tile
/// and all of that stays behind.</para>
///
/// <para><b>Where each object goes is decided from where it was authored</b> — its position before
/// <see cref="StreetWidthTuner"/> shrank the blocks, recovered by dividing that scaling back out about the
/// centre it was applied around (for the medians this reproduces the pack's positions to 1e-4 units):</para>
/// <list type="bullet">
/// <item><b>In the street, or on a block?</b> Beyond the kerb line (<see cref="CityTiles.BlockHalfSpan"/> from
/// its tile's centre) is street. In the pack that separates cleanly: every median object is 44.9–45.8 units
/// out, everything else at most 34.8. Street scenery goes under <see cref="CityTiles.StreetContainer"/> and is
/// put back where it was authored. Untied, the tuner scaled it towards whichever tile centre was nearer, which
/// walked each median ~9 units onto one side's verge at a block scale of 0.8. Block scenery goes under
/// <see cref="CityTiles.SceneryContainer"/> and keeps the position the tuning gave it.</item>
/// <item><b>Which tile?</b> Block scenery: the tile whose square footprint holds its mesh centre — not its
/// pivot, because every road plate is pivoted on a tile corner. Street scenery stands on the line two
/// footprints share, where that test is a coin toss, so it goes to the tile on the line's west side (south,
/// for a line running east–west). A median strip and its trees therefore always stay together, and each tile
/// owns the medians along its east and north edges.</item>
/// </list>
///
/// <para><b>A group that straddles tiles is split, not dragged whole onto one of them.</b> A group node that is
/// nothing but a Transform is recreated under each tile it covers, with the same name and world transform, and
/// its children are moved into the copies, so each child keeps its local values exactly. A node with components
/// of its own cannot be separated from its children; it moves whole to where most of it belongs, and the report
/// names it.</para>
///
/// <para><b>It edits the scene, never a prefab.</b> Unity will not reparent an object that is part of a prefab
/// instance, and the city is one, so its root is unpacked in the scene first — the outermost layer only, so
/// every tile stays a linked instance of its patch prefab. Doing the same inside the city prefab would need no
/// unpack but would break the scene: its position overrides on this scenery (thousands, written by the tuner)
/// are relative to the city root, and would be re-read relative to each tile.</para>
///
/// <para><b>It can be run again.</b> Scenery already under a tile's Scenery or Street container is re-sorted by
/// the same rules, so a second run changes nothing, and a scene tied before street scenery had its own container
/// is corrected in place. The <see cref="CityTiles.VergeContainer"/> belongs to <see cref="VergeTreePlanter"/>
/// and is never touched. The whole change is a single undo step.</para>
/// </summary>
public static class CitySceneryParenter
{
    private const string k_UndoName = "Tie city scenery to tiles";

    /// <summary>The containers this tool sorts scenery into, and re-sorts on a later run.</summary>
    private static readonly string[] k_SortedContainers = { CityTiles.SceneryContainer, CityTiles.StreetContainer };

    /// <summary>
    /// How far outside a footprint a street object may be and still count as on its edge: authored medians lie
    /// within a unit of the line, and tiles within a fraction of a unit of the grid.
    /// </summary>
    private const float k_LineTolerance = 1f;

    /// <summary>
    /// A block object whose runner-up tile is within this distance of its own is reported: which tile it lands on
    /// is decided by float noise.
    /// </summary>
    private const float k_AmbiguousMargin = 1f;

    /// <summary>Where one renderer belongs.</summary>
    private struct Placement
    {
        public int tile;
        public string container;
        public bool restore;          // street scenery the tuning has moved off its line
        public Vector3 authoredPivot; // world position it is restored to
    }

    /// <summary>A top-level object to sort, and where it hangs now (tile -1 and no container: loose).</summary>
    private struct Unit
    {
        public Transform node;
        public int tile;
        public string container;
    }

    private struct Move
    {
        public Transform transform;
        public int tile;
        public string container;
        public Transform[] groups; // split groups it came out of, outermost first; each is recreated at the target
    }

    private sealed class Plan
    {
        public readonly Dictionary<Renderer, Placement> placements = new Dictionary<Renderer, Placement>();
        public readonly List<Move> moves = new List<Move>();
        public readonly List<Transform> splitGroups = new List<Transform>(); // outermost first
        public readonly List<string> keptWhole = new List<string>();
        public int alreadyTied;
        public float smallestMargin = float.MaxValue;
        public string smallestMarginName;

        public int StreetCount => placements.Values.Count(p => p.container == CityTiles.StreetContainer);
        public int RestoreCount => placements.Values.Count(p => p.restore);
    }

    /// <summary>How <see cref="StreetWidthTuner"/> has scaled the scenery, so it can be divided back out.</summary>
    private sealed class Tuning
    {
        public StreetWidthTuner tuner; // null: nothing in the scene scales scenery

        /// <summary>The factor the tuner has scaled this renderer by, about the centre it scaled it around.</summary>
        public float AppliedFactor(Renderer renderer, string container)
        {
            if (tuner == null || !tuner.MovesLooseScenery || tuner.IgnoresObject(renderer.name))
            {
                return 1f;
            }
            float share = container == null ? 1f : CityTiles.BlockShare(container);
            return CityTiles.ScaleFor(tuner.AppliedBlockScale, share);
        }
    }

    [MenuItem("Tools/Swarm/Report city scenery ties")]
    public static void Report()
    {
        if (!Prepare(out CityTiles.City city, out Tuning tuning))
        {
            return;
        }

        List<Transform> loose = city.LooseChildren();
        Plan plan = MakePlan(city, tuning);
        Debug.Log($"CitySceneryParenter: {city.Tiles.Count} tiles under {city.Root.name}; {loose.Count} loose objects " +
                  "beside them" + (loose.Count > 0 ? $" ({Families(loose)})" : "") + $", {plan.alreadyTied} already tied.",
                  city.Root);
        Debug.Log($"  {plan.StreetCount} renderers stand in the street, {plan.placements.Count - plan.StreetCount} on a " +
                  $"block. {plan.RestoreCount} street renderers have been pulled off their line by the street-width " +
                  "tuning and would go back to the middle of the street.");

        if (plan.moves.Count == 0 && plan.RestoreCount == 0)
        {
            Debug.Log("  Nothing to do: all scenery is tied, and the medians are in the middle of the street.");
            return;
        }

        if (plan.moves.Count > 0)
        {
            int[] perTile = CountPerTile(city, plan);
            Debug.Log($"  Tying would move {plan.moves.Count} objects, onto {perTile.Count(n => n > 0)} tiles " +
                      $"({perTile.Min()}-{perTile.Max()} per tile), splitting {plan.splitGroups.Count} groups that " +
                      "straddle tiles.");
        }
        LogPlanWarnings(plan);
        if (plan.moves.Count > 0 && PrefabUtility.IsPartOfPrefabInstance(city.Root.gameObject))
        {
            Debug.Log($"  {PrefabUtility.GetOutermostPrefabInstanceRoot(city.Root.gameObject).name} is a prefab instance, " +
                      "so tying unpacks it in this scene first (outermost layer only; the tiles stay linked to their " +
                      "patch prefabs).");
        }
        Debug.Log("  Run Tools/Swarm/Tie city scenery to tiles to apply.");
    }

    [MenuItem("Tools/Swarm/Tie city scenery to tiles")]
    public static void Tie()
    {
        if (EditorApplication.isPlayingOrWillChangePlaymode)
        {
            Debug.LogError("CitySceneryParenter: exit Play mode first. This restructures the scene, and in Play the " +
                           "city is static-batched and its goal patches are runtime instances.");
            return;
        }
        if (!Prepare(out CityTiles.City city, out Tuning tuning))
        {
            return;
        }

        // A split group is recreated with the original's local scale, which is only the same world scale if the
        // tile it lands under is unscaled relative to the city root it came from.
        foreach (Transform tile in city.Tiles)
        {
            if ((tile.localScale - Vector3.one).sqrMagnitude > 1e-8f)
            {
                Debug.LogError($"CitySceneryParenter: {tile.name} is scaled ({tile.localScale}), so scenery moved under " +
                               "it could not keep its size exactly. Nothing changed.", tile);
                return;
            }
        }

        Plan preview = MakePlan(city, tuning);
        if (preview.moves.Count == 0 && preview.RestoreCount == 0)
        {
            Debug.Log("CitySceneryParenter: nothing to do — all scenery is tied, and the medians are in the middle of " +
                      "the street.", city.Root);
            return;
        }

        bool unpack = preview.moves.Count > 0 && PrefabUtility.IsPartOfPrefabInstance(city.Root.gameObject);
        string unpackNote = unpack
            ? $"\n\n{PrefabUtility.GetOutermostPrefabInstanceRoot(city.Root.gameObject).name} is a prefab instance, so it " +
              "is unpacked in this scene first. Only its outermost layer: every tile stays linked to its patch prefab, " +
              "and no prefab asset is modified."
            : "";
        if (!EditorUtility.DisplayDialog(k_UndoName,
                $"Move {preview.moves.Count} objects onto the {city.Tiles.Count} tiles they stand on, and put " +
                $"{preview.RestoreCount} street objects back in the middle of the street.{unpackNote}\n\n" +
                "One undo step reverts all of it.",
                "Tie", "Cancel"))
        {
            return;
        }

        Undo.IncrementCurrentGroup();
        int undoGroup = Undo.GetCurrentGroup();
        Undo.SetCurrentGroupName(k_UndoName);

        if (unpack)
        {
            // Unity refuses to reparent anything inside a prefab instance, so peel instance layers off the city
            // root until it is a plain object. The tiles below it stay instances of their own prefabs.
            for (int layer = 0; PrefabUtility.IsPartOfPrefabInstance(city.Root.gameObject); layer++)
            {
                if (layer == 8)
                {
                    Undo.CollapseUndoOperations(undoGroup);
                    Debug.LogError($"CitySceneryParenter: could not unpack {city.Root.name} into a plain object. " +
                                   "Undo to revert the layers that were unpacked.", city.Root);
                    return;
                }
                GameObject outermost = PrefabUtility.GetOutermostPrefabInstanceRoot(city.Root.gameObject);
                PrefabUtility.UnpackPrefabInstance(outermost, PrefabUnpackMode.OutermostRoot, InteractionMode.UserAction);
            }

            // Planned again against the unpacked hierarchy, so nothing below holds a reference from before it.
            city = CityTiles.FindCity(SceneManager.GetActiveScene(), out string error);
            if (city == null)
            {
                Undo.CollapseUndoOperations(undoGroup);
                Debug.LogError("CitySceneryParenter: after unpacking, " + error + " Undo to revert the unpack.");
                return;
            }
        }
        Plan plan = unpack ? MakePlan(city, tuning) : preview;

        // Positions first. They are world space, so the reparenting below keeps them.
        foreach (KeyValuePair<Renderer, Placement> entry in plan.placements)
        {
            if (entry.Value.restore)
            {
                Transform t = entry.Key.transform;
                Undo.RecordObject(t, k_UndoName);
                t.position = entry.Value.authoredPivot;
                if (PrefabUtility.IsPartOfPrefabInstance(t))
                {
                    PrefabUtility.RecordPrefabInstancePropertyModifications(t);
                }
            }
        }

        Dictionary<(Transform, string), Transform> containers = new Dictionary<(Transform, string), Transform>();
        Dictionary<(Transform, string, Transform), Transform> copies = new Dictionary<(Transform, string, Transform), Transform>();
        foreach (Move move in plan.moves)
        {
            Transform tile = city.Tiles[move.tile];
            Transform parent = Container(tile, move.container, containers);
            foreach (Transform group in move.groups)
            {
                if (!copies.TryGetValue((tile, move.container, group), out Transform copy))
                {
                    copy = CopyGroup(group, parent);
                    copies[(tile, move.container, group)] = copy;
                }
                parent = copy;
            }
            Undo.SetTransformParent(move.transform, parent, true, k_UndoName);
        }

        // Innermost first, so a group nested inside another split group is gone before its parent is checked.
        for (int i = plan.splitGroups.Count - 1; i >= 0; i--)
        {
            if (plan.splitGroups[i].childCount == 0)
            {
                Undo.DestroyObjectImmediate(plan.splitGroups[i].gameObject);
            }
        }
        foreach (Transform tile in city.Tiles)
        {
            foreach (string containerName in k_SortedContainers)
            {
                Transform container = tile.Find(containerName);
                if (container != null && container.childCount == 0)
                {
                    Undo.DestroyObjectImmediate(container.gameObject); // emptied by a re-sort
                }
            }
        }

        Undo.CollapseUndoOperations(undoGroup);
        EditorSceneManager.MarkSceneDirty(city.Root.gameObject.scene);

        // The tuner caches what it moves and in which frame. Rescanning maps the tied scenery in its tile's frame,
        // with its container's share, straight away — so a tile moved later in this session carries its scenery's
        // tuning with it, and a later scale change leaves the restored medians where they are.
        if (tuning.tuner != null)
        {
            tuning.tuner.Rescan();
        }

        int[] perTile = CountPerTile(city, plan);
        Debug.Log($"CitySceneryParenter: moved {plan.moves.Count} objects onto {perTile.Count(n => n > 0)} of " +
                  $"{city.Tiles.Count} tiles, splitting {plan.splitGroups.Count} groups that straddled tiles, and put " +
                  $"{plan.RestoreCount} street objects back in the middle of the street. Save the scene to keep this.",
                  city.Root);
        LogPlanWarnings(plan);
    }

    /// <summary>The city in the active scene, and the street-width tuning to divide out of its scenery.</summary>
    private static bool Prepare(out CityTiles.City city, out Tuning tuning)
    {
        tuning = null;
        city = CityTiles.FindCity(SceneManager.GetActiveScene(), out string error);
        if (city == null)
        {
            Debug.LogError("CitySceneryParenter: " + error);
            return false;
        }

        StreetWidthTuner[] tuners = Object.FindObjectsByType<StreetWidthTuner>(FindObjectsInactive.Include,
                                                                               FindObjectsSortMode.None);
        if (tuners.Length > 1)
        {
            Debug.LogError($"CitySceneryParenter: {tuners.Length} StreetWidthTuners in the scene, so there is no single " +
                           "scaling to divide out of the scenery. Nothing changed.");
            return false;
        }

        tuning = new Tuning { tuner = tuners.Length == 1 ? tuners[0] : null };
        if (tuning.tuner != null && tuning.tuner.AppliedBlockScale > 1f)
        {
            // Grown blocks reach past the kerb line, so "beyond it is street" stops being true of what is there.
            Debug.LogError($"CitySceneryParenter: StreetWidthTuner is at block scale {tuning.tuner.AppliedBlockScale:F3}, " +
                           "above 1, where a block's own scenery reaches into the street and cannot be told apart from " +
                           "the medians. Set it to 1 or below first. Nothing changed.", tuning.tuner);
            return false;
        }
        return true;
    }

    /// <summary>Everything to sort: the city root's loose children, and whatever already hangs in a sorted container.</summary>
    private static List<Unit> Units(CityTiles.City city)
    {
        List<Unit> units = new List<Unit>();
        foreach (Transform child in city.LooseChildren())
        {
            units.Add(new Unit { node = child, tile = -1 });
        }
        for (int i = 0; i < city.Tiles.Count; i++)
        {
            foreach (string containerName in k_SortedContainers)
            {
                Transform container = city.Tiles[i].Find(containerName);
                if (container == null)
                {
                    continue;
                }
                foreach (Transform child in container)
                {
                    units.Add(new Unit { node = child, tile = i, container = containerName });
                }
            }
        }
        return units;
    }

    private static Plan MakePlan(CityTiles.City city, Tuning tuning)
    {
        Plan plan = new Plan();
        List<Unit> units = Units(city);
        foreach (Unit unit in units)
        {
            foreach (Renderer renderer in unit.node.GetComponentsInChildren<Renderer>(true))
            {
                plan.placements[renderer] = Place(city, tuning, renderer, unit, plan);
            }
        }

        List<Transform> groups = new List<Transform>();
        foreach (Unit unit in units)
        {
            PlanNode(city, unit, unit.node, groups, plan);
        }
        return plan;
    }

    /// <summary>Where one renderer belongs, decided from its authored position (see the class summary).</summary>
    private static Placement Place(CityTiles.City city, Tuning tuning, Renderer renderer, Unit unit, Plan plan)
    {
        Transform t = renderer.transform;
        Vector3 pivot = t.position;

        // Divide the tuning back out about the centre the tuner scaled it around: the nearest tile's for loose
        // scenery, its own tile's once tied. Height is never scaled.
        float factor = tuning.AppliedFactor(renderer, unit.container);
        Vector3 centre = city.Centres[unit.tile >= 0 ? unit.tile : NearestCentre(city, pivot)];
        Vector3 authoredPivot = new Vector3(centre.x + (pivot.x - centre.x) / factor,
                                            pivot.y,
                                            centre.z + (pivot.z - centre.z) / factor);
        Vector3 authored = authoredPivot + (MeshCentre(renderer) - pivot);

        int footprint = CityTiles.FootprintOwner(city.Centres, authored, out float margin);
        Vector3 fromCentre = authored - city.Centres[footprint];
        if (Mathf.Max(Mathf.Abs(fromCentre.x), Mathf.Abs(fromCentre.z)) > CityTiles.BlockHalfSpan)
        {
            return new Placement
            {
                tile = LineOwner(city, authored),
                container = CityTiles.StreetContainer,
                restore = (authoredPivot - pivot).sqrMagnitude > 1e-8f,
                authoredPivot = authoredPivot,
            };
        }

        if (margin < plan.smallestMargin)
        {
            plan.smallestMargin = margin;
            plan.smallestMarginName = renderer.name;
        }
        return new Placement { tile = footprint, container = CityTiles.SceneryContainer };
    }

    /// <summary>
    /// Decide where <paramref name="node"/> goes: whole, if everything in it belongs in one place; otherwise split
    /// (a pure group) or kept whole where most of it belongs (anything else).
    /// </summary>
    private static void PlanNode(CityTiles.City city, Unit unit, Transform node, List<Transform> groups, Plan plan)
    {
        Dictionary<(int, string), int> targets = new Dictionary<(int, string), int>();
        Renderer[] renderers = node.GetComponentsInChildren<Renderer>(true);
        foreach (Renderer renderer in renderers)
        {
            Placement placement = plan.placements[renderer];
            (int, string) key = (placement.tile, placement.container);
            targets[key] = targets.TryGetValue(key, out int n) ? n + 1 : 1;
        }
        if (renderers.Length == 0)
        {
            if (unit.tile >= 0)
            {
                plan.alreadyTied++;
                return; // nothing in it to place, and it is already on a tile
            }
            targets[(CityTiles.FootprintOwner(city.Centres, node.position, out _), CityTiles.SceneryContainer)] = 1;
        }

        if (targets.Count > 1 && IsPureGroup(node))
        {
            plan.splitGroups.Add(node);
            groups.Add(node);
            foreach (Transform child in node)
            {
                PlanNode(city, unit, child, groups, plan);
            }
            groups.RemoveAt(groups.Count - 1);
            return;
        }

        (int tile, string container) target = targets.OrderByDescending(kv => kv.Value).First().Key;
        if (targets.Count > 1)
        {
            plan.keptWhole.Add($"{node.name} belongs in {targets.Count} places and cannot be split, so it goes whole " +
                               $"to {city.Tiles[target.tile].name}/{target.container}");
        }
        if (target.tile == unit.tile && target.container == unit.container)
        {
            plan.alreadyTied++;
            return;
        }
        plan.moves.Add(new Move { transform = node, tile = target.tile, container = target.container, groups = groups.ToArray() });
    }

    /// <summary>
    /// The tile a street object belongs to. Of the tiles whose footprint (widened by <see cref="k_LineTolerance"/>)
    /// reaches it, the westernmost — or, between rows, the southernmost — so everything on one line segment agrees.
    /// </summary>
    private static int LineOwner(CityTiles.City city, Vector3 point)
    {
        float reach = CityTiles.Pitch / 2f + k_LineTolerance;
        int best = -1;
        for (int i = 0; i < city.Centres.Count; i++)
        {
            Vector3 c = city.Centres[i];
            if (Mathf.Abs(point.x - c.x) > reach || Mathf.Abs(point.z - c.z) > reach)
            {
                continue;
            }
            if (best < 0)
            {
                best = i;
                continue;
            }
            Vector3 b = city.Centres[best];
            bool otherRow = Mathf.Abs(c.z - b.z) > CityTiles.Pitch / 2f;
            if (otherRow ? c.z < b.z : c.x < b.x)
            {
                best = i;
            }
        }
        return best >= 0 ? best : CityTiles.FootprintOwner(city.Centres, point, out _);
    }

    /// <summary>The straight-line nearest tile centre — the rule the tuner scaled untied scenery around.</summary>
    private static int NearestCentre(CityTiles.City city, Vector3 point)
    {
        int best = 0;
        float bestSqr = float.MaxValue;
        for (int i = 0; i < city.Centres.Count; i++)
        {
            float dx = city.Centres[i].x - point.x;
            float dz = city.Centres[i].z - point.z;
            float sqr = dx * dx + dz * dz;
            if (sqr < bestSqr)
            {
                bestSqr = sqr;
                best = i;
            }
        }
        return best;
    }

    /// <summary>
    /// Where a renderer's geometry is: the centre of its mesh's bounds, in world space. Read off the mesh rather than
    /// <see cref="Renderer.bounds"/> so an inactive or disabled object is placed as reliably as a visible one; the
    /// pivot is the fallback only for a renderer with no mesh filter.
    /// </summary>
    private static Vector3 MeshCentre(Renderer renderer)
    {
        MeshFilter filter = renderer.GetComponent<MeshFilter>();
        return filter != null && filter.sharedMesh != null
            ? renderer.transform.TransformPoint(filter.sharedMesh.bounds.center)
            : renderer.transform.position;
    }

    /// <summary>
    /// A node that is nothing but a Transform with children, so a copy of it per tile loses nothing. Prefab instance
    /// roots are excluded: a copy would not be an instance.
    /// </summary>
    private static bool IsPureGroup(Transform node)
    {
        return node.childCount > 0
            && node.GetComponents<Component>().Length == 1
            && !PrefabUtility.IsAnyPrefabInstanceRoot(node.gameObject);
    }

    /// <summary>The tile's container of this name, created on first use at the tile's own origin.</summary>
    private static Transform Container(Transform tile, string containerName,
                                       Dictionary<(Transform, string), Transform> containers)
    {
        if (containers.TryGetValue((tile, containerName), out Transform container))
        {
            return container;
        }

        container = tile.Find(containerName);
        if (container == null)
        {
            GameObject go = new GameObject(containerName);
            go.layer = tile.gameObject.layer;
            GameObjectUtility.SetStaticEditorFlags(go, GameObjectUtility.GetStaticEditorFlags(tile.gameObject));
            container = go.transform;
            container.SetParent(tile, false);
            Undo.RegisterCreatedObjectUndo(go, k_UndoName);
        }
        containers[(tile, containerName)] = container;
        return container;
    }

    /// <summary>
    /// An empty copy of <paramref name="group"/> under <paramref name="parent"/>, at the same world pose, so children
    /// moved into it keep their local values exactly.
    /// </summary>
    private static Transform CopyGroup(Transform group, Transform parent)
    {
        GameObject go = new GameObject(group.name);
        go.layer = group.gameObject.layer;
        go.tag = group.tag;
        GameObjectUtility.SetStaticEditorFlags(go, GameObjectUtility.GetStaticEditorFlags(group.gameObject));
        go.SetActive(group.gameObject.activeSelf);

        Transform copy = go.transform;
        copy.SetParent(parent, false);
        copy.SetPositionAndRotation(group.position, group.rotation);
        copy.localScale = group.localScale; // valid because Tie refuses a scaled tile
        Undo.RegisterCreatedObjectUndo(go, k_UndoName);
        return copy;
    }

    private static int[] CountPerTile(CityTiles.City city, Plan plan)
    {
        int[] perTile = new int[city.Tiles.Count];
        foreach (Move move in plan.moves)
        {
            perTile[move.tile]++;
        }
        return perTile;
    }

    private static void LogPlanWarnings(Plan plan)
    {
        foreach (string kept in plan.keptWhole)
        {
            Debug.LogWarning("  " + kept + ".");
        }
        if (plan.smallestMargin < k_AmbiguousMargin)
        {
            Debug.LogWarning($"  {plan.smallestMarginName} stands within {plan.smallestMargin:F2} u of the edge between two " +
                             "tiles, so which of them it went to is arbitrary. Check it by eye.");
        }
    }

    /// <summary>"48 x Road_Structure, 12 x Green_Belt_Tile, ..." — trailing instance numbers stripped.</summary>
    private static string Families(List<Transform> nodes)
    {
        return string.Join(", ", nodes.GroupBy(n => n.name.TrimEnd('0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ' ', '_'))
                                      .OrderByDescending(g => g.Count())
                                      .Select(g => $"{g.Count()} x {g.Key}"));
    }
}
