using System;
using System.Collections.Generic;
using System.Globalization;
using System.Text;
using UnityEngine;
using UnityEngine.Rendering;

/// <summary>
/// Dresses a generated city (<see cref="ProceduralCityBuilder"/>) the way DiamondCityWorld's tiles are dressed: a
/// footpath along every kerb, grass medians planted with trees down some streets, and street furniture -- street
/// lights, tree-lined streets, hydrants, bus stops, benches with bins, and mail boxes, newspaper machines and phone
/// booths near the corners. VoronoiCityWorld carries it (<c>VoronoiCityWorldBuilder.EnsureDressing</c> adds it);
/// ProceduralCityWorld, the density study's scene, does not, so the study's look and the bench are unchanged.
///
/// <para><b>It is derived from the layout file every time the builder runs</b> -- Bake, Build, and at Play whenever the
/// bake's dressing is stale (<see cref="IsCurrent"/>) -- and drawn from the layout's seed, with every decision on its own
/// random stream keyed on the street. So a regenerated or rebaked city is dressed to match with nothing to redo by hand,
/// one layout is always dressed the same way, and S3's coverage levels (one seed, one street network) share their
/// medians, lights and bus stops, losing only the spots their extra buildings stand on.</para>
///
/// <para><b>Nothing it places has a collider</b>, and nothing is on <c>Obstacle</c>. The force law sees only Obstacle
/// colliders, so a prop a drone can hit but the swarm cannot see is a drone lost (internal_3's street-light pole). A
/// prop is never instantiated: its meshes are copied onto fresh objects, so no collider, light or script can come
/// with it. None of it is in the layout file either, so the studies' line-of-sight models (citysim, analyse.py) do not
/// see the trees or the lamps.</para>
///
/// <para><b>Border trees</b> are DiamondCityWorld's verge trees (<see cref="VergeTreePlanter"/>: a row 4 m apart round a
/// third of its tiles and round every goal): a row on the footpath all round <see cref="Rules.borderShare"/> of the blocks
/// and every goal's block, which screens a goal's walkers -- the hat among them -- from a drone low in the street. Only
/// low: trees 4 m tall hide what stands behind them from a camera level with their tops, and the swarm flies at about
/// 20 m (in VoronoiCityWorld at 0.0227, 12% of the walkers a drone at 8 m sees from the goal's streets, 2% at 19.5 m).
/// About 1,200 trees there, +0.05 ms of GPU per FPV render once merged. Planted after the street furniture, so a lamp or
/// a bus stop keeps its place. Since every goal has a row and only some blocks do, a row is a goal cue unless borderShare
/// is 1, as in DiamondCityWorld.</para>
///
/// <para><b>Goals are kept clear</b>, by what their patches need. On the lawns, nothing stands within a goal's clear radius
/// plus <see cref="Rules.goalMargin"/>, so its ring and the ground round it are as the studies modelled them. On the
/// streets it depends on the layout: a goal whose block holds its patch (citygen's <c>goal_cell</c>, VoronoiCityWorld's
/// layouts since 2026-10-26) keeps the street props off its walkers' loops only, so its streets are lit, furnished and
/// planted like any others and a bare stretch cannot give it away; a goal whose ring stands across the streets (the layouts
/// made before) keeps them out of that whole disc, as it always did. The footpath runs through either way. A goal drawn
/// at Play from a block-shaped city (<see cref="BlockGoals"/>) keeps nothing clear: it changes every session, so a
/// dressing shaped round it would mark it, and its walkers take the lane the props leave free.</para>
///
/// <para><b>Placement</b>. The street network is <see cref="StreetSurface.Stretches"/>' -- the road's own setbacks,
/// crossings and merged junctions -- so nothing stands on a crossing or in a junction. A block's kerb is the intersection
/// of the half-planes its streets leave it (citygen's <c>offset_convex(cell, s/2)</c>), its footpath the part of it within
/// <see cref="Rules.pavementWidth"/> of an edge. A prop stands on the footpath of the street it serves, clear of every
/// building box (from the catalogue, as the builder places them) and of the props before it; a spot that fails slides a
/// little along the street and is otherwise skipped, and counted by reason in the summary.</para>
///
/// <para><b>Block-shaped buildings</b> (<see cref="PrismBuilding"/>, S4's variant B) stand at the back of the footpath,
/// their footprint the kerb inset by the layout's footpath width. So the footpath is paved out to the facades whatever
/// that width (the layout's, measured off the buildings, overrides <see cref="Rules.pavementWidth"/> for that layout
/// only), a block a building covers has no lawn, and props keep off the true footprint rather than a box round it, by
/// only <see cref="FacadeClearance"/>: a bench at the back of the footpath stands against the wall. A layout without
/// them is dressed exactly as before.</para>
///
/// <para><b>Cost</b>. An FPV render is CPU culling and submission (CLAUDE.md), so the props are the pack's cheap ones
/// (MC_Tree is 537 triangles, against Tree9's 8.7k; every other prop shares one material) and cast no shadows; the
/// footpath and the medians are one mesh each; and at Play the props are merged (<see cref="MergeForPlay"/>) into one
/// mesh per <see cref="Rules.mergeCell"/> cell per material, about 100 renderers for about 900. Measured headless
/// (2026-10-26, 60 drone views at 1152x648, main-thread ms per FPV render): the bare city 0.25 ms at α 0.0227 and 0.44 ms
/// at 0.10; the dressing adds 0.04 ms merged, against 0.08-0.09 ms as separate static-batched props, with or without a
/// LODGroup each (which is why there are none). The bake is saved unmerged, so it stays editable and is not
/// static-batched at load (that would leave nothing to merge).</para>
/// </summary>
[DisallowMultipleComponent]
[RequireComponent(typeof(ProceduralCityBuilder))]
public class CityDressing : MonoBehaviour
{
    public const string RootName = "Dressing";
    /// <summary>Bump when the rules change what a layout is dressed with, so a bake dressed before is redressed at Play.</summary>
    private const int DressingVersion = 4;
    /// <summary>A border tree counts against the other props as this share of its canopy, so a row's canopies may close up
    /// while its trunks keep off each other and off the furniture.</summary>
    private const float BorderTreeShrink = 0.25f;
    /// <summary>m beyond a walker's loop kept free of street props: about a walker's half-width and a step.</summary>
    private const float WalkerBody = 0.6f;
    /// <summary>m a prop keeps off a block-shaped building's facade, which is the back of its footpath.</summary>
    private const float FacadeClearance = 0.05f;
    /// <summary>Bump when the defaults in <see cref="Rules"/> change. A scene keeps the rules it was saved with, so without
    /// this a changed default never reaches it: <c>VoronoiCityWorldBuilder.EnsureDressing</c> resets older rules to the
    /// current defaults, and says so (a rule set by hand is lost).</summary>
    public const int RulesVersion = 3;

    [Tooltip("Off: no footpaths, medians or props -- the bare city the studies modelled.")]
    public bool dress = true;
    [Tooltip("Mixed into the layout's seed: another salt dresses the same city differently.")]
    public int seedSalt;

    [Header("Props (their meshes are copied, nothing else)")]
    [Tooltip("The road light that lines the pack's streets (BnP_Street_Light_02 in MC_Patch_16_Scaled); its arm is turned over the road.")]
    public GameObject streetLight;
    [Tooltip("MC_Tree: 537 triangles. Not Tree9 (8.7k), the largest single cost of an FPV render in the tile cities.")]
    public GameObject tree;
    [Tooltip("The border rows' tree; empty: the street tree above (MC_Tree, 4.0 m, a 3 m canopy). DiamondCityWorld's " +
             "verges plant Tree9_2 (4.6 m): as rows here it hid 23% of a goal's walkers from a drone at 8 m against MC_Tree's " +
             "12% (5% and 2% at 19.5 m), but its 16x the triangles cost about 1.3 ms of GPU per FPV render against 0.05 ms " +
             "(2026-10-26, 60 street views at 1152x648, merged as at Play).")]
    public GameObject borderTree;
    public GameObject hydrant;
    [Tooltip("Faces the road, its back to the block.")]
    public GameObject bench;
    public GameObject bin;
    [Tooltip("Its open side to the road.")]
    public GameObject busStop;
    [Tooltip("Mail boxes, newspaper machines, phone booths: one near some corners, at the back of the footpath.")]
    public GameObject[] cornerProps = new GameObject[0];

    [Header("Lawns (inside the footpaths)")]
    [Tooltip("The trees planted on the lawns, a grove of one species at a time: the pack's green-belt models, each a bark " +
             "and a leaves object (VoronoiCityWorldBuilder pairs them out of City_Pack_01_Scaled).")]
    public PropModel[] lawnTrees = new PropModel[0];
    [Tooltip("MC_Bush, a hedge segment (a long low block, not a shrub): laid end to end along building fronts.")]
    public GameObject hedge;
    [Tooltip("The pack park's one-head lamp (MC_Street_Lamp), at the sitting areas.")]
    public GameObject parkLamp;
    [Tooltip("MC_Rounded_Grass: the bed in the middle of a sitting area, with a tree on it.")]
    public GameObject flowerBed;
    [Tooltip("Replaces any blended (transparent-queue) material on a copied prop -- the trees' leaves -- with this cutout " +
             "one (Swarm/Double-sided Cutout). Merged at Play, blended leaves would draw in index order rather than by " +
             "distance, and overlapping trees would show through each other.")]
    public Material cutoutLeaves;

    /// <summary>A prop made of several objects that are not one hierarchy (a green-belt tree: its bark and its leaves).</summary>
    [Serializable]
    public class PropModel
    {
        public GameObject[] parts = new GameObject[0];
    }

    [Header("Surfaces")]
    [Tooltip("The footpath (MC_Footpath).")]
    public Material pavementMaterial;
    [Tooltip("The medians' grass (DiamondParkGrass, the diamond parks' lawn).")]
    public Material grassMaterial;
    [Tooltip("Metres of footpath per texture repeat: the pack footpath's own scale (VoronoiCityWorldBuilder fits it).")]
    public float pavementUvMetres = 4f;
    [Tooltip("Metres of grass per UV unit: 76, as on the diamond parks' planes, which the grass material tiles 10.86 times.")]
    public float grassUvMetres = 76f;

    public Rules rules = new Rules();

    [Tooltip("Written by Dress: what the dressing in the scene was made for. Play redresses when it no longer matches.")]
    [SerializeField] private string dressedFor = "";
    // 0, not RulesVersion, as its initializer: a scene saved before the field existed must read as out of date.
    [SerializeField, HideInInspector] private int rulesVersion;

    /// <summary>Whether <see cref="rules"/> were set at the current <see cref="RulesVersion"/>.</summary>
    public bool RulesCurrent => rulesVersion >= RulesVersion;

    /// <summary>The current default rules, marked current.</summary>
    public void ResetRules()
    {
        rules = new Rules();
        rulesVersion = RulesVersion;
    }

    [Serializable]
    public class Rules
    {
        [Header("Footpaths along the kerbs")]
        public float pavementWidth = 3.5f;
        [Tooltip("m above y = 0; the road is at 0.02 and the terrain just below 0.")]
        public float pavementHeight = 0.1f;

        [Header("Medians: grass strips with trees down the middle of some streets")]
        [Range(0f, 1f)] public float medianShare = 0.2f;
        [Tooltip("m of plain road (between the crossings) a street needs to take a median.")]
        public float medianMinStreet = 45f;
        public float medianMinPiece = 20f;
        [Tooltip("m between a median's nose and the crossing, so turning traffic and the zebra stay clear.")]
        public float medianEndGap = 3f;
        public float medianWidth = 2.2f;
        public float medianHeight = 0.15f;
        public float medianTreeSpacing = 10f;

        [Header("Street furniture")]
        [Tooltip("m between street lights along a street, alternating sides.")]
        public float lightSpacing = 40f;
        [Tooltip("m of plain road a street needs to be lit.")]
        public float lightMinStreet = 15f;
        [Range(0f, 1f)] public float treeLinedShare = 0.12f;
        public float streetTreeSpacing = 16f;
        [Tooltip("m from the kerb to a street tree's trunk.")]
        public float streetTreeInset = 1.4f;
        public Vector2 treeScale = new Vector2(0.9f, 1.15f);
        [Tooltip("Of each street end, each side: a hydrant just past the crossing.")]
        [Range(0f, 1f)] public float hydrantShare = 0.08f;
        [Range(0f, 1f)] public float busStopShare = 0.1f;
        public float busStopMinStreet = 60f;
        [Tooltip("Of each street side: a bench and a bin.")]
        [Range(0f, 1f)] public float benchShare = 0.1f;
        public float benchMinStreet = 25f;
        [Tooltip("Of each street end, each side: one of the corner props.")]
        [Range(0f, 1f)] public float cornerShare = 0.06f;

        [Header("Border trees: DiamondCityWorld's verge rows, on the footpath all round a block")]
        [Tooltip("Of the blocks: a row round the whole block. DiamondCityWorld plants 0.328 of its tiles " +
                 "(VergeTreePlanter.tileFraction). 1: every block, and a row is no longer a goal cue.")]
        [Range(0f, 1f)] public float borderShare = 0.33f;
        [Tooltip("Every goal's block gets a row, whatever borderShare draws: it screens the walkers, the hat among them, " +
                 "from the streets round the goal, as DiamondCityWorld's goal verges do.")]
        public bool borderAtGoals = true;
        [Tooltip("m between the trees of a row: DiamondCityWorld's 4.")]
        public float borderSpacing = 4f;
        [Tooltip("Random shift along the kerb, as a fraction of the spacing either way: DiamondCityWorld's 0.23.")]
        [Range(0f, 0.5f)] public float borderJitter = 0.23f;
        [Tooltip("Random size change, as a fraction either way: DiamondCityWorld's 0.23.")]
        [Range(0f, 0.5f)] public float borderScaleJitter = 0.23f;
        [Tooltip("m from the kerb to a row tree's trunk, as for the street trees.")]
        public float borderInset = 1.4f;
        [Tooltip("m a row stops short of each end of its street's plain road, so the junctions stay open: " +
                 "DiamondCityWorld's 5 short of each corner.")]
        public float borderCornerClearance = 5f;

        [Header("Lawns: groves, single trees, hedges and sitting areas inside the footpaths")]
        [Tooltip("m² of lawn per grove drawn: a block draws round(lawn / this) groves, whatever stands on it.")]
        public float groveArea = 1700f;
        public Vector2Int groveTrees = new Vector2Int(3, 6);
        [Tooltip("m: a grove's trees stand within this of its centre.")]
        public float groveRadius = 7f;
        [Tooltip("m between two trunks in a grove.")]
        public float groveTrunkGap = 4f;
        [Tooltip("m² of lawn per single tree drawn.")]
        public float singleTreeArea = 1200f;
        [Tooltip("Park trees grow bigger than the street trees: the same models at 1.1-1.6 fill a lawn without more of them.")]
        public Vector2 lawnTreeScale = new Vector2(1.1f, 1.6f);
        [Tooltip("m between a lawn tree's trunk and the footpath.")]
        public float lawnEdgeMargin = 1.5f;
        [Tooltip("Of the city buildings: a hedge along one front, its segments end to end.")]
        [Range(0f, 1f)] public float hedgeShare = 0.15f;
        [Tooltip("m between a facade and its hedge.")]
        public float hedgeOffset = 0.6f;
        [Tooltip("Of the blocks with at least sittingMinLawn of lawn: a sitting area (a paved round, a bed with a tree, " +
                 "benches facing it, a bin, a lamp, a path to the footpath lined with an avenue, and one formal planting: " +
                 "a ring of trees, a round hedge inside a ring, or a grid of trees behind it).")]
        [Range(0f, 1f)] public float sittingShare = 0.4f;
        public float sittingMinLawn = 1000f;
        public float sittingRadius = 6f;
        public float pathWidth = 2.4f;
        [Tooltip("m: the longest path a sitting area may take to the footpath.")]
        public float pathMaxLength = 60f;
        [Tooltip("m between the pairs of trees lining a sitting area's path.")]
        public float avenueSpacing = 7f;
        [Tooltip("m between the trees of a sitting area's grid (bosque) planting, 3 rows of 4.")]
        public float bosqueSpacing = 6f;

        [Header("Keep-outs")]
        [Tooltip("m beyond each goal's clear radius kept free of props and medians.")]
        public float goalMargin = 3f;
        public float buildingClearance = 0.5f;
        [Tooltip("m between two props.")]
        public float propGap = 0.4f;
        [Tooltip("m between a prop and a crossing or junction.")]
        public float junctionGap = 1.5f;

        [Header("Cost")]
        [Tooltip("m: at Play the props are merged into one mesh per cell this size per material (0: left separate and " +
                 "static-batched, which costs an FPV render about twice as much).")]
        public float mergeCell = 120f;
    }

    /// <summary>Whether the scene's dressing is the one <see cref="Dress"/> would make of <paramref name="file"/> now.</summary>
    public bool IsCurrent(ProceduralCityBuilder.LayoutFile file, StreetSurface.Settings road)
    {
        return dressedFor == Signature(file, road) && (!dress || transform.Find(RootName) != null);
    }

    /// <summary>Removes the dressing and makes it again from <paramref name="file"/> under <c>Dressing</c>: merged at Play
    /// (<see cref="MergeForPlay"/>), marked static in edit mode (a bake) but not batching-static. Returns a summary for
    /// the builder's log line.</summary>
    public string Dress(ProceduralCityBuilder builder, ProceduralCityBuilder.LayoutFile file)
    {
        StreetSurface.Settings road = builder.RoadSettings;
        Clear();
        dressedFor = Signature(file, road);
#if UNITY_EDITOR
        if (!Application.isPlaying) UnityEditor.EditorUtility.SetDirty(this);
#endif
        if (!dress) return "no dressing (off)";
        if (file.streets == null || file.streets.Length == 0) return "no streets to dress";
        var root = new GameObject(RootName).transform;
        root.SetParent(transform, false);
        string summary = new Work(this, builder, file, road, root).Run();
        if (Application.isPlaying)
        {
            summary += "; " + MergeForPlay();
        }
#if UNITY_EDITOR
        else
        {
            var flags = (UnityEditor.StaticEditorFlags)~0 & ~UnityEditor.StaticEditorFlags.BatchingStatic;
            foreach (Transform t in root.GetComponentsInChildren<Transform>(true))
            {
                UnityEditor.GameObjectUtility.SetStaticEditorFlags(t.gameObject, flags);
            }
        }
#endif
        return summary;
    }

    private const string MergedName = "Merged props";

    /// <summary>At Play, once: every prop's meshes merged into one mesh per <see cref="Rules.mergeCell"/> cell per
    /// material, under <c>Dressing/Merged props</c>, and the props themselves switched off (the footpath and median meshes
    /// are one renderer each already). A summary, or why nothing was merged.</summary>
    public string MergeForPlay()
    {
        Transform root = transform.Find(RootName);
        if (!Application.isPlaying || root == null || root.Find(MergedName) != null) return "";
        if (rules.mergeCell <= 0f)
        {
            StaticBatchingUtility.Combine(root.gameObject);
            return "props left separate (mergeCell 0), static-batched";
        }
        var groups = new Dictionary<(int, int, Material), List<CombineInstance>>();
        var folders = new List<GameObject>();
        int props = 0;
        foreach (Transform folder in root)
        {
            if (folder.GetComponent<Renderer>() != null) continue;
            foreach (MeshRenderer mr in folder.GetComponentsInChildren<MeshRenderer>())
            {
                MeshFilter mf = mr.GetComponent<MeshFilter>();
                Mesh mesh = mf != null ? mf.sharedMesh : null;
                if (mesh == null) continue;
                if (!mesh.isReadable && !Application.isEditor)
                {
                    StaticBatchingUtility.Combine(root.gameObject);
                    return $"props left separate ({mesh.name} cannot be read in a player), static-batched";
                }
                Vector3 c = mr.bounds.center;
                Material[] mats = mr.sharedMaterials;
                for (int s = 0; s < mesh.subMeshCount && s < mats.Length; s++)
                {
                    var key = (Mathf.FloorToInt(c.x / rules.mergeCell), Mathf.FloorToInt(c.z / rules.mergeCell), mats[s]);
                    if (!groups.TryGetValue(key, out List<CombineInstance> list)) groups[key] = list = new List<CombineInstance>();
                    list.Add(new CombineInstance { mesh = mesh, subMeshIndex = s, transform = mr.transform.localToWorldMatrix });
                }
            }
            props += folder.childCount;
            folders.Add(folder.gameObject);
        }
        var merged = new GameObject(MergedName).transform;
        merged.SetParent(root, false);
        merged.SetPositionAndRotation(Vector3.zero, Quaternion.identity);   // the instances carry world matrices
        foreach (var kv in groups)
        {
            var mesh = new Mesh { name = $"{MergedName} {kv.Key.Item1},{kv.Key.Item2} {kv.Key.Item3?.name}",
                                  indexFormat = IndexFormat.UInt32 };
            mesh.CombineMeshes(kv.Value.ToArray(), true, true);
            var go = new GameObject(mesh.name);
            go.transform.SetParent(merged, false);
            go.AddComponent<MeshFilter>().sharedMesh = mesh;
            MeshRenderer mr = go.AddComponent<MeshRenderer>();
            mr.sharedMaterial = kv.Key.Item3;
            mr.shadowCastingMode = ShadowCastingMode.Off;
            mr.receiveShadows = false;
        }
        foreach (GameObject f in folders) f.SetActive(false);
        return $"{props} prop meshes merged into {groups.Count}";
    }

    /// <summary>Removes the dressing.</summary>
    public void Clear()
    {
        Transform old = transform.Find(RootName);
        if (old == null) return;
        if (Application.isPlaying)
        {
            old.SetParent(null);        // out of the way of the new one before Destroy, which waits for the frame's end
            Destroy(old.gameObject);
        }
        else
        {
            DestroyImmediate(old.gameObject);
        }
    }

#if UNITY_EDITOR
    [ContextMenu("Redress the baked city")]
    private void RedressFromMenu()
    {
        ProceduralCityBuilder builder = GetComponent<ProceduralCityBuilder>();
        if (builder.Redress(out string summary)) Debug.Log("CityDressing: " + summary, this);
        else Debug.LogError("CityDressing: " + summary, this);
    }
#endif

    private string Signature(ProceduralCityBuilder.LayoutFile file, StreetSurface.Settings road)
    {
        var sb = new StringBuilder();
        void Add(object o) => sb.Append(Convert.ToString(o, CultureInfo.InvariantCulture)).Append('|');
        Add(DressingVersion);
        Add(dress);
        Add(seedSalt);
        Add(file.layout.id);
        Add(file.layout.seed);
        Add(file.boxes.Length);
        Add(file.streets?.Length ?? 0);
        Add(road.crossingDepth);
        Add(road.maxSetback);
        Add(road.minStrip);
        Add(JsonUtility.ToJson(rules));
        Add(pavementUvMetres);
        Add(grassUvMetres);
        foreach (UnityEngine.Object o in new UnityEngine.Object[] { streetLight, tree, hydrant, bench, bin, busStop,
                                                                   pavementMaterial, grassMaterial, hedge, parkLamp,
                                                                   flowerBed, cutoutLeaves, borderTree })
        {
            Add(o != null ? o.name : "-");
        }
        foreach (GameObject o in cornerProps ?? new GameObject[0]) Add(o != null ? o.name : "-");
        foreach (PropModel m in lawnTrees ?? new PropModel[0])
        {
            foreach (GameObject o in m?.parts ?? new GameObject[0]) Add(o != null ? o.name : "-");
        }
        uint h = 2166136261;
        foreach (char ch in sb.ToString())
        {
            h ^= ch;
            h *= 16777619;
        }
        return $"{file.layout.id} #{h:x8}";
    }

    // ------------------------------------------------------------------ the work of one Dress

    private enum Facing
    {
        Any,        // a random yaw (trees)
        Along,      // its longer side along the street, either way round
        Arm,        // its arm over the road (lights): the top's centroid off the pole's
        Back,       // facing the road, its back to the block: away from its tallest part (a bench's backrest)
        Open,       // facing the road: towards its seat, the bulk of its mid-height vertices (a shelter, whose back panel is flat)
    }

    /// <summary>An oriented rectangle on the ground: centre and two half-axis vectors.</summary>
    private struct Obb
    {
        public Vector2 c, u, v;

        public Obb(Vector2 c, Vector2 u, Vector2 v)
        {
            this.c = c;
            this.u = u;
            this.v = v;
        }

        public Vector2 Extent => new Vector2(Mathf.Abs(u.x) + Mathf.Abs(v.x), Mathf.Abs(u.y) + Mathf.Abs(v.y));

        public Vector2 Corner(int k) => c + ((k & 1) == 0 ? -u : u) + ((k & 2) == 0 ? -v : v);

        /// <summary>Whether they come within <paramref name="gap"/> of each other (separating axes).</summary>
        public static bool Near(Obb a, Obb b, float gap)
        {
            Vector2 d = b.c - a.c;
            foreach (Vector2 axis in new[] { a.u, a.v, b.u, b.v })
            {
                float len = axis.magnitude;
                if (len < 1e-6f) continue;
                Vector2 n = axis / len;
                float ra = Mathf.Abs(Vector2.Dot(a.u, n)) + Mathf.Abs(Vector2.Dot(a.v, n));
                float rb = Mathf.Abs(Vector2.Dot(b.u, n)) + Mathf.Abs(Vector2.Dot(b.v, n));
                if (Mathf.Abs(Vector2.Dot(d, n)) > ra + rb + gap) return false;
            }
            return true;
        }
    }

    /// <summary>Rectangles hashed on a grid, for "is anything within this gap".</summary>
    private sealed class Grid
    {
        private const float Cell = 16f;
        private readonly Dictionary<(int, int), List<int>> cells = new Dictionary<(int, int), List<int>>();
        private readonly List<Obb> items = new List<Obb>();

        public void Add(Obb o)
        {
            int i = items.Count;
            items.Add(o);
            Vector2 e = o.Extent;
            for (int x = Key(o.c.x - e.x); x <= Key(o.c.x + e.x); x++)
            {
                for (int z = Key(o.c.y - e.y); z <= Key(o.c.y + e.y); z++)
                {
                    if (!cells.TryGetValue((x, z), out List<int> list)) cells[(x, z)] = list = new List<int>();
                    list.Add(i);
                }
            }
        }

        public bool Near(Obb o, float gap)
        {
            Vector2 e = o.Extent + Vector2.one * gap;
            for (int x = Key(o.c.x - e.x); x <= Key(o.c.x + e.x); x++)
            {
                for (int z = Key(o.c.y - e.y); z <= Key(o.c.y + e.y); z++)
                {
                    if (!cells.TryGetValue((x, z), out List<int> list)) continue;
                    foreach (int i in list)
                    {
                        if (Obb.Near(items[i], o, gap)) return true;
                    }
                }
            }
            return false;
        }

        private static int Key(float v) => Mathf.FloorToInt(v / Cell);
    }

    /// <summary>Convex footprints (block-shaped buildings), for "is anything within this gap" against a rectangle.</summary>
    private sealed class Footprints
    {
        private readonly List<Vector2[]> polys = new List<Vector2[]>();
        private readonly List<Rect> bounds = new List<Rect>();

        public int Count => polys.Count;
        public IReadOnlyList<Vector2[]> Polys => polys;

        public void Add(Vector2[] poly)
        {
            Vector2 lo = poly[0], hi = poly[0];
            foreach (Vector2 p in poly)
            {
                lo = Vector2.Min(lo, p);
                hi = Vector2.Max(hi, p);
            }
            polys.Add(poly);
            bounds.Add(Rect.MinMaxRect(lo.x, lo.y, hi.x, hi.y));
        }

        /// <summary>Whether <paramref name="o"/> comes within <paramref name="gap"/> of any footprint (separating axes:
        /// the footprint's edge normals and the rectangle's two axes).</summary>
        public bool Near(Obb o, float gap)
        {
            Vector2 e = o.Extent + Vector2.one * gap;
            var box = Rect.MinMaxRect(o.c.x - e.x, o.c.y - e.y, o.c.x + e.x, o.c.y + e.y);
            for (int i = 0; i < polys.Count; i++)
            {
                if (bounds[i].Overlaps(box) && !Separated(polys[i], o, gap)) return true;
            }
            return false;
        }

        private static bool Separated(Vector2[] poly, Obb o, float gap)
        {
            for (int i = 0; i < poly.Length + 2; i++)
            {
                Vector2 n;
                if (i < poly.Length)
                {
                    Vector2 d = poly[(i + 1) % poly.Length] - poly[i];
                    n = new Vector2(-d.y, d.x);
                }
                else
                {
                    n = i == poly.Length ? o.u : o.v;
                }
                float len = n.magnitude;
                if (len < 1e-6f) continue;
                n /= len;
                float lo = float.PositiveInfinity, hi = float.NegativeInfinity;
                foreach (Vector2 p in poly)
                {
                    float t = Vector2.Dot(p, n);
                    lo = Mathf.Min(lo, t);
                    hi = Mathf.Max(hi, t);
                }
                float c = Vector2.Dot(o.c, n), r = Mathf.Abs(Vector2.Dot(o.u, n)) + Mathf.Abs(Vector2.Dot(o.v, n));
                if (c - r > hi + gap || c + r < lo - gap) return true;
            }
            return false;
        }
    }

    /// <summary>A block: its kerb, a convex polygon counter-clockwise seen from above, and its lawn, the kerb polygon less
    /// the footpath (empty where the footpath covers the block).</summary>
    private sealed class Block
    {
        public int id;
        public List<Vector2> poly;
        public Vector2 inside;
        public List<Vector2> lawn = new List<Vector2>();
        public float lawnArea;
        public Vector2 lawnMin, lawnMax;

        public bool Contains(Vector2 p, float margin) => Inside(poly, p, margin);
        public bool OnLawn(Vector2 p, float margin) => lawn.Count >= 3 && Inside(lawn, p, margin);
    }

    /// <summary>Whether <paramref name="p"/> is at least <paramref name="margin"/> inside a convex counter-clockwise
    /// polygon.</summary>
    private static bool Inside(List<Vector2> poly, Vector2 p, float margin)
    {
        for (int i = 0; i < poly.Count; i++)
        {
            Vector2 a = poly[i], d = (poly[(i + 1) % poly.Count] - a).normalized;
            if (Vector2.Dot(new Vector2(-d.y, d.x), p - a) < margin) return false;
        }
        return true;
    }

    /// <summary>A prop as measured once: its meshes relative to its source's position (world rotation and scale, as it
    /// stands in its prefab), the centroid of its lowest slice (pole, trunk, legs: the point that is stood on a spot), and
    /// its body and base rectangles about that point.</summary>
    private sealed class Proto
    {
        public string name;
        public readonly List<(Mesh mesh, Material[] mats, Vector3 pos, Quaternion rot, Vector3 scale)> parts =
            new List<(Mesh, Material[], Vector3, Quaternion, Vector3)>();
        public Vector2 anchor;
        public float minY, height;
        public Vector2 bodyMin, bodyMax, baseMin, baseMax;     // relative to the anchor
        public Vector2 front;                                  // unit, on one of its axes

        public Obb Body(Vector2 at, float yaw, float k) => Box(bodyMin, bodyMax, at, yaw, k);
        public Obb Base(Vector2 at, float yaw, float k) => Box(baseMin, baseMax, at, yaw, k);

        private static Obb Box(Vector2 min, Vector2 max, Vector2 at, float yaw, float k)
        {
            Vector2 c = Turn((min + max) * 0.5f * k, yaw), h = (max - min) * 0.5f * k;
            return new Obb(at + c, Turn(new Vector2(h.x, 0f), yaw), Turn(new Vector2(0f, h.y), yaw));
        }
    }

    /// <summary>(x, z) turned as a yaw of <paramref name="yaw"/> degrees about +Y turns it (clockwise seen from above).</summary>
    private static Vector2 Turn(Vector2 v, float yaw)
    {
        Vector3 w = Quaternion.AngleAxis(yaw, Vector3.up) * new Vector3(v.x, 0f, v.y);
        return new Vector2(w.x, w.z);
    }

    /// <summary>The yaw that turns <paramref name="from"/> onto <paramref name="to"/> (both (x, z)).</summary>
    private static float YawFromTo(Vector2 from, Vector2 to) => -Vector2.SignedAngle(from, to);

    /// <summary>A goal as the dressing keeps clear of it: its disc (the clear radius plus <see cref="Rules.goalMargin"/>),
    /// its block, its buildings and its walkers' loops, and whether that block holds them all (set once the blocks are
    /// built; see the class summary).</summary>
    private sealed class GoalZone
    {
        public Vector2 c;
        public float radius;
        public int block;
        public readonly List<Obb> parts = new List<Obb>();
        public readonly List<Obb> walks = new List<Obb>();
        public bool held;
    }

    private sealed class Work
    {
        private readonly CityDressing d;
        private Rules r;
        private readonly ProceduralCityBuilder.LayoutFile file;
        private readonly Transform root;
        private readonly List<StreetSurface.Stretch> stretches;
        private readonly Dictionary<int, Block> blocks = new Dictionary<int, Block>();
        private readonly Grid buildings = new Grid();
        private readonly Footprints prisms = new Footprints();
        private readonly List<(Obb box, bool city, string name)> boxes = new List<(Obb, bool, string)>();
        private readonly Grid props = new Grid();
        private readonly List<GoalZone> goals = new List<GoalZone>();
        private readonly Dictionary<string, Transform> folders = new Dictionary<string, Transform>();
        private readonly Dictionary<string, int> placed = new Dictionary<string, int>();
        private readonly List<string> kinds = new List<string>();
        private readonly Dictionary<string, int> skipped = new Dictionary<string, int>();
        private readonly HashSet<int> medianStreets = new HashSet<int>();
        private readonly int seed;
        private readonly List<string> notes = new List<string>();
        private float medianMetres, pavementMetres;
        private int medianCount;

        public Work(CityDressing d, ProceduralCityBuilder builder, ProceduralCityBuilder.LayoutFile file,
                    StreetSurface.Settings road, Transform root)
        {
            this.d = d;
            r = d.rules;
            this.file = file;
            this.root = root;
            seed = unchecked(file.layout.seed ^ (d.seedSalt * 486187739));
            var segments = new List<(Vector2, Vector2, float)>();
            foreach (ProceduralCityBuilder.LayoutStreet s in file.streets)
            {
                bool ok = s.xz != null && s.xz.Length >= 4;
                segments.Add(ok ? (new Vector2(s.xz[0], s.xz[1]), new Vector2(s.xz[2], s.xz[3]), s.width)
                                : (Vector2.zero, Vector2.zero, s.width));
            }
            stretches = StreetSurface.Stretches(segments, road);
            var byIndex = new Dictionary<int, GoalZone>();
            foreach (ProceduralCityBuilder.LayoutGoal g in file.layout.goals ?? new ProceduralCityBuilder.LayoutGoal[0])
            {
                if (g.kind == BlockGoals.Kind) continue;     // drawn at Play from the city as it stands: it clears nothing
                var zone = new GoalZone { c = new Vector2(g.centre[0], g.centre[1]), radius = g.clear_radius + r.goalMargin,
                                          block = g.block_id };
                goals.Add(zone);
                byIndex[g.index] = zone;
            }
            // How far out a walker's loop runs: WalkerPatrol walks baseOffset + U(0, margin) / 2 off its building.
            float walk = builder.baseOffset + 0.5f * builder.margin + WalkerBody;
            foreach (ProceduralCityBuilder.LayoutBox b in file.boxes)
            {
                if (PrismBuilding.IsPrism(b))
                {
                    if (PrismBuilding.Valid(b, out _)) prisms.Add(PrismBuilding.Footprint(b));
                    continue;
                }
                BuildingCatalogue.Entry e = builder.catalogue != null ? builder.catalogue.Find(b.design) : null;
                if (e == null) continue;
                Quaternion rot = Quaternion.Euler(-90f, b.yaw, 0f);   // as ProceduralCityBuilder.Place stands it
                Vector3 u = rot * new Vector3(0.5f * e.Width * b.size, 0f, 0f);
                Vector3 v = rot * new Vector3(0f, 0.5f * e.Depth * b.size, 0f);
                var box = new Obb(new Vector2(b.c[0], b.c[2]), new Vector2(u.x, u.z), new Vector2(v.x, v.z));
                buildings.Add(box);
                boxes.Add((box, b.role == "city", b.name ?? ""));
                if (b.role == "city" || !byIndex.TryGetValue(b.goal, out GoalZone zone)) continue;
                zone.parts.Add(box);
                if (!b.patrol) continue;
                var loops = new Obb(box.c, box.u + box.u.normalized * walk, box.v + box.v.normalized * walk);
                zone.walks.Add(loops);
                zone.parts.Add(loops);
            }
        }

        public string Run()
        {
            BuildBlocks();
            FitToPrisms();
            int held = 0;
            foreach (GoalZone g in goals)
            {
                if (g.parts.Count > 0 && blocks.TryGetValue(g.block, out Block b))
                {
                    g.held = g.parts.TrueForAll(o => Within(b, o));
                }
                held += g.held ? 1 : 0;
            }
            if (goals.Count > 0)
            {
                notes.Add(held == goals.Count ? "every goal stands in a block of its own"
                          : held == 0 ? "the goals' rings stand across the streets, which are left bare round them"
                          : $"{held} of {goals.Count} goals stand in a block of their own");
            }
            Pave();
            Medians();
            Proto light = Measure(d.streetLight, Facing.Arm), tree = Measure(d.tree, Facing.Any),
                  hydrant = Measure(d.hydrant, Facing.Along), bench = Measure(d.bench, Facing.Back),
                  bin = Measure(d.bin, Facing.Along), shelter = Measure(d.busStop, Facing.Open);
            var corner = new List<Proto>();
            foreach (GameObject g in d.cornerProps ?? new GameObject[0])
            {
                Proto p = Measure(g, Facing.Along);
                if (p != null) corner.Add(p);
            }
            MedianTrees(tree);
            BusStops(shelter, bin);
            Lights(light);
            StreetTrees(tree);
            Benches(bench, bin);
            EndProps("Hydrants", 1, r.hydrantShare, 1.5f, hydrant == null ? null : new List<Proto> { hydrant }, false);
            EndProps("Corner props", 2, r.cornerShare, 4f, corner, true);
            BorderTrees(d.borderTree != null ? Measure(d.borderTree, Facing.Any) : tree);
            var lawnTrees = new List<Proto>();
            foreach (PropModel m in d.lawnTrees ?? new PropModel[0])
            {
                Proto p = Measure(m, Facing.Any);
                if (p != null) lawnTrees.Add(p);
            }
            if (lawnTrees.Count == 0 && tree != null) lawnTrees.Add(tree);
            Lawns(lawnTrees, Measure(d.hedge, Facing.Along), bench, bin, Measure(d.parkLamp, Facing.Arm),
                  Measure(d.flowerBed, Facing.Any));
            return Summary();
        }

        // ---------------------------------------------------------------- blocks and their footpaths

        private void BuildBlocks()
        {
            var sides = new Dictionary<int, List<int>>();
            for (int k = 0; k < file.streets.Length; k++)
            {
                int[] ids = file.streets[k].blocks;
                if (ids == null) continue;
                foreach (int id in ids)
                {
                    if (id < 0) continue;
                    if (!sides.TryGetValue(id, out List<int> list)) sides[id] = list = new List<int>();
                    list.Add(k);
                }
            }
            if (sides.Count == 0) notes.Add("the layout names no blocks beside its streets, so there are no footpaths");
            foreach (var kv in sides)
            {
                Vector2 inside = Vector2.zero, lo = Vector2.positiveInfinity, hi = Vector2.negativeInfinity;
                foreach (int k in kv.Value)
                {
                    foreach (Vector2 p in new[] { stretches[k].a, stretches[k].b })
                    {
                        inside += p;
                        lo = Vector2.Min(lo, p);
                        hi = Vector2.Max(hi, p);
                    }
                }
                inside /= 2 * kv.Value.Count;
                var poly = new List<Vector2>
                {
                    new Vector2(lo.x - 1f, lo.y - 1f), new Vector2(hi.x + 1f, lo.y - 1f),
                    new Vector2(hi.x + 1f, hi.y + 1f), new Vector2(lo.x - 1f, hi.y + 1f),
                };
                foreach (int k in kv.Value)
                {
                    StreetSurface.Stretch s = stretches[k];
                    if (s.Length < 1e-3f) continue;
                    Vector2 n = s.Left;
                    if (Vector2.Dot(n, inside - s.a) < 0f) n = -n;
                    poly = Clip(poly, n, Vector2.Dot(n, s.a) + 0.5f * s.width);
                }
                poly = Dedupe(poly);
                if (poly.Count < 3) continue;
                var block = new Block { id = kv.Key, poly = poly, inside = inside };
                List<Vector2> lawn = poly;
                for (int i = 0; i < poly.Count && lawn.Count >= 3; i++)
                {
                    Vector2 a = poly[i], e = (poly[(i + 1) % poly.Count] - a).normalized, n = new Vector2(-e.y, e.x);
                    lawn = Clip(lawn, n, Vector2.Dot(n, a) + r.pavementWidth);
                }
                lawn = Dedupe(lawn);
                if (lawn.Count >= 3)
                {
                    block.lawn = lawn;
                    block.lawnMin = block.lawnMax = lawn[0];
                    for (int i = 0; i < lawn.Count; i++)
                    {
                        Vector2 p = lawn[i], q = lawn[(i + 1) % lawn.Count];
                        block.lawnArea += 0.5f * (p.x * q.y - q.x * p.y);
                        block.lawnMin = Vector2.Min(block.lawnMin, p);
                        block.lawnMax = Vector2.Max(block.lawnMax, p);
                    }
                }
                blocks[kv.Key] = block;
            }
        }

        /// <summary>For block-shaped buildings: the footpath runs to their facades (its width the median over them of how
        /// far each stands inside its kerb, replacing <see cref="Rules.pavementWidth"/> for this layout when they differ,
        /// and the blocks rebuilt), and a block whose lawn a building covers, to within half a metre, has none.</summary>
        private void FitToPrisms()
        {
            if (prisms.Count == 0) return;
            var insets = new List<float>();
            foreach (Vector2[] poly in prisms.Polys)
            {
                Block b = BlockAt(Centroid(poly));
                if (b == null) continue;
                float inset = float.PositiveInfinity;
                foreach (Vector2 p in poly)
                {
                    for (int i = 0; i < b.poly.Count; i++)
                    {
                        Vector2 a = b.poly[i], e = (b.poly[(i + 1) % b.poly.Count] - a).normalized;
                        inset = Mathf.Min(inset, Vector2.Dot(new Vector2(-e.y, e.x), p - a));
                    }
                }
                insets.Add(inset);
            }
            if (insets.Count > 0)
            {
                insets.Sort();
                float width = insets[insets.Count / 2];
                if (width > 0.5f && Mathf.Abs(width - r.pavementWidth) > 0.01f)
                {
                    r = JsonUtility.FromJson<Rules>(JsonUtility.ToJson(r));
                    r.pavementWidth = width;
                    blocks.Clear();
                    BuildBlocks();
                    notes.Add($"footpaths {width:0.##} m wide, out to the block-shaped buildings' facades");
                }
            }
            int covered = 0;
            foreach (Vector2[] poly in prisms.Polys)
            {
                Block b = BlockAt(Centroid(poly));
                if (b == null || b.lawn.Count < 3) continue;
                var footprint = new List<Vector2>(poly);
                if (!b.lawn.TrueForAll(p => Inside(footprint, p, -0.5f))) continue;
                b.lawn = new List<Vector2>();
                b.lawnArea = 0f;
                covered++;
            }
            notes.Add($"{covered} of {prisms.Count} block-shaped buildings fill their block to the footpath, which leaves " +
                      "it no lawn");
        }

        private static Vector2 Centroid(Vector2[] poly)
        {
            Vector2 c = Vector2.zero;
            foreach (Vector2 p in poly) c += p;
            return c / Mathf.Max(poly.Length, 1);
        }

        /// <summary>Whether a rectangle comes within <paramref name="gap"/> of a building: a catalogue box, or a
        /// block-shaped building's footprint, which a prop may stand against (<see cref="FacadeClearance"/>).</summary>
        private bool AtBuilding(Obb o, float gap) =>
            buildings.Near(o, gap) || (prisms.Count > 0 && prisms.Near(o, Mathf.Min(gap, FacadeClearance)));

        /// <summary>One mesh: each block's footpath, the part of its kerb polygon within the footpath's width of an edge --
        /// for a convex polygon exactly the union over its edges of the polygon clipped to that edge's band, each piece
        /// convex. Where pieces overlap they are coplanar and mapped alike, so nothing shows. A kerb face along the road,
        /// another down to the lawn.</summary>
        private void Pave()
        {
            if (d.pavementMaterial == null)
            {
                if (blocks.Count > 0) notes.Add("no pavement material, so no footpaths");
                return;
            }
            var m = new MeshBuffer();
            float y = r.pavementHeight, w = r.pavementWidth, uv = Mathf.Max(d.pavementUvMetres, 0.1f);
            foreach (Block b in blocks.Values)
            {
                List<Vector2> poly = b.poly, inner = b.lawn;
                for (int i = 0; i < poly.Count; i++)
                {
                    Vector2 a = poly[i], e = (poly[(i + 1) % poly.Count] - a).normalized, n = new Vector2(-e.y, e.x);
                    m.Floor(0, Clip(poly, -n, -(Vector2.Dot(n, a) + w)), y, uv);
                    m.Wall(0, a, poly[(i + 1) % poly.Count], 0f, y, -n, uv);
                    pavementMetres += (poly[(i + 1) % poly.Count] - a).magnitude;
                }
                for (int i = 0; inner.Count >= 3 && i < inner.Count; i++)
                {
                    Vector2 a = inner[i], b2 = inner[(i + 1) % inner.Count], e = (b2 - a).normalized;
                    m.Wall(0, a, b2, -0.05f, y, new Vector2(-e.y, e.x), uv);
                }
            }
            m.Make(root, "Footpaths", d.pavementMaterial);
        }

        // ---------------------------------------------------------------- medians

        private void Medians()
        {
            if (d.grassMaterial == null || d.pavementMaterial == null)
            {
                if (r.medianShare > 0f) notes.Add("no grass or pavement material, so no medians");
                return;
            }
            var m = new MeshBuffer();
            float hw = 0.5f * r.medianWidth, y = r.medianHeight, uv = Mathf.Max(d.pavementUvMetres, 0.1f);
            for (int k = 0; k < stretches.Count; k++)
            {
                StreetSurface.Stretch s = stretches[k];
                var rng = Rng(10, k, 0);
                bool pick = rng.NextDouble() < r.medianShare;
                if (!pick || s.absorbed || file.streets[k].edge || s.end - s.start < r.medianMinStreet) continue;
                foreach ((float t0, float t1) in FreeRuns(s, s.start + r.medianEndGap, s.end - r.medianEndGap, hw + 0.3f))
                {
                    if (t1 - t0 < r.medianMinPiece) continue;
                    Vector2 p0 = s.At(t0, -hw), p1 = s.At(t1, -hw), p2 = s.At(t1, hw), p3 = s.At(t0, hw);
                    var top = new List<Vector2> { p0, p1, p2, p3 };
                    m.Floor(0, top, y, Mathf.Max(d.grassUvMetres, 0.1f));
                    for (int i = 0; i < 4; i++)
                    {
                        Vector2 a = top[i], b = top[(i + 1) % 4];
                        Vector2 mid = (a + b) * 0.5f, centre = (p0 + p2) * 0.5f;
                        m.Wall(1, a, b, 0f, y, (mid - centre).normalized, uv);
                    }
                    pieces.Add((k, t0, t1));
                    medianMetres += t1 - t0;
                    medianCount++;
                }
                medianStreets.Add(k);
            }
            if (medianCount > 0) m.Make(root, "Medians", d.grassMaterial, d.pavementMaterial);
        }

        private readonly List<(int k, float t0, float t1)> pieces = new List<(int, float, float)>();

        private void MedianTrees(Proto tree)
        {
            if (tree == null) return;
            foreach ((int k, float t0, float t1) in pieces)
            {
                StreetSurface.Stretch s = stretches[k];
                var rng = Rng(11, k, Mathf.RoundToInt(t0));
                float len = t1 - t0 - 5f;
                int n = Mathf.Max(1, Mathf.FloorToInt(len / r.medianTreeSpacing) + 1);
                float step = n > 1 ? len / (n - 1) : 0f;
                for (int i = 0; i < n; i++)
                {
                    float t = n > 1 ? t0 + 2.5f + i * step : 0.5f * (t0 + t1);
                    if (i > 0 && i < n - 1) t += (float)(rng.NextDouble() - 0.5);
                    TreeAt("Median trees", tree, s.At(t, 0f), r.medianHeight, rng, 0f, Vector2.zero, _ => true);
                }
            }
        }

        /// <summary>The runs of [t0, t1] along a street's centre line at least <paramref name="half"/> from every building and
        /// outside every goal, sampled every half metre.</summary>
        private List<(float, float)> FreeRuns(StreetSurface.Stretch s, float t0, float t1, float half)
        {
            var runs = new List<(float, float)>();
            float start = float.NaN;
            for (float t = t0; t <= t1 + 1e-3f; t += 0.5f)
            {
                Vector2 p = s.At(Mathf.Min(t, t1), 0f);
                bool ok = !InGoal(p, half) &&
                          !AtBuilding(new Obb(p, s.Dir * 0.25f, s.Left * 0.01f), half);
                if (ok && float.IsNaN(start)) start = t;
                if (!ok && !float.IsNaN(start))
                {
                    runs.Add((start, t - 0.5f));
                    start = float.NaN;
                }
            }
            if (!float.IsNaN(start)) runs.Add((start, t1));
            return runs;
        }

        // ---------------------------------------------------------------- along the footpaths

        /// <summary>The sides of a street that have a footpath: +1 (its left, seen along a -> b) and -1, each with its block.</summary>
        private List<(int side, Block block)> Sides(int k)
        {
            var list = new List<(int, Block)>();
            StreetSurface.Stretch s = stretches[k];
            int[] ids = file.streets[k].blocks;
            if (ids == null || s.absorbed) return list;
            foreach (int id in ids)
            {
                if (id < 0 || !blocks.TryGetValue(id, out Block b)) continue;
                list.Add((Vector2.Dot(s.Left, b.inside - s.a) >= 0f ? 1 : -1, b));
            }
            list.Sort((x, y) => -x.Item1.CompareTo(y.Item1));     // +1 first, whatever order the layout lists them
            return list;
        }

        private float Usable(StreetSurface.Stretch s) => s.end - s.start - 2f * r.junctionGap;

        private void Lights(Proto light)
        {
            if (light == null) return;
            for (int k = 0; k < stretches.Count; k++)
            {
                StreetSurface.Stretch s = stretches[k];
                List<(int side, Block block)> sides = Sides(k);
                float len = Usable(s);
                if (sides.Count == 0 || s.end - s.start < r.lightMinStreet) continue;
                var rng = Rng(20, k, 0);
                int first = rng.Next(2);
                int n = Mathf.Max(1, Mathf.FloorToInt(len / r.lightSpacing) + 1);
                float t = s.start + r.junctionGap + 0.5f * (len - (n - 1) * r.lightSpacing);
                for (int i = 0; i < n; i++, t += r.lightSpacing)
                {
                    var (side, block) = sides[(i + first) % sides.Count];
                    OnFootpath("Street lights", light, s, side, block, t, Kerb, 3f, Facing.Arm, rng, 1f);
                }
            }
        }

        private void StreetTrees(Proto tree)
        {
            if (tree == null) return;
            for (int k = 0; k < stretches.Count; k++)
            {
                StreetSurface.Stretch s = stretches[k];
                var rng = Rng(30, k, 0);
                bool pick = rng.NextDouble() < r.treeLinedShare;
                float len = Usable(s);
                if (!pick || medianStreets.Contains(k) || len < 30f) continue;
                int n = Mathf.Max(1, Mathf.FloorToInt(len / r.streetTreeSpacing) + 1);
                float lo = s.start + r.junctionGap, hi = s.end - r.junctionGap;
                foreach (var (side, block) in Sides(k))
                {
                    float t = lo + 0.5f * (len - (n - 1) * r.streetTreeSpacing);
                    for (int i = 0; i < n; i++, t += r.streetTreeSpacing)
                    {
                        Vector2 at = s.At(t, side * (0.5f * s.width + r.streetTreeInset));
                        TreeAt("Street trees", tree, at, r.pavementHeight, rng, 2.5f, s.Dir,
                               bas => FootpathFits(bas, s, side, block, lo, hi));
                    }
                }
            }
        }

        private void BusStops(Proto shelter, Proto bin)
        {
            if (shelter == null) return;
            for (int k = 0; k < stretches.Count; k++)
            {
                StreetSurface.Stretch s = stretches[k];
                var rng = Rng(40, k, 0);
                bool pick = rng.NextDouble() < r.busStopShare;
                List<(int side, Block block)> sides = Sides(k);
                if (!pick || sides.Count == 0 || s.end - s.start < r.busStopMinStreet) continue;
                var (side, block) = sides[rng.Next(sides.Count)];
                float t = 0.5f * (s.start + s.end) + (float)(rng.NextDouble() - 0.5) * 0.3f * (s.end - s.start);
                if (!OnFootpath("Bus stops", shelter, s, side, block, t, Middle, 4f, Facing.Open, rng, 1f, out Obb body))
                {
                    continue;
                }
                if (bin != null)
                {
                    float along = Mathf.Abs(Vector2.Dot(body.u, s.Dir)) + Mathf.Abs(Vector2.Dot(body.v, s.Dir));
                    float tb = Vector2.Dot(body.c - s.a, s.Dir) + along + 0.9f;
                    OnFootpath("Bins", bin, s, side, block, tb, Back, 0f, Facing.Along, rng, 1f);
                }
            }
        }

        private void Benches(Proto bench, Proto bin)
        {
            if (bench == null) return;
            for (int k = 0; k < stretches.Count; k++)
            {
                StreetSurface.Stretch s = stretches[k];
                foreach (var (side, block) in Sides(k))
                {
                    var rng = Rng(50, k, side);
                    bool pick = rng.NextDouble() < r.benchShare;
                    if (!pick || s.end - s.start < r.benchMinStreet) continue;
                    float t = s.start + r.junctionGap + Usable(s) * (0.25f + 0.5f * (float)rng.NextDouble());
                    if (!OnFootpath("Benches", bench, s, side, block, t, Back, 2f, Facing.Back, rng, 1f, out Obb body))
                    {
                        continue;
                    }
                    if (bin == null) continue;
                    float along = Mathf.Abs(Vector2.Dot(body.u, s.Dir)) + Mathf.Abs(Vector2.Dot(body.v, s.Dir));
                    float sign = rng.NextDouble() < 0.5 ? 1f : -1f;
                    float tb = Vector2.Dot(body.c - s.a, s.Dir) + sign * (along + 0.9f);
                    OnFootpath("Bins", bin, s, side, block, tb, Back, 0f, Facing.Along, rng, 1f);
                }
            }
        }

        /// <summary>At each end of each street side, with chance <paramref name="share"/>, one of <paramref name="protos"/>
        /// <paramref name="fromEnd"/> metres past the crossing, at the kerb or at the back of the footpath.</summary>
        private void EndProps(string kind, int stream, float share, float fromEnd, List<Proto> protos, bool back)
        {
            if (protos == null || protos.Count == 0) return;
            for (int k = 0; k < stretches.Count; k++)
            {
                StreetSurface.Stretch s = stretches[k];
                foreach (var (side, block) in Sides(k))
                {
                    for (int end = 0; end < 2; end++)
                    {
                        var rng = Rng(60 + stream, k, 2 * side + end);
                        bool pick = rng.NextDouble() < share;
                        Proto p = protos[rng.Next(protos.Count)];
                        if (!pick || Usable(s) < 2f * fromEnd) continue;
                        float t = end == 0 ? s.start + r.junctionGap + fromEnd : s.end - r.junctionGap - fromEnd;
                        OnFootpath(kind, p, s, side, block, t, back ? (Inset)Back : (Inset)Kerb, 1.5f, Facing.Along, rng, 1f);
                    }
                }
            }
        }

        // ---------------------------------------------------------------- the lawns

        private const float LawnY = -0.02f;                // the lawn is the Terrain, just under 0
        private int sittingAreas;

        /// <summary>Plants every block's lawn: the sitting areas first (they need the most room), then hedges along the
        /// buildings, the groves and the single trees. Each block draws a fixed number of each from its lawn's own area, every
        /// one on a stream keyed on the block, so a level whose buildings take a spot loses that spot and nothing else.</summary>
        private void Lawns(List<Proto> trees, Proto hedge, Proto bench, Proto bin, Proto lamp, Proto bed)
        {
            var ids = new List<int>(blocks.Keys);
            ids.Sort();
            if (trees.Count > 0 && bench != null && d.pavementMaterial != null)
            {
                var paving = new MeshBuffer();
                foreach (int id in ids) SittingArea(blocks[id], paving, trees, bench, bin, lamp, bed, hedge);
                if (sittingAreas > 0) paving.Make(root, "Lawn paving", d.pavementMaterial);
            }
            if (hedge != null) Hedges(hedge);
            if (trees.Count == 0) return;
            foreach (int id in ids) Groves(blocks[id], trees);
            foreach (int id in ids) SingleTrees(blocks[id], trees);
        }

        /// <summary>Groves of one species: round(lawn / groveArea) of them, each of groveTrees trees within groveRadius of
        /// its centre and groveTrunkGap apart.</summary>
        private void Groves(Block b, List<Proto> trees)
        {
            int n = Mathf.RoundToInt(b.lawnArea / Mathf.Max(r.groveArea, 1f));
            float gap2 = r.groveTrunkGap * r.groveTrunkGap;
            for (int g = 0; g < n; g++)
            {
                var rng = Rng(80, b.id, g);
                bool found = OnLawnAt(b, rng, r.lawnEdgeMargin, out Vector2 c);
                Proto species = trees[rng.Next(trees.Count)];
                int count = rng.Next(r.groveTrees.x, r.groveTrees.y + 1);
                var trunks = new List<Vector2>();
                for (int t = 0; t < count; t++)
                {
                    Vector2 spot = Vector2.zero;
                    bool free = false;
                    for (int tries = 0; tries < 4; tries++)          // always four draws: the stream keeps step
                    {
                        Vector2 q = c + InDisc(rng, r.groveRadius);
                        if (!free && trunks.TrueForAll(o => (o - q).sqrMagnitude >= gap2))
                        {
                            spot = q;
                            free = true;
                        }
                    }
                    if (LawnTree(species, spot, b, rng, found && free)) trunks.Add(spot);
                }
            }
        }

        /// <summary>round(lawn / singleTreeArea) trees standing alone, each of a random species.</summary>
        private void SingleTrees(Block b, List<Proto> trees)
        {
            int n = Mathf.RoundToInt(b.lawnArea / Mathf.Max(r.singleTreeArea, 1f));
            for (int i = 0; i < n; i++)
            {
                var rng = Rng(81, b.id, i);
                bool found = OnLawnAt(b, rng, r.lawnEdgeMargin, out Vector2 at);
                LawnTree(trees[rng.Next(trees.Count)], at, b, rng, found);
            }
        }

        /// <summary>A lawn tree at a random yaw and size, its trunk lawnEdgeMargin inside the lawn; its canopy counts at
        /// 0.6 against other props, so a grove can close up, or not at all when <paramref name="overlap"/> (a formal
        /// planting laid out round a sitting area). The yaw and size are drawn whether or not it is tried.</summary>
        private bool LawnTree(Proto tree, Vector2 at, Block b, System.Random rng, bool attempt, bool overlap = false,
                              string kind = "Lawn trees")
        {
            float yaw = (float)rng.NextDouble() * 360f;
            float k = Mathf.Lerp(r.lawnTreeScale.x, r.lawnTreeScale.y, (float)rng.NextDouble());
            return attempt && Try(kind, tree, at, yaw, LawnY, k, Vector2.zero, 0f,
                                  bas => b.OnLawn(bas.c, r.lawnEdgeMargin), out _, shrink: 0.6f, overlap: overlap);
        }

        /// <summary>A hedge along one front of hedgeShare of the city buildings: segments laid end to end along the facade,
        /// hedgeOffset off it, as long as the front less a metre at each end. A segment not wholly on the lawn (a building
        /// near the kerb) is left out.</summary>
        private void Hedges(Proto hedge)
        {
            Vector2 size = hedge.bodyMax - hedge.bodyMin;
            float segment = Mathf.Max(size.x, size.y), thick = Mathf.Min(size.x, size.y);
            if (segment < 0.1f) return;
            foreach (var (box, city, name) in boxes)
            {
                if (!city) continue;
                var rng = Rng(82, Key(name), 0);
                bool pick = rng.NextDouble() < r.hedgeShare;
                int side = rng.Next(4);
                if (!pick) continue;
                Vector2 n = side < 2 ? box.u : box.v, t = side < 2 ? box.v : box.u;
                if ((side & 1) == 1) n = -n;
                float depth = n.magnitude, length = t.magnitude;
                int count = Mathf.FloorToInt((2f * length - 2f) / segment);
                Block b = BlockAt(box.c);
                if (b == null || count < 1 || depth < 1e-3f) continue;
                Vector2 nu = n / depth, tu = t / length;
                float yaw = YawFromTo(hedge.front, tu);
                for (int j = 0; j < count; j++)
                {
                    Vector2 at = box.c + nu * (depth + r.hedgeOffset + 0.5f * thick) + tu * ((j - 0.5f * (count - 1)) * segment);
                    // At 0.8 its segments clear each other, so it still keeps off a sitting area's path and planting.
                    Try("Hedges", hedge, at, yaw, LawnY, 1f, Vector2.zero, 0f, bas => OnLawn(bas, b, 0.2f), out _,
                        shrink: 0.8f, clearance: 0.05f);
                }
            }
        }

        /// <summary>Whether every corner of a base stands on the block's lawn, <paramref name="margin"/> inside it.</summary>
        private static bool OnLawn(Obb bas, Block b, float margin)
        {
            for (int c = 0; c < 4; c++)
            {
                if (!b.OnLawn(bas.Corner(c), margin)) return false;
            }
            return true;
        }

        /// <summary>In sittingShare of the blocks with sittingMinLawn of lawn: a paved round of sittingRadius, joined to the
        /// footpath by a path, furnished and planted by <see cref="Furnish"/>. The first of thirty drawn spots where it and
        /// its path stand clear of everything; the first fifteen are drawn deep in the lawn, so the path is long enough
        /// to be an avenue.</summary>
        private void SittingArea(Block b, MeshBuffer paving, List<Proto> trees, Proto bench, Proto bin, Proto lamp, Proto bed,
                                 Proto hedge)
        {
            var rng = Rng(90, b.id, 0);
            bool pick = rng.NextDouble() < r.sittingShare;
            if (!pick || b.lawnArea < r.sittingMinLawn) return;
            float R = r.sittingRadius;
            for (int tries = 0; tries < 30; tries++)
            {
                if (!OnLawnAt(b, rng, R + (tries < 15 ? 12f : 3f), out Vector2 c)) continue;
                var round = new Obb(c, new Vector2(R + 2f, 0f), new Vector2(0f, R + 2f));
                if (AtBuilding(round, 1f) || InGoal(c, R + 4f) || props.Near(round, 0f)) continue;
                if (!PathOut(b, c, R, out Obb path, out Vector2 dir)) continue;
                Furnish(b, c, dir, path, paving, trees[rng.Next(trees.Count)], rng, bench, bin, lamp, bed, hedge);
                return;
            }
        }

        /// <summary>The shortest path from a round at <paramref name="c"/> to the edge of the lawn that stands clear of the
        /// buildings, the goals and the props; it runs half a metre under the footpath, so the two meet without a seam.</summary>
        private bool PathOut(Block b, Vector2 c, float radius, out Obb path, out Vector2 dir)
        {
            var exits = new List<(float d, Vector2 q)>();
            for (int i = 0; i < b.lawn.Count; i++)
            {
                Vector2 q = ClosestOnSegment(c, b.lawn[i], b.lawn[(i + 1) % b.lawn.Count]);
                exits.Add(((q - c).magnitude, q));
            }
            exits.Sort((x, y) => x.d.CompareTo(y.d));
            foreach (var (dist, q) in exits)
            {
                if (dist > r.pathMaxLength) break;
                dir = (q - c) / dist;
                Vector2 from = c + dir * (radius - 0.5f), to = q + dir * 0.5f;
                float len = (to - from).magnitude;
                path = new Obb(0.5f * (from + to), dir * 0.5f * len, new Vector2(-dir.y, dir.x) * 0.5f * r.pathWidth);
                if (!AtBuilding(path, 0.5f) && !InGoal(path.c, 0.5f * len + r.pathWidth) && !props.Near(path, 0.2f))
                {
                    return true;
                }
            }
            path = default;
            dir = Vector2.zero;
            return false;
        }

        /// <summary>A sitting area: the paved round and its path; a bed with a tree in the middle, three benches facing it, a
        /// bin and a park lamp; and a formal planting of one species. The path is an avenue, and round the round stands one
        /// of a ring of trees, a round hedge inside a ring of trees, or a grid of trees (a bosque) behind it -- the hedged
        /// round where the grid does not fit.</summary>
        private void Furnish(Block b, Vector2 c, Vector2 dir, Obb path, MeshBuffer paving, Proto species, System.Random rng,
                             Proto bench, Proto bin, Proto lamp, Proto bed, Proto hedge)
        {
            sittingAreas++;
            float R = r.sittingRadius, y = r.pavementHeight, uv = Mathf.Max(d.pavementUvMetres, 0.1f);
            var rim = new List<Vector2>();
            for (int i = 0; i < 24; i++) rim.Add(c + new Vector2(Mathf.Cos(i * Mathf.PI / 12f), Mathf.Sin(i * Mathf.PI / 12f)) * R);
            paving.Floor(0, rim, y, uv);
            for (int i = 0; i < rim.Count; i++)
            {
                Vector2 p = rim[i], q = rim[(i + 1) % rim.Count];
                paving.Wall(0, p, q, -0.05f, y, (0.5f * (p + q) - c).normalized, uv);
            }
            paving.Floor(0, new List<Vector2> { path.Corner(0), path.Corner(1), path.Corner(3), path.Corner(2) }, y, uv);
            Vector2 side = path.v.normalized;
            paving.Wall(0, path.Corner(0), path.Corner(1), -0.05f, y, -side, uv);
            paving.Wall(0, path.Corner(2), path.Corner(3), -0.05f, y, side, uv);

            float phi = Mathf.Atan2(dir.y, dir.x);
            Vector2 At(float angle, float radius) => c + new Vector2(Mathf.Cos(phi + angle), Mathf.Sin(phi + angle)) * radius;
            bool Anywhere(Obb _) => true;
            float treeY = y;
            if (bed != null)
            {
                float width = Mathf.Max(bed.bodyMax.x - bed.bodyMin.x, bed.bodyMax.y - bed.bodyMin.y);
                float kb = Mathf.Clamp(3.6f / Mathf.Max(width, 0.1f), 0.2f, 5f);
                if (Try("Flower beds", bed, c, (float)rng.NextDouble() * 360f, y, kb, Vector2.zero, 0f, Anywhere, out _,
                        overlap: true))
                {
                    treeY = y + bed.height * kb;
                }
            }
            float treeK = Mathf.Lerp(r.lawnTreeScale.x, r.lawnTreeScale.y, (float)rng.NextDouble());
            Try("Lawn trees", species, c, (float)rng.NextDouble() * 360f, treeY, treeK, Vector2.zero, 0f, Anywhere, out _,
                overlap: true);
            for (int i = 1; i <= 3; i++)                        // the path comes in at angle 0
            {
                Vector2 at = At(i * 0.5f * Mathf.PI, R - 1.3f);
                if (!Try("Park benches", bench, at, YawFromTo(bench.front, c - at), y, 1f, Vector2.zero, 0f, Anywhere,
                         out Obb seat, overlap: true) || i != 2 || bin == null)
                {
                    continue;
                }
                Vector2 along = new Vector2(-(c - at).y, (c - at).x).normalized;
                float reach = Mathf.Abs(Vector2.Dot(seat.u, along)) + Mathf.Abs(Vector2.Dot(seat.v, along));
                Try("Park bins", bin, at + along * (reach + 0.8f), (float)rng.NextDouble() * 360f, y, 1f, Vector2.zero, 0f,
                    Anywhere, out _, overlap: true);
            }
            if (lamp != null)
            {
                Vector2 at = At(1.25f * Mathf.PI, R - 0.6f);
                Try("Park lamps", lamp, at, YawFromTo(lamp.front, c - at), y, 1f, Vector2.zero, 0f, Anywhere, out _,
                    overlap: true);
            }
            // The round and its path keep everything planted after them off (a square of 0.75 R stands in for the round:
            // with a canopy's margin, nothing then reaches the paving). The formal planting is laid out round them, so it
            // is placed without the spacing test, but still clear of buildings, goals and the footpath.
            props.Add(new Obb(c, new Vector2(0.75f * R, 0f), new Vector2(0f, 0.75f * R)));
            props.Add(path);
            Vector2 across = new Vector2(-dir.y, dir.x);
            float pathEnd = Vector2.Dot(path.c - c, dir) + path.u.magnitude;      // where the path meets the footpath
            for (float t = R + 2.5f; t <= pathEnd - 2.5f; t += r.avenueSpacing)
            {
                foreach (float s in new[] { -1f, 1f })
                {
                    LawnTree(species, c + dir * t + across * (s * (0.5f * r.pathWidth + 1.6f)), b, rng, true, overlap: true,
                             kind: "Avenue trees");
                }
            }
            int formation = rng.Next(3);
            if (formation == 2 && !Bosque(b, c, dir, species, rng)) formation = 1;
            if (formation == 2)
            {
                bosques++;
                return;
            }
            bool hedged = formation == 1 && hedge != null;
            if (hedged)
            {
                HedgedRound(b, c, dir, hedge);
                hedgedRounds++;
            }
            for (int i = 0; i < 7; i++)                         // 45 to 315 degrees: clear of the avenue
            {
                LawnTree(species, At((45f + 45f * i) * Mathf.Deg2Rad, R + (hedged ? 5.5f : 3.5f)), b, rng, true, overlap: true,
                         kind: "Ring trees");
            }
        }

        private int bosques, hedgedRounds;

        /// <summary>A grid of trees bosqueSpacing apart, square to the path: three rows of four, or else two of four, or two
        /// of three; on the far side of the round, or else on either side of it. The first that stands wholly on the lawn,
        /// clear of buildings, goals and the props already placed, is planted; false (nothing planted) if none does.</summary>
        private bool Bosque(Block b, Vector2 c, Vector2 dir, Proto species, System.Random rng)
        {
            float s = r.bosqueSpacing, near = r.sittingRadius + 4.5f;
            foreach ((int rows, int cols) in new[] { (3, 4), (2, 4), (2, 3) })
            {
                foreach (Vector2 away in new[] { -dir, new Vector2(-dir.y, dir.x), new Vector2(dir.y, -dir.x) })
                {
                    float deep = 0.5f * (rows - 1) * s, wide = 0.5f * (cols - 1) * s;
                    Vector2 across = new Vector2(-away.y, away.x), middle = c + away * (near + deep);
                    var area = new Obb(middle, away * (deep + 2f), across * (wide + 2f));
                    bool fits = !AtBuilding(area, 0.5f) && !InGoal(middle, area.Extent.magnitude) &&
                                !props.Near(new Obb(middle, away * deep, across * wide), 0.2f);
                    for (int k = 0; fits && k < 4; k++) fits = b.OnLawn(area.Corner(k), 0.5f);
                    if (!fits) continue;
                    for (int row = 0; row < rows; row++)
                    {
                        for (int col = 0; col < cols; col++)
                        {
                            LawnTree(species, c + away * (near + row * s) + across * (col * s - wide), b, rng, true,
                                     overlap: true, kind: "Bosque trees");
                        }
                    }
                    return true;
                }
            }
            return false;
        }

        /// <summary>A round hedge just outside the paving, its segments tangent to the circle, with two entrances on the
        /// path's axis (the path, and opposite it).</summary>
        private void HedgedRound(Block b, Vector2 c, Vector2 dir, Proto hedge)
        {
            Vector2 size = hedge.bodyMax - hedge.bodyMin;
            float segment = Mathf.Max(size.x, size.y), thick = Mathf.Min(size.x, size.y);
            if (segment < 0.1f) return;
            float radius = r.sittingRadius + 1f + 0.5f * thick;
            int n = Mathf.Max(8, Mathf.FloorToInt(2f * Mathf.PI * radius / segment));
            float phi = Mathf.Atan2(dir.y, dir.x);
            float opening = (0.5f * r.pathWidth + 0.8f + 0.5f * segment) / radius * Mathf.Rad2Deg;
            for (int i = 0; i < n; i++)
            {
                float a = (i + 0.5f) * 360f / n;                  // degrees from the path
                if (Mathf.Abs(Mathf.DeltaAngle(a, 0f)) < opening || Mathf.Abs(Mathf.DeltaAngle(a, 180f)) < opening) continue;
                float w = phi + a * Mathf.Deg2Rad;
                Vector2 at = c + new Vector2(Mathf.Cos(w), Mathf.Sin(w)) * radius;
                Vector2 tangent = new Vector2(-Mathf.Sin(w), Mathf.Cos(w));
                Try("Hedges", hedge, at, YawFromTo(hedge.front, tangent), LawnY, 1f, Vector2.zero, 0f,
                    bas => OnLawn(bas, b, 0.2f), out _, overlap: true);
            }
        }

        /// <summary>The first of eight random points of a block's lawn at least <paramref name="margin"/> inside it (always
        /// eight draws, so the stream keeps step); false if none is.</summary>
        private static bool OnLawnAt(Block b, System.Random rng, float margin, out Vector2 at)
        {
            at = Vector2.zero;
            bool found = false;
            for (int i = 0; i < 8; i++)
            {
                var q = new Vector2(Mathf.Lerp(b.lawnMin.x, b.lawnMax.x, (float)rng.NextDouble()),
                                    Mathf.Lerp(b.lawnMin.y, b.lawnMax.y, (float)rng.NextDouble()));
                if (!found && b.OnLawn(q, margin))
                {
                    at = q;
                    found = true;
                }
            }
            return found;
        }

        private static Vector2 InDisc(System.Random rng, float radius)
        {
            float a = (float)(rng.NextDouble() * 2.0 * Math.PI), rad = radius * Mathf.Sqrt((float)rng.NextDouble());
            return new Vector2(Mathf.Cos(a), Mathf.Sin(a)) * rad;
        }

        private static Vector2 ClosestOnSegment(Vector2 p, Vector2 a, Vector2 b)
        {
            Vector2 ab = b - a;
            float t = ab.sqrMagnitude < 1e-9f ? 0f : Mathf.Clamp01(Vector2.Dot(p - a, ab) / ab.sqrMagnitude);
            return a + ab * t;
        }

        private Block BlockAt(Vector2 p)
        {
            foreach (Block b in blocks.Values)
            {
                if (b.Contains(p, 0f)) return b;
            }
            return null;
        }

        /// <summary>A stable key for a name (string.GetHashCode is not promised to be).</summary>
        private static int Key(string s)
        {
            unchecked
            {
                uint h = 2166136261;
                foreach (char ch in s)
                {
                    h ^= ch;
                    h *= 16777619;
                }
                return (int)h;
            }
        }

        // ---------------------------------------------------------------- placing one prop

        /// <summary>Where across the footpath a prop's anchor goes, given its base's reach towards the road and towards the
        /// block from the anchor.</summary>
        private delegate float Inset(float towardsRoad, float towardsBlock);

        private float Kerb(float towardsRoad, float towardsBlock) => 0.3f + towardsRoad;

        private float Back(float towardsRoad, float towardsBlock) => r.pavementWidth - 0.25f - towardsBlock;

        private float Middle(float towardsRoad, float towardsBlock) =>
            Mathf.Clamp(0.5f * r.pavementWidth, Kerb(towardsRoad, towardsBlock), Mathf.Max(Kerb(towardsRoad, towardsBlock),
                                                                                          Back(towardsRoad, towardsBlock)));

        private bool OnFootpath(string kind, Proto p, StreetSurface.Stretch s, int side, Block block, float t, Inset inset,
                                float slide, Facing facing, System.Random rng, float k) =>
            OnFootpath(kind, p, s, side, block, t, inset, slide, facing, rng, k, out _);

        private bool OnFootpath(string kind, Proto p, StreetSurface.Stretch s, int side, Block block, float t, Inset inset,
                                float slide, Facing facing, System.Random rng, float k, out Obb body)
        {
            Vector2 road = -s.Left * side;                       // from the footpath towards the street
            float yaw = facing == Facing.Along
                ? YawFromTo(p.front, s.Dir) + (rng.NextDouble() < 0.5 ? 0f : 180f)
                : YawFromTo(p.front, road);
            Obb b0 = p.Base(Vector2.zero, yaw, k);
            float towardsRoad = 0f, towardsBlock = 0f;
            for (int c = 0; c < 4; c++)
            {
                towardsRoad = Mathf.Max(towardsRoad, Vector2.Dot(b0.Corner(c), road));
                towardsBlock = Mathf.Max(towardsBlock, -Vector2.Dot(b0.Corner(c), road));
            }
            float across = side * (0.5f * s.width + inset(towardsRoad, towardsBlock));
            float lo = s.start + r.junctionGap, hi = s.end - r.junctionGap;
            return Try(kind, p, s.At(t, across), yaw, r.pavementHeight, k, s.Dir, slide,
                       bas => FootpathFits(bas, s, side, block, lo, hi), out body, street: true);
        }

        /// <summary>A tree in a street (a median's, or a footpath's), at a random yaw and size, its trunk where
        /// <paramref name="fits"/> allows (any canopy may overhang the road).</summary>
        private void TreeAt(string kind, Proto tree, Vector2 at, float y, System.Random rng, float slide, Vector2 dir,
                            Func<Obb, bool> fits)
        {
            float yaw = (float)rng.NextDouble() * 360f;
            float k = Mathf.Lerp(r.treeScale.x, r.treeScale.y, (float)rng.NextDouble());
            Try(kind, tree, at, yaw, y, k, dir, slide, fits, out _, street: true);
        }

        /// <summary>DiamondCityWorld's verge rows: a tree every borderSpacing (borderJitter of it either way) borderInset
        /// from the kerb, all round every goal's block (borderAtGoals) and borderShare of the others, stopping
        /// borderCornerClearance short of each end of the plain road so the junctions stay open. Planted after the street
        /// furniture, which keeps its places; a tree that would stand on a prop, at a building or on a goal's walkers is
        /// left out. A block is drawn once, so a row runs all the way round it.</summary>
        private void BorderTrees(Proto tree)
        {
            if (tree == null || (r.borderShare <= 0f && !r.borderAtGoals)) return;
            var goalBlocks = new HashSet<int>();
            foreach (GoalZone g in goals) goalBlocks.Add(g.block);
            float spacing = Mathf.Max(r.borderSpacing, 0.5f);
            for (int k = 0; k < stretches.Count; k++)
            {
                StreetSurface.Stretch s = stretches[k];
                foreach (var (side, block) in Sides(k))
                {
                    bool picked = (r.borderAtGoals && goalBlocks.Contains(block.id)) ||
                                  Rng(70, block.id, 0).NextDouble() < r.borderShare;
                    float lo = s.start + r.junctionGap + r.borderCornerClearance;
                    float hi = s.end - r.junctionGap - r.borderCornerClearance;
                    if (!picked || hi < lo) continue;
                    var rng = Rng(71, k, side);
                    int n = Mathf.FloorToInt((hi - lo) / spacing) + 1;
                    float first = 0.5f * (lo + hi - (n - 1) * spacing);
                    for (int i = 0; i < n; i++)
                    {
                        float shift = (2f * (float)rng.NextDouble() - 1f) * r.borderJitter * spacing;
                        float yaw = (float)rng.NextDouble() * 360f;
                        float size = 1f + (2f * (float)rng.NextDouble() - 1f) * r.borderScaleJitter;
                        float t = Mathf.Clamp(first + i * spacing + shift, lo, hi);
                        Try("Border trees", tree, s.At(t, side * (0.5f * s.width + r.borderInset)), yaw, r.pavementHeight,
                            size, Vector2.zero, 0f,
                            bas => FootpathFits(bas, s, side, block, s.start + r.junctionGap, s.end - r.junctionGap),
                            out _, shrink: BorderTreeShrink, street: true);
                    }
                }
            }
        }

        /// <summary>Whether a base stands on this street's footpath: inside the block's kerb, within the footpath's width of
        /// the street's kerb, and between its crossings.</summary>
        private bool FootpathFits(Obb bas, StreetSurface.Stretch s, int side, Block block, float lo, float hi)
        {
            for (int c = 0; c < 4; c++)
            {
                Vector2 p = bas.Corner(c);
                float across = side * Vector2.Dot(p - s.a, s.Left) - 0.5f * s.width;
                float along = Vector2.Dot(p - s.a, s.Dir);
                if (!block.Contains(p, 0.15f) || across < 0.15f || across > r.pavementWidth - 0.1f ||
                    along < lo || along > hi)
                {
                    return false;
                }
            }
            return true;
        }

        /// <summary>Stands a prop at <paramref name="at"/> (its anchor), or up to <paramref name="slide"/> either way along
        /// <paramref name="dir"/>; false, counted by reason, if no spot fits. Against the other props it counts as its body
        /// times <paramref name="shrink"/> (a tree's canopy may overlap its neighbours'), or not at all when
        /// <paramref name="overlap"/> (the pieces of one sitting area, a hedge's segments); <paramref name="clearance"/> overrides the building
        /// clearance (a hedge along a facade). A <paramref name="street"/> prop keeps off the goals as
        /// <see cref="InGoalWay"/> says, any other out of their discs.</summary>
        private bool Try(string kind, Proto p, Vector2 at, float yaw, float y, float k, Vector2 dir, float slide,
                         Func<Obb, bool> fits, out Obb body, float shrink = 1f, float clearance = -1f, bool overlap = false,
                         bool street = false)
        {
            string why = null;
            body = default;
            foreach (float shift in slide > 0f ? new[] { 0f, slide, -slide } : new[] { 0f })
            {
                Vector2 q = at + dir * shift;
                Obb b = p.Body(q, yaw, k), mine = new Obb(b.c, b.u * shrink, b.v * shrink);
                string reason = !fits(p.Base(q, yaw, k)) ? "off its footpath or lawn"
                              : AtBuilding(b, clearance >= 0f ? clearance : r.buildingClearance) ? "at a building"
                              : (street ? InGoalWay(b) : InGoal(b.c, b.Extent.magnitude)) ? "in a goal patch"
                              : !overlap && props.Near(mine, r.propGap) ? "at another prop"
                              : null;
                if (reason == null)
                {
                    props.Add(mine);
                    Spawn(kind, p, q, yaw, y, k);
                    body = b;
                    return true;
                }
                why = why ?? reason;            // the spot asked for, not the slides
            }
            skipped[why] = (skipped.TryGetValue(why, out int n) ? n : 0) + 1;
            return false;
        }

        private bool InGoal(Vector2 p, float pad)
        {
            foreach (GoalZone g in goals)
            {
                if ((p - g.c).sqrMagnitude < (g.radius + pad) * (g.radius + pad)) return true;
            }
            return false;
        }

        /// <summary>Whether a street prop's body stands in a goal's way: on its walkers' loops where its block holds the
        /// goal, so its streets are dressed like any others; anywhere in its disc where the goal's ring stands across the
        /// streets (the layouts made before citygen's goal_cell), as every prop always was.</summary>
        private bool InGoalWay(Obb body)
        {
            foreach (GoalZone g in goals)
            {
                if (!g.held)
                {
                    float reach = g.radius + body.Extent.magnitude;
                    if ((body.c - g.c).sqrMagnitude < reach * reach) return true;
                    continue;
                }
                foreach (Obb w in g.walks)
                {
                    if (Obb.Near(w, body, 0f)) return true;
                }
            }
            return false;
        }

        /// <summary>Whether every corner of <paramref name="o"/> stands inside the block's kerb.</summary>
        private static bool Within(Block b, Obb o)
        {
            for (int k = 0; k < 4; k++)
            {
                if (!b.Contains(o.Corner(k), 0f)) return false;
            }
            return true;
        }

        /// <summary>The prop's meshes on fresh objects (one, or a root with one per mesh), its anchor on
        /// <paramref name="at"/> and its lowest point at <paramref name="y"/>.</summary>
        private void Spawn(string kind, Proto p, Vector2 at, float yaw, float y, float k)
        {
            if (!folders.TryGetValue(kind, out Transform folder))
            {
                folder = new GameObject(kind).transform;
                folder.SetParent(root, false);
                folders[kind] = folder;
                kinds.Add(kind);
                placed[kind] = 0;
            }
            string name = $"{p.name}_{placed[kind]:D3}";
            placed[kind]++;
            Quaternion q = Quaternion.AngleAxis(yaw, Vector3.up);
            Vector3 anchor = q * new Vector3(p.anchor.x, 0f, p.anchor.y) * k;
            Vector3 origin = new Vector3(at.x - anchor.x, y - p.minY * k, at.y - anchor.z);
            // One object per mesh, straight under the kind's folder, with no root to hold a tree's bark and leaves together:
            // the bake is saved with the scene, and a root per tree would be a third of the lawns' objects.
            for (int i = 0; i < p.parts.Count; i++)
            {
                Part(p.parts[i], folder, p.parts.Count == 1 ? name : $"{name}_{i}", origin, q, k);
            }
        }

        /// <summary>One mesh of a prop. Its parents are unrotated and unscaled, so the source's world scale is its own.</summary>
        private static GameObject Part((Mesh mesh, Material[] mats, Vector3 pos, Quaternion rot, Vector3 scale) part,
                                       Transform parent, string name, Vector3 origin, Quaternion q, float k)
        {
            var go = new GameObject(name);
            go.transform.SetParent(parent, false);
            go.transform.SetPositionAndRotation(origin + q * (part.pos * k), q * part.rot);
            go.transform.localScale = part.scale * k;
            go.AddComponent<MeshFilter>().sharedMesh = part.mesh;
            MeshRenderer mr = go.AddComponent<MeshRenderer>();
            mr.sharedMaterials = part.mats;
            mr.shadowCastingMode = ShadowCastingMode.Off;
            mr.receiveShadows = false;
            return go;
        }

        // ---------------------------------------------------------------- measuring a prop

        /// <summary>The prop as it stands in its source, or null (noted) if it has no mesh.</summary>
        private Proto Measure(GameObject src, Facing facing) =>
            src == null ? null : Measure(src.name, new[] { src }, facing);

        /// <summary>A prop made of several objects (a green-belt tree's bark and leaves), as they stand relative to the first.</summary>
        private Proto Measure(PropModel model, Facing facing)
        {
            if (model?.parts == null || model.parts.Length == 0 || model.parts[0] == null) return null;
            return Measure(model.parts[0].name, model.parts, facing);
        }

        private Proto Measure(string name, GameObject[] sources, Facing facing)
        {
            var p = new Proto { name = name };
            Vector3 origin = sources[0].transform.position;
            var pts = new List<Vector3>();
            foreach (GameObject src in sources)
            {
                if (src == null) continue;
                foreach (MeshFilter f in src.GetComponentsInChildren<MeshFilter>(true))
                {
                    MeshRenderer mr = f.GetComponent<MeshRenderer>();
                    if (f.sharedMesh == null || mr == null || !mr.enabled || !ActiveUnder(f.transform, src.transform)) continue;
                    Transform t = f.transform;
                    var part = (f.sharedMesh, Swap(mr.sharedMaterials), t.position - origin, t.rotation, t.lossyScale);
                    p.parts.Add(part);
                    Matrix4x4 m = Matrix4x4.TRS(part.Item3, part.Item4, part.Item5);
                    foreach (Vector3 v in Vertices(f.sharedMesh)) pts.Add(m.MultiplyPoint3x4(v));
                }
            }
            if (pts.Count == 0)
            {
                notes.Add($"{name} has no mesh to copy");
                return null;
            }
            float minY = float.MaxValue, maxY = float.MinValue;
            foreach (Vector3 v in pts)
            {
                minY = Mathf.Min(minY, v.y);
                maxY = Mathf.Max(maxY, v.y);
            }
            float h = Mathf.Max(maxY - minY, 1e-3f);
            Vector2 Centroid(float y0, float y1, out Vector2 lo, out Vector2 hi)
            {
                Vector2 sum = Vector2.zero;
                lo = Vector2.positiveInfinity;
                hi = Vector2.negativeInfinity;
                int n = 0;
                foreach (Vector3 v in pts)
                {
                    float f = (v.y - minY) / h;
                    if (f < y0 || f > y1) continue;
                    var xz = new Vector2(v.x, v.z);
                    sum += xz;
                    lo = Vector2.Min(lo, xz);
                    hi = Vector2.Max(hi, xz);
                    n++;
                }
                return n > 0 ? sum / n : Vector2.zero;
            }
            Vector2 anchor = Centroid(0f, 0.15f, out Vector2 baseLo, out Vector2 baseHi);
            Centroid(0f, 1f, out Vector2 bodyLo, out Vector2 bodyHi);
            Vector2 centre = 0.5f * (bodyLo + bodyHi);
            Vector2 f2;
            switch (facing)
            {
                case Facing.Arm:
                    f2 = Centroid(0.7f, 1f, out _, out _) - anchor;
                    break;
                case Facing.Back:
                    f2 = centre - Centroid(0.7f, 1f, out _, out _);
                    break;
                case Facing.Open:
                    f2 = Centroid(0.2f, 0.6f, out _, out _) - centre;
                    break;
                case Facing.Along:
                    f2 = bodyHi.x - bodyLo.x >= bodyHi.y - bodyLo.y ? Vector2.right : Vector2.up;
                    break;
                default:
                    f2 = Vector2.up;
                    break;
            }
            // The pack's models are square to their own axes: snap what was measured to the nearer one.
            p.front = Mathf.Abs(f2.x) >= Mathf.Abs(f2.y) ? new Vector2(Mathf.Sign(f2.x), 0f) : new Vector2(0f, Mathf.Sign(f2.y));
            if (f2.sqrMagnitude < 1e-8f) p.front = Vector2.up;
            p.anchor = anchor;
            p.minY = minY;
            p.height = h;
            p.bodyMin = bodyLo - anchor;
            p.bodyMax = bodyHi - anchor;
            p.baseMin = baseLo - anchor;
            p.baseMax = baseHi - anchor;
            return p;
        }

        /// <summary>A part's materials, any blended one (the trees' leaves) replaced by <see cref="cutoutLeaves"/>.</summary>
        private Material[] Swap(Material[] mats)
        {
            if (d.cutoutLeaves == null) return mats;
            var o = (Material[])mats.Clone();
            for (int i = 0; i < o.Length; i++)
            {
                if (o[i] != null && o[i].renderQueue > (int)RenderQueue.GeometryLast) o[i] = d.cutoutLeaves;
            }
            return o;
        }

        private static bool ActiveUnder(Transform t, Transform top)
        {
            for (; t != null; t = t.parent)
            {
                if (!t.gameObject.activeSelf) return false;
                if (t == top) return true;
            }
            return true;
        }

        /// <summary>A mesh's vertices, or its bounds' corners where they cannot be read (an unreadable mesh in a player).</summary>
        private static IEnumerable<Vector3> Vertices(Mesh mesh)
        {
            Vector3[] vs = null;
            if (mesh.isReadable || Application.isEditor)
            {
                try
                {
                    vs = mesh.vertices;
                }
                catch (Exception)
                {
                    vs = null;
                }
            }
            if (vs != null && vs.Length > 0) return vs;
            Bounds b = mesh.bounds;
            var corners = new Vector3[8];
            for (int i = 0; i < 8; i++)
            {
                corners[i] = new Vector3((i & 1) == 0 ? b.min.x : b.max.x, (i & 2) == 0 ? b.min.y : b.max.y,
                                         (i & 4) == 0 ? b.min.z : b.max.z);
            }
            return corners;
        }

        // ---------------------------------------------------------------- bookkeeping

        /// <summary>Its own random stream for every decision, keyed on what is being decided and where: a street that keeps its
        /// id keeps its dressing whatever else changes.</summary>
        private System.Random Rng(int what, int street, int slot)
        {
            unchecked
            {
                uint h = 2166136261;
                foreach (int x in new[] { seed, what, street, slot })
                {
                    h ^= (uint)x;
                    h *= 16777619;
                    h ^= h >> 13;
                }
                return new System.Random((int)h);
            }
        }

        private string Summary()
        {
            var sb = new StringBuilder();
            sb.Append($"{pavementMetres:F0} m of footpath round {blocks.Count} blocks, {medianCount} medians " +
                      $"({medianMetres:F0} m), {sittingAreas} sitting areas ({hedgedRounds} hedged, {bosques} with a bosque, " +
                      $"{sittingAreas - hedgedRounds - bosques} ringed)");
            foreach (string kind in kinds) sb.Append($", {placed[kind]} {kind.ToLowerInvariant()}");
            int total = 0;
            foreach (int n in placed.Values) total += n;
            sb.Append($" ({total} props");
            if (skipped.Count > 0)
            {
                sb.Append("; spots skipped: ");
                bool first = true;
                foreach (var kv in skipped)
                {
                    sb.Append(first ? "" : ", ").Append($"{kv.Value} {kv.Key}");
                    first = false;
                }
            }
            sb.Append(')');
            foreach (string n in notes) sb.Append("; ").Append(n);
            return sb.ToString();
        }
    }

    // ------------------------------------------------------------------ geometry

    /// <summary>The part of a convex polygon where n.x >= c (Sutherland-Hodgman; keeps its winding).</summary>
    private static List<Vector2> Clip(List<Vector2> poly, Vector2 n, float c)
    {
        var o = new List<Vector2>(poly.Count + 1);
        for (int i = 0; i < poly.Count; i++)
        {
            Vector2 p = poly[i], q = poly[(i + 1) % poly.Count];
            float dp = Vector2.Dot(n, p) - c, dq = Vector2.Dot(n, q) - c;
            if (dp >= 0f) o.Add(p);
            if ((dp >= 0f) != (dq >= 0f)) o.Add(p + (q - p) * (dp / (dp - dq)));
        }
        return o;
    }

    private static List<Vector2> Dedupe(List<Vector2> poly)
    {
        var o = new List<Vector2>(poly.Count);
        foreach (Vector2 p in poly)
        {
            if (o.Count == 0 || (p - o[o.Count - 1]).sqrMagnitude > 1e-6f) o.Add(p);
        }
        while (o.Count > 1 && (o[0] - o[o.Count - 1]).sqrMagnitude <= 1e-6f) o.RemoveAt(o.Count - 1);
        return o;
    }

    /// <summary>Floors and walls for the footpath and median meshes, in submeshes, world-mapped.</summary>
    private sealed class MeshBuffer
    {
        private readonly List<Vector3> verts = new List<Vector3>();
        private readonly List<Vector3> normals = new List<Vector3>();
        private readonly List<Vector2> uvs = new List<Vector2>();
        private readonly List<int>[] tris = { new List<int>(), new List<int>() };

        /// <summary>A convex polygon at height <paramref name="y"/>, facing up, mapped (x, z) / <paramref name="metres"/>.</summary>
        public void Floor(int sub, List<Vector2> poly, float y, float metres)
        {
            poly = Dedupe(poly);
            if (poly.Count < 3) return;
            int i0 = verts.Count;
            foreach (Vector2 p in poly)
            {
                verts.Add(new Vector3(p.x, y, p.y));
                normals.Add(Vector3.up);
                uvs.Add(p / metres);
            }
            for (int i = 1; i + 1 < poly.Count; i++) Tri(sub, i0, i0 + i, i0 + i + 1, Vector3.up);
        }

        /// <summary>A vertical quad over a - b from <paramref name="y0"/> to <paramref name="y1"/>, facing
        /// <paramref name="facing"/> ((x, z)).</summary>
        public void Wall(int sub, Vector2 a, Vector2 b, float y0, float y1, Vector2 facing, float metres)
        {
            float len = (b - a).magnitude;
            if (len < 1e-3f) return;
            var n = new Vector3(facing.x, 0f, facing.y);
            int i0 = verts.Count;
            verts.Add(new Vector3(a.x, y0, a.y));
            verts.Add(new Vector3(b.x, y0, b.y));
            verts.Add(new Vector3(b.x, y1, b.y));
            verts.Add(new Vector3(a.x, y1, a.y));
            for (int i = 0; i < 4; i++) normals.Add(n);
            uvs.Add(new Vector2(0f, y0 / metres));
            uvs.Add(new Vector2(len / metres, y0 / metres));
            uvs.Add(new Vector2(len / metres, y1 / metres));
            uvs.Add(new Vector2(0f, y1 / metres));
            Tri(sub, i0, i0 + 1, i0 + 2, n);
            Tri(sub, i0, i0 + 2, i0 + 3, n);
        }

        /// <summary>A triangle wound to face <paramref name="facing"/> (Unity: clockwise seen from the front).</summary>
        private void Tri(int sub, int a, int b, int c, Vector3 facing)
        {
            Vector3 n = Vector3.Cross(verts[b] - verts[a], verts[c] - verts[a]);
            if (n.sqrMagnitude < 1e-12f) return;
            tris[sub].Add(a);
            if (Vector3.Dot(n, facing) > 0f)
            {
                tris[sub].Add(b);
                tris[sub].Add(c);
            }
            else
            {
                tris[sub].Add(c);
                tris[sub].Add(b);
            }
        }

        /// <summary>The mesh on a new child of <paramref name="parent"/>, one material per submesh, no collider, no shadows.</summary>
        public void Make(Transform parent, string name, params Material[] materials)
        {
            var mesh = new Mesh { name = name, indexFormat = IndexFormat.UInt32 };
            mesh.SetVertices(verts);
            mesh.SetNormals(normals);
            mesh.SetUVs(0, uvs);
            mesh.subMeshCount = materials.Length;
            for (int s = 0; s < materials.Length; s++) mesh.SetTriangles(tris[s], s);
            mesh.RecalculateTangents();
            mesh.RecalculateBounds();
            var go = new GameObject(name);
            go.transform.SetParent(parent, false);
            go.AddComponent<MeshFilter>().sharedMesh = mesh;
            MeshRenderer mr = go.AddComponent<MeshRenderer>();
            mr.sharedMaterials = materials;
            mr.shadowCastingMode = ShadowCastingMode.Off;
            mr.receiveShadows = false;
        }
    }
}
