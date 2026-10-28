using System;
using System.Collections.Generic;
using System.IO;
using UnityEngine;

/// <summary>
/// Builds a city from a generated layout file (Assets/Scripts/Experiment/layouts/&lt;id&gt;.json, written by the Python
/// generator citysim/citygen.py): every building a copy of a ScaledCity pack building from the
/// <see cref="BuildingCatalogue"/>, the streets as flat meshes, and the goals as <see cref="ProceduralGoal"/> roots
/// whose ring buildings carry <see cref="WalkerPatrol"/>s. Sits on its own scene root (<c>ProceduralCity</c>), never on
/// <c>gameManager</c>, which <see cref="SwarmManager"/> moves to DontDestroyOnLoad.
///
/// <para><b>Placement is by collider centre.</b> A pack building is authored local +Z up with the pivot at its base
/// (see the catalogue), so it is given the rotation <c>Euler(-90, yaw, 0)</c>, its design's lossy scale with the two
/// horizontal axes times the layout's size, and the position that puts its BoxCollider's centre exactly on the
/// layout's box centre; the result is asserted. That is the frame the layout's half-axes were written in
/// (citysim/layout.py), so CityObstacleExport of the built scene reproduces the layout's boxes.</para>
///
/// <para><b>Nothing here may collide unless the swarm can see it</b>: buildings stay on <c>Obstacle</c>, streets have no
/// collider, and the only street furniture is a <see cref="CityDressing"/>'s, which copies meshes and never colliders — in
/// the tile cities, lamps and hydrants on Default killed drones the avoidance law cannot see (internal_3's one non-pair
/// crash). A root with a CityDressing is dressed after every build, bake and adopt (an adopt redresses only a stale
/// dressing); one without (ProceduralCityWorld, the density study's look) is not.</para>
///
/// <para><b>The goals' order and randomness come from the layout.</b> <see cref="PlacedGoals"/> is in the layout's goal
/// order (the recorder's goal index). The session's goal seed (<see cref="GoalSession"/>) is the layout's seed, set
/// before any goal root is activated, so the hats, the target walkers and the walkers' starts are the same every
/// time a layout is flown, whichever interface flies it. Each goal root is instantiated under an inactive holder and
/// activated only once its buildings and patrols are in place: <see cref="GoalSpecialWalker"/> picks its patrol in
/// Awake, before any patrol's Start.</para>
///
/// <para>The layout's arena and spawn must stand on the scene's Terrain: past it there is no ground for the altitude
/// ceiling (<see cref="TerrainHeightSampler"/>), so the build refuses a layout that leaves it.</para>
///
/// <para><b>A building may instead be a block-shaped prism</b> (<c>type: ConvexPrism</c>, S4's variant B from
/// citysim/blockcity.py): a footprint and a height with no catalogue design, built by <see cref="PrismBuilding"/> in
/// <see cref="prismStyle"/>'s materials, standing on its footprint's centroid. A layout of prisms needs no catalogue.</para>
///
/// <para><b>A city can be baked into the scene</b> (<see cref="Bake"/>, edit mode): the buildings and streets are built
/// and saved with it, so the editor shows the city without pressing Play — VoronoiCityWorld is made this way. The goals
/// cannot be baked whole: <see cref="GoalSpecialWalker"/> deals its hat in Awake, from state shared across the session,
/// so a goal root saved active would deal at scene load in no particular order. A bake therefore holds only the goals'
/// buildings (under <c>Goal buildings</c>), and at Play <see cref="Awake"/> adopts it: the goal roots are made exactly as a
/// fresh build makes them, with the baked buildings moved under them. The layout FILE stays the truth, because the
/// analysis tools read it: a bake is adopted only if it is of the layout this Awake would build and every building still
/// stands on its box; otherwise the city is rebuilt from the file, with a warning.</para>
/// </summary>
[DisallowMultipleComponent]
public class ProceduralCityBuilder : MonoBehaviour
{
    public const string LayoutEnv = "CITY_LAYOUT";
    /// <summary>"0" turns the ring buildings' pedestrians off whatever spawnWalkers says (the headless bench sets it:
    /// the builder runs in Awake, before the bench's own setup).</summary>
    public const string WalkersEnv = "CITY_WALKERS";
    public const string LayoutFolder = "Scripts/Experiment/layouts";

    [Tooltip("The layout: an id or file name under Assets/Scripts/Experiment/layouts, a path under Assets/, or an " +
             "absolute path. The CITY_LAYOUT environment variable overrides it (the headless bench).")]
    public string layoutPath = "";
    public BuildingCatalogue catalogue;
    [Tooltip("ProceduralGoalRoot.prefab: GoalSpecialWalker (with its hat prefabs) and ProceduralGoal on the root.")]
    public GameObject goalRootPrefab;
    [Tooltip("The ordinary pedestrian every patrol spawns (SimpleWalker).")]
    public GameObject walkerPrefab;
    public bool buildOnAwake = true;
    [Tooltip("Off: no WalkerPatrol on the ring buildings (the headless bench flies without pedestrians).")]
    public bool spawnWalkers = true;
    [Tooltip("Move swarmSpawn's start to the layout's spawn point.")]
    public bool setSpawn = true;

    [Header("Patrols (WalkerPatrol on each ring building)")]
    public int numWalkers = 8;
    public float walkerSpeed = 2f;
    public float baseOffset = 0.5f;
    public float margin = 3f;
    public float cornerRadius = 1f;

    public enum StreetStyle
    {
        /// <summary>One flat quad per street in <see cref="streetMaterial"/> (the density study's look).</summary>
        Flat,
        /// <summary>Road: lane-marked stretches, plain junctions, zebra crossings (<see cref="StreetSurface"/>).</summary>
        Textured,
    }

    [Header("Streets")]
    public StreetStyle streetStyle = StreetStyle.Flat;
    [Tooltip("Flat: the street colour. Textured: the lane-marked road (MC_Asphalt_Road Texture_M_01, as on ParkRoad).")]
    public Material streetMaterial;
    [Tooltip("Textured: junctions and crossings (MC_Asphalt_Cross and Corner). Without it the streets are Flat.")]
    public Material junctionMaterial;
    public float streetHeight = 0.02f;
    [Tooltip("Textured: road texture tiles across a street (ParkRoad lays one per 7 m lane).")]
    public int streetTilesAcross = 2;
    [Tooltip("Textured: depth of a zebra crossing along its street (m).")]
    public float crossingDepth = 3.3f;

    [Header("Block-shaped buildings (ConvexPrism: S4's variant B)")]
    [Tooltip("Their materials and facade grid (BlockShapedCityWorldBuilder fills it). Unused by catalogue layouts.")]
    public PrismBuilding.Style prismStyle = new PrismBuilding.Style();

    [Tooltip("Written by Bake: the layout (as layoutPath named it) whose buildings and streets this scene holds. Adopted " +
             "at Play while it is still the layout to build; empty when nothing is baked.")]
    [SerializeField] private string bakedLayout = "";

    private const string BuildingsName = "Buildings", StreetsName = "Streets", GoalsName = "Goals",
                         GoalBuildingsName = "Goal buildings", DressingName = CityDressing.RootName;
    private const float AdoptTolerance = 1e-2f;   // m: how far a baked collider centre may sit from its box
    private readonly List<GameObject> placedGoals = new List<GameObject>();
    private Dictionary<string, PrismBuilding.Look> prismLooks = new Dictionary<string, PrismBuilding.Look>();
    private string prismNote = "";

    /// <summary>The layout whose city is baked into the scene, or "".</summary>
    public string BakedLayout => bakedLayout;

    /// <summary>The goal roots in the layout's goal order. Empty until Build has run.</summary>
    public IReadOnlyList<GameObject> PlacedGoals => placedGoals;

    /// <summary>The parsed layout file of the last build (null before).</summary>
    public LayoutFile Layout { get; private set; }

    /// <summary>The layout file the last build read.</summary>
    public string ResolvedPath { get; private set; } = "";

    /// <summary>How the Textured road is drawn: its setbacks, crossings and merged junctions, which a
    /// <see cref="CityDressing"/> keeps its props off (<see cref="StreetSurface.Stretches"/>).</summary>
    public StreetSurface.Settings RoadSettings => new StreetSurface.Settings
    {
        height = streetHeight,
        tilesAcross = Mathf.Max(streetTilesAcross, 1),
        crossingDepth = crossingDepth,
        maxSetback = 1.5f,
        minStrip = 1f,
    };

    void Awake()
    {
        if (!Application.isPlaying) return;
        bool baked = HasBake;
        if (!baked && !buildOnAwake) return;
        string error;
        if (baked)
        {
            if (Adopt(out error)) return;
            Debug.LogWarning($"ProceduralCityBuilder: not using the city baked into the scene ({error}); building " +
                             $"'{LayoutName}' from its file instead.", this);
        }
        if (!Build(out error))
        {
            Debug.LogError("ProceduralCityBuilder: " + error, this);
        }
    }

    /// <summary>The layout to build: CITY_LAYOUT if set, else <see cref="layoutPath"/>.</summary>
    public string LayoutName
    {
        get
        {
            string name = Environment.GetEnvironmentVariable(LayoutEnv);
            return string.IsNullOrEmpty(name) ? layoutPath : name;
        }
    }

    private bool HasBake => !string.IsNullOrEmpty(bakedLayout) && transform.Find(BuildingsName) != null;

    // ------------------------------------------------------------------ the file

    [Serializable]
    public class LayoutFile
    {
        public LayoutBox[] boxes;
        public LayoutStreet[] streets;
        public LayoutRecord layout;
    }

    [Serializable]
    public class LayoutBox
    {
        public string name;
        public string tile;
        public float[] c;            // collider centre, world
        public string design;
        public float size;           // horizontal scale of the design
        public int yaw;              // 0 or 90
        public string role;          // city, goal_centre, goal_ring
        public int goal;             // -1 for city buildings
        public bool patrol;
        public string type;          // "ConvexPrism" (PrismBuilding), else a catalogue box
        public float[] poly_xz;      // ConvexPrism: the footprint, flat (x, z), counter-clockwise
        public float height;         // ConvexPrism
    }

    [Serializable]
    public class LayoutStreet
    {
        public int id;
        public float[] xz;           // flattened polyline
        public float width;
        public bool edge;
        public int[] blocks;         // the blocks either side (citygen's cell ids; -1 beyond the arena's edge)
    }

    [Serializable]
    public class LayoutGoal
    {
        public int index;
        public int block_id = -1;    // the block (citygen's cell id) it stands in
        public float[] centre;
        public string centre_box;
        public string[] ring_boxes;
        public float w_c, R_c, r_g, c_g, clear_radius, ring_size;
        public int n_b;
    }

    [Serializable]
    public class LayoutRecord
    {
        public string generator_version;
        public string id;
        public int seed;
        public string mode;
        public string design;
        public float size, p, s, f, g, alpha_achieved, alpha_target;
        public float[] arena;
        public float[] spawn;
        public float spawn_yaw;
        public LayoutGoal[] goals;
    }

    /// <summary>The file a layout name resolves to, or null.</summary>
    public static string Resolve(string name)
    {
        if (string.IsNullOrWhiteSpace(name)) return null;
        name = name.Trim();
        string folder = Path.Combine(Application.dataPath, LayoutFolder);
        string[] candidates =
        {
            name,
            Path.Combine(folder, name),
            Path.Combine(folder, name + ".json"),
            Path.Combine(Application.dataPath, name),
        };
        foreach (string c in candidates)
        {
            if (File.Exists(c)) return Path.GetFullPath(c);
        }
        return null;
    }

    // ------------------------------------------------------------------ build

    private delegate GameObject Obtain(LayoutBox b, Transform parent, out string error);

    /// <summary>Builds the layout under this root, replacing any previous build (a bake included). False with an error
    /// on failure.</summary>
    public bool Build(out string error)
    {
        if (!Load(out LayoutFile file, out string path, out error)) return false;
        Clear();
        Layout = file;
        ResolvedPath = path;
        Transform buildings = Child(BuildingsName);
        BuildStreets(file.streets, Child(StreetsName));
        int nCity = PlaceCity(file, buildings, out error);
        if (nCity < 0) return false;
        if (!BuildGoals(file, Child(GoalsName), Place, out error)) return false;
        if (Application.isPlaying)
        {
            StaticBatchingUtility.Combine(buildings.gameObject);
        }
        Finish(file, path, buildings, nCity, "built", Dress(file));
        return true;
    }

    /// <summary>The street dressing, if this root has a <see cref="CityDressing"/>: its summary for the log line, or "".</summary>
    private string Dress(LayoutFile file)
    {
        CityDressing dressing = GetComponent<CityDressing>();
        return dressing == null ? "" : dressing.Dress(this, file);
    }

#if UNITY_EDITOR
    /// <summary>Edit mode: dresses the baked city again (its <see cref="CityDressing"/>, after a change to its rules),
    /// leaving the buildings and streets as they are. False with the reason in <paramref name="summary"/>.</summary>
    public bool Redress(out string summary)
    {
        CityDressing dressing = GetComponent<CityDressing>();
        if (Application.isPlaying || dressing == null || !HasBake)
        {
            summary = Application.isPlaying ? "redress in edit mode." : dressing == null ? "no CityDressing on this root."
                    : "nothing is baked to dress.";
            return false;
        }
        if (!Load(bakedLayout, out LayoutFile file, out string path, out summary)) return false;
        summary = $"dressed {file.layout.id}: {dressing.Dress(this, file)}";
        UnityEditor.SceneManagement.EditorSceneManager.MarkSceneDirty(gameObject.scene);
        return true;
    }
#endif

#if UNITY_EDITOR
    /// <summary>Edit mode: builds the layout's buildings and streets into the scene to be saved with it, the goals'
    /// buildings under <c>Goal buildings</c> without their goal roots (Awake makes those at Play; see the class
    /// summary). Replaces any previous build. False with an error on failure.</summary>
    public bool Bake(out string error)
    {
        if (Application.isPlaying)
        {
            error = "bake in edit mode; at Play the city is built (or the bake adopted) in Awake.";
            return false;
        }
        if (!Load(out LayoutFile file, out string path, out error)) return false;
        Clear();
        Layout = file;
        ResolvedPath = path;
        Transform buildings = Child(BuildingsName);
        BuildStreets(file.streets, Child(StreetsName));
        int nCity = PlaceCity(file, buildings, out error);
        if (nCity < 0) return false;
        Transform goalBuildings = Child(GoalBuildingsName);
        foreach (LayoutGoal g in file.layout.goals ?? new LayoutGoal[0])
        {
            var holder = new GameObject($"goal_{g.index}");
            holder.transform.SetParent(goalBuildings, false);
            holder.transform.position = new Vector3(g.centre[0], 0f, g.centre[1]);
            foreach (LayoutBox b in file.boxes)
            {
                if (b.goal == g.index && b.role != "city" && Place(b, holder.transform, out error) == null) return false;
            }
        }
        bakedLayout = LayoutName;
        UnityEditor.EditorUtility.SetDirty(this);
        Finish(file, path, buildings, nCity, "baked", Dress(file));
        return true;
    }
#endif

    /// <summary>At Play: keeps the baked city and makes the goals round its goal buildings, if the bake is of the
    /// layout to build and every building still stands on its box. False (nothing changed) with the reason otherwise.</summary>
    private bool Adopt(out string error)
    {
        string want = Resolve(LayoutName), have = Resolve(bakedLayout);
        if (want == null || have == null || !string.Equals(want, have, StringComparison.OrdinalIgnoreCase))
        {
            error = $"it is of '{bakedLayout}', and the layout to build is '{LayoutName}'";
            return false;
        }
        if (!Load(out LayoutFile file, out string path, out error)) return false;
        Transform buildings = transform.Find(BuildingsName), goalBuildings = transform.Find(GoalBuildingsName);
        var baked = new Dictionary<string, Transform>();
        foreach (Transform t in buildings) baked[t.name] = t;
        if (goalBuildings != null)
        {
            foreach (Transform goal in goalBuildings)
            {
                foreach (Transform t in goal) baked[t.name] = t;
            }
        }
        foreach (LayoutBox b in file.boxes)
        {
            if (!baked.TryGetValue(b.name, out Transform t))
            {
                error = $"{b.name} is in {Path.GetFileName(path)} but not in the scene: rebake it";
                return false;
            }
            bool stands;
            if (PrismBuilding.IsPrism(b))
            {
                MeshCollider mesh = t.GetComponent<MeshCollider>();
                stands = mesh != null && mesh.sharedMesh != null &&
                         (t.position - PrismBuilding.Origin(b)).sqrMagnitude <= AdoptTolerance * AdoptTolerance;
            }
            else
            {
                BoxCollider box = t.GetComponent<BoxCollider>();
                Vector3 c = new Vector3(b.c[0], b.c[1], b.c[2]);
                stands = box != null && (t.TransformPoint(box.center) - c).sqrMagnitude <= AdoptTolerance * AdoptTolerance;
            }
            if (!stands)
            {
                error = $"{b.name} no longer stands where {Path.GetFileName(path)} puts it: rebake it";
                return false;
            }
        }
        if (baked.Count != file.boxes.Length)
        {
            error = $"the scene holds {baked.Count} buildings and {Path.GetFileName(path)} {file.boxes.Length}: rebake it";
            return false;
        }

        Layout = file;
        ResolvedPath = path;
        placedGoals.Clear();
        Transform goalsRoot = transform.Find(GoalsName);
        if (goalsRoot == null) goalsRoot = Child(GoalsName);
        Obtain move = (LayoutBox b, Transform parent, out string err) =>
        {
            Transform t = baked[b.name];
            t.SetParent(parent, true);
            err = null;
            return t.gameObject;
        };
        if (!BuildGoals(file, goalsRoot, move, out error)) return false;
        if (goalBuildings != null) Destroy(goalBuildings.gameObject);      // emptied: every building moved to its goal
        int nCity = 0;
        foreach (LayoutBox b in file.boxes) nCity += b.role == "city" ? 1 : 0;
        CityDressing dressing = GetComponent<CityDressing>();
        string dressed = dressing == null ? ""
            : dressing.IsCurrent(file, RoadSettings) ? "kept the baked dressing; " + dressing.MergeForPlay()
            : "the bake's dressing is stale or missing, so it was dressed at Play (rebake to see it in the editor): " +
              dressing.Dress(this, file);
        Finish(file, path, buildings, nCity, "adopted the baked", dressed);
        return true;
    }

    /// <summary>Resolves, reads and checks the layout to build (<see cref="LayoutName"/>).</summary>
    private bool Load(out LayoutFile file, out string path, out string error) => Load(LayoutName, out file, out path, out error);

    /// <summary>Resolves, reads and checks the layout <paramref name="name"/>.</summary>
    private bool Load(string name, out LayoutFile file, out string path, out string error)
    {
        file = null;
        path = Resolve(name);
        if (path == null)
        {
            error = $"no layout file for '{name}' (looked under Assets/{LayoutFolder} and Assets/).";
            return false;
        }
        try
        {
            file = JsonUtility.FromJson<LayoutFile>(File.ReadAllText(path));
        }
        catch (Exception e)
        {
            error = $"cannot read {path}: {e.Message}";
            return false;
        }
        if (file == null || file.boxes == null || file.layout == null)
        {
            error = $"{path} is not a layout file (no boxes / layout record).";
            return false;
        }
        if (catalogue == null && Array.Exists(file.boxes, b => !PrismBuilding.IsPrism(b)))
        {
            error = "no BuildingCatalogue assigned (Tools/Swarm/Export building catalogue writes one).";
            return false;
        }
        if (file.layout.goals != null && file.layout.goals.Length > 0 && goalRootPrefab == null)
        {
            error = "the layout has goals but no goalRootPrefab is assigned.";
            return false;
        }
        if (!OnTerrain(file.layout, out error)) return false;
        foreach (LayoutBox b in file.boxes)
        {
            if (PrismBuilding.IsPrism(b))
            {
                if (!PrismBuilding.Valid(b, out error)) return false;
                continue;
            }
            if (catalogue.Find(b.design) == null)
            {
                error = $"design '{b.design}' (box {b.name}) is not in the catalogue.";
                return false;
            }
        }
        error = null;
        return true;
    }

    /// <summary>The city buildings under <paramref name="parent"/>; their number, or -1 with an error.</summary>
    private int PlaceCity(LayoutFile file, Transform parent, out string error)
    {
        DealPrisms(file);
        int n = 0;
        foreach (LayoutBox b in file.boxes)
        {
            if (b.role != "city") continue;
            if (Place(b, parent, out error) == null) return -1;
            n++;
        }
        error = null;
        return n;
    }

    /// <summary>Every prism's colour and window pattern, before any is placed (<see cref="PrismBuilding.Deal"/>: from the
    /// layout's seed, so a layout always looks the same).</summary>
    private void DealPrisms(LayoutFile file)
    {
        prismLooks = PrismBuilding.Deal(file.boxes, prismStyle, file.layout.seed, file.layout.p, out int clashes);
        prismNote = prismLooks.Count == 0 ? ""
            : $", {prismLooks.Count} block-shaped in {Mathf.Max(prismStyle.Colours, 1)} colours" +
              (clashes > 0 ? $" ({clashes} beside a building of their own colour)" : "");
    }

    /// <summary>The goals, in layout order, each a <see cref="goalRootPrefab"/> with its buildings (from
    /// <paramref name="obtain"/>) under it, activated only once complete.</summary>
    private bool BuildGoals(LayoutFile file, Transform goalsRoot, Obtain obtain, out string error)
    {
        bool walkers = spawnWalkers && Environment.GetEnvironmentVariable(WalkersEnv) != "0";
        // The session's goal seed first (the hats, target walkers and walker starts follow from it).
        GoalSession.Set(file.layout.seed, "layout");
        GameObject holder = new GameObject("goal holder (inactive)");
        holder.SetActive(false);
        holder.transform.SetParent(goalsRoot, false);
        var roots = new List<GameObject>();
        foreach (LayoutGoal g in file.layout.goals ?? new LayoutGoal[0])
        {
            GameObject root = Instantiate(goalRootPrefab, holder.transform);
            root.name = $"goal_{g.index}";
            root.transform.SetPositionAndRotation(new Vector3(g.centre[0], 0f, g.centre[1]), Quaternion.identity);
            ProceduralGoal pg = root.GetComponent<ProceduralGoal>();
            if (pg == null) pg = root.AddComponent<ProceduralGoal>();
            pg.index = g.index;
            pg.ringRadius = g.r_g;
            pg.occluderWidth = g.w_c;
            pg.clearance = g.c_g;
            pg.clearRadius = g.clear_radius;
            pg.centreBox = g.centre_box ?? "";
            pg.ringBoxes = g.ring_boxes ?? new string[0];
            float ringHalfDiag = 0f;
            foreach (LayoutBox b in file.boxes)
            {
                if (b.goal != g.index || b.role == "city") continue;
                GameObject building = obtain(b, root.transform, out error);
                if (building == null)
                {
                    DestroyNow(holder);
                    return false;
                }
                if (b.role == "goal_ring")
                {
                    Bounds bb = building.GetComponent<Collider>().bounds;
                    ringHalfDiag = Mathf.Max(ringHalfDiag, new Vector2(bb.extents.x, bb.extents.z).magnitude);
                    if (walkers && b.patrol) AddPatrol(building);
                }
            }
            pg.patchRadius = g.r_g + ringHalfDiag;
            roots.Add(root);
        }
        // Activating in layout order: each GoalSpecialWalker.Awake deals its hat in that order.
        foreach (GameObject root in roots)
        {
            root.transform.SetParent(goalsRoot, true);
            placedGoals.Add(root);
        }
        DestroyNow(holder);
        error = null;
        return true;
    }

    /// <summary>What every path ends with: edit-mode static flags, the spawn, the session record, the log line (with the
    /// dressing's summary, if any).</summary>
    private void Finish(LayoutFile file, string path, Transform buildings, int nCity, string verb, string dressing)
    {
#if UNITY_EDITOR
        if (!Application.isPlaying)
        {
            foreach (Transform t in buildings.GetComponentsInChildren<Transform>(true))
            {
                UnityEditor.GameObjectUtility.SetStaticEditorFlags(t.gameObject, (UnityEditor.StaticEditorFlags)~0);
            }
            TerrainHeightSampler.Invalidate();
        }
#endif
        if (setSpawn && file.layout.spawn != null && file.layout.spawn.Length >= 3)
        {
            swarmSpawn spawner = FindObjectOfType<swarmSpawn>(true);
            if (spawner != null)
            {
                spawner.start_x = Mathf.RoundToInt(file.layout.spawn[0]);
                spawner.start_y = Mathf.RoundToInt(file.layout.spawn[1]);
                spawner.start_z = Mathf.RoundToInt(file.layout.spawn[2]);
#if UNITY_EDITOR
                if (!Application.isPlaying) UnityEditor.EditorUtility.SetDirty(spawner);
#endif
            }
        }

        CityLayoutInfo.Set(new CityLayoutInfo.Record
        {
            id = file.layout.id ?? "",
            path = path,
            seed = file.layout.seed,
            design = file.layout.design ?? "",
            size = file.layout.size,
            p = file.layout.p,
            s = file.layout.s,
            f = file.layout.f,
            alphaAchieved = file.layout.alpha_achieved,
            scene = gameObject.scene.name,
            generatorVersion = file.layout.generator_version ?? "",
        });

        Debug.Log($"ProceduralCityBuilder: {verb} {file.layout.id} ({nCity} city buildings{prismNote}, " +
                  $"{file.layout.goals?.Length ?? 0} goals, " +
                  $"{file.streets?.Length ?? 0} streets, alpha {file.layout.alpha_achieved:F3}, goal seed {file.layout.seed}) " +
                  $"from {path}" + (string.IsNullOrEmpty(dressing) ? "" : $"; dressing: {dressing}"), this);
    }

    /// <summary>Removes the previous build or bake (this root's Buildings, Streets, Goals, Goal buildings and Dressing).</summary>
    public void Clear()
    {
        placedGoals.Clear();
        prismNote = "";
        foreach (string n in new[] { BuildingsName, StreetsName, GoalsName, GoalBuildingsName, DressingName })
        {
            Transform t = transform.Find(n);
            if (t != null) DestroyNow(t.gameObject);
        }
        Layout = null;
        bakedLayout = "";
    }

    private Transform Child(string n)
    {
        var go = new GameObject(n);
        go.transform.SetParent(transform, false);
        return go.transform;
    }

    private static void DestroyNow(GameObject go)
    {
        if (Application.isPlaying) Destroy(go);
        else DestroyImmediate(go);
    }

    /// <summary>One building, its collider centred on the box centre. Null with an error if that fails.</summary>
    private GameObject Place(LayoutBox b, Transform parent, out string error)
    {
        if (PrismBuilding.IsPrism(b))
        {
            error = null;
            prismLooks.TryGetValue(b.name, out PrismBuilding.Look look);
            return PrismBuilding.Create(b, parent, prismStyle, look);
        }
        BuildingCatalogue.Entry e = catalogue.Find(b.design);
        GameObject go = Instantiate(e.source, parent);
        go.name = b.name;
        go.SetActive(true);
        Transform t = go.transform;
        Quaternion rot = Quaternion.Euler(-90f, b.yaw, 0f);
        Vector3 scale = new Vector3(e.lossyScale.x * b.size, e.lossyScale.y * b.size, e.lossyScale.z);
        Vector3 c = new Vector3(b.c[0], b.c[1], b.c[2]);
        t.rotation = rot;
        t.localScale = scale;          // the containers above are unscaled at the origin
        t.position = c - rot * Vector3.Scale(e.colliderCenter, scale);
        BoxCollider box = go.GetComponent<BoxCollider>();
        if (box == null)
        {
            error = $"the catalogue's {b.design} has no BoxCollider.";
            return null;
        }
        box.enabled = true;
        Vector3 got = t.TransformPoint(box.center);
        if ((got - c).sqrMagnitude > 1e-4f)
        {
            error = $"{b.name}: its collider centre landed at {got}, not {c} -- the catalogue's frame is wrong.";
            return null;
        }
        error = null;
        return go;
    }

    private void AddPatrol(GameObject building)
    {
        WalkerPatrol p = building.GetComponent<WalkerPatrol>();
        if (p == null) p = building.AddComponent<WalkerPatrol>();
        p.shape = WalkerPatrol.PathShape.Rectangle;
        p.walkerPrefab = walkerPrefab;
        p.numWalkers = numWalkers;
        p.speedMetersPerSec = walkerSpeed;
        p.baseOffset = baseOffset;
        p.margin = margin;
        p.cornerRadius = cornerRadius;
        p.verticalOffset = 0f;
    }

    /// <summary>One mesh of flat quads along every street centreline (no collider).</summary>
    private void BuildStreets(LayoutStreet[] streets, Transform parent)
    {
        if (streets == null || streets.Length == 0) return;
        if (streetStyle == StreetStyle.Textured)
        {
            if (junctionMaterial != null && streetMaterial != null)
            {
                BuildRoads(streets, parent);
                return;
            }
            Debug.LogWarning("ProceduralCityBuilder: Textured streets need a street and a junction material " +
                             "(the Modular City Pack's asphalt); drawing them flat.", this);
        }
        var verts = new List<Vector3>();
        var tris = new List<int>();
        var uvs = new List<Vector2>();
        foreach (LayoutStreet s in streets)
        {
            if (s.xz == null) continue;
            for (int k = 0; k + 3 < s.xz.Length; k += 2)
            {
                var a = new Vector3(s.xz[k], streetHeight, s.xz[k + 1]);
                var b = new Vector3(s.xz[k + 2], streetHeight, s.xz[k + 3]);
                Vector3 d = b - a;
                if (d.sqrMagnitude < 1e-6f) continue;
                Vector3 side = Vector3.Cross(Vector3.up, d.normalized) * (0.5f * s.width);
                int i = verts.Count;
                verts.Add(a - side);
                verts.Add(a + side);
                verts.Add(b + side);
                verts.Add(b - side);
                float len = d.magnitude / Mathf.Max(s.width, 1e-3f);
                uvs.Add(new Vector2(0, 0));
                uvs.Add(new Vector2(1, 0));
                uvs.Add(new Vector2(1, len));
                uvs.Add(new Vector2(0, len));
                tris.AddRange(new[] { i, i + 2, i + 1, i, i + 3, i + 2 });
            }
        }
        var mesh = new Mesh { name = "Procedural streets", indexFormat = UnityEngine.Rendering.IndexFormat.UInt32 };
        mesh.SetVertices(verts);
        mesh.SetUVs(0, uvs);
        mesh.SetTriangles(tris, 0);
        mesh.RecalculateNormals();
        mesh.RecalculateBounds();
        var go = new GameObject("Street surface");
        go.transform.SetParent(parent, false);
        go.AddComponent<MeshFilter>().sharedMesh = mesh;
        MeshRenderer r = go.AddComponent<MeshRenderer>();
        r.sharedMaterial = streetMaterial;
        r.shadowCastingMode = UnityEngine.Rendering.ShadowCastingMode.Off;
    }

    /// <summary>The streets as road (<see cref="StreetSurface"/>): one mesh, stretches in the street material and
    /// junctions and crossings in the junction material. No collider, no shadows.</summary>
    private void BuildRoads(LayoutStreet[] streets, Transform parent)
    {
        var segments = new List<(Vector2, Vector2, float)>();
        foreach (LayoutStreet s in streets)
        {
            if (s.xz == null) continue;
            for (int k = 0; k + 3 < s.xz.Length; k += 2)
            {
                segments.Add((new Vector2(s.xz[k], s.xz[k + 1]), new Vector2(s.xz[k + 2], s.xz[k + 3]), s.width));
            }
        }
        Mesh mesh = StreetSurface.Build(segments, RoadSettings);
        var go = new GameObject("Street surface");
        go.transform.SetParent(parent, false);
        go.AddComponent<MeshFilter>().sharedMesh = mesh;
        MeshRenderer r = go.AddComponent<MeshRenderer>();
        r.sharedMaterials = new[] { streetMaterial, junctionMaterial };
        r.shadowCastingMode = UnityEngine.Rendering.ShadowCastingMode.Off;
    }

    /// <summary>Whether the arena's corners and the spawn stand on one of the scene's Terrains.</summary>
    private static bool OnTerrain(LayoutRecord r, out string error)
    {
        error = null;
        Terrain[] terrains = Terrain.activeTerrains;
        if (terrains.Length == 0)
        {
            error = "the scene has no Terrain, so no altitude ceiling: keep the city scenes' four terrain tiles.";
            return false;
        }
        var pts = new List<Vector2>();
        if (r.arena != null && r.arena.Length == 4)
        {
            pts.Add(new Vector2(r.arena[0], r.arena[2]));
            pts.Add(new Vector2(r.arena[1], r.arena[2]));
            pts.Add(new Vector2(r.arena[0], r.arena[3]));
            pts.Add(new Vector2(r.arena[1], r.arena[3]));
        }
        if (r.spawn != null && r.spawn.Length >= 3) pts.Add(new Vector2(r.spawn[0], r.spawn[2]));
        foreach (Vector2 p in pts)
        {
            bool on = false;
            foreach (Terrain t in terrains)
            {
                Vector3 o = t.transform.position, s = t.terrainData.size;
                if (p.x >= o.x - 1e-3f && p.x <= o.x + s.x + 1e-3f && p.y >= o.z - 1e-3f && p.y <= o.z + s.z + 1e-3f)
                {
                    on = true;
                    break;
                }
            }
            if (!on)
            {
                error = $"the layout's point ({p.x:F0}, {p.y:F0}) is off every Terrain: there would be no ground for " +
                        "the altitude ceiling there.";
                return false;
            }
        }
        return true;
    }
}
