using System.Collections.Generic;
using System.IO;
using System.Linq;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;
using UnityEngine.SceneManagement;

/// <summary>
/// Makes the four diamonds (<see cref="DiamondCityBuilder"/>) formal parks built on one shared base, each distinct from
/// the others, writing straight into their prefabs.
///
/// <para><b>The structure.</b> <c>Diamond_B</c> was restyled by hand into a cross-shaped lawn of grass planes with a
/// fountain at its centre, inside the street shells of its four blocks. That became <see cref="BasePath"/>, and every
/// diamond is now its own root holding three children: <c>Base</c>, a nested instance of it; <c>Plazas</c>, the two
/// small plazas standing across the north–south corridors; and <c>Park</c>, the trees and tip buildings. Editing the base
/// changes all four. It is a nested prefab rather than a Unity Prefab Variant on purpose: a variant's root is a new
/// object, so turning the existing prefabs into variants would orphan the root-position overrides DiamondCityWorld holds
/// for each placed diamond, and every diamond would jump to its prefab's stored position.</para>
///
/// <para><b>What a run does</b>, all of it idempotent: creates the base from <see cref="BaseSourceLetter"/> if it does not
/// exist; puts any diamond not yet on the base onto it, keeping plazas it already has (Diamond_B's, made by hand) or
/// drafting its spec's two plazas (see <see cref="DraftPlaza"/>); gives the base's grass planes the terrain's grass; and
/// rebuilds every diamond's <c>Park</c> from nothing. So the layout is this file, not hand edits: change a constant or a
/// spec and run it again. Hand edits to the base and to the plazas survive a run; hand edits under <c>Park</c> do not.
/// Deleting a diamond's prefab gets <see cref="DiamondCityBuilder"/>'s tile-copy draft back, which the next run puts on
/// the base again.</para>
///
/// <para><b>The park</b> (~115–120 trees) is measured from what the diamond holds — the lawn is the union of the grass
/// planes, the centre is the fountain, the axes run through it, and whatever stands on the lawn (plaza footpaths,
/// buildings, props) is a keep-out — and a spot that fails the lawn or keep-out test is skipped and counted, never
/// nudged. A ring round the fountain, open on the four axes; an avenue (a pair of rows) down each axis, from the ring or
/// the far side of a plaza out towards the tip; a staggered double row inside both long edges of two opposite arms, and
/// a small grid grove either side of the avenue in the other two; and at each of the four tips a single 50-unit building
/// on a footpath slab slightly larger than its footprint.</para>
///
/// <para><b>What makes each diamond distinct</b> is its <see cref="DiamondSpec"/>: its plazas, its four tip buildings,
/// which arms carry the double rows and which the groves, which tree model each arrangement uses, and its seed. No patch
/// is the source of anything in two diamonds, every tip building is a different design from its diamond's plazas and
/// from every other diamond's tips, and everything comes from a patch at least three slots from its diamond in the
/// <see cref="DiamondCityBuilder"/> layout, so nothing stands in sight of its twin. The pack has ~38 building designs,
/// so a few designs do recur between one diamond's plazas and another's.</para>
///
/// <para><b>The trees are the four models in <see cref="TreeFolder"/></b> — <c>Tree9_2</c> is the one
/// DiamondCityWorld's <see cref="VergeTreePlanter"/> plants on the verges — all at <see cref="TreeScale"/> times their
/// prefab size, where they are 4.1–4.7 units tall, street trees. Each arrangement uses one model, and its mirror image
/// across the fountain the same one, so a row reads as one planting; the groves mix them all. They are scenery, not
/// obstacles — no collider, not on <c>Obstacle</c> — like every other tree in the city, and marked Batching Static as
/// the planter does.</para>
///
/// <para><b>The tip buildings are copies of ScaledCity buildings</b> (not prefab instances), so each keeps the UV-baked
/// mesh <see cref="BuildingUvBaker"/> made for its 0.25-width scale, its 50-unit height, its <c>BoxCollider</c> on
/// <c>Obstacle</c> and a name in one of the building families <see cref="ObstacleLayerAuditor"/> keeps on that layer.
/// Its footpath is a generated quad with the pack footpath's own texel density: a scaled-down copy of the 76.2-unit
/// footpath slab would shrink its paving five-fold.</para>
///
/// <para><b>The grass planes get <see cref="GrassMaterialPath"/></b>, built from the terrain's grass layer
/// (<see cref="TerrainLayerPath"/>, which all four DiamondCityWorld terrains use): same texture, same metres per repeat
/// (the plane's size over the layer's tile size), matte like the layer. The planes also leave the <c>Obstacle</c> layer.
/// They carry MeshColliders, and <see cref="OlfatiSaber"/> takes any collider on that layer as a cylinder round its
/// bounds: each 76-unit plane stood over the park as a ~54-unit-radius repulsion field, the <c>Road_Structure</c>
/// failure <see cref="ObstacleLayerAuditor"/> describes.</para>
///
/// <para><b>It also keeps each diamond from being taken for a tile</b>: a plaza's kerb (<c>Carbs_NN</c>) is renamed
/// <c>Kerb_NN</c> and hidden, and its <c>MC_Patch</c> nodes become <c>Block</c>, as <see cref="DiamondCityBuilder"/>
/// does for its drafts. A kerb under the diamonds' root stops <see cref="CityTiles.FindCity(Scene, out string)"/>
/// finding the city at all — which silently stops goal-patch verge planting at runtime — and
/// <see cref="CityObstacleExport"/> lists every <c>MC_Patch</c> node as a tile.</para>
/// </summary>
public static class DiamondParkBuilder
{
    public const string BasePath = DiamondCityBuilder.DiamondFolder + "/Diamond_Base.prefab";

    /// <summary>The diamond restyled by hand, whose lawn, fountain and street shells the base is made from.</summary>
    private const char BaseSourceLetter = 'B';

    private const string GeneratedFolder = DiamondCityBuilder.DiamondFolder + "/Generated";
    private const string GrassMaterialPath = "Assets/Materials/DiamondParkGrass.mat";
    private const string TerrainLayerPath = "Assets/BaseLayer.terrainlayer";
    private const string TreeFolder = "Assets/Tree9";
    private const string PatchFolder = "Assets/Prefabs/ScaledCity/Patches";

    private const string BaseName = "Base";
    private const string PlazasName = "Plazas";
    private const string ParkName = "Park";
    private const string GrassName = "Grass";
    private const string GrassPlanePrefix = "Plane";
    private const string FountainPrefix = "fountain";
    private const string RoadPlatePrefix = "Road_Structure_";
    private const string InteriorStreetPrefix = "Roads_Street_";
    private static readonly string[] FootpathPrefixes = { "FootPath_", "Footpath_" };

    /// <summary>A pack footpath's height above its tile's road plate, for a patch that has no plate to measure by.</summary>
    private const float FootpathHeight = 0.15f;

    /// <summary>The building families <see cref="ObstacleLayerAuditor"/> keeps on <c>Obstacle</c>.</summary>
    private static readonly string[] BuildingPrefixes =
    {
        "BnP_Small_Building_", "BnP_Large_Building_", "BnP_Apartment_", "Skyscraper_",
    };

    /// <summary>Renderers a tree may stand next to: the ground, and the kerbs. The street medians are skipped too.</summary>
    private static readonly string[] GroundPrefixes =
    {
        GrassPlanePrefix, RoadPlatePrefix, "Green_Belt", CityTiles.KerbPrefix, DiamondCityBuilder.DiamondKerbPrefix,
    };

    private const StaticEditorFlags Everything = (StaticEditorFlags)(-1); // what the city's tiles carry

    // ---------------------------------------------------------------- plazas

    /// <summary>
    /// A drafted plaza's block content about its kerb, as Diamond_B's hand-made plazas were: their footpaths are 49.5
    /// units across where a block's is 76.2. Buildings and props move in, keeping their size.
    /// </summary>
    private const float PlazaScale = 0.65f;
    private const float PlazaMinGap = 1f;    // between two of a drafted plaza's buildings
    private const float PlazaMinMargin = 0.5f; // from a building to its plaza's footpath edge

    /// <summary>How far a placed building's base may sit from the diamond's ground before the build refuses it.</summary>
    private const float MaxBaseHeight = 1f;

    // ---------------------------------------------------------------- trees

    /// <summary>Times the prefab's own size, so a park tree stands 7.4–8.5 units tall against a street tree's ~4.6.</summary>
    private const float TreeScale = 1.8f;
    private const float TreeScaleJitter = 0.1f; // either way

    /// <summary>How close a trunk may stand to anything on the lawn, and to the lawn's edge.</summary>
    private const float TreeClearance = 3f;
    private const float LawnMargin = 2.5f;

    /// <summary>A multiple of four, so the ring is open where each axis meets it.</summary>
    private const int RingTrees = 12;
    private const float RingRadius = 12f;

    private const float AvenueHalfWidth = 6.5f;
    private const float AvenueSpacing = 7f;
    private const float AvenueGapToRing = 10f;
    private const float AvenueStandOff = 6f; // from a plaza, or from a tip building's footpath

    private const float EdgeRowInset = 4f;     // outer row from the lawn's edge
    private const float EdgeRowGap = 6f;       // outer row to inner row
    private const float EdgeRowSpacing = 8.5f; // about, in the outer row; the inner row stands in its gaps
    private const float EdgeRowStartGap = 6f;  // past the corner where the arm leaves the cross, or past a plaza
    private const float EdgeRowEndGap = 5f;    // short of the tip

    private const int GroveAlong = 3;  // columns, along the arm
    private const int GroveAcross = 2; // rows
    private const float GroveSpacing = 9f;
    private const float GroveInset = 4f; // from the lawn's edge and the plaza
    private const float GroveGapToAvenue = 5f;

    /// <summary>One arrangement: where its trees go, and which model they are (null: a random one each).</summary>
    private readonly struct TreeGroup
    {
        public readonly string Name;
        public readonly string Species;
        public readonly List<Vector2> Spots;

        public TreeGroup(string name, string species, List<Vector2> spots)
        {
            Name = name;
            Species = species;
            Spots = spots;
        }
    }

    // ---------------------------------------------------------------- tips

    private enum Tip
    {
        North,
        East,
        South,
        West,
    }

    private static readonly Tip[] Tips = { Tip.North, Tip.East, Tip.South, Tip.West };

    private readonly struct TipBuilding
    {
        public readonly Tip Tip;
        public readonly string Patch;    // the ScaledCity patch prefab it is copied from
        public readonly string Building; // its name there

        public TipBuilding(Tip tip, string patch, string building)
        {
            Tip = tip;
            Patch = patch;
            Building = building;
        }
    }

    private const float TipInset = 3f;       // the footpath's outer edge from the lawn's edge
    private const float FootpathMargin = 2f; // footpath beyond the building on every side

    private sealed class TipSource
    {
        public TipBuilding Spec;
        public Transform Building;
        public Renderer Footpath;
    }

    /// <summary>A linear map from the horizontal plane (x, z) to texture space, fitted to a pack footpath.</summary>
    private readonly struct UvMap
    {
        private readonly Vector3 u; // u = u.x * x + u.y * z + u.z
        private readonly Vector3 v;

        public UvMap(Vector3 u, Vector3 v)
        {
            this.u = u;
            this.v = v;
        }

        public Vector2 At(float x, float z) => new Vector2(u.x * x + u.y * z + u.z, v.x * x + v.y * z + v.z);
    }

    // ---------------------------------------------------------------- the diamonds

    /// <summary>Everything that makes one diamond different from the others.</summary>
    private sealed class DiamondSpec
    {
        public char Letter;
        public int Seed;

        /// <summary>Double rows in the north and south arms and groves in the west and east, or the other way round.</summary>
        public bool RowsNorthSouth;

        // Tree models by prefab name in TreeFolder; the groves mix every model there.
        public string Ring;
        public string NorthSouthAvenues;
        public string EastWestAvenues;
        public string Rows;

        /// <summary>West, then east: drafted only into a diamond that has no plazas of its own.</summary>
        public string[] Plazas;

        public TipBuilding[] TipBuildings;

        public string PrefabPath => $"{DiamondCityBuilder.DiamondFolder}/Diamond_{Letter}.prefab";
    }

    /// <summary>
    /// Slots are (column, row) in <see cref="DiamondCityBuilder"/>'s layout; every source listed is at least three slots
    /// from its diamond. B's plazas are the ones made by hand; its other values are what it was first dressed with.
    /// </summary>
    private static readonly DiamondSpec[] Diamonds =
    {
        new DiamondSpec
        {
            Letter = 'A', Seed = 2, RowsNorthSouth = false,
            Ring = "Tree9_2", NorthSouthAvenues = "Tree9_5", EastWestAvenues = "Tree9_3", Rows = "Tree9_4",
            Plazas = new[] { "MC_Patch_35_Scaled", "MC_Patch_26_Scaled" },
            TipBuildings = new[]
            {
                new TipBuilding(Tip.North, "MC_Patch_25_Scaled", "BnP_Apartment_F_006"),
                new TipBuilding(Tip.East, "MC_Patch_45_Scaled", "BnP_Large_Building_I_007"),
                new TipBuilding(Tip.South, "MC_Patch_11_Scaled", "Skyscraper_J_002"),
                new TipBuilding(Tip.West, "MC_Patch_42_Scaled", "BnP_Apartment_D_012"),
            },
        },
        new DiamondSpec
        {
            Letter = 'B', Seed = 1, RowsNorthSouth = true,
            Ring = "Tree9_3", NorthSouthAvenues = "Tree9_2", EastWestAvenues = "Tree9_4", Rows = "Tree9_5",
            Plazas = new[] { "MC_Patch_02_Scaled", "MC_Patch_43_Scaled" },
            TipBuildings = new[]
            {
                new TipBuilding(Tip.North, "MC_Patch_05_Scaled", "BnP_Small_Building_D_004"),
                new TipBuilding(Tip.East, "MC_Patch_03_Scaled", "BnP_Apartment_H_000"),
                new TipBuilding(Tip.South, "MC_Patch_16_Scaled", "BnP_Large_Building_K_002"),
                new TipBuilding(Tip.West, "MC_Patch_19_Scaled", "BnP_Apartment_E_002"),
            },
        },
        new DiamondSpec
        {
            Letter = 'C', Seed = 3, RowsNorthSouth = true,
            Ring = "Tree9_4", NorthSouthAvenues = "Tree9_3", EastWestAvenues = "Tree9_5", Rows = "Tree9_2",
            Plazas = new[] { "MC_Patch_12_Scaled", "MC_Patch_06_Scaled" },
            TipBuildings = new[]
            {
                new TipBuilding(Tip.North, "MC_Patch_31_Scaled", "BnP_Apartment_I_004"),
                new TipBuilding(Tip.East, "MC_Patch_46_Scaled", "BnP_Small_Building_B_007"),
                new TipBuilding(Tip.South, "MC_Patch_07_Scaled", "Skyscraper_E_000"),
                new TipBuilding(Tip.West, "MC_Patch_32_Scaled", "BnP_Apartment_C_002"),
            },
        },
        new DiamondSpec
        {
            Letter = 'D', Seed = 4, RowsNorthSouth = false,
            Ring = "Tree9_5", NorthSouthAvenues = "Tree9_4", EastWestAvenues = "Tree9_2", Rows = "Tree9_3",
            Plazas = new[] { "MC_Patch_30_Scaled", "MC_Patch_38_Scaled" },
            TipBuildings = new[]
            {
                new TipBuilding(Tip.North, "MC_Patch_48_Scaled", "Skyscraper_D_002"),
                new TipBuilding(Tip.East, "MC_Patch_39_Scaled", "BnP_Large_Building_J_004"),
                new TipBuilding(Tip.South, "MC_Patch_14_Scaled", "BnP_Large_Building_L_005"),
                new TipBuilding(Tip.West, "MC_Patch_39_Scaled", "Skyscraper_I_000"),
            },
        },
    };

    // ---------------------------------------------------------------- entry points

    [MenuItem("Tools/Swarm/Build diamond parks")]
    private static void BuildFromMenu()
    {
        PrefabStage stage = PrefabStageUtility.GetCurrentPrefabStage();
        if (stage != null && (stage.assetPath == BasePath || Diamonds.Any(d => d.PrefabPath == stage.assetPath)))
        {
            EditorUtility.DisplayDialog("Build diamond parks",
                $"{Path.GetFileName(stage.assetPath)} is open in Prefab Mode. Save and close it first: the parks are written " +
                "into the prefab assets, and saving the open stage afterwards would overwrite them.", "OK");
            return;
        }
        Report(Build(out string error), error);
    }

    [MenuItem("Tools/Swarm/Build diamond parks", true)]
    private static bool CanBuild()
    {
        return !EditorApplication.isPlayingOrWillChangePlaymode;
    }

    /// <summary>Batch-mode entry point: <c>-executeMethod DiamondParkBuilder.BuildInBatch</c>. Exits 0 on success.</summary>
    public static void BuildInBatch()
    {
        string summary = Build(out string error);
        Report(summary, error);
        EditorApplication.Exit(summary != null ? 0 : 1);
    }

    private static void Report(string summary, string error)
    {
        if (summary == null)
        {
            Debug.LogError("DiamondParkBuilder: " + error);
        }
        else
        {
            Debug.Log("DiamondParkBuilder: " + summary);
        }
    }

    /// <summary>
    /// Puts every diamond on the base and rebuilds every park. Returns an account of what it did, or null with
    /// <paramref name="error"/> set. Everything it needs is found before anything is changed; a failure part-way leaves
    /// the prefabs already saved as they are, and a re-run carries on from there.
    /// </summary>
    public static string Build(out string error)
    {
        // Every tree model in the folder, by name; a prefab without a Tree component is not one.
        List<GameObject> trees = AssetDatabase.FindAssets("t:Prefab", new[] { TreeFolder })
                                              .Select(g => AssetDatabase.LoadAssetAtPath<GameObject>(AssetDatabase.GUIDToAssetPath(g)))
                                              .Where(p => p != null && p.GetComponent<Tree>() != null)
                                              .OrderBy(p => p.name, System.StringComparer.Ordinal)
                                              .ToList();
        string missing = Diamonds.SelectMany(d => new[] { d.Ring, d.NorthSouthAvenues, d.EastWestAvenues, d.Rows })
                                 .FirstOrDefault(s => trees.All(p => p.name != s));
        TerrainLayer grassLayer = AssetDatabase.LoadAssetAtPath<TerrainLayer>(TerrainLayerPath);
        if (trees.Count == 0 || missing != null || grassLayer == null || grassLayer.diffuseTexture == null)
        {
            error = trees.Count == 0 ? $"no tree prefabs in {TreeFolder}."
                  : missing != null ? $"no tree prefab named {missing} in {TreeFolder}."
                  : grassLayer == null ? $"no terrain layer at {TerrainLayerPath}."
                  : $"{TerrainLayerPath} has no diffuse texture.";
            return null;
        }
        Dictionary<char, List<TipSource>> tips = new Dictionary<char, List<TipSource>>();
        foreach (DiamondSpec spec in Diamonds)
        {
            if (AssetDatabase.LoadAssetAtPath<GameObject>(spec.PrefabPath) == null)
            {
                error = $"no prefab at {spec.PrefabPath}; build the diamond city first (Tools/Swarm/Build diamond city).";
                return null;
            }
            tips[spec.Letter] = FindTipSources(spec, out error);
            if (tips[spec.Letter] == null)
            {
                return null;
            }
        }

        string structure = EnsureStructure(out error);
        if (structure == null)
        {
            return null;
        }
        string grass = DressBase(grassLayer, out error);
        if (grass == null)
        {
            return null;
        }
        List<string> parks = new List<string>();
        foreach (DiamondSpec spec in Diamonds)
        {
            string park = DressDiamond(spec, trees, tips[spec.Letter], out error);
            if (park == null)
            {
                error = $"Diamond_{spec.Letter}: {error}";
                return null;
            }
            parks.Add(park);
        }
        AssetDatabase.SaveAssets();
        return $"{structure}; {grass}.\n" + string.Join("\n", parks);
    }

    private static List<TipSource> FindTipSources(DiamondSpec spec, out string error)
    {
        int obstacle = LayerMask.NameToLayer("Obstacle");
        List<TipSource> sources = new List<TipSource>();
        foreach (Tip tip in Tips)
        {
            TipBuilding[] matches = spec.TipBuildings.Where(t => t.Tip == tip).ToArray();
            if (matches.Length != 1)
            {
                error = $"Diamond_{spec.Letter} has {matches.Length} tip buildings for its {tip} tip, not one.";
                return null;
            }
            TipBuilding building = matches[0];
            string path = $"{PatchFolder}/{building.Patch}.prefab";
            GameObject patch = AssetDatabase.LoadAssetAtPath<GameObject>(path);
            if (patch == null)
            {
                error = $"no patch prefab at {path}.";
                return null;
            }
            Transform source = patch.GetComponentsInChildren<Transform>(true).FirstOrDefault(t => t.name == building.Building);
            Renderer footpath = patch.GetComponentsInChildren<Renderer>(true)
                                     .FirstOrDefault(r => HasAnyPrefix(r.name, FootpathPrefixes) && r.GetComponent<MeshFilter>());
            if (source == null || footpath == null)
            {
                error = source == null ? $"{building.Patch} has no {building.Building}." : $"{building.Patch} has no footpath.";
                return null;
            }
            if (source.gameObject.layer != obstacle || source.GetComponent<Collider>() == null)
            {
                error = $"{building.Building} in {building.Patch} is not a collider on Obstacle, so the swarm would not avoid it.";
                return null;
            }
            sources.Add(new TipSource { Spec = building, Building = source, Footpath = footpath });
        }
        error = null;
        return sources;
    }

    // ---------------------------------------------------------------- the base, and putting diamonds on it

    /// <summary>
    /// Creates the base if it is missing and puts every diamond not yet on it onto it. Returns what it did, or null
    /// with <paramref name="error"/> set.
    /// </summary>
    private static string EnsureStructure(out string error)
    {
        List<string> notes = new List<string>();
        DiamondSpec baseSource = Diamonds.First(d => d.Letter == BaseSourceLetter);
        GameObject basePrefab = AssetDatabase.LoadAssetAtPath<GameObject>(BasePath);
        if (basePrefab == null)
        {
            basePrefab = CreateBase(baseSource, out error);
            if (basePrefab == null)
            {
                return null;
            }
            notes.Add($"created {Path.GetFileName(BasePath)} from Diamond_{BaseSourceLetter}");
        }

        // Where a drafted plaza goes: on the base source's own plazas, which block the corridors.
        Vector2[] plazaCentres = null;
        foreach (DiamondSpec spec in Diamonds)
        {
            GameObject root = PrefabUtility.LoadPrefabContents(spec.PrefabPath);
            try
            {
                if (IsOnBase(root.transform))
                {
                    continue;
                }
                List<Transform> plazas = FindPlazas(root.transform);
                if (plazas.Count == 0 && plazaCentres == null)
                {
                    plazaCentres = PlazaCentres(baseSource, out error);
                    if (plazaCentres == null)
                    {
                        return null;
                    }
                }
                string note = PutOnBase(root.transform, spec, basePrefab, plazas, plazaCentres, out error);
                if (note == null)
                {
                    return null;
                }
                PrefabUtility.SaveAsPrefabAsset(root, spec.PrefabPath, out bool saved);
                if (!saved)
                {
                    error = $"could not save {spec.PrefabPath}.";
                    return null;
                }
                notes.Add(note);
            }
            finally
            {
                PrefabUtility.UnloadPrefabContents(root);
            }
        }
        error = null;
        return notes.Count == 0 ? "every diamond already on the base" : string.Join("; ", notes);
    }

    /// <summary>
    /// Saves the base source's lawn, fountain and street shells — everything but its plazas and park — as the base,
    /// its root at the origin so it sits exactly where they stood.
    /// </summary>
    private static GameObject CreateBase(DiamondSpec source, out string error)
    {
        GameObject root = PrefabUtility.LoadPrefabContents(source.PrefabPath);
        try
        {
            if (IsOnBase(root.transform))
            {
                error = $"{BasePath} is missing but Diamond_{source.Letter} is already built on it. Restore it from git.";
                return null;
            }
            if (root.transform.Find(GrassName) == null)
            {
                error = $"Diamond_{source.Letter} has no {GrassName} child to make the base's lawn from.";
                return null;
            }
            foreach (Transform child in root.transform.Cast<Transform>().ToList())
            {
                if (child.name == ParkName || IsPlaza(child))
                {
                    Object.DestroyImmediate(child.gameObject);
                }
            }
            root.name = Path.GetFileNameWithoutExtension(BasePath);
            root.transform.localPosition = Vector3.zero;
            root.transform.localRotation = Quaternion.identity;
            root.transform.localScale = Vector3.one;
            GameObject saved = PrefabUtility.SaveAsPrefabAsset(root, BasePath, out bool ok);
            error = ok ? null : $"could not save {BasePath}.";
            return ok ? saved : null;
        }
        finally
        {
            PrefabUtility.UnloadPrefabContents(root);
        }
    }

    /// <summary>
    /// Replaces a diamond's content with the base, keeping the plazas it already has or drafting its spec's. The root
    /// itself is kept, which is what keeps DiamondCityWorld's placement of it valid.
    /// </summary>
    private static string PutOnBase(Transform root, DiamondSpec spec, GameObject basePrefab, List<Transform> plazas,
                                    Vector2[] plazaCentres, out string error)
    {
        // The plazas move to their container first, so clearing the rest cannot take one with it.
        Transform plazaParent = CreateChild(root, PlazasName);
        foreach (Transform plaza in plazas)
        {
            plaza.SetParent(plazaParent, true);
        }
        foreach (Transform child in root.Cast<Transform>().ToList())
        {
            if (child != plazaParent)
            {
                Object.DestroyImmediate(child.gameObject);
            }
        }

        GameObject baseInstance = (GameObject)PrefabUtility.InstantiatePrefab(basePrefab, root);
        baseInstance.name = BaseName;
        baseInstance.transform.localPosition = Vector3.zero;
        baseInstance.transform.localRotation = Quaternion.identity;
        baseInstance.transform.localScale = Vector3.one;
        baseInstance.transform.SetSiblingIndex(0);

        if (plazas.Count > 0)
        {
            error = null;
            return $"Diamond_{spec.Letter} put on the base, keeping its {plazas.Count} plaza(s)";
        }
        List<string> drafted = new List<string>();
        for (int i = 0; i < 2; i++)
        {
            string note = DraftPlaza(spec.Plazas[i], plazaParent, root, plazaCentres[i], out error);
            if (note == null)
            {
                error = $"Diamond_{spec.Letter}'s plaza from {spec.Plazas[i]}: {error}";
                return null;
            }
            drafted.Add(note);
        }
        error = null;
        return $"Diamond_{spec.Letter} put on the base with plazas drafted from {string.Join(" and ", drafted)}";
    }

    /// <summary>
    /// A ScaledCity patch made into a plaza the way Diamond_B's were by hand: its block centred on
    /// <paramref name="centre"/> with its road plate at the diamond's ground, and drawn in about its kerb by
    /// <see cref="PlazaScale"/> — footpaths scaled, buildings and props moved in at their own size — with its road plate,
    /// interior streets and kerb hidden, so the lawn shows between its footpaths. (The streets would sit just under the
    /// lawn, a few centimetres from z-fighting it.) Seven patches carry their content tens of units below their pivot,
    /// which the city cancels on the tile instance; placing by the road plate is what keeps those out of the ground.
    /// Checked afterwards: no two buildings closer than <see cref="PlazaMinGap"/>, none off the footpath. Returns a
    /// note, or null with <paramref name="error"/> set.
    /// </summary>
    private static string DraftPlaza(string patchName, Transform parent, Transform frame, Vector2 centre, out string error)
    {
        string path = $"{PatchFolder}/{patchName}.prefab";
        GameObject prefab = AssetDatabase.LoadAssetAtPath<GameObject>(path);
        if (prefab == null)
        {
            error = $"no patch prefab at {path}.";
            return null;
        }
        GameObject plaza = (GameObject)PrefabUtility.InstantiatePrefab(prefab, parent);
        Transform kerb = CityTiles.FindKerb(plaza.transform);
        if (kerb == null)
        {
            error = $"{patchName} has no {CityTiles.KerbPrefix}NN kerb to centre it by.";
            return null;
        }
        Vector3 target = frame.TransformPoint(new Vector3(centre.x, 0f, centre.y));
        plaza.transform.position += new Vector3(target.x - kerb.position.x, target.y - PatchGround(plaza.transform),
                                                target.z - kerb.position.z);
        Vector3 k = kerb.position;

        // Only a renderer with no renderer above it moves: anything below one travels with it. No container in the
        // pack carries a renderer, so in practice that is every leaf.
        List<Transform> leaves = plaza.GetComponentsInChildren<Renderer>(true)
                                      .Select(r => r.transform)
                                      .Where(t => !Ancestors(t, plaza.transform).Any(a => a.GetComponent<Renderer>() != null))
                                      .ToList();
        foreach (Transform t in leaves)
        {
            if (t == kerb || t.name.StartsWith(RoadPlatePrefix) || t.name.StartsWith(InteriorStreetPrefix))
            {
                t.gameObject.SetActive(false);
                PrefabUtility.RecordPrefabInstancePropertyModifications(t.gameObject);
                continue;
            }
            Vector3 p = t.position;
            t.position = new Vector3(k.x + (p.x - k.x) * PlazaScale, p.y, k.z + (p.z - k.z) * PlazaScale);
            if (HasAnyPrefix(t.name, FootpathPrefixes))
            {
                // Scale whichever local axes lie flat; the pack's -90-about-X import makes that x and y, but read it.
                Vector3 s = t.localScale;
                for (int axis = 0; axis < 3; axis++)
                {
                    Vector3 unit = Vector3.zero;
                    unit[axis] = 1f;
                    if (Mathf.Abs(Vector3.Dot(t.rotation * unit, Vector3.up)) < 0.5f)
                    {
                        s[axis] *= PlazaScale;
                    }
                }
                t.localScale = s;
            }
            PrefabUtility.RecordPrefabInstancePropertyModifications(t);
        }

        // Checked in the diamond's frame.
        float half = CityTiles.BlockHalfSpan * PlazaScale;
        Rect footpath = new Rect(centre - half * Vector2.one, 2f * half * Vector2.one);
        List<Renderer> renderers = plaza.GetComponentsInChildren<Renderer>().Where(r => HasAnyPrefix(r.name, BuildingPrefixes)).ToList();
        foreach (Renderer r in renderers)
        {
            float baseY = frame.InverseTransformPoint(r.bounds.min).y;
            if (Mathf.Abs(baseY) > MaxBaseHeight)
            {
                error = $"{r.name} stands {baseY:F1} units off the ground.";
                return null;
            }
        }
        List<KeyValuePair<string, Rect>> buildings = renderers.Select(r => new KeyValuePair<string, Rect>(r.name, Footprint(r.bounds, frame)))
                                                              .ToList();
        for (int i = 0; i < buildings.Count; i++)
        {
            Rect a = buildings[i].Value;
            if (a.xMin < footpath.xMin + PlazaMinMargin || a.xMax > footpath.xMax - PlazaMinMargin ||
                a.yMin < footpath.yMin + PlazaMinMargin || a.yMax > footpath.yMax - PlazaMinMargin)
            {
                error = $"{buildings[i].Key} ends up off the plaza's footpath.";
                return null;
            }
            for (int j = i + 1; j < buildings.Count; j++)
            {
                Rect b = buildings[j].Value;
                float gap = Mathf.Max(a.xMin - b.xMax, b.xMin - a.xMax, a.yMin - b.yMax, b.yMin - a.yMax);
                if (gap < PlazaMinGap)
                {
                    error = $"{buildings[i].Key} and {buildings[j].Key} end up {gap:F1} units apart.";
                    return null;
                }
            }
        }
        error = null;
        return $"{patchName} ({buildings.Count} buildings)";
    }

    /// <summary>
    /// The world height of a patch's ground: its road plate, which lies at road level whatever offset the patch's
    /// content is baked at; failing that, its footpath less <see cref="FootpathHeight"/>.
    /// </summary>
    private static float PatchGround(Transform patch)
    {
        Renderer[] renderers = patch.GetComponentsInChildren<Renderer>(true);
        Renderer plate = renderers.FirstOrDefault(r => r.name.StartsWith(RoadPlatePrefix));
        if (plate != null)
        {
            return plate.bounds.center.y;
        }
        Renderer footpath = renderers.FirstOrDefault(r => HasAnyPrefix(r.name, FootpathPrefixes));
        return footpath != null ? footpath.bounds.center.y - FootpathHeight : patch.position.y;
    }

    /// <summary>The footpath centres of the base source's plazas, west first, in its frame.</summary>
    private static Vector2[] PlazaCentres(DiamondSpec source, out string error)
    {
        GameObject root = PrefabUtility.LoadPrefabContents(source.PrefabPath);
        try
        {
            Vector2[] centres = FindPlazas(root.transform)
                .Select(p => p.GetComponentsInChildren<Renderer>().FirstOrDefault(r => HasAnyPrefix(r.name, FootpathPrefixes)))
                .Where(r => r != null)
                .Select(r => Footprint(r.bounds, root.transform).center)
                .OrderBy(c => c.x)
                .ToArray();
            error = centres.Length == 2 ? null : $"Diamond_{source.Letter} has {centres.Length} plaza footpaths, not two, " +
                                                 "so there is nowhere to put the other diamonds' plazas.";
            return centres.Length == 2 ? centres : null;
        }
        finally
        {
            PrefabUtility.UnloadPrefabContents(root);
        }
    }

    /// <summary>Whether <paramref name="root"/> already holds the base under <see cref="BaseName"/>.</summary>
    private static bool IsOnBase(Transform root)
    {
        Transform b = root.Find(BaseName);
        return b != null && PrefabUtility.IsOutermostPrefabInstanceRoot(b.gameObject) &&
               PrefabUtility.GetPrefabAssetPathOfNearestInstanceRoot(b.gameObject) == BasePath;
    }

    /// <summary>A diamond's plazas: ScaledCity patch instances, at its top level or under <see cref="PlazasName"/>.</summary>
    private static List<Transform> FindPlazas(Transform root)
    {
        Transform container = root.Find(PlazasName);
        return root.Cast<Transform>()
                   .Concat(container != null ? container.Cast<Transform>() : Enumerable.Empty<Transform>())
                   .Where(IsPlaza)
                   .ToList();
    }

    private static bool IsPlaza(Transform t)
    {
        return PrefabUtility.IsOutermostPrefabInstanceRoot(t.gameObject) &&
               PrefabUtility.GetPrefabAssetPathOfNearestInstanceRoot(t.gameObject).StartsWith(PatchFolder + "/");
    }

    // ---------------------------------------------------------------- the grass, on the base

    private static string DressBase(TerrainLayer grassLayer, out string error)
    {
        GameObject root = PrefabUtility.LoadPrefabContents(BasePath);
        try
        {
            Transform grass = root.transform.Find(GrassName);
            List<Renderer> planes = grass == null
                ? new List<Renderer>()
                : grass.GetComponentsInChildren<Renderer>(true).Where(r => r.name.StartsWith(GrassPlanePrefix)).ToList();
            if (planes.Count == 0)
            {
                error = $"{BasePath} has no {GrassName}/{GrassPlanePrefix}* renderers.";
                return null;
            }
            Vector2 planeSize = new Vector2(planes[0].bounds.size.x, planes[0].bounds.size.z);
            if (planes.Any(p => Mathf.Abs(p.bounds.size.x - planeSize.x) > 0.01f || Mathf.Abs(p.bounds.size.z - planeSize.y) > 0.01f))
            {
                error = "the grass planes are not all one size, so no single tiling gives them all the terrain's texel density.";
                return null;
            }
            Material material = GrassMaterial(grassLayer, planeSize);
            int moved = 0;
            foreach (Renderer plane in planes)
            {
                plane.sharedMaterials = Enumerable.Repeat(material, plane.sharedMaterials.Length).ToArray();
                if (plane.gameObject.layer != 0)
                {
                    plane.gameObject.layer = 0; // Default: see the class summary
                    moved++;
                }
                GameObjectUtility.SetStaticEditorFlags(plane.gameObject,
                    GameObjectUtility.GetStaticEditorFlags(plane.gameObject) | StaticEditorFlags.BatchingStatic);
            }
            int kerbs = 0;
            int nodes = 0;
            RenameTileMarkers(root.transform, ref kerbs, ref nodes);
            PrefabUtility.SaveAsPrefabAsset(root, BasePath, out bool saved);
            error = saved ? null : $"could not save {BasePath}.";
            return saved
                ? $"{planes.Count} base grass planes on {Path.GetFileName(GrassMaterialPath)} at " +
                  $"{material.mainTextureScale.x:F3} repeats ({moved} moved off Obstacle)"
                : null;
        }
        finally
        {
            PrefabUtility.UnloadPrefabContents(root);
        }
    }

    private static Material GrassMaterial(TerrainLayer layer, Vector2 planeSize)
    {
        Material material = AssetDatabase.LoadAssetAtPath<Material>(GrassMaterialPath);
        bool created = material == null;
        Shader standard = Shader.Find("Standard");
        if (created)
        {
            material = new Material(standard);
        }
        material.shader = standard;
        material.color = Color.white;
        material.SetTexture("_MainTex", layer.diffuseTexture);
        material.SetTextureScale("_MainTex", new Vector2(planeSize.x / layer.tileSize.x, planeSize.y / layer.tileSize.y));
        material.SetTextureOffset("_MainTex", Vector2.zero);
        material.SetFloat("_Glossiness", layer.smoothness);
        material.SetFloat("_Metallic", layer.metallic);
        if (layer.normalMapTexture != null)
        {
            material.SetTexture("_BumpMap", layer.normalMapTexture);
            material.SetFloat("_BumpScale", layer.normalScale);
            material.EnableKeyword("_NORMALMAP");
        }
        else
        {
            material.SetTexture("_BumpMap", null);
            material.DisableKeyword("_NORMALMAP");
        }
        if (created)
        {
            AssetDatabase.CreateAsset(material, GrassMaterialPath);
        }
        else
        {
            EditorUtility.SetDirty(material);
        }
        return material;
    }

    // ---------------------------------------------------------------- a diamond's park

    /// <summary>
    /// Rebuilds one diamond's park and saves it. Everything is measured in the prefab root's frame, which is the
    /// diamond's: its origin is the centre of the four blocks.
    /// </summary>
    private static string DressDiamond(DiamondSpec spec, List<GameObject> trees, List<TipSource> tips, out string error)
    {
        GameObject rootObject = PrefabUtility.LoadPrefabContents(spec.PrefabPath);
        try
        {
            Transform root = rootObject.transform;
            Transform oldPark = root.Find(ParkName);
            if (oldPark != null)
            {
                Object.DestroyImmediate(oldPark.gameObject);
            }

            // The lawn, read from the base.
            Transform grass = root.GetComponentsInChildren<Transform>(true).FirstOrDefault(t => t.name == GrassName);
            List<Renderer> planes = grass == null
                ? new List<Renderer>()
                : grass.GetComponentsInChildren<Renderer>(true).Where(r => r.name.StartsWith(GrassPlanePrefix)).ToList();
            if (planes.Count == 0)
            {
                error = $"no {GrassName}/{GrassPlanePrefix}* renderers to take the lawn from.";
                return null;
            }
            List<Rect> lawn = planes.Select(p => Footprint(p.bounds, root)).ToList();
            float groundY = planes.Max(p => root.InverseTransformPoint(p.bounds.max).y);

            int kerbsHidden = 0;
            int nodesRenamed = 0;
            RenameTileMarkers(root, ref kerbsHidden, ref nodesRenamed);

            // The plazas batch with the rest of the city; a patch prefab's own objects are not marked for it.
            Transform plazas = root.Find(PlazasName);
            if (plazas != null)
            {
                foreach (Renderer r in plazas.GetComponentsInChildren<Renderer>(true))
                {
                    GameObjectUtility.SetStaticEditorFlags(r.gameObject,
                        GameObjectUtility.GetStaticEditorFlags(r.gameObject) | StaticEditorFlags.BatchingStatic);
                }
            }

            Renderer fountain = root.GetComponentsInChildren<Transform>(true)
                                    .Where(t => t.name.StartsWith(FountainPrefix, System.StringComparison.OrdinalIgnoreCase))
                                    .SelectMany(t => t.GetComponentsInChildren<Renderer>())
                                    .FirstOrDefault();
            if (fountain == null)
            {
                error = $"no {FountainPrefix}* to centre the park on.";
                return null;
            }
            Vector2 centre = Footprint(fountain.bounds, root).center;

            List<Rect> keepOuts = KeepOuts(root);

            Transform park = CreateChild(root, ParkName);
            Transform tipsParent = CreateChild(park, "TipBuildings");
            Transform treesParent = CreateChild(park, "Trees");

            // Tips first: their footpaths end the avenues and are keep-outs for every tree.
            Dictionary<Tip, Rect> tipFootpaths = new Dictionary<Tip, Rect>();
            List<string> tipNotes = new List<string>();
            foreach (TipSource tip in tips)
            {
                Rect footpath = PlaceTip(tip, tipsParent, root, lawn, centre, out string note, out error);
                if (note == null)
                {
                    return null;
                }
                tipFootpaths[tip.Spec.Tip] = footpath;
                keepOuts.Add(footpath);
                tipNotes.Add(note);
            }

            // Trees, arrangement by arrangement.
            List<TreeGroup> groups = new List<TreeGroup> { new TreeGroup("FountainRing", spec.Ring, Ring(centre)) };
            foreach (Tip tip in Tips)
            {
                bool northSouth = tip == Tip.North || tip == Tip.South;
                groups.Add(new TreeGroup($"Avenue_{tip}", northSouth ? spec.NorthSouthAvenues : spec.EastWestAvenues,
                                         Avenue(tip, centre, tipFootpaths[tip], keepOuts, out float plazaFarEdge)));
                groups.Add(northSouth == spec.RowsNorthSouth
                    ? new TreeGroup($"EdgeRows_{tip}", spec.Rows, EdgeRows(tip, centre, lawn, plazaFarEdge))
                    : new TreeGroup($"Groves_{tip}", null, Groves(tip, centre, lawn, plazaFarEdge)));
            }

            System.Random rng = new System.Random(spec.Seed);
            int planted = 0;
            int offLawn = 0;
            int crowded = 0;
            List<string> counts = new List<string>();
            Dictionary<string, int> perModel = trees.ToDictionary(t => t.name, t => 0);
            foreach (TreeGroup group in groups)
            {
                Transform parent = CreateChild(treesParent, group.Name);
                int here = 0;
                foreach (Vector2 spot in group.Spots)
                {
                    // Every spot draws its numbers whether or not it is planted, so one skipped spot does not
                    // reshuffle every tree after it.
                    float yaw = (float)rng.NextDouble() * 360f;
                    float size = 1f + ((float)rng.NextDouble() * 2f - 1f) * TreeScaleJitter;
                    GameObject tree = group.Species != null ? trees.First(t => t.name == group.Species)
                                                            : trees[rng.Next(trees.Count)];
                    if (!InLawn(spot, lawn, LawnMargin))
                    {
                        offLawn++;
                        continue;
                    }
                    if (Distance(spot, keepOuts) < TreeClearance)
                    {
                        crowded++;
                        continue;
                    }
                    PlantTree(tree, parent, new Vector3(spot.x, groundY, spot.y), yaw, size);
                    perModel[tree.name]++;
                    here++;
                }
                planted += here;
                counts.Add($"{group.Name} {here}/{group.Spots.Count}");
            }

            PrefabUtility.SaveAsPrefabAsset(rootObject, spec.PrefabPath, out bool saved);
            if (!saved)
            {
                error = $"could not save {spec.PrefabPath}.";
                return null;
            }
            error = null;
            return $"Diamond_{spec.Letter}: planted {planted} trees ({string.Join(", ", counts)}; {offLawn} spot(s) off the " +
                   $"lawn, {crowded} too close to something; by model {string.Join(", ", perModel.Select(m => $"{m.Key} {m.Value}"))}) " +
                   $"round the fountain at ({centre.x:F1}, {centre.y:F1}); tips: {string.Join("; ", tipNotes)}" +
                   (kerbsHidden + nodesRenamed > 0
                       ? $"; {kerbsHidden} stray kerb(s) renamed and hidden, {nodesRenamed} MC_Patch node(s) renamed Block"
                       : "") + ".";
        }
        finally
        {
            PrefabUtility.UnloadPrefabContents(rootObject);
        }
    }

    // ---------------------------------------------------------------- tile markers

    /// <summary>
    /// Renames what the city tools recognise a tile by, as <see cref="DiamondCityBuilder"/> does for its drafts, and
    /// hides a plaza's kerb: round a plaza it is either a stray full-size outline or, drafted, a second kerb inside it.
    /// </summary>
    private static void RenameTileMarkers(Transform root, ref int kerbsHidden, ref int nodesRenamed)
    {
        foreach (Transform t in root.GetComponentsInChildren<Transform>(true))
        {
            if (t.name.StartsWith(CityTiles.KerbPrefix))
            {
                t.name = DiamondCityBuilder.DiamondKerbPrefix + t.name.Substring(CityTiles.KerbPrefix.Length);
                t.gameObject.SetActive(false);
                kerbsHidden++;
            }
            else if (t.name.StartsWith(DiamondCityBuilder.TileNamePrefix))
            {
                t.name = DiamondCityBuilder.DiamondContentPrefix + t.name.Substring(DiamondCityBuilder.TileNamePrefix.Length);
                nodesRenamed++;
            }
        }
    }

    // ---------------------------------------------------------------- tips

    /// <summary>
    /// Places a tip's building and footpath on the axis through <paramref name="centre"/>, the footpath's outer edge
    /// <see cref="TipInset"/> in from the lawn's edge, the building's long side along it. Returns the footpath's
    /// footprint, with <paramref name="note"/> describing it, or a null note with <paramref name="error"/> set.
    /// </summary>
    private static Rect PlaceTip(TipSource source, Transform parent, Transform root, List<Rect> lawn, Vector2 centre,
                                 out string note, out string error)
    {
        Tip tip = source.Spec.Tip;
        Vector2 axis = Axis(tip);
        bool alongX = tip == Tip.North || tip == Tip.South; // the direction the lawn's edge runs at this tip
        float reach = Reach(lawn, centre, axis);

        Transform group = CreateChild(parent, $"Tip_{tip}");
        GameObject building = Object.Instantiate(source.Building.gameObject, group);
        building.name = $"{source.Building.name} (tip {tip})"; // keeps its building-family prefix
        Transform b = building.transform;
        b.rotation = root.rotation * source.Building.rotation;
        b.localScale = source.Building.lossyScale; // its patch's hierarchy is unscaled above it
        // Heights are kept from the patch, measured from its road plate: its footpath and its building's base.
        float ground = PatchGround(source.Building.root);
        float pivotY = source.Building.position.y - ground;
        float footpathY = source.Footpath.bounds.center.y - ground;
        b.position = root.TransformPoint(new Vector3(0f, pivotY, 0f));
        Rect footprint = Footprint(RendererBounds(building), root);
        if ((footprint.width >= footprint.height) != alongX)
        {
            b.rotation = Quaternion.AngleAxis(90f, root.up) * b.rotation;
            footprint = Footprint(RendererBounds(building), root);
        }
        foreach (Transform part in building.GetComponentsInChildren<Transform>(true))
        {
            GameObjectUtility.SetStaticEditorFlags(part.gameObject, Everything);
        }
        float baseY = root.InverseTransformPoint(RendererBounds(building).min).y;
        if (Mathf.Abs(baseY) > MaxBaseHeight)
        {
            note = null;
            error = $"{source.Spec.Building} would stand {baseY:F1} units off the ground at the {tip} tip.";
            return default;
        }

        // The footpath: long side along the edge, outer edge TipInset in from it.
        Vector2 size = footprint.size + 2f * FootpathMargin * Vector2.one;
        float depth = alongX ? size.y : size.x;
        Vector2 middle = centre + axis * (reach - TipInset - depth / 2f);
        Rect footpath = new Rect(middle - size / 2f, size);

        // Centre the building on it.
        Vector3 shift = new Vector3(middle.x - footprint.center.x, 0f, middle.y - footprint.center.y);
        b.position += root.TransformVector(shift);
        footprint.center = middle;

        Mesh mesh = FootpathMesh($"{root.name}_Footpath_{tip}", size, FitUv(source.Footpath, out float residual));
        if (mesh == null)
        {
            note = null;
            error = $"could not write the footpath mesh for the {tip} tip.";
            return default;
        }
        GameObject slab = new GameObject("TipFootpath");
        SceneManager.MoveGameObjectToScene(slab, root.gameObject.scene);
        slab.transform.SetParent(group, false);
        slab.transform.localPosition = new Vector3(middle.x, footpathY, middle.y);
        slab.AddComponent<MeshFilter>().sharedMesh = mesh;
        slab.AddComponent<MeshRenderer>().sharedMaterials = source.Footpath.sharedMaterials;
        GameObjectUtility.SetStaticEditorFlags(slab, Everything);

        note = $"{tip} {source.Spec.Building} {footprint.width:F1}x{footprint.height:F1} on a {size.x:F1}x{size.y:F1} " +
               $"footpath at ({middle.x:F1}, {middle.y:F1})" + (residual > 1e-3f ? $" (footpath UV fit residual {residual:F4})" : "");
        error = null;
        return footpath;
    }

    /// <summary>
    /// The linear map a pack footpath uses from ground position to texture, so a generated slab of any size shows its
    /// paving at the same scale. Least squares over the mesh's vertices; <paramref name="residual"/> is the RMS miss,
    /// which is zero for a planar mapping.
    /// </summary>
    private static UvMap FitUv(Renderer footpath, out float residual)
    {
        Mesh mesh = footpath.GetComponent<MeshFilter>().sharedMesh;
        Vector3[] vertices = mesh.vertices;
        Vector2[] uvs = mesh.uv;
        Matrix4x4 toWorld = footpath.transform.localToWorldMatrix;

        // Normal equations for [x z 1] . p = u (and = v).
        double[,] a = new double[3, 3];
        double[] bu = new double[3];
        double[] bv = new double[3];
        Vector3[] rows = new Vector3[vertices.Length];
        for (int i = 0; i < vertices.Length; i++)
        {
            Vector3 w = toWorld.MultiplyPoint3x4(vertices[i]);
            rows[i] = new Vector3(w.x, w.z, 1f);
            for (int r = 0; r < 3; r++)
            {
                for (int c = 0; c < 3; c++)
                {
                    a[r, c] += rows[i][r] * rows[i][c];
                }
                bu[r] += rows[i][r] * uvs[i].x;
                bv[r] += rows[i][r] * uvs[i].y;
            }
        }
        Vector3 pu = Solve3(a, bu);
        Vector3 pv = Solve3(a, bv);

        double sq = 0;
        for (int i = 0; i < vertices.Length; i++)
        {
            float du = Vector3.Dot(rows[i], pu) - uvs[i].x;
            float dv = Vector3.Dot(rows[i], pv) - uvs[i].y;
            sq += du * du + dv * dv;
        }
        residual = (float)System.Math.Sqrt(sq / System.Math.Max(1, vertices.Length));
        return new UvMap(pu, pv);
    }

    private static Vector3 Solve3(double[,] a, double[] b)
    {
        double Det(double[,] m) =>
            m[0, 0] * (m[1, 1] * m[2, 2] - m[1, 2] * m[2, 1]) -
            m[0, 1] * (m[1, 0] * m[2, 2] - m[1, 2] * m[2, 0]) +
            m[0, 2] * (m[1, 0] * m[2, 1] - m[1, 1] * m[2, 0]);

        double d = Det(a);
        double[] x = new double[3];
        for (int k = 0; k < 3; k++)
        {
            double[,] m = (double[,])a.Clone();
            for (int r = 0; r < 3; r++)
            {
                m[r, k] = b[r];
            }
            x[k] = Det(m) / d;
        }
        return new Vector3((float)x[0], (float)x[1], (float)x[2]);
    }

    /// <summary>
    /// A flat quad of <paramref name="size"/> about its origin, facing up, textured by <paramref name="uv"/>, saved as
    /// its own asset in <see cref="GeneratedFolder"/>. An existing asset is rewritten in place, so its GUID and every
    /// reference to it survive a re-run.
    /// </summary>
    private static Mesh FootpathMesh(string name, Vector2 size, UvMap uv)
    {
        EnsureFolder(GeneratedFolder);
        string path = $"{GeneratedFolder}/{name}.asset";
        Mesh mesh = AssetDatabase.LoadAssetAtPath<Mesh>(path);
        bool created = mesh == null;
        if (created)
        {
            mesh = new Mesh();
        }
        mesh.Clear();
        mesh.name = name;
        float hx = size.x / 2f;
        float hz = size.y / 2f;
        Vector3[] vertices =
        {
            new Vector3(-hx, 0f, -hz), new Vector3(-hx, 0f, hz), new Vector3(hx, 0f, hz), new Vector3(hx, 0f, -hz),
        };
        mesh.vertices = vertices;
        mesh.normals = Enumerable.Repeat(Vector3.up, 4).ToArray();
        mesh.uv = vertices.Select(p => uv.At(p.x, p.z)).ToArray();
        mesh.triangles = new[] { 0, 1, 2, 0, 2, 3 }; // clockwise seen from above
        mesh.RecalculateBounds();
        mesh.RecalculateTangents();
        if (created)
        {
            AssetDatabase.CreateAsset(mesh, path);
        }
        else
        {
            EditorUtility.SetDirty(mesh);
        }
        return AssetDatabase.LoadAssetAtPath<Mesh>(path);
    }

    // ---------------------------------------------------------------- tree arrangements

    private static List<Vector2> Ring(Vector2 centre)
    {
        List<Vector2> spots = new List<Vector2>();
        for (int k = 0; k < RingTrees; k++)
        {
            float angle = (k + 0.5f) * 2f * Mathf.PI / RingTrees; // half a step off each axis
            spots.Add(centre + RingRadius * new Vector2(Mathf.Cos(angle), Mathf.Sin(angle)));
        }
        return spots;
    }

    /// <summary>
    /// A pair of rows either side of the axis, from past the ring — or past the far side of whatever stands across the
    /// avenue, which in the west and east arms is a plaza — to <see cref="AvenueStandOff"/> short of the tip's footpath.
    /// <paramref name="plazaFarEdge"/> is where that obstruction ends, as a distance along the axis.
    /// </summary>
    private static List<Vector2> Avenue(Tip tip, Vector2 centre, Rect tipFootpath, List<Rect> keepOuts, out float plazaFarEdge)
    {
        Vector2 axis = Axis(tip);
        Vector2 side = new Vector2(-axis.y, axis.x);
        float band = AvenueHalfWidth + TreeClearance;

        plazaFarEdge = 0f;
        foreach (Rect r in keepOuts)
        {
            if (r == tipFootpath)
            {
                continue;
            }
            Vector2 along = Extent(r, centre, axis);
            Vector2 across = Extent(r, centre, side);
            if (along.y > 0f && across.x < band && across.y > -band)
            {
                plazaFarEdge = Mathf.Max(plazaFarEdge, along.y);
            }
        }
        float start = Mathf.Max(RingRadius + AvenueGapToRing, plazaFarEdge + AvenueStandOff);
        float end = Extent(tipFootpath, centre, axis).x - AvenueStandOff;

        List<Vector2> spots = new List<Vector2>();
        for (float s = start; s <= end + 1e-3f; s += AvenueSpacing)
        {
            spots.Add(centre + axis * s + side * AvenueHalfWidth);
            spots.Add(centre + axis * s - side * AvenueHalfWidth);
        }
        return spots;
    }

    /// <summary>
    /// A staggered double row inside each long edge of an arm: from just past the corner where the arm leaves the cross,
    /// or past the plaza if that is further out, to just short of the tip, at about <see cref="EdgeRowSpacing"/> in the
    /// outer row.
    /// </summary>
    private static List<Vector2> EdgeRows(Tip tip, Vector2 centre, List<Rect> lawn, float plazaFarEdge)
    {
        Vector2 axis = Axis(tip);
        Vector2 side = new Vector2(-axis.y, axis.x);
        float reach = Reach(lawn, centre, axis);

        // The arm's two long edges, measured near its tip; then where each edge begins, found by walking out along
        // the axis just beyond the arm's side, which stays in the lawn until it leaves the cross's other arms.
        Vector2 nearTip = centre + axis * (reach - 10f);
        float left = Reach(lawn, nearTip, side);
        float right = Reach(lawn, nearTip, -side);
        float cornerLeft = Reach(lawn, centre + side * (left + 5f), axis);
        float cornerRight = Reach(lawn, centre - side * (right + 5f), axis);

        List<Vector2> spots = new List<Vector2>();
        foreach ((float edge, float corner, float sign) in new[] { (left, cornerLeft, 1f), (right, cornerRight, -1f) })
        {
            float a = Mathf.Max(corner, plazaFarEdge) + EdgeRowStartGap;
            float b = reach - EdgeRowEndGap;
            int count = Mathf.RoundToInt((b - a) / EdgeRowSpacing) + 1;
            if (b <= a || count < 2)
            {
                continue;
            }
            float step = (b - a) / (count - 1);
            for (int k = 0; k < count; k++)
            {
                spots.Add(centre + axis * (a + k * step) + side * sign * (edge - EdgeRowInset));
            }
            for (int k = 0; k < count - 1; k++)
            {
                spots.Add(centre + axis * (a + (k + 0.5f) * step) + side * sign * (edge - EdgeRowInset - EdgeRowGap));
            }
        }
        return spots;
    }

    /// <summary>
    /// A <see cref="GroveAlong"/> x <see cref="GroveAcross"/> grid either side of the avenue in an arm, centred in the
    /// lawn between the plaza (or the fountain) and the arm's end, and between the avenue and the arm's edge.
    /// </summary>
    private static List<Vector2> Groves(Tip tip, Vector2 centre, List<Rect> lawn, float plazaFarEdge)
    {
        Vector2 axis = Axis(tip);
        Vector2 side = new Vector2(-axis.y, axis.x);
        float reach = Reach(lawn, centre, axis);
        float along = ((plazaFarEdge + GroveInset) + (reach - GroveInset)) / 2f;

        List<Vector2> spots = new List<Vector2>();
        foreach (float sign in new[] { 1f, -1f })
        {
            float edge = Reach(lawn, centre + axis * along, side * sign);
            float across = ((AvenueHalfWidth + GroveGapToAvenue) + (edge - GroveInset)) / 2f;
            for (int i = 0; i < GroveAlong; i++)
            {
                for (int j = 0; j < GroveAcross; j++)
                {
                    float u = (i - (GroveAlong - 1) / 2f) * GroveSpacing;
                    float v = (j - (GroveAcross - 1) / 2f) * GroveSpacing;
                    spots.Add(centre + axis * (along + u) + side * sign * (across + v));
                }
            }
        }
        return spots;
    }

    private static void PlantTree(GameObject prefab, Transform parent, Vector3 localPosition, float yaw, float size)
    {
        GameObject tree = (GameObject)PrefabUtility.InstantiatePrefab(prefab, parent);
        Transform t = tree.transform;
        t.localPosition = localPosition;
        t.localRotation = Quaternion.AngleAxis(yaw, Vector3.up) * prefab.transform.localRotation;
        t.localScale = prefab.transform.localScale * (TreeScale * size);
        // Batched with the rest of the city once Play starts, as VergeTreePlanter's trees are.
        foreach (Transform part in tree.GetComponentsInChildren<Transform>(true))
        {
            GameObjectUtility.SetStaticEditorFlags(part.gameObject,
                GameObjectUtility.GetStaticEditorFlags(part.gameObject) | StaticEditorFlags.BatchingStatic);
        }
    }

    // ---------------------------------------------------------------- geometry, in the diamond's frame (x, z)

    private static Vector2 Axis(Tip tip)
    {
        switch (tip)
        {
            case Tip.North: return Vector2.up;
            case Tip.East: return Vector2.right;
            case Tip.South: return Vector2.down;
            default: return Vector2.left;
        }
    }

    /// <summary>Footprints of everything standing on the lawn, excluding the ground itself and the street medians.</summary>
    private static List<Rect> KeepOuts(Transform root)
    {
        List<Rect> rects = new List<Rect>();
        foreach (Renderer r in root.GetComponentsInChildren<Renderer>())
        {
            if (!r.enabled || HasAnyPrefix(r.name, GroundPrefixes) || UnderContainer(r.transform, root, CityTiles.StreetContainer))
            {
                continue;
            }
            rects.Add(Footprint(r.bounds, root));
        }
        return rects;
    }

    private static bool UnderContainer(Transform t, Transform root, string container)
    {
        return Ancestors(t, root).Any(p => p.name == container);
    }

    /// <summary>The parents of <paramref name="t"/>, nearest first, up to but excluding <paramref name="root"/>.</summary>
    private static IEnumerable<Transform> Ancestors(Transform t, Transform root)
    {
        for (Transform p = t.parent; p != null && p != root; p = p.parent)
        {
            yield return p;
        }
    }

    /// <summary>The horizontal footprint of world-space <paramref name="bounds"/> in <paramref name="frame"/>.</summary>
    private static Rect Footprint(Bounds bounds, Transform frame)
    {
        float xMin = float.MaxValue, xMax = float.MinValue, zMin = float.MaxValue, zMax = float.MinValue;
        for (int corner = 0; corner < 4; corner++)
        {
            Vector3 p = frame.InverseTransformPoint(new Vector3((corner & 1) == 0 ? bounds.min.x : bounds.max.x, bounds.center.y,
                                                                (corner & 2) == 0 ? bounds.min.z : bounds.max.z));
            xMin = Mathf.Min(xMin, p.x);
            xMax = Mathf.Max(xMax, p.x);
            zMin = Mathf.Min(zMin, p.z);
            zMax = Mathf.Max(zMax, p.z);
        }
        return Rect.MinMaxRect(xMin, zMin, xMax, zMax);
    }

    private static Bounds RendererBounds(GameObject go)
    {
        Renderer[] renderers = go.GetComponentsInChildren<Renderer>();
        Bounds bounds = renderers[0].bounds;
        foreach (Renderer r in renderers)
        {
            bounds.Encapsulate(r.bounds);
        }
        return bounds;
    }

    /// <summary>How far <paramref name="rect"/> runs along <paramref name="direction"/> from <paramref name="from"/>: (nearest, furthest).</summary>
    private static Vector2 Extent(Rect rect, Vector2 from, Vector2 direction)
    {
        float a = Vector2.Dot(rect.min - from, direction);
        float b = Vector2.Dot(rect.max - from, direction);
        float c = Vector2.Dot(new Vector2(rect.xMin, rect.yMax) - from, direction);
        float d = Vector2.Dot(new Vector2(rect.xMax, rect.yMin) - from, direction);
        return new Vector2(Mathf.Min(a, b, c, d), Mathf.Max(a, b, c, d));
    }

    /// <summary>How far the lawn runs from <paramref name="from"/> along <paramref name="direction"/> before it ends.</summary>
    private static float Reach(List<Rect> lawn, Vector2 from, Vector2 direction)
    {
        const float step = 0.1f;
        float s = 0f;
        while (s < 1000f && InUnion(from + direction * (s + step), lawn))
        {
            s += step;
        }
        return s;
    }

    private static bool InUnion(Vector2 p, List<Rect> rects) => rects.Any(r => r.Contains(p));

    /// <summary>Whether <paramref name="p"/>, and every point <paramref name="margin"/> from it, lies on the lawn.</summary>
    private static bool InLawn(Vector2 p, List<Rect> lawn, float margin)
    {
        if (!InUnion(p, lawn))
        {
            return false;
        }
        for (int k = 0; k < 8; k++)
        {
            float angle = k * Mathf.PI / 4f;
            if (!InUnion(p + margin * new Vector2(Mathf.Cos(angle), Mathf.Sin(angle)), lawn))
            {
                return false;
            }
        }
        return true;
    }

    private static float Distance(Vector2 p, List<Rect> rects)
    {
        float nearest = float.MaxValue;
        foreach (Rect r in rects)
        {
            float dx = Mathf.Max(r.xMin - p.x, 0f, p.x - r.xMax);
            float dz = Mathf.Max(r.yMin - p.y, 0f, p.y - r.yMax);
            nearest = Mathf.Min(nearest, Mathf.Sqrt(dx * dx + dz * dz));
        }
        return nearest;
    }

    // ---------------------------------------------------------------- plumbing

    private static Transform CreateChild(Transform parent, string name)
    {
        GameObject go = new GameObject(name);
        SceneManager.MoveGameObjectToScene(go, parent.gameObject.scene); // a new object starts in the active scene
        go.transform.SetParent(parent, false);
        return go.transform;
    }

    private static bool HasAnyPrefix(string name, string[] prefixes)
    {
        foreach (string prefix in prefixes)
        {
            if (name.StartsWith(prefix))
            {
                return true;
            }
        }
        return false;
    }

    private static void EnsureFolder(string folder)
    {
        if (AssetDatabase.IsValidFolder(folder))
        {
            return;
        }
        string parent = Path.GetDirectoryName(folder).Replace('\\', '/');
        EnsureFolder(parent);
        AssetDatabase.CreateFolder(parent, Path.GetFileName(folder));
    }
}
