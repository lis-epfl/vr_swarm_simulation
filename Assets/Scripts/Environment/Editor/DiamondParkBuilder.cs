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
/// paved blocks whose buildings screen the north–south streets; and <c>Park</c>, the trees and tip buildings. Editing the base
/// changes all four. It is a nested prefab rather than a Unity Prefab Variant on purpose: a variant's root is a new
/// object, so turning the existing prefabs into variants would orphan the root-position overrides DiamondCityWorld holds
/// for each placed diamond, and every diamond would jump to its prefab's stored position.</para>
///
/// <para><b>What a run does</b>, all of it idempotent: creates the base from <see cref="BaseSourceLetter"/> if it does not
/// exist; puts any diamond not yet on the base onto it; gives any diamond without them its spec's two plazas (see
/// <see cref="EnsurePlazas"/>); gives the base's grass planes the terrain's grass; and rebuilds every diamond's
/// <c>Park</c>, and every plaza's furniture, from nothing. So the layout is this file, not hand edits: change a constant
/// or a spec and run it again. Hand edits to the base and to the plazas' buildings survive a run; hand edits under
/// <c>Park</c> or a plaza's <c>Props</c> do not. To change a plaza's
/// screen, edit the spec and run <c>Tools/Swarm/Build diamond parks, redrafting plazas</c> (or delete that diamond's
/// <c>Plazas</c> to redraft only its own). Deleting a diamond's prefab gets
/// <see cref="DiamondCityBuilder"/>'s tile-copy draft back, which the next run puts on the base again.</para>
///
/// <para><b>A plaza</b> (<see cref="DraftPlaza"/>) is what Diamond_B's were first made by hand as: a footpath, here
/// exactly a city block's at the city's block scale, with buildings standing across the north–south street the diamond
/// blocks — the diamonds exist to cut the long sight lines down those streets, and the plazas are what does it. It is no
/// patch: no road ring, kerb, verge or interior streets. Each carries a <see cref="DiamondPlaza"/>, through which
/// <see cref="GoalPatchReplacer"/> may replace it with a goal, which comes out exactly the plaza's size. Its furniture —
/// park lamps, benches, bins, hedges and planters, copied from the pack's own park block — is laid out afresh on every
/// run from where its buildings stand (<see cref="FurnishPlaza"/>), and goes with the plaza when a goal replaces it.</para>
///
/// <para><b>The park</b> (~115–120 trees) is measured from what the diamond holds — the lawn is the union of the grass
/// planes, the centre is the fountain, the axes run through it, and whatever stands on the lawn (plaza footpaths,
/// buildings, props) is a keep-out — and a spot that fails the lawn or keep-out test is skipped and counted, never
/// nudged. A ring round the fountain, open on the four axes; an avenue (a pair of rows) down each axis, from the ring or
/// the far side of a plaza out towards the tip; a staggered double row inside both long edges of two opposite arms, and
/// a small grid grove either side of the avenue in the other two; and at each of the four tips a single 50-unit building
/// on a footpath slab slightly larger than its footprint. Then its furniture (<see cref="FurnishPark"/>), laid out by
/// the trees and kept clear of their trunks: a round of paving about the fountain with benches facing the water, and
/// benches, bins and lamps down the avenues, under the double rows and along the fronts of the groves.</para>
///
/// <para><b>What makes each diamond distinct</b> is its <see cref="DiamondSpec"/>: its screens, its four tip buildings,
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
/// <c>Kerb_NN</c>, and its <c>MC_Patch</c> nodes become <c>Block</c>, as <see cref="DiamondCityBuilder"/> does for its
/// drafts. A kerb under the diamonds' root stops <see cref="CityTiles.FindCity(Scene, out string)"/> finding the city at
/// all — which silently stops goal-patch verge planting at runtime — and <see cref="CityObstacleExport"/> lists every
/// <c>MC_Patch</c> node as a tile. That is why a plaza is found by its <see cref="DiamondPlaza"/> instead.</para>
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
    /// The city's block scale (<see cref="StreetWidthTuner"/> in DiamondCityWorld). A plaza's footpath is a block's
    /// footpath at this scale, which is the size a goal patch replacing it is brought to at runtime. Change it with the
    /// city's.
    /// </summary>
    private const float PlazaBlockScale = 0.8f;

    /// <summary>
    /// How far a plaza's ground stands above the diamond's. The lawn is 4 cm up, so a goal replacing a plaza, standing at
    /// road level, would have its interior streets under the grass; this puts them 6 cm over it. See
    /// <see cref="DiamondPlaza"/>.
    /// </summary>
    private const float PlazaLift = 0.1f;

    /// <summary>The patch whose footpath a plaza's paving copies (a plain slab, with no interior streets).</summary>
    private const string PlazaFootpathSource = "MC_Patch_02_Scaled";

    /// <summary>
    /// Half the open width of a north–south street where a plaza stands across it: from the building zone of the column
    /// on one side to that of the column on the other, street and verges together, at the city's block scale.
    /// </summary>
    private const float CorridorHalfWidth = CityTiles.Pitch / 2f - CityTiles.BlockHalfSpan * PlazaBlockScale;

    /// <summary>The share of a street's open width a plaza's screen must block: a majority, as on Diamond_B's.</summary>
    private const float ScreenCoverage = 0.6f;

    /// <summary>How much of it a screen is steered towards blocking; past this, spacing its buildings out counts for more.</summary>
    private const float ScreenPreferredCoverage = 0.8f;

    private const float ScreenClearance = 3f; // at least this far between two of a screen's buildings
    private const float ScreenSpill = 6f;     // how far past the street's open width a building's centre may stand
    private const int ScreenAttempts = 20000; // random layouts tried at most
    private const int ScreenCandidates = 400; // qualifying layouts compared before the best is kept
    private const float PlazaMinMargin = 1f;  // from a building to its plaza's footpath edge

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

    private enum Arrangement
    {
        Ring,
        Avenue,
        EdgeRows,
        Groves,
    }

    /// <summary>
    /// One arrangement: where its trees go, and which model they are (null: a random one each), and what kind of
    /// arrangement it is in which arm, which is what the park's furniture is laid out by.
    /// </summary>
    private readonly struct TreeGroup
    {
        public readonly string Name;
        public readonly string Species;
        public readonly List<Vector2> Spots;
        public readonly Arrangement Kind;
        public readonly Tip Arm; // the ring's is North, and means nothing

        public TreeGroup(string name, string species, List<Vector2> spots, Arrangement kind, Tip arm)
        {
            Name = name;
            Species = species;
            Spots = spots;
            Kind = kind;
            Arm = arm;
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

    /// <summary>A building on a plaza's screen: the ScaledCity patch prefab it is copied from, and its name there.</summary>
    private readonly struct ScreenBuilding
    {
        public readonly string Patch;
        public readonly string Building;

        public ScreenBuilding(string patch, string building)
        {
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

    // ---------------------------------------------------------------- furniture, on the plazas and in the park

    /// <summary>
    /// The patch whose park block (<c>Garden_02</c>) all the furniture is copied from, as the buildings are copied from
    /// theirs, so it is exactly the city's own: meshes, materials, and the small colliders on Default.
    /// </summary>
    private const string PropPatch = "MC_Patch_16_Scaled";

    private const string LampProp = "BnP_Street_Lamp_Double_000"; // a 3 m park lamp with two arms
    private const string ParkLampProp = "BnP_Street_Lamp_005";     // one head, as down the pack park's walks
    private const string BenchProp = "BnP_Bench_000 1";
    private const string BinProp = "BnP_Trash_Can_0123";
    private const string BedProp = "Rectangle_Grass_000"; // a strip of grass in the paving, which a hedge stands on
    private const string HedgeProp = "Bush_006";
    private static readonly string[] PlanterProps = { "BnP_Plant_000 1", "BnP_Plant_012 1", "BnP_Plant_017 1" };

    private const string PropsName = "Props";
    private const string PlazaFootpathName = "PlazaFootpath";

    // Stations along each of a plaza's four edges, measured from the edge's middle, and how far in from the edge each
    // line of furniture stands; the lamps' arms reach across their line, as they do in the pack's park.
    private static readonly float[] LampStations = { -25f, -15f, -5f, 5f, 15f, 25f };
    private static readonly float[] BenchStations = { -10f, 10f };
    private static readonly float[] BedStations = { -20f, 0f, 20f };
    private const float LampInset = 1f;
    private const float FurnitureInset = 2.6f; // benches, bins and beds, just inside the lamps
    private const float BinBesideBench = 1.6f; // centre to centre, on the side towards the middle of the edge

    private const float CornerPlanterInset = 1.5f; // a planter this far in from each of a corner's edges, and one either side of it
    private const float BuildingPlanterOffset = 1f; // a planter off each corner of a building, this far out on either axis

    private const float PropBuildingClearance = 1f; // no nearer a building, or anything else standing on the lawn, than this
    private const float PropSpacing = 0.3f;         // nor another prop
    private const float PropEdgeMargin = 0.1f;      // and wholly on the paving

    // The park's: a round of paving about the fountain, inside the ring of trees, and what stands on it. Angles are
    // about the fountain; the benches stand in pairs either side of each diagonal, so the four axes stay open as ways in.
    private const float FountainPavingRadius = 8f;
    private const int FountainPavingSides = 48;
    private const float FountainPavingLift = 0.05f;  // above the lawn
    private const float FountainBenchRadius = 6.2f;  // to a bench's middle; it faces the water
    private const float FountainBenchSpread = 20f;   // degrees either side of a diagonal; a bin stands on the diagonal
    private const float FountainLampRadius = 7.3f;   // a lamp behind each bin
    private const float FountainPlanterRadius = 7.3f;
    private const float FountainPlanterSpread = 11f; // degrees either side of an axis, flanking the way in

    // Measured across an arm from its axis, as the trees are.
    private const float AvenueLampOffset = 5f;    // the avenue's trees stand at AvenueHalfWidth
    private const float AvenueBenchOffset = 4.6f; // to a bench's middle, its back to the trees
    private const float RowBenchOffset = 2f;      // in front of a double row's inner line, towards the axis

    private const float PropTrunkClearance = 1f;
    private const float LawnPropMargin = 0.5f; // in from the lawn's edge

    private sealed class PropKit
    {
        public Transform Lamp;
        public Transform ParkLamp;
        public Transform Bench;
        public Transform Bin;
        public Transform Bed;
        public Transform Hedge;
        public Transform[] Planters;
        public Renderer Paving; // the plazas' paving, which the fountain's is too
    }

    /// <summary>How a copied prop is turned: left as it stood in its patch, long side along a direction, or its front towards one.</summary>
    private enum PropTurn
    {
        AsIs,
        LongAlong,
        FrontTowards,
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

        /// <summary>
        /// The buildings screening the street each plaza stands across, scattered over it by <see cref="ScatterScreen"/>.
        /// Drafted only into a diamond without its plazas, or when asked to redraft; see <see cref="EnsurePlazas"/>.
        /// </summary>
        public ScreenBuilding[] WestScreen;
        public ScreenBuilding[] EastScreen;

        public TipBuilding[] TipBuildings;

        public string PrefabPath => $"{DiamondCityBuilder.DiamondFolder}/Diamond_{Letter}.prefab";
    }

    private static ScreenBuilding S(string patch, string building) => new ScreenBuilding(patch, building);

    /// <summary>
    /// Slots are (column, row) in <see cref="DiamondCityBuilder"/>'s layout; every source listed is at least three slots
    /// from its diamond, and no building design appears twice in one diamond. B's screens are the buildings its
    /// hand-made plazas had, from patches 02 and 43; its other values are what it was first dressed with. A screen's
    /// buildings must be wide enough between them to block <see cref="ScreenCoverage"/> of the street's ~30-unit open
    /// width, so two to four of the pack's narrow buildings.
    /// </summary>
    private static readonly DiamondSpec[] Diamonds =
    {
        new DiamondSpec
        {
            Letter = 'A', Seed = 2, RowsNorthSouth = false,
            Ring = "Tree9_2", NorthSouthAvenues = "Tree9_5", EastWestAvenues = "Tree9_3", Rows = "Tree9_4",
            WestScreen = new[]
            {
                S("MC_Patch_35_Scaled", "Skyscraper_D_003"), S("MC_Patch_35_Scaled", "BnP_Large_Building_D_001"),
                S("MC_Patch_35_Scaled", "Skyscraper_C_003"),
            },
            EastScreen = new[]
            {
                S("MC_Patch_26_Scaled", "BnP_Small_Building_A_002"), S("MC_Patch_26_Scaled", "BnP_Apartment_I_000"),
                S("MC_Patch_26_Scaled", "BnP_Small_Building_F_006"), S("MC_Patch_04_Scaled", "BnP_Small_Building_C_002"),
            },
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
            WestScreen = new[]
            {
                S("MC_Patch_02_Scaled", "Skyscraper_A_000"), S("MC_Patch_02_Scaled", "BnP_Apartment_G_003"),
                S("MC_Patch_02_Scaled", "BnP_Small_Building_E_026"), S("MC_Patch_02_Scaled", "BnP_Large_Building_C_012"),
            },
            EastScreen = new[]
            {
                S("MC_Patch_43_Scaled", "BnP_Apartment_C_006"), S("MC_Patch_43_Scaled", "BnP_Large_Building_A_003"),
            },
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
            WestScreen = new[]
            {
                S("MC_Patch_12_Scaled", "BnP_Apartment_D_007"), S("MC_Patch_12_Scaled", "Skyscraper_H_002"),
                S("MC_Patch_12_Scaled", "BnP_Apartment_D_011"),
            },
            EastScreen = new[]
            {
                S("MC_Patch_06_Scaled", "Skyscraper_J_000"), S("MC_Patch_06_Scaled", "BnP_Apartment_F_008"),
                S("MC_Patch_06_Scaled", "BnP_Small_Building_E_032"),
            },
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
            WestScreen = new[]
            {
                S("MC_Patch_30_Scaled", "BnP_Large_Building_B_001"), S("MC_Patch_17_Scaled", "BnP_Apartment_C_005"),
            },
            EastScreen = new[]
            {
                S("MC_Patch_38_Scaled", "Skyscraper_G_000"), S("MC_Patch_38_Scaled", "BnP_Large_Building_E_004"),
                S("MC_Patch_38_Scaled", "BnP_Small_Building_F_013"),
            },
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
        if (PrefabModeOpen())
        {
            return;
        }
        Report(Build(out string error), error);
    }

    [MenuItem("Tools/Swarm/Build diamond parks, redrafting plazas")]
    private static void RedraftFromMenu()
    {
        if (PrefabModeOpen() ||
            !EditorUtility.DisplayDialog("Build diamond parks, redrafting plazas",
                "Redraw every diamond's two plazas from its spec, as well as rebuilding the parks? Anything changed by " +
                "hand under a diamond's Plazas is lost.", "Redraft", "Cancel"))
        {
            return;
        }
        Report(Build(out string error, redraftPlazas: true), error);
    }

    [MenuItem("Tools/Swarm/Build diamond parks", true)]
    [MenuItem("Tools/Swarm/Build diamond parks, redrafting plazas", true)]
    private static bool CanBuild()
    {
        return !EditorApplication.isPlayingOrWillChangePlaymode;
    }

    private static bool PrefabModeOpen()
    {
        PrefabStage stage = PrefabStageUtility.GetCurrentPrefabStage();
        if (stage != null && (stage.assetPath == BasePath || Diamonds.Any(d => d.PrefabPath == stage.assetPath)))
        {
            EditorUtility.DisplayDialog("Build diamond parks",
                $"{Path.GetFileName(stage.assetPath)} is open in Prefab Mode. Save and close it first: the parks are written " +
                "into the prefab assets, and saving the open stage afterwards would overwrite them.", "OK");
            return true;
        }
        return false;
    }

    /// <summary>
    /// Batch-mode entry point: <c>-executeMethod DiamondParkBuilder.BuildInBatch</c>, with the environment variable
    /// <c>DIAMOND_REDRAFT_PLAZAS=1</c> to redraft the plazas too. Exits 0 on success.
    /// </summary>
    public static void BuildInBatch()
    {
        bool redraft = System.Environment.GetEnvironmentVariable("DIAMOND_REDRAFT_PLAZAS") == "1";
        string summary = Build(out string error, redraft);
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
    /// Puts every diamond on the base and rebuilds every park, and with <paramref name="redraftPlazas"/> every diamond's
    /// plazas as well. Returns an account of what it did, or null with <paramref name="error"/> set. Everything it needs is
    /// found before anything is changed; a failure part-way leaves the prefabs already saved as they are, and a re-run
    /// carries on from there.
    /// </summary>
    public static string Build(out string error, bool redraftPlazas = false)
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
        PropKit props = LoadProps(out error);
        if (props == null)
        {
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

        string structure = EnsureStructure(redraftPlazas, out error);
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
            string park = DressDiamond(spec, trees, tips[spec.Letter], props, out error);
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
            Transform source = LoadBuilding(building.Patch, building.Building, out error);
            Renderer footpath = source != null ? LoadFootpath(building.Patch, out error) : null;
            if (footpath == null)
            {
                return null;
            }
            sources.Add(new TipSource { Spec = building, Building = source, Footpath = footpath });
        }

        // The screens are only read when a plaza is drafted, but a typo in one should not wait for that to be found.
        foreach (ScreenBuilding entry in spec.WestScreen.Concat(spec.EastScreen))
        {
            if (LoadBuilding(entry.Patch, entry.Building, out error) == null)
            {
                error = $"Diamond_{spec.Letter}'s screen: {error}";
                return null;
            }
        }
        error = null;
        return sources;
    }

    // ---------------------------------------------------------------- the base, and putting diamonds on it

    /// <summary>
    /// Creates the base if it is missing, puts every diamond not yet on it onto it, and gives each its plazas (all of
    /// them afresh with <paramref name="redraftPlazas"/>). Returns what it did, or null with <paramref name="error"/> set.
    /// </summary>
    private static string EnsureStructure(bool redraftPlazas, out string error)
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

        foreach (DiamondSpec spec in Diamonds)
        {
            GameObject root = PrefabUtility.LoadPrefabContents(spec.PrefabPath);
            try
            {
                bool changed = false;
                if (!IsOnBase(root.transform))
                {
                    PutOnBase(root.transform, basePrefab);
                    notes.Add($"Diamond_{spec.Letter} put on the base");
                    changed = true;
                }
                if (!EnsurePlazas(root.transform, spec, redraftPlazas, out string plazaNote, out error))
                {
                    error = $"Diamond_{spec.Letter}: {error}";
                    return null;
                }
                if (plazaNote != null)
                {
                    notes.Add(plazaNote);
                    changed = true;
                }
                if (!changed)
                {
                    continue;
                }
                PrefabUtility.SaveAsPrefabAsset(root, spec.PrefabPath, out bool saved);
                if (!saved)
                {
                    error = $"could not save {spec.PrefabPath}.";
                    return null;
                }
            }
            finally
            {
                PrefabUtility.UnloadPrefabContents(root);
            }
        }
        error = null;
        return notes.Count == 0 ? "every diamond already on the base with its plazas" : string.Join("; ", notes);
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
                if (child.name == ParkName || child.name == PlazasName || IsPatchInstance(child))
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
    /// Replaces a diamond's content with the base. The root itself is kept, which is what keeps DiamondCityWorld's
    /// placement of it valid.
    /// </summary>
    private static void PutOnBase(Transform root, GameObject basePrefab)
    {
        foreach (Transform child in root.Cast<Transform>().ToList())
        {
            Object.DestroyImmediate(child.gameObject);
        }
        GameObject baseInstance = (GameObject)PrefabUtility.InstantiatePrefab(basePrefab, root);
        baseInstance.name = BaseName;
        baseInstance.transform.localPosition = Vector3.zero;
        baseInstance.transform.localRotation = Quaternion.identity;
        baseInstance.transform.localScale = Vector3.one;
        baseInstance.transform.SetSiblingIndex(0);
    }

    /// <summary>
    /// Gives a diamond its spec's two plazas unless it already has the two <see cref="DraftPlaza"/> makes and
    /// <paramref name="redraft"/> is off. Anything else — none, or the ScaledCity patches the diamonds' plazas used to be —
    /// is replaced. Each stands across one of the two north–south streets the diamond blocks, the
    /// pair symmetric about the fountain: both are set at the fountain's distance from the further street, so the further
    /// plaza is centred on its street and the nearer one still spans its own. <paramref name="note"/> is null when nothing
    /// needed doing.
    /// </summary>
    private static bool EnsurePlazas(Transform root, DiamondSpec spec, bool redraft, out string note, out string error)
    {
        note = null;
        Transform container = root.Find(PlazasName);
        List<Transform> patches = FindPatchPlazas(root);
        int drafted = container == null
            ? 0
            : container.Cast<Transform>().Count(t => t.GetComponent<DiamondPlaza>() != null && !IsPatchInstance(t));
        if (drafted == 2 && patches.Count == 0 && !redraft)
        {
            error = null;
            return true;
        }
        foreach (Transform patch in patches)
        {
            Object.DestroyImmediate(patch.gameObject);
        }
        if (container != null)
        {
            Object.DestroyImmediate(container.gameObject);
        }
        Transform parent = CreateChild(root, PlazasName);
        parent.SetSiblingIndex(Mathf.Min(1, root.childCount - 1));

        Renderer fountain = FindFountain(root);
        Renderer paving = LoadFootpath(PlazaFootpathSource, out error);
        if (fountain == null || paving == null)
        {
            error = fountain == null ? $"no {FountainPrefix}* to place the plazas about." : error;
            return false;
        }
        Vector2 f = Footprint(fountain.bounds, root).center;
        float offset = Mathf.Max(Mathf.Abs(-CityTiles.Pitch / 2f - f.x), Mathf.Abs(CityTiles.Pitch / 2f - f.x));

        List<string> notes = new List<string>();
        foreach ((string side, float sign, ScreenBuilding[] screen) in new[] { ("West", -1f, spec.WestScreen), ("East", 1f, spec.EastScreen) })
        {
            // Named for its diamond too: a goal replacing it is named for it, and so is its entry in CityObstacleExport.
            Vector2 centre = new Vector2(f.x + sign * offset, f.y);
            string plazaNote = DraftPlaza($"{spec.Letter}_{side}", screen, paving, parent, root, centre,
                                          sign * CityTiles.Pitch / 2f, spec.Seed * 31 + (sign > 0f ? 1 : 0), out error);
            if (plazaNote == null)
            {
                error = $"{side.ToLowerInvariant()} plaza: {error}";
                return false;
            }
            notes.Add(plazaNote);
        }
        note = $"Diamond_{spec.Letter} plazas drafted: {string.Join("; ", notes)}";
        error = null;
        return true;
    }

    /// <summary>
    /// A plaza, the way Diamond_B's were first made by hand: a footpath the size of a city block at the city's block
    /// scale — the size a goal patch replacing it comes out at — centred on <paramref name="centre"/>, with a screen of
    /// buildings standing across the north–south street at <paramref name="streetX"/>, and a <see cref="DiamondPlaza"/>
    /// so <see cref="GoalPatchReplacer"/> can replace it. It has no road ring, kerb or verge; the park's lawn is round it.
    ///
    /// <para>Blocking the street is the plaza's purpose — the diamonds exist to cut the long sight lines down the
    /// north–south streets — and it is enough that most of it is blocked, as on Diamond_B's hand-made plazas (52% and
    /// 74% of the street's open width). So the buildings, long side across the street, are scattered over the plaza
    /// rather than packed into a wall: see <see cref="ScatterScreen"/>. The street's open width runs from the building
    /// zone of the column on one side to that of the column on the other (<see cref="CorridorHalfWidth"/>), verges
    /// included.</para>
    ///
    /// <para>The plaza's ground is <see cref="PlazaLift"/> above the diamond's, for the goal that may replace it; its
    /// paving and buildings stand on that ground at their pack heights.</para>
    /// </summary>
    private static string DraftPlaza(string side, ScreenBuilding[] screen, Renderer paving, Transform parent, Transform root,
                                     Vector2 centre, float streetX, int seed, out string error)
    {
        Transform plaza = CreateChild(parent, $"Plaza_{side}");
        plaza.localPosition = new Vector3(centre.x, PlazaLift, centre.y); // the container sits at the diamond's origin
        float size = 2f * CityTiles.BlockHalfSpan * PlazaBlockScale;

        Mesh mesh = FootpathMesh("Diamond_Plaza_Footpath", new Vector2(size, size), FitUv(paving, out _));
        if (mesh == null)
        {
            error = "could not write the plaza footpath mesh.";
            return null;
        }
        GameObject slab = new GameObject(PlazaFootpathName);
        SceneManager.MoveGameObjectToScene(slab, root.gameObject.scene);
        slab.transform.SetParent(plaza, false);
        slab.transform.localPosition = new Vector3(0f, paving.bounds.center.y - PatchGround(paving.transform.root), 0f);
        slab.AddComponent<MeshFilter>().sharedMesh = mesh;
        slab.AddComponent<MeshRenderer>().sharedMaterials = paving.sharedMaterials;
        GameObjectUtility.SetStaticEditorFlags(slab, Everything);

        // The screen: copy the buildings, long side across the street, then scatter them.
        List<GameObject> copies = new List<GameObject>();
        List<Rect> footprints = new List<Rect>();
        foreach (ScreenBuilding entry in screen)
        {
            Transform source = LoadBuilding(entry.Patch, entry.Building, out error);
            if (source == null)
            {
                return null;
            }
            GameObject copy = CopyBuilding(source, plaza, root, true, PlazaLift, $"plaza {side}", out Rect footprint, out error);
            if (copy == null)
            {
                return null;
            }
            copies.Add(copy);
            footprints.Add(footprint);
        }
        Rect[] layout = ScatterScreen(footprints, centre, size, streetX, seed, out float coverage, out float widestGap,
                                      out float spread);
        if (layout == null)
        {
            error = $"no way to scatter its {copies.Count} buildings that blocks {ScreenCoverage:P0} of the street at x " +
                    $"{streetX:F1}; give it wider buildings or more of them.";
            return null;
        }
        for (int i = 0; i < copies.Count; i++)
        {
            MoveFootprint(copies[i], root, footprints[i], layout[i].center);
        }

        plaza.gameObject.AddComponent<DiamondPlaza>().Initialise(plaza, 0f);
        error = null;
        return $"{side} {copies.Count} buildings blocking {coverage:P0} of the street (widest gap {widestGap:F1}, " +
               $"buildings at least {spread:F1} apart)";
    }

    /// <summary>
    /// Where a screen's buildings stand, as footprints in the diamond's frame: scattered over the plaza — each on the
    /// footpath and centred within <see cref="ScreenSpill"/> of the street's open width, no two closer than
    /// <see cref="ScreenClearance"/> — and together blocking at least <see cref="ScreenCoverage"/> of that width. Drawn
    /// from <paramref name="seed"/>, so a plaza comes out the same every time it is drafted. Of the layouts that qualify
    /// it keeps the one that blocks the most, up to <see cref="ScreenPreferredCoverage"/>, weighed against how far apart
    /// the buildings stand, so the screen is neither a wall nor a scatter with the street left mostly open. Null if no
    /// layout qualifies.
    /// </summary>
    private static Rect[] ScatterScreen(List<Rect> footprints, Vector2 centre, float size, float streetX, int seed,
                                        out float coverage, out float widestGap, out float spread)
    {
        System.Random rng = new System.Random(seed);
        float inner = size / 2f - PlazaMinMargin;
        float s0 = streetX - CorridorHalfWidth;
        float s1 = streetX + CorridorHalfWidth;
        Rect[] best = null;
        float bestScore = float.MinValue;
        coverage = widestGap = spread = 0f;

        int qualifying = 0;
        for (int attempt = 0; attempt < ScreenAttempts && qualifying < ScreenCandidates; attempt++)
        {
            Rect[] rects = new Rect[footprints.Count];
            bool fits = true;
            for (int i = 0; i < rects.Length && fits; i++)
            {
                Vector2 s = footprints[i].size;
                float xLo = Mathf.Max(s0 - ScreenSpill, centre.x - inner + s.x / 2f);
                float xHi = Mathf.Min(s1 + ScreenSpill, centre.x + inner - s.x / 2f);
                float zLo = centre.y - inner + s.y / 2f;
                float zHi = centre.y + inner - s.y / 2f;
                if (xHi < xLo || zHi < zLo)
                {
                    return null; // too big for the plaza
                }
                Vector2 c = new Vector2(Range(rng, xLo, xHi), Range(rng, zLo, zHi));
                rects[i] = new Rect(c - s / 2f, s);
                for (int j = 0; j < i && fits; j++)
                {
                    fits = Gap(rects[i], rects[j]) >= ScreenClearance;
                }
            }
            if (!fits)
            {
                continue;
            }
            float blocked = Coverage(rects, s0, s1, out float gap);
            if (blocked < ScreenCoverage)
            {
                continue;
            }
            qualifying++;
            float apart = float.MaxValue;
            for (int i = 0; i < rects.Length; i++)
            {
                for (int j = i + 1; j < rects.Length; j++)
                {
                    apart = Mathf.Min(apart, Gap(rects[i], rects[j]));
                }
            }
            float score = Mathf.Min(blocked, ScreenPreferredCoverage) + 0.02f * Mathf.Min(apart, 10f);
            if (score > bestScore)
            {
                bestScore = score;
                best = rects;
                coverage = blocked;
                widestGap = gap;
                spread = rects.Length > 1 ? apart : 0f;
            }
        }
        return best;
    }

    /// <summary>
    /// The share of [<paramref name="from"/>, <paramref name="to"/>] that the rects' x-extents cover, and the widest part
    /// of it they leave open.
    /// </summary>
    private static float Coverage(Rect[] rects, float from, float to, out float widestGap)
    {
        float covered = 0f;
        float reached = from;
        widestGap = 0f;
        foreach (Rect r in rects.OrderBy(r => r.xMin))
        {
            float a = Mathf.Clamp(r.xMin, from, to);
            float b = Mathf.Clamp(r.xMax, from, to);
            if (a > reached)
            {
                widestGap = Mathf.Max(widestGap, a - reached);
            }
            if (b > reached)
            {
                covered += b - Mathf.Max(a, reached);
                reached = b;
            }
        }
        widestGap = Mathf.Max(widestGap, to - reached);
        return covered / (to - from);
    }

    /// <summary>How far apart two footprints stand: positive when separated on either axis, negative when they overlap.</summary>
    private static float Gap(Rect a, Rect b)
    {
        return Mathf.Max(a.xMin - b.xMax, b.xMin - a.xMax, a.yMin - b.yMax, b.yMin - a.yMax);
    }

    private static float Range(System.Random rng, float min, float max)
    {
        return min + (float)rng.NextDouble() * (max - min);
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

    /// <summary>The fountain the park is centred on, from the base.</summary>
    private static Renderer FindFountain(Transform root)
    {
        return root.GetComponentsInChildren<Transform>(true)
                   .Where(t => t.name.StartsWith(FountainPrefix, System.StringComparison.OrdinalIgnoreCase))
                   .SelectMany(t => t.GetComponentsInChildren<Renderer>())
                   .FirstOrDefault();
    }

    /// <summary>Whether <paramref name="root"/> already holds the base under <see cref="BaseName"/>.</summary>
    private static bool IsOnBase(Transform root)
    {
        Transform b = root.Find(BaseName);
        return b != null && PrefabUtility.IsOutermostPrefabInstanceRoot(b.gameObject) &&
               PrefabUtility.GetPrefabAssetPathOfNearestInstanceRoot(b.gameObject) == BasePath;
    }

    /// <summary>
    /// Plazas of the old kind: ScaledCity patch instances, at a diamond's top level (as Diamond_B's were made by hand) or
    /// under <see cref="PlazasName"/>.
    /// </summary>
    private static List<Transform> FindPatchPlazas(Transform root)
    {
        Transform container = root.Find(PlazasName);
        return root.Cast<Transform>()
                   .Concat(container != null ? container.Cast<Transform>() : Enumerable.Empty<Transform>())
                   .Where(IsPatchInstance)
                   .ToList();
    }

    private static bool IsPatchInstance(Transform t)
    {
        return PrefabUtility.IsOutermostPrefabInstanceRoot(t.gameObject) &&
               PrefabUtility.GetPrefabAssetPathOfNearestInstanceRoot(t.gameObject).StartsWith(PatchFolder + "/");
    }

    // ---------------------------------------------------------------- buildings copied from the pack

    /// <summary>
    /// A building in a ScaledCity patch prefab, checked to be one the swarm avoids: a collider on <c>Obstacle</c>. Null,
    /// with <paramref name="error"/> set, if there is no such building.
    /// </summary>
    private static Transform LoadBuilding(string patchName, string buildingName, out string error)
    {
        string path = $"{PatchFolder}/{patchName}.prefab";
        GameObject patch = AssetDatabase.LoadAssetAtPath<GameObject>(path);
        Transform source = patch == null
            ? null
            : patch.GetComponentsInChildren<Transform>(true).FirstOrDefault(t => t.name == buildingName);
        if (source == null)
        {
            error = patch == null ? $"no patch prefab at {path}." : $"{patchName} has no {buildingName}.";
            return null;
        }
        if (source.gameObject.layer != LayerMask.NameToLayer("Obstacle") || source.GetComponent<Collider>() == null)
        {
            error = $"{buildingName} in {patchName} is not a collider on Obstacle, so the swarm would not avoid it.";
            return null;
        }
        // The ScaledCity fork hides a few buildings; a copy of one would be neither seen nor avoided.
        for (Transform t = source; t != null; t = t.parent)
        {
            if (!t.gameObject.activeSelf)
            {
                error = $"{buildingName} in {patchName} is hidden there (inactive), so a copy would be neither seen nor avoided.";
                return null;
            }
        }
        error = null;
        return source;
    }

    /// <summary>A ScaledCity patch prefab's footpath, whose paving a generated slab copies.</summary>
    private static Renderer LoadFootpath(string patchName, out string error)
    {
        string path = $"{PatchFolder}/{patchName}.prefab";
        GameObject patch = AssetDatabase.LoadAssetAtPath<GameObject>(path);
        Renderer footpath = patch == null
            ? null
            : patch.GetComponentsInChildren<Renderer>(true)
                   .FirstOrDefault(r => HasAnyPrefix(r.name, FootpathPrefixes) && r.GetComponent<MeshFilter>());
        error = footpath != null ? null : patch == null ? $"no patch prefab at {path}." : $"{patchName} has no footpath.";
        return footpath;
    }

    /// <summary>
    /// A copy of a ScaledCity building (not a prefab instance, so it keeps the UV-baked mesh <see cref="BuildingUvBaker"/>
    /// made for its scale, its <c>BoxCollider</c> on <c>Obstacle</c>, and a name in one of the building families
    /// <see cref="ObstacleLayerAuditor"/> keeps on that layer), under <paramref name="parent"/>. Its long side runs along x
    /// (<paramref name="alongX"/>) or z, and it stands as high above <paramref name="groundY"/> as it stood above its own
    /// patch's ground. <paramref name="footprint"/> is where it stands, in <paramref name="root"/>'s frame; move it with
    /// <see cref="MoveFootprint"/>. Null, with <paramref name="error"/> set, if its base does not end up on the ground.
    /// </summary>
    private static GameObject CopyBuilding(Transform source, Transform parent, Transform root, bool alongX, float groundY,
                                           string label, out Rect footprint, out string error)
    {
        GameObject building = Object.Instantiate(source.gameObject, parent);
        building.name = $"{source.name} ({label})"; // keeps its building-family prefix
        Transform b = building.transform;
        b.rotation = root.rotation * source.rotation;
        b.localScale = source.lossyScale; // its patch's hierarchy, and ours, are unscaled above it
        // Heights are measured from the patch's road plate, since seven patches bake their content below the pivot.
        float pivotY = source.position.y - PatchGround(source.root);
        b.position = root.TransformPoint(new Vector3(0f, groundY + pivotY, 0f));
        footprint = Footprint(RendererBounds(building), root);
        if ((footprint.width >= footprint.height) != alongX)
        {
            b.rotation = Quaternion.AngleAxis(90f, root.up) * b.rotation;
            footprint = Footprint(RendererBounds(building), root);
        }
        foreach (Transform part in building.GetComponentsInChildren<Transform>(true))
        {
            GameObjectUtility.SetStaticEditorFlags(part.gameObject, Everything);
        }
        float baseY = root.InverseTransformPoint(RendererBounds(building).min).y - groundY;
        if (Mathf.Abs(baseY) > MaxBaseHeight)
        {
            error = $"{source.name} would stand {baseY:F1} units off the ground.";
            return null;
        }
        error = null;
        return building;
    }

    /// <summary>Moves a building or prop so its footprint is centred on <paramref name="centre"/>; returns the new footprint.</summary>
    private static Rect MoveFootprint(GameObject building, Transform root, Rect footprint, Vector2 centre)
    {
        Vector3 shift = new Vector3(centre.x - footprint.center.x, 0f, centre.y - footprint.center.y);
        building.transform.position += root.TransformVector(shift);
        footprint.center = centre;
        return footprint;
    }

    // ---------------------------------------------------------------- plaza furniture

    private static PropKit LoadProps(out string error)
    {
        string path = $"{PatchFolder}/{PropPatch}.prefab";
        GameObject patch = AssetDatabase.LoadAssetAtPath<GameObject>(path);
        if (patch == null)
        {
            error = $"no patch prefab at {path}.";
            return null;
        }
        Transform[] all = patch.GetComponentsInChildren<Transform>(true);
        PropKit kit = new PropKit
        {
            Lamp = LoadProp(all, LampProp, out error),
            ParkLamp = error == null ? LoadProp(all, ParkLampProp, out error) : null,
            Bench = error == null ? LoadProp(all, BenchProp, out error) : null,
            Bin = error == null ? LoadProp(all, BinProp, out error) : null,
            Bed = error == null ? LoadProp(all, BedProp, out error) : null,
            Hedge = error == null ? LoadProp(all, HedgeProp, out error) : null,
        };
        List<Transform> planters = new List<Transform>();
        foreach (string name in PlanterProps)
        {
            if (error == null)
            {
                planters.Add(LoadProp(all, name, out error));
            }
        }
        kit.Planters = planters.ToArray();
        if (error == null)
        {
            kit.Paving = LoadFootpath(PlazaFootpathSource, out error);
        }
        return error == null ? kit : null;
    }

    /// <summary>
    /// A prop in <see cref="PropPatch"/>: shown there, and not on <c>Obstacle</c>, where the swarm would take it for a
    /// building and give it a cylinder the size of its bounds.
    /// </summary>
    private static Transform LoadProp(Transform[] patch, string name, out string error)
    {
        Transform source = patch.FirstOrDefault(t => t.name == name);
        error = source == null ? $"{PropPatch} has no {name}."
              : source.GetComponentInChildren<Renderer>(true) == null ? $"{name} in {PropPatch} has no renderer."
              : source.GetComponentsInChildren<Transform>(true).Any(t => t.gameObject.layer == LayerMask.NameToLayer("Obstacle"))
                  ? $"{name} in {PropPatch} is on Obstacle, so the swarm would steer round it as a building."
              : null;
        for (Transform t = source; error == null && t != null; t = t.parent)
        {
            if (!t.gameObject.activeSelf)
            {
                error = $"{name} in {PropPatch} is hidden there (inactive).";
            }
        }
        return error == null ? source : null;
    }

    /// <summary>
    /// Furnishes a plaza the way the pack furnishes its park block, as a paved square in a park: a line of park lamps just
    /// inside each edge, and inside those benches looking out over the lawn, a bin beside each, and grass beds planted
    /// with a hedge; a group of planters in each corner; and a planter off each corner of every building. Rebuilt under
    /// the plaza's <see cref="PropsName"/> on every run from where its buildings stand, so a redrafted screen is furnished
    /// to match. A spot nearer a building than <see cref="PropBuildingClearance"/>, on another prop, or off the paving is
    /// skipped and counted, never nudged. Returns what it placed, or null with <paramref name="error"/> set.
    /// </summary>
    private static string FurnishPlaza(Transform plaza, Transform root, PropKit kit, out string error)
    {
        Transform old = plaza.Find(PropsName);
        if (old != null)
        {
            Object.DestroyImmediate(old.gameObject);
        }
        Transform slab = plaza.Find(PlazaFootpathName);
        if (slab == null || slab.GetComponent<Renderer>() == null)
        {
            error = $"no {PlazaFootpathName} to furnish.";
            return null;
        }
        Bounds slabBounds = slab.GetComponent<Renderer>().bounds;
        Rect paving = Footprint(slabBounds, root);
        int obstacle = LayerMask.NameToLayer("Obstacle");
        List<Rect> buildings = plaza.GetComponentsInChildren<Collider>(true)
                                    .Where(c => c.gameObject.layer == obstacle)
                                    .Select(c => Footprint(RendererBounds(c.gameObject), root))
                                    .ToList();
        Furnisher furnisher = new Furnisher(CreateChild(plaza, PropsName), root, root.InverseTransformPoint(slabBounds.max).y,
            r => r.xMin >= paving.xMin + PropEdgeMargin && r.xMax <= paving.xMax - PropEdgeMargin &&
                 r.yMin >= paving.yMin + PropEdgeMargin && r.yMax <= paving.yMax - PropEdgeMargin &&
                 buildings.All(b => Separation(r, b) >= PropBuildingClearance));
        bool Place(Transform source, string kind, Vector2 at, PropTurn turn, Vector2 direction, bool free = false) =>
            furnisher.Place(source, kind, at, turn, direction, free);
        int planters = 0;

        Vector2 centre = paving.center;
        float half = paving.width / 2f;
        Vector2[] inwards = { Vector2.down, Vector2.left, Vector2.up, Vector2.right }; // the north, east, south and west edges
        foreach (Vector2 inward in inwards)
        {
            Vector2 middle = centre - inward * half;
            Vector2 along = new Vector2(-inward.y, inward.x);
            Vector2 At(float s, float inset) => middle + along * s + inward * inset;

            foreach (float s in LampStations)
            {
                Place(kit.Lamp, "Lamp", At(s, LampInset), PropTurn.LongAlong, inward);
            }
            foreach (float s in BedStations)
            {
                if (Place(kit.Bed, "Bed", At(s, FurnitureInset), PropTurn.LongAlong, along))
                {
                    Place(kit.Hedge, "Hedge", At(s, FurnitureInset), PropTurn.LongAlong, along, free: true);
                }
            }
            foreach (float s in BenchStations)
            {
                if (Place(kit.Bench, "Bench", At(s, FurnitureInset), PropTurn.FrontTowards, -inward))
                {
                    Place(kit.Bin, "Bin", At(s - Mathf.Sign(s) * BinBesideBench, FurnitureInset), PropTurn.AsIs, Vector2.zero);
                }
            }
        }

        // Planters: a group in each corner, then one off each corner of every building, cycling through the models.
        List<Vector2> planterSpots = new List<Vector2>();
        foreach (Vector2 corner in new[] { new Vector2(1f, 1f), new Vector2(1f, -1f), new Vector2(-1f, -1f), new Vector2(-1f, 1f) })
        {
            Vector2 c = centre + corner * half;
            planterSpots.Add(c - corner * CornerPlanterInset);
            planterSpots.Add(c - corner * CornerPlanterInset - new Vector2(corner.x, 0f) * CornerPlanterInset);
            planterSpots.Add(c - corner * CornerPlanterInset - new Vector2(0f, corner.y) * CornerPlanterInset);
        }
        foreach (Rect b in buildings)
        {
            float o = BuildingPlanterOffset;
            planterSpots.Add(new Vector2(b.xMin - o, b.yMin - o));
            planterSpots.Add(new Vector2(b.xMax + o, b.yMin - o));
            planterSpots.Add(new Vector2(b.xMax + o, b.yMax + o));
            planterSpots.Add(new Vector2(b.xMin - o, b.yMax + o));
        }
        foreach (Vector2 spot in planterSpots)
        {
            if (Place(kit.Planters[planters % kit.Planters.Length], "Planter", spot, PropTurn.AsIs, Vector2.zero))
            {
                planters++;
            }
        }

        error = null;
        return $"{plaza.name} {furnisher.Summary()}";
    }

    /// <summary>
    /// Copies props into place one spot at a time, standing on <see cref="SurfaceY"/>. A spot its test refuses, or
    /// within <see cref="PropSpacing"/> of a prop it has already placed, is skipped and counted, never nudged.
    /// </summary>
    private sealed class Furnisher
    {
        private readonly Transform parent;
        private readonly Transform root;
        private readonly System.Func<Rect, bool> fits;
        private readonly List<Rect> placed = new List<Rect>();
        private readonly List<string> kinds = new List<string>();
        private readonly Dictionary<string, int> counts = new Dictionary<string, int>();
        private int skipped;

        /// <summary>The height, in the root's frame, that the props placed next stand on.</summary>
        public float SurfaceY;

        public Furnisher(Transform parent, Transform root, float surfaceY, System.Func<Rect, bool> fits)
        {
            this.parent = parent;
            this.root = root;
            SurfaceY = surfaceY;
            this.fits = fits;
        }

        /// <summary>Places a prop, or skips it; whether it stood. A <paramref name="free"/> one is not tested.</summary>
        public bool Place(Transform source, string kind, Vector2 at, PropTurn turn, Vector2 direction, bool free = false)
        {
            GameObject prop = CopyProp(source, parent, root, SurfaceY, turn, direction);
            Rect r = MoveFootprint(prop, root, Footprint(RendererBounds(prop), root), at);
            if (!free && (!fits(r) || placed.Any(p => Separation(r, p) < PropSpacing)))
            {
                Object.DestroyImmediate(prop);
                skipped++;
                return false;
            }
            if (!counts.TryGetValue(kind, out int n))
            {
                kinds.Add(kind);
            }
            prop.name = $"{kind}_{n:D2}";
            counts[kind] = n + 1;
            placed.Add(r);
            return true;
        }

        public string Summary()
        {
            return string.Join(", ", kinds.Select(k => $"{counts[k]} {k.ToLowerInvariant()}{(k.EndsWith("ch") ? "es" : "s")}")) +
                   (skipped > 0 ? $" ({skipped} spot(s) skipped)" : "");
        }
    }

    /// <summary>
    /// A copy of a prop, standing on <paramref name="surfaceY"/> in <paramref name="root"/>'s frame, turned as
    /// <paramref name="turn"/> says. Where it stands is left to <see cref="MoveFootprint"/>.
    /// </summary>
    private static GameObject CopyProp(Transform source, Transform parent, Transform root, float surfaceY, PropTurn turn,
                                       Vector2 direction)
    {
        GameObject prop = Object.Instantiate(source.gameObject, parent);
        Transform t = prop.transform;
        t.rotation = root.rotation * source.rotation;
        t.localScale = source.lossyScale;
        float yaw = 0f;
        if (turn == PropTurn.LongAlong)
        {
            Rect r = Footprint(RendererBounds(prop), root);
            yaw = (r.width >= r.height) == (Mathf.Abs(direction.x) >= Mathf.Abs(direction.y)) ? 0f : 90f;
        }
        else if (turn == PropTurn.FrontTowards)
        {
            // The pack's models are square to their own axes, so the measured front is snapped to one of them first. A
            // turn about up is clockwise seen from above; SignedAngle in (x, z) counts the other way.
            Vector2 front = Front(prop, root);
            front = Mathf.Abs(front.x) >= Mathf.Abs(front.y) ? new Vector2(Mathf.Sign(front.x), 0f) : new Vector2(0f, Mathf.Sign(front.y));
            yaw = -Vector2.SignedAngle(front, direction);
        }
        t.rotation = Quaternion.AngleAxis(yaw, root.up) * t.rotation;
        // Its lowest point on the surface, wherever its pivot is: the pack stands its props on a footpath 0.15 up.
        t.position = root.TransformPoint(new Vector3(0f, surfaceY, 0f));
        t.position += root.up * (surfaceY - root.InverseTransformPoint(RendererBounds(prop).min).y);
        foreach (Transform part in prop.GetComponentsInChildren<Transform>(true))
        {
            GameObjectUtility.SetStaticEditorFlags(part.gameObject, Everything);
        }
        return prop;
    }

    /// <summary>
    /// Which way a bench faces, in <paramref name="root"/>'s frame: away from its backrest, the only part of it standing
    /// above the seat. Read off the mesh, since the pack's models are authored in more than one orientation.
    /// </summary>
    private static Vector2 Front(GameObject prop, Transform root)
    {
        Bounds bounds = RendererBounds(prop);
        float cut = bounds.min.y + 0.7f * bounds.size.y;
        Vector2 sum = Vector2.zero;
        int n = 0;
        foreach (MeshFilter filter in prop.GetComponentsInChildren<MeshFilter>())
        {
            foreach (Vector3 v in filter.sharedMesh.vertices)
            {
                Vector3 w = filter.transform.TransformPoint(v);
                if (w.y > cut)
                {
                    Vector3 local = root.InverseTransformPoint(w);
                    sum += new Vector2(local.x, local.z);
                    n++;
                }
            }
        }
        return n == 0 ? Vector2.up : (Footprint(bounds, root).center - sum / n).normalized;
    }

    /// <summary>How far apart two footprints are: zero when they touch or overlap.</summary>
    private static float Separation(Rect a, Rect b)
    {
        float dx = Mathf.Max(0f, a.xMin - b.xMax, b.xMin - a.xMax);
        float dz = Mathf.Max(0f, a.yMin - b.yMax, b.yMin - a.yMax);
        return Mathf.Sqrt(dx * dx + dz * dz);
    }

    /// <summary>
    /// Furnishes the park the way a formal park is. Round the fountain, inside its ring of trees, a round of the plazas'
    /// paving, with benches facing the water in pairs either side of each diagonal, a bin between each pair and a lamp
    /// behind it, and planters flanking the four ways in along the axes. Down each avenue, pairs of lamps and pairs of
    /// benches facing the walk, in alternate gaps between its trees; under the inner line of each double row, a bench
    /// with a bin in every other gap, facing into the park; and along the front of each grove, a bench between each two
    /// of its trees, facing the avenue. It is laid out by the trees' arrangements and placed after them, so it changes
    /// nothing about the planting. A spot off the lawn, within <see cref="PropTrunkClearance"/> of a trunk or
    /// <see cref="PropBuildingClearance"/> of anything else standing on the lawn, or on another prop, is skipped and
    /// counted. Returns what it placed, or null with <paramref name="error"/> set.
    /// </summary>
    private static string FurnishPark(Transform park, Transform root, PropKit kit, Vector2 centre, float groundY,
                                      List<Rect> lawn, List<Rect> keepOuts, List<TreeGroup> groups, List<Vector2> trunks,
                                      out string error)
    {
        Transform parent = CreateChild(park, PropsName);
        Furnisher furnisher = new Furnisher(parent, root, groundY,
            r => RectOnLawn(r, lawn, LawnPropMargin) &&
                 keepOuts.All(k => Separation(r, k) >= PropBuildingClearance) &&
                 trunks.All(t => Separation(r, new Rect(t, Vector2.zero)) >= PropTrunkClearance));
        int planters = 0;

        // The fountain's paving: the plazas' paving at its own texel density, and ground rather than a prop.
        Mesh disc = FootpathMesh("Diamond_Fountain_Paving", Circle(FountainPavingRadius, FountainPavingSides),
                                 FitUv(kit.Paving, out _));
        if (disc == null)
        {
            error = "could not write the fountain paving mesh.";
            return null;
        }
        GameObject slab = new GameObject("FountainPaving");
        SceneManager.MoveGameObjectToScene(slab, root.gameObject.scene);
        slab.transform.SetParent(parent, false);
        slab.transform.localPosition = new Vector3(centre.x, groundY + FountainPavingLift, centre.y);
        slab.AddComponent<MeshFilter>().sharedMesh = disc;
        slab.AddComponent<MeshRenderer>().sharedMaterials = kit.Paving.sharedMaterials;
        GameObjectUtility.SetStaticEditorFlags(slab, Everything);

        furnisher.SurfaceY = groundY + FountainPavingLift;
        Vector2 Around(float degrees, float radius) =>
            centre + radius * new Vector2(Mathf.Cos(degrees * Mathf.Deg2Rad), Mathf.Sin(degrees * Mathf.Deg2Rad));
        foreach (float diagonal in new[] { 45f, 135f, 225f, 315f })
        {
            foreach (float spread in new[] { -FountainBenchSpread, FountainBenchSpread })
            {
                Vector2 at = Around(diagonal + spread, FountainBenchRadius);
                furnisher.Place(kit.Bench, "Bench", at, PropTurn.FrontTowards, (centre - at).normalized);
            }
            furnisher.Place(kit.Bin, "Bin", Around(diagonal, FountainBenchRadius), PropTurn.AsIs, Vector2.zero);
            furnisher.Place(kit.ParkLamp, "Lamp", Around(diagonal, FountainLampRadius), PropTurn.AsIs, Vector2.zero);
        }
        foreach (float axis in new[] { 0f, 90f, 180f, 270f })
        {
            foreach (float spread in new[] { -FountainPlanterSpread, FountainPlanterSpread })
            {
                if (furnisher.Place(kit.Planters[planters % kit.Planters.Length], "Planter",
                                    Around(axis + spread, FountainPlanterRadius), PropTurn.AsIs, Vector2.zero))
                {
                    planters++;
                }
            }
        }

        // The arms, on the lawn, arrangement by arrangement, in each arm's frame: along its axis, and across it.
        furnisher.SurfaceY = groundY;
        foreach (TreeGroup group in groups.Where(g => g.Kind != Arrangement.Ring))
        {
            Vector2 axis = Axis(group.Arm);
            Vector2 side = new Vector2(-axis.y, axis.x);
            Vector2 At(float along, float across) => centre + axis * along + side * across;
            List<Vector2> spots = group.Spots
                                       .Select(p => new Vector2(Vector2.Dot(p - centre, axis), Vector2.Dot(p - centre, side)))
                                       .ToList();
            if (group.Kind == Arrangement.Avenue)
            {
                List<float> stations = Stations(spots.Select(s => s.x));
                for (int i = 0; i + 1 < stations.Count; i++)
                {
                    float along = (stations[i] + stations[i + 1]) / 2f;
                    foreach (float sign in new[] { 1f, -1f })
                    {
                        if (i % 2 == 0)
                        {
                            furnisher.Place(kit.ParkLamp, "Lamp", At(along, sign * AvenueLampOffset), PropTurn.AsIs, Vector2.zero);
                        }
                        else if (furnisher.Place(kit.Bench, "Bench", At(along, sign * AvenueBenchOffset), PropTurn.FrontTowards,
                                                 -sign * side) && sign > 0f)
                        {
                            furnisher.Place(kit.Bin, "Bin", At(along + BinBesideBench, sign * AvenueBenchOffset), PropTurn.AsIs,
                                            Vector2.zero);
                        }
                    }
                }
                continue;
            }

            // A double row's inner line, or a grove's front row: the trees on each side nearest the axis.
            bool rows = group.Kind == Arrangement.EdgeRows;
            foreach (float sign in new[] { 1f, -1f })
            {
                List<Vector2> mine = spots.Where(s => s.y * sign > 0f).ToList();
                if (mine.Count < 2)
                {
                    continue;
                }
                float line = mine.Min(s => Mathf.Abs(s.y));
                List<float> stops = Stations(mine.Where(s => Mathf.Abs(Mathf.Abs(s.y) - line) < 0.5f).Select(s => s.x));
                float across = sign * (rows ? line - RowBenchOffset : line);
                for (int i = 0; i + 1 < stops.Count; i++)
                {
                    if (rows && i % 2 == 1)
                    {
                        continue;
                    }
                    float along = (stops[i] + stops[i + 1]) / 2f;
                    if (furnisher.Place(kit.Bench, "Bench", At(along, across), PropTurn.FrontTowards, -sign * side) &&
                        (rows || i == 0))
                    {
                        furnisher.Place(kit.Bin, "Bin", At(along - BinBesideBench, across), PropTurn.AsIs, Vector2.zero);
                    }
                }
            }
        }

        error = null;
        return $"park {furnisher.Summary()}";
    }

    /// <summary>Distinct positions along a line, in order; spots a hair apart are one.</summary>
    private static List<float> Stations(IEnumerable<float> values)
    {
        return values.Select(v => Mathf.Round(v * 100f) / 100f).Distinct().OrderBy(v => v).ToList();
    }

    /// <summary>Whether a footprint lies on the lawn, <paramref name="margin"/> in from its edge.</summary>
    private static bool RectOnLawn(Rect r, List<Rect> lawn, float margin)
    {
        return InUnion(new Vector2(r.xMin - margin, r.yMin - margin), lawn) &&
               InUnion(new Vector2(r.xMax + margin, r.yMin - margin), lawn) &&
               InUnion(new Vector2(r.xMax + margin, r.yMax + margin), lawn) &&
               InUnion(new Vector2(r.xMin - margin, r.yMax + margin), lawn);
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
    /// Rebuilds one diamond's park and its plazas' furniture, and saves it. Everything is measured in the prefab root's
    /// frame, which is the diamond's: its origin is the centre of the four blocks.
    /// </summary>
    private static string DressDiamond(DiamondSpec spec, List<GameObject> trees, List<TipSource> tips, PropKit props,
                                       out string error)
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

            int kerbsRenamed = 0;
            int nodesRenamed = 0;
            RenameTileMarkers(root, ref kerbsRenamed, ref nodesRenamed);

            // The plazas' furniture, laid out afresh round their buildings. It stays on the paving, so it changes
            // nothing the trees are measured against.
            Transform plazas = root.Find(PlazasName);
            List<string> furnished = new List<string>();
            foreach (DiamondPlaza plaza in plazas != null
                         ? plazas.GetComponentsInChildren<DiamondPlaza>(true).OrderBy(p => p.name, System.StringComparer.Ordinal)
                         : Enumerable.Empty<DiamondPlaza>())
            {
                string note = FurnishPlaza(plaza.transform, root, props, out error);
                if (note == null)
                {
                    error = $"{plaza.name}: {error}";
                    return null;
                }
                furnished.Add(note);
            }

            // The plazas batch with the rest of the city; a patch prefab's own objects are not marked for it.
            if (plazas != null)
            {
                foreach (Renderer r in plazas.GetComponentsInChildren<Renderer>(true))
                {
                    GameObjectUtility.SetStaticEditorFlags(r.gameObject,
                        GameObjectUtility.GetStaticEditorFlags(r.gameObject) | StaticEditorFlags.BatchingStatic);
                }
            }

            Renderer fountain = FindFountain(root);
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
            List<TreeGroup> groups = new List<TreeGroup>
            {
                new TreeGroup("FountainRing", spec.Ring, Ring(centre), Arrangement.Ring, Tip.North),
            };
            foreach (Tip tip in Tips)
            {
                bool northSouth = tip == Tip.North || tip == Tip.South;
                groups.Add(new TreeGroup($"Avenue_{tip}", northSouth ? spec.NorthSouthAvenues : spec.EastWestAvenues,
                                         Avenue(tip, centre, tipFootpaths[tip], keepOuts, out float plazaFarEdge),
                                         Arrangement.Avenue, tip));
                groups.Add(northSouth == spec.RowsNorthSouth
                    ? new TreeGroup($"EdgeRows_{tip}", spec.Rows, EdgeRows(tip, centre, lawn, plazaFarEdge), Arrangement.EdgeRows, tip)
                    : new TreeGroup($"Groves_{tip}", null, Groves(tip, centre, lawn, keepOuts, plazaFarEdge), Arrangement.Groves, tip));
            }

            System.Random rng = new System.Random(spec.Seed);
            int planted = 0;
            int offLawn = 0;
            int crowded = 0;
            List<Vector2> trunks = new List<Vector2>();
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
                    trunks.Add(spot);
                    perModel[tree.name]++;
                    here++;
                }
                planted += here;
                counts.Add($"{group.Name} {here}/{group.Spots.Count}");
            }

            // The park's furniture, last: it keeps clear of the trees, never the other way round.
            string parkFurniture = FurnishPark(park, root, props, centre, groundY, lawn, keepOuts, groups, trunks, out error);
            if (parkFurniture == null)
            {
                return null;
            }
            furnished.Insert(0, parkFurniture);

            PrefabUtility.SaveAsPrefabAsset(rootObject, spec.PrefabPath, out bool saved);
            if (!saved)
            {
                error = $"could not save {spec.PrefabPath}.";
                return null;
            }
            error = null;
            return $"Diamond_{spec.Letter}: planted {planted} trees ({string.Join(", ", counts)}; {offLawn} spot(s) off the " +
                   $"lawn, {crowded} too close to something; by model {string.Join(", ", perModel.Select(m => $"{m.Key} {m.Value}"))}) " +
                   $"round the fountain at ({centre.x:F1}, {centre.y:F1}); tips: {string.Join("; ", tipNotes)}; " +
                   $"furnished {string.Join("; ", furnished)}" +
                   (kerbsRenamed + nodesRenamed > 0
                       ? $"; {kerbsRenamed} kerb(s) renamed Kerb_, {nodesRenamed} MC_Patch node(s) renamed Block"
                       : "") + ".";
        }
        finally
        {
            PrefabUtility.UnloadPrefabContents(rootObject);
        }
    }

    // ---------------------------------------------------------------- tile markers

    /// <summary>
    /// Renames what the city tools recognise a tile by, as <see cref="DiamondCityBuilder"/> does for its drafts. A kerb
    /// keeps its geometry: round a plaza it is the block's kerb, as on any tile.
    /// </summary>
    private static void RenameTileMarkers(Transform root, ref int kerbsRenamed, ref int nodesRenamed)
    {
        foreach (Transform t in root.GetComponentsInChildren<Transform>(true))
        {
            if (t.name.StartsWith(CityTiles.KerbPrefix))
            {
                t.name = DiamondCityBuilder.DiamondKerbPrefix + t.name.Substring(CityTiles.KerbPrefix.Length);
                kerbsRenamed++;
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
        GameObject building = CopyBuilding(source.Building, group, root, alongX, 0f, $"tip {tip}", out Rect footprint, out error);
        if (building == null)
        {
            note = null;
            error = $"the {tip} tip: {error}";
            return default;
        }
        // The slab keeps the pack footpath's height above its patch's road plate, as the building keeps its own.
        float footpathY = source.Footpath.bounds.center.y - PatchGround(source.Building.root);

        // The footpath: long side along the edge, outer edge TipInset in from it.
        Vector2 size = footprint.size + 2f * FootpathMargin * Vector2.one;
        float depth = alongX ? size.y : size.x;
        Vector2 middle = centre + axis * (reach - TipInset - depth / 2f);
        Rect footpath = new Rect(middle - size / 2f, size);

        // Centre the building on it.
        footprint = MoveFootprint(building, root, footprint, middle);

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

    /// <summary>A flat rectangle of <paramref name="size"/> about its origin; see the outline overload.</summary>
    private static Mesh FootpathMesh(string name, Vector2 size, UvMap uv)
    {
        float hx = size.x / 2f;
        float hz = size.y / 2f;
        return FootpathMesh(name, new[] { new Vector2(-hx, -hz), new Vector2(-hx, hz), new Vector2(hx, hz), new Vector2(hx, -hz) }, uv);
    }

    /// <summary>
    /// A flat convex polygon about its origin, facing up, textured by <paramref name="uv"/>, saved as its own asset in
    /// <see cref="GeneratedFolder"/>. <paramref name="outline"/> is (x, z), clockwise seen from above, which makes a fan
    /// from its first corner face up. An existing asset is rewritten in place, so its GUID and every reference to it
    /// survive a re-run.
    /// </summary>
    private static Mesh FootpathMesh(string name, Vector2[] outline, UvMap uv)
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
        mesh.vertices = outline.Select(p => new Vector3(p.x, 0f, p.y)).ToArray();
        mesh.normals = Enumerable.Repeat(Vector3.up, outline.Length).ToArray();
        mesh.uv = outline.Select(p => uv.At(p.x, p.y)).ToArray();
        mesh.triangles = Enumerable.Range(1, outline.Length - 2).SelectMany(i => new[] { 0, i, i + 1 }).ToArray();
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

    /// <summary>A regular polygon round the origin, clockwise seen from above, as <see cref="FootpathMesh(string, Vector2[], UvMap)"/> wants.</summary>
    private static Vector2[] Circle(float radius, int sides)
    {
        return Enumerable.Range(0, sides)
                         .Select(k => -k * 2f * Mathf.PI / sides)
                         .Select(a => radius * new Vector2(Mathf.Cos(a), Mathf.Sin(a)))
                         .ToArray();
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
    /// lawn between the arm's end and whatever stands nearer the fountain in the grove's own strip of it — a plaza
    /// reaching in from beside the arm, or at least the plaza or fountain across the avenue — and between the avenue and
    /// the arm's edge.
    /// </summary>
    private static List<Vector2> Groves(Tip tip, Vector2 centre, List<Rect> lawn, List<Rect> keepOuts, float plazaFarEdge)
    {
        Vector2 axis = Axis(tip);
        Vector2 side = new Vector2(-axis.y, axis.x);
        float reach = Reach(lawn, centre, axis);

        List<Vector2> spots = new List<Vector2>();
        foreach (float sign in new[] { 1f, -1f })
        {
            // The strip the grove stands in, measured across the arm where it is widest; then how far out along the
            // axis anything in that strip reaches.
            float edge = Reach(lawn, centre + axis * (reach - GroveInset), side * sign);
            float stripNear = AvenueHalfWidth + GroveGapToAvenue;
            float stripFar = edge - GroveInset;
            float start = plazaFarEdge;
            foreach (Rect r in keepOuts)
            {
                Vector2 alongR = Extent(r, centre, axis);
                Vector2 acrossR = Extent(r, centre, side * sign);
                if (alongR.y > 0f && acrossR.x < stripFar && acrossR.y > stripNear && alongR.x < reach)
                {
                    start = Mathf.Max(start, alongR.y);
                }
            }
            float along = ((start + GroveInset) + (reach - GroveInset)) / 2f;
            float across = (stripNear + stripFar) / 2f;
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
