using System.Collections.Generic;
using System.IO;
using System.Linq;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;
using UnityEngine.SceneManagement;

/// <summary>
/// <c>Tools/Swarm/Build Voronoi city world</c>: makes <c>VoronoiCityWorld.unity</c>, DiamondCityWorld's own buildings on
/// the density study's Voronoi streets. The layout is <c>citygen mixcity</c>'s
/// (<c>layouts/VoronoiCity__&lt;id&gt;.json</c>): every building drawn from DiamondCityWorld's 37 designs in the shares
/// they stand there, at the ScaledCity scale, spread evenly over the blocks, to the city's own coverage over its tile
/// grid by default, with its tile pitch as the block spacing -- the same city in everything but its streets, which no
/// longer run straight for kilometres. The goals are the study's (an occluder ringed by patrolled buildings), built from
/// the same mix so a ring of identical buildings does not give one away. The streets are road
/// (<see cref="ProceduralCityBuilder.StreetStyle.Textured"/>: ParkRoad's asphalt, plain junctions, zebra crossings).
///
/// <para>The scene itself is <see cref="ProceduralCityWorldBuilder"/>'s: DiamondCityWorld with its city taken out and a
/// <see cref="ProceduralCityBuilder"/>, here with the city BAKED in (<see cref="ProceduralCityBuilder.Bake"/>), so it is
/// in the editor without Play; Play adopts the bake and makes only the goals. The recorder, the bench, analyse.py and the
/// replica read it as any generated layout (by the id in the session). Tune the density and see it with
/// <see cref="VoronoiCityWindow"/> (<c>Tools/Swarm/Voronoi city…</c>), which generates a layout and rebakes the open
/// scene. Rebuild the scene after changing DiamondCityWorld, and regenerate after restyling its buildings and
/// re-exporting its city obstacles. Batch: <c>-executeMethod VoronoiCityWorldBuilder.Run</c> (CITY_LAYOUT overrides
/// the layout).</para>
/// </summary>
public static class VoronoiCityWorldBuilder
{
    public const string ScenePath = "Assets/Scenes/VoronoiCityWorld.unity";
    /// <summary>citygen mixcity's file names (citygen.MIXCITY_PREFIX).</summary>
    public const string LayoutPattern = "VoronoiCity__*.json";
    public const string RoadMaterialPath = "Assets/Modular City Pack/Materials/MC_Asphalt_Road Texture_M_01.mat";
    public const string JunctionMaterialPath = "Assets/Modular City Pack/Materials/MC_Asphalt_Cross and Corner.mat";
    public const string VoronoiRoadMaterialPath = "Assets/Materials/VoronoiRoad.mat";
    public const string VoronoiJunctionMaterialPath = "Assets/Materials/VoronoiJunction.mat";
    private const float AsphaltTint = 0.5f;

    [MenuItem("Tools/Swarm/Build Voronoi city world")]
    private static void BuildFromMenu()
    {
        if (!EditorSceneManager.SaveCurrentModifiedScenesIfUserWantsTo()) return;
        if (File.Exists(ScenePath) &&
            !EditorUtility.DisplayDialog("Build Voronoi city world",
                $"Rebuild {ScenePath} from DiamondCityWorld? This replaces the scene, including hand edits.",
                "Rebuild", "Cancel"))
        {
            return;
        }
        string layout = ProceduralCityWorldBuilder.DefaultLayout(LayoutPattern);
        if (!ProceduralCityWorldBuilder.Build(ScenePath, layout, true, out string summary, Configure))
        {
            Debug.LogError("VoronoiCityWorldBuilder: " + summary + NoLayoutHint(layout));
        }
    }

    [MenuItem("Tools/Swarm/Build Voronoi city world", true)]
    private static bool CanBuild() => !EditorApplication.isPlayingOrWillChangePlaymode;

    /// <summary>Batchmode: builds the scene (CITY_LAYOUT, else the first mixcity layout) and exits 0 / 1.</summary>
    public static void Run()
    {
        string layout = System.Environment.GetEnvironmentVariable(ProceduralCityBuilder.LayoutEnv);
        if (string.IsNullOrEmpty(layout)) layout = ProceduralCityWorldBuilder.DefaultLayout(LayoutPattern);
        bool ok = ProceduralCityWorldBuilder.Build(ScenePath, layout, true, out string summary, Configure);
        if (!ok) Debug.LogError("VoronoiCityWorldBuilder: " + summary + NoLayoutHint(layout));
        EditorApplication.Exit(ok ? 0 : 1);
    }

    /// <summary>Batchmode: bakes CITY_LAYOUT (else the first mixcity layout) into the scene as it stands, without
    /// rebuilding it from DiamondCityWorld, so the scene's own settings are kept -- the Voronoi city window's Bake,
    /// from the command line (<c>-executeMethod VoronoiCityWorldBuilder.Rebake</c>). Saves and exits 0 / 1.</summary>
    public static void Rebake()
    {
        string layout = System.Environment.GetEnvironmentVariable(ProceduralCityBuilder.LayoutEnv);
        if (string.IsNullOrEmpty(layout)) layout = ProceduralCityWorldBuilder.DefaultLayout(LayoutPattern);
        bool ok = false;
        string error = "no layout." + NoLayoutHint(layout);
        if (!string.IsNullOrEmpty(layout))
        {
            Scene scene = EditorSceneManager.OpenScene(ScenePath, OpenSceneMode.Single);
            var builder = Object.FindFirstObjectByType<ProceduralCityBuilder>(FindObjectsInactive.Include);
            if (builder == null)
            {
                error = $"{ScenePath} has no ProceduralCityBuilder.";
            }
            else
            {
                builder.layoutPath = layout;
                EnsureDressing(builder);
                ok = builder.Bake(out error);
                if (ok && !EditorSceneManager.SaveScene(scene))
                {
                    ok = false;
                    error = $"could not save {ScenePath}.";
                }
            }
        }
        if (ok) Debug.Log($"VoronoiCityWorldBuilder: baked {layout} into {ScenePath}.");
        else Debug.LogError("VoronoiCityWorldBuilder: " + error);
        EditorApplication.Exit(ok ? 0 : 1);
    }

    /// <summary>Road streets: ParkRoad's lane-marked asphalt for the stretches, the pack's crossing texture for the
    /// junctions and crossings, both through this scene's own lighter copies (<see cref="Tinted"/>), which ParkRoad is
    /// moved onto too so the spawn road matches. Without the pack (it is git-ignored) the streets stay flat. The
    /// landmarks round the city (<see cref="LandmarkBuilder"/>) are added first, so a rebuild keeps them.</summary>
    private static void Configure(ProceduralCityBuilder builder)
    {
        if (!LandmarkBuilder.AddTo(builder.gameObject.scene, out string landmarks))
        {
            Debug.LogWarning("VoronoiCityWorldBuilder: no landmarks: " + landmarks);
        }
        EnsureDressing(builder);
        var packRoad = AssetDatabase.LoadAssetAtPath<Material>(RoadMaterialPath);
        Material road = Tinted(VoronoiRoadMaterialPath, RoadMaterialPath);
        Material junction = Tinted(VoronoiJunctionMaterialPath, JunctionMaterialPath);
        if (road == null || junction == null)
        {
            Debug.LogWarning($"VoronoiCityWorldBuilder: no {RoadMaterialPath} / {JunctionMaterialPath} (the Modular City " +
                             "Pack); the streets stay flat.");
            return;
        }
        builder.streetStyle = ProceduralCityBuilder.StreetStyle.Textured;
        builder.streetMaterial = road;
        builder.junctionMaterial = junction;
        foreach (GameObject root in builder.gameObject.scene.GetRootGameObjects())
        {
            if (root.name != "ParkRoad") continue;
            foreach (MeshRenderer r in root.GetComponentsInChildren<MeshRenderer>(true))
            {
                Material[] mats = r.sharedMaterials;
                for (int i = 0; i < mats.Length; i++)
                {
                    if (mats[i] == packRoad) mats[i] = road;
                }
                r.sharedMaterials = mats;
            }
        }
    }

    /// <summary>
    /// Gives the builder its street dressing (<see cref="CityDressing"/>: footpaths, medians with trees, street furniture),
    /// adding the component if missing and filling only what is unassigned, so a hand choice in its inspector survives:
    /// the pack's props (<see cref="PropFolder"/>), the road light that lines the pack's streets (out of
    /// <see cref="StreetLightPatch"/>, which has no prefab of its own), the pack footpath at its own texture scale
    /// (<see cref="DiamondParkBuilder.FootpathRepeatMetres"/>), and the diamond parks' grass for the medians. Every
    /// bake and build then dresses the city it makes. Without the pack (git-ignored) the props stay unassigned and are
    /// skipped.
    /// </summary>
    public static CityDressing EnsureDressing(ProceduralCityBuilder builder)
    {
        CityDressing d = builder.GetComponent<CityDressing>();
        if (d == null) d = Undo.AddComponent<CityDressing>(builder.gameObject);
        Undo.RecordObject(d, "Voronoi city dressing");
        if (!d.RulesCurrent)
        {
            d.ResetRules();
            Debug.Log($"VoronoiCityWorldBuilder: the dressing's rules were older than version {CityDressing.RulesVersion}; " +
                      "reset to the current defaults.", d);
        }
        GameObject Prop(string name) => AssetDatabase.LoadAssetAtPath<GameObject>($"{PropFolder}/{name}.prefab");
        if (d.tree == null) d.tree = Prop("MC_Tree");
        if (d.hydrant == null) d.hydrant = Prop("MC_Hydrant");
        if (d.bench == null) d.bench = Prop("MC_Bench");
        if (d.bin == null) d.bin = Prop("MC_Trash_Can");
        if (d.busStop == null) d.busStop = Prop("MC_Bus_Stop");
        if (d.cornerProps == null || d.cornerProps.Length == 0 || d.cornerProps.Any(p => p == null))
        {
            d.cornerProps = new[] { "MC_Mail_Box", "MC_Newspaper_Machine", "MC_Telephone_Booth" }
                            .Select(Prop).Where(p => p != null).ToArray();
        }
        // The extras' props (placed only where CityDressing.Extras turns them on: BlockShapedCityWorld).
        if (d.trafficSignal == null) d.trafficSignal = Prop("MC_Traffic_Signal");
        if (d.frontageBench == null) d.frontageBench = Prop("MC_Bench_Single");
        if (d.bollard == null) d.bollard = Prop("MC_Street_Lamp_Small");
        if (d.planters == null || d.planters.Length == 0 || d.planters.Any(p => p == null))
        {
            d.planters = new[] { "MC_Plant", "MC_Plant2", "MC_Plant3" }.Select(Prop).Where(p => p != null).ToArray();
        }
        if (d.kerbProps == null || d.kerbProps.Length == 0 || d.kerbProps.Any(p => p == null))
        {
            d.kerbProps = new[] { "MC_Speed_Limit_Sign", "MC_Taxi_Pole" }.Select(Prop).Where(p => p != null).ToArray();
        }
        if (d.streetLight == null) d.streetLight = StreetLight();
        if (d.pavementMaterial == null)
        {
            float metres = DiamondParkBuilder.FootpathRepeatMetres(out Material footpath, out string error);
            if (footpath != null && metres > 0f)
            {
                d.pavementMaterial = footpath;
                d.pavementUvMetres = metres;
            }
            else
            {
                Debug.LogWarning($"VoronoiCityWorldBuilder: no pack footpath to pave with ({error}).");
            }
        }
        if (d.grassMaterial == null) d.grassMaterial = AssetDatabase.LoadAssetAtPath<Material>(GrassMaterialPath);
        if (d.hedge == null) d.hedge = Prop("MC_Bush");
        if (d.parkLamp == null) d.parkLamp = Prop("MC_Street_Lamp");
        if (d.flowerBed == null) d.flowerBed = Prop("MC_Rounded_Grass");
        if (d.lawnTrees == null || d.lawnTrees.Length == 0 ||
            d.lawnTrees.Any(m => m?.parts == null || m.parts.Length == 0 || m.parts.Any(p => p == null)))
        {
            d.lawnTrees = GreenBeltTrees();
        }
        if (d.cutoutLeaves == null) d.cutoutLeaves = CutoutLeaves();
        EditorUtility.SetDirty(d);
        return d;
    }

    public const string GreenBeltPrefab = "Assets/Prefabs/ScaledCity/City_Pack_01_Scaled.prefab";
    private const string LeavesMaterialPath = "Assets/Materials/VoronoiLeaves.mat";
    private const string PackLeavesPath = "Assets/Modular City Pack/Materials/MC_Plants_Alpha.mat";
    private const string CutoutShader = "Swarm/Double-sided Cutout";

    /// <summary>The pack's tree models, as DiamondCityWorld's street medians use them: in <see cref="GreenBeltPrefab"/>
    /// every tree is a <c>Bark_*</c> object beside a <c>Leaves_*</c> one, not one hierarchy, and each bark mesh always goes
    /// with one leaves mesh. One model per bark mesh: its first active bark object and the leaves object nearest it.</summary>
    private static CityDressing.PropModel[] GreenBeltTrees()
    {
        var city = AssetDatabase.LoadAssetAtPath<GameObject>(GreenBeltPrefab);
        if (city == null) return new CityDressing.PropModel[0];
        bool Active(Transform t)
        {
            for (; t != null; t = t.parent)
            {
                if (!t.gameObject.activeSelf) return false;
            }
            return true;
        }
        MeshFilter[] filters = city.GetComponentsInChildren<MeshFilter>(true)
                                   .Where(f => f.sharedMesh != null && Active(f.transform)).ToArray();
        MeshFilter[] leaves = filters.Where(f => f.name.StartsWith("Leaves")).ToArray();
        var models = new List<CityDressing.PropModel>();
        foreach (var bark in filters.Where(f => f.name.StartsWith("Bark")).GroupBy(f => f.sharedMesh)
                                    .OrderBy(g => g.Key.name).Select(g => g.First()))
        {
            MeshFilter leaf = leaves.OrderBy(l => (l.transform.position - bark.transform.position).sqrMagnitude)
                                    .FirstOrDefault();
            if (leaf == null || (leaf.transform.position - bark.transform.position).magnitude > 3f) continue;
            models.Add(new CityDressing.PropModel { parts = new[] { bark.gameObject, leaf.gameObject } });
        }
        if (models.Count == 0) Debug.LogWarning($"VoronoiCityWorldBuilder: no bark and leaves pairs in {GreenBeltPrefab}.");
        return models.ToArray();
    }

    /// <summary>The trees' leaves as a cutout (<see cref="CutoutShader"/>, Assets/Shaders/DoubleSidedCutout.shader), made
    /// from the pack's MC_Plants_Alpha if missing: the dressing merges its trees at Play, and the pack's blended leaves
    /// would then draw in index order rather than by distance. The pack's own material is left alone.</summary>
    private static Material CutoutLeaves()
    {
        var m = AssetDatabase.LoadAssetAtPath<Material>(LeavesMaterialPath);
        if (m != null) return m;
        var src = AssetDatabase.LoadAssetAtPath<Material>(PackLeavesPath);
        Shader shader = Shader.Find(CutoutShader);
        if (src == null || shader == null)
        {
            Debug.LogWarning($"VoronoiCityWorldBuilder: no {PackLeavesPath} or no shader {CutoutShader}; the trees keep " +
                             "their blended leaves.");
            return null;
        }
        m = new Material(shader) { name = Path.GetFileNameWithoutExtension(LeavesMaterialPath) };
        m.mainTexture = src.mainTexture;
        m.mainTextureScale = src.mainTextureScale;
        m.mainTextureOffset = src.mainTextureOffset;
        m.color = src.color;
        m.SetFloat("_Cutoff", 0.4f);
        AssetDatabase.CreateAsset(m, LeavesMaterialPath);
        return m;
    }

    public const string PropFolder = "Assets/Modular City Pack/Prefabs/Props";
    public const string StreetLightPatch = "Assets/Prefabs/ScaledCity/Patches/MC_Patch_16_Scaled.prefab";
    /// <summary>The pack's road light (BnP_Street_Light_*, by its mesh: the others of that name are traffic signals).</summary>
    private const string StreetLightMesh = "BnP_Street_Light_02";
    private const string GrassMaterialPath = "Assets/Materials/DiamondParkGrass.mat";

    /// <summary>The first active road light in <see cref="StreetLightPatch"/>, or null.</summary>
    private static GameObject StreetLight()
    {
        var patch = AssetDatabase.LoadAssetAtPath<GameObject>(StreetLightPatch);
        if (patch == null) return null;
        foreach (MeshFilter f in patch.GetComponentsInChildren<MeshFilter>(true))
        {
            if (f.sharedMesh == null || f.sharedMesh.name != StreetLightMesh) continue;
            bool active = true;
            for (Transform t = f.transform; t != null; t = t.parent) active &= t.gameObject.activeSelf;
            if (active) return f.gameObject;
        }
        Debug.LogWarning($"VoronoiCityWorldBuilder: no {StreetLightMesh} in {StreetLightPatch}; no street lights.");
        return null;
    }

    /// <summary>A copy of the pack material at <see cref="AsphaltTint"/>, made if missing (so a tint set by hand on it
    /// survives a rebuild). The pack's own materials tint the albedo to 0.196, which takes asphalt to ~5% and the
    /// white paint of the lane marks and crossings to ~15%: from a drone they read as black. The pack's materials are
    /// left alone; ParkRoad and the tile cities use them.</summary>
    private static Material Tinted(string path, string sourcePath)
    {
        var m = AssetDatabase.LoadAssetAtPath<Material>(path);
        if (m != null) return m;
        var src = AssetDatabase.LoadAssetAtPath<Material>(sourcePath);
        if (src == null) return null;
        m = new Material(src) { name = Path.GetFileNameWithoutExtension(path) };
        m.color = new Color(AsphaltTint, AsphaltTint, AsphaltTint, 1f);
        AssetDatabase.CreateAsset(m, path);
        return m;
    }

    private static string NoLayoutHint(string layout) =>
        string.IsNullOrEmpty(layout)
            ? " Generate one first: python -B -m citysim.citygen mixcity --city DiamondCityWorld (in Assets/Scripts/Experiment)."
            : "";
}
