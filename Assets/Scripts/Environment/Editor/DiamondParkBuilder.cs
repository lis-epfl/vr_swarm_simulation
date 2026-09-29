using System.Collections.Generic;
using System.IO;
using System.Linq;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;
using UnityEngine.SceneManagement;

/// <summary>
/// Dresses <c>Diamond_B</c> as a formal park, writing straight into its prefab. The diamond was restyled by hand into a
/// cross-shaped lawn of grass planes, a fountain at its centre and two small plazas (nested ScaledCity patches) across
/// the north–south corridors; this adds what makes that read as a park, and makes the lawn look like the terrain's grass.
///
/// <para><b>Everything it adds hangs under one <c>Park</c> child that is rebuilt from nothing on every run</b>, so the
/// layout is this file, not hand edits: change a constant and run it again. Hand edits elsewhere in the prefab survive
/// a run; hand edits under <c>Park</c> do not. The layout is measured from what the prefab already holds — the lawn is
/// the union of the grass planes, the centre is the fountain, the axes run through it, and whatever stands on the lawn
/// (plaza footpaths, buildings, props) is a keep-out — so moving a plane or the fountain and re-running follows the
/// change. A spot that fails the lawn or keep-out test is skipped and counted, never nudged.</para>
///
/// <para><b>The layout</b> (~115 trees): a ring round the fountain, open on the four axes; an avenue (a pair of rows)
/// down each axis, from the ring or the far side of a plaza out towards the tip; a staggered double row inside both
/// long edges of the north and south arms; a small grid grove either side of the avenue in the west and east arms; and
/// at each of the four tips a single 50-unit building on a footpath slab slightly larger than its footprint.</para>
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
/// Each comes from a patch nowhere near the diamond, so it does not stand beside its twin. Its footpath is a generated
/// quad with the pack footpath's own texel density: a scaled-down copy of the 76.2-unit footpath slab would shrink its
/// paving five-fold.</para>
///
/// <para><b>The grass planes get <see cref="GrassMaterialPath"/></b>, built from the terrain's grass layer
/// (<see cref="TerrainLayerPath"/>, which all four DiamondCityWorld terrains use): same texture, same metres per repeat
/// (the plane's size over the layer's tile size), matte like the layer. The planes also leave the <c>Obstacle</c> layer.
/// They carry MeshColliders, and <see cref="OlfatiSaber"/> takes any collider on that layer as a cylinder round its
/// bounds: each 76-unit plane stood over the park as a ~54-unit-radius repulsion field, the <c>Road_Structure</c>
/// failure <see cref="ObstacleLayerAuditor"/> describes.</para>
///
/// <para><b>It also keeps the diamond from being taken for a tile</b> (see <see cref="DiamondCityBuilder"/>): a
/// nested patch's kerb (<c>Carbs_NN</c>) is renamed <c>Kerb_NN</c> and hidden, and its <c>MC_Patch</c> nodes become
/// <c>Block</c>. A kerb under the diamonds' root stops <see cref="CityTiles.FindCity(Scene, out string)"/> finding the
/// city at all — which silently stops goal-patch verge planting at runtime — and <see cref="CityObstacleExport"/> lists
/// every <c>MC_Patch</c> node as a tile.</para>
/// </summary>
public static class DiamondParkBuilder
{
    public const string PrefabPath = DiamondCityBuilder.DiamondFolder + "/Diamond_B.prefab";

    private const string GeneratedFolder = DiamondCityBuilder.DiamondFolder + "/Generated";
    private const string GrassMaterialPath = "Assets/Materials/DiamondParkGrass.mat";
    private const string TerrainLayerPath = "Assets/BaseLayer.terrainlayer";
    private const string TreeFolder = "Assets/Tree9";
    private const string PatchFolder = "Assets/Prefabs/ScaledCity/Patches";

    private const string ParkName = "Park";
    private const string GrassName = "Grass";
    private const string GrassPlanePrefix = "Plane";
    private const string FountainPrefix = "fountain";
    private static readonly string[] FootpathPrefixes = { "FootPath_", "Footpath_" };

    /// <summary>Renderers a tree may stand next to: the ground, and the kerbs. The street medians are skipped too.</summary>
    private static readonly string[] GroundPrefixes =
    {
        GrassPlanePrefix, "Road_Structure_", "Green_Belt", CityTiles.KerbPrefix, DiamondCityBuilder.DiamondKerbPrefix,
    };

    private const StaticEditorFlags Everything = (StaticEditorFlags)(-1); // what the city's tiles carry

    // ---------------------------------------------------------------- trees

    /// <summary>Times the prefab's own size, so a park tree stands 7.4–8.5 units tall against a street tree's ~4.6.</summary>
    private const float TreeScale = 1.8f;
    private const float TreeScaleJitter = 0.1f; // either way
    private const int TreeSeed = 1;

    // The model each arrangement uses, by prefab name in TreeFolder; null mixes every model there.
    private const string RingSpecies = "Tree9_3"; // the roundest canopy, and the one standing most nearly over its trunk
    private const string NorthSouthAvenueSpecies = "Tree9_2";
    private const string EastWestAvenueSpecies = "Tree9_4";
    private const string EdgeRowSpecies = "Tree9_5";
    private const string GroveSpecies = null;

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

    private const float EdgeRowInset = 4f; // outer row from the lawn's edge
    private const float EdgeRowGap = 6f;   // outer row to inner row
    private const int EdgeRowTrees = 5;    // in the outer row; the inner row has one fewer, in the gaps
    private const float EdgeRowStartGap = 6f; // past the corner where the arm leaves the cross
    private const float EdgeRowEndGap = 5f;   // short of the tip

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

    /// <summary>
    /// Compact footprints (7.7–12.5 units a side), four different designs, and from patches that stand nowhere near
    /// Diamond_B in the <see cref="DiamondCityBuilder"/> layout.
    /// </summary>
    private static readonly TipBuilding[] TipBuildings =
    {
        new TipBuilding(Tip.North, "MC_Patch_05_Scaled", "BnP_Small_Building_D_004"),
        new TipBuilding(Tip.East, "MC_Patch_03_Scaled", "BnP_Apartment_H_000"),
        new TipBuilding(Tip.South, "MC_Patch_16_Scaled", "BnP_Large_Building_K_002"),
        new TipBuilding(Tip.West, "MC_Patch_19_Scaled", "BnP_Apartment_E_002"),
    };

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

    [MenuItem("Tools/Swarm/Dress Diamond B park")]
    private static void DressFromMenu()
    {
        PrefabStage stage = PrefabStageUtility.GetCurrentPrefabStage();
        if (stage != null && stage.assetPath == PrefabPath)
        {
            EditorUtility.DisplayDialog("Dress Diamond B park",
                "Diamond_B is open in Prefab Mode. Save and close it first: the park is written into the prefab asset, " +
                "and saving the open stage afterwards would overwrite it.", "OK");
            return;
        }
        string summary = Dress(out string error);
        if (summary == null)
        {
            Debug.LogError("DiamondParkBuilder: " + error);
        }
        else
        {
            Debug.Log("DiamondParkBuilder: " + summary);
        }
    }

    [MenuItem("Tools/Swarm/Dress Diamond B park", true)]
    private static bool CanDress()
    {
        return !EditorApplication.isPlayingOrWillChangePlaymode;
    }

    /// <summary>Batch-mode entry point: <c>-executeMethod DiamondParkBuilder.DressInBatch</c>. Exits 0 on success.</summary>
    public static void DressInBatch()
    {
        string summary = Dress(out string error);
        if (summary == null)
        {
            Debug.LogError("DiamondParkBuilder: " + error);
        }
        else
        {
            Debug.Log("DiamondParkBuilder: " + summary);
        }
        EditorApplication.Exit(summary != null ? 0 : 1);
    }

    /// <summary>
    /// Rebuilds Diamond_B's park and saves the prefab. Returns a one-line account of what it did, or null with
    /// <paramref name="error"/> set, in which case the prefab is left as it was.
    /// </summary>
    public static string Dress(out string error)
    {
        // Every tree model in the folder, by name; a prefab without a Tree component is not one.
        List<GameObject> trees = AssetDatabase.FindAssets("t:Prefab", new[] { TreeFolder })
                                              .Select(g => AssetDatabase.LoadAssetAtPath<GameObject>(AssetDatabase.GUIDToAssetPath(g)))
                                              .Where(p => p != null && p.GetComponent<Tree>() != null)
                                              .OrderBy(p => p.name, System.StringComparer.Ordinal)
                                              .ToList();
        string missing = new[] { RingSpecies, NorthSouthAvenueSpecies, EastWestAvenueSpecies, EdgeRowSpecies, GroveSpecies }
                         .FirstOrDefault(s => s != null && trees.All(p => p.name != s));
        TerrainLayer grassLayer = AssetDatabase.LoadAssetAtPath<TerrainLayer>(TerrainLayerPath);
        if (trees.Count == 0 || missing != null || grassLayer == null || grassLayer.diffuseTexture == null)
        {
            error = trees.Count == 0 ? $"no tree prefabs in {TreeFolder}."
                  : missing != null ? $"no tree prefab named {missing} in {TreeFolder}."
                  : grassLayer == null ? $"no terrain layer at {TerrainLayerPath}."
                  : $"{TerrainLayerPath} has no diffuse texture.";
            return null;
        }
        List<TipSource> tips = FindTipSources(out error);
        if (tips == null)
        {
            return null;
        }

        GameObject root = PrefabUtility.LoadPrefabContents(PrefabPath);
        try
        {
            string summary = Populate(root, trees, grassLayer, tips, out error);
            if (summary == null)
            {
                return null;
            }
            PrefabUtility.SaveAsPrefabAsset(root, PrefabPath, out bool saved);
            if (!saved)
            {
                error = $"could not save {PrefabPath}.";
                return null;
            }
            AssetDatabase.SaveAssets();
            return summary;
        }
        finally
        {
            PrefabUtility.UnloadPrefabContents(root);
        }
    }

    private static List<TipSource> FindTipSources(out string error)
    {
        int obstacle = LayerMask.NameToLayer("Obstacle");
        List<TipSource> sources = new List<TipSource>();
        foreach (TipBuilding spec in TipBuildings)
        {
            string path = $"{PatchFolder}/{spec.Patch}.prefab";
            GameObject patch = AssetDatabase.LoadAssetAtPath<GameObject>(path);
            if (patch == null)
            {
                error = $"no patch prefab at {path}.";
                return null;
            }
            Transform building = patch.GetComponentsInChildren<Transform>(true).FirstOrDefault(t => t.name == spec.Building);
            Renderer footpath = patch.GetComponentsInChildren<Renderer>(true)
                                     .FirstOrDefault(r => HasAnyPrefix(r.name, FootpathPrefixes) && r.GetComponent<MeshFilter>());
            if (building == null || footpath == null)
            {
                error = building == null ? $"{spec.Patch} has no {spec.Building}." : $"{spec.Patch} has no footpath.";
                return null;
            }
            if (building.gameObject.layer != obstacle || building.GetComponent<Collider>() == null)
            {
                error = $"{spec.Building} in {spec.Patch} is not a collider on Obstacle, so the swarm would not avoid it.";
                return null;
            }
            sources.Add(new TipSource { Spec = spec, Building = building, Footpath = footpath });
        }
        error = null;
        return sources;
    }

    /// <summary>
    /// Does the work on the loaded prefab contents. Everything is measured in the prefab root's frame, which is the
    /// diamond's: its origin is the centre of the four blocks.
    /// </summary>
    private static string Populate(GameObject rootObject, List<GameObject> trees, TerrainLayer grassLayer,
                                   List<TipSource> tips, out string error)
    {
        Transform root = rootObject.transform;

        Transform oldPark = root.Find(ParkName);
        if (oldPark != null)
        {
            Object.DestroyImmediate(oldPark.gameObject);
        }

        // The lawn.
        Transform grass = root.Find(GrassName);
        List<Renderer> planes = grass == null
            ? new List<Renderer>()
            : grass.GetComponentsInChildren<Renderer>(true).Where(r => r.name.StartsWith(GrassPlanePrefix)).ToList();
        if (planes.Count == 0)
        {
            error = $"{PrefabPath} has no {GrassName}/{GrassPlanePrefix}* renderers to take the lawn from.";
            return null;
        }
        Vector2 planeSize = new Vector2(planes[0].bounds.size.x, planes[0].bounds.size.z);
        if (planes.Any(p => Mathf.Abs(p.bounds.size.x - planeSize.x) > 0.01f || Mathf.Abs(p.bounds.size.z - planeSize.y) > 0.01f))
        {
            error = "the grass planes are not all one size, so no single tiling gives them all the terrain's texel density.";
            return null;
        }
        Material grassMaterial = GrassMaterial(grassLayer, planeSize);
        int grassLayerChanges = 0;
        foreach (Renderer plane in planes)
        {
            plane.sharedMaterials = Enumerable.Repeat(grassMaterial, plane.sharedMaterials.Length).ToArray();
            if (plane.gameObject.layer != 0)
            {
                plane.gameObject.layer = 0; // Default: see the class summary
                grassLayerChanges++;
            }
            GameObjectUtility.SetStaticEditorFlags(plane.gameObject,
                GameObjectUtility.GetStaticEditorFlags(plane.gameObject) | StaticEditorFlags.BatchingStatic);
        }
        List<Rect> lawn = planes.Select(p => Footprint(p.bounds, root)).ToList();
        float groundY = planes.Max(p => root.InverseTransformPoint(p.bounds.max).y);

        int kerbsHidden = 0;
        int nodesRenamed = 0;
        RenameTileMarkers(root, ref kerbsHidden, ref nodesRenamed);

        Renderer fountain = root.Cast<Transform>()
                                .Where(t => t.name.StartsWith(FountainPrefix, System.StringComparison.OrdinalIgnoreCase))
                                .SelectMany(t => t.GetComponentsInChildren<Renderer>())
                                .FirstOrDefault();
        if (fountain == null)
        {
            error = $"{PrefabPath} has no {FountainPrefix}* child to centre the park on.";
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
        List<TreeGroup> groups = new List<TreeGroup> { new TreeGroup("FountainRing", RingSpecies, Ring(centre)) };
        foreach (Tip tip in new[] { Tip.North, Tip.East, Tip.South, Tip.West })
        {
            bool northSouth = tip == Tip.North || tip == Tip.South;
            groups.Add(new TreeGroup($"Avenue_{tip}", northSouth ? NorthSouthAvenueSpecies : EastWestAvenueSpecies,
                                     Avenue(tip, centre, tipFootpaths[tip], keepOuts, out float plazaFarEdge)));
            groups.Add(northSouth
                ? new TreeGroup($"EdgeRows_{tip}", EdgeRowSpecies, EdgeRows(tip, centre, lawn))
                : new TreeGroup($"Groves_{tip}", GroveSpecies, Groves(tip, centre, lawn, plazaFarEdge)));
        }

        System.Random rng = new System.Random(TreeSeed);
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
                // Every spot draws its numbers whether or not it is planted, so one skipped spot does not reshuffle
                // every tree after it.
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

        error = null;
        return $"planted {planted} trees ({string.Join(", ", counts)}; {offLawn} spot(s) off the lawn, {crowded} too close " +
               $"to something; by model {string.Join(", ", perModel.Select(m => $"{m.Key} {m.Value}"))}) round the fountain " +
               $"at ({centre.x:F1}, {centre.y:F1}); tips: {string.Join("; ", tipNotes)}; " +
               $"{planes.Count} grass planes now use {GrassMaterialPath} at {grassMaterial.mainTextureScale.x:F3} repeats " +
               $"({grassLayerChanges} moved off Obstacle); {kerbsHidden} stray kerb(s) renamed and hidden, " +
               $"{nodesRenamed} MC_Patch node(s) renamed Block.";
    }

    // ---------------------------------------------------------------- the grass

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

    // ---------------------------------------------------------------- tile markers

    /// <summary>
    /// Renames what the city tools recognise a tile by, as <see cref="DiamondCityBuilder"/> does for its drafts, and
    /// hides a nested patch's kerb: at full block size round a shrunk plaza it is a stray outline on the lawn.
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
        // Heights are kept from the patch, measured from its root: its footpath and its building's base.
        float pivotY = source.Building.position.y - source.Building.root.position.y;
        float footpathY = source.Footpath.bounds.center.y - source.Footpath.transform.root.position.y;
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

        // The footpath: long side along the edge, outer edge TipInset in from it.
        Vector2 size = footprint.size + 2f * FootpathMargin * Vector2.one;
        float depth = alongX ? size.y : size.x;
        Vector2 middle = centre + axis * (reach - TipInset - depth / 2f);
        Rect footpath = new Rect(middle - size / 2f, size);

        // Centre the building on it.
        Vector3 shift = new Vector3(middle.x - footprint.center.x, 0f, middle.y - footprint.center.y);
        b.position += root.TransformVector(shift);
        footprint.center = middle;

        Mesh mesh =FootpathMesh($"{Path.GetFileNameWithoutExtension(PrefabPath)}_Footpath_{tip}", size,
                                 FitUv(source.Footpath, out float residual));
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
    /// A staggered double row inside each long edge of the north or south arm: from just past the corner where the
    /// arm leaves the cross, to just short of the tip.
    /// </summary>
    private static List<Vector2> EdgeRows(Tip tip, Vector2 centre, List<Rect> lawn)
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
            float a = corner + EdgeRowStartGap;
            float b = reach - EdgeRowEndGap;
            float step = (b - a) / (EdgeRowTrees - 1);
            for (int k = 0; k < EdgeRowTrees; k++)
            {
                spots.Add(centre + axis * (a + k * step) + side * sign * (edge - EdgeRowInset));
            }
            for (int k = 0; k < EdgeRowTrees - 1; k++)
            {
                spots.Add(centre + axis * (a + (k + 0.5f) * step) + side * sign * (edge - EdgeRowInset - EdgeRowGap));
            }
        }
        return spots;
    }

    /// <summary>
    /// A <see cref="GroveAlong"/> x <see cref="GroveAcross"/> grid either side of the avenue in the west or east arm,
    /// centred in the lawn between the plaza and the arm's end, and between the avenue and the arm's edge.
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
        for (Transform p = t.parent; p != null && p != root; p = p.parent)
        {
            if (p.name == container)
            {
                return true;
            }
        }
        return false;
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
