using System;
using System.Collections.Generic;
using System.IO;
using UnityEditor;
using UnityEngine;
using UnityEngine.SceneManagement;

/// <summary>
/// The scenic city's ground (called by <see cref="ScenicPatchCityWorldBuilder"/>): the roads, footpaths and lawns in
/// <c>Swarm/Scenic Ground</c>, and the land round the city repainted.
///
/// <para><b>The land round the city</b> was one 512-pixel photograph of grass laid every 7 m over a flat square
/// clearing, with the woods beyond it ending on straight lines. It is the four city Terrain tiles, whose TerrainData
/// five other scenes share, so this scene is given copies of its own (<c>Assets/Scenic/Terrain/CityTile_*.asset</c>)
/// and only those are changed:
/// the paint becomes three photographed grounds (Poly Haven, CC0) mixed by noise at several scales -- a green grass,
/// an aerial photograph of grass and bare earth laid 47 m to a repeat, a dry grass -- with worn strips along the road in
/// and round the city's edge, so nowhere is one texture left alone long enough to show its repeat; and the woods' edge
/// is thinned raggedly and clumps of the same trees are scattered into the clearing.
/// <b>Heights are not touched</b> (the tiles stay flat), so the altitude ceiling and the height floor, which are
/// measured from the terrain, are what they were. The trees are the tiles' own prototype (Tree9_3, which has no
/// collider) and none is planted within <see cref="TreeFreeCity"/> of the city, <see cref="TreeFreeRoad"/> of the road
/// in or <see cref="TreeFreeSpawn"/> of the spawn, so the drones never fly among them.</para>
///
/// <para><b>The landmarks' ground</b> (<see cref="Landmarks"/>) is repainted the same way: the twelve Terrain tiles
/// round the city, which Landmarks.prefab shares with five other scenes, are replaced in this scene by copies of its own
/// (<c>Assets/Scenic/Terrain/Ring_*.asset</c>), the prefab instance unpacked first so that nothing done here can reach
/// the prefab (LandmarkBuilder's <i>Place models</i> applies a scene instance's overrides to it). The mountain keeps
/// its rock and snow where they were painted, in photographed rock and snow; the hills and the turbine ridge, painted
/// before with a dark generated forest texture, are grassland with real copses where that paint was, few on the ridge
/// and none within <see cref="TreeFreeTurbine"/> of a turbine; the steepest slopes break into rock. Heights are not
/// touched, nor the turbines, the mast, the lake or the mountain's own trees. Past the ring a skirt of ground
/// (<see cref="Skirt"/>) runs on to the horizon, where the ground used to stop in a straight edge 2 km out.</para>
///
/// <para><b>Every tile is drawn in <c>Swarm/Scenic Terrain</c></b> (<c>ScenicTerrain.mat</c>): its layers laid in world
/// metres, so tiles join without a seam, sampled at two scales against the repeat, and projected from the side on
/// steep ground, so the mountain's rock does not stretch into streaks. Four layers per tile, one pass: the city tiles
/// grass, sand, field and dry grass; the ring the same with rock for sand, and on the mountain snow for dry grass. Dry
/// grass fades out towards the mountain's tiles so their joint shows no edge. The beaches are photographed sand now, so
/// if the lake is edited in this scene, <c>ClearTreesUnderLakes</c> (which looks for Landmarks' own Sand layer) adds
/// that layer as a fifth: rebuild the scene after such an edit.</para>
/// </summary>
public static class ScenicGroundBuilder
{
    private const string Folder = ScenicPatchCityWorldBuilder.Folder;
    private const string Textures = Folder + "/Textures";
    private const string Materials = Folder + "/Materials";
    private const string TerrainFolder = Folder + "/Terrain";
    private const string SandLayer = "Assets/Landmarks/Terrain/Sand.terrainlayer";
    private const string TreePrefab = "Assets/Tree9/Tree9_3.prefab";
    private const string TerrainMaterialPath = Materials + "/ScenicTerrain.mat";

    /// <summary>m round a turbine kept clear of the copies' trees: open grass under the rotors, as on a wind farm.</summary>
    private const float TreeFreeTurbine = 140f;
    /// <summary>The ring's outer edge (m from the centre, each way) and how far the skirt runs on past it.</summary>
    private const float RingEdge = 2000f, SkirtReach = 1000f;

    /// <summary>m from the city's streets, from the road in, and from the spawn inside which no tree is planted.</summary>
    private const float TreeFreeCity = 75f, TreeFreeRoad = 45f, TreeFreeSpawn = 110f;

    // ------------------------------------------------------------------ roads, footpaths, lawns

    /// <summary>Gives the builder and its dressing the scenic ground materials (before the city is baked) and the road
    /// in (ParkRoad) the scenic road. A summary.</summary>
    public static string Surfaces(ProceduralCityBuilder builder, Scene scene)
    {
        EnsureFolder(Materials);
        Texture2D noise = Texture($"{Textures}/Noise.png", false, false);
        Texture2D asphalt = Texture($"{Textures}/Ground/asphalt_02_diff.jpg", true, false);
        Texture2D asphaltNormal = Texture($"{Textures}/Ground/asphalt_02_nor.jpg", false, true);
        Texture roadPattern = builder.streetMaterial != null ? builder.streetMaterial.mainTexture : null;
        Texture junctionPattern = builder.junctionMaterial != null ? builder.junctionMaterial.mainTexture : null;
        if (roadPattern == null || junctionPattern == null)
        {
            throw new InvalidOperationException("the source scene's street and junction materials have no texture to take the road markings from.");
        }

        Material Road(string name, Material from)
        {
            Material m = Ground(name);
            m.SetTexture("_MainTex", from.mainTexture);
            m.SetTextureScale("_MainTex", from.mainTextureScale);
            m.SetTextureOffset("_MainTex", from.mainTextureOffset);
            m.SetFloat("_Marks", 1f);
            m.EnableKeyword("SCENIC_MARKS");
            m.SetTexture("_SurfaceTex", asphalt);
            m.SetTexture("_SurfaceBump", asphaltNormal);
            m.SetFloat("_SurfaceMetres", 3f);
            m.SetTexture("_Noise", noise);
            m.SetFloat("_Stains", 0.35f);
            // the photograph is of old, pale asphalt: a city street's is darker, and the paint shows better on it
            m.SetColor("_Color", new Color(0.78f, 0.78f, 0.78f).gamma);
            EditorUtility.SetDirty(m);
            return m;
        }

        Material road = Road("ScenicRoad", builder.streetMaterial), junction = Road("ScenicJunction", builder.junctionMaterial);

        Material footpath = Ground("ScenicFootpath");
        footpath.SetTexture("_MainTex", Texture($"{Textures}/Ground/concrete_pavement_diff.jpg", true, false));
        footpath.SetTexture("_BumpMap", Texture($"{Textures}/Ground/concrete_pavement_nor.jpg", false, true));
        footpath.SetFloat("_Marks", 0f);
        footpath.DisableKeyword("SCENIC_MARKS");
        footpath.SetTexture("_Noise", noise);
        footpath.SetFloat("_Stains", 0.3f);
        footpath.SetFloat("_StainMetres", 29f);
        // No lighter than BlockShapedPatchCityWorld's footpath, which was darkened so the pedestrians show on it.
        footpath.SetColor("_Color", new Color(0.9f, 0.9f, 0.9f).gamma);
        EditorUtility.SetDirty(footpath);

        Material lawn = Ground("ScenicLawn");
        lawn.SetTexture("_MainTex", Texture($"{Textures}/Terrain/leafy_grass_diff.jpg", true, false));
        lawn.SetTexture("_BumpMap", Texture($"{Textures}/Terrain/leafy_grass_nor.jpg", false, true));
        lawn.SetFloat("_Marks", 0f);
        lawn.DisableKeyword("SCENIC_MARKS");
        lawn.SetTexture("_Noise", noise);
        lawn.SetFloat("_Stains", 0.45f);
        lawn.SetFloat("_StainMetres", 23f);
        EditorUtility.SetDirty(lawn);

        Texture oldRoad = roadPattern, oldJunction = junctionPattern;
        Undo.RecordObject(builder, "Scenic ground");
        builder.streetMaterial = road;
        builder.junctionMaterial = junction;
        EditorUtility.SetDirty(builder);
        CityDressing dressing = builder.GetComponent<CityDressing>();
        if (dressing != null)
        {
            Undo.RecordObject(dressing, "Scenic ground");
            dressing.pavementMaterial = footpath;
            dressing.pavementUvMetres = 1.8f;                     // the photograph is of 1.8 m of paving
            dressing.grassMaterial = lawn;
            lawn.mainTextureScale = Vector2.one * (dressing.grassUvMetres / 2f);      // and of 2 m of grass
            EditorUtility.SetDirty(dressing);
        }

        // The road from the spawn to the city is the pack's tiles, in the same two textures.
        int swapped = 0;
        foreach (GameObject root in scene.GetRootGameObjects())
        {
            if (root.GetComponent<ProceduralCityBuilder>() != null || root.GetComponentInChildren<Terrain>() != null) continue;
            foreach (MeshRenderer r in root.GetComponentsInChildren<MeshRenderer>(true))
            {
                Material[] mats = r.sharedMaterials;
                bool changed = false;
                for (int k = 0; k < mats.Length; k++)
                {
                    // not one already swapped (the tools call this again before a bake or a redress)
                    if (mats[k] == null || mats[k] == road || mats[k] == junction || !mats[k].HasProperty("_MainTex") ||
                        mats[k].shader == road.shader) continue;
                    Texture t = mats[k].mainTexture;
                    if (t != oldRoad && t != oldJunction) continue;
                    Material scenic = Road(t == oldRoad ? "ScenicRoad_" + mats[k].name : "ScenicJunction_" + mats[k].name, mats[k]);
                    mats[k] = scenic;
                    changed = true;
                }
                if (!changed) continue;
                Undo.RecordObject(r, "Scenic ground");
                r.sharedMaterials = mats;
                swapped++;
            }
        }
        return $"roads, footpaths and lawns in Swarm/Scenic Ground ({swapped} renderers of the road in)";
    }

    private static Material Ground(string name)
    {
        string path = $"{Materials}/{name}.mat";
        var m = AssetDatabase.LoadAssetAtPath<Material>(path);
        if (m == null)
        {
            m = new Material(Shader.Find("Swarm/Scenic Ground")) { name = name };
            AssetDatabase.CreateAsset(m, path);
        }
        return m;
    }

    /// <summary>A repeating, mipmapped texture, as colour, as data or as a normal map.</summary>
    private static Texture2D Texture(string path, bool srgb, bool normalMap)
    {
        var importer = AssetImporter.GetAtPath(path) as TextureImporter;
        if (importer == null) throw new InvalidOperationException($"no texture at {path}.");
        TextureImporterType type = normalMap ? TextureImporterType.NormalMap : TextureImporterType.Default;
        if (importer.textureType != type || importer.sRGBTexture != (srgb && !normalMap) || !importer.mipmapEnabled ||
            importer.wrapMode != TextureWrapMode.Repeat || importer.anisoLevel != 8 || importer.filterMode != FilterMode.Trilinear)
        {
            importer.textureType = type;
            importer.sRGBTexture = srgb && !normalMap;
            importer.mipmapEnabled = true;
            importer.wrapMode = TextureWrapMode.Repeat;
            importer.filterMode = FilterMode.Trilinear;
            importer.anisoLevel = 8;
            importer.SaveAndReimport();
        }
        var texture = AssetDatabase.LoadAssetAtPath<Texture2D>(path);
        if (texture == null) throw new InvalidOperationException($"{path} did not import as a texture.");
        return texture;
    }

    private static void EnsureFolder(string folder)
    {
        if (AssetDatabase.IsValidFolder(folder)) return;
        EnsureFolder(Path.GetDirectoryName(folder).Replace('\\', '/'));
        AssetDatabase.CreateFolder(Path.GetDirectoryName(folder).Replace('\\', '/'), Path.GetFileName(folder));
    }

    // ------------------------------------------------------------------ the land round the city

    /// <summary>Gives every city Terrain tile in <paramref name="scene"/> its scenic copy, repainted and replanted. The
    /// city is baked by now: its streets' bounds are where the city ends. The copies are
    /// <c>Assets/Scenic/Terrain/&lt;tilePrefix&gt;_&lt;x&gt;_&lt;z&gt;.asset</c>: each scene with a city of its own needs its
    /// own, since the worn strips and the clearing follow its city. The tiles must still be the source scene's (the paint
    /// and the trees are worked out from theirs). A summary.</summary>
    public static string Surroundings(ProceduralCityBuilder builder, Scene scene, string tilePrefix = "CityTile")
    {
        EnsureFolder(TerrainFolder);
        TerrainLayer grass = Layer("ScenicGrass", "forrest_ground_01", 5.5f);
        TerrainLayer field = Layer("ScenicField", "aerial_grass_rock", 47f);
        TerrainLayer dry = Layer("ScenicDryGrass", "withered_grass", 6.5f);
        TerrainLayer sand = Layer("ScenicSand", "coast_sand_01", 8f);
        var lakeSand = AssetDatabase.LoadAssetAtPath<TerrainLayer>(SandLayer);
        Material look = TerrainMaterial();

        // Where the city, the road in and the spawn are, in world (x, z).
        Rect city = Footprint(builder.transform);
        var roads = new List<Rect>();
        foreach (GameObject root in scene.GetRootGameObjects())
        {
            if (root.name == "ParkRoad") roads.Add(Footprint(root.transform));
        }
        swarmSpawn spawner = UnityEngine.Object.FindFirstObjectByType<swarmSpawn>(FindObjectsInactive.Include);
        Vector2 spawn = spawner != null ? new Vector2(spawner.start_x, spawner.start_z) : new Vector2(city.xMin - 90f, city.center.y);

        int tiles = 0, removed = 0, planted = 0;
        foreach (GameObject root in scene.GetRootGameObjects())
        {
            foreach (Terrain t in root.GetComponentsInChildren<Terrain>(true))
            {
                if (t.terrainData == null || t.GetComponentInParent<LandmarkSet>() != null) continue;
                string source = AssetDatabase.GetAssetPath(t.terrainData);
                Vector3 o = t.transform.position;
                string path = $"{TerrainFolder}/{tilePrefix}_{Signed(o.x)}_{Signed(o.z)}.asset";
                if (source != path)
                {
                    // Over a tile from an earlier build the file is replaced under its .meta, so its GUID (and with
                    // it every rebuild's diff) stays put.
                    if (File.Exists(path))
                    {
                        File.Copy(source, path, true);
                        AssetDatabase.ImportAsset(path, ImportAssetOptions.ForceUpdate);
                    }
                    else if (!AssetDatabase.CopyAsset(source, path))
                    {
                        throw new InvalidOperationException($"could not copy {source} to {path}.");
                    }
                }
                var data = AssetDatabase.LoadAssetAtPath<TerrainData>(path);
                Paint(data, o, new[] { grass, sand, field, dry }, lakeSand, city, roads);
                Replant(data, o, city, roads, spawn, ref removed, ref planted);
                EditorUtility.SetDirty(data);
                Undo.RecordObject(t, "Scenic terrain");
                t.terrainData = data;
                UseLook(t, look);
                var collider = t.GetComponent<TerrainCollider>();
                if (collider != null)
                {
                    Undo.RecordObject(collider, "Scenic terrain");
                    collider.terrainData = data;
                }
                EditorUtility.SetDirty(t);
                tiles++;
            }
        }
        TerrainHeightSampler.Invalidate();
        return $"{tiles} terrain tiles repainted in 3 grounds ({removed} trees thinned from the woods' edge, {planted} planted in clumps; " +
               $"city {city}, road in {(roads.Count > 0 ? roads[0].ToString() : "none")}, spawn {spawn})";
    }

    private static string Signed(float v) => (v < 0f ? "m" : "") + Mathf.RoundToInt(Mathf.Abs(v));

    private static Rect Footprint(Transform root)
    {
        bool any = false;
        var b = new Bounds();
        foreach (Renderer r in root.GetComponentsInChildren<Renderer>(true))
        {
            if (!any) b = r.bounds;
            else b.Encapsulate(r.bounds);
            any = true;
        }
        return any ? Rect.MinMaxRect(b.min.x, b.min.z, b.max.x, b.max.z) : new Rect(root.position.x, root.position.z, 0f, 0f);
    }

    private static TerrainLayer Layer(string name, string texture, float tile)
    {
        string path = $"{TerrainFolder}/{name}.terrainlayer";
        var layer = AssetDatabase.LoadAssetAtPath<TerrainLayer>(path);
        if (layer == null)
        {
            layer = new TerrainLayer();
            AssetDatabase.CreateAsset(layer, path);
        }
        layer.diffuseTexture = Texture($"{Textures}/Terrain/{texture}_diff.jpg", true, false);
        layer.normalMapTexture = Texture($"{Textures}/Terrain/{texture}_nor.jpg", false, true);
        layer.tileSize = new Vector2(tile, tile);
        layer.tileOffset = Vector2.zero;
        layer.metallic = 0f;
        layer.smoothness = 0f;
        layer.specular = Color.black;
        layer.normalScale = 1f;
        EditorUtility.SetDirty(layer);
        return layer;
    }

    /// <summary>Smooth noise in 0..1 (four octaves of Perlin), <paramref name="scale"/> metres to its largest feature.</summary>
    private static float Fbm(float x, float z, float scale, float offset)
    {
        float sum = 0f, amp = 1f, total = 0f, f = 1f / scale;
        for (int k = 0; k < 4; k++)
        {
            sum += amp * Mathf.PerlinNoise(x * f + offset + 137.3f * k, z * f - offset + 71.9f * k);
            total += amp;
            amp *= 0.5f;
            f *= 2.03f;
        }
        return Mathf.Clamp01(sum / total);
    }

    /// <summary>1 away from the mountain's tiles, falling to 0 at their joint (z = -1000), where the mountain's tiles
    /// carry snow in the slot the others give dry grass.</summary>
    private static float ToMountain(float z) => Step(-1000f, -850f, z);

    /// <summary>0 below <paramref name="lo"/>, 1 above <paramref name="hi"/>, smooth between (not Mathf.SmoothStep,
    /// which interpolates between its first two arguments).</summary>
    private static float Step(float lo, float hi, float x)
    {
        float t = Mathf.Clamp01((x - lo) / (hi - lo));
        return t * t * (3f - 2f * t);
    }

    /// <summary>m from <paramref name="p"/> to <paramref name="r"/>, 0 inside.</summary>
    private static float Outside(Rect r, Vector2 p)
    {
        float dx = Mathf.Max(r.xMin - p.x, 0f, p.x - r.xMax), dz = Mathf.Max(r.yMin - p.y, 0f, p.y - r.yMax);
        return Mathf.Sqrt(dx * dx + dz * dz);
    }

    /// <summary>The tile's paint: grass and field mixed at two scales, dry patches, worn ground along the road in and
    /// round the city, and the sand it already had (the lake's beaches) kept as it was.</summary>
    private static void Paint(TerrainData data, Vector3 origin, TerrainLayer[] layers, TerrainLayer sandLayer, Rect city, List<Rect> roads)
    {
        int res = data.alphamapResolution;
        TerrainLayer[] before = data.terrainLayers;
        int sandBefore = Array.IndexOf(before, sandLayer);
        float[,,] old = data.GetAlphamaps(0, 0, res, res);
        var paint = new float[res, res, layers.Length];
        Vector3 size = data.size;
        for (int y = 0; y < res; y++)
        {
            for (int x = 0; x < res; x++)
            {
                float wx = origin.x + (x + 0.5f) / res * size.x, wz = origin.z + (y + 0.5f) / res * size.z;
                var p = new Vector2(wx, wz);
                Mix(wx, wz, out float field, out float dry, out float bare, out float fine);
                // worn ground: a strip round the city's streets and along the road in
                float edge = Outside(city, p);
                float worn = edge > 0f ? Mathf.Clamp01(1f - edge / (5f + 6f * fine)) : 0f;
                foreach (Rect road in roads)
                {
                    float d = Outside(road, p);
                    if (d > 0f) worn = Mathf.Max(worn, Mathf.Clamp01(1f - d / (2.5f + 5f * fine)));
                }
                dry = Mathf.Max(dry, 0.8f * worn) * ToMountain(wz);
                bare = Mathf.Max(bare, 0.45f * worn * worn);
                float sand = Mathf.Max(sandBefore >= 0 ? old[y, x, sandBefore] : 0f, bare * (1f - dry * 0.5f));
                float rest = 1f - sand;
                paint[y, x, 1] = sand;
                paint[y, x, 3] = rest * dry;
                paint[y, x, 2] = rest * (1f - dry) * field;
                paint[y, x, 0] = rest * (1f - dry) * (1f - field);
            }
        }
        data.terrainLayers = layers;
        data.SetAlphamaps(0, 0, paint);
    }

    // ------------------------------------------------------------------ the terrain material

    /// <summary><c>Swarm/Scenic Terrain</c>, made if missing, its values set on every build.</summary>
    public static Material TerrainMaterial()
    {
        EnsureFolder(Materials);
        var m = AssetDatabase.LoadAssetAtPath<Material>(TerrainMaterialPath);
        if (m == null)
        {
            m = new Material(Shader.Find("Swarm/Scenic Terrain")) { name = "ScenicTerrain" };
            AssetDatabase.CreateAsset(m, TerrainMaterialPath);
        }
        m.SetTexture("_MacroNoise", Texture($"{Textures}/Noise.png", false, false));
        m.SetFloat("_TerrainSize", 1000f);
        m.SetFloat("_MacroMetres", 700f);
        m.SetFloat("_MacroAmount", 0.35f);
        m.SetFloat("_BombMetres", 90f);
        m.SetFloat("_SteepStart", 0.85f);
        m.SetFloat("_NormalStrength", 1f);
        EditorUtility.SetDirty(m);
        return m;
    }

    /// <summary>The tile drawn in the scenic terrain material at every distance (the base map, a low-resolution
    /// picture of the paint Unity draws instead past basemapDistance, would show the old flat look on the far
    /// hills), and not instanced (the material builds no instanced mesh).</summary>
    private static void UseLook(Terrain t, Material look)
    {
        Undo.RecordObject(t, "Scenic terrain");
        t.materialTemplate = look;
        t.drawInstanced = false;
        t.basemapDistance = 20000f;
        EditorUtility.SetDirty(t);
    }

    // ------------------------------------------------------------------ the landmarks' ground

    /// <summary>The ring of landmark tiles in this scene's own copies, repainted and planted; the Landmarks prefab
    /// instance unpacked first. With <paramref name="reuse"/>, a tile whose copy already exists is given that copy as it
    /// stands (the ring does not depend on the city, so the scenic practice scene shares the scenic scene's). A summary.</summary>
    public static string Landmarks(Scene scene, bool reuse = false)
    {
        LandmarkSet set = null;
        foreach (GameObject root in scene.GetRootGameObjects())
        {
            set = root.GetComponentInChildren<LandmarkSet>(true);
            if (set != null) break;
        }
        if (set == null || set.ground == null) return "no landmarks to repaint";
        GameObject instance = PrefabUtility.GetOutermostPrefabInstanceRoot(set.gameObject);
        if (instance != null)
        {
            PrefabUtility.UnpackPrefabInstance(instance, PrefabUnpackMode.Completely, InteractionMode.AutomatedAction);
        }

        EnsureFolder(TerrainFolder);
        TerrainLayer grass = Layer("ScenicGrass", "forrest_ground_01", 5.5f);
        TerrainLayer field = Layer("ScenicField", "aerial_grass_rock", 47f);
        TerrainLayer dry = Layer("ScenicDryGrass", "withered_grass", 6.5f);
        TerrainLayer rock = Layer("ScenicRock", "aerial_rocks_02", 26f);
        TerrainLayer snow = Layer("ScenicSnow", "snow_field_aerial", 40f);
        Material look = TerrainMaterial();
        var turbines = new List<Vector2>();
        if (set.windFarm != null)
        {
            foreach (Transform t in set.windFarm) turbines.Add(new Vector2(t.position.x, t.position.z));
        }
        var treePrefab = AssetDatabase.LoadAssetAtPath<GameObject>(TreePrefab);

        int tiles = 0, planted = 0, shared = 0;
        foreach (Terrain t in set.ground.GetComponentsInChildren<Terrain>(true))
        {
            if (t.terrainData == null) continue;
            string source = AssetDatabase.GetAssetPath(t.terrainData);
            string path = $"{TerrainFolder}/Ring_{t.terrainData.name}.asset";
            var existing = reuse ? AssetDatabase.LoadAssetAtPath<TerrainData>(path) : null;
            if (existing != null)
            {
                Undo.RecordObject(t, "Scenic terrain");
                t.terrainData = existing;
                var c = t.GetComponent<TerrainCollider>();
                if (c != null)
                {
                    Undo.RecordObject(c, "Scenic terrain");
                    c.terrainData = existing;
                }
                UseLook(t, look);
                tiles++;
                shared++;
                continue;
            }
            if (source != path)
            {
                if (File.Exists(path))
                {
                    File.Copy(source, path, true);
                    AssetDatabase.ImportAsset(path, ImportAssetOptions.ForceUpdate);
                }
                else if (!AssetDatabase.CopyAsset(source, path))
                {
                    throw new InvalidOperationException($"could not copy {source} to {path}.");
                }
            }
            var data = AssetDatabase.LoadAssetAtPath<TerrainData>(path);
            Vector3 o = t.transform.position;
            bool mountain = o.z < -1500f, ridge = o.x > 999f;
            float[,,] forest = RepaintRing(data, o, mountain,
                                           mountain ? new[] { grass, rock, field, snow } : new[] { grass, rock, field, dry });
            if (!mountain) planted += PlantCopses(data, o, forest, ridge, turbines, treePrefab);
            EditorUtility.SetDirty(data);
            Undo.RecordObject(t, "Scenic terrain");
            t.terrainData = data;
            var collider = t.GetComponent<TerrainCollider>();
            if (collider != null)
            {
                Undo.RecordObject(collider, "Scenic terrain");
                collider.terrainData = data;
            }
            UseLook(t, look);
            tiles++;
        }
        TerrainHeightSampler.Invalidate();
        if (shared > 0) return $"landmarks unpacked, {tiles} ring tiles ({shared} the existing scenic copies, as they stand)";
        return $"landmarks unpacked, {tiles} ring tiles repainted, {planted} trees in copses where the forest paint was";
    }

    /// <summary>A ring tile's paint from the generator's: its rock and snow kept where they were, outcrops on the steepest
    /// slopes, and the rest the city's grounds by the same noise as round the city. Returns the old forest weights.</summary>
    private static float[,,] RepaintRing(TerrainData data, Vector3 origin, bool mountain, TerrainLayer[] layers)
    {
        int res = data.alphamapResolution;
        TerrainLayer[] before = data.terrainLayers;
        int Find(string word) => Array.FindIndex(before, l => l != null && l.name.IndexOf(word, StringComparison.OrdinalIgnoreCase) >= 0);
        int iForest = Find("Forest"), iRock = Find("Rock"), iSnow = Find("Snow");
        float[,,] old = data.GetAlphamaps(0, 0, res, res);
        var forest = new float[res, res, 1];
        var paint = new float[res, res, 4];
        Vector3 size = data.size;
        for (int y = 0; y < res; y++)
        {
            for (int x = 0; x < res; x++)
            {
                float u = (x + 0.5f) / res, v = (y + 0.5f) / res;
                float wx = origin.x + u * size.x, wz = origin.z + v * size.z;
                float f0 = iForest >= 0 ? old[y, x, iForest] : 0f;
                float r0 = iRock >= 0 ? old[y, x, iRock] : 0f;
                float s0 = iSnow >= 0 ? old[y, x, iSnow] : 0f;
                forest[y, x, 0] = f0;
                float slope = data.GetSteepness(u, v);
                float snow = mountain ? s0 : 0f;
                float outcrop = 0.9f * Step(31f, 41f, slope) * Step(0.5f, 0.66f, Fbm(wx, wz, 55f, 401f));
                float rock = Mathf.Max(r0, outcrop) * (1f - snow);
                Mix(wx, wz, out float fieldW, out float dryW, out _, out _);
                fieldW *= 1f - 0.6f * f0;                   // under the copses the darker ground
                dryW *= (1f - f0) * ToMountain(wz);
                if (mountain) dryW = 0f;
                float rest = Mathf.Max(0f, 1f - rock - snow);
                paint[y, x, 1] = rock;
                paint[y, x, 3] = mountain ? snow : rest * dryW;
                paint[y, x, 2] = rest * (1f - dryW) * fieldW;
                paint[y, x, 0] = rest * (1f - dryW) * (1f - fieldW);
            }
        }
        data.terrainLayers = layers;
        data.SetAlphamaps(0, 0, paint);
        return forest;
    }

    /// <summary>Copses where the generator painted forest: a jittered grid kept in patches, sparser on the turbine
    /// ridge and none under the rotors or on the rock. The mountain's own woods are left as they are.</summary>
    private static int PlantCopses(TerrainData data, Vector3 origin, float[,,] forest, bool ridge, List<Vector2> turbines, GameObject tree)
    {
        if (tree == null) return 0;
        if (data.treePrototypes.Length == 0)
        {
            data.treePrototypes = new[] { new TreePrototype { prefab = tree } };
            data.RefreshPrototypes();
        }
        const float spacing = 11f;
        Vector3 size = data.size;
        int res = forest.GetLength(0);
        var rng = new System.Random(Mathf.RoundToInt(origin.x) * 7919 + Mathf.RoundToInt(origin.z) * 31 + 5);
        var trees = new List<TreeInstance>(data.treeInstances);
        int planted = 0;
        for (float zz = spacing * 0.5f; zz < size.z; zz += spacing)
        {
            for (float xx = spacing * 0.5f; xx < size.x; xx += spacing)
            {
                // every spot draws all its numbers, so a change to the rule moves trees in and out, not the whole wood
                float jx = (float)rng.NextDouble() - 0.5f, jz = (float)rng.NextDouble() - 0.5f;
                double keep = rng.NextDouble(), size01 = rng.NextDouble(), turn = rng.NextDouble(), width01 = rng.NextDouble();
                float u = Mathf.Clamp01((xx + jx * spacing) / size.x), v = Mathf.Clamp01((zz + jz * spacing) / size.z);
                float f0 = forest[Mathf.Min(res - 1, (int)(v * res)), Mathf.Min(res - 1, (int)(u * res)), 0];
                if (f0 < 0.25f) continue;
                float wx = origin.x + u * size.x, wz = origin.z + v * size.z;
                float p = Step(0.25f, 0.6f, f0) * Step(0.4f, 0.56f, Fbm(wx, wz, 150f, 613f)) * (ridge ? 0.4f : 0.92f);
                if (keep >= p || data.GetSteepness(u, v) > 48f) continue;
                bool underRotor = false;
                foreach (Vector2 tb in turbines) underRotor |= (new Vector2(wx, wz) - tb).sqrMagnitude < TreeFreeTurbine * TreeFreeTurbine;
                if (underRotor) continue;
                float s = Mathf.Lerp(0.9f, 1.15f, (float)size01);
                trees.Add(new TreeInstance
                {
                    position = new Vector3(u, 0f, v),
                    prototypeIndex = 0,
                    heightScale = s,
                    widthScale = s * (0.85f + 0.3f * (float)width01),
                    rotation = (float)turn * 2f * Mathf.PI,
                    color = Color.white,
                    lightmapColor = Color.white,
                });
                planted++;
            }
        }
        data.SetTreeInstances(trees.ToArray(), true);
        return planted;
    }

    // ------------------------------------------------------------------ the ground past the ring

    /// <summary>A skirt of ground from the ring's outer edge out <see cref="SkirtReach"/>, its inner edge at the ring's
    /// own heights (tucked under them) falling to the ground's level, so a drone looking out of the city sees land to
    /// the far clip rather than the ring's straight edge against the sky. One mesh, one draw, no collider. A summary.</summary>
    public static string Skirt(Scene scene)
    {
        const string name = "Distant ground";
        foreach (GameObject root in scene.GetRootGameObjects())
        {
            if (root.name == name) UnityEngine.Object.DestroyImmediate(root);
        }
        var terrains = new List<Terrain>();
        foreach (GameObject root in scene.GetRootGameObjects()) terrains.AddRange(root.GetComponentsInChildren<Terrain>(true));
        float EdgeHeight(Vector2 p)
        {
            // just inside the ring, on whichever tile is there
            Vector2 q = p * ((RingEdge - 0.5f) / RingEdge);
            foreach (Terrain t in terrains)
            {
                Vector3 o = t.transform.position, s = t.terrainData.size;
                if (q.x >= o.x && q.x <= o.x + s.x && q.y >= o.z && q.y <= o.z + s.z) return t.SampleHeight(new Vector3(q.x, 0f, q.y)) + o.y;
            }
            return 0f;
        }

        const float step = 25f;
        float[] rows = { -25f, 0f, 60f, 200f, 450f, SkirtReach };
        int per = Mathf.RoundToInt(2f * RingEdge / step), ring = 4 * per;
        var edge = new Vector2[ring];
        for (int k = 0; k < ring; k++)
        {
            int side = k / per;
            float t = (k % per) * step - RingEdge;
            edge[k] = side == 0 ? new Vector2(t, -RingEdge) : side == 1 ? new Vector2(RingEdge, t)
                    : side == 2 ? new Vector2(-t, RingEdge) : new Vector2(-RingEdge, -t);
        }
        var verts = new List<Vector3>(ring * rows.Length);
        var uvs = new List<Vector2>(ring * rows.Length);
        foreach (float d in rows)
        {
            for (int k = 0; k < ring; k++)
            {
                Vector2 p = edge[k] * ((RingEdge + d) / RingEdge);
                float h = EdgeHeight(edge[k]);
                float y = d < 0f ? h - 2f : h * (1f - Step(0f, 700f, d)) - 0.3f;
                verts.Add(new Vector3(p.x, y, p.y));
                uvs.Add(p / 47f);
            }
        }
        var tris = new List<int>(ring * (rows.Length - 1) * 6);
        for (int r = 0; r + 1 < rows.Length; r++)
        {
            for (int k = 0; k < ring; k++)
            {
                int a = r * ring + k, b = r * ring + (k + 1) % ring, c = a + ring, e = b + ring;
                // the edge runs counter-clockwise seen from above and the rows outward: (a, b, c) is clockwise from above
                tris.AddRange(new[] { a, b, c, b, e, c });
            }
        }
        // Rewritten in place over an earlier build's mesh, so its GUID stays put.
        string meshPath = $"{TerrainFolder}/DistantGround.asset";
        var mesh = AssetDatabase.LoadAssetAtPath<Mesh>(meshPath);
        bool fresh = mesh == null;
        if (fresh) mesh = new Mesh();
        mesh.Clear();
        mesh.name = name;
        mesh.indexFormat = UnityEngine.Rendering.IndexFormat.UInt32;
        mesh.SetVertices(verts);
        mesh.SetUVs(0, uvs);
        mesh.SetTriangles(tris, 0);
        mesh.RecalculateNormals();
        mesh.RecalculateBounds();
        if (fresh) AssetDatabase.CreateAsset(mesh, meshPath);
        EditorUtility.SetDirty(mesh);

        Material far = Ground("ScenicFarGround");
        far.SetTexture("_MainTex", Texture($"{Textures}/Terrain/aerial_grass_rock_diff.jpg", true, false));
        far.SetTexture("_BumpMap", Texture($"{Textures}/Terrain/aerial_grass_rock_nor.jpg", false, true));
        far.SetFloat("_Marks", 0f);
        far.DisableKeyword("SCENIC_MARKS");
        far.SetTexture("_Noise", Texture($"{Textures}/Noise.png", false, false));
        far.SetFloat("_Stains", 0.5f);
        far.SetFloat("_StainMetres", 400f);
        far.SetColor("_Color", Color.white);
        EditorUtility.SetDirty(far);

        var go = new GameObject(name);
        SceneManager.MoveGameObjectToScene(go, scene);
        go.AddComponent<MeshFilter>().sharedMesh = mesh;
        MeshRenderer mr = go.AddComponent<MeshRenderer>();
        mr.sharedMaterial = far;
        mr.shadowCastingMode = UnityEngine.Rendering.ShadowCastingMode.Off;
        mr.receiveShadows = false;
        GameObjectUtility.SetStaticEditorFlags(go, StaticEditorFlags.ContributeGI | StaticEditorFlags.OccludeeStatic);
        Undo.RegisterCreatedObjectUndo(go, "Distant ground");
        return $"distant ground out to {RingEdge + SkirtReach:0} m ({verts.Count} vertices)";
    }

    /// <summary>The grounds' mix at a point, shared by the city's tiles and the ring: how much of the grass is field, how
    /// much is dry, how much bare, and the fine noise.</summary>
    private static void Mix(float wx, float wz, out float field, out float dry, out float bare, out float fine)
    {
        float broad = Fbm(wx, wz, 260f, 11f), mid = Fbm(wx, wz, 70f, 53f);
        fine = Fbm(wx, wz, 17f, 97f);
        field = Mathf.Clamp01(0.2f + 0.7f * Step(0.35f, 0.65f, 0.6f * broad + 0.4f * mid));
        dry = 0.85f * Step(0.52f, 0.72f, 0.55f * Fbm(wx, wz, 120f, 211f) + 0.45f * fine);
        bare = 0.5f * Step(0.66f, 0.8f, 0.5f * mid + 0.5f * fine);
    }

    /// <summary>The tile's trees: the woods' edge thinned raggedly, and clumps planted in the clearing.</summary>
    private static void Replant(TerrainData data, Vector3 origin, Rect city, List<Rect> roads, Vector2 spawn, ref int removed, ref int planted)
    {
        TreeInstance[] trees = data.treeInstances;
        if (trees.Length == 0 || data.treePrototypes.Length == 0) return;
        Vector3 size = data.size;
        // Where the woods are, on a 20 m grid; and how far each cell is from open ground.
        const float cell = 20f;
        int n = Mathf.CeilToInt(size.x / cell);
        var wooded = new bool[n, n];
        foreach (TreeInstance t in trees)
        {
            wooded[Mathf.Min(n - 1, (int)(t.position.z * size.z / cell)), Mathf.Min(n - 1, (int)(t.position.x * size.x / cell))] = true;
        }
        var toOpen = new float[n, n];
        for (int z = 0; z < n; z++)
        {
            for (int x = 0; x < n; x++)
            {
                float best = 6f;
                for (int dz = -5; dz <= 5 && wooded[z, x]; dz++)
                {
                    for (int dx = -5; dx <= 5; dx++)
                    {
                        int zz = z + dz, xx = x + dx;
                        if (zz < 0 || xx < 0 || zz >= n || xx >= n || wooded[zz, xx]) continue;
                        best = Mathf.Min(best, Mathf.Sqrt(dx * dx + dz * dz));
                    }
                }
                toOpen[z, x] = wooded[z, x] ? best * cell : 0f;
            }
        }

        var rng = new System.Random(Mathf.RoundToInt(origin.x) * 31 + Mathf.RoundToInt(origin.z));
        var kept = new List<TreeInstance>(trees.Length + 512);
        float minScale = float.MaxValue, maxScale = 0f;
        foreach (TreeInstance t in trees)
        {
            minScale = Mathf.Min(minScale, t.heightScale);
            maxScale = Mathf.Max(maxScale, t.heightScale);
            float wx = origin.x + t.position.x * size.x, wz = origin.z + t.position.z * size.z;
            float d = toOpen[Mathf.Min(n - 1, (int)(t.position.z * size.z / cell)), Mathf.Min(n - 1, (int)(t.position.x * size.x / cell))];
            // Near the edge, a tree stays where the noise is high: bays and headlands instead of a ruled line.
            float keep = Mathf.Clamp01(d / 90f) + Step(0.4f, 0.62f, Fbm(wx, wz, 95f, 301f));
            if (keep < 1f && rng.NextDouble() > keep * keep)
            {
                removed++;
                continue;
            }
            kept.Add(t);
        }

        // Clumps in the clearing: where the noise is high, clear of the city, the road in, the spawn and the sand.
        TreeInstance model = trees[0];
        int res = data.alphamapResolution;
        float[,,] paint = data.GetAlphamaps(0, 0, res, res);
        int tries = Mathf.RoundToInt(size.x * size.z / 140f);
        for (int k = 0; k < tries; k++)
        {
            float u = (float)rng.NextDouble(), v = (float)rng.NextDouble();
            int cx = Mathf.Min(n - 1, (int)(u * size.x / cell)), cz = Mathf.Min(n - 1, (int)(v * size.z / cell));
            if (wooded[cz, cx]) continue;
            float wx = origin.x + u * size.x, wz = origin.z + v * size.z;
            var p = new Vector2(wx, wz);
            if (Outside(city, p) < TreeFreeCity || (p - spawn).magnitude < TreeFreeSpawn) continue;
            bool onRoad = false;
            foreach (Rect road in roads) onRoad |= Outside(road, p) < TreeFreeRoad;
            if (onRoad) continue;
            if (paint[Mathf.Min(res - 1, (int)(v * res)), Mathf.Min(res - 1, (int)(u * res)), 1] > 0.05f) continue;
            float clump = Fbm(wx, wz, 85f, 517f) * 0.7f + Fbm(wx, wz, 23f, 733f) * 0.3f;
            if (clump < 0.53f || rng.NextDouble() > (clump - 0.53f) * 9f) continue;
            TreeInstance t = model;
            t.position = new Vector3(u, 0f, v);
            float s = Mathf.Lerp(minScale, maxScale, (float)rng.NextDouble());
            t.heightScale = s;
            t.widthScale = s * Mathf.Lerp(0.9f, 1.1f, (float)rng.NextDouble());
            t.rotation = (float)rng.NextDouble() * Mathf.PI * 2f;
            kept.Add(t);
            planted++;
        }
        data.SetTreeInstances(kept.ToArray(), true);
    }
}
