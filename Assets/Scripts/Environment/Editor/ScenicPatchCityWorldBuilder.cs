using System;
using System.IO;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;
using UnityEngine.Rendering;
using UnityEngine.SceneManagement;

/// <summary>
/// <c>Tools/Swarm/Build scenic patch city world</c>: makes <c>ScenicPatchCityWorld.unity</c>, BlockShapedPatchCityWorld
/// with a more realistic look -- the same layout file, the same buildings where the swarm and the analysis are concerned
/// (their colliders, the streets, the dressing's rules, the goals drawn at Play), and the same swarm, rig, input, screens
/// and recorder.
///
/// <para><b>It is BlockShapedPatchCityWorld copied file and all</b>, for <see cref="PracticeBlockWorldBuilder"/>'s
/// reason: that scene's swarm tuning (S6, S8, S10, S11) was written into its SwarmManager by <c>swarmtune apply</c> and
/// is set by no builder. So <b>rebuild after changing BlockShapedPatchCityWorld</b>, or the two drift apart; a rebuild
/// discards hand edits to the scenic scene. Only what is listed here differs from the source scene.</para>
///
/// <para><b>Buildings</b>: <see cref="ScenicCity"/> on the city's root draws every block-shaped building as
/// <see cref="ScenicBuilding"/>'s, in photographed wall materials (<see cref="ScenicFacades"/>); the city is baked
/// again from the same layout file, so its colliders are PrismBuilding's as before.</para>
///
/// <para><b>Ground</b>: <see cref="ScenicGroundBuilder"/>: the roads, footpaths and lawns in <c>Swarm/Scenic Ground</c>;
/// the four city Terrain tiles and the landmarks' twelve replaced by this scene's own copies, repainted and replanted
/// (heights untouched) and drawn in <c>Swarm/Scenic Terrain</c>, the Landmarks prefab instance unpacked so nothing
/// reaches the prefab; and ground on past the ring to the horizon.</para>
///
/// <para><b>Props</b>: <see cref="ScenicPropsBuilder"/>: the same street furniture in the same places, its materials
/// swapped by the dressing (<see cref="CityDressing.materialSwaps"/>) for matte, weathered ones, and the leaves lit
/// through.</para>
///
/// <para><b>Light</b>: a photographed sky (Poly Haven's kloofendal_48d_partly_cloudy_puresky, CC0) with the sun turned
/// to stand where the scene's light comes from and raised to the panorama's 48 degrees, and <see cref="CityLighting"/>
/// on the city's root: sun shadows, the sky the buildings hide and haze, drawn from maps baked once at Play.</para>
///
/// <para>Batch: <c>-executeMethod ScenicPatchCityWorldBuilder.Run</c>.</para>
/// </summary>
public static class ScenicPatchCityWorldBuilder
{
    public const string ScenePath = "Assets/Scenes/ScenicPatchCityWorld.unity";
    private const string SourceScenePath = BlockShapedPatchCityWorldBuilder.ScenePath;
    public const string Folder = "Assets/Scenic";

    // The panorama (Assets/Scenic/Sky): where its sun stands, measured off the HDR (probe: sky_analyse.py).
    private const string SkyTexture = Folder + "/Sky/CitySky_kloofendal_48d.jpg";
    private const string SkyMaterial = Folder + "/Sky/CitySky.mat";
    private const float SkySunU = 0.5949f;
    private const float SkySunElevation = 47.87f;

    // Linear. A soft day: the sky gives as much light as the sun, so what stands in shade is lit nearly as well as
    // what stands in sun. The user asked for that (2026-11-11): a pedestrian must not be much harder to pick out on
    // one side of a street than on the other. The panorama's own sun is about three times its sky.
    private static readonly Color SunColor = new Color(1.00f, 0.95f, 0.86f);
    private const float SunIntensity = 0.85f;
    private static readonly Color AmbientSky = new Color(0.54f, 0.60f, 0.71f);
    private static readonly Color AmbientHorizon = new Color(0.44f, 0.46f, 0.50f);
    private static readonly Color AmbientGround = new Color(0.27f, 0.26f, 0.24f);

    /// <summary>How far the buildings rise above their 50 m colliders, as a share of the design
    /// (<see cref="ScenicFacades.extraHeight"/>). At 1 the tallest stands 88 m and, of the street points that see a
    /// landmark over the plain city's roofs, about 14% lose it; at 0.5 (69 m) about 8%, at 0.3 about 5% (probe:
    /// landmarks.py, which takes every footprint to its full height, so these overstate it). The landmarks were made
    /// taller so they would show over the roofs, hence half.</summary>
    private const float ExtraHeight = 0.5f;

    // The walls (probe: prep_assets.py): name, metres to a repeat, paintable, mean colour (linear).
    private static readonly (string name, float metres, bool paintable, Color mean)[] Walls =
    {
        ("red_brick_03", 1.0f, false, new Color(0.155f, 0.100f, 0.083f)),
        ("brick_wall_006", 3.0f, false, new Color(0.434f, 0.193f, 0.101f)),
        ("brown_brick_02", 1.0f, false, new Color(0.494f, 0.303f, 0.157f)),
        ("painted_brick", 1.8f, true, new Color(0.604f, 0.604f, 0.604f)),
        ("concrete_wall_008", 2.71f, false, new Color(0.266f, 0.237f, 0.162f)),
        ("concrete_wall_004", 2.0f, false, new Color(0.181f, 0.171f, 0.126f)),
        ("white_stucco", 2.0f, true, new Color(0.604f, 0.604f, 0.604f)),
        ("painted_plaster_wall", 2.0f, true, new Color(0.604f, 0.604f, 0.604f)),
        ("stone_tile_wall", 2.0f, false, new Color(0.406f, 0.381f, 0.357f)),
        ("granite_tile_03", 1.8f, false, new Color(0.132f, 0.107f, 0.087f)),
        ("marble_01", 1.5f, false, new Color(0.443f, 0.338f, 0.194f)),
        ("exterior_wall_cladding_03", 1.96f, true, new Color(0.603f, 0.603f, 0.603f)),
        ("concrete_tile_facade", 2.09f, false, new Color(0.038f, 0.040f, 0.046f)),
        ("bitumen", 20.0f, false, new Color(0.043f, 0.038f, 0.035f)),
    };
    private const string FacadesAsset = Folder + "/ScenicFacades.asset";
    private const string FacadeMaterial = Folder + "/ScenicFacade.mat";
    private const string Textures = Folder + "/Textures";

    [MenuItem("Tools/Swarm/Build scenic patch city world")]
    private static void BuildFromMenu()
    {
        if (!EditorSceneManager.SaveCurrentModifiedScenesIfUserWantsTo()) return;
        if (File.Exists(ScenePath) &&
            !EditorUtility.DisplayDialog("Build scenic patch city world",
                $"Rebuild {ScenePath} from {SourceScenePath}? This replaces the scenic scene, including anything " +
                "changed in it by hand.", "Rebuild", "Cancel"))
        {
            return;
        }
        Build();
    }

    [MenuItem("Tools/Swarm/Build scenic patch city world", true)]
    private static bool CanBuild() => !EditorApplication.isPlayingOrWillChangePlaymode;

    /// <summary>Batchmode: builds the scene and exits 0 / 1.</summary>
    public static void Run() => EditorApplication.Exit(Build() ? 0 : 1);

    /// <summary>Rebuilds the scenic scene from BlockShapedPatchCityWorld, saves it and leaves it open. On failure it
    /// logs why and puts back whatever scenic scene was there before. The source scene is never opened.</summary>
    public static bool Build()
    {
        SceneSetup[] previousSetup = EditorSceneManager.GetSceneManagerSetup();
        byte[] previousScene = File.Exists(ScenePath) ? File.ReadAllBytes(ScenePath) : null;

        // Nothing may hold the scenic scene open while its file is replaced underneath it.
        EditorSceneManager.NewScene(NewSceneSetup.EmptyScene, NewSceneMode.Single);
        File.Copy(SourceScenePath, ScenePath, true);
        AssetDatabase.ImportAsset(ScenePath, ImportAssetOptions.ForceUpdate);
        Scene scene = EditorSceneManager.OpenScene(ScenePath, OpenSceneMode.Single);

        string summary, error = null;
        try
        {
            summary = Populate(scene, out error);
            // The copy carries the source scene's occlusion data, which belongs to that scene's own objects.
            if (summary != null)
            {
                summary += FlightOcclusionBaker.BakeForOpenScene(out string occlusion)
                    ? $"; occlusion: {occlusion}"
                    : $"; occlusion NOT baked ({occlusion}): run Tools/Swarm/Bake flight occlusion";
            }
            if (summary != null && !EditorSceneManager.SaveScene(scene))
            {
                summary = null;
                error = $"could not save {ScenePath}.";
            }
        }
        catch (Exception e)
        {
            summary = null;
            error = e.ToString();
        }

        if (summary == null)
        {
            Debug.LogError("ScenicPatchCityWorldBuilder: " + error + " The scenic scene was left as it was.");
            EditorSceneManager.NewScene(NewSceneSetup.EmptyScene, NewSceneMode.Single);
            if (previousScene != null)
            {
                File.WriteAllBytes(ScenePath, previousScene);
                AssetDatabase.ImportAsset(ScenePath, ImportAssetOptions.ForceUpdate);
            }
            else
            {
                AssetDatabase.DeleteAsset(ScenePath);
            }
            if (previousSetup.Length > 0 && Array.TrueForAll(previousSetup, s => !string.IsNullOrEmpty(s.path)))
            {
                EditorSceneManager.RestoreSceneManagerSetup(previousSetup);
            }
            return false;
        }

        AssetDatabase.SaveAssets();
        Debug.Log($"ScenicPatchCityWorldBuilder: built {ScenePath} from {SourceScenePath}: {summary}");
        return true;
    }

    /// <summary>Turns a fresh copy of BlockShapedPatchCityWorld into the scenic scene. A one-line account of what it did,
    /// or null with <paramref name="error"/> set.</summary>
    private static string Populate(Scene scene, out string error)
    {
        ProceduralCityBuilder builder = UnityEngine.Object.FindFirstObjectByType<ProceduralCityBuilder>(FindObjectsInactive.Include);
        Light sun = RenderSettings.sun;
        if (builder == null || sun == null)
        {
            error = $"{SourceScenePath} has no {(builder == null ? "ProceduralCityBuilder" : "sun (RenderSettings.sun)")}.";
            return null;
        }
        // The buildings first: the city is baked again from the same layout with the scenic facades on it.
        string dressed = Dress(builder, scene);
        if (!builder.Bake(out error)) return null;
        string rest = Finish(builder, scene, sun, "CityTile", false);
        error = null;
        return $"baked {builder.Layout.layout.id} ({builder.Layout.boxes.Length} buildings); {dressed}; {rest}";
    }

    /// <summary>The scenic look's first half, before the city is baked: <see cref="ScenicCity"/> and its facades on the
    /// builder, the dressing and style as BlockShapedPatchCityWorld has them, the ground's and the props' materials.
    /// Shared with <see cref="PracticeBlockWorldBuilder"/>'s scenic practice scene. A summary.</summary>
    internal static string Dress(ProceduralCityBuilder builder, Scene scene)
    {
        ScenicCity scenic = builder.GetComponent<ScenicCity>();
        if (scenic == null) scenic = Undo.AddComponent<ScenicCity>(builder.gameObject);
        scenic.facades = Facades();
        EditorUtility.SetDirty(scenic);
        BlockShapedCityWorldBuilder.EnsureDressing(builder);
        BlockShapedCityWorldBuilder.EnsureStyle(builder);
        string ground = ScenicGroundBuilder.Surfaces(builder, scene);
        string props = ScenicPropsBuilder.Props(builder, scene);
        return $"{scenic.facades.walls.Length} wall materials; {ground}; {props}";
    }

    /// <summary>The scenic look's second half, once the city is baked: the land round it in this scene's own terrain
    /// copies (<paramref name="tilePrefix"/>), the landmarks' ring (the existing scenic copies when
    /// <paramref name="reuseRing"/>), the distant ground, the light and the screens. A summary.</summary>
    internal static string Finish(ProceduralCityBuilder builder, Scene scene, Light sun, string tilePrefix, bool reuseRing)
    {
        string land = ScenicGroundBuilder.Surroundings(builder, scene, tilePrefix);
        string landmarks = ScenicGroundBuilder.Landmarks(scene, reuseRing);
        string skirt = ScenicGroundBuilder.Skirt(scene);
        string light = Light(builder, sun);
        string screens = Screens();
        EditorSceneManager.MarkSceneDirty(scene);
        return $"{screens}; {land}; {landmarks}; {skirt}; {light}";
    }

    /// <summary>For the tools that bake or redress a city in place (the Block-shaped city window): on a scenic city,
    /// puts its ground and prop materials back on the builder and its dressing before they run, so a dressing whose
    /// rules were reset, or a field cleared by hand, cannot take the look away. Nothing on any other city.</summary>
    public static void KeepLook(ProceduralCityBuilder builder)
    {
        if (builder == null || builder.GetComponent<ScenicCity>() == null) return;
        Scene scene = builder.gameObject.scene;
        ScenicGroundBuilder.Surfaces(builder, scene);
        ScenicPropsBuilder.Props(builder, scene);
    }

    // ------------------------------------------------------------------ the pilot's screens

    /// <summary>The feed screens unlit (<c>Swarm/Feed Screen</c>): this scene's brighter sky would otherwise lie over
    /// every feed as a veil, a different one on each screen of the ring.</summary>
    private static string Screens()
    {
        const string path = Folder + "/Materials/FeedScreen.mat";
        var material = AssetDatabase.LoadAssetAtPath<Material>(path);
        if (material == null)
        {
            if (!AssetDatabase.IsValidFolder(Folder + "/Materials")) AssetDatabase.CreateFolder(Folder, "Materials");
            material = new Material(Shader.Find("Swarm/Feed Screen")) { name = "FeedScreen" };
            AssetDatabase.CreateAsset(material, path);
        }
        int n = 0;
        foreach (ScreenSpawn spawn in UnityEngine.Object.FindObjectsByType<ScreenSpawn>(FindObjectsInactive.Include, FindObjectsSortMode.None))
        {
            Undo.RecordObject(spawn, "Scenic feed screens");
            spawn.feedScreenMaterial = material;
            EditorUtility.SetDirty(spawn);
            n++;
        }
        return n > 0 ? "feed screens unlit" : "no ScreenSpawn to give the unlit feed screen";
    }

    // ------------------------------------------------------------------ facades

    /// <summary>The facades asset and its material, made if missing, with every wall's textures imported alike (one
    /// size, one format, one mip count: they are stacked into texture arrays).</summary>
    private static ScenicFacades Facades()
    {
        var walls = new ScenicFacades.Wall[Walls.Length];
        for (int i = 0; i < Walls.Length; i++)
        {
            (string name, float metres, bool paintable, Color mean) = Walls[i];
            walls[i] = new ScenicFacades.Wall
            {
                name = name, metres = metres, paintable = paintable, mean = mean,
                albedo = Import($"{Textures}/Walls/{name}_albedo.jpg", true, TextureImporterFormat.DXT1, 8),
                normal = Import($"{Textures}/Walls/{name}_nrm.jpg", false, TextureImporterFormat.BC7, 8),
            };
        }
        var material = AssetDatabase.LoadAssetAtPath<Material>(FacadeMaterial);
        if (material == null)
        {
            material = new Material(Shader.Find("Swarm/Scenic Facade")) { name = Path.GetFileNameWithoutExtension(FacadeMaterial) };
            AssetDatabase.CreateAsset(material, FacadeMaterial);
        }
        // Rooms seen from the street by day are dim: about a seventh of what stands in the open, the shops less, so a
        // pedestrian is seen against dark glass, as against the plain prisms' shopfronts.
        material.SetFloat("_InteriorExposure", 0.16f);
        material.SetFloat("_ShopExposure", 0.08f);
        material.SetFloat("_ReflectionBoost", 1f);
        material.SetFloat("_Weathering", 1f);
        material.SetFloat("_DetailPixels", 7f);
        EditorUtility.SetDirty(material);
        var facades = AssetDatabase.LoadAssetAtPath<ScenicFacades>(FacadesAsset);
        if (facades == null)
        {
            facades = ScriptableObject.CreateInstance<ScenicFacades>();
            AssetDatabase.CreateAsset(facades, FacadesAsset);
        }
        facades.walls = walls;
        facades.extraHeight = ExtraHeight;
        facades.material = material;
        facades.noise = Import($"{Textures}/Noise.png", false, TextureImporterFormat.RGBA32, 1);
        EditorUtility.SetDirty(facades);
        if (!facades.Apply()) throw new InvalidOperationException("the wall textures could not be stacked into arrays (see the console).");
        return facades;
    }

    /// <summary>The texture at <paramref name="path"/>, imported repeating and mipmapped in <paramref name="format"/>.</summary>
    private static Texture2D Import(string path, bool srgb, TextureImporterFormat format, int aniso, bool normalMap = false,
                                    int maxSize = 1024)
    {
        var importer = AssetImporter.GetAtPath(path) as TextureImporter;
        if (importer == null) throw new InvalidOperationException($"no texture at {path}.");
        TextureImporterType type = normalMap ? TextureImporterType.NormalMap : TextureImporterType.Default;
        TextureImporterPlatformSettings platform = importer.GetPlatformTextureSettings("Standalone");
        bool formatSet = normalMap || (platform.overridden && platform.format == format && platform.maxTextureSize == maxSize);
        if (importer.textureType != type || importer.sRGBTexture != srgb || !importer.mipmapEnabled ||
            importer.wrapMode != TextureWrapMode.Repeat || importer.anisoLevel != aniso || importer.maxTextureSize != maxSize ||
            importer.filterMode != FilterMode.Trilinear || !formatSet)
        {
            importer.textureType = type;
            importer.sRGBTexture = srgb;
            importer.alphaSource = format == TextureImporterFormat.RGBA32 ? TextureImporterAlphaSource.FromInput : TextureImporterAlphaSource.None;
            importer.alphaIsTransparency = false;
            importer.mipmapEnabled = true;
            importer.wrapMode = TextureWrapMode.Repeat;
            importer.filterMode = FilterMode.Trilinear;
            importer.anisoLevel = aniso;
            importer.maxTextureSize = maxSize;
            importer.npotScale = TextureImporterNPOTScale.None;
            if (!normalMap)
            {
                platform.overridden = true;
                platform.format = format;
                platform.maxTextureSize = maxSize;
                importer.SetPlatformTextureSettings(platform);
            }
            importer.SaveAndReimport();
        }
        var texture = AssetDatabase.LoadAssetAtPath<Texture2D>(path);
        if (texture == null) throw new InvalidOperationException($"{path} did not import as a texture.");
        return texture;
    }

    // ------------------------------------------------------------------ light

    private static string Light(ProceduralCityBuilder builder, Light sun)
    {
        // The sun keeps the bearing the scene had (the landmarks were laid out for it: the mountain south, lit from
        // the north-west) and takes the panorama's height.
        Vector3 toSun = -sun.transform.forward;
        float bearing = Mathf.Atan2(toSun.x, toSun.z) * Mathf.Rad2Deg;
        Undo.RecordObject(sun.transform, "Scenic sun");
        sun.transform.rotation = Quaternion.Euler(SkySunElevation, bearing + 180f, 0f);
        Undo.RecordObject(sun, "Scenic sun");
        sun.color = SunColor.gamma;
        sun.intensity = SunIntensity;
        sun.shadows = LightShadows.None;

        Material sky = SkyMaterialAsset(bearing);
        RenderSettings.skybox = sky;
        RenderSettings.sun = sun;
        RenderSettings.fog = false;
        // The editor's view of the three ambient colours; at Play CityLighting sets the probe itself.
        RenderSettings.ambientMode = AmbientMode.Trilight;
        RenderSettings.ambientSkyColor = AmbientSky.gamma;
        RenderSettings.ambientEquatorColor = AmbientHorizon.gamma;
        RenderSettings.ambientGroundColor = AmbientGround.gamma;
        RenderSettings.ambientIntensity = 1f;
        RenderSettings.defaultReflectionMode = DefaultReflectionMode.Skybox;
        // The copy points at another scene's baked ambient and reflection data.
        Lightmapping.lightingDataAsset = null;

        CityLighting lighting = builder.GetComponent<CityLighting>();
        if (lighting == null) lighting = Undo.AddComponent<CityLighting>(builder.gameObject);
        lighting.sun = sun;
        lighting.skyColor = AmbientSky;
        lighting.horizonColor = AmbientHorizon;
        lighting.groundColor = AmbientGround;
        lighting.casterShader = Shader.Find("Hidden/Swarm/City Caster");
        lighting.bakeShader = Shader.Find("Hidden/Swarm/City Bake");
        lighting.compositeShader = Shader.Find("Hidden/Swarm/City Lighting Composite");
        EditorUtility.SetDirty(lighting);
        return $"sun at {SkySunElevation:F1} deg up, bearing {bearing:F0} deg; sky {Path.GetFileName(SkyTexture)}; CityLighting on {builder.name}";
    }

    /// <summary>The sky material, made if missing, turned so the panorama's sun stands at <paramref name="bearing"/>
    /// (degrees from north towards east).</summary>
    private static Material SkyMaterialAsset(float bearing)
    {
        var importer = (TextureImporter)AssetImporter.GetAtPath(SkyTexture);
        if (importer == null) throw new InvalidOperationException($"no sky panorama at {SkyTexture}.");
        if (importer.maxTextureSize != 8192 || importer.wrapModeV != TextureWrapMode.Clamp || importer.mipmapEnabled == false)
        {
            importer.textureType = TextureImporterType.Default;
            importer.sRGBTexture = true;
            importer.npotScale = TextureImporterNPOTScale.None;
            importer.maxTextureSize = 8192;
            importer.mipmapEnabled = true;
            importer.wrapModeU = TextureWrapMode.Repeat;
            importer.wrapModeV = TextureWrapMode.Clamp;
            importer.filterMode = FilterMode.Trilinear;
            importer.anisoLevel = 1;
            importer.textureCompression = TextureImporterCompression.CompressedHQ;
            importer.SaveAndReimport();
        }
        var material = AssetDatabase.LoadAssetAtPath<Material>(SkyMaterial);
        if (material == null)
        {
            material = new Material(Shader.Find("Swarm/City Sky")) { name = Path.GetFileNameWithoutExtension(SkyMaterial) };
            AssetDatabase.CreateAsset(material, SkyMaterial);
        }
        material.mainTexture = AssetDatabase.LoadAssetAtPath<Texture2D>(SkyTexture);
        material.SetFloat("_Rotation", Mathf.Repeat(SkySunU - bearing / 360f, 1f));
        material.SetFloat("_Exposure", 1f);
        // Below the horizon, beyond the far clip: the land there seen through 2 km of haze, so the ground meets it
        // with no line.
        material.SetColor("_GroundColor", new Color(0.47f, 0.52f, 0.47f));
        EditorUtility.SetDirty(material);
        return material;
    }
}
