using System;
using System.IO;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;
using UnityEngine.SceneManagement;

/// <summary>
/// <c>Tools/Swarm/Build block-shaped city world</c>: makes <c>BlockShapedCityWorld.unity</c>, S4 Split and Surround's
/// variant B to fly (docs/split_study/README.md): small Voronoi blocks, each built over to the back of a footpath round
/// its kerb, one building per block. The layout is <c>citysim.blockcity</c>'s (<c>layouts/BlockShaped__&lt;id&gt;.json</c>,
/// written by <c>blockcity one --scene</c> or by <see cref="BlockShapedCityWindow"/>); by default S4's p 40 with 18 m
/// streets, seed 1 -- the streets of BlockCityWorld, which flies variant R on them.
///
/// <para>The buildings are simple prisms (<see cref="PrismBuilding"/>), not pack models: plain walls with a grid of
/// windows, ten colours dealt so that neighbours differ, three window patterns, shopfronts on the ground floor. They are
/// there to let a pilot tell the buildings apart and match one across two feeds. The facade textures and materials are
/// made here if missing, under <see cref="FacadeFolder"/> (<c>Tools/Swarm/Regenerate block facades</c> makes them
/// again after a change to the palette or the patterns).</para>
///
/// <para>The rest of the scene is BlockCityWorld's (<see cref="BlockCityWorldBuilder.ConfigureS4"/>): DiamondCityWorld's
/// rig, the landmarks, the street dressing on VoronoiCityWorld's road surfaces (here the footpath runs out to the
/// facades, and its props stand on it), S4's cohesion cutoff of 30 and the ceiling at 45 m under the 50 m roofs. The
/// layout has no goals: at every Play the builder draws three from its buildings (<see cref="BlockGoals"/>), each a
/// building with pedestrians on the footpaths round its neighbours, so nothing about them is baked. Batch:
/// <c>-executeMethod BlockShapedCityWorldBuilder.Run</c> builds the scene,
/// <c>.Rebake</c> bakes another layout into it (CITY_LAYOUT names either's layout).</para>
/// </summary>
public static class BlockShapedCityWorldBuilder
{
    public const string ScenePath = "Assets/Scenes/BlockShapedCityWorld.unity";
    /// <summary>blockcity's scene layouts (blockcity.SCENE_PREFIX).</summary>
    public const string LayoutPattern = "BlockShaped__*.json";
    /// <summary>What a build bakes unless CITY_LAYOUT says otherwise: S4's p 40 s 18, seed 1 (S4's own file, copied).</summary>
    public const string S4Layout = "BlockShaped__bbB_p40_s18_seed1.json";
    public const string FacadeFolder = "Assets/Materials/BlockFacades";

    // ------------------------------------------------------------------ the facades
    // Texture space is metres on the wall: a pattern is one bay across (Style.bayWidth) and one floor up
    // (Style.floorHeight), the ground floor one bay across and Style.groundFloorHeight up. Alpha is smoothness (the
    // Standard shader reads it from the albedo's alpha): glass shines, walls do not.

    private const float PxPerMetre = 256f / 3f;
    private static readonly (string name, Color colour)[] Palette =
    {
        ("Brick", new Color(0.66f, 0.32f, 0.26f)),
        ("Sand", new Color(0.86f, 0.74f, 0.50f)),
        ("Sage", new Color(0.55f, 0.68f, 0.50f)),
        ("Slate", new Color(0.40f, 0.52f, 0.70f)),
        ("Cream", new Color(0.92f, 0.90f, 0.84f)),
        ("Terracotta", new Color(0.85f, 0.50f, 0.30f)),
        ("Plum", new Color(0.56f, 0.42f, 0.62f)),
        ("Graphite", new Color(0.36f, 0.37f, 0.40f)),
        ("Teal", new Color(0.28f, 0.58f, 0.58f)),
        ("Rose", new Color(0.88f, 0.62f, 0.62f)),
    };
    private static readonly string[] Patterns = { "Punched", "Ribbon", "Tall" };
    private static readonly Color Frame = new Color(0.30f, 0.31f, 0.33f, 0.35f);
    private static readonly Color Sill = new Color(0.82f, 0.81f, 0.78f, 0.25f);
    private static readonly Color Stone = new Color(0.55f, 0.53f, 0.50f, 0.15f);

    [MenuItem("Tools/Swarm/Build block-shaped city world")]
    private static void BuildFromMenu()
    {
        if (!EditorSceneManager.SaveCurrentModifiedScenesIfUserWantsTo()) return;
        if (File.Exists(ScenePath) &&
            !EditorUtility.DisplayDialog("Build block-shaped city world",
                $"Rebuild {ScenePath} from DiamondCityWorld? This replaces the scene, including hand edits.",
                "Rebuild", "Cancel"))
        {
            return;
        }
        string layout = DefaultLayout();
        if (!ProceduralCityWorldBuilder.Build(ScenePath, layout, true, out string summary, Configure))
        {
            Debug.LogError("BlockShapedCityWorldBuilder: " + summary + NoLayoutHint(layout));
        }
    }

    [MenuItem("Tools/Swarm/Build block-shaped city world", true)]
    private static bool CanBuild() => !EditorApplication.isPlayingOrWillChangePlaymode;

    [MenuItem("Tools/Swarm/Regenerate block facades")]
    private static void RegenerateFromMenu()
    {
        if (!EditorUtility.DisplayDialog("Regenerate block facades",
                $"Paint the facade textures under {FacadeFolder} again from the palette and patterns in " +
                "BlockShapedCityWorldBuilder? Hand edits to those textures are lost; the materials are kept.",
                "Regenerate", "Cancel"))
        {
            return;
        }
        MakeFacades(true);
        Debug.Log($"BlockShapedCityWorldBuilder: painted the facades under {FacadeFolder} again.");
    }

    /// <summary>Batchmode: builds the scene (CITY_LAYOUT, else the first BlockShaped__ layout) and exits 0 / 1.</summary>
    public static void Run()
    {
        string layout = Environment.GetEnvironmentVariable(ProceduralCityBuilder.LayoutEnv);
        if (string.IsNullOrEmpty(layout)) layout = DefaultLayout();
        bool ok = ProceduralCityWorldBuilder.Build(ScenePath, layout, true, out string summary, Configure);
        if (!ok) Debug.LogError("BlockShapedCityWorldBuilder: " + summary + NoLayoutHint(layout));
        EditorApplication.Exit(ok ? 0 : 1);
    }

    /// <summary>Batchmode: bakes CITY_LAYOUT (else the first BlockShaped__ layout) into the scene as it stands, keeping
    /// the scene's own settings -- the window's Bake from the command line. Saves and exits 0 / 1.</summary>
    public static void Rebake()
    {
        string layout = Environment.GetEnvironmentVariable(ProceduralCityBuilder.LayoutEnv);
        if (string.IsNullOrEmpty(layout)) layout = DefaultLayout();
        bool ok = false;
        string error = "no layout." + NoLayoutHint(layout);
        if (!string.IsNullOrEmpty(layout))
        {
            Scene scene = EditorSceneManager.OpenScene(ScenePath, OpenSceneMode.Single);
            var builder = UnityEngine.Object.FindFirstObjectByType<ProceduralCityBuilder>(FindObjectsInactive.Include);
            if (builder == null)
            {
                error = $"{ScenePath} has no ProceduralCityBuilder.";
            }
            else
            {
                builder.layoutPath = layout;
                EnsureDressing(builder);
                EnsureStyle(builder);
                ok = builder.Bake(out error);
                if (ok && !EditorSceneManager.SaveScene(scene))
                {
                    ok = false;
                    error = $"could not save {ScenePath}.";
                }
            }
        }
        if (ok) Debug.Log($"BlockShapedCityWorldBuilder: baked {layout} into {ScenePath}.");
        else Debug.LogError("BlockShapedCityWorldBuilder: " + error);
        EditorApplication.Exit(ok ? 0 : 1);
    }

    /// <summary><see cref="S4Layout"/> if it is there, else the first BlockShaped__ layout (sorted), else "".</summary>
    private static string DefaultLayout() =>
        File.Exists(Path.Combine(Application.dataPath, ProceduralCityBuilder.LayoutFolder, S4Layout))
            ? S4Layout
            : ProceduralCityWorldBuilder.DefaultLayout(LayoutPattern);

    /// <summary>BlockCityWorld's S4 configuration, the dressing's short-street rules, then the prisms' materials.</summary>
    private static void Configure(ProceduralCityBuilder builder)
    {
        BlockCityWorldBuilder.ConfigureS4(builder);
        CityDressing d = builder.GetComponent<CityDressing>();
        if (d != null) ShortStreetRules(d.rules);
        EnsureStyle(builder);
    }

    /// <summary>The builder's dressing (<see cref="VoronoiCityWorldBuilder.EnsureDressing"/>), with
    /// <see cref="ShortStreetRules"/> whenever its rules are new or were reset to the defaults (a RulesVersion bump).</summary>
    public static CityDressing EnsureDressing(ProceduralCityBuilder builder)
    {
        CityDressing existing = builder.GetComponent<CityDressing>();
        bool fresh = existing == null || !existing.RulesCurrent;
        CityDressing d = VoronoiCityWorldBuilder.EnsureDressing(builder);
        if (fresh) ShortStreetRules(d.rules);
        return d;
    }

    /// <summary>The dressing's rules for S4's short streets. The defaults are sized for VoronoiCityWorld, ~70 m of plain
    /// road between a street's crossings; at p 40 s 18 the setbacks and crossings take ~17 m of a ~23 m street, and
    /// the defaults put 2 benches and no bus stop in the whole city. More of the street ends and sides draw furniture
    /// here, and shorter streets qualify; every prop is still checked against the others and the facades.</summary>
    private static void ShortStreetRules(CityDressing.Rules r)
    {
        r.benchShare = 0.3f;
        r.benchMinStreet = 8f;
        r.busStopShare = 0.12f;
        r.busStopMinStreet = 18f;
        r.cornerShare = 0.15f;
        r.hydrantShare = 0.1f;
    }

    /// <summary>Gives the builder its prism materials (<see cref="ProceduralCityBuilder.prismStyle"/>), making the
    /// facades if missing; a slot already filled by hand is kept.</summary>
    public static void EnsureStyle(ProceduralCityBuilder builder)
    {
        MakeFacades(false);
        Undo.RecordObject(builder, "Block facades");
        PrismBuilding.Style style = builder.prismStyle ?? (builder.prismStyle = new PrismBuilding.Style());
        int n = Palette.Length * Patterns.Length;
        if (style.facades == null || style.facades.Length != n || Array.Exists(style.facades, m => m == null))
        {
            style.facades = new Material[n];
            style.patterns = Patterns.Length;
            for (int c = 0; c < Palette.Length; c++)
            {
                for (int p = 0; p < Patterns.Length; p++)
                {
                    style.facades[c * Patterns.Length + p] = AssetDatabase.LoadAssetAtPath<Material>(FacadePath(c, p, ".mat"));
                }
            }
        }
        if (style.groundFloor == null) style.groundFloor = AssetDatabase.LoadAssetAtPath<Material>($"{FacadeFolder}/GroundFloor.mat");
        if (style.roof == null) style.roof = AssetDatabase.LoadAssetAtPath<Material>($"{FacadeFolder}/Roof.mat");
        EditorUtility.SetDirty(builder);
    }

    private static string FacadePath(int colour, int pattern, string ext) =>
        $"{FacadeFolder}/Facade_{Palette[colour].name}_{Patterns[pattern]}{ext}";

    /// <summary>Every texture and material under <see cref="FacadeFolder"/>, each made only if missing (textures painted
    /// again when <paramref name="repaint"/>).</summary>
    private static void MakeFacades(bool repaint)
    {
        if (!AssetDatabase.IsValidFolder(FacadeFolder))
        {
            AssetDatabase.CreateFolder(Path.GetDirectoryName(FacadeFolder).Replace('\\', '/'), Path.GetFileName(FacadeFolder));
        }
        var style = new PrismBuilding.Style();
        int w = Mathf.RoundToInt(style.bayWidth * PxPerMetre), h = Mathf.RoundToInt(style.floorHeight * PxPerMetre);
        for (int c = 0; c < Palette.Length; c++)
        {
            for (int p = 0; p < Patterns.Length; p++)
            {
                Color wall = Palette[c].colour;
                int pattern = p;
                Texture2D tex = Texture(FacadePath(c, p, ".png"), w, h, repaint, (x, y) => Upper(pattern, wall, x, y));
                Lit(FacadePath(c, p, ".mat"), tex, Color.white, 0f);
            }
        }
        int hg = Mathf.RoundToInt(style.groundFloorHeight * PxPerMetre);
        Texture2D ground = Texture($"{FacadeFolder}/GroundFloor.png", w, hg, repaint, Ground);
        Lit($"{FacadeFolder}/GroundFloor.mat", ground, Color.white, 0f);
        Lit($"{FacadeFolder}/Roof.mat", null, new Color(0.42f, 0.42f, 0.43f), 0.1f);
    }

    /// <summary>A texture painted from <paramref name="pixel"/> (metres from the bottom left, RGBA with smoothness in A),
    /// saved as PNG and imported repeating, mipmapped and anisotropic, if missing or <paramref name="repaint"/>.</summary>
    private static Texture2D Texture(string path, int w, int h, bool repaint, Func<float, float, Color> pixel)
    {
        if (!repaint && File.Exists(path)) return AssetDatabase.LoadAssetAtPath<Texture2D>(path);
        var tex = new Texture2D(w, h, TextureFormat.RGBA32, false);
        var px = new Color[w * h];
        for (int j = 0; j < h; j++)
        {
            for (int i = 0; i < w; i++) px[j * w + i] = pixel((i + 0.5f) / PxPerMetre, (j + 0.5f) / PxPerMetre);
        }
        tex.SetPixels(px);
        File.WriteAllBytes(path, tex.EncodeToPNG());
        UnityEngine.Object.DestroyImmediate(tex);
        AssetDatabase.ImportAsset(path, ImportAssetOptions.ForceUpdate);
        var importer = (TextureImporter)AssetImporter.GetAtPath(path);
        importer.textureType = TextureImporterType.Default;
        importer.sRGBTexture = true;
        importer.alphaSource = TextureImporterAlphaSource.FromInput;
        importer.alphaIsTransparency = false;
        importer.mipmapEnabled = true;
        importer.wrapMode = TextureWrapMode.Repeat;
        importer.filterMode = FilterMode.Trilinear;
        importer.anisoLevel = 8;
        importer.SaveAndReimport();
        return AssetDatabase.LoadAssetAtPath<Texture2D>(path);
    }

    /// <summary>A Standard material, made if missing: <paramref name="tex"/> as albedo with smoothness from its alpha, or
    /// a plain colour at <paramref name="gloss"/>.</summary>
    private static void Lit(string path, Texture2D tex, Color colour, float gloss)
    {
        var m = AssetDatabase.LoadAssetAtPath<Material>(path);
        bool made = m == null;
        if (made) m = new Material(Shader.Find("Standard")) { name = Path.GetFileNameWithoutExtension(path) };
        else if (m.mainTexture == tex || tex == null) return;
        m.color = colour;
        m.SetFloat("_Metallic", 0f);
        if (tex != null)
        {
            m.mainTexture = tex;
            m.SetFloat("_SmoothnessTextureChannel", 1f);
            m.EnableKeyword("_SMOOTHNESS_TEXTURE_ALBEDO_CHANNEL_A");
            m.SetFloat("_GlossMapScale", 1f);
        }
        else
        {
            m.SetFloat("_Glossiness", gloss);
        }
        if (made) AssetDatabase.CreateAsset(m, path);
        else EditorUtility.SetDirty(m);
    }

    /// <summary>A floor of one window pattern on a wall of one colour, 3 m by 3.5 m: a slab edge along the bottom, then
    /// punched windows (one per bay, split down the middle), a ribbon (four panes running nearly the bay's width) or a
    /// tall pair with transoms.</summary>
    private static Color Upper(int pattern, Color wall, float x, float y)
    {
        wall = Grain(wall, x, y, 0.12f);
        if (y < 0.22f) return Color.Lerp(wall, new Color(0.78f, 0.78f, 0.76f, 0.2f), 0.55f);
        switch (pattern)
        {
            case 0:
                return Window(x, y, 0.80f, 2.20f, 0.95f, 2.85f, new[] { 1.5f }, null, true) ?? wall;
            case 1:
                return Window(x, y, 0.15f, 2.85f, 1.00f, 2.55f, new[] { 0.825f, 1.5f, 2.175f }, null, true) ?? wall;
            default:
                return Window(x, y, 0.45f, 1.20f, 0.55f, 3.05f, null, 2.55f, true) ??
                       Window(x, y, 1.80f, 2.55f, 0.55f, 3.05f, null, 2.55f, true) ?? wall;
        }
    }

    /// <summary>The ground floor, 3 m by 4.5 m, the same on every building: a plinth, stone piers, a shopfront with a
    /// door and a transom, and a fascia with a sign band above it.</summary>
    private static Color Ground(float x, float y)
    {
        if (y < 0.35f) return Grain(new Color(0.30f, 0.29f, 0.28f, 0.15f), x, y, 0.15f);
        if (y > 3.40f)
        {
            bool sign = y > 3.62f && y < 4.22f && x > 0.30f && x < 2.70f;
            return Grain(sign ? new Color(0.22f, 0.24f, 0.27f, 0.3f) : new Color(0.46f, 0.46f, 0.47f, 0.2f), x, y, 0.2f);
        }
        if (x < 0.18f || x > 2.82f) return Grain(Stone, x, y, 0.15f);
        return Window(x, y, 0.18f, 2.82f, 0.35f, 3.40f, new[] { 1.05f }, 2.78f, false, 0.5f) ?? Stone;
    }

    /// <summary>A framed window over [x0, x1] x [y0, y1] with vertical mullions and a transom, or null outside it (and
    /// outside its sill). Glass is darker low down and catches the sky higher up.</summary>
    private static Color? Window(float x, float y, float x0, float x1, float y0, float y1, float[] mullions, float? transom,
                                 bool sill, float darker = 1f)
    {
        const float f = 0.07f, bar = 0.05f;
        if (sill && y >= y0 - 0.08f && y < y0 && x >= x0 - 0.08f && x <= x1 + 0.08f) return Sill;
        if (x < x0 || x > x1 || y < y0 || y > y1) return null;
        if (x < x0 + f || x > x1 - f || y < y0 + f || y > y1 - f) return Frame;
        if (mullions != null)
        {
            foreach (float m in mullions)
            {
                if (Mathf.Abs(x - m) < 0.5f * bar) return Frame;
            }
        }
        if (transom.HasValue && Mathf.Abs(y - transom.Value) < 0.5f * bar) return Frame;
        float t = (y - y0) / (y1 - y0);
        Color glass = Color.Lerp(new Color(0.10f, 0.13f, 0.17f), new Color(0.24f, 0.30f, 0.36f), t * t);
        glass = Color.Lerp(glass, glass * darker, 1f - t);
        glass.a = 0.85f;
        return glass;
    }

    /// <summary>A little fixed grain on an opaque surface (±3%), so a wall does not read as flat colour.</summary>
    private static Color Grain(Color c, float x, float y, float smoothness)
    {
        float n = Mathf.Sin(x * 127.1f + y * 311.7f) * 43758.5453f;
        float k = 1f + 0.06f * (n - Mathf.Floor(n) - 0.5f);
        return new Color(c.r * k, c.g * k, c.b * k, smoothness);
    }

    private static string NoLayoutHint(string layout) =>
        string.IsNullOrEmpty(layout)
            ? " Write one first (in Assets/Scripts/Experiment): python -B -m citysim.blockcity one --variant B --p 40 " +
              "--s 18 --seed 1 --scene"
            : "";
}
