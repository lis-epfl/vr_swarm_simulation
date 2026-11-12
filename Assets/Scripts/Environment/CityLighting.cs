using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Rendering;

/// <summary>
/// Sun shadows, the sky the buildings hide, and haze for a generated city, at no cost to the CPU per drone-camera
/// render. The scenic city (ScenicPatchCityWorldBuilder) carries it; no other scene does, and with it disabled the
/// scene renders as any other.
///
/// <para><b>Why not Unity's shadows.</b> An FPV render is CPU culling and submission (CLAUDE.md), and the feeds make
/// ~300 of them a second. A shadow-casting sun adds a shadow pass to every one: the city culled and drawn again from the
/// light, per cascade. The city never moves, so here its shadow is drawn <b>once</b>, at the start of Play, into one depth
/// map seen from the sun, and every later render only reads it.</para>
///
/// <para><b>How a render uses it.</b> Each drone camera gets a command buffer after its opaque geometry
/// (<c>Hidden/Swarm/City Lighting Composite</c>): two full-screen draws that read the depth buffer the render has just
/// written, rebuild each pixel's world position and facet normal from it, and multiply the picture by the ratio between
/// what the surface's own shader assumed (the whole sky, no shadow) and what reaches it, then blend the haze over.
/// That ratio does not depend on the surface's colour, so it is right for every material in the scene without touching
/// one of them: the terrain, the pack's props, the trees, the pedestrians and the other drones are all shadowed by the
/// buildings, and none of them needed a new shader. Nothing is drawn a second time, and no depth pre-pass is asked for:
/// the feed's own depth buffer is bound as a texture while a spare depth buffer of the same size stands in.</para>
///
/// <para><b>What is baked</b> (<see cref="Bake"/>, once, a few tens of milliseconds): the sun's depth map over the city;
/// a height map of the buildings and from it a small volume of how much sky each point in the air still sees, which
/// darkens the street canyons and the feet of the facades; and a blurred map of where street furniture stands, the soft
/// patch on the ground under a tree or a bench. Only this root's static renderers cast: pedestrians and drones do not
/// (the pedestrians keep their own ground patch, WalkerGroundShadows), and they are still shadowed by the city.</para>
///
/// <para><b>The ambient light is set here too</b>, as a probe built from three colours (sky, horizon, ground), so the
/// scene's shaders and this pass agree on it exactly and nothing depends on a lighting bake. The sky reflected in
/// glass is a cube drawn once from the skybox.</para>
///
/// <para>Windows / Direct3D 11 is what it is written for and tested on; the row order and depth range of other
/// platforms are handled where they are known (<c>SystemInfo</c>), not tested.</para>
/// </summary>
[DisallowMultipleComponent]
[DefaultExecutionOrder(-800)]
public class CityLighting : MonoBehaviour
{
    [Header("The sun (empty: RenderSettings.sun)")]
    public Light sun;

    [Header("Ambient light: a probe built from three colours (linear)")]
    public bool setAmbient = true;
    [ColorUsage(false, true)] public Color skyColor = new Color(0.54f, 0.60f, 0.71f);
    [ColorUsage(false, true)] public Color horizonColor = new Color(0.44f, 0.46f, 0.50f);
    [ColorUsage(false, true)] public Color groundColor = new Color(0.27f, 0.26f, 0.24f);
    [Tooltip("Draw the skybox into a cube once and use it as the scene's reflection (what glass shows).")]
    public bool skyReflection = true;
    public int reflectionSize = 256;

    [Header("Sun shadows: one depth map of the city, drawn once")]
    public bool sunShadows = true;
    public int shadowMapSize = 8192;
    [Tooltip("How much of the sun a shadow takes away. Kept well under 1 on purpose (the user's call, 2026-11-11): a " +
             "pedestrian in the shade has to read nearly as well as one in the sun.")]
    [Range(0f, 1f)] public float shadowStrength = 0.5f;
    [Tooltip("Texels between the nine taps: the width of a shadow's soft edge.")]
    public float shadowSoftness = 1.25f;
    [Tooltip("m: a surface is tested this far off itself along its normal, so it does not shadow itself.")]
    public float shadowNormalOffset = 0.12f;
    [Tooltip("m along the sun's direction.")]
    public float shadowDepthBias = 0.12f;

    [Header("Sky hidden by the buildings")]
    public bool skyOcclusion = true;
    [Range(0f, 1f)] public float skyOcclusionStrength = 0.5f;
    [Tooltip("How much of the hidden sky's light comes back off the facades (their albedo, roughly).")]
    [Range(0f, 0.9f)] public float bounce = 0.5f;
    [Tooltip("The least share of the sky's light left anywhere.")]
    [Range(0f, 1f)] public float leastSky = 0.5f;
    [Tooltip("Texels across the city each way, and slices up.")]
    public Vector2Int volumeSize = new Vector2Int(512, 32);
    [Tooltip("m: how far a point looks for buildings hiding its sky.")]
    public float volumeReach = 90f;
    [Tooltip("m off the surface, along its normal, at which the volume is read: about one texel, to clear the wall.")]
    public float volumeNormalOffset = 2.5f;
    [Tooltip("The soft patch on the ground under street furniture and trees. 0: none.")]
    [Range(0f, 1f)] public float canopyStrength = 0.4f;
    [Tooltip("m: how far that patch spreads.")]
    public float canopyBlur = 1.0f;

    [Header("Haze")]
    public bool haze = true;
    [Tooltip("Per metre at the base height: 0.00025 hides about a sixth of what is 700 m off.")]
    public float hazeDensity = 0.00025f;
    [Tooltip("Per metre: how fast the haze thins with height.")]
    public float hazeHeightFalloff = 0.004f;
    [Range(0f, 1f)] public float hazeMax = 0.8f;
    public Color hazeColor = new Color(0.33f, 0.36f, 0.43f);
    public Color hazeSunColor = new Color(0.50f, 0.54f, 0.62f);
    public float hazeSunExponent = 4f;

    [Header("Shaders (Hidden/Swarm/City ...)")]
    public Shader casterShader;
    public Shader bakeShader;
    public Shader compositeShader;

    /// <summary>The active instance, or null: ScreenSpawn asks it for the feeds' depth format.</summary>
    public static CityLighting Active { get; private set; }

    /// <summary>What <see cref="Bake"/> last did, for the log and the probes.</summary>
    public string BakeSummary { get; private set; } = "";

    private const string ShadowKeyword = "TransparentCutout";
    private static readonly int CasterVP = Shader.PropertyToID("_CityCasterVP");
    private static readonly int DepthTex = Shader.PropertyToID("_CityDepthTex");

    private RenderTexture shadowMap, skyVolume, canopyMap, reflectionCube;
    private Material casterOpaque, bakeMaterial, compositeMaterial;
    private readonly Dictionary<Material, Material> cutoutCasters = new Dictionary<Material, Material>();
    private Mesh triangle;
    private bool ready;

    private sealed class Hook
    {
        public CommandBuffer commands;
        public RenderTexture target;
    }

    private readonly Dictionary<Camera, Hook> hooks = new Dictionary<Camera, Hook>();
    private readonly Dictionary<Camera, bool> droneCameras = new Dictionary<Camera, bool>();
    private readonly Dictionary<Vector2Int, RenderTexture> spareDepth = new Dictionary<Vector2Int, RenderTexture>();
    private AmbientMode ambientModeBefore;
    private SphericalHarmonicsL2 ambientProbeBefore;
    private float ambientIntensityBefore;
    private DefaultReflectionMode reflectionModeBefore;
    private Texture reflectionBefore;
    private bool ambientSet, reflectionSet;

    [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.SubsystemRegistration)]
    private static void ResetStatics() => Active = null;

    private void OnEnable()
    {
        if (!Application.isPlaying) return;
        Active = this;
        Camera.onPreCull += OnCameraPreCull;
    }

    private void OnDisable()
    {
        if (!Application.isPlaying) return;
        Camera.onPreCull -= OnCameraPreCull;
        foreach (KeyValuePair<Camera, Hook> kv in hooks)
        {
            if (kv.Key != null) kv.Key.RemoveCommandBuffer(CameraEvent.AfterForwardOpaque, kv.Value.commands);
            kv.Value.commands.Release();
        }
        hooks.Clear();
        droneCameras.Clear();
        if (Active == this) Active = null;
    }

    private void OnDestroy()
    {
        if (ambientSet)
        {
            RenderSettings.ambientMode = ambientModeBefore;
            RenderSettings.ambientProbe = ambientProbeBefore;
            RenderSettings.ambientIntensity = ambientIntensityBefore;
        }
        if (reflectionSet)
        {
            RenderSettings.defaultReflectionMode = reflectionModeBefore;
            RenderSettings.customReflectionTexture = reflectionBefore;
        }
        Release(ref shadowMap);
        Release(ref skyVolume);
        Release(ref canopyMap);
        Release(ref reflectionCube);
        foreach (RenderTexture rt in spareDepth.Values) rt.Release();
        spareDepth.Clear();
        foreach (Material m in cutoutCasters.Values) Destroy(m);
        if (casterOpaque != null) Destroy(casterOpaque);
        if (bakeMaterial != null) Destroy(bakeMaterial);
        if (compositeMaterial != null) Destroy(compositeMaterial);
        if (triangle != null) Destroy(triangle);
    }

    private static void Release(ref RenderTexture rt)
    {
        if (rt == null) return;
        rt.Release();
        Destroy(rt);
        rt = null;
    }

    // After every Awake -- ProceduralCityBuilder has built or adopted the city and merged its dressing by then -- and,
    // at this execution order, before the Starts that spawn the pedestrians and the drones.
    private void Start()
    {
        if (sun == null) sun = RenderSettings.sun;
        casterShader = casterShader != null ? casterShader : Shader.Find("Hidden/Swarm/City Caster");
        bakeShader = bakeShader != null ? bakeShader : Shader.Find("Hidden/Swarm/City Bake");
        compositeShader = compositeShader != null ? compositeShader : Shader.Find("Hidden/Swarm/City Lighting Composite");
        if (sun == null || casterShader == null || bakeShader == null || compositeShader == null)
        {
            Debug.LogError("CityLighting: " + (sun == null ? "no sun (a directional light, or RenderSettings.sun)" : "its shaders are missing") +
                           "; the city is drawn without it.", this);
            enabled = false;
            return;
        }
        casterOpaque = new Material(casterShader) { hideFlags = HideFlags.HideAndDontSave };
        bakeMaterial = new Material(bakeShader) { hideFlags = HideFlags.HideAndDontSave };
        compositeMaterial = new Material(compositeShader) { hideFlags = HideFlags.HideAndDontSave };
        triangle = new Mesh { name = "City lighting triangle", hideFlags = HideFlags.HideAndDontSave };
        triangle.vertices = new[] { new Vector3(-1f, -1f, 0f), new Vector3(3f, -1f, 0f), new Vector3(-1f, 3f, 0f) };
        triangle.triangles = new[] { 0, 1, 2 };
        triangle.bounds = new Bounds(Vector3.zero, Vector3.one * 1e6f);

        SetAmbient();
        Bake();
        ready = true;
    }

    /// <summary>Puts the inspector's values in force again: after changing one at Play, call this (or tick the component
    /// off and on). The baked maps are kept; <see cref="Bake"/> remakes them.</summary>
    [ContextMenu("Apply settings")]
    public void Apply()
    {
        if (!Application.isPlaying || !ready) return;
        SetAmbient();
        SetGlobals();
    }

    private void OnValidate()
    {
        if (Application.isPlaying && ready) Apply();
    }

    // ------------------------------------------------------------------ ambient and reflection

    /// <summary>a + b y + c y^2 in the normal's height y, as the shader constants Unity derives from a probe: the sky's
    /// colour straight up, the horizon's sideways, the ground's straight down.</summary>
    public static SphericalHarmonicsL2 ProbeFrom(Color sky, Color horizon, Color ground)
    {
        var sh = new SphericalHarmonicsL2();
        for (int ch = 0; ch < 3; ch++)
        {
            float a = horizon[ch], b = 0.5f * (sky[ch] - ground[ch]), c = 0.5f * (sky[ch] + ground[ch]) - horizon[ch];
            // ShadeSH9 evaluates (sh0 - sh6) + sh1 y + 3 sh6 z^2 + sh8 (x^2 - y^2) from these; y^2 = 1/2 - z^2/2 - (x^2 - y^2)/2.
            sh[ch, 0] = a + c / 3f;
            sh[ch, 1] = b;
            sh[ch, 6] = -c / 6f;
            sh[ch, 8] = -c / 2f;
        }
        return sh;
    }

    private void SetAmbient()
    {
        if (setAmbient)
        {
            if (!ambientSet)
            {
                ambientModeBefore = RenderSettings.ambientMode;
                ambientProbeBefore = RenderSettings.ambientProbe;
                ambientIntensityBefore = RenderSettings.ambientIntensity;
                ambientSet = true;
            }
            RenderSettings.ambientMode = AmbientMode.Custom;
            RenderSettings.ambientIntensity = 1f;
            RenderSettings.ambientProbe = ProbeFrom(skyColor, horizonColor, groundColor);
        }
        if (skyReflection && reflectionCube == null && RenderSettings.skybox != null)
        {
            int size = Mathf.ClosestPowerOfTwo(Mathf.Clamp(reflectionSize, 32, 1024));
            reflectionCube = new RenderTexture(size, size, 0, RenderTextureFormat.ARGBHalf, RenderTextureReadWrite.Linear)
            {
                name = "City sky reflection", dimension = TextureDimension.Cube, useMipMap = true, autoGenerateMips = false,
                filterMode = FilterMode.Trilinear, hideFlags = HideFlags.HideAndDontSave,
            };
            reflectionCube.Create();
            var go = new GameObject("City sky reflection camera") { hideFlags = HideFlags.HideAndDontSave };
            Camera cam = go.AddComponent<Camera>();
            cam.enabled = false;
            cam.cullingMask = 0;
            cam.clearFlags = CameraClearFlags.Skybox;
            cam.allowHDR = true;
            cam.nearClipPlane = 0.1f;
            cam.farClipPlane = 10f;
            cam.transform.position = transform.position;
            bool drawn = cam.RenderToCubemap(reflectionCube, 63);
            Destroy(go);
            if (drawn)
            {
                reflectionCube.GenerateMips();
                reflectionModeBefore = RenderSettings.defaultReflectionMode;
                reflectionBefore = RenderSettings.customReflectionTexture;
                reflectionSet = true;
                RenderSettings.defaultReflectionMode = DefaultReflectionMode.Custom;
                RenderSettings.customReflectionTexture = reflectionCube;
            }
            else
            {
                Debug.LogWarning("CityLighting: the skybox could not be drawn into a reflection cube; reflections stay as the scene has them.", this);
                Release(ref reflectionCube);
            }
        }
    }

    // ------------------------------------------------------------------ the bake

    private struct Caster
    {
        public Mesh mesh;
        public Matrix4x4 matrix;
        public int submesh;
        public Material source;
        public bool building;
    }

    /// <summary>The city's static renderers under this root, submesh by submesh: every MeshRenderer but the pedestrians'
    /// (<see cref="SpawnedContentRoot"/>). A static-batched renderer is drawn from its batch's mesh.</summary>
    private List<Caster> Casters(out Bounds bounds)
    {
        var list = new List<Caster>();
        bounds = new Bounds();
        bool any = false;
        Transform dressing = transform.Find(CityDressing.RootName);
        foreach (MeshRenderer r in GetComponentsInChildren<MeshRenderer>(false))
        {
            if (!r.enabled || SpawnedContentRoot.Covers(r.transform)) continue;
            MeshFilter f = r.GetComponent<MeshFilter>();
            Mesh mesh = f != null ? f.sharedMesh : null;
            if (mesh == null) continue;
            bool batched = r.isPartOfStaticBatch;
            Matrix4x4 matrix = batched ? Matrix4x4.identity : r.localToWorldMatrix;
            int first = batched ? r.subMeshStartIndex : 0;
            Material[] mats = r.sharedMaterials;
            bool building = dressing == null || !r.transform.IsChildOf(dressing);
            for (int s = 0; s < mats.Length && first + s < mesh.subMeshCount; s++)
            {
                list.Add(new Caster { mesh = mesh, matrix = matrix, submesh = first + s, source = mats[s], building = building });
            }
            if (!any) bounds = r.bounds;
            else bounds.Encapsulate(r.bounds);
            any = true;
        }
        return list;
    }

    private Material CasterFor(Material source, out int cutoutPass)
    {
        cutoutPass = 0;
        if (source == null || source.GetTag("RenderType", false) != ShadowKeyword || !source.HasProperty("_MainTex")) return casterOpaque;
        cutoutPass = 1;
        if (!cutoutCasters.TryGetValue(source, out Material m))
        {
            m = new Material(casterShader) { hideFlags = HideFlags.HideAndDontSave };
            m.mainTexture = source.mainTexture;
            m.mainTextureScale = source.mainTextureScale;
            m.mainTextureOffset = source.mainTextureOffset;
            m.SetFloat("_Cutoff", source.HasProperty("_Cutoff") ? source.GetFloat("_Cutoff") : 0.4f);
            cutoutCasters[source] = m;
        }
        return m;
    }

    /// <summary>World to (u, v, depth as the platform stores it) for a box whose three axes and ranges are given, and
    /// the clip matrix that draws into the same texels.</summary>
    private static void Maps(Vector3 right, Vector2 rangeU, Vector3 up, Vector2 rangeV, Vector3 forward, Vector2 rangeD,
                             out Matrix4x4 toTexture, out Matrix4x4 toClip)
    {
        Vector4 Row(Vector3 axis, Vector2 range) =>
            new Vector4(axis.x, axis.y, axis.z, -range.x) / (range.y - range.x);

        Vector4 u = Row(right, rangeU), v = Row(up, rangeV), d = Row(forward, rangeD);
        bool reversed = SystemInfo.usesReversedZBuffer;
        Vector4 stored = reversed ? new Vector4(-d.x, -d.y, -d.z, 1f - d.w) : d;
        toTexture = new Matrix4x4();
        toTexture.SetRow(0, u);
        toTexture.SetRow(1, v);
        toTexture.SetRow(2, stored);
        toTexture.SetRow(3, new Vector4(0f, 0f, 0f, 1f));
        // Clip space: x from u; y so that v = 0 is the first row of the texture; z as stored (Direct3D's 0..1), or -1..1.
        bool topFirst = SystemInfo.graphicsUVStartsAtTop;
        toClip = new Matrix4x4();
        toClip.SetRow(0, new Vector4(2f * u.x, 2f * u.y, 2f * u.z, 2f * u.w - 1f));
        toClip.SetRow(1, topFirst ? new Vector4(-2f * v.x, -2f * v.y, -2f * v.z, 1f - 2f * v.w)
                                  : new Vector4(2f * v.x, 2f * v.y, 2f * v.z, 2f * v.w - 1f));
        toClip.SetRow(2, topFirst ? stored : new Vector4(2f * d.x, 2f * d.y, 2f * d.z, 2f * d.w - 1f));
        toClip.SetRow(3, new Vector4(0f, 0f, 0f, 1f));
    }

    /// <summary>Draws the sun's depth map, the sky-visibility volume and the canopy map from the city as it stands, and
    /// puts them in force. Called once from Start; call it again if the city is rebuilt during Play.</summary>
    [ContextMenu("Bake again")]
    public void Bake()
    {
        if (!Application.isPlaying || casterOpaque == null) return;
        float t0 = Time.realtimeSinceStartup;
        List<Caster> casters = Casters(out Bounds city);
        Release(ref shadowMap);
        Release(ref skyVolume);
        Release(ref canopyMap);
        if (casters.Count == 0)
        {
            Debug.LogWarning("CityLighting: nothing under this root to cast a shadow; only the haze is drawn.", this);
            city = new Bounds(transform.position, Vector3.one);
        }
        // The region: the city's footprint and a margin, from the ground to above the roofs.
        float margin = 12f;
        Vector3 min = new Vector3(city.min.x - margin, Mathf.Min(city.min.y, 0f) - 1f, city.min.z - margin);
        Vector3 max = new Vector3(city.max.x + margin, city.max.y + 2f, city.max.z + margin);
        Vector3 size = max - min;
        var cmd = new CommandBuffer { name = "City lighting bake" };
        int drawn = 0;

        // ---- the sun's depth map
        Matrix4x4 worldToShadow = Matrix4x4.identity;
        Vector3 toSun = -sun.transform.forward;
        if (sunShadows && casters.Count > 0 && toSun.y > 0.02f)
        {
            Vector3 f = -toSun;
            Vector3 r = Vector3.Cross(Vector3.up, f).normalized;
            Vector3 u = Vector3.Cross(f, r);
            Vector2 ru = new Vector2(float.MaxValue, float.MinValue), rv = ru, rd = ru;
            for (int k = 0; k < 8; k++)
            {
                var p = new Vector3((k & 1) == 0 ? min.x : max.x, (k & 2) == 0 ? min.y : max.y, (k & 4) == 0 ? min.z : max.z);
                Stretch(ref ru, Vector3.Dot(p, r));
                Stretch(ref rv, Vector3.Dot(p, u));
                Stretch(ref rd, Vector3.Dot(p, f));
            }
            rd += new Vector2(-5f, 5f);
            Maps(r, ru, u, rv, f, rd, out worldToShadow, out Matrix4x4 clip);
            int n = Mathf.Min(Mathf.ClosestPowerOfTwo(Mathf.Clamp(shadowMapSize, 256, 16384)), SystemInfo.maxTextureSize);
            shadowMap = new RenderTexture(n, n, 16, RenderTextureFormat.Shadowmap, RenderTextureReadWrite.Linear)
            {
                name = "City sun shadow map", filterMode = FilterMode.Bilinear, wrapMode = TextureWrapMode.Clamp,
                hideFlags = HideFlags.HideAndDontSave,
            };
            shadowMap.Create();
            cmd.SetRenderTarget(shadowMap);
            cmd.ClearRenderTarget(true, false, Color.clear, 1f);
            cmd.SetGlobalMatrix(CasterVP, clip);
            foreach (Caster c in casters)
            {
                Material m = CasterFor(c.source, out int cut);
                cmd.DrawMesh(c.mesh, c.matrix, m, c.submesh, cut);
                drawn++;
            }
            shadowTexel = new Vector4(1f / n, 1f / n, shadowSoftness, shadowDepthBias / (rd.y - rd.x));
        }
        else
        {
            shadowMap = new RenderTexture(4, 4, 16, RenderTextureFormat.Shadowmap, RenderTextureReadWrite.Linear)
            {
                name = "City sun shadow map (none)", hideFlags = HideFlags.HideAndDontSave,
            };
            shadowMap.Create();
            cmd.SetRenderTarget(shadowMap);
            cmd.ClearRenderTarget(true, false, Color.clear, 1f);
            shadowTexel = new Vector4(0.25f, 0.25f, 0f, 0f);
        }

        // ---- heights from above: the buildings', for the sky; the props', for the patch under them
        int hSize = 2048, cSize = 4096;
        RenderTexture heights = Heights(cmd, casters, true, hSize, min, size, "City building heights");
        RenderTexture props = Heights(cmd, casters, false, cSize, min, size, "City prop heights");

        // ---- how much sky each point in the air sees
        int vx = Mathf.ClosestPowerOfTwo(Mathf.Clamp(volumeSize.x, 32, 1024)), vy = Mathf.Clamp(volumeSize.y, 4, 128);
        if (skyOcclusion && casters.Count > 0)
        {
            skyVolume = new RenderTexture(vx, vx, 0, RenderTextureFormat.R8, RenderTextureReadWrite.Linear)
            {
                name = "City sky visibility", dimension = TextureDimension.Tex3D, volumeDepth = vy,
                filterMode = FilterMode.Bilinear, wrapMode = TextureWrapMode.Clamp, hideFlags = HideFlags.HideAndDontSave,
            };
            skyVolume.Create();
            cmd.SetGlobalTexture("_CityBakeSource", heights);
            cmd.SetGlobalVector("_CityBakeSize", new Vector4(vx, vx, 1f / vx, 1f / vx));
            for (int k = 0; k < vy; k++)
            {
                cmd.SetRenderTarget(skyVolume, 0, CubemapFace.Unknown, k);
                cmd.SetGlobalVector("_CityBakeRegion", new Vector4(size.x, size.z, min.y + (k + 0.5f) / vy * size.y, volumeReach));
                cmd.DrawMesh(triangle, Matrix4x4.identity, bakeMaterial, 0, 0);
            }
        }
        else
        {
            skyVolume = new RenderTexture(1, 1, 0, RenderTextureFormat.R8, RenderTextureReadWrite.Linear)
            {
                name = "City sky visibility (none)", dimension = TextureDimension.Tex3D, volumeDepth = 1,
                hideFlags = HideFlags.HideAndDontSave,
            };
            skyVolume.Create();
            cmd.SetRenderTarget(skyVolume, 0, CubemapFace.Unknown, 0);
            cmd.ClearRenderTarget(false, true, Color.white);
        }

        // ---- the patch under the props
        canopyMap = new RenderTexture(cSize, cSize, 0, RenderTextureFormat.R8, RenderTextureReadWrite.Linear)
        {
            name = "City canopy map", filterMode = FilterMode.Bilinear, wrapMode = TextureWrapMode.Clamp,
            useMipMap = true, autoGenerateMips = true, hideFlags = HideFlags.HideAndDontSave,
        };
        canopyMap.Create();
        RenderTexture across = RenderTexture.GetTemporary(cSize, cSize, 0, RenderTextureFormat.R8, RenderTextureReadWrite.Linear);
        float texelsPerMetre = cSize / Mathf.Max(size.x, size.z);
        cmd.SetGlobalVector("_CityBakeSize", new Vector4(cSize, cSize, 1f / cSize, 1f / cSize));
        cmd.SetGlobalVector("_CityBakeBlur", new Vector4(Mathf.Max(canopyBlur * texelsPerMetre, 1f), 0.3f, 0.7f, 0f));
        cmd.SetGlobalTexture("_CityBakeSource", props);
        cmd.SetRenderTarget(across);
        cmd.DrawMesh(triangle, Matrix4x4.identity, bakeMaterial, 0, 1);
        cmd.SetGlobalTexture("_CityBakeSource", across);
        cmd.SetRenderTarget(canopyMap);
        cmd.DrawMesh(triangle, Matrix4x4.identity, bakeMaterial, 0, 2);

        Graphics.ExecuteCommandBuffer(cmd);
        cmd.Release();
        RenderTexture.ReleaseTemporary(across);
        RenderTexture.ReleaseTemporary(heights);
        RenderTexture.ReleaseTemporary(props);
        RenderTexture.active = null;

        this.worldToShadow = worldToShadow;
        regionMin = min;
        regionSize = size;
        SetGlobals();
        SphericalHarmonicsL2 probe = RenderSettings.ambientProbe;
        BakeSummary = $"ambient {RenderSettings.ambientMode} (green: up {probe[1, 0] - probe[1, 6] + probe[1, 1] - probe[1, 8]:F2}, side {probe[1, 0] - probe[1, 6] + probe[1, 8]:F2}, down {probe[1, 0] - probe[1, 6] - probe[1, 1] - probe[1, 8]:F2}), " +
                      $"{casters.Count} static submeshes, sun map {shadowMap.width} ({drawn} drawn), sky volume " +
                      $"{skyVolume.width}x{skyVolume.volumeDepth}x{skyVolume.height}, canopy {canopyMap.width}, region " +
                      $"{size.x:F0}x{size.y:F0}x{size.z:F0} m, {(Time.realtimeSinceStartup - t0) * 1000f:F0} ms";
        Debug.Log("CityLighting: baked " + BakeSummary, this);
    }

    private Matrix4x4 worldToShadow = Matrix4x4.identity;
    private Vector4 shadowTexel;
    private Vector3 regionMin, regionSize = Vector3.one;

    private static void Stretch(ref Vector2 range, float v)
    {
        range.x = Mathf.Min(range.x, v);
        range.y = Mathf.Max(range.y, v);
    }

    /// <summary>The height (m) of the highest caster over each texel of the region, 0 where there is none.</summary>
    private RenderTexture Heights(CommandBuffer cmd, List<Caster> casters, bool buildings, int n, Vector3 min, Vector3 size, string name)
    {
        RenderTexture rt = RenderTexture.GetTemporary(n, n, 24, RenderTextureFormat.RHalf, RenderTextureReadWrite.Linear);
        rt.name = name;
        rt.filterMode = FilterMode.Bilinear;
        rt.wrapMode = TextureWrapMode.Clamp;
        // Looking down: u along x, v along z, depth from the top of the region.
        Maps(Vector3.right, new Vector2(min.x, min.x + size.x), Vector3.forward, new Vector2(min.z, min.z + size.z),
             Vector3.down, new Vector2(-(min.y + size.y), -min.y), out _, out Matrix4x4 clip);
        cmd.SetRenderTarget(rt);
        cmd.ClearRenderTarget(true, true, Color.clear, 1f);
        cmd.SetGlobalMatrix(CasterVP, clip);
        foreach (Caster c in casters)
        {
            if (c.building != buildings) continue;
            Material m = CasterFor(c.source, out int cut);
            cmd.DrawMesh(c.mesh, c.matrix, m, c.submesh, 2 + cut);
        }
        return rt;
    }

    // ------------------------------------------------------------------ what the composite reads

    private void SetGlobals()
    {
        Shader.SetGlobalTexture("_CitySunShadowMap", shadowMap);
        Shader.SetGlobalMatrix("_CityWorldToShadow", worldToShadow);
        Vector4 texel = shadowTexel;
        texel.z = shadowSoftness;
        Shader.SetGlobalVector("_CityShadowTexel", texel);
        Shader.SetGlobalTexture("_CitySkyVolume", skyVolume);
        Shader.SetGlobalVector("_CityVolumeMin", regionMin);
        Shader.SetGlobalVector("_CityVolumeInvSize", new Vector4(1f / regionSize.x, 1f / regionSize.y, 1f / regionSize.z, 0f));
        Shader.SetGlobalTexture("_CityCanopyMap", canopyMap);
        Shader.SetGlobalVector("_CityRegion", new Vector4(regionMin.x, regionMin.z, 1f / regionSize.x, 1f / regionSize.z));

        Vector3 toSun = -sun.transform.forward;
        Shader.SetGlobalVector("_CitySunDir", toSun);
        Color sunColor = GraphicsSettings.lightsUseLinearIntensity ? sun.color.linear * sun.intensity : (sun.color * sun.intensity).linear;
        Shader.SetGlobalVector("_CitySunColor", sun.isActiveAndEnabled ? (Vector4)sunColor : Vector4.zero);

        SphericalHarmonicsL2 sh = RenderSettings.ambientProbe;
        string[] channel = { "r", "g", "b" };
        for (int ch = 0; ch < 3; ch++)
        {
            Shader.SetGlobalVector("_CitySHA" + channel[ch], new Vector4(sh[ch, 3], sh[ch, 1], sh[ch, 2], sh[ch, 0] - sh[ch, 6]));
            Shader.SetGlobalVector("_CitySHB" + channel[ch], new Vector4(sh[ch, 4], sh[ch, 5], sh[ch, 6] * 3f, sh[ch, 7]));
        }
        Shader.SetGlobalVector("_CitySHC", new Vector4(sh[0, 8], sh[1, 8], sh[2, 8], 1f));

        Shader.SetGlobalVector("_CityShade", new Vector4(sunShadows ? shadowStrength : 0f, skyOcclusion ? skyOcclusionStrength : 0f,
                                                         bounce, canopyStrength));
        Shader.SetGlobalVector("_CityOffsets", new Vector4(shadowNormalOffset, volumeNormalOffset, leastSky, 0.3f));
        Shader.SetGlobalVector("_CityHaze", new Vector4(haze ? hazeDensity : 0f, hazeHeightFalloff, 0f, hazeMax));
        Shader.SetGlobalVector("_CityHazeColor", hazeColor.linear);
        Color glow = hazeSunColor.linear;
        Shader.SetGlobalVector("_CityHazeSunColor", new Vector4(glow.r, glow.g, glow.b, hazeSunExponent));
    }

    // ------------------------------------------------------------------ the drone cameras

    /// <summary>The depth format a feed's RenderTexture should have while this is active: a float depth keeps the
    /// positions rebuilt from it exact out to the far plane, where 24 bits are a metre apart.</summary>
    public static UnityEngine.Experimental.Rendering.GraphicsFormat FeedDepthFormat =>
        UnityEngine.Experimental.Rendering.GraphicsFormat.D32_SFloat_S8_UInt;

    /// <summary>Has <paramref name="cam"/> drawn with the city's lighting, as the drone cameras are without asking: a
    /// camera of another kind that draws the city into a RenderTexture with a depth buffer (a probe's, a spectator's).</summary>
    public void Include(Camera cam)
    {
        if (cam != null) droneCameras[cam] = true;
    }

    private bool IsDroneCamera(Camera cam)
    {
        if (!droneCameras.TryGetValue(cam, out bool drone))
        {
            drone = cam.GetComponent<FPVCameraScript>() != null;
            droneCameras[cam] = drone;
        }
        return drone;
    }

    private void OnCameraPreCull(Camera cam)
    {
        if (!ready || cam == null) return;
        RenderTexture target = cam.targetTexture;
        if (target == null || target.depth == 0 || target.dimension != TextureDimension.Tex2D || !IsDroneCamera(cam)) return;
        if (!hooks.TryGetValue(cam, out Hook hook) || hook.target != target)
        {
            if (hook != null)
            {
                cam.RemoveCommandBuffer(CameraEvent.AfterForwardOpaque, hook.commands);
                hook.commands.Release();
            }
            hook = new Hook { target = target, commands = new CommandBuffer { name = "City lighting" } };
            var key = new Vector2Int(target.width, target.height);
            if (!spareDepth.TryGetValue(key, out RenderTexture spare) || spare == null)
            {
                spare = new RenderTexture(key.x, key.y, 16, RenderTextureFormat.Depth)
                {
                    name = $"City lighting spare depth {key.x}x{key.y}", hideFlags = HideFlags.HideAndDontSave,
                };
                spare.Create();
                spareDepth[key] = spare;
            }
            CommandBuffer c = hook.commands;
            // The feed's own depth, as a texture, while a spare depth buffer stands in for it: the two draws test nothing.
            c.SetGlobalTexture(DepthTex, target, RenderTextureSubElement.Depth);
            c.SetRenderTarget(new RenderTargetIdentifier(target), new RenderTargetIdentifier(spare));
            c.DrawMesh(triangle, Matrix4x4.identity, compositeMaterial, 0, 0);
            if (haze) c.DrawMesh(triangle, Matrix4x4.identity, compositeMaterial, 0, 1);
            c.SetRenderTarget(BuiltinRenderTextureType.CameraTarget);
            cam.AddCommandBuffer(CameraEvent.AfterForwardOpaque, c);
            hooks[cam] = hook;
        }

        Transform t = cam.transform;
        float tanV = Mathf.Tan(0.5f * cam.fieldOfView * Mathf.Deg2Rad), tanH = tanV * cam.aspect;
        Shader.SetGlobalVector("_CityTargetSize", new Vector4(target.width, target.height, 1f / target.width, 1f / target.height));
        Shader.SetGlobalVector("_CityCamPos", t.position);
        Shader.SetGlobalVector("_CityRayX", t.right * tanH);
        Shader.SetGlobalVector("_CityRayY", t.up * tanV);
        Shader.SetGlobalVector("_CityRayZ", t.forward);
        Shader.SetGlobalVector("_CityZ", new Vector4(cam.nearClipPlane, cam.farClipPlane, SystemInfo.usesReversedZBuffer ? 1f : 0f, 0f));
    }
}
