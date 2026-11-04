using System;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Rendering;
using UnityEngine.SceneManagement;

/// <summary>
/// A dim room round the pilot, fixed to the room the pilot sits in rather than to the virtual world, as a
/// visual anchor against the rig yaw. It comes in two styles (<see cref="style"/>): a grid floor inside a
/// ring wall with vertical ribs (<see cref="RoomStyle.FloorAndWall"/>), or a grid floor and a grid ceiling
/// that both fade into <see cref="baseColor"/> with distance, so the room has no edge anywhere
/// (<see cref="RoomStyle.FloorAndCeiling"/>). The style can be switched in Play to compare the two.
///
/// In the hull modes the yaw stick rotates the OVRCameraRig (<c>PyUniSharingFast.UpdateBodyYaw</c>), so
/// the ring of feed screens sweeps round the pilot while their vestibular system says they are still.
/// Behind the screens there used to be only the Arena box, uniform black: with no other structure in
/// view the screens are the frame the eye judges motion by, and a turn reads as the pilot turning. The
/// background is what is taken as the frame of reference (Ohmi, Howard and Landolt 1987), so a backdrop
/// that holds still behind the screens makes the sweeping screens read as objects moving instead. With
/// one drone the rig turns with the drone and its screen, so during a turn nothing in view moves but
/// the feed's own picture. The wall's ribs hold yaw directly; the grids hold it through their lines'
/// vanishing points, which any yaw would move.
///
/// It hangs under the rig's tracking space, never under the Arena: every rig yaw and
/// <c>CalibrateToCentre</c>'s height shift carry it along, so it never moves relative to the real room,
/// and leaning gives real parallax against it. The floor is <see cref="floorBelowEyes"/> under the
/// tracking origin, which under the EyeLevel origin these scenes use is the eye height at recenter;
/// under a floor-level origin it is the real floor.
///
/// It is drawn first (Background queue) and writes no depth (Sprites/Default writes none), so every
/// screen draws over it whatever the layout, radius or scale: a depth-writing floor 1.2 m down would cut
/// off the bottom of the panorama and of a four-row FORMATION_WALL. The ceiling's default height is above
/// the top of every layout at its default size, so no screen hangs past it. Two things would hide a room
/// drawn that way, and both are undone while it is on. The eye cameras clear to the skybox, which the
/// built-in pipeline draws after the opaque queue into every pixel with no depth, so they clear to
/// <see cref="baseColor"/> instead. The Arena box encloses everything and is drawn after it, so its
/// renderers are off; its colliders and transform, which ScreenSpawn and PyUniSharingFast read, are
/// untouched. Disabling the component puts both back, which is the view without it.
///
/// It must live on the feed-screen layer (<see cref="ScreenSpawn.screenLayerName"/>): the eye cameras
/// are culled to that layer and draw nothing else, and the FPV cameras exclude it. It carries no
/// collider, so neither the scene-plane raycast nor the swarm's obstacle query can hit it.
///
/// Added automatically to any scene with an Arena; add one by hand to tune it (the bootstrap then
/// leaves that one alone), or disable it to fly without it.
/// </summary>
public class CockpitRoom : MonoBehaviour
{
    // Values spelled out: Unity serialises enums by integer.
    public enum RoomStyle
    {
        FloorAndWall = 0,
        FloorAndCeiling = 1,
    }

    [Tooltip("FloorAndWall: a grid floor inside a ring wall with vertical ribs. FloorAndCeiling: a grid " +
             "floor and a grid ceiling, both fading into the base colour with distance. Can be switched " +
             "in Play.")]
    public RoomStyle style = RoomStyle.FloorAndCeiling;

    [Tooltip("Height of the floor below the tracking origin, in metres. Under the EyeLevel origin the " +
             "scenes use, that origin is the pilot's eye height at recenter, so this is a seated eye " +
             "height. Ignored under a floor-level origin, where the floor is the real one.")]
    public float floorBelowEyes = 1.2f;

    [Tooltip("Floor (and ceiling) grid spacing, in metres.")]
    public float gridSpacing = 1f;

    [Tooltip("Width of a grid line, in metres.")]
    public float gridLineWidth = 0.025f;

    [Header("Floor and wall")]
    [Tooltip("Radius of the wall, and of the floor it stands on, in metres. Beyond the curved " +
             "panorama's 5 m, so stereo puts it behind every screen.")]
    public float wallRadius = 9f;

    [Tooltip("Height of the wall above the floor, in metres. Its ribs fade out towards the top, so " +
             "it has no top edge.")]
    public float wallHeight = 6f;

    [Tooltip("Angle between the wall's vertical ribs, in degrees, rounded so a whole number goes " +
             "round. The ribs are what hold still against a yaw.")]
    public float ribSpacingDeg = 15f;

    [Tooltip("Width of a wall rib, in metres.")]
    public float ribWidth = 0.06f;

    [Tooltip("Strength of the floor grid at the wall, as a fraction of its strength under the pilot.")]
    [Range(0f, 1f)] public float floorFarAlpha = 0.3f;

    [Header("Floor and ceiling")]
    [Tooltip("Height of the ceiling above the eyes, in metres (above floorBelowEyes over the floor " +
             "under a floor-level origin). The default clears the top of every screen layout at its " +
             "default size (the panorama's is 1.5 m up), so no screen hangs past it.")]
    public float ceilingAboveEyes = 1.8f;

    [Tooltip("Horizontal distance from the pilot at which the floor and ceiling grids have faded into " +
             "the base colour, in metres. They fade smoothly all the way out, so neither has an edge.")]
    public float fadeDistance = 15f;

    [Tooltip("Strength of the ceiling grid, as a fraction of the floor's.")]
    [Range(0f, 1f)] public float ceilingStrength = 1f;

    [Header("Colours")]
    [Tooltip("The room's ground colour: what the eye cameras clear to, and what every fade ends in. " +
             "The default is the Arena box as it renders (its black Standard material still reflects " +
             "the ambient sky, to sRGB 19/21/26), so the lines are the only thing the room adds.")]
    public Color baseColor = new Color(0.075f, 0.082f, 0.102f, 1f);

    [Tooltip("The colour of a line at full strength, drawn over baseColor. Keep it close to the base: " +
             "the room is an anchor, and the screens should stay what the eye goes to.")]
    public Color lineColor = new Color(0.17f, 0.176f, 0.196f, 1f);

    /// <summary>True while a room is drawn in place of the Arena box (read by ExperimentRecorder).</summary>
    public static bool Active { get; private set; }

    /// <summary>The style drawn while <see cref="Active"/>.</summary>
    public static RoomStyle ActiveStyle { get; private set; }

    /// <summary>The style in force, or "off" (ExperimentRecorder's display events and session file).</summary>
    public static string InForce => Active ? ActiveStyle.ToString() : "off";

    // Round the wall and the floor's rim alike, so the floor's edge meets the wall's base exactly.
    private const int Segments = 96;
    private const int FloorRings = 12;
    private const int WallRows = 8;
    // The fading discs are wider and their fade is a curve, so they get more rings.
    private const int FadeRings = 30;

    private GameObject arena;
    private OVRCameraRig rig;
    private GameObject root;
    private Material floorMaterial;
    private Material wallMaterial;
    private Material ceilingMaterial;
    // Everything Build makes, destroyed together on a rebuild.
    private readonly List<UnityEngine.Object> owned = new List<UnityEngine.Object>();
    private readonly List<Renderer> hiddenArenaRenderers = new List<Renderer>();
    private readonly List<(Camera camera, CameraClearFlags flags, Color background)> clearedCameras =
        new List<(Camera, CameraClearFlags, Color)>();
    // What the current meshes and textures were built with, so an inspector change in Play rebuilds them.
    private int builtShape;

    [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.BeforeSceneLoad)]
    private static void RegisterBootstrap()
    {
        Active = false;
        // -= first so a disabled domain reload cannot stack the handler.
        SceneManager.sceneLoaded -= OnSceneLoaded;
        SceneManager.sceneLoaded += OnSceneLoaded;
    }

    private static void OnSceneLoaded(Scene scene, LoadSceneMode mode)
    {
        if (FindObjectOfType<CockpitRoom>(true) != null) return;
        if (GameObject.FindGameObjectWithTag("Arena") == null) return;
        new GameObject("CockpitRoom").AddComponent<CockpitRoom>();
    }

    private void OnDisable()
    {
        if (root != null) root.SetActive(false);
        Disengage();
    }

    private void OnDestroy()
    {
        Disengage();
        DestroyVisual();
    }

    // Start as well as LateUpdate, so the room is up before any Update reads Active (the recorder's
    // first display event): the bootstrap adds this after the rig's Awake and before every Start.
    private void Start() => Refresh();

    private void LateUpdate() => Refresh();

    private void Refresh()
    {
        if (arena == null)
        {
            arena = GameObject.FindGameObjectWithTag("Arena");
            if (arena == null) return;
        }
        if (rig == null || rig.trackingSpace == null)
        {
            rig = FindObjectOfType<OVRCameraRig>();
            if (rig == null || rig.trackingSpace == null) return;
        }

        int shape = ShapeHash();
        if (root == null || shape != builtShape) Build(shape);
        if (!Active) Engage();
        ActiveStyle = style;

        if (floorMaterial != null) floorMaterial.color = lineColor;
        if (wallMaterial != null) wallMaterial.color = lineColor;
        if (ceilingMaterial != null)
        {
            ceilingMaterial.color = new Color(lineColor.r, lineColor.g, lineColor.b, lineColor.a * ceilingStrength);
        }
        foreach (var cleared in clearedCameras)
        {
            if (cleared.camera != null) cleared.camera.backgroundColor = baseColor;
        }
    }

    private int ShapeHash() => HashCode.Combine(
        HashCode.Combine(floorBelowEyes, wallRadius, wallHeight, gridSpacing, gridLineWidth, ribSpacingDeg,
                         ribWidth, floorFarAlpha),
        style, ceilingAboveEyes, fadeDistance);

    private int RibCount() => Mathf.Max(1, Mathf.RoundToInt(360f / Mathf.Max(ribSpacingDeg, 0.1f)));

    /// <summary>
    /// Hides the Arena box and clears the eye cameras to the room's base colour, remembering both so
    /// <see cref="Disengage"/> can put them back. Only the renderers under the Arena at this moment are
    /// hidden; the feed screens hang under ScreenSpawn's own parent, not the Arena.
    /// </summary>
    private void Engage()
    {
        hiddenArenaRenderers.Clear();
        foreach (Renderer r in arena.GetComponentsInChildren<Renderer>())
        {
            if (!r.enabled) continue;
            r.enabled = false;
            hiddenArenaRenderers.Add(r);
        }

        clearedCameras.Clear();
        foreach (Camera c in rig.GetComponentsInChildren<Camera>(true))
        {
            clearedCameras.Add((c, c.clearFlags, c.backgroundColor));
            c.clearFlags = CameraClearFlags.SolidColor;
            c.backgroundColor = baseColor;
        }

        root.SetActive(true);
        Active = true;
    }

    private void Disengage()
    {
        if (!Active) return;
        foreach (Renderer r in hiddenArenaRenderers)
        {
            if (r != null) r.enabled = true;
        }
        hiddenArenaRenderers.Clear();
        foreach (var cleared in clearedCameras)
        {
            if (cleared.camera == null) continue;
            cleared.camera.clearFlags = cleared.flags;
            cleared.camera.backgroundColor = cleared.background;
        }
        clearedCameras.Clear();
        Active = false;
    }

    private void Build(int shape)
    {
        DestroyVisual();
        builtShape = shape;

        root = new GameObject("CockpitRoomVisual");
        root.transform.SetParent(rig.trackingSpace, false);
        bool floorLevelOrigin = OVRManager.instance != null &&
                                OVRManager.instance.trackingOriginType != OVRManager.TrackingOrigin.EyeLevel;
        root.transform.localPosition = new Vector3(0f, floorLevelOrigin ? 0f : -floorBelowEyes, 0f);
        owned.Add(root);

        ScreenSpawn screenSpawn = FindObjectOfType<ScreenSpawn>();
        int layer = LayerMask.NameToLayer(screenSpawn != null ? screenSpawn.screenLayerName : "UI");
        if (layer < 0)
        {
            Debug.LogWarning("[CockpitRoom] Feed-screen layer not found; the headset eye cameras " +
                             "will not draw the room.");
            layer = 0;
        }
        root.layer = layer;

        Texture2D grid = LineTexture(128, 128, gridLineWidth / gridSpacing, true);
        floorMaterial = NewMaterial(grid);
        if (style == RoomStyle.FloorAndWall)
        {
            float ribPitch = 2f * Mathf.PI * wallRadius / RibCount();
            wallMaterial = NewMaterial(LineTexture(256, 4, ribWidth / ribPitch, false));
            AddPart("Floor", DiscMesh("CockpitRoomFloor", wallRadius, FloorRings,
                                      t => Mathf.Lerp(1f, floorFarAlpha, t)), floorMaterial, layer, 0f);
            AddPart("Wall", WallMesh(), wallMaterial, layer, 0f);
        }
        else
        {
            // One mesh for both: a smoothstep from 1 under the pilot to 0 at fadeDistance, flat at both
            // ends, so the fade has no visible start and the rim no edge.
            Mesh fading = DiscMesh("CockpitRoomFadingGrid", fadeDistance, FadeRings,
                                   t => 1f - t * t * (3f - 2f * t));
            ceilingMaterial = NewMaterial(grid);
            AddPart("Floor", fading, floorMaterial, layer, 0f);
            AddPart("Ceiling", fading, ceilingMaterial, layer, floorBelowEyes + ceilingAboveEyes);
        }
    }

    private void DestroyVisual()
    {
        foreach (UnityEngine.Object o in owned)
        {
            if (o != null) Destroy(o);
        }
        owned.Clear();
        root = null;
        floorMaterial = null;
        wallMaterial = null;
        ceilingMaterial = null;
    }

    private void AddPart(string name, Mesh mesh, Material material, int layer, float height)
    {
        var part = new GameObject(name) { layer = layer };
        part.transform.SetParent(root.transform, false);
        part.transform.localPosition = new Vector3(0f, height, 0f);
        part.AddComponent<MeshFilter>().sharedMesh = mesh;
        MeshRenderer mr = part.AddComponent<MeshRenderer>();
        mr.sharedMaterial = material;
        mr.shadowCastingMode = ShadowCastingMode.Off;
        mr.receiveShadows = false;
        mr.lightProbeUsage = LightProbeUsage.Off;
        mr.reflectionProbeUsage = ReflectionProbeUsage.Off;
    }

    /// <summary>
    /// Sprites/Default, as BodyYawCompass uses: unlit, premultiplied, double-sided (so the ceiling is
    /// seen from below), no depth write, stereo-instanced, and always included in builds. Queued as
    /// Background so it is drawn before every screen; the line colour is the material colour, applied
    /// over the texture's line mask and the mesh's vertex alpha (the fades).
    /// </summary>
    private Material NewMaterial(Texture2D texture)
    {
        var material = new Material(Shader.Find("Sprites/Default"))
        {
            mainTexture = texture,
            color = lineColor,
            renderQueue = (int)RenderQueue.Background,
        };
        owned.Add(material);
        return material;
    }

    /// <summary>
    /// White lines on transparent, one line per tile along u (and along v for a grid), centred on the
    /// tile edge so they meet across the wrap, with a one-pixel soft edge. Mipmapped with anisotropic
    /// filtering, so distant grid lines average into a faint even tone instead of shimmering.
    /// </summary>
    private Texture2D LineTexture(int width, int height, float lineFraction, bool grid)
    {
        var texture = new Texture2D(width, height, TextureFormat.RGBA32, true)
        {
            name = grid ? "CockpitRoomGrid" : "CockpitRoomRib",
            wrapMode = TextureWrapMode.Repeat,
            filterMode = FilterMode.Trilinear,
            anisoLevel = 16,
        };
        var pixels = new Color32[width * height];
        for (int y = 0; y < height; y++)
        {
            float ay = grid ? LineAlpha(y, height, lineFraction) : 0f;
            for (int x = 0; x < width; x++)
            {
                float a = Mathf.Max(LineAlpha(x, width, lineFraction), ay);
                pixels[y * width + x] = new Color32(255, 255, 255, (byte)Mathf.RoundToInt(255f * a));
            }
        }
        texture.SetPixels32(pixels);
        texture.Apply(true, true);
        owned.Add(texture);
        return texture;
    }

    private static float LineAlpha(int i, int size, float lineFraction)
    {
        float distance = Mathf.Min(i + 0.5f, size - i - 0.5f);
        return Mathf.Clamp01(0.5f * lineFraction * size + 0.5f - distance);
    }

    /// <summary>
    /// A horizontal disc of the given radius centred under (or over) the pilot, UVs in grid tiles (offset
    /// half a tile so the pilot sits in the middle of one, not on a line), vertex alpha
    /// <paramref name="alphaAt"/> of the fraction of the radius out.
    /// </summary>
    private Mesh DiscMesh(string name, float radius, int rings, Func<float, float> alphaAt)
    {
        var vertices = new List<Vector3> { Vector3.zero };
        var colors = new List<Color> { new Color(1f, 1f, 1f, alphaAt(0f)) };
        var triangles = new List<int>();
        for (int ring = 1; ring <= rings; ring++)
        {
            float t = (float)ring / rings;
            float alpha = alphaAt(t);
            for (int s = 0; s < Segments; s++)
            {
                float angle = 2f * Mathf.PI * s / Segments;
                vertices.Add(new Vector3(radius * t * Mathf.Cos(angle), 0f, radius * t * Mathf.Sin(angle)));
                colors.Add(new Color(1f, 1f, 1f, alpha));
            }
        }
        for (int s = 0; s < Segments; s++)
        {
            triangles.AddRange(new[] { 0, 1 + (s + 1) % Segments, 1 + s });
        }
        for (int ring = 1; ring < rings; ring++)
        {
            int inner = 1 + (ring - 1) * Segments;
            int outer = inner + Segments;
            for (int s = 0; s < Segments; s++)
            {
                int s1 = (s + 1) % Segments;
                triangles.AddRange(new[] { inner + s, inner + s1, outer + s1, inner + s, outer + s1, outer + s });
            }
        }

        var uvs = new Vector2[vertices.Count];
        for (int i = 0; i < uvs.Length; i++)
        {
            uvs[i] = new Vector2(vertices[i].x / gridSpacing + 0.5f, vertices[i].z / gridSpacing + 0.5f);
        }
        return NewMesh(name, vertices, uvs, colors, triangles);
    }

    /// <summary>
    /// An open cylinder of radius wallRadius standing on the floor, u counting ribs round it (a seam
    /// column repeats the first at u = RibCount so the last rib closes), vertex alpha easing from 1 at
    /// the base to 0 at the top.
    /// </summary>
    private Mesh WallMesh()
    {
        int ribs = RibCount();
        var vertices = new List<Vector3>();
        var uvs = new List<Vector2>();
        var colors = new List<Color>();
        var triangles = new List<int>();
        for (int row = 0; row <= WallRows; row++)
        {
            float t = (float)row / WallRows;
            float alpha = 1f - t * t * (3f - 2f * t);
            for (int s = 0; s <= Segments; s++)
            {
                float angle = 2f * Mathf.PI * s / Segments;
                vertices.Add(new Vector3(wallRadius * Mathf.Cos(angle), t * wallHeight, wallRadius * Mathf.Sin(angle)));
                uvs.Add(new Vector2((float)s / Segments * ribs, t));
                colors.Add(new Color(1f, 1f, 1f, alpha));
            }
        }
        for (int row = 0; row < WallRows; row++)
        {
            int below = row * (Segments + 1);
            int above = below + Segments + 1;
            for (int s = 0; s < Segments; s++)
            {
                triangles.AddRange(new[] { below + s, above + s, above + s + 1, below + s, above + s + 1, below + s + 1 });
            }
        }
        return NewMesh("CockpitRoomWall", vertices, uvs.ToArray(), colors, triangles);
    }

    private Mesh NewMesh(string name, List<Vector3> vertices, Vector2[] uvs, List<Color> colors, List<int> triangles)
    {
        var mesh = new Mesh { name = name };
        mesh.SetVertices(vertices);
        mesh.uv = uvs;
        mesh.SetColors(colors);
        mesh.SetTriangles(triangles, 0);
        mesh.RecalculateBounds();
        owned.Add(mesh);
        return mesh;
    }
}
