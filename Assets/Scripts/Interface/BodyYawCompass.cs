using UnityEngine;
using UnityEngine.Rendering;
using UnityEngine.SceneManagement;

/// <summary>
/// A small flat compass arrow lying below and a little in front of the pilot, pointing along the body
/// yaw (<see cref="PyUniSharingFast.BodyYawDegrees"/>) — the heading that aims the panorama and the
/// VR velocity frame, not wherever the head happens to be looking. Glance down to see which way
/// "forward" is.
///
/// The direction is expressed in the arena's display frame, not as a world heading: a yaw ψ sits at
/// azimuth −ψ, i.e. along <c>(cos ψ, 0, −sin ψ)</c> — the same mapping
/// <c>PyUniSharingFast.UpdateCurvedScreenPose</c> and <c>ScreenSpawn</c>'s circle layouts use, so the
/// arrow points at the panorama centre after a calibration and at a drone's screen when that drone
/// flies on the body heading. It is pushed out from the OVRCameraRig along that same direction, so
/// it stays in front of the pilot as the body heading turns.
///
/// It must live on the feed-screen layer (<see cref="ScreenSpawn.screenLayerName"/>): the eye
/// cameras are culled to that layer and draw nothing else. It carries no collider, so the scene-plane
/// raycast cannot hit it.
///
/// Added automatically to any scene with an Arena; add one by hand to tune it (the bootstrap then
/// leaves that one alone), or disable it to hide the arrow.
/// </summary>
public class BodyYawCompass : MonoBehaviour
{
    [Tooltip("How far below the screens' centre height (the Arena origin, which is the pilot's " +
             "eye level) the arrow lies, in metres. Below every layout's bottom row so it never " +
             "sits in front of a feed.")]
    public float depthBelowScreens = 1.1f;

    [Tooltip("Horizontal distance from the OVRCameraRig to the arrow's centre, along the body " +
             "heading, in metres.")]
    public float forwardOffset = 0.5f;

    [Tooltip("Tip-to-tail length of the arrow, in metres.")]
    public float arrowLength = 0.45f;

    [Tooltip("Arrow colour at the tip. Blended over the black arena, so alpha sets how bright it " +
             "reads; a low value keeps it a faint shadow rather than a bright mark.")]
    public Color color = new Color(1f, 1f, 1f, 0.3f);

    [Tooltip("Brightness at the tail as a fraction of the tip's. The arrow fades linearly from " +
             "tip to tail, so the tip reads as the pointing end.")]
    [Range(0f, 1f)] public float tailFade = 0.15f;

    [Tooltip("Width of the soft edge that fades the arrow out to nothing, as a fraction of its " +
             "length. 0 = a hard edge.")]
    [Range(0f, 0.3f)] public float edgeSoftness = 0.08f;

    private GameObject arena;
    private Transform rig;
    private GameObject visual;
    private Material material;
    // What the current mesh was built with, so an inspector change in Play rebuilds it.
    private float builtTailFade = -1f;
    private float builtEdgeSoftness = -1f;

    [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.BeforeSceneLoad)]
    private static void RegisterBootstrap()
    {
        // -= first so a disabled domain reload cannot stack the handler.
        SceneManager.sceneLoaded -= OnSceneLoaded;
        SceneManager.sceneLoaded += OnSceneLoaded;
    }

    private static void OnSceneLoaded(Scene scene, LoadSceneMode mode)
    {
        if (FindObjectOfType<BodyYawCompass>(true) != null) return;
        if (GameObject.FindGameObjectWithTag("Arena") == null) return;
        new GameObject("BodyYawCompass").AddComponent<BodyYawCompass>();
    }

    private void OnEnable()
    {
        if (visual != null) visual.SetActive(true);
    }

    private void OnDisable()
    {
        if (visual != null) visual.SetActive(false);
    }

    private void OnDestroy()
    {
        if (visual != null)
        {
            Destroy(visual.GetComponent<MeshFilter>().sharedMesh);
            Destroy(visual);
        }
        if (material != null) Destroy(material);
    }

    private void LateUpdate()
    {
        if (arena == null)
        {
            arena = GameObject.FindGameObjectWithTag("Arena");
            if (arena == null) return;
        }
        if (rig == null)
        {
            OVRCameraRig ovrRig = FindObjectOfType<OVRCameraRig>();
            if (ovrRig != null) rig = ovrRig.transform;
        }

        if (visual == null) BuildVisual();

        // 0 deg is a real heading, so an unseeded body yaw would point the arrow north with
        // confidence. Hide it until PyUniSharingFast has seeded one.
        bool show = PyUniSharingFast.BodyYawValid;
        if (visual.activeSelf != show) visual.SetActive(show);
        if (!show) return;

        Quaternion rotation = Quaternion.Euler(0f, PyUniSharingFast.BodyYawDegrees + 90f, 0f);

        // Horizontally from the rig (ScreenSpawn puts it on the Arena, but it is the pilot's actual
        // position), vertically from the Arena, which is where the screens hang.
        Vector3 position = rig != null ? rig.position : arena.transform.position;
        position.y = arena.transform.position.y - depthBelowScreens;
        position += rotation * Vector3.forward * forwardOffset;

        visual.transform.SetPositionAndRotation(position, rotation);
        visual.transform.localScale = Vector3.one * arrowLength;
        material.color = color;
        if (tailFade != builtTailFade || edgeSoftness != builtEdgeSoftness)
        {
            MeshFilter mf = visual.GetComponent<MeshFilter>();
            Destroy(mf.sharedMesh);
            mf.sharedMesh = BuildMesh();
        }
    }

    private void BuildVisual()
    {
        visual = new GameObject("BodyYawCompassArrow");

        ScreenSpawn screenSpawn = FindObjectOfType<ScreenSpawn>();
        int layer = LayerMask.NameToLayer(screenSpawn != null ? screenSpawn.screenLayerName : "UI");
        if (layer >= 0)
        {
            visual.layer = layer;
        }
        else
        {
            Debug.LogWarning("[BodyYawCompass] Feed-screen layer not found; the headset eye " +
                             "cameras will not draw the arrow.");
        }

        visual.AddComponent<MeshFilter>().sharedMesh = BuildMesh();

        // Sprites/Default: unlit, alpha-blended, double-sided, and always included in builds.
        material = new Material(Shader.Find("Sprites/Default")) { color = color };
        MeshRenderer mr = visual.AddComponent<MeshRenderer>();
        mr.sharedMaterial = material;
        mr.shadowCastingMode = ShadowCastingMode.Off;
        mr.receiveShadows = false;
        mr.lightProbeUsage = LightProbeUsage.Off;
        mr.reflectionProbeUsage = ReflectionProbeUsage.Off;
    }

    /// <summary>
    /// A unit-length dart in the XZ plane, tip along local +Z, centred on the origin. The shading is
    /// all vertex colour (Sprites/Default multiplies it by the material colour): alpha falls off
    /// linearly from tip to tail, and a feather strip round the outline fades from the edge's own
    /// alpha to zero, so the arrow has no hard edge. Winding is irrelevant: Sprites/Default does not
    /// cull.
    /// </summary>
    private Mesh BuildMesh()
    {
        builtTailFade = tailFade;
        builtEdgeSoftness = edgeSoftness;

        // Clockwise seen from above, so each edge's outward normal is (-dz, dx).
        Vector3[] outline =
        {
            new Vector3(0f, 0f, 0.5f),      // tip
            new Vector3(0.28f, 0f, -0.5f),  // right barb
            new Vector3(0f, 0f, -0.25f),    // tail notch
            new Vector3(-0.28f, 0f, -0.5f), // left barb
        };
        int n = outline.Length;
        bool feather = edgeSoftness > 0f;

        var vertices = new Vector3[feather ? 2 * n : n];
        var colors = new Color[vertices.Length];
        for (int i = 0; i < n; i++)
        {
            vertices[i] = outline[i];
            colors[i] = new Color(1f, 1f, 1f, Mathf.Lerp(tailFade, 1f, outline[i].z + 0.5f));
        }

        var triangles = new System.Collections.Generic.List<int> { 0, 1, 2, 0, 2, 3 };

        if (feather)
        {
            for (int i = 0; i < n; i++)
            {
                Vector3 prev = outline[(i + n - 1) % n];
                Vector3 next = outline[(i + 1) % n];
                Vector3 n1 = EdgeNormal(prev, outline[i]);
                Vector3 n2 = EdgeNormal(outline[i], next);
                // Mitred so the strip keeps its width along both edges; capped so the sharp tip
                // does not throw a long spike.
                Vector3 miter = (n1 + n2).normalized;
                float length = Mathf.Min(edgeSoftness / Mathf.Max(Vector3.Dot(miter, n1), 1e-3f),
                                         2.5f * edgeSoftness);
                vertices[n + i] = outline[i] + miter * length;
                colors[n + i] = new Color(1f, 1f, 1f, 0f);
            }
            for (int i = 0; i < n; i++)
            {
                int j = (i + 1) % n;
                triangles.AddRange(new[] { i, j, n + j, i, n + j, n + i });
            }
        }

        var mesh = new Mesh { name = "BodyYawCompass" };
        mesh.vertices = vertices;
        mesh.colors = colors;
        mesh.SetTriangles(triangles, 0);
        mesh.RecalculateNormals();
        mesh.RecalculateBounds();
        return mesh;
    }

    private static Vector3 EdgeNormal(Vector3 from, Vector3 to)
    {
        Vector3 d = to - from;
        return new Vector3(-d.z, 0f, d.x).normalized;
    }
}
