using System.Collections.Generic;
using UnityEngine;

/// <summary>
/// Two extra cameras for recording instruction videos (Video/README.md), spawned at Play by
/// Tools/Video/Capture rig at Play, or added to a scene by hand:
///
///  * Overhead (tag VideoOverhead): orthographic, straight down over the swarm, with each drone's
///    camera frustum drawn for this camera only, as DroneFrustumGizmos draws it in the Scene view:
///    green while the drone is on the hull (its feed is on a screen), red while it is inside the
///    swarm. The frustums are lifted above the roofs (markerHeight) so no building hides them;
///    from straight above a lifted copy looks exactly like the original.
///  * Interface (tag VideoInterface): just behind and above the pilot's head, looking at the ring
///    of feed screens, with the eye camera's culling and clear so it sees what the pilot sees.
///
/// Both render into their own RenderTextures, so the Game view and the headset are untouched;
/// the Recorder copies those (Tools/Video/Record at Play). Nothing here touches the flight, the
/// drones or the screens. Keys while playing: F5 preview on/off, F6 overhead follows the swarm /
/// holds still, F7 overhead north-up / pilot-heading-up, PageUp/PageDown overhead zoom.
/// </summary>
[DefaultExecutionOrder(1000)]
public class VideoCaptureRig : MonoBehaviour
{
    public const string OverheadTag = "VideoOverhead";
    public const string InterfaceTag = "VideoInterface";

    [Header("Output")]
    public int width = 1920;
    public int height = 1080;
    [Tooltip("Draw small previews of both cameras in the Game view's corner (not recorded).")]
    public bool showPreview = true;
    [Tooltip("Leave the cameras disabled and let a caller Render() them (the headless VideoShotRunner).")]
    public bool manualRender = false;

    [Header("Overhead")]
    public bool overheadEnabled = true;
    [Tooltip("Half the picture's height, metres.")]
    public float orthoSize = 45f;
    public bool followSwarm = true;
    [Tooltip("Hold the overhead view over fixedCentre (x, z) instead of following the swarm.")]
    public bool useFixedCentre = false;
    public Vector2 fixedCentre;
    [Tooltip("Turn the picture so the pilot's forward (body yaw) points up, as in the video's schematics.")]
    public bool headingUp = true;
    [Tooltip("With headingUp off, the world heading that points up the picture (0 = north, 90 = east).")]
    public float upYaw = 0f;
    public float followSmoothTime = 0.6f;
    public float cameraHeight = 400f;
    [Tooltip("Height the markers are drawn at: above every roof so nothing hides them.")]
    public float markerHeight = 150f;
    [Tooltip("Marker size as a fraction of orthoSize.")]
    public float markerScale = 0.03f;
    [Tooltip("Length of each camera frustum, metres; 0 hides them.")]
    public float coneLength = 12f;
    [Range(0f, 1f)] public float coneAlpha = 0.3f;
    [Tooltip("Ring the pedestrians wearing hats (handy for editing; it shows where the answer is).")]
    public bool highlightHats = false;

    [Header("Interface")]
    public bool interfaceEnabled = true;
    public float interfaceBack = 0.9f;
    public float interfaceUp = 0.45f;
    public float interfaceFov = 90f;
    [Tooltip("The camera aims at the point this far ahead of the eye...")]
    public float interfaceLookAhead = 2f;
    [Tooltip("...and this far below eye level.")]
    public float interfaceLookDown = 0.25f;
    [Tooltip("Seconds the interface camera takes to follow the pilot's head round.")]
    public float headYawSmoothTime = 1.2f;
    [Tooltip("Draw a simple head and shoulders where the pilot sits (for this camera only).")]
    public bool drawPilot = true;

    // The video's palette (Video/tools/vidkit.py DRONE_HUES), so a recording matches the schematics.
    static readonly Color32[] Hues =
    {
        new Color32(255, 120, 98, 255), new Color32(255, 176, 59, 255), new Color32(240, 226, 90, 255),
        new Color32(140, 220, 110, 255), new Color32(86, 214, 180, 255), new Color32(86, 190, 242, 255),
        new Color32(120, 140, 255, 255), new Color32(186, 120, 255, 255), new Color32(240, 112, 200, 255),
        new Color32(200, 200, 210, 255),
    };
    static readonly Color HiddenColour = new Color32(110, 116, 128, 255);
    // DroneFrustumGizmos' colours: on the hull (feed shown) and inside the swarm.
    static readonly Color EdgeGreen = new Color(0.2f, 1.0f, 0.4f, 1.0f);
    static readonly Color InsideRed = new Color(1.0f, 0.25f, 0.25f, 1.0f);
    static readonly Color HatColour = new Color32(255, 214, 64, 255);

    Camera overhead, iface;
    RenderTexture overheadRT, ifaceRT;
    Mesh arrowMesh, discMesh, pyramidMesh, pyramidWire, ringMesh, sphereMesh, capsuleMesh;
    Material markerMat, pilotMat;
    MaterialPropertyBlock mpb;

    readonly List<GameObject> drones = new List<GameObject>();
    float nextDroneScan, nextWalkerScan, nextEyeSync;
    SpecialWalker[] hatWalkers = new SpecialWalker[0];
    Vector3 followVel;
    Vector3 followPos;
    bool followSeeded;
    float smoothHeadYaw, headYawVel;
    bool headSeeded;
    Transform eye;
    Camera eyeCam;

    public Camera OverheadCamera => overhead;
    public Camera InterfaceCamera => iface;
    public RenderTexture OverheadTexture => overheadRT;
    public RenderTexture InterfaceTexture => ifaceRT;

    void Start()
    {
        mpb = new MaterialPropertyBlock();
        var sprite = Shader.Find("Sprites/Default");
        markerMat = new Material(sprite) { renderQueue = 3100 };
        pilotMat = new Material(sprite) { color = new Color32(70, 76, 92, 255) };
        arrowMesh = BuildArrow();
        discMesh = BuildDisc(1f, 24);
        ringMesh = BuildRing(0.8f, 1f, 32);
        pyramidMesh = BuildPyramid(false);
        pyramidWire = BuildPyramid(true);
        sphereMesh = PrimitiveMesh(PrimitiveType.Sphere);
        capsuleMesh = PrimitiveMesh(PrimitiveType.Capsule);

        if (overheadEnabled)
        {
            overhead = MakeCamera("Video Overhead Camera", OverheadTag, out overheadRT);
            overhead.orthographic = true;
            overhead.orthographicSize = orthoSize;
            overhead.nearClipPlane = 1f;
            overhead.farClipPlane = cameraHeight + 200f;
            int ui = LayerMask.NameToLayer("UI");
            overhead.cullingMask = ui >= 0 ? ~(1 << ui) : ~0;
            overhead.clearFlags = CameraClearFlags.SolidColor;
            overhead.backgroundColor = new Color32(35, 47, 38, 255);
        }
        if (interfaceEnabled)
        {
            iface = MakeCamera("Video Interface Camera", InterfaceTag, out ifaceRT);
            iface.fieldOfView = interfaceFov;
            iface.nearClipPlane = 0.05f;
        }
    }

    Camera MakeCamera(string name, string tag, out RenderTexture rt)
    {
        var go = new GameObject(name);
        go.transform.SetParent(transform, false);
        try { go.tag = tag; }
        catch (UnityException)
        {
            Debug.LogWarning($"[VideoCaptureRig] Tag '{tag}' is not defined; run Tools/Video/Add capture tags " +
                             "so the Recorder can find this camera.");
        }
        var cam = go.AddComponent<Camera>();
        cam.enabled = !manualRender;
        cam.stereoTargetEye = StereoTargetEyeMask.None;
        cam.allowHDR = false;
        cam.allowMSAA = true;
        rt = new RenderTexture(width, height, 24, RenderTextureFormat.ARGB32) { antiAliasing = 4, name = name };
        rt.Create();
        cam.targetTexture = rt;
        return cam;
    }

    void Update()
    {
        if (Input.GetKeyDown(KeyCode.F5)) showPreview = !showPreview;
        if (Input.GetKeyDown(KeyCode.F6)) followSwarm = !followSwarm;
        if (Input.GetKeyDown(KeyCode.F7)) headingUp = !headingUp;
        if (Input.GetKey(KeyCode.PageUp)) orthoSize = Mathf.Max(8f, orthoSize * (1f - 0.8f * Time.unscaledDeltaTime));
        if (Input.GetKey(KeyCode.PageDown)) orthoSize = Mathf.Min(600f, orthoSize * (1f + 0.8f * Time.unscaledDeltaTime));
    }

    void LateUpdate()
    {
        if (Time.unscaledTime >= nextDroneScan)
        {
            nextDroneScan = Time.unscaledTime + 1f;
            drones.Clear();
            drones.AddRange(GameObject.FindGameObjectsWithTag("DroneBase"));
            drones.Sort((a, b) => string.CompareOrdinal(a.name, b.name));
        }
        if (highlightHats && Time.unscaledTime >= nextWalkerScan)
        {
            nextWalkerScan = Time.unscaledTime + 2f;
            hatWalkers = FindObjectsOfType<SpecialWalker>();
        }
        if (overhead != null) UpdateOverhead();
        if (iface != null) UpdateInterface();
    }

    // -----------------------------------------------------------------------------------------
    void UpdateOverhead()
    {
        Vector3 centre = Vector3.zero;
        int n = 0;
        foreach (var d in drones)
        {
            if (!IsAlive(d, out var parent)) continue;
            centre += parent.position;
            n++;
        }
        if (n > 0) centre /= n;
        if (n > 0 && (followSwarm || !followSeeded))
        {
            if (!followSeeded) { followPos = centre; followSeeded = true; }
            followPos = Vector3.SmoothDamp(followPos, centre, ref followVel, followSmoothTime, Mathf.Infinity,
                                           Time.unscaledDeltaTime);
        }
        float yaw = headingUp && PyUniSharingFast.BodyYawValid ? PyUniSharingFast.BodyYawDegrees : upYaw;
        overhead.orthographicSize = orthoSize;
        Vector3 at = useFixedCentre ? new Vector3(fixedCentre.x, 0f, fixedCentre.y) : followPos;
        overhead.transform.position = new Vector3(at.x, cameraHeight, at.z);
        overhead.transform.rotation = Quaternion.Euler(90f, yaw, 0f);

        float s = orthoSize * markerScale;
        bool hullMode = HullModeActive();
        for (int i = 0; i < drones.Count; i++)
        {
            var d = drones[i];
            if (!IsAlive(d, out var parent)) continue;
            Transform fpv = d.transform.Find("FPV");
            Camera fcam = fpv != null ? fpv.GetComponent<Camera>() : null;
            Vector3 fwd = fpv != null ? fpv.forward : parent.forward;
            fwd.y = 0f;
            if (fwd.sqrMagnitude < 1e-6f) fwd = Vector3.forward;
            var rot = Quaternion.LookRotation(fwd.normalized, Vector3.up);
            var pos = new Vector3(parent.position.x, markerHeight, parent.position.z);
            // As DroneFrustumGizmos: green on the hull (BoundaryEstimate), red inside.
            bool edge = true;
            if (hullMode && SwarmRegistry.TryGet(d, out var entry) && entry.attitude != null)
                edge = entry.attitude.BoundaryEstimate;
            Color c = edge ? EdgeGreen : InsideRed;
            if (coneLength > 0f && fpv != null)
            {
                float vfov = fcam != null ? fcam.fieldOfView : 46.2f;
                float aspect = fcam != null && fcam.aspect > 0f ? fcam.aspect : 16f / 9f;
                float tanV = Mathf.Tan(vfov * 0.5f * Mathf.Deg2Rad);
                var frus = Matrix4x4.TRS(pos - Vector3.up * 2f, fpv.rotation,
                                         new Vector3(tanV * aspect * coneLength, tanV * coneLength, coneLength));
                Draw(pyramidMesh, frus, new Color(c.r, c.g, c.b, coneAlpha), overhead);
                Draw(pyramidWire, frus, c, overhead);
            }
            Draw(discMesh, Matrix4x4.TRS(pos, rot, Vector3.one * s * 1.25f), new Color(0.04f, 0.05f, 0.06f, 1f), overhead);
            Draw(discMesh, Matrix4x4.TRS(pos + Vector3.up * 0.1f, rot, Vector3.one * s), Color.white, overhead);
            Draw(arrowMesh, Matrix4x4.TRS(pos + Vector3.up * 0.2f, rot, Vector3.one * s), c, overhead);
        }
        if (highlightHats)
        {
            foreach (var w in hatWalkers)
            {
                if (w == null) continue;
                var p = w.HatCentre;
                Draw(ringMesh, Matrix4x4.TRS(new Vector3(p.x, markerHeight, p.z), Quaternion.identity,
                                             Vector3.one * s * 1.6f), HatColour, overhead);
            }
        }
    }

    // -----------------------------------------------------------------------------------------
    void UpdateInterface()
    {
        if (eye == null)
        {
            var rig = FindObjectOfType<OVRCameraRig>();
            if (rig != null && rig.centerEyeAnchor != null) eye = rig.centerEyeAnchor;
            else if (Camera.main != null) eye = Camera.main.transform;
            if (eye == null) return;
            eyeCam = eye.GetComponent<Camera>();
        }
        // CockpitRoom and the SPHERE display change the eye camera's clear at runtime, so keep in step.
        if (eyeCam != null && Time.unscaledTime >= nextEyeSync)
        {
            nextEyeSync = Time.unscaledTime + 1f;
            iface.cullingMask = eyeCam.cullingMask;
            iface.clearFlags = eyeCam.clearFlags;
            iface.backgroundColor = eyeCam.backgroundColor;
            iface.farClipPlane = Mathf.Max(50f, eyeCam.farClipPlane);
        }
        Vector3 f = eye.forward;
        f.y = 0f;
        if (f.sqrMagnitude < 1e-6f) f = Vector3.forward;
        float headYaw = Mathf.Atan2(f.x, f.z) * Mathf.Rad2Deg;
        if (!headSeeded) { smoothHeadYaw = headYaw; headSeeded = true; }
        smoothHeadYaw = Mathf.SmoothDampAngle(smoothHeadYaw, headYaw, ref headYawVel, headYawSmoothTime, Mathf.Infinity,
                                              Time.unscaledDeltaTime);
        var yawRot = Quaternion.Euler(0f, smoothHeadYaw, 0f);
        Vector3 fwd = yawRot * Vector3.forward;
        Vector3 e = eye.position;
        iface.transform.position = e - fwd * interfaceBack + Vector3.up * interfaceUp;
        iface.fieldOfView = interfaceFov;
        iface.transform.rotation = Quaternion.LookRotation(
            (e + fwd * interfaceLookAhead - Vector3.up * interfaceLookDown) - iface.transform.position);

        if (drawPilot)
        {
            // On the screens' layer, which the eye camera's culling mask (copied above) always includes.
            int ui = LayerMask.NameToLayer("UI");
            int layer = ui >= 0 ? ui : eye.gameObject.layer;
            var head = Matrix4x4.TRS(e - Vector3.up * 0.05f, yawRot, new Vector3(0.17f, 0.21f, 0.19f));
            Graphics.DrawMesh(sphereMesh, head, pilotMat, layer, iface);
            var shoulders = Matrix4x4.TRS(e - Vector3.up * 0.33f, yawRot * Quaternion.Euler(0f, 0f, 90f),
                                          new Vector3(0.17f, 0.22f, 0.14f));
            Graphics.DrawMesh(capsuleMesh, shoulders, pilotMat, layer, iface);
        }
    }

    // -----------------------------------------------------------------------------------------
    static bool IsAlive(GameObject d, out Transform parent)
    {
        parent = null;
        if (d == null || !d.activeInHierarchy) return false;
        if (!SwarmRegistry.TryGet(d, out var e) || e.droneParent == null) return false;
        parent = e.droneParent;
        var vc = e.velocityControl;
        return vc == null || vc.State == null || vc.State.IsAlive;
    }

    static int DroneIndex(GameObject d, int fallback)
    {
        // "Drone N": colour by N, so a drone keeps its colour when another is lost.
        string name = d.name;
        int sp = name.LastIndexOf(' ');
        if (sp >= 0 && int.TryParse(name.Substring(sp + 1), out int k)) return Mathf.Max(0, k);
        return fallback;
    }

    static bool HullModeActive()
    {
        var sm = SwarmManager.Instance;
        if (sm == null) return false;
        var algo = sm.GetSelectedAttitudeAlgorithm();
        if (algo != SwarmManager.AttitudeAlgorithm.LOCAL_CONVEXHULL && algo != SwarmManager.AttitudeAlgorithm.GLOBAL_CONVEXHULL)
            return false;
        return !SwarmRegistry.TryGetLoneDrone(out _, out _);
    }

    void Draw(Mesh mesh, Matrix4x4 m, Color c, Camera cam)
    {
        mpb.SetColor("_Color", c);
        Graphics.DrawMesh(mesh, m, markerMat, 0, cam, 0, mpb, false, false);
    }

    void OnGUI()
    {
        if (!showPreview) return;
        float w = Screen.width * 0.22f, h = w * height / width, m = 10f;
        float y = m;
        if (overheadRT != null) { GUI.DrawTexture(new Rect(Screen.width - w - m, y, w, h), overheadRT); y += h + m; }
        if (ifaceRT != null) GUI.DrawTexture(new Rect(Screen.width - w - m, y, w, h), ifaceRT);
        GUI.Label(new Rect(Screen.width - w - m, y + h + 2, w, 22),
                  "F5 preview  F6 follow  F7 heading-up  PgUp/PgDn zoom");
    }

    void OnDestroy()
    {
        if (overheadRT != null) overheadRT.Release();
        if (ifaceRT != null) ifaceRT.Release();
    }

    // -- meshes, all in the XZ plane, +Z forward ---------------------------------------------
    static Mesh BuildArrow()
    {
        var m = new Mesh { name = "VideoArrow" };
        m.vertices = new[] { new Vector3(0, 0, 2.6f), new Vector3(-0.85f, 0, 0.55f), new Vector3(0.85f, 0, 0.55f) };
        m.triangles = new[] { 0, 2, 1 };
        m.RecalculateBounds();
        return m;
    }

    static Mesh BuildDisc(float r, int n)
    {
        var v = new Vector3[n + 1];
        var t = new int[n * 3];
        for (int i = 0; i < n; i++)
        {
            float a = 2f * Mathf.PI * i / n;
            v[i + 1] = new Vector3(Mathf.Sin(a) * r, 0, Mathf.Cos(a) * r);
            t[i * 3] = 0; t[i * 3 + 1] = i + 1; t[i * 3 + 2] = (i + 1) % n + 1;
        }
        var m = new Mesh { name = "VideoDisc", vertices = v, triangles = t };
        m.RecalculateBounds();
        return m;
    }

    static Mesh BuildRing(float r0, float r1, int n)
    {
        var v = new Vector3[n * 2];
        var t = new int[n * 6];
        for (int i = 0; i < n; i++)
        {
            float a = 2f * Mathf.PI * i / n;
            var d = new Vector3(Mathf.Sin(a), 0, Mathf.Cos(a));
            v[i * 2] = d * r0;
            v[i * 2 + 1] = d * r1;
            int j = (i + 1) % n;
            t[i * 6] = i * 2; t[i * 6 + 1] = i * 2 + 1; t[i * 6 + 2] = j * 2 + 1;
            t[i * 6 + 3] = i * 2; t[i * 6 + 4] = j * 2 + 1; t[i * 6 + 5] = j * 2;
        }
        var m = new Mesh { name = "VideoRing", vertices = v, triangles = t };
        m.RecalculateBounds();
        return m;
    }

    // The unit frustum DroneFrustumGizmos draws: apex at the origin, base at z = 1 spanning +-1 in x and y.
    static Mesh BuildPyramid(bool wire)
    {
        var v = new[] { Vector3.zero, new Vector3(-1, -1, 1), new Vector3(1, -1, 1), new Vector3(1, 1, 1), new Vector3(-1, 1, 1) };
        var m = new Mesh { name = wire ? "VideoFrustumWire" : "VideoFrustum", vertices = v };
        if (wire)
            m.SetIndices(new[] { 0, 1, 0, 2, 0, 3, 0, 4, 1, 2, 2, 3, 3, 4, 4, 1 }, MeshTopology.Lines, 0);
        else
            m.triangles = new[] { 0, 2, 1, 0, 3, 2, 0, 4, 3, 0, 1, 4, 1, 2, 3, 1, 3, 4 };
        m.RecalculateBounds();
        return m;
    }

    static Mesh PrimitiveMesh(PrimitiveType type)
    {
        var go = GameObject.CreatePrimitive(type);
        var mesh = go.GetComponent<MeshFilter>().sharedMesh;
        Destroy(go);
        return mesh;
    }
}
