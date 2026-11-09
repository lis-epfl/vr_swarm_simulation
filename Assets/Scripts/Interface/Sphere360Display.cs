using System;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Rendering;

/// <summary>
/// <see cref="InterfaceManager.DisplayMode.SPHERE"/>: one drone carrying a 360 camera, shown on a sphere round the
/// pilot, the way the DJI Avata 360 and the Antigravity A1 show theirs in goggles (docs/single_drone_360_camera.md,
/// build B). A hidden camera at the drone renders a cube map at video rate; the pilot stays in the arena inside a
/// sphere that looks the cube up in each pixel's direction, so the head turns freely at the headset's rate inside the
/// latest cube. Points worth keeping:
/// <list type="bullet">
/// <item>The cube's faces stay aligned with the world whatever the camera's rotation (measured), so the display does
/// the turning: <see cref="DisplayToCity"/> takes a direction in the arena to a direction in the city. It includes the
/// display's 90 deg (a screen for world heading psi sits at display yaw psi + 90, as BodyYawCompass draws it).</item>
/// <item>The sphere's shader looks up by each eye's own view ray, so the view is mono and at infinity and leaning the
/// head changes nothing: turning only, as with any 360 video.</item>
/// <item>Physics runs at 50 Hz with no Rigidbody interpolation, so the camera's position and the drone's heading are
/// interpolated here (Unity's own scheme, one physics step behind), which leaves the flight untouched. The rig itself
/// is still turned in 50 Hz steps by PyUniSharingFast's lone-drone lock; the sphere takes each step back, so the city
/// turns at the interpolated rate and nothing judders on a 72 Hz display.</item>
/// <item>A cube RenderTexture must be a power of two: at any other size it fails to create, yet RenderToCubemap
/// still returns true and draws black. The size is therefore an enum, and creation is checked.</item>
/// <item>Each face counts as one FPV render in FPVCameraScript's counters (and so in _perf.csv's fpvRenders): a
/// 2048 cube costs about what ten feeds do, 6.9 ms of main thread in BlockShapedPatchCityWorld.</item>
/// </list>
/// </summary>
[DefaultExecutionOrder(-100)]   // FixedUpdate before VelocityControl's: see RecordPhysicsStep
public class Sphere360Display : MonoBehaviour
{
    // Values spelled out: Unity serialises enum fields by integer.
    public enum CubeSize
    {
        Px1024 = 1024,
        Px2048 = 2048,
        Px4096 = 4096
    }

    public enum HeadingMode
    {
        TurnWithDrone = 0,
        WorldFixed = 1
    }

    [Serializable]
    public class Settings
    {
        [Tooltip("Edge of each cube face, in pixels. 2048 is ~18 px/deg at a face's centre against the headset's ~22, " +
                 "about as many pixels on a walker as a feed at scale 1.3 gives (the write-up's section 6.2). 4096 " +
                 "does not fit one frame at 72 Hz without splitFaces.")]
        public CubeSize cubeSize = CubeSize.Px2048;

        [Tooltip("Refreshes of the whole cube per second. 36 lands on every second frame at 72 fps (30 would land " +
                 "unevenly) and on every frame at the runtime's 36 fps half rate.")]
        [Range(1f, 72f)] public float refreshHz = 36f;

        [Tooltip("Render half the faces on one frame and the rest on the next, from the same position, so a refresh " +
                 "costs two half frames rather than one whole one. On by default: replaying a single-drone flight " +
                 "in the patch city on a 72 Hz grid, a 2048 cube took the main thread's p95 / p99 from 11.1 / 12.3 " +
                 "ms of the 13.9 ms frame to 6.9 / 7.9. The price is that the halves are a frame apart in time, " +
                 "which only a moving walker could show (~2 cm).")]
        public bool splitFaces = true;

        [Tooltip("TurnWithDrone: the view turns with the drone's heading, so the stick's forward stays in front of a " +
                 "seated pilot, as the single drone's screen does. WorldFixed: the city stays put in the room and the " +
                 "pilot turns their head to look where the drone flies; BodyYawCompass marks the heading. The " +
                 "calibrate key recentres either on the heading.")]
        public HeadingMode heading = HeadingMode.TurnWithDrone;

        [Tooltip("0: the whole sphere, as the products show it. Above 0: only a band this many degrees either side " +
                 "of the gimbal pitch is shown (23.1 is the feeds' half height), for comparing coverage with the " +
                 "swarm's ring of feeds. The recorder's scoring applies the same band.")]
        [Range(0f, 90f)] public float bandHalfDeg = 0f;
    }

    public const string ShaderName = "Swarm/Sphere 360 View";
    private const float MaxSphereRadius = 100f;
    private const int AllFaces = 63;
    // +X, -X, +Y then -Y, +Z, -Z: the two halves measured 3.8 and 3.6 ms of main thread in the patch city.
    private const int FirstHalf = 0b000111;
    private const int SecondHalf = 0b111000;

    /// <summary>True while a sphere is shown (read by ExperimentRecorder).</summary>
    public static bool Active { get; private set; }

    public static Sphere360Display Instance { get; private set; }

    /// <summary>
    /// Takes a direction in the arena (Unity world, where the head and the gaze are logged) to the direction in the
    /// city the sphere shows there.
    /// </summary>
    public static Quaternion DisplayToCity { get; private set; } = Quaternion.identity;

    /// <summary>
    /// The sphere's yaw in degrees: a direction at yaw theta in the arena shows the city at yaw theta - SphereYaw.
    /// _head.csv logs it, so the analysis can turn the head and the gaze into directions in the city.
    /// </summary>
    public static float SphereYawDegrees { get; private set; }

    /// <summary>The drone carrying the camera.</summary>
    public GameObject Drone { get; private set; }

    /// <summary>Where the cube now on the sphere was rendered from (valid once <see cref="HasRendered"/>).</summary>
    public Vector3 LastRenderPosition { get; private set; }

    public bool HasRendered { get; private set; }

    /// <summary>Complete cube refreshes so far.</summary>
    public long Refreshes { get; private set; }

    public Settings Config => settings;

    /// <summary>The cube's edge in pixels, or 0 before it exists.</summary>
    public int CubeEdge => cube != null ? cube.width : 0;

    /// <summary>The cube itself, for checks that read it back.</summary>
    public RenderTexture Cube => cube;

    private Settings settings;
    private Rigidbody body;
    private VelocityControl velocity;
    private FPVCameraScript fpvScript;
    private Camera cubeCamera;
    private RenderTexture cube;
    private GameObject sphere;
    private Material material;
    private OVRCameraRig rig;
    private GameObject arena;
    private CockpitRoom cockpitRoom;
    private readonly List<Camera> disabledFpvCameras = new List<Camera>();
    private readonly List<Renderer> hiddenArenaRenderers = new List<Renderer>();
    private readonly List<(Camera camera, CameraClearFlags flags, Color background)> clearedCameras =
        new List<(Camera, CameraClearFlags, Color)>();

    // The last two physics states (see RecordPhysicsStep) and what they interpolate to this frame.
    private bool havePrevious;
    private Vector3 previousPosition;
    private float previousHeadingDeg;
    private Vector3 position;
    private float headingDeg;

    private float lastRefreshTime = float.NegativeInfinity;
    private float smoothedFrameTime = 1f / 72f;
    private bool secondHalfDue;
    private Vector3 halfRenderPosition;
    private bool renderFailed;

    private int latchedCalibration = -1;
    private float worldFixedOffset;

    [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.SubsystemRegistration)]
    private static void ResetStatics()
    {
        // Statics survive a Play session in the editor when domain reload is off.
        Active = false;
        Instance = null;
        DisplayToCity = Quaternion.identity;
        SphereYawDegrees = 0f;
    }

    /// <summary>
    /// Starts the display for the swarm InterfaceManager was handed, which should be one drone. Called once, from
    /// <see cref="InterfaceManager.SetSwarm"/>, straight after the drones are instantiated.
    /// </summary>
    public void Begin(List<GameObject> swarm, Settings config)
    {
        if (Active)
        {
            Debug.LogWarning("Sphere360Display: already showing a drone; ignoring a second Begin.", this);
            return;
        }
        settings = config ?? new Settings();

        Drone = null;
        if (swarm != null)
        {
            foreach (GameObject d in swarm)
            {
                if (d != null)
                {
                    Drone = d;
                    break;
                }
            }
        }
        if (Drone == null)
        {
            Debug.LogError("Sphere360Display: no drone to carry the 360 camera.", this);
            enabled = false;
            return;
        }
        if (swarm.Count != 1)
        {
            // InterfaceManager never hands it a swarm (swarmSpawn spawns one drone under SPHERE); a 360 camera is a
            // single-drone condition, so refuse rather than show one drone of many.
            Debug.LogError($"Sphere360Display: SPHERE is a one-drone display, but was handed {swarm.Count} drones. " +
                           "Not starting.", this);
            Drone = null;
            enabled = false;
            return;
        }

        Transform droneParent = Drone.transform.Find("DroneParent");
        Transform fpv = Drone.transform.Find("FPV");
        Camera fpvCamera = fpv != null ? fpv.GetComponent<Camera>() : null;
        body = droneParent != null ? droneParent.GetComponent<Rigidbody>() : null;
        velocity = droneParent != null ? droneParent.GetComponent<VelocityControl>() : null;
        fpvScript = fpv != null ? fpv.GetComponent<FPVCameraScript>() : null;
        if (body == null || velocity == null || fpvCamera == null)
        {
            Debug.LogError($"Sphere360Display: {Drone.name} lacks a DroneParent Rigidbody/VelocityControl or an FPV " +
                           "camera.", this);
            enabled = false;
            return;
        }

        if (!CreateCube(fpvCamera)) return;
        DisableFeedCameras(swarm);
        TakeOverBackdrop();
        CreateSphere();

        PyUniSharingFast stitching = FindObjectOfType<PyUniSharingFast>();
        if (stitching != null && stitching.isActiveAndEnabled && stitching.StitchingEnabled)
        {
            Debug.LogWarning("Sphere360Display: stitching is on, but a lone drone gives the stitcher nothing. Untick " +
                             "PyUniSharingFast's enableImageWriting and enablePanoramaReading.", this);
        }

        Instance = this;
        Active = true;
        Debug.Log($"Sphere360Display: {Drone.name} carries a {CubeEdge}-pixel cube at {settings.refreshHz:F0} Hz" +
                  $"{(settings.splitFaces ? ", split over two frames" : "")}, heading {settings.heading}" +
                  $"{(settings.bandHalfDeg > 0f ? $", band +-{settings.bandHalfDeg:F1} deg" : "")}.", this);
    }

    private bool CreateCube(Camera fpvCamera)
    {
        // CopyFrom brings the FPV camera's culling mask (the city, without the UI layer the sphere is on), its 2000 m
        // far clip (the landmarks stand 1.25-1.45 km out), its near clip and its skybox clear. Not its tag: the FPV
        // camera is tagged MainCamera, which PyUniSharingFast falls back to.
        cubeCamera = new GameObject("Camera360").AddComponent<Camera>();
        cubeCamera.CopyFrom(fpvCamera);
        cubeCamera.enabled = false;
        cubeCamera.targetTexture = null;
        cubeCamera.allowHDR = false;
        cubeCamera.allowMSAA = false;
        cubeCamera.stereoTargetEye = StereoTargetEyeMask.None;
        cubeCamera.useOcclusionCulling = ProceduralCityBuilder.OcclusionUsable;
        cubeCamera.transform.rotation = Quaternion.identity;

        // Mips only where the cube is denser than the headset's ~22 px/deg, i.e. minified on the sphere: there they
        // stop it aliasing, as the feeds' mips do; at 2048 and below the sphere magnifies the cube and they would buy
        // nothing for their ~1 ms of GPU.
        int size = (int)settings.cubeSize;
        bool mips = size > 2048;
        cube = new RenderTexture(size, size, 24)
        {
            name = "Sphere360Cube",
            dimension = TextureDimension.Cube,
            useMipMap = mips,
            autoGenerateMips = mips,
            filterMode = mips ? FilterMode.Trilinear : FilterMode.Bilinear,
            wrapMode = TextureWrapMode.Clamp,
        };
        if (!cube.Create() || !cube.IsCreated())
        {
            Debug.LogError($"Sphere360Display: the {size}-pixel cube RenderTexture could not be created, so there is " +
                           "nothing to show. A cube must be a power of two.", this);
            enabled = false;
            return false;
        }
        return true;
    }

    // The prefab leaves every FPV camera enabled with no target texture, so with no ScreenSpawn to drive them they
    // would redraw the city every frame for nobody. Only the Camera component goes: FPVCameraScript keeps moving the
    // FPV transform, which WalkerPatrol's sleep test and PyUniSharingFast's lone-drone recentre still read.
    private void DisableFeedCameras(List<GameObject> swarm)
    {
        foreach (GameObject d in swarm)
        {
            Transform fpv = d != null ? d.transform.Find("FPV") : null;
            Camera c = fpv != null ? fpv.GetComponent<Camera>() : null;
            if (c == null || !c.enabled) continue;
            c.enabled = false;
            disabledFpvCameras.Add(c);
        }
    }

    /// <summary>
    /// Puts the rig in the arena, as ScreenSpawn.SpawnScreens does, and takes over what is behind the sphere:
    /// CockpitRoom off (the sphere covers it), the Arena box hidden and the eye cameras cleared to black, mirroring
    /// CockpitRoom.Engage. The sphere writes no depth, so either the skybox or the box would otherwise draw over it.
    /// </summary>
    private void TakeOverBackdrop()
    {
        GameObject player = GameObject.FindGameObjectWithTag("Player");
        rig = player != null ? player.GetComponent<OVRCameraRig>() : null;
        if (rig == null) rig = FindObjectOfType<OVRCameraRig>();
        arena = GameObject.FindGameObjectWithTag("Arena");
        if (rig == null) Debug.LogWarning("Sphere360Display: no OVRCameraRig in the scene.", this);
        if (arena == null) Debug.LogWarning("Sphere360Display: no GameObject tagged 'Arena' in the scene.", this);
        if (rig != null && arena != null) rig.transform.position = arena.transform.position;

        cockpitRoom = FindObjectOfType<CockpitRoom>();
        if (cockpitRoom != null && cockpitRoom.enabled) cockpitRoom.enabled = false;
        else cockpitRoom = null;

        if (arena != null)
        {
            foreach (Renderer r in arena.GetComponentsInChildren<Renderer>())
            {
                if (!r.enabled) continue;
                r.enabled = false;
                hiddenArenaRenderers.Add(r);
            }
        }
        if (rig != null)
        {
            foreach (Camera c in rig.GetComponentsInChildren<Camera>(true))
            {
                clearedCameras.Add((c, c.clearFlags, c.backgroundColor));
                c.clearFlags = CameraClearFlags.SolidColor;
                c.backgroundColor = Color.black;
            }
        }
    }

    private void CreateSphere()
    {
        Shader shader = Shader.Find(ShaderName);
        if (shader == null)
        {
            Debug.LogError($"Sphere360Display: shader '{ShaderName}' not found (a player build needs it in Always " +
                           "Included Shaders).", this);
            return;
        }
        material = new Material(shader) { name = "Sphere360View" };
        material.SetTexture("_Cube", cube);

        sphere = GameObject.CreatePrimitive(PrimitiveType.Sphere);
        sphere.name = "Sphere360";
        Collider c = sphere.GetComponent<Collider>();
        if (c != null) Destroy(c);

        // On the feed screens' layer, the only one the eye cameras draw.
        ScreenSpawn screens = GetComponent<ScreenSpawn>();
        string layerName = screens != null ? screens.screenLayerName : "UI";
        int layer = LayerMask.NameToLayer(layerName);
        if (layer >= 0) sphere.layer = layer;
        else Debug.LogWarning($"Sphere360Display: layer '{layerName}' does not exist; the eye cameras may not draw " +
                              "the sphere.", this);

        // The lookup is by direction, so the radius only has to sit inside the eye cameras' clip planes.
        float radius = MaxSphereRadius;
        if (rig != null)
        {
            foreach (Camera eye in rig.GetComponentsInChildren<Camera>(true))
            {
                radius = Mathf.Min(radius, 0.5f * eye.farClipPlane);
            }
        }
        sphere.transform.localScale = Vector3.one * (2f * radius);   // the primitive's radius is 0.5

        MeshRenderer mr = sphere.GetComponent<MeshRenderer>();
        mr.sharedMaterial = material;
        mr.shadowCastingMode = ShadowCastingMode.Off;
        mr.receiveShadows = false;
        mr.lightProbeUsage = LightProbeUsage.Off;
        mr.reflectionProbeUsage = ReflectionProbeUsage.Off;
        mr.allowOcclusionWhenDynamic = false;
        PlaceSphere();
    }

    /// <summary>
    /// Records the state the coming physics step starts from. Runs before VelocityControl's FixedUpdate (the execution
    /// order), so the heading setpoint is still the previous step's; together with the values read in LateUpdate this
    /// gives the last two physics states, as Rigidbody interpolation would.
    /// </summary>
    private void FixedUpdate()
    {
        if (!Active || body == null || velocity == null) return;
        previousPosition = body.position;
        previousHeadingDeg = velocity.HeadingSetpointRad * Mathf.Rad2Deg;
        havePrevious = true;
    }

    private void LateUpdate()
    {
        if (!Active) return;
        InterpolatePose();
        StepRefresh();
        UpdateDisplayRotation();
        PlaceSphere();
    }

    // The body's position (DroneParent), not the FPV mount 0.4 m forward of it, which would swing the eye round on
    // every turn: the products' lenses sit on the body's axis.
    private void InterpolatePose()
    {
        Vector3 current = body.position;
        float currentHeading = velocity.HeadingSetpointRad * Mathf.Rad2Deg;
        if (!havePrevious)
        {
            position = current;
            headingDeg = currentHeading;
            return;
        }
        float a = Time.fixedDeltaTime > 0f ? Mathf.Clamp01((Time.time - Time.fixedTime) / Time.fixedDeltaTime) : 1f;
        position = Vector3.Lerp(previousPosition, current, a);
        headingDeg = Mathf.LerpAngle(previousHeadingDeg, currentHeading, a);
    }

    // Due once a refresh interval has passed, less half a frame, which locks to every second frame at 72 fps and to
    // every frame at 36. A dead drone's camera holds its last picture.
    private void StepRefresh()
    {
        smoothedFrameTime = Mathf.Lerp(smoothedFrameTime, Time.unscaledDeltaTime, 0.1f);
        if (secondHalfDue)
        {
            secondHalfDue = false;
            RenderFaces(SecondHalf, halfRenderPosition);
            Refreshes++;
            return;
        }
        if (velocity.State != null && !velocity.State.IsAlive) return;
        float interval = 1f / Mathf.Max(1f, settings.refreshHz);
        if (Time.unscaledTime - lastRefreshTime < interval - 0.5f * smoothedFrameTime) return;
        lastRefreshTime = Time.unscaledTime;

        if (settings.splitFaces)
        {
            // Both halves from one position, so the static city meets itself across the split; only what moved in
            // the frame between them can show it.
            halfRenderPosition = position;
            RenderFaces(FirstHalf, halfRenderPosition);
            secondHalfDue = true;
        }
        else
        {
            RenderFaces(AllFaces, position);
            Refreshes++;
        }
    }

    private void RenderFaces(int mask, Vector3 from)
    {
        cubeCamera.transform.position = from;
        bool ok;
        if (fpvScript != null) fpvScript.SetOwnBodyHidden(true);
        try
        {
            ok = cubeCamera.RenderToCubemap(cube, mask);
        }
        finally
        {
            if (fpvScript != null) fpvScript.SetOwnBodyHidden(false);
        }
        if (!ok && !renderFailed)
        {
            renderFailed = true;
            Debug.LogError("Sphere360Display: RenderToCubemap failed; the sphere will not refresh.", this);
        }
        FPVCameraScript.CountRenders(FaceCount(mask));
        LastRenderPosition = from;
        HasRendered = true;
    }

    private static int FaceCount(int mask)
    {
        int n = 0;
        for (; mask != 0; mask &= mask - 1) n++;
        return n;
    }

    /// <summary>
    /// Sets the display->city rotation. B is the body yaw PyUniSharingFast has turned the rig to, in 50 Hz steps (its
    /// lone-drone lock follows the heading setpoint); h is that setpoint interpolated. Turning with the drone, the
    /// sphere sits at 90 + (B - h): world-fixed at the display's 90 deg whenever the two agree, and taking back the
    /// step in between, so the city turns smoothly in front of the pilot. World-fixed, it turns with the rig instead,
    /// at the offset latched at the start and at every recentre, so the city holds still in the room.
    /// </summary>
    private void UpdateDisplayRotation()
    {
        float turning = 90f + Mathf.DeltaAngle(headingDeg, PyUniSharingFast.BodyYawDegrees);
        float yaw = turning;
        if (settings.heading == HeadingMode.WorldFixed && rig != null)
        {
            float rigYaw = rig.transform.eulerAngles.y;
            if (latchedCalibration != PyUniSharingFast.CalibrationCount)
            {
                latchedCalibration = PyUniSharingFast.CalibrationCount;
                worldFixedOffset = Mathf.DeltaAngle(rigYaw, turning);
            }
            yaw = rigYaw + worldFixedOffset;
        }
        yaw = Mathf.Repeat(yaw, 360f);

        SphereYawDegrees = yaw;
        DisplayToCity = Quaternion.Euler(0f, -yaw, 0f);
        if (material != null)
        {
            material.SetMatrix("_DisplayToCity", Matrix4x4.Rotate(DisplayToCity));
            material.SetFloat("_BandHalfDeg", settings.bandHalfDeg);
            material.SetFloat("_BandCentreDeg", FPVCameraScript.SharedPitch);
        }
    }

    // Centred on the head, though at this radius and with a lookup by direction a frame's lag would not show.
    private void PlaceSphere()
    {
        if (sphere == null) return;
        Transform head = rig != null ? rig.centerEyeAnchor : null;
        if (head != null) sphere.transform.position = head.position;
        else if (rig != null) sphere.transform.position = rig.transform.position;
        else if (arena != null) sphere.transform.position = arena.transform.position;
        sphere.transform.rotation = Quaternion.identity;
    }

    private void OnDisable()
    {
        if (Instance == this)
        {
            Active = false;
            Instance = null;
        }
        if (sphere != null) sphere.SetActive(false);
        RestoreBackdrop();
    }

    private void OnDestroy()
    {
        RestoreBackdrop();
        if (sphere != null) Destroy(sphere);
        if (material != null) Destroy(material);
        if (cubeCamera != null) Destroy(cubeCamera.gameObject);
        if (cube != null)
        {
            cube.Release();
            Destroy(cube);
        }
    }

    private void RestoreBackdrop()
    {
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
        foreach (Camera c in disabledFpvCameras)
        {
            if (c != null) c.enabled = true;
        }
        disabledFpvCameras.Clear();
        if (cockpitRoom != null)
        {
            cockpitRoom.enabled = true;
            cockpitRoom = null;
        }
    }
}
