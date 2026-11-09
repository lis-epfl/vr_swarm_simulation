using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class InterfaceManager : MonoBehaviour
{
    
    // Values spelled out: Unity serialises enum fields by integer, and every scene stores 0.
    public enum DisplayMode
    {
        SCREENS = 0,
        SINGLE = 1,
        BIRDSEYE = 2,
        SPHERE = 3      // one drone's 360 camera on a sphere round the pilot (Sphere360Display)
    }

    [Header("Interface Display Mode")]
    [Tooltip("SCREENS: a feed screen per drone. SPHERE: one drone carries a 360 camera, shown on a sphere round the " +
             "pilot. SPHERE is single-drone only: swarmSpawn then spawns one drone at its start position whatever " +
             "its pattern says. Set the recorder's condition to SingleSphere. Read when the swarm spawns; changing " +
             "it in Play does nothing.")]
    public DisplayMode displayMode = DisplayMode.SCREENS;

    [Tooltip("The 360 camera and its sphere, used only in the SPHERE display mode.")]
    public Sphere360Display.Settings sphere360 = new Sphere360Display.Settings();

    [Header("Screen Display Settings")]
    [Tooltip("Layout for the per-drone feed screens. AUTO follows the configuration instead of " +
             "naming a layout: OUTER_CIRCLE for a horizontal swarm looking out, FORMATION_MAP " +
             "once the gimbal is pitched below -60 deg, FORMATION_WALL in the vertical plane. " +
             "Works off simulated and real drones alike; the resolved layout is shown below.")]
    public ScreenSpawn.ScreenStyle screenStyle = ScreenSpawn.ScreenStyle.OFF;

    // Read-out, and also what AUTO holds until the configuration can first be read — which in a
    // real-drone scene is not until an aircraft has streamed. No feed means no visible screen
    // either, so that opening window is not something a pilot can see; what it protects against
    // is a mid-session dropout re-choosing the layout and resetting its tuned radius/scale.
    [Tooltip("Read-only: the layout AUTO has resolved to. Ignored unless the style above is AUTO.")]
    [SerializeField] private ScreenSpawn.ScreenStyle autoStyle = ScreenSpawn.ScreenStyle.OUTER_CIRCLE;

    [Tooltip("Rearrange the feeds into FORMATION_MAP when the gimbal is pitched below -60 deg. " +
             "Untick to keep OUTER_CIRCLE at any pitch; the vertical plane still gets FORMATION_WALL.")]
    public bool formationMapAtNadir = true;

    [Header("Screen Parameters")]
    [Tooltip("Each feed's render-texture size, used only with stitching off (PyUniSharingFast's " +
             "enableImageWriting and enablePanoramaReading both unticked). While stitching, the feeds " +
             "adopt its blockImageWidth/Height, because the same texture is the stitcher's input.")]
    public int width = 640;
    public int height = 360;
    
    [Header("Active Display Parameters")]
    public float radius = 2.0f;
    public float scale = 1.0f;
    public Vector3 offset = new Vector3(0.5f, -0.3f, 0.0f);
    public Vector3 lookAtOffset = new Vector3(0.0f, 0.0f, 0.0f);
    
    [Header("Special Settings")]
    public bool invertBottomScreen = false;
    public bool doubleView = false;
    public float rotatingCircleDistance = 2.0f;

    [Header("Formation Grid Settings (FORMATION_WALL / FORMATION_MAP)")]
    [Tooltip("Clearance between neighbouring screens, as a multiple of the screen's own size. " +
             "1.0 makes them touch exactly; below 1 is clamped away, since the point of the " +
             "style is that the screens do not overlap.")]
    public float formationWallPadding = 1.08f;
    [Tooltip("Columns in the grid. 0 = auto, which shapes the grid like the formation itself " +
             "(a wall five drones wide and two tall lays out 5x2).")]
    public int formationWallColumns = 0;
    [Tooltip("Widest azimuth the grid may span, in degrees. Feeds that do not fit go into extra " +
             "rows rather than wrapping around the pilot. Lower the scale to fit more columns " +
             "into the same span. 0 = unbounded.")]
    public float formationWallMaxSpanDeg = 120.0f;
    [Tooltip("Time constant (s) of the low-pass on the wall's heading and on each screen's " +
             "glide between grid cells. 0 = snap. FORMATION_MAP eases only the glide — its " +
             "heading is the pilot's own, and lagging that slides the panel out of view.")]
    public float formationWallSmoothTime = 0.15f;
    [Tooltip("FORMATION_MAP only: roll each feed so its imagery lines up with the map frame. " +
             "A nadir image is drawn in its own drone's heading frame, so without this two " +
             "feeds show the same ground rotated differently as soon as the headings disagree.")]
    public bool formationMapRollScreens = true;

    [Tooltip("Time constant (s) of the low-pass on each screen's display heading, for the styles " +
             "that place a screen at its own drone's yaw (the circle styles; the two formation " +
             "grids deliberately bypass it — see BuildFormationGridLayout). It puts a screen's " +
             "POSITION in phase with the IMAGE inside it: a sim FPV camera Slerps onto its drone's " +
             "heading at FPVCameraScript.smoothSpeed while the screen used to be placed from the " +
             "raw 50 Hz body heading, so the screen led its own picture. Keep this at " +
             "1 / smoothSpeed. Real feeds reuse the same number as the best available estimate of " +
             "the DJI gimbal's own lag, which is not measured — it also smooths the staircase " +
             "their telemetry pushes arrive as. 0 = no filter.")]
    public float displayYawSmoothTime = 0.1f;

    public delegate void OnInterfaceParamsChanged();
    public event OnInterfaceParamsChanged interfaceParamsChanged;
    
    private ViewManager viewManager;
    private ScreenSpawn spawnScreens;
    private BirdsEyeCamera birdsEyeCamera;
    private ThirdPersonSwarmCamera thirdPersonCam;
    private visualiseOlfatiSaber visualiseOlfatiSaber;

    private List<GameObject> swarm;
    private bool screensSpawned = false;
    private Sphere360Display sphereDisplay;
    private bool sphereRefused;     // SPHERE was set but more than one drone spawned (see StartSphere)
    private bool modeChangeWarned;

    /// <summary>
    /// The layout actually in force: <see cref="screenStyle"/> itself, or — while it is AUTO —
    /// the one the configuration resolves to. Everything that consumes the style must read this
    /// rather than the field, because AUTO is a choice of rule and not a layout: the placement
    /// switches in <see cref="ScreenSpawn"/> have no case for it and would leave the screens
    /// wherever they last were.
    /// </summary>
    public ScreenSpawn.ScreenStyle ResolvedScreenStyle
        => screenStyle == ScreenSpawn.ScreenStyle.AUTO ? autoStyle : screenStyle;

    // Awake is called before Start
    void Awake()
    {
        viewManager = GetComponent<ViewManager>();
        spawnScreens = GetComponent<ScreenSpawn>();

        // Get the BirdsEyeCamera
        GameObject birdsEyeCameraObject = GameObject.FindGameObjectWithTag("BirdsEyeCamera");
        if (birdsEyeCameraObject != null)
        {
            birdsEyeCamera = birdsEyeCameraObject.GetComponent<BirdsEyeCamera>();
        }

        GameObject thirdPersonCamObject = GameObject.FindGameObjectWithTag("ThirdPersonCamera");
        if (thirdPersonCamObject != null)
        {
            thirdPersonCam = thirdPersonCamObject.GetComponent<ThirdPersonSwarmCamera>();
        }

        visualiseOlfatiSaber = GetComponent<visualiseOlfatiSaber>();
    }


    // Start is called before the first frame update
    void Start()
    {
        // Subscribe ScreenSpawn to parameter changes
        if (spawnScreens != null)
        {
            interfaceParamsChanged += spawnScreens.OnInterfaceParamsChanged;
        }
        
        // Initialize ScreenSpawn parameters
        UpdateScreenSpawnParameters();
    }

    void Update()
    {
        RefreshAutoScreenStyle();

        // If screens are not spawned and display mode is SCREENS, spawn screens (never beside a sphere: the mode
        // is read once, when the swarm spawns)
        if (displayMode == DisplayMode.SCREENS && !screensSpawned && sphereDisplay == null)
        {
            if (spawnScreens != null && spawnScreens.IsSpawned)
            {
                // Screens were already spawned externally (e.g. by ImageSharing)
                screensSpawned = true;
            }
            else
            {
                spawnScreens.SpawnScreens(swarm);
                screensSpawned = true;
            }
        }
    }

    // Called whenever a value is changed in the Inspector
    public void OnValidate()
    {
        // Resolve first, so switching the style TO Auto in the inspector takes effect on this
        // edit rather than on the next frame. Playing only: outside play mode there is no swarm
        // and no feed producer, so every reading is at its default and resolving would do
        // nothing but overwrite the read-out with a meaningless OUTER_CIRCLE.
        if (Application.isPlaying)
        {
            RefreshAutoScreenStyle();
        }

        // Update ScreenSpawn parameters immediately
        UpdateScreenSpawnParameters();

        // Trigger the event to notify all subscribed scripts
        interfaceParamsChanged?.Invoke();
    }

    /// <summary>
    /// Points <see cref="autoStyle"/> at the layout belonging to the configuration the swarm is
    /// currently in, and pushes it down when it changes. Only the change pushes: the resolution
    /// runs every frame, and re-notifying on an unchanged style would re-run ScreenSpawn's whole
    /// parameter reset (and restart the wall's easing) once per frame.
    ///
    /// The rule and its two readings both live in <see cref="SwarmPlaneController"/>, which is
    /// also what <c>driveDisplayConfiguration</c> pushes through, so the automatic layout and the
    /// automatic stitcher always describe the same configuration. Nothing here is sim-specific —
    /// that resolver answers for the real-drone feeds as well, which is why AUTO works in a scene
    /// with no swarm in it at all.
    /// </summary>
    private void RefreshAutoScreenStyle()
    {
        if (screenStyle != ScreenSpawn.ScreenStyle.AUTO) return;

        // Indeterminate: a real-drone scene whose aircraft have not streamed yet. Hold the
        // layout rather than reading the empty state as "horizontal, looking out" — see
        // TryResolveDisplayConfiguration.
        if (!SwarmPlaneController.TryResolveDisplayConfiguration(out bool vertical, out bool nadir))
        {
            return;
        }

        ScreenSpawn.ScreenStyle resolved = AllowedStyle(SwarmPlaneController.StyleForConfiguration(vertical, nadir));
        if (resolved == autoStyle) return;

        autoStyle = resolved;
        UpdateScreenSpawnParameters();
        interfaceParamsChanged?.Invoke();
    }

    /// <summary>
    /// Changes the screen layout at runtime, on the same path an inspector edit takes.
    /// <see cref="SwarmPlaneController"/> calls this when the swarming plane changes:
    /// OUTER_CIRCLE only spreads the screens out while the drones' yaws are spread out, so a
    /// shared-heading formation needs FORMATION_WALL instead.
    ///
    /// This field stays the single source of truth for the layout — pushing the style into
    /// ScreenSpawn directly would be overwritten by the next parameter change, and would skip
    /// the panorama-fallback substitution ScreenSpawn re-applies on top of it.
    ///
    /// Refused while the style is AUTO, which is the same request answered continuously rather
    /// than on a change, and answered over both feed sources. Honouring it would also be
    /// one-way: it writes a concrete style into the field, so the first configuration change
    /// would silently end AUTO for the rest of the session.
    /// </summary>
    public void SetScreenStyle(ScreenSpawn.ScreenStyle style)
    {
        style = AllowedStyle(style);
        if (screenStyle == style || screenStyle == ScreenSpawn.ScreenStyle.AUTO) return;

        screenStyle = style;
        UpdateScreenSpawnParameters();
        interfaceParamsChanged?.Invoke();
    }

    // The one place formationMapAtNadir is read, so AUTO and the SwarmPlaneController push agree.
    // Swaps in OUTER_CIRCLE rather than refusing the request: a refusal would leave the wall up
    // when the vertical plane is switched off with the gimbal already down.
    private ScreenSpawn.ScreenStyle AllowedStyle(ScreenSpawn.ScreenStyle style)
        => style == ScreenSpawn.ScreenStyle.FORMATION_MAP && !formationMapAtNadir
            ? ScreenSpawn.ScreenStyle.OUTER_CIRCLE : style;

    // Called by ScreenSpawn to update display parameters with defaults
    public void UpdateDisplayParameters(float newRadius, float newScale, Vector3 newOffset, Vector3 newLookAtOffset)
    {
        radius = newRadius;
        scale = newScale;
        offset = newOffset;
        lookAtOffset = newLookAtOffset;
    }

    // Update ScreenSpawn with parameters from InterfaceManager
    private void UpdateScreenSpawnParameters()
    {
        if (spawnScreens != null)
        {
            // Resolved, never the raw field: AUTO is not a layout ScreenSpawn can place.
            spawnScreens.screenStyle = ResolvedScreenStyle;
            spawnScreens.width = width;
            spawnScreens.height = height;
            spawnScreens.radius = radius;
            spawnScreens.scale = scale;
            spawnScreens.offset = offset;
            spawnScreens.lookAtOffset = lookAtOffset;
            spawnScreens.invertBottomScreen = invertBottomScreen;
            spawnScreens.doubleView = doubleView;
            spawnScreens.rotatingCircleDistance = rotatingCircleDistance;
            spawnScreens.formationWallPadding = formationWallPadding;
            spawnScreens.formationWallColumns = formationWallColumns;
            spawnScreens.formationWallMaxSpanDeg = formationWallMaxSpanDeg;
            spawnScreens.formationWallSmoothTime = formationWallSmoothTime;
            spawnScreens.displayYawSmoothTime = displayYawSmoothTime;
            spawnScreens.formationMapRollScreens = formationMapRollScreens;
        }
    }

    // Assign the swarm list to this script and other relevant scripts
    public void SetSwarm(List<GameObject> swarmList)
    {
        swarm = swarmList;

        if (viewManager != null)
        {
            viewManager.swarm = swarmList;
        }
        if (visualiseOlfatiSaber != null)
        {
            visualiseOlfatiSaber.swarm = swarmList;
        }
        if (birdsEyeCamera != null)
        {
            birdsEyeCamera.swarm = swarmList;
        }
        if (thirdPersonCam != null)
        {
            thirdPersonCam.swarm = swarmList;
        }
        if (displayMode == DisplayMode.SCREENS && !screensSpawned)
        {
            if (spawnScreens != null)
            {
                if (!spawnScreens.IsSpawned)
                {
                    spawnScreens.SpawnScreens(swarm);
                }
                screensSpawned = true;
            }
        }
        StartSphere();
    }

    // SPHERE replaces the screens rather than sitting beside them: no screen is spawned, so ScreenSpawn never moves
    // the rig or drives the FPV cameras, and Sphere360Display does both its own way. It is a single-drone condition:
    // swarmSpawn spawns one drone under it, and a swarm handed here anyway gets its feed screens, never 360 cameras.
    private void StartSphere()
    {
        if (displayMode != DisplayMode.SPHERE || sphereDisplay != null || sphereRefused || swarm == null ||
            swarm.Count == 0) return;
        if (swarm.Count != 1)
        {
            sphereRefused = true;
            Debug.LogError($"InterfaceManager: the SPHERE display is one drone's 360 camera, and {swarm.Count} drones " +
                           "spawned; showing their feed screens instead.", this);
            if (spawnScreens != null && !spawnScreens.IsSpawned) spawnScreens.SpawnScreens(swarm);
            screensSpawned = true;
            return;
        }
        sphereDisplay = GetComponent<Sphere360Display>();
        if (sphereDisplay == null) sphereDisplay = gameObject.AddComponent<Sphere360Display>();
        sphereDisplay.Begin(swarm, sphere360);
    }

    void LateUpdate()
    {
        if (swarm == null || modeChangeWarned || sphereRefused) return;
        if ((displayMode == DisplayMode.SPHERE) != (sphereDisplay != null))
        {
            modeChangeWarned = true;
            Debug.LogWarning($"InterfaceManager: displayMode is read when the swarm spawns; the change to " +
                             $"{displayMode} takes effect on the next Play.", this);
        }
    }
}