using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class ScreenSpawn : MonoBehaviour
{
    public enum ScreenStyle
    {
        OFF,
        OUTER_CIRCLE,
        INNER_CIRCLE,
        BOTTOM_CIRCLE,
        ROTATING_CIRCLE,
        REAL_DRONE,

        // Curved video wall for SwarmPlaneController's vertical plane (the nadir configuration,
        // the other shared-heading one, has its own style below).
        // OUTER_CIRCLE places each screen at its own drone's yaw,
        // which collapses to a single stack of screens once every drone points the same way;
        // this style keeps the yaw (the wall as a whole sits in the direction the swarm is
        // looking) and resolves the collision from the drones' relative positions inside the
        // swarming plane. See UpdateFormationWallScreen.
        //
        // Appended rather than inserted next to OUTER_CIRCLE on purpose: Unity serialises
        // enum fields by integer value, so inserting a member would silently re-point every
        // scene and prefab that already stores a later style.
        FORMATION_WALL,

        // The nadir counterpart of FORMATION_WALL: a horizontal swarm with the gimbal pitched
        // straight down, stitched by PLANAR. Same grid machinery, but the frame it ranks in and
        // the frame it hangs in are both the pilot's own body heading rather than the swarming
        // plane, and each feed is rolled so the imagery is map-aligned. See
        // BuildFormationGridLayout / UpdateFormationMapScreen.
        //
        // The panel stands in front of the pilot exactly like the wall — the feeds are NOT laid
        // on the floor. A ground layout would be geometrically honest and useless: it puts the
        // whole display outside the comfortable gaze cone, and the pilot flies by looking down at
        // their feet instead of at the panorama. Tilting the map up onto a vertical panel keeps
        // the map reading (ahead is up, starboard is right) at the cost of a perspective that was
        // never real anyway.
        FORMATION_MAP
    }

    [Header("Display Settings")]
    [HideInInspector] public ScreenStyle screenStyle = ScreenStyle.OFF;

    [Header("VR Parameters")]
    [HideInInspector] public int width = 640;
    [HideInInspector] public int height = 360;

    [Header("Active Display Parameters")]
    [HideInInspector] public float radius = 2.0f;
    [HideInInspector] public float scale = 1.0f;
    [HideInInspector] public Vector3 offset = new Vector3(0.5f, -0.3f, 0.0f);
    [HideInInspector] public Vector3 lookAtOffset = new Vector3(0.0f, 0.0f, 0.0f);

    [Header("Special Settings")]
    [HideInInspector] public bool invertBottomScreen = false;
    [HideInInspector] public bool doubleView = false;
    [HideInInspector] public float rotatingCircleDistance = 2.0f;
    [HideInInspector] public int numScreens = 2;

    [Header("Formation Wall / Map Settings")]
    // Shared by FORMATION_WALL and FORMATION_MAP: the two styles differ in which frame they rank
    // and hang the grid in, not in how the grid itself is built, so a second copy of these four
    // would only ever be kept equal by hand.
    //
    // Clearance between neighbouring cells, as a multiple of the screen's own size. 1.0 makes
    // them touch exactly; anything above leaves a gap. Values below 1 are clamped away, since
    // the whole point of the style is that the screens do not overlap.
    [HideInInspector] public float formationWallPadding = 1.08f;
    // Columns in the grid. 0 = auto, which shapes the grid like the formation itself.
    [HideInInspector] public int formationWallColumns = 0;
    // Widest azimuth the wall may span, in degrees. Extra feeds go into extra rows rather than
    // wrapping around the pilot. 0 = unbounded.
    [HideInInspector] public float formationWallMaxSpanDeg = 120.0f;
    // Time constant (s) of the low-pass on the wall's azimuth and on each screen's glide
    // between cells. 0 = snap.
    [HideInInspector] public float formationWallSmoothTime = 0.15f;

    // FORMATION_MAP only: roll each feed so its imagery is map-aligned (see the roll derivation
    // in BuildFormationGridLayout). Off leaves every screen upright, which is tidier but shows
    // the same ground rotated differently on adjacent feeds whenever the headings disagree.
    [HideInInspector] public bool formationMapRollScreens = true;

    [Header("Rendering")]
    [Tooltip("Layer the spawned feed screens are placed on, so the headset eye cameras " +
             "can be culled to render only these screens (and the panorama) rather than " +
             "the full world geometry. Must be an existing layer name (default 'UI').")]
    public string screenLayerName = "UI";

    // GameObject references
    private OVRCameraRig cameraRig;
    private List<GameObject> swarm = new List<GameObject>();
    private List<GameObject> screens = new List<GameObject>();
    private GameObject arena;
    private GameObject screenParent;

    // Per-drone references resolved once at spawn so the per-frame update never
    // searches by name or calls GetComponent. Bound by spawn index (screen_i is
    // wired to swarm[i]'s FPV render texture at spawn), which stays valid even
    // though the shared swarm list itself may later be re-sorted in place
    // (AttitudeAlgorithm's LOCAL_CONVEXHULL sorts it every tick).
    private struct DroneScreenBinding
    {
        public GameObject drone;
        public GameObject screen;
        public Camera fpvCamera;
        public VelocityControl velocityControl;
        public AttitudeAlgorithm attitude;
    }
    private readonly List<DroneScreenBinding> bindings = new List<DroneScreenBinding>();

    // Default parameters for each display mode
    private float outerCircleRadius = 2.0f;
    private float outerCircleScale = 1.0f;
    private Vector3 outerCircleOffset = new Vector3(0.0f, 0.0f, 0.0f);
    private Vector3 outerCircleLookAtOffset = new Vector3(0.0f, 0.0f, 0.0f);

    private float innerCircleRadius = 0.45f;
    private float innerCircleScale = 0.2f;
    private Vector3 innerCircleOffset = new Vector3(0.0f, -0.3f, 0.0f);
    private Vector3 innerCircleLookAtOffset = new Vector3(0.5f, -0.3f, 0.0f);

    private float bottomCircleRadius = 0.6f;
    private float bottomCircleScale = 0.25f;
    private Vector3 bottomCircleOffset = new Vector3(0.0f, -0.5f, 0.0f);
    private Vector3 bottomCircleLookAtOffset = new Vector3(0.0f, -0.5f, 0.0f);

    private float rotatingCircleRadius = 0.2f;
    private float rotatingCircleScale = 0.1f;
    private Vector3 rotatingCircleOffset = new Vector3(0.0f, -0.3f, 0.0f);
    private Vector3 rotatingCircleLookAtOffset = new Vector3(0.0f, 0.0f, 0.0f);

    private float realDroneRadius = 2.0f;
    private float realDroneScale = 1.0f;
    private Vector3 realDroneOffset = new Vector3(0.0f, 0.0f, 0.0f);
    private Vector3 realDroneLookAtOffset = new Vector3(0.0f, 0.0f, 0.0f);

    // A grid needs both more standoff and smaller screens than the single ring of
    // OUTER_CIRCLE: at radius 3 / scale 0.55 a padded 16:9 screen subtends ~20 deg, so the
    // 120 deg span budget holds 6 columns and four rows stack ~1.8 m of height. Halving the
    // scale is what buys more columns, since the pitch is proportional to the screen width.
    private float formationWallRadius = 2.0f;
    private float formationWallScale = 0.55f;
    private Vector3 formationWallOffset = new Vector3(0.0f, 0.0f, 0.0f);
    private Vector3 formationWallLookAtOffset = new Vector3(0.0f, 0.0f, 0.0f);

    // Same grid, same standoff — but a rolled screen sweeps out its own diagonal, so the map's
    // cells are up to ~1.84x taller than the wall's for a 16:9 feed at 45 deg of roll. Starting
    // one notch smaller keeps a nadir formation inside the same span budget; the roll expansion
    // is computed exactly per frame (see BuildFormationGridLayout), this is only the default.
    private float formationMapRadius = 2.0f;
    private float formationMapScale = 0.45f;
    private Vector3 formationMapOffset = new Vector3(0.0f, 0.0f, 0.0f);
    private Vector3 formationMapLookAtOffset = new Vector3(0.0f, 0.0f, 0.0f);

    private SwarmManager swarmManager;
    private bool pointInwards = false;
    private ScreenStyle previousScreenStyle;
    private InterfaceManager interfaceManager;

    // Drones whose individual feeds are suppressed because they are currently
    // composited into the stitched panorama (driven by PyUniSharingFast). Matched
    // by GameObject reference. Pass null/empty to show all feeds again.
    private readonly HashSet<GameObject> stitchedDronesToHide = new HashSet<GameObject>();

    public bool IsSpawned => screens.Count > 0;

    // Set which drones' individual feeds to hide because they already appear in
    // the stitched panorama. Called by PyUniSharingFast; null/empty restores all.
    public void SetStitchedDronesHidden(IEnumerable<GameObject> drones)
    {
        stitchedDronesToHide.Clear();
        if (drones != null)
        {
            foreach (var d in drones)
            {
                if (d != null) stitchedDronesToHide.Add(d);
            }
        }
    }

    // Function to spawn screens for the drones in the swarm
    public void SpawnScreens(List<GameObject> swarm = null)
    {
        // The per-drone feed resolution must match the block images PyUniSharingFast
        // captures for the stitcher. PyUniSharingFast is the single source of truth:
        // if its block resolution differs, adopt it here once, at spawn time (done
        // before the render textures / aspect ratios below are built from width/height).
        PyUniSharingFast stitchSharing = FindObjectOfType<PyUniSharingFast>();
        if (stitchSharing != null &&
            (width != stitchSharing.BlockImageWidth || height != stitchSharing.BlockImageHeight))
        {
            Debug.Log($"[ScreenSpawn] Overriding feed resolution {width}x{height} with " +
                      $"PyUniSharingFast block resolution {stitchSharing.BlockImageWidth}x{stitchSharing.BlockImageHeight}.");
            width = stitchSharing.BlockImageWidth;
            height = stitchSharing.BlockImageHeight;
        }

        // Find the OVRCameraRig in the scene if not already assigned. Fall back to
        // FindObjectOfType so it resolves even if the rig isn't tagged 'Player'.
        if (cameraRig == null)
        {
            cameraRig = GameObject.FindGameObjectWithTag("Player")?.GetComponent<OVRCameraRig>();
            if (cameraRig == null)
            {
                cameraRig = FindObjectOfType<OVRCameraRig>();
            }
            if (cameraRig == null)
            {
                Debug.LogWarning("No OVRCameraRig found in the scene!");
            }
        }

        // Find the arena in the scene
        if (arena == null)
        {
            arena = GameObject.FindGameObjectWithTag("Arena");
            if (arena == null)
            {
                Debug.LogWarning("No GameObject with tag 'Arena' found in the scene!");
            }
        } 
    
        // Get the swarm manager instance and add the event listener
        if (swarm != null)
        {
            // Get the swarm manager instance
            swarmManager = SwarmManager.Instance;

            // Add the swarmParamsChanged event listener
            swarmManager.swarmParamsChanged += OnSwarmParamsChanged;
        }

        // Store the swarm list
        this.swarm = swarm;

        // Create an empty GameObject to serve as the parent for all screens
        screenParent = new GameObject("ScreenParent");

        // Resolve the layer the feed screens live on so the headset eye cameras
        // can be culled to render only these (and the panorama). Resolve once and
        // warn if the layer is missing, rather than silently leaving screens on
        // Default (where the eye-camera cull couldn't exclude the world geometry).
        int screenLayer = LayerMask.NameToLayer(screenLayerName);
        if (screenLayer < 0)
        {
            Debug.LogWarning($"[ScreenSpawn] Layer '{screenLayerName}' does not exist; " +
                             "feed screens will stay on the Default layer. Add the layer " +
                             "(Project Settings > Tags and Layers) or fix screenLayerName.");
        }

        // Determine how many screens to create
        int count = (swarm != null) ? swarm.Count : numScreens;

        for (int i = 0; i < count; i++)
        {
            int droneNumber = i;

            // Create a screen using a quad
            GameObject screen = GameObject.CreatePrimitive(PrimitiveType.Quad);

            // Name the screen
            screen.name = "screen_" + droneNumber;

            // Parent the screen under the screenParent GameObject
            screen.transform.parent = screenParent.transform;

            // Set the tag of the screen to 'Screen'
            screen.tag = "Screen";

            // Put the screen on the feed-screen layer so the headset eye cameras
            // can render only these. Quads have no children, so setting the layer
            // on the screen itself is enough.
            if (screenLayer >= 0)
            {
                screen.layer = screenLayer;
            }

            // Add the screen to the screens list
            screens.Add(screen);

            // Create a render texture
            RenderTexture rt = new RenderTexture(width, height, 24);

            // Name the render texture 'rt_' followed by the drone number
            rt.name = "rt_" + droneNumber;

            // Create a new Material object
            Material screenMaterial = new Material(Shader.Find("Standard"));

            // Set the color to black, then white for real drones
            screenMaterial.color = Color.black;
            if (screenStyle == ScreenStyle.REAL_DRONE)
            {
                screenMaterial.color = Color.white;
            }

            // Set the smoothness to 0
            screenMaterial.SetFloat("_Glossiness", 0f);

            // Name the material 'screenMaterial'
            screenMaterial.name = "screenMaterial" + droneNumber;

            // Set the emission map to the render texture
            screenMaterial.SetTexture("_EmissionMap", rt);

            // Set the emission color to white
            screenMaterial.SetColor("_EmissionColor", Color.white);

            // Set the emission global illumination to baked
            screenMaterial.globalIlluminationFlags = MaterialGlobalIlluminationFlags.BakedEmissive;

            // Turn on emission for the material
            screenMaterial.EnableKeyword("_EMISSION");

            // Set the screens material to the screen material
            screen.GetComponent<Renderer>().material = screenMaterial;

            // set the scale to match the aspect ratio of the feed
            screen.transform.localScale = new Vector3((float)width / height, 1f, 1f);

            if (swarm != null && i < swarm.Count)
            {
                // Find the camera object on the drone called 'FPV'
                GameObject drone = swarm[i];
                Transform camera = drone.transform.Find("FPV");

                // Get the camera and set the aspect ratio and field of view
                Camera cam = camera.GetComponent<Camera>();
                float aspect = (float)width / height;
                cam.aspect = aspect;

                // DJI Mini 3 Pro is specced at 82.1 deg diagonal FOV. Unity's
                // Camera.fieldOfView is vertical, so convert the diagonal spec to
                // the vertical FOV for the current aspect ratio (~46.4 deg at 16:9).
                const float djiDiagonalFov = 82.1f;
                float diagHalfRad = djiDiagonalFov * 0.5f * Mathf.Deg2Rad;
                float vertHalfRad = Mathf.Atan(Mathf.Tan(diagHalfRad) / Mathf.Sqrt(aspect * aspect + 1f));
                cam.fieldOfView = vertHalfRad * 2f * Mathf.Rad2Deg;

                // The feed RT doubles as the stitch-capture source in
                // PyUniSharingFast, so it's needed regardless of screen style.
                cam.targetTexture = rt;

                Transform droneParent = drone.transform.Find("DroneParent");
                bindings.Add(new DroneScreenBinding
                {
                    drone = drone,
                    screen = screen,
                    fpvCamera = cam,
                    velocityControl = droneParent != null ? droneParent.GetComponent<VelocityControl>() : null,
                    attitude = droneParent != null ? droneParent.GetComponent<AttitudeAlgorithm>() : null,
                });
            }
        }

        // Place the screens based on the orientation of the drones
        UpdateScreenPositions();

        // Move the camera rig to the centre of the arena
        if (cameraRig != null && arena != null)
        {
            cameraRig.transform.position = arena.transform.position;
        }
    }

    // Update display parameters when screen style changes
    private void UpdateDisplayParameters()
    {
        switch (screenStyle)
        {
            case ScreenStyle.OUTER_CIRCLE:
                radius = outerCircleRadius;
                scale = outerCircleScale;
                offset = outerCircleOffset;
                lookAtOffset = outerCircleLookAtOffset;
                break;
            case ScreenStyle.INNER_CIRCLE:
                radius = innerCircleRadius;
                scale = innerCircleScale;
                offset = innerCircleOffset;
                lookAtOffset = innerCircleLookAtOffset;
                break;
            case ScreenStyle.BOTTOM_CIRCLE:
                radius = bottomCircleRadius;
                scale = bottomCircleScale;
                offset = bottomCircleOffset;
                lookAtOffset = bottomCircleLookAtOffset;
                break;
            case ScreenStyle.ROTATING_CIRCLE:
                radius = rotatingCircleRadius;
                scale = rotatingCircleScale;
                offset = rotatingCircleOffset;
                lookAtOffset = rotatingCircleLookAtOffset;
                break;
            case ScreenStyle.REAL_DRONE:
                radius = realDroneRadius;
                scale = realDroneScale;
                offset = realDroneOffset;
                lookAtOffset = realDroneLookAtOffset;
                break;
            case ScreenStyle.FORMATION_WALL:
                radius = formationWallRadius;
                scale = formationWallScale;
                offset = formationWallOffset;
                lookAtOffset = formationWallLookAtOffset;
                break;
            case ScreenStyle.FORMATION_MAP:
                radius = formationMapRadius;
                scale = formationMapScale;
                offset = formationMapOffset;
                lookAtOffset = formationMapLookAtOffset;
                break;
        }

        // The wall eases towards its cells, so a style change (or a radius/scale change that
        // moves every cell at once) must not be animated from wherever the screens happened
        // to be sitting under the previous layout.
        InvalidateFormationGrid();
    }

    // Get default parameters for a given screen style and send them to InterfaceManager
    public void SendDefaultParametersToInterfaceManager(ScreenStyle style)
    {
        // Get reference to InterfaceManager if not already set
        if (interfaceManager == null)
        {
            interfaceManager = GetComponent<InterfaceManager>();
        }

        if (interfaceManager == null)
        {
            Debug.LogWarning("InterfaceManager not found on this GameObject!");
            return;
        }

        float defaultRadius = 2.0f;
        float defaultScale = 1.0f;
        Vector3 defaultOffset = Vector3.zero;
        Vector3 defaultLookAtOffset = Vector3.zero;

        switch (style)
        {
            case ScreenStyle.OUTER_CIRCLE:
                defaultRadius = outerCircleRadius;
                defaultScale = outerCircleScale;
                defaultOffset = outerCircleOffset;
                defaultLookAtOffset = outerCircleLookAtOffset;
                break;
            case ScreenStyle.INNER_CIRCLE:
                defaultRadius = innerCircleRadius;
                defaultScale = innerCircleScale;
                defaultOffset = innerCircleOffset;
                defaultLookAtOffset = innerCircleLookAtOffset;
                break;
            case ScreenStyle.BOTTOM_CIRCLE:
                defaultRadius = bottomCircleRadius;
                defaultScale = bottomCircleScale;
                defaultOffset = bottomCircleOffset;
                defaultLookAtOffset = bottomCircleLookAtOffset;
                break;
            case ScreenStyle.ROTATING_CIRCLE:
                defaultRadius = rotatingCircleRadius;
                defaultScale = rotatingCircleScale;
                defaultOffset = rotatingCircleOffset;
                defaultLookAtOffset = rotatingCircleLookAtOffset;
                break;
            case ScreenStyle.REAL_DRONE:
                defaultRadius = realDroneRadius;
                defaultScale = realDroneScale;
                defaultOffset = realDroneOffset;
                defaultLookAtOffset = realDroneLookAtOffset;
                break;
            case ScreenStyle.FORMATION_WALL:
                defaultRadius = formationWallRadius;
                defaultScale = formationWallScale;
                defaultOffset = formationWallOffset;
                defaultLookAtOffset = formationWallLookAtOffset;
                break;
            case ScreenStyle.FORMATION_MAP:
                defaultRadius = formationMapRadius;
                defaultScale = formationMapScale;
                defaultOffset = formationMapOffset;
                defaultLookAtOffset = formationMapLookAtOffset;
                break;
        }

        // Call InterfaceManager to update its display parameters
        interfaceManager.UpdateDisplayParameters(defaultRadius, defaultScale, defaultOffset, defaultLookAtOffset);
    }

    // Update the position of the screens based on the drone orientation
    void UpdateScreenPositions()
    {
        if (bindings.Count == 0)
        {
            return;
        }

        // Per-frame, not per-drone: the gate depends only on the selected
        // attitude algorithm.
        bool boundaryGate = IsBoundaryGateActive();

        // The two grid styles are the ones whose placement is not a pure function of their own
        // drone — a cell index only means something relative to the rest of the visible set —
        // so the whole grid is solved once here, before any screen is placed.
        if (IsFormationGrid(screenStyle))
        {
            BuildFormationGridLayout();
        }

        for (int i = 0; i < bindings.Count; i++)
        {
            DroneScreenBinding binding = bindings[i];
            GameObject screen = binding.screen;
            if (screen == null)
            {
                continue;
            }

            // Hide the feed for any drone currently composited into the stitched
            // panorama (mirrors the BoundaryEstimate gate below). Applies to every
            // screen style. A destroyed drone also just hides its screen.
            if (IsFeedSuppressed(binding))
            {
                screen.SetActive(false);
            }
            else
            {
                switch (screenStyle)
                {
                    case ScreenStyle.OFF:
                        HideScreen(screen);
                        break;
                    case ScreenStyle.OUTER_CIRCLE:
                        UpdateOuterCircleScreen(screen, binding, boundaryGate);
                        break;
                    case ScreenStyle.FORMATION_WALL:
                        UpdateFormationWallScreen(screen, i);
                        break;
                    case ScreenStyle.FORMATION_MAP:
                        UpdateFormationMapScreen(screen, i);
                        break;
                    case ScreenStyle.INNER_CIRCLE:
                        UpdateInnerCircleScreen(screen, binding);
                        break;
                    case ScreenStyle.BOTTOM_CIRCLE:
                        UpdateBottomCircleScreen(screen, binding);
                        break;
                    case ScreenStyle.ROTATING_CIRCLE:
                        UpdateRotatingCircleScreen(screen, binding);
                        break;
                }
            }

            // An FPV camera only needs to render while its feed screen is
            // visible; otherwise it would draw the full world every frame for
            // nothing. The stitch capture path (PyUniSharingFast) renders
            // disabled cameras on demand at its own send rate.
            Camera cam = binding.fpvCamera;
            if (cam != null && cam.enabled != screen.activeSelf)
            {
                cam.enabled = screen.activeSelf;
            }
        }
    }

    private void HideScreen(GameObject screen)
    {
        screen.SetActive(false);
    }

    // A feed is suppressed when its drone is gone, or when that drone is currently
    // composited into the stitched panorama. Shared by the placement loop and by
    // BuildFormationGridLayout: the grid is only non-overlapping if it is solved over
    // exactly the set of screens that is about to be shown.
    private bool IsFeedSuppressed(DroneScreenBinding binding)
    {
        return binding.drone == null
            || (stitchedDronesToHide.Count > 0 && stitchedDronesToHide.Contains(binding.drone));
    }

    // The convex-hull attitude modes are the only ones that populate
    // BoundaryEstimate; under NONE/SIMPLE it stays false for every drone. The
    // OUTER_CIRCLE feed gate must therefore only consult the flag when a hull
    // mode is active, otherwise it would hide every feed. Defaults to false (show
    // all feeds) when the SwarmManager can't be resolved.
    private bool IsBoundaryGateActive()
    {
        SwarmManager sm = swarmManager != null ? swarmManager : SwarmManager.Instance;
        if (sm == null)
        {
            return false;
        }
        SwarmManager.AttitudeAlgorithm algo = sm.GetSelectedAttitudeAlgorithm();
        return algo == SwarmManager.AttitudeAlgorithm.LOCAL_CONVEXHULL
            || algo == SwarmManager.AttitudeAlgorithm.GLOBAL_CONVEXHULL;
    }

    private void UpdateOuterCircleScreen(GameObject screen, DroneScreenBinding binding, bool boundaryGate)
    {
        // Hide interior (non-boundary) drones — but only when an attitude hull
        // algorithm is actually computing BoundaryEstimate. Under attitude modes
        // NONE/SIMPLE the flag is never set (stays false for every drone), so
        // gating on it would blank all feeds — e.g. a lone drone that fell back to
        // OUTER_CIRCLE because its single feed couldn't stitch would show nothing.
        if (boundaryGate && binding.attitude != null && !binding.attitude.BoundaryEstimate)
        {
            screen.SetActive(false);
            return;
        }

        // Get the drone's yaw
        StateFinder stateFinder = binding.velocityControl.State;
        float radians = -stateFinder.Angles.y; // Already in radians

        // Calculate the position on outer circle
        float x = arena.transform.position.x + radius * Mathf.Cos(radians);
        float z = arena.transform.position.z + radius * Mathf.Sin(radians);
        float y = arena.transform.position.y + offset.y;

        // Position and rotate the screen
        screen.transform.position = new Vector3(x, y, z);
        screen.transform.LookAt(arena.transform.position + lookAtOffset);
        screen.transform.Rotate(0, 180f, 0); // Face outward
        screen.SetActive(true);
    }

    // --- FORMATION_WALL / FORMATION_MAP --------------------------------------
    // OUTER_CIRCLE reads one number per drone (its yaw) and needs nothing else, because in
    // the radially-outward ring the yaws are spread around the circle and therefore already
    // separate the screens. Under a shared heading — SwarmPlaneController's vertical wall, or
    // a nadir formation — every yaw is the same number and every screen lands on the same
    // arc position. These two styles keep yaw as the thing that aims the display and take the
    // *separation* from the drones' relative positions in the formation instead.
    //
    // Deliberately naive: the drones are ranked into a grid rather than placed at scaled-down
    // copies of their true in-plane coordinates. A proportional mapping preserves the
    // formation's shape but guarantees nothing about spacing — two drones a metre apart in a
    // 40 m wall would still overlap — whereas ranking gives non-overlap by construction and
    // still preserves the reading that matters ("that feed is the drone up and to the left").
    //
    // They share every step of that construction and differ in exactly three places, all of
    // which follow from what the drones are looking at:
    //
    //  - **The frame.** The wall ranks in the swarming plane (SwarmPlaneController.GetPlaneAxes,
    //    the same basis the planar centre-drone rule uses) and hangs at the circular mean of the
    //    drones' yaws. The map ranks in the *ground* plane and hangs at the pilot's own body
    //    heading. In nadir the swarm plane is horizontal, so GetPlaneAxes degenerates to the
    //    world (X, Z) pair and the wall's circular mean is whatever the attitude algorithm
    //    leaves behind — on a radially-outward ring the resultant collapses entirely and the
    //    azimuth is simply held. Body yaw has neither problem: it is always defined, and since
    //    it is where the pilot is facing (and what CalibrateToCentre aims the head at) the panel
    //    lands in front of them by construction rather than by luck.
    //  - **Which way is up.** The wall's rows are altitude, which needs no interpretation. The
    //    map's rows are distance along the pilot's heading, furthest ahead at the top — the
    //    formation's ground plan tilted up onto a vertical panel.
    //  - **Roll.** A nadir feed is already a map, drawn in its own drone's heading frame (with
    //    the gimbal at -90 the camera's up axis lands on the drone's forward, see
    //    FPVCameraScript). Adjacent feeds therefore show the same ground rotated differently the
    //    moment the headings disagree, and no amount of grid placement fixes that. The wall has
    //    no equivalent problem: its cameras look along the plane normal, where a shared heading
    //    already means a shared image frame.

    // Grid cell each binding occupies this frame, as offsets centred on the wall's own axis,
    // so a short bottom row ends up centred instead of left-aligned. Parallel to `bindings`;
    // wallPlaced[i] == false means "not in the visible set this frame".
    private float[] wallColOffset = new float[0];
    private float[] wallRowOffset = new float[0];
    private bool[] wallPlaced = new bool[0];

    // Eased screen positions, so a cell swap glides instead of teleporting.
    private Vector3[] wallSmoothedPos = new Vector3[0];
    private bool[] wallSmoothedValid = new bool[0];

    // FORMATION_MAP: per-screen roll in degrees about its own view axis, and the eased *cell*
    // offsets it glides through. The map eases in cell space rather than in world position
    // because its azimuth is the pilot's own heading — a world-space low-pass would let the
    // whole panel swing out of view during a turn and drift back afterwards, which is exactly
    // the motion the pilot is trying to fly against. Cell-space easing keeps the panel rigidly
    // in front of them and still makes a cell swap read as a swap rather than a teleport.
    private float[] wallRollDeg = new float[0];
    private float[] wallSmoothedCol = new float[0];
    private float[] wallSmoothedRow = new float[0];

    // Azimuth the wall is centred on, in the same negated-yaw convention as every other style
    // here (screen at radius * (cos a, sin a) around the arena centre).
    private float wallAnchorAzimuth = 0.0f;
    private bool wallAnchorInitialised = false;

    // Cell pitch, solved once per frame in BuildFormationGridLayout from the live screen size.
    private float wallAzimuthStep = 0.0f;
    private float wallRowStep = 0.0f;

    private struct WallEntry
    {
        public int binding;
        public float across;  // in-plane horizontal coordinate relative to the centroid, metres
        public float up;      // in-plane vertical coordinate relative to the centroid, metres
        public float rollDeg; // FORMATION_MAP: this feed's rotation away from the map frame
    }
    private readonly List<WallEntry> wallEntries = new List<WallEntry>();

    // Static so the sort takes no per-frame delegate allocation. Both fall back to the binding
    // index, which keeps two drones at identical coordinates from trading places every frame.
    private static readonly IComparer<WallEntry> ByUpDescending = Comparer<WallEntry>.Create(
        (a, b) =>
        {
            int c = b.up.CompareTo(a.up);
            return c != 0 ? c : a.binding.CompareTo(b.binding);
        });
    private static readonly IComparer<WallEntry> ByAcrossAscending = Comparer<WallEntry>.Create(
        (a, b) =>
        {
            int c = a.across.CompareTo(b.across);
            return c != 0 ? c : a.binding.CompareTo(b.binding);
        });

    private static bool IsFormationGrid(ScreenStyle style)
    {
        return style == ScreenStyle.FORMATION_WALL || style == ScreenStyle.FORMATION_MAP;
    }

    private void BuildFormationGridLayout()
    {
        EnsureWallArrays();
        for (int i = 0; i < wallPlaced.Length; i++)
        {
            wallPlaced[i] = false;
        }

        bool mapStyle = screenStyle == ScreenStyle.FORMATION_MAP;

        // The frame the grid is ranked in. FORMATION_MAP builds its own from the pilot's body
        // heading rather than asking GetPlaneAxes, even though the swarm plane is horizontal in
        // nadir and GetPlaneAxes would answer: that answer is the fixed world (X, Z) pair, which
        // ranks the formation north-up and leaves the map's "ahead" meaning nothing to the pilot.
        //
        // `planeUp` is the horizontal heading direction in StateFinder's yaw convention
        // (forward == (sin yaw, 0, cos yaw)) and `planeRight` is 90 deg clockwise of it — the
        // same pair GetPlaneAxes returns for a vertical plane, so the shared ranking below reads
        // `up` as "ahead" and `across` as "to starboard" without knowing which style it is in.
        Vector3 planeRight, planeUp;
        float mapFrameYawDeg = 0.0f;
        if (mapStyle)
        {
            mapFrameYawDeg = PyUniSharingFast.BodyYawDegrees;
            float yawRad = mapFrameYawDeg * Mathf.Deg2Rad;
            planeUp = new Vector3(Mathf.Sin(yawRad), 0.0f, Mathf.Cos(yawRad));
            planeRight = new Vector3(Mathf.Cos(yawRad), 0.0f, -Mathf.Sin(yawRad));
        }
        else
        {
            SwarmPlaneController swarmPlane = SwarmPlaneController.Instance;
            if (swarmPlane != null)
            {
                swarmPlane.GetPlaneAxes(out planeRight, out planeUp);
            }
            else
            {
                planeRight = Vector3.right;
                planeUp = Vector3.forward;
            }
        }

        wallEntries.Clear();
        Vector3 centroid = Vector3.zero;
        float yawSin = 0.0f;
        float yawCos = 0.0f;

        for (int i = 0; i < bindings.Count; i++)
        {
            DroneScreenBinding binding = bindings[i];
            if (binding.screen == null || IsFeedSuppressed(binding))
            {
                continue;
            }

            // Guarded, unlike the older styles: a drone with no VelocityControl contributes no
            // yaw and no ranking key, so it costs its own screen rather than the whole layout.
            StateFinder state = binding.velocityControl != null ? binding.velocityControl.State : null;
            if (state == null)
            {
                continue;
            }

            wallEntries.Add(new WallEntry { binding = i });
            centroid += WallSamplePosition(binding);

            // Only the wall aims itself with the swarm's own heading; the map takes the pilot's,
            // so skip the trig rather than accumulate a resultant nothing reads.
            if (!mapStyle)
            {
                float azimuth = -state.Angles.y;
                yawSin += Mathf.Sin(azimuth);
                yawCos += Mathf.Cos(azimuth);
            }
        }

        int n = wallEntries.Count;
        if (n == 0)
        {
            return;
        }
        centroid /= n;

        // A rolled screen sweeps out more than its own width and height, so the cell it needs is
        // the axis-aligned bounding box of the rotated quad. Taken per entry and maxed rather
        // than from the largest |roll| in the set: the width term w|cos d| + h|sin d| peaks at
        // atan(h/w), not at the largest angle, so the biggest roll is not always the widest cell.
        float rolledWidth = (float)width / height * scale;
        float rolledHeight = scale;

        float minAcross = float.MaxValue, maxAcross = float.MinValue;
        float minUp = float.MaxValue, maxUp = float.MinValue;
        for (int k = 0; k < n; k++)
        {
            WallEntry entry = wallEntries[k];
            Vector3 rel = WallSamplePosition(bindings[entry.binding]) - centroid;
            entry.across = Vector3.Dot(rel, planeRight);
            entry.up = Vector3.Dot(rel, planeUp);

            if (mapStyle && formationMapRollScreens)
            {
                // How far this feed's imagery is turned away from the map frame. Read off the FPV
                // camera rather than StateFinder: the camera is what produced the pixels and it
                // Slerps toward the drone heading (FPVCameraScript), so during a turn the body has
                // already moved on from what the frame shows.
                entry.rollDeg = Mathf.DeltaAngle(mapFrameYawDeg, WallSampleYawDeg(bindings[entry.binding]));

                float c = Mathf.Abs(Mathf.Cos(entry.rollDeg * Mathf.Deg2Rad));
                float s = Mathf.Abs(Mathf.Sin(entry.rollDeg * Mathf.Deg2Rad));
                float w = (float)width / height * scale;
                rolledWidth = Mathf.Max(rolledWidth, w * c + scale * s);
                rolledHeight = Mathf.Max(rolledHeight, w * s + scale * c);
            }

            wallEntries[k] = entry;

            if (entry.across < minAcross) minAcross = entry.across;
            if (entry.across > maxAcross) maxAcross = entry.across;
            if (entry.up < minUp) minUp = entry.up;
            if (entry.up > maxUp) maxUp = entry.up;
        }

        if (mapStyle)
        {
            // Snapped, not eased. The map's azimuth is the pilot's own heading, so easing it is
            // easing the panel away from wherever they are looking; the glide that a cell swap
            // needs is applied to the cell offsets instead (see UpdateFormationMapScreen).
            // Negated to match the display frame every style here places screens in — a screen at
            // azimuth -yaw is what CalibrateToCentre aims the head at for that same yaw.
            wallAnchorAzimuth = WrapPi(-mapFrameYawDeg * Mathf.Deg2Rad);
            wallAnchorInitialised = true;
        }
        else
        {
            // Circular mean, not a plain average: the latter tears at the +/-pi wrap, which is
            // exactly where a wall flown on a northerly heading sits. When the yaws cancel out —
            // a radially-outward ring, where this style has nothing useful to say anyway and
            // OUTER_CIRCLE is the right choice — the resultant collapses and we hold the previous
            // azimuth rather than snapping the wall to atan2(0, 0) == 0.
            float resultant = Mathf.Sqrt(yawSin * yawSin + yawCos * yawCos) / n;
            if (resultant > 0.05f)
            {
                float target = Mathf.Atan2(yawSin, yawCos);
                if (!wallAnchorInitialised)
                {
                    wallAnchorAzimuth = target;
                    wallAnchorInitialised = true;
                }
                else
                {
                    wallAnchorAzimuth = WrapPi(
                        wallAnchorAzimuth + WallSmoothAlpha() * WrapPi(target - wallAnchorAzimuth));
                }
            }
        }

        // Non-overlap is geometric rather than a heuristic: the column pitch is the angle whose
        // chord at `radius` is one padded screen width, and the row pitch one padded screen
        // height. Recomputed every frame from the live scale so it keeps holding while the
        // operator drags the scale slider. Screens wider than the wall's own diameter can't be
        // separated at all — the clamp caps the pitch at 180 deg rather than producing NaN.
        float padding = Mathf.Max(1.0f, formationWallPadding);
        float cellWidth = rolledWidth * padding;
        float cellHeight = rolledHeight * padding;
        float r = Mathf.Max(0.01f, radius);
        wallAzimuthStep = 2.0f * Mathf.Asin(Mathf.Clamp(cellWidth / (2.0f * r), 0.0f, 1.0f));
        wallRowStep = cellHeight;

        int columns = formationWallColumns > 0
            ? Mathf.Min(formationWallColumns, n)
            : AutoColumnCount(n, maxAcross - minAcross, maxUp - minUp);

        // Bound how far around the pilot the wall may wrap. A formation eight drones wide asks
        // for eight columns, which at ~20 deg of pitch is 142 deg of azimuth — the outermost
        // feeds end up beside the pilot's ears, and unlike the OUTER_CIRCLE ring (where a screen
        // behind you means a drone behind you) that placement carries no information, it is just
        // where the grid ran out of room. Overflow goes into extra rows instead, which stay in
        // front. Honoured for an explicit column count too: the operator is choosing the shape
        // of the grid, not asking for screens they cannot see.
        if (wallAzimuthStep > 1e-4f && formationWallMaxSpanDeg > 0.0f)
        {
            int spanLimit = 1 + Mathf.FloorToInt(formationWallMaxSpanDeg * Mathf.Deg2Rad / wallAzimuthStep);
            columns = Mathf.Min(columns, Mathf.Max(1, spanLimit));
        }

        int rows = Mathf.CeilToInt(n / (float)columns);

        // Rank into rows top-down, then each row left-to-right. Two sorts rather than one
        // composite key: banding by rank keeps the rows exactly `columns` wide (so the cell
        // pitch is all that non-overlap depends on), where banding by a coordinate threshold
        // would let an unevenly spread formation pile six drones into one row.
        wallEntries.Sort(0, n, ByUpDescending);
        for (int row = 0; row < rows; row++)
        {
            int start = row * columns;
            int count = Mathf.Min(columns, n - start);
            if (count <= 0)
            {
                break;
            }

            wallEntries.Sort(start, count, ByAcrossAscending);
            for (int c = 0; c < count; c++)
            {
                int b = wallEntries[start + c].binding;
                wallColOffset[b] = c - (count - 1) * 0.5f;
                wallRowOffset[b] = (rows - 1) * 0.5f - row;
                wallRollDeg[b] = wallEntries[start + c].rollDeg;
                wallPlaced[b] = true;
            }
        }
    }

    private void UpdateFormationWallScreen(GameObject screen, int index)
    {
        if (arena == null || index >= wallPlaced.Length || !wallPlaced[index])
        {
            screen.SetActive(false);
            if (index < wallSmoothedValid.Length)
            {
                wallSmoothedValid[index] = false;
            }
            return;
        }

        float azimuth = wallAnchorAzimuth + wallColOffset[index] * wallAzimuthStep;
        float r = Mathf.Max(0.01f, radius);
        Vector3 target = new Vector3(
            arena.transform.position.x + r * Mathf.Cos(azimuth),
            arena.transform.position.y + offset.y + wallRowOffset[index] * wallRowStep,
            arena.transform.position.z + r * Mathf.Sin(azimuth));

        // Cells are re-ranked from live positions, so two drones crossing over in the formation
        // swap cells. Easing between cells makes that read as a swap rather than a teleport,
        // and the crossing is the only thing that ever puts two screens on top of each other.
        if (!wallSmoothedValid[index])
        {
            wallSmoothedPos[index] = target;
            wallSmoothedValid[index] = true;
        }
        else
        {
            wallSmoothedPos[index] = Vector3.Lerp(wallSmoothedPos[index], target, WallSmoothAlpha());
        }

        // Aimed at the wall's own vertical centre, not at the arena origin the single-row styles
        // use. With rows stacked either side of `offset.y`, aiming everything at the origin tilts
        // the whole grid down by however far the wall was raised; aiming at its centre keeps the
        // tilt symmetric, so the top row leans down and the bottom row up by the same amount.
        // `lookAtOffset` is still the operator's control on top of that.
        Vector3 aim = arena.transform.position + lookAtOffset;
        aim.y += offset.y;

        screen.transform.position = wallSmoothedPos[index];
        screen.transform.LookAt(aim);
        screen.transform.Rotate(0, 180f, 0); // textured face towards the pilot, as OUTER_CIRCLE
        screen.SetActive(true);
    }

    // FORMATION_MAP. Same cell geometry as the wall; the differences are that the azimuth is the
    // pilot's own (so the easing moves to cell space, see wallSmoothedCol) and that the screen is
    // rolled to put its imagery in the map's frame.
    private void UpdateFormationMapScreen(GameObject screen, int index)
    {
        if (arena == null || index >= wallPlaced.Length || !wallPlaced[index])
        {
            screen.SetActive(false);
            if (index < wallSmoothedValid.Length)
            {
                wallSmoothedValid[index] = false;
            }
            return;
        }

        if (!wallSmoothedValid[index])
        {
            wallSmoothedCol[index] = wallColOffset[index];
            wallSmoothedRow[index] = wallRowOffset[index];
            wallSmoothedValid[index] = true;
        }
        else
        {
            float alpha = WallSmoothAlpha();
            wallSmoothedCol[index] += alpha * (wallColOffset[index] - wallSmoothedCol[index]);
            wallSmoothedRow[index] += alpha * (wallRowOffset[index] - wallSmoothedRow[index]);
        }

        // Column offsets are *subtracted*, unlike FORMATION_WALL, so that a drone further to
        // starboard lands further to the pilot's right. In this display frame azimuth increases
        // to the pilot's LEFT: a screen sits at world offset r(cos a, 0, sin a) with a = -yaw,
        // and the head faces a == -bodyYaw (CalibrateToCentre aims it there), so the pilot's
        // right-hand direction is -d/da of that circle. OUTER_CIRCLE is the proof rather than
        // the theory: a drone yawed clockwise of the view centre gets the more negative azimuth
        // and does appear to the pilot's right, which is the whole point of the style.
        //
        // A mirrored map is not a cosmetic complaint — "the obstacle is on the right of the
        // mosaic" has to mean the pilot's right or the display is worse than no display.
        float azimuth = wallAnchorAzimuth - wallSmoothedCol[index] * wallAzimuthStep;
        float r = Mathf.Max(0.01f, radius);
        screen.transform.position = new Vector3(
            arena.transform.position.x + r * Mathf.Cos(azimuth),
            arena.transform.position.y + offset.y + wallSmoothedRow[index] * wallRowStep,
            arena.transform.position.z + r * Mathf.Sin(azimuth));

        Vector3 aim = arena.transform.position + lookAtOffset;
        aim.y += offset.y;
        screen.transform.LookAt(aim);
        screen.transform.Rotate(0, 180f, 0); // textured face towards the pilot, as OUTER_CIRCLE

        // Roll the imagery into the map frame, about the screen's own view axis.
        //
        // Sign, derived rather than tuned: a nadir feed is a plan view in its drone's heading
        // frame, so a feature at bearing `b` (clockwise from the map frame's forward) is drawn at
        // `b - droneYaw` clockwise from image-up. The panel wants it at `b` clockwise from panel-
        // up, so the image must turn clockwise by `droneYaw - mapYaw` = wallRollDeg. After the
        // 180 flip the screen's local +Z points away from the pilot, and a positive rotation
        // about an axis pointing away from the viewer reads counter-clockwise in Unity's
        // left-handed convention (the same reason +90 of yaw takes forward onto right when seen
        // from above) — hence the negation. This is the one thing here that cannot be checked
        // without a headset: if the feeds come out counter-rotated, flip this sign, not the
        // ranking. Untick formationMapRollScreens to leave every screen upright.
        if (wallRollDeg[index] != 0.0f)
        {
            screen.transform.Rotate(0.0f, 0.0f, -wallRollDeg[index], Space.Self);
        }

        screen.SetActive(true);
    }

    // Shapes the grid like the formation instead of like a square, so the screens keep the
    // arrangement the pilot would see out of the window: a wall five drones wide and two tall
    // lays out 5x2, not the 4x3 a near-square grid would pick.
    //
    // Estimates the ROW count from the aspect and derives the columns from it, rather than the
    // other way round. Rows are the small number, so rounding it to an integer costs little,
    // whereas rounding the columns directly overshoots — a 5x2 wall reads as ratio 4, and
    // round(sqrt(10 * 4)) is 6 columns, which splits the ten drones 6/4 across rows that are
    // really 5 and 5. Rounding two rows and dividing recovers 5 exactly.
    //
    // The degenerate spreads answer themselves: a single horizontal line of drones asks for one
    // row of n columns and a vertical line for n rows of one, both of which are worth getting
    // right because a one-row wall is a perfectly ordinary formation.
    private static int AutoColumnCount(int n, float spreadAcross, float spreadUp)
    {
        if (n <= 1)
        {
            return 1;
        }
        if (spreadUp <= 1e-3f)
        {
            return n;      // one horizontal line
        }
        if (spreadAcross <= 1e-3f)
        {
            return 1;      // one vertical line
        }

        int rows = Mathf.Clamp(Mathf.RoundToInt(Mathf.Sqrt(n * spreadUp / spreadAcross)), 1, n);
        return Mathf.Clamp(Mathf.CeilToInt(n / (float)rows), 1, n);
    }

    // The FPV camera, not the "Drone N" root: it is what actually produces the feed, and it is
    // the same transform PyUniSharingFast measures its planar centre-drone selection from, so
    // the wall's notion of "centre of the formation" matches the panorama's.
    private static Vector3 WallSamplePosition(DroneScreenBinding binding)
    {
        return binding.fpvCamera != null
            ? binding.fpvCamera.transform.position
            : binding.drone.transform.position;
    }

    // Heading of the frame this feed's pixels were drawn in, degrees. The FPV camera for the same
    // reason WallSamplePosition uses it, and because its yaw is what the roll has to undo.
    private static float WallSampleYawDeg(DroneScreenBinding binding)
    {
        if (binding.fpvCamera != null)
        {
            return binding.fpvCamera.transform.eulerAngles.y;
        }
        return binding.velocityControl != null && binding.velocityControl.State != null
            ? binding.velocityControl.State.Angles.y * Mathf.Rad2Deg
            : 0.0f;
    }

    private void EnsureWallArrays()
    {
        if (wallPlaced.Length == bindings.Count)
        {
            return;
        }

        wallColOffset = new float[bindings.Count];
        wallRowOffset = new float[bindings.Count];
        wallPlaced = new bool[bindings.Count];
        wallSmoothedPos = new Vector3[bindings.Count];
        wallSmoothedValid = new bool[bindings.Count];
        wallRollDeg = new float[bindings.Count];
        wallSmoothedCol = new float[bindings.Count];
        wallSmoothedRow = new float[bindings.Count];
    }

    // Frame-rate-independent exponential low-pass coefficient, matching the convention used by
    // SwarmPlaneController / AttitudeAlgorithm. Snaps outside play mode, where deltaTime is not
    // a meaningful step and an eased layout would simply never arrive.
    private float WallSmoothAlpha()
    {
        float dt = Time.deltaTime;
        if (!Application.isPlaying || formationWallSmoothTime <= 0.0f || dt <= 0.0f)
        {
            return 1.0f;
        }
        return 1.0f - Mathf.Exp(-dt / formationWallSmoothTime);
    }

    private void InvalidateFormationGrid()
    {
        wallAnchorInitialised = false;
        for (int i = 0; i < wallSmoothedValid.Length; i++)
        {
            wallSmoothedValid[i] = false;
        }
    }

    private static float WrapPi(float angle)
    {
        while (angle > Mathf.PI) angle -= 2f * Mathf.PI;
        while (angle < -Mathf.PI) angle += 2f * Mathf.PI;
        return angle;
    }

    private void UpdateInnerCircleScreen(GameObject screen, DroneScreenBinding binding)
    {

        // Get the drone's yaw
        StateFinder stateFinder = binding.velocityControl.State;
        float radians = -stateFinder.Angles.y; // Already in radians

        // Calculate the position on inner circle
        float x = arena.transform.position.x + radius * Mathf.Cos(radians) + offset.x;
        float z = arena.transform.position.z + radius * Mathf.Sin(radians);
        float y = arena.transform.position.y + offset.y;

        // Position and rotate the screen
        screen.transform.position = new Vector3(x, y, z);
        screen.transform.LookAt(arena.transform.position + lookAtOffset);
        screen.SetActive(true);
    }

    // Update the bottom circle screen positions
    private void UpdateBottomCircleScreen(GameObject screen, DroneScreenBinding binding)
    {

        // Get the drone's yaw
        StateFinder stateFinder = binding.velocityControl.State;
        float radians = -stateFinder.Angles.y; // Already in radians

        // Calculate the position on bottom circle
        float x = arena.transform.position.x + radius * Mathf.Cos(radians) + offset.x;
        float z = arena.transform.position.z + radius * Mathf.Sin(radians) + offset.z;
        float y = arena.transform.position.y + offset.y;

        // Position and rotate the screen
        screen.transform.position = new Vector3(x, y, z);
        screen.transform.LookAt(arena.transform.position + lookAtOffset + offset);
        if (invertBottomScreen)
        {
            screen.transform.Rotate(0, 180f, 0); // Invert the screen
        }

        // If double view is true then find the screens where the relative position is in the positive x direction and reverse the direction
        if (doubleView)
        {
            Vector3 relativePos = screen.transform.position - (arena.transform.position + offset);
            if (relativePos.x > 0)
            {
                screen.transform.Rotate(0, 180f, 0); // Reverse the direction
            }
        }

        screen.SetActive(true);
    }

    private void UpdateRotatingCircleScreen(GameObject screen, DroneScreenBinding binding)
    {
        if (cameraRig == null)
        {
            screen.SetActive(false);
            return;
        }

        // Get the drone's yaw
        StateFinder stateFinder = binding.velocityControl.State;
        float radians = -stateFinder.Angles.y; // Already in radians

        // Calculate base position on inner circle
        float x = arena.transform.position.x + radius * Mathf.Cos(radians);
        float z = arena.transform.position.z + radius * Mathf.Sin(radians);
        float y = arena.transform.position.y + offset.y;
        Vector3 basePosition = new Vector3(x, y, z);

        // Get the player's forward direction (only using horizontal direction).
        // The OVRCameraRig transform is just the tracking-space origin and does not
        // rotate with the head, so read the head look direction from the HMD's
        // centre-eye anchor (falling back to the rig transform if unavailable).
        Transform headTransform = cameraRig.centerEyeAnchor != null
            ? cameraRig.centerEyeAnchor
            : cameraRig.transform;
        Vector3 playerForward = headTransform.forward;
        playerForward.y = 0; // Zero out vertical component
        playerForward.Normalize();

        // Offset the screen position in the player's forward direction
        Vector3 offsetPosition = basePosition + playerForward * rotatingCircleDistance;

        // Position and rotate the screen
        screen.transform.position = offsetPosition;
        screen.transform.LookAt(arena.transform.position + playerForward * rotatingCircleDistance + lookAtOffset);
        screen.SetActive(true);
    }

    // Update the position of the screens based on real drone orientation, called from ImageSharing.cs
    public void UpdateRealDroneScreen(int i, float yaw)
    {
        // screens[i] is "screen_i" by construction (SpawnScreens creates them in
        // index order), so no name search is needed.
        if (i < 0 || i >= screens.Count)
        {
            return;
        }
        GameObject screen = screens[i];

        // Calculate the screen position based on the yaw of the real drone
        float radians = -yaw * Mathf.Deg2Rad;

        // Calculate the position on real drone circle
        float x = arena.transform.position.x + radius * Mathf.Cos(radians);
        float z = arena.transform.position.z + radius * Mathf.Sin(radians);
        float y = arena.transform.position.y;

        // Position and rotate the screen
        screen.transform.position = new Vector3(x, y, z);
        screen.transform.LookAt(arena.transform.position);
        screen.transform.Rotate(0, 180f, 0); // Face outward
        screen.SetActive(true);
    }

    // Update the positions of the screens based on the drone orientation
    void Update()
    {
        UpdateScreenPositions();
    }

    void OnValidate()
    {
        // Only update display parameters if the screen style changed
        if (screenStyle != previousScreenStyle)
        {
            previousScreenStyle = screenStyle;
            UpdateDisplayParameters();
        }
        UpdateScreenScale();
    }

    // Called when InterfaceManager parameters change
    public void OnInterfaceParamsChanged()
    {
        // InterfaceManager has just pushed its configured style straight into `screenStyle`,
        // which wipes any fallback substitution — re-apply it before the change detection
        // below, so editing an unrelated inspector field while the panorama is hidden
        // doesn't blank the feeds until the next panorama transition.
        screenStyle = ResolveScreenStyle(screenStyle);

        if (screenStyle != previousScreenStyle)
        {
            previousScreenStyle = screenStyle;
            UpdateDisplayParameters();
            
            // Send default parameters back to InterfaceManager
            SendDefaultParametersToInterfaceManager(screenStyle);
        }
        
        // Update screen scale when parameters change
        UpdateScreenScale();
        
        // Update screen positions if screens are already spawned
        UpdateScreenPositions();
    }

    // Update the parameters from the swarm manager
    void OnSwarmParamsChanged()
    {
        // Update the screen scales when swarm parameters change
        UpdateScreenScale();
    }

    // Update the scale of each screen with a factor
    public void UpdateScreenScale()
    {
        for (int i = 0; i < screens.Count; i++)
        {
            GameObject screen = screens[i];
            screen.transform.localScale = new Vector3((float)width / height, 1f, 1f) * scale;
        }
    }

    // Get the scale based on screen style
    public float GetScreenScale()
    {
        return scale;
    }

    // --- Panorama-quality fallback -------------------------------------------
    // Toggle the individual per-drone feed screens on/off as a fallback for when the
    // stitched panorama is judged bad, or the pilot switches it off (called by
    // PyUniSharingFast). Reuses the already-spawned screens; the normal
    // Update()/UpdateScreenPositions() loop then shows or hides the feeds.
    //
    // The caller's fallback style is a *substitute for a layout that shows nothing*,
    // not a layout of its own, so it only applies when the configured style is OFF.
    // Overwriting the style unconditionally loses the operator's choice (someone who
    // configured FORMATION_WALL asked for the feeds in a wall) and desyncs
    // InterfaceManager, whose inspector still reads the configured style: the feeds
    // come back in the caller's default layout when the panorama is toggled off, and
    // only recover once the style is nudged in the inspector and pushed down again.
    private ScreenStyle styleBeforeFallback = ScreenStyle.OFF;
    private ScreenStyle fallbackStyleWhenOff = ScreenStyle.OUTER_CIRCLE;
    private bool fallbackFeedsActive = false;

    public void ShowFallbackFeeds(bool on, ScreenStyle fallbackStyle)
    {
        if (on == fallbackFeedsActive)
        {
            return; // no change
        }

        if (!IsSpawned)
        {
            Debug.LogWarning("[ScreenSpawn] ShowFallbackFeeds called but no screens are spawned; cannot display individual feeds.");
        }

        if (on)
        {
            // Only read back in a scene with no InterfaceManager — see ConfiguredScreenStyle.
            styleBeforeFallback = screenStyle;
        }
        fallbackStyleWhenOff = fallbackStyle;
        fallbackFeedsActive = on;

        ScreenStyle wanted = ResolveScreenStyle(ConfiguredScreenStyle());
        bool layoutChanged = wanted != screenStyle;
        screenStyle = wanted;
        // Keep previousScreenStyle in sync so OnValidate doesn't fight us.
        previousScreenStyle = screenStyle;

        if (layoutChanged)
        {
            // Resets radius/scale/offset to the new style's defaults and restarts the
            // wall's easing — neither of which a fallback that kept the configured
            // layout has any business doing, since its screens are already in place.
            UpdateDisplayParameters();
            UpdateScreenScale();
        }
        UpdateScreenPositions();
    }

    // The style the operator configured. InterfaceManager owns it and pushes it into
    // `screenStyle` whenever its parameters change, so ask it rather than reading our own
    // field back: while a fallback substitution is in place `screenStyle` holds the
    // substitute. A snapshot taken when the fallback engaged is no better — it would undo a
    // style change made while it was active — so that snapshot is only the answer in a scene
    // with no InterfaceManager to ask (DJIScene drives ScreenSpawn from ImageSharing).
    private ScreenStyle ConfiguredScreenStyle()
    {
        if (interfaceManager == null)
        {
            interfaceManager = GetComponent<InterfaceManager>();
        }
        return interfaceManager != null ? interfaceManager.screenStyle : styleBeforeFallback;
    }

    // Configured style + the fallback substitution, which is the only thing that may
    // override it and only when it would show nothing at all.
    private ScreenStyle ResolveScreenStyle(ScreenStyle configured)
    {
        return fallbackFeedsActive && configured == ScreenStyle.OFF
            ? fallbackStyleWhenOff
            : configured;
    }
}