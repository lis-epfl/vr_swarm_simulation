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
        REAL_DRONE
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

                // Set the camera's target texture to the render texture
                if (screenStyle != ScreenStyle.OFF || screenStyle != ScreenStyle.REAL_DRONE)
                {
                    cam.GetComponent<Camera>().targetTexture = rt;
                }
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
        }
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
        }

        // Call InterfaceManager to update its display parameters
        interfaceManager.UpdateDisplayParameters(defaultRadius, defaultScale, defaultOffset, defaultLookAtOffset);
    }

    // Update the position of the screens based on the drone orientation
    void UpdateScreenPositions()
    {
        if (swarm == null || swarm.Count == 0 || screens.Count == 0)
        {
            return;
        }

        for (int i = 0; i < swarm.Count; i++)
        {
            GameObject drone = swarm.Find(d => d.name == "Drone " + i);
            GameObject droneChild = drone.transform.Find("DroneParent").gameObject;
            GameObject screen = screens.Find(s => s.name == "screen_" + i);

            // Hide the feed for any drone currently composited into the stitched
            // panorama (mirrors the BoundaryEstimate gate below). Applies to every
            // screen style.
            if (stitchedDronesToHide.Count > 0 && stitchedDronesToHide.Contains(drone))
            {
                screen.SetActive(false);
                continue;
            }

            switch (screenStyle)
            {
                case ScreenStyle.OFF:
                    HideScreen(screen);
                    break;
                case ScreenStyle.OUTER_CIRCLE:
                    UpdateOuterCircleScreen(screen, droneChild);
                    break;
                case ScreenStyle.INNER_CIRCLE:
                    UpdateInnerCircleScreen(screen, droneChild);
                    break;
                case ScreenStyle.BOTTOM_CIRCLE:
                    UpdateBottomCircleScreen(screen, droneChild);
                    break;
                case ScreenStyle.ROTATING_CIRCLE:
                    UpdateRotatingCircleScreen(screen, droneChild);
                    break;
            }
        }
    }

    private void HideScreen(GameObject screen)
    {
        screen.SetActive(false);
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

    private void UpdateOuterCircleScreen(GameObject screen, GameObject droneChild)
    {
        // Hide interior (non-boundary) drones — but only when an attitude hull
        // algorithm is actually computing BoundaryEstimate. Under attitude modes
        // NONE/SIMPLE the flag is never set (stays false for every drone), so
        // gating on it would blank all feeds — e.g. a lone drone that fell back to
        // OUTER_CIRCLE because its single feed couldn't stitch would show nothing.
        if (IsBoundaryGateActive())
        {
            AttitudeAlgorithm attitudeControl = droneChild.GetComponent<AttitudeAlgorithm>();
            if (!attitudeControl.BoundaryEstimate)
            {
                screen.SetActive(false);
                return;
            }
        }

        // Get the drone's yaw
        StateFinder stateFinder = droneChild.GetComponent<VelocityControl>().State;
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

    private void UpdateInnerCircleScreen(GameObject screen, GameObject droneChild)
    {

        // Get the drone's yaw
        StateFinder stateFinder = droneChild.GetComponent<VelocityControl>().State;
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
    private void UpdateBottomCircleScreen(GameObject screen, GameObject droneChild)
    {

        // Get the drone's yaw
        StateFinder stateFinder = droneChild.GetComponent<VelocityControl>().State;
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

    private void UpdateRotatingCircleScreen(GameObject screen, GameObject droneChild)
    {
        if (cameraRig == null)
        {
            screen.SetActive(false);
            return;
        }

        // Get the drone's yaw
        StateFinder stateFinder = droneChild.GetComponent<VelocityControl>().State;
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
        // Find the screen
        GameObject screen = screens.Find(s => s.name == "screen_" + i);

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
    // Toggle the individual per-drone feed screens on/off as a fallback for
    // when the stitched panorama is judged bad (called by PyUniSharingFast).
    // Reuses the already-spawned screens: on enable it switches to a visible
    // screen style, on disable it restores the previous style. The normal
    // Update()/UpdateScreenPositions() loop then shows or hides the feeds.
    private ScreenStyle styleBeforeFallback = ScreenStyle.OFF;
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
            styleBeforeFallback = screenStyle;
            screenStyle = fallbackStyle;
        }
        else
        {
            screenStyle = styleBeforeFallback;
        }

        // Keep previousScreenStyle in sync so OnValidate doesn't fight us.
        previousScreenStyle = screenStyle;
        fallbackFeedsActive = on;

        UpdateDisplayParameters();
        UpdateScreenScale();
        UpdateScreenPositions();
    }
}