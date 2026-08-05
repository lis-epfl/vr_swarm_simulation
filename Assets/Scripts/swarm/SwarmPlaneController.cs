using System.Collections.Generic;
using UnityEngine;

/// <summary>
/// Owns the swarming plane. Normally the swarm is constrained to the horizontal plane
/// (SwarmManager.is3D == false, plane normal = world up). This controller adds a toggleable
/// <b>vertical</b> mode: the swarm re-forms into a wall perpendicular to the heading of the drone
/// at the centre of stitching, so the pilot ends up looking at a billboard of drones rather than
/// standing inside a ring of them.
///
/// Nothing is teleported. The mode change only swaps the plane the swarm algorithm is constrained
/// to (OlfatiSaber / Reynolds project relative positions onto it and add a restoring term along its
/// normal), so the drones fly into the wall under the same algorithm that holds the horizontal
/// formation together.
///
/// The anchor drone's identity is captured once on entry — once every drone shares a heading, the
/// "camera yaw closest to the body yaw" rule that picks the stitch centre has no unique answer. Its
/// <i>heading</i>, however, is tracked live, so yawing the anchor re-aims the whole wall.
/// </summary>
[DefaultExecutionOrder(-100)] // must publish the plane before any drone's FixedUpdate reads it
public class SwarmPlaneController : MonoBehaviour
{
    public static SwarmPlaneController Instance { get; private set; }

    [Header("Toggle")]
    [Tooltip("Keyboard key that toggles vertical-plane swarming. TogglePlaneMode() is public so a " +
             "controller switch can drive the same thing.")]
    public KeyCode togglePlaneKey = KeyCode.V;

    [Header("Plane")]
    [Tooltip("Time constant (s) of the low-pass on the plane heading. The plane follows the anchor " +
             "drone's live heading; smoothing keeps the whole formation from chasing its jitter. " +
             "0 = no smoothing.")]
    public float planeNormalFilterTime = 0.5f;

    [Header("Display")]
    [Tooltip("Switch the stitcher and the screen layout with the swarming plane: vertical gets " +
             "PLANAR + FORMATION_WALL, horizontal gets STABSTITCH + OUTER_CIRCLE. Untick to keep " +
             "whatever PyUniSharingFast and InterfaceManager are configured with, e.g. to compare " +
             "two stitchers on the same formation.")]
    public bool driveDisplayConfiguration = true;

    [Header("Status (read-only)")]
    [SerializeField] private bool planeModeActive = false;
    [SerializeField] private string anchorDroneName = "";

    // World-space unit normal of the swarming plane. Vector3.up in horizontal mode, the anchor
    // drone's (flattened) heading in vertical mode.
    private Vector3 planeNormal = Vector3.up;

    // Heading the plane normal points along, in the same [-pi, pi] yaw space as StateFinder.Angles.y.
    // Low-passed towards the anchor's live yaw each tick.
    private float planeYaw = 0.0f;

    private Transform anchorParent;          // the anchor's "DroneParent" (carries VelocityControl)
    private GameObject anchorRoot;           // the anchor's "Drone N" root
    private VelocityControl anchorControl;

    // Display components driven by the mode change; see ApplyDisplayConfiguration.
    private PyUniSharingFast sharing;
    private InterfaceManager interfaceManager;

    public bool PlaneModeActive => planeModeActive;

    /// <summary>World-space unit normal of the plane the swarm is constrained to.</summary>
    public Vector3 PlaneNormal => planeNormal;

    /// <summary>Point the plane passes through (the anchor drone's position).</summary>
    public Vector3 PlaneOrigin => anchorParent != null ? anchorParent.position : Vector3.zero;

    /// <summary>
    /// Anchor drone's altitude. The rest of the swarm leashes its height setpoint to this, which is
    /// what bounds the wall's vertical extent and stops the formation drifting off as a whole.
    /// </summary>
    public float AnchorAltitude => anchorParent != null ? anchorParent.position.y : 0f;

    /// <summary>Anchor heading in radians, in StateFinder.Angles.y's [-pi, pi] yaw space.</summary>
    public float AnchorYaw => planeYaw;

    /// <summary>
    /// In-plane basis: <paramref name="right"/> is the horizontal axis of the plane, <paramref name="up"/>
    /// the vertical one. Used to project the swarm into 2D for the convex-hull boundary estimate.
    /// </summary>
    public void GetPlaneAxes(out Vector3 right, out Vector3 up)
    {
        right = Vector3.Cross(Vector3.up, planeNormal);
        if (right.sqrMagnitude < 1e-6f)
        {
            // Degenerate only if the plane is horizontal, where the caller uses (x, z) anyway.
            right = Vector3.right;
            up = Vector3.forward;
            return;
        }
        right.Normalize();
        up = Vector3.Cross(planeNormal, right).normalized;
    }

    /// <summary>True when <paramref name="droneParent"/> is the anchor drone's DroneParent object.</summary>
    public bool IsAnchor(GameObject droneParent)
    {
        return planeModeActive && anchorParent != null && anchorParent.gameObject == droneParent;
    }

    void Awake()
    {
        if (Instance == null)
        {
            Instance = this;
        }
        else if (Instance != this)
        {
            Destroy(this);
        }
    }

    void OnDestroy()
    {
        if (Instance == this) Instance = null;
    }

    void Update()
    {
        if (Input.GetKeyDown(togglePlaneKey))
        {
            TogglePlaneMode();
        }
    }

    void FixedUpdate()
    {
        if (!planeModeActive) return;

        // The anchor can be destroyed or crash mid-mode; try to pick a new one rather than
        // leaving the plane frozen around a corpse.
        if (!IsAnchorUsable() && !ResolveAnchor())
        {
            Debug.LogWarning("SwarmPlaneController: lost the anchor drone, reverting to horizontal swarming.");
            SetPlaneMode(false);
            return;
        }

        // Frame-rate-independent circular low-pass, matching AttitudeAlgorithm's target-heading filter.
        float rawYaw = anchorControl.State.Angles.y;
        if (planeNormalFilterTime > 0.0f)
        {
            float alpha = 1.0f - Mathf.Exp(-Time.fixedDeltaTime / planeNormalFilterTime);
            planeYaw = WrapAngle(planeYaw + alpha * WrapAngle(rawYaw - planeYaw));
        }
        else
        {
            planeYaw = rawYaw;
        }

        planeNormal = YawToForward(planeYaw);
    }

    public void TogglePlaneMode() => SetPlaneMode(!planeModeActive);

    /// <summary>
    /// Enters or leaves vertical-plane swarming. Entering resolves the anchor drone and seeds the
    /// plane from its current heading (unfiltered, so the plane starts exactly perpendicular to it).
    /// </summary>
    public void SetPlaneMode(bool active)
    {
        if (active == planeModeActive) return;

        if (active)
        {
            if (!ResolveAnchor())
            {
                Debug.LogWarning("SwarmPlaneController: no anchor drone found, staying in horizontal swarming.");
                return;
            }

            planeModeActive = true;
            planeYaw = anchorControl.State.Angles.y;
            planeNormal = YawToForward(planeYaw);
            Debug.Log($"SwarmPlaneController: vertical-plane swarming ON, anchored on {anchorDroneName}.");
        }
        else
        {
            planeModeActive = false;
            planeNormal = Vector3.up;
            anchorParent = null;
            anchorRoot = null;
            anchorControl = null;
            anchorDroneName = "";
            Debug.Log("SwarmPlaneController: vertical-plane swarming OFF.");
        }

        ApplyDisplayConfiguration(planeModeActive);
    }

    /// <summary>
    /// Points the stitcher and the screen layout at the configuration the new swarming plane
    /// calls for. Both choices follow from the plane rather than from taste:
    ///
    /// <list type="bullet">
    /// <item>the wall is one dominant plane with no parallax, where PLANAR's pose-driven
    /// homographies are exact and need no image content, while the horizontal ring is exactly
    /// the parallax-heavy case StabStitch++'s TPS warps exist for;</item>
    /// <item>OUTER_CIRCLE places each screen at its own drone's yaw, which works only because
    /// the ring spreads those yaws — in plane mode every drone shares the anchor's heading and
    /// the screens stack on one arc position, which is what FORMATION_WALL is for.</item>
    /// </list>
    ///
    /// Only ever called on an actual mode change, so the operator's inspector choices stand
    /// until the mode is first toggled. Both components refuse a change they cannot honour
    /// (the DJI scene has no camera pose for PLANAR), so neither call is asserted here.
    /// </summary>
    private void ApplyDisplayConfiguration(bool vertical)
    {
        if (!driveDisplayConfiguration) return;

        // Resolved lazily and cached: this runs once per mode change, but FindObjectOfType is
        // far too slow to reach for casually, and neither component is guaranteed to exist
        // (the swarm runs headless in some scenes).
        if (sharing == null) sharing = FindObjectOfType<PyUniSharingFast>();
        if (interfaceManager == null) interfaceManager = FindObjectOfType<InterfaceManager>();

        if (sharing != null)
        {
            sharing.SetStitcherType(vertical
                ? PyUniSharingFast.stitcherType.PLANAR
                : PyUniSharingFast.stitcherType.STABSTITCH);
        }

        if (interfaceManager != null)
        {
            interfaceManager.SetScreenStyle(vertical
                ? ScreenSpawn.ScreenStyle.FORMATION_WALL
                : ScreenSpawn.ScreenStyle.OUTER_CIRCLE);
        }
    }

    /// <summary>
    /// Picks the anchor: the drone at the centre of stitching. PyUniSharingFast publishes that
    /// selection every frame whether or not stitching is actually running; if the component is
    /// absent entirely, fall back to the alive drone whose heading is closest to the pilot body yaw
    /// (which is the same rule PyUniSharingFast applies to the FPV cameras).
    /// </summary>
    private bool ResolveAnchor()
    {
        Transform centre = PyUniSharingFast.CentreStitchDrone;
        if (centre == null || !AdoptAnchor(centre.gameObject))
        {
            if (!AdoptAnchor(FindDroneClosestToBodyYaw())) return false;
        }
        return true;
    }

    // Accepts a "Drone N" root and caches its DroneParent components.
    private bool AdoptAnchor(GameObject droneRoot)
    {
        if (droneRoot == null) return false;
        if (!SwarmRegistry.TryGet(droneRoot, out SwarmRegistry.Entry entry)) return false;
        if (entry.velocityControl == null || entry.velocityControl.State == null) return false;
        if (!entry.velocityControl.State.IsAlive) return false;

        anchorRoot = droneRoot;
        anchorParent = entry.droneParent;
        anchorControl = entry.velocityControl;
        anchorDroneName = droneRoot.name;
        return true;
    }

    private bool IsAnchorUsable()
    {
        return anchorParent != null
            && anchorControl != null
            && anchorControl.State != null
            && anchorControl.State.IsAlive;
    }

    private GameObject FindDroneClosestToBodyYaw()
    {
        float bodyYaw = PyUniSharingFast.BodyYawDegrees;
        GameObject best = null;
        float bestDiff = float.MaxValue;

        foreach (GameObject drone in GameObject.FindGameObjectsWithTag("DroneBase"))
        {
            if (!SwarmRegistry.TryGet(drone, out SwarmRegistry.Entry entry)) continue;
            if (entry.velocityControl == null || entry.velocityControl.State == null) continue;
            if (!entry.velocityControl.State.IsAlive) continue;

            float yawDeg = entry.velocityControl.State.Angles.y * Mathf.Rad2Deg;
            float diff = Mathf.Abs(Mathf.DeltaAngle(yawDeg, bodyYaw));
            if (diff < bestDiff)
            {
                bestDiff = diff;
                best = drone;
            }
        }

        return best;
    }

    // Matches StateFinder's yaw convention: forward == (sin yaw, 0, cos yaw).
    private static Vector3 YawToForward(float yawRadians)
    {
        return new Vector3(Mathf.Sin(yawRadians), 0.0f, Mathf.Cos(yawRadians));
    }

    private static float WrapAngle(float angle)
    {
        while (angle > Mathf.PI) angle -= 2f * Mathf.PI;
        while (angle < -Mathf.PI) angle += 2f * Mathf.PI;
        return angle;
    }
}
