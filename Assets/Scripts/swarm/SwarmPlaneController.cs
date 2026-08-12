using System.Collections.Generic;
using UnityEngine;

/// <summary>
/// Owns the swarming plane. Normally the swarm is constrained to the horizontal plane
/// (SwarmManager.is3D == false, plane normal = world up). This controller adds a toggleable
/// <b>vertical</b> mode: the swarm re-forms into a wall perpendicular to a heading the pilot steers
/// with the yaw stick, so the pilot ends up looking at a billboard of drones rather than standing
/// inside a ring of them.
///
/// Nothing is teleported. The mode change only swaps the plane the swarm algorithm is constrained
/// to (OlfatiSaber / Reynolds project relative positions onto it and add a restoring term along its
/// normal), so the drones fly into the wall under the same algorithm that holds the horizontal
/// formation together.
///
/// <para><b>No drone is privileged.</b> The plane's heading is a setpoint <i>this component</i> owns
/// — seeded from the swarm's mean heading on entry, then advanced only by the yaw stick — and every
/// drone converges on it through the same feed-forward + P law
/// (<see cref="AttitudeAlgorithm.ApplyPlaneModeAttitude"/>). Its position along the normal is the
/// swarm centroid and its reference altitude is latched from that centroid, so neither is any one
/// member's. This mirrors the real fleet, where a single PC-owned <c>target_yaw</c> is held by every
/// aircraft (DJI_Swarm <c>joystick_controller.heading_hold_rate</c>) and the wall is pinned to the
/// centroid (<c>swarm_plane.py</c>).</para>
///
/// <para>The earlier design nominated one <i>anchor</i> drone: it took the yaw stick directly while
/// the rest P-tracked its compass, its swarm force was zeroed, and it supplied both the plane offset
/// and the vertical reference. That drone turned at the full stick rate while the wall lagged behind
/// it through the yaw filter, the inner rate loop and drag — one important drone and n-1 followers.
/// Steering the setpoint instead costs nothing the anchor gave: the stick still re-aims the whole
/// wall, and now it re-aims all of it at once.</para>
/// </summary>
[DefaultExecutionOrder(-100)] // must publish the plane before any drone's FixedUpdate reads it
public class SwarmPlaneController : MonoBehaviour
{
    public static SwarmPlaneController Instance { get; private set; }

    [Header("Toggle")]
    [Tooltip("Keyboard key that toggles vertical-plane swarming. TogglePlaneMode() is public so a " +
             "controller switch can drive the same thing.")]
    public KeyCode togglePlaneKey = KeyCode.V;

    [Header("Heading")]
    [Tooltip("Yaw-stick gain: deg/s the shared target heading advances at full stick. Matches the " +
             "real fleet's YAW_RATE_DEG_S. This is a feed-forward on every drone's yaw command, so " +
             "the rate actually achieved is still bounded by each drone's VelocityControl.maxYawRate " +
             "— deliberately, see maxTargetLeadDeg.")]
    public float targetYawRateDegPerSec = 60.0f;

    [Tooltip("Anti-windup: while the stick is deflected the target heading may lead the swarm's mean " +
             "heading by at most this many degrees. The stick gain exceeds what the drones can turn " +
             "at, so without the clamp a sustained turn banks up a heading debt they keep paying off " +
             "after the stick is centred — overshoot, then a wag. 0 = no clamp.")]
    public float maxTargetLeadDeg = 25.0f;

    [Header("Display")]
    [Tooltip("Switch the stitcher and the screen layout with the configuration: vertical plane " +
             "gets PLANAR + FORMATION_WALL, horizontal-with-the-gimbal-down gets PLANAR + " +
             "FORMATION_MAP, and horizontal looking out gets STABSTITCH + OUTER_CIRCLE. Untick " +
             "to keep whatever PyUniSharingFast and InterfaceManager are configured with, e.g. " +
             "to compare two stitchers on the same formation.")]
    public bool driveDisplayConfiguration = true;

    [Header("Status (read-only)")]
    [SerializeField] private bool planeModeActive = false;
    [SerializeField] private float targetHeadingDeg = 0.0f;
    [SerializeField] private int swarmDroneCount = 0;

    // World-space unit normal of the swarming plane. Vector3.up in horizontal mode, the target
    // heading in vertical mode.
    private Vector3 planeNormal = Vector3.up;

    // The shared target heading, in the same [-pi, pi] yaw space as StateFinder.Angles.y. A
    // commanded setpoint, not a measurement: nothing reads a compass into it.
    private float planeYaw = 0.0f;

    // Feed-forward that goes with planeYaw (rad/s) — the yaw stick, published so every drone gets
    // the identical value.
    private float targetYawRate = 0.0f;

    // Wall reference altitude (the vertical leash centre) and the climb-stick rate it moves at.
    private float referenceAltitude = 0.0f;
    private float referenceAltitudeRate = 3.0f;

    // Per-tick swarm aggregates, recomputed in FixedUpdate ahead of every drone's.
    private Vector3 swarmCentroid = Vector3.zero;
    private float swarmMeanYaw = 0.0f;

    // Swarm roster. swarmSpawn owns the live list that every drone's SwarmAlgorithm and
    // AttitudeAlgorithm already share, so holding that reference tracks joins and losses for free.
    // Scenes with hand-placed drones fall back to a tag search, rate-limited because
    // FindGameObjectsWithTag is far too slow for a per-tick call.
    private List<GameObject> roster;
    private bool rosterIsShared = false;
    private float lastRosterScan = float.NegativeInfinity;
    private const float RosterRescanInterval = 1.0f;

    // Display components driven by the mode change; see ApplyDisplayConfiguration.
    private PyUniSharingFast sharing;
    private InterfaceManager interfaceManager;

    // Last configuration the display was pointed at, so a change is what drives it and not the
    // mere fact of being in a configuration. Latched on the first Update rather than in Start:
    // SwarmManager pushes the inspector's gimbal pitch to FPVCameraScript in *its* Start, and
    // this component runs at -100, so a value latched in Start would read a stale level gimbal
    // and then fire a spurious switch on the first frame of a scene that starts in nadir.
    private bool displayStateLatched = false;
    private bool lastVerticalPlane = false;
    private bool lastNadirGimbal = false;

    public bool PlaneModeActive => planeModeActive;

    /// <summary>World-space unit normal of the plane the swarm is constrained to.</summary>
    public Vector3 PlaneNormal => planeNormal;

    /// <summary>
    /// Point the plane passes through: the swarm's centroid. Every drone is pulled onto the plane
    /// through this one offset along the normal, which makes the pull zero-sum — so the wall cannot
    /// drift along its own normal under its own restoring term. That is the property a pinned anchor
    /// drone provided, without making one drone the thing the wall is built around.
    /// </summary>
    public Vector3 PlaneOrigin => swarmCentroid;

    /// <summary>
    /// Altitude the wall's vertical leash is measured against
    /// (<see cref="VelocityControl.verticalReferenceAltitude"/>). Latched from the swarm's centroid
    /// altitude on entry and then moved only by the climb stick — never re-read from the drones.
    /// A live centroid would leave the formation's mean altitude a free mode: the leash would bound
    /// each drone's spread about the mean while the mean itself drifted on whatever net vertical bias
    /// the swarm forces carry (cohesion and the plane pull are zero-sum, ground repulsion is not).
    /// Same construction, and the same reason, as <c>swarm_plane.py</c>'s <c>alt_ref</c>.
    /// </summary>
    public float ReferenceAltitude => referenceAltitude;

    /// <summary>
    /// The shared target heading in radians, in StateFinder.Angles.y's [-pi, pi] yaw space. Every
    /// drone in the wall converges on this one value; it is not any drone's measured heading.
    /// </summary>
    public float TargetYaw => planeYaw;

    /// <summary>
    /// Feed-forward yaw rate (rad/s) belonging to <see cref="TargetYaw"/> — the yaw stick, handed to
    /// every drone verbatim so the whole wall starts turning on the same tick rather than waiting for
    /// each drone's P term to notice the setpoint moved.
    /// </summary>
    public float TargetYawRate => targetYawRate;

    /// <summary>
    /// In-plane basis: <paramref name="right"/> is the horizontal axis of the plane, <paramref name="up"/>
    /// the vertical one. Used to project the swarm into 2D for the convex-hull boundary estimate.
    /// </summary>
    public void GetPlaneAxes(out Vector3 right, out Vector3 up)
    {
        PlaneAxesFromNormal(planeNormal, out right, out up);
    }

    /// <summary>
    /// The in-plane basis belonging to an arbitrary plane normal, in the same convention
    /// <see cref="GetPlaneAxes"/> returns. Returns false — and the world (x, z) pair — when the
    /// plane is horizontal and the basis is therefore not determined by the normal alone.
    ///
    /// Static and public because ScreenSpawn's FORMATION_WALL needs this same basis in scenes that
    /// have no SwarmPlaneController at all (the real-drone scenes contain no swarm), where it
    /// derives the normal from the drones' shared heading instead. Sharing the construction is what
    /// keeps the display's notion of "along the wall" identical to the one the convex-hull boundary
    /// estimate is computed in.
    /// </summary>
    public static bool PlaneAxesFromNormal(Vector3 normal, out Vector3 right, out Vector3 up)
    {
        right = Vector3.Cross(Vector3.up, normal);
        if (right.sqrMagnitude < 1e-6f)
        {
            // Degenerate only if the plane is horizontal, where the caller uses (x, z) anyway.
            right = Vector3.right;
            up = Vector3.forward;
            return false;
        }
        right.Normalize();
        up = Vector3.Cross(normal, right).normalized;
        return true;
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
        // Polled rather than event-driven. The gimbal has three ways in (the SwarmManager
        // inspector, SetGimbalPitchNormalized off the joystick dial, and FPVCameraScript's own
        // fallback field) and only the first raises swarmParamsChanged, so an event subscription
        // would silently miss the dial — which is the one a pilot in a headset can actually reach.
        //
        // Ahead of the toggle so the first Update latches the starting configuration before
        // anything can change it: a V press on frame one would otherwise be swallowed by the
        // latch instead of switching the display.
        RefreshDisplayConfiguration();

        if (Input.GetKeyDown(togglePlaneKey))
        {
            TogglePlaneMode();
        }
    }

    void FixedUpdate()
    {
        if (!planeModeActive) return;

        // Losing the whole swarm is the only way the plane can be lost now: no single drone holds it
        // up, so no single drone's death can take it down.
        if (!UpdateSwarmAggregates())
        {
            Debug.LogWarning("SwarmPlaneController: no alive drones left, reverting to horizontal swarming.");
            SetPlaneMode(false);
            return;
        }

        IntegrateTargetYaw();
        planeNormal = YawToForward(planeYaw);
        IntegrateReferenceAltitude();
    }

    public void TogglePlaneMode() => SetPlaneMode(!planeModeActive);

    /// <summary>
    /// Enters or leaves vertical-plane swarming. Entering seeds the target heading from where the
    /// swarm already points and latches the wall's reference altitude, so the flip itself commands
    /// neither a turn nor a climb.
    /// </summary>
    public void SetPlaneMode(bool active)
    {
        if (active == planeModeActive) return;

        if (active)
        {
            if (!UpdateSwarmAggregates())
            {
                Debug.LogWarning("SwarmPlaneController: no alive drones found, staying in horizontal swarming.");
                return;
            }

            planeModeActive = true;
            // Seed from the swarm's *mean* heading, not from a member's: that is the one heading the
            // choice of which drone to read cannot bias, and it is already where the formation points,
            // so entering the mode asks nobody to turn.
            planeYaw = swarmMeanYaw;
            targetYawRate = 0.0f;
            planeNormal = YawToForward(planeYaw);
            referenceAltitude = swarmCentroid.y;
            referenceAltitudeRate = ResolveAltitudeRate();
            targetHeadingDeg = planeYaw * Mathf.Rad2Deg;
            Debug.Log($"SwarmPlaneController: vertical-plane swarming ON, {swarmDroneCount} drones, "
                    + $"target heading {targetHeadingDeg:F1} deg, reference altitude "
                    + $"{referenceAltitude:F1} m.");
        }
        else
        {
            planeModeActive = false;
            planeNormal = Vector3.up;
            targetYawRate = 0.0f;
            Debug.Log("SwarmPlaneController: vertical-plane swarming OFF.");
        }

        RefreshDisplayConfiguration();
    }

    /// <summary>
    /// Advances the shared target heading by the yaw stick, with anti-windup against the swarm's own
    /// mean heading.
    ///
    /// The stick is a feed-forward, so the target says where the wall is being asked to point, not
    /// where it is. Clamping the lead is what keeps a sustained turn from banking up a heading debt
    /// the drones then keep paying off after the stick is centred: the stick gain exceeds what
    /// VelocityControl.maxYawRate lets them turn at, exactly as on the real fleet, and the clamp
    /// rather than the gain is what holds the formation's headings together through a turn.
    ///
    /// The clamp acts only while the stick is deflected. Windup can only accumulate while
    /// integrating, and at centre stick the hold keeps its full authority — a disturbance that pushes
    /// the wall off heading never drags the setpoint along with it.
    /// </summary>
    private void IntegrateTargetYaw()
    {
        float normYaw = InputManager.Instance != null ? InputManager.Instance.InputStatus["yaw"] : 0.0f;
        targetYawRate = normYaw * targetYawRateDegPerSec * Mathf.Deg2Rad;

        planeYaw = WrapAngle(planeYaw + targetYawRate * Time.fixedDeltaTime);

        if (targetYawRate != 0.0f && maxTargetLeadDeg > 0.0f)
        {
            float maxLead = maxTargetLeadDeg * Mathf.Deg2Rad;
            float lead = WrapAngle(planeYaw - swarmMeanYaw);
            if (lead > maxLead)
            {
                planeYaw = WrapAngle(swarmMeanYaw + maxLead);
            }
            else if (lead < -maxLead)
            {
                planeYaw = WrapAngle(swarmMeanYaw - maxLead);
            }
        }

        targetHeadingDeg = planeYaw * Mathf.Rad2Deg;
    }

    /// <summary>
    /// Moves the wall's reference altitude with the climb stick, at the rate that stick moves each
    /// drone's own height setpoint (VelocityControl.SetNormalisedAltitudeRate). The rate is read off
    /// a drone rather than exposed as a second knob: one that disagreed would let the vertical leash
    /// clip a climb the pilot is actually commanding.
    /// </summary>
    private void IntegrateReferenceAltitude()
    {
        if (InputManager.Instance == null) return;

        float normAlt = Mathf.Clamp(InputManager.Instance.InputStatus["throttle"], -1.0f, 1.0f);
        if (normAlt == 0.0f) return;

        referenceAltitude += normAlt * referenceAltitudeRate * Time.fixedDeltaTime;
    }

    /// <summary>
    /// Recomputes the swarm-wide quantities the plane is built from — the centroid, and the circular
    /// mean of the alive drones' headings — once per tick, ahead of every drone's FixedUpdate (hence
    /// the execution order). Returns false when no alive drone is left to build a wall from.
    /// </summary>
    private bool UpdateSwarmAggregates()
    {
        EnsureRoster();
        if (roster == null) return false;

        Vector3 positionSum = Vector3.zero;
        float sumSin = 0.0f;
        float sumCos = 0.0f;
        int count = 0;

        foreach (GameObject drone in roster)
        {
            if (!SwarmRegistry.TryGet(drone, out SwarmRegistry.Entry entry)) continue;

            VelocityControl droneControl = entry.velocityControl;
            if (droneControl == null || droneControl.State == null || !droneControl.State.IsAlive) continue;

            positionSum += entry.droneParent.position;

            // Circular mean via sin/cos components: averaging the angles directly crosses the +-pi
            // seam and returns a heading no drone holds (the mean of +179 and -179 is 180, not 0).
            float yaw = droneControl.State.Angles.y;
            sumSin += Mathf.Sin(yaw);
            sumCos += Mathf.Cos(yaw);
            count++;
        }

        swarmDroneCount = count;
        if (count == 0) return false;

        swarmCentroid = positionSum / count;
        swarmMeanYaw = Mathf.Atan2(sumSin, sumCos);
        return true;
    }

    /// <summary>
    /// Resolves the swarm roster. swarmSpawn's list is live — it is the same object every drone's
    /// SwarmAlgorithm holds — so once found it never needs refreshing. Without a spawner (hand-placed
    /// drones, or a scene that builds its own swarm) fall back to the tag search, rate-limited.
    /// </summary>
    private void EnsureRoster()
    {
        if (rosterIsShared) return;
        if (roster != null && Time.time - lastRosterScan < RosterRescanInterval) return;
        lastRosterScan = Time.time;

        swarmSpawn spawner = FindObjectOfType<swarmSpawn>();
        if (spawner != null && spawner.swarm != null && spawner.swarm.Count > 0)
        {
            roster = spawner.swarm;
            rosterIsShared = true;
            return;
        }

        GameObject[] tagged = GameObject.FindGameObjectsWithTag("DroneBase");
        roster = tagged.Length > 0 ? new List<GameObject>(tagged) : null;
    }

    /// <summary>
    /// The climb-stick rate the drones themselves use. Every drone runs the same flight profile in
    /// practice, so the first one that has a VelocityControl answers for the swarm; the fallback is
    /// VelocityControl's own default.
    /// </summary>
    private float ResolveAltitudeRate()
    {
        if (roster != null)
        {
            foreach (GameObject drone in roster)
            {
                if (!SwarmRegistry.TryGet(drone, out SwarmRegistry.Entry entry)) continue;
                if (entry.velocityControl != null) return entry.velocityControl.maxAltitudeRate;
            }
        }
        return 3.0f;
    }

    /// <summary>
    /// Points the stitcher and the screen layout at the configuration the swarm is currently in,
    /// but only when that configuration has actually changed — so the operator's inspector choices
    /// stand until something moves, exactly as when this only watched the plane toggle.
    ///
    /// The configuration is two bits, because the swarm has three of them: the swarming plane
    /// (this component's own) and whether the gimbal is pitched down far enough to be imaging the
    /// ground rather than a facade (<see cref="FPVCameraScript.NadirPitch"/>, the same test
    /// <c>ScenePlaneMode.Auto</c> uses to aim its raycast — one threshold, so the panorama and
    /// the feeds can never end up in different configurations).
    /// </summary>
    private void RefreshDisplayConfiguration()
    {
        if (!driveDisplayConfiguration) return;

        bool vertical = planeModeActive;
        bool nadir = FPVCameraScript.SharedPitch <= FPVCameraScript.NadirPitch;

        if (!displayStateLatched)
        {
            lastVerticalPlane = vertical;
            lastNadirGimbal = nadir;
            displayStateLatched = true;
            return;
        }

        if (vertical == lastVerticalPlane && nadir == lastNadirGimbal) return;

        lastVerticalPlane = vertical;
        lastNadirGimbal = nadir;
        ApplyDisplayConfiguration(vertical, nadir);
    }

    /// <summary>
    /// Applies one of the three configurations. None of the pairings is taste:
    ///
    /// <list type="bullet">
    /// <item><b>Vertical plane</b> — PLANAR + FORMATION_WALL. The wall is one dominant plane with
    /// no parallax, where PLANAR's pose-driven homographies are exact and need no image content;
    /// OUTER_CIRCLE would stack every screen on one arc position because the drones share a
    /// heading.</item>
    /// <item><b>Horizontal, gimbal down</b> — PLANAR + FORMATION_MAP. Same argument for the
    /// stitcher (the ground is the dominant plane, and a nadir view of it has no parallax to
    /// speak of), and the layout is the wall's nadir counterpart: ranked in the ground plane
    /// about the pilot's heading, with the feeds rolled into that frame.</item>
    /// <item><b>Horizontal, looking out</b> — STABSTITCH + OUTER_CIRCLE. The radially-outward
    /// ring is exactly the parallax-heavy case StabStitch++'s TPS warps exist for, and its spread
    /// yaws are what make OUTER_CIRCLE separate the screens in the first place.</item>
    /// </list>
    ///
    /// The plane wins over the gimbal when both are set: a wall flying with its cameras pointed at
    /// the ground is not a configuration anything here is built for, and the plane is the one the
    /// pilot toggled deliberately. Both components refuse a change they cannot honour (the DJI
    /// scene has no camera pose for PLANAR), so neither call is asserted here.
    /// </summary>
    private void ApplyDisplayConfiguration(bool vertical, bool nadir)
    {
        // Resolved lazily and cached: this runs once per configuration change, but
        // FindObjectOfType is far too slow to reach for casually, and neither component is
        // guaranteed to exist (the swarm runs headless in some scenes).
        if (sharing == null) sharing = FindObjectOfType<PyUniSharingFast>();
        if (interfaceManager == null) interfaceManager = FindObjectOfType<InterfaceManager>();

        bool planar = vertical || nadir;

        if (sharing != null)
        {
            sharing.SetStitcherType(planar
                ? PyUniSharingFast.stitcherType.PLANAR
                : PyUniSharingFast.stitcherType.STABSTITCH);
        }

        if (interfaceManager != null)
        {
            ScreenSpawn.ScreenStyle style = vertical ? ScreenSpawn.ScreenStyle.FORMATION_WALL
                                          : nadir    ? ScreenSpawn.ScreenStyle.FORMATION_MAP
                                                     : ScreenSpawn.ScreenStyle.OUTER_CIRCLE;
            interfaceManager.SetScreenStyle(style);
        }
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
