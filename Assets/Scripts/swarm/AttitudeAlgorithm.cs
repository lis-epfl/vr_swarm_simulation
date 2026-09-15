using System.Collections;
using System.Collections.Generic;
using Unity.XR.CoreUtils;
using UnityEngine;

/// <summary>
/// 
/// </summary>
public class AttitudeAlgorithm : MonoBehaviour
{
    public VelocityControl vc;
    public List<GameObject> swarm;
    public List<GameObject> neighbours;
    public int NumNeighbours = 3;
    public int NumDimensions = 2;
    public bool BoundaryEstimate = false;
    public bool PointInwards = false;
    [Tooltip("Outer heading-hold gain (1/s) — this component is the analogue of the real fleet's " +
             "PC-side heading_hold_rate, not of the aircraft's flight controller, which lives in " +
             "VelocityControl.headingHoldKp. Deliberately NOT the fleet's KP_YAW (0.8): the fleet " +
             "chose that for a 0.15-0.5 s transport delay each way that the sim does not model, and " +
             "at 0.8 the approach is exponential with a 1.25 s time constant that asks for the full " +
             "rate only beyond 94 deg of error. At 3 a large error flies at " +
             "VelocityControl.maxYawRate (75 deg/s) until about 25 deg out, then brakes within " +
             "maxAlpha: a 90 deg step settles within 1 deg in 2.2 s with no overshoot, relying on " +
             "VelocityControl's back-calculation. In ScaledCityWorld a drone promoted to the hull " +
             "now reaches 5 deg of its outward heading in a median 1.2-2.0 s, from 3.0-3.3 s at 0.8 " +
             "with the rate loop's old 35 deg/s ceiling. 4 is ~0.15 s faster; 5 starts to overshoot " +
             "and to ring with one tick of delay.")]
    public float YawCorrectionFactor = 3.0f;
    public float NeighborYawSmoothingFactor = 0.1f;
    [Tooltip("Seconds the hull-membership reading must persist before BoundaryEstimate flips. Prevents feed flicker.")]
    public float BoundaryHysteresisTime = 0.5f;
    [Tooltip("Time constant (s) of the low-pass on the hull-derived target heading. Keeps the yaw " +
             "setpoint steady while the hull deforms during manoeuvres; 0 = no smoothing.")]
    public float TargetHeadingFilterTime = 0.3f;

    [Header("Fast ejection when the swarm closes over a drone")]
    [Tooltip("Predicted depth inside the hull at which ejection runs at the full speedup below, as " +
             "a multiple of the live Olfati-Saber d_ref, in swarm units. 0.25 d_ref is half the " +
             "swarm's equilibrium spacing (which settles at about 0.5 d_ref), i.e. the point where " +
             "a neighbour has clearly got outside this drone. Smaller = quicker to give up.")]
    public float BoundaryDepthRatio = 0.25f;
    [Tooltip("Seconds of look-ahead applied to how fast the drone is being swallowed. This is what " +
             "catches a drone blocked by an obstacle while its neighbours stream past it: the drone " +
             "itself is stationary, so its depth is still small and only the rate the hull sweeps " +
             "over it is large. 0 = react to depth alone.")]
    public float BoundaryDepthLeadTime = 0.8f;
    [Tooltip("Time constant (s) of the low-pass on the swallow rate. The hull's vertex set changes " +
             "discretely, so the raw derivative steps whenever a neighbour joins or leaves it.")]
    public float BoundaryDepthRateFilterTime = 0.1f;
    [Tooltip("Ceiling on that accrual rate, i.e. the most the hysteresis may be shortened by. " +
             "5 turns a 0.5 s debounce into a 0.1 s floor. 1 restores the plain fixed hysteresis.")]
    public float BoundaryMaxEjectSpeedup = 5.0f;

    private string droneName;
    private SwarmManager swarmManager;
    private SwarmManager.AttitudeAlgorithm selectedAttitudeAlgorithm;
    private float inputYawRate = 0.0f;
    private float smoothedNeighborYaw = 0.0f;
    private float boundaryTimer = 0.0f;
    // Smoothed hull-derived heading setpoint, held across frames where the drone momentarily
    // drops off the hull so the yaw command stays continuous instead of chattering to zero.
    private float targetHeading = 0.0f;
    private bool hasTargetHeading = false;
    private bool wasPlaneMode = false;
    // Latched per boundary episode: set the first tick the heading is within tolerance of the
    // target, cleared only when BoundaryEstimate drops. See BoundaryFeedReady.
    private bool boundaryFeedReady = false;
    private float feedHeadingToleranceRad = 5.0f * Mathf.Deg2Rad;

    // How deep inside the hull this drone currently sits, and how fast that depth is growing.
    // Both are measured in the hull's own frame, which is the whole point: the drone this feature
    // exists for is the one blocked by an obstacle while its neighbours fly past, and that drone is
    // barely moving. Its own velocity says "not going inward"; the hull edge sweeping over it says
    // otherwise. hasBoundaryDepth distinguishes "measured zero" from "no hull to measure against",
    // so the first sample after a gap seeds the filter instead of being differenced against stale
    // state from before it.
    private float boundaryDepth = 0.0f;
    private float boundaryDepthRate = 0.0f;
    private bool hasBoundaryDepth = false;

    // Shared global hull: every drone would otherwise rebuild the identical
    // full-swarm hull every tick (O(n² log n) total). The first drone whose
    // FixedUpdate runs in a physics tick rebuilds it; the rest reuse it.
    // Transforms don't move between FixedUpdates of the same tick, so the shared
    // hull is exactly what each drone would have computed itself.
    private static readonly List<Vector2> sharedHullPositions = new List<Vector2>();
    private static IList<Vector2> sharedGlobalHull;
    private static float sharedGlobalHullTime = float.NegativeInfinity;
    private static List<GameObject> sharedGlobalHullSwarm;

    // Outward heading of each hull vertex in degrees, index-aligned with sharedGlobalHull (NaN for a
    // vertex with no usable bisector). Kept in hull order rather than sorted in place, because the
    // look-direction gap fill has to hand each shift back to the vertex it belongs to. Reused, like the
    // lists below, so the per-tick pass allocates nothing.
    private static readonly List<float> sharedHullHeadings = new List<float>();
    // The same headings, shifted and sorted, for the max-gap read-out.
    private static readonly List<float> sharedSortedHeadings = new List<float>();

    // ---- Look-direction gap fill ---------------------------------------------------------------
    // The one control quantity the shared pass produces; see UpdateLookGapShift. Per hull vertex,
    // index-aligned with sharedGlobalHull: the turn in radians added to that vertex's hull-derived
    // target heading. Exact zeros for every vertex but the pair bracketing the pilot's body yaw, and
    // for all of them while the rule is off or gated, which is what leaves the plain hull rule untouched.
    private static readonly List<float> sharedLookShiftRad = new List<float>();
    private static bool lookGapFillActive = false;

    // ---- Swarm-shape metrics -------------------------------------------------------------------
    // Read-only derivations off the hull block that is already built once per physics tick. They
    // drive no force and gate no display; they exist so the shape of the swarm is measurable, which
    // is the only way to tell whether the hollow-core feature is doing anything. Computed whether or
    // not that feature is enabled, precisely so the disabled state is what the enabled one is
    // compared against. The look-gap read-outs hold the gap fill to the same rule: the shift itself
    // is private control state above, and they only describe it.
    /// <summary>Alive drones the shape was measured over.</summary>
    public static int SharedAliveCount;
    /// <summary>Vertices of the whole-swarm convex hull — the drones whose feeds the pilot sees.</summary>
    public static int SharedHullVertexCount;
    /// <summary>Alive drones that are not hull vertices, i.e. the ones hidden inside the swarm.</summary>
    public static int SharedInteriorCount;
    /// <summary>
    /// Largest angular gap, in degrees, between the headings the hull vertices are being driven to:
    /// their outward bisectors, plus the look-direction gap fill's shift while it is acting. This is
    /// the blind sector: 360 when there is no usable hull.
    /// </summary>
    public static float SharedMaxGapDeg;
    /// <summary>
    /// How far, in degrees, the pilot's body yaw is from the nearest heading a hull vertex is driven to
    /// before the gap fill's shift (outward, or inward under PointInwards) — the look-direction blind
    /// spot the fill closes, as it stands with no fill applied. NaN while there is no body yaw or no
    /// usable hull to measure it against. Measured whether or not the rule is enabled, so a run with it
    /// off is the baseline.
    /// </summary>
    public static float SharedLookGapRawDeg = float.NaN;
    /// <summary>
    /// <see cref="SharedLookGapRawDeg"/> after the gap fill's shift, i.e. against the headings the
    /// drones are actually driven to. Equal to it whenever the rule is not turning anything.
    /// </summary>
    public static float SharedLookGapDeg = float.NaN;
    /// <summary>
    /// True on ticks the look-direction gap fill is enabled and none of its gates hold. The shift can
    /// still be zero: a drone may already be covering the look direction.
    /// </summary>
    public static bool SharedLookGapFillActive => lookGapFillActive;
    /// <summary>Mean nearest-neighbour separation in metres. Compare against d_ref * ScaleFactor.</summary>
    public static float SharedMeanNearestNeighbourM;
    /// <summary>Mean distance from the centroid in metres.</summary>
    public static float SharedRingRadiusM;
    /// <summary>Centroid in the same 2D space the hull is built in (see ProjectForHull).</summary>
    public static Vector2 SharedCentroid;
    /// <summary>Time.fixedTime the metrics above were last recomputed at.</summary>
    public static float SharedShapeTime = float.NegativeInfinity;

    /// <summary>
    /// The shared hull, in the 2D space <see cref="ProjectForHull"/> builds it in — world (x, z)
    /// normally, the plane's own axes in vertical-plane mode. Exposed for gizmos and diagnostics;
    /// treat it as read-only, it is rebuilt every tick.
    /// </summary>
    public static IList<Vector2> SharedHull => sharedGlobalHull;

    /// <summary>
    /// How far inside the swarm hull this drone currently sits, in metres. 0 while it is on the
    /// boundary. Read-only; exposed for gizmos and diagnostics.
    /// </summary>
    public float BoundaryDepthM => boundaryDepth;

    /// <summary>
    /// How fast <see cref="BoundaryDepthM"/> is growing, in metres per second — the rate the swarm
    /// is closing over this drone. Positive means being swallowed, negative means climbing back out
    /// toward the rim. Read-only; exposed for gizmos and diagnostics.
    /// </summary>
    public float BoundarySwallowRate => boundaryDepthRate;

    /// <summary>
    /// True once this boundary drone has turned onto its target heading, i.e. its FPV view is worth
    /// showing. A drone promoted from the interior is still pointing wherever it drifted to and
    /// swings round to face outward; without this its feed screen appears at the start of that swing.
    ///
    /// Latched for the boundary episode rather than re-tested every tick: once shown, the screen
    /// stays up until <see cref="BoundaryEstimate"/> drops. A continuous test would blank feeds in
    /// ordinary flight — the hull-derived target moves as the formation deforms, and in plane mode a
    /// yaw-stick turn leads the setpoint by up to <c>maxTargetLeadDeg</c> — which is flicker, not the
    /// arrival swing this exists to hide.
    /// </summary>
    public bool BoundaryFeedReady => BoundaryEstimate && boundaryFeedReady;

    [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.SubsystemRegistration)]
    private static void ResetSharedHullOnLoad()
    {
        sharedHullPositions.Clear();
        sharedHullHeadings.Clear();
        sharedSortedHeadings.Clear();
        sharedLookShiftRad.Clear();
        lookGapFillActive = false;
        sharedGlobalHull = null;
        sharedGlobalHullTime = float.NegativeInfinity;
        sharedGlobalHullSwarm = null;

        SharedAliveCount = 0;
        SharedHullVertexCount = 0;
        SharedInteriorCount = 0;
        SharedMaxGapDeg = 360.0f;
        SharedLookGapRawDeg = float.NaN;
        SharedLookGapDeg = float.NaN;
        SharedMeanNearestNeighbourM = 0.0f;
        SharedRingRadiusM = 0.0f;
        SharedCentroid = Vector2.zero;
        SharedShapeTime = float.NegativeInfinity;
    }
    
    // Awake is called before Start
    void Awake()
    {
        // Automatically assign the SwarmManager if not already set
        swarmManager = swarmManager ?? SwarmManager.Instance;
    }

    // Start is called before the first frame update
    void Start()
    {
        droneName = transform.parent.name;

        swarmManager.swarmParamsChanged += OnSwarmParamsChanged;
        // Initialize all parameters from current values in SwarmManager
        OnSwarmParamsChanged();

        if (vc == null)
        {
            Debug.LogError("VelocityControl component not assigned in AttitudeAlgorithm script on " + droneName);
        }
    }

    void FixedUpdate()
    {
        readInputs();

        // Vertical-plane swarming replaces the hull-facing heading rule entirely: the whole wall
        // points one way, along the plane's shared target heading.
        SwarmPlaneController plane = SwarmPlaneController.Instance;
        bool planeMode = plane != null && plane.PlaneModeActive;
        if (planeMode != wasPlaneMode)
        {
            // The hull-derived heading means something different on each side of the switch.
            hasTargetHeading = false;
            wasPlaneMode = planeMode;
        }
        if (planeMode)
        {
            ApplyPlaneModeAttitude(plane);
            return;
        }

        float commandedYawRate = 0.0f;
        switch(selectedAttitudeAlgorithm)
        {
            case SwarmManager.AttitudeAlgorithm.NONE:
                commandedYawRate = 0.0f;
                break;
            case SwarmManager.AttitudeAlgorithm.SIMPLE:
                commandedYawRate = getYawRateFromNeighborMean();
                break;
            case SwarmManager.AttitudeAlgorithm.LOCAL_CONVEXHULL:
                commandedYawRate = getYawRateFromLocalConvexHull();
                break;
            case SwarmManager.AttitudeAlgorithm.GLOBAL_CONVEXHULL:
                commandedYawRate = getYawRateFromGlobalConvexHull();
                break;
            default:
                Debug.LogError("Unknown Attitude Control Algorithm selected.");
                break;
        }
        
        // In the convex-hull modes the controller yaw command steers the pilot's body /
        // panorama view (the OVRCameraRig, rotated in PyUniSharingFast.UpdateBodyYaw) rather
        // than the drones — the drones hold their hull-facing heading via attitude_control_yaw.
        // Feeding inputYawRate into the drones as well would spin them redundantly, so suppress it.
        bool convexHull = selectedAttitudeAlgorithm == SwarmManager.AttitudeAlgorithm.LOCAL_CONVEXHULL
                       || selectedAttitudeAlgorithm == SwarmManager.AttitudeAlgorithm.GLOBAL_CONVEXHULL;

        // Set the desired yaw rate in the velocity control script
        vc.desiredYawRate = convexHull ? 0.0f : inputYawRate;
        vc.attitude_control_yaw = commandedYawRate;

    }

    public void Reset()
    {
        vc.attitude_control_yaw = 0.0f;
        boundaryTimer = 0.0f;
        BoundaryEstimate = false;
        boundaryFeedReady = false;
        hasTargetHeading = false;
        ClearBoundaryDepth();
    }

    /// <summary>
    /// Sets the desired yaw rate from an external controller (in radians)
    /// </summary>
    /// <param name="yawRateRadians"></param>
    public void SetYawRateFromCommand(float yawRateRadians)
    {
        inputYawRate = yawRateRadians;
    }

    /// <summary>
    /// Attitude rule for vertical-plane swarming: converge on the plane's shared target heading, so
    /// the whole wall faces one way and is readable from the pilot's seat.
    ///
    /// Every drone runs this identical law — the plane's stick feed-forward, plus a P correction on
    /// its <i>own</i> heading error against the same setpoint. There is deliberately no special case:
    /// this used to hand the yaw stick straight to an anchor drone and make the rest P-track that
    /// drone's live compass, which turned the anchor at the full stick rate while the wall trailed it
    /// through the yaw filter, the inner rate loop and drag. Steering the setpoint instead turns every
    /// drone at the same rate, and is what the real fleet does
    /// (DJI_Swarm <c>joystick_controller.heading_hold_rate</c>: one shared target heading, per-drone
    /// feed-forward + P, clamped).
    ///
    /// VelocityControl sums the two channels, low-passes them and clamps to maxYawRate, which supplies
    /// the rate limit of that helper. The real fleet's yaw deadband has no counterpart here on purpose:
    /// it exists to stop a jittering compass dithering the nose, and StateFinder's heading is exact.
    /// </summary>
    private void ApplyPlaneModeAttitude(SwarmPlaneController plane)
    {
        // BoundaryEstimate still gates the feed displays, and the usual XZ hull of a vertical wall
        // collapses to a line, so recompute it in the plane's own axes.
        UpdatePlaneBoundaryEstimate();

        // The feed-forward comes from the plane, not from this drone's own inputYawRate: the stick
        // gain then cannot differ between drones whose flight profiles differ (inputYawRate scales
        // by the per-drone maxYawRate), and the setpoint and the feed-forward driving it stay one pair.
        vc.desiredYawRate = plane.TargetYawRate;
        vc.attitude_control_yaw = YawCorrectionFactor * WrapAngle(plane.TargetYaw - vc.State.Angles.y);

        UpdateBoundaryFeedReady(true, plane.TargetYaw);
    }

    /// <summary>
    /// Debounced boundary flag from the swarm hull taken in the swarming plane's own axes. Only the
    /// convex-hull attitude modes publish a boundary today (the display gating keys off that), so
    /// this keeps the same contract while in plane mode.
    /// </summary>
    private void UpdatePlaneBoundaryEstimate()
    {
        bool hullMode = selectedAttitudeAlgorithm == SwarmManager.AttitudeAlgorithm.LOCAL_CONVEXHULL
                     || selectedAttitudeAlgorithm == SwarmManager.AttitudeAlgorithm.GLOBAL_CONVEXHULL;
        if (!hullMode || swarm == null || swarm.Count == 0)
        {
            ClearBoundaryDepth();
            UpdateBoundaryEstimate(false);
            return;
        }

        EnsureSharedGlobalHull(swarm);
        if (sharedGlobalHull == null)
        {
            ClearBoundaryDepth();
            UpdateBoundaryEstimate(false);
            return;
        }

        Vector2 planePosition = ProjectForHull(transform.position);
        UpdateBoundaryDepth(sharedGlobalHull, planePosition);
        UpdateBoundaryEstimate(sharedGlobalHull.Contains(planePosition));
    }

    /// <summary>
    /// Projects a world position into the 2D space the convex hull is built in: the XZ plane
    /// normally, or the swarming plane's own (horizontal, vertical) axes while in plane mode.
    /// Every caller must use this so hull membership can still be tested by exact equality.
    /// </summary>
    private static Vector2 ProjectForHull(Vector3 position)
    {
        SwarmPlaneController plane = SwarmPlaneController.Instance;
        if (plane == null || !plane.PlaneModeActive)
        {
            return new Vector2(position.x, position.z);
        }

        plane.GetPlaneAxes(out Vector3 planeRight, out Vector3 planeUp);
        Vector3 offset = position - plane.PlaneOrigin;
        return new Vector2(Vector3.Dot(offset, planeRight), Vector3.Dot(offset, planeUp));
    }

    /// <summary>
    /// Wraps an angle to the [-pi, pi] range to avoid discontinuities.
    /// </summary>
    private float WrapAngle(float angle)
    {
        while (angle > Mathf.PI)  angle -= 2f * Mathf.PI;
        while (angle < -Mathf.PI) angle += 2f * Mathf.PI;
        return angle;
    }

    /// <summary>
    /// Computes the desired yaw rate based on the mean yaw of all drones in the swarm (consensus-based).
    /// Uses circular mean (via unit vectors) to handle the +-pi wraparound correctly.
    /// </summary>
    private float getYawRateFromNeighborMean()
    {
        if (swarm == null || swarm.Count == 0)
        {
            return 0.0f;
        }

        // Circular mean: accumulate sin/cos components to avoid +-pi discontinuity
        float sumSin = 0.0f;
        float sumCos = 0.0f;
        int count = 0;
        foreach (GameObject drone in swarm)
        {
            if (drone != null && drone != transform.parent.gameObject)
            {
                VelocityControl droneVC = SwarmRegistry.TryGet(drone, out SwarmRegistry.Entry entry)
                    ? entry.velocityControl
                    : null;
                if (droneVC != null && droneVC.State != null && droneVC.State.IsAlive)
                {
                    float yaw = droneVC.State.Angles.y;
                    sumSin += Mathf.Sin(yaw);
                    sumCos += Mathf.Cos(yaw);
                    count++;
                }
            }
        }

        if (count == 0)
        {
            return 0.0f;
        }

        float meanSwarmYaw = Mathf.Atan2(sumSin / count, sumCos / count);

        // Apply low-pass filter on the unit vector components to avoid discontinuities
        // smoothedNeighborYaw = Mathf.Atan2(
        //     Mathf.Lerp(Mathf.Sin(smoothedNeighborYaw), Mathf.Sin(meanSwarmYaw), NeighborYawSmoothingFactor),
        //     Mathf.Lerp(Mathf.Cos(smoothedNeighborYaw), Mathf.Cos(meanSwarmYaw), NeighborYawSmoothingFactor)
        // );

        // Wrap error to [-pi, pi] to avoid discontinuity in correction
        float error = WrapAngle(meanSwarmYaw - vc.State.Angles.y);
        float targetYawRate = YawCorrectionFactor * error;

        return targetYawRate;
    }

    /// <summary>
    /// Local-hull boundary algorithm: builds a convex hull from only the current drone and its
    /// NumNeighbours nearest neighbours. Cheap, but tends to flag interior drones as boundary
    /// because the local point set is small (with few neighbours nearly every drone is a vertex).
    /// </summary>
    /// <returns>The desired yaw rate in radians.</returns>
    private float getYawRateFromLocalConvexHull()
    {
         // Log an error if the number of dimensions is not 2
        if (NumDimensions != 2)
        {
            Debug.LogError("The number of dimensions must be 2");
            return 0.0f;
        }

        // Sort the swarm by the distance to the current drone and get the closest numNeighbours
        swarm.Sort((a, b) =>
        {
            // Positions come from the spawn-time registry so the comparator doesn't
            // re-run a "DroneParent" string search twice per comparison.
            Vector3 aPos = SwarmRegistry.TryGet(a, out SwarmRegistry.Entry ea) ? ea.droneParent.position : a.transform.position;
            Vector3 bPos = SwarmRegistry.TryGet(b, out SwarmRegistry.Entry eb) ? eb.droneParent.position : b.transform.position;

            // Sort by the distance to the current drone
            return Vector3.Distance(aPos, transform.position).CompareTo(Vector3.Distance(bPos, transform.position));
        });

        // Get the closest numNeighbours
        neighbours = swarm.GetRange(1, (int)Mathf.Min(NumNeighbours, swarm.Count - 1));

        // Collect positions of the current drone and its neighbours
        List<Vector2> positions2D = new List<Vector2>
        {
            // Add the position of the current drone
            ProjectForHull(transform.position)
        };

        // Add the positions of the neighbours
        foreach (GameObject neighbour in neighbours)
        {
            if (!SwarmRegistry.TryGet(neighbour, out SwarmRegistry.Entry entry))
            {
                continue;
            }
            positions2D.Add(ProjectForHull(entry.droneParent.position));
        }

        // Compute the convex hull of the local point set (current drone + nearest neighbours)
        IList<Vector2> convexHull = ConvexHull.ComputeConvexHull(positions2D);

        Vector2 currentDronePosition = ProjectForHull(transform.position);
        // hullIsGlobal: false. Depth inside a four-point neighbour hull says little about how far
        // into the *swarm* a drone has been pushed, so this mode keeps the plain fixed hysteresis.
        return getYawRateFromHull(convexHull, currentDronePosition, false);
    }

    /// <summary>
    /// Global-hull boundary algorithm: builds the convex hull of the whole swarm, so only drones
    /// on the true outer ring are flagged as boundary. More correct than the local variant for
    /// deciding which feeds belong in the panorama / OUTER_CIRCLE, at O(n log n) per drone.
    /// </summary>
    /// <returns>The desired yaw rate in radians.</returns>
    private float getYawRateFromGlobalConvexHull()
    {
        // Log an error if the number of dimensions is not 2
        if (NumDimensions != 2)
        {
            Debug.LogError("The number of dimensions must be 2");
            return 0.0f;
        }

        if (swarm == null || swarm.Count == 0)
        {
            return 0.0f;
        }

        // The full-swarm hull is identical for every drone within a physics tick,
        // so it is built once per tick and shared (see EnsureSharedGlobalHull).
        EnsureSharedGlobalHull(swarm);

        // Everyone crashed (or filtered out): no hull to build.
        if (sharedGlobalHull == null)
        {
            ClearBoundaryDepth();
            UpdateBoundaryEstimate(false);
            return 0.0f;
        }

        Vector2 currentDronePosition = ProjectForHull(transform.position);
        return getYawRateFromHull(sharedGlobalHull, currentDronePosition, true);
    }

    /// <summary>
    /// Rebuilds the shared full-swarm hull if this is the first drone to need it this physics
    /// tick (or the swarm list changed). Positions are read once from the spawn-time registry;
    /// the current drone's own position is included and matches currentDronePosition exactly,
    /// since both read the same DroneParent transform and transforms don't move mid-tick.
    /// </summary>
    public static void EnsureSharedGlobalHull(List<GameObject> swarm)
    {
        if (swarm == null)
        {
            return;
        }
        if (sharedGlobalHullSwarm == swarm && sharedGlobalHullTime == Time.fixedTime)
        {
            return;
        }
        sharedGlobalHullSwarm = swarm;
        sharedGlobalHullTime = Time.fixedTime;

        sharedHullPositions.Clear();
        foreach (GameObject drone in swarm)
        {
            if (!SwarmRegistry.TryGet(drone, out SwarmRegistry.Entry entry))
            {
                continue;
            }
            // Skip crashed drones: they stay in the swarm list where they fell, and their stale
            // XZ position would otherwise stay a hull vertex and warp every neighbour's heading.
            VelocityControl droneVC = entry.velocityControl;
            if (droneVC != null && droneVC.State != null && !droneVC.State.IsAlive)
            {
                continue;
            }
            sharedHullPositions.Add(ProjectForHull(entry.droneParent.position));
        }

        // ComputeConvexHull cannot handle an empty point set. sortInPlace avoids its
        // defensive copy — the positions list is rebuilt from scratch next tick anyway.
        // Note it reorders sharedHullPositions but keeps every point, which is all the metrics need.
        sharedGlobalHull = sharedHullPositions.Count > 0
            ? ConvexHull.ComputeConvexHull(sharedHullPositions, sortInPlace: true)
            : null;

        // Headings first: the look-direction gap fill and the max-gap read-out both work off them, and
        // the read-out has to describe the headings after the fill has shifted them.
        ComputeHullHeadings();
        UpdateLookGapShift();
        UpdateSharedShapeMetrics();
    }

    /// <summary>
    /// Recomputes the swarm-shape read-outs from the hull block just built. Pure measurement: it
    /// writes only the Shared* statics, and nothing in the control or display path reads them.
    /// </summary>
    private static void UpdateSharedShapeMetrics()
    {
        SharedShapeTime = sharedGlobalHullTime;
        SharedAliveCount = sharedHullPositions.Count;
        SharedHullVertexCount = sharedGlobalHull != null ? sharedGlobalHull.Count : 0;
        SharedInteriorCount = Mathf.Max(0, SharedAliveCount - SharedHullVertexCount);

        if (SharedAliveCount == 0)
        {
            SharedCentroid = Vector2.zero;
            SharedRingRadiusM = 0.0f;
            SharedMeanNearestNeighbourM = 0.0f;
            SharedMaxGapDeg = 360.0f;
            return;
        }

        Vector2 sum = Vector2.zero;
        for (int i = 0; i < sharedHullPositions.Count; i++)
        {
            sum += sharedHullPositions[i];
        }
        SharedCentroid = sum / SharedAliveCount;

        float radiusSum = 0.0f;
        float nearestSum = 0.0f;
        for (int i = 0; i < sharedHullPositions.Count; i++)
        {
            radiusSum += (sharedHullPositions[i] - SharedCentroid).magnitude;

            float nearest = float.PositiveInfinity;
            for (int j = 0; j < sharedHullPositions.Count; j++)
            {
                if (i == j) continue;
                nearest = Mathf.Min(nearest, (sharedHullPositions[i] - sharedHullPositions[j]).magnitude);
            }
            if (!float.IsPositiveInfinity(nearest))
            {
                nearestSum += nearest;
            }
        }
        SharedRingRadiusM = radiusSum / SharedAliveCount;
        SharedMeanNearestNeighbourM = SharedAliveCount > 1 ? nearestSum / SharedAliveCount : 0.0f;

        SharedMaxGapDeg = ComputeMaxHeadingGapDeg();
    }

    /// <summary>
    /// Fills <see cref="sharedHullHeadings"/> with the outward heading of every hull vertex, in hull
    /// order. Outward even under <see cref="PointInwards"/>: the gaps between the headings are the
    /// same either way, so <see cref="UpdateLookGapShift"/> turns the look direction round instead.
    /// </summary>
    private static void ComputeHullHeadings()
    {
        sharedHullHeadings.Clear();

        // Fewer than three vertices is a degenerate hull (a point or a line), which neither consumer
        // measures, so there is nothing worth computing.
        if (sharedGlobalHull == null || sharedGlobalHull.Count < 3)
        {
            return;
        }

        for (int i = 0; i < sharedGlobalHull.Count; i++)
        {
            // -bisector is the outward direction, matching getYawRateFromHull's rawTargetHeading.
            Vector2 outward = -ConvexHull.ComputeBisector(sharedGlobalHull, sharedGlobalHull[i], false);
            sharedHullHeadings.Add(outward.sqrMagnitude < 1e-12f
                ? float.NaN
                : Mathf.Atan2(outward.x, outward.y) * Mathf.Rad2Deg);
        }
    }

    /// <summary>
    /// Look-direction gap fill. The hull rule points each boundary drone along its outward vertex
    /// bisector, and nothing in that rule knows where the pilot is looking. When the swarm splits
    /// around a building — the front drone stopped and swallowed, its two neighbours passing either
    /// side ahead of the lagging swarm — those two become sharp hull corners, their bisectors point
    /// 40-50 deg off the flight direction, and nothing faces the way the pilot is flying.
    ///
    /// The two hull vertices whose headings are the circular neighbours of the pilot's body yaw ψ —
    /// the ends of the hull edge facing it, the nearer of which is normally the drone
    /// SelectStitchCameras centres the panorama on — are each turned towards ψ by
    /// <c>s = clamp(d_near − lookGapCoverageDeg, 0, lookGapMaxShiftDeg)</c>, where d_near is ψ's
    /// distance from the nearer of the two. Every other vertex gets exactly zero. Three properties
    /// fall out of that one expression rather than being special-cased:
    /// <list type="bullet">
    /// <item>It is its own gate: while a drone already faces within the coverage angle of ψ, s is 0.</item>
    /// <item>It cannot overshoot: s never exceeds d_near, so neither drone crosses ψ and the headings
    /// keep their order around the circle.</item>
    /// <item>It adds no edge to dither across: the bracketing pair only changes as ψ crosses a heading,
    /// which is exactly where d_near, and so s, is zero. A vertex joining or leaving the hull still
    /// steps the target, as it always did, and the target-heading low-pass smooths it.</item>
    /// </list>
    ///
    /// Body yaw rather than the live head direction: body yaw already aims the panorama and the VR
    /// velocity frame, so it is where the pilot is flying, whereas the head moves every time the pilot
    /// glances at a side screen, and following it would turn the drones under their gaze.
    ///
    /// Computed once per tick beside the read-outs, so the drones and the numbers that describe them
    /// cannot disagree. <see cref="SharedLookGapRawDeg"/> and <see cref="SharedLookGapDeg"/> are
    /// written whenever there is a look direction to measure, whether or not the rule is enabled.
    /// </summary>
    private static void UpdateLookGapShift()
    {
        int count = sharedGlobalHull != null ? sharedGlobalHull.Count : 0;

        sharedLookShiftRad.Clear();
        for (int i = 0; i < count; i++)
        {
            sharedLookShiftRad.Add(0.0f);
        }
        lookGapFillActive = false;
        SharedLookGapRawDeg = float.NaN;
        SharedLookGapDeg = float.NaN;

        // Measurable at all: a pilot heading, a real hull, and headings that are yaws. In vertical-plane
        // mode the hull is built in the plane's own axes, where a bisector's angle is not a heading.
        SwarmPlaneController plane = SwarmPlaneController.Instance;
        if (!PyUniSharingFast.BodyYawValid
            || count < 3
            || sharedHullHeadings.Count != count
            || (plane != null && plane.PlaneModeActive))
        {
            return;
        }

        SwarmManager manager = SwarmManager.Instance;
        bool pointInwards = manager != null && manager.GetPointInwards();

        // The headings are stored outward. Drones driven inward face the other way, so bracket the
        // opposite of the look direction instead; the shift is the same signed turn in either frame.
        float lookDeg = PyUniSharingFast.BodyYawDegrees + (pointInwards ? 180.0f : 0.0f);

        // The circular neighbours of the look direction: the nearest heading below it and the nearest
        // above it. A scan rather than a sort, and indifferent to where the ±180 seam falls.
        int below = -1;
        int above = -1;
        float toBelow = float.PositiveInfinity;
        float toAbove = float.PositiveInfinity;
        for (int i = 0; i < count; i++)
        {
            float heading = sharedHullHeadings[i];
            if (float.IsNaN(heading))
            {
                // A vertex with no usable bisector leaves a hole in the circle the pair is taken from.
                return;
            }

            float fromBelow = Mathf.Repeat(lookDeg - heading, 360.0f);
            if (fromBelow < toBelow)
            {
                toBelow = fromBelow;
                below = i;
            }

            float fromAbove = Mathf.Repeat(heading - lookDeg, 360.0f);
            if (fromAbove < toAbove)
            {
                toAbove = fromAbove;
                above = i;
            }
        }

        float nearDeg = Mathf.Min(toBelow, toAbove);
        SharedLookGapRawDeg = nearDeg;
        SharedLookGapDeg = nearDeg;

        bool enabled = manager != null
            && manager.GetFillLookDirectionGap()
            // A local neighbour hull makes almost every drone a vertex, so there is no swarm-wide pair.
            && manager.GetSelectedAttitudeAlgorithm() == SwarmManager.AttitudeAlgorithm.GLOBAL_CONVEXHULL
            // A camera looking down sees the ground under its drone whichever way the drone faces.
            && FPVCameraScript.SharedPitch > FPVCameraScript.NadirPitch;
        if (!enabled)
        {
            return;
        }
        lookGapFillActive = true;

        float shiftDeg = Mathf.Clamp(nearDeg - manager.GetLookGapCoverageDeg(),
                                     0.0f, Mathf.Max(0.0f, manager.GetLookGapMaxShiftDeg()));
        // below == above only when the look direction sits on a heading, where there is nothing to fill.
        if (shiftDeg <= 0.0f || below == above)
        {
            return;
        }

        SharedLookGapDeg = nearDeg - shiftDeg;

        // Headings increase from 'below' through the look direction to 'above', so the one below turns
        // up towards it and the one above turns down.
        float shiftRad = shiftDeg * Mathf.Deg2Rad;
        sharedLookShiftRad[below] = shiftRad;
        sharedLookShiftRad[above] = -shiftRad;
    }

    /// <summary>
    /// The look-direction gap fill's turn, in radians, for the hull vertex at
    /// <paramref name="position"/>. Zero unless <paramref name="convexHull"/> is this tick's shared
    /// hull: the shifts are indexed by its vertices, and a local neighbour hull has no swarm-wide pair.
    /// </summary>
    private static float LookGapShiftFor(IList<Vector2> convexHull, Vector2 position)
    {
        if (!lookGapFillActive || convexHull == null || convexHull != sharedGlobalHull)
        {
            return 0.0f;
        }

        // Exact equality, the same test getYawRateFromHull's hull-membership check relies on.
        int index = convexHull.IndexOf(position);
        return index >= 0 && index < sharedLookShiftRad.Count ? sharedLookShiftRad[index] : 0.0f;
    }

    /// <summary>
    /// Largest angular gap between the headings the hull vertices are driven to — the pilot's widest
    /// unobserved sector. Measured off the hull's own bisectors (plus the look-direction gap fill's
    /// shift while it is acting) rather than off the drones' live yaws, so it reports the coverage the
    /// formation's *shape* affords and is not confounded by drones still turning onto their targets.
    /// </summary>
    private static float ComputeMaxHeadingGapDeg()
    {
        // Fewer than three vertices is a degenerate hull (a point or a line): there is no enclosed
        // direction set to speak of, so the honest answer is that everything is a gap.
        if (sharedGlobalHull == null || sharedGlobalHull.Count < 3)
        {
            return 360.0f;
        }

        sharedSortedHeadings.Clear();
        for (int i = 0; i < sharedHullHeadings.Count; i++)
        {
            float heading = sharedHullHeadings[i];
            if (float.IsNaN(heading))
            {
                continue;
            }

            // Measure what the drones are driven to. A shifted heading can land past ±180, and the
            // wrap-around pair below needs every heading inside one 360 window.
            float shiftRad = lookGapFillActive ? sharedLookShiftRad[i] : 0.0f;
            if (shiftRad != 0.0f)
            {
                heading = Mathf.Repeat(heading + shiftRad * Mathf.Rad2Deg + 180.0f, 360.0f) - 180.0f;
            }
            sharedSortedHeadings.Add(heading);
        }

        if (sharedSortedHeadings.Count < 2)
        {
            return 360.0f;
        }

        sharedSortedHeadings.Sort();

        float maxGap = 0.0f;
        for (int i = 0; i < sharedSortedHeadings.Count; i++)
        {
            int next = (i + 1) % sharedSortedHeadings.Count;
            float gap = sharedSortedHeadings[next] - sharedSortedHeadings[i];
            // The wrap-around pair closes the circle; every other gap is already positive.
            if (gap < 0.0f) gap += 360.0f;
            maxGap = Mathf.Max(maxGap, gap);
        }
        return maxGap;
    }

    /// <summary>
    /// Shared tail of both convex-hull algorithms. Given a hull and the current drone's position,
    /// debounces the boundary flag and maintains a smoothed target heading that faces the drone
    /// outward (or inward when PointInwards is set). The target is only *recomputed* while the
    /// drone is a hull vertex, but it is *held* (and still tracked) while the debounced
    /// BoundaryEstimate says the drone is still a boundary drone — so a momentary hull dropout
    /// during a manoeuvre no longer zeroes the yaw command and lets the heading drift.
    /// </summary>
    private float getYawRateFromHull(IList<Vector2> convexHull, Vector2 currentDronePosition, bool hullIsGlobal)
    {
        // Check if the current drone is a vertex of the convex hull
        bool onHullNow = convexHull.Contains(currentDronePosition);

        // Measure how far inside the swarm the drone is before debouncing: that is what lets a
        // drone the formation has closed over be given up on faster than one grazing the rim. Only
        // meaningful against the whole-swarm hull, so the local variant opts out rather than
        // measuring depth inside a handful of neighbours.
        if (hullIsGlobal)
        {
            UpdateBoundaryDepth(convexHull, currentDronePosition);
        }
        else
        {
            ClearBoundaryDepth();
        }

        // Publish the (debounced) boundary flag used for display gating (panorama / OUTER_CIRCLE).
        UpdateBoundaryEstimate(onHullNow);

        if (onHullNow)
        {
            // Interior angle bisector of a convex vertex points toward the swarm centroid (inward).
            Vector2 inwardBisector = ConvexHull.ComputeBisector(convexHull, currentDronePosition, false);

            // Default goal: face outward (away from centroid). PointInwards flips it to face the centroid.
            Vector2 targetDir = PointInwards ? inwardBisector : -inwardBisector;

            // forward == (sin yaw, cos yaw) in XZ, so Angles.y == Atan2(forward.x, forward.z).
            // Express the target direction in that same yaw space so the sign convention matches
            // VelocityControl.
            float rawTargetHeading = Mathf.Atan2(targetDir.x, targetDir.y);

            // Look-direction gap fill (see UpdateLookGapShift): turns the two drones either side of the
            // pilot's body yaw towards it when neither faces it. Zero for every other drone, and for all
            // of them while the rule is off. Applied before the low-pass so that a change of bracketing
            // pair is smoothed like any other hull deformation.
            float lookGapShift = LookGapShiftFor(convexHull, currentDronePosition);
            if (lookGapShift != 0.0f)
            {
                rawTargetHeading = WrapAngle(rawTargetHeading + lookGapShift);
            }

            if (!hasTargetHeading || TargetHeadingFilterTime <= 0.0f)
            {
                targetHeading = rawTargetHeading;
                hasTargetHeading = true;
            }
            else
            {
                // Frame-rate-independent circular low-pass: hull deformation during manoeuvres
                // makes the raw bisector heading thrash; the drones should not chase every jump.
                float alpha = 1.0f - Mathf.Exp(-Time.fixedDeltaTime / TargetHeadingFilterTime);
                targetHeading = WrapAngle(targetHeading + alpha * WrapAngle(rawTargetHeading - targetHeading));
            }
        }
        else if (!BoundaryEstimate)
        {
            // Debounce agrees the drone is genuinely interior: release the held heading.
            hasTargetHeading = false;
        }
        // else: momentarily off the hull but still a boundary drone per the debounced flag —
        // keep correcting toward the last hull-derived heading instead of free-drifting.

        UpdateBoundaryFeedReady(hasTargetHeading, targetHeading);

        if (!hasTargetHeading)
        {
            return 0.0f;
        }

        float error = WrapAngle(targetHeading - vc.State.Angles.y);
        return YawCorrectionFactor * error;
    }

    /// <summary>
    /// Tracks how deep inside <paramref name="convexHull"/> this drone sits and how fast that depth
    /// is growing. Called every tick a hull exists — not only while the membership reading disagrees
    /// — so the derivative stays continuous across the on-hull to off-hull transition, which is the
    /// exact moment it has to be right.
    /// </summary>
    private void UpdateBoundaryDepth(IList<Vector2> convexHull, Vector2 currentDronePosition)
    {
        float depth = ConvexHull.DistanceInsideHull(convexHull, currentDronePosition);

        if (!hasBoundaryDepth || Time.fixedDeltaTime <= 0.0f)
        {
            // First sample since a gap in the hull: seed the level and claim no rate rather than
            // differencing against state from before the gap.
            boundaryDepth = depth;
            boundaryDepthRate = 0.0f;
            hasBoundaryDepth = true;
            return;
        }

        float rawRate = (depth - boundaryDepth) / Time.fixedDeltaTime;
        boundaryDepth = depth;

        if (BoundaryDepthRateFilterTime <= 0.0f)
        {
            boundaryDepthRate = rawRate;
            return;
        }

        // Same frame-rate-independent low-pass the target heading uses. Needed here because the
        // hull's vertex set changes discretely: a neighbour joining or leaving it steps the depth,
        // and the raw derivative of a step is a spike that would eject a drone spuriously.
        float alpha = 1.0f - Mathf.Exp(-Time.fixedDeltaTime / BoundaryDepthRateFilterTime);
        boundaryDepthRate += alpha * (rawRate - boundaryDepthRate);
    }

    /// <summary>
    /// Forgets the depth measurement. "No hull to measure against" is not the same reading as
    /// "measured zero depth", and conflating them would hand the next real sample a bogus rate.
    /// </summary>
    private void ClearBoundaryDepth()
    {
        boundaryDepth = 0.0f;
        boundaryDepthRate = 0.0f;
        hasBoundaryDepth = false;
    }

    /// <summary>
    /// How much faster than real time the boundary debounce should accrue evidence, given how badly
    /// the current reading disagrees with the published flag.
    ///
    /// The plain fixed hysteresis is blind to the *size* of the disagreement: a drone shoved into
    /// the middle of the formation by an obstacle and a drone that wobbled 5 cm off the rim both
    /// take the full <see cref="BoundaryHysteresisTime"/>. This scales the accrual by a predicted
    /// depth — where the drone will be a short lead time from now if the swarm keeps closing over
    /// it — so the first case is given up on quickly and the second still isn't.
    ///
    /// The ramp is <b>quadratic</b> in that predicted depth and saturates at
    /// <see cref="BoundaryDepthRatio"/>. Squaring keeps the toe flat so a shallow excursion is
    /// nearly unaffected, and saturating means the knob names the depth at which the speedup is
    /// *fully* applied rather than merely the depth that doubles it — under an unbounded linear ramp
    /// the clamp sat four spacings out, where nothing ever reached it, and the rule did almost
    /// nothing.
    ///
    /// <b>What the numbers actually are, measured against ScaledCityWorld's geometry</b> (169
    /// buildings, obstacle cylinder radius 2.3–9.7 m; 10 drones at the 5.4 m equilibrium spacing;
    /// scale 2.70 m). A swarm flying past a building is *not* the clean picture of one drone dipping
    /// a few centimetres off an otherwise steady rim:
    /// <list type="bullet">
    /// <item>Ordinary cruise churn already takes drones 1.6–4.7 m inside the hull for 0.5–7.6 s at a
    /// time. Off-hull episodes shorter than 0.5 s essentially do not occur, so the plain hysteresis
    /// was already ejecting every one of them — the speedup changes when, not whether.</item>
    /// <item>A building encounter looks much the same: peak depth 1.7–3.9 m. Depth alone therefore
    /// does <i>not</i> separate "squeezing past a building" from "ordinary formation churn", and it
    /// was never going to — both are a drone one lattice cell inside a ten-drone blob.</item>
    /// <item>On a genuine swallow this lands at 0.24 s against the old fixed 0.50 s.</item>
    /// </list>
    ///
    /// 0.25 is the most aggressive setting that still protects the marginal case. Dropping the
    /// saturation to 0.20 d_ref reaches 0.21 s but stops protecting a slow 0.2 m creep off the rim,
    /// and 0.15 also ejects a 0.2 m bob that returns within 0.4 s. That is a poor trade: the
    /// encounters being shortened last 2.4–12.8 s, so the 30 ms on offer is about 1% of the episode,
    /// bought by giving up the flicker margin this debounce exists for.
    ///
    /// <b>Known weak spot.</b> The rule helps least at the widest buildings, which is the opposite of
    /// what intuition suggests. At a 19.4 m cylinder a ten-drone swarm (~16 m across) cannot split
    /// around it: instead of flowing past and swallowing the blocked drone, the whole formation
    /// stalls and piles up, and the encounter depths come out *shallower* (median peak 1.67 m) than
    /// at a 13.5 m building (3.35 m), where it does split cleanly. Nothing here detects "the swarm
    /// has stopped"; a pilot who wants the feeds culled in that case has to widen the formation.
    ///
    /// Three properties, all of which fall out rather than being special-cased:
    /// <list type="bullet">
    /// <item>Depth is 0 by construction whenever the drone <i>is</i> a hull vertex, so this returns
    /// 1 on the rejoin direction. Rejoining the boundary still costs the full hysteresis and cannot
    /// flicker — there is no branch on direction here and none should be added.</item>
    /// <item>Clamping the predicted <i>depth</i> at zero (rather than clamping the rate) is what
    /// makes retreat work: a drone climbing back out has a negative swallow rate, so its prediction
    /// collapses to 0 and it gets the full unaccelerated hysteresis instead of being ejected on its
    /// way back to the rim.</item>
    /// <item>Scaling by the lattice spacing keeps the knob dimensionless, so it survives a change of
    /// scene scale or swarm size without retuning.</item>
    /// </list>
    ///
    /// That scale is the <b>commanded</b> spacing (Olfati-Saber's live <c>d_ref</c>), deliberately
    /// not the measured mean nearest-neighbour separation. The measured figure co-varies with the
    /// very thing being measured: a swarm squeezing past an obstacle compresses, which shrinks the
    /// yardstick, which inflates <c>depth / scale</c> and ejects drones faster for reasons having
    /// nothing to do with their own situation — and it does so in exactly the manoeuvre this rule
    /// exists for. Measured on this scene's numbers (d_ref 1.08, scaleFactor 10), a formation
    /// compressed to 60% of its equilibrium spacing drove the peak speedup from 2.2x to 4.4x and
    /// halved the ejection time for a drone sitting at a constant depth. <c>d_ref</c> is a setpoint
    /// and has no such feedback; it is also always available, where
    /// <see cref="SharedMeanNearestNeighbourM"/> is only fresh on ticks a hull was built, and it
    /// tracks the pilot's spread stick the moment it moves rather than after the swarm converges.
    ///
    /// The 0.25 default is that argument's one wrinkle: the swarm's equilibrium spacing settles at
    /// about <i>half</i> d_ref (measured, and noted twice in <see cref="SwarmManager"/>), so a
    /// distance meant to read as "half a neighbour spacing" is a quarter of d_ref, not a half. This
    /// is the same reason <c>coreRadiusFraction</c> is measured rather than predicted from d_ref —
    /// there the factor of two could not be folded into a constant because the core has to track
    /// where the drones actually are, whereas a lattice cell is precisely what d_ref defines. Cf.
    /// <c>coreStandoffRatio</c>, which is a multiple of the live d_ref for this same reason.
    /// </summary>
    private float BoundaryEjectSpeedup()
    {
        if (!hasBoundaryDepth || BoundaryMaxEjectSpeedup <= 1.0f)
        {
            return 1.0f;
        }

        if (swarmManager == null)
        {
            return 1.0f;
        }

        // d_ref is in swarm units; depth is in world metres, hence the scaleFactor.
        float scale = BoundaryDepthRatio * swarmManager.GetDRef() * swarmManager.GetScaleFactor();
        if (scale <= 0.0f)
        {
            return 1.0f;
        }

        float predictedDepth = Mathf.Max(0.0f, boundaryDepth + BoundaryDepthLeadTime * boundaryDepthRate);
        float frac = Mathf.Clamp01(predictedDepth / scale);
        return 1.0f + (BoundaryMaxEjectSpeedup - 1.0f) * frac * frac;
    }

    /// <summary>
    /// Symmetric debounce for <see cref="BoundaryEstimate"/>: a hull-membership reading that
    /// disagrees with the published flag must persist for <see cref="BoundaryHysteresisTime"/>
    /// before we commit the flip. Display gating (panorama / OUTER_CIRCLE) therefore stays stable
    /// even as a drone jitters across the hull edge.
    ///
    /// The wait is shortened — never lengthened — in proportion to how deep inside the swarm the
    /// drone has been pushed; see <see cref="BoundaryEjectSpeedup"/>. That is a continuous scaling
    /// of the existing timer rather than a second threshold, deliberately: a threshold would give
    /// drones a new edge to dither across, which is the failure this debounce exists to prevent.
    /// </summary>
    private void UpdateBoundaryEstimate(bool onHullNow)
    {
        if (onHullNow == BoundaryEstimate)
        {
            boundaryTimer = 0.0f;
            return;
        }

        boundaryTimer += Time.fixedDeltaTime * BoundaryEjectSpeedup();
        if (boundaryTimer >= BoundaryHysteresisTime)
        {
            BoundaryEstimate = onHullNow;
            boundaryTimer = 0.0f;
        }

        // Every path that drops the flag goes through here, including the early returns that have
        // no hull or no target to test against, so the latch is released in one place.
        if (!BoundaryEstimate)
        {
            boundaryFeedReady = false;
        }
    }

    /// <summary>
    /// Advances the <see cref="BoundaryFeedReady"/> latch against the heading this drone is being
    /// driven to. Called after the boundary flag has been updated for the tick, so a drone promoted
    /// this tick is tested against the target computed in the same tick.
    /// </summary>
    private void UpdateBoundaryFeedReady(bool haveTarget, float targetYaw)
    {
        if (!BoundaryEstimate || boundaryFeedReady || !haveTarget)
        {
            return;
        }

        if (Mathf.Abs(WrapAngle(targetYaw - vc.State.Angles.y)) <= feedHeadingToleranceRad)
        {
            boundaryFeedReady = true;
        }
    }

    void OnSwarmParamsChanged()
    {
        selectedAttitudeAlgorithm = swarmManager.GetSelectedAttitudeAlgorithm();
        NumNeighbours = swarmManager.GetNumNeighbours();
        NumDimensions = swarmManager.GetNumDimensions();
        PointInwards = swarmManager.GetPointInwards();
        // 180 must always pass, and pi in degrees-to-radians need not compare >= WrapAngle's pi, so
        // the disabled setting is made unconditional rather than left to a float comparison.
        float toleranceDeg = swarmManager.GetFeedHeadingToleranceDeg();
        feedHeadingToleranceRad = toleranceDeg >= 180.0f ? float.PositiveInfinity : Mathf.Max(0.0f, toleranceDeg) * Mathf.Deg2Rad;

        // Parameter/algorithm changes invalidate the held hull heading (e.g. PointInwards flips
        // the goal 180 degrees); drop it so the next hull pass rebuilds it from scratch. The depth
        // measurement goes with it: switching attitude algorithm changes which hull it was taken
        // against, so differencing across the switch would invent a swallow rate.
        hasTargetHeading = false;
        ClearBoundaryDepth();
    }

    void OnDestroy()
    {
        if (swarmManager != null)
        {
            swarmManager.swarmParamsChanged -= OnSwarmParamsChanged;        
        }
    }

    private void readInputs()
    {
        if (InputManager.Instance != null)
        {
            float normYaw = InputManager.Instance.InputStatus["yaw"];
            inputYawRate = normYaw * vc.maxYawRate;
        }
    }
}
