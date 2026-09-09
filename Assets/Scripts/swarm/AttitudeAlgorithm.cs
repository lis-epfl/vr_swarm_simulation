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
    public float YawCorrectionFactor = 1.0f;
    public float NeighborYawSmoothingFactor = 0.1f;
    [Tooltip("Seconds the hull-membership reading must persist before BoundaryEstimate flips. Prevents feed flicker.")]
    public float BoundaryHysteresisTime = 0.5f;
    [Tooltip("Time constant (s) of the low-pass on the hull-derived target heading. Keeps the yaw " +
             "setpoint steady while the hull deforms during manoeuvres; 0 = no smoothing.")]
    public float TargetHeadingFilterTime = 0.3f;

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

    // Shared global hull: every drone would otherwise rebuild the identical
    // full-swarm hull every tick (O(n² log n) total). The first drone whose
    // FixedUpdate runs in a physics tick rebuilds it; the rest reuse it.
    // Transforms don't move between FixedUpdates of the same tick, so the shared
    // hull is exactly what each drone would have computed itself.
    private static readonly List<Vector2> sharedHullPositions = new List<Vector2>();
    private static IList<Vector2> sharedGlobalHull;
    private static float sharedGlobalHullTime = float.NegativeInfinity;
    private static List<GameObject> sharedGlobalHullSwarm;

    // Scratch for the outward headings of the hull vertices, reused so the per-tick metrics pass
    // allocates nothing.
    private static readonly List<float> sharedHullHeadings = new List<float>();

    // ---- Swarm-shape metrics -------------------------------------------------------------------
    // Read-only derivations off the hull block that is already built once per physics tick. They
    // drive no force and gate no display; they exist so the shape of the swarm is measurable, which
    // is the only way to tell whether the hollow-core feature is doing anything. Computed whether or
    // not that feature is enabled, precisely so the disabled state is what the enabled one is
    // compared against.
    /// <summary>Alive drones the shape was measured over.</summary>
    public static int SharedAliveCount;
    /// <summary>Vertices of the whole-swarm convex hull — the drones whose feeds the pilot sees.</summary>
    public static int SharedHullVertexCount;
    /// <summary>Alive drones that are not hull vertices, i.e. the ones hidden inside the swarm.</summary>
    public static int SharedInteriorCount;
    /// <summary>
    /// Largest angular gap, in degrees, between the outward headings the hull vertices are being
    /// driven to. This is the blind sector: 360 when there is no usable hull.
    /// </summary>
    public static float SharedMaxGapDeg;
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

    [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.SubsystemRegistration)]
    private static void ResetSharedHullOnLoad()
    {
        sharedHullPositions.Clear();
        sharedHullHeadings.Clear();
        sharedGlobalHull = null;
        sharedGlobalHullTime = float.NegativeInfinity;
        sharedGlobalHullSwarm = null;

        SharedAliveCount = 0;
        SharedHullVertexCount = 0;
        SharedInteriorCount = 0;
        SharedMaxGapDeg = 360.0f;
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
        hasTargetHeading = false;
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
            UpdateBoundaryEstimate(false);
            return;
        }

        EnsureSharedGlobalHull(swarm);
        if (sharedGlobalHull == null)
        {
            UpdateBoundaryEstimate(false);
            return;
        }

        UpdateBoundaryEstimate(sharedGlobalHull.Contains(ProjectForHull(transform.position)));
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
        return getYawRateFromHull(convexHull, currentDronePosition);
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
            UpdateBoundaryEstimate(false);
            return 0.0f;
        }

        Vector2 currentDronePosition = ProjectForHull(transform.position);
        return getYawRateFromHull(sharedGlobalHull, currentDronePosition);
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
    /// Largest angular gap between the outward headings of the hull vertices — the pilot's widest
    /// unobserved sector. Measured off the hull's own bisectors rather than off the drones' live
    /// yaws, so it reports the coverage the formation's *shape* affords and is not confounded by
    /// drones still turning onto their targets.
    /// </summary>
    private static float ComputeMaxHeadingGapDeg()
    {
        // Fewer than three vertices is a degenerate hull (a point or a line): there is no enclosed
        // direction set to speak of, so the honest answer is that everything is a gap.
        if (sharedGlobalHull == null || sharedGlobalHull.Count < 3)
        {
            return 360.0f;
        }

        sharedHullHeadings.Clear();
        for (int i = 0; i < sharedGlobalHull.Count; i++)
        {
            // -bisector is the outward direction, matching getYawRateFromHull's rawTargetHeading.
            Vector2 outward = -ConvexHull.ComputeBisector(sharedGlobalHull, sharedGlobalHull[i], false);
            if (outward.sqrMagnitude < 1e-12f)
            {
                continue;
            }
            sharedHullHeadings.Add(Mathf.Atan2(outward.x, outward.y) * Mathf.Rad2Deg);
        }

        if (sharedHullHeadings.Count < 2)
        {
            return 360.0f;
        }

        sharedHullHeadings.Sort();

        float maxGap = 0.0f;
        for (int i = 0; i < sharedHullHeadings.Count; i++)
        {
            int next = (i + 1) % sharedHullHeadings.Count;
            float gap = sharedHullHeadings[next] - sharedHullHeadings[i];
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
    private float getYawRateFromHull(IList<Vector2> convexHull, Vector2 currentDronePosition)
    {
        // Check if the current drone is a vertex of the convex hull
        bool onHullNow = convexHull.Contains(currentDronePosition);

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

        if (!hasTargetHeading)
        {
            return 0.0f;
        }

        float error = WrapAngle(targetHeading - vc.State.Angles.y);
        return YawCorrectionFactor * error;
    }

    /// <summary>
    /// Symmetric debounce for <see cref="BoundaryEstimate"/>: a hull-membership reading that
    /// disagrees with the published flag must persist for <see cref="BoundaryHysteresisTime"/>
    /// before we commit the flip. Display gating (panorama / OUTER_CIRCLE) therefore stays stable
    /// even as a drone jitters across the hull edge.
    /// </summary>
    private void UpdateBoundaryEstimate(bool onHullNow)
    {
        if (onHullNow == BoundaryEstimate)
        {
            boundaryTimer = 0.0f;
            return;
        }

        boundaryTimer += Time.fixedDeltaTime;
        if (boundaryTimer >= BoundaryHysteresisTime)
        {
            BoundaryEstimate = onHullNow;
            boundaryTimer = 0.0f;
        }
    }

    void OnSwarmParamsChanged()
    {
        selectedAttitudeAlgorithm = swarmManager.GetSelectedAttitudeAlgorithm();
        NumNeighbours = swarmManager.GetNumNeighbours();
        NumDimensions = swarmManager.GetNumDimensions();
        PointInwards = swarmManager.GetPointInwards();

        // Parameter/algorithm changes invalidate the held hull heading (e.g. PointInwards flips
        // the goal 180 degrees); drop it so the next hull pass rebuilds it from scratch.
        hasTargetHeading = false;
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
