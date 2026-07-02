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
                VelocityControl droneVC = drone.GetNamedChild("DroneParent").GetComponent<VelocityControl>();
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
            // Get the child of the neighbours
            GameObject aChild = a.transform.Find("DroneParent").gameObject;
            GameObject bChild = b.transform.Find("DroneParent").gameObject;

            // Sort by the distance to the current drone
            return Vector3.Distance(aChild.transform.position, transform.position).CompareTo(Vector3.Distance(bChild.transform.position, transform.position));
        });

        // Get the closest numNeighbours
        neighbours = swarm.GetRange(1, (int)Mathf.Min(NumNeighbours, swarm.Count - 1));

        // Collect positions of the current drone and its neighbours
        List<Vector2> positions2D = new List<Vector2>
        {
            // Add the position of the current drone
            new Vector2(transform.position.x, transform.position.z)
        };

        // Add the positions of the neighbours
        foreach (GameObject neighbour in neighbours)
        {
            GameObject neighbourChild = neighbour.transform.Find("DroneParent").gameObject;
            Vector3 position = neighbourChild.transform.position;
            positions2D.Add(new Vector2(position.x, position.z));
        }

        // Compute the convex hull of the local point set (current drone + nearest neighbours)
        IList<Vector2> convexHull = ConvexHull.ComputeConvexHull(positions2D);

        Vector2 currentDronePosition = new Vector2(transform.position.x, transform.position.z);
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

        // Collect every drone's position. The current drone is part of the swarm list, so its own
        // position is included (and matches currentDronePosition below since both read the same
        // DroneParent transform within this frame).
        List<Vector2> positions2D = new List<Vector2>(swarm.Count);
        foreach (GameObject drone in swarm)
        {
            Transform droneParent = drone.transform.Find("DroneParent");
            if (droneParent == null)
            {
                continue;
            }
            // Skip crashed drones: they stay in the swarm list where they fell, and their stale
            // XZ position would otherwise stay a hull vertex and warp every neighbour's heading.
            VelocityControl droneVC = droneParent.GetComponent<VelocityControl>();
            if (droneVC != null && droneVC.State != null && !droneVC.State.IsAlive)
            {
                continue;
            }
            Vector3 position = droneParent.position;
            positions2D.Add(new Vector2(position.x, position.z));
        }

        // Everyone crashed (or filtered out): no hull to build, and ComputeConvexHull
        // cannot handle an empty point set.
        if (positions2D.Count == 0)
        {
            UpdateBoundaryEstimate(false);
            return 0.0f;
        }

        // Compute the convex hull of the entire swarm
        IList<Vector2> convexHull = ConvexHull.ComputeConvexHull(positions2D);

        Vector2 currentDronePosition = new Vector2(transform.position.x, transform.position.z);
        return getYawRateFromHull(convexHull, currentDronePosition);
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
