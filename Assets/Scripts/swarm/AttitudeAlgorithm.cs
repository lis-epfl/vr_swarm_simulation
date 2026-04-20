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

    private string droneName;
    private SwarmManager swarmManager;
    private SwarmManager.AttitudeAlgorithm selectedAttitudeAlgorithm;
    private float inputYawRate = 0.0f;
    private float smoothedNeighborYaw = 0.0f;
    
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
            case SwarmManager.AttitudeAlgorithm.CONVEXHULL:
                commandedYawRate = getYawRateFromConvexHull();
                break;
            default:
                Debug.LogError("Unknown Attitude Control Algorithm selected.");
                break;
        }
        
        // Set the desired yaw rate in the velocity control script
        vc.desiredYawRate = inputYawRate;
        vc.attitude_control_yaw = commandedYawRate;

    }

    public void Reset()
    {
        vc.attitude_control_yaw = 0.0f;
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
    /// Computes the desired yaw rate based on the convex hull algorithm.
    /// </summary>
    /// <returns>The desired yaw rate in radians based on the convex hull algorithm.</returns>
    private float getYawRateFromConvexHull()
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

        // Compute the convex hull of the drone positions
        IList<Vector2> convexHull = ConvexHull.ComputeConvexHull(positions2D);

        // Now you have the convex hull of the closest neighbours including the current drone
        // You can use the convexHull list for further processing

        // Check if the current drone is in the convex hull
        Vector2 currentDronePosition = new Vector2(transform.position.x, transform.position.z);
        bool isInConvexHull = convexHull.Contains(currentDronePosition);

        // Check if not in the convex hull
        if (!isInConvexHull)
        {
            // Set the boundary estimate to false
            BoundaryEstimate = false;
            return 0.0f;
        }

        // Compute the bisector at the current drone's position
        Vector2 bisector = ConvexHull.ComputeBisector(convexHull, currentDronePosition, PointInwards);

        // Set the desired yaw rate to be the the angle between the bisector and the current heading
        float desiredYawRateDegrees = Vector2.SignedAngle(new Vector2(transform.forward.x, transform.forward.z), bisector);

        // Convert the desired yaw rate from degrees to radians
        float commandedYawRate = desiredYawRateDegrees * Mathf.Deg2Rad;

        if (commandedYawRate > 0)
        {
            commandedYawRate = Mathf.PI - commandedYawRate;
        }
        else
        {
            commandedYawRate = -Mathf.PI - commandedYawRate;
        }

        BoundaryEstimate = true;
        return commandedYawRate;
    }

    void OnSwarmParamsChanged()
    {
        selectedAttitudeAlgorithm = swarmManager.GetSelectedAttitudeAlgorithm();
        NumNeighbours = swarmManager.GetNumNeighbours();
        NumDimensions = swarmManager.GetNumDimensions();
        PointInwards = swarmManager.GetPointInwards();
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
