using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class AttitudeAlgorithm : MonoBehaviour
{
    public VelocityControl vc;
    public List<GameObject> swarm;
    public List<GameObject> neighbours;
    public int NumNeighbours = 3;
    public int NumDimensions = 2;
    public bool BoundaryEstimate = false;
    public bool PointInwards = false;

    private string droneName;    
    private SwarmManager swarmManager;
    private SwarmManager.AttitudeAlgorithm selectedAttitudeAlgorithm;
    private float inputYawRate = 0.0f;
    
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
    }

    void FixedUpdate()
    {
        float desiredYawRateRadians = 0.0f;
        switch(selectedAttitudeAlgorithm)
        {
            case SwarmManager.AttitudeAlgorithm.NONE:
                // Force yaw rate to zero
                desiredYawRateRadians = 0.0f;
                break;
            case SwarmManager.AttitudeAlgorithm.SIMPLE:
                desiredYawRateRadians = inputYawRate;
                break;
            case SwarmManager.AttitudeAlgorithm.CONVEXHULL:
                desiredYawRateRadians = getYawRateFromConvexHull();
                break;
            default:
                Debug.LogError("Unknown Attitude Control Algorithm selected.");
                break;
        }
        
        // Set the desired yaw rate in the velocity control script
        vc.attitude_control_yaw = desiredYawRateRadians;

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
        float desiredYawRateRadians = desiredYawRateDegrees * Mathf.Deg2Rad;

        if (desiredYawRateRadians > 0)
        {
            desiredYawRateRadians = Mathf.PI - desiredYawRateRadians;
        }
        else
        {
            desiredYawRateRadians = -Mathf.PI - desiredYawRateRadians;
        }

        BoundaryEstimate = true;
        return desiredYawRateRadians;
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
}
