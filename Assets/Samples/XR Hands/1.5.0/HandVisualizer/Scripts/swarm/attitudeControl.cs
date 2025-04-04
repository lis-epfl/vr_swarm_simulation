using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class AttitudeControl : MonoBehaviour
{
    public VelocityControl vc;
    public List<GameObject> swarm;
    public List<GameObject> neighbours;
    public int numNeighbours = 3;
    public int numDimensions = 2;
    public bool boundaryEstimate = false;

    private string droneName;    
    
    void Start()
    {
        droneName = transform.parent.name;
    }

    void FixedUpdate()
    {
        
        // Log an error if the number of dimensions is not 2
        if (numDimensions != 2)
        {
            Debug.LogError("The number of dimensions must be 2");
            return;
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
        neighbours = swarm.GetRange(1, (int)Mathf.Min(numNeighbours, swarm.Count - 1));

        // Collect positions of the current drone and its neighbours
        List<Vector2> positions2D = new List<Vector2>();

        // Add the position of the current drone
        positions2D.Add(new Vector2(transform.position.x, transform.position.z));

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
            boundaryEstimate = false;
            
            return;
        }

        // Find the location of the current drone in the convex hull
        int locationInConvexHull = convexHull.IndexOf(currentDronePosition);

        // Get the edges adjacent to the current drone
        int nextIndex = (locationInConvexHull + 1) % convexHull.Count;
        int previousIndex = ((locationInConvexHull - 1) + convexHull.Count) % convexHull.Count;

        Vector2 edge1 = convexHull[nextIndex] - convexHull[locationInConvexHull];
        Vector2 edge2 = convexHull[previousIndex] - convexHull[locationInConvexHull];

        // Get the bisector of the first and last edge
        Vector2 bisector = (edge1 + edge2).normalized;

        // Set the desired yaw rate to be the the angle between the bisector and the current heading
        float desiredYawRateDegrees = Vector2.SignedAngle(new Vector2(transform.forward.x, transform.forward.z), bisector);

        // Convert the desired yaw rate from degrees to radians
        float desiredYawRateRadians = desiredYawRateDegrees * Mathf.Deg2Rad;

        if (desiredYawRateRadians > 0)
        {
            desiredYawRateRadians = Mathf.PI - desiredYawRateRadians;
        } else
        {
            desiredYawRateRadians = -Mathf.PI - desiredYawRateRadians;
        }

        // Set the desired yaw rate in the velocity control script
        vc.attitude_control_yaw = desiredYawRateRadians;

        // Set the boundary estimate to true
        boundaryEstimate = true;


    }
}
