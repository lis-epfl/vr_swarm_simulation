using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class AttitudeControl : MonoBehaviour
{
    public VelocityControl vc;
    public List<GameObject> swarm;
    public List<GameObject> neighbours;
    public int numNeighbours = 3;
    public int numDimensions = 3; // Ensure this is 3
    public bool boundaryEstimate = false; // Add this field back

    // Add a field to store the target orientation for VelocityControl
    // Make sure VelocityControl reads this value
    public Quaternion targetOrientation; 

    private string droneName;    
    
    void Start()
    {
        droneName = transform.parent.name;
        targetOrientation = transform.rotation; // Initialize with current rotation
    }

    void FixedUpdate()
    {
        // Ensure we are in 3D mode (optional check, could rely on swarmManager setting)
        if (numDimensions != 3)
        {
            Debug.LogError("AttitudeControl requires numDimensions to be 3.");
            return; 
        }
        
        // Sort the swarm by the distance to the current drone
        swarm.Sort((a, b) => 
        {
            GameObject aChild = a.transform.Find("DroneParent").gameObject;
            GameObject bChild = b.transform.Find("DroneParent").gameObject;
            return Vector3.Distance(aChild.transform.position, transform.position).CompareTo(Vector3.Distance(bChild.transform.position, transform.position));
        });

        // Get the closest numNeighbours (excluding self)
        // Ensure we don't try to get more neighbours than available in the swarm (excluding self)
        int maxPossibleNeighbours = swarm.Count > 0 ? swarm.Count - 1 : 0; 
        int actualNeighbourCount = (int)Mathf.Min(numNeighbours, maxPossibleNeighbours);

        // --- Simple 3D Boundary Estimation ---
        // If the drone has fewer neighbours than desired, consider it on the boundary.
        boundaryEstimate = actualNeighbourCount < numNeighbours;
        // --- End Boundary Estimation ---

        if (actualNeighbourCount <= 0) {
            // No neighbours, maintain current orientation or default
            targetOrientation = transform.rotation; 
            if (vc != null) vc.targetOrientation = targetOrientation; // Pass to VelocityControl
            return; // Exit if no neighbours
        }
        neighbours = swarm.GetRange(1, actualNeighbourCount);

        // --- 3D Attitude Logic ---
        // ... (rest of the 3D attitude logic as implemented previously) ...
        
        // 1. Collect 3D positions of neighbours
        List<Vector3> neighbourPositions = new List<Vector3>();
        Vector3 positionSum = Vector3.zero;
        foreach (GameObject neighbour in neighbours)
        {
            GameObject neighbourChild = neighbour.transform.Find("DroneParent").gameObject;
            Vector3 pos = neighbourChild.transform.position;
            neighbourPositions.Add(pos);
            positionSum += pos;
        }

        // 2. Calculate the centroid of the neighbours
        Vector3 centroid = positionSum / neighbourPositions.Count;

        // 3. Calculate the vector from the drone to the centroid
        Vector3 directionToCentroid = centroid - transform.position;

        // 4. Determine desired forward direction
        Vector3 desiredForward;
        if (boundaryEstimate) {
            // If on boundary, point away from centroid
             desiredForward = -directionToCentroid.normalized;
        } else {
            // If inside, maybe point towards centroid or align with neighbours (example: still point away)
            // You might want different logic for interior drones.
            desiredForward = -directionToCentroid.normalized; 
            // Alternative: Align with average neighbour forward direction? Requires getting neighbour orientations.
            // Vector3 avgNeighbourForward = Vector3.zero;
            // foreach (GameObject neighbour in neighbours) { avgNeighbourForward += neighbour.transform.Find("DroneParent").forward; }
            // desiredForward = (avgNeighbourForward / neighbours.Count).normalized;
        }


        // Ensure desiredForward is not zero
        if (desiredForward == Vector3.zero)
        {
            desiredForward = transform.forward; // Maintain current forward direction
        }

        // 5. Determine desired up direction
        Vector3 desiredUp = Vector3.up; 

        // 6. Calculate the target orientation
        targetOrientation = Quaternion.LookRotation(desiredForward, desiredUp);

        // 7. Pass the target orientation to the VelocityControl script
        if (vc != null)
        {
             vc.targetOrientation = targetOrientation; 
        } else {
             Debug.LogWarning($"VelocityControl (vc) not assigned on {droneName}");
        }

        // Note: The boundaryEstimate flag is now set, but the core logic
        // might not fully utilize it yet depending on desired behavior.
        // The example above uses it to decide the forward direction.
    }
}
