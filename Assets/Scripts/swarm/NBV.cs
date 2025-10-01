using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class NBV : MonoBehaviour
{
    public List<GameObject> swarm;

    [Header("Formation Parameters")]
    public float radius = 1.0f; // The radius of the circle
    public float height = 1.0f; // The fixed height for the swarm
    public Vector3 centerPoint = new Vector3(0, 0, 0); // The reference point for the circle
    public float movementSpeed = 1.0f; // How fast the drones move into position

    // Your NBV-specific parameters can still be used for other logic
    public float viewDistance;
    public float informationGainWeight;

    void FixedUpdate()
    {
        // Ensure there are drones in the swarm to avoid division by zero
        if (swarm == null || swarm.Count == 0)
        {
            return;
        }

        // Calculate the angle between each drone
        // 360 degrees in a circle, divided by the number of drones
        float angleStep = 360.0f / swarm.Count;

        // Loop through each drone in the swarm list
        for (int i = 0; i < swarm.Count; i++)
        {
            // Calculate the angle for this specific drone
            // We multiply by Mathf.Deg2Rad to convert degrees to radians for the trig functions
            float angle = i * angleStep * Mathf.Deg2Rad;

            // Calculate the target position using trigonometry
            // x = center.x + radius * cos(angle)
            // z = center.z + radius * sin(angle)
            float x = centerPoint.x + radius * Mathf.Cos(angle);
            float z = centerPoint.z + radius * Mathf.Sin(angle);
            
            // Create the final target position vector
            Vector3 targetPosition = new Vector3(x, height, z);
            //  debugging print
            // Debug.Log("Drone " + i + " target position: " + targetPosition);

            // Get the current drone GameObject and find the DroneParent child (like in Reynolds)
            GameObject drone = swarm[i];
            GameObject droneChild = drone.transform.Find("DroneParent").gameObject;
            
            // Debug both positions to see the difference
            // Debug.Log("Drone " + i + " parent position: " + drone.transform.position + ", child position: " + droneChild.transform.position + ", target: " + targetPosition);
            
            // Move the drone child from its current position towards the target position
            // Time.deltaTime makes the movement frame-rate independent
            droneChild.transform.position = Vector3.MoveTowards(droneChild.transform.position, targetPosition, movementSpeed * Time.deltaTime);
        }
    }
}