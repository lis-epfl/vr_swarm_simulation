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

    [Header("Control Parameters")]
    public float proportionalGain = 1.0f; // Tune this value

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

        // Get the name of the parent of this gameObject
        string droneName = gameObject.transform.parent.name;
        string[] splitName = droneName.Split(' ');
        int i = int.Parse(splitName[1]);
        
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
        // droneChild.transform.position = Vector3.MoveTowards(droneChild.transform.position, targetPosition, movementSpeed * Time.deltaTime);

        // Get the max speed from the VelocityControl script
        float maxSpeed = droneChild.GetComponent<VelocityControl>().GetMaxSpeed();

        // Compute a velocity command towards the target position
        // Vector3 direction = (targetPosition - droneChild.transform.position).normalized;
        // Vector3 velocityCommand = direction * maxSpeed;

        // Proportional control
        Vector3 positionError = targetPosition - droneChild.transform.position;
        Vector3 velocityCommand = positionError * proportionalGain;

        // Clamp the velocity command to the maximum speed
        // Clamp the velocity to maximum allowed
        if (velocityCommand.magnitude > maxSpeed)
        {
            velocityCommand = velocityCommand.normalized * maxSpeed;
        }

        // Send commands to VelocityControl - ONLY X and Z
        GetComponent<VelocityControl>().swarm_vx = velocityCommand.x;
        GetComponent<VelocityControl>().swarm_vz = velocityCommand.z;
        // Don't set swarm_vy - let VelocityControl handle height!

        // Set the desired height for VelocityControl to use
        GetComponent<VelocityControl>().desired_height = height;
    }
}