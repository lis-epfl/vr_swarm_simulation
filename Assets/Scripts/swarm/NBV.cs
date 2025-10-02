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

    [Header("Debug Options")]
    public bool debug_bool = false; // Toggle debug arrows for yaw visualization

    [Header("Camera Control")]
    // public float cameraPitch = 0.0f; // Camera pitch angle in degrees (+ = looking down, - = looking up)
    [Range(-90f, 35f)] public float cameraPitch = 0.0f; // Limited range in Unity Inspector
    public bool enableCameraPitchControl = false; // Toggle camera pitch control

    // Private variables
    private float lastCameraPitch = float.MinValue; // To track changes

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

        // Attitude control - point toward center
        CalculateYawTowardCenter(droneChild);

        // NEW: Camera pitch control
        if (enableCameraPitchControl)
        {
            ControlCameraPitch(drone);
        }
    }

    void CalculateYawTowardCenter(GameObject droneChild)
    {
        // Get current drone position (only X and Z matter for yaw)
        Vector3 dronePosition = droneChild.transform.position;

        // Calculate direction from drone to center point (only X and Z)
        Vector2 dronePos2D = new Vector2(dronePosition.x, dronePosition.z);
        Vector2 centerPos2D = new Vector2(centerPoint.x, centerPoint.z);
        Vector2 directionToCenter = (centerPos2D - dronePos2D).normalized;

        // Get current drone heading (where it's currently pointing)
        Vector2 currentHeading = new Vector2(droneChild.transform.forward.x, droneChild.transform.forward.z);

        // Calculate the angle difference between current heading and desired direction
        float desiredYawRateDegrees = Vector2.SignedAngle(currentHeading, directionToCenter);

        // Add dead zone to prevent oscillations
        float yawDeadZone = 5f; // degrees - tune this value
        if (Mathf.Abs(desiredYawRateDegrees) < yawDeadZone)
        {
            // Close enough - don't send yaw commands
            GetComponent<VelocityControl>().attitude_control_yaw = 0.0f;

            // Debug visualization
            DrawYawDebugArrows(droneChild, currentHeading, directionToCenter);
            return;
        }

        // Convert to radians
        float desiredYawRateRadians = desiredYawRateDegrees * Mathf.Deg2Rad;

        // Apply the same transformation as in attitudeControl.cs (lines 100-107)
        if (desiredYawRateRadians > 0)
        {
            desiredYawRateRadians = Mathf.PI - desiredYawRateRadians;
        }
        else
        {
            desiredYawRateRadians = -Mathf.PI - desiredYawRateRadians;
        }

        // Send the yaw command to VelocityControl
        GetComponent<VelocityControl>().attitude_control_yaw = -1 * desiredYawRateRadians;

        // Visualize the drone's current heading and desired direction
        Vector3 dronePos3D = droneChild.transform.position;

        // Debug visualization
        DrawYawDebugArrows(droneChild, currentHeading, directionToCenter);
    }


    // Modified camera control - only update when needed
    void ControlCameraPitch(GameObject drone)
    {
        // Find the FPV camera
        Camera camera = drone.transform.Find("FPV")?.GetComponent<Camera>();
        
        if (camera != null)
        {
            // Get the FPVCameraScript component
            FPVCameraScript fpvScript = camera.GetComponent<FPVCameraScript>();
            
            if (fpvScript != null)
            {
                // Set our desired pitch in the camera script
                fpvScript.SetAdditionalPitch(cameraPitch);

                // Debug.Log($"Set additional pitch to: {cameraPitch} degrees on {drone.name}");
            }
            else
            {
                Debug.LogWarning($"FPVCameraScript not found on camera of {drone.name}");
            }
            
            DrawCameraPitchArrow(camera);
        }
    }


    

    // Debug visualization for camera pitch direction
    void DrawCameraPitchArrow(Camera camera)
    {
        if (debug_bool == true)
        {
            Vector3 cameraPosition = camera.transform.position;
            Vector3 cameraForward = camera.transform.forward;

            // Draw camera's forward direction (BLUE arrow for pitch)
            Debug.DrawRay(cameraPosition, cameraForward * 5.0f, Color.blue, 0.1f);

            // Draw a shorter arrow showing just the pitch component
            Vector3 pitchDirection = new Vector3(0, cameraForward.y, Vector3.Project(cameraForward, Vector3.forward).z).normalized;
            Debug.DrawRay(cameraPosition, pitchDirection * 1.5f, Color.cyan, 0.1f);
        }
    }
     // Debug visualization function - easy to comment out the entire function
    void DrawYawDebugArrows(GameObject droneChild, Vector2 currentHeading, Vector2 directionToCenter)
    {
        if (debug_bool == true)
        {
            Vector3 dronePos3D = droneChild.transform.position;

            // Draw current heading (RED arrow)
            Vector3 currentHeading3D = new Vector3(currentHeading.x, 0, currentHeading.y);
            Debug.DrawRay(dronePos3D, currentHeading3D * 5.0f, Color.red, 0.1f);

            // Draw desired direction to center (GREEN arrow)
            Vector3 directionToCenter3D = new Vector3(directionToCenter.x, 0, directionToCenter.y);
            Debug.DrawRay(dronePos3D, directionToCenter3D * 3.0f, Color.green, 0.1f);
        }
    }
}