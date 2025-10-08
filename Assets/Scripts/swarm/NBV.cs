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

    // // Your NBV-specific parameters can still be used for other logic
    public float viewDistance;
    public float informationGainWeight;

    [Header("Control Parameters")]
    public float proportionalGain = 1.0f; // Tune this value

    [Header("Obstacle Avoidance")]
    public LayerMask obstacleLayerMask = (1 << 10); // Set to layer 10 (Obstacle layer)
    public float avoidanceDistance = 3.0f; // How far to stay away from obstacles
    public float avoidanceForce = 2.0f; // How strong the avoidance force is
    public bool enableObstacleAvoidance = true; // Toggle on/off
    
    [Header("Advanced Avoidance Tuning")]
    public float escapeForceMultiplier = 5.0f; // Extra force when very close to obstacle
    public float minForceRatio = 0.5f; // Minimum force as ratio of avoidanceForce (prevents getting stuck)
    public bool useFormationOverride = true; // Temporarily reduce formation force when avoiding obstacles


    [Header("Inter-Drone Avoidance")]
    public float minInterDroneDistance = 2.0f; // Minimum distance between drones
    public float interDroneAvoidanceForce = 3.0f; // How strong the inter-drone avoidance is
    public bool enableInterDroneAvoidance = true; // Toggle on/off
    public LayerMask droneLayerMask = (1 << 9); // Layer for drones (you mentioned drones are on layer 9)

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
        // Get the max speed from the VelocityControl script
        float maxSpeed = droneChild.GetComponent<VelocityControl>().GetMaxSpeed();

        // Proportional control
        Vector3 positionError = targetPosition - droneChild.transform.position;
        Vector3 velocityCommand = positionError * proportionalGain;

        // NEW: Add obstacle avoidance
        Vector3 avoidanceVector = Vector3.zero;
        Vector3 interDroneAvoidanceVector = Vector3.zero;
        bool isAvoiding = false;
        
        if (enableObstacleAvoidance)
        {
            avoidanceVector = CalculateObstacleAvoidance(droneChild);
            isAvoiding = avoidanceVector.magnitude > 0.1f;
        }
        
        // NEW: Add inter-drone avoidance
        if (enableInterDroneAvoidance)
        {
            interDroneAvoidanceVector = CalculateInterDroneAvoidance(droneChild);
            if (interDroneAvoidanceVector.magnitude > 0.1f)
            {
                isAvoiding = true;
            }
        }
        
        // Formation override: reduce formation pull when avoiding obstacles OR other drones
        if (useFormationOverride && isAvoiding)
        {
            float reductionFactor = 0.3f; // Reduce formation force to 30% when avoiding
            velocityCommand *= reductionFactor;
            
            if (debug_bool)
            {
                NBVDebugger.LogFormationOverride(droneChild.name, avoidanceVector.magnitude, interDroneAvoidanceVector.magnitude);
            }
        }
        
        // Combine all forces
        velocityCommand += avoidanceVector + interDroneAvoidanceVector;

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

    Vector3 CalculateObstacleAvoidance(GameObject droneChild)
    {
        Vector3 avoidance = Vector3.zero;
        Vector3 dronePos = droneChild.transform.position;
        
        // Find all colliders within avoidance distance on the obstacle layer
        Collider[] obstacles = Physics.OverlapSphere(dronePos, avoidanceDistance, obstacleLayerMask);
        
        // Debug logging
        if (debug_bool && obstacles.Length > 0)
        {
            NBVDebugger.LogObstacleAvoidance(droneChild.name, obstacles.Length, avoidanceDistance, 0, 0, Vector3.zero);
        }
        
        foreach (Collider obstacle in obstacles)
        {
            // Get the closest point on the obstacle to the drone
            Vector3 closestPoint = obstacle.ClosestPoint(dronePos);
            
            // Calculate direction away from the closest point
            Vector3 directionAway = dronePos - closestPoint;
            float distanceToObstacle = directionAway.magnitude;
            
            // Only apply avoidance if we're actually close to the obstacle
            if (distanceToObstacle > 0.1f && distanceToObstacle < avoidanceDistance) // TODO! Perhaps change 0.1f to a larger value
            {
                // Normalize direction
                Vector3 avoidanceForceVector = directionAway.normalized;
                
                // IMPROVED: Multiple force calculation methods
                float forceMultiplier = CalculateAvoidanceForce(distanceToObstacle);
                
                // Add the avoidance force
                Vector3 currentAvoidance = avoidanceForceVector * forceMultiplier;
                avoidance += currentAvoidance;
                
                // Debug visualization
                if (debug_bool)
                {
                    NBVDebugger.DrawObstacleAvoidanceDebug(dronePos, closestPoint, directionAway, obstacle, avoidanceDistance);
                    NBVDebugger.LogObstacleAvoidance(droneChild.name, 1, avoidanceDistance, distanceToObstacle, forceMultiplier, avoidanceForceVector);
                }
            }
        }
        
        return avoidance;
    }

    // NEW: Improved force calculation with multiple strategies
    float CalculateAvoidanceForce(float distanceToObstacle)
    {
        // Method 1: Exponential decay (stronger near obstacle)
        float exponentialForce = avoidanceForce * Mathf.Exp(-2.0f * (distanceToObstacle / avoidanceDistance));
        
        // Method 2: Inverse square law (physics-based)
        float inverseSquareForce = avoidanceForce / (1.0f + distanceToObstacle * distanceToObstacle);
        
        // Method 3: Minimum force threshold (prevents getting stuck)
        float minForceThreshold = avoidanceForce * minForceRatio;
        
        // Method 4: Escape boost (stronger force when very close)
        float escapeBoost = distanceToObstacle < (avoidanceDistance * 0.3f) ? 
            avoidanceForce * escapeForceMultiplier : 0.0f;
        
        // Combine methods: use the strongest force
        float finalForce = Mathf.Max(
            exponentialForce,
            inverseSquareForce + escapeBoost,
            minForceThreshold
        );
        
        return finalForce;
    }

    // NEW: Calculate avoidance from other drones
    Vector3 CalculateInterDroneAvoidance(GameObject droneChild)
    {
        Vector3 avoidance = Vector3.zero;
        Vector3 dronePos = droneChild.transform.position;
        
        // Find all drone colliders within minimum distance
        Collider[] nearbyDrones = Physics.OverlapSphere(dronePos, minInterDroneDistance, droneLayerMask);
        
        int nearbyCount = 0;
        
        foreach (Collider otherDroneCollider in nearbyDrones)
        {
            // Skip if it's the same drone or related to this drone
            if (otherDroneCollider.transform.IsChildOf(droneChild.transform) || 
                droneChild.transform.IsChildOf(otherDroneCollider.transform) ||
                otherDroneCollider.transform == droneChild.transform)
            {
                continue;
            }
            
            // Get the other drone's position
            Vector3 otherDronePos = otherDroneCollider.transform.position;
            
            // Calculate direction away from the other drone
            Vector3 directionAway = dronePos - otherDronePos;
            float distanceToOtherDrone = directionAway.magnitude;
            
            // Only apply avoidance if we're too close
            if (distanceToOtherDrone > 0.1f && distanceToOtherDrone < minInterDroneDistance)
            {
                // Normalize direction
                Vector3 avoidanceForceVector = directionAway.normalized;
                
                // Calculate force - stronger when closer
                float forceMultiplier = CalculateInterDroneAvoidanceForce(distanceToOtherDrone);
                
                // Add the avoidance force
                Vector3 currentAvoidance = avoidanceForceVector * forceMultiplier;
                avoidance += currentAvoidance;
                
                nearbyCount++;
                
                // Debug visualization
                if (debug_bool)
                {
                    NBVDebugger.DrawInterDroneAvoidanceDebug(dronePos, otherDronePos, directionAway, minInterDroneDistance);
                    NBVDebugger.LogInterDroneAvoidance(droneChild.name, 1, distanceToOtherDrone, forceMultiplier);
                }
            }
        }
        
        if (debug_bool && nearbyCount > 0)
        {
            NBVDebugger.LogInterDroneAvoidance(droneChild.name, nearbyCount, 0, 0);
        }
        
        return avoidance;
    }

    // Calculate inter-drone avoidance force
    float CalculateInterDroneAvoidanceForce(float distanceToOtherDrone)
    {
        // Similar to obstacle avoidance but tuned for drone-to-drone
        
        // Method 1: Inverse square law (physics-based)
        float inverseSquareForce = interDroneAvoidanceForce / (1.0f + distanceToOtherDrone * distanceToOtherDrone);
        
        // Method 2: Linear decay with minimum threshold
        float linearForce = interDroneAvoidanceForce * (1.0f - (distanceToOtherDrone / minInterDroneDistance));
        
        // Method 3: Exponential force for very close encounters
        float exponentialForce = interDroneAvoidanceForce * Mathf.Exp(-3.0f * (distanceToOtherDrone / minInterDroneDistance));
        
        // Use the strongest force
        float finalForce = Mathf.Max(inverseSquareForce, linearForce, exponentialForce);
        
        return finalForce;
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
            if (debug_bool)
            {
                NBVDebugger.DrawYawDebugArrows(droneChild, currentHeading, directionToCenter);
            }
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
        if (debug_bool)
        {
            NBVDebugger.DrawYawDebugArrows(droneChild, currentHeading, directionToCenter);
        }
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
            
            if (debug_bool)
            {
                NBVDebugger.DrawCameraPitchArrow(camera);
            }
        }
    }

}