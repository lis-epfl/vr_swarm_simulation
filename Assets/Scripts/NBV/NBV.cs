using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System;
using System.Runtime.InteropServices;

public class NBV : MonoBehaviour
{
    // Windows API for shared memory
    [DllImport("kernel32.dll", SetLastError = true)]
    private static extern IntPtr OpenFileMapping(uint dwDesiredAccess, bool bInheritHandle, string lpName);

    [DllImport("kernel32.dll", SetLastError = true, CharSet = CharSet.Auto)]
    private static extern IntPtr CreateFileMapping(IntPtr hFile, IntPtr lpAttributes, uint protect, uint maxSizeHigh, uint maxSizeLow, string name);

    [DllImport("kernel32.dll", SetLastError = true)]
    private static extern IntPtr MapViewOfFile(IntPtr hFileMappingObject, uint dwDesiredAccess, uint dwFileOffsetHigh, uint dwFileOffsetLow, uint dwNumberOfBytesToMap);

    [DllImport("kernel32.dll", SetLastError = true)]
    private static extern bool UnmapViewOfFile(IntPtr lpBaseAddress);

    [DllImport("kernel32.dll", SetLastError = true)]
    private static extern bool CloseHandle(IntPtr hObject);

    private const uint FILE_MAP_ALL_ACCESS = 0xF001F;
    private const uint PAGE_READWRITE = 0x04;

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

    [Header("Vision System Integration")]
    [SerializeField] private bool enableVisionSystem = true; // Enable/disable vision-based adjustments
    [SerializeField] private float visionInfluenceStrength = 0.1f; // How much vision affects positions (reduced for safety)
    [SerializeField] private bool logVisionCommands = true; // Debug logging - ENABLED FOR DEBUGGING
    [SerializeField] private bool usePerDroneCommands = true; // NEW: Use individual drone commands instead of shared
    
    [Header("NBV Mode")]
    [SerializeField] private bool useNBVMode = false; // Toggle: false = vision offsets, true = absolute NBV targets
    [SerializeField] private bool logNBVCommands = true; // Debug logging for NBV commands

    // Static shared vision command (legacy mode - all drones use the same command)
    private static Vector3 sharedVisionCommand = Vector3.zero;
    
    // Per-drone vision commands (new mode - each drone has its own command)
    private static Vector3[] perDroneVisionCommands = new Vector3[10]; // Max 10 drones
    private static bool perDroneCommandsUpdated = false;
    
    // NBV target positions (absolute positions from Python)
    private static Vector3[] nbvTargetPositions = new Vector3[10]; // Max 10 drones
    private static bool hasNBVTargets = false;
    
    private static bool isVisionCoordinator = false; // Only one drone manages shared memory
    
    // Vision system constants
    private const string commandMemoryName = "NBVCommandSharedMemory";

    // Private variables
    private float lastCameraPitch = float.MinValue; // To track changes
    
    // Vision system private variables (for coordinator only)
    private float lastVisionCheckTime = 0f;
    private System.IntPtr commandFileMap;
    private System.IntPtr commandPtr;
    private bool visionSystemInitialized = false;

    void Start()
    {
        if (logVisionCommands || logNBVCommands)
            Debug.Log($"NBV.cs: Starting on GameObject '{gameObject.name}' with vision system {(enableVisionSystem ? "ENABLED" : "DISABLED")}, NBV mode {(useNBVMode ? "ENABLED" : "DISABLED")}");
            
        if (enableVisionSystem)
        {
            // The sharedVisionManager becomes the vision coordinator
            string objectName = gameObject.name;
            if (objectName.Contains("sharedVisionManager") || objectName.Contains("0") || objectName.Contains("DroneParent"))
            {
                isVisionCoordinator = true;
                InitializeVisionSystem();
                if (logVisionCommands || logNBVCommands)
                    Debug.Log($"NBV.cs: '{gameObject.name}' designated as vision coordinator");
            }
            else if (logVisionCommands || logNBVCommands)
            {
                Debug.Log($"NBV.cs: '{gameObject.name}' will receive shared vision commands");
            }
        }
    }

    void OnDestroy()
    {
        if (isVisionCoordinator)
        {
            CleanupVisionSystem();
        }
    }

    void FixedUpdate()
    {
        // Only the vision coordinator reads from shared memory
        // Check frequently (every 0.1 seconds) for immediate response to Python commands
        if (enableVisionSystem && isVisionCoordinator && Time.time - lastVisionCheckTime >= 0.1f)
        {
            if (logVisionCommands || logNBVCommands)
            {
                Debug.Log($"NBV Coordinator ({gameObject.name}): Checking for commands... (NBV mode: {useNBVMode})");
            }
            UpdateVisionCommand();
            lastVisionCheckTime = Time.time;
        }

        // Ensure there are drones in the swarm to avoid division by zero
        if (swarm == null || swarm.Count == 0)
        {
            return;
        }

        // Calculate the angle between each drone
        // 360 degrees in a circle, divided by the number of drones
        // float angleStep = 360.0f / swarm.Count;
        float angleStep = 30.0f;
        

        // Get the name of the parent of this gameObject
        string droneName = gameObject.transform.parent.name;
        string[] splitName = droneName.Split(' ');
        int i = int.Parse(splitName[1]);

        // Calculate the angle for this specific drone
        // We multiply by Mathf.Deg2Rad to convert degrees to radians for the trig functions
        // add +45 degrees buffer so that drone 0 starts at 45 degrees (northeast) instead of 0 degrees (east)
        // Add buffer_angle as variable
        float buffer_angle = 45.0f;
        float angle = (i * angleStep + buffer_angle) * Mathf.Deg2Rad;

        // Calculate the target position using trigonometry
        // x = center.x + radius * cos(angle)
        // z = center.z + radius * sin(angle)
        float x = centerPoint.x + radius * Mathf.Cos(angle);
        float z = centerPoint.z + radius * Mathf.Sin(angle);

        // Create the final target position vector
        Vector3 targetPosition = new Vector3(x, height, z);
        
        // Apply NBV mode or vision system offset if enabled
        if (enableVisionSystem && useNBVMode && hasNBVTargets && i < nbvTargetPositions.Length)
        {
            // NBV MODE: Use absolute target position from Python
            Vector3 nbvTarget = nbvTargetPositions[i];
            
            // Only use NBV target if it's not zero (valid target)
            if (nbvTarget.magnitude > 0.001f)
            {
                targetPosition = nbvTarget;
                
                if (logNBVCommands && i < 3) // Log for first 3 drones
                {
                    Debug.Log($"NBV Drone {i}: Using absolute NBV target = {nbvTarget}");
                }
                
                // Set desired height from NBV target
                GetComponent<VelocityControl>().desired_height = nbvTarget.y;
            }
            else
            {
                // No valid NBV target, use base height
                GetComponent<VelocityControl>().desired_height = height;
            }
        }
        else if (enableVisionSystem && !useNBVMode)
        {
            // VISION OFFSET MODE: Apply vision-based adjustments
            Vector3 visionCommand = Vector3.zero;
            Vector3 visionOffset = Vector3.zero;
            
            if (usePerDroneCommands && i < perDroneVisionCommands.Length)
            {
                // NEW: Use individual drone command
                visionCommand = perDroneVisionCommands[i];
                visionOffset = visionCommand * visionInfluenceStrength;
                
                if (logVisionCommands && visionCommand.magnitude > 0.001f && i < 3) // Log for first 3 drones
                {
                    Debug.Log($"NBV Drone {i}: Individual vision command = {visionCommand}, offset = {visionOffset}");
                }
            }
            else
            {
                // LEGACY: Use shared command with multiplier
                int droneMultiplier = i + 1; // +1 to avoid multiplying by 0 for drone 0
                Vector3 individualVisionCommand = sharedVisionCommand * droneMultiplier;
                visionOffset = individualVisionCommand * visionInfluenceStrength;
                
                if (logVisionCommands && sharedVisionCommand.magnitude > 0.001f && i < 3) // Log for first 3 drones
                {
                    Debug.Log($"NBV Drone {i}: Shared vision command {sharedVisionCommand}, multiplied by {droneMultiplier} = {individualVisionCommand}, offset = {visionOffset}");
                }
            }
            
            // Apply X and Z offsets to target position (for horizontal movement)
            targetPosition.x += visionOffset.x;
            targetPosition.z += visionOffset.z;
            
            // Apply Y offset to height (for vertical movement) - TEMP OFFSET, NOT CUMULATIVE!
            // Use a temporary variable so we don't accumulate the offset every frame
            float adjustedHeight = height + visionOffset.y;
            
            if (logVisionCommands && visionOffset.magnitude > 0.001f && i < 3) // Log for first 3 drones
            {
                Debug.Log($"NBV Drone {i}: Target position = {targetPosition}, base height = {height}, adjusted height = {adjustedHeight}");
            }
            
            // Set the desired height with vision offset for VelocityControl to use
            GetComponent<VelocityControl>().desired_height = adjustedHeight;
        }
        else
        {
            // No vision system or NBV mode - use base height
            GetComponent<VelocityControl>().desired_height = height;
        }
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

        // NOTE: desired_height is now set in the vision system section above

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
        // NEW: Orient toward closest object point instead of center
        CalculateYawTowardClosestObject(droneChild);
    }
    
    void CalculateYawTowardClosestObject(GameObject droneChild)
    {
        // Get current drone position
        Vector3 dronePosition = droneChild.transform.position;
        
        // Try to find the closest point on any obstacle/building
        Vector3 targetPoint = FindClosestObjectPoint(dronePosition);
        
        // If no object found, fall back to center point
        if (targetPoint == Vector3.zero)
        {
            targetPoint = centerPoint;
        }
        
        // Calculate direction from drone to target point (only X and Z for yaw)
        Vector2 dronePos2D = new Vector2(dronePosition.x, dronePosition.z);
        Vector2 targetPos2D = new Vector2(targetPoint.x, targetPoint.z);
        Vector2 directionToTarget = (targetPos2D - dronePos2D).normalized;
        
        // Get current drone heading (where it's currently pointing)
        Vector2 currentHeading = new Vector2(droneChild.transform.forward.x, droneChild.transform.forward.z);
        
        // Calculate the angle difference between current heading and desired direction
        float desiredYawRateDegrees = Vector2.SignedAngle(currentHeading, directionToTarget);
        
        // Add dead zone to prevent oscillations
        float yawDeadZone = 5f; // degrees - tune this value
        if (Mathf.Abs(desiredYawRateDegrees) < yawDeadZone)
        {
            // Close enough - don't send yaw commands
            GetComponent<VelocityControl>().attitude_control_yaw = 0.0f;
            
            // Debug visualization
            if (debug_bool)
            {
                NBVDebugger.DrawYawDebugArrows(droneChild, currentHeading, directionToTarget);
                // Draw line to target point
                Debug.DrawLine(dronePosition, targetPoint, Color.cyan, 0.1f);
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
        
        // Debug visualization
        if (debug_bool)
        {
            NBVDebugger.DrawYawDebugArrows(droneChild, currentHeading, directionToTarget);
            // Draw line to target point
            Debug.DrawLine(dronePosition, targetPoint, Color.yellow, 0.1f);
        }
    }
    
    Vector3 FindClosestObjectPoint(Vector3 dronePosition)
    {
        // Strategy: Use sphere overlap to find nearby obstacles, then get closest point
        float searchRadius = 50f; // Search within 50m radius
        
        // Find all obstacles within search radius
        Collider[] obstacles = Physics.OverlapSphere(dronePosition, searchRadius, obstacleLayerMask);
        
        if (obstacles.Length == 0)
        {
            // No obstacles found, return zero (will fall back to center)
            return Vector3.zero;
        }
        
        // Find the closest point among all obstacles
        Vector3 closestPoint = Vector3.zero;
        float closestDistance = float.MaxValue;
        
        foreach (Collider obstacle in obstacles)
        {
            // Get the closest point on this obstacle's surface
            Vector3 pointOnObstacle = obstacle.ClosestPoint(dronePosition);
            float distance = Vector3.Distance(dronePosition, pointOnObstacle);
            
            // Update if this is closer
            if (distance < closestDistance)
            {
                closestDistance = distance;
                closestPoint = pointOnObstacle;
            }
        }
        
        // Debug visualization
        if (debug_bool && closestPoint != Vector3.zero)
        {
            Debug.DrawLine(dronePosition, closestPoint, Color.green, 0.1f);
            Debug.DrawSphere(closestPoint, 0.3f, Color.red);
        }
        
        return closestPoint;
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

    // ==================== VISION SYSTEM METHODS ====================

    private void InitializeVisionSystem()
    {
        try
        {
            // CREATE the shared memory for commands (instead of opening)
            // This way Unity creates it first, and Python can open it later
            
            // Calculate size based on command mode
            uint commandMemorySize;
            if (usePerDroneCommands)
            {
                // Per-drone commands: flag(4) + (10 drones × 12 bytes per command)
                commandMemorySize = 4 + (10 * 12); // 124 bytes total
            }
            else
            {
                // Single shared command: flag(4) + single command(12)
                commandMemorySize = 16; // 16 bytes total
            }
            
            // CREATE shared memory (not open)
            commandFileMap = CreateFileMapping(new IntPtr(-1), IntPtr.Zero, PAGE_READWRITE, 0, commandMemorySize, commandMemoryName);
            if (commandFileMap == IntPtr.Zero)
            {
                if (logVisionCommands || logNBVCommands)
                    Debug.LogWarning($"NBV: Could not create command shared memory '{commandMemoryName}'. Vision system disabled.");
                enableVisionSystem = false;
                return;
            }
            
            commandPtr = MapViewOfFile(commandFileMap, FILE_MAP_ALL_ACCESS, 0, 0, commandMemorySize);
            if (commandPtr == IntPtr.Zero)
            {
                if (logVisionCommands || logNBVCommands)
                    Debug.LogWarning("NBV: Could not map command shared memory. Vision system disabled.");
                CloseHandle(commandFileMap);
                enableVisionSystem = false;
                return;
            }
            
            // Initialize command flag to 0
            Marshal.WriteInt32(commandPtr, 0, 0);

            visionSystemInitialized = true;
            if (logVisionCommands || logNBVCommands)
                Debug.Log($"NBV: Vision system initialized successfully (memory size: {commandMemorySize} bytes, per-drone: {usePerDroneCommands})");
        }
        catch (System.Exception e)
        {
            if (logVisionCommands || logNBVCommands)
                Debug.LogError($"NBV: Failed to initialize vision system: {e.Message}");
            enableVisionSystem = false;
        }
    }

    private void UpdateVisionCommand()
    {
        if (!visionSystemInitialized || commandPtr == IntPtr.Zero)
        {
            if (logVisionCommands || logNBVCommands)
                Debug.LogWarning("NBV Coordinator: Vision system not initialized or commandPtr is null");
            return;
        }

        try
        {
            // Read the command flag (first 4 bytes)
            int commandFlag = Marshal.ReadInt32(commandPtr, 0);
            
            // DEBUG: Read first 16 bytes to see what's actually in memory
            byte[] debugBytes = new byte[16];
            Marshal.Copy(commandPtr, debugBytes, 0, 16);
            
            if (logVisionCommands || logNBVCommands)
            {
                Debug.Log($"NBV Coordinator: Read flag = {commandFlag}, Memory dump: {BitConverter.ToString(debugBytes)}");
            }
            
            // Only read if there's a new command (flag > 0)
            if (commandFlag > 0)
            {
                Debug.Log($"NBV Coordinator: New command detected! Flag = {commandFlag}, useNBVMode = {useNBVMode}");
                
                if (usePerDroneCommands)
                {
                    // NEW: Read individual commands for each drone
                    ReadPerDroneCommands();
                }
                else
                {
                    // LEGACY: Read single shared command
                    ReadSharedCommand();
                }
                
                // IMPORTANT: Reset flag to 0 to acknowledge we've read the command(s)
                // This creates a handshake with Python to prevent race conditions
                Marshal.WriteInt32(commandPtr, 0, 0);
                
                if (logVisionCommands || logNBVCommands)
                {
                    Debug.Log($"NBV Coordinator: Acknowledged command(s), reset flag to 0 (was {commandFlag})");
                }
            }
        }
        catch (System.Exception e)
        {
            // if (logVisionCommands)
            //     Debug.LogError($"NBV: Error reading vision command: {e.Message}");
        }
    }
    
    private void ReadSharedCommand()
    {
        // Read the position command (3 floats starting at byte 4)
        byte[] commandBytes = new byte[12];
        Marshal.Copy(IntPtr.Add(commandPtr, 4), commandBytes, 0, 12);
        
        float x = BitConverter.ToSingle(commandBytes, 0);
        float y = BitConverter.ToSingle(commandBytes, 4);
        float z = BitConverter.ToSingle(commandBytes, 8);
        
        Vector3 newCommand = new Vector3(x, y, z);
        
        // Update the shared vision command (affects all drones)
        if (Vector3.Distance(newCommand, sharedVisionCommand) > 0.001f)
        {
            sharedVisionCommand = newCommand;
            
            // if (logVisionCommands)
            // {
            //     Debug.Log($"NBV Coordinator: Broadcasting shared vision command to all drones: ({sharedVisionCommand.x:F2}, {sharedVisionCommand.y:F2}, {sharedVisionCommand.z:F2})");
            // }
        }
    }
    
    private void ReadPerDroneCommands()
    {
        int commandSize = 12; // 3 floats (x, y, z)
        int maxDrones = useNBVMode ? nbvTargetPositions.Length : perDroneVisionCommands.Length;
        
        // Read commands for all drones
        for (int droneId = 0; droneId < maxDrones; droneId++)
        {
            // Calculate position for this drone's command
            int commandPosition = 4 + (droneId * commandSize); // Start after flag
            
            // Read the position command (3 floats)
            byte[] commandBytes = new byte[12];
            Marshal.Copy(IntPtr.Add(commandPtr, commandPosition), commandBytes, 0, 12);
            
            float x = BitConverter.ToSingle(commandBytes, 0);
            float y = BitConverter.ToSingle(commandBytes, 4);
            float z = BitConverter.ToSingle(commandBytes, 8);
            
            Vector3 newCommand = new Vector3(x, y, z);
            
            if (useNBVMode)
            {
                // NBV MODE: Store as absolute target position
                if (Vector3.Distance(newCommand, nbvTargetPositions[droneId]) > 0.001f)
                {
                    nbvTargetPositions[droneId] = newCommand;
                    hasNBVTargets = true;
                    
                    if (logNBVCommands && droneId < 3) // Log first 3 drones
                    {
                        Debug.Log($"NBV Coordinator: Drone {droneId} NBV target updated: ({newCommand.x:F2}, {newCommand.y:F2}, {newCommand.z:F2})");
                    }
                }
            }
            else
            {
                // VISION OFFSET MODE: Store as vision offset (existing behavior)
                if (Vector3.Distance(newCommand, perDroneVisionCommands[droneId]) > 0.001f)
                {
                    perDroneVisionCommands[droneId] = newCommand;
                    perDroneCommandsUpdated = true;
                    
                    if (logVisionCommands && droneId < 3) // Log first 3 drones
                    {
                        Debug.Log($"NBV Coordinator: Updated drone {droneId} vision offset: ({newCommand.x:F2}, {newCommand.y:F2}, {newCommand.z:F2})");
                    }
                }
            }
        }
        
        if (useNBVMode && hasNBVTargets && logNBVCommands)
        {
            Debug.Log($"NBV Coordinator: Updated NBV targets for {maxDrones} drones");
        }
        else if (!useNBVMode && perDroneCommandsUpdated && logVisionCommands)
        {
            Debug.Log($"NBV Coordinator: Updated per-drone vision commands for {maxDrones} drones");
        }
    }

    private void CleanupVisionSystem()
    {
        try
        {
            if (commandPtr != IntPtr.Zero)
            {
                UnmapViewOfFile(commandPtr);
                commandPtr = IntPtr.Zero;
            }

            if (commandFileMap != IntPtr.Zero)
            {
                CloseHandle(commandFileMap);
                commandFileMap = IntPtr.Zero;
            }

            // if (logVisionCommands)
            //     Debug.Log("NBV: Vision system cleaned up");
        }
        catch (System.Exception e)
        {
            Debug.LogError($"NBV: Error during vision system cleanup: {e.Message}");
        }
    }

}