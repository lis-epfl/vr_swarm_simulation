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

    [DllImport("kernel32.dll", SetLastError = true)]
    private static extern IntPtr MapViewOfFile(IntPtr hFileMappingObject, uint dwDesiredAccess, uint dwFileOffsetHigh, uint dwFileOffsetLow, uint dwNumberOfBytesToMap);

    [DllImport("kernel32.dll", SetLastError = true)]
    private static extern bool UnmapViewOfFile(IntPtr lpBaseAddress);

    [DllImport("kernel32.dll", SetLastError = true)]
    private static extern bool CloseHandle(IntPtr hObject);

    private const uint FILE_MAP_ALL_ACCESS = 0xF001F;

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
    [SerializeField] private bool logVisionCommands = false; // Debug logging
    [SerializeField] private bool usePerDroneCommands = true; // NEW: Use individual drone commands instead of shared

    // Static shared vision command (legacy mode - all drones use the same command)
    private static Vector3 sharedVisionCommand = Vector3.zero;
    
    // Per-drone vision commands (new mode - each drone has its own command)
    private static Vector3[] perDroneVisionCommands = new Vector3[10]; // Max 10 drones
    private static bool perDroneCommandsUpdated = false;
    
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

    // Add NBVMode enum for clarity
    private enum NBVMode { Hold, Formation, NBVCommand }
    private NBVMode currentMode = NBVMode.Hold;
    private int lastCommandFlag = -1;
    private float vicinityThreshold = 1.0f; // meters

    void Start()
    {
        // if (logVisionCommands)
        //     Debug.Log($"NBV.cs: Starting on GameObject '{gameObject.name}' with vision system {(enableVisionSystem ? "ENABLED" : "DISABLED")}");
            
        if (enableVisionSystem)
        {
            // The sharedVisionManager becomes the vision coordinator
            string objectName = gameObject.name;
            if (objectName.Contains("sharedVisionManager") || objectName.Contains("0"))
            {
                isVisionCoordinator = true;
                InitializeVisionSystem();
                // if (logVisionCommands)
                //     Debug.Log($"NBV.cs: '{gameObject.name}' designated as vision coordinator");
            }
            // else if (logVisionCommands)
            // {
            //     Debug.Log($"NBV.cs: '{gameObject.name}' will receive shared vision commands");
            // }
        }

        // Set initial mode and flag to formation (9)
        currentMode = NBVMode.Formation;
        lastCommandFlag = 9;
        // If vision system is enabled and initialized, set flag in shared memory
        if (enableVisionSystem && isVisionCoordinator && visionSystemInitialized && commandPtr != IntPtr.Zero)
        {
            Marshal.WriteInt32(commandPtr, 0, 9);
            Debug.Log("[NBV] Set initial shared memory flag to 9 (Formation mode)");
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
        if (enableVisionSystem && isVisionCoordinator && Time.time - lastVisionCheckTime >= 0.1f)
        {
            UpdateVisionCommand();
            lastVisionCheckTime = Time.time;
        }

        // Check command flag from shared memory (if vision system is enabled and initialized)
        int commandFlag = 0;
        if (visionSystemInitialized && commandPtr != IntPtr.Zero)
        {
            commandFlag = Marshal.ReadInt32(commandPtr, 0);
        }

        // Switch mode based on flag
        if (commandFlag != lastCommandFlag)
        {
            lastCommandFlag = commandFlag;
            if (commandFlag == 9)
                currentMode = NBVMode.Formation;
            else if (commandFlag == 3)
                currentMode = NBVMode.NBVCommand;
            else if (commandFlag == 0)
                currentMode = NBVMode.Hold;
            // 1 and 2 are busy/data ready, don't change mode
        }

        // Debug: print current mode and flag
        Debug.Log($"[NBV] Frame: Mode={currentMode}, SharedMemoryFlag={lastCommandFlag}");

        // Ensure there are drones in the swarm to avoid division by zero
        if (swarm == null || swarm.Count == 0)
        {
            return;
        }

        // Get drone index
        string droneName = gameObject.transform.parent.name;
        string[] splitName = droneName.Split(' ');
        int i = int.Parse(splitName[1]);

        Vector3 targetPosition = Vector3.zero;
        float adjustedHeight = height;

        if (currentMode == NBVMode.Formation)
        {
            // Formation logic
            float angleStep = 30.0f;
            float buffer_angle = 45.0f;
            float angle = (i * angleStep + buffer_angle) * Mathf.Deg2Rad;
            float x = centerPoint.x + radius * Mathf.Cos(angle);
            float z = centerPoint.z + radius * Mathf.Sin(angle);
            targetPosition = new Vector3(x, height, z);
        }
        else if (currentMode == NBVMode.NBVCommand)
        {
            // Use NBV command as target
            if (usePerDroneCommands && i < perDroneVisionCommands.Length)
            {
                targetPosition = perDroneVisionCommands[i];
                adjustedHeight = targetPosition.y;
            }
            else
            {
                targetPosition = sharedVisionCommand;
                adjustedHeight = targetPosition.y;
            }
        }
        else if (currentMode == NBVMode.Hold)
        {
            // Hold current position
            targetPosition = transform.position;
            adjustedHeight = transform.position.y;
        }

        // Set desired height for VelocityControl
        GetComponent<VelocityControl>().desired_height = adjustedHeight;

        // Get the current drone GameObject and find the DroneParent child
        GameObject drone = swarm[i];
        GameObject droneChild = drone.transform.Find("DroneParent").gameObject;
        float maxSpeed = droneChild.GetComponent<VelocityControl>().GetMaxSpeed();

        // Proportional control
        Vector3 positionError = targetPosition - droneChild.transform.position;
        Vector3 velocityCommand = positionError * proportionalGain;

        // Obstacle/inter-drone avoidance (unchanged)
        Vector3 avoidanceVector = Vector3.zero;
        Vector3 interDroneAvoidanceVector = Vector3.zero;
        bool isAvoiding = false;
        if (enableObstacleAvoidance)
        {
            avoidanceVector = CalculateObstacleAvoidance(droneChild);
            isAvoiding = avoidanceVector.magnitude > 0.1f;
        }
        if (enableInterDroneAvoidance)
        {
            interDroneAvoidanceVector = CalculateInterDroneAvoidance(droneChild);
            if (interDroneAvoidanceVector.magnitude > 0.1f)
            {
                isAvoiding = true;
            }
        }
        if (useFormationOverride && isAvoiding)
        {
            float reductionFactor = 0.3f;
            velocityCommand *= reductionFactor;
        }
        velocityCommand += avoidanceVector + interDroneAvoidanceVector;
        if (velocityCommand.magnitude > maxSpeed)
        {
            velocityCommand = velocityCommand.normalized * maxSpeed;
        }
        GetComponent<VelocityControl>().swarm_vx = velocityCommand.x;
        GetComponent<VelocityControl>().swarm_vz = velocityCommand.z;
        // desired_height is already set above
        CalculateYawTowardCenter(droneChild);
        if (enableCameraPitchControl)
        {
            ControlCameraPitch(drone);
        }

        // Vicinity check for mode transitions
        if (currentMode == NBVMode.Formation || currentMode == NBVMode.NBVCommand)
        {
            bool allInVicinity = true;
            for (int idx = 0; idx < swarm.Count; idx++)
            {
                GameObject d = swarm[idx];
                GameObject dChild = d.transform.Find("DroneParent").gameObject;
                Vector3 target = currentMode == NBVMode.Formation ?
                    new Vector3(centerPoint.x + radius * Mathf.Cos((idx * 30.0f + 45.0f) * Mathf.Deg2Rad), height, centerPoint.z + radius * Mathf.Sin((idx * 30.0f + 45.0f) * Mathf.Deg2Rad)) :
                    (usePerDroneCommands && idx < perDroneVisionCommands.Length ? perDroneVisionCommands[idx] : sharedVisionCommand);
                float dist = Vector3.Distance(dChild.transform.position, target);
                if (dist > vicinityThreshold)
                {
                    allInVicinity = false;
                    break;
                }
            }
            // If all drones are in vicinity, switch flag to 0 (hold)
            if (allInVicinity && visionSystemInitialized && commandPtr != IntPtr.Zero)
            {
                Marshal.WriteInt32(commandPtr, 0, 0);
                currentMode = NBVMode.Hold;
            }
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

    // ==================== VISION SYSTEM METHODS ====================

    private void InitializeVisionSystem()
    {
        try
        {
            // Open the shared memory for commands
            commandFileMap = OpenFileMapping(FILE_MAP_ALL_ACCESS, false, commandMemoryName);
            if (commandFileMap == IntPtr.Zero)
            {
                // if (logVisionCommands)
                //     Debug.LogWarning($"NBV: Could not open command shared memory '{commandMemoryName}'. Vision system disabled.");
                enableVisionSystem = false;
                return;
            }

            // Map the command memory - calculate size based on command mode
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
            
            commandPtr = MapViewOfFile(commandFileMap, FILE_MAP_ALL_ACCESS, 0, 0, commandMemorySize);
            if (commandPtr == IntPtr.Zero)
            {
                // if (logVisionCommands)
                //     Debug.LogWarning("NBV: Could not map command shared memory. Vision system disabled.");
                CloseHandle(commandFileMap);
                enableVisionSystem = false;
                return;
            }

            visionSystemInitialized = true;
            Debug.Log($"[NBV] Vision system initialized: isVisionCoordinator={isVisionCoordinator}, commandPtr={(commandPtr != IntPtr.Zero)}");
            if (enableVisionSystem && isVisionCoordinator && commandPtr != IntPtr.Zero)
            {
                Marshal.WriteInt32(commandPtr, 0, 9);
                Debug.Log("[NBV] Set initial shared memory flag to 9 (Formation mode) in InitializeVisionSystem");
            }
        }
        catch (System.Exception e)
        {
            if (logVisionCommands)
                Debug.LogError($"NBV: Failed to initialize vision system: {e.Message}");
            enableVisionSystem = false;
        }
    }

    private void UpdateVisionCommand()
    {
        if (!visionSystemInitialized || commandPtr == IntPtr.Zero)
        {
            // if (logVisionCommands)
            //     Debug.LogWarning("NBV Coordinator: Vision system not initialized or commandPtr is null");
            return;
        }

        try
        {
            // Read the command flag (first 4 bytes)
            int commandFlag = Marshal.ReadInt32(commandPtr, 0);
            
            // if (logVisionCommands)
            // {
            //     Debug.Log($"NBV Coordinator: Read flag = {commandFlag}");
            // }
            
            // Only read if there's a new command (flag > 0)
            if (commandFlag > 0)
            {
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
                
                // if (logVisionCommands)
                // {
                //     Debug.Log($"NBV Coordinator: Acknowledged command(s), reset flag to 0 (was {commandFlag})");
                // }
            }
            else if (logVisionCommands)
            {
                Debug.Log("NBV Coordinator: No new command (flag = 0)");
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
        int maxDrones = perDroneVisionCommands.Length;
        
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
            
            // Update this drone's command
            if (Vector3.Distance(newCommand, perDroneVisionCommands[droneId]) > 0.001f)
            {
                perDroneVisionCommands[droneId] = newCommand;
                perDroneCommandsUpdated = true;
                
                // if (logVisionCommands && droneId < 3) // Log first 3 drones
                // {
                //     Debug.Log($"NBV Coordinator: Updated drone {droneId} command: ({newCommand.x:F2}, {newCommand.y:F2}, {newCommand.z:F2})");
                // }
            }
        }
        
        // if (logVisionCommands && perDroneCommandsUpdated)
        // {
        //     Debug.Log($"NBV Coordinator: Updated per-drone vision commands for {maxDrones} drones");
        // }
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