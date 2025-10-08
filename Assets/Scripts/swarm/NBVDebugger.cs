using UnityEngine;

/// <summary>
/// Debug utility class for NBV (Next Best View) swarm visualization.
/// Handles all debug drawing, logging, and visualization for obstacle avoidance,
/// inter-drone avoidance, yaw control, and camera pitch debugging.
/// </summary>
public static class NBVDebugger
{
    #region Obstacle Avoidance Debug

    /// <summary>
    /// Visualizes obstacle avoidance calculations including detection sphere, 
    /// closest points, and avoidance force vectors.
    /// </summary>
    public static void DrawObstacleAvoidanceDebug(Vector3 dronePos, Vector3 closestPoint, 
        Vector3 avoidanceDirection, Collider obstacle, float avoidanceDistance)
    {
        // Draw line from drone to closest point on obstacle (MAGENTA)
        Debug.DrawLine(dronePos, closestPoint, Color.magenta, 0.1f);
        
        // Draw avoidance force direction (YELLOW)
        if (avoidanceDirection.magnitude > 0.1f)
        {
            Debug.DrawRay(dronePos, avoidanceDirection.normalized * 2.0f, Color.yellow, 0.1f);
        }
        
        // Draw detection sphere around drone (YELLOW wireframe)
        DrawWireSphere(dronePos, avoidanceDistance, Color.yellow);
        
        // Draw small sphere at closest point (RED)
        DrawWireSphere(closestPoint, 0.2f, Color.red);
    }

    /// <summary>
    /// Logs detailed obstacle avoidance information for debugging.
    /// </summary>
    public static void LogObstacleAvoidance(string droneName, int obstacleCount, float avoidanceDistance, 
        float distanceToObstacle, float forceMultiplier, Vector3 avoidanceForceVector)
    {
        /// Debug.Log($"Drone {droneName} found {obstacleCount} obstacles within {avoidanceDistance} units");
        /// Debug.Log($"Distance: {distanceToObstacle:F2}, Force: {forceMultiplier:F2}, Direction: {avoidanceForceVector}");
    }

    #endregion

    #region Inter-Drone Avoidance Debug

    /// <summary>
    /// Visualizes inter-drone avoidance including connection lines and avoidance vectors.
    /// </summary>
    public static void DrawInterDroneAvoidanceDebug(Vector3 dronePos, Vector3 otherDronePos, 
        Vector3 avoidanceDirection, float minInterDroneDistance)
    {
        // Draw line between drones (CYAN)
        Debug.DrawLine(dronePos, otherDronePos, Color.cyan, 0.1f);
        
        // Draw avoidance force direction (PURPLE/MAGENTA)
        if (avoidanceDirection.magnitude > 0.1f)
        {
            Debug.DrawRay(dronePos, avoidanceDirection.normalized * 1.5f, new Color(1f, 0f, 1f), 0.1f);
        }
        
        // Draw minimum distance sphere around drone (LIGHT BLUE)
        DrawWireSphere(dronePos, minInterDroneDistance, Color.cyan);
    }

    /// <summary>
    /// Logs inter-drone avoidance information.
    /// </summary>
    public static void LogInterDroneAvoidance(string droneName, int nearbyCount, 
        float distanceToOtherDrone, float forceMultiplier)
    {
        /// Debug.Log($"Drone {droneName} avoiding {nearbyCount} nearby drones");
        /// Debug.Log($"Inter-drone avoidance: Distance: {distanceToOtherDrone:F2}, Force: {forceMultiplier:F2}");
    }

    #endregion

    #region Formation Override Debug

    /// <summary>
    /// Logs formation override status when avoiding obstacles or other drones.
    /// </summary>
    public static void LogFormationOverride(string droneName, float obstacleAvoidanceMagnitude, 
        float interDroneAvoidanceMagnitude)
    {
        /// Debug.Log($"Drone {droneName}: Formation override active, obstacle avoidance: {obstacleAvoidanceMagnitude:F2}, inter-drone avoidance: {interDroneAvoidanceMagnitude:F2}");
    }

    #endregion

    #region Yaw Control Debug

    /// <summary>
    /// Visualizes drone yaw control with current heading and desired direction arrows.
    /// </summary>
    public static void DrawYawDebugArrows(GameObject droneChild, Vector2 currentHeading, Vector2 directionToCenter)
    {
        Vector3 dronePos3D = droneChild.transform.position;

        // Draw current heading (RED arrow)
        Vector3 currentHeading3D = new Vector3(currentHeading.x, 0, currentHeading.y);
        Debug.DrawRay(dronePos3D, currentHeading3D * 5.0f, Color.red, 0.1f);

        // Draw desired direction to center (GREEN arrow)
        Vector3 directionToCenter3D = new Vector3(directionToCenter.x, 0, directionToCenter.y);
        Debug.DrawRay(dronePos3D, directionToCenter3D * 3.0f, Color.green, 0.1f);
    }

    #endregion

    #region Camera Pitch Debug

    /// <summary>
    /// Visualizes camera pitch direction with forward direction and pitch component arrows.
    /// </summary>
    public static void DrawCameraPitchArrow(Camera camera)
    {
        Vector3 cameraPosition = camera.transform.position;
        Vector3 cameraForward = camera.transform.forward;

        // Draw camera's forward direction (BLUE arrow for pitch)
        Debug.DrawRay(cameraPosition, cameraForward * 5.0f, Color.blue, 0.1f);

        // Draw a shorter arrow showing just the pitch component
        Vector3 pitchDirection = new Vector3(0, cameraForward.y, Vector3.Project(cameraForward, Vector3.forward).z).normalized;
        Debug.DrawRay(cameraPosition, pitchDirection * 1.5f, Color.cyan, 0.1f);
    }

    #endregion

    #region Utility Methods

    /// <summary>
    /// Draws a wireframe sphere using Debug.DrawLine for visualization.
    /// Creates circles on XZ and XY planes to approximate a sphere.
    /// </summary>
    public static void DrawWireSphere(Vector3 center, float radius, Color color)
    {
        int segments = 16;
        float angleStep = 360f / segments;
        
        for (int i = 0; i < segments; i++)
        {
            float angle1 = i * angleStep * Mathf.Deg2Rad;
            float angle2 = (i + 1) * angleStep * Mathf.Deg2Rad;
            
            // Draw circle on XZ plane (horizontal)
            Vector3 point1 = center + new Vector3(Mathf.Cos(angle1) * radius, 0, Mathf.Sin(angle1) * radius);
            Vector3 point2 = center + new Vector3(Mathf.Cos(angle2) * radius, 0, Mathf.Sin(angle2) * radius);
            Debug.DrawLine(point1, point2, color, 0.1f);
            
            // Draw circle on XY plane (vertical)
            point1 = center + new Vector3(Mathf.Cos(angle1) * radius, Mathf.Sin(angle1) * radius, 0);
            point2 = center + new Vector3(Mathf.Cos(angle2) * radius, Mathf.Sin(angle2) * radius, 0);
            Debug.DrawLine(point1, point2, color, 0.1f);
        }
    }

    /// <summary>
    /// Checks if debug mode is enabled for the given NBV component.
    /// </summary>
    public static bool IsDebugEnabled(NBV nbv)
    {
        return nbv != null && nbv.debug_bool;
    }

    #endregion

    #region Debug Color Constants

    // Color constants for consistent debugging
    public static readonly Color ObstacleDetectionColor = Color.yellow;
    public static readonly Color ObstacleAvoidanceColor = Color.yellow;
    public static readonly Color ObstacleClosePointColor = Color.red;
    public static readonly Color ObstacleConnectionColor = Color.magenta;
    
    public static readonly Color InterDroneConnectionColor = Color.cyan;
    public static readonly Color InterDroneAvoidanceColor = new Color(1f, 0f, 1f); // Purple/Magenta
    public static readonly Color InterDroneDetectionColor = Color.cyan;
    
    public static readonly Color CurrentHeadingColor = Color.red;
    public static readonly Color DesiredDirectionColor = Color.green;
    
    public static readonly Color CameraForwardColor = Color.blue;
    public static readonly Color CameraPitchColor = Color.cyan;

    #endregion
}