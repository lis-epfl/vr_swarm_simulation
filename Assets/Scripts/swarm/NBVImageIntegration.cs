using UnityEngine;

/// <summary>
/// NBVImageIntegration.cs - Integration layer between NBV.cs and NBVImageCapture.cs
/// 
/// This component shows how to integrate the vision-based position commands
/// into your existing NBV drone control system.
/// 
/// Usage:
/// 1. Attach this to the same GameObject as NBV.cs
/// 2. Make sure NBVImageCapture.cs is active in the scene
/// 3. Run NBVImageProcessor.py to start Python processing
/// 
/// The integration allows the vision system to influence drone movement
/// while preserving the existing NBV formation and obstacle avoidance logic.
/// </summary>
[RequireComponent(typeof(NBV))]
public class NBVImageIntegration : MonoBehaviour
{
    [Header("Vision Integration Settings")]
    [SerializeField] private bool enableVisionControl = false; // DISABLED: NBV.cs now handles vision directly
    [SerializeField] private float visionInfluenceStrength = 0.5f; // How much vision commands affect movement (0-1)
    [SerializeField] private float visionCommandTimeout = 5.0f; // Timeout for vision commands in seconds
    [SerializeField] private bool smoothVisionCommands = true; // Apply smoothing to vision commands
    [SerializeField] private float smoothingSpeed = 2.0f; // Speed of smoothing
    
    [Header("Debug")]
    [SerializeField] private bool showVisionInfluence = true;
    [SerializeField] private bool logVisionCommands = true; // Enable logging to see commands
    
    // Component references
    private SwarmManager swarmManager;
    private NBV nbvScript;
    private NBVImageCapture imageCapture;
    
    // Vision command state
    private Vector3 currentVisionCommand = Vector3.zero;
    private Vector3 smoothedVisionCommand = Vector3.zero;
    private float lastVisionCommandTime = 0f;
    private Vector3 lastReceivedCommand = Vector3.zero;
    
    // Integration state
    private Vector3 originalCenterPoint;
    private bool hasInitialized = false;

    void Start()
    {
        InitializeComponents();
    }

    void Update()
    {
        if (!hasInitialized)
            return;
            
        UpdateVisionCommands();
        ApplyVisionInfluence();
        
        if (showVisionInfluence && enableVisionControl)
        {
            DrawDebugVisualization();
        }
    }

    private void InitializeComponents()
    {
        // Find SwarmManager (this is what controls the actual drone positions)
        swarmManager = FindObjectOfType<SwarmManager>();
        if (swarmManager == null)
        {
            Debug.LogError("NBVImageIntegration: SwarmManager not found in scene!");
            enabled = false;
            return;
        }
        
        // Find NBV script in the scene (for reference/debugging)
        nbvScript = FindObjectOfType<NBV>();
        if (nbvScript == null)
        {
            Debug.LogWarning("NBVImageIntegration: NBV script not found in scene.");
        }
        
        // Find image capture component in scene
        imageCapture = FindObjectOfType<NBVImageCapture>();
        if (imageCapture == null)
        {
            Debug.LogWarning("NBVImageIntegration: NBVImageCapture not found in scene. Vision integration disabled.");
            enableVisionControl = false;
            return;
        }
        
        // Store original center point from SwarmManager
        originalCenterPoint = swarmManager.centerPoint;
        
        hasInitialized = true;
        
        Debug.Log("NBVImageIntegration: Initialized successfully with SwarmManager");
    }

    private void UpdateVisionCommands()
    {
        if (!enableVisionControl || imageCapture == null)
            return;
            
        // Get the latest vision command from image capture
        Vector3 newCommand = imageCapture.GetCurrentPositionCommand();
        
        // Check if we received a new command
        if (Vector3.Distance(newCommand, lastReceivedCommand) > 0.001f)
        {
            lastReceivedCommand = newCommand;
            currentVisionCommand = newCommand;
            lastVisionCommandTime = Time.time;
            
            if (logVisionCommands)
                Debug.Log($"Vision command received: {newCommand}");
        }
        
        // Check for timeout
        if (Time.time - lastVisionCommandTime > visionCommandTimeout)
        {
            currentVisionCommand = Vector3.zero;
        }
        
        // Apply smoothing if enabled
        if (smoothVisionCommands)
        {
            smoothedVisionCommand = Vector3.Lerp(
                smoothedVisionCommand, 
                currentVisionCommand, 
                smoothingSpeed * Time.deltaTime
            );
        }
        else
        {
            smoothedVisionCommand = currentVisionCommand;
        }
    }

    private void ApplyVisionInfluence()
    {
        if (!enableVisionControl || swarmManager == null)
            return;
            
        // Get the command to apply (smoothed or direct)
        Vector3 commandToApply = smoothVisionCommands ? smoothedVisionCommand : currentVisionCommand;
        
        // Apply vision influence to the SwarmManager center point (this controls the actual drones!)
        Vector3 visionOffset = commandToApply * visionInfluenceStrength;
        Vector3 newCenterPoint = originalCenterPoint + visionOffset;
        
        // Update SwarmManager center point (this will affect all drones!)
        swarmManager.centerPoint = newCenterPoint;
        
        if (logVisionCommands && commandToApply.magnitude > 0.001f)
        {
            Debug.Log($"Applying vision influence to SwarmManager: Offset={visionOffset}, NewCenter={newCenterPoint}");
        }
    }

    private void DrawDebugVisualization()
    {
        if (!hasInitialized)
            return;
            
        // Draw original center point (using cross pattern since DrawWireSphere doesn't exist)
        DrawDebugCross(originalCenterPoint, 0.5f, Color.blue);
        
        // Draw current center point (with vision influence)
        DrawDebugCross(nbvScript.centerPoint, 0.5f, Color.green);
        
        // Draw line showing vision influence
        if (smoothedVisionCommand.magnitude > 0.001f)
        {
            Vector3 visionOffset = smoothedVisionCommand * visionInfluenceStrength;
            Debug.DrawLine(originalCenterPoint, originalCenterPoint + visionOffset, Color.red, Time.deltaTime);
        }
        
        // Draw vision command direction
        if (currentVisionCommand.magnitude > 0.001f)
        {
            Vector3 arrowStart = nbvScript.centerPoint + Vector3.up * 2f;
            Vector3 arrowEnd = arrowStart + currentVisionCommand.normalized * 2f;
            
            Debug.DrawLine(arrowStart, arrowEnd, Color.magenta, Time.deltaTime);
            
            // Draw arrow head
            Vector3 right = Vector3.Cross(currentVisionCommand.normalized, Vector3.up).normalized * 0.3f;
            Vector3 back = -currentVisionCommand.normalized * 0.3f;
            Debug.DrawLine(arrowEnd, arrowEnd + back + right, Color.magenta, Time.deltaTime);
            Debug.DrawLine(arrowEnd, arrowEnd + back - right, Color.magenta, Time.deltaTime);
        }
    }
    
    // Helper method to draw a 3D cross pattern instead of a wire sphere
    private void DrawDebugCross(Vector3 center, float size, Color color)
    {
        // Draw X axis
        Debug.DrawLine(center - Vector3.right * size, center + Vector3.right * size, color, Time.deltaTime);
        // Draw Y axis  
        Debug.DrawLine(center - Vector3.up * size, center + Vector3.up * size, color, Time.deltaTime);
        // Draw Z axis
        Debug.DrawLine(center - Vector3.forward * size, center + Vector3.forward * size, color, Time.deltaTime);
    }

    #region Public Interface
    
    /// <summary>
    /// Enable or disable vision control
    /// </summary>
    public void SetVisionControlEnabled(bool enabled)
    {
        enableVisionControl = enabled;
        
        if (!enabled)
        {
            // Reset to original center point
            if (nbvScript != null)
                nbvScript.centerPoint = originalCenterPoint;
        }
        
        Debug.Log($"Vision control {(enabled ? "enabled" : "disabled")}");
    }
    
    /// <summary>
    /// Adjust how much vision commands influence movement
    /// </summary>
    public void SetVisionInfluenceStrength(float strength)
    {
        visionInfluenceStrength = Mathf.Clamp01(strength);
        Debug.Log($"Vision influence strength set to {visionInfluenceStrength}");
    }
    
    /// <summary>
    /// Get the current vision command
    /// </summary>
    public Vector3 GetCurrentVisionCommand()
    {
        return currentVisionCommand;
    }
    
    /// <summary>
    /// Get the smoothed vision command
    /// </summary>
    public Vector3 GetSmoothedVisionCommand()
    {
        return smoothedVisionCommand;
    }
    
    /// <summary>
    /// Check if vision system is active and receiving commands
    /// </summary>
    public bool IsVisionActive()
    {
        return enableVisionControl && 
               imageCapture != null && 
               imageCapture.IsInitialized() &&
               (Time.time - lastVisionCommandTime) < visionCommandTimeout;
    }
    
    /// <summary>
    /// Reset the center point to original position
    /// </summary>
    public void ResetCenterPoint()
    {
        if (nbvScript != null)
        {
            nbvScript.centerPoint = originalCenterPoint;
            currentVisionCommand = Vector3.zero;
            smoothedVisionCommand = Vector3.zero;
        }
    }
    
    #endregion

    #region Unity Events
    
    void OnValidate()
    {
        visionInfluenceStrength = Mathf.Clamp01(visionInfluenceStrength);
        visionCommandTimeout = Mathf.Max(0.1f, visionCommandTimeout);
        smoothingSpeed = Mathf.Max(0.1f, smoothingSpeed);
    }
    
    void OnDrawGizmosSelected()
    {
        if (!hasInitialized)
            return;
            
        // Draw original center point
        Gizmos.color = Color.blue;
        Gizmos.DrawWireSphere(originalCenterPoint, 0.5f);
        
        // Draw current center point
        if (nbvScript != null)
        {
            Gizmos.color = Color.green;
            Gizmos.DrawWireSphere(nbvScript.centerPoint, 0.5f);
        }
        
        // Draw vision influence
        if (enableVisionControl && smoothedVisionCommand.magnitude > 0.001f)
        {
            Gizmos.color = Color.red;
            Vector3 visionOffset = smoothedVisionCommand * visionInfluenceStrength;
            Gizmos.DrawLine(originalCenterPoint, originalCenterPoint + visionOffset);
        }
    }
    
    #endregion
}