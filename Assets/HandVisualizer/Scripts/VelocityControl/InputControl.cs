using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.XR.Hands.Samples.VisualizerSample; // Add this for HandProcessor access

public class InputControl : MonoBehaviour {

    public VelocityControl vc; // Assign a representative drone's VelocityControl component
    public OlfatiSaber olfatiSaber; // Assign a representative drone's OlfatiSaber component

    public SwarmManager.SwarmAlgorithm currentAlgorithm;

    private float abs_height = 1;

    [Header("Calibration Settings")]
    public bool useAutomaticCalibrationPoint = false; // This flag is largely superseded by the new initial auto-calibration
    public Transform playerViewpointTransform; // Assign player's viewpoint transform (e.g., Main Camera or Player GameObject) for automatic calibration

    [Header("Hand Control Settings")]
    public float velocityScale = 5.0f; // User's value
    public float maxVelocity = 10.0f; // Maximum velocity in m/s
    public float yawRateScale = 0.5f; // Sensitivity for yaw control (degrees of hand rotation to degrees/sec of yaw rate)
    public float handDeadZoneRadius = 0.05f; // 5cm radius for the dead zone (10cm diameter)
    
    // Calibration state
    private bool isCalibrated = false;
    private bool isCalibrating = false; // This flag was for the 3-2-1 countdown, can be repurposed or removed if that system is fully gone.
    private Vector3 calibrationCenterPoint = Vector3.zero;
    private float calibrationTimer = 0f; // For 3-2-1 countdown
    private int calibrationCountdown = 3; // For 3-2-1 countdown

    // New flag for the initial 5-second auto-calibration
    private bool initialAutoCalibrationStarted = false;
    private bool isCalibrationCoroutineRunning = false; // To prevent coroutine overlap

    // Y-axis safety
    private float initialCalibratedYPosition;
    private bool isYPositionInitialized = false;
    public float yRecoverySpeed = 0.3f; // Speed to return to initialY if below
    public float yDeadZone = 0.05f;     // Small zone around initialY to consider "at" initialY

    // Palm angle for Yaw - REMOVED CALIBRATION FIELDS
    // private float initialPalmLineAngleWorld; // REMOVED
    // private bool isPalmAngleCalibrated = false; // REMOVED


    // Use this for initialization
    void Start()
    {
        // Auto-assign dependencies if not set in Inspector
        if (vc == null)
        {
            vc = FindObjectOfType<VelocityControl>();
            if (vc == null)
            {
                Debug.LogError("InputControl: VelocityControl (vc) could not be found automatically. Please assign it or ensure one exists in the scene.");
            }
        }
        if (olfatiSaber == null)
        {
            olfatiSaber = FindObjectOfType<OlfatiSaber>();
            if (olfatiSaber == null)
            {
                Debug.LogError("InputControl: OlfatiSaber could not be found automatically. Please assign it or ensure one exists in the scene.");
            }
        }
        if (playerViewpointTransform == null)
        {
            if (Camera.main != null)
            {
                playerViewpointTransform = Camera.main.transform;
            }
            else
            {
                Debug.LogError("InputControl: Main Camera could not be found automatically for playerViewpointTransform. Please assign it or ensure a Camera is tagged 'MainCamera'.");
            }
        }

        // Ensure vc and olfatiSaber are assigned, otherwise log error
        if (vc == null)
        {
            Debug.LogError("VelocityControl (vc) is not assigned in InputControl. Hand control and Y-axis safety might not work correctly.");
        }
        if (olfatiSaber == null)
        {
            Debug.LogError("OlfatiSaber is not assigned in InputControl. Hand control might not work correctly.");
        }
        // Removed: Immediate call to PerformAutomaticCalibration based on useAutomaticCalibrationPoint.
        // The new logic in FixedUpdate will handle initial automatic calibration.
        // if (useAutomaticCalibrationPoint)
        // {
        //     PerformAutomaticCalibration();
        // }
    }
    
    void PerformAutomaticCalibration()
    {
        // This method assumes HandProcessor.ArePalmsTracked was checked by the calling coroutine
        // if it's crucial for this specific call.
        // The primary check is done in DelayedAutomaticCalibration before calling this.

        // Set the calibration center point to the current hand midpoint
        Vector3 leftPalmPos = HandProcessor.LeftPalmPosition;
        Vector3 rightPalmPos = HandProcessor.RightPalmPosition;
        calibrationCenterPoint = (leftPalmPos + rightPalmPos) * 0.5f;
        isCalibrated = true; // Mark as calibrated       
        
        // Initialize Y position safety floor based on a representative drone's current height
        if (vc != null)
        {
            initialCalibratedYPosition = vc.transform.position.y;
            isYPositionInitialized = true;
        }
        else
        {
            Debug.LogError("VelocityControl (vc) not assigned. Cannot set initial Y position for automatic calibration's Y-safety.");
            isYPositionInitialized = false;
        }

        Debug.Log($"Calibration complete! Center point set to: {calibrationCenterPoint}. Initial Y swarm position for safety: {initialCalibratedYPosition:F2}m.");
        Debug.Log("Swarm can now be controlled by hand movement. Yaw control is active if hands are tracked (no calibration needed).");
    }

    // Update is called once per frame
    void FixedUpdate () {

        // New: Initial automatic calibration when hands are seen for the first time
        if (!isCalibrated && !initialAutoCalibrationStarted && HandProcessor.ArePalmsTracked)
        {
            StartCoroutine(DelayedAutomaticCalibration(true)); // Pass true for initial attempt
            initialAutoCalibrationStarted = true; // Mark that the process has been initiated
        }

        // 'C' key for manual re-calibration (uses the same 5-second delay mechanism)
        if (Input.GetKeyDown(KeyCode.C))
        {
            Debug.Log("'C' key pressed. Initiating manual re-calibration.");
            // Reset calibration state before starting manual C-key calibration
            // isCalibrated = false; // Allow re-calibration even if already calibrated
            // initialAutoCalibrationStarted = true; // Prevent initial auto-cal from re-triggering if C is pressed early
            StartCoroutine(DelayedAutomaticCalibration(false)); // Pass false for manual attempt
        }
        
        if (currentAlgorithm == SwarmManager.SwarmAlgorithm.REYNOLDS)
            {
                // Default velocity control for Reynolds
                if (vc != null)
                {
                    vc.desired_vx = 0.0f;
                    vc.desired_vy = 0.0f;
                    vc.desired_yaw = 0.0f;
                }
            }
        else if (currentAlgorithm == SwarmManager.SwarmAlgorithm.OLFATI_SABER)
            {
                if (olfatiSaber == null) return; // Do nothing if olfatiSaber script is not assigned

                // Hand-controlled Olfati-Saber velocity control
                if (isCalibrated && HandProcessor.ArePalmsTracked)
                {
                    UpdateSwarmVelocityFromHands();
                }
                else
                {
                    // Default to zero velocity if not calibrated or hands not tracked
                    olfatiSaber.desired_vx = 0.0f;
                    olfatiSaber.desired_vy = 0.0f; // Corresponds to World Z (via -desired_vy in OlfatiSaber)
                    olfatiSaber.desired_vz = 0.0f; // Corresponds to World Y
                    olfatiSaber.desired_yaw = 0.0f; // No yaw rate
                }
                // The global olfatiSaber.desired_yaw = 0.0f; line that was here previously is removed
                // as yaw is now actively controlled or zeroed within the conditions above.
            }
    }

    // Modified coroutine to handle both initial and manual 5s delayed calibration
    private System.Collections.IEnumerator DelayedAutomaticCalibration(bool isInitialAttempt = false)
    {
        if (isCalibrationCoroutineRunning)
        {
            Debug.Log("Calibration coroutine already running. New request ignored.");
            yield break;
        }
        isCalibrationCoroutineRunning = true;

        if (isInitialAttempt)
        {
            Debug.Log("Initial hands detected. Automatic calibration will occur in 5 seconds. Hold hands at desired center.");
        }
        else
        {
            Debug.Log("Manual calibration: Waiting 5 seconds before calibrating to current hand midpoint...");
        }
        
        yield return new WaitForSeconds(5f);

        if (HandProcessor.ArePalmsTracked)
        {
            PerformAutomaticCalibration(); // This sets isCalibrated = true
            if (isInitialAttempt)
            {
                Debug.Log("Initial automatic calibration successful.");
            }
            else
            {
                Debug.Log("Manual re-calibration successful.");
            }
        }
        else
        {
            Debug.LogWarning($"Calibration after 5s delay failed: Hands not tracked.");
            if (isInitialAttempt)
            {
                Debug.LogWarning("Initial auto-calibration will re-attempt if hands are seen again and not yet calibrated.");
                initialAutoCalibrationStarted = false; // Allow the initial trigger to run again if it failed
            }
            // isCalibrated remains false or its previous state if manual calibration failed.
        }
        isCalibrationCoroutineRunning = false;
    }

    // The old HandleCalibration, CalibrationDelayRoutine, StartCalibration, CompleteCalibration methods
    // for the 3-2-1 countdown are removed as 'C' key now uses DelayedAutomaticCalibration.
    /*
    private void HandleCalibration()
    {
        // ... old code ...
    }

    private System.Collections.IEnumerator CalibrationDelayRoutine()
    {
        // ... old code ...
    }

    private void StartCalibration() 
    {
        // ... old code ...
    }

    private void CompleteCalibration() 
    {
        // ... old code ...
    }
    */

    private void UpdateSwarmVelocityFromHands()
    {
        if (olfatiSaber == null) return;

        if (!HandProcessor.ArePalmsTracked) 
        {
            olfatiSaber.desired_vx = 0.0f;
            olfatiSaber.desired_vy = 0.0f; 
            olfatiSaber.desired_vz = 0.0f;
            olfatiSaber.desired_yaw = 0.0f;
            return;
        }

        // Get current hand midpoint
        Vector3 leftPalmPos = HandProcessor.LeftPalmPosition;
        Vector3 rightPalmPos = HandProcessor.RightPalmPosition;
        Vector3 currentMidpoint = (leftPalmPos + rightPalmPos) * 0.5f;

        // Calculate displacement from calibration center (this is in world coordinates)
        Vector3 displacement = currentMidpoint - calibrationCenterPoint;

        // Check if inside the dead zone for translational movement
        if (displacement.magnitude <= handDeadZoneRadius)
        {
            olfatiSaber.desired_vx = 0.0f;
            olfatiSaber.desired_vy = 0.0f;
            olfatiSaber.desired_vz = 0.0f;
        }
        else
        {
            // Calculate target velocity based on displacement (outside dead zone)
            // We can scale the displacement from the edge of the deadzone, or just use the full displacement
            // For simplicity, using full displacement and then scaling.
            Vector3 targetVelocity = displacement * velocityScale;

            if (targetVelocity.magnitude > maxVelocity)
            {
                targetVelocity = targetVelocity.normalized * maxVelocity;
            }

            // If drone 0 print calibration center point, current midpoint, and displacement, and target velocity
            if (transform.parent.name == "Drone 0")
            {
                Debug.Log($"Calibration Center: {calibrationCenterPoint}, Current Midpoint: {currentMidpoint}, Displacement: {displacement}, Target Velocity: {targetVelocity}");
            }


            
            // Apply to Olfati-Saber desired velocities
            // Hand Z movement (targetVelocity.z) controls OlfatiSaber's local X
            olfatiSaber.desired_vx = targetVelocity.z;
            // Hand X movement (targetVelocity.x) controls OlfatiSaber's local Z (via desired_vy)
            olfatiSaber.desired_vy = targetVelocity.x; 

            // Y-axis (World Y / Up-Down) control with safety floor
            if (isYPositionInitialized && vc != null)
            {
                float hand_commanded_world_y_velocity = targetVelocity.y; // Hand Y displacement controls OlfatiSaber's local Y
                float current_world_y = vc.transform.position.y;

                if (current_world_y < initialCalibratedYPosition - yDeadZone) {
                    olfatiSaber.desired_vz = Mathf.Max(hand_commanded_world_y_velocity, yRecoverySpeed);
                } else if (current_world_y <= initialCalibratedYPosition + yDeadZone) {
                    olfatiSaber.desired_vz = Mathf.Max(0.0f, hand_commanded_world_y_velocity);
                } else {
                    olfatiSaber.desired_vz = hand_commanded_world_y_velocity;
                }
            }
            else
            {
                olfatiSaber.desired_vz = 0.0f;
            }
        }

        // Yaw control based on palm line angle (no calibration, always active if hands tracked)
        Vector3 currentPalmConnection = HandProcessor.RightPalmPosition - HandProcessor.LeftPalmPosition;
        // Calculate angle of the palm connection line relative to the World Z-axis in the XZ plane.
        // Atan2(x, z) gives angle where 0 is along +Z, positive towards +X.
        float currentPalmAngleRelativeToWorldZ_rad = Mathf.Atan2(currentPalmConnection.x, currentPalmConnection.z);
        float currentPalmAngleRelativeToWorldZ_deg = currentPalmAngleRelativeToWorldZ_rad * Mathf.Rad2Deg;
        
        olfatiSaber.desired_yaw = currentPalmAngleRelativeToWorldZ_deg * yawRateScale;


        // Debug output (optional)
        if (Time.fixedTime % 0.5f < Time.fixedDeltaTime) 
        {
            // Debug.Log($"Hand Disp:({displacement.x:F2},{displacement.y:F2},{displacement.z:F2}), TargetVel:({targetVelocity.x:F2},{targetVelocity.y:F2},{targetVelocity.z:F2}) -> OS_desiredV:({olfatiSaber.desired_vx:F2}, {olfatiSaber.desired_vy:F2}, {olfatiSaber.desired_vz:F2}), OS_YawRate: {olfatiSaber.desired_yaw:F2}");
        }
    }

    // Public method to reset calibration (can be called from other scripts)
    public void ResetCalibration()
    {
        isCalibrated = false;
        // isCalibrating = false; // This flag was for the 3-2-1 system
        isYPositionInitialized = false;
        calibrationCenterPoint = Vector3.zero;
        
        initialAutoCalibrationStarted = false; // Allow initial auto-calibration to re-trigger
        if(isCalibrationCoroutineRunning) // Stop any ongoing calibration coroutine
        {
            StopCoroutine("DelayedAutomaticCalibration"); // Use string name if it's the only one
            isCalibrationCoroutineRunning = false;
        }

        Debug.Log("Calibration reset. Initial automatic hand calibration will re-attempt if hands are detected. Y-axis safety floor deactivated. Use 'C' for manual re-calibration.");
    }

    // Public method to check calibration status
    public bool IsCalibrated()
    {
        return isCalibrated;
    }

    // Public method to get current center point
    public Vector3 GetCalibrationCenterPoint()
    {
        return calibrationCenterPoint;
    }
}
