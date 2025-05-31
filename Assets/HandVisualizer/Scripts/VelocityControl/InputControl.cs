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
    public bool useAutomaticCalibrationPoint = false; // Set to true for automatic calibration
    public Transform playerViewpointTransform; // Assign player's viewpoint transform (e.g., Main Camera or Player GameObject) for automatic calibration

    [Header("Hand Control Settings")]
    public float velocityScale = 5.0f; // User's value
    public float maxVelocity = 10.0f; // Maximum velocity in m/s
    public float yawRateScale = 0.5f; // Sensitivity for yaw control (degrees of hand rotation to degrees/sec of yaw rate)
    public float handDeadZoneRadius = 0.05f; // 5cm radius for the dead zone (10cm diameter)
    
    // Calibration state
    private bool isCalibrated = false;
    private bool isCalibrating = false;
    private Vector3 calibrationCenterPoint = Vector3.zero;
    private float calibrationTimer = 0f;
    private int calibrationCountdown = 3;

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

        if (useAutomaticCalibrationPoint)
        {
            PerformAutomaticCalibration();
        }
    }
    
    void PerformAutomaticCalibration()
    {
        if (playerViewpointTransform == null)
        {
            Debug.LogError("PlayerViewpointTransform not assigned in InputControl for automatic calibration. Automatic calibration failed.");
            return;
        }
        if (vc == null)
        {
            Debug.LogError("VelocityControl (vc) not assigned. Cannot set initial Y position for automatic calibration. Automatic calibration failed.");
            return;
        }

        // Calculate calibration point 30cm in front and 30cm below the player's viewpoint
        // calibrationCenterPoint = playerViewpointTransform.position + playerViewpointTransform.forward * 0.2f - playerViewpointTransform.up * 0.3f;
        // calibrationCenterPoint = playerViewpointTransform.forward * 0.2f - playerViewpointTransform.up * 0.3f;
        // Set the calibration center point to the current hand midpoint
        Vector3 leftPalmPos = HandProcessor.LeftPalmPosition;
        Vector3 rightPalmPos = HandProcessor.RightPalmPosition;
        calibrationCenterPoint = (leftPalmPos + rightPalmPos) * 0.5f;
        isCalibrated = true; // Mark as calibrated       
        

        // Initialize Y position safety floor based on a representative drone's current height
        initialCalibratedYPosition = vc.transform.position.y;
        isYPositionInitialized = true;

        Debug.Log($"Automatic calibration complete! Center point set to: {calibrationCenterPoint}.");
        
        // Calibrate palm angle for yaw if hands are tracked - REMOVED YAW CALIBRATION LOGIC
        // if (HandProcessor.ArePalmsTracked) { // REMOVED
        //     Vector3 palmConnection = HandProcessor.RightPalmPosition - HandProcessor.LeftPalmPosition; // REMOVED
        //     initialPalmLineAngleWorld = Mathf.Atan2(palmConnection.z, palmConnection.x) * Mathf.Rad2Deg; // REMOVED
        //     isPalmAngleCalibrated = true; // REMOVED
        //     Debug.Log($"Automatic calibration: Initial palm line angle for yaw set to {initialPalmLineAngleWorld:F2} degrees."); // REMOVED
        // } else { // REMOVED
        //     isPalmAngleCalibrated = false; // REMOVED
        //     Debug.LogWarning("Automatic calibration: Hands not tracked, palm angle for yaw not calibrated. Yaw control via hand angle will be disabled until manual calibration with hands."); // REMOVED
        // } // REMOVED
        Debug.Log("Swarm can now be controlled by hand movement. Yaw control is active if hands are tracked (no calibration needed).");
    }

    // Update is called once per frame
    void FixedUpdate () {

        // Handle 'C' key calibration only if not using automatic calibration and not already calibrated by it
        if (!useAutomaticCalibrationPoint)
        {
            // HandleCalibration();
            // PerformAutomaticCalibration(); // Automatically calibrate if automatic calibration is enabled
        }

        // With this:
        if (Input.GetKeyDown(KeyCode.C))
        {
            StartCoroutine(DelayedAutomaticCalibration());
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

    // Add this coroutine method anywhere in the class:
    private System.Collections.IEnumerator DelayedAutomaticCalibration()
    {
        Debug.Log("Waiting 5 seconds before automatic calibration...");
        yield return new WaitForSeconds(5f);
        PerformAutomaticCalibration();
    }

    private void HandleCalibration()
    {
        // Check for C key press to start calibration if manual mode is active
        if (!useAutomaticCalibrationPoint && Input.GetKeyDown(KeyCode.C) && !isCalibrating && !isCalibrated)
        {
            StartCoroutine(CalibrationDelayRoutine());
        }

        // Handle calibration countdown (this part is shared if manual calibration was initiated
        if (isCalibrating)
        {
            calibrationTimer += Time.fixedDeltaTime;

            if (calibrationTimer >= 1.0f) // Every second
            {
                calibrationTimer = 0f;
                calibrationCountdown--;

                if (calibrationCountdown > 0)
                {
                    Debug.Log($"Calibration countdown: {calibrationCountdown}");
                }
                else
                {
                    CompleteCalibration();
                }
            }
        }
    }

    private System.Collections.IEnumerator CalibrationDelayRoutine()
    {
        Debug.Log("Calibration will start in 2 seconds...");
        yield return new WaitForSeconds(2f);
        StartCalibration();
    }

    private void StartCalibration() // This is for manual 'C' key calibration
    {
        if (!HandProcessor.ArePalmsTracked)
        {
            Debug.LogWarning("Cannot calibrate: Hands not tracked!");
            return;
        }

        isCalibrating = true;
        isCalibrated = false;
        isYPositionInitialized = false; // Reset Y position initialization
        // isPalmAngleCalibrated = false; // REMOVED Reset palm angle calibration
        calibrationCountdown = 3;
        calibrationTimer = 0f;
        
        Debug.Log("Starting hand calibration...");
        Debug.Log($"Calibration countdown: {calibrationCountdown}");
    }

    private void CompleteCalibration() // This is for manual 'C' key calibration
    {
        if (!HandProcessor.ArePalmsTracked)
        {
            Debug.LogWarning("Calibration failed: Hands not tracked!");
            isCalibrating = false;
            return;
        }

        // Calculate midpoint between palms
        Vector3 leftPalmPos = HandProcessor.LeftPalmPosition;
        Vector3 rightPalmPos = HandProcessor.RightPalmPosition;
        calibrationCenterPoint = (leftPalmPos + rightPalmPos) * 0.5f;
        
        isCalibrating = false;
        isCalibrated = true;

        // Initialize Y position safety floor
        if (vc != null)
        {
            initialCalibratedYPosition = vc.transform.position.y;
            isYPositionInitialized = true;
            Debug.Log($"Manual calibration complete! Center point: {calibrationCenterPoint}. Initial Y swarm position set to: {initialCalibratedYPosition:F2}m");
        }
        else
        {
            Debug.LogError("VelocityControl (vc) not assigned. Cannot set initial Y position for safety floor.");
            isYPositionInitialized = false;
            Debug.LogWarning("Calibration complete, but Y-axis safety floor is NOT active.");
        }

        // Calibrate palm angle for yaw - REMOVED YAW CALIBRATION
        // Vector3 palmConnection = HandProcessor.RightPalmPosition - HandProcessor.LeftPalmPosition; // REMOVED
        // initialPalmLineAngleWorld = Mathf.Atan2(palmConnection.z, palmConnection.x) * Mathf.Rad2Deg; // REMOVED
        // isPalmAngleCalibrated = true; // REMOVED
        // Debug.Log($"Manual calibration: Initial palm line angle for yaw set to {initialPalmLineAngleWorld:F2} degrees."); // REMOVED
        
        Debug.Log("Swarm can now be controlled by hand movement. Yaw control is active if hands are tracked (no calibration needed).");
    }

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
        isCalibrating = false;
        isYPositionInitialized = false;
        // isPalmAngleCalibrated = false; // REMOVED
        calibrationCenterPoint = Vector3.zero;
        Debug.Log("Calibration reset. Y-axis safety floor deactivated until next calibration. Automatic calibration will re-run if enabled and scene reloads/script re-enables, or use 'C' for manual. Yaw control remains active if hands are tracked.");
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
