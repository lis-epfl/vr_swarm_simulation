using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.XR.Hands.Samples.VisualizerSample; // Add this for HandProcessor access

public class InputControl : MonoBehaviour {

    public VelocityControl vc;
    public OlfatiSaber olfatiSaber;

    public SwarmManager.SwarmAlgorithm currentAlgorithm;

    private float abs_height = 1;

    [Header("Hand Control Settings")]
    public float velocityScale = 40.0f; // 5 m/s at 25cm distance = 20 ratio
    public float maxVelocity = 10.0f; // Maximum velocity in m/s
    
    // Calibration state
    private bool isCalibrated = false;
    private bool isCalibrating = false;
    private Vector3 calibrationCenterPoint = Vector3.zero;
    private float calibrationTimer = 0f;
    private int calibrationCountdown = 3;

    // Use this for initialization
    void Start()
    {
        
    }
    
    // Update is called once per frame
    void FixedUpdate () {
        
        // Handle calibration input
        HandleCalibration();
        
        if (currentAlgorithm == SwarmManager.SwarmAlgorithm.REYNOLDS)
        {
            // Default velocity control for Reynolds
            vc.desired_vx = 0.0f; //Input.GetAxisRaw ("Pitch")*4.0f;
            vc.desired_vy = 0.0f; //Input.GetAxisRaw ("Roll")*4.0f;
            vc.desired_yaw = 0.0f; //Input.GetAxisRaw ("Yaw")*0.5f;
            abs_height += 0.0f; //Input.GetAxisRaw("Throttle") * 0.1f;
            // vc.desired_height = abs_height;
        }
        else if (currentAlgorithm == SwarmManager.SwarmAlgorithm.OLFATI_SABER)
        {
            // Hand-controlled Olfati-Saber velocity control
            if (isCalibrated && HandProcessor.ArePalmsTracked)
            {
                UpdateSwarmVelocityFromHands();
            }
            else
            {
                // Default to zero velocity if not calibrated or hands not tracked
                olfatiSaber.desired_vx = 0.0f;
                olfatiSaber.desired_vy = 0.0f;
                olfatiSaber.desired_vz = 0.0f; // Note: This maps to world Y in your system
            }
            
            olfatiSaber.desired_yaw = 0.0f; //Input.GetAxisRaw ("Yaw")*0.5f;
            // Don't override desired_height here - let it be controlled by the drone's height controller
        }
    }

    private void HandleCalibration()
    {
        // Check for C key press to start calibration
        if (Input.GetKeyDown(KeyCode.C) && !isCalibrating)
        {
            StartCoroutine(CalibrationDelayRoutine());
        }

        // Handle calibration countdown
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

    // Add this coroutine for the 2 seconds wait
    private System.Collections.IEnumerator CalibrationDelayRoutine()
    {
        Debug.Log("Calibration will start in 2 seconds...");
        yield return new WaitForSeconds(2f);
        StartCalibration();
    }

    private void StartCalibration()
    {
        if (!HandProcessor.ArePalmsTracked)
        {
            Debug.LogWarning("Cannot calibrate: Hands not tracked!");
            return;
        }

        isCalibrating = true;
        isCalibrated = false;
        calibrationCountdown = 3;
        calibrationTimer = 0f;
        
        Debug.Log("Starting hand calibration...");
        Debug.Log($"Calibration countdown: {calibrationCountdown}");
    }

    private void CompleteCalibration()
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
        
        Debug.Log("Calibration complete!");
        Debug.Log($"Center point set at: {calibrationCenterPoint}");
        Debug.Log("Swarm can now be controlled by hand movement.");
    }

    private void UpdateSwarmVelocityFromHands()
    {
        if (!HandProcessor.ArePalmsTracked)
        {
            // If hands are lost, stop the swarm
            olfatiSaber.desired_vx = 0.0f;
            olfatiSaber.desired_vy = 0.0f;
            olfatiSaber.desired_vz = 0.0f;
            return;
        }

        // Get current hand midpoint
        Vector3 leftPalmPos = HandProcessor.LeftPalmPosition;
        Vector3 rightPalmPos = HandProcessor.RightPalmPosition;
        Vector3 currentMidpoint = (leftPalmPos + rightPalmPos) * 0.5f;

        // Calculate displacement from calibration center
        Vector3 displacement = currentMidpoint - calibrationCenterPoint;

        // Calculate velocity magnitude based on displacement
        float displacementMagnitude = displacement.magnitude;
        float velocityMagnitude = displacementMagnitude * velocityScale;
        
        // Clamp to maximum velocity
        velocityMagnitude = Mathf.Min(velocityMagnitude, maxVelocity);

        // Calculate velocity direction (normalized displacement)
        Vector3 velocityDirection = Vector3.zero;
        if (displacementMagnitude > 0.001f) // Avoid division by zero
        {
            velocityDirection = displacement.normalized;
        }

        // Calculate final velocity vector
        Vector3 targetVelocity = velocityDirection * velocityMagnitude;

        // Apply to Olfati-Saber desired velocities
        // Note: Your system maps desired_vy to world Z and desired_vz to world Y
        olfatiSaber.desired_vx = targetVelocity.x;
        olfatiSaber.desired_vz = targetVelocity.z; // World Z mapped to desired_vy
        olfatiSaber.desired_vy = 0.0f; // Keep vertical component zero for stability

        // Debug output (only occasionally to avoid spam)
        if (Time.fixedTime % 0.5f < Time.fixedDeltaTime) // Every 0.5 seconds
        {
            //Debug.Log($"Hand displacement: {displacement.magnitude:F3}m, " +
                     //$"Velocity: ({targetVelocity.x:F2}, {targetVelocity.y:F2}, {targetVelocity.z:F2}) m/s");
        }
    }

    // Public method to reset calibration (can be called from other scripts)
    public void ResetCalibration()
    {
        isCalibrated = false;
        isCalibrating = false;
        calibrationCenterPoint = Vector3.zero;
        Debug.Log("Calibration reset.");
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
