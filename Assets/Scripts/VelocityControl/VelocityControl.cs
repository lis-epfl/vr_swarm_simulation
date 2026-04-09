using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class VelocityControl : MonoBehaviour
{
    public enum ControlStyle { Custom, Soft, Racing }

    // Predefined profiles — values hardcoded here, applied at runtime when not Custom
    private static readonly float[] k_MaxPitch       = { 0f,    0.15f,  0.4f }; // Custom placeholder, Soft ~9°, Racing ~30°
    private static readonly float[] k_MaxRoll        = { 0f,    0.15f,  0.4f };
    private static readonly float[] k_MaxYawRate     = { 0f,    0.6f,   1.5f  };
    private static readonly float[] k_MaxSpeed       = { 0f,    3.0f,   15.0f };
    private static readonly float[] k_MaxAscentRate  = { 0f,    1.5f,   3.0f  };
    private static readonly float[] k_MaxDescentRate = { 0f,    1.5f,   2.5f  };
    private static readonly float[] k_MaxAlpha       = { 0f,    6.0f,   15.0f };

    [Header("Control Style")]
    public ControlStyle controlStyle = ControlStyle.Custom;

    public StateFinder State;

    public GameObject PropFL;
    public GameObject PropFR;
    public GameObject PropRR;
    public GameObject PropRL;

    private float gravity = 9.81f;
    private float timeConstantAcceleration = 0.5f;
    private float timeConstantOmegaXYRate = 0.1f; // Normal-person coordinates (roll/pitch)
    private float timeConstantAlphaXYRate = 0.05f; // Normal-person coordinates (roll/pitch)
    private float timeConstantAlphaZRate = 0.05f; // Normal-person coordinates (yaw)

    [Header("Rates & Limits")]
    public float maxPitch = 0.175f; // 10 Degrees in radians, otherwise small-angle approximation dies 
    public float maxRoll = 0.175f; // 10 Degrees in radians, otherwise small-angle approximation dies
    public float maxYawRate = 1.0f;
    public float maxAlpha = 10.0f;
    public float maxSpeed = 10.0f;
    public float MaxAscentRate = 3.0f; // Maximum ascent rate in m/s
    public float MaxDescentRate = 3.0f; // Maximum descent rate in m/s
    public float MinHeight = 0.5f;
    

    //must set this
    [Header("Setpoints")]
    public float desired_height = 4.0f;
    // User velocity commands in body frame (x = sideways/roll axis, z = forward/pitch axis)
    private float userVelX = 0.0f;
    private float userVelZ = 0.0f;
    public float desiredYawRate = 0.0f;
    public float attitude_control_yaw = 0.0f;
    // Swarm acceleration feedforward (world frame, set by SwarmAlgorithm)
    [HideInInspector] public Vector3 swarmAcceleration = Vector3.zero;

    // PD coefficients for height control
    [Header("Filters & Coefficients")]
    public float HeightKp = 2.0f;
    public float HeightKd = 1.0f;
    public float heightDerivFilterCoeff = 0.2f;
    public float yawFilterCoefficient = 0.15f;
    public float SwarmAccelFilterCoefficient = 0.3f;

    private float previousHeightError = 0.0f;
    private float filteredHeightErrorDerivative = 0.0f;
    private float userAltitudeRate = 0.0f;

    private float targetYawRate = 0.0f;
    private float filteredYawRate = 0.0f;

    [Header("Other")]
    public SwarmManager.SwarmAlgorithm currentAlgorithm;
    public float initial_height = 14.0f;

    private float speedScale = 500.0f;
    private Vector3 filteredSwarmAccel = Vector3.zero;
    private Vector3 initialPosition;
    private Quaternion initialRotation;
    
    // Use this for initialization
    void Start() {
        ApplyControlStyle();

        State.GetState ();
        Rigidbody rb = GetComponent<Rigidbody> ();
        Vector3 desiredForce = new Vector3 (0.0f, gravity * State.Mass, 0.0f);
        rb.AddForce (desiredForce, ForceMode.Acceleration);

        initial_height = State.Altitude;
        desired_height = initial_height;

        initialPosition = transform.position;
        initialRotation = transform.rotation;
    }

    // Called in editor when any field is changed in Inspector
    void OnValidate() => ApplyControlStyle();

    /// <summary>
    /// Overwrites the rate/limit fields with the values defined in the selected profile.
    /// Custom leaves all fields untouched so the Inspector values are used directly.
    /// </summary>
    public void ApplyControlStyle()
    {
        if (controlStyle == ControlStyle.Custom)
            return;

        int i = (int)controlStyle;
        maxPitch       = k_MaxPitch[i];
        maxRoll        = k_MaxRoll[i];
        maxYawRate     = k_MaxYawRate[i];
        maxSpeed       = k_MaxSpeed[i];
        MaxAscentRate  = k_MaxAscentRate[i];
        MaxDescentRate = k_MaxDescentRate[i];
        maxAlpha       = k_MaxAlpha[i];
    }

    // Update is called once per frame
    void FixedUpdate() {
        State.GetState ();
        
        // NOTE: I'm using stupid vector order (sideways, up, forward) at the end

        Vector3 desiredTheta;
        Vector3 desiredOmega;


        // --- Height control (PD) ---
        desired_height += userAltitudeRate * Time.deltaTime;
        desired_height = Mathf.Max(desired_height, MinHeight);

        float currentHeightError = desired_height - State.Altitude;
        float rawHeightErrorDerivative = (currentHeightError - previousHeightError) / Time.deltaTime;
        filteredHeightErrorDerivative = filteredHeightErrorDerivative * (1.0f - heightDerivFilterCoeff) + rawHeightErrorDerivative * heightDerivFilterCoeff;
        float altitudeCommand = HeightKp * currentHeightError + HeightKd * filteredHeightErrorDerivative;

        // --- User velocity controller (world frame) ---
        // Use StateFinder velocity (includes sensor noise) transformed to world frame
        Vector3 userWorldVel = transform.TransformDirection(new Vector3(userVelX, 0f, userVelZ));
        Vector3 worldVelocity = transform.TransformDirection(State.VelocityVector);
        Vector3 userVelError = worldVelocity - userWorldVel;
        Vector3 userAccel = userVelError * -1.0f / timeConstantAcceleration;

        // --- Swarm feedforward acceleration (world frame) ---
        // Swarm output is already an acceleration (cohesion + consensus + obstacle).
        // Filter it lightly to smooth inter-frame jitter, then add directly.
        filteredSwarmAccel = Vector3.Lerp(filteredSwarmAccel, swarmAcceleration, SwarmAccelFilterCoefficient);

        Vector3 desiredAcceleration = userAccel + filteredSwarmAccel;

        // World-frame acceleration → desired pitch/roll
        desiredTheta = new Vector3(desiredAcceleration.z / gravity, 0.0f, -desiredAcceleration.x / gravity);

        // Clamp the desired angles to the maximum allowed values
        desiredTheta.x = Mathf.Clamp(desiredTheta.x, -maxPitch, maxPitch);
        desiredTheta.z = Mathf.Clamp(desiredTheta.z, -maxRoll, maxRoll);

        Vector3 thetaError = State.Angles - desiredTheta;

        desiredOmega = thetaError * -1.0f / timeConstantOmegaXYRate;

        // Add the yaw rate contributions from user input and the autonomous control
        targetYawRate = desiredYawRate + attitude_control_yaw;

        // Apply the low-pass filter to reduce oscillations in yaw control
        filteredYawRate = filteredYawRate * (1.0f - yawFilterCoefficient) + targetYawRate * yawFilterCoefficient;

        // Clamp the filtered yaw rate to the maximum allowed value
        filteredYawRate = Mathf.Clamp(filteredYawRate, -maxYawRate, maxYawRate);

        // Use the filtered yaw rate for further calculations
        desiredOmega.y = filteredYawRate;

        Vector3 omegaError = State.AngularVelocityVector - desiredOmega;

        Vector3 desiredAlpha = Vector3.Scale(omegaError, new Vector3(-1.0f / timeConstantAlphaXYRate, -1.0f / timeConstantAlphaZRate, -1.0f / timeConstantAlphaXYRate));
        desiredAlpha = Vector3.Min(desiredAlpha, Vector3.one * maxAlpha);
        desiredAlpha = Vector3.Max(desiredAlpha, Vector3.one * maxAlpha * -1.0f);

        // float desiredThrust = (gravity + desiredAcceleration.y) / (Mathf.Cos(State.Angles.z) * Mathf.Cos(State.Angles.x));
        float desiredThrust = (gravity + altitudeCommand + desiredAcceleration.y) / (Mathf.Cos(State.Angles.z) * Mathf.Cos(State.Angles.x));
        desiredThrust = Mathf.Min(desiredThrust, 2.7f * gravity);
        desiredThrust = Mathf.Max(desiredThrust, 0.0f);

        Vector3 desiredTorque = Vector3.Scale(desiredAlpha, State.Inertia);
        Vector3 desiredForce = new Vector3(0.0f, desiredThrust * State.Mass, 0.0f);

        Rigidbody rb = GetComponent<Rigidbody>();

        rb.AddRelativeTorque(desiredTorque, ForceMode.Acceleration);
        rb.AddRelativeForce(desiredForce, ForceMode.Acceleration);

        //prop transforms
        PropFL.transform.Rotate(Vector3.forward * Time.deltaTime * desiredThrust * speedScale);
        PropFR.transform.Rotate(Vector3.forward * Time.deltaTime * desiredThrust * speedScale);
        PropRR.transform.Rotate(Vector3.forward * Time.deltaTime * desiredThrust * speedScale);
        PropRL.transform.Rotate(Vector3.forward * Time.deltaTime * desiredThrust * speedScale);

        // Update previous values
        previousHeightError = currentHeightError;

    }

    public void Reset()
    {

        State.Reset();
        State.Position = initialPosition;
        State.Angles = initialRotation.eulerAngles;
        State.VelocityVector = Vector3.zero;
        State.AngularVelocityVector = Vector3.zero;

        userVelX = 0.0f;
        userVelZ = 0.0f;
        desiredYawRate = 0.0f;
        desired_height = initial_height;

        transform.position = initialPosition;
        transform.rotation = initialRotation;

        enabled = true;
    }

    // Return max speed
    public float GetMaxSpeed()    => maxSpeed;
    public float GetMaxYawRate()  => maxYawRate;

    /// <summary>
    /// Set horizontal velocity commands from a normalised input in [-1, 1].
    /// Scaled against maxSpeed so the full stick always maps to the current limit.
    /// </summary>
    public void SetNormalisedVelocity(float normVx, float normVy)
    {
        userVelX = Mathf.Clamp(normVx, -1f, 1f) * maxSpeed;
        userVelZ = Mathf.Clamp(normVy, -1f, 1f) * maxSpeed;
    }

    /// <summary>
    /// Set yaw rate command from a normalised input in [-1, 1].
    /// Scaled against maxYawRate.
    /// </summary>
    public void SetNormalisedYawRate(float normYaw)
    {
        desiredYawRate = Mathf.Clamp(normYaw, -1f, 1f) * maxYawRate;
    }

    /// <summary>
    /// Set altitude rate command from a normalised input in [-1, 1].
    /// Positive = ascend (scaled against MaxAscentRate), negative = descend (MaxDescentRate).
    /// </summary>
    public void SetNormalisedAltitudeRate(float normAlt)
    {
        normAlt = Mathf.Clamp(normAlt, -1f, 1f);
        userAltitudeRate = normAlt >= 0f ? normAlt * MaxAscentRate : normAlt * MaxDescentRate;
    }
}
