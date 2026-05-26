using System.Collections;
using System.Collections.Generic;
using System.IO;
using UnityEngine;

public class VelocityControl : MonoBehaviour
{
    [Header("Control Style")]
    public FlightProfile activeProfile;

    public StateFinder State;

    public GameObject PropFL;
    public GameObject PropFR;
    public GameObject PropRR;
    public GameObject PropRL;

    private float gravity = 9.81f;
    private float timeConstantOmegaXYRate = 0.1f; // Normal-person coordinates (roll/pitch)
    private float timeConstantAlphaXYRate = 0.05f; // Normal-person coordinates (roll/pitch)
    private float timeConstantAlphaZRate = 0.05f; // Normal-person coordinates (yaw)

    [Header("Rates & Limits")]
    public float maxPitch = 0.175f; // 10 Degrees in radians, otherwise small-angle approximation dies
    public float maxRoll = 0.175f; // 10 Degrees in radians, otherwise small-angle approximation dies
    public float maxYawRate = 1.0f;
    public float maxAlpha = 10.0f;
    public float maxSpeed = 10.0f;
    public float maxAltitudeRate = 3.0f; // Maximum altitude rate in m/s
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
    // Last-frame horizontal (XZ) acceleration magnitudes — read by FlightHUD
    [HideInInspector] public float lastUserAccelMag  = 0f;
    [HideInInspector] public float lastSwarmAccelMag = 0f;
    [HideInInspector] public float lastThrustClamped = 0f;

    // PD coefficients for height control
    [Header("Filters & Coefficients")]
    public float HeightKp = 2.0f;
    public float HeightKd = 1.0f;
    public float heightDerivFilterCoeff = 0.2f;
    public float yawFilterCoefficient = 0.15f;
    public float SwarmAccelFilterCoefficient = 0.3f;
    [Tooltip("Time constant (s) of the velocity → acceleration P-controller. " +
             "Larger = softer velocity response = more angle budget left for swarm corrections. " +
             "Saturation threshold ≈ g × maxPitch × tau.")]
    public float timeConstantAcceleration = 0.5f;

    private float previousHeightError = 0.0f;
    private float filteredHeightErrorDerivative = 0.0f;
    private float userAltitudeRate = 0.0f;

    private float targetYawRate = 0.0f;
    private float filteredYawRate = 0.0f;

    [Header("Other")]
    public SwarmManager.SwarmAlgorithm currentAlgorithm;
    public float initial_height = 14.0f;
    public bool logToCSV = false;
    public string logDirectory = "C:/Users/ahebert/Desktop";

    private float speedScale = 500.0f;
    private Vector3 worldFilteredSwarmAccel = Vector3.zero;
    private Vector3 initialPosition;
    private Quaternion initialRotation;

    private StreamWriter csvStreamWriter;
    
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

        if (logToCSV)
        {
            string path = Path.Combine(Application.persistentDataPath, "control_log_" + gameObject.name + ".csv");
            csvStreamWriter = new StreamWriter(path, false, System.Text.Encoding.UTF8); // Overwrite existing file
            csvStreamWriter.WriteLine("Time;UserAccelX;UserAccelY;UserAccelZ;SwarmAccelX;SwarmAccelY;SwarmAccelZ;DesiredThetaX;DesiredThetaY;DesiredThetaZ;DesiredOmegaX;DesiredOmegaY;DesiredOmegaZ;DesiredAlphaX;DesiredAlphaY;DesiredAlphaZ;DesiredThrust;DesiredTorqueX;DesiredTorqueY;DesiredTorqueZ;DesiredForceX;DesiredForceY;DesiredForceZ");
        }
    }

    // Called in editor when any field is changed in Inspector
    void OnValidate() => ApplyControlStyle();

    /// <summary>
    /// Applies the active flight profile to all rate/limit fields.
    /// If no profile is assigned, fields remain unchanged.
    /// </summary>
    public void ApplyControlStyle()
    {
        if (activeProfile == null)
            return;

        maxPitch               = activeProfile.maxPitch;
        maxRoll                = activeProfile.maxRoll;
        maxYawRate             = activeProfile.maxYawRate;
        maxSpeed               = activeProfile.maxSpeed;
        maxAltitudeRate        = activeProfile.maxAltitudeRate;
        maxAlpha               = activeProfile.maxAlpha;
        timeConstantAcceleration = activeProfile.timeConstantAccel;
    }

    // Update is called once per frame
    void FixedUpdate() {
        State.GetState ();

        if (!State.IsAlive)
        {
            userAltitudeRate = 0f;
            return;
        }

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

        // --- User velocity controller ---
        // Force any "ghost" y  component zoming from Unity's transform due to drone's tilt to zero
        Vector3 bodyVelocity = State.VelocityVector;
        Vector3 bodyUserVelCommand = new Vector3(userVelX, 0f, userVelZ);
        Vector3 userVelError = transform.TransformDirection(bodyVelocity - bodyUserVelCommand);
        Vector3 worldUserAccel = userVelError * -1.0f / timeConstantAcceleration;
        worldUserAccel.y = 0f; // Add altitude command to vertical acceleration

        // --- Swarm feedforward acceleration (world frame) ---
        // Swarm already computes acceleration in world frame
        worldFilteredSwarmAccel = Vector3.Lerp(worldFilteredSwarmAccel, swarmAcceleration, SwarmAccelFilterCoefficient);

        Vector3 desiredAcceleration = worldUserAccel + worldFilteredSwarmAccel;

        // Convert combined acceleration back to body frame before mapping to pitch/roll.
        Vector3 bodyDesiredAccel = transform.InverseTransformDirection(desiredAcceleration);
        desiredTheta = new Vector3(bodyDesiredAccel.z / gravity, 0.0f, -bodyDesiredAccel.x / gravity);

        // Per-axis clamp respects asymmetric pitch/roll limits
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
        Vector3 desiredAlphaClamped = Vector3.Min(desiredAlpha, Vector3.one * maxAlpha);
        desiredAlphaClamped = Vector3.Max(desiredAlphaClamped, Vector3.one * maxAlpha * -1.0f);

        // float desiredThrust = (gravity + desiredAcceleration.y) / (Mathf.Cos(State.Angles.z) * Mathf.Cos(State.Angles.x));
        float desiredThrust = (gravity + altitudeCommand + desiredAcceleration.y) / (Mathf.Cos(State.Angles.z) * Mathf.Cos(State.Angles.x));
        float  desiredThrustClamped = Mathf.Min(desiredThrust, 2.7f * gravity);
        desiredThrustClamped = Mathf.Max(desiredThrustClamped, 0.0f);
        lastThrustClamped = desiredThrustClamped;

        Vector3 desiredTorque = Vector3.Scale(desiredAlphaClamped, State.Inertia);
        Vector3 desiredForce = new Vector3(0.0f, desiredThrustClamped * State.Mass, 0.0f);

        Rigidbody rb = GetComponent<Rigidbody>();

        rb.AddRelativeTorque(desiredTorque, ForceMode.Acceleration);
        rb.AddRelativeForce(desiredForce, ForceMode.Acceleration);

        //prop transforms
        PropFL.transform.Rotate(Vector3.forward * Time.deltaTime * desiredThrustClamped * speedScale);
        PropFR.transform.Rotate(Vector3.forward * Time.deltaTime * desiredThrustClamped * speedScale);
        PropRR.transform.Rotate(Vector3.forward * Time.deltaTime * desiredThrustClamped * speedScale);
        PropRL.transform.Rotate(Vector3.forward * Time.deltaTime * desiredThrustClamped * speedScale);

        // Update previous values
        previousHeightError = currentHeightError;

        if (logToCSV)
            dumpToCSVFile(worldUserAccel, worldFilteredSwarmAccel, desiredTheta, desiredOmega, desiredAlpha, desiredAlphaClamped, desiredThrust, desiredThrustClamped, desiredTorque, desiredForce);
    }

    private void dumpToCSVFile(
        Vector3 userAccel, 
        Vector3 swarmAccel, 
        Vector3 desiredTheta, 
        Vector3 desiredOmega, 
        Vector3 desiredAlpha, 
        Vector3 desiredAlphaClamped, 
        float desiredThrust, 
        float desiredThrustClamped,
        Vector3 desiredTorque,
        Vector3 desiredForce)
    {
        if (csvStreamWriter != null)
        {
            string line = $"{System.DateTimeOffset.UtcNow.ToUnixTimeMilliseconds()};" +
                          $"{userAccel.x};{userAccel.y};{userAccel.z};" +
                          $"{swarmAccel.x};{swarmAccel.y};{swarmAccel.z};" +
                          $"{desiredTheta.x};{desiredTheta.y};{desiredTheta.z};" +
                          $"{desiredOmega.x};{desiredOmega.y};{desiredOmega.z};" +
                          $"{desiredAlpha.x};{desiredAlpha.y};{desiredAlpha.z};" +
                          $"{desiredAlphaClamped.x};{desiredAlphaClamped.y};{desiredAlphaClamped.z};" +
                          $"{desiredThrust};" +
                          $"{desiredThrustClamped};" +
                          $"{desiredTorque.x};{desiredTorque.y};{desiredTorque.z};" +
                          $"{desiredForce.x};{desiredForce.y};{desiredForce.z}";
            csvStreamWriter.WriteLine(line);
            csvStreamWriter.Flush();
        }
    }

     /// <summary>
        /// Adjusts the drone's control parameters based on the current CWL level.
        /// - If CWL is Low: incrementally increase difficulty by tightening limits (lower max speed; sharper turns).

    void OnDestroy()
    {
        csvStreamWriter?.Close();
    }

    public void Reset()
    {
        ResetToPos(initialPosition, initialRotation);
    }

    public void ResetToPos(Vector3 newPos, Quaternion? newRot = null)
    {
        State.ResetToPos(newPos, newRot ?? initialRotation);
        userVelX = 0.0f;
        userVelZ = 0.0f;
        desiredYawRate = 0.0f;
        desired_height = newPos.y;

        transform.position = newPos;
        transform.rotation = newRot ?? initialRotation;

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
        userAltitudeRate = normAlt * maxAltitudeRate;
    }
}
