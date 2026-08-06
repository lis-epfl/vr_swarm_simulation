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
    // User velocity commands (x = sideways/roll axis, z = forward/pitch axis).
    // Interpreted in body or world frame depending on userCommandInWorldFrame.
    private float userVelX = 0.0f;
    private float userVelZ = 0.0f;
    // When true, the velocity command is interpreted in world frame (fixed axes); when false,
    // in the drone's body frame (relative to heading). Set by InputManager via SwarmAlgorithm.
    [HideInInspector] public bool userCommandInWorldFrame = false;
    // Yaw (degrees) the world-frame command is expressed relative to. 0 = fixed world axes (World frame);
    // the pilot body yaw for the VR frame. Set by SwarmAlgorithm each tick.
    [HideInInspector] public float commandReferenceYaw = 0f;
    public float desiredYawRate = 0.0f;
    public float attitude_control_yaw = 0.0f;
    // Swarm acceleration feedforward (world frame, set by SwarmAlgorithm)
    [HideInInspector] public Vector3 swarmAcceleration = Vector3.zero;
    // When true, the vertical channel is handed to the swarm: the altitude-hold PD is replaced by a
    // vertical *velocity* loop (symmetric with the horizontal one), so the swarm's vertical
    // acceleration drives the drone instead of being fought by the height setpoint. Set every tick
    // by SwarmAlgorithm; true only while swarming in a non-horizontal plane.
    [HideInInspector] public bool verticalSwarmAuthority = false;
    // Altitude the vertical leash is measured against — SwarmPlaneController.ReferenceAltitude, the
    // wall's own reference rather than any drone's. Set by SwarmAlgorithm.
    [HideInInspector] public float verticalReferenceAltitude = 0f;
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

    [Tooltip("Seconds. Converts the swarm's vertical acceleration into a climb rate for the height " +
             "setpoint while the swarm owns the vertical channel (verticalSwarmAuthority). Larger = " +
             "the wall forms faster vertically. The resulting rate is clamped to maxAltitudeRate.")]
    public float swarmVerticalSetpointGain = 0.5f;

    [Tooltip("Metres. Hard limit on how far above or below the wall's reference altitude the height " +
             "setpoint may be driven while swarming in a tilted plane — i.e. the wall's half-height. " +
             "This is the absolute bound on vertical drift, so keep it near the formation size you expect.")]
    public float swarmVerticalLeash = 30.0f;

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

    private Rigidbody rb;  // cached: FixedUpdate applies force/torque every tick

    void Awake() {
        rb = GetComponent<Rigidbody> ();
    }

    // Use this for initialization
    void Start() {
        ApplyControlStyle();

        State.GetState ();
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


        // --- Swarm feedforward acceleration (world frame) ---
        // Swarm already computes acceleration in world frame. Filtered here, ahead of the height
        // loop, because the vertical channel consumes it below.
        worldFilteredSwarmAccel = Vector3.Lerp(worldFilteredSwarmAccel, swarmAcceleration, SwarmAccelFilterCoefficient);

        // --- Height control (PD) ---
        // The throttle stick always moves the setpoint. While the swarm owns the vertical channel
        // (a tilted swarming plane, where the formation's spread is mostly vertical) the swarm's
        // vertical acceleration moves the setpoint as well, rather than being added into thrust.
        //
        // Driving the *setpoint* is what keeps this stable. Handing the channel over as a velocity
        // command — the obvious symmetric counterpart of the horizontal controller — leaves the
        // vertical axis with no position feedback at all, and a velocity loop has no DC gain on
        // position: any sustained bias (ground repulsion, an asymmetric formation, drones dying off
        // the bottom of the wall) then integrates into a permanent climb instead of settling at a
        // bounded offset. Through the setpoint the PD keeps its disturbance rejection — at formation
        // equilibrium the swarm force is zero, the setpoint stops moving, and the PD holds the drone
        // exactly where the wall wants it.
        float heightRate = userAltitudeRate;
        if (verticalSwarmAuthority)
        {
            heightRate += Mathf.Clamp(swarmVerticalSetpointGain * worldFilteredSwarmAccel.y,
                                      -maxAltitudeRate, maxAltitudeRate);
        }

        desired_height += heightRate * Time.deltaTime;

        if (verticalSwarmAuthority)
        {
            // The wall has a finite vertical extent, so leash the setpoint to the wall's reference
            // altitude. This is the absolute bound on drift, whatever its source.
            desired_height = Mathf.Clamp(desired_height,
                                         verticalReferenceAltitude - swarmVerticalLeash,
                                         verticalReferenceAltitude + swarmVerticalLeash);
        }

        desired_height = Mathf.Max(desired_height, MinHeight);

        float currentHeightError = desired_height - State.Altitude;
        float rawHeightErrorDerivative = (currentHeightError - previousHeightError) / Time.deltaTime;
        filteredHeightErrorDerivative = filteredHeightErrorDerivative * (1.0f - heightDerivFilterCoeff) + rawHeightErrorDerivative * heightDerivFilterCoeff;
        float altitudeCommand = HeightKp * currentHeightError + HeightKd * filteredHeightErrorDerivative;

        // --- User velocity controller ---
        // The command can be interpreted in the body frame (moves relative to the drone's
        // heading) or the world frame (moves along fixed world axes, optionally rotated by a
        // reference heading — see commandReferenceYaw, which the VR frame sets to the pilot body
        // yaw). Either way the velocity error is expressed in world frame before being turned
        // into an acceleration.
        Vector3 bodyVelocity = State.VelocityVector;
        Vector3 userVelCommand = new Vector3(userVelX, 0f, userVelZ);

        Vector3 worldUserVelError;
        if (userCommandInWorldFrame)
        {
            // Command is in world frame, expressed relative to commandReferenceYaw (0 = fixed
            // world axes); rotate it into world space, then compare against the world velocity.
            Vector3 worldCommand = Quaternion.Euler(0f, commandReferenceYaw, 0f) * userVelCommand;
            Vector3 worldVelocity = transform.TransformDirection(bodyVelocity);
            worldUserVelError = worldVelocity - worldCommand;
        }
        else
        {
            // Command is in body frame; take the error in body frame, then rotate to world.
            worldUserVelError = transform.TransformDirection(bodyVelocity - userVelCommand);
        }

        Vector3 worldUserAccel = worldUserVelError * -1.0f / timeConstantAcceleration;
        // Force any "ghost" y component coming from the drone's tilt to zero (altitude handled separately).
        worldUserAccel.y = 0f;

        // worldFilteredSwarmAccel was filtered above, ahead of the height loop that consumes it.
        Vector3 desiredAcceleration = worldUserAccel + worldFilteredSwarmAccel;

        if (verticalSwarmAuthority)
        {
            // The swarm's vertical force is already being applied through the height setpoint above,
            // so it must not go into thrust as well. Dropping it here also keeps it out of the tilt
            // map below, where InverseTransformDirection would otherwise bleed it into pitch/roll in
            // proportion to the drone's tilt — corrupting the horizontal command exactly when the
            // swarm is pushing hardest vertically. The thrust clamp's [0, 2.7g] asymmetry stops
            // mattering for the same reason: the vertical demand now goes through the PD, which is
            // bounded by the setpoint rather than by the raw swarm force.
            desiredAcceleration.y = 0f;
        }

        // Convert combined acceleration back to body frame before mapping to pitch/roll.
        Vector3 bodyDesiredAccel = transform.InverseTransformDirection(desiredAcceleration);
        desiredTheta = new Vector3(bodyDesiredAccel.z / gravity, 0.0f, -bodyDesiredAccel.x / gravity);

        // Circular tilt limit: cap the combined pitch/roll magnitude while preserving direction,
        // so the acceleration envelope is the same in every horizontal direction. Unlike an
        // independent per-axis clamp (a square envelope, ~41% larger on the diagonal), a circular
        // envelope is rotation-invariant, so a world-frame command maps to the same achievable
        // acceleration on every drone regardless of heading — which keeps the formation together.
        Vector2 horizTilt = new Vector2(desiredTheta.x, desiredTheta.z);
        float maxTilt = Mathf.Min(maxPitch, maxRoll);
        if (horizTilt.magnitude > maxTilt)
        {
            horizTilt = horizTilt.normalized * maxTilt;
            desiredTheta.x = horizTilt.x;
            desiredTheta.z = horizTilt.y;
        }

        Vector3 thetaError = State.Angles - desiredTheta;

        desiredOmega = thetaError * -1.0f / timeConstantOmegaXYRate;

        // Add the yaw rate contributions from user input and the autonomous control
        targetYawRate = desiredYawRate + attitude_control_yaw;

        // Apply the low-pass filter to reduce oscillations in yaw control
        filteredYawRate = filteredYawRate * (1.0f - yawFilterCoefficient) + targetYawRate * yawFilterCoefficient;

        // Clamp the filtered yaw rate to the maximum allowed value
        filteredYawRate = Mathf.Clamp(filteredYawRate, -maxYawRate, maxYawRate);

        // filteredYawRate is a heading rate about the WORLD vertical, but desiredOmega is a
        // body-frame angular-velocity command (differenced against the body-frame
        // State.AngularVelocityVector below). When the drone tilts to translate, body-Y is no
        // longer world-up, so writing the yaw rate straight into desiredOmega.y under-rotates the
        // heading and bleeds the command into pitch/roll — heading drifts during motion. Express
        // the world-vertical yaw rate in the body frame instead. desiredOmega.y currently holds a
        // meaningless -thetaError.y/tau term (desiredTheta.y is always 0), so clear it first.
        desiredOmega.y = 0.0f;
        Vector3 upBody = transform.InverseTransformDirection(Vector3.up);
        // The pitch/roll rate commands themselves also have a world-vertical component while the
        // drone is tilted, so aggressive tilt reversals rotate the heading even with a zero yaw
        // command — far faster than the outer heading loop can correct. Subtract that component in
        // the yaw channel so the commanded body rate's world-vertical projection equals
        // filteredYawRate exactly (Dot(desiredOmega, upBody) == filteredYawRate), instead of
        // filteredYawRate plus the tilt-correction leak.
        float tiltHeadingLeak = desiredOmega.x * upBody.x + desiredOmega.z * upBody.z;
        desiredOmega += (filteredYawRate - tiltHeadingLeak) * upBody;

        Vector3 omegaError = State.AngularVelocityVector - desiredOmega;

        Vector3 desiredAlpha = Vector3.Scale(omegaError, new Vector3(-1.0f / timeConstantAlphaXYRate, -1.0f / timeConstantAlphaZRate, -1.0f / timeConstantAlphaXYRate));

        // Circular angular-acceleration limit, for the same reason as the tilt limit above.
        // A per-axis clamp is a square envelope (~41% larger on the diagonal), so a drone whose
        // body axes align with the required tilt-change direction saturates at maxAlpha while one
        // at 45 degrees to it gets up to maxAlpha*sqrt(2). For a world-frame command the body-frame
        // direction of the tilt change depends on yaw, so a square clamp makes the angular response
        // heading-dependent. Clamp the (pitch, roll) magnitude instead so the rate of tilting is the
        // same in every direction; clamp yaw independently since it is a separate axis/time constant.
        Vector3 desiredAlphaClamped = desiredAlpha;
        Vector2 horizAlpha = new Vector2(desiredAlpha.x, desiredAlpha.z);
        if (horizAlpha.magnitude > maxAlpha)
        {
            horizAlpha = horizAlpha.normalized * maxAlpha;
            desiredAlphaClamped.x = horizAlpha.x;
            desiredAlphaClamped.z = horizAlpha.y;
        }
        desiredAlphaClamped.y = Mathf.Clamp(desiredAlphaClamped.y, -maxAlpha, maxAlpha);

        // float desiredThrust = (gravity + desiredAcceleration.y) / (Mathf.Cos(State.Angles.z) * Mathf.Cos(State.Angles.x));
        float desiredThrust = (gravity + altitudeCommand + desiredAcceleration.y) / (Mathf.Cos(State.Angles.z) * Mathf.Cos(State.Angles.x));
        float  desiredThrustClamped = Mathf.Min(desiredThrust, 2.7f * gravity);
        desiredThrustClamped = Mathf.Max(desiredThrustClamped, 0.0f);
        lastThrustClamped = desiredThrustClamped;

        Vector3 desiredTorque = Vector3.Scale(desiredAlphaClamped, State.Inertia);
        Vector3 desiredForce = new Vector3(0.0f, desiredThrustClamped * State.Mass, 0.0f);

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
        Debug.Log("Resetting drone to initial position and state.");
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
    /// The input is magnitude-limited (not clamped per axis) so the commanded speed is the same
    /// in every direction — a full diagonal maps to maxSpeed, not maxSpeed·√2.
    /// </summary>
    public void SetNormalisedVelocity(float normVx, float normVy)
    {
        Vector2 norm = new Vector2(normVx, normVy);
        if (norm.magnitude > 1f)
            norm.Normalize();

        userVelX = norm.x * maxSpeed;
        userVelZ = norm.y * maxSpeed;
    }

    /// <summary>
    /// Selects whether the velocity command is interpreted in the world frame (fixed axes,
    /// true) or the drone's body frame (relative to heading, false). When in world frame,
    /// referenceYawDegrees rotates the command about the world vertical (0 = fixed axes;
    /// the pilot body yaw for the VR frame).
    /// </summary>
    public void SetCommandFrame(bool useWorldFrame, float referenceYawDegrees = 0f)
    {
        userCommandInWorldFrame = useWorldFrame;
        commandReferenceYaw = referenceYawDegrees;
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
