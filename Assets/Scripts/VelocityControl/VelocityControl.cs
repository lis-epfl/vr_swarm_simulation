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
    // One rate-loop constant for all three body axes, not the two this used to have. The
    // world-vertical yaw command is projected into the body frame via upBody below, so it lands on
    // all three axes at once; that projection is exact only while the three share a gain. Unequal
    // time constants rotate the commanded vector away from upBody and leak heading on every tilt
    // change — which is what made the drones wander in yaw whenever the velocity command changed.
    // The value preserves today's pitch/roll response through the ForceMode fix on the torque
    // below: 0.05 / 0.3893, the inertia factor that used to be applied twice.
    private float timeConstantAlphaRate = 0.1284f;

    [Header("Rates & Limits")]
    // Defaults below match DroneReduced.prefab's serialized values, so the two cannot disagree.
    // 25 degrees in radians. The tilt map is small-angle (it commands a/g as an angle), so at this
    // limit the delivered horizontal acceleration runs roughly 11% above the command.
    public float maxPitch = 0.436332f;
    public float maxRoll = 0.436332f;
    // 75 deg/s in rad/s — the DJI Mini 3 Pro's maximum yaw rate. This is the airframe's limit; the
    // real fleet's PC additionally clamps its own commands at 40 deg/s (MAX_YAW_RATE_DEG_S), which
    // the sim does not reproduce.
    public float maxYawRate = 1.309f;
    // Angular-acceleration budget, rad/s^2. Scaled by 0.3893 from the 8.68 this used to hold: the
    // torque below is applied with ForceMode.Force now, so this is the limit that is actually
    // achieved, where before the inertia pre-multiply silently turned it into 3.38 on pitch/roll
    // and 6.67 on yaw. Same delivered authority, honestly stated, and isotropic for the first time.
    // On the yaw axis it is the NET acceleration: the Rigidbody's angularDrag is compensated on top
    // of it (see the rate loop), because spent fighting that drag it capped yaw at 35 deg/s.
    public float maxAlpha = 3.38f;
    public float maxSpeed = 9.31f;
    public float maxAltitudeRate = 3.34f; // Maximum altitude rate in m/s
    public float MinHeight = 0.5f;

    //must set this
    [Header("Setpoints")]
    // Overwritten in Start with the spawn altitude, so this value never flies.
    public float desired_height = 25.3f;
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
    // Obstacle avoidance for the *pilot's* command: the inward component of the commanded velocity
    // is faded out near an obstacle (see OlfatiSaber.ProjectCommandVelocity). Wired by
    // SwarmAlgorithm where an Olfati-Saber swarm exists; null everywhere else, which is a no-op.
    [HideInInspector] public OlfatiSaber obstacleShield = null;
    // When true, the vertical channel is handed to the swarm: the altitude-hold PD is replaced by a
    // vertical *velocity* loop (symmetric with the horizontal one), so the swarm's vertical
    // acceleration drives the drone instead of being fought by the height setpoint. Set every tick
    // by SwarmAlgorithm; true only while swarming in a non-horizontal plane.
    [HideInInspector] public bool verticalSwarmAuthority = false;
    // Altitude the vertical leash is measured against — SwarmPlaneController.ReferenceAltitude, the
    // wall's own reference rather than any drone's. Set by SwarmAlgorithm.
    [HideInInspector] public float verticalReferenceAltitude = 0f;
    // Metres above the terrain surface this drone may be commanded to; 0 = no ceiling. Pushed by
    // SwarmAlgorithm from SwarmManager.maxHeightAboveTerrain, where it is tuned per scene, so a
    // scene with no swarm (the DJI real-drone ones) leaves it at 0 and is unaffected.
    [HideInInspector] public float maxHeightAboveTerrain = 0f;
    // Last-frame horizontal (XZ) acceleration magnitudes — read by FlightHUD
    [HideInInspector] public float lastUserAccelMag  = 0f;
    [HideInInspector] public float lastSwarmAccelMag = 0f;
    [HideInInspector] public float lastThrustClamped = 0f;

    // PD coefficients for height control
    [Header("Filters & Coefficients")]
    public float HeightKp = 5.0f;
    public float HeightKd = 2.0f;
    public float heightDerivFilterCoeff = 0.5f;

    [Tooltip("Low-pass on the OUTER yaw rate command only — the heading-hold P term is added after " +
             "it, because that term is disturbance rejection and must not be lagged. 1 = off, " +
             "which is the setting the real fleet runs: joystick_controller smooths pitch, roll " +
             "and gimbal pitch (STICK_SMOOTHING_ALPHA) but feeds the yaw stick in raw, because " +
             "heading_hold_rate is what makes smoothing unnecessary. Lower it only for a source of " +
             "stepping yaw commands the heading hold cannot absorb.")]
    public float yawFilterCoefficient = 1.0f;

    [Tooltip("Heading-hold gain (1/s): heading error (rad) -> yaw rate (rad/s). This is the DJI " +
             "flight controller's own loop, which the sim had no counterpart for at all — " +
             "VelocityControl had only a yaw RATE loop, so nothing held an absolute heading and a " +
             "disturbance was pulled back only by AttitudeAlgorithm's outer P, at well over a " +
             "second. The yaw rate loop has unity DC gain (its angularDrag is compensated), so the " +
             "closed-loop time constant is about 1/headingHoldKp. 5 reproduces the response 8 gave " +
             "before that compensation, when the loop only delivered 0.58 of its command: 0.38 s " +
             "rise for a small step with 0.5% overshoot. Useful range 4-6; 8 overshoots 5%, 10 9%.")]
    public float headingHoldKp = 5.0f;

    [Tooltip("Anti-windup, degrees: how far the integrated heading setpoint may lead the MEASURED " +
             "heading. The analogue of the fleet's MAX_TARGET_LEAD_DEG and of " +
             "SwarmPlaneController.maxTargetLeadDeg, one level down — those bound the swarm's " +
             "shared setpoint against the swarm mean, this bounds each drone's own setpoint " +
             "against its own heading, so a rate-saturated turn cannot bank up heading debt it " +
             "keeps paying off after the stick is centred. The back-calculation in FixedUpdate " +
             "already holds a saturated turn's lead near maxYawRate/headingHoldKp (15 deg), so this " +
             "is the hard bound behind it; keep headingHoldKp * this * Deg2Rad >= maxYawRate.")]
    public float maxHeadingHoldErrorDeg = 25.0f;

    public float SwarmAccelFilterCoefficient = 0.3f;
    [Tooltip("Time constant (s) of the velocity → acceleration P-controller. " +
             "Larger = softer velocity response = more angle budget left for swarm corrections. " +
             "Saturation threshold ≈ g × maxPitch × tau.")]
    public float timeConstantAcceleration = 0.75f;

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

    // Cached ground sample for the altitude ceiling. The terrain is static scenery, so the only
    // thing that can change the answer is the drone moving over a different part of it — hence a
    // resample on horizontal movement rather than one per tick. The terrains in the city scenes are
    // flat, so in practice this samples once per drone per flight.
    private const float GroundResampleDistance = 4.0f;
    private bool groundSampleValid = false;
    private float groundSampleHeight = 0.0f;
    private Vector2 groundSampleXZ = Vector2.zero;

    private float targetYawRate = 0.0f;
    private float filteredYawRate = 0.0f;

    // Absolute heading setpoint (radians, in StateFinder.Angles.y's [-pi, pi] space) — the integral
    // of the commanded yaw rate. This is the piece the sim was missing: on the real aircraft the PC
    // sends a yaw RATE and the DJI flight controller holds heading off its own IMU whenever that
    // rate is zero (YawControlMode.ANGULAR_VELOCITY). AttitudeAlgorithm is the analogue of the PC's
    // heading_hold_rate, NOT of the flight controller, so without this there was no heading loop
    // below it at all and anything that knocked the nose off heading came back only at the outer
    // loop's pace — a slow, visible swing of every screen on the ring.
    private float headingSetpoint = 0.0f;
    /// <summary>
    /// The commanded heading: radians, [-pi, pi], StateFinder.Angles.y's convention (Unity yaw,
    /// clockwise from +Z). A lone drone's body yaw follows this (PyUniSharingFast.UpdateBodyYaw).
    /// </summary>
    public float HeadingSetpointRad => headingSetpoint;
    // Last tick's heading-hold diagnostics, for the CSV log.
    private float lastEffectiveYawRate = 0.0f;
    private float lastHeadingError = 0.0f;

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

        // Seed, don't ramp: a heading setpoint starting at 0 would command a turn on the first tick
        // of every scene whose drones do not happen to spawn facing world north.
        headingSetpoint = State.Angles.y;

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
            csvStreamWriter.WriteLine("Time;UserAccelX;UserAccelY;UserAccelZ;SwarmAccelX;SwarmAccelY;SwarmAccelZ;DesiredThetaX;DesiredThetaY;DesiredThetaZ;DesiredOmegaX;DesiredOmegaY;DesiredOmegaZ;DesiredAlphaX;DesiredAlphaY;DesiredAlphaZ;DesiredThrust;DesiredTorqueX;DesiredTorqueY;DesiredTorqueZ;DesiredForceX;DesiredForceY;DesiredForceZ;Yaw;HeadingSetpoint;HeadingError;FilteredYawRate;EffectiveYawRate;TiltHeadingLeak;WorldVerticalRate");
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
            // DroneHealthMonitor parks a dead drone, so whatever setpoint it held before means
            // nothing now. Track the measured heading while dead so the tick it is revived on
            // commands no turn. State.GetState() has already run, so Angles.y is fresh.
            headingSetpoint = State.Angles.y;
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
        bool hasCeiling = TryGetAltitudeCeiling(out float altitudeCeiling);

        float heightRate = userAltitudeRate;

        // Once the FORMATION is as high as it may go, a climb stick has nothing left to command,
        // and leaving it in the sum is not harmless: it is added and clamped away every tick, which
        // cancels the swarm's own vertical term, so a wall flown up into the ceiling pancakes flat
        // against it instead of re-forming below it and drones end up in each other's cells (60 s
        // of full climb stick in plane mode cost 2 of 10 drones to a drone-drone collision at the
        // cap — the same failure the MinHeight floor already produces at the bottom).
        //
        // The test is the WALL's reference altitude, not this drone's own setpoint.
        // SwarmPlaneController integrates the same stick into that reference and clamps it to the
        // same ceiling, so the reference being pinned is exactly "the pilot has asked for more
        // altitude than the formation may have". Testing the drone's own setpoint instead does
        // nothing: the swarm pushes it a tick's worth below the cap and the stick, re-enabled by
        // that very dip, puts it straight back. Horizontal formations keep the plain clamp below —
        // there is no vertical swarm term there, so there is nothing to re-form.
        if (hasCeiling && verticalSwarmAuthority && heightRate > 0.0f
            && verticalReferenceAltitude >= altitudeCeiling - 0.01f)
        {
            heightRate = 0.0f;
        }

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

        // Altitude ceiling: a hard cap on the setpoint, measured from the terrain under the drone
        // (SwarmManager.maxHeightAboveTerrain). It is the symmetric counterpart of MinHeight below,
        // and it is applied to the setpoint rather than to the stick so that nothing winds up:
        // desired_height is clamped in place, so pushing the stick back down moves the drone on the
        // first tick instead of first paying off however long the pilot held climb.
        if (hasCeiling)
        {
            desired_height = Mathf.Min(desired_height, altitudeCeiling);
        }

        // The floor wins if the two ever cross — a ceiling below MinHeight is a misconfiguration,
        // and flying into the ground is the worse way to resolve it.
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

        // Resolve the command into world frame first, so the obstacle shield below sees it in the
        // same frame as the obstacle normals. The body-frame branch used to take its error in body
        // frame and rotate afterwards; that is algebraically the same thing, and the y component is
        // zeroed below either way.
        Vector3 worldCommand = userCommandInWorldFrame
            // Command is in world frame, expressed relative to commandReferenceYaw (0 = fixed
            // world axes); rotate it into world space.
            ? Quaternion.Euler(0f, commandReferenceYaw, 0f) * userVelCommand
            // Command is in body frame (relative to this drone's heading).
            : transform.TransformDirection(userVelCommand);

        // Fade out whatever part of the command heads into a nearby obstacle. This has to happen to
        // the *command* rather than being fought with force: the tilt clamp below applies to the
        // pilot + swarm sum while this P-loop ahead of it is unbounded, so full stick at a wall
        // demands several times the actuator limit inward and simply outvotes any repulsion the
        // drone could produce. Tangential and outward components pass through untouched, so flying
        // around or away is never denied.
        if (obstacleShield != null)
            worldCommand = obstacleShield.ProjectCommandVelocity(worldCommand);

        Vector3 worldUserVelError = transform.TransformDirection(bodyVelocity) - worldCommand;

        Vector3 worldUserAccel = worldUserVelError * -1.0f / timeConstantAcceleration;
        // Force any "ghost" y component coming from the drone's tilt to zero (altitude handled separately).
        worldUserAccel.y = 0f;

        // worldFilteredSwarmAccel was filtered above, ahead of the height loop that consumes it.
        Vector3 desiredAcceleration = worldUserAccel + worldFilteredSwarmAccel;

        // Horizontal magnitudes for the HUD: how much of the shared tilt budget each source is
        // asking for. Both are pre-clamp, so a user figure well above g*tan(maxPitch) is the
        // signature of the pilot loop saturating and crowding the swarm out of the sum.
        lastUserAccelMag = new Vector2(worldUserAccel.x, worldUserAccel.z).magnitude;
        lastSwarmAccelMag = new Vector2(worldFilteredSwarmAccel.x, worldFilteredSwarmAccel.z).magnitude;

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
        else if (hasCeiling && State.Altitude >= altitudeCeiling && desiredAcceleration.y > 0f)
        {
            // Above the ceiling the swarm must not push the drone higher either. In a horizontal
            // formation its vertical force goes straight into thrust (see desiredThrust below),
            // where the clamped setpoint does not bound it — only the height PD opposes it, and a
            // sustained upward force from the lattice would simply sit on top of the cap. The
            // downward half is left alone, so a drone held at the ceiling still settles back into
            // the formation. In plane mode the branch above has already zeroed it, because there
            // the swarm drives the setpoint instead, which the ceiling clamps.
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

        // Add the yaw rate contributions from user input and the autonomous control. This pair is
        // the OUTER loop, and it is the exact counterpart of the real fleet's PC-side command:
        // desiredYawRate is the stick feed-forward (ff_rate) and attitude_control_yaw is the outer
        // heading P (the fleet's KP_YAW * err, at the sim's own gain) that AttitudeAlgorithm computes.
        targetYawRate = desiredYawRate + attitude_control_yaw;

        // Command prefilter on the outer rate, and nothing downstream of it. It exists to keep a
        // stepping yaw command from stepping the torque; it must NOT lag the heading-hold P term
        // below, which is disturbance rejection and wants full bandwidth.
        filteredYawRate = filteredYawRate * (1.0f - yawFilterCoefficient) + targetYawRate * yawFilterCoefficient;

        // --- Heading hold: the flight controller's own loop --------------------------------------
        // Anti-windup, hard bound; the back-calculation below does the routine work. Unconditional,
        // unlike the real fleet's version (integrate_target_heading
        // gates on ff_rate != 0), and that difference is deliberate rather than an oversight: the
        // fleet gates because KP_YAW * 25 deg = 20 deg/s sits BELOW its 40 deg/s rate clamp, so a
        // pinned setpoint would cap its correction below the actuator limit. Here
        // headingHoldKp * maxHeadingHoldErrorDeg is well above maxYawRate, so the rate clamp
        // binds first and the lead clamp costs no authority in any regime — while closing the
        // windup hole the gate leaves open for a drone pinned by an obstacle. The invariant that
        // makes it free is headingHoldKp * maxHeadingHoldErrorDeg * Deg2Rad >= maxYawRate; if
        // either number is lowered past that, put the fleet's ff != 0 gate back.
        float maxLead = maxHeadingHoldErrorDeg * Mathf.Deg2Rad;
        float headingError = WrapAngle(headingSetpoint - State.Angles.y);
        if (headingError > maxLead)
        {
            headingSetpoint = WrapAngle(State.Angles.y + maxLead);
            headingError = maxLead;
        }
        else if (headingError < -maxLead)
        {
            headingSetpoint = WrapAngle(State.Angles.y - maxLead);
            headingError = -maxLead;
        }

        // Feed-forward plus P, exactly as heading_hold_rate. No deadband: the fleet's
        // YAW_ERR_DEADBAND_DEG exists because a magnetic compass jitters a degree or two, and
        // StateFinder's heading is exact (the attitude noise it injects is on x and z only) — the
        // same argument AttitudeAlgorithm.ApplyPlaneModeAttitude already records.
        float unclampedYawRate = filteredYawRate + headingHoldKp * headingError;

        // The actuator limit applies to the SUM, not to the feed-forward alone — as it does on the
        // fleet, where heading_hold_rate clamps ff + P. Clamping the feed-forward on its own (which
        // is where this line used to sit) let the total command exceed the limit whenever the hold
        // was correcting.
        float effectiveYawRate = Mathf.Clamp(unclampedYawRate, -maxYawRate, maxYawRate);
        lastEffectiveYawRate = effectiveYawRate;
        lastHeadingError = headingError;

        // Advance the setpoint by the rate we are actually asking the aircraft to fly — the
        // FILTERED one, not the raw sum (integrating the raw rate while feeding the filtered one
        // forward leaves the setpoint permanently ahead by tau_EMA * rate, all of it owed when the
        // stick centres) — and only by as much of it as survived the clamp above.
        //
        // That second part is back-calculation, and it is what lets AttitudeAlgorithm command a turn
        // briskly. Integrating the full rate while the clamp holds the aircraft at maxYawRate walks
        // the setpoint ahead of the nose — through the spin-up, and for as long as the outer command
        // exceeds the limit — and every degree of that lead is flown past the target on arrival:
        // 12-19 deg of overshoot, in a replica of this loop, at outer gains of 2-3. Subtracting what
        // the clamp removed holds the lead near (maxYawRate - rate) / headingHoldKp instead, so it
        // closes as the aircraft reaches speed and there is nothing left to overshoot with.
        //
        // The correction may only slow the advance, never reverse or speed it up. A disturbance
        // large enough to saturate the P term would otherwise drag the setpoint toward wherever it
        // knocked the nose — the setpoint would forget the heading it is meant to hold — whereas
        // this way a centred stick still returns the aircraft all the way.
        float advanceRate = Mathf.Clamp(filteredYawRate - (unclampedYawRate - effectiveYawRate),
                                        Mathf.Min(0.0f, filteredYawRate), Mathf.Max(0.0f, filteredYawRate));
        headingSetpoint = WrapAngle(headingSetpoint + advanceRate * Time.fixedDeltaTime);

        // effectiveYawRate is a heading rate about the WORLD vertical, but desiredOmega is a
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
        // effectiveYawRate exactly (Dot(desiredOmega, upBody) == effectiveYawRate), instead of
        // effectiveYawRate plus the tilt-correction leak.
        //
        // That identity holds in COMMANDED-rate space. It used not to survive into achieved-rate
        // space, because the torque below was pre-multiplied by the inertia tensor and then applied
        // with ForceMode.Acceleration, which ignores it: pitch/roll came out at 0.3893 of the
        // command and yaw at 0.7688, so the commanded vector was rotated off upBody by the physics
        // and a tilt change leaked heading however carefully this line was written. The single
        // timeConstantAlphaRate, the ForceMode.Force below and the upBody-aligned clamp together
        // make the achieved rate an isotropic copy of the commanded one, so the cancellation now
        // survives. What remains is the rate loop's tracking LAG, which no algebraic cancellation
        // can remove — that is what the heading hold above absorbs, model-free, exactly as the real
        // flight controller does without knowing anything about the aircraft's inertia.
        float tiltHeadingLeak = desiredOmega.x * upBody.x + desiredOmega.z * upBody.z;
        desiredOmega += (effectiveYawRate - tiltHeadingLeak) * upBody;

        Vector3 omegaError = State.AngularVelocityVector - desiredOmega;

        Vector3 desiredAlpha = omegaError * -1.0f / timeConstantAlphaRate;

        // Clamp in the frame the command was built in, not in body axes. desiredOmega is
        // (tilt correction) + (yaw rate about upBody), and the clamp this replaces scaled the
        // (x, z) pair circularly while clamping y independently — which rescales those two parts by
        // different factors and destroys the cancellation above exactly when tilt demand is highest.
        // That is not an edge case: a full-stick step commands maxTilt/timeConstantOmegaXYRate of
        // body rate, far past what maxAlpha can deliver, so the pitch/roll channel sits at its clamp
        // for the best part of a second on every stick movement.
        //
        // Splitting about upBody keeps the world-vertical component intact through tilt saturation,
        // and reduces to the old circular (x, z) clamp when the drone is level. The circular form is
        // kept for the tilt part for its original reason: a per-axis clamp is a square envelope
        // (~41% larger on the diagonal), so a drone whose body axes align with the required
        // tilt-change direction would saturate at maxAlpha while one at 45 degrees to it got
        // maxAlpha*sqrt(2), making the angular response heading-dependent under a world-frame
        // command. Both parts saturating can put the total magnitude at sqrt(2)*maxAlpha — the same
        // latitude the old clamp allowed, and bounded, so it is left alone.
        float alphaYaw = Vector3.Dot(desiredAlpha, upBody);
        Vector3 alphaTilt = desiredAlpha - alphaYaw * upBody;
        if (alphaTilt.magnitude > maxAlpha)
            alphaTilt = alphaTilt.normalized * maxAlpha;

        // The yaw part gets its own law, because the shared one could not fly maxYawRate at all.
        // PhysX damps angular velocity by (1 - angularDrag * dt) every step, AFTER adding the
        // torque's velocity change (measured in batchmode), and at the prefab's drag of 5 holding
        // 75 deg/s takes 7.3 rad/s^2 of torque against a maxAlpha of 3.38: the airframe topped out at
        // 34.9 deg/s, and below saturation delivered only 0.58 of its command. That drag models
        // nothing a DJI flight controller's yaw rate loop fails to overcome, so it is cancelled here
        // and maxAlpha bounds the NET acceleration — the aircraft's real yaw authority. The gain
        // 1/tau + drag is the closed-loop pole the droopy loop already had, so the bandwidth is
        // unchanged and only the droop goes (headingHoldKp was rescaled to match).
        //
        // This does not reopen the tilt leak "one time constant for all three axes" closed. That
        // rule is about BODY axes, whose yaw command is a projection spread over all three. The
        // world-vertical component of angular velocity is decoupled from the tilt part outright:
        // damping is isotropic, alphaTilt is perpendicular to upBody, so its rate evolves only
        // through alphaYaw and its own damping — confirmed to five decimals on a tilted body under
        // simultaneous tilt torque. Unlike the analytic leak fix CLAUDE.md records as rejected,
        // nothing is hard-coded: drag and step are read live, and it stays exact under saturation.
        // No model-free loop could have done this job: the heading hold removes droop, but it
        // cannot buy torque the clamp refuses.
        float worldVerticalRate = Vector3.Dot(State.AngularVelocityVector, upBody);
        float drag = rb.angularDrag;
        float dampingRetention = Mathf.Max(1.0f - drag * Time.fixedDeltaTime, 0.1f);
        float alphaYawNet = Mathf.Clamp((effectiveYawRate - worldVerticalRate) * (1.0f / timeConstantAlphaRate + drag),
                                        -maxAlpha, maxAlpha);
        alphaYaw = (alphaYawNet + drag * worldVerticalRate) / dampingRetention;
        Vector3 desiredAlphaClamped = alphaTilt + alphaYaw * upBody;

        // float desiredThrust = (gravity + desiredAcceleration.y) / (Mathf.Cos(State.Angles.z) * Mathf.Cos(State.Angles.x));
        float desiredThrust = (gravity + altitudeCommand + desiredAcceleration.y) / (Mathf.Cos(State.Angles.z) * Mathf.Cos(State.Angles.x));
        float  desiredThrustClamped = Mathf.Min(desiredThrust, 2.7f * gravity);
        desiredThrustClamped = Mathf.Max(desiredThrustClamped, 0.0f);
        lastThrustClamped = desiredThrustClamped;

        Vector3 desiredTorque = Vector3.Scale(desiredAlphaClamped, State.Inertia);
        Vector3 desiredForce = new Vector3(0.0f, desiredThrustClamped * State.Mass, 0.0f);

        // ForceMode.Force, not Acceleration. desiredTorque is already I*alpha, and Acceleration
        // *ignores* the inertia tensor — so the pre-multiply was never cancelled and the achieved
        // angular acceleration was alpha*I: pitch/roll at 0.3893 of the command, yaw at 0.7688.
        // That anisotropy is what broke the tiltHeadingLeak cancellation above (exact in commanded
        // space, impossible once the achieved rate is an anisotropically scaled copy) and it also
        // made the "rotation-invariant" alpha envelope 1.97x larger on yaw than on tilt.
        // timeConstantAlphaRate and maxAlpha are scaled to reproduce the old pitch/roll response
        // exactly, so this is a fidelity fix rather than a retune.
        //
        // The linear line below is the same bug and is deliberately NOT changed here: desiredForce
        // is thrust*Mass applied as an acceleration, so the achieved vertical acceleration is three
        // times the computed thrust. The height PD absorbs it by sitting at an offset setpoint, so
        // fixing it means retuning HeightKp/HeightKd and re-reading the 2.7g clamp — its own change,
        // with its own altitude-hold comparison.
        rb.AddRelativeTorque(desiredTorque, ForceMode.Force);
        rb.AddRelativeForce(desiredForce, ForceMode.Acceleration);

        //prop transforms
        PropFL.transform.Rotate(Vector3.forward * Time.deltaTime * desiredThrustClamped * speedScale);
        PropFR.transform.Rotate(Vector3.forward * Time.deltaTime * desiredThrustClamped * speedScale);
        PropRR.transform.Rotate(Vector3.forward * Time.deltaTime * desiredThrustClamped * speedScale);
        PropRL.transform.Rotate(Vector3.forward * Time.deltaTime * desiredThrustClamped * speedScale);

        // Update previous values
        previousHeightError = currentHeightError;

        if (logToCSV)
            dumpToCSVFile(worldUserAccel, worldFilteredSwarmAccel, desiredTheta, desiredOmega, desiredAlpha, desiredAlphaClamped, desiredThrust, desiredThrustClamped, desiredTorque, desiredForce,
                          tiltHeadingLeak, Vector3.Dot(State.AngularVelocityVector, upBody));
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
        Vector3 desiredForce,
        float tiltHeadingLeak,
        float worldVerticalRate)
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
                          $"{desiredForce.x};{desiredForce.y};{desiredForce.z};" +
                          // Appended, never inserted, so existing parsers keep working. These six
                          // are what the yaw-stability verification reads: WorldVerticalRate is the
                          // disturbance itself, and HeadingSetpoint must not move at all through a
                          // pure-translation manoeuvre.
                          $"{State.Angles.y};{headingSetpoint};{lastHeadingError};" +
                          $"{filteredYawRate};{lastEffectiveYawRate};" +
                          $"{tiltHeadingLeak};{worldVerticalRate}";
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

        // From the transform, NOT from State.Angles.y: StateFinder.ResetToPos has just written
        // newRot.eulerAngles into Angles, which is DEGREES in [0, 360) where the rest of the
        // codebase reads Angles as radians in [-pi, pi]. That only survives until the next
        // GetState(), but this runs before it.
        headingSetpoint = HeadingFromTransform();

        enabled = true;
    }

    /// <summary>
    /// This drone's heading in StateFinder's convention — radians, [-pi, pi]. Read straight off the
    /// transform rather than taken from State.Angles.y because the reset path needs it before the
    /// next GetState(), and at that instant State.Angles holds euler degrees (see
    /// StateFinder.ResetToPos). Projects forward onto the horizontal plane for the same reason
    /// StateFinder and FPVCameraScript do: reading eulerAngles.y directly drifts with tilt.
    /// </summary>
    private float HeadingFromTransform()
    {
        Vector3 fwd = transform.forward;
        fwd.y = 0f;
        float rawYaw = fwd.sqrMagnitude > 1e-6f
            ? Quaternion.LookRotation(fwd, Vector3.up).eulerAngles.y
            : transform.eulerAngles.y;
        return ((rawYaw > 180f) ? rawYaw - 360f : rawYaw) * Mathf.Deg2Rad;
    }

    /// <summary>
    /// Wraps an angle to [-pi, pi]. Same construction as AttitudeAlgorithm's and
    /// SwarmPlaneController's — the heading setpoint lives in StateFinder.Angles.y's space and has
    /// to cross the seam the same way they do.
    /// </summary>
    private static float WrapAngle(float angle)
    {
        while (angle > Mathf.PI)  angle -= 2f * Mathf.PI;
        while (angle < -Mathf.PI) angle += 2f * Mathf.PI;
        return angle;
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

    /// <summary>
    /// World altitude this drone may be commanded up to: the terrain surface under it plus
    /// <see cref="maxHeightAboveTerrain"/>. False when there is no ceiling to apply — the knob is
    /// off, or the scene has no Terrain to measure from.
    ///
    /// <para>The ground sample is cached and refreshed only once the drone has moved
    /// <see cref="GroundResampleDistance"/> horizontally, because terrain is static scenery and the
    /// answer cannot change while the drone hovers. On the city scenes' flat terrain it never
    /// changes at all, which is what keeps a per-drone per-tick ceiling free.</para>
    /// </summary>
    private bool TryGetAltitudeCeiling(out float ceiling)
    {
        ceiling = 0f;
        if (maxHeightAboveTerrain <= 0f) return false;

        Vector3 position = transform.position;
        Vector2 xz = new Vector2(position.x, position.z);

        if (!groundSampleValid ||
            (xz - groundSampleXZ).sqrMagnitude > GroundResampleDistance * GroundResampleDistance)
        {
            if (!TerrainHeightSampler.TryGetHeight(position, out groundSampleHeight))
            {
                // No Terrain in this scene: nothing to measure a height above, so no ceiling.
                groundSampleValid = false;
                return false;
            }

            groundSampleXZ = xz;
            groundSampleValid = true;
        }

        ceiling = groundSampleHeight + maxHeightAboveTerrain;
        return true;
    }
}
