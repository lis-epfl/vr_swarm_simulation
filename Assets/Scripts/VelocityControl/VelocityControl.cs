using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class VelocityControl : MonoBehaviour {

    public StateFinder state;

    public GameObject propFL;
    public GameObject propFR;
    public GameObject propRR;
    public GameObject propRL;

    private float gravity = 9.81f;
    private float time_constant_z_velocity = 1.0f; // Normal-person coordinates
    private float time_constant_acceleration = 0.5f;
    private float time_constant_omega_xy_rate = 0.1f; // Normal-person coordinates (roll/pitch)
    private float time_constant_alpha_xy_rate = 0.05f; // Normal-person coordinates (roll/pitch)
    private float time_constant_alpha_z_rate = 0.05f; // Normal-person coordinates (yaw)

    private float max_pitch = 0.175f; // 10 Degrees in radians, otherwise small-angle approximation dies 
    private float max_roll = 0.175f; // 10 Degrees in radians, otherwise small-angle approximation dies
    private float max_alpha = 10.0f;
    //must set this
    public float desired_height = 4.0f;
    public float desired_vx = 0.0f;
    public float desired_vy = 0.0f;
    public float desired_yaw = 0.0f;
    public float attitude_control_yaw = 0.0f;

    private float targetYawRate = 0.0f;
    private float filteredYawRate = 0.0f;
    public float yawFilterCoefficient = 0.2f;

    public float swarm_vx = 0.0f;
    public float swarm_vy = 0.0f;
    public float swarm_vz = 0.0f;

    //must set this
    public float initial_height = 4.0f;

    private bool wait = false;
    private bool flag = true;

    private float speedScale = 500.0f;

    // Use this for initialization
    void Start () {
        state.GetState ();
        Rigidbody rb = GetComponent<Rigidbody> ();
        Vector3 desiredForce = new Vector3 (0.0f, gravity * state.Mass, 0.0f);
        rb.AddForce (desiredForce, ForceMode.Acceleration);
    }

    // Update is called once per frame
    void FixedUpdate () {
        state.GetState ();
        
        // NOTE: I'm using stupid vector order (sideways, up, forward) at the end
        
        Vector3 desiredTheta;
        Vector3 desiredOmega;

        float heightError = state.Altitude - desired_height;

        Vector3 desiredVelocity = new Vector3(0.0f, -1.0f * heightError / time_constant_z_velocity, 0.0f);
        Vector3 swarmVelocity = new Vector3 (swarm_vx, 0.0f, swarm_vz);

        // NOTE: In world frame y is up

        Vector3 totalTargetVelocityWorld = desiredVelocity + swarmVelocity;


        // Transform the desired velocity from the world frame to the body frame
        Vector3 totalTargetVelocity = transform.InverseTransformDirection(totalTargetVelocityWorld);


        // Get the name of the drone
        string droneName = transform.parent.name;


        

        Vector3 velocityError = state.VelocityVector - totalTargetVelocity;

        Vector3 desiredAcceleration = velocityError * -1.0f / time_constant_acceleration;

        desiredTheta = new Vector3 (desiredAcceleration.z / gravity, 0.0f, -desiredAcceleration.x / gravity);
        if (desiredTheta.x > max_pitch) {
            desiredTheta.x = max_pitch;
        } else if (desiredTheta.x < -1.0f * max_pitch) {
            desiredTheta.x = -1.0f * max_pitch;
        }
        if (desiredTheta.z > max_roll) {
            desiredTheta.z = max_roll;
        } else if (desiredTheta.z < -1.0f * max_roll) {
            desiredTheta.z = -1.0f * max_roll;
        }

        Vector3 thetaError = state.Angles - desiredTheta;

        desiredOmega = thetaError * -1.0f / time_constant_omega_xy_rate;

        // Add the yaw rate contributions from user input and the autonomous control
        targetYawRate = desired_yaw + attitude_control_yaw;

        // Apply the low-pass filter to reduce oscillations in yaw control
        filteredYawRate = filteredYawRate * (1.0f - yawFilterCoefficient) + targetYawRate * yawFilterCoefficient;

        // Use the filtered yaw rate for further calculations
        desiredOmega.y = filteredYawRate;

        Vector3 omegaError = state.AngularVelocityVector - desiredOmega;

        Vector3 desiredAlpha = Vector3.Scale(omegaError, new Vector3(-1.0f/time_constant_alpha_xy_rate, -1.0f/time_constant_alpha_z_rate, -1.0f/time_constant_alpha_xy_rate));
        desiredAlpha = Vector3.Min (desiredAlpha, Vector3.one * max_alpha);
        desiredAlpha = Vector3.Max (desiredAlpha, Vector3.one * max_alpha * -1.0f);

        float desiredThrust = (gravity + desiredAcceleration.y) / (Mathf.Cos (state.Angles.z) * Mathf.Cos (state.Angles.x));
        desiredThrust = Mathf.Min (desiredThrust, 2.0f * gravity);
        desiredThrust = Mathf.Max (desiredThrust, 0.0f);

        Vector3 desiredTorque = Vector3.Scale (desiredAlpha, state.Inertia);
        Vector3 desiredForce = new Vector3 (0.0f, desiredThrust * state.Mass, 0.0f);

        Rigidbody rb = GetComponent<Rigidbody>();

        rb.AddRelativeTorque (desiredTorque, ForceMode.Acceleration);
        rb.AddRelativeForce (desiredForce , ForceMode.Acceleration);

        //prop transforms
        propFL.transform.Rotate(Vector3.forward * Time.deltaTime * desiredThrust * speedScale);
        propFR.transform.Rotate(Vector3.forward * Time.deltaTime * desiredThrust * speedScale);
        propRR.transform.Rotate(Vector3.forward * Time.deltaTime * desiredThrust * speedScale);
        propRL.transform.Rotate(Vector3.forward * Time.deltaTime * desiredThrust * speedScale);

    }

    public void Reset() {

        state.VelocityVector = Vector3.zero;
        state.AngularVelocityVector = Vector3.zero;

        desired_vx = 0.0f;
        desired_vy = 0.0f;
        desired_yaw = 0.0f;
        desired_height = initial_height;

        state.Reset ();
    
        enabled = true;
    }

    IEnumerator Waiting(float time) {
        wait = true;
        yield return new WaitForSeconds(time);
        wait = false;
    }
}
