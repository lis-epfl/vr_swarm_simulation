using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class VelocityControl : MonoBehaviour
{

    public StateFinder State;

    public GameObject PropFL;
    public GameObject PropFR;
    public GameObject PropRR;
    public GameObject PropRL;

    private float gravity = 9.81f;
    private float timeConstantVelocityZ = 1.0f; // Normal-person coordinates
    private float timeConstantAcceleration = 0.5f;
    private float timeConstantOmegaXYRate = 0.1f; // Normal-person coordinates (roll/pitch)
    private float timeConstantAlphaXYRate = 0.05f; // Normal-person coordinates (roll/pitch)
    private float timeConstantAlphaZRate = 0.05f; // Normal-person coordinates (yaw)

    private float maxPitch = 0.175f; // 10 Degrees in radians, otherwise small-angle approximation dies 
    private float maxRoll = 0.175f; // 10 Degrees in radians, otherwise small-angle approximation dies
    private float maxYawRate = 0.5f;
    private float maxAlpha = 10.0f;
    private float maxSpeed = 5.0f;

    //must set this
    public float desired_height = 4.0f;
    public float desired_vx = 0.0f;
    public float desired_vy = 0.0f;
    public float desiredYawRate = 0.0f;
    public float attitude_control_yaw = 0.0f;

    // PD coeff for height control
    public float HeightKp = 2.0f;
    public float HeightKd = 1.0f;
    public float HeightKi = 0.1f;
    private float previousHeightError = 0.0f;
    private float cumulativeHeightError = 0.0f;

    private float targetYawRate = 0.0f;
    private float filteredYawRate = 0.0f;
    public float yawFilterCoefficient = 0.15f;

    public float swarm_vx = 0.0f;
    public float swarm_vy = 0.0f;
    public float swarm_vz = 0.0f;

    public SwarmManager.SwarmAlgorithm currentAlgorithm;

    //must set this
    public float initial_height = 14.0f;

    private bool wait = false;
    private bool flag = true;

    private float speedScale = 500.0f;

    private Vector3 filteredVelocity = Vector3.zero;
    private float filterCoefficient = 0.01f;
    
    // Use this for initialization
    void Start() {
        State.GetState ();
        Rigidbody rb = GetComponent<Rigidbody> ();
        Vector3 desiredForce = new Vector3 (0.0f, gravity * State.Mass, 0.0f);
        rb.AddForce (desiredForce, ForceMode.Acceleration);

        initial_height = State.Altitude + 4.0f;
        desired_height = State.Altitude + 4.0f;
    }

    // Update is called once per frame
    void FixedUpdate() {
        State.GetState ();
        
        // NOTE: I'm using stupid vector order (sideways, up, forward) at the end

        Vector3 desiredTheta;
        Vector3 desiredOmega;
        Vector3 desiredVelocity;
        Vector3 swarmVelocity;

        desired_height = desired_height + swarm_vy * Time.deltaTime;

        float currentHeightError = desired_height - State.Altitude;

        // Low pass filter for height control
        // heightError = heightError * (1.0f - filterCoefficient) + (State.Altitude - initial_height) * filterCoefficient;
        float heightErrorDerivative = (currentHeightError - previousHeightError) / Time.deltaTime;
        cumulativeHeightError += currentHeightError * Time.deltaTime;
        float altitudeCommand = HeightKp * currentHeightError + HeightKd * heightErrorDerivative + HeightKi * cumulativeHeightError;

        // Get the velocity commands from Reynolds and add the user commands
        if (currentAlgorithm == SwarmManager.SwarmAlgorithm.REYNOLDS)
        {
            desiredVelocity = new Vector3(desired_vx, -1.0f * currentHeightError / timeConstantVelocityZ, desired_vy);
            swarmVelocity = new Vector3(swarm_vx, 0.0f, swarm_vy);
        }
        // Get the velocity commands from Olfati-Saber, user commands are already included
        else if (currentAlgorithm == SwarmManager.SwarmAlgorithm.OLFATI_SABER)
        {
            //desiredVelocity = new Vector3(0.0f, -1.0f * currentHeightError / timeConstantVelocityZ, 0.0f);
            desiredVelocity = Vector3.zero;
            swarmVelocity = new Vector3(swarm_vx, 0.0f, swarm_vz);
        }
        else
        {
            desiredVelocity = new Vector3(desired_vx, -1.0f * currentHeightError / timeConstantVelocityZ, desired_vy);
            swarmVelocity = new Vector3(0.0f, 0.0f, 0.0f);
        }



        // NOTE: In world frame y is up

        Vector3 totalTargetVelocityWorld = desiredVelocity + swarmVelocity;


        // Transform the desired velocity from the world frame to the body frame
        Vector3 totalTargetVelocity = transform.InverseTransformDirection(totalTargetVelocityWorld);


        // Apply the low-pass filter to reduce oscillations in velocity control
        filteredVelocity = filteredVelocity * (1.0f - filterCoefficient) + totalTargetVelocity * filterCoefficient;

        Vector3 velocityError = State.VelocityVector - filteredVelocity;

        Vector3 desiredAcceleration = velocityError * -1.0f / timeConstantAcceleration;

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
        float desiredThrust = (gravity + altitudeCommand) / (Mathf.Cos(State.Angles.z) * Mathf.Cos(State.Angles.x));
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

        State.VelocityVector = Vector3.zero;
        State.AngularVelocityVector = Vector3.zero;

        desired_vx = 0.0f;
        desired_vy = 0.0f;
        desiredYawRate = 0.0f;
        desired_height = initial_height;

        State.Reset();

        enabled = true;
    }

    IEnumerator Waiting(float time)
    {
        wait = true;
        yield return new WaitForSeconds(time);
        wait = false;
    }

    // Return max speed
    public float GetMaxSpeed()
    {
        return maxSpeed;

    }
}
