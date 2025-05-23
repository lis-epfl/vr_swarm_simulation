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
    private Vector3 filteredVelocity = Vector3.zero; 
    public float filterCoefficient = 0.1f; 

    public SwarmManager.SwarmAlgorithm currentAlgorithm;

    //must set this
    public float initial_height = 14.0f;

    private bool wait = false;
    private bool flag = true;
    private bool takeoff = false;


    private float speedScale = 500.0f;

    public Quaternion targetOrientation = Quaternion.identity; // Add this line, initialize to identity

    // Use this for initialization
    void Start () {
        state.GetState ();
        Rigidbody rb = GetComponent<Rigidbody> ();
        Vector3 desiredForce = new Vector3 (0.0f, gravity * state.Mass, 0.0f);
        rb.AddForce (desiredForce, ForceMode.Acceleration);

        initial_height = state.Altitude + 4.0f;
        desired_height = state.Altitude + 4.0f;
    }

    // Update is called once per frame
    void FixedUpdate () {
        state.GetState ();
        

        Vector3 desiredTheta;
        Vector3 desiredOmega;
        Vector3 desiredVelocity; // Base desired velocity (user input + height control)
        Vector3 swarmVelocity;   // Velocity component from swarm algorithm
        Vector3 totalTargetVelocityWorld; // Final target velocity

        // Calculate height error and the corresponding corrective velocity
        float heightError = state.Altitude - desired_height;
        float heightControlVelocity = -1.0f * heightError / time_constant_z_velocity;

        // Set the base desired velocity
        if (currentAlgorithm == SwarmManager.SwarmAlgorithm.REYNOLDS) 
        {
            // For Reynolds: Use user input for horizontal (desired_vx, desired_vy)
            // AND use the height control velocity for the base vertical component.
            desiredVelocity = new Vector3(desired_vx, heightControlVelocity, desired_vy); // Restore height control here
        } 
        else // Olfati-Saber or other algorithms
        {
            // For Olfati-Saber: Use height control for vertical base, zero for horizontal base.
            // Horizontal movement comes entirely from swarm_vx/vz in this case.
            if (heightError < 0.01f)
            {
                takeoff = true;
            }

            if (takeoff == false)
            {
                // If the drone is too low, set the desired velocity to zero
                // This prevents the drone from trying to go down when it's already low enough
                desiredVelocity = new Vector3(0.0f, heightControlVelocity, 0.0f);
            }
            else 
            {
                // If the drone is at a safe height, allow it to move horizontally
                desiredVelocity = new Vector3(0.0f, heightControlVelocity, 0.0f);
            }
            // desiredVelocity = new Vector3(0.0f, heightControlVelocity, 0.0f);
        }
        
        // Get the velocity contribution from the active swarm algorithm (includes swarm_vy)
        swarmVelocity = new Vector3 (swarm_vx, swarm_vy, swarm_vz); 

        // Combine the base desired velocity (with height control) and the swarm algorithm's velocity
        totalTargetVelocityWorld = desiredVelocity + swarmVelocity; 
        // Now, totalTargetVelocityWorld.y = heightControlVelocity + swarm_vy
                // Get the name of the drone
        string droneName = transform.parent.name;
        // get drone name
        // if (droneName == "Drone 0")
        // {
        //     Debug.Log("Desired Velocity: " + desiredVelocity);
        //     Debug.Log("Swarm Velocity: " + swarmVelocity);
        //     Debug.Log("Total Target Velocity (World): " + totalTargetVelocityWorld);
        // }

        // Transform the final target velocity from the world frame to the body frame
        Vector3 totalTargetVelocity = transform.InverseTransformDirection(totalTargetVelocityWorld);

        filteredVelocity = filteredVelocity * (1.0f - filterCoefficient) + totalTargetVelocity * filterCoefficient;         
 


        // --- The rest of the FixedUpdate remains the same ---
        Vector3 velocityError = state.VelocityVector - filteredVelocity;

//         // NOTE: I'm using stupid vector order (sideways, up, forward) at the end
        
//         Vector3 desiredTheta;
//         Vector3 desiredOmega;
//         Vector3 desiredVelocity;

//         float heightError = state.Altitude - desired_height;

//         // If reynolds algorithm is selected add the velocity commands from the user, otherwise handled in Olfati-Saber Script
//         if (currentAlgorithm == SwarmManager.SwarmAlgorithm.REYNOLDS) 
//         {
//             desiredVelocity = new Vector3(desired_vx, -1.0f * heightError / time_constant_z_velocity, desired_vy);
//         } 
//         else
//         {
//             desiredVelocity = new Vector3(0.0f, -1.0f * heightError / time_constant_z_velocity, 0.0f);
//         }
        
//         Vector3 swarmVelocity = new Vector3 (swarm_vx, 0.0f, swarm_vz);

//         // NOTE: In world frame y is up

//         Vector3 totalTargetVelocityWorld = desiredVelocity + swarmVelocity;


//         // Transform the desired velocity from the world frame to the body frame
//         Vector3 totalTargetVelocity = transform.InverseTransformDirection(totalTargetVelocityWorld);


//         // Get the name of the drone
//         string droneName = transform.parent.name;


        

//         Vector3 velocityError = state.VelocityVector - totalTargetVelocity;


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
        desiredThrust = Mathf.Min (desiredThrust, 2.7f * gravity);
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


        // --- Add logic here to use targetOrientation ---
        // Example: Calculate torque needed to reach targetOrientation
        // This is a simplified example using proportional control for torque.
        // You'll likely need a more robust PD or PID controller.

        Quaternion currentRotation = transform.rotation;
        Quaternion rotationDifference = targetOrientation * Quaternion.Inverse(currentRotation);

        rotationDifference.ToAngleAxis(out float angleDegrees, out Vector3 axis);
        Vector3 angularVelocityError = axis.normalized * (angleDegrees * Mathf.Deg2Rad); // Error in radians

        // Reduce angular velocity error over time (damping) - requires Rigidbody access (rb)
        // Vector3 currentAngularVelocity = rb.angularVelocity; 
        // Vector3 torque = angularVelocityError * attitudeCorrectionTorque - currentAngularVelocity * attitudeDamping; // Example PD control
        
        // Simplified: Apply torque proportional to the angular error
        // You'll need to define and tune attitudeCorrectionTorque and potentially attitudeDamping
        // float attitudeCorrectionTorque = 5.0f; // Example value - TUNE THIS
        // float attitudeDamping = 0.5f; // Example value - TUNE THIS
        // Rigidbody rb = GetComponent<Rigidbody>(); // Get the Rigidbody if not already cached
        // if (rb != null) {
        //     Vector3 torque = angularVelocityError * attitudeCorrectionTorque; 
        //     rb.AddTorque(torque);
        // }
        
        // --- End of attitude control logic example ---
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
