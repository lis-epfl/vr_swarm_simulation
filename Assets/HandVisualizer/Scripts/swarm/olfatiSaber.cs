using System.Collections;
using System.Collections.Generic;
using System.Security;
using UnityEngine;
using System.Net;
using System.Net.Sockets;
using System.Text;

// Data structure for Olfati-Saber parameters to be sent via UDP
[System.Serializable]
public class OlfatiSaberUdpParameters
{
    public float d_ref;
    public float r0_coh;
    public float delta;
    public float a;
    public float b;
    public float c;
    public float gamma;
    public float c_vm;
    public float scaleFactor;
    // Add any other parameters you deem "weights" here
}

public class OlfatiSaber : MonoBehaviour
{
    public List<GameObject> swarm;

    [Header("Olfati-Saber Model Parameters")]
    public float d_ref = 7.0f;
    public float r0_coh = 150.0f;
    public float delta = 0.1f;
    public float a = 0.9f;
    public float b = 1.5f;
    public float c; // This is typically calculated or set by SwarmManager
    public float gamma = 1.0f;
    public float c_vm = 1.0f;
    public float scaleFactor = 10.0f;

    [Header("Migration and Obstacle Avoidance")]
    public float maxMigrationDistance = 10.0f;
    public float detectionRadius = 5.0f;  // Radius to detect obstacles
    public float obstacleAvoidanceForceWeight = 2.0f; // Weight for obstacle avoidance force
    public float maxAvoidForce = 10.0f;   // Maximum avoidance force
    public string obstacleTag = "Obstacle"; // Tag for obstacles

    [Header("Desired State Inputs")]
    public float desired_height = 4.0f;
    public float desired_vx = 0.0f;
    public float desired_vy = 0.0f;
    public float desired_vz = 0.0f;
    public float desired_yaw = 0.0f;
    
    private Vector3 velocityMatching = new Vector3(0, 0, 0);
    private Vector3 cohesion = new Vector3(0, 0, 0);
    private Vector3 obstacle = new Vector3(0, 0, 0);

    private Vector3 swarmInput = new Vector3(0, 0, 0);

    private string droneName;

    [Header("UDP Settings")]
    public string targetIPAddress = "192.168.100.176"; // IP address of the receiver
    public int targetPort = 11000; // Port the receiver is listening on

    private UdpClient udpClient;

    void Start()
    {
        droneName = transform.parent.name;

        // Initialize UDP client
        try
        {
            udpClient = new UdpClient();
            Debug.Log($"OlfatiSaber UDP client initialized for {droneName}. Target: {targetIPAddress}:{targetPort}");
            // Note: UDP sends to an IP address and port over the network, not directly to a USB port.
            // The receiving application must be listening on this IP and port.
            Debug.LogWarning("UDP sends data over the network to an IP Address and Port, not directly to a USB port.");

            // Send initial parameters.
            // Be aware that 'c' might not be set by SwarmManager yet if called here.
            // Consider calling SendCurrentParameters() after 'c' is definitively set.
            SendCurrentParameters(); 
        }
        catch (System.Exception e)
        {
            Debug.LogError($"Failed to initialize UDP client for {droneName}: {e.Message}");
            udpClient = null; // Ensure client is null if initialization failed
        }
    }

    // Example: Call SendCurrentParameters() on a key press for testing
    // void Update()
    // {
    //     if (Input.GetKeyDown(KeyCode.U))
    //     {
    //         SendCurrentParameters();
    //     }
    // }

    public void SendCurrentParameters()
    {
        if (udpClient == null)
        {
            Debug.LogWarning($"UDP client not initialized for {droneName}. Cannot send parameters.");
            return;
        }

        OlfatiSaberUdpParameters parametersToSend = new OlfatiSaberUdpParameters
        {
            d_ref = this.d_ref,
            r0_coh = this.r0_coh,
            delta = this.delta,
            a = this.a,
            b = this.b,
            c = this.c, // Ensure 'c' has its correct value when sending
            gamma = this.gamma,
            c_vm = this.c_vm,
            scaleFactor = this.scaleFactor
        };

        string jsonParameters = JsonUtility.ToJson(parametersToSend);
        byte[] data = Encoding.UTF8.GetBytes(jsonParameters);

        try
        {
            udpClient.Send(data, data.Length, targetIPAddress, targetPort);
            // Debug.Log($"Sent Olfati-Saber parameters from {droneName} to {targetIPAddress}:{targetPort}: {jsonParameters}");
        }
        catch (SocketException e)
        {
            Debug.LogError($"SocketException sending UDP data from {droneName}: {e.Message} to {targetIPAddress}:{targetPort}");
        }
        catch (System.Exception e)
        {
            Debug.LogError($"Error sending UDP data from {droneName}: {e.Message}");
        }
    }

    void OnDestroy()
    {
        if (udpClient != null)
        {
            udpClient.Close();
            udpClient = null;
            Debug.Log($"OlfatiSaber UDP client closed for {droneName}.");
        }
    }

    void FixedUpdate()
    {
        // Reset the vectors
        velocityMatching = new Vector3(0, 0, 0);
        cohesion = new Vector3(0, 0, 0);
        obstacle = new Vector3(0, 0, 0);

        // Get the position and velocity of the current drone
        Vector3 position = transform.position;
        Rigidbody rb = GetComponent<Rigidbody>(); // Get Rigidbody once
        if (rb == null) return; // Should not happen if drone is physics based
        Vector3 velocity = rb.velocity;

        //Calculate the velocity matching force given the difference between the desired v_x, v_y, and v_z and the current velocity
        Vector3 desired_velocity_vm = new Vector3(desired_vx, desired_vz, -desired_vy); // Renamed to avoid conflict if desired_velocity is used elsewhere
        velocityMatching = c_vm * (desired_velocity_vm - velocity);           
        
        // Calculate the cohesion force for each of the neighbours
        if (swarm != null) // Add null check for swarm list
        {
            foreach (GameObject neighbour in swarm)
            {
                if (neighbour == null) continue; // Add null check for individual neighbour

                // Get the child of the neighbour
                Transform neighbourChildTransform = neighbour.transform.Find("DroneParent");
                if (neighbourChildTransform == null) continue; // Check if DroneParent exists
                GameObject neighbourChild = neighbourChildTransform.gameObject;
                
                // Skip the current drone
                if (neighbourChild == gameObject)
                {
                    continue;
                }

                // Get the position of the neighbour
                Vector3 neighbourPosition = neighbourChild.transform.position;

                // Relative Position
                Vector3 relativePosition = neighbourPosition - position;
                float distance = relativePosition.magnitude;

                if (distance < 0.001f) distance = 0.001f; // Avoid division by zero later

                // Relative Velocity
                Rigidbody neighbourRb = neighbourChild.GetComponent<Rigidbody>();
                if (neighbourRb == null) continue; // Ensure neighbour has Rigidbody
                Vector3 relativeVelocity = neighbourRb.velocity - velocity;

                // Scale the distance to fit the cohesion function
                // The 'r' parameter in your Olfati-Saber functions expects the scaled distance
                float scaled_distance_for_function = distance / scaleFactor; 
                
                // Cohesion
                Vector3 thisCohesion = GetCohesionForce(scaled_distance_for_function) * relativePosition.normalized;
                cohesion += thisCohesion;

                // // Check if this is drone 0
                // if (droneName == "Drone 0")
                // {
                //     // Print the neighbour name, relative position, distance, relative position normalized and cohesion
                //     Debug.Log("Neighbour: " + neighbourChild.name + " Relative Position: " + relativePosition + " Distance: " + distance + " Relative Position Normalized: " + relativePosition.normalized + " thisCohesion: " + thisCohesion + " Cohesion: " + cohesion + " Cohesion Magnitude: " + cohesion.magnitude);
                // }
            }
        }


        // if (droneName == "Drone 0")
        // {
        //     // Print the algorithm parameters: d_ref, r0_coh, delta, a, b, c
        //     Debug.Log("d_ref: " + d_ref + " r0_coh: " + r0_coh + " delta: " + delta + " a: " + a + " b: " + b + " c: " + c);
        // }

        // TODO: Add the obstacle avoidance force here
        // obstacle = obstacleAvoidanceForce(rb); // Call if implemented

        // Get the total velocity command
        swarmInput = velocityMatching + cohesion + obstacle;

        // To ensure this script does not cause net vertical drift of the swarm's center of mass,
        // set its contribution to vertical velocity to zero.
        swarmInput.y = 0.0f;


        var vc = GetComponent<VelocityControl>(); // Get VelocityControl once
        if (vc != null)
        {
            vc.swarm_vx = swarmInput.x;
            vc.swarm_vy = swarmInput.y;
            vc.swarm_vz = swarmInput.z;        
        }
    }

    private Vector3 obstacleAvoidanceForce(Rigidbody droneRB)
    {
        Vector3 avoidForce = Vector3.zero;
        // Collider[] hitColliders = Physics.OverlapSphere(droneRB.transform.position, detectionRadius);
        // float radius = 0f;
        // if(droneRB.transform.parent != null && droneRB.transform.parent.GetComponent<interactionHandler>() != null)
        // {
        //     radius = droneRB.transform.parent.GetComponent<interactionHandler>().radiusOfCollider;
        // }


        // foreach (var hitCollider in hitColliders)
        // {
        //     if (hitCollider.CompareTag(obstacleTag) && hitCollider.gameObject != droneRB.gameObject) // Don't avoid self
        //     {
        //         Vector3 closestPoint = hitCollider.ClosestPoint(droneRB.transform.position);
        //         Vector3 directionToObstacle = droneRB.transform.position - closestPoint;
        //         float distanceToObstacle = directionToObstacle.magnitude;
        //         // if (radius > 0 && distanceToObstacle > radius) distanceToObstacle -= radius; // Adjust by radius only if not already inside
        //         // else if (radius > 0 && distanceToObstacle <= radius) distanceToObstacle = 0.001f; // Very close or overlapping

        //         if (distanceToObstacle < detectionRadius && distanceToObstacle > 0.001f) // distance > 0 to avoid issues at contact
        //         {
        //             // Force inversely proportional to distance (or square of distance)
        //             float forceMagnitude = obstacleAvoidanceForceWeight / distanceToObstacle; // Simpler inverse
        //             // float forceMagnitude = obstacleAvoidanceForceWeight / (distanceToObstacle * distanceToObstacle); // Inverse square
        //             forceMagnitude = Mathf.Min(forceMagnitude, maxAvoidForce); // Clamp force
        //             avoidForce += directionToObstacle.normalized * forceMagnitude;
        //         }
        //     }
        // }
        // if(droneRB.transform.parent != null && droneRB.transform.parent.GetComponent<varibaleManager>() != null)
        // {
        //    droneRB.transform.parent.GetComponent<varibaleManager>().lastObstacle = avoidForce;
        // }
        return avoidForce;
    }

    public float GetCohesionForce(float r) // 'r' here is the scaled_distance_for_function
    {
        // Ensure c is valid, especially if a or b can be zero or negative.
        if (float.IsNaN(c) || float.IsInfinity(c)) {
            // Debug.LogError($"Invalid 'c' value ({c}) in GetCohesionForce for {droneName}. Defaulting c calculation or returning 0.");
            // Optionally, try to calculate c if a and b are valid:
            if (a > 0 && b > 0 && a != b) c = Mathf.Abs(b - a) / (2 * Mathf.Sqrt(a * b)); // Ensure a*b is positive
            else c = 0; // Fallback c
        }
        if (r0_coh == 0) return 0f; // Avoid division by zero

        float neighbourWeightDerivative = GetNeighbourWeightDerivative(r); // r is scaled_distance
        float cohesionIntensity = GetCohesionIntensity(r); // r is scaled_distance
        float neighbourWeight = GetNeighbourWeight(r); // r is scaled_distance
        float cohesionIntensityDerivative = GetCohesionIntensityDerivative(r); // r is scaled_distance

        // The formula from Olfati-Saber paper for u_i^alpha (cohesion term) often looks like:
        // - Sum_j [ phi_alpha( ||q_j - q_i||_sigma ) * n_ij + a_ij(q_i, q_j) * (p_j - p_i) ]
        // Your GetCohesionForce seems to be calculating the magnitude of the force based on scalar functions.
        // The original formula is: (1.0f / r0_coh) * neighbourWeightDerivative * cohesionIntensity + neighbourWeight * cohesionIntensityDerivative;
        // Ensure this correctly represents the force magnitude as per the model.
        return (1.0f / r0_coh) * neighbourWeightDerivative * cohesionIntensity + neighbourWeight * cohesionIntensityDerivative;
    }

    // Cohesion intensity function
    public float GetCohesionIntensity(float r) // 'r' here is the scaled_distance_for_function
    {
        if (float.IsNaN(c) || float.IsInfinity(c)) c = 0; // Fallback for c
        float diff = r - d_ref; // d_ref is also a scaled distance if r is scaled
        
        float term_sqrt_1_val = 1 + Mathf.Pow(diff + c, 2);
        float term_sqrt_2_val = 1 + c * c;

        // Ensure arguments to Sqrt are non-negative, though Pow(..., 2) should handle this.
        term_sqrt_1_val = Mathf.Max(0, term_sqrt_1_val);
        term_sqrt_2_val = Mathf.Max(0, term_sqrt_2_val);

        return ((a + b) / 2.0f) * (Mathf.Sqrt(term_sqrt_1_val) - Mathf.Sqrt(term_sqrt_2_val)) + ((a - b) * diff) / 2.0f;
    }

    // Derivative of cohesion intensity function
    float GetCohesionIntensityDerivative(float r) // 'r' here is the scaled_distance_for_function
    {
        if (float.IsNaN(c) || float.IsInfinity(c)) c = 0; // Fallback for c
        float diff = r - d_ref;
        
        float denominator_val = 1 + Mathf.Pow(diff + c, 2);
        if (denominator_val <= 0) return (a - b) / 2.0f; // Avoid division by zero or sqrt of negative

        return ((a + b) / 2.0f) * (diff + c) / Mathf.Sqrt(denominator_val) + (a - b) / 2.0f;
    }
    
    // Neighbor weight function
    public float GetNeighbourWeight(float r) // 'r' here is the scaled_distance_for_function
    {
        if (r0_coh == 0) return 0f; // Avoid division by zero
        // 'r' is already scaled_distance. So r_ratio is (distance/scaleFactor) / r0_coh
        // This means r0_coh should also be considered in the context of scaled distances.
        // Or, the input 'r' to this function should be the unscaled distance, and scaling happens internally.
        // Assuming 'r' is already scaled:
        float r_ratio_val = r / r0_coh; // If r is scaled distance, r0_coh should be a "scaled reference cohesion range"

        if (r_ratio_val < delta)
        {
            return 1.0f;
        }
        else if (r_ratio_val < 1.0f)
        {
            if (1.0f - delta == 0) return 0f; // Avoid division by zero
            float arg = Mathf.PI * (r_ratio_val - delta) / (1.0f - delta);
            return Mathf.Pow(0.5f * (1.0f + Mathf.Cos(arg)), 2);
        }
        else
        {
            return 0.0f;
        }
    }

    // Derivative of neighbor weight function
    float GetNeighbourWeightDerivative(float r) // 'r' here is the scaled_distance_for_function
    {
        if (r0_coh == 0) return 0f; // Avoid division by zero
        float r_ratio_val = r / r0_coh; // Similar consideration for r_ratio as in GetNeighbourWeight

        if (r_ratio_val < delta)
        {
            return 0.0f;
        }
        else if (r_ratio_val < 1.0f)
        {
            if (1.0f - delta == 0) return 0f; // Avoid division by zero
            float arg = Mathf.PI * (r_ratio_val - delta) / (1.0f - delta);
            // The derivative of (0.5 * (1 + cos(arg)))^2 w.r.t. arg is:
            // 2 * (0.5 * (1 + cos(arg))) * (0.5 * (-sin(arg))) = -0.5 * (1 + cos(arg)) * sin(arg)
            // Chain rule: d(arg)/dr_ratio_val = PI / (1-delta)
            // Chain rule: d(r_ratio_val)/dr = 1/r0_coh
            // So, derivative w.r.t r is:
            // -0.5 * (1 + Mathf.Cos(arg)) * Mathf.Sin(arg) * (Mathf.PI / (1.0f - delta)) * (1.0f / r0_coh)
            // Your formula: 0.5f*(-Mathf.PI) / (1 - delta) * (1 + Mathf.Cos(arg)) * Mathf.Sin(arg);
            // This is missing the (1/r0_coh) term from the chain rule if 'r' is the input to this function.
            // If 'r' is actually 'r_ratio', then the 1/r0_coh is implicitly handled.
            // Let's assume the input 'r' is scaled_distance, and the derivative is w.r.t this 'r'.
            
            // Corrected derivative w.r.t. 'r' (scaled_distance)
            return -0.5f * (1.0f + Mathf.Cos(arg)) * Mathf.Sin(arg) * (Mathf.PI / (1.0f - delta)) * (1.0f / r0_coh);
        }
        else
        {
            return 0.0f;
        }
    }
}