using System.Collections;
using System.Collections.Generic;
using System.Security;
using UnityEngine;
using UnityEngine.XR.Hands.Samples.VisualizerSample; // To access HandProcessor

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
    public float scaleFactor = 10.0f; // Scales distance for cohesion function input

    [Header("Migration and Obstacle Avoidance")]
    public float maxMigrationDistance = 10.0f;
    public float detectionRadius = 5.0f;  // Radius to detect obstacles
    public float obstacleAvoidanceForceWeight = 2.0f; // Weight for obstacle avoidance force
    public float maxAvoidForce = 10.0f;   // Maximum avoidance force
    public string obstacleTag = "Obstacle"; // Tag for obstacles

    [Header("Desired State Inputs")]
    public float desired_height = 4.0f;
    public float desired_vx = 0.0f;
    public float desired_vy = 0.0f; // Note: In your velocity matching, this seems to be used for Z in world
    public float desired_vz = 0.0f; // Note: In your velocity matching, this seems to be used for Y in world
    public float desired_yaw = 0.0f;

    [Header("Hand Interaction Parameters")]
    public float cohesionAxisModifier = 1.0f; // Similar to Reynolds, for tuning hand influence
    public float baselineLength = 0.5f; // Example baseline for palm distance
    public float baselineWidth  = 0.1f; // Example baseline for width scale
    
    private Vector3 velocityMatching = new Vector3(0, 0, 0);
    private Vector3 cohesion = new Vector3(0, 0, 0);
    private Vector3 obstacle = new Vector3(0, 0, 0); // Obstacle force, currently calculated but not used in swarmInput

    private Vector3 swarmInput = new Vector3(0, 0, 0);

    private string droneName;

    void Start()
    {
        droneName = transform.parent.name;
        // It's good practice to ensure 'c' is set, typically by SwarmManager via swarmAlgorithm script
        // If SwarmManager calculates c, this script will receive it.
        // If not, you might need a local calculation or ensure it's set.
        // For example: if (a != 0 && b != 0) c = (b - a) / (2 * Mathf.Sqrt(a * b));
    }


    void FixedUpdate()
    {
        // Reset the vectors
        velocityMatching = Vector3.zero;
        cohesion = Vector3.zero;
        obstacle = Vector3.zero; // Reset obstacle force, though it's not added to swarmInput yet

        // Get the position and velocity of the current drone
        Vector3 position = transform.position;
        Vector3 velocity = GetComponent<Rigidbody>().velocity;

        //Calculate the velocity matching force
        // Original: Vector3 desired_velocity = new Vector3(desired_vx, desired_vz, -desired_vy);
        // Assuming standard Unity coordinates for desired_v: X, Y (up), Z (forward)
        // And your drone's local frame might be different.
        // Let's assume desired_vx, desired_vy (for up/down), desired_vz are in world frame.
        // The velocityMatching calculation seems to map desired_vy to world Z and desired_vz to world Y.
        // For clarity, let's use more descriptive names if these are world targets:
        // Vector3 world_desired_velocity = new Vector3(desired_vx, desired_height_velocity_component_or_vz, desired_forward_velocity_component_or_vy);
        // For now, sticking to your existing mapping:
        Vector3 targetVelocityForVM = new Vector3(desired_vx, desired_vz, -desired_vy); // This seems to be a mix of axes. Ensure this is intended.
        velocityMatching = c_vm * (targetVelocityForVM - velocity);           
        
        // Calculate the cohesion force for each of the neighbours
        int validNeighbourCount = 0;
        foreach (GameObject neighbour in swarm)
        {
            GameObject neighbourChild = neighbour.transform.Find("DroneParent")?.gameObject;
            
            if (neighbourChild == null || neighbourChild == gameObject)
            {
                continue;
            }
            validNeighbourCount++;

            Vector3 neighbourPosition = neighbourChild.transform.position;
            Vector3 relativePosition = neighbourPosition - position;
            float distance = relativePosition.magnitude;

            if (distance < 0.001f) distance = 0.001f; // Avoid division by zero

            // Scale the distance for the cohesion function input
            float scaledDistanceForFunction = distance / scaleFactor;
            
            Vector3 individualCohesionForce = GetCohesionForce(scaledDistanceForFunction) * relativePosition.normalized;
            cohesion += individualCohesionForce;
        }

        if (validNeighbourCount > 0)
        {
            // Optionally, average cohesion, though Olfati-Saber sums weighted influences.
            // The GetCohesionForce itself contains weighting.
            // So, direct summation is usually correct.
        }

        // --- Hand Orientation and Scaling for Cohesion ---
        Vector3 leftPalmPos = HandProcessor.LeftPalmPosition;
        Vector3 rightPalmPos = HandProcessor.RightPalmPosition;
        float handLength = HandProcessor.HandEllipsoidLength;
        float handWidth = HandProcessor.HandEllipsoidWidth;
        bool palmsTracked = HandProcessor.ArePalmsTracked;

        Vector3 handVector = Vector3.forward; // Default to world forward
        Quaternion handOrientation = Quaternion.identity; 

        if (palmsTracked)
        {
             handVector = rightPalmPos - leftPalmPos;
             if (handVector.sqrMagnitude > 0.001f) // Check if palms are sufficiently separated
             {
                 handOrientation = Quaternion.LookRotation(handVector.normalized, Vector3.up);
             }
        }
        
        float lengthFactor = (baselineLength > 0.001f && handLength > 0) ? handLength / baselineLength : 1.0f;
        float widthFactor  = (baselineWidth > 0.001f && handWidth > 0)  ? handWidth / baselineWidth   : 1.0f;

        lengthFactor = Mathf.Clamp(lengthFactor, 0.1f, 10.0f);
        widthFactor = Mathf.Clamp(widthFactor, 0.1f, 10.0f);

        if (cohesion.sqrMagnitude > 0.0001f) // Only apply if there's a cohesion force
        {
            // Transform Cohesion to Hand-Local Space
            Vector3 localCohesion = Quaternion.Inverse(handOrientation) * cohesion;

            // Apply Scaling Factors in Hand-Local Space
            // Dividing by factor: if hands expand (factor > 1), cohesion along that axis is reduced.
            if (widthFactor * cohesionAxisModifier > 0.001f) // Avoid division by zero/small numbers
            {
                localCohesion.x /= (widthFactor * cohesionAxisModifier);
                localCohesion.y /= (widthFactor * cohesionAxisModifier); // Assuming Y is the other axis in the hand plane
            }
            if (lengthFactor * cohesionAxisModifier > 0.001f)
            {
                localCohesion.z /= (lengthFactor * cohesionAxisModifier); // Z is along the hand vector
            }

            // Transform Scaled Cohesion Back to World Space
            cohesion = handOrientation * localCohesion;
        }
        // --- End Hand Interaction for Cohesion ---

        // Calculate obstacle avoidance force (currently not added to swarmInput)
        // if (GetComponent<Rigidbody>()) // Ensure Rigidbody exists
        // {
        //     obstacle = obstacleAvoidanceForce(GetComponent<Rigidbody>());
        // }


        // Get the total velocity command
        // Note: The 'obstacle' vector is calculated but not added here. Add it if needed.
        swarmInput = velocityMatching + cohesion; // + obstacle; if you implement and want to use it

        GetComponent<VelocityControl>().swarm_vx = swarmInput.x;
        GetComponent<VelocityControl>().swarm_vy = swarmInput.y; // This is world Y (up/down)
        GetComponent<VelocityControl>().swarm_vz = swarmInput.z; // This is world Z (forward/backward)      
        
    }

    private Vector3 obstacleAvoidanceForce(Rigidbody droneRB)
    {
        Vector3 avoidForce = Vector3.zero;
        Collider[] hitColliders = Physics.OverlapSphere(droneRB.transform.position, detectionRadius);
        // float radius = 0; // Placeholder, get actual radius if needed
        // if (droneRB.gameObject.transform.parent.GetComponent<interactionHandler>() != null)
        // {
        //     radius = droneRB.gameObject.transform.parent.GetComponent<interactionHandler>().radiusOfCollider;
        // }

        foreach (var hitCollider in hitColliders)
        {
            if (hitCollider.CompareTag(obstacleTag) && !hitCollider.isTrigger) // Ensure not hitting self or triggers
            {
                Vector3 closestPoint = hitCollider.ClosestPoint(droneRB.transform.position);
                Vector3 directionToObstacle = droneRB.transform.position - closestPoint;
                float distanceToObstacle = directionToObstacle.magnitude; // - radius; // Adjust if radius is significant

                if (distanceToObstacle < detectionRadius && distanceToObstacle > 0.001f) // distance > 0 to avoid issues at contact
                {
                    // Force inversely proportional to distance (or square of distance)
                    float forceMagnitude = obstacleAvoidanceForceWeight / distanceToObstacle;
                    forceMagnitude = Mathf.Min(forceMagnitude, maxAvoidForce); // Clamp force
                    avoidForce += directionToObstacle.normalized * forceMagnitude;
                }
            }
        }
        // if (droneRB.transform.parent.GetComponent<varibaleManager>() != null)
        // {
        //     droneRB.transform.parent.GetComponent<varibaleManager>().lastObstacle = avoidForce;
        // }
        return avoidForce;
    }


    public float GetCohesionForce(float r)
    {
        // Ensure c is valid, especially if a or b can be zero or negative.
        // This calculation of c: (b-a) / (2 * Mathf.Sqrt(a*b)) assumes a and b are positive.
        // SwarmManager typically provides 'c'. If 'c' is NaN or problematic, this function will fail.
        // Add a check for c if necessary: if (float.IsNaN(c)) c = 0; (or a default)

        float neighbourWeightDerivative = GetNeighbourWeightDerivative(r);
        float cohesionIntensity = GetCohesionIntensity(r);
        float neighbourWeight = GetNeighbourWeight(r);
        float cohesionIntensityDerivative = GetCohesionIntensityDerivative(r);

        // Original formula: 1 / r0_coh * neighbourWeightDerivative * cohesionIntensity + neighbourWeight * cohesionIntensityDerivative;
        // This can lead to very large or very small numbers.
        // Consider if r0_coh should be part of the scaling or if the terms need normalization.
        // For now, using the direct formula:
        return (1.0f / r0_coh) * neighbourWeightDerivative * cohesionIntensity + neighbourWeight * cohesionIntensityDerivative;
    }

    // Cohesion intensity function
    public float GetCohesionIntensity(float r)
    {
        // if (float.IsNaN(c)) { Debug.LogError("c is NaN in GetCohesionIntensity"); return 0; }
        float diff = r - d_ref;
        float term1_sqrt = 1 + Mathf.Pow(diff + c, 2);
        float term2_sqrt = 1 + c * c;

        if (term1_sqrt < 0) term1_sqrt = 0; // Should not happen with Pow(..., 2)
        if (term2_sqrt < 0) term2_sqrt = 0;

        return ((a + b) / 2.0f) * (Mathf.Sqrt(term1_sqrt) - Mathf.Sqrt(term2_sqrt)) + ((a - b) * diff) / 2.0f;
    }

    // Derivative of cohesion intensity function
    float GetCohesionIntensityDerivative(float r)
    {
        // if (float.IsNaN(c)) { Debug.LogError("c is NaN in GetCohesionIntensityDerivative"); return 0; }
        float diff = r - d_ref;
        float denominator_sqrt = 1 + Mathf.Pow(diff + c, 2);
        if (denominator_sqrt <= 0) return (a - b) / 2.0f; // Avoid division by zero or sqrt of negative

        return ((a + b) / 2.0f) * (diff + c) / Mathf.Sqrt(denominator_sqrt) + (a - b) / 2.0f;
    }
    
    // Neighbor weight function
    public float GetNeighbourWeight(float r)
    {
        if (r0_coh == 0) return 0; // Avoid division by zero
        float r_ratio = r / r0_coh;

        if (r_ratio < delta)
        {
            return 1.0f;
        }
        else if (r_ratio < 1.0f)
        {
            if (1 - delta == 0) return 0; // Avoid division by zero
            float arg = Mathf.PI * (r_ratio - delta) / (1 - delta);
            return Mathf.Pow(0.5f * (1.0f + Mathf.Cos(arg)), 2);
        }
        else
        {
            return 0.0f;
        }
    }

    // Derivative of neighbor weight function
    float GetNeighbourWeightDerivative(float r)
    {
        if (r0_coh == 0) return 0; // Avoid division by zero
        float r_ratio = r / r0_coh;

        if (r_ratio < delta)
        {
            return 0.0f;
        }
        else if (r_ratio < 1.0f)
        {
            if (1 - delta == 0) return 0; // Avoid division by zero
            float arg = Mathf.PI * (r_ratio - delta) / (1 - delta);
            // Original: 0.5f*(-Mathf.PI) / (1 - delta) * (1 + Mathf.Cos(arg)) * Mathf.Sin(arg);
            // Corrected based on common bump function derivative form ( (1+cos(x))^2 derivative is 2*(1+cos(x))*(-sin(x)) * (inner derivative) )
            // The derivative of (0.5 * (1 + cos(arg)))^2 = 2 * (0.5 * (1 + cos(arg))) * (0.5 * (-sin(arg))) * (d(arg)/dr_ratio) * (dr_ratio/dr)
            // d(arg)/dr_ratio = PI / (1-delta)
            // dr_ratio/dr = 1/r0_coh
            // So: (1 + cos(arg)) * (-0.5 * sin(arg)) * (PI / (1-delta)) * (1/r0_coh)
            // The formula in the paper might be slightly different or simplified. Sticking to your provided one:
            return (0.5f * (-Mathf.PI) / ( (1.0f - delta) * r0_coh ) ) * (1.0f + Mathf.Cos(arg)) * Mathf.Sin(arg); // Added 1/r0_coh for chain rule on r
        }
        else
        {
            return 0.0f;
        }
    }
}