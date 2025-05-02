using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.XR.Hands.Samples.VisualizerSample; // To access HandProcessor

public class Reynolds : MonoBehaviour
{
    public List<GameObject> swarm;

    // --- Base Weights (Keep these at 1.0 initially unless needed) --- TO CHANGE IN THE INSPECTOR
    public float cohesionWeight = 1.0f;  
    public float separationWeight = 1.0f; 
    public float alignmentWeight = 1.0f;

    // --- Multipliers (Tune these first) ---
    public float cohesionMultiplier = 2000.0f; // DECREASED: Less pull together
    public float separationMultiplier = 10000.0f; // INCREASED: More push apart

    // --- Axis Modifiers (Tune these to enhance expansion) ---
    public float cohesionAxisModifier = 1.0f; // INCREASED: Cohesion weakens more with expansion
    public float separationAxisModifier = 1.0f; // INCREASED: Separation strengthens more with expansion

    private Vector3 cohesion = Vector3.zero;
    private Vector3 separation = Vector3.zero;
    private Vector3 alignment = Vector3.zero;
    private Vector3 swarmInput = Vector3.zero;

    // Define baseline values (adjust based on your expectations)
    private float baselineLength = 0.5f; // example baseline for palm distance
    private float baselineWidth  = 0.1f; // example baseline for width scale

    void FixedUpdate()
    {
        // Reset the vectors
        cohesion = Vector3.zero;
        separation = Vector3.zero;
        alignment = Vector3.zero;       

        int neighbourCount = 0; // Keep track of actual neighbours processed

        // Calculate the relative position and velocity of each drone to the current drone
        foreach (GameObject neighbour in swarm)
        {
            // Ensure neighbour has DroneParent and Rigidbody, and isn't self
            Transform neighbourChildTransform = neighbour.transform.Find("DroneParent");
            if (neighbourChildTransform == null) continue; 
            GameObject neighbourChild = neighbourChildTransform.gameObject;
            if (neighbourChild == gameObject) continue; // Use direct comparison
            Rigidbody neighbourRb = neighbourChild.GetComponent<Rigidbody>();
            Rigidbody selfRb = GetComponent<Rigidbody>(); // Get self Rigidbody once
            if (neighbourRb == null || selfRb == null) continue;

            neighbourCount++; // Increment count only for valid neighbours

            Vector3 neighbourPosition = neighbourChild.transform.position;
            Vector3 relativePosition = neighbourPosition - transform.position;
            float distance = relativePosition.magnitude;
            // Prevent division by zero if distance is very small
            if (distance < 0.001f) distance = 0.001f; 
            
            Vector3 relativeVelocity = neighbourRb.velocity - selfRb.velocity;

            // Cohesion: steer toward average position (sum of relative positions)
            cohesion += relativePosition;
            // Separation: steer away from neighbours (sum of inverse-distance weighted vectors)
            separation -= relativePosition.normalized / distance; // Normalize first for direction, then scale by inverse distance
            // Alignment: match velocity (sum of relative velocities)
            alignment += relativeVelocity;
        }

        // Average the forces only if there were neighbours
        if(neighbourCount > 0)
        {
            cohesion /= neighbourCount;
            separation /= neighbourCount;
            alignment /= neighbourCount;
        }

        // --- Hand Orientation and Scaling ---

        // 1. Get Hand Data from HandProcessor's static variables
        Vector3 leftPalmPos = HandProcessor.LeftPalmPosition;   // CHANGED
        Vector3 rightPalmPos = HandProcessor.RightPalmPosition;  // CHANGED
        float handLength = HandProcessor.HandEllipsoidLength;
        float handWidth = HandProcessor.HandEllipsoidWidth;
        bool palmsTracked = HandProcessor.ArePalmsTracked; // Get tracking status

        // 2. Determine Hand Orientation (only if palms are tracked)
        Vector3 handVector = Vector3.forward; // Default to world forward
        Quaternion handOrientation = Quaternion.identity; 

        if (palmsTracked) // Check if palms are tracked before calculating orientation
        {
             handVector = rightPalmPos - leftPalmPos; // CHANGED to use palm positions
             if (handVector.sqrMagnitude > 0.001f) // Check if palms are sufficiently separated
             {
                 // Use LookRotation: forward direction is handVector, up is world up
                 handOrientation = Quaternion.LookRotation(handVector.normalized, Vector3.up);
             }
             // If sqrMagnitude is too small, handOrientation remains identity (world aligned)
        }
        // If palms are not tracked, handOrientation remains identity (world aligned)
        
        // 3. Calculate Scaling Factors
        // Use safety checks for division by zero
        float lengthFactor = (baselineLength > 0.001f && handLength > 0) ? handLength / baselineLength : 1.0f;
        float widthFactor  = (baselineWidth > 0.001f && handWidth > 0)  ? handWidth / baselineWidth   : 1.0f;

        // Clamp factors to prevent extreme values (optional, but recommended)
        lengthFactor = Mathf.Clamp(lengthFactor, 0.1f, 10.0f);
        widthFactor = Mathf.Clamp(widthFactor, 0.1f, 10.0f);

        // 4. Transform Forces to Hand-Local Space
        Vector3 localCohesion = Quaternion.Inverse(handOrientation) * cohesion;
        Vector3 localSeparation = Quaternion.Inverse(handOrientation) * separation;
        // Note: Alignment is kept in world space for simplicity, could be transformed too if needed.

        // 5. Apply Scaling Factors in Hand-Local Space
        // Length factor affects local Z (along handVector)
        // Width factor affects local X and Y (perpendicular to handVector)
        localCohesion.x /= (widthFactor * cohesionAxisModifier);
        localCohesion.y /= (widthFactor * cohesionAxisModifier);
        localCohesion.z /= (lengthFactor * cohesionAxisModifier);

        localSeparation.x *= (widthFactor * separationAxisModifier);
        localSeparation.y *= (widthFactor * separationAxisModifier);
        localSeparation.z *= (lengthFactor * separationAxisModifier);

        // 6. Transform Scaled Forces Back to World Space
        Vector3 scaledWorldCohesion = handOrientation * localCohesion;
        Vector3 scaledWorldSeparation = handOrientation * localSeparation;

        // 7. Apply Base Weights and Multipliers
        scaledWorldCohesion *= cohesionWeight * cohesionMultiplier;
        scaledWorldSeparation *= separationWeight * separationMultiplier;
        alignment *= alignmentWeight; // Apply alignment weight (no multiplier here)

        // 8. Combine Forces
        swarmInput = scaledWorldCohesion + scaledWorldSeparation + alignment;

        // Pass the final computed vector to the VelocityControl script
        var vc = GetComponent<VelocityControl>();
        if(vc != null)
        {
            vc.swarm_vx = swarmInput.x;
            vc.swarm_vy = swarmInput.y;
            vc.swarm_vz = swarmInput.z;
        }
        else
        {
            Debug.LogWarning($"VelocityControl script not found on {gameObject.name}");
        }
    }
}

