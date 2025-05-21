using System.Collections;
using System.Collections.Generic;
using System.Security;
using UnityEngine;
using UnityEngine.XR.Hands.Samples.VisualizerSample;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System;

public class OlfatiSaber : MonoBehaviour
{
    public List<GameObject> swarm;
    private Vector3 cohesion = Vector3.zero; // This is calculated from GetCohesionForce

    [Header("Olfati-Saber Model Parameters")]
    public float d_ref = 7.0f;
    public float r0_coh = 20.0f; // MODIFIED: Was 150.0f, try a smaller value
    public float delta = 0.1f;
    public float a = 0.9f;
    public float b = 1.5f;
    public float c;
    public float gamma = 1.0f;
    public float c_vm = 1.0f;
    public float scaleFactor = 10.0f;
    public float cohesionMultiplier = 2.0f;

    [Header("Hand Interaction Parameters")]
    public float baselineLength = 0.5f; // Adjust this to your neutral hand separation (palm-to-palm)
    public float baselineWidth = 0.1f;  // Adjust this to a neutral hand width measure
    public float handCohesionAxisModifier = 1.0f; // Tune this: >1 weakens cohesion more with expansion, <1 weakens it less.

    [Header("Migration and Obstacle Avoidance")]
    public float maxMigrationDistance = 10.0f;
    public float detectionRadius = 5.0f;
    public float obstacleAvoidanceForceWeight = 2.0f;
    public float maxAvoidForce = 10.0f;
    public string obstacleTag = "Obstacle";
    // Add this new parameter to your class
    //[Header("Force Multipliers")]

    [Header("Desired State Inputs (for Velocity Matching)")]
    public float desired_vx = 0.0f;
    public float desired_vy = 0.0f; // This is mapped to world Z for desired_velocity
    public float desired_vz = 0.0f; // This is mapped to world Y for desired_velocity
    public float desired_yaw = 0.0f;
    public float desired_height = 4.0f;  // Add this line - default value matches typical VelocityControl value
    
    private Vector3 velocityMatching = Vector3.zero;
    private Vector3 obstacle = Vector3.zero;
    private Vector3 swarmInput = Vector3.zero;

    private string droneName; // Class member for drone name

    // vector3 cohesionmultiplier
    //public Vector3 cohesionMultiplier = new Vector3(5.0f, 5.0f, 5.0f);
  

    

    void Start()
    {
        // Initialize droneName once
        if (transform.parent != null)
        {
            droneName = transform.parent.name;
        }
        else
        {
            droneName = gameObject.name;
            Debug.LogWarning("OlfatiSaber script on " + gameObject.name + " does not have a parent. Using GameObject name.");
        }
        
       
    }


    void FixedUpdate()
    {
        // Reset the vectors
        velocityMatching = Vector3.zero;
        cohesion = Vector3.zero;
        obstacle = Vector3.zero;

        // Get the position and velocity of the current drone
        Vector3 position = transform.position;
        Rigidbody rb = GetComponent<Rigidbody>();
        if (rb == null) return;
        Vector3 velocity = rb.velocity;

        // Calculate the velocity matching force
        // Note: desired_vz (OlfatiSaber field) contributes to world Y of desired_velocity for VM.
        // Ensure desired_vz is 0.0f in Inspector if no vertical velocity is intended from THIS component.
        Vector3 desired_velocity_for_vm = new Vector3(desired_vx, desired_vz, -desired_vy);
        velocityMatching = c_vm * (desired_velocity_for_vm - velocity); 

        if (this.droneName == "Drone 0") // Use class member droneName
        {
            Debug.Log($"OlfatiSaber Drone 0 - VM Target: {desired_velocity_for_vm}, CurrentVel: {velocity}, Resulting VM: {velocityMatching}");
            Debug.Log("cohesionMultiplier: " + cohesionMultiplier);
        }          
        
        // Calculate the base cohesion force for each of the neighbours
        if (swarm != null)
        {
            foreach (GameObject neighbourGO in swarm)
            {
                if (neighbourGO == null) continue;

                Transform neighbourChildTransform = neighbourGO.transform.Find("DroneParent");
                if (neighbourChildTransform == null) continue;
                GameObject neighbourChild = neighbourChildTransform.gameObject;
            
                if (neighbourChild == gameObject) continue;

                Vector3 neighbourPosition = neighbourChild.transform.position;
                Vector3 relativePosition = neighbourPosition - position;
                float distance = relativePosition.magnitude;

                if (distance < 0.001f) distance = 0.001f;

                // Scale the distance for the Olfati-Saber math functions
                float scaled_distance_for_function = distance / scaleFactor;
            
                Vector3 individualCohesionForce = GetCohesionForce(scaled_distance_for_function) * relativePosition.normalized;
                cohesion += individualCohesionForce;
            }
        }
       
       

        // --- Hand Interaction for Cohesion (Copied and Adapted from Reynolds-like logic) ---
        if (HandProcessor.ArePalmsTracked)
        {
            Vector3 leftPalmPos = HandProcessor.LeftPalmPosition;
            Vector3 rightPalmPos = HandProcessor.RightPalmPosition;
            float currentHandEllipsoidLength = HandProcessor.HandEllipsoidLength;
            float currentHandEllipsoidWidth = HandProcessor.HandEllipsoidWidth;

            Quaternion handOrientation = Quaternion.identity;
            Vector3 handVector = rightPalmPos - leftPalmPos;

            if (handVector.sqrMagnitude > 0.001f) // Ensure handVector is not zero
            {
                handOrientation = Quaternion.LookRotation(handVector.normalized, Vector3.up);
            }

            // Calculate scaling factors based on current hand dimensions vs baselines
            float lengthFactor = (baselineLength > 0.001f && currentHandEllipsoidLength > 0) ? currentHandEllipsoidLength / baselineLength : 1.0f;
            float widthFactor = (baselineWidth > 0.001f && currentHandEllipsoidWidth > 0) ? currentHandEllipsoidWidth / baselineWidth : 1.0f;

            // Clamp factors to prevent extreme scaling
            lengthFactor = Mathf.Clamp(lengthFactor, 0.1f, 10.0f);
            widthFactor = Mathf.Clamp(widthFactor, 0.1f, 10.0f);

            // --- Debugging: Output factors and pre-scaled cohesion ---
            // if (this.droneName == "Drone 0") // Or any specific drone for less spam
            // {
            //     Debug.Log($"Hand Factors: LengthF={lengthFactor:F2}, WidthF={widthFactor:F2}. Pre-Scale Cohesion: {cohesion.magnitude:F2}");
            // }

            // Transform cohesion to hand-local space
            Vector3 localCohesion = Quaternion.Inverse(handOrientation) * cohesion;

            // Apply the cohesion multiplier BEFORE scaling by hand factors
            localCohesion *= cohesionMultiplier;

            // Fix the safeAxisModifier calculation - it was forcing a minimum of 100!
            float safeAxisModifier = Mathf.Max(0.1f, handCohesionAxisModifier); // Changed from 100f to 0.001f

            localCohesion.x /= (widthFactor * safeAxisModifier);
            localCohesion.y /= (widthFactor * safeAxisModifier);
            localCohesion.z /= (lengthFactor * safeAxisModifier);

            // Transform scaled cohesion back to world space
            cohesion = handOrientation * localCohesion;
            Debug.Log($"Hand Orientation: {handOrientation}, Local Cohesion: {localCohesion}, Scaled Cohesion: {cohesion}");
            Debug.Log($"Cohesion: {cohesion}");

            // --- Debugging: Output post-scaled cohesion ---
            // if (this.droneName == "Drone 0")
            // {
            //     Debug.Log($"Post-Scale Cohesion: {cohesion.magnitude:F2}");
            // }
        }
        // If palms are not tracked, cohesion remains unscaled by hands.
        // --- End Hand Interaction for Cohesion ---

        // obstacle = obstacleAvoidanceForce(rb); // Implement if needed

        swarmInput = velocityMatching + cohesion + obstacle;

        // Ensure OlfatiSaber script does not command net vertical velocity to VelocityControl
        // Vertical positioning is primarily handled by VelocityControl's height controller.
        swarmInput.y = 0.0f;

        var vc = GetComponent<VelocityControl>();
        if (vc != null)
        {
            vc.swarm_vx = swarmInput.x;
            vc.swarm_vy = swarmInput.y; // Will be 0.0f
            vc.swarm_vz = swarmInput.z;        
        }
        
    }
    
    
    // Cohesion force calculation
    public float GetCohesionForce(float r)
    {
        if (float.IsNaN(c) || float.IsInfinity(c)) c = 0; // Basic fallback for c
        if (r0_coh == 0) return 0f; // Avoid division by zero if r0_coh is set to 0

        float neighbourWeightDerivative = GetNeighbourWeightDerivative(r);
        float cohesionIntensityVal = GetCohesionIntensity(r); // Renamed to avoid conflict
        float neighbourWeight = GetNeighbourWeight(r);
        float cohesionIntensityDerivative = GetCohesionIntensityDerivative(r);

        // Ensure r0_coh is not zero before division
        float r0_coh_safe = (r0_coh == 0) ? 1.0f : r0_coh; // Use 1.0f as a fallback if r0_coh is zero, though it should be positive

        return (1.0f / r0_coh_safe) * neighbourWeightDerivative * cohesionIntensityVal + neighbourWeight * cohesionIntensityDerivative;
    }

    // Cohesion intensity function
    public float GetCohesionIntensity(float r)
    {
        if (float.IsNaN(c) || float.IsInfinity(c)) c = 0;
        float diff = r - d_ref;
        float term_sqrt_1_val = 1 + Mathf.Pow(diff + c, 2);
        float term_sqrt_2_val = 1 + c * c;
        term_sqrt_1_val = Mathf.Max(0, term_sqrt_1_val);
        term_sqrt_2_val = Mathf.Max(0, term_sqrt_2_val);
        return ((a + b) / 2.0f) * (Mathf.Sqrt(term_sqrt_1_val) - Mathf.Sqrt(term_sqrt_2_val)) + ((a - b) * diff) / 2.0f;
    }

    // Derivative of cohesion intensity function
    float GetCohesionIntensityDerivative(float r)
    {
        if (float.IsNaN(c) || float.IsInfinity(c)) c = 0;
        float diff = r - d_ref;
        float denominator_val = 1 + Mathf.Pow(diff + c, 2);
        if (denominator_val <= 0) return (a - b) / 2.0f;
        return ((a + b) / 2.0f) * (diff + c) / Mathf.Sqrt(denominator_val) + (a - b) / 2.0f;
    }
    
    // Neighbor weight function
    public float GetNeighbourWeight(float r)
    {
        if (r0_coh == 0) return 0f;
        float r_ratio_val = r / r0_coh;
        if (r_ratio_val < delta) return 1.0f;
        if (r_ratio_val < 1.0f)
        {
            if (1.0f - delta == 0) return 0f;
            float arg = Mathf.PI * (r_ratio_val - delta) / (1.0f - delta);
            return Mathf.Pow(0.5f * (1.0f + Mathf.Cos(arg)), 2);
        }
        return 0.0f;
    }

    // Derivative of neighbor weight function
    float GetNeighbourWeightDerivative(float r)
    {
        if (r0_coh == 0) return 0f;
        float r_ratio_val = r / r0_coh;
        if (r_ratio_val < delta) return 0.0f;
        if (r_ratio_val < 1.0f)
        {
            if (1.0f - delta == 0) return 0f;
            float arg = Mathf.PI * (r_ratio_val - delta) / (1.0f - delta);
            // Derivative w.r.t. 'r' (scaled_distance)
            return -0.5f * (1.0f + Mathf.Cos(arg)) * Mathf.Sin(arg) * (Mathf.PI / (1.0f - delta)) * (1.0f / r0_coh);
        }
        return 0.0f;
    }
    
    // Data structure for serialization
    [Serializable]
    private class OlfatiSaberData
    {
        // Drone identification
        public string droneId;
        public float timestamp;
        
        // Core model parameters
        public float d_ref;
        public float r0_coh;
        public float delta;
        public float a;
        public float b;
        public float c;
        public float gamma;
        public float c_vm;
        
        // Hand interaction parameters
        public float baselineLength;
        public float baselineWidth;
        public float handCohesionAxisModifier;
        
        // Current forces as Vector3 (will be serialized as x,y,z)
        public Vector3 velocityMatching;
        public Vector3 cohesion;
        public Vector3 obstacle;
        public Vector3 swarmInput;
        
        // Position and desired state
        public Vector3 position;
        public float desired_vx;
        public float desired_vy;
        public float desired_vz;
        public float desired_height;
    }
}