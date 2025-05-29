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
    private Vector3 cohesion = Vector3.zero;

    [Header("Olfati-Saber Model Parameters")]
    public float d_ref = 300.0f; // Keep as fallback/baseline
    
    // NEW: Axis-specific reference distances
    [Header("Axis-Specific Reference Distances")]
    public float d_ref_x = 7.0f;
    public float d_ref_y = 7.0f;
    public float d_ref_z = 7.0f;
    
    public float r0_coh = 20.0f;
    public float delta = 0.1f;
    public float a = 0.9f;
    public float b = 1.5f;
    public float c;
    public float gamma = 1.0f;
    public float c_vm = 1.0f;
    public float scaleFactor = 10.0f;
    public float cohesionMultiplier = 2.0f;

    [Header("Hand Interaction Parameters")]
    public float baselineLength = 0.5f;
    public float baselineWidth = 0.1f;
    public float handDistanceScaleFactor = 0.001f; // How much hand movement affects distance

    [Header("Migration and Obstacle Avoidance")]
    public float maxMigrationDistance = 10.0f;
    public float detectionRadius = 7.5f;
    public float obstacleAvoidanceForceWeight = 2.0f;
    public float maxAvoidForce = 10.0f;
    public string obstacleTag = "Obstacle";

    [Header("Desired State Inputs (for Velocity Matching)")]
    public float desired_vx = 0.0f;
    public float desired_vy = 0.0f;
    public float desired_vz = 0.0f;
    public float desired_yaw = 0.0f;
    public float desired_height = 4.0f;
    
    private Vector3 velocityMatching = Vector3.zero;
    private Vector3 obstacle = Vector3.zero;
    private Vector3 swarmInput = Vector3.zero;
    private string droneName;

    // Public properties to expose calculated factors
    public float CurrentWidthFactor { get; private set; } = 1.0f;
    public float CurrentLengthFactor { get; private set; } = 1.0f;

    void Start()
    {
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

        Vector3 position = transform.position;
        Rigidbody rb = GetComponent<Rigidbody>();
        if (rb == null) return;
        Vector3 velocity = rb.velocity;

        // --- 1. Rotate Target Velocity for Velocity Matching ---
        // desired_vx is along swarm's local X (side), desired_vz is swarm's local Y (up/down), -desired_vy is swarm's local Z (forward)
        Vector3 localDesiredVelocityForVM = new Vector3(desired_vx, desired_vz, -desired_vy); 
        Quaternion swarmYawRotation = Quaternion.Euler(0, desired_yaw, 0); // Rotation based on hand yaw input
        Vector3 worldDesiredVelocityForVM = swarmYawRotation * localDesiredVelocityForVM;
        velocityMatching = c_vm * (worldDesiredVelocityForVM - velocity);

        // Update reference distances based on hand tracking (magnitudes of d_ref_x,y,z)
        UpdateReferenceDistancesFromHands();

        // Calculate cohesion forces using axis-specific reference distances in the swarm's yawed frame
        if (swarm != null)
        {
            Quaternion inverseSwarmYawRotation = Quaternion.Inverse(swarmYawRotation);
            foreach (GameObject neighbourGO in swarm)
            {
                if (neighbourGO == null) continue;

                // Assuming OlfatiSaber script is on the same GameObject as the Rigidbody and represents the drone's center
                // If DroneParent is a child visual, use neighbourGO.transform.position if OlfatiSaber is on the root drone object.
                // For consistency, let's assume OlfatiSaber is on the object whose position matters for swarming.
                // The original code used neighbourGO.transform.Find("DroneParent").gameObject.transform.position
                // Let's stick to that if "DroneParent" is the intended swarming agent center.
                // If OlfatiSaber is on "DroneParent", then neighbourGO is the "DroneParent".
                // If OlfatiSaber is on the root, and "DroneParent" is a child, then we need the "DroneParent" of the neighbour.

                GameObject neighbourSwarmAgent = neighbourGO; // Assuming OlfatiSaber is on the swarming agent itself
                // If your setup requires finding "DroneParent" for each neighbour:
                // Transform neighbourChildTransform = neighbourGO.transform.Find("DroneParent");
                // if (neighbourChildTransform == null) continue;
                // GameObject neighbourSwarmAgent = neighbourChildTransform.gameObject;
            
                if (neighbourSwarmAgent == gameObject) continue; // Don't interact with self

                Vector3 neighbourPosition = neighbourSwarmAgent.transform.position;
                Vector3 relativePositionWorld = neighbourPosition - position; // In world coordinates
                float worldDistance = relativePositionWorld.magnitude;

                if (worldDistance < 0.001f) worldDistance = 0.001f;

                // --- 2. Transform relative position to swarm's yawed frame ---
                Vector3 relativePositionInSwarmFrame = inverseSwarmYawRotation * relativePositionWorld;

                // --- 3. Calculate cohesion force in swarm's yawed frame ---
                // GetAxisSpecificCohesionForce now expects relativePositionInSwarmFrame
                Vector3 cohesionForceInSwarmFrame = GetAxisSpecificCohesionForce(relativePositionInSwarmFrame, worldDistance);
                
                // --- 4. Transform cohesion force back to world space ---
                Vector3 cohesionForceWorld = swarmYawRotation * cohesionForceInSwarmFrame;
                cohesion += cohesionForceWorld;
            }
        }

        swarmInput = velocityMatching + cohesion + obstacle;
        // swarmInput.y = 0.0f; // Original code had this, re-evaluate if vertical swarm control is intended through OlfatiSaber forces

        var vc = GetComponent<VelocityControl>();
        if (vc != null)
        {
            vc.swarm_vx = swarmInput.x;
            vc.swarm_vy = swarmInput.y; // This is world Y for the swarm input
            vc.swarm_vz = swarmInput.z; // This is world Z for the swarm input
        }
    }

    // NEW: Update reference distances based on hand movements
    private void UpdateReferenceDistancesFromHands()
    {
        if (HandProcessor.ArePalmsTracked)
        {
            Vector3 leftPalmPos = HandProcessor.LeftPalmPosition;
            Vector3 rightPalmPos = HandProcessor.RightPalmPosition;
            float currentHandEllipsoidLength = HandProcessor.HandEllipsoidLength;
            float currentHandEllipsoidWidth = HandProcessor.HandEllipsoidWidth;

            Quaternion handOrientation = Quaternion.identity;
            Vector3 handVector = rightPalmPos - leftPalmPos;

            if (handVector.sqrMagnitude > 0.001f)
            {
                handOrientation = Quaternion.LookRotation(handVector.normalized, Vector3.up);
            }

            // Calculate scaling factors
            float lengthFactor = (baselineLength > 0.001f && currentHandEllipsoidLength > 0) ? 
                currentHandEllipsoidLength / baselineLength : 1.0f;
            float widthFactor = (baselineWidth > 0.001f && currentHandEllipsoidWidth > 0) ? 
                currentHandEllipsoidWidth / baselineWidth : 1.0f;

            // Clamp factors
            lengthFactor = Mathf.Clamp(lengthFactor, 0.1f, 10.0f);
            widthFactor = Mathf.Clamp(widthFactor, 0.1f, 10.0f);

            // Update public properties
            CurrentLengthFactor = lengthFactor;
            CurrentWidthFactor = widthFactor;

            // Transform to hand-local space to determine which axis should be affected
            Vector3 localHandDirection = Quaternion.Inverse(handOrientation) * Vector3.forward;

            // Update reference distances based on hand orientation and scaling
            // X and Y axes scale with width factor, Z axis scales with length factor
            d_ref_x = d_ref * widthFactor * handDistanceScaleFactor - 0.3f;
            d_ref_y = d_ref * widthFactor * handDistanceScaleFactor - 0.3f;
            d_ref_z = d_ref * lengthFactor * handDistanceScaleFactor;

            Debug.Log($"Drone {droneName} - d_ref_x: {d_ref_x:F2}, d_ref_y: {d_ref_y:F2}, d_ref_z: {d_ref_z:F2}");

            // Debug output for one drone
            if (this.droneName == "Drone 0")
            {
                //Debug.Log($"Hand factors - Length: {lengthFactor:F2}, Width: {widthFactor:F2}");
                //Debug.Log($"d_ref distances - X: {d_ref_x:F2}, Y: {d_ref_y:F2}, Z: {d_ref_z:F2}");
            }
        }
        else
        {
            // Reset to baseline when hands not tracked
            d_ref_x = d_ref;
            d_ref_y = d_ref;
            d_ref_z = d_ref;
            CurrentLengthFactor = 1.0f; // Reset exposed factors
            CurrentWidthFactor = 1.0f;  // Reset exposed factors
        }
    }

    // MODIFIED: Calculate cohesion force using axis-specific reference distances in the swarm's yawed frame
    private Vector3 GetAxisSpecificCohesionForce(Vector3 relativePositionInSwarmFrame, float worldDistance)
    {
        // normalizedRelativePos should also be in the swarm frame for directional application of axis forces
        Vector3 normalizedRelativePosInSwarmFrame = Vector3.zero;
        if (relativePositionInSwarmFrame.sqrMagnitude > 0.00001f) // Avoid normalization of zero vector
        {
            normalizedRelativePosInSwarmFrame = relativePositionInSwarmFrame.normalized;
        }
        
        Vector3 cohesionForceInSwarmFrame = Vector3.zero;

        // Scale distance components for function calculations based on the swarm's yawed frame
        // These are distances along the swarm's local X, Y, Z axes
        float scaled_distance_swarm_x = Mathf.Abs(relativePositionInSwarmFrame.x / scaleFactor);
        float scaled_distance_swarm_y = Mathf.Abs(relativePositionInSwarmFrame.y / scaleFactor); // Swarm's local Y
        float scaled_distance_swarm_z = Mathf.Abs(relativePositionInSwarmFrame.z / scaleFactor); // Swarm's local Z

        // Calculate force for each axis using its specific reference distance
        // d_ref_x, d_ref_y, d_ref_z are defined relative to the swarm's desired orientation (magnitudes set by hands)
        float forceSwarmX = GetCohesionForceWithCustomDRef(scaled_distance_swarm_x, d_ref_x);
        float forceSwarmY = GetCohesionForceWithCustomDRef(scaled_distance_swarm_y, d_ref_y); 
        float forceSwarmZ = GetCohesionForceWithCustomDRef(scaled_distance_swarm_z, d_ref_z);

        // Apply the force in the direction of each axis component *within the swarm's frame*
        cohesionForceInSwarmFrame.x = forceSwarmX * normalizedRelativePosInSwarmFrame.x;
        cohesionForceInSwarmFrame.y = forceSwarmY * normalizedRelativePosInSwarmFrame.y;
        cohesionForceInSwarmFrame.z = forceSwarmZ * normalizedRelativePosInSwarmFrame.z;

        return cohesionForceInSwarmFrame; // This force is in the swarm's yawed reference frame
    }

    // NEW: Modified cohesion force calculation with custom d_ref
    private float GetCohesionForceWithCustomDRef(float r, float custom_d_ref)
    {
        if (float.IsNaN(c) || float.IsInfinity(c)) c = 0;
        if (r0_coh == 0) return 0f;

        float neighbourWeightDerivative = GetNeighbourWeightDerivative(r);
        float cohesionIntensityVal = GetCohesionIntensityWithCustomDRef(r, custom_d_ref);
        float neighbourWeight = GetNeighbourWeight(r);
        float cohesionIntensityDerivative = GetCohesionIntensityDerivativeWithCustomDRef(r, custom_d_ref);

        float r0_coh_safe = (r0_coh == 0) ? 1.0f : r0_coh;

        return (1.0f / r0_coh_safe) * neighbourWeightDerivative * cohesionIntensityVal + 
               neighbourWeight * cohesionIntensityDerivative;
    }

    // NEW: Cohesion intensity with custom d_ref
    private float GetCohesionIntensityWithCustomDRef(float r, float custom_d_ref)
    {
        if (float.IsNaN(c) || float.IsInfinity(c)) c = 0;
        float diff = r - custom_d_ref;
        float term_sqrt_1_val = 1 + Mathf.Pow(diff + c, 2);
        float term_sqrt_2_val = 1 + c * c;
        term_sqrt_1_val = Mathf.Max(0, term_sqrt_1_val);
        term_sqrt_2_val = Mathf.Max(0, term_sqrt_2_val);
        return ((a + b) / 2.0f) * (Mathf.Sqrt(term_sqrt_1_val) - Mathf.Sqrt(term_sqrt_2_val)) + 
               ((a - b) * diff) / 2.0f;
    }

    // NEW: Cohesion intensity derivative with custom d_ref
    private float GetCohesionIntensityDerivativeWithCustomDRef(float r, float custom_d_ref)
    {
        if (float.IsNaN(c) || float.IsInfinity(c)) c = 0;
        float diff = r - custom_d_ref;
        float denominator_val = 1 + Mathf.Pow(diff + c, 2);
        if (denominator_val <= 0) return (a - b) / 2.0f;
        return ((a + b) / 2.0f) * (diff + c) / Mathf.Sqrt(denominator_val) + (a - b) / 2.0f;
    }

    // Keep original methods for compatibility
    public float GetCohesionForce(float r)
    {
        return GetCohesionForceWithCustomDRef(r, d_ref / scaleFactor);
    }

    public float GetCohesionIntensity(float r)
    {
        return GetCohesionIntensityWithCustomDRef(r, d_ref / scaleFactor);
    }

    float GetCohesionIntensityDerivative(float r)
    {
        return GetCohesionIntensityDerivativeWithCustomDRef(r, d_ref / scaleFactor);
    }
    
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