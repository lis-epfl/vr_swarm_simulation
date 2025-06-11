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
    public float d_ref_x = 7.0f; // Base for hand's local X (sideways)
    public float d_ref_y = 7.0f; // Base for world Y (vertical)
    public float d_ref_z = 7.0f; // Base for hand's local Z (along direction)
    
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
    public float detectionRadius = 5.0f;
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

    // Store initial base reference distances
    private float base_d_ref_x;
    private float base_d_ref_y;
    private float base_d_ref_z;

    // Public properties to expose calculated factors
    public float CurrentWidthFactor { get; private set; } = 1.0f;
    public float CurrentLengthFactor { get; private set; } = 1.0f;
    private float currentHandYawAngle = 0.0f; // Stores the yaw angle of the hands

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

        // Store the initial values from the inspector to use as baselines for scaling
        base_d_ref_x = d_ref_x;
        base_d_ref_y = d_ref_y;
        base_d_ref_z = d_ref_z;
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

        // Calculate velocity matching
        Vector3 desired_velocity_for_vm = new Vector3(desired_vx, desired_vz, -desired_vy); // Assuming desired_vy from input is world Z, desired_vz is world Y
        velocityMatching = c_vm * (desired_velocity_for_vm - velocity);

        // Update reference distances based on hand tracking
        UpdateReferenceDistancesFromHands();

        // Calculate cohesion forces using axis-specific reference distances
        if (swarm != null)
        {
            foreach (GameObject neighbourGO in swarm)
            {
                if (neighbourGO == null) continue;

                // Assuming the OlfatiSaber script is on the "DroneParent" which is a child of the main swarm list GameObjects
                Transform neighbourChildTransform = neighbourGO.transform.Find("DroneParent"); 
                if (neighbourChildTransform == null)
                {
                    // Fallback if "DroneParent" is not found, try getting OlfatiSaber from neighbourGO directly
                    // This depends on your hierarchy. If OlfatiSaber is on neighbourGO:
                    // OlfatiSaber neighbourOlfati = neighbourGO.GetComponent<OlfatiSaber>();
                    // if (neighbourOlfati == null || neighbourOlfati.gameObject == gameObject) continue;
                    // neighbourPosition = neighbourGO.transform.position;

                    // Using the provided structure:
                    // Debug.LogWarning($"DroneParent not found on {neighbourGO.name}. Skipping cohesion calculation for this neighbour.");
                    continue;
                }
                GameObject neighbourChild = neighbourChildTransform.gameObject;
            
                if (neighbourChild == gameObject) continue; // Don't calculate cohesion with self

                Vector3 neighbourPosition = neighbourChild.transform.position;
                Vector3 relativePosition = neighbourPosition - position;
                float distance = relativePosition.magnitude;

                if (distance < 0.001f) distance = 0.001f; // Avoid division by zero

                // Calculate cohesion force for each axis separately
                Vector3 individualCohesionForce = GetAxisSpecificCohesionForce(relativePosition, distance);
                cohesion += individualCohesionForce;
            }
        }

        swarmInput = velocityMatching + cohesion + obstacle;
        swarmInput.y = 0.0f; // RESTORED: Ensure no vertical swarm input from OlfatiSaber

        var vc = GetComponent<VelocityControl>();
        if (vc != null)
        {
            vc.swarm_vx = swarmInput.x;
            vc.swarm_vy = swarmInput.y; // This will now be 0.0f
            vc.swarm_vz = swarmInput.z;
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

            Vector3 handVector = rightPalmPos - leftPalmPos;

            // Calculate scaling factors
            CurrentLengthFactor = (baselineLength > 0.001f && currentHandEllipsoidLength > 0) ? 
                currentHandEllipsoidLength / baselineLength : 1.0f;
            CurrentWidthFactor = (baselineWidth > 0.001f && currentHandEllipsoidWidth > 0) ? 
                currentHandEllipsoidWidth / baselineWidth : 1.0f;

            // Clamp factors
            CurrentLengthFactor = Mathf.Clamp(CurrentLengthFactor, 0.1f, 10.0f);
            CurrentWidthFactor = Mathf.Clamp(CurrentWidthFactor, 0.1f, 10.0f);

            // Calculate hand yaw angle based on X-Z plane projection
            Vector3 handVectorXZ = new Vector3(handVector.x, 0f, handVector.z);
            if (handVectorXZ.sqrMagnitude > 0.001f)
            {
                currentHandYawAngle = Vector3.SignedAngle(Vector3.forward, handVectorXZ.normalized, Vector3.up);
            }
            else
            {
                currentHandYawAngle = 0f; // Default yaw if hands are vertically aligned or too close
            }

            // Update d_ref values based on factors
            // d_ref_x is for the hand's local X axis (scales with width)
            d_ref_x = base_d_ref_x * CurrentWidthFactor;
            // d_ref_y is for the world Y axis (vertical).
            // It's set to base_d_ref_y. If you want no vertical cohesion effect from d_ref_y,
            // ensure base_d_ref_y is initialized from an Inspector value of d_ref_y that leads to
            // zero force (e.g., if d_ref_y itself was 0, or handled by swarmInput.y = 0 above).
            // For now, we rely on swarmInput.y = 0.0f to nullify vertical cohesion.
            d_ref_y = base_d_ref_y; 
            // d_ref_z is for the hand's local Z axis (scales with length)
            d_ref_z = base_d_ref_z * CurrentLengthFactor;

            // Ensure distances don't become negative or zero
            d_ref_x = Mathf.Max(0.01f, d_ref_x);
            d_ref_y = Mathf.Max(0.01f, d_ref_y);
            d_ref_z = Mathf.Max(0.01f, d_ref_z);

            // Debug.Log($"Drone {droneName} - HandYaw: {currentHandYawAngle:F2}, d_ref_x: {d_ref_x:F2}, d_ref_y: {d_ref_y:F2}, d_ref_z: {d_ref_z:F2}");
        }
        else
        {
            // Reset to initial base values and yaw when hands not tracked
            d_ref_x = base_d_ref_x;
            d_ref_y = base_d_ref_y; // d_ref_y will be its base value (e.g. 7.0f)
            d_ref_z = base_d_ref_z;
            CurrentLengthFactor = 1.0f;
            CurrentWidthFactor = 1.0f;
            currentHandYawAngle = 0.0f;
        }
    }

    // NEW: Calculate cohesion force using axis-specific reference distances and hand yaw
    private Vector3 GetAxisSpecificCohesionForce(Vector3 relativePosition, float distance)
    {
        // Create rotation based on hand yaw
        Quaternion handYawRotation = Quaternion.Euler(0, currentHandYawAngle, 0);
        // Transform relative position to hand-oriented local space (for X and Z axes)
        Vector3 localRelativePosition = Quaternion.Inverse(handYawRotation) * relativePosition;

        // Scaled distances for cohesion functions
        // X and Z are in the hand-rotated frame, Y is world vertical
        float scaled_distance_hand_x = Mathf.Abs(localRelativePosition.x / scaleFactor);
        float scaled_distance_world_y = Mathf.Abs(relativePosition.y / scaleFactor); // Use world relativePosition.y
        float scaled_distance_hand_z = Mathf.Abs(localRelativePosition.z / scaleFactor);

        // Calculate force magnitudes along each axis
        // d_ref_ values are in world units, so divide by scaleFactor for cohesion functions
        float forceHandX = GetCohesionForceWithCustomDRef(scaled_distance_hand_x, d_ref_x / scaleFactor);
        float forceWorldY = GetCohesionForceWithCustomDRef(scaled_distance_world_y, d_ref_y / scaleFactor);
        float forceHandZ = GetCohesionForceWithCustomDRef(scaled_distance_hand_z, d_ref_z / scaleFactor);

        // Construct force components
        // X and Z components are in the local hand-oriented frame
        Vector3 localXZForceComponent = Vector3.zero;
        localXZForceComponent.x = forceHandX * Mathf.Sign(localRelativePosition.x);
        localXZForceComponent.z = forceHandZ * Mathf.Sign(localRelativePosition.z);
        
        // Y component is in world frame
        float worldYForce = forceWorldY * Mathf.Sign(relativePosition.y);

        // Rotate XZ force components from local hand-oriented frame back to world space
        Vector3 worldXZForce = handYawRotation * localXZForceComponent;

        // Combine with world Y force
        Vector3 totalWorldCohesionForce = new Vector3(worldXZForce.x, worldYForce, worldXZForce.z);
        
        return totalWorldCohesionForce * cohesionMultiplier; // Apply cohesionMultiplier here
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