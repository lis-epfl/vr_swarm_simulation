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

    [Header("Olfati-Saber Model Parameters")]
    public float d_ref = 7.0f;
    public float r0_coh = 150.0f;
    public float delta = 0.1f;
    public float a = 0.9f;
    public float b = 1.5f;
    public float c;
    public float gamma = 1.0f;
    public float c_vm = 1.0f;
    public float scaleFactor = 10.0f;

    [Header("Hand Interaction Parameters")]
    public float baselineLength = 0.5f; // Example: typical distance between palms for neutral scaling
    public float baselineWidth = 0.1f;  // Example: typical hand width measure for neutral scaling
    public float handCohesionAxisModifier = 1.0f; // Similar to Reynolds, modulates cohesion scaling by hands

    [Header("Migration and Obstacle Avoidance")]
    public float maxMigrationDistance = 10.0f;
    public float detectionRadius = 5.0f;
    public float obstacleAvoidanceForceWeight = 2.0f;
    public float maxAvoidForce = 10.0f;
    public string obstacleTag = "Obstacle";

    [Header("Desired State Inputs (for Velocity Matching)")]
    public float desired_vx = 0.0f;
    public float desired_vy = 0.0f; // This is mapped to world Z for desired_velocity
    public float desired_vz = 0.0f; // This is mapped to world Y for desired_velocity
    public float desired_yaw = 0.0f;
    public float desired_height = 4.0f;  // Add this line - default value matches typical VelocityControl value
    
    private Vector3 velocityMatching = Vector3.zero;
    private Vector3 cohesion = Vector3.zero;
    private Vector3 obstacle = Vector3.zero;
    private Vector3 swarmInput = Vector3.zero;

    private string droneName; // Class member for drone name

    [Header("UDP Network Settings")]
    public bool enableUdpTransmission = true;
    public string udpTargetIP = "127.0.0.1";
    public int udpTargetPort = 8051;
    public float udpTransmitFrequency = 0.1f; // seconds between transmissions (10 Hz)
    
    private UdpClient udpClient;
    private IPEndPoint remoteEndPoint;
    private float udpTransmitTimer = 0f;

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
        
        // Initialize UDP client
        if (enableUdpTransmission)
        {
            try
            {
                udpClient = new UdpClient();
                remoteEndPoint = new IPEndPoint(IPAddress.Parse(udpTargetIP), udpTargetPort);
                Debug.Log($"UDP client initialized for {droneName}. Target: {udpTargetIP}:{udpTargetPort}");
            }
            catch (Exception e)
            {
                Debug.LogError($"Failed to initialize UDP client: {e.Message}");
                enableUdpTransmission = false;
            }
        }
    }

    void OnDestroy()
    {
        // Clean up UDP client
        if (udpClient != null)
        {
            udpClient.Close();
            udpClient = null;
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
        
        // --- Hand Interaction: Scale Cohesion (similar to Reynolds) ---
        if (HandProcessor.ArePalmsTracked)
        {
            Vector3 leftPalmPos = HandProcessor.LeftPalmPosition;
            Vector3 rightPalmPos = HandProcessor.RightPalmPosition;
            float currentHandEllipsoidLength = HandProcessor.HandEllipsoidLength; // Used for lengthFactor
            float currentHandEllipsoidWidth = HandProcessor.HandEllipsoidWidth;   // Used for widthFactor

            Quaternion handOrientation = Quaternion.identity;
            Vector3 handVector = rightPalmPos - leftPalmPos;

            if (handVector.sqrMagnitude > 0.001f)
            {
                handOrientation = Quaternion.LookRotation(handVector.normalized, Vector3.up);
            }

            float lengthFactor = (baselineLength > 0.001f && currentHandEllipsoidLength > 0) ? currentHandEllipsoidLength / baselineLength : 1.0f;
            float widthFactor = (baselineWidth > 0.001f && currentHandEllipsoidWidth > 0) ? currentHandEllipsoidWidth / baselineWidth : 1.0f;

            lengthFactor = Mathf.Clamp(lengthFactor, 0.1f, 10.0f);
            widthFactor = Mathf.Clamp(widthFactor, 0.1f, 10.0f);

            // Transform cohesion to hand-local space
            Vector3 localCohesion = Quaternion.Inverse(handOrientation) * cohesion;

            // Scale components in hand-local space
            // Division means force decreases as factor increases (hands expand)
            float safeAxisModifier = Mathf.Max(0.001f, handCohesionAxisModifier); // Avoid division by zero

            localCohesion.x /= (widthFactor * safeAxisModifier);
            localCohesion.y /= (widthFactor * safeAxisModifier); // Scaling local Y by widthFactor
            localCohesion.z /= (lengthFactor * safeAxisModifier); // Scaling local Z by lengthFactor

            // Transform scaled cohesion back to world space
            cohesion = handOrientation * localCohesion;
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
        
        // UDP transmission logic
        if (enableUdpTransmission)
        {
            udpTransmitTimer += Time.fixedDeltaTime;
            if (udpTransmitTimer >= udpTransmitFrequency)
            {
                SendOlfatiSaberParameters();
                udpTransmitTimer = 0f;
            }
        }
    }
    
    private void SendOlfatiSaberParameters()
    {
        try
        {
            // Create a structured JSON message containing model parameters and state
            string jsonMessage = JsonUtility.ToJson(new OlfatiSaberData
            {
                droneId = this.droneName,
                timestamp = Time.time,
                
                // Model parameters
                d_ref = this.d_ref,
                r0_coh = this.r0_coh,
                delta = this.delta,
                a = this.a,
                b = this.b,
                c = this.c,
                gamma = this.gamma,
                c_vm = this.c_vm,
                
                // Hand interaction parameters
                baselineLength = this.baselineLength,
                baselineWidth = this.baselineWidth,
                handCohesionAxisModifier = this.handCohesionAxisModifier,
                
                // Current forces
                velocityMatching = this.velocityMatching,
                cohesion = this.cohesion,
                obstacle = this.obstacle,
                swarmInput = this.swarmInput,
                
                // Position and desired state
                position = transform.position,
                desired_vx = this.desired_vx,
                desired_vy = this.desired_vy,
                desired_vz = this.desired_vz,
                desired_height = this.desired_height
            });
            
            // Convert JSON string to bytes and send
            byte[] data = Encoding.UTF8.GetBytes(jsonMessage);
            udpClient.Send(data, data.Length, remoteEndPoint);
            
            if (this.droneName == "Drone 0") // Only log for one drone to avoid console spam
            {
                Debug.Log($"UDP sent {data.Length} bytes for {droneName}");
            }
        }
        catch (Exception e)
        {
            Debug.LogError($"Error sending UDP data: {e.Message}");
        }
    }
    
    // Cohesion force calculation
    public float GetCohesionForce(float r)
    {
        if (float.IsNaN(c) || float.IsInfinity(c)) c = 0; // Basic fallback for c
        if (r0_coh == 0) return 0f;

        float neighbourWeightDerivative = GetNeighbourWeightDerivative(r);
        float cohesionIntensityVal = GetCohesionIntensity(r); // Renamed to avoid conflict
        float neighbourWeight = GetNeighbourWeight(r);
        float cohesionIntensityDerivative = GetCohesionIntensityDerivative(r);

        return (1.0f / r0_coh) * neighbourWeightDerivative * cohesionIntensityVal + neighbourWeight * cohesionIntensityDerivative;
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