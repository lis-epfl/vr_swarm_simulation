using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.XR.Hands.Samples.VisualizerSample; // To access HandProcessor
using System.Net;
using System.Net.Sockets;
using System.Text;

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
    public Vector3 scaledWorldSeparation = Vector3.zero;
    public Vector3 scaledWorldCohesion = Vector3.zero;

    private Vector3 cohesion = Vector3.zero;
    private Vector3 separation = Vector3.zero;
    private Vector3 alignment = Vector3.zero;
    private Vector3 swarmInput = Vector3.zero;

    // Define baseline values (adjust based on your expectations)
    private float baselineLength = 0.5f; // example baseline for palm distance
    private float baselineWidth  = 0.1f; // example baseline for width scale

    // [Header("UDP Settings")]
    // public string targetIPAddress = "192.168.100.176"; // IP address of the receiver
    // public int targetPort = 11001; // Port for Reynolds data (use a different port than OlfatiSaber if sending to same IP)
    // private UdpClient udpClient;
    // private string droneName; // To identify which drone is sending, if needed for logging

    void Start()
    {
        // droneName = transform.parent != null ? transform.parent.name : gameObject.name;

        // Initialize UDP client
        // try
        // {
        //     udpClient = new UdpClient();
        //     Debug.Log($"Reynolds UDP client initialized for {droneName}. Target: {targetIPAddress}:{targetPort}");
        //     Debug.LogWarning("UDP sends data over the network to an IP Address and Port.");

        //     // Send initial parameters
        //     //SendCurrentParameters();
        // }
        // catch (System.Exception e)
        // {
        //     Debug.LogError($"Failed to initialize Reynolds UDP client for {droneName}: {e.Message}");
        //     udpClient = null; // Ensure client is null if initialization failed
        // }
    }

    // public void SendCurrentParameters()
    // {
    //     if (udpClient == null)
    //     {
    //         Debug.LogWarning($"Reynolds UDP client not initialized for {droneName}. Cannot send parameters.");
    //         return;
    //     }

    //     ReynoldsUdpParameters parametersToSend = new ReynoldsUdpParameters
    //     {
    //         cohesionWeight = this.cohesionWeight,
    //         separationWeight = this.separationWeight,
    //         alignmentWeight = this.alignmentWeight,
    //         cohesionMultiplier = this.cohesionMultiplier,
    //         separationMultiplier = this.separationMultiplier,
    //         cohesionAxisModifier = this.cohesionAxisModifier,
    //         separationAxisModifier = this.separationAxisModifier
    //     };

    //     string jsonParameters = JsonUtility.ToJson(parametersToSend);
    //     byte[] data = Encoding.UTF8.GetBytes(jsonParameters);

    //     try
    //     {
    //         udpClient.Send(data, data.Length, targetIPAddress, targetPort);
    //         // Debug.Log($"Sent Reynolds parameters from {droneName} to {targetIPAddress}:{targetPort}: {jsonParameters}");
    //     }
    //     catch (SocketException e)
    //     {
    //         Debug.LogError($"SocketException sending Reynolds UDP data from {droneName}: {e.Message} to {targetIPAddress}:{targetPort}");
    //     }
    //     catch (System.Exception e)
    //     {
    //         Debug.LogError($"Error sending Reynolds UDP data from {droneName}: {e.Message}");
    //     }
    // }

    // void OnDestroy()
    // {
    //     if (udpClient != null)
    //     {
    //         udpClient.Close();
    //         udpClient = null;
    //         Debug.Log($"Reynolds UDP client closed for {droneName}.");
    //     }
    // }

    void FixedUpdate()
    {
        // Reset the vectors
        cohesion = Vector3.zero;
        separation = Vector3.zero;
        alignment = Vector3.zero;       

        int neighbourCount = 0; // Keep track of actual neighbours processed

        Rigidbody selfRb = GetComponent<Rigidbody>(); // Get self Rigidbody once
        if (selfRb == null)
        {
            // Debug.LogError($"Reynolds script on {gameObject.name} is missing a Rigidbody.", this);
            return; // Cannot perform calculations without self Rigidbody
        }


        // Calculate the relative position and velocity of each drone to the current drone
        if (swarm != null) // Check if swarm list is assigned
        {
            foreach (GameObject neighbour in swarm)
            {
                if (neighbour == null) continue; // Skip null entries in swarm list

                // Ensure neighbour has DroneParent and Rigidbody, and isn't self
                Transform neighbourChildTransform = neighbour.transform.Find("DroneParent");
                if (neighbourChildTransform == null) continue; 
                GameObject neighbourChild = neighbourChildTransform.gameObject;
                if (neighbourChild == gameObject) continue; // Use direct comparison
                
                Rigidbody neighbourRb = neighbourChild.GetComponent<Rigidbody>();
                if (neighbourRb == null) continue;

                neighbourCount++; // Increment count only for valid neighbours

                Vector3 neighbourPosition = neighbourChild.transform.position;
                Vector3 relativePosition = neighbourPosition - transform.position; // Position of this GameObject
                float distance = relativePosition.magnitude;
                
                if (distance < 0.001f) distance = 0.001f; 
                
                Vector3 relativeVelocity = neighbourRb.velocity - selfRb.velocity;

                cohesion += relativePosition;
                separation -= relativePosition.normalized / distance; 
                alignment += relativeVelocity; // In Reynolds, this is often neighbourRb.velocity, not relative.
                                               // Using relativeVelocity as per your existing code.
            }
        }


        // Average the forces only if there were neighbours
        if(neighbourCount > 0)
        {
            cohesion /= neighbourCount;
            separation /= neighbourCount;
            alignment /= neighbourCount;
        }

        // --- Hand Orientation and Scaling ---
        Vector3 leftPalmPos = Vector3.zero;
        Vector3 rightPalmPos = Vector3.zero;
        float handLength = 0f;
        float handWidth = 0f;
        bool palmsTracked = false;

        if (HandProcessor.ArePalmsTracked) // Check if HandProcessor has valid data
        {
            leftPalmPos = HandProcessor.LeftPalmPosition;
            rightPalmPos = HandProcessor.RightPalmPosition;
            handLength = HandProcessor.HandEllipsoidLength;
            handWidth = HandProcessor.HandEllipsoidWidth;
            palmsTracked = true;
        }
        
        Quaternion handOrientation = Quaternion.identity; 

        if (palmsTracked)
        {
             Vector3 handVector = rightPalmPos - leftPalmPos;
             if (handVector.sqrMagnitude > 0.001f)
             {
                 handOrientation = Quaternion.LookRotation(handVector.normalized, Vector3.up);
             }
        }
        
        float lengthFactor = (baselineLength > 0.001f && handLength > 0) ? handLength / baselineLength : 1.0f;
        float widthFactor  = (baselineWidth > 0.001f && handWidth > 0)  ? handWidth / baselineWidth   : 1.0f;

        lengthFactor = Mathf.Clamp(lengthFactor, 0.1f, 10.0f);
        widthFactor = Mathf.Clamp(widthFactor, 0.1f, 10.0f);

        Vector3 localCohesion = Quaternion.Inverse(handOrientation) * cohesion;
        Vector3 localSeparation = Quaternion.Inverse(handOrientation) * separation;
        
        // Apply Axis Modifiers carefully. Division for cohesion, multiplication for separation.
        // Ensure modifiers are not zero if used in denominator.
        float safeCohesionAxisModifier = Mathf.Max(0.001f, cohesionAxisModifier); // Avoid division by zero
        
        localCohesion.x /= (widthFactor * safeCohesionAxisModifier);
        localCohesion.y /= (widthFactor * safeCohesionAxisModifier);
        localCohesion.z /= (lengthFactor * safeCohesionAxisModifier);

        localSeparation.x *= (widthFactor * separationAxisModifier);
        localSeparation.y *= (widthFactor * separationAxisModifier);
        localSeparation.z *= (lengthFactor * separationAxisModifier);

        scaledWorldCohesion = handOrientation * localCohesion*cohesionWeight * cohesionMultiplier;
        scaledWorldSeparation = handOrientation * localSeparation*separationWeight * separationMultiplier;

        // scaledWorldCohesion *= cohesionWeight * cohesionMultiplier;
        // scaledWorldSeparation *= separationWeight * separationMultiplier;
        alignment *= alignmentWeight; 

        swarmInput = scaledWorldCohesion + scaledWorldSeparation + alignment;

        var vc = GetComponent<VelocityControl>();
        if(vc != null)
        {
            vc.swarm_vx = swarmInput.x;
            vc.swarm_vy = swarmInput.y;
            vc.swarm_vz = swarmInput.z;
        }
        else
        {
            // Debug.LogWarning($"VelocityControl script not found on {gameObject.name}", this);
        }
    }
}

// Data structure for Reynolds parameters to be sent via UDP
// [System.Serializable]
// public class ReynoldsUdpParameters
// {
//     public float cohesionWeight;
//     public float separationWeight;
//     public float alignmentWeight;
//     public float cohesionMultiplier;
//     public float separationMultiplier;
//     public float cohesionAxisModifier;
//     public float separationAxisModifier;
//     // Add any other parameters you want to send
// }

