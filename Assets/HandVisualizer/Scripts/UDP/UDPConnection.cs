using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.Net;
using System.Net.Sockets;
using System.Text;

// Data structure for Reynolds axis-specific weights to be sent via UDP
[System.Serializable]
public class ReynoldsAxisWeightsUdpData
{
    public Vector3 cohesionWeights; // X, Y, Z cohesion weights
    public Vector3 separationWeights; // X, Y, Z separation weights
    // You can add other general parameters here if needed in the futur

}

public class UDPConnection : MonoBehaviour
{
    [Header("UDP Settings")]
    public string targetIPAddress = "192.168.100.176"; // Target IP address
    public int targetPort = 11002;             // Target port (ensure it's different if Reynolds still sends)

    [Header("Reynolds General Weights (Per Axis)")]
    public Vector3 cohesionWeights = new Vector3(1.0f, 1.0f, 1.0f);
    public Vector3 separationWeights = new Vector3(1.0f, 1.0f, 1.0f);

    // Time interval between sending UDP packets (in seconds)
    public float sendInterval = 0.1f; // Send 10 times per second

    public List<GameObject> swarm;



    private float timeSinceLastSend = 0f;
    private UdpClient udpClient;
    private GameObject drone0;
    private Reynolds reynoldsScript; // Reference to the Reynolds script on drone0

    void Start()
    {        
        try
        {
            udpClient = new UdpClient();
            Debug.Log($"UDP_Connection: UDP client initialized. Target: {targetIPAddress}:{targetPort}");
            Debug.LogWarning("UDP_Connection: UDP sends data over the network to an IP Address and Port.");
        }
        catch (System.Exception e)
        {
            Debug.LogError($"UDP_Connection: Failed to initialize UDP client: {e.Message}");
            udpClient = null; // Ensure client is null if initialization failed
            this.enabled = false; // Disable script if UDP can't be set up
        }

        // Set the Reynolds script reference
        SetReynoldsScript();
    }

    void Update()
    {
        if (udpClient == null)
        {
            return; // Don't proceed if UDP client isn't initialized
        }

        if (reynoldsScript == null)
        {
            SetReynoldsScript(); // Try to set the Reynolds script again
            if (reynoldsScript == null)
            {
                Debug.Log("UDP_Connection: Reynolds script is not set. Cannot send data.");
               
                return; // Exit if Reynolds script is still not set
            }
        }

        timeSinceLastSend += Time.deltaTime;

        if (timeSinceLastSend >= sendInterval)
        {
            SendReynoldsWeights();
            timeSinceLastSend = 0f; // Reset timer
        }
    }

    void SendReynoldsWeights()
    {
        ReynoldsAxisWeightsUdpData parametersToSend = new ReynoldsAxisWeightsUdpData
        {
            cohesionWeights = reynoldsScript.scaledWorldCohesion,
            separationWeights = reynoldsScript.scaledWorldSeparation
        };

        string jsonParameters = JsonUtility.ToJson(parametersToSend);
        byte[] data = Encoding.UTF8.GetBytes(jsonParameters);

        try
        {
            udpClient.Send(data, data.Length, targetIPAddress, targetPort);
            //Debug.Log($"UDP_Connection: Sent Reynolds axis weights to {targetIPAddress}:{targetPort}: {jsonParameters}");
        }
        catch (SocketException e)
        {
            Debug.LogError($"UDP_Connection: SocketException sending UDP data: {e.Message} to {targetIPAddress}:{targetPort}");
        }
        catch (System.Exception e)
        {
            Debug.LogError($"UDP_Connection: Error sending UDP data: {e.Message}");
        }
    }

    void OnDestroy()
    {
        if (udpClient != null)
        {
            udpClient.Close();
            udpClient = null;
            Debug.Log("UDP_Connection: UDP client closed.");
        }
    }

    // Optional: Method to update weights dynamically if needed from other scripts
    public void UpdateWeights(Vector3 newCohesionWeights, Vector3 newSeparationWeights)
    {
        this.cohesionWeights = newCohesionWeights;
        this.separationWeights = newSeparationWeights;
    }

    public void SetReynoldsScript()
    {
        // Get drone0 from the swarm list
        if (swarm != null && swarm.Count > 0)
        {
            drone0 = swarm[0];
            // get the droneParent child
            Transform droneParentTransform = drone0.transform.Find("DroneParent");
            // Get the reynolds script from drone0
            reynoldsScript = droneParentTransform.GetComponent<Reynolds>();
        }
        else
        {
            Debug.LogError("UDP_Connection: Swarm list is empty or not assigned.");
        }
    }
}
