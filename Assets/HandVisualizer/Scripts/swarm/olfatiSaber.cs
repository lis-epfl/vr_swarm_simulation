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
    public string algorithmType = "Reynolds";
    public Vector3 cohesionWeights; // X, Y, Z cohesion weights
    public Vector3 separationWeights; // X, Y, Z separation weights
}

// Data structure for Olfati-Saber axis-specific reference distances to be sent via UDP
[System.Serializable]
public class OlfatiSaberAxisDistancesUdpData
{
    public string algorithmType = "OlfatiSaber";
    public float d_ref_x; // X axis reference distance
    public float d_ref_y; // Y axis reference distance
    public float d_ref_z; // Z axis reference distance
    public float baseD_ref; // Base reference distance
}

public class UDPConnection : MonoBehaviour
{
    [Header("UDP Settings")]
    public string targetIPAddress = "192.168.100.176"; // Target IP address
    public int targetPort = 11002;             // Target port

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
    private OlfatiSaber olfatiSaberScript; // Reference to the OlfatiSaber script on drone0
    private SwarmManager swarmManager;

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

        // Get SwarmManager reference
        swarmManager = SwarmManager.Instance;
        if (swarmManager == null)
        {
            Debug.LogError("UDP_Connection: SwarmManager instance not found.");
        }

        // Set the algorithm script references
        SetAlgorithmScripts();
    }

    void Update()
    {
        if (udpClient == null || swarmManager == null)
        {
            return; // Don't proceed if UDP client isn't initialized or SwarmManager is missing
        }

        // Check if we need to set algorithm scripts again
        if ((swarmManager.swarmAlgorithm == SwarmManager.SwarmAlgorithm.REYNOLDS && reynoldsScript == null) ||
            (swarmManager.swarmAlgorithm == SwarmManager.SwarmAlgorithm.OLFATI_SABER && olfatiSaberScript == null))
        {
            SetAlgorithmScripts();
        }

        timeSinceLastSend += Time.deltaTime;

        if (timeSinceLastSend >= sendInterval)
        {
            SendSwarmData();
            timeSinceLastSend = 0f; // Reset timer
        }
    }

    void SendSwarmData()
    {
        // Check which algorithm is currently being used
        switch (swarmManager.swarmAlgorithm)
        {
            case SwarmManager.SwarmAlgorithm.REYNOLDS:
                SendReynoldsWeights();
                break;
            case SwarmManager.SwarmAlgorithm.OLFATI_SABER:
                SendOlfatiSaberDistances();
                break;
        }
    }

    void SendReynoldsWeights()
    {
        if (reynoldsScript == null)
        {
            Debug.LogWarning("UDP_Connection: Reynolds script is not set. Cannot send Reynolds data.");
            return;
        }

        ReynoldsAxisWeightsUdpData parametersToSend = new ReynoldsAxisWeightsUdpData
        {
            algorithmType = "Reynolds",
            cohesionWeights = reynoldsScript.scaledWorldCohesion,
            separationWeights = reynoldsScript.scaledWorldSeparation
        };

        string jsonParameters = JsonUtility.ToJson(parametersToSend);
        SendUdpData(jsonParameters);
    }

    void SendOlfatiSaberDistances()
    {
        if (olfatiSaberScript == null)
        {
            Debug.LogWarning("UDP_Connection: OlfatiSaber script is not set. Cannot send OlfatiSaber data.");
            return;
        }

        OlfatiSaberAxisDistancesUdpData parametersToSend = new OlfatiSaberAxisDistancesUdpData
        {
            algorithmType = "OlfatiSaber",
            d_ref_x = olfatiSaberScript.d_ref_x,
            d_ref_y = olfatiSaberScript.d_ref_y,
            d_ref_z = olfatiSaberScript.d_ref_z,
            baseD_ref = olfatiSaberScript.d_ref
        };

        string jsonParameters = JsonUtility.ToJson(parametersToSend);
        SendUdpData(jsonParameters);
    }

    void SendUdpData(string jsonData)
    {
        byte[] data = Encoding.UTF8.GetBytes(jsonData);

        try
        {
            udpClient.Send(data, data.Length, targetIPAddress, targetPort);
            //Debug.Log($"UDP_Connection: Sent data to {targetIPAddress}:{targetPort}: {jsonData}");
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

    public void SetAlgorithmScripts()
    {
        // Get drone0 from the swarm list
        if (swarm != null && swarm.Count > 0)
        {
            drone0 = swarm[0];
            // get the droneParent child
            Transform droneParentTransform = drone0.transform.Find("DroneParent");
            
            if (droneParentTransform != null)
            {
                // Get both algorithm scripts from drone0
                reynoldsScript = droneParentTransform.GetComponent<Reynolds>();
                olfatiSaberScript = droneParentTransform.GetComponent<OlfatiSaber>();
                
                if (reynoldsScript == null)
                {
                    Debug.LogWarning("UDP_Connection: Reynolds script not found on drone0.");
                }
                
                if (olfatiSaberScript == null)
                {
                    Debug.LogWarning("UDP_Connection: OlfatiSaber script not found on drone0.");
                }
            }
            else
            {
                Debug.LogError("UDP_Connection: DroneParent not found on drone0.");
            }
        }
        else
        {
            Debug.LogError("UDP_Connection: Swarm list is empty or not assigned.");
        }
    }
}
