using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.IO; // Required for file operations
using UnityEngine.XR.Hands.Samples.VisualizerSample; // Required for HandProcessor

// Data structure for Reynolds axis-specific weights to be sent via UDP
[System.Serializable]
public class ReynoldsAxisWeightsUdpData
{
    public string algorithmType = "Reynolds";
    public Vector3 cohesionWeights; // X,
    public float swarm_vy; // Commanded swarm velocity Y (vertical)
    public float swarm_vz; // Commanded swarm velocity Z
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
    public float swarm_vx; // Commanded swarm velocity X
    public float swarm_vy; // Commanded swarm velocity Y (vertical)
    public float swarm_vz; // Commanded swarm velocity Z
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

    private StreamWriter csvWriter;
    private string csvFilePath;

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

        // Initialize CSV writer
        InitializeCsvWriter();
    }

    void InitializeCsvWriter()
    {
        csvFilePath = Path.Combine(Application.persistentDataPath, "data_UDP_collection.csv");
        try
        {
            // Create or overwrite the file and write the header
            csvWriter = new StreamWriter(csvFilePath, false); // false to overwrite
            string header = "Timestamp,AlgorithmType," +
                            "LeftPalmPosX,LeftPalmPosY,LeftPalmPosZ," +
                            "RightPalmPosX,RightPalmPosY,RightPalmPosZ," +
                            "WidthFactor,LengthFactor," +
                            "SwarmVX_Command,SwarmVY_Command,SwarmVZ_Command," + // Common velocity commands
                            "CohesionWeightX,CohesionWeightY,CohesionWeightZ," + // Reynolds specific
                            "SeparationWeightX,SeparationWeightY,SeparationWeightZ," + // Reynolds specific
                            "DRefX,DRefY,DRefZ,BaseDRef"; // Olfati-Saber specific
            csvWriter.WriteLine(header);
            csvWriter.Flush(); // Ensure header is written immediately
            Debug.Log($"UDP_Connection: CSV logging initialized. File at: {csvFilePath}");
        }
        catch (System.Exception e)
        {
            Debug.LogError($"UDP_Connection: Failed to initialize CSV writer: {e.Message}");
            csvWriter = null;
        }
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
        // olfatiSaberScript is needed for desired velocities and factors
        // if (olfatiSaberScript == null) 
        // {
        //     Debug.LogWarning("UDP_Connection: OlfatiSaber script is not set. Cannot send desired swarm velocities or hand factors for Reynolds packet.");
        // }

        ReynoldsAxisWeightsUdpData parametersToSend = new ReynoldsAxisWeightsUdpData
        {
            algorithmType = "Reynolds",
            cohesionWeights = reynoldsScript.scaledWorldCohesion,
            separationWeights = reynoldsScript.scaledWorldSeparation,
            swarm_vx = olfatiSaberScript != null ? olfatiSaberScript.desired_vx : 0f,
            swarm_vy = olfatiSaberScript != null ? olfatiSaberScript.desired_vz : 0f,
            swarm_vz = olfatiSaberScript != null ? -olfatiSaberScript.desired_vy : 0f
        };

        string jsonParameters = JsonUtility.ToJson(parametersToSend);
        SendUdpData(jsonParameters);

        // Log to CSV
        if (csvWriter != null)
        {
            Vector3 leftPalmPos = HandProcessor.ArePalmsTracked ? HandProcessor.LeftPalmPosition : Vector3.zero;
            Vector3 rightPalmPos = HandProcessor.ArePalmsTracked ? HandProcessor.RightPalmPosition : Vector3.zero;
            float widthFactor = olfatiSaberScript != null ? olfatiSaberScript.CurrentWidthFactor : 1.0f;
            float lengthFactor = olfatiSaberScript != null ? olfatiSaberScript.CurrentLengthFactor : 1.0f;

            string csvLine = string.Format("{0},{1},{2},{3},{4},{5},{6},{7},{8},{9},{10},{11},{12},{13},{14},{15},{16},{17},{18},{19},{20},{21},{22}",
                Time.time.ToString("F3"), // Timestamp
                parametersToSend.algorithmType,
                leftPalmPos.x.ToString("F3"), leftPalmPos.y.ToString("F3"), leftPalmPos.z.ToString("F3"),
                rightPalmPos.x.ToString("F3"), rightPalmPos.y.ToString("F3"), rightPalmPos.z.ToString("F3"),
                widthFactor.ToString("F2"),
                lengthFactor.ToString("F2"),
                parametersToSend.swarm_vx.ToString("F3"),
                parametersToSend.swarm_vy.ToString("F3"), // This is world Y / Olfati's desired_vz
                parametersToSend.swarm_vz.ToString("F3"), // This is Olfati's local Z / -desired_vy
                parametersToSend.cohesionWeights.x.ToString("F3"), parametersToSend.cohesionWeights.y.ToString("F3"), parametersToSend.cohesionWeights.z.ToString("F3"),
                parametersToSend.separationWeights.x.ToString("F3"), parametersToSend.separationWeights.y.ToString("F3"), parametersToSend.separationWeights.z.ToString("F3"),
                "N/A", "N/A", "N/A", "N/A" // Olfati-Saber specific fields
            );
            csvWriter.WriteLine(csvLine);
        }
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
            baseD_ref = olfatiSaberScript.d_ref,
            swarm_vx = olfatiSaberScript.desired_vx,
            swarm_vy = olfatiSaberScript.desired_vz, 
            swarm_vz = -olfatiSaberScript.desired_vy
        };

        string jsonParameters = JsonUtility.ToJson(parametersToSend);
        SendUdpData(jsonParameters);

        // Log to CSV
        if (csvWriter != null)
        {
            Vector3 leftPalmPos = HandProcessor.ArePalmsTracked ? HandProcessor.LeftPalmPosition : Vector3.zero;
            Vector3 rightPalmPos = HandProcessor.ArePalmsTracked ? HandProcessor.RightPalmPosition : Vector3.zero;
            // Factors are directly from olfatiSaberScript
            float widthFactor = olfatiSaberScript.CurrentWidthFactor;
            float lengthFactor = olfatiSaberScript.CurrentLengthFactor;

            string csvLine = string.Format("{0},{1},{2},{3},{4},{5},{6},{7},{8},{9},{10},{11},{12},{13},{14},{15},{16},{17},{18},{19},{20},{21},{22}",
                Time.time.ToString("F3"), // Timestamp
                parametersToSend.algorithmType,
                leftPalmPos.x.ToString("F3"), leftPalmPos.y.ToString("F3"), leftPalmPos.z.ToString("F3"),
                rightPalmPos.x.ToString("F3"), rightPalmPos.y.ToString("F3"), rightPalmPos.z.ToString("F3"),
                widthFactor.ToString("F2"),
                lengthFactor.ToString("F2"),
                parametersToSend.swarm_vx.ToString("F3"),
                parametersToSend.swarm_vy.ToString("F3"), // This is world Y / Olfati's desired_vz
                parametersToSend.swarm_vz.ToString("F3"), // This is Olfati's local Z / -desired_vy
                "N/A", "N/A", "N/A", // Reynolds specific
                "N/A", "N/A", "N/A", // Reynolds specific
                parametersToSend.d_ref_x.ToString("F3"), parametersToSend.d_ref_y.ToString("F3"), parametersToSend.d_ref_z.ToString("F3"), parametersToSend.baseD_ref.ToString("F3")
            );
            csvWriter.WriteLine(csvLine);
        }
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

        // Close CSV writer
        if (csvWriter != null)
        {
            csvWriter.Flush(); // Ensure all data is written
            csvWriter.Close();
            csvWriter = null;
            Debug.Log("UDP_Connection: CSV writer closed.");
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
