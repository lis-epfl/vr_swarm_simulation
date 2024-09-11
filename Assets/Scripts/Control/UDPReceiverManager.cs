using System;
using System.Net;
using System.Net.Sockets;
using System.Text;
using UnityEngine;

public class UDPReceiverManager : MonoBehaviour
{
    public int port = 5055;
    private UdpClient udpClient;
    private IPEndPoint endPoint;

    // Shared data for all objects to read from
    public static JoystickData sharedJoystickData;

    void Start()
    {
        // Initialize the UDP client
        udpClient = new UdpClient(port);
        endPoint = new IPEndPoint(IPAddress.Any, port);
    }

    void Update()
    {
        // Check if there is data available from the UDP socket
        if (udpClient.Available > 0)
        {
            byte[] data = udpClient.Receive(ref endPoint);
            string message = Encoding.UTF8.GetString(data);

            ProcessInput(message);
        }
    }

    // Function to process the JSON data received from the UDP socket
    void ProcessInput(string message)
    {
        // Convert the JSON string to a JoystickData object
        sharedJoystickData = JsonUtility.FromJson<JoystickData>(message);
    }

    private void OnApplicationQuit()
    {
        udpClient.Close();
    }
}

// Data structures for parsing the JSON data
[Serializable]
public class JoystickData
{
    public LinearVelocity linear;
    public AngularVelocity angular;
}

[Serializable]
public class LinearVelocity
{
    public float x;
    public float y;
    public float z;
}

[Serializable]
public class AngularVelocity
{
    public float x;
    public float y;
    public float z;
}
