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

    /// <summary>
    /// Presses of the RC's mark button (C2 under <c>rcjoy bridge --profile sim</c>) received since
    /// this play session's first packet; only ever increases. The bridge sends a cumulative count, so
    /// this is the count's rise: the first packet (and a restarted bridge counting from 0 again) is
    /// only a baseline, which is how a press made before Play never fires late. Consumers keep their
    /// own last-seen value and act on the difference. Stays 0 on the Taranis path, which sends no count.
    /// </summary>
    public static int MarkPresses { get; private set; }
    private static int lastRawMarks = -1;

    // Statics survive Play sessions when domain reload is off.
    [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.SubsystemRegistration)]
    private static void ResetStatics()
    {
        sharedJoystickData = null;
        MarkPresses = 0;
        lastRawMarks = -1;
    }

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

        int raw = sharedJoystickData.marks;
        if (lastRawMarks >= 0 && raw > lastRawMarks) MarkPresses += raw - lastRawMarks;
        lastRawMarks = raw;
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
    public Switches switches;
    // Cumulative mark-button presses (rcjoy bridge --profile sim only; absent, so 0, otherwise).
    // Read through UDPReceiverManager.MarkPresses, not directly.
    public int marks;
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

[Serializable]
public class Switches
{
    public int s1;
    public float s2; // Gimbal pitch dial (raw -1..1), mapped to swarm gimbal pitch in Unity.
}
