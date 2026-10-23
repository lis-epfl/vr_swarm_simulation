using System;
using System.Net;
using System.Net.Sockets;
using System.Text;
using UnityEngine;

// Reads the socket before InputManager reads sharedJoystickData: one frame less latency, and IsStale
// is refreshed on the first frame after a long one instead of reading stale for a frame.
[DefaultExecutionOrder(-200)]
public class UDPReceiverManager : MonoBehaviour
{
    public int port = 5055;
    [Tooltip("Seconds without a packet before IsStale goes true and InputManager flies neutral sticks.")]
    public float staleAfterSeconds = 0.5f;
    private UdpClient udpClient;
    private IPEndPoint endPoint;

    // A flood must not spin the main thread; the socket buffer only holds a few hundred packets.
    private const int MaxReadsPerFrame = 4096;
    // Two packets in one frame is ordinary send jitter. Report skips only once they add up to
    // input a pilot would feel (4 = 0.2 s at the bridge's 20 Hz) inside one window.
    private const float SkipLogWindowSeconds = 2f;
    private const int SkipLogMin = 4;
    private int skipped;
    private float skipWindowStart;
    private bool reportedStale;

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

    /// <summary>
    /// True once a packet has arrived and none has for <see cref="staleAfterSeconds"/>: the sender
    /// stopped (RC dropped, bridge stopped; the bridge never pads a gap with neutral packets).
    /// <see cref="sharedJoystickData"/> still holds the last packet; InputManager flies neutral sticks
    /// instead of it. False before the first packet, when sharedJoystickData is null anyway.
    /// </summary>
    public static bool IsStale =>
        lastPacketTime >= 0f && Time.realtimeSinceStartup - lastPacketTime > staleAfter;
    private static float lastPacketTime = -1f;
    private static float staleAfter = 0.5f;

    // Statics survive Play sessions when domain reload is off.
    [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.SubsystemRegistration)]
    private static void ResetStatics()
    {
        sharedJoystickData = null;
        MarkPresses = 0;
        lastRawMarks = -1;
        lastPacketTime = -1f;
    }

    void Start()
    {
        // Initialize the UDP client
        udpClient = new UdpClient(port);
        endPoint = new IPEndPoint(IPAddress.Any, port);
        skipWindowStart = Time.realtimeSinceStartup;
    }

    void Update()
    {
        staleAfter = staleAfterSeconds;

        // Read everything queued and act on the newest packet only. Reading one per frame turned any
        // frame slower than the sender (the first frames of Play, a heavy view, < 20 fps) into a FIFO
        // backlog in the socket buffer, and the swarm flew seconds-old sticks, in order, until it
        // drained. Skipping loses nothing: sticks, knobs and s1 are levels, and marks is cumulative.
        byte[] newest = null;
        int reads = 0;
        while (reads < MaxReadsPerFrame && udpClient.Available > 0)
        {
            try
            {
                newest = udpClient.Receive(ref endPoint);
            }
            catch (SocketException)
            {
                break;
            }
            reads++;
        }

        float now = Time.realtimeSinceStartup;
        if (newest != null)
        {
            ProcessInput(Encoding.UTF8.GetString(newest));
            if (reportedStale)
            {
                reportedStale = false;
                Debug.Log($"[UDPReceiverManager] joystick back after {now - lastPacketTime:F1} s");
            }
            lastPacketTime = now;
        }
        else if (!reportedStale && IsStale)
        {
            reportedStale = true;
            Debug.LogWarning($"[UDPReceiverManager] joystick lost (no packet for {staleAfter:F1} s): sticks neutral");
        }

        skipped += Mathf.Max(0, reads - 1);
        if (now - skipWindowStart >= SkipLogWindowSeconds)
        {
            if (skipped >= SkipLogMin)
                Debug.Log($"[UDPReceiverManager] skipped {skipped} stale joystick packets in {now - skipWindowStart:F1} s " +
                          "(the scene ran slower than the sender)");
            skipped = 0;
            skipWindowStart = now;
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
