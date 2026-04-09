
using UnityEngine;
using Tobii.Research.Unity;
using System.Runtime.InteropServices;
using System;
using Unity.VisualScripting;
using System.IO;
using Experiment;
using System.Collections.Generic;

public class PySender : MonoBehaviour
{
    [SerializeField]
    [Tooltip("Name of the shared memory block for metadata. Should match the name used in the Python receiver.")]
    private string sharedMemoryMetadataName = "TobiiUnityMetadata";

    [SerializeField]
    [Tooltip("Name of the shared memory block for gaze data. Should match the name used in the Python receiver.")]
    private string sharedMemoryGazeDataName = "TobiiUnityGazeData";

    [SerializeField]
    [Tooltip("Name of the shared memory block for N-Back data. Should match the name used in the Python receiver.")]
    private string sharedMemoryNBackDataName = "ExperimentUnityNBackData";

    [SerializeField]
    [Tooltip("Name of the shared memory block for drone data. Should match the name used in the Python receiver.")]
    private string sharedMemoryDroneDataName = "ExperimentUnityDroneData";

    [SerializeField]
    [Tooltip("Name of the shared memory block for user input data. Should match the name used in the Python receiver.")]
    private string sharedMemoryUserInputDataName = "ExperimentUnityUserInputData";

    [SerializeField]
    [Tooltip("Maximum number of buffer blocks to be used for the circular buffer (Gaze data)")]
    private int maxBufferBlocks = 100;

    [SerializeField]
    [Tooltip("Key to toggle sending data.")]
    private KeyCode toggleSendingKey = KeyCode.None;

    [SerializeField]
    [Tooltip("Name of the shared memory block for gate status. Should match the name used in the Python receiver.")]
    private string sharedMemoryGateStatusName = "ExperimentUnityGateStatus";

    [SerializeField]
    [Tooltip("Name of the shared memory block for gate layout. Should match the name used in the Python receiver.")]
    private string sharedMemoryGateLayoutName = "ExperimentUnityGateLayout";

    [SerializeField]
    [Tooltip("Update rate for drone data in Hz.")]
    private int droneDataUpdateRate = 30;
    private long lastDroneDataUpdateTime = 0;
    private readonly List<StateFinder> droneStates = new List<StateFinder>();
    private List<GameObject> swarm = new List<GameObject>();
    public List<GameObject> Swarm 
    { 
        get => swarm; 
        set
        {
            swarm = value;
            cacheSwarmStates();
        }
    }

    public static PySender Instance { get; private set; }
    public bool IsCalibrationOk
    {
        get { return isCalibrationOk; }
        set { isCalibrationOk = value; }
    }

    public bool EyeDataStreaming { 
        get { return isSending; } 
        set {
            isSending = value; 
        } 
    }

    private enum FileMapAccessType : uint
    {
        Write = 0x02,
        Read = 0x04,
        AllAccess = 0xF001F
    }

    private CustomEyeTracker eyeTracker;
    private bool isReady = false;
    private bool isSending = false;
    private bool isCalibrationOk = false;
    private bool isBufferOverflowed = false;
    private const uint k_PageReadWrite = 0x04;
    private long gazeTimestampAdjustment = -1;

    private int metadataSize = -1;
    private int gazeDataSize = -1;
    private IntPtr gazeDataStartPtr = IntPtr.Zero;
    private int gazeDataHead = 0;
    private IntPtr metadataFileMap = IntPtr.Zero;
    private IntPtr gazeDataFileMap = IntPtr.Zero;
    private IntPtr metadataPtr = IntPtr.Zero;
    private IntPtr nbackDataFileMap = IntPtr.Zero;
    private IntPtr nbackDataPtr = IntPtr.Zero;
    private IntPtr droneDataFileMap = IntPtr.Zero;
    private IntPtr droneDataPtr = IntPtr.Zero;
    private IntPtr userInputDataFileMap = IntPtr.Zero;
    private IntPtr userInputDataPtr = IntPtr.Zero;

    // Gate shared memory
    private IntPtr gateStatusFileMap = IntPtr.Zero;
    private IntPtr gateStatusPtr = IntPtr.Zero;
    private IntPtr gateLayoutFileMap = IntPtr.Zero;
    private IntPtr gateLayoutPtr = IntPtr.Zero;

    private const int MAX_GATES = 32;
    private const int GateStatusHeaderSize = 11; // long(8) + byte(1) + byte(1) + byte(1)
    private const int GateStatusEntrySize = 10;  // byte(1) + byte(1) + long(8)
    private const int GateStatusBlockSize = GateStatusHeaderSize + MAX_GATES * GateStatusEntrySize;
    private const int GateLayoutHeaderSize = 1;  // byte(1)
    private const int GateLayoutEntrySize = 32;  // Vec3f(12) + Vec3f(12) + float(4) + float(4)
    private const int GateLayoutBlockSize = GateLayoutHeaderSize + MAX_GATES * GateLayoutEntrySize;


    [DllImport("kernel32.dll", SetLastError = true, CharSet = CharSet.Auto)]
    private static extern IntPtr CreateFileMapping(IntPtr hFile, IntPtr lpFileMappingAttributes, uint flProtect, uint dwMaximumSizeHigh, uint dwMaximumSizeLow, string lpName);

    [DllImport("kernel32.dll", SetLastError = true)]
    private static extern IntPtr MapViewOfFile(IntPtr hFileMappingObject, uint dwDesiredAccess, uint dwFileOffsetHigh, uint dwFileOffsetLow, UIntPtr dwNumberOfBytesToMap);

    [DllImport("kernel32.dll", SetLastError = true)]
    private static extern bool UnmapViewOfFile(IntPtr lpBaseAddress);

    [DllImport("kernel32.dll", SetLastError = true)]
    private static extern bool CloseHandle(IntPtr hObject);

    [DllImport("kernel32.dll", SetLastError = true, CharSet = CharSet.Auto)]
    private static extern IntPtr OpenFileMapping(uint dwDesiredAccess, bool bInheritHandle, string lpName);

    void Awake()
    {
        Instance = this;
    }
    void Start()
    {
        Debug.Log("PySender started.");
        eyeTracker = CustomEyeTracker.Instance;

        // Initialize shared memory metadata and gaze data blocks here
        metadataSize = Marshal.SizeOf(typeof(PySenderData.CustomMetadata));
        gazeDataSize = Marshal.SizeOf(typeof(PySenderData.CustomGazeData));

        metadataFileMap = CreateFileMapping(new IntPtr(-1), IntPtr.Zero, k_PageReadWrite, 0, (uint)metadataSize, sharedMemoryMetadataName);
        if (metadataFileMap != IntPtr.Zero)
        {
            metadataPtr = MapViewOfFile(metadataFileMap, (uint)FileMapAccessType.AllAccess, 0, 0, UIntPtr.Zero);
            if (metadataPtr == IntPtr.Zero)
            {
                Debug.LogError("Could not map view of file for metadata. Error: " + Marshal.GetLastWin32Error());
                return;
            }
        }
        // Initialize metadata to default values
        PySenderData.CustomMetadata initialMetadata = new PySenderData.CustomMetadata
        {
            IsSenderReady = 1,
            IsCalibrationOk = (byte)(isCalibrationOk ? 1 : 0),
            Head = gazeDataHead
        };
        writeMetadata(initialMetadata);
        gazeDataFileMap = CreateFileMapping(new IntPtr(-1), IntPtr.Zero, k_PageReadWrite, 0, (uint)(gazeDataSize * maxBufferBlocks), sharedMemoryGazeDataName);
        if (gazeDataFileMap == IntPtr.Zero)
        {
            // Maybe it already exists, try to open it
            gazeDataFileMap = OpenFileMapping((uint)FileMapAccessType.Write, true, sharedMemoryGazeDataName);
        }
        if (gazeDataFileMap != IntPtr.Zero)
        {
            gazeDataStartPtr = MapViewOfFile(gazeDataFileMap, (uint)FileMapAccessType.Write, 0, 0, UIntPtr.Zero);
            if (gazeDataStartPtr == IntPtr.Zero)
            {
                Debug.LogError("Could not map view of file for gaze data. Error: " + Marshal.GetLastWin32Error());
                return;
            }
        }
        else
        {
            Debug.LogError("Could not create or open file mapping for gaze data. Error: " + Marshal.GetLastWin32Error());
            return;
        }
        // Gate Status shared memory (dynamic, updated on state changes)
        gateStatusFileMap = CreateFileMapping(new IntPtr(-1), IntPtr.Zero, k_PageReadWrite, 0, (uint)GateStatusBlockSize, sharedMemoryGateStatusName);
        if (gateStatusFileMap != IntPtr.Zero)
            gateStatusPtr = MapViewOfFile(gateStatusFileMap, (uint)FileMapAccessType.AllAccess, 0, 0, UIntPtr.Zero);
        else
            Debug.LogError("[PySender] Failed to create GateStatus shared memory. Error: " + Marshal.GetLastWin32Error());

        // Gate Layout shared memory (static, written once when course is generated)
        gateLayoutFileMap = CreateFileMapping(new IntPtr(-1), IntPtr.Zero, k_PageReadWrite, 0, (uint)GateLayoutBlockSize, sharedMemoryGateLayoutName);
        if (gateLayoutFileMap != IntPtr.Zero)
            gateLayoutPtr = MapViewOfFile(gateLayoutFileMap, (uint)FileMapAccessType.AllAccess, 0, 0, UIntPtr.Zero);
        else
            Debug.LogError("[PySender] Failed to create GateLayout shared memory. Error: " + Marshal.GetLastWin32Error());

        isReady = true;
        Time.fixedDeltaTime = 0.01f; // Set fixed update to 100Hz for more frequent data sending
    }

    void FixedUpdate()
    {
        if (Input.GetKeyDown(toggleSendingKey))
        {
            isSending = !isSending;
            Debug.Log("PySender sending toggled to: " + isSending);
        }

        if (Input.GetKeyDown(KeyCode.Space))
        {   
            // For testing purposes, write one temp data point
            isBufferOverflowed = !checkEnoughSpaceInBuffer();
            writeToCircularBuffer(MockupData.mockupGazeData);
        }


        while (eyeTracker.GazeDataAvailable && eyeTracker.EyeOpennessDataAvailable)
        {
            GazeData gazeData = (GazeData) eyeTracker.NextData;
            CustomEyeTracker.EyeOpennessData eyeOpennessData = eyeTracker.NextEyeOpennessData;
            if (gazeTimestampAdjustment < 0)
            {
                gazeTimestampAdjustment = DateTimeOffset.UtcNow.ToUnixTimeMilliseconds() - gazeData.TimeStamp / 1000;
            }
            if (isSending && checkReceiverReady())
            {
                updateGazeData(gazeData, eyeOpennessData);
            }
        }

        // Update drone data at specified rate
        long currentTime = DateTimeOffset.UtcNow.ToUnixTimeMilliseconds();
        if (isSending && (currentTime - lastDroneDataUpdateTime >= (1000 / droneDataUpdateRate)))
        {
            updateSwarmData();
            updateInputCommands();
            lastDroneDataUpdateTime = currentTime;
        }

    }


    void OnDestroy()
    {
       // Destroy memory mappings
        if (metadataPtr != IntPtr.Zero)
        {
            if (isReady)
            {
                // Set IsSenderReady to 0
                PySenderData.CustomMetadata finalMetadata = new PySenderData.CustomMetadata
                {
                    IsSenderReady = 0,
                    IsCalibrationOk = (byte)(isCalibrationOk ? 1 : 0),
                    Head = gazeDataHead,
                };
                writeMetadata(finalMetadata);
            }
            UnmapViewOfFile(metadataPtr);
            metadataPtr = IntPtr.Zero;
        }
        if (metadataFileMap != IntPtr.Zero)
        {
            CloseHandle(metadataFileMap);
            metadataFileMap = IntPtr.Zero;
        }
        if (gazeDataStartPtr != IntPtr.Zero)
        {
            UnmapViewOfFile(gazeDataStartPtr);
            gazeDataStartPtr = IntPtr.Zero;
        }
        if (gazeDataFileMap != IntPtr.Zero)
        {
            CloseHandle(gazeDataFileMap);
            gazeDataFileMap = IntPtr.Zero;
        }
        if (nbackDataPtr != IntPtr.Zero)
        {
            UnmapViewOfFile(nbackDataPtr);
            nbackDataPtr = IntPtr.Zero;
        }
        if (nbackDataFileMap != IntPtr.Zero)
        {
            CloseHandle(nbackDataFileMap);
            nbackDataFileMap = IntPtr.Zero;
        }
        if (droneDataPtr != IntPtr.Zero)
        {
            UnmapViewOfFile(droneDataPtr);
            droneDataPtr = IntPtr.Zero;
        }
        if (droneDataFileMap != IntPtr.Zero)
        {
            CloseHandle(droneDataFileMap);
            droneDataFileMap = IntPtr.Zero;
        }
        if (userInputDataPtr != IntPtr.Zero)
        {
            UnmapViewOfFile(userInputDataPtr);
            userInputDataPtr = IntPtr.Zero;
        }
        if (userInputDataFileMap != IntPtr.Zero)
        {
            CloseHandle(userInputDataFileMap);
            userInputDataFileMap = IntPtr.Zero;
        }
        if (gateStatusPtr != IntPtr.Zero)
        {
            UnmapViewOfFile(gateStatusPtr);
            gateStatusPtr = IntPtr.Zero;
        }
        if (gateStatusFileMap != IntPtr.Zero)
        {
            CloseHandle(gateStatusFileMap);
            gateStatusFileMap = IntPtr.Zero;
        }
        if (gateLayoutPtr != IntPtr.Zero)
        {
            UnmapViewOfFile(gateLayoutPtr);
            gateLayoutPtr = IntPtr.Zero;
        }
        if (gateLayoutFileMap != IntPtr.Zero)
        {
            CloseHandle(gateLayoutFileMap);
            gateLayoutFileMap = IntPtr.Zero;
        }
    }

    public bool InitializeNBackDataSharedMemory(uint sequenceLength)
    {
        nbackDataFileMap = CreateFileMapping
        (
            new IntPtr(-1), IntPtr.Zero, 
            k_PageReadWrite, 
            0, 
            (uint)(sizeof(long) + Marshal.SizeOf(typeof(PySenderData.NBackData)) * sequenceLength), 
            sharedMemoryNBackDataName
        );
        if (nbackDataFileMap == IntPtr.Zero)
        {
            // Maybe it already exists, try to open it
            nbackDataFileMap = OpenFileMapping((uint)FileMapAccessType.Write, false, sharedMemoryNBackDataName);
        }
        if (nbackDataFileMap != IntPtr.Zero)
        {
            nbackDataPtr = MapViewOfFile(nbackDataFileMap, (uint)FileMapAccessType.Write, 0, 0, UIntPtr.Zero);
            if (nbackDataPtr == IntPtr.Zero)
            {
                Debug.LogError("Could not map view of file for N-Back data. Error: " + Marshal.GetLastWin32Error());
                return false;
            }
            return true;
        }
        else
        {
            Debug.LogError("Could not create or open file mapping for N-Back data. Error: " + Marshal.GetLastWin32Error());
            return false;
        }
    }

    public bool UpdateNBackData(long timeStamp, PySenderData.NBackData[] nBackSequence)
    {
        if (nbackDataPtr == IntPtr.Zero)
        {
            return false;
        }
        // Write timestamp
        Marshal.WriteInt64(nbackDataPtr, timeStamp);
        // Write N-Back sequence
        IntPtr sequenceStartPtr = IntPtr.Add(nbackDataPtr, sizeof(long));
        for (int i = 0; i < nBackSequence.Length; i++)
        {
            IntPtr currentDataPtr = IntPtr.Add(sequenceStartPtr, i * Marshal.SizeOf(typeof(PySenderData.NBackData)));
            Marshal.StructureToPtr(nBackSequence[i], currentDataPtr, false);
        }
        return true;
    }

    public bool UpdateNBackData(long timeStamp, PySenderData.NBackData singleStimulus, int index)
    {
        if (nbackDataPtr == IntPtr.Zero)
        {
            return false;
        }
        // Write timestamp
        Marshal.WriteInt64(nbackDataPtr, timeStamp);
        // Write latest stimulus at specified index
        IntPtr sequenceStartPtr = IntPtr.Add(nbackDataPtr, sizeof(long));
        IntPtr currentDataPtr = IntPtr.Add(sequenceStartPtr, index * Marshal.SizeOf(typeof(PySenderData.NBackData)));
        Marshal.StructureToPtr(singleStimulus, currentDataPtr, false);
        return true;
    }

    public bool InitializeDroneDataSharedMemory(uint droneCount)
    {
        droneDataFileMap = CreateFileMapping
        (
            new IntPtr(-1), IntPtr.Zero, 
            k_PageReadWrite, 
            0, 
            (uint)(sizeof(long) + Marshal.SizeOf(typeof(PySenderData.DroneData)) * droneCount), 
            sharedMemoryDroneDataName
        );
        if (droneDataFileMap == IntPtr.Zero)
        {
            // Maybe it already exists, try to open it
            droneDataFileMap = OpenFileMapping((uint)FileMapAccessType.Write, false, sharedMemoryDroneDataName);
        }
        if (droneDataFileMap != IntPtr.Zero)
        {
            droneDataPtr = MapViewOfFile(droneDataFileMap, (uint)FileMapAccessType.Write, 0, 0, UIntPtr.Zero);
            if (droneDataPtr == IntPtr.Zero)
            {
                Debug.LogError("Could not map view of file for drone data. Error: " + Marshal.GetLastWin32Error());
                return false;
            }
            return true;
        }
        else
        {
            Debug.LogError("Could not create or open file mapping for drone data. Error: " + Marshal.GetLastWin32Error());
            return false;
        }
    }

    private void updateSwarmData()
    {
        if (Swarm == null || Swarm.Count == 0 || droneDataPtr == IntPtr.Zero)
        {
            return;
        }

        // Write timestamp
        long currentTimeStamp = DateTimeOffset.UtcNow.ToUnixTimeMilliseconds();
        Marshal.WriteInt64(droneDataPtr, currentTimeStamp);

        // Write each drone directly into shared memory (no array allocation)
        IntPtr arrayStartPtr = IntPtr.Add(droneDataPtr, sizeof(long));
        int stride = Marshal.SizeOf(typeof(PySenderData.DroneData));

        for (int i = 0; i < Swarm.Count; i++)
        {
            StateFinder state = droneStates[i];

            PySenderData.DroneData data = new PySenderData.DroneData
            {
                Timestamp = currentTimeStamp,
                Id = (byte)i,
                IsAlive = (byte)((state != null && state.IsAlive) ? 1 : 0),
                Position = new PySenderData.Vec3f(state.GroundTruthPosition),
                Orientation = new PySenderData.Vec3f(state.Angles),
                Velocity = new PySenderData.Vec3f(state.VelocityVector),
                AngularVelocity = new PySenderData.Vec3f(state.AngularVelocityVector),
                Acceleration = new PySenderData.Vec3f(state.Acceleration),
            };

            IntPtr currentDataPtr = IntPtr.Add(arrayStartPtr, i * stride);
            Marshal.StructureToPtr(data, currentDataPtr, false);
        }
    }

    public bool InitializeUserInputDataSharedMemory()
    {
        int blockSize = Marshal.SizeOf(typeof(PySenderData.UserControlInputData));
        userInputDataFileMap = CreateFileMapping(
            new IntPtr(-1), IntPtr.Zero,
            k_PageReadWrite,
            0, sizeof(long) + (uint)blockSize,
            sharedMemoryUserInputDataName
        );
        if (userInputDataFileMap == IntPtr.Zero)
        {
            userInputDataFileMap = OpenFileMapping((uint)FileMapAccessType.Write, false, sharedMemoryUserInputDataName);
        }
        if (userInputDataFileMap != IntPtr.Zero)
        {
            userInputDataPtr = MapViewOfFile(userInputDataFileMap, (uint)FileMapAccessType.Write, 0, 0, UIntPtr.Zero);
            if (userInputDataPtr == IntPtr.Zero)
            {
                Debug.LogError("Could not map view of file for user input data. Error: " + Marshal.GetLastWin32Error());
                return false;
            }
            return true;
        }
        else
        {
            Debug.LogError("Could not create or open file mapping for user input data. Error: " + Marshal.GetLastWin32Error());
            return false;
        }
    }

    private void updateInputCommands()
    {
        if (userInputDataPtr == IntPtr.Zero || InputManager.Instance == null) return;

        long currentTimeStamp = DateTimeOffset.UtcNow.ToUnixTimeMilliseconds();
        Marshal.WriteInt64(userInputDataPtr, currentTimeStamp);

        Dictionary<string, float> inputs = InputManager.Instance.InputStatusRaw;
        PySenderData.UserControlInputData data = new PySenderData.UserControlInputData
        {
            Timestamp = currentTimeStamp,
            AltitudeRate = inputs["throttle"],
            Yaw          = inputs["yaw"],
            Pitch        = inputs["pitch"],
            Roll         = inputs["roll"],
            SwarmSpread  = inputs["spread"],
        };
        Marshal.StructureToPtr(data, userInputDataPtr + sizeof(long), false);
    }

    private void cacheSwarmStates()
    {
        if (Swarm == null || Swarm.Count == 0)
        {
            return;
        }
        droneStates.Clear();
        for (int i = 0; i < Swarm.Count; i++)
        {
            GameObject drone = Swarm[i];
            if (drone == null)
            {
                droneStates.Add(null);
                continue;
            }

            Transform parent = drone.transform.Find("DroneParent");
            StateFinder s = parent != null ? parent.GetComponent<VelocityControl>().State : null;
            droneStates.Add(s);
        }
    }

    private void updateGazeData(GazeData gazeData, CustomEyeTracker.EyeOpennessData eyeOpennessData)
    {
        PySenderData.CustomGazeData dataToSend = new PySenderData.CustomGazeData
        {
            TimeStamp = gazeTimestampAdjustment > 0 ? gazeData.TimeStamp / 1000 + gazeTimestampAdjustment : gazeData.TimeStamp,
            LeftGazePoint = new PySenderData.Vec3f(gazeData.Left.GazePointInUserCoordinates),
            RightGazePoint = new PySenderData.Vec3f(gazeData.Right.GazePointInUserCoordinates),
            LeftGazeOnScreen = new PySenderData.Vec2f(gazeData.Left.GazePointOnDisplayArea),
            RightGazeOnScreen = new PySenderData.Vec2f(gazeData.Right.GazePointOnDisplayArea),
            LeftGazeValid = (byte)(gazeData.Left.GazePointValid ? 1 : 0),
            RightGazeValid = (byte)(gazeData.Right.GazePointValid ? 1 : 0),
            LeftPupilDiameter = gazeData.Left.PupilDiameter,
            RightPupilDiameter = gazeData.Right.PupilDiameter,
            LeftOpennessValid = (byte)(eyeOpennessData.LeftValid ? 1 : 0),
            RightOpennessValid = (byte)(eyeOpennessData.RightValid ? 1 : 0),
            LeftEyeOpenness = eyeOpennessData.LeftEyeOpenness,
            RightEyeOpenness = eyeOpennessData.RightEyeOpenness
        };
        writeToCircularBuffer(dataToSend);
    }

    private void writeToCircularBuffer(PySenderData.CustomGazeData data)
    {
        bool lastOverflowStatus = isBufferOverflowed;
        isBufferOverflowed = !checkEnoughSpaceInBuffer();
        if (!isBufferOverflowed)
        {
            IntPtr gazeDataCurrentPtr = IntPtr.Add(gazeDataStartPtr, gazeDataHead * gazeDataSize);
            Marshal.StructureToPtr(data, gazeDataCurrentPtr, false);
            gazeDataHead = (gazeDataHead + 1) % maxBufferBlocks;
            // Update head in metadata
            Marshal.WriteInt32(IntPtr.Add(metadataPtr, Marshal.OffsetOf(typeof(PySenderData.CustomMetadata), "Head").ToInt32()), gazeDataHead);
        }
        else if (!lastOverflowStatus)
        {
            Debug.LogWarning("Cannot write to circular buffer because it is full. Data is being dropped.");
        }
    }

    private bool checkReceiverReady()
    {
        if (metadataPtr == IntPtr.Zero)
        {
            return false;
        }
        byte receiverReady = Marshal.ReadByte(IntPtr.Add(metadataPtr, Marshal.OffsetOf(typeof(PySenderData.CustomMetadata), "IsReceiverReady").ToInt32()));
        return receiverReady == 1;
    }

    private bool checkEnoughSpaceInBuffer()
    {
        if (metadataPtr == IntPtr.Zero)
        {
            return false;
        }
        int tail = Marshal.ReadInt32(IntPtr.Add(metadataPtr, Marshal.OffsetOf(typeof(PySenderData.CustomMetadata), "Tail").ToInt32()));
        return ((gazeDataHead + 1) % maxBufferBlocks) != tail;
    }

    private void writeMetadata(PySenderData.CustomMetadata metadata)
    {
        // Only wrtite sender related data
        if (metadataPtr == IntPtr.Zero)
        {
            return;
        }
        Marshal.WriteByte(IntPtr.Add(metadataPtr, Marshal.OffsetOf(typeof(PySenderData.CustomMetadata), "IsSenderReady").ToInt32()), metadata.IsSenderReady);
        Marshal.WriteByte(IntPtr.Add(metadataPtr, Marshal.OffsetOf(typeof(PySenderData.CustomMetadata), "IsCalibrationOk").ToInt32()), metadata.IsCalibrationOk);
        Marshal.WriteInt32(IntPtr.Add(metadataPtr, Marshal.OffsetOf(typeof(PySenderData.CustomMetadata), "Head").ToInt32()), metadata.Head);
    }

    private PySenderData.CustomMetadata readMetadata()
    {
        if (metadataPtr == IntPtr.Zero)
        {
            return default(PySenderData.CustomMetadata);
        }
        return Marshal.PtrToStructure<PySenderData.CustomMetadata>(metadataPtr);
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Gate shared memory
    // ─────────────────────────────────────────────────────────────────────────

    /// <summary>
    /// Writes the static gate layout to shared memory. Call once after course generation.
    /// Memory layout: [GateCount:1] then per gate [CenterPosition:12][ForwardDirection:12][Width:4][Height:4]
    /// </summary>
    public void WriteGateLayout(List<RingGate> gates)
    {
        if (gateLayoutPtr == IntPtr.Zero) return;

        int count = Mathf.Min(gates.Count, MAX_GATES);
        Marshal.WriteByte(gateLayoutPtr, 0, (byte)count);

        int stride = Marshal.SizeOf(typeof(PySenderData.GateLayoutEntry));
        for (int i = 0; i < count; i++)
        {
            var gate = gates[i];
            if (gate == null) continue;

            var entry = new PySenderData.GateLayoutEntry
            {
                CenterPosition  = new PySenderData.Vec3f(gate.transform.position),
                ForwardDirection = new PySenderData.Vec3f(gate.transform.forward),
                Width  = gate.gateWidth,
                Height = gate.gateHeight
            };

            IntPtr entryPtr = IntPtr.Add(gateLayoutPtr, GateLayoutHeaderSize + i * stride);
            Marshal.StructureToPtr(entry, entryPtr, false);
        }

        Debug.Log($"[PySender] Gate layout written: {count} gates.");
    }

    /// <summary>
    /// Updates the dynamic gate status in shared memory.
    /// Memory layout: [Timestamp:8][TotalGates:1][ActiveGateIndex:1][IsCourseRunning:1]
    ///   then per gate [PassCount:1][GateState:1][FirstPassTimestamp:8]
    /// </summary>
    public void UpdateGateStatus(RingGateManager manager)
    {
        if (gateStatusPtr == IntPtr.Zero) return;

        long ts = DateTimeOffset.UtcNow.ToUnixTimeMilliseconds();
        int count = Mathf.Min(manager.gates.Count, MAX_GATES);

        // Header
        Marshal.WriteInt64(gateStatusPtr, 0, ts);
        Marshal.WriteByte(gateStatusPtr, 8, (byte)count);
        int activeIdx = manager.CurrentGateIndex;
        Marshal.WriteByte(gateStatusPtr, 9, (byte)(activeIdx < 0 || activeIdx >= count ? 255 : activeIdx));
        Marshal.WriteByte(gateStatusPtr, 10, (byte)(manager.IsCourseRunning ? 1 : 0));

        // Per-gate entries
        int stride = Marshal.SizeOf(typeof(PySenderData.GateStatusEntry));
        for (int i = 0; i < count; i++)
        {
            var gate = manager.gates[i];
            if (gate == null) continue;

            var visual = gate.GetComponent<RingGateVisual>();
            byte stateCode = 0;
            if (visual != null)
            {
                stateCode = visual.CurrentState switch
                {
                    GateVisualState.Idle            => 0,
                    GateVisualState.Next            => 1,
                    GateVisualState.PartialComplete => 2,
                    GateVisualState.Completed       => 3,
                    _                               => 0
                };
            }

            var entry = new PySenderData.GateStatusEntry
            {
                PassCount          = (byte)Mathf.Min(gate.TotalPasses, 255),
                GateState          = stateCode,
                FirstPassTimestamp = gate.FirstPassUnixMs
            };

            IntPtr entryPtr = IntPtr.Add(gateStatusPtr, GateStatusHeaderSize + i * stride);
            Marshal.StructureToPtr(entry, entryPtr, false);
        }
    }
}