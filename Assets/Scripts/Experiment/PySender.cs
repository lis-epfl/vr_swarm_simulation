
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
    private string sharedMemoryMetadataName = "TobiiUnityMetadata";

    [SerializeField]
    private string sharedMemoryGazeDataName = "TobiiUnityGazeData";

    [SerializeField]
    private string sharedMemoryNBackDataName = "ExperimentUnityNBackData";

    [SerializeField]
    private string sharedMemoryDroneDataName = "ExperimentUnityDroneData";

    [SerializeField]
    private uint maxBufferBlocks = 100;

    [SerializeField]
    private KeyCode toggleSendingKey = KeyCode.None;

    [SerializeField]
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

    private enum FileMapAccessType : uint
    {
        Write = 0x02,
        Read = 0x04,
        AllAccess = 0xF001F
    }

    private EyeTracker eyeTracker;
    private bool isReady = false;
    private bool isSending = false;
    private bool isCalibrationOk = false;
    private bool isBufferOverflowed = false;
    private const uint k_PageReadWrite = 0x04;

    private int metadataSize = -1;
    private int gazeDataSize = -1;
    private IntPtr gazeDataStartPtr = IntPtr.Zero;
    private IntPtr gazeDataCurrentPtr = IntPtr.Zero;
    private IntPtr metadataFileMap = IntPtr.Zero;
    private IntPtr gazeDataFileMap = IntPtr.Zero;
    private IntPtr metadataPtr = IntPtr.Zero;
    private IntPtr nbackDataFileMap = IntPtr.Zero;
    private IntPtr nbackDataPtr = IntPtr.Zero;
    private IntPtr droneDataFileMap = IntPtr.Zero;
    private IntPtr droneDataPtr = IntPtr.Zero;


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
        eyeTracker = EyeTracker.Instance;

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
            IsStreamReady = 1,
            IsCalibrationOk = (byte)(isCalibrationOk ? 1 : 0),
            ActiveDataCnt = 0
        };
        updateMetadata(initialMetadata);

        gazeDataFileMap = CreateFileMapping(new IntPtr(-1), IntPtr.Zero, k_PageReadWrite, 0, (uint)(gazeDataSize * maxBufferBlocks), sharedMemoryGazeDataName);
        if (gazeDataFileMap == IntPtr.Zero)
        {
            // Maybe it already exists, try to open it
            gazeDataFileMap = OpenFileMapping((uint)FileMapAccessType.Write, false, sharedMemoryGazeDataName);
        }
        if (gazeDataFileMap != IntPtr.Zero)
        {
            gazeDataStartPtr = MapViewOfFile(gazeDataFileMap, (uint)FileMapAccessType.Write, 0, 0, UIntPtr.Zero);
            if (gazeDataStartPtr == IntPtr.Zero)
            {
                Debug.LogError("Could not map view of file for gaze data. Error: " + Marshal.GetLastWin32Error());
                return;
            }
            gazeDataCurrentPtr = gazeDataStartPtr;
        }
        else
        {
            Debug.LogError("Could not create or open file mapping for gaze data. Error: " + Marshal.GetLastWin32Error());
            return;
        }
        isReady = true;

    }

    void Update()
    {
        if (Input.GetKeyDown(toggleSendingKey))
        {
            isSending = !isSending;
            Debug.Log("PySender sending toggled to: " + isSending);
        }

        if (Input.GetKeyDown(KeyCode.Space))
        {   
            // For testing purposes, write one temp data point
            writeToCircularBuffer(MockupData.mockupGazeData);
            updateMetadataCnt(1);
        }

        if (!isSending)
        {
            return;
        }

        var gazeData = eyeTracker.NextData;
        if (gazeData != default(IGazeData))
        {
            updateGazeData((GazeData)gazeData);
        }

        // Update drone data at specified rate
        long currentTime = DateTimeOffset.UtcNow.ToUnixTimeMilliseconds();
        if (currentTime - lastDroneDataUpdateTime >= (1000 / droneDataUpdateRate))
        {
            updateSwarmData();
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
                // Set IsStreamReady to 0
                PySenderData.CustomMetadata finalMetadata = new PySenderData.CustomMetadata
                {
                    IsStreamReady = 0,
                    IsCalibrationOk = (byte)(isCalibrationOk ? 1 : 0),
                    ActiveDataCnt = 0
                };
                updateMetadata(finalMetadata);
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
                Position = new PySenderData.Vec3f(state.Position),
                Orientation = new PySenderData.Vec3f(state.Angles)
            };

            IntPtr currentDataPtr = IntPtr.Add(arrayStartPtr, i * stride);
            Marshal.StructureToPtr(data, currentDataPtr, false);
        }
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

    private void updateGazeData(GazeData gazeData)
    {
        PySenderData.CustomGazeData dataToSend = new PySenderData.CustomGazeData
        {
            TimeStamp = gazeData.TimeStamp,
            LeftGazePoint = new PySenderData.Vec3f(gazeData.Left.GazeOriginInUserCoordinates),
            RightGazePoint = new PySenderData.Vec3f(gazeData.Right.GazeOriginInUserCoordinates),
            LeftGazeOnScreen = new PySenderData.Vec2f(gazeData.Left.GazePointOnDisplayArea),
            RightGazeOnScreen = new PySenderData.Vec2f(gazeData.Right.GazePointOnDisplayArea),
            LeftGazeValid = (byte)(gazeData.Left.GazePointValid ? 1 : 0),
            RightGazeValid = (byte)(gazeData.Right.GazePointValid ? 1 : 0),
            LeftPupilDiameter = gazeData.Left.PupilDiameter,
            RightPupilDiameter = gazeData.Right.PupilDiameter
        };
        if (isBufferOverflowed)
        {
            isBufferOverflowed = (updateMetadataCnt(0) >= maxBufferBlocks);
            return;
        }
        if (!isBufferOverflowed)
        {   
            writeToCircularBuffer(dataToSend);
            int count = updateMetadataCnt(1);
            if (count == -1)
            {
                Debug.LogError("Failed to update metadata count. Error: " + Marshal.GetLastWin32Error());
            } 
            else if (count >= maxBufferBlocks)
            {
                isBufferOverflowed = true;
                Debug.LogWarning("Auto-stopping PySender sending due to buffer full.");
            }
        }
    }

    private void writeToCircularBuffer(PySenderData.CustomGazeData data)
    {
        Marshal.StructureToPtr(data, gazeDataCurrentPtr, false);
        gazeDataCurrentPtr = IntPtr.Add(gazeDataCurrentPtr, gazeDataSize);
        long offset = gazeDataCurrentPtr.ToInt64() - gazeDataStartPtr.ToInt64();
        if (offset >= gazeDataSize * maxBufferBlocks)
        {
            gazeDataCurrentPtr = gazeDataStartPtr;
        }
    }

    private int updateMetadataCnt(byte dataIncr)
    {
        if (metadataPtr == IntPtr.Zero)
        {
            return -1;
        }

        IntPtr cntPtr = IntPtr.Add(metadataPtr, Marshal.OffsetOf(typeof(PySenderData.CustomMetadata), "ActiveDataCnt").ToInt32());
        byte currentCnt = Marshal.ReadByte(cntPtr);
        currentCnt += dataIncr;
        Marshal.WriteByte(cntPtr, 0, currentCnt);
        // if (Marshal.GetLastWin32Error() != 0)
        // {
        //     return false;
        // }
        return currentCnt;
    }

    private void updateMetadata(PySenderData.CustomMetadata metadata)
    {
        if (metadataPtr == IntPtr.Zero)
        {
            return;
        }
        Marshal.StructureToPtr(metadata, metadataPtr, false);
    }
}