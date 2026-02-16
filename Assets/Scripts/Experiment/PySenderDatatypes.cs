using UnityEngine;
using System.Runtime.InteropServices;

namespace Experiment
{
    public class PySenderData
    {
        // The "StructLayout" attributes ensure that the data structures are laid out in memory exactly as defined,
        [StructLayout(LayoutKind.Sequential, Pack = 1)]
        public struct Vec2f
        {
            public float x;
            public float y;

            public Vec2f(Vector2 vec)
            {
                x = vec.x;
                y = vec.y;
            }
        }

        [StructLayout(LayoutKind.Sequential, Pack = 1)]
        public struct Vec3f
        {
            public float x;
            public float y;
            public float z;

            public Vec3f(Vector3 vec)
            {
                x = vec.x;
                y = vec.y;
                z = vec.z;
            }
        }

        [StructLayout(LayoutKind.Sequential, Pack = 1)]
        public struct CustomGazeData
        {
            public long TimeStamp;
            public Vec3f LeftGazePoint;
            public Vec3f RightGazePoint;
            public Vec2f LeftGazeOnScreen;
            public Vec2f RightGazeOnScreen;
            public byte LeftGazeValid;
            public byte RightGazeValid;
            public float LeftPupilDiameter;
            public float RightPupilDiameter;
            public byte LeftOpennessValid;
            public byte RightOpennessValid;
            public float LeftEyeOpenness;
            public float RightEyeOpenness;
        }

        [StructLayout(LayoutKind.Sequential, Pack = 1)]
        public struct CustomMetadata
        {
            public byte IsStreamReady;
            public byte IsCalibrationOk;
            public byte ActiveDataCnt;
        }

        [StructLayout(LayoutKind.Sequential, Pack = 1)]
        public struct NBackData
        {
            public long TimeStamp;
            public long ResponseTimeStamp;
            public byte NBackLevel;
            public byte Stimulus;
            public byte ParticipantResponse;
            public byte IsCorrect;

            public NBackData(byte stimulus, byte nback_level)
            {
                Stimulus = stimulus;
                TimeStamp = 0;
                ParticipantResponse = 0;
                IsCorrect = 0;
                ResponseTimeStamp = 0;
                NBackLevel = nback_level;
            }
        }

        [StructLayout(LayoutKind.Sequential, Pack = 1)]
        public struct DroneData
        {
            public long Timestamp;
            public byte Id;
            public Vec3f Position;
            public Vec3f Orientation;
            public Vec3f Velocity;
            public Vec3f AngularVelocity;
            public Vec3f Acceleration;
        }

        [StructLayout(LayoutKind.Sequential, Pack = 1)]
        public struct UserControlInputData
        {
            public float Throttle;
            public float Yaw;
            public float Pitch;
            public float Roll;
            public float SwarmSpread;
        }
    };

    public class MockupData {
        static public PySenderData.CustomGazeData mockupGazeData = new PySenderData.CustomGazeData
        {
            TimeStamp = 1234567890123,
            LeftGazePoint = new PySenderData.Vec3f {x=1.1f, y=1.2f, z=1.3f},
            RightGazePoint = new PySenderData.Vec3f {x=2.1f, y=2.2f, z=2.3f},
            LeftGazeOnScreen = new PySenderData.Vec2f {x=3.1f, y=3.2f},
            RightGazeOnScreen = new PySenderData.Vec2f {x=4.1f, y=4.2f},
            LeftGazeValid = 0,
            RightGazeValid = 1,
            LeftPupilDiameter = 5.5f,
            RightPupilDiameter = 6.6f
        };
    }
}