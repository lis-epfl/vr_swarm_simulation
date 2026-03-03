//-----------------------------------------------------------------------
// Improved version of the Tobii Pro Eye Tracker script, which
// now includes eye openess data.
// Copyright © 2019 Tobii Pro AB. All rights reserved.
//-----------------------------------------------------------------------

using System.Threading;
using UnityEngine;
using Tobii.Research.Unity;
using Tobii.Research;

namespace Experiment
{
    public class CustomEyeTracker : EyeTrackerBase
    {
        #region Public Properties

        /// <summary>
        /// Get <see cref="CustomEyeTracker"/> instance. This is assigned
        /// in Awake(), so call earliest in Start().
        /// </summary>
        public static CustomEyeTracker Instance { get; private set; }

        /// <summary>
        /// Get the latest gaze data. If there are new arrivals,
        /// they will be processed before returning.
        /// </summary>
        public IGazeData LatestGazeData
        {
            get
            {
                if (UnprocessedGazeDataCount > 0)
                {
                    // We have more data.
                    ProcessGazeEvents();
                }

                return _latestGazeData;
            }
        }

        public EyeOpennessData LatestEyeOpennessData
        {
            get
            {
                if (_originalEyeOpennessData.Count > 0)
                {
                    // We have more data.
                    ProcessEyeOpennessEvents();
                }

                return _latestEyeOpennessData;
            }
        }

        public struct EyeOpennessData
        {
            public bool LeftValid { get; private set; }
            public bool RightValid { get; private set; }
            public float LeftEyeOpenness { get; private set; }
            public float RightEyeOpenness { get; private set; }

            public EyeOpennessData(EyeOpennessDataEventArgs eventArgs)
            {
                LeftValid = eventArgs.LeftEyeValidity == Validity.Valid ? true : false;
                RightValid = eventArgs.RightEyeValidity == Validity.Valid ? true : false;
                LeftEyeOpenness = eventArgs.LeftEyeValue;
                RightEyeOpenness = eventArgs.RightEyeValue;
            }
        }

        /// <summary>
        /// Get the latest processed processed gaze data.
        /// Don't care if there a newer one has arrived.
        /// </summary>
        public IGazeData LatestProcessedGazeData { get { return _latestGazeData; } }

        /// <summary>
        /// Pop and get the next gaze data object from the queue.
        /// </summary>
        public IGazeData NextData
        {
            get
            {
                if (_gazeDataQueue.Count < 1)
                {
                    return default(IGazeData);
                }

                return _gazeDataQueue.Next;
            }
        }

        public override bool SubscribeToGazeData
        {
            get
            {
                return _subscribeToGaze;
            }

            set
            {
                _subscribeToGaze = value;
                base.SubscribeToGazeData = value;
            }
        }

        public EyeOpennessData NextEyeOpennessData
        {
            get
            {
                if (_originalEyeOpennessData.Count < 1)
                {
                    return default(EyeOpennessData);
                }

                return _eyeOpennessDataQueue.Next;
            }
        }

        public bool GazeDataAvailable { get { return _gazeDataQueue.Count > 0; } }
        public bool EyeOpennessDataAvailable { get { return _eyeOpennessDataQueue.Count > 0; } }

        public bool SubscribeToEyeOpenness
        {
            get
            {
                return _subscribeToEyeOpenness;
            }

            set
            {
                _subscribeToEyeOpenness = value;
                UpdateSubscriptions();
            }
        }

        public override int GazeDataCount { get { return _gazeDataQueue.Count; } }

        public override int UnprocessedGazeDataCount { get { return _originalGazeData.Count; } }

        #endregion Public Properties

        #region Inspector Properties

        [SerializeField]
        [Tooltip("Connect to the first found eye tracker. Otherwise use provided serial number.")]
        private bool _connectToFirst;

        [SerializeField]
        [Tooltip("Check for this specific eyetracker serial number. Matches start of string so a partial start of a serial number can be used.")]
        private string _eyeTrackerSerialStart = "IS";

        [SerializeField]
        private int maxBufferBlocks = 100;

        [SerializeField]
        [Tooltip("Checking this will subscribe to eye openness data.")]
        private bool _subscribeToEyeOpenness = true;

        #endregion Inspector Properties

        #region Private Fields

        /// <summary>
        /// Locked access and size management.
        /// </summary>
        private LockedQueue<GazeDataEventArgs> _originalGazeData;

        /// <summary>
        /// Size managed queue.
        /// </summary>
        private SizedQueue<IGazeData> _gazeDataQueue;

        /// <summary>
        /// Hold the latest processed gaze data. Initialized to an invalid object.
        /// </summary>
        private IGazeData _latestGazeData = new GazeData();

        /// <summary>
        /// Hold the latest processed eye openness data. Initialized to an invalid object.
        /// </summary>
        private EyeOpennessData _latestEyeOpennessData = new EyeOpennessData();
        
        private LockedQueue<EyeOpennessDataEventArgs> _originalEyeOpennessData;

        /// <summary>
        /// Size managed queue.
        /// </summary>
        private SizedQueue<EyeOpennessData> _eyeOpennessDataQueue;


        private bool _subscribingToEyeOpenness;

        #endregion Private Fields

        #region Unity Methods

        protected override void OnAwake()
        {
            Instance = this;
            _originalGazeData = new LockedQueue<GazeDataEventArgs>(maxCount: maxBufferBlocks);
            _gazeDataQueue = new SizedQueue<IGazeData>(maxCount: maxBufferBlocks);
            _originalEyeOpennessData = new LockedQueue<EyeOpennessDataEventArgs>(maxCount: maxBufferBlocks);
            _eyeOpennessDataQueue = new SizedQueue<EyeOpennessData>(maxCount: maxBufferBlocks);
            base.OnAwake();
        }

        protected override void OnStart()
        {
            base.OnStart();
        }

        protected override void OnUpdate()
        {
            base.OnUpdate();
            if (SubscribeToEyeOpenness)
            {
                ProcessEyeOpennessEvents();
            }
        }

        #endregion Unity Methods

        #region Private Eye Tracking Methods

        protected override void ProcessGazeEvents()
        {
            const int maxIterations = 20;

            var gazeData = _latestGazeData;

            for (int i = 0; i < maxIterations; i++)
            {
                var originalGaze = _originalGazeData.Next;

                // Queue empty
                if (originalGaze == null)
                {
                    break;
                }

                gazeData = new GazeData(originalGaze);
                _gazeDataQueue.Next = gazeData;
            }

            var queueCount = UnprocessedGazeDataCount;
            if (queueCount > 0)
            {
                Debug.LogWarning("We didn't manage to empty the queue: " + queueCount + " items left...");
            }

            _latestGazeData = gazeData;
        }

        protected void ProcessEyeOpennessEvents()
        {
            const int maxIterations = 20;

            var eyeOpennessData = _latestEyeOpennessData;

            for (int i = 0; i < maxIterations; i++)
            {
                var originalEyeOpenness = _originalEyeOpennessData.Next;

                // Queue empty
                if (originalEyeOpenness == null)
                {
                    break;
                }

                eyeOpennessData = new EyeOpennessData(originalEyeOpenness);
                _eyeOpennessDataQueue.Next = eyeOpennessData;
            }

            var queueCount = _originalEyeOpennessData.Count;
            if (queueCount > 0)
            {
                Debug.LogWarning("We didn't manage to empty the eye openness queue: " + queueCount + " items left...");
            }

            _latestEyeOpennessData = eyeOpennessData;
        }

        protected override void StartAutoConnectThread()
        {
            if (_autoConnectThread != null)
            {
                return;
            }

            _autoConnectThread = new Thread(() =>
            {
                AutoConnectThreadRunning = true;

                while (AutoConnectThreadRunning)
                {
                    var eyeTrackers = EyeTrackingOperations.FindAllEyeTrackers();

                    foreach (var eyeTrackerEntry in eyeTrackers)
                    {
                        if (_connectToFirst || eyeTrackerEntry.SerialNumber.StartsWith(_eyeTrackerSerialStart))
                        {
                            FoundEyeTracker = eyeTrackerEntry;
                            AutoConnectThreadRunning = false;
                            return;
                        }
                    }

                    Thread.Sleep(200);
                }
            });

            _autoConnectThread.IsBackground = true;
            _autoConnectThread.Start();
        }

        protected override void UpdateSubscriptions()
        {
            if (_eyeTracker == null)
            {
                return;
            }

            if (_subscribeToGaze && !_subscribingToGazeData)
            {
                _eyeTracker.GazeDataReceived += GazeDataReceivedCallback;
                _subscribingToGazeData = true;
            }
            else if (!_subscribeToGaze && _subscribingToGazeData)
            {
                _eyeTracker.GazeDataReceived -= GazeDataReceivedCallback;
                _subscribingToGazeData = false;
            }
            if (_subscribeToEyeOpenness && !_subscribingToEyeOpenness)
            {
                _eyeTracker.EyeOpennessDataReceived += EyeOpennessReceivedCallback;
                _subscribingToEyeOpenness = true;
            }
            else if (!_subscribeToEyeOpenness && _subscribingToEyeOpenness)
            {
                _eyeTracker.EyeOpennessDataReceived -= EyeOpennessReceivedCallback;
                _subscribingToEyeOpenness = false;
            }
        }

        private void GazeDataReceivedCallback(object sender, GazeDataEventArgs eventArgs)
        {
            _originalGazeData.Next = eventArgs;
        }

        private void EyeOpennessReceivedCallback(object sender, EyeOpennessDataEventArgs eventArgs)
        {
            _originalEyeOpennessData.Next = eventArgs;
        }

        #endregion Private Eye Tracking Methods
    }
}