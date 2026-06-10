using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.InputSystem;

namespace Experiment
{
    public class ExperimentFSMRacingGate : ExperimentFSMBase
    {
        private enum ExperimentStateRacingGate
        {
            Idle,
            IdleSilent,
            Wait,
            Calibration,
            Welcome,
            RcControls,
            FlyingInstructions,
            FlyingPracticeInstr,
            FlyingPractice,
            RaceInstructions,
            RacePractice,
            ReadyScreen,
            Trial,
            Countdown,
            Finished,
            WaitForUser,
            Feedback
        }

        [Serializable]
        private class StepConfig
        {
            public ExperimentStateRacingGate state;
            public List<GameObject> enableOnState = new List<GameObject>();
        }

        [Header("References")]
        [SerializeField] private TimerCountdown timer;
        [SerializeField] private ViewManager viewManager;
        [SerializeField] private swarmSpawn swarmSpawner;
        [SerializeField] private RingGateManager ringGateManager;
        [SerializeField] private CourseStartTrigger courseStartTrigger;
        [SerializeField] private CWLController cwlController;
        [SerializeField] private CustomCalibration eyeTrackerCalibration;
        [SerializeField] private CustomTrackBoxGuide eyeTrackerGuide;
        [SerializeField] private CourseDemo courseDemo;
        [SerializeField] private ReadyOverlayManager readyOverlayManager;
        [SerializeField] private FeedbackOverlayManager feedbackManager;

        [Header("Step Visuals")]
        [SerializeField] private List<StepConfig> stateConfigs = new List<StepConfig>();
        [SerializeField] private ExperimentStateRacingGate initialState = ExperimentStateRacingGate.Welcome;

        [Header("Settings")]
        [SerializeField] private int totalTrialNumber = 4;
        [SerializeField] private int randomSeed = -1;
        [SerializeField] private bool isCwlActive = true;
        [SerializeField] private FlightProfile defaultFlightProfile;

        [Header("Debug")]
        [SerializeField] private bool isDebugMode = false;
        [SerializeField] private bool skipCalibration = false;
        [SerializeField] private bool skipFlyingPractice = false;
        [SerializeField] private float practiceTime = 0.25f; // in minutes
        [SerializeField] private float CountdownTime = 0.25f; // in minutes

        public float PracticeTime
        {
            set
            {
                practiceTime = value;
                flyingTaskPracticeTime = (int)TimeSpan.FromMinutes(value).TotalMilliseconds;
            }
            get => practiceTime;
        }

        private struct ExperimentSettings
        {
            public float FlightPracticeDuration; // in minutes
            public float CountdownTimeBetweentotalTrialNumber; // in minutes
            public int NumberOfTrials;
            public int NumberOfSegments;
            public int NumberOfGatesPerSegment;
        }

        private ExperimentSettings defaultSettings = new ExperimentSettings
        {
            FlightPracticeDuration = 0.75f,
            CountdownTimeBetweentotalTrialNumber = 0.75f,
            NumberOfTrials = 5,
            NumberOfSegments = 5,
            NumberOfGatesPerSegment = 8
        };

        private ExperimentStateRacingGate state = ExperimentStateRacingGate.Idle;
        private ExperimentStateRacingGate nextState = ExperimentStateRacingGate.Idle;
        private ExperimentStateRacingGate previousState = ExperimentStateRacingGate.Idle;

        private System.Random rng;
        private bool hasUserClicked = false;
        private bool isTransitionRequested = false;
        private bool _demoDotMoving = false;
        private bool CountdownEllapsed = false;
         private bool calibrationDone = false;
        private bool isHeadPositionOk = false;
        private int lastSwitchState = -1;
        private long stateEnterTime = 0;
        private int idleRequestDelay = -1; // in seconds, -1 means no delay

        private Vector3 k_swarmSpawnCourse = new Vector3(400f, 40f, 40f);
        private Vector3 k_swarmSpawnPractice = new Vector3(300, 30f, 200f);

        private int currentTrial = 0;
        private int flyingTaskPracticeTime = (int)TimeSpan.FromMinutes(0.25).TotalMilliseconds;

        void Start()
        {
            // Auto-discover CWL controller if not assigned
            if (cwlController == null)
                cwlController = FindObjectOfType<CWLController>();

            // Assign swarm to CWL controller when swarm is created
            if (cwlController != null && swarmSpawner != null && swarmSpawner.swarm.Count > 0)
            {
                cwlController.Swarm = swarmSpawner.swarm;
                Debug.Log("[ExperimentFSMRacingGate] Assigned swarm to CWL controller");
            }

            // Subscribe to course start trigger to enable CWL when race actually begins
            if (courseStartTrigger != null)
            {
                courseStartTrigger.onCourseTriggered.AddListener(OnCourseTriggered);
            }

            if (viewManager != null)
                viewManager.ToggleAllViews(false);

            DisableAllActiveObjects();
            if (timer != null)
                timer.OnCountdownFinished += () => CountdownEllapsed = true;
            rng = (randomSeed >= 0) ? new System.Random(randomSeed) : new System.Random();
            if (InputManager.Instance != null)
            {
                InputManager.Instance.LockControl();
            }
            if (isDebugMode)
            {
                Debug.LogWarning("Racing Gate Experiment is running in DEBUG MODE. Some steps may be skipped and practice times may be shortened.");
                flyingTaskPracticeTime = (int)TimeSpan.FromMinutes(practiceTime).TotalMilliseconds;
                timer.Minutes = CountdownTime;
                timer.Seconds = 0;
                ringGateManager.ShowLapTimer = true;
            }
            else
            {
                flyingTaskPracticeTime = (int)TimeSpan.FromMinutes(defaultSettings.FlightPracticeDuration).TotalMilliseconds;
                timer.Minutes = defaultSettings.CountdownTimeBetweentotalTrialNumber;
                timer.Seconds = 0;
                totalTrialNumber = defaultSettings.NumberOfTrials;
            }
            TransitionTo(initialState);
        }

        void Update()
        {
            ReadUserInput();

            switch (state)
            {
                case ExperimentStateRacingGate.Idle:
                case ExperimentStateRacingGate.IdleSilent:
                    if (idleRequestDelay > 0 && DateTimeOffset.Now.ToUnixTimeSeconds() - stateEnterTime / 1000 > idleRequestDelay)
                    {
                        TransitionTo(nextState);
                    }
                    break;

                case ExperimentStateRacingGate.Wait:
                    if (isTransitionRequested)
                    {
                        isTransitionRequested = false;
                        TransitionTo(nextState);
                    }
                    break;

                case ExperimentStateRacingGate.WaitForUser:
                    if (hasUserClicked)
                    {
                        hasUserClicked = false;
                        TransitionTo(nextState);
                    }
                    break;

                case ExperimentStateRacingGate.Countdown:
                    if (CountdownEllapsed)
                    {
                        CountdownEllapsed = false;
                        if (!feedbackManager.IsSubmitted)
                        {
                            feedbackManager.ForceSubmit();
                        }
                        TransitionTo(nextState);
                    }
                    break;

                case ExperimentStateRacingGate.Welcome:
                    nextState = ExperimentStateRacingGate.RcControls;
                    TransitionTo(ExperimentStateRacingGate.Wait);
                    break;

                case ExperimentStateRacingGate.RcControls:
                    if (skipCalibration)
                    {
                        nextState = ExperimentStateRacingGate.FlyingPracticeInstr;
                    }
                    else
                    {
                        nextState = ExperimentStateRacingGate.Calibration;
                    }
                    TransitionTo(ExperimentStateRacingGate.Wait);
                    break;

                case ExperimentStateRacingGate.Calibration:
                    if (!isHeadPositionOk)
                    {
                        TransitionTo(ExperimentStateRacingGate.Wait);
                        isHeadPositionOk = true;
                    }
                    if (calibrationDone)
                    {
                        TransitionTo(ExperimentStateRacingGate.FlyingPracticeInstr);
                    }
                    break;

                case ExperimentStateRacingGate.RaceInstructions:
                    if (isTransitionRequested)
                    {
                        isTransitionRequested = false;
                        if (!_demoDotMoving)
                        {
                            // First click: start the dot moving
                            _demoDotMoving = true;
                            if (courseDemo != null)
                                courseDemo.BeginMoving();
                        }
                        else
                        {
                            // Second click: proceed to ReadyScreen
                            _demoDotMoving = false;
                            TransitionTo(ExperimentStateRacingGate.ReadyScreen);
                        }
                    }
                    break;

                case ExperimentStateRacingGate.FlyingPracticeInstr:
                    if (skipFlyingPractice)
                    {
                        TransitionTo(ExperimentStateRacingGate.Feedback);
                    } else {
                        nextState = ExperimentStateRacingGate.FlyingPractice;
                        TransitionTo(ExperimentStateRacingGate.Wait);
                    }
                    break;

                case ExperimentStateRacingGate.FlyingPractice:
                    if (DateTimeOffset.Now.ToUnixTimeMilliseconds() - stateEnterTime > flyingTaskPracticeTime)
                    {
                        nextState = ExperimentStateRacingGate.Feedback;
                        idleRequestDelay = 2; // 2 seconds delay before transitioning to next state
                        TransitionTo(ExperimentStateRacingGate.Idle);
                    }
                    break;


                case ExperimentStateRacingGate.ReadyScreen:
                    nextState = ExperimentStateRacingGate.Trial;
                    TransitionTo(ExperimentStateRacingGate.Wait);
                    break;

                case ExperimentStateRacingGate.Trial:
                    if ((ringGateManager != null && ringGateManager.IsCourseCompleted) || isTransitionRequested)
                    {
                        isTransitionRequested = false;
                        idleRequestDelay = 2;
                        TransitionTo(ExperimentStateRacingGate.Idle);
                    }
                    break;

                case ExperimentStateRacingGate.Feedback:
                    if (feedbackManager.IsSubmitted)
                    {
                        idleRequestDelay = 1;
                        nextState = ExperimentStateRacingGate.Finished;
                        TransitionTo(ExperimentStateRacingGate.IdleSilent);
                    }
                    if (isTransitionRequested)
                    {
                        isTransitionRequested = false;
                        TransitionTo(ExperimentStateRacingGate.RaceInstructions);
                    }
                    break;
            }
        }

        private void TransitionTo(ExperimentStateRacingGate next)
        {
            Debug.Log($"[RacingGate FSM] Transitioning from {state} to {next}");
            ExitState(state);
            EnterState(next);
        }

        private void EnterState(ExperimentStateRacingGate s)
        {
            // Generate course before applying state visuals so the demo spline is ready when OnEnable fires
            if (s == ExperimentStateRacingGate.RaceInstructions && ringGateManager != null)
                ringGateManager.GenerateNewCourse();

            ApplyStateVisuals(s);
            switch (s)
            {
                case ExperimentStateRacingGate.Idle:
                case ExperimentStateRacingGate.Wait:
                    InputManager.Instance.LockControl();
                    break;

                case ExperimentStateRacingGate.WaitForUser:
                    hasUserClicked = false;
                    InputManager.Instance.UnlockControl();
                    break;

                case ExperimentStateRacingGate.Calibration:
                    if (!isHeadPositionOk)
                    {
                        if (eyeTrackerGuide != null)
                            eyeTrackerGuide.TrackBoxGuideActive = true;
                        nextState = ExperimentStateRacingGate.Calibration;
                    }
                    else if (eyeTrackerCalibration != null)
                    {
                        if (eyeTrackerGuide != null)
                            eyeTrackerGuide.TrackBoxGuideActive = false;
                        nextState = ExperimentStateRacingGate.FlyingPracticeInstr;
                        calibrationDone =  false;
                        eyeTrackerCalibration.StartCalibration(
                            resultCallback: (calibResult) =>
                            {
                                if (calibResult)
                                    Debug.Log("Calibration successful");
                                else
                                    Debug.LogError("Calibration failed!");
                                PySender.Instance.IsCalibrationOk = calibResult;
                                calibrationDone = true;
                            }
                        );
                    }
                    InputManager.Instance.LockControl();
                    break;

                case ExperimentStateRacingGate.FlyingPracticeInstr:
                    InputManager.Instance.LockControl();
                    if (viewManager != null)
                        viewManager.ToggleAllViews(true);
                    // Reset swarm to practice spawn position
                    if (swarmSpawner != null)
                    {
                        swarmSpawner.DisableHealthMonitoring();
                        swarmSpawner.ResetToPos(k_swarmSpawnPractice);
                        swarmSpawner.EnableHealthMonitoring();
                    }
                    break;

                case ExperimentStateRacingGate.FlyingPractice:
                    PySender.Instance.EyeDataStreaming = true;
                    InputManager.Instance.UnlockControl();
                    if (cwlController != null)
                        cwlController.SetCWLFeedbackEnabled(false);
                    if (swarmSpawner != null)
                        swarmSpawner.DisableStuckDetection();
                    break;

                case ExperimentStateRacingGate.RaceInstructions:
                    if (viewManager != null)
                        viewManager.ToggleAllViews(false);
                    if (swarmSpawner != null)
                    {
                        swarmSpawner.DisableHealthMonitoring();
                        swarmSpawner.ResetToPos(k_swarmSpawnCourse);
                        swarmSpawner.EnableHealthMonitoring();
                    }
                    InputManager.Instance.LockControl();
                    break;

                case ExperimentStateRacingGate.ReadyScreen:
                    InputManager.Instance.LockControl();
                    PySender.Instance.EyeDataStreaming = true;
                    currentTrial++;
                    if (readyOverlayManager != null)
                        readyOverlayManager.SetTrialNumber(currentTrial, totalTrialNumber);
                    if (feedbackManager != null)
                        feedbackManager.SetTrialNumber(currentTrial);
                    ResetTrialComponents();
                    if (ringGateManager != null)
                        ringGateManager.GenerateNewCourse();
                    break;

                case ExperimentStateRacingGate.Trial:
                    Debug.Log($"Starting Trial {currentTrial}");
                    if (currentTrial >= totalTrialNumber)
                    {
                        nextState = ExperimentStateRacingGate.Feedback;
                    } else
                    {
                        nextState = ExperimentStateRacingGate.Countdown;
                    }
                    // Assign swarm to CWL controller before trial
                    if (cwlController != null && swarmSpawner != null && swarmSpawner.swarm.Count > 0)
                    {
                        cwlController.Swarm = swarmSpawner.swarm;
                        cwlController.SetCWLFeedbackEnabled(false);
                    }
                    if (viewManager != null)
                        viewManager.ToggleAllViews(true);
                    if (swarmSpawner != null)
                        swarmSpawner.EnableStuckDetection();
                    InputManager.Instance.UnlockControl();
                    break;

                case ExperimentStateRacingGate.Countdown:
                    if (timer != null)
                        timer.BeginCountdown();
                    // Re-enable warmup phase for next trial, but keep current parameter values
                    if (cwlController != null)
                        cwlController.EnableWarmup();
                    if (viewManager != null)
                        viewManager.ToggleAllViews(false);
                    nextState = ExperimentStateRacingGate.ReadyScreen;
                    InputManager.Instance.LockControl();
                    break;

                case ExperimentStateRacingGate.Finished:
                    PySender.Instance.EyeDataStreaming = false;
                    // Disable CWL feedback at the end of all trials
                    if (cwlController != null)
                        cwlController.SetCWLFeedbackEnabled(false);
                    if (viewManager != null)
                        viewManager.ToggleAllViews(false);
                    InputManager.Instance.LockControl();
                    currentTrial = 0;
                    break;

                case ExperimentStateRacingGate.Feedback:
                    break;
            }
            state = s;
            stateEnterTime = DateTimeOffset.Now.ToUnixTimeMilliseconds();
        }

        private void ExitState(ExperimentStateRacingGate s)
        {
            previousState = s;
            switch (s)
            {
                case ExperimentStateRacingGate.Idle:
                case ExperimentStateRacingGate.IdleSilent:
                case ExperimentStateRacingGate.Wait:
                case ExperimentStateRacingGate.WaitForUser:
                    DisableAllActiveObjects();
                    break;

                case ExperimentStateRacingGate.Countdown:
                    if (timer != null)
                        timer.StopCountdown();
                    DisableAllActiveObjects();
                    break;

                case ExperimentStateRacingGate.FlyingPracticeInstr:
                    if (skipFlyingPractice)
                        DisableAllActiveObjects();
                    break;

                case ExperimentStateRacingGate.Feedback:
                    DisableAllActiveObjects();
                    break;

                case ExperimentStateRacingGate.RaceInstructions:
                    // Demo GameObjects are hidden by DisableAllActiveObjects via stateConfigs;
                    // CourseDemo.OnDisable() handles cleanup automatically.
                    DisableAllActiveObjects();
                    _demoDotMoving = false;
                    break;

                case ExperimentStateRacingGate.FlyingPractice:
                    PySender.Instance.EyeDataStreaming = false;
                    break;

                case ExperimentStateRacingGate.RacePractice:
                    PySender.Instance.EyeDataStreaming = false;
                    break;

                case ExperimentStateRacingGate.Trial:
                    Debug.Log($"End of Trial #{currentTrial}");
                    PySender.Instance.EyeDataStreaming = false;
                    if (cwlController != null)
                        cwlController.SetCWLFeedbackEnabled(false);
                    break;
            }
        }

        private void ApplyStateVisuals(ExperimentStateRacingGate state)
        {
            for (int i = 0; i < stateConfigs.Count; i++)
            {
                StepConfig cfg = stateConfigs[i];
                if (cfg.state != state) continue;

                for (int e = 0; e < cfg.enableOnState.Count; e++)
                {
                    GameObject go = cfg.enableOnState[e];
                    if (go != null)
                    {
                        go.SetActive(true);
                        if (go.GetComponent<Camera>() != null)
                        {
                            Camera cam = go.GetComponent<Camera>();
                            cam.targetDisplay = 0;
                        }
                    }
                }
            }
        }

        private void DisableAllActiveObjects()
        {
            for (int i = 0; i < stateConfigs.Count; i++)
            {
                StepConfig cfg = stateConfigs[i];
                for (int e = 0; e < cfg.enableOnState.Count; e++)
                {
                    GameObject go = cfg.enableOnState[e];
                    if (go != null)
                        go.SetActive(false);
                }
            }
        }

        private void ReadUserInput()
        {
            if (InputManager.Instance != null)
            {
                if (lastSwitchState < 0 && InputManager.Instance.InputStatus["userSwitch"] > 0)
                {
                    hasUserClicked = true;
                }
                lastSwitchState = InputManager.Instance.InputStatus["userSwitch"] > 0 ? 1 : -1;
            }
            if (Input.GetKeyDown(KeyCode.Return))
                isTransitionRequested = true;
            if (Input.GetKeyDown(KeyCode.Tab))
                hasUserClicked = true;
        }

        /// <summary>
        /// Callback for when the course start trigger fires. Enables CWL feedback and warmup phase
        /// when drones enter the course, signaling the actual start of the race.
        /// </summary>
        private void OnCourseTriggered()
        {
            if (cwlController != null)
            {
                cwlController.SetCWLFeedbackEnabled(isCwlActive);
                if (isCwlActive)
                    cwlController.EnableWarmup();
                else
                    cwlController.SetDefaultProfile(defaultFlightProfile);
                Debug.Log("[ExperimentFSMRacingGate] Course triggered - enabling CWL feedback and warmup phase");
            }
        }

        public override ExperimentFSMBase.ExperimentStateSnapshot GetStateSnapshot()
        {
            return new ExperimentFSMBase.ExperimentStateSnapshot
            {
                state = state.ToString(),
                previousState = previousState.ToString(),
                nextState = nextState.ToString(),
                currentTask = 0,
                currentTrial = currentTrial,
                totalTaskNumber = 1,
                totalTrialNumber = totalTrialNumber,
                stateEnterTimestamp = stateEnterTime,
            };
        }

        public override string[] GetAvailableStates()
        {
            return Enum.GetNames(typeof(ExperimentStateRacingGate));
        }

        public override void NotifyOperatorClicked()
        {
            isTransitionRequested = true;
        }

        public override bool RequestTransitionTo(string stateName, out string error)
        {
            error = string.Empty;
            if (string.IsNullOrWhiteSpace(stateName))
            {
                error = "invalid_state";
                return false;
            }

            if (!Enum.TryParse(stateName, true, out ExperimentStateRacingGate requestedState))
            {
                error = "unknown_state";
                return false;
            }

            TransitionTo(requestedState);
            return true;
        }

        // ─────────────────────────────────────────────────────────────────────────
        // Reset methods
        // ─────────────────────────────────────────────────────────────────────────

        /// <summary>
        /// Performs a complete reset of all trial components.
        /// Call this at the start of each new trial to ensure clean state.
        /// </summary>
        public void ResetTrialComponents()
        {
            // Disable health monitoring during reset to prevent false "dead" markings
            if (swarmSpawner != null)
            {
                swarmSpawner.DisableHealthMonitoring();
                swarmSpawner.ResetToPos(k_swarmSpawnCourse);
                swarmSpawner.EnableHealthMonitoring();
            }

            // Reset course gates and visuals
            if (ringGateManager != null)
            {
                ringGateManager.ResetAll();
            }

            // Reset the course trigger so it can fire for this trial
            if (courseStartTrigger != null)
            {
                courseStartTrigger.ResetTrigger();
            }
        }
    }
}
