using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.InputSystem;

namespace Experiment
{
    public class ExperimentFSMRacingGate : MonoBehaviour
    {
        [Serializable]
        public class ExperimentStateSnapshot
        {
            public long timestamp => DateTimeOffset.UtcNow.ToUnixTimeMilliseconds();
            public string state;
            public string previousState;
            public string nextState;
            public int currentTask;
            public int currentTrial;
            public long stateEnterTimestamp;
        }

        private enum ExperimentStateRacingGate
        {
            Idle,
            IdleSilent,
            Wait,
            Welcome,
            FlyingInstructions,
            FlyingPractice,
            RaceInstructions,
            RacePractice,
            ExperimentBegin,
            Trial,
            Rest,
            Finished,
            WaitForUser,
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
        [SerializeField] private CWLController cwlController;

        [Header("Step Visuals")]
        [SerializeField] private List<StepConfig> stateConfigs = new List<StepConfig>();

        [Header("Settings")]
        [SerializeField] private int trialsPerTask = 4;
        [SerializeField] private int randomSeed = -1;

        [Header("Debug")]
        [SerializeField] private bool isDebugMode = false;
        [SerializeField] private bool skipFlyingPractice = false;
        [SerializeField] private bool skipRacePractice = false;
        [SerializeField] private float practiceTime = 0.25f; // in minutes
        [SerializeField] private float restTime = 0.25f; // in minutes

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
            public float RestTimeBetweenTrials; // in seconds
        }

        private ExperimentSettings defaultSettings = new ExperimentSettings
        {
            FlightPracticeDuration = 3.0f,
            RestTimeBetweenTrials = 8.0f,
        };

        private AudioSource audioSource;
        private ExperimentStateRacingGate state = ExperimentStateRacingGate.Idle;
        private ExperimentStateRacingGate nextState = ExperimentStateRacingGate.Idle;
        private ExperimentStateRacingGate previousState = ExperimentStateRacingGate.Idle;

        private System.Random rng;
        private bool hasUserClicked = false;
        private bool isTransitionRequested = false;
        private bool restEllapsed = false;
        private int lastSwitchState = -1;
        private long stateEnterTime = 0;
        private int idleRequestDelay = -1; // in seconds, -1 means no delay

        private int currentTask = 0;
        private int currentTrial = 0;
        private bool isHeadPositionOk = false;
        private int flyingTaskPracticeTime = (int)TimeSpan.FromMinutes(0.25).TotalMilliseconds;

        void Start()
        {
            audioSource = GetComponent<AudioSource>();

            // Auto-discover CWL controller if not assigned
            if (cwlController == null)
                cwlController = FindObjectOfType<CWLController>();

            if (viewManager != null)
                viewManager.ToggleAllViews(false);
            DisableAllActiveObjects();
            TransitionTo(ExperimentStateRacingGate.Welcome);
            if (audioSource == null)
            {
                Debug.LogWarning("AudioSource not set up for ExperimentFSMRacingGate. No sound will be played during trials.");
            }
            if (timer != null)
                timer.OnCountdownFinished += () => restEllapsed = true;
            rng = (randomSeed >= 0) ? new System.Random(randomSeed) : new System.Random();
            if (InputManager.Instance != null)
            {
                InputManager.Instance.LockControl();
            }
            if (isDebugMode)
            {
                Debug.LogWarning("Racing Gate Experiment is running in DEBUG MODE. Some steps may be skipped and practice times may be shortened.");
                flyingTaskPracticeTime = (int)TimeSpan.FromMinutes(practiceTime).TotalMilliseconds;
                timer.Minutes = restTime;
                timer.Seconds = 0;
            }
            else
            {
                flyingTaskPracticeTime = (int)TimeSpan.FromMinutes(defaultSettings.FlightPracticeDuration).TotalMilliseconds;
                timer.Minutes = defaultSettings.RestTimeBetweenTrials;
                timer.Seconds = 0;
            }
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

                case ExperimentStateRacingGate.Rest:
                    if (restEllapsed)
                    {
                        restEllapsed = false;
                        TransitionTo(nextState);
                    }
                    break;

                case ExperimentStateRacingGate.Welcome:
                    nextState = ExperimentStateRacingGate.FlyingInstructions;
                    TransitionTo(ExperimentStateRacingGate.Wait);
                    break;

                case ExperimentStateRacingGate.FlyingInstructions:
                    nextState = ExperimentStateRacingGate.FlyingPractice;
                    TransitionTo(ExperimentStateRacingGate.Wait);
                    break;

                case ExperimentStateRacingGate.FlyingPractice:
                    if (skipFlyingPractice)
                    {
                        TransitionTo(ExperimentStateRacingGate.RaceInstructions);
                    }
                    if (DateTimeOffset.Now.ToUnixTimeMilliseconds() - stateEnterTime > flyingTaskPracticeTime)
                    {
                        nextState = ExperimentStateRacingGate.RaceInstructions;
                        idleRequestDelay = 2; // 2 seconds delay before transitioning to next state
                        TransitionTo(ExperimentStateRacingGate.Idle);
                    }
                    break;

                case ExperimentStateRacingGate.RaceInstructions:
                    if (skipRacePractice)
                    {
                        nextState = ExperimentStateRacingGate.ExperimentBegin;
                        TransitionTo(ExperimentStateRacingGate.Wait);
                    }
                    else
                    {
                        nextState = ExperimentStateRacingGate.RacePractice;
                        TransitionTo(ExperimentStateRacingGate.WaitForUser);
                    }
                    break;

                case ExperimentStateRacingGate.RacePractice:
                    // TODO: Implement race practice completion logic
                    // This should wait for the course to complete, then transition
                    break;

                case ExperimentStateRacingGate.ExperimentBegin:
                    nextState = ExperimentStateRacingGate.Trial;
                    TransitionTo(ExperimentStateRacingGate.Wait);
                    break;

                case ExperimentStateRacingGate.Trial:
                    // TODO: Implement trial completion logic
                    // This should wait for the course run to finish, then check if more trials are needed
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

                case ExperimentStateRacingGate.FlyingPractice:
                    if (viewManager != null)
                        viewManager.ToggleAllViews(true);
                    PySender.Instance.EyeDataStreaming = true;
                    InputManager.Instance.UnlockControl();
                    break;

                case ExperimentStateRacingGate.RaceInstructions:
                    if (viewManager != null)
                        viewManager.ToggleAllViews(false);
                    InputManager.Instance.LockControl();
                    break;

                case ExperimentStateRacingGate.RacePractice:
                    // TODO: Start race practice run
                    if (ringGateManager != null)
                    {
                        ringGateManager.ResetAll();
                        ringGateManager.StartCourse();
                    }
                    if (viewManager != null)
                        viewManager.ToggleAllViews(true);
                    PySender.Instance.EyeDataStreaming = true;
                    InputManager.Instance.UnlockControl();
                    break;

                case ExperimentStateRacingGate.ExperimentBegin:
                    InputManager.Instance.LockControl();
                    break;

                case ExperimentStateRacingGate.Trial:
                    // TODO: Start race trial
                    currentTrial++;
                    Debug.Log($"Starting Trial {currentTrial} of Task {currentTask}");
                    if (ringGateManager != null)
                    {
                        ringGateManager.ResetAll();
                        ringGateManager.StartCourse();
                    }
                    if (viewManager != null)
                        viewManager.ToggleAllViews(true);
                    PySender.Instance.EyeDataStreaming = true;
                    InputManager.Instance.UnlockControl();
                    break;

                case ExperimentStateRacingGate.Rest:
                    if (timer != null)
                        timer.BeginCountdown();
                    if (viewManager != null)
                        viewManager.ToggleAllViews(false);
                    InputManager.Instance.LockControl();
                    break;

                case ExperimentStateRacingGate.Finished:
                    PySender.Instance.EyeDataStreaming = false;
                    if (viewManager != null)
                        viewManager.ToggleAllViews(false);
                    InputManager.Instance.LockControl();
                    currentTask = 0;
                    currentTrial = 0;
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

                case ExperimentStateRacingGate.Rest:
                    if (timer != null)
                        timer.StopCountdown();
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

        // External transition requests
        public void RequestTransitionToWait() => TransitionTo(ExperimentStateRacingGate.Wait);
        public void RequestTransitionToIdle() => TransitionTo(ExperimentStateRacingGate.Idle);
        public void RequestTransitionToFlyingPractice() => TransitionTo(ExperimentStateRacingGate.FlyingPractice);
        public void RequestTransitionToRacePractice() => TransitionTo(ExperimentStateRacingGate.RacePractice);
        public void RequestTransitionToTrial() => TransitionTo(ExperimentStateRacingGate.Trial);
        public void RequestTransitionToRest() => TransitionTo(ExperimentStateRacingGate.Rest);

        public ExperimentStateSnapshot GetStateSnapshot()
        {
            return new ExperimentStateSnapshot
            {
                state = state.ToString(),
                previousState = previousState.ToString(),
                nextState = nextState.ToString(),
                currentTask = currentTask,
                currentTrial = currentTrial,
                stateEnterTimestamp = stateEnterTime,
            };
        }

        public string[] GetAvailableStates()
        {
            return Enum.GetNames(typeof(ExperimentStateRacingGate));
        }

        public void NotifyOperatorClicked()
        {
            isTransitionRequested = true;
        }

        public bool RequestTransitionTo(string stateName, out string error)
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
        // CWL Feedback Control
        // ─────────────────────────────────────────────────────────────────────────

        /// <summary>
        /// Enable the CWL adaptive feedback system.
        /// When enabled, the system will receive CWL inferences and adjust drone parameters
        /// to maintain a medium (optimal) cognitive workload level.
        /// </summary>
        public void EnableCWLFeedback()
        {
            if (cwlController != null)
                cwlController.SetCWLFeedbackEnabled(true);
            else
                Debug.LogWarning("[ExperimentFSMRacingGate] CWL Controller not found");
        }

        /// <summary>
        /// Disable the CWL adaptive feedback system.
        /// When disabled, no parameter adjustments will be made in response to CWL inferences.
        /// </summary>
        public void DisableCWLFeedback()
        {
            if (cwlController != null)
                cwlController.SetCWLFeedbackEnabled(false);
            else
                Debug.LogWarning("[ExperimentFSMRacingGate] CWL Controller not found");
        }

        /// <summary>
        /// Reset all drone parameters to their medium (optimal) values.
        /// </summary>
        public void ResetDroneParametersToMedium()
        {
            if (cwlController != null)
                cwlController.ResetToMedium();
            else
                Debug.LogWarning("[ExperimentFSMRacingGate] CWL Controller not found");
        }

        /// <summary>
        /// Get whether the CWL feedback system is currently enabled.
        /// </summary>
        public bool IsCWLFeedbackEnabled()
        {
            return cwlController != null && cwlController.IsCWLFeedbackEnabled;
        }
    }
}
