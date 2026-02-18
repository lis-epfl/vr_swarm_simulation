using System;
using System.Collections;
using System.Collections.Generic;
using System.Diagnostics.Tracing;
using Tobii.Research.Unity;
using Unity.VisualScripting;
using UnityEngine;
using UnityEngine.InputSystem;

namespace Experiment
{
    public class ExperimentFSM : MonoBehaviour
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
            public int[] nbackLevelsOrder;
            public int currentNBackLevel;
            public long stateEnterTimestamp;
        }

        private enum ExperimentState
        {
            Idle,
            IdleSilent,
            Wait,
            Welcome,
            RcControls,
            Calibration,
            FlyingInstructions,
            FlyingPractice,
            NBackInstructions,
            NBackPractice,
            ExperimentBegin,
            Task,
            Countdown,
            Trial,
            Finished,
            WaitForUser,
        }

        [Serializable]
        private class StepConfig
        {
            public ExperimentState state;
            public List<GameObject> enableOnState = new List<GameObject>();
        }

        [Header("References")]
        [SerializeField] private TimerCountdown timer;
        [SerializeField] private ViewManager viewManager;
        [SerializeField] private CustomCalibration eyeTrackerCalibration;
        [SerializeField] private CustomTrackBoxGuide eyeTrackerGuide;
        [SerializeField] private swarmSpawn swarmSpawner;
        [SerializeField] private TaskOverlayManager taskOverlayManager;

        [Header("Step Visuals")]
        [SerializeField] private List<StepConfig> stateConfigs = new List<StepConfig>();

        [Header("Settings")]
        [SerializeField] private int maxPracticeLevel = 2;
        [SerializeField] private int trialsPerTask = 4;
        [SerializeField] private int randomSeed = -1;
        [SerializeField] private int[] nBackLevelsOrder = new int[] { 0, 1, 2 };


        [Header("Debug")]
        [SerializeField] private bool isDebugMode = false;
        [SerializeField] private bool skipCalibration = false;
        [SerializeField] private bool skipFlyingPractice = false;
        [SerializeField] private bool skipNBackPractice = false;
        [SerializeField] private float practiceTime = 0.25f; // in minutes
        [SerializeField] private float countdownTime = 0.25f; // in minutes
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
            public int NBackSequenceLength;
            public int TrialsPerTask;
            public float RestTimeBetweenTasks; // in minutes
            public float RestTimeBetweenTrials; // in sconds
        }

        private ExperimentSettings defaultSettings = new ExperimentSettings
        {
            FlightPracticeDuration = 3.0f,
            NBackSequenceLength = 20,
            RestTimeBetweenTasks = 2.0f,
            RestTimeBetweenTrials = 8.0f,
        };

        private NBackTask nBackTask;
        private AudioSource audioSource;
        private ExperimentState state = ExperimentState.Idle;
        private ExperimentState nextState = ExperimentState.Idle;
        private ExperimentState previousState = ExperimentState.Idle;

        private System.Random rng;
        private bool hasUserClicked = false;
        private bool isTransitionRequested = false;
        private bool countdownEllapsed = false;
        private int lastSwitchState = -1;
        private long stateEnterTime = 0;
        private int idleRequestDelay = -1; // in seconds, -1 means no delay

        private int currentPracticeLevel = 0;
        private bool nBackCompletedFlag = false;
        private int currentTask = 0;
        private int currentTrial = 0;
        private bool calibrationDone = false;
        private bool isHeadPositionOk = false;
        private int flyingTaskPracticeTime = (int)TimeSpan.FromMinutes(0.25).TotalMilliseconds;

        private const String k_trialFinishClipName = "TrialFinishSound";
        private const String k_lastTrialClipName = "TrialStartSoundFinal";
        private String trialStartClip = "TrialStartSound#";

        void Start()
        {
            // Retrieve components from this object
            nBackTask = GetComponent<NBackTask>();
            audioSource = GetComponent<AudioSource>();
            if (viewManager != null)
                viewManager.ToggleAllViews(false);
            DisableAllActiveObjects();
            TransitionTo(ExperimentState.Welcome);
            if (nBackTask != null)
            {
                nBackTask.Completed += () => nBackCompletedFlag = true;
            }
            if (audioSource == null)
            {
                Debug.LogWarning("AudioSource not set up for ExperimentFSM. No sound will be played at the end of trials.");
            }
            if (timer != null)
                timer.OnCountdownFinished += () => countdownEllapsed = true;
            rng = (randomSeed >= 0) ? new System.Random(randomSeed) : new System.Random();
            if (InputManager.Instance != null)
            {
                InputManager.Instance.LockControl();
            }
            if (isDebugMode)
            {
                Debug.LogWarning("Experiment is running in DEBUG MODE. Some steps may be skipped and practice times may be shortened.");
                flyingTaskPracticeTime = (int)TimeSpan.FromMinutes(practiceTime).TotalMilliseconds;
                timer.Minutes = countdownTime;
                timer.Seconds = 0;
            }
            else
            {
                flyingTaskPracticeTime = (int)TimeSpan.FromMinutes(defaultSettings.FlightPracticeDuration).TotalMilliseconds;
                timer.Minutes = defaultSettings.RestTimeBetweenTasks;
                timer.Seconds = 0;
                nBackTask.TotalStimuli = (uint)defaultSettings.NBackSequenceLength;
                nBackTask.InitialDelay = defaultSettings.RestTimeBetweenTrials;
            }
        }

        void Update()
        {
            ReadUserInput();

            switch (state)
            {
                case ExperimentState.Idle:
                case ExperimentState.IdleSilent:
                    if (idleRequestDelay > 0 && DateTimeOffset.Now.ToUnixTimeSeconds() - stateEnterTime / 1000 > idleRequestDelay)
                    {
                        TransitionTo(nextState);
                    }
                    break;

                case ExperimentState.Wait:
                    if (isTransitionRequested)
                    {
                        isTransitionRequested = false;
                        TransitionTo(nextState);
                    }
                    break;
                case ExperimentState.WaitForUser:
                    if (hasUserClicked)
                    {
                        hasUserClicked = false;
                        TransitionTo(nextState);
                    }
                    break;
                
                case ExperimentState.Countdown:
                    if (countdownEllapsed)
                    {
                        countdownEllapsed = false;
                        TransitionTo(ExperimentState.Task);
                    }
                    break;

                case ExperimentState.Welcome:
                    nextState = ExperimentState.FlyingInstructions;
                    TransitionTo(ExperimentState.Wait);
                    break;

                case ExperimentState.RcControls:
                    if (skipCalibration)
                    {
                        nextState = ExperimentState.FlyingPractice;
                    }
                    else
                    {                        
                        nextState = ExperimentState.Calibration;
                    }
                    TransitionTo(ExperimentState.Wait);
                    break;

                case ExperimentState.FlyingInstructions:
                    nextState = ExperimentState.RcControls;
                    TransitionTo(ExperimentState.Wait);
                    break;

                case ExperimentState.Calibration:
                    if (!isHeadPositionOk)
                    {
                        TransitionTo(ExperimentState.Wait);
                        isHeadPositionOk = true;
                    }
                    else if (calibrationDone)
                    {
                        TransitionTo(nextState);
                    }
                    break;

                case ExperimentState.FlyingPractice:
                    if (skipFlyingPractice)
                    {
                        TransitionTo(ExperimentState.NBackInstructions);
                    }
                    if (DateTimeOffset.Now.ToUnixTimeMilliseconds() - stateEnterTime > flyingTaskPracticeTime)
                    {
                        nextState = ExperimentState.NBackInstructions;
                        idleRequestDelay = 2; // 2 seconds delay before transitioning to next state
                        TransitionTo(ExperimentState.Idle);
                    }
                    break;
                
                case ExperimentState.NBackInstructions:
                    if (skipNBackPractice)
                    {
                        nextState = ExperimentState.ExperimentBegin;
                        TransitionTo(ExperimentState.Wait);
                    }
                    else 
                    {
                        nextState = ExperimentState.NBackPractice;
                        TransitionTo(ExperimentState.WaitForUser);
                    }
                    break;

                case ExperimentState.NBackPractice:
                    if (nBackCompletedFlag)
                    {
                        nBackCompletedFlag = false;
                        TransitionTo(nextState);
                    }
                    break;
                
                case ExperimentState.ExperimentBegin:
                    // Transition to the main experiment task or trial state as needed
                    nextState = ExperimentState.Task; // or ExperimentState.Trial based on your design
                    TransitionTo(ExperimentState.Wait);
                    break;

                case ExperimentState.Task:
                    nextState = ExperimentState.Trial;
                    TransitionTo(ExperimentState.Wait);
                    break;

                case ExperimentState.Trial:
                    // Wait for current nback to finish
                    if (nBackCompletedFlag)
                    {
                        nBackCompletedFlag = false;
                        // Check if another trial required
                        if (currentTrial < trialsPerTask)
                        {
                            idleRequestDelay = 5; // 5 seconds delay before transitioning to next task
                            TransitionTo(ExperimentState.IdleSilent);
                        }
                        else
                        {
                            idleRequestDelay = 2;
                            TransitionTo(ExperimentState.Idle);
                        }
                    }
                    break;
            }
        }

        private void TransitionTo(ExperimentState next)
        {
            Debug.Log($"Transitioning from {state} to {next}");
            ExitState(state);
            EnterState(next);
        }

        private void EnterState(ExperimentState s)
        {
            ApplyStateVisuals(s);
            switch (s)
            {
                case ExperimentState.Idle:
                case ExperimentState.Wait:
                    InputManager.Instance.LockControl();
                    break;
                case ExperimentState.WaitForUser:
                    hasUserClicked = false;
                    InputManager.Instance.UnlockControl();
                    break;
                case ExperimentState.Calibration:
                    if (!isHeadPositionOk)
                    {
                        if (eyeTrackerGuide != null)
                            eyeTrackerGuide.TrackBoxGuideActive = true;
                        nextState = ExperimentState.Calibration;
                    }
                    else if (eyeTrackerCalibration != null)
                    {
                        if (eyeTrackerGuide != null)
                            eyeTrackerGuide.TrackBoxGuideActive = false;
                        nextState = ExperimentState.FlyingPractice;
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
                case ExperimentState.FlyingPractice:
                    if (viewManager != null)
                        viewManager.ToggleAllViews(true);
                    PySender.Instance.EyeDataStreaming = true;
                    InputManager.Instance.UnlockControl();
                    break;
                case ExperimentState.NBackInstructions:
                    if (viewManager != null)
                        viewManager.ToggleAllViews(false);
                    InputManager.Instance.LockControl();
                    break;
                case ExperimentState.NBackPractice:
                    nBackTask.IsPracticeMode = true;
                    PySender.Instance.EyeDataStreaming = true;
                    if (currentPracticeLevel >= maxPracticeLevel)
                    {
                        nextState = ExperimentState.ExperimentBegin;
                    }
                    else
                    {
                        nextState = ExperimentState.NBackInstructions;
                    }
                    nBackTask.setNBack(currentPracticeLevel);
                    nBackTask.beginTask();
                    hasUserClicked = false;
                    InputManager.Instance.UnlockControl();
                    break;
                case ExperimentState.ExperimentBegin:
                    nBackTask.IsPracticeMode = false;
                    nBackLevelsOrder = shuffleArray(nBackLevelsOrder);
                    Debug.Log("Shuffled N-Back Levels Order: " + string.Join(", ", nBackLevelsOrder));
                    InputManager.Instance.LockControl();
                    break;
                case ExperimentState.Task:
                    currentTask++;
                    if (swarmSpawner != null)
                        swarmSpawner.Reset();
                    Debug.Log($"Starting Task {currentTask} (N-Back Level: {nBackLevelsOrder[currentTask-1]})");
                    taskOverlayManager.SetTaskNumber(currentTask);
                    taskOverlayManager.setNBackLevel(nBackLevelsOrder[currentTask-1]);
                    nBackTask.setNBack(nBackLevelsOrder[currentTask-1]);
                    if (viewManager != null)
                        viewManager.ToggleAllViews(true);
                    break;
                case ExperimentState.Trial:
                    Debug.Log($"Starting Trial {currentTrial} of Task {currentTask}");
                    hasUserClicked = false;
                    PySender.Instance.EyeDataStreaming = true;
                    nBackTask.beginTask();
                    InputManager.Instance.UnlockControl();
                    String audioClip = trialStartClip.Replace("#", currentTrial.ToString());
                    if (currentTrial < trialsPerTask)
                    {
                        nextState = ExperimentState.Trial;
                    }
                    else
                    {
                        audioClip = k_lastTrialClipName;
                        nextState = (currentTask >= nBackLevelsOrder.Length) ? ExperimentState.Finished : ExperimentState.Countdown;
                    }
                    if (audioSource != null)
                    {   
                        AudioClip clip = Resources.Load<AudioClip>("Audio/" + audioClip);
                        if (clip != null)
                            audioSource.PlayOneShot(clip);
                    }
                    break;
                case ExperimentState.Countdown:
                    PySender.Instance.EyeDataStreaming = false;
                    if (timer != null)
                        timer.BeginCountdown();
                    if (viewManager != null)
                        viewManager.ToggleAllViews(false);
                    InputManager.Instance.LockControl();
                    break;
                case ExperimentState.Finished:
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

        private int[] shuffleArray(int[] array)
        {
            int n = array.Length;
            while (n > 1) 
            {
                int k = rng.Next(n--);
                int temp = array[n];
                array[n] = array[k];
                array[k] = temp;
            }
            return array;
        }


        private void ExitState(ExperimentState s)
        {
            previousState = s;
            switch (s)
            {
                case ExperimentState.Idle:
                case ExperimentState.IdleSilent:
                case ExperimentState.Wait:
                case ExperimentState.WaitForUser:
                    DisableAllActiveObjects();
                    break;
                case ExperimentState.Countdown:
                    if (timer != null)
                        timer.StopCountdown();
                    break;
                case ExperimentState.FlyingPractice:
                    PySender.Instance.EyeDataStreaming = false;
                    break;
                case ExperimentState.NBackPractice:
                    PySender.Instance.EyeDataStreaming = false;
                    currentPracticeLevel++;
                    break;
                case ExperimentState.Task:
                    Debug.Log($"Ending Task #{currentTask}");
                    currentTrial = 1;
                    break;
                case ExperimentState.Trial:
                    Debug.Log($"End of Trial #{currentTrial}");
                    PySender.Instance.EyeDataStreaming = false;
                    if (audioSource != null)
                    {
                        AudioClip clip = Resources.Load<AudioClip>("Audio/" + k_trialFinishClipName);
                        if (clip != null)
                            audioSource.PlayOneShot(clip);
                    }
                    currentTrial++;
                    break;
            }
        }
        private void ApplyStateVisuals(ExperimentState state)
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

        private void DisableActiveStateObjects(ExperimentState s)
        {
            for (int i = 0; i < stateConfigs.Count; i++)
            {
                StepConfig cfg = stateConfigs[i];
                if (cfg.state != s) continue;

                for (int e = 0; e < cfg.enableOnState.Count; e++)
                {
                    GameObject go = cfg.enableOnState[e];
                    if (go != null)
                        go.SetActive(false);
                }

                break;
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
        public void RequestTransitionToWait() => TransitionTo(ExperimentState.Wait);
        public void RequestTransitionToIdle() => TransitionTo(ExperimentState.Idle);
        public void RequestTransitionToFlyingPractice() => TransitionTo(ExperimentState.FlyingPractice);
        public void RequestTransitionToNBackPractice() => TransitionTo(ExperimentState.NBackPractice);
        public void RequestTransitionToCountdown() => TransitionTo(ExperimentState.Countdown);
        public void RequestTransitionToTask() => TransitionTo(ExperimentState.Task);
        public void RequestTransitionToTrial() => TransitionTo(ExperimentState.Trial);

        public ExperimentStateSnapshot GetStateSnapshot()
        {
            return new ExperimentStateSnapshot
            {
                state = state.ToString(),
                previousState = previousState.ToString(),
                nextState = nextState.ToString(),
                currentTask = currentTask,
                currentTrial = currentTrial,
                nbackLevelsOrder = nBackLevelsOrder,
                currentNBackLevel = (currentTask > 0 && currentTask <= nBackLevelsOrder.Length) ? nBackLevelsOrder[currentTask - 1] : currentPracticeLevel,
                stateEnterTimestamp = stateEnterTime,
            };
        }

        public string[] GetAvailableStates()
        {
            return Enum.GetNames(typeof(ExperimentState));
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

            if (!Enum.TryParse(stateName, true, out ExperimentState requestedState))
            {
                error = "unknown_state";
                return false;
            }

            TransitionTo(requestedState);
            return true;
        }
    }
}