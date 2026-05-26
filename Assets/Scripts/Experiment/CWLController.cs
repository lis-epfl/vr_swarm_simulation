using System.Collections.Generic;
using UnityEngine;

namespace Experiment
{
    /// <summary>
    /// Drone control parameters that can be adjusted by CWL feedback.
    /// </summary>
    public struct DroneParams
    {
        public float maxSpeed;
        public float maxYawRate;
        public float maxPitch;
        public float maxRoll;
        public float maxAltitudeRate;
        public float maxAlpha;
    }

    /// <summary>
    /// CWL (Cognitive Workload Level) feedback controller.
    /// Receives CWL inferences from Python and adjusts drone control parameters
    /// to maintain workload at optimal (Medium) level.
    /// Uses exponential step growth: exponentialBase=2.0 for exponential, 1.0 for linear.
    /// </summary>
    public class CWLController : MonoBehaviour
    {
        public enum CWLLevel { Low, Medium, High }
        public enum StepMode { Linear, Proportional }

        [Header("Feedback System")]
        [SerializeField] private bool _cwlFeedbackEnabled = true;

        [Header("CWL Adjustment Range")]
        [SerializeField] private FlightProfile _minProfile;
        [SerializeField] private FlightProfile _maxProfile;
        [SerializeField] private FlightProfile _defaultProfile;

        private List<GameObject> swarm;
        public List<GameObject> Swarm
        {
            get => swarm;
            set => swarm = value;
        }

        private CWLLevel _lastCWLLevel = CWLLevel.Medium;

        [Header("Adaptive Algorithm")]
        [SerializeField] [Min(1)] private int _numSteps = 16;           // Total anchor steps across the full range
        [SerializeField] [Min(1)] private int _maxStepSize = 4;         // Steps to move at maximum model confidence
        [SerializeField] [Min(0)] private int _warmupUpdates = 2;       // Inferences to skip before adjusting
        [SerializeField] private StepMode _stepMode = StepMode.Proportional;
        [SerializeField] private bool _enableSharpSwitchBuffer = true;  // Buffer 0 steps on direct Low↔High switch

        // Precomputed anchor step arrays (one array per parameter)
        private float[] _stepsSpeed;
        private float[] _stepsYawRate;
        private float[] _stepsPitch;
        private float[] _stepsRoll;
        private float[] _stepsAltRate;
        private float[] _stepsAlpha;

        // Current step index (only updated when feedback is enabled)
        private int _currentStepIdx;

        // Recommended step index (always updated, for logging during non-adaptive experiments)
        private int _recommendedStepIdx;

        // Warmup phase tracking
        private int _warmupCounter = 0;
        private bool _warmupComplete = false;

        // Last level received (including Medium); used to detect sharp Low↔High switches
        private CWLLevel? _lastDirection;

        // Linear mode step counter (starts at 1, increments each same-direction call, resets on direction change)
        private int _linearStepCount = 1;

        private void Start()
        {
            if (_minProfile == null || _maxProfile == null)
            {
                Debug.LogError("[CWLController] _minProfile and _maxProfile must be assigned.");
                return;
            }
            PrecomputeAnchorSteps();
        }

        /// <summary>
        /// Toggle the CWL feedback system on/off.
        /// When disabled, no parameter adjustments are made.
        /// </summary>
        public void SetCWLFeedbackEnabled(bool enabled)
        {
            _cwlFeedbackEnabled = enabled;
            Debug.Log($"[CWLController] CWL Feedback System: {(enabled ? "ENABLED" : "DISABLED")}");
        }

        public bool IsCWLFeedbackEnabled => _cwlFeedbackEnabled;
        public FlightProfile MaxProfile => _maxProfile;

        /// <summary>
        /// Enable the warmup phase. Called at the start of each trial to skip initial CWL inferences
        /// while the system stabilizes. Keeps current drone parameter values but resets warmup counter.
        /// </summary>
        public void EnableWarmup()
        {
            _warmupCounter = 0;
            _warmupComplete = false;
            Debug.Log("[CWLController] Warmup phase enabled");
        }

        public void SetDefaultProfile(FlightProfile profile)
        {
            _defaultProfile = profile;
            ResetToDefaultProfile();
            Debug.Log("[CWLController] CWL feedback disabled, default profile set, and parameters resetted");
        }

        /// <summary>
        /// Receive CWL inference from Python and adjust drone parameters.
        /// Called via API endpoint POST /api/cwl/level
        /// lowProb/medProb/highProb are the raw softmax probabilities from the model.
        /// </summary>
        public void OnCWLInference(string cwlLevelString, float lowProb = 0.334f, float medProb = 0.333f, float highProb = 0.333f)
        {
            if (!System.Enum.TryParse(cwlLevelString, true, out CWLLevel cwlLevel))
            {
                Debug.LogError($"[CWLController] Invalid CWL level: {cwlLevelString}. Use 'Low', 'Medium', or 'High'");
                return;
            }

            _lastCWLLevel = cwlLevel;

            // Warmup phase: skip adjustments until warmup period is complete (but always compute recommendations)
            bool applyNow = _warmupComplete;
            if (!_warmupComplete)
            {
                if (_warmupCounter < _warmupUpdates)
                {
                    _warmupCounter++;
                    Debug.Log($"[CWLController] Warmup phase {_warmupCounter}/{_warmupUpdates}");
                    if (_warmupCounter == _warmupUpdates)
                    {
                        _warmupComplete = true;
                        Debug.Log("[CWLController] Warmup phase complete, CWL adjustments will be enabled starting from next inference");
                    }
                }
            }

            // Always compute the recommended step (for logging), but only apply if enabled and warmup complete
            AdjustDroneParameters(cwlLevel, lowProb, medProb, highProb, applyNow && _cwlFeedbackEnabled);
        }

        /// <summary>
        /// Build an evenly spaced array of step values from min to max.
        /// Returns array of length _numSteps where arr[0]=min and arr[_numSteps-1]=max.
        /// </summary>
        private float[] BuildStepArray(float min, float max)
        {
            var arr = new float[_numSteps];
            for (int i = 0; i < _numSteps; i++)
            {
                arr[i] = (_numSteps == 1)
                    ? min
                    : Mathf.Lerp(min, max, (float)i / (_numSteps - 1));
            }
            return arr;
        }

        /// <summary>
        /// Find the index in a step array whose value is closest to the target value.
        /// </summary>
        private int FindClosestIndex(float[] steps, float value)
        {
            int best = 0;
            float bestDist = Mathf.Abs(steps[0] - value);
            for (int i = 1; i < steps.Length; i++)
            {
                float d = Mathf.Abs(steps[i] - value);
                if (d < bestDist)
                {
                    bestDist = d;
                    best = i;
                }
            }
            return best;
        }

        /// <summary>
        /// Precompute all 6 anchor step arrays from the min/max profiles,
        /// and initialize the shared step index to the closest position to the default profile.
        /// </summary>
        private void PrecomputeAnchorSteps()
        {
            _stepsSpeed    = BuildStepArray(_minProfile.maxSpeed,        _maxProfile.maxSpeed);
            _stepsYawRate  = BuildStepArray(_minProfile.maxYawRate,      _maxProfile.maxYawRate);
            _stepsPitch    = BuildStepArray(_minProfile.maxPitch,        _maxProfile.maxPitch);
            _stepsRoll     = BuildStepArray(_minProfile.maxRoll,         _maxProfile.maxRoll);
            _stepsAltRate  = BuildStepArray(_minProfile.maxAltitudeRate, _maxProfile.maxAltitudeRate);
            _stepsAlpha    = BuildStepArray(_minProfile.maxAlpha,        _maxProfile.maxAlpha);

            // Initialize step indices to the position closest to the default profile (using speed as reference)
            FlightProfile d = _defaultProfile != null ? _defaultProfile : _minProfile;
            _currentStepIdx = FindClosestIndex(_stepsSpeed, d.maxSpeed);
            _recommendedStepIdx = _currentStepIdx;

            _linearStepCount = 1;
            _lastDirection = null;
        }

        private void AdjustDroneParameters(CWLLevel cwlLevel, float lowProb, float medProb, float highProb, bool applyToSwarm)
        {
            // Medium is a no-op; still update direction so Low→Medium→High doesn't trigger buffer
            if (cwlLevel == CWLLevel.Medium)
            {
                _lastDirection = cwlLevel;
                return;
            }

            // Detect direction change (any non-Medium level different from last non-Medium level)
            bool directionChanged = _lastDirection.HasValue && _lastDirection.Value != CWLLevel.Medium && _lastDirection.Value != cwlLevel;

            // Sharp Low↔High switch → buffer this call (0 steps) if enabled
            bool isSharpSwitch = _enableSharpSwitchBuffer && directionChanged && _lastDirection.Value != CWLLevel.Medium;
            _lastDirection = cwlLevel;
            if (isSharpSwitch)
            {
                if (_stepMode == StepMode.Linear)
                    _linearStepCount = 1;
                Debug.Log($"[CWLController] CWL={cwlLevel} | sharp direction switch — buffering (0 steps)");
                return;
            }

            // Reset linear step count on direction change
            if (directionChanged && _stepMode == StepMode.Linear)
                _linearStepCount = 1;

            // Determine step count based on selected mode
            int stepCount;
            if (_stepMode == StepMode.Linear)
            {
                stepCount = _linearStepCount;
                _linearStepCount = Mathf.Min(_linearStepCount + 1, _maxStepSize);
            }
            else // Proportional
            {
                stepCount = ComputeStepCount(lowProb, medProb, highProb, cwlLevel);
            }

            int delta = (cwlLevel == CWLLevel.High) ? -stepCount : +stepCount;

            // Always update recommended step (for logging)
            _recommendedStepIdx = Mathf.Clamp(_recommendedStepIdx + delta, 0, _numSteps - 1);

            // Only apply to swarm if feedback is enabled
            if (applyToSwarm && swarm != null && swarm.Count > 0)
            {
                _currentStepIdx = _recommendedStepIdx;

                // Write step values to all drones
                foreach (GameObject drone in swarm)
                {
                    Transform droneParent = drone.transform.Find("DroneParent");
                    if (droneParent == null) continue;
                    VelocityControl ctrl = droneParent.GetComponent<VelocityControl>();
                    if (ctrl == null) continue;

                    ctrl.maxSpeed        = _stepsSpeed[_currentStepIdx];
                    ctrl.maxYawRate      = _stepsYawRate[_currentStepIdx];
                    ctrl.maxPitch        = _stepsPitch[_currentStepIdx];
                    ctrl.maxRoll         = _stepsRoll[_currentStepIdx];
                    ctrl.maxAltitudeRate = _stepsAltRate[_currentStepIdx];
                    ctrl.maxAlpha        = _stepsAlpha[_currentStepIdx];
                }
            }

            string applyStatus = applyToSwarm ? "applied" : "logged only";
            Debug.Log($"[CWLController] CWL={cwlLevel} | mode={_stepMode} | steps={stepCount} | recommended_idx={_recommendedStepIdx} | {applyStatus}");
        }

        private int ComputeStepCount(float lowProb, float medProb, float highProb, CWLLevel cwlLevel)
        {
            float winnerProb  = cwlLevel == CWLLevel.Low ? lowProb : highProb;
            float secondBest  = cwlLevel == CWLLevel.Low
                ? Mathf.Max(medProb, highProb)
                : Mathf.Max(lowProb, medProb);
            float margin = Mathf.Clamp01(winnerProb - secondBest);
            return Mathf.Clamp(1 + Mathf.RoundToInt(margin * (_maxStepSize - 1)), 1, _maxStepSize);
        }

        /// <summary>
        /// Returns the current CWL level (last inference received).
        /// </summary>
        public CWLLevel GetCurrentCWLLevel => _lastCWLLevel;

        /// <summary>
        /// Returns the total number of steps in the anchor arrays.
        /// </summary>
        public int GetNumSteps => _numSteps;

        /// <summary>
        /// Returns the current step index (shared across all parameters). Only updated when feedback is enabled.
        /// </summary>
        public int GetCurrentStepIdx => _currentStepIdx;

        /// <summary>
        /// Returns the recommended step index (what the model recommends, always updated for logging).
        /// </summary>
        public int GetRecommendedStepIdx => _recommendedStepIdx;

        /// <summary>
        /// Reset all drone parameters to the midpoint between min and max (neutral starting point).
        /// The system will then naturally settle to the optimal equilibrium based on CWL feedback.
        /// </summary>
        public void ResetToDefaultProfile()
        {
            if (swarm == null || swarm.Count == 0)
            {
                Debug.LogWarning("[CWLController] No drones found in swarm");
                return;
            }

            if (_minProfile == null || _maxProfile == null)
            {
                Debug.LogWarning("[CWLController] _minProfile and _maxProfile must be assigned");
                return;
            }

            // Rebuild anchor steps and reset index to default profile position
            PrecomputeAnchorSteps();

            // Write the initial anchor values to all drones
            foreach (GameObject drone in swarm)
            {
                Transform droneParent = drone.transform.Find("DroneParent");
                if (droneParent == null)
                    continue;

                VelocityControl ctrl = droneParent.GetComponent<VelocityControl>();
                if (ctrl == null)
                    continue;

                ctrl.maxSpeed        = _stepsSpeed[_currentStepIdx];
                ctrl.maxYawRate      = _stepsYawRate[_currentStepIdx];
                ctrl.maxPitch        = _stepsPitch[_currentStepIdx];
                ctrl.maxRoll         = _stepsRoll[_currentStepIdx];
                ctrl.maxAltitudeRate = _stepsAltRate[_currentStepIdx];
                ctrl.maxAlpha        = _stepsAlpha[_currentStepIdx];
            }

            _lastCWLLevel  = CWLLevel.Medium;
            _warmupCounter = 0;
            _warmupComplete = false;
            Debug.Log("[CWLController] Reset to medium (current={_currentStepIdx}, recommended={_recommendedStepIdx})");
        }

        public Dictionary<string, float> GetCurrentControlLimits()
        {
            if (swarm == null || swarm.Count == 0)
            {
                Debug.LogWarning("[CWLController] No drones found in swarm");
                return new Dictionary<string, float>();
            }

            // Assuming all drones have the same parameters, we can just read from the first one
            VelocityControl ctrl = swarm[0].transform.Find("DroneParent").gameObject.GetComponent<VelocityControl>();
            if (ctrl == null)
            {
                Debug.LogWarning($"[CWLController] Drone {swarm[0].name} has no VelocityControl component");
                return new Dictionary<string, float>();
            }

            return new Dictionary<string, float>
            {
                {"maxSpeed", ctrl.maxSpeed},
                {"maxYawRate", ctrl.maxYawRate},
                {"maxPitch", ctrl.maxPitch},
                {"maxRoll", ctrl.maxRoll},
                {"maxAltitudeRate", ctrl.maxAltitudeRate},
                {"maxAlpha", ctrl.maxAlpha}
            };
        }
    }
}
