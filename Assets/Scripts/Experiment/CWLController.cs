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
        public enum StepMode { Exponential, Linear }

        [Header("Feedback System")]
        [SerializeField] private bool _cwlFeedbackEnabled = true;

        [Header("CWL Adjustment Range")]
        [SerializeField] private FlightProfile _minProfile;   // Low difficulty (Soft) — used when CWL=High
        [SerializeField] private FlightProfile _maxProfile;   // High difficulty (Racing) — used when CWL=Low
        [SerializeField] private FlightProfile _defaultProfile; // Reset target (Average)

        private List<GameObject> swarm;
        public List<GameObject> Swarm
        {
            get => swarm;
            set => swarm = value;
        }

        private CWLLevel _lastCWLLevel = CWLLevel.Medium;

        [Header("Adaptive Algorithm")]
        [SerializeField] private StepMode _stepMode = StepMode.Exponential;
        [SerializeField] private float _exponentialBase = 2.0f;    // Exponential mode: multiplier per same-direction call
        [SerializeField] [Min(1)] private int _numSteps = 16;      // Number of anchor steps covering the full range
        [SerializeField] [Min(1)] private int _maxStepSize = 4;    // Max steps to move on a single inference
        [SerializeField] [Min(0)] private int _warmupUpdates = 2;  // Number of CWL inferences to skip before adjusting

        // Precomputed anchor step arrays (one array per parameter)
        private float[] _stepsSpeed;
        private float[] _stepsYawRate;
        private float[] _stepsPitch;
        private float[] _stepsRoll;
        private float[] _stepsAltRate;
        private float[] _stepsAlpha;

        // Current step index (shared across all parameters since they move together)
        private int _currentStepIdx;

        // Warmup phase tracking
        private int _warmupCounter = 0;
        private bool _warmupComplete = false;

        // Number of steps to move on next adjustment; grows exponentially on same direction
        private float _stepsToMove = 1.0f;
        private CWLLevel? _lastDirection;

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
        /// </summary>
        public void OnCWLInference(string cwlLevelString)
        {
            if (!_cwlFeedbackEnabled)
            {
                Debug.Log($"[CWLController] Received CWL inference '{cwlLevelString}' but feedback system is DISABLED");
                return;
            }

            if (!System.Enum.TryParse(cwlLevelString, true, out CWLLevel cwlLevel))
            {
                Debug.LogError($"[CWLController] Invalid CWL level: {cwlLevelString}. Use 'Low', 'Medium', or 'High'");
                return;
            }

            _lastCWLLevel = cwlLevel;

            // Warmup phase: skip adjustments until warmup period is complete
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
                    return;
                }
            }

            // Determine adjustment direction
            // If CWL is Low: increase difficulty (move towards max)
            // If CWL is High: decrease difficulty (move towards min)
            // If CWL is Medium: no adjustment needed
            AdjustDroneParameters(cwlLevel);
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

            // Initialize step index to the position closest to the default profile (using speed as reference)
            FlightProfile d = _defaultProfile != null ? _defaultProfile : _minProfile;
            _currentStepIdx = FindClosestIndex(_stepsSpeed, d.maxSpeed);

            _stepsToMove = 1.0f;
            _lastDirection = null;
        }

        private void AdjustDroneParameters(CWLLevel cwlLevel)
        {
            if (swarm == null || swarm.Count == 0)
            {
                Debug.LogWarning("[CWLController] No drones found in swarm");
                return;
            }

            // On direction change, reset step count (keep current index)
            bool directionChanged = _lastDirection.HasValue && _lastDirection.Value != cwlLevel;
            if (directionChanged)
                _stepsToMove = 1.0f;

            // Medium is a no-op
            if (cwlLevel == CWLLevel.Medium)
                return;

            // Number of discrete steps to move this call
            int n = Mathf.Max(1, Mathf.RoundToInt(_stepsToMove));
            int delta = (cwlLevel == CWLLevel.High) ? -n : +n;   // High → toward min, Low → toward max

            // Move shared index and clamp to valid range
            _currentStepIdx = Mathf.Clamp(_currentStepIdx + delta, 0, _numSteps - 1);

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

            // Update state: direction and grow step count for next call
            _lastDirection = cwlLevel;
            if (_stepMode == StepMode.Exponential)
                _stepsToMove = Mathf.Min(_stepsToMove * _exponentialBase, _maxStepSize);
            else
                _stepsToMove = Mathf.Min(_stepsToMove + 1f, _maxStepSize);

            Debug.Log($"[CWLController] CWL={cwlLevel} | moved {n} step(s) | index={_currentStepIdx} | " +
                      $"next move={_stepsToMove:F1} step(s)");
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
        /// Returns the current step index (shared across all parameters).
        /// </summary>
        public int GetCurrentStepIdx => _currentStepIdx;

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
            Debug.Log("[CWLController] Reset to medium (anchor step indices and algorithm state)");
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
