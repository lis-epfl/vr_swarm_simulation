using System.Collections.Generic;
using UnityEngine;

namespace Experiment
{
    /// <summary>
    /// Adaptive CWL (Cognitive Workload Level) feedback system.
    /// Receives CWL inference from Python and adjusts drone control parameters
    /// to maintain the CWL at a medium (optimal) level.
    ///
    /// Architecture:
    /// - Each adjustable parameter (maxSpeed, maxYawRate, etc.) has min/medium/max values
    /// - When CWL is LOW: increase parameters towards max (increase difficulty)
    /// - When CWL is HIGH: decrease parameters towards min (decrease difficulty)
    /// - When CWL is MEDIUM: hold steady (no adjustment needed)
    /// - Adjustments happen incrementally per call using step sizes
    /// </summary>
    public class CWLController : MonoBehaviour
    {
        public enum CWLLevel { Low, Medium, High }

        [Header("Feedback System")]
        [SerializeField] private bool _cwlFeedbackEnabled = true;

        [System.Serializable]
        public struct ParameterRange
        {
            [Tooltip("Minimum value (used when CWL is High, difficulty needs to decrease)")]
            public float min;

            [Tooltip("Maximum value (used when CWL is Low, difficulty needs to increase)")]
            public float max;

            [Tooltip("Amount to adjust per step (will settle at optimal equilibrium)")]
            public float stepSize;

            public ParameterRange(float min, float max, float stepSize)
            {
                this.min = min;
                this.max = max;
                this.stepSize = stepSize;
            }
        }

        [Header("Parameter Ranges (min/max, will settle at optimal equilibrium)")]
        [SerializeField] private ParameterRange maxSpeedRange = new(3.0f, 15.0f, 0.5f);
        [SerializeField] private ParameterRange maxYawRateRange = new(0.6f, 1.5f, 0.05f);
        [SerializeField] private ParameterRange maxPitchRange = new(0.15f, 0.45f, 0.01f);
        [SerializeField] private ParameterRange maxRollRange = new(0.15f, 0.45f, 0.01f);
        [SerializeField] private ParameterRange maxAltitudeRateRange = new(1.5f, 5.0f, 0.1f);
        [SerializeField] private ParameterRange maxAlphaRange = new(6.0f, 15.0f, 0.5f);

        private List<GameObject> swarm;
        public List<GameObject> Swarm
        {
            get => swarm;
            set => swarm = value;
        }

        private CWLLevel _lastCWLLevel = CWLLevel.Medium;

        [Header("Adaptive Algorithm")]
        [SerializeField] private float _exponentialBase = 2.0f;

        private CWLAdaptiveAlgorithm _adaptiveAlgorithm;

        private void Awake()
        {
        }

        private void Start()
        {
            var config = new CWLAdaptiveAlgorithmConfig
            {
                maxSpeedRange        = maxSpeedRange,
                maxYawRateRange      = maxYawRateRange,
                maxPitchRange        = maxPitchRange,
                maxRollRange         = maxRollRange,
                maxAltitudeRateRange = maxAltitudeRateRange,
                maxAlphaRange        = maxAlphaRange,
            };
            _adaptiveAlgorithm = new CWLAdaptiveAlgorithm(config, _exponentialBase);
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

            // Determine adjustment direction
            // If CWL is Low: increase difficulty (move towards max)
            // If CWL is High: decrease difficulty (move towards min)
            // If CWL is Medium: no adjustment needed
            AdjustDroneParameters(cwlLevel);
        }

        /// <summary>
        /// Adjusts all drone parameters based on the current CWL level using the adaptive algorithm.
        /// The algorithm applies exponential step growth with direction change detection.
        /// </summary>
        private void AdjustDroneParameters(CWLLevel cwlLevel)
        {
            if (swarm == null || swarm.Count == 0)
            {
                Debug.LogWarning("[CWLController] No drones found in swarm");
                return;
            }

            foreach (GameObject drone in swarm)
            {
                Transform droneParentTransform = drone.transform.Find("DroneParent");
                if (droneParentTransform == null)
                {
                    Debug.LogWarning($"[CWLController] Drone {drone.name} has no 'DroneParent' child");
                    continue;
                }

                VelocityControl ctrl = droneParentTransform.GetComponent<VelocityControl>();
                if (ctrl == null)
                {
                    Debug.LogWarning($"[CWLController] DroneParent on {drone.name} has no VelocityControl component");
                    continue;
                }

                // Read current parameters
                DroneParams current = new DroneParams
                {
                    maxSpeed        = ctrl.maxSpeed,
                    maxYawRate      = ctrl.maxYawRate,
                    maxPitch        = ctrl.maxPitch,
                    maxRoll         = ctrl.maxRoll,
                    maxAltitudeRate = ctrl.maxAltitudeRate,
                    maxAlpha        = ctrl.maxAlpha,
                };

                // Apply adaptive adjustment
                DroneParams updated = _adaptiveAlgorithm.Apply(cwlLevel, current);

                // Write updated parameters back
                ctrl.maxSpeed        = updated.maxSpeed;
                ctrl.maxYawRate      = updated.maxYawRate;
                ctrl.maxPitch        = updated.maxPitch;
                ctrl.maxRoll         = updated.maxRoll;
                ctrl.maxAltitudeRate = updated.maxAltitudeRate;
                ctrl.maxAlpha        = updated.maxAlpha;
            }

            float multiplierBefore = _adaptiveAlgorithm.StepMultiplier / _exponentialBase;
            Debug.Log($"[CWLController] Applied CWL adjustment: {cwlLevel} | step multiplier after: {_adaptiveAlgorithm.StepMultiplier:F3}x (was {multiplierBefore:F3}x)");
        }

        /// <summary>
        /// Returns the current CWL level (last inference received).
        /// </summary>
        public CWLLevel GetCurrentCWLLevel => _lastCWLLevel;

        /// <summary>
        /// Reset all drone parameters to the midpoint between min and max (neutral starting point).
        /// The system will then naturally settle to the optimal equilibrium based on CWL feedback.
        /// </summary>
        public void ResetToMedium()
        {
            if (swarm == null || swarm.Count == 0)
            {
                Debug.LogWarning("[CWLController] No drones found in swarm");
                return;
            }

            foreach (GameObject drone in swarm)
            {
                VelocityControl ctrl = drone.transform.Find("DroneParent").gameObject.GetComponent<VelocityControl>();
                if (ctrl == null) continue;

                // Reset to midpoint of each parameter range
                ctrl.maxSpeed = (maxSpeedRange.min + maxSpeedRange.max) / 2f;
                ctrl.maxYawRate = (maxYawRateRange.min + maxYawRateRange.max) / 2f;
                ctrl.maxPitch = (maxPitchRange.min + maxPitchRange.max) / 2f;
                ctrl.maxRoll = (maxRollRange.min + maxRollRange.max) / 2f;
                ctrl.maxAltitudeRate = (maxAltitudeRateRange.min + maxAltitudeRateRange.max) / 2f;
                ctrl.maxAlpha = (maxAlphaRange.min + maxAlphaRange.max) / 2f;
            }

            _lastCWLLevel = CWLLevel.Medium;
            _adaptiveAlgorithm?.Reset();
            Debug.Log("[CWLController] Reset all drone parameters to midpoint (neutral starting point)");
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
