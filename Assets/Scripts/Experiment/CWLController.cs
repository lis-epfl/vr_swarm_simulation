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
        [SerializeField] private ParameterRange maxPitchRange = new(0.15f, 0.4f, 0.01f);
        [SerializeField] private ParameterRange maxRollRange = new(0.15f, 0.4f, 0.01f);
        [SerializeField] private ParameterRange maxAscentRateRange = new(1.5f, 3.0f, 0.1f);
        [SerializeField] private ParameterRange maxDescentRateRange = new(1.5f, 2.5f, 0.1f);
        [SerializeField] private ParameterRange maxAlphaRange = new(6.0f, 15.0f, 0.5f);

        private swarmSpawn _swarmSpawn;
        private CWLLevel _lastCWLLevel = CWLLevel.Medium;

        private void Awake()
        {
            _swarmSpawn = FindObjectOfType<swarmSpawn>();
            if (_swarmSpawn == null)
                Debug.LogWarning("[CWLController] swarmSpawn not found in scene");
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
                return;

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
        /// Adjusts all drone parameters based on the current CWL level.
        /// </summary>
        private void AdjustDroneParameters(CWLLevel cwlLevel)
        {
            if (_swarmSpawn == null || _swarmSpawn.swarm == null || _swarmSpawn.swarm.Count == 0)
            {
                Debug.LogWarning("[CWLController] No drones found in swarm");
                return;
            }

            foreach (GameObject drone in _swarmSpawn.swarm)
            {
                VelocityControl velocityControl = drone.GetComponent<VelocityControl>();
                if (velocityControl == null)
                {
                    Debug.LogWarning($"[CWLController] Drone {drone.name} has no VelocityControl component");
                    continue;
                }

                AdjustDroneParameter(velocityControl, cwlLevel);
            }

            Debug.Log($"[CWLController] Applied CWL adjustment: {cwlLevel}");
        }

        /// <summary>
        /// Adjusts a single drone's parameters based on CWL feedback.
        /// - If CWL is Low: increase parameters towards max (increase difficulty)
        /// - If CWL is High: decrease parameters towards min (decrease difficulty)
        /// - If CWL is Medium: no adjustment (optimal equilibrium)
        /// </summary>
        private void AdjustDroneParameter(VelocityControl ctrl, CWLLevel cwlLevel)
        {
            // Helper function to adjust towards direction
            float AdjustInDirection(float current, float min, float max, float stepSize, bool increaseDirection)
            {
                if (increaseDirection)
                {
                    // Move towards max
                    if (current >= max)
                        return max;
                    return Mathf.Min(current + stepSize, max);
                }
                else
                {
                    // Move towards min
                    if (current <= min)
                        return min;
                    return Mathf.Max(current - stepSize, min);
                }
            }

            // No adjustment if CWL is Medium (system has found equilibrium)
            if (cwlLevel == CWLLevel.Medium)
                return;

            bool isIncreasing = cwlLevel == CWLLevel.Low; // Low CWL → increase difficulty

            // Adjust each parameter incrementally
            ctrl.maxSpeed = AdjustInDirection(ctrl.maxSpeed, maxSpeedRange.min, maxSpeedRange.max, maxSpeedRange.stepSize, isIncreasing);
            ctrl.maxYawRate = AdjustInDirection(ctrl.maxYawRate, maxYawRateRange.min, maxYawRateRange.max, maxYawRateRange.stepSize, isIncreasing);
            ctrl.maxPitch = AdjustInDirection(ctrl.maxPitch, maxPitchRange.min, maxPitchRange.max, maxPitchRange.stepSize, isIncreasing);
            ctrl.maxRoll = AdjustInDirection(ctrl.maxRoll, maxRollRange.min, maxRollRange.max, maxRollRange.stepSize, isIncreasing);
            ctrl.MaxAscentRate = AdjustInDirection(ctrl.MaxAscentRate, maxAscentRateRange.min, maxAscentRateRange.max, maxAscentRateRange.stepSize, isIncreasing);
            ctrl.MaxDescentRate = AdjustInDirection(ctrl.MaxDescentRate, maxDescentRateRange.min, maxDescentRateRange.max, maxDescentRateRange.stepSize, isIncreasing);
            ctrl.maxAlpha = AdjustInDirection(ctrl.maxAlpha, maxAlphaRange.min, maxAlphaRange.max, maxAlphaRange.stepSize, isIncreasing);
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
            if (_swarmSpawn == null || _swarmSpawn.swarm == null || _swarmSpawn.swarm.Count == 0)
            {
                Debug.LogWarning("[CWLController] No drones found in swarm");
                return;
            }

            foreach (GameObject drone in _swarmSpawn.swarm)
            {
                VelocityControl ctrl = drone.GetComponent<VelocityControl>();
                if (ctrl == null) continue;

                // Reset to midpoint of each parameter range
                ctrl.maxSpeed = (maxSpeedRange.min + maxSpeedRange.max) / 2f;
                ctrl.maxYawRate = (maxYawRateRange.min + maxYawRateRange.max) / 2f;
                ctrl.maxPitch = (maxPitchRange.min + maxPitchRange.max) / 2f;
                ctrl.maxRoll = (maxRollRange.min + maxRollRange.max) / 2f;
                ctrl.MaxAscentRate = (maxAscentRateRange.min + maxAscentRateRange.max) / 2f;
                ctrl.MaxDescentRate = (maxDescentRateRange.min + maxDescentRateRange.max) / 2f;
                ctrl.maxAlpha = (maxAlphaRange.min + maxAlphaRange.max) / 2f;
            }

            _lastCWLLevel = CWLLevel.Medium;
            Debug.Log("[CWLController] Reset all drone parameters to midpoint (neutral starting point)");
        }
    }
}
