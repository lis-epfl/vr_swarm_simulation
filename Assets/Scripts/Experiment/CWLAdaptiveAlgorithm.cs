using UnityEngine;

namespace Experiment
{
    /// <summary>
    /// Value struct holding a drone's six adjustable flight parameters.
    /// Used to pass current and updated values between CWLController and the algorithm.
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
    /// Configuration snapshot for CWLAdaptiveAlgorithm.
    /// Holds the six ParameterRange values (min, max, stepSize) for each drone parameter.
    /// Captured from CWLController at Start() time.
    /// </summary>
    public struct CWLAdaptiveAlgorithmConfig
    {
        public CWLController.ParameterRange maxSpeedRange;
        public CWLController.ParameterRange maxYawRateRange;
        public CWLController.ParameterRange maxPitchRange;
        public CWLController.ParameterRange maxRollRange;
        public CWLController.ParameterRange maxAltitudeRateRange;
        public CWLController.ParameterRange maxAlphaRange;
    }

    /// <summary>
    /// Adaptive algorithm for CWL-based drone parameter adjustment.
    ///
    /// Implements exponential step growth with direction change detection:
    /// - When CWL direction changes (Low ↔ High), step multiplier resets to 1.0 (tiny steps, noise filter)
    /// - When CWL direction persists, step multiplier grows exponentially (e.g., 1.0x → 2.0x → 4.0x → 8.0x)
    /// - Parameters are clamped to [min, max] and never reset to midpoint on direction change
    /// - Medium is always a no-op; it does not affect direction state
    ///
    /// Usage:
    /// 1. Create: var algo = new CWLAdaptiveAlgorithm(config, exponentialBase: 2.0f)
    /// 2. On each CWL update: newParams = algo.Apply(cwlLevel, currentParams)
    /// 3. On session reset: algo.Reset()
    /// </summary>
    public class CWLAdaptiveAlgorithm
    {
        // --- Configuration (immutable after construction) ---
        private readonly CWLAdaptiveAlgorithmConfig _config;
        private readonly float _exponentialBase;

        // --- Mutable state ---
        /// <summary>
        /// Last non-Medium level received (Low or High). Null before any non-Medium level arrives.
        /// Used to detect direction changes.
        /// </summary>
        private CWLController.CWLLevel? _lastDirection;

        /// <summary>
        /// Exponential multiplier applied to stepSize on each call.
        /// Resets to 1.0 when direction changes, then grows by exponentialBase each same-direction update.
        /// </summary>
        private float _stepMultiplier;

        // --- Public diagnostics ---
        /// <summary>
        /// Current step multiplier (e.g., 1.0x, 2.0x, 4.0x). Read-only.
        /// Useful for debug UI or logging.
        /// </summary>
        public float StepMultiplier => _stepMultiplier;

        /// <summary>
        /// Last direction that was applied (Low or High). Null before any non-Medium level arrives.
        /// </summary>
        public CWLController.CWLLevel? LastDirection => _lastDirection;

        // --- Constructor ---
        /// <summary>
        /// Initialize the adaptive algorithm with parameter ranges and exponential base.
        /// </summary>
        /// <param name="config">The six ParameterRange values (min, max, stepSize) for each drone parameter.</param>
        /// <param name="exponentialBase">Multiplier for step growth each update (default 2.0 = double each time).</param>
        public CWLAdaptiveAlgorithm(CWLAdaptiveAlgorithmConfig config, float exponentialBase = 2.0f)
        {
            _config = config;
            _exponentialBase = exponentialBase;
            _stepMultiplier = 1.0f;
            _lastDirection = null;
        }

        // --- Public API ---

        /// <summary>
        /// Apply one adaptive adjustment step.
        ///
        /// Returns a new DroneParams with values adjusted according to the CWL level and current state:
        /// - Medium: no change, no state update
        /// - Low: increase params (toward max), or reset multiplier if direction changed
        /// - High: decrease params (toward min), or reset multiplier if direction changed
        ///
        /// After applying the step, the multiplier grows exponentially (or resets if direction changed).
        /// </summary>
        /// <param name="level">The CWL level from the Python model (Low, Medium, or High).</param>
        /// <param name="current">The current drone parameter values.</param>
        /// <returns>New parameter values with step applied and clamped to [min, max].</returns>
        public DroneParams Apply(CWLController.CWLLevel level, DroneParams current)
        {
            // Medium is always a no-op; never affects direction state or multiplier
            if (level == CWLController.CWLLevel.Medium)
                return current;

            // Determine if this is a direction change (High ↔ Low)
            bool isHigh = level == CWLController.CWLLevel.High;
            bool directionChanged = _lastDirection.HasValue && _lastDirection.Value != level;

            // Direction changed: reset multiplier to 1.0 (start with tiny steps again)
            if (directionChanged)
                _stepMultiplier = 1.0f;

            // Apply scaled step to each of the six parameters
            DroneParams result;
            result.maxSpeed        = Step(current.maxSpeed,        _config.maxSpeedRange,        isHigh);
            result.maxYawRate      = Step(current.maxYawRate,      _config.maxYawRateRange,      isHigh);
            result.maxPitch        = Step(current.maxPitch,        _config.maxPitchRange,        isHigh);
            result.maxRoll         = Step(current.maxRoll,         _config.maxRollRange,         isHigh);
            result.maxAltitudeRate = Step(current.maxAltitudeRate, _config.maxAltitudeRateRange, isHigh);
            result.maxAlpha        = Step(current.maxAlpha,        _config.maxAlphaRange,        isHigh);

            // Save direction and grow multiplier AFTER the step is applied.
            // This ensures the very first call after a direction reset uses stepSize * 1.0,
            // and only subsequent same-direction calls use 2.0x, 4.0x, 8.0x, etc.
            _lastDirection = level;
            _stepMultiplier *= _exponentialBase;

            return result;
        }

        /// <summary>
        /// Reset the algorithm state (direction memory and multiplier).
        /// Call this when an experiment session resets or restarts.
        /// </summary>
        public void Reset()
        {
            _stepMultiplier = 1.0f;
            _lastDirection = null;
        }

        // --- Private helpers ---

        /// <summary>
        /// Apply one adaptive step to a single parameter value.
        /// </summary>
        /// <param name="current">Current parameter value.</param>
        /// <param name="range">The ParameterRange (min, max, stepSize) for this parameter.</param>
        /// <param name="decrease">True for High (move toward min), False for Low (move toward max).</param>
        /// <returns>New value clamped to [range.min, range.max].</returns>
        private float Step(float current, CWLController.ParameterRange range, bool decrease)
        {
            float delta = range.stepSize * _stepMultiplier;
            float next = decrease
                ? current - delta
                : current + delta;
            return Mathf.Clamp(next, range.min, range.max);
        }
    }
}
