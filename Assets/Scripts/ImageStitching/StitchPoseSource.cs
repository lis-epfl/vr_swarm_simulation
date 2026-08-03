using System.Collections.Generic;
using UnityEngine;

/// <summary>
/// Injects realistic GNSS error into the camera poses published to the planar stitcher.
///
/// Exists so the pose-only backbone can be measured against the error it would actually
/// face on real aircraft, without touching the flight loop. <see cref="StateFinder"/>
/// deliberately is not modified: its <c>Position</c> feeds <c>VelocityControl</c>, so
/// adding drift there would change how the drones fly and confound the experiment. Its
/// existing noise is also white and redrawn every <c>GetState()</c>, which is not what
/// GNSS error looks like.
///
/// Model: first-order Gauss-Markov (Ornstein-Uhlenbeck) per axis per drone, plus a
/// shared common-mode term.
/// <code>
///   a = exp(-dt/tau)
///   b &lt;- a*b + sigma*sqrt(1-a^2)*N(0,1)      per-drone bias
///   B &lt;- a_c*B + sigma_c*sqrt(1-a_c^2)*N(0,1) common-mode bias
///   out = truth + (1-k)*b + k*B + white
/// </code>
///
/// Two points that matter more than the magnitudes:
///
/// <b>Common mode dominates.</b> Receivers operating close together share almost all of
/// their error (ionosphere, ephemeris), and a purely common-mode bias translates the
/// whole mosaic rigidly -- invisible to the pilot and harmless to stitch quality. Only
/// the differential part <c>(1-k)*b</c> causes seams. Modelling 3 m of fully independent
/// error would size a future refiner against a scenario roughly 3x worse than reality.
///
/// <b>Attitude error is not negligible.</b> At 100 m range a 0.3 deg yaw error is ~0.5 m
/// of ground shift, comparable to the differential position error. Without the yaw term
/// a noise study reads as far more position-dominated than it really is.
/// </summary>
public class StitchPoseSource
{
    [System.Serializable]
    public class Settings
    {
        [Tooltip("Horizontal (x/z) GNSS sigma, metres.")]
        public float sigmaHorizontal = 1.5f;

        [Tooltip("Vertical (y) GNSS sigma, metres. Real GNSS vertical error is 2-3x worse " +
                 "than horizontal; modelling it as isotropic understates nadir error and " +
                 "overstates facade error.")]
        public float sigmaVertical = 3.0f;

        [Tooltip("Correlation time of the per-drone bias, seconds. This is a slow wander, " +
                 "not per-frame jitter.")]
        public float tau = 60f;

        [Range(0f, 1f)]
        [Tooltip("Fraction of the error that is common to the whole formation. Common-mode " +
                 "error translates the mosaic rigidly and costs nothing; only the remaining " +
                 "(1-k) differential part actually causes seams.")]
        public float commonModeFraction = 0.7f;

        [Tooltip("Correlation time of the common-mode bias, seconds.")]
        public float commonModeTau = 300f;

        [Tooltip("Per-sample white noise on top of the drifting bias, metres.")]
        public float whiteSigma = 0.05f;

        [Tooltip("Yaw bias sigma, degrees. Small but consequential at range.")]
        public float yawBiasSigmaDeg = 0.3f;

        [Tooltip("Correlation time of the yaw bias, seconds.")]
        public float yawTau = 30f;

        [Tooltip("0 = seed from the clock (different every run). Non-zero is reproducible, " +
                 "which is what you want when comparing configurations.")]
        public int seed = 0;
    }

    private class DroneState
    {
        public Vector3 bias;
        public float yawBias;
    }

    private readonly Settings settings;
    private readonly Dictionary<int, DroneState> perDrone = new Dictionary<int, DroneState>();
    private Vector3 commonBias;
    private System.Random rng;
    private float lastStepTime = -1f;

    public StitchPoseSource(Settings settings)
    {
        this.settings = settings;
        rng = new System.Random(settings.seed != 0
            ? settings.seed
            : System.Environment.TickCount);
    }

    /// <summary>Re-seeds and clears all accumulated bias.</summary>
    public void Reset()
    {
        perDrone.Clear();
        commonBias = Vector3.zero;
        lastStepTime = -1f;
        rng = new System.Random(settings.seed != 0
            ? settings.seed
            : System.Environment.TickCount);
    }

    /// <summary>
    /// Advances the common-mode bias. Call once per frame, before <see cref="Apply"/>.
    /// Separated from the per-drone step so every drone in a frame shares one common-mode
    /// sample, which is the whole point of the term.
    /// </summary>
    public void Step(float time)
    {
        float dt = (lastStepTime < 0f) ? 0f : Mathf.Max(0f, time - lastStepTime);
        lastStepTime = time;
        if (dt <= 0f) return;

        commonBias = StepOU(commonBias, dt, settings.commonModeTau,
                            settings.sigmaHorizontal, settings.sigmaVertical);
    }

    /// <summary>
    /// Applies the current error to one camera's true pose.
    /// </summary>
    public void Apply(int droneId, float dt, ref Vector3 position, ref Quaternion rotation)
    {
        if (!perDrone.TryGetValue(droneId, out DroneState s))
        {
            s = new DroneState();
            perDrone[droneId] = s;
        }

        if (dt > 0f)
        {
            s.bias = StepOU(s.bias, dt, settings.tau,
                            settings.sigmaHorizontal, settings.sigmaVertical);

            float a = Mathf.Exp(-dt / Mathf.Max(1e-3f, settings.yawTau));
            s.yawBias = a * s.yawBias
                      + settings.yawBiasSigmaDeg * Mathf.Sqrt(Mathf.Max(0f, 1f - a * a))
                        * Gaussian();
        }

        float k = Mathf.Clamp01(settings.commonModeFraction);
        Vector3 offset = (1f - k) * s.bias + k * commonBias;
        offset += new Vector3(Gaussian(), Gaussian(), Gaussian()) * settings.whiteSigma;

        position += offset;
        rotation = Quaternion.AngleAxis(s.yawBias, Vector3.up) * rotation;
    }

    private Vector3 StepOU(Vector3 state, float dt, float tau, float sigmaH, float sigmaV)
    {
        float a = Mathf.Exp(-dt / Mathf.Max(1e-3f, tau));
        float g = Mathf.Sqrt(Mathf.Max(0f, 1f - a * a));
        return new Vector3(
            a * state.x + sigmaH * g * Gaussian(),
            a * state.y + sigmaV * g * Gaussian(),
            a * state.z + sigmaH * g * Gaussian());
    }

    // Box-Muller. System.Random rather than UnityEngine.Random so the stream is private
    // to this object and a fixed seed reproduces a run regardless of what else draws.
    private float Gaussian()
    {
        double u1 = 1.0 - rng.NextDouble();
        double u2 = rng.NextDouble();
        return (float)(System.Math.Sqrt(-2.0 * System.Math.Log(u1))
                       * System.Math.Sin(2.0 * System.Math.PI * u2));
    }
}
