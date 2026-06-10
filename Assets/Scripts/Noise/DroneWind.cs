using UnityEngine;

public class DroneWind : MonoBehaviour
{
    public float spatialVarianceSigma = 0.05f;

    private Rigidbody rb;
    private RandomPulseNoise windSource;
    private System.Random rng;

    void Start()
    {
        rb = GetComponent<Rigidbody>();
        windSource = FindObjectOfType<RandomPulseNoise>();
        rng = new System.Random(GetInstanceID());
    }

    void FixedUpdate()
    {
        if (windSource == null || rb == null) return;

        Vector3 wind = windSource.CurrentWindForce;
        if (wind == Vector3.zero) return;

        // Shared gust + small per-drone micro-turbulence
        Vector3 perturbation = new Vector3(
            NextGaussian() * spatialVarianceSigma,
            NextGaussian() * spatialVarianceSigma * 0.3f,
            NextGaussian() * spatialVarianceSigma);
        Vector3 droneWind = wind + wind.magnitude * perturbation;

        rb.AddForce(droneWind, ForceMode.Impulse);
    }

    private float NextGaussian()
    {
        float u, v, S;
        do
        {
            u = 2.0f * (float)rng.NextDouble() - 1.0f;
            v = 2.0f * (float)rng.NextDouble() - 1.0f;
            S = u * u + v * v;
        } while (S >= 1.0f);
        float fac = Mathf.Sqrt(-2.0f * Mathf.Log(S) / S);
        return u * fac;
    }
}
