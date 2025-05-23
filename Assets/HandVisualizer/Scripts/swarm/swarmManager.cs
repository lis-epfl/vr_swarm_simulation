using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class SwarmManager : MonoBehaviour
{
    public static SwarmManager Instance;

    public enum SwarmAlgorithm
    {
        REYNOLDS,
        OLFATI_SABER
    }

    [Header("Swarm Algorithm")]
    public SwarmAlgorithm swarmAlgorithm;

    [Header("Reynolds Parameters")]
    public float cohesionWeight = 1.0f;
    public float separationWeight = 15.0f;
    public float alignmentWeight = 1.0f;

    [Header("Olfati-Saber Parameters")]
    public float d_ref = 7.0f;
    public float r0_coh = 20.0f;
    public float delta = 0.1f;
    public float a = 0.3f;
    public float b = 0.5f;
    private float c;
    public float gamma = 1.0f;
    public float c_vm = 1.0f;
    public float scaleFactor = 10.0f;
    public float cohesionMultiplier = 2.0f;
    public float handDistanceScaleFactor = 1.0f;
    public float velocityProportionFactor = 0.5f;

    public enum AttitudeControl
    {
        NONE,
        CONVEXHULL,
    }

    [Header("Attitude Control")]
    public AttitudeControl attitudeControlType;
    public int numNeighbours = 5;
    public int numDimensions = 3;

    [Header("Velocity Control")]
    public float filterCoefficient = 0.1f;


    public delegate void OnSwarmParamsChanged();
    public event OnSwarmParamsChanged swarmParamsChanged;

    void Awake()
    {
        if (Instance == null)
        {
            Instance = this;
            DontDestroyOnLoad(gameObject);
        }
        else
        {
            Destroy(gameObject);
        }
    }

    // Called whenever a value is changed in the Inspector
    private void OnValidate()
    {
        // Trigger the event to notify all subscribed drones
        swarmParamsChanged?.Invoke();
    }

    // Getters for the Reynolds parameters
    public float GetCohesionWeight() => cohesionWeight;
    public float GetSeparationWeight() => separationWeight;
    public float GetAlignmentWeight() => alignmentWeight;

    // Getters for the Olfati-Saber parameters
    public float GetDRef() => d_ref;
    public float GetR0Coh() => r0_coh;
    public float GetDelta() => delta;
    public float GetA() => a;
    public float GetB() => b;
    public float GetC() => (b - a) / (2 * Mathf.Sqrt(a * b));
    public float GetGamma() => gamma;
    public float GetCVM() => c_vm;
    public float getScaleFactor() => scaleFactor;
    public float GetCohesionMultiplier() => cohesionMultiplier;
    public float GetHandDistanceScaleFactor() => handDistanceScaleFactor;
    public float GetVelocityProportionFactor() => velocityProportionFactor;

    // Getters for the attitude control
    public int GetNumNeighbours() => numNeighbours;
    public int GetNumDimensions() => numDimensions;

    // Getters for the velocity control
    public float GetFilterCoefficient() => filterCoefficient;

}
