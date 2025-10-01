using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class SwarmManager : MonoBehaviour
{
    public static SwarmManager Instance;

    public enum SwarmAlgorithm
    {
        REYNOLDS,
        OLFATI_SABER,
        NBV // NEW ADVAITH NBV
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
    public float d_obs = 4.0f;
    public float r0_obs = 6.0f;
    public float lambda_obs = 1.0f;
    public float c_obs = 4.3f;
    public float scaleFactor = 10.0f;

    
    [Header("NBV Parameters")] // NEW ADVAITH NBV
    public float viewDistance = 10.0f;
    public float informationGainWeight = 2.0f;
    // ... any other parameters you need
    public float radius = 20.0f;
    public float height = 150.0f;
    public Vector3 centerPoint = new Vector3(0, 0, 0);
    public float movementSpeed = 5.0f; // END OF NEW ADVAITH NBV

    public enum AttitudeControl
    {
        NONE,
        CONVEXHULL,
    }

    [Header("Attitude Control")]
    public AttitudeControl attitudeControlType;
    public int numNeighbours = 5;
    public int numDimensions = 2;


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
    public float GetDObs() => d_obs;
    public float GetR0Obs() => r0_obs;
    public float GetLambdaObs() => lambda_obs;
    public float GetCObs() => c_obs;
    public float getScaleFactor() => scaleFactor;

    // Getters for the attitude control
    public int GetNumNeighbours() => numNeighbours;
    public int GetNumDimensions() => numDimensions;

    // Getters for NBV parameters // NEW ADVAITH NBV
    public float GetViewDistance() => viewDistance;
    public float GetInformationGainWeight() => informationGainWeight;
    public float GetRadius() => radius;
    public float GetHeight() => height;
    public Vector3 GetCenterPoint() => centerPoint;
    public float GetMovementSpeed() => movementSpeed; // END OF NEW ADVAITH NBV

}
