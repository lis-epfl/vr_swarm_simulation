using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.Mathematics;
using UnityEditor.Rendering.LookDev;

public class SwarmAlgorithm : MonoBehaviour
{
    public List<GameObject> swarm;

    [SerializeField] private bool isSwarmSpreadEnabled = true;

    private SwarmManager swarmManager;

    // Use the SwarmAlgorithm enum from SwarmManager
    private SwarmManager.SwarmAlgorithm currentAlgorithm;


    private Reynolds reynoldsAlgorithm;
    private OlfatiSaber olfatiSaberAlgorithm;
    public float desired_height = 4.0f;

    // Normalised inputs [-1, 1] forwarded to VelocityControl
    private float normPitch = 0.0f;
    private float normRoll  = 0.0f;
    private float desired_alittude_rate = 0.0f;

    // Controller scripts
    private GameObject controller;

    // Velocity control script
    private VelocityControl velocityControl;

    // Awake is called before Start
    void Awake()
{
        // Automatically assign the SwarmManager if not already set
        swarmManager = swarmManager ?? SwarmManager.Instance;

        // Get references to the algorithm components
        reynoldsAlgorithm = GetComponent<Reynolds>();
        olfatiSaberAlgorithm = GetComponent<OlfatiSaber>();

        // Get the controller scripts
        controller = transform.parent.Find("Controller").gameObject;

        // Get the velocity control script
        velocityControl = GetComponent<VelocityControl>();
    }


    // Start is called before the first frame update
    void Start()
    {

        swarmManager.swarmParamsChanged += OnSwarmParamsChanged;

        // Initialize parameters for the first time
        OnSwarmParamsChanged();
    }

    void FixedUpdate()
    {
        readInputs();
        Vector3 swarmAccel = Vector3.zero;

        switch (currentAlgorithm)
        {
            case SwarmManager.SwarmAlgorithm.REYNOLDS:
                swarmAccel = reynoldsAlgorithm.GetSwarmVelocityCommand(swarm);
                break;

            case SwarmManager.SwarmAlgorithm.OLFATI_SABER:
                swarmAccel = olfatiSaberAlgorithm.GetSwarmAcceleration(swarm);
                break;
        }

        velocityControl.swarmAcceleration = swarmAccel;
    }

    // Cleanup when the script is destroyed
    void OnDestroy()
    {
        if (swarmManager != null)
        {
            swarmManager.swarmParamsChanged -= OnSwarmParamsChanged;        
        }
    }


    // Update the swarming parameters
    void OnSwarmParamsChanged()
    {

        // Get swarm algorithm selection
        currentAlgorithm = swarmManager.swarmAlgorithm;


        // Check the current algorithm and enable/disable the corresponding algorithm
        switch (currentAlgorithm)
        {
            // Reynolds algorithm and parameters
            case SwarmManager.SwarmAlgorithm.REYNOLDS:
                UpdateReynoldsParameters();
                velocityControl.currentAlgorithm = SwarmManager.SwarmAlgorithm.REYNOLDS;
                break;

            // Olfati-Saber algorithm and parameters
            case SwarmManager.SwarmAlgorithm.OLFATI_SABER:
                UpdateOlfatiSaberParameters();
                velocityControl.currentAlgorithm = SwarmManager.SwarmAlgorithm.OLFATI_SABER;
                break;

        }

    }

    public void Reset()
    {
        desired_alittude_rate = 0.0f;
        normPitch = 0.0f;
        normRoll  = 0.0f;
        velocityControl.Reset();
    }

    public void SetSwarmSpread(float spread)
    {
        if (olfatiSaberAlgorithm != null)
        {
            olfatiSaberAlgorithm.d_ref = spread;
        }
    }
    // Enable an algorithm script
    private void EnableAlgorithm(MonoBehaviour algorithm)
    {
        if (algorithm != null)
        {
            algorithm.enabled = true;
        }
    }

    // Disable an algorithm script
    private void DisableAlgorithm(MonoBehaviour algorithm)
    {
        if (algorithm != null)
        {
            algorithm.enabled = false;
        }
    }

    private void readInputs()
    {
        if (InputManager.Instance != null)
        {
            Dictionary<string, float> inputStatus = InputManager.Instance.InputStatus;

            normPitch             = inputStatus["pitch"];
            normRoll              = inputStatus["roll"];
            desired_alittude_rate = inputStatus["throttle"];

            velocityControl.SetNormalisedVelocity(normRoll, normPitch); // Drones are facing forward in z
            velocityControl.SetNormalisedAltitudeRate(desired_alittude_rate);

            if (inputStatus["spread"] > 0 && isSwarmSpreadEnabled)
                SetSwarmSpread(inputStatus["spread"]);
        }
    }

    // Update the Reynolds parameters
    private void UpdateReynoldsParameters()
    {
        if (reynoldsAlgorithm != null)
        {
            reynoldsAlgorithm.Is3D = swarmManager.GetDimensions();
            reynoldsAlgorithm.CohesionWeight = swarmManager.GetCohesionWeight();
            reynoldsAlgorithm.SeparationWeight = swarmManager.GetSeparationWeight();
            reynoldsAlgorithm.AlignmentWeight = swarmManager.GetAlignmentWeight();
        }
    }

    // Update the Olfati-Saber parameters
    private void UpdateOlfatiSaberParameters()
    {
        if (olfatiSaberAlgorithm != null)
        {
            olfatiSaberAlgorithm.Is3D = swarmManager.GetDimensions();
            olfatiSaberAlgorithm.d_ref = swarmManager.GetDRef();
            olfatiSaberAlgorithm.r0_coh = swarmManager.GetR0Coh();
            olfatiSaberAlgorithm.delta = swarmManager.GetDelta();
            olfatiSaberAlgorithm.a = swarmManager.GetA();
            olfatiSaberAlgorithm.b = swarmManager.GetB();
            olfatiSaberAlgorithm.c = swarmManager.GetC();
            olfatiSaberAlgorithm.gamma = swarmManager.GetGamma();
            olfatiSaberAlgorithm.c_vm = swarmManager.GetCVM();
            olfatiSaberAlgorithm.d_obs = swarmManager.GetDObs();
            olfatiSaberAlgorithm.lambda_obs = swarmManager.GetLambdaObs();
            olfatiSaberAlgorithm.c_obs = swarmManager.GetCObs();
            olfatiSaberAlgorithm.ScaleFactor = swarmManager.GetScaleFactor();
        }
    }


    public Vector3 GetSwarmCenter()
    {
        Vector3 center = Vector3.zero;
        foreach (GameObject drone in swarm)
        {
            center += drone.transform.position;
        }
        center /= swarm.Count;
        return center;
    }
}
