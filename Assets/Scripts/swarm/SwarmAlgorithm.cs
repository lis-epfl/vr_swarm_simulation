using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.Mathematics;
using UnityEditor.Rendering.LookDev;

public class SwarmAlgorithm : MonoBehaviour
{
    public List<GameObject> swarm;

    private SwarmManager swarmManager;

    // Use the SwarmAlgorithm enum from SwarmManager
    private SwarmManager.SwarmAlgorithm currentAlgorithm;
    private SwarmManager.AttitudeAlgorithm attitudeAlgorithm;

    private Reynolds reynoldsAlgorithm;
    private OlfatiSaber olfatiSaberAlgorithm;
    // Default altitude and velocity
    public float desired_height = 4.0f;
    public float desired_vx = 0.0f;
    public float desired_vy = 0.0f;
    public float desired_alittude_rate = 0.0f;

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
        Vector3 velocityCommand = Vector3.zero;
        switch (currentAlgorithm)
        {
            case SwarmManager.SwarmAlgorithm.REYNOLDS:
                velocityCommand = reynoldsAlgorithm.GetSwarmVelocityCommand(swarm);
                break;

            case SwarmManager.SwarmAlgorithm.OLFATI_SABER:
                // Unity left-handed + z forward coordinate system
                Vector3 desiredVelocityInput = new Vector3(desired_vy, 0, desired_vx);
                switch (attitudeAlgorithm)
                {
                    case SwarmManager.AttitudeAlgorithm.SIMPLE:
                        // Rotate the desired velocity input based on the estimated swarm heading
                        float currentYaw = GetComponent<VelocityControl>().State.Angles.y;
                        desiredVelocityInput = Quaternion.Euler(0, currentYaw, 0) * desiredVelocityInput;
                        break;
                }
                velocityCommand = olfatiSaberAlgorithm.GetSwarmVelocityCommand(swarm, desiredVelocityInput);
                break;
        }
        // Set the desired velocities in the velocity control script
        velocityControl.swarm_vx = velocityCommand.x;
        velocityControl.swarm_vy = velocityCommand.y + desired_alittude_rate;
        velocityControl.swarm_vz = velocityCommand.z;

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
        attitudeAlgorithm = swarmManager.GetSelectedAttitudeAlgorithm();

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
        desired_vx = 0.0f;
        desired_vy = 0.0f;
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
        // Read the desired velocities from the input manager
        if (InputManager.Instance != null)
        {
            Dictionary<string, float> inputStatus = InputManager.Instance.InputStatus;
            desired_vx = inputStatus["pitch"];
            desired_vy = inputStatus["roll"];
            desired_alittude_rate = inputStatus["throttle"];
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
            olfatiSaberAlgorithm.r0_obs = swarmManager.GetR0Obs();
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
