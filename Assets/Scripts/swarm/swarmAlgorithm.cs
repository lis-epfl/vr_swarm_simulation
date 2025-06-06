using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class swarmAlgorithm : MonoBehaviour
{
    public List<GameObject> swarm;
    
    private SwarmManager swarmManager;

    // Use the SwarmAlgorithm enum from SwarmManager
    private SwarmManager.SwarmAlgorithm currentAlgorithm;

    private Reynolds reynoldsAlgorithm;
    private OlfatiSaber olfatiSaberAlgorithm;
    private AttitudeControl attitudeControl;

    // Default altitude and velocity
    public float desired_height = 4.0f;
    public float desired_vx = 0.0f;
    public float desired_vy = 0.0f;
    public float desired_yaw = 0.0f;

    // Get the AttitudeControl enum from SwarmManager
    private SwarmManager.AttitudeControl attitudeControlType;

    // Controller scripts
    private GameObject controller;
    private ReadController readController;
    private InputControl inputControl;

    // Velocity control script
    private VelocityControl velocityControl;

    // Start is called before the first frame update
    void Start()
    {

        // Automatically assign the SwarmManager if not already set
        swarmManager = swarmManager ?? SwarmManager.Instance;

        swarmManager.swarmParamsChanged += OnSwarmParamsChanged;

        // Get references to the algorithm components
        reynoldsAlgorithm = GetComponent<Reynolds>();
        olfatiSaberAlgorithm = GetComponent<OlfatiSaber>();

        // Assign the swarm list to both algorithms
        reynoldsAlgorithm.swarm = swarm;
        olfatiSaberAlgorithm.swarm = swarm;
        
        // Initialize the attitude control script
        attitudeControl = GetComponent<AttitudeControl>();

        // Assign the swarm list to the attitude control script
        attitudeControl.swarm = swarm;

        // Get the controller scripts
        controller = transform.parent.Find("Controller").gameObject;
        readController = controller.GetComponent<ReadController>();
        inputControl = controller.GetComponent<InputControl>();

        // Get the velocity control script
        velocityControl = GetComponent<VelocityControl>();

        // Initialize parameters for the first time
        OnSwarmParamsChanged();

    }

    // Cleanup when the script is destroyed
    void OnDestroy()
    {
        swarmManager.swarmParamsChanged -= OnSwarmParamsChanged;
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
                EnableAlgorithm(reynoldsAlgorithm);
                DisableAlgorithm(olfatiSaberAlgorithm);
                UpdateReynoldsParameters();
                readController.currentAlgorithm = SwarmManager.SwarmAlgorithm.REYNOLDS;
                inputControl.currentAlgorithm = SwarmManager.SwarmAlgorithm.REYNOLDS;
                velocityControl.currentAlgorithm = SwarmManager.SwarmAlgorithm.REYNOLDS;

                break;

            // Olfati-Saber algorithm and parameters
            case SwarmManager.SwarmAlgorithm.OLFATI_SABER:                
                EnableAlgorithm(olfatiSaberAlgorithm);
                DisableAlgorithm(reynoldsAlgorithm);
                UpdateOlfatiSaberParameters();
                readController.currentAlgorithm = SwarmManager.SwarmAlgorithm.OLFATI_SABER;
                inputControl.currentAlgorithm = SwarmManager.SwarmAlgorithm.OLFATI_SABER;
                velocityControl.currentAlgorithm = SwarmManager.SwarmAlgorithm.OLFATI_SABER;
                break;
        }

        // Get the attitude control selection
        attitudeControlType = swarmManager.attitudeControlType;

        // Check the attitude control type and enable/disable the corresponding algorithm
        switch (attitudeControlType)
        {
            case SwarmManager.AttitudeControl.NONE:
                DisableAlgorithm(attitudeControl);
                break;
            case SwarmManager.AttitudeControl.CONVEXHULL:
                EnableAlgorithm(attitudeControl);
                UpdateAttitudeControlParameters();
                break;
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

    // Update the Reynolds parameters
    private void UpdateReynoldsParameters()
    {
        if (reynoldsAlgorithm != null)
        {
            reynoldsAlgorithm.cohesionWeight = swarmManager.GetCohesionWeight();
            reynoldsAlgorithm.separationWeight = swarmManager.GetSeparationWeight();
            reynoldsAlgorithm.alignmentWeight = swarmManager.GetAlignmentWeight();
        }
    }

    // Update the Olfati-Saber parameters
    private void UpdateOlfatiSaberParameters()
    {
        if (olfatiSaberAlgorithm != null)
        {
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
            olfatiSaberAlgorithm.scaleFactor = swarmManager.getScaleFactor();
        }
    }

    // Update the attitude control parameters
    private void UpdateAttitudeControlParameters()
    {
        if (attitudeControl != null)
        {
            attitudeControl.numNeighbours = swarmManager.GetNumNeighbours();
            attitudeControl.numDimensions = swarmManager.GetNumDimensions();
        }
    }
}
