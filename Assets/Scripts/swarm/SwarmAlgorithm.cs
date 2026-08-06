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
        ApplyPlaneConstraint();
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

    /// <summary>
    /// Pushes the swarming plane onto the active algorithm. This has to happen every tick rather
    /// than through OnSwarmParamsChanged (which only fires on inspector edits) because the plane
    /// tracks the pilot-steered target heading and the swarm's own centroid.
    /// </summary>
    private void ApplyPlaneConstraint()
    {
        SwarmPlaneController plane = SwarmPlaneController.Instance;
        bool planeMode = plane != null && plane.PlaneModeActive;

        bool is3D = !planeMode && swarmManager.GetDimensions();
        Vector3 planeNormal = planeMode ? plane.PlaneNormal : Vector3.up;
        // Every drone is pulled toward the same offset along the normal — the swarm centroid's — so
        // the restoring pull is zero-sum and the wall cannot slide along its own normal. The
        // neighbour-consensus form the horizontal mode uses would do the same job here up to a gain,
        // but a shared target makes it explicit that no member defines where the wall sits.
        float planeOffsetTarget = planeMode ? Vector3.Dot(plane.PlaneOrigin, planeNormal) : 0f;

        if (reynoldsAlgorithm != null)
        {
            reynoldsAlgorithm.Is3D = is3D;
            reynoldsAlgorithm.PlaneNormal = planeNormal;
            reynoldsAlgorithm.HasPlaneOffsetTarget = planeMode;
            reynoldsAlgorithm.PlaneOffsetTarget = planeOffsetTarget;
        }
        if (olfatiSaberAlgorithm != null)
        {
            olfatiSaberAlgorithm.Is3D = is3D;
            olfatiSaberAlgorithm.PlaneNormal = planeNormal;
            olfatiSaberAlgorithm.HasPlaneOffsetTarget = planeMode;
            olfatiSaberAlgorithm.PlaneOffsetTarget = planeOffsetTarget;
        }

        // A vertical plane puts the formation's spread on the vertical axis, which the altitude-hold
        // PD would fight, so the swarm takes the vertical channel from every drone without exception
        // (see VelocityControl.verticalSwarmAuthority). The absolute vertical reference that channel
        // needs comes from SwarmPlaneController.ReferenceAltitude — latched from the centroid on entry
        // and moved by the climb stick — rather than from one drone left behind on altitude hold.
        velocityControl.verticalSwarmAuthority = planeMode;
        velocityControl.verticalReferenceAltitude = planeMode ? plane.ReferenceAltitude : 0f;
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
        // Mirror into the SwarmManager so the live spread is visible in the inspector.
        if (swarmManager != null)
        {
            swarmManager.SetDRef(spread);
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

            // Forward the command frame. Body rotates by each drone's heading; World uses fixed axes;
            // VR rotates by the pilot body heading (OVRCameraRig yaw integrated in PyUniSharingFast).
            InputManager.CommandFrame frame = InputManager.Instance.ActiveCommandFrame;

            // In the convex-hull attitude modes the controller yaw command steers the pilot's body
            // (the OVRCameraRig, rotated in PyUniSharingFast.UpdateBodyYaw) instead of the drones, so
            // the velocity stick must follow that body heading — forward on the stick always matches
            // where the rig faces. Force the VR frame here regardless of the inspector setting.
            if (swarmManager != null)
            {
                SwarmManager.AttitudeAlgorithm attitude = swarmManager.GetSelectedAttitudeAlgorithm();
                if (attitude == SwarmManager.AttitudeAlgorithm.LOCAL_CONVEXHULL
                 || attitude == SwarmManager.AttitudeAlgorithm.GLOBAL_CONVEXHULL)
                {
                    frame = InputManager.CommandFrame.VR;
                }
            }

            bool worldFrame = frame != InputManager.CommandFrame.Body;
            float referenceYaw = frame == InputManager.CommandFrame.VR ? PyUniSharingFast.BodyYawDegrees : 0f;
            velocityControl.SetCommandFrame(worldFrame, referenceYaw);

            if (inputStatus["spread"] > 0 && isSwarmSpreadEnabled)
                SetSwarmSpread(inputStatus["spread"]);
        }
    }

    // Update the Reynolds parameters
    private void UpdateReynoldsParameters()
    {
        if (reynoldsAlgorithm != null)
        {
            // Is3D is owned by ApplyPlaneConstraint (it depends on the live plane mode, not just
            // the inspector setting), so it is deliberately not pushed here.
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
            // Is3D is owned by ApplyPlaneConstraint (see above).
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
