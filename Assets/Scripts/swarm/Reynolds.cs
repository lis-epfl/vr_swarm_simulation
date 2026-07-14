using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Reynolds : MonoBehaviour
{
    public bool Is3D = true;
    public float CohesionWeight = 1.0f;
    public float SeparationWeight = 1.0f;
    public float AlignmentWeight = 1.0f;
    private Vector3 cohesion = new Vector3(0, 0, 0);
    private Vector3 separation = new Vector3(0, 0, 0);
    private Vector3 alignment = new Vector3(0, 0, 0);
    private Vector3 swarmInput = new Vector3(0, 0, 0);
    private VelocityControl selfVelocityControl;

    // Awake, not Start: this component may sit disabled (SwarmAlgorithm toggles
    // the algorithm components), and GetSwarmVelocityCommand can be called
    // before Start would run — but Awake runs regardless of the enabled flag.
    void Awake()
    {
        selfVelocityControl = GetComponent<VelocityControl>();
    }

    // Update is called once per frame
    public Vector3 GetSwarmVelocityCommand(List<GameObject> swarm)
    {

        // Reset the vectors
        cohesion = new Vector3(0, 0, 0);
        separation = new Vector3(0, 0, 0);
        alignment = new Vector3(0, 0, 0);

        StateFinder currentDroneState = selfVelocityControl.State;

        // Calculate the relative position and velocity of each drone to the current drone
        foreach (GameObject neighbour in swarm)
        {

            // Get the neighbour's cached components (resolved once at spawn)
            if (!SwarmRegistry.TryGet(neighbour, out SwarmRegistry.Entry entry) || entry.velocityControl == null)
            {
                continue;
            }

            // Skip the current drone
            if (entry.droneParent.gameObject == gameObject)
            {
                continue;
            }

            // Get the position of the neighbour
            StateFinder neighbourState = entry.velocityControl.State;

            if (!neighbourState.IsAlive)
                continue;

            Vector3 neighbourPosition = neighbourState.Position;

            // Relative Position
            Vector3 relativePosition = neighbourPosition - currentDroneState.Position;

            // Set the y-component to zero if in 2D mode
            if (!Is3D)
            {
                relativePosition.y = 0;
            }

            // Get the distance to the neighbour
            float distance = relativePosition.magnitude;

            // Relative Velocity
            Vector3 relativeVelocity = transform.TransformDirection(neighbourState.VelocityVector) - transform.TransformDirection(currentDroneState.VelocityVector);
            
            // Set the y-component to zero if in 2D mode
            if (!Is3D)
            {
                relativeVelocity.y = 0;
            }

            // Cohesion
            cohesion += relativePosition;

            // Separation
            separation -= relativePosition / distance;

            // Alignment
            alignment += relativeVelocity;

        }

        // Multiply by coefficients and normalize by the number of drones
        cohesion *= CohesionWeight / swarm.Count;
        separation *= SeparationWeight / swarm.Count;
        alignment *= AlignmentWeight / swarm.Count;

        // Sum the total and send it to the velocity control script as the swarm input
        swarmInput = cohesion + separation + alignment;

        return swarmInput;
    }
}
