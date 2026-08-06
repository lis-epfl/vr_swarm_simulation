using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Reynolds : MonoBehaviour
{
    public bool Is3D = true;

    // Unit normal of the plane the swarm is constrained to when !Is3D. Vector3.up gives the
    // horizontal formation; SwarmPlaneController swings it onto the pilot-steered target heading for
    // a vertical wall. Pushed every tick by SwarmAlgorithm.
    public Vector3 PlaneNormal = Vector3.up;

    // When set, drones are pulled onto the plane at PlaneOffsetTarget along the normal — one value
    // shared by the whole swarm — rather than each toward the consensus of its own neighbours.
    // See the matching fields in OlfatiSaber.
    public bool HasPlaneOffsetTarget = false;
    public float PlaneOffsetTarget = 0.0f;

    public float CohesionWeight = 1.0f;
    public float SeparationWeight = 1.0f;
    public float AlignmentWeight = 1.0f;

    // Gain of the term that pulls a drone back onto the swarming plane, toward the mean neighbour
    // offset along the normal. Reynolds has no such term of its own, so without it nothing holds
    // the swarm in the plane at all — the projections below only stop it being pushed *out*.
    public float PlaneWeight = 1.0f;
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

        float totalNeighbourPlaneOffset = 0f;
        int aliveNeighbourCount = 0;

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

            // Drop the out-of-plane component if constrained to a plane
            if (!Is3D)
            {
                relativePosition -= PlaneNormal * Vector3.Dot(relativePosition, PlaneNormal);
                totalNeighbourPlaneOffset += Vector3.Dot(neighbourPosition, PlaneNormal);
                aliveNeighbourCount++;
            }

            // Get the distance to the neighbour
            float distance = relativePosition.magnitude;

            // Relative Velocity
            Vector3 relativeVelocity = transform.TransformDirection(neighbourState.VelocityVector) - transform.TransformDirection(currentDroneState.VelocityVector);

            // Drop the out-of-plane component if constrained to a plane
            if (!Is3D)
            {
                relativeVelocity -= PlaneNormal * Vector3.Dot(relativeVelocity, PlaneNormal);
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

        // Constrained mode: pull back onto the plane, toward the mean neighbour offset along the
        // normal. Mirrors OlfatiSaber's plane correction; the projections above only remove the
        // out-of-plane forcing, they don't restore drift.
        Vector3 planeCorrection = Vector3.zero;
        if (!Is3D && (HasPlaneOffsetTarget || aliveNeighbourCount > 0))
        {
            float targetOffset = HasPlaneOffsetTarget
                ? PlaneOffsetTarget
                : totalNeighbourPlaneOffset / aliveNeighbourCount;
            planeCorrection = PlaneWeight
                            * (targetOffset - Vector3.Dot(currentDroneState.Position, PlaneNormal))
                            * PlaneNormal;
        }

        // Sum the total and send it to the velocity control script as the swarm input
        swarmInput = cohesion + separation + alignment + planeCorrection;

        return swarmInput;
    }
}
