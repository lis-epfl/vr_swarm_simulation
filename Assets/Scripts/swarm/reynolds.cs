using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Reynolds : MonoBehaviour
{
    public List<GameObject> swarm;

    public float cohesionWeight = 1.0f;
    public float separationWeight = 1.0f;
    public float alignmentWeight = 1.0f;

    private Vector3 cohesion = new Vector3(0, 0, 0);
    private Vector3 separation = new Vector3(0, 0, 0);
    private Vector3 alignment = new Vector3(0, 0, 0);
    private Vector3 swarmInput = new Vector3(0, 0, 0);


    // Update is called once per frame
    void FixedUpdate()
    {

        // Reset the vectors
        cohesion = new Vector3(0, 0, 0);
        separation = new Vector3(0, 0, 0);
        alignment = new Vector3(0, 0, 0);       
        
        // Calculate the relative position and velocity of each drone to the current drone
        foreach (GameObject neighbour in swarm)
        {
            
            // Get the child of the neighbour
            GameObject neighbourChild = neighbour.transform.Find("DroneParent").gameObject;

            // Skip the current drone
            if (neighbourChild == gameObject)
            {
                continue;
            }

            // Get the position of the neighbour
            Vector3 neighbourPosition = neighbourChild.transform.position;
            
            // Relative Position
            Vector3 relativePosition = neighbourPosition - transform.position;
            float distance = relativePosition.magnitude;           

            // Relative Velocity
            Vector3 relativeVelocity = neighbourChild.GetComponent<Rigidbody>().velocity - GetComponent<Rigidbody>().velocity;

            // Cohesion
            cohesion += relativePosition;

            // Separation
            separation -= relativePosition / distance;

            // Alignment
            alignment += relativeVelocity;

        }

        // Multiply by coefficients and normalize by the number of drones
        cohesion *= cohesionWeight / swarm.Count;
        separation *= separationWeight / swarm.Count;
        alignment *= alignmentWeight / swarm.Count;

        // Sum the total and send it to the velocity control script as the swarm input
        swarmInput = cohesion + separation + alignment;
        GetComponent<VelocityControl>().swarm_vx = swarmInput.x;
        GetComponent<VelocityControl>().swarm_vy = swarmInput.y;
        GetComponent<VelocityControl>().swarm_vz = swarmInput.z;

    }
}
