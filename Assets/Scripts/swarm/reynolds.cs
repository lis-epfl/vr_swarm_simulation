using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class reynolds : MonoBehaviour
{
    public List<GameObject> swarm;

    public float cohesionWeight = 1.0f;
    public float separationWeight = 1.0f;
    public float alignmentWeight = 1.0f;

    private Vector3 cohesion = new Vector3(0, 0, 0);
    private Vector3 separation = new Vector3(0, 0, 0);
    private Vector3 alignment = new Vector3(0, 0, 0);
    private Vector3 swarmInput = new Vector3(0, 0, 0);

    // Start is called before the first frame update
    void Start()
    {

    }

    // Update is called once per frame
    void Update()
    {

        // Reset the vectors
        cohesion = new Vector3(0, 0, 0);
        separation = new Vector3(0, 0, 0);
        alignment = new Vector3(0, 0, 0);

        Debug.Log("Swarm Size: " + swarm.Count);

        // Calculate the relative position and velocity of each drone to the current drone
        foreach (GameObject drone in swarm)
        {
            // Relative Position
            Vector3 relativePosition = drone.transform.position - transform.position;
            float distance = relativePosition.magnitude;

            // Skip the current drone
            if (distance == 0)
            {
                continue;
            }

            // Relative Velocity
            Vector3 relativeVelocity = drone.GetComponent<Rigidbody>().velocity - GetComponent<Rigidbody>().velocity;

            // Cohesion
            cohesion += relativePosition;

            // Separation
            separation -= relativePosition / distance;

            // Alignment
            alignment += relativeVelocity;


            Debug.Log("Cohesion: " + cohesion);
            Debug.Log("Separation: " + separation);
            Debug.Log("Alignment: " + alignment);
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

        Debug.Log("Swarm Input: " + swarmInput);

    }
}
