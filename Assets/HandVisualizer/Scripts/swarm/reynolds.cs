using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.XR.Hands.Samples.VisualizerSample; // To access HandProcessor

public class Reynolds : MonoBehaviour
{
    public List<GameObject> swarm;

    public float cohesionWeight = 1000.0f;
    public float separationWeight = 5000.0f;
    public float alignmentWeight = 1.0f;

    // New overall multipliers that let you scale the final cohesion and separation forces
    public float cohesionMultiplier = 1000.0f;
    public float separationMultiplier = 5000.0f;

    // New modifiers that affect how the cohesion and separation change along the adjusted axes.
    // For cohesion, this modifier scales the factor you use when modifying the force.
    // Likewise for separation. 
    public float cohesionAxisModifier = 1.0f;
    public float separationAxisModifier = 1.0f;

    private Vector3 cohesion = Vector3.zero;
    private Vector3 separation = Vector3.zero;
    private Vector3 alignment = Vector3.zero;
    private Vector3 swarmInput = Vector3.zero;

    // Update is called once per frame
    void FixedUpdate()
    {
        // Reset the vectors
        cohesion = Vector3.zero;
        separation = Vector3.zero;
        alignment = Vector3.zero;       

        // Calculate the relative position and velocity of each drone to the current drone
        foreach (GameObject neighbour in swarm)
        {
            GameObject neighbourChild = neighbour.transform.Find("DroneParent").gameObject;
            if (neighbourChild == gameObject)
                continue;

            Vector3 neighbourPosition = neighbourChild.transform.position;
            Vector3 relativePosition = neighbourPosition - transform.position;
            float distance = relativePosition.magnitude;
            Vector3 relativeVelocity = neighbourChild.GetComponent<Rigidbody>().velocity - GetComponent<Rigidbody>().velocity;

            // Cohesion: steer toward average position
            cohesion += relativePosition;
            // Separation: steer away from neighbours
            separation -= relativePosition / distance;
            // Alignment: match velocity
            alignment += relativeVelocity;
        }

        int count = swarm.Count;
        if(count > 0)
        {
            cohesion *= cohesionWeight / count;
            separation *= separationWeight / count;
            alignment *= alignmentWeight / count;
        }

        // Read the hand shape data from HandProcessor
        float handLength = HandProcessor.HandEllipsoidLength; // palm distance → now affects z axis
        float handWidth = HandProcessor.HandEllipsoidWidth;   // finger-determined thickness → now affects x axis

        // Define baseline values (adjust based on your expectations)
        float baselineLength = 0.5f; // example baseline for palm distance
        float baselineWidth  = 0.1f; // example baseline for width scale

        // Compute factors – if hands are further apart, we want less cohesion along that axis and more separation.
        // Now, use handWidth for x and y, and handLength for z.
        float lengthFactor = (handLength > 0) ? handLength / baselineLength : 1.0f;
        float widthFactor  = (handWidth > 0)  ? handWidth / baselineWidth   : 1.0f;

        // Modify cohesion and separation based on hand shape factors for all three axes.
        // Use widthFactor for x and y, and lengthFactor for z.
        cohesion.x /= (widthFactor * cohesionAxisModifier);   // larger thickness (handWidth) → lower cohesion in x
        cohesion.y /= (widthFactor * cohesionAxisModifier);   // larger thickness (handWidth) → lower cohesion in y (NEW)
        cohesion.z /= (lengthFactor * cohesionAxisModifier);    // larger palm distance → lower cohesion in z

        separation.x *= (widthFactor * separationAxisModifier); // larger thickness → increased separation in x
        separation.y *= (widthFactor * separationAxisModifier); // larger thickness → increased separation in y (NEW)
        separation.z *= (lengthFactor * separationAxisModifier);  // larger palm distance → increased separation in z

        // Apply overall multipliers
        cohesion *= cohesionMultiplier;
        separation *= separationMultiplier; 

        // Combine forces (alignment remains unchanged)
        swarmInput = cohesion + separation + alignment;

        // Pass the final computed vector to the VelocityControl script
        var vc = GetComponent<VelocityControl>();
        if(vc != null)
        {
            vc.swarm_vx = swarmInput.x;
            vc.swarm_vy = swarmInput.y;
            vc.swarm_vz = swarmInput.z;
        }
    }
}
