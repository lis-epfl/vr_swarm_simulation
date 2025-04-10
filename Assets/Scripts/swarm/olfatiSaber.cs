using System.Collections;
using System.Collections.Generic;
using System.Security;
using UnityEngine;

public class OlfatiSaber : MonoBehaviour
{
    public List<GameObject> swarm;

    public float d_ref = 7.0f;
    public float r0_coh = 150.0f;
    public float delta = 0.1f;
    public float a = 0.9f;
    public float b = 1.5f;
    public float c;
    public float gamma = 1.0f;
    public float c_vm = 1.0f;


    public float maxMigrationDistance = 10.0f;


    public float detectionRadius = 5.0f;  // Radius to detect obstacles
    public float obstacleAvoidanceForceWeight = 2.0f; // Weight for obstacle avoidance force
    public float maxAvoidForce = 10.0f;   // Maximum avoidance force
    public string obstacleTag = "Obstacle"; // Tag for obstacles

    public float desired_height = 4.0f;
    public float desired_vx = 0.0f;
    public float desired_vy = 0.0f;
    public float desired_vz = 0.0f;
    public float desired_yaw = 0.0f;
    
    private Vector3 velocityMatching = new Vector3(0, 0, 0);
    private Vector3 cohesion = new Vector3(0, 0, 0);
    private Vector3 obstacle = new Vector3(0, 0, 0);

    private Vector3 swarmInput = new Vector3(0, 0, 0);

    private string droneName;

    void Start()
    {
        droneName = transform.parent.name;
    }


    void FixedUpdate()
    {

        // Reset the vectors
        velocityMatching = new Vector3(0, 0, 0);
        cohesion = new Vector3(0, 0, 0);
        obstacle = new Vector3(0, 0, 0);

        // Get the position and velocity of the current drone
        Vector3 position = transform.position;
        Vector3 velocity = GetComponent<Rigidbody>().velocity;

        //Calculate the velocity matching force given the difference between the desired v_x, v_y, and v_z and the current velocity
        Vector3 desired_velocity = new Vector3(desired_vx, desired_vz, -desired_vy);
        velocityMatching = c_vm * (desired_velocity - velocity);           
        
        // Calculate the cohesion force for each of the neighbours
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
            Vector3 relativePosition = neighbourPosition - position;
            float distance = relativePosition.magnitude;

            // Relative Velocity
            Vector3 relativeVelocity = neighbourChild.GetComponent<Rigidbody>().velocity - velocity;

            // Cohesion
            Vector3 thisCohesion = GetCohesionForce(distance) * relativePosition.normalized;
            cohesion += thisCohesion;

            // // Check if this is drone 0
            // if (droneName == "Drone 0")
            // {
            //     // Print the neighbour name, relative position, distance, relative position normalized and cohesion
            //     Debug.Log("Neighbour: " + neighbourChild.name + " Relative Position: " + relativePosition + " Distance: " + distance + " Relative Position Normalized: " + relativePosition.normalized + " thisCohesion: " + thisCohesion + " Cohesion: " + cohesion + " Cohesion Magnitude: " + cohesion.magnitude);
            // }
        }

        // if (droneName == "Drone 0")
        // {
        //     // Print the algorithm parameters: d_ref, r0_coh, delta, a, b, c
        //     Debug.Log("d_ref: " + d_ref + " r0_coh: " + r0_coh + " delta: " + delta + " a: " + a + " b: " + b + " c: " + c);
        // }

        // TODO: Add the obstacle avoidance force here

        // Get the total velocity command
        swarmInput = velocityMatching + cohesion + obstacle;

        GetComponent<VelocityControl>().swarm_vx = swarmInput.x;
        GetComponent<VelocityControl>().swarm_vy = swarmInput.y;
        GetComponent<VelocityControl>().swarm_vz = swarmInput.z;        
        
    }

    private Vector3 obstacleAvoidanceForce(Rigidbody droneRB)
    {

        Vector3 avoidForce = Vector3.zero;
        // Collider[] hitColliders = Physics.OverlapSphere(droneRB.transform.position, detectionRadius);
        // float radius = droneRB.gameObject.transform.parent.GetComponent<interactionHandler>().radiusOfCollider;

        // foreach (var hitCollider in hitColliders)
        // {
        //     if (hitCollider.CompareTag(obstacleTag))
        //     {
        //         // Find the closest point on the collider to the agent
        //         Vector3 closestPoint = hitCollider.ClosestPoint(droneRB.transform.position);
        //         Vector3 directionToObstacle = droneRB.transform.position - closestPoint;

        //         float distanceToObstacle = directionToObstacle.magnitude-radius;

        //         if (distanceToObstacle < detectionRadius && distanceToObstacle > 0)
        //         {
        //             float forceMagnitude = Mathf.Min(maxAvoidForce, 1 / (distanceToObstacle*distanceToObstacle) * obstacleAvoidanceForceWeight);
        //             avoidForce += directionToObstacle.normalized * forceMagnitude;

        //         }
        //     }
        // }

        // droneRB.transform.parent.GetComponent<varibaleManager>().lastObstacle = avoidForce;
        return avoidForce;
    }

    public float GetCohesionForce(float r)
    {
        float neighbourWeightDerivative = GetNeighbourWeightDerivative(r);
        float cohesionIntensity = GetCohesionIntensity(r);
        float neighbourWeight = GetNeighbourWeight(r);
        float cohesionIntensityDerivative = GetCohesionIntensityDerivative(r);

        return 1 / r0_coh * neighbourWeightDerivative * cohesionIntensity + neighbourWeight * cohesionIntensityDerivative;
    }

    // Cohesion intensity function
    public float GetCohesionIntensity(float r)
    {
        float diff = r - d_ref;
        return ((a + b) / 2) * (Mathf.Sqrt(1 + Mathf.Pow(diff + c, 2)) - Mathf.Sqrt(1 + c * c)) + ((a - b) * diff) / 2;
    }

    // Derivative of cohesion intensity function
    float GetCohesionIntensityDerivative(float r)
    {
        float diff = r - d_ref;
        return ((a + b) / 2) * (diff + c) / Mathf.Sqrt(1 + Mathf.Pow(diff + c, 2)) + (a - b) / 2;
    }
    
    // Neighbor weight function
    public float GetNeighbourWeight(float r)
    {
        float r_ratio = r / r0_coh;

        if (r_ratio < delta)
        {
            return 1.0f;
        }
        else if (r_ratio < 1.0f)
        {
            float arg = Mathf.PI * (r_ratio - delta) / (1 - delta);
            return Mathf.Pow(0.5f * (1.0f + Mathf.Cos(arg)), 2);
        }
        else
        {
            return 0.0f;
        }
    }

    // Derivative of neighbor weight function
    float GetNeighbourWeightDerivative(float r)
    {
        float r_ratio = r / r0_coh;

        if (r_ratio < delta)
        {
            return 0.0f;
        }
        else if (r_ratio < 1.0f)
        {
            float arg = Mathf.PI * (r_ratio - delta) / (1 - delta);
            return 0.5f*(-Mathf.PI) / (1 - delta) * (1 + Mathf.Cos(arg)) * Mathf.Sin(arg);
        }
        else
        {
            return 0.0f;
        }
    }
}