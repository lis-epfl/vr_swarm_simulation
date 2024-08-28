using System.Collections;
using System.Collections.Generic;
using System.Security;
using UnityEngine;

public class OlfatiSaber : MonoBehaviour
{
    public List<GameObject> swarm;

    public float d_ref = 7.0f;
    public float r0_coh = 20.0f;
    public float delta = 0.1f;
    public float a = 0.3f;
    public float b = 0.5f;
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

        // Calcualte the cohesion force for each of the neighbours
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
            cohesion += GetCohesionForce(distance, d_ref, a, b, c, r0_coh, delta) * relativePosition.normalized;

        }

        // TODO: Add the obstacle avoidance force here

        // Get the total velocity command
        swarmInput = velocityMatching + cohesion + obstacle;

        GetComponent<VelocityControl>().swarm_vx = swarmInput.x;
        GetComponent<VelocityControl>().swarm_vy = swarmInput.y;
        GetComponent<VelocityControl>().swarm_vz = swarmInput.z;
        
    }

    // private Vector3 obstacleAvoidanceForce(Rigidbody droneRB)
    // {

    //     Vector3 avoidForce = Vector3.zero;
    //     Collider[] hitColliders = Physics.OverlapSphere(droneRB.transform.position, detectionRadius);
    //     float radius = droneRB.gameObject.transform.parent.GetComponent<interactionHandler>().radiusOfCollider;

    //     foreach (var hitCollider in hitColliders)
    //     {
    //         if (hitCollider.CompareTag(obstacleTag))
    //         {
    //             // Find the closest point on the collider to the agent
    //             Vector3 closestPoint = hitCollider.ClosestPoint(droneRB.transform.position);
    //             Vector3 directionToObstacle = droneRB.transform.position - closestPoint;

    //             float distanceToObstacle = directionToObstacle.magnitude-radius;

    //             if (distanceToObstacle < detectionRadius && distanceToObstacle > 0)
    //             {
    //                 float forceMagnitude = Mathf.Min(maxAvoidForce, 1 / (distanceToObstacle*distanceToObstacle) * obstacleAvoidanceForceWeight);
    //                 avoidForce += directionToObstacle.normalized * forceMagnitude;

    //             }
    //         }
    //     }

    //     droneRB.transform.parent.GetComponent<varibaleManager>().lastObstacle = avoidForce;
    //     return avoidForce;
    // }

    private float GetCohesionForce(float r, float d_ref, float a, float b, float c, float r0, float delta)
    {
        float neighbourWeightDer = GetNeighbourWeightDer(r, r0, delta);
        float cohesionIntensity = GetCohesionIntensity(r, d_ref, a, b, c);
        float neighbourWeight = GetNeighbourWeight(r, r0, delta);
        float cohesionIntensityDer = GetCohesionIntensityDer(r, d_ref, a, b, c);

        return 1 / r0 *neighbourWeightDer * cohesionIntensity + neighbourWeight * cohesionIntensityDer;
    }

    private float GetCohesionIntensity(float r, float d_ref, float a, float b, float c)
    {
        float diff = r - d_ref;

        return (a + b) / 2 * Mathf.Sqrt(1 + Mathf.Pow(diff + c, 2)) - Mathf.Sqrt(1 + (c * c)) + (a - b) * diff / 2;
    }

    private float GetCohesionIntensityDer(float r, float d_ref, float a, float b, float c)
    {
        float diff = r - d_ref;
        return (a + b) / 2 * (diff + c) / Mathf.Sqrt(1 + Mathf.Pow(diff + c, 2)) + (a - b) / 2; 
    }

    private float GetNeighbourWeight(float r, float r0, float delta)
    {
        float rRatio = r / r0;

        if (rRatio < delta)
        {
            return 1;
        }
        else if (rRatio < 1)
        {
            return 0.25f * Mathf.Pow(1 + Mathf.Cos(Mathf.PI * (rRatio - delta) / (1 - delta)), 2);
        }
        else
        {
            return 0;
        }
    }

    private float GetNeighbourWeightDer(float r, float r0, float delta)
    {
        float rRatio = r / r0;

        if (rRatio < delta)
        {
            return 1;
        }
        else if (rRatio < 1)
        {
            float arg = Mathf.PI * (rRatio - delta) / (1 - delta);
            return -0.5f * Mathf.PI / (1 - delta) * (1 + Mathf.Cos(arg)) * Mathf.Sin(arg);
        }
        else
        {
            return 0;
        }
    }
}