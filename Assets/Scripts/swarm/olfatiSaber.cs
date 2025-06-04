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
    public float d_obs = 5.0f;
    public float r0_obs = 6.0f;
    public float lambda_obs = 1.0f;
    public float c_obs = 4.3f;
    public float scaleFactor = 10.0f;


    public float maxMigrationDistance = 10.0f;



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

    private string obstacleLayer = "Obstacle";

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

            // Scale the distance to fit the cohesion function
            distance = distance / scaleFactor;
            
            // Cohesion
            Vector3 thisCohesion = GetCohesionForce(distance, d_ref, r0_coh) * relativePosition.normalized;
            cohesion += thisCohesion;

        }

        // Get the obstacle avoidance force
        obstacle = GetObstacleForce();

        // Get the total velocity command
        swarmInput = velocityMatching + cohesion + obstacle;

        // Log the forces for debugging and the overall swarm input
        // Debug.Log($"Velocity Matching Force for {droneName}: {velocityMatching}");
        // Debug.Log($"Cohesion Force for {droneName}: {cohesion}");
        // Debug.Log($"Obstacle Avoidance Force for {droneName}: {obstacle}");
        // Debug.Log($"Swarm Input for {droneName}: {swarmInput}");

        GetComponent<VelocityControl>().swarm_vx = swarmInput.x;
        GetComponent<VelocityControl>().swarm_vy = swarmInput.y;
        GetComponent<VelocityControl>().swarm_vz = swarmInput.z;        
        
    }

    private Vector3 GetObstacleForce()
    {


        Vector3 ObsCoh = Vector3.zero;
        Vector3 ObsVel = Vector3.zero;

        // Get the drone velocity
        Vector3 droneVelocity = GetComponent<Rigidbody>().velocity;

        // Find the closest point on each obstacle with the 'Obstacle' tag and log the distance
        Collider[] obstacles = Physics.OverlapSphere(transform.position, d_obs, LayerMask.GetMask(obstacleLayer));
        foreach (Collider obstacleCollider in obstacles)
        {
            // Find the closest point on the obstacle
            Vector3 closestPoint = obstacleCollider.ClosestPointOnBounds(transform.position);

            // Calculate the distance to the closest point
            Vector3 directionToObstacle = closestPoint - transform.position;
            float distanceToObstacle = directionToObstacle.magnitude;

            // Scale the distance to fit the obstacle avoidance function
            distanceToObstacle = distanceToObstacle / scaleFactor;

            // If the distance is less than d_obs, calculate the obstacle avoidance force
            if (distanceToObstacle < r0_obs)
            {
                // Calculate the obstacle velocity, vel_obs, using the logic above
                float s = 1 / (distanceToObstacle + 1);
                Vector3 pos_obs = s * transform.position + (1 - s) * closestPoint;
                float s_der = 1 * Vector3.Dot(droneVelocity, (pos_obs - transform.position).normalized) / Mathf.Pow(1 + distanceToObstacle, 2);
                Vector3 vel_obs = s * droneVelocity - 1 * (s_der / s) * (pos_obs - transform.position).normalized;

                ObsCoh += GetCohesionForce(distanceToObstacle, d_obs, r0_obs) * directionToObstacle.normalized;
                ObsVel += (vel_obs - droneVelocity);
            }

        }

        Vector3 obstacleAvoidanceForce = c_obs * ObsCoh + c_vm * ObsVel;

        return obstacleAvoidanceForce;
    }

    public float GetCohesionForce(float r, float ref_d = -1, float r0 = -1)
    {
        // Use default values if parameters are not provided
        if (ref_d == -1) ref_d = d_ref;
        if (r0 == -1) r0 = r0_coh;

        float neighbourWeightDerivative = GetNeighbourWeightDerivative(r, r0);
        float cohesionIntensity = GetCohesionIntensity(r, ref_d);
        float neighbourWeight = GetNeighbourWeight(r, r0);
        float cohesionIntensityDerivative = GetCohesionIntensityDerivative(r, ref_d);

        return 1 / r0_coh * neighbourWeightDerivative * cohesionIntensity + neighbourWeight * cohesionIntensityDerivative;
    }

    // Cohesion intensity function
    public float GetCohesionIntensity(float r, float ref_d=-1)
    {
        // Use default value if ref_d is not provided
        if (ref_d == -1) ref_d = d_ref;

        float diff = r - ref_d;
        return ((a + b) / 2) * (Mathf.Sqrt(1 + Mathf.Pow(diff + c, 2)) - Mathf.Sqrt(1 + c * c)) + ((a - b) * diff) / 2;
    }

    // Derivative of cohesion intensity function
    float GetCohesionIntensityDerivative(float r, float ref_d=-1)
    {
        // Use default value if ref_d is not provided
        if (ref_d == -1) ref_d = d_ref;

        // Derivative of the cohesion intensity function
        float diff = r - ref_d;
        return ((a + b) / 2) * (diff + c) / Mathf.Sqrt(1 + Mathf.Pow(diff + c, 2)) + (a - b) / 2;
    }
    
    // Neighbor weight function
    public float GetNeighbourWeight(float r, float r0=-1)
    {

        // Use default value if r0 is not provided
        if (r0 == -1) r0 = r0_coh;

        float r_ratio = r / r0;

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
    float GetNeighbourWeightDerivative(float r, float r0=-1)
    {
        // Use default value if r0 is not provided
        if (r0 == -1) r0 = r0_coh;

        float r_ratio = r / r0;

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