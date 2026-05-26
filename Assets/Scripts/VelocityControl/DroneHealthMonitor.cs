using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class DroneHealthMonitor : MonoBehaviour
{
    public StateFinder State;
    private VelocityControl velocityControl;
    private OlfatiSaber olfatiSaber;
    private List<GameObject> swarm;

    // Health check thresholds
    public float maxPitchAngle = 0.5f; // ~30 degrees in radians, drone crashed if exceeded
    public float maxRollAngle = 0.5f;  // ~30 degrees in radians, drone crashed if exceeded
    public float groundCollisionThreshold = 0.3f; // How close to ground before marked dead

    // Cohesion distance threshold multiplier
    public float cohesionDistanceMultiplier = 1.0f;

    // Stabilization delay before health checks start
    public float stabilizationDelay = 1.0f;
    private float elapsedTime = 0.0f;

    [Header("Dead Drone Handling")]
    [Tooltip("Y position to park dead drones (far below the course so they do not interfere)")]
    public float parkingY = -100f;

    [Header("Collision Detection")]
    public float droneCollisionRadius = 0.5f;

    [Header("Stuck Detection")]
    public float stuckVelocityThreshold = 0.3f;
    public float stuckObstacleRadius = 1.5f;
    public float stuckTimeThreshold = 1.5f;
    public float stuckDetectionDelay = 3.0f;

    private Rigidbody rb;
    private Renderer[] droneRenderers;
    private bool wasAlive = true;
    private float stuckTimer = 0f;
    private float stuckDetectionTimer = 0f;
    private int k_ObstacleLayerMask;
    private bool stuckDetectionEnabled = false;

    void Start()
    {
        velocityControl = GetComponent<VelocityControl>();
        olfatiSaber = GetComponent<OlfatiSaber>();
        rb = GetComponent<Rigidbody>();
        droneRenderers = GetComponentsInChildren<Renderer>(includeInactive: true);
        elapsedTime = 0.0f;
        wasAlive = true;
        k_ObstacleLayerMask = LayerMask.GetMask("Obstacle");
    }

    void FixedUpdate()
    {
        if (State == null)
            return;

        if (!State.IsAlive)
        {
            if (wasAlive)
            {
                ParkDrone();
                wasAlive = false;
            }
            return;
        }

        // Accumulate time for stabilization delay
        elapsedTime += Time.deltaTime;

        // Wait for stabilization period before checking health
        if (elapsedTime < stabilizationDelay)
            return;

        // Set swarm reference if not already set (gets populated by SwarmSpawn)
        if (swarm == null || swarm.Count == 0)
        {
            swarm = velocityControl.GetComponent<SwarmAlgorithm>().swarm;
            if (swarm == null || swarm.Count == 0)
                return;
        }

        // Run all health checks
        if (IsTooFarFromSwarm() || IsCrashed() || IsTooCloseToGround() || IsCollidingWithAnotherDrone() || IsStuckOnObstacle())
        {
            State.IsAlive = false;
            ParkDrone();
            wasAlive = false;
        }
    }

    private void ParkDrone()
    {
        if (rb != null)
        {
            rb.velocity = Vector3.zero;
            rb.angularVelocity = Vector3.zero;
            rb.isKinematic = true;
        }

        Vector3 pos = transform.position;
        transform.position = new Vector3(pos.x, parkingY, pos.z);
        transform.rotation = Quaternion.identity;

        foreach (Renderer r in droneRenderers)
        {
            if (r != null) r.enabled = false;
        }
    }

    private void UnparkDrone()
    {
        foreach (Renderer r in droneRenderers)
        {
            if (r != null) r.enabled = true;
        }

        if (rb != null)
            rb.isKinematic = false;
    }

    private bool IsCollidingWithAnotherDrone()
    {
        if (swarm == null || swarm.Count == 0)
            return false;

        foreach (GameObject drone in swarm)
        {
            Transform droneParent = drone.transform.Find("DroneParent");
            if (droneParent == null || droneParent.gameObject == gameObject)
                continue;

            VelocityControl vc = droneParent.GetComponent<VelocityControl>();
            if (vc == null || !vc.State.IsAlive)
                continue;

            float dist = Vector3.Distance(State.GroundTruthPosition, vc.State.GroundTruthPosition);
            if (dist < droneCollisionRadius)
            {
                vc.State.IsAlive = false;
                Debug.LogWarning($"{gameObject.name} collided with {droneParent.parent.name}");
                return true;
            }
        }
        return false;
    }

    private bool IsStuckOnObstacle()
    {
        if (!stuckDetectionEnabled)
        {
            stuckDetectionTimer = 0f;
            stuckTimer = 0f;
            return false;
        }

        stuckDetectionTimer += Time.deltaTime;
        if (stuckDetectionTimer < stuckDetectionDelay)
            return false;

        bool slowEnough = rb.velocity.magnitude < stuckVelocityThreshold;
        bool touchingObstacle = Physics.OverlapSphere(transform.position, stuckObstacleRadius, k_ObstacleLayerMask).Length > 0;

        if (slowEnough && touchingObstacle)
            stuckTimer += Time.deltaTime;
        else
            stuckTimer = 0f;

        if (stuckTimer >= stuckTimeThreshold)
        {
            Debug.LogWarning($"{gameObject.name} stuck on obstacle for {stuckTimer:F1}s");
            return true;
        }
        return false;
    }

    private bool IsTooFarFromSwarm()
    {
        if (swarm == null || swarm.Count == 0)
            return false;

        Vector3 currentPos = State.GroundTruthPosition;
        Vector3 swarmCenter = GetSwarmCenterGroundTruth();

        float distanceToCenter = Vector3.Distance(currentPos, swarmCenter);

        // Use r0_coh from OlfatiSaber with multiplier for safety margin
        float maxDistance = olfatiSaber.r0_coh * olfatiSaber.ScaleFactor * cohesionDistanceMultiplier;

        bool tooFar = distanceToCenter > maxDistance;

        if (tooFar)
        {
            Debug.LogWarning($"{gameObject.name} is too far from swarm. Distance: {distanceToCenter:F2}, Max: {maxDistance:F2}");
        }

        return tooFar;
    }

    private bool IsCrashed()
    {
        float pitch = Mathf.Abs(State.Angles.x);
        float roll = Mathf.Abs(State.Angles.z);

        bool crashed = pitch > maxPitchAngle || roll > maxRollAngle;

        if (crashed)
        {
            Debug.LogWarning($"{gameObject.name} crashed! Pitch: {pitch:F3}, Roll: {roll:F3}");
        }

        return crashed;
    }

    private bool IsTooCloseToGround()
    {
        float altitude = State.Altitude;
        bool tooLow = altitude < groundCollisionThreshold;

        if (tooLow)
        {
            Debug.LogWarning($"{gameObject.name} hit the ground! Altitude: {altitude:F3}");
        }

        return tooLow;
    }

    private Vector3 GetSwarmCenterGroundTruth()
    {
        Vector3 center = Vector3.zero;
        int validCount = 0;

        foreach (GameObject drone in swarm)
        {
            if (drone == null)
                continue;

            Transform droneParent = drone.transform.Find("DroneParent");
            if (droneParent == null)
                continue;

            VelocityControl vc = droneParent.GetComponent<VelocityControl>();
            if (vc == null || vc.State == null)
                continue;

            if (!vc.State.IsAlive)
                continue;

            center += vc.State.GroundTruthPosition;
            validCount++;
        }

        if (validCount > 0)
            center /= validCount;

        return center;
    }

    public void Reset()
    {
        UnparkDrone();
        State.IsAlive = true;
        elapsedTime = 0.0f;
        wasAlive = true;
        stuckTimer = 0f;
        stuckDetectionTimer = 0f;
    }

    public void EnableStuckDetection()
    {
        stuckDetectionEnabled = true;
        stuckDetectionTimer = 0f;
        stuckTimer = 0f;
    }

    public void DisableStuckDetection()
    {
        stuckDetectionEnabled = false;
        stuckDetectionTimer = 0f;
        stuckTimer = 0f;
    }
}
