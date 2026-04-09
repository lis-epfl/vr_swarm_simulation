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

    void Start()
    {
        velocityControl = GetComponent<VelocityControl>();
        olfatiSaber = GetComponent<OlfatiSaber>();
        elapsedTime = 0.0f;
    }

    void FixedUpdate()
    {
        if (State == null || !State.IsAlive)
            return;

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
        if (IsTooFarFromSwarm() || IsCrashed() || IsTooCloseToGround())
        {
            State.IsAlive = false;
        }
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

            center += vc.State.GroundTruthPosition;
            validCount++;
        }

        if (validCount > 0)
            center /= validCount;

        return center;
    }

    public void Reset()
    {
        State.IsAlive = true;
        elapsedTime = 0.0f;
    }
}
