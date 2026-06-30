using System;
using System.Collections;
using System.Collections.Generic;
using Unity.VisualScripting;
using UnityEngine;

public enum SpawnPattern
{
    Grid,
    Circle
}

public class swarmSpawn : MonoBehaviour
{
    public GameObject dronePrefab;
    public List<GameObject> swarm = new List<GameObject>();
    public SpawnPattern spawnPattern = SpawnPattern.Grid;
    public int dronesAlongX = 3;
    public int dronesAlongZ = 3;
    public float droneSpacing = 3.0f;

    [Header("Circle pattern")]
    public int circleDroneCount = 10;
    public float circleRadius = 5.0f;
    public int start_x = 0;
    public int start_y = 0;
    public int start_z = 0;
    public bool randomYaw = false;
    public bool RandomizeMassAndInertia = false;
    public int YawDegrees = 0;
    public bool trackBirdsEye = false;
    public GameObject swarmParent;
    private InterfaceManager interfaceManager;
    private ScreenSpawn ScreenSpawn;
    private visualiseOlfatiSaber visualiseOlfatiSaber;
    private ViewManager viewManager;
    private int droneNumber = 0;

    // Awake is called before Start
    void Awake()
    {
        // Get the InterfaceManager
        interfaceManager = GetComponent<InterfaceManager>();
        
        // Get the ScreenSpawn script
        ScreenSpawn = GetComponent<ScreenSpawn>();

        viewManager = GetComponent<ViewManager>();

        //Get the visualize olfati-saber script if any
        visualiseOlfatiSaber = GetComponent<visualiseOlfatiSaber>();
    }
    
    // Start is called before the first frame update
    void Start()
    {
        


        // Create an empty GameObject to serve as the parent for all drones
        swarmParent = new GameObject("SwarmParent");

        // Spawn drones in the chosen pattern
        foreach (Vector3 dronePosition in GetSpawnPositions())
        {
            SpawnDrone(dronePosition);
        }

        // Add the swarm list to the reynolds script of each drone
        foreach (GameObject drone in swarm)
        {

            // Get the drone number from the name
            string[] splitName = drone.name.Split(' ');
            droneNumber = int.Parse(splitName[1]);

            // Find the DroneParent object
            Transform droneParent = drone.transform.Find("DroneParent");

            // Add the swarm to the necessary scripts
            droneParent.GetComponent<SwarmAlgorithm>().swarm = swarm;
            droneParent.GetComponent<AttitudeAlgorithm>().swarm = swarm;

            // Per-drone mass/inertia variation for realism
            Rigidbody rb = droneParent.GetComponent<Rigidbody>();
            if (rb != null && RandomizeMassAndInertia)
            {
                rb.mass *= UnityEngine.Random.Range(0.92f, 1.08f);
                rb.inertiaTensor *= UnityEngine.Random.Range(0.95f, 1.05f);
            }
        }

        // Set the swarm in interface scripts
        interfaceManager.SetSwarm(swarm);
        
    }

    // Update is called once per frame
    void Update()
    {

    }

    /// <summary>
    /// Computes the spawn positions for the configured pattern.
    /// Grid: dronesAlongX × dronesAlongZ lattice anchored at (start_x, start_y, start_z).
    /// Circle: circleDroneCount drones evenly spaced on a ring of circleRadius around the start position.
    /// </summary>
    private IEnumerable<Vector3> GetSpawnPositions()
    {
        if (spawnPattern == SpawnPattern.Circle)
        {
            int count = Mathf.Max(1, circleDroneCount);

            // One drone at the centre, the remaining (count - 1) evenly spaced on the ring.
            yield return new Vector3(start_x, start_y, start_z);

            int ringCount = count - 1;
            for (int i = 0; i < ringCount; i++)
            {
                float angle = i * Mathf.PI * 2f / ringCount;
                yield return new Vector3(
                    start_x + Mathf.Cos(angle) * circleRadius,
                    start_y,
                    start_z + Mathf.Sin(angle) * circleRadius);
            }
        }
        else
        {
            for (int x = 0; x < dronesAlongX; x++)
            {
                for (int z = 0; z < dronesAlongZ; z++)
                {
                    yield return new Vector3(
                        start_x + x * droneSpacing,
                        start_y,
                        start_z + z * droneSpacing);
                }
            }
        }
    }

    /// <summary>
    /// Instantiates a single drone at the given position, applies yaw, and registers it in the swarm.
    /// </summary>
    private void SpawnDrone(Vector3 dronePosition)
    {
        Quaternion droneRotation = Quaternion.Euler(0, YawDegrees, 0);
        if (randomYaw)
        {
            // Generate a random yaw angle and convert it to a quaternion
            float yaw = UnityEngine.Random.Range(0f, 360f);
            droneRotation = Quaternion.Euler(0, yaw, 0);
        }

        // Instantiate the drone prefab at the drone position and rotation
        GameObject drone = Instantiate(dronePrefab);
        drone.name = "Drone " + swarm.Count;
        StateFinder state = drone.GetComponent<StateFinder>();
        if (state != null)
        {
            state.Altitude = dronePosition.y;
            state.Position = dronePosition;
            state.GroundTruthPosition = dronePosition;
        }

        // Move the drone to the position
        Transform droneParent = drone.transform.Find("DroneParent");
        droneParent.position = dronePosition;
        droneParent.rotation = droneRotation;

        // Parent the drone under the swarmParent GameObject
        drone.transform.parent = swarmParent.transform;

        // Add the drone to the swarm list
        swarm.Add(drone);
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

#if UNITY_EDITOR
    private void OnDrawGizmos()
    {
        // Birds-eye scene view tracking when in play mode
        if (Application.isPlaying && trackBirdsEye && swarm != null && swarm.Count > 0)
        {
            Vector3 center = GetSwarmCenterGroundTruth();
            UnityEditor.SceneView sv = UnityEditor.SceneView.lastActiveSceneView;
            if (sv != null)
            {
                sv.LookAt(center, Quaternion.Euler(90, 0, 0), 40f);
            }
        }

        uint index = 0;
        float armSpacing = spawnPattern == SpawnPattern.Circle ? circleRadius : droneSpacing;

        foreach (Vector3 pos in GetSpawnPositions())
        {
            index++;

            // Drone body
            Gizmos.color = new Color(0.3f, 0.8f, 1f, 0.9f);
            Gizmos.DrawSphere(pos, 0.25f);

            // Arm cross
            Gizmos.color = new Color(0.3f, 0.8f, 1f, 0.4f);
            float armLen = armSpacing * 0.35f;
            Gizmos.DrawLine(pos + Vector3.left  * armLen, pos + Vector3.right   * armLen);
            Gizmos.DrawLine(pos + Vector3.back  * armLen, pos + Vector3.forward * armLen);

            // Label for drone number on top of the sphere
            UnityEditor.Handles.color = new Color(0.3f, 0.8f, 1f, 0.9f);
            UnityEditor.Handles.Label(pos + Vector3.up * 0.5f, $"{index}");
        }

        if (spawnPattern == SpawnPattern.Circle)
        {
            // Ring outline around the start position
            Vector3 ringCenter = new Vector3(start_x, start_y, start_z);
            Gizmos.color = new Color(0.3f, 0.8f, 1f, 0.25f);
            Vector3 prev = ringCenter + new Vector3(circleRadius, 0f, 0f);
            const int segments = 64;
            for (int s = 1; s <= segments; s++)
            {
                float a = s * Mathf.PI * 2f / segments;
                Vector3 next = ringCenter + new Vector3(Mathf.Cos(a) * circleRadius, 0f, Mathf.Sin(a) * circleRadius);
                Gizmos.DrawLine(prev, next);
                prev = next;
            }
        }
        else if (dronesAlongX > 0 && dronesAlongZ > 0)
        {
            // Grid bounding box
            Vector3 origin = new Vector3(start_x, start_y, start_z);
            Vector3 corner = origin + new Vector3((dronesAlongX - 1) * droneSpacing, 0f, (dronesAlongZ - 1) * droneSpacing);
            Vector3 center = (origin + corner) * 0.5f;
            Vector3 size   = new(
                Mathf.Max((dronesAlongX - 1) * droneSpacing, droneSpacing * 0.5f),
                0.05f,
                Mathf.Max((dronesAlongZ - 1) * droneSpacing, droneSpacing * 0.5f));

            Gizmos.color = new Color(0.3f, 0.8f, 1f, 0.08f);
            Gizmos.DrawCube(center, size);
        }

    }
#endif

    public void Reset()
    {
        Debug.Log("Resetting swarm to initial positions.");
        if (swarm.Count > 0)
        {
            foreach (GameObject drone in swarm)
            {
            Transform droneParent = drone.transform.Find("DroneParent");
            Transform stateFinderTransform = drone.transform.Find("StateFinder");

            // Add the swarm to the necessary scripts
            droneParent.GetComponent<SwarmAlgorithm>().Reset();   
            droneParent.GetComponent<AttitudeAlgorithm>().Reset();  
            stateFinderTransform.GetComponent<StateFinder>().Reset();
            }
        }
    }

    public void ResetToPos(Vector3 newStartPosition)
    {
        Debug.Log($"Resetting swarm to new position: {newStartPosition}");
        if (swarm.Count > 0)
        {
            // Offsets of each spawn slot relative to the original start position.
            Vector3 startOrigin = new Vector3(start_x, start_y, start_z);
            List<Vector3> offsets = new List<Vector3>();
            foreach (Vector3 pos in GetSpawnPositions())
                offsets.Add(pos - startOrigin);

            int index = 0;
            foreach (GameObject drone in swarm)
            {
                Transform droneParent = drone.transform.Find("DroneParent");

                // Add the swarm to the necessary scripts
                droneParent.GetComponent<SwarmAlgorithm>().Reset();
                droneParent.GetComponent<AttitudeAlgorithm>().Reset();

                // Move the drone to the new position, preserving the spawn-pattern layout
                Vector3 offset = offsets.Count > 0 ? offsets[index % offsets.Count] : Vector3.zero;
                index++;

                // Reset velocity control and state finder data
                VelocityControl vc = droneParent.GetComponent<VelocityControl>();
                if (vc != null)
                {
                    vc.ResetToPos(newStartPosition + offset);
                }
                // Reset DroneHealthMonitor to prevent false "dead" markings during repositioning
                DroneHealthMonitor healthMonitor = droneParent.GetComponent<DroneHealthMonitor>();
                if (healthMonitor != null)
                {
                    healthMonitor.Reset();
                }
            }
        }
    }

    /// <summary>
    /// Disables health monitoring temporarily during critical operations.
    /// Prevents drones from being marked as dead during repositioning.
    /// </summary>
    public void DisableHealthMonitoring()
    {
        if (swarm.Count > 0)
        {
            foreach (GameObject drone in swarm)
            {
                Transform droneParent = drone.transform.Find("DroneParent");
                DroneHealthMonitor healthMonitor = droneParent.GetComponent<DroneHealthMonitor>();
                if (healthMonitor != null)
                {
                    healthMonitor.enabled = false;
                }
            }
        }
    }

    /// <summary>
    /// Re-enables health monitoring after critical operations.
    /// Should be called after drones have settled in their new positions.
    /// </summary>
    public void EnableHealthMonitoring()
    {
        if (swarm.Count > 0)
        {
            foreach (GameObject drone in swarm)
            {
                Transform droneParent = drone.transform.Find("DroneParent");
                DroneHealthMonitor healthMonitor = droneParent.GetComponent<DroneHealthMonitor>();
                if (healthMonitor != null)
                {
                    healthMonitor.enabled = true;
                    healthMonitor.Reset();
                }
            }
        }
    }

    /// <summary>
    /// Resets all DroneHealthMonitor instances to clear their elapsed time.
    /// Call this to restart the stabilization delay on all drones.
    /// </summary>
    public void ResetAllHealthMonitors()
    {
        if (swarm.Count > 0)
        {
            foreach (GameObject drone in swarm)
            {
                Transform droneParent = drone.transform.Find("DroneParent");
                DroneHealthMonitor healthMonitor = droneParent.GetComponent<DroneHealthMonitor>();
                if (healthMonitor != null)
                {
                    healthMonitor.Reset();
                }
            }
        }
    }

    /// <summary>
    /// Enables stuck-on-gate detection for all drones.
    /// Call this when entering a trial.
    /// </summary>
    public void EnableStuckDetection()
    {
        if (swarm.Count > 0)
        {
            foreach (GameObject drone in swarm)
            {
                Transform droneParent = drone.transform.Find("DroneParent");
                DroneHealthMonitor healthMonitor = droneParent.GetComponent<DroneHealthMonitor>();
                if (healthMonitor != null)
                {
                    healthMonitor.EnableStuckDetection();
                }
            }
        }
    }

    /// <summary>
    /// Disables stuck-on-gate detection for all drones.
    /// Call this when exiting a trial (e.g., entering FlyingPractice).
    /// </summary>
    public void DisableStuckDetection()
    {
        if (swarm.Count > 0)
        {
            foreach (GameObject drone in swarm)
            {
                Transform droneParent = drone.transform.Find("DroneParent");
                DroneHealthMonitor healthMonitor = droneParent.GetComponent<DroneHealthMonitor>();
                if (healthMonitor != null)
                {
                    healthMonitor.DisableStuckDetection();
                }
            }
        }
    }
}
