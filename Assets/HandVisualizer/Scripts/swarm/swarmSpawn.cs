using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class swarmSpawn : MonoBehaviour
{
    public GameObject dronePrefab;
    public List<GameObject> swarm = new List<GameObject>();
    
    [Header("Grid Dimensions")] // Added header for clarity
    public int dronesAlongX = 3;
    public int dronesAlongY = 1; // Added Y dimension control
    public int dronesAlongZ = 3;
    
    [Header("Spawning Parameters")] // Added header for clarity
    public float droneSpacing = 3.0f;
    public int start_x = 0;
    public int start_y = 0; // Renamed for consistency, represents the base Y level
    public int start_z = 0;
    public bool randomYaw = true;
    
    [Header("Optional Components")] // Added header for clarity
    public bool addScreens = true;
    public bool addTPVCamera = true;
    public bool streamUDP = true;
    
    public GameObject swarmParent;
    private screenSpawn screenSpawn;
    private int droneNumber = 0;

    // Start is called before the first frame update
    void Start()
    {
        // Get the screenSpawn script if it exists on the same GameObject
        screenSpawn = GetComponent<screenSpawn>();
        if (screenSpawn == null && addScreens)
        {
            Debug.LogWarning("screenSpawn script not found, cannot add screens.");
            addScreens = false; // Disable screen adding if script is missing
        }

        // Create an empty GameObject to serve as the parent for all drones
        swarmParent = new GameObject("SwarmParent");
        
        // Spawn drones in a 3D grid
        for (int x = 0; x < dronesAlongX; x++)
        {
            for (int y = 0; y < dronesAlongY; y++) // Added Y loop
            {
                for (int z = 0; z < dronesAlongZ; z++)
                {
                    // Assign the drone position in 3D grid
                    Vector3 dronePosition = new Vector3(
                        start_x + x * droneSpacing, 
                        start_y + y * droneSpacing, // Added Y position calculation
                        start_z + z * droneSpacing
                    );

                    Quaternion droneRotation = Quaternion.identity; // Default to no rotation
                    if (randomYaw)
                    {
                        // Generate a random yaw angle and convert it to a quaternion
                        float randomYawAngle = Random.Range(0f, 360f);
                        droneRotation = Quaternion.Euler(0, randomYawAngle, 0);
                    }

                    // Instantiate the drone prefab 
                    GameObject drone = Instantiate(dronePrefab);
                    // Use swarm.Count to ensure unique names even with multiple spawners
                    drone.name = "Drone " + swarm.Count; 

                    // Find the child object that holds the main components and position/rotate it
                    Transform droneParentTransform = drone.transform.Find("DroneParent"); 
                    if (droneParentTransform != null)
                    {
                        droneParentTransform.position = dronePosition;
                        droneParentTransform.rotation = droneRotation;
                    }
                    else
                    {
                        // Fallback if "DroneParent" doesn't exist: position the root object
                        Debug.LogWarning($"Drone prefab for {drone.name} does not have a 'DroneParent' child. Positioning root object instead.");
                        drone.transform.position = dronePosition;
                        drone.transform.rotation = droneRotation;
                    }


                    // Parent the drone under the swarmParent GameObject
                    drone.transform.parent = swarmParent.transform;
                    
                    // Add the drone to the swarm list
                    swarm.Add(drone);
                }
            }
        }
        
        // Add the swarm list to the relevant script of each drone
        foreach (GameObject drone in swarm)
        {
            // Find the DroneParent object again (or use the root if fallback occurred)
            Transform droneControllerTransform = drone.transform.Find("DroneParent") ?? drone.transform;

            // Try to get the swarmAlgorithm component
            swarmAlgorithm algorithmScript = droneControllerTransform.GetComponent<swarmAlgorithm>();
            if (algorithmScript != null)
            {
                 algorithmScript.swarm = swarm; // Assign the complete swarm list
            }
            else
            {
                Debug.LogError($"swarmAlgorithm script not found on {droneControllerTransform.name} for {drone.name}");
            }
           
            // Assign other necessary references if needed here...
        }

        // Spawn screens for each drone in the swarm
        if (addScreens && screenSpawn != null)
        {
            screenSpawn.SpawnScreens(swarm);
        }

        // Add a TPV camera to follow the swarm (if component exists)
        if (addTPVCamera)
        {
            // Get the SwarmFollowCamera script from this GameObject
            SwarmFollowCamera swarmFollowCamera = GetComponent<SwarmFollowCamera>();
            if (swarmFollowCamera != null)
            {
                // Setup the camera to follow the swarm
                swarmFollowCamera.SetupCamera(swarm);
            }
            else
            {
                 Debug.LogWarning("SwarmFollowCamera script not found, cannot add TPV camera.");
            }
        }

        if(streamUDP)
        {
            // Get the UDPConnection script from this GameObject
            UDPConnection udpConnection = GetComponent<UDPConnection>();

            // set the swarm list in the UDPConnection script
            if (udpConnection != null)
            {
                udpConnection.swarm = swarm;
            }
            else
            {
                Debug.LogWarning("UDPConnection script not found, cannot set swarm list.");
            }

        }
    }

    // Update is called once per frame
    void Update()
    {
        // Usually empty for a spawner script
    }
}
