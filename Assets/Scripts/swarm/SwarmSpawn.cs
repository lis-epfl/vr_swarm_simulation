using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class swarmSpawn : MonoBehaviour
{
    public GameObject dronePrefab;
    public List<GameObject> swarm = new List<GameObject>();
    public int dronesAlongX = 3;
    public int dronesAlongZ = 3;
    public float droneSpacing = 3.0f;
    public int start_x = 0;
    public int start_y = 0;
    public int start_z = 0;
    public bool randomYaw = true;
    public int YawDegrees = 0;
    public bool UsePySender = false;
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
        
        // Spawn drones in a grid
        for (int x = 0; x < dronesAlongX; x++)
        {
            for (int z = 0; z < dronesAlongZ; z++)
            {
                // Assign the drone position
                Vector3 dronePosition = new Vector3(start_x + x * droneSpacing, start_y, start_z + z * droneSpacing);

                Quaternion droneRotation = Quaternion.Euler(0, YawDegrees, 0);
                if (randomYaw)
                {
                    // Generate a random yaw angle and convert it to a quaternion
                    float randomYaw = Random.Range(0f, 360f);
                    droneRotation = Quaternion.Euler(0, randomYaw, 0);
                }

                // Instantiate the drone prefab at the drone position and rotation
                GameObject drone = Instantiate(dronePrefab);
                drone.name = "Drone " + swarm.Count;

                // Move the drone to the position
                Transform droneParent = drone.transform.Find("DroneParent");
                droneParent.position = dronePosition;
                droneParent.rotation = droneRotation;

                // Parent the drone under the swarmParent GameObject
                drone.transform.parent = swarmParent.transform;
                
                // Add the drone to the swarm list
                swarm.Add(drone);
            }
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
        }

        // Set the swarm in interface scripts
        interfaceManager.SetSwarm(swarm);

        if (UsePySender)
        {
            PySender.Instance.InitializeDroneDataSharedMemory((uint)swarm.Count);
            PySender.Instance.Swarm = swarm;
        }
        
    }

    // Update is called once per frame
    void Update()
    {
        
    }

    public void Reset()
    {
        if (swarm.Count > 0)
        {
            foreach (GameObject drone in swarm)
            {
            Transform droneParent = drone.transform.Find("DroneParent");

            // Add the swarm to the necessary scripts
            droneParent.GetComponent<SwarmAlgorithm>().Reset();   
            droneParent.GetComponent<AttitudeAlgorithm>().Reset();  
            }
        }
    }
}
