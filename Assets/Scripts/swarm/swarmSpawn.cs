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
    public GameObject swarmParent;
    private screenSpawn screenSpawn;
    private visualiseOlfatiSaber visualiseOlfatiSaber;
    private int droneNumber = 0;

    // Start is called before the first frame update
    void Start()
    {
        
        // Get the screenSpawn script
        screenSpawn = GetComponent<screenSpawn>();

        // Create an empty GameObject to serve as the parent for all drones
        swarmParent = new GameObject("SwarmParent");
        
        // Spawn drones in a grid
        for (int x = 0; x < dronesAlongX; x++)
        {
            for (int z = 0; z < dronesAlongZ; z++)
            {
                // Assign the drone position
                Vector3 dronePosition = new Vector3(start_x + x * droneSpacing, start_y, start_z + z * droneSpacing);

                Quaternion droneRotation = Quaternion.Euler(0, 0, 0);
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

            // Add the swarm to the swarmAlgorithm script
            droneParent.GetComponent<swarmAlgorithm>().swarm = swarm;            
        }

        visualiseOlfatiSaber = GetComponent<visualiseOlfatiSaber>();
        visualiseOlfatiSaber.swarm = swarm;
        
        
        // Spawn screens for each drone in the swarm
        screenSpawn.SpawnScreens(swarm);
        
    }

    // Update is called once per frame
    void Update()
    {
        
    }
}
