using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class swarmSpawn : MonoBehaviour
{
    public GameObject dronePrefab;
    public List<GameObject> swarm = new List<GameObject>();
    public int gridLength = 3;
    public float gridSpacing = 3.0f;

    // Start is called before the first frame update
    void Start()
    {
        // Spawn drones in a grid
        for (int x = 0; x < gridLength; x++)
        {
            for (int z = 0; z < gridLength; z++)
            {
                Vector3 dronePosition = new Vector3(x * gridSpacing, 0, z * gridSpacing);
                GameObject drone = Instantiate(dronePrefab, dronePosition, Quaternion.identity);
                drone.name = "Drone " + swarm.Count;
                
                swarm.Add(drone);

                Debug.Log("Spawned " + drone.name);
            }
        }

        Debug.Log("Swarm Size: " + swarm.Count);
        
        // Add the swarm list to the reynolds script of each drone
        foreach (GameObject drone in swarm)
        {
            Debug.Log("Adding swarm to " + drone.name);

            // Find the DroneParent object
            Transform droneParent = drone.transform.Find("DroneParent");
            droneParent.GetComponent<reynolds>().swarm = swarm;
            
        }
    }

    // Update is called once per frame
    void Update()
    {
        
    }
}
