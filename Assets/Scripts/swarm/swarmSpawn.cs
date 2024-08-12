using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class swarmSpawn : MonoBehaviour
{
    public GameObject dronePrefab;
    public List<GameObject> swarm = new List<GameObject>();
    public int gridLength = 3;
    public float gridSpacing = 3.0f;
    private int droneNumber = 0;

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

            // Get the drone number from the name
            string[] splitName = drone.name.Split(' ');
            droneNumber = int.Parse(splitName[1]);
            
            // Find the DroneParent object
            Transform droneParent = drone.transform.Find("DroneParent");
            droneParent.GetComponent<reynolds>().swarm = swarm;


            // Create a quad, render texture, and screen material for each drone

            // Create a quad
            GameObject quad = GameObject.CreatePrimitive(PrimitiveType.Quad);
            // position the quad at 6.6, -3, 9.8
            quad.transform.position = new Vector3(6.6f, -3.0f, 9.8f);
            // Name the quad 'quad_' followed by the drone number
            quad.name = "quad_" + droneNumber;

            // Create a render texture
            RenderTexture rt = new RenderTexture(256, 256, 24);
            // Name the render texture 'rt_' followed by the drone number
            rt.name = "rt_" + droneNumber;

            // Load all assets from the Resources folder
            Object[] resources = Resources.LoadAll("", typeof(Object));

            Debug.Log("Resources in the Resources folder:");
            // Print the names of all assets in the Resources folder
            foreach (Object resource in resources)
            {
                Debug.Log(resource.name);
            }
            Debug.Log("End of Resources");

            // Load the material from the Resources folder
            Material screen = Resources.Load<Material>("Screen");

            if (screen != null)
            {
                // Set the emission color to the render texture
                screen.SetTexture("_EmissionColor", rt);

                // Name the material 'screen_' followed by the drone number
                screen.name = "screen_" + droneNumber;
            }
            else
            {
                Debug.LogError("Material 'Screen' not found in Resources folder");
            }

            // Set the quad's material to the screen material
            quad.GetComponent<Renderer>().material = screen;
            
            
        }
        
    }

    // Update is called once per frame
    void Update()
    {
        
    }
}
