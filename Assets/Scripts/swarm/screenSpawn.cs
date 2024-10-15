using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class screenSpawn : MonoBehaviour
{

    public List<GameObject> swarm = new List<GameObject>();
    public List<GameObject> screens = new List<GameObject>();
    public GameObject arena;
    public float circleRadius = 2.0f;
    // Function to spawn screens for the drones in the swarm
    public void SpawnScreens(List<GameObject> swarm)
    {
        
        // Store the swarm list
        this.swarm = swarm;
        
        for (int i = 0; i < swarm.Count; i++)
        {
            GameObject drone = swarm[i];
            int droneNumber = i;

            // Create a screen using a quad
            GameObject screen = GameObject.CreatePrimitive(PrimitiveType.Quad);

            // Name the screen
            screen.name = "screen_" + droneNumber;

            // Add the screen to the screens list
            screens.Add(screen);

            // Create a render texture
            RenderTexture rt = new RenderTexture(256, 256, 24);

            // Name the render texture 'rt_' followed by the drone number
            rt.name = "rt_" + droneNumber;

            // Create a new Material object
            Material screenMaterial = new Material(Shader.Find("Standard"));

            // Set the color to black
            screenMaterial.color = Color.black;

            // Set the smoothness to 0
            screenMaterial.SetFloat("_Glossiness", 0f);

            // Name the material 'screenMaterial'
            screenMaterial.name = "screenMaterial" + droneNumber;

            // Set the emission map to the render texture
            screenMaterial.SetTexture("_EmissionMap", rt);

            // Set the emission color to white
            screenMaterial.SetColor("_EmissionColor", Color.white);

            // Set the emission global illumination to baked
            screenMaterial.globalIlluminationFlags = MaterialGlobalIlluminationFlags.BakedEmissive;

            // Turn on emission for the material
            screenMaterial.EnableKeyword("_EMISSION");

            // Set the screens material to the screen material
            screen.GetComponent<Renderer>().material = screenMaterial;

            // Find the camera object on the drone called 'FPV'
            Transform camera = drone.transform.Find("FPV");

            // Set the camera's target texture to the render texture
            camera.GetComponent<Camera>().targetTexture = rt;
        }

        // Place the screens based on the orientation of the drones
        UpdateScreenPositions();
    }

    // Function to update the positions and orientations of the screens
    private void UpdateScreenPositions()
    {
        for (int i = 0; i < swarm.Count; i++)
        {
            GameObject drone = swarm[i];
            GameObject screen = screens[i]; // Check this
            GameObject droneChild = drone.transform.Find("DroneParent").gameObject;

            // Calculate the screen position based on the yaw of the drone
            float yaw = droneChild.transform.eulerAngles.y; // Get the yaw angle of the drone (in degrees)
            float radians = yaw * Mathf.Deg2Rad; // Convert yaw to radians

            // Calculate the position on circle relative to the arena
            float x = arena.transform.position.x + circleRadius * Mathf.Cos(radians);
            float z = arena.transform.position.z + circleRadius * Mathf.Sin(radians);
            float y = arena.transform.position.y - 0.3f;

            // Position the quad at the calculated coordinates
            screen.transform.position = new Vector3(x, y, z);

            // if (drone.name == "Drone 1")
            // {
            //     Debug.Log("Drone 1 Yaw: " + yaw);
            //     Debug.Log("Drone 1 Radians: " + radians);
            //     Debug.Log("Drone 1 Screen Position: (" + x + ", " + y + ", " + z + ")");
            // }

            // Rotate the screen to face the center of the arena
            screen.transform.LookAt(arena.transform.position);
            screen.transform.Rotate(0, 180f, 0); // Adjust rotation to face outward
        }
    }

    // Update the positions of the screens based on the drone orientation
    void Update()
    {
        UpdateScreenPositions();

    }
}
