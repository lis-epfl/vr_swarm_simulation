using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class screenSpawn : MonoBehaviour
{

    public List<GameObject> swarm = new List<GameObject>();
    public List<GameObject> screens = new List<GameObject>();
    public GameObject arena;
    public GameObject screenParent;
    public float circleRadius = 2.0f;
    public int width = 640;
    public int height = 360;


    // Function to spawn screens for the drones in the swarm
    public void SpawnScreens(List<GameObject> swarm)
    {
        
        // Store the swarm list
        this.swarm = swarm;

        // Create an empty GameObject to serve as the parent for all screens
        screenParent = new GameObject("ScreenParent");
        
        for (int i = 0; i < swarm.Count; i++)
        {
            GameObject drone = swarm[i];
            int droneNumber = i;

            // Create a screen using a quad
            GameObject screen = GameObject.CreatePrimitive(PrimitiveType.Quad);

            // Name the screen
            screen.name = "screen_" + droneNumber;

            // Parent the screen under the screenParent GameObject
            screen.transform.parent = screenParent.transform;

            // Add the screen to the screens list
            screens.Add(screen);

            // Create a render texture
            RenderTexture rt = new RenderTexture(width, height, 24);

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

            // set the scale to match the aspect ratio of the feed
            screen.transform.localScale = new Vector3((float)width/height, 1f, 1f);

            // Find the camera object on the drone called 'FPV'
            Transform camera = drone.transform.Find("FPV");

            // Get the camera and set the aspect ratio and field of view
            Camera cam = camera.GetComponent<Camera>();
            cam.aspect = (float)width/height;
            cam.fieldOfView = 82.1f;

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
            // Find the drone and screen by name, seems like the order in swarm changes, hence the need to find the drone by name
            GameObject drone = swarm.Find(d => d.name == "Drone " + i);
            GameObject droneChild = drone.transform.Find("DroneParent").gameObject;
            GameObject screen = screens.Find(s => s.name == "screen_" + i);

            // Check the boundaryEstimate of the drone
            AttitudeControl attitudeControl = droneChild.GetComponent<AttitudeControl>();
            if (attitudeControl.boundaryEstimate)
            {
                // Calculate the screen position based on the yaw of the drone
                float yaw = droneChild.transform.eulerAngles.y; // Get the yaw angle of the drone (in degrees)
                float radians = -yaw * Mathf.Deg2Rad; // Convert yaw to radians

                // Calculate the position on circle relative to the arena
                float x = arena.transform.position.x + circleRadius * Mathf.Cos(radians);
                float z = arena.transform.position.z + circleRadius * Mathf.Sin(radians);
                float y = arena.transform.position.y - 0.3f;

                // Position the quad at the calculated coordinates
                screen.transform.position = new Vector3(x, y, z);

                // Rotate the screen to face the center of the arena
                screen.transform.LookAt(arena.transform.position);
                screen.transform.Rotate(0, 180f, 0); // Adjust rotation to face outward

                // Ensure the screen is active
                screen.SetActive(true);
            }
            else
            {                
               // Hide the screen if the drone is not on the swarm boundary
                screen.SetActive(false);
            }
            
        }
    }

    // Update the positions of the screens based on the drone orientation
    void Update()
    {
        UpdateScreenPositions();

    }
}
