using System.Collections;
using System.Collections.Generic;
using System.Diagnostics;
// using System.Numerics;
using UnityEngine;

public class screenSpawn : MonoBehaviour
{

    public List<GameObject> swarm = new List<GameObject>();
    public List<GameObject> screens = new List<GameObject>();
    public GameObject arena;
    public GameObject screenParent;
    public int width = 640;
    public int height = 360;
    public float outerRadius = 2.0f;
    public float outerScale = 1.0f;
    public float innerRadius = 0.2f;
    public float innerScale = 0.1f;
    public Vector3 offset = new Vector3(0.5f, -0.3f, 0.0f);
    private SwarmManager swarmManager;
    private bool pointInwards = false;
    private bool displayScreens = false;


    // Function to spawn screens for the drones in the swarm
    public void SpawnScreens(List<GameObject> swarm)
    {
        // Get the swarm manager instance
        swarmManager = SwarmManager.Instance;

        // Add the swarmParamsChanged event listener
        swarmManager.swarmParamsChanged += OnSwarmParamsChanged;

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
            screen.transform.localScale = new Vector3((float)width / height, 1f, 1f);

            // Find the camera object on the drone called 'FPV'
            Transform camera = drone.transform.Find("FPV");

            // Get the camera and set the aspect ratio and field of view
            Camera cam = camera.GetComponent<Camera>();
            cam.aspect = (float)width / height;
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


            if (!displayScreens)
            {
                // Hide all screens if displayScreens is false
                screen.SetActive(false);
                continue;
            }

            // Check which swarm algorithm is being used
            SwarmManager.SwarmAlgorithm swarmAlgorithm = swarmManager.swarmAlgorithm;

            // If Reynolds or Olfati-saber check the boundaryEstimate to position screens
            if (swarmAlgorithm == SwarmManager.SwarmAlgorithm.REYNOLDS || swarmAlgorithm == SwarmManager.SwarmAlgorithm.OLFATI_SABER)
            {
                // Check the boundaryEstimate of the drone
                AttitudeControl attitudeControl = droneChild.GetComponent<AttitudeControl>();
                if (attitudeControl.boundaryEstimate)
                {
                    // Calculate the screen position based on the yaw of the drone
                    float yaw = droneChild.transform.eulerAngles.y; // Get the yaw angle of the drone (in degrees)
                    float radians = -yaw * Mathf.Deg2Rad; // Convert yaw to radians

                    float x, z, y;

                    if (!pointInwards)
                    {
                        // Calculate the position on circle relative to the arena
                        x = arena.transform.position.x + outerRadius * Mathf.Cos(radians);
                        z = arena.transform.position.z + outerRadius * Mathf.Sin(radians);
                        y = arena.transform.position.y + offset.y;

                        // Position the quad at the calculated coordinates
                        screen.transform.position = new Vector3(x, y, z);

                        // Rotate the screen to face the center of the arena
                        screen.transform.LookAt(arena.transform.position);
                        screen.transform.Rotate(0, 180f, 0); // Adjust rotation to face outward
                    }
                    else
                    {
                        // Calculate the position on inner circle relative to the arena
                        x = arena.transform.position.x + innerRadius * Mathf.Cos(radians) + offset.x;
                        z = arena.transform.position.z + innerRadius * Mathf.Sin(radians);
                        y = arena.transform.position.y + offset.y;

                        // Position the quad at the calculated coordinates
                        screen.transform.position = new Vector3(x, y, z);

                        // Rotate the screen to face radially outward from the center of the arena
                        screen.transform.LookAt(arena.transform.position + offset);
                    }

                    // Ensure the screen is active
                    screen.SetActive(true);
                }
                else
                {
                    // Hide the screen if the drone is not on the swarm boundary
                    screen.SetActive(false);
                }
            }
            // If NBV is selected then set all screens to active and use the inner circle positioning
            else if (swarmAlgorithm == SwarmManager.SwarmAlgorithm.NBV)
            {
                // Calculate the position on inner circle relative to the arena
                float yaw = droneChild.transform.eulerAngles.y; // Get the yaw angle of the drone (in degrees)
                float radians = -yaw * Mathf.Deg2Rad; // Convert yaw to radians

                float x = arena.transform.position.x + innerRadius * Mathf.Cos(radians) + offset.x;
                float z = arena.transform.position.z + innerRadius * Mathf.Sin(radians);
                float y = arena.transform.position.y + offset.y;

                // Position the quad at the calculated coordinates
                screen.transform.position = new Vector3(x, y, z);

                // Rotate the screen to face radially outward from the center of the arena
                screen.transform.LookAt(arena.transform.position + offset);

                // Ensure the screen is active
                screen.SetActive(true);
            }
        }
    }

    // Update the positions of the screens based on the drone orientation
    void Update()
    {
        UpdateScreenPositions();

    }

    void OnValidate()
    {
        UpdateScreenScale();
    }
    
    // Update the parameters from the swarm manager
    void OnSwarmParamsChanged()
    {

        // Check if the displayScreens parameter has changed
        displayScreens = swarmManager.GetDisplayScreens();

        // Check if screens should be facing inwards
        pointInwards = swarmManager.GetPointInwards();

        // Update the screen scales
        UpdateScreenScale();

    }

    // Update the scale of each screen with a factor
    public void UpdateScreenScale()
    {
        // Get the screen scale
        float screenScale = GetScreenScale();

        for (int i = 0; i < screens.Count; i++)
        {
            GameObject screen = screens[i];
            screen.transform.localScale = new Vector3((float)width / height, 1f, 1f) * screenScale;
        }
    }
    
    // Get the scale of the screens by checking the swarm algorithm
    public float GetScreenScale()
    {
        SwarmManager.SwarmAlgorithm swarmAlgorithm = swarmManager.swarmAlgorithm;
        if (swarmAlgorithm == SwarmManager.SwarmAlgorithm.NBV)
        {
            return innerScale;
        }
        else
        {
            return pointInwards ? innerScale : outerScale;
        }
    }

}
