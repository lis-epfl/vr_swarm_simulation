using System.Collections;
using System.Collections.Generic;
using System.Diagnostics;
// using System.Numerics;
using UnityEngine;

public class screenSpawn : MonoBehaviour
{

    public enum DisplayMode
    {
        OFF,
        OUTER_CIRCLE,
        INNER_CIRCLE
    }
    [Header("Display Settings")]
    public DisplayMode displayMode = DisplayMode.OFF;

    [Header("Screen Parameters")]
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

            switch (displayMode)
            {
                case DisplayMode.OFF:
                    HideScreen(screen);
                    break;
                case DisplayMode.OUTER_CIRCLE:
                    UpdateOuterCircleScreen(screen, droneChild);
                    break;
                case DisplayMode.INNER_CIRCLE:
                    UpdateInnerCircleScreen(screen, droneChild);
                    break;
            }
        }
    }

    private void HideScreen(GameObject screen)
    {
        screen.SetActive(false);
    }

    private void UpdateOuterCircleScreen(GameObject screen, GameObject droneChild)
    {
        // Check if the drone is on the boundary
        AttitudeControl attitudeControl = droneChild.GetComponent<AttitudeControl>();
        if (!attitudeControl.boundaryEstimate)
        {
            screen.SetActive(false);
            return;
        }

        // Get the drone's yaw
        float yaw = droneChild.transform.eulerAngles.y;
        float radians = -yaw * Mathf.Deg2Rad;

        // Calculate the position on outer circle
        float x = arena.transform.position.x + outerRadius * Mathf.Cos(radians);
        float z = arena.transform.position.z + outerRadius * Mathf.Sin(radians);
        float y = arena.transform.position.y + offset.y;

        // Position and rotate the screen
        screen.transform.position = new Vector3(x, y, z);
        screen.transform.LookAt(arena.transform.position);
        screen.transform.Rotate(0, 180f, 0); // Face outward
        screen.SetActive(true);
    }

    private void UpdateInnerCircleScreen(GameObject screen, GameObject droneChild)
    {
        float yaw = droneChild.transform.eulerAngles.y;
        float radians = -yaw * Mathf.Deg2Rad;

        // Calculate the position on inner circle
        float x = arena.transform.position.x + innerRadius * Mathf.Cos(radians) + offset.x;
        float z = arena.transform.position.z + innerRadius * Mathf.Sin(radians);
        float y = arena.transform.position.y + offset.y;

        // Position and rotate the screen
        screen.transform.position = new Vector3(x, y, z);
        screen.transform.LookAt(arena.transform.position + offset);
        screen.SetActive(true);
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
        // Update the screen scales when swarm parameters change
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
    
    // Get the scale based on display mode
    public float GetScreenScale()
    {
        switch (displayMode)
        {
            case DisplayMode.INNER_CIRCLE:
                return innerScale;
            case DisplayMode.OUTER_CIRCLE:
                return outerScale;
            default:
                return 1.0f;
        }
    }

}
