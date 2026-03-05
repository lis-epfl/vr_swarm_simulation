using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class ScreenSpawn : MonoBehaviour
{
    public enum ScreenStyle
    {
        OFF,
        OUTER_CIRCLE,
        INNER_CIRCLE,
        BOTTOM_CIRCLE,
        ROTATING_CIRCLE,
        REAL_DRONE
    }

    [Header("Display Settings")]
    [HideInInspector] public ScreenStyle screenStyle = ScreenStyle.OFF;

    [Header("VR Parameters")]
    [HideInInspector] public int width = 640;
    [HideInInspector] public int height = 360;

    [Header("Active Display Parameters")]
    [HideInInspector] public float radius = 2.0f;
    [HideInInspector] public float scale = 1.0f;
    [HideInInspector] public Vector3 offset = new Vector3(0.5f, -0.3f, 0.0f);
    [HideInInspector] public Vector3 lookAtOffset = new Vector3(0.0f, 0.0f, 0.0f);

    [Header("Special Settings")]
    [HideInInspector] public bool invertBottomScreen = false;
    [HideInInspector] public bool doubleView = false;
    [HideInInspector] public float rotatingCircleDistance = 2.0f;
    [HideInInspector] public int numScreens = 2;

    // GameObject references
    private OVRPlayerController player;
    private List<GameObject> swarm = new List<GameObject>();
    private List<GameObject> screens = new List<GameObject>();
    private GameObject arena;
    private GameObject screenParent;

    // Default parameters for each display mode
    private float outerCircleRadius = 2.0f;
    private float outerCircleScale = 1.0f;
    private Vector3 outerCircleOffset = new Vector3(0.0f, 0.0f, 0.0f);
    private Vector3 outerCircleLookAtOffset = new Vector3(0.0f, 0.0f, 0.0f);

    private float innerCircleRadius = 0.45f;
    private float innerCircleScale = 0.2f;
    private Vector3 innerCircleOffset = new Vector3(0.0f, -0.3f, 0.0f);
    private Vector3 innerCircleLookAtOffset = new Vector3(0.5f, -0.3f, 0.0f);

    private float bottomCircleRadius = 0.6f;
    private float bottomCircleScale = 0.25f;
    private Vector3 bottomCircleOffset = new Vector3(0.0f, -0.5f, 0.0f);
    private Vector3 bottomCircleLookAtOffset = new Vector3(0.0f, -0.5f, 0.0f);

    private float rotatingCircleRadius = 0.2f;
    private float rotatingCircleScale = 0.1f;
    private Vector3 rotatingCircleOffset = new Vector3(0.0f, -0.3f, 0.0f);
    private Vector3 rotatingCircleLookAtOffset = new Vector3(0.0f, 0.0f, 0.0f);

    private float realDroneRadius = 2.0f;
    private float realDroneScale = 1.0f;
    private Vector3 realDroneOffset = new Vector3(0.0f, 0.0f, 0.0f);
    private Vector3 realDroneLookAtOffset = new Vector3(0.0f, 0.0f, 0.0f);

    private SwarmManager swarmManager;
    private bool pointInwards = false;
    private ScreenStyle previousScreenStyle;
    private InterfaceManager interfaceManager;

    public bool IsSpawned => screens.Count > 0;

    // Function to spawn screens for the drones in the swarm
    public void SpawnScreens(List<GameObject> swarm = null)
    {
        // Find the OVRPlayerController in the scene if not already assigned
        if (player == null)
        {
            player = GameObject.FindGameObjectWithTag("Player")?.GetComponent<OVRPlayerController>();
            if (player == null)
            {
                Debug.LogWarning("No OVRPlayerController found in the scene!");
            }
        }

        // Find the arena in the scene
        if (arena == null)
        {
            arena = GameObject.FindGameObjectWithTag("Arena");
            if (arena == null)
            {
                Debug.LogWarning("No GameObject with tag 'Arena' found in the scene!");
            }
        } 
    
        // Get the swarm manager instance and add the event listener
        if (swarm != null)
        {
            // Get the swarm manager instance
            swarmManager = SwarmManager.Instance;

            // Add the swarmParamsChanged event listener
            swarmManager.swarmParamsChanged += OnSwarmParamsChanged;
        }

        // Store the swarm list
        this.swarm = swarm;

        // Create an empty GameObject to serve as the parent for all screens
        screenParent = new GameObject("ScreenParent");

        // Determine how many screens to create
        int count = (swarm != null) ? swarm.Count : numScreens;

        for (int i = 0; i < count; i++)
        {
            int droneNumber = i;

            // Create a screen using a quad
            GameObject screen = GameObject.CreatePrimitive(PrimitiveType.Quad);

            // Name the screen
            screen.name = "screen_" + droneNumber;

            // Parent the screen under the screenParent GameObject
            screen.transform.parent = screenParent.transform;

            // Set the tag of the screen to 'Screen'
            screen.tag = "Screen";

            // Add the screen to the screens list
            screens.Add(screen);

            // Create a render texture
            RenderTexture rt = new RenderTexture(width, height, 24);

            // Name the render texture 'rt_' followed by the drone number
            rt.name = "rt_" + droneNumber;

            // Create a new Material object
            Material screenMaterial = new Material(Shader.Find("Standard"));

            // Set the color to white
            screenMaterial.color = Color.white;

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

            if (swarm != null && i < swarm.Count)
            {
                // Find the camera object on the drone called 'FPV'
                GameObject drone = swarm[i];
                Transform camera = drone.transform.Find("FPV");

                // Get the camera and set the aspect ratio and field of view
                Camera cam = camera.GetComponent<Camera>();
                cam.aspect = (float)width / height;
                cam.fieldOfView = 82.1f;

                // Set the camera's target texture to the render texture
                if (screenStyle != ScreenStyle.OFF || screenStyle != ScreenStyle.REAL_DRONE)
                {
                    cam.GetComponent<Camera>().targetTexture = rt;
                }
            }
        }

        // Place the screens based on the orientation of the drones
        UpdateScreenPositions();

        // Move the player to the centre of the arena
        if (player != null && arena != null)
        {
            player.transform.position = arena.transform.position;
        }
    }

    // Update display parameters when screen style changes
    private void UpdateDisplayParameters()
    {
        switch (screenStyle)
        {
            case ScreenStyle.OUTER_CIRCLE:
                radius = outerCircleRadius;
                scale = outerCircleScale;
                offset = outerCircleOffset;
                lookAtOffset = outerCircleLookAtOffset;
                break;
            case ScreenStyle.INNER_CIRCLE:
                radius = innerCircleRadius;
                scale = innerCircleScale;
                offset = innerCircleOffset;
                lookAtOffset = innerCircleLookAtOffset;
                break;
            case ScreenStyle.BOTTOM_CIRCLE:
                radius = bottomCircleRadius;
                scale = bottomCircleScale;
                offset = bottomCircleOffset;
                lookAtOffset = bottomCircleLookAtOffset;
                break;
            case ScreenStyle.ROTATING_CIRCLE:
                radius = rotatingCircleRadius;
                scale = rotatingCircleScale;
                offset = rotatingCircleOffset;
                lookAtOffset = rotatingCircleLookAtOffset;
                break;
            case ScreenStyle.REAL_DRONE:
                radius = realDroneRadius;
                scale = realDroneScale;
                offset = realDroneOffset;
                lookAtOffset = realDroneLookAtOffset;
                break;
        }
    }

    // Get default parameters for a given screen style and send them to InterfaceManager
    public void SendDefaultParametersToInterfaceManager(ScreenStyle style)
    {
        // Get reference to InterfaceManager if not already set
        if (interfaceManager == null)
        {
            interfaceManager = GetComponent<InterfaceManager>();
        }

        if (interfaceManager == null)
        {
            Debug.LogWarning("InterfaceManager not found on this GameObject!");
            return;
        }

        float defaultRadius = 2.0f;
        float defaultScale = 1.0f;
        Vector3 defaultOffset = Vector3.zero;
        Vector3 defaultLookAtOffset = Vector3.zero;

        switch (style)
        {
            case ScreenStyle.OUTER_CIRCLE:
                defaultRadius = outerCircleRadius;
                defaultScale = outerCircleScale;
                defaultOffset = outerCircleOffset;
                defaultLookAtOffset = outerCircleLookAtOffset;
                break;
            case ScreenStyle.INNER_CIRCLE:
                defaultRadius = innerCircleRadius;
                defaultScale = innerCircleScale;
                defaultOffset = innerCircleOffset;
                defaultLookAtOffset = innerCircleLookAtOffset;
                break;
            case ScreenStyle.BOTTOM_CIRCLE:
                defaultRadius = bottomCircleRadius;
                defaultScale = bottomCircleScale;
                defaultOffset = bottomCircleOffset;
                defaultLookAtOffset = bottomCircleLookAtOffset;
                break;
            case ScreenStyle.ROTATING_CIRCLE:
                defaultRadius = rotatingCircleRadius;
                defaultScale = rotatingCircleScale;
                defaultOffset = rotatingCircleOffset;
                defaultLookAtOffset = rotatingCircleLookAtOffset;
                break;
            case ScreenStyle.REAL_DRONE:
                defaultRadius = realDroneRadius;
                defaultScale = realDroneScale;
                defaultOffset = realDroneOffset;
                defaultLookAtOffset = realDroneLookAtOffset;
                break;
        }

        // Call InterfaceManager to update its display parameters
        interfaceManager.UpdateDisplayParameters(defaultRadius, defaultScale, defaultOffset, defaultLookAtOffset);
    }

    // Update the position of the screens based on the drone orientation
    void UpdateScreenPositions()
    {
        if (swarm == null || swarm.Count == 0 || screens.Count == 0)
        {
            return;
        }

        for (int i = 0; i < swarm.Count; i++)
        {
            GameObject drone = swarm.Find(d => d.name == "Drone " + i);
            GameObject droneChild = drone.transform.Find("DroneParent").gameObject;
            GameObject screen = screens.Find(s => s.name == "screen_" + i);

            switch (screenStyle)
            {
                case ScreenStyle.OFF:
                    HideScreen(screen);
                    break;
                case ScreenStyle.OUTER_CIRCLE:
                    UpdateOuterCircleScreen(screen, droneChild);
                    break;
                case ScreenStyle.INNER_CIRCLE:
                    UpdateInnerCircleScreen(screen, droneChild);
                    break;
                case ScreenStyle.BOTTOM_CIRCLE:
                    UpdateBottomCircleScreen(screen, droneChild);
                    break;
                case ScreenStyle.ROTATING_CIRCLE:
                    UpdateRotatingCircleScreen(screen, droneChild);
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
        AttitudeAlgorithm attitudeControl = droneChild.GetComponent<AttitudeAlgorithm>();
        if (!attitudeControl.BoundaryEstimate)
        {
            screen.SetActive(false);
            return;
        }

        // Get the drone's yaw
        StateFinder stateFinder = droneChild.GetComponent<VelocityControl>().State;
        float yaw = stateFinder.Angles.y;
        float radians = -yaw * Mathf.Deg2Rad;

        // Calculate the position on outer circle
        float x = arena.transform.position.x + radius * Mathf.Cos(radians);
        float z = arena.transform.position.z + radius * Mathf.Sin(radians);
        float y = arena.transform.position.y + offset.y;

        // Position and rotate the screen
        screen.transform.position = new Vector3(x, y, z);
        screen.transform.LookAt(arena.transform.position + lookAtOffset);
        screen.transform.Rotate(0, 180f, 0); // Face outward
        screen.SetActive(true);
    }

    private void UpdateInnerCircleScreen(GameObject screen, GameObject droneChild)
    {

        // Get the drone's yaw
        StateFinder stateFinder = droneChild.GetComponent<VelocityControl>().State;
        float yaw = stateFinder.Angles.y;
        float radians = -yaw * Mathf.Deg2Rad;

        // Calculate the position on inner circle
        float x = arena.transform.position.x + radius * Mathf.Cos(radians) + offset.x;
        float z = arena.transform.position.z + radius * Mathf.Sin(radians);
        float y = arena.transform.position.y + offset.y;

        // Position and rotate the screen
        screen.transform.position = new Vector3(x, y, z);
        screen.transform.LookAt(arena.transform.position + lookAtOffset);
        screen.SetActive(true);
    }

    // Update the bottom circle screen positions
    private void UpdateBottomCircleScreen(GameObject screen, GameObject droneChild)
    {

        // Get the drone's yaw
        StateFinder stateFinder = droneChild.GetComponent<VelocityControl>().State;
        float yaw = stateFinder.Angles.y;
        float radians = -yaw * Mathf.Deg2Rad;

        // Calculate the position on bottom circle
        float x = arena.transform.position.x + radius * Mathf.Cos(radians) + offset.x;
        float z = arena.transform.position.z + radius * Mathf.Sin(radians) + offset.z;
        float y = arena.transform.position.y + offset.y;

        // Position and rotate the screen
        screen.transform.position = new Vector3(x, y, z);
        screen.transform.LookAt(arena.transform.position + lookAtOffset + offset);
        if (invertBottomScreen)
        {
            screen.transform.Rotate(0, 180f, 0); // Invert the screen
        }

        // If double view is true then find the screens where the relative position is in the positive x direction and reverse the direction
        if (doubleView)
        {
            Vector3 relativePos = screen.transform.position - (arena.transform.position + offset);
            if (relativePos.x > 0)
            {
                screen.transform.Rotate(0, 180f, 0); // Reverse the direction
            }
        }

        screen.SetActive(true);
    }

    private void UpdateRotatingCircleScreen(GameObject screen, GameObject droneChild)
    {
        if (player == null)
        {
            screen.SetActive(false);
            return;
        }

        // Get the drone's yaw
        StateFinder stateFinder = droneChild.GetComponent<VelocityControl>().State;
        float yaw = stateFinder.Angles.y;
        float radians = -yaw * Mathf.Deg2Rad;

        // Calculate base position on inner circle
        float x = arena.transform.position.x + radius * Mathf.Cos(radians);
        float z = arena.transform.position.z + radius * Mathf.Sin(radians);
        float y = arena.transform.position.y + offset.y;
        Vector3 basePosition = new Vector3(x, y, z);

        // Get player's forward direction (only using horizontal direction)
        Vector3 playerForward = player.transform.forward;
        playerForward.y = 0; // Zero out vertical component
        playerForward.Normalize();

        // Offset the screen position in the player's forward direction
        Vector3 offsetPosition = basePosition + playerForward * rotatingCircleDistance;

        // Position and rotate the screen
        screen.transform.position = offsetPosition;
        screen.transform.LookAt(arena.transform.position + playerForward * rotatingCircleDistance + lookAtOffset);
        screen.SetActive(true);
    }

    // Update the position of the screens based on real drone orientation, called from ImageSharing.cs
    public void UpdateRealDroneScreen(int i, float yaw)
    {
        // Find the screen
        GameObject screen = screens.Find(s => s.name == "screen_" + i);

        // Calculate the screen position based on the yaw of the real drone
        float radians = -yaw * Mathf.Deg2Rad;

        // Calculate the position on real drone circle
        float x = arena.transform.position.x + radius * Mathf.Cos(radians);
        float z = arena.transform.position.z + radius * Mathf.Sin(radians);
        float y = arena.transform.position.y;

        // Position and rotate the screen
        screen.transform.position = new Vector3(x, y, z);
        screen.transform.LookAt(arena.transform.position);
        screen.transform.Rotate(0, 180f, 0); // Face outward
        screen.SetActive(true);
    }

    // Update the positions of the screens based on the drone orientation
    void Update()
    {
        UpdateScreenPositions();
    }

    void OnValidate()
    {
        // Only update display parameters if the screen style changed
        if (screenStyle != previousScreenStyle)
        {
            previousScreenStyle = screenStyle;
            UpdateDisplayParameters();
        }
        UpdateScreenScale();
    }

    // Called when InterfaceManager parameters change
    public void OnInterfaceParamsChanged()
    {
        if (screenStyle != previousScreenStyle)
        {
            previousScreenStyle = screenStyle;
            UpdateDisplayParameters();
            
            // Send default parameters back to InterfaceManager
            SendDefaultParametersToInterfaceManager(screenStyle);
        }
        
        // Update screen scale when parameters change
        UpdateScreenScale();
        
        // Update screen positions if screens are already spawned
        UpdateScreenPositions();
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
        for (int i = 0; i < screens.Count; i++)
        {
            GameObject screen = screens[i];
            screen.transform.localScale = new Vector3((float)width / height, 1f, 1f) * scale;
        }
    }

    // Get the scale based on screen style
    public float GetScreenScale()
    {
        return scale;
    }
}