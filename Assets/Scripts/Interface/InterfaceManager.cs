using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class InterfaceManager : MonoBehaviour
{
    
    public enum DisplayMode
    {
        SCREENS,
        SINGLE,
        BIRDSEYE
    }
    
    [Header("Interface Display Mode")]
    public DisplayMode displayMode = DisplayMode.SCREENS;
    
    [Header("Screen Display Settings")]
    public ScreenSpawn.ScreenStyle screenStyle = ScreenSpawn.ScreenStyle.OFF;
    
    [Header("Screen Parameters")]
    public int width = 640;
    public int height = 360;
    
    [Header("Active Display Parameters")]
    public float radius = 2.0f;
    public float scale = 1.0f;
    public Vector3 offset = new Vector3(0.5f, -0.3f, 0.0f);
    public Vector3 lookAtOffset = new Vector3(0.0f, 0.0f, 0.0f);
    
    [Header("Special Settings")]
    public bool invertBottomScreen = false;
    public bool doubleView = false;
    public float rotatingCircleDistance = 2.0f;
    
    public delegate void OnInterfaceParamsChanged();
    public event OnInterfaceParamsChanged interfaceParamsChanged;
    
    private ViewManager viewManager;
    private ScreenSpawn spawnScreens;
    private BirdsEyeCamera birdsEyeCamera;
    private visualiseOlfatiSaber visualiseOlfatiSaber;

    private List<GameObject> swarm;
    private bool screensSpawned = false;
    
    void Start()
    {
        viewManager = GetComponent<ViewManager>();
        spawnScreens = GetComponent<ScreenSpawn>();

        // Get the BirdsEyeCamera
        GameObject birdsEyeCameraObject = GameObject.FindGameObjectWithTag("BirdsEyeCamera");
        if (birdsEyeCameraObject != null)
        {
            birdsEyeCamera = birdsEyeCameraObject.GetComponent<BirdsEyeCamera>();
        }

        visualiseOlfatiSaber = GetComponent<visualiseOlfatiSaber>();
        
        // Subscribe ScreenSpawn to parameter changes
        if (spawnScreens != null)
        {
            interfaceParamsChanged += spawnScreens.OnInterfaceParamsChanged;
        }
        
        // Initialize ScreenSpawn parameters
        UpdateScreenSpawnParameters();
    }

    void Update()
    {
        // If screens are not spawned and display mode is SCREENS, spawn screens
        if (displayMode == DisplayMode.SCREENS && !screensSpawned)
        {
            spawnScreens.SpawnScreens(swarm);
            screensSpawned = true;
        }
    }

    // Called whenever a value is changed in the Inspector
    public void OnValidate()
    {
        // Update ScreenSpawn parameters immediately
        UpdateScreenSpawnParameters();
        
        // Trigger the event to notify all subscribed scripts
        interfaceParamsChanged?.Invoke();
    }

    // Called by ScreenSpawn to update display parameters with defaults
    public void UpdateDisplayParameters(float newRadius, float newScale, Vector3 newOffset, Vector3 newLookAtOffset)
    {
        radius = newRadius;
        scale = newScale;
        offset = newOffset;
        lookAtOffset = newLookAtOffset;
    }

    // Update ScreenSpawn with parameters from InterfaceManager
    private void UpdateScreenSpawnParameters()
    {
        if (spawnScreens != null)
        {
            spawnScreens.screenStyle = screenStyle;
            spawnScreens.width = width;
            spawnScreens.height = height;
            spawnScreens.radius = radius;
            spawnScreens.scale = scale;
            spawnScreens.offset = offset;
            spawnScreens.lookAtOffset = lookAtOffset;
            spawnScreens.invertBottomScreen = invertBottomScreen;
            spawnScreens.doubleView = doubleView;
            spawnScreens.rotatingCircleDistance = rotatingCircleDistance;
        }
    }

    // Assign the swarm list to this script and other relevant scripts
    public void SetSwarm(List<GameObject> swarmList)
    {
        swarm = swarmList;

        if (viewManager != null)
        {
            viewManager.swarm = swarmList;
        }
        if (visualiseOlfatiSaber != null)
        {
            visualiseOlfatiSaber.swarm = swarmList;
        }
        if (birdsEyeCamera != null)
        {
            birdsEyeCamera.swarm = swarmList;
        }
        if (displayMode == DisplayMode.SCREENS && !screensSpawned)
        {
            if (spawnScreens != null)
            {
                spawnScreens.SpawnScreens(swarm);
                screensSpawned = true;
            }
        }
    }
}