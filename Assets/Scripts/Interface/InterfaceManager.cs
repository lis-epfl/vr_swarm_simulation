using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class interfaceManager : MonoBehaviour
{
    
    public enum DisplayMode
    {
        SCREENS,
        SINGLE,
        BIRDSEYE
    }
    public DisplayMode displayMode = DisplayMode.SCREENS;
    
    private ViewManager viewManager;
    private ScreenSpawn spawnScreens;
    private BirdsEyeCamera birdsEyeCamera;
    private visualiseOlfatiSaber visualiseOlfatiSaber;

    private List<GameObject> swarm;
    private bool screensSpawned = false;
    
    // Start is called before the first frame update
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

                 
    }

    // Update is called once per frame
    void Update()
    {
        // If screens are not spawned and display mode is SCREENS, spawn screens
        if (displayMode == DisplayMode.SCREENS && !screensSpawned)
        {
            spawnScreens.SpawnScreens(swarm);
            screensSpawned = true;
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