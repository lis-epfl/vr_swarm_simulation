using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class sceneSetup : MonoBehaviour
{
    
    private ScreenSpawn ScreenSpawn;
    
    // Start is called before the first frame update
    void Start()
    {
        // Get the ScreenSpawn script
        ScreenSpawn = GetComponent<ScreenSpawn>();

        // Call the SpawnScreens function from the ScreenSpawn script]
        ScreenSpawn.SpawnScreens();
        
    }

    // Update is called once per frame
    void Update()
    {
        
    }
}
