using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class sceneSetup : MonoBehaviour
{
    
    private screenSpawn screenSpawn;
    
    // Start is called before the first frame update
    void Start()
    {
        // Get the screenSpawn script
        screenSpawn = GetComponent<screenSpawn>();

        // Call the SpawnScreens function from the screenSpawn script]
        screenSpawn.SpawnScreens();
        
    }

    // Update is called once per frame
    void Update()
    {
        
    }
}
