using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class movePlayer : MonoBehaviour
{
    // Reference to the camera GameObject that the player should follow.
    public GameObject swarmFollowCamera;

    // Start is called before the first frame update
    void Start()
    {
        // Check if the swarmFollowCamera is assigned in the Inspector.
        if(swarmFollowCamera == null)
        {
            Debug.LogError("swarmFollowCamera is not assigned in the inspector!");

            // Get the swarmFollowCamera from the scene
            swarmFollowCamera = GameObject.Find("SwarmCamera");
        }
    }

    // Update is called once per frame
    void Update()
    {
        // If the swarmFollowCamera is assigned, update the player's position.
        if (swarmFollowCamera != null)
        {
            transform.position = swarmFollowCamera.transform.position;
        }
        else
        {
            swarmFollowCamera = GameObject.Find("SwarmCamera");
        }
    }
}

