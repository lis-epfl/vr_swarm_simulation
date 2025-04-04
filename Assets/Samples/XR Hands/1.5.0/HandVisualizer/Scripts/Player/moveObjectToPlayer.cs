using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class moveObjectToPlayer : MonoBehaviour
{
    // Reference to the player GameObject that the object should follow.
    public GameObject player;

    // Start is called before the first frame update
    void Start()
    {
        // Check if the player is assigned in the Inspector.
        if (player == null)
        {
            Debug.LogError("Player is not assigned in the inspector!");

            // Attempt to find the player GameObject in the scene by name.
            player = GameObject.Find("Player");
        }
    }

    // Update is called once per frame
    void Update()
    {
        // If the player is assigned, update the object's position to follow the player.
        if (player != null)
        {
            transform.position = player.transform.position;
        }
        else
        {
            // Attempt to find the player GameObject again if it was lost.
            player = GameObject.Find("Player");
        }
    }
}