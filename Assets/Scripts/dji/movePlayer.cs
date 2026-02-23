using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class movePlayer : MonoBehaviour
{
    public OVRPlayerController playerController;
    
    public GameObject Arena;
    public GameObject Arena2;
    
    public enum ArenaLocation
    {
        Arena1,
        Arena2
    }
    public ArenaLocation currentArena = ArenaLocation.Arena1;
    
    public float arena1Direction = 0f;
    public float arena2Direction = 0f;
    
    private ArenaLocation previousArena;
    private Vector3 arenaDisplacement;

    void Start()
    {
        if (Arena != null && Arena2 != null)
        {
            // Calculate the displacement vector between arena centers
            arenaDisplacement = Arena2.transform.position - Arena.transform.position;
        }
        else
        {
            Debug.LogWarning("Arena or Arena2 is not assigned!");
        }
        
        // Initialize previous arena state
        previousArena = currentArena;
    }

    void Update()
    {
        // Check if the arena selection has changed
        if (currentArena != previousArena)
        {
            TeleportPlayer();
            previousArena = currentArena;
        }
    }
    
    void TeleportPlayer()
    {
        if (playerController == null)
        {
            Debug.LogWarning("OVRPlayerController is not assigned!");
            return;
        }
        
        if (Arena == null || Arena2 == null)
        {
            Debug.LogWarning("Arena references are missing!");
            return;
        }
        
        // Determine teleport direction and rotation based on current arena
        if (currentArena == ArenaLocation.Arena2)
        {
            // Move player by the displacement to Arena2
            playerController.transform.position += arenaDisplacement;
            SetPlayerRotation(arena2Direction);
        }
        else // currentArena == ArenaLocation.Arena1
        {
            // Move player back by the displacement to Arena1
            playerController.transform.position -= arenaDisplacement;
            SetPlayerRotation(arena1Direction);
        }
    }
    
    void SetPlayerRotation(float yRotation)
    {
        // Set the player's Y-axis rotation
        Vector3 currentRotation = playerController.transform.eulerAngles;
        playerController.transform.eulerAngles = new Vector3(currentRotation.x, yRotation, currentRotation.z);
    }
}