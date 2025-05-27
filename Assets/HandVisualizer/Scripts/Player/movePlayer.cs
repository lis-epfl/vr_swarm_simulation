using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class movePlayer : MonoBehaviour
{
    // Reference to the GameObject that has the Camera component for the swarm view.
    // This GameObject is expected to be named "SwarmCamera" and created by SwarmFollowCamera.cs
    private GameObject swarmCameraGameObject; 
    private SwarmFollowCamera swarmFollowCameraScript;

    private bool hasSetInitialRotation = false;

    void Start()
    {
        // Attempt to find the SwarmFollowCamera script and the Camera GameObject it manages
        swarmFollowCameraScript = FindObjectOfType<SwarmFollowCamera>();
        if (swarmFollowCameraScript == null)
        {
            Debug.LogWarning("movePlayer: SwarmFollowCamera script not found in Start. Player positioning and initial rotation might be delayed or fail.");
        }

        swarmCameraGameObject = GameObject.Find("SwarmCamera");
        if (swarmCameraGameObject == null) {
            Debug.LogWarning("movePlayer: 'SwarmCamera' GameObject not found in Start. Player positioning might be delayed or fail.");
        }
        
        // Attempt initial rotation. If dependencies aren't ready, Update will keep trying.
        SetInitialRotationTowardsSwarm();
    }

    void Update()
    {
        // Ensure we have references, try to re-acquire if lost or not found initially
        if (swarmFollowCameraScript == null) {
            swarmFollowCameraScript = FindObjectOfType<SwarmFollowCamera>();
            if (swarmFollowCameraScript == null) {
                // Still not found, can't proceed with rotation logic that depends on it.
                // Depending on setup, player position might also be affected.
            }
        }

        if (swarmCameraGameObject == null) {
            swarmCameraGameObject = GameObject.Find("SwarmCamera");
            if (swarmCameraGameObject == null && swarmFollowCameraScript != null) {
                // If script found but GO not, this is unusual as script creates the GO.
                // Could be a timing issue or naming mismatch.
            }
        }

        // If the swarmCameraGameObject is assigned, update the player's position to match it.
        if (swarmCameraGameObject != null)
        {
            transform.position = swarmCameraGameObject.transform.position;
        }
        else
        {
            // Log error less frequently if it persists
            if(Time.frameCount % 60 == 0) // Log once per second
                Debug.LogError("movePlayer: swarmCameraGameObject is null in Update. Player position not updated.");
        }
        
        // Try to set initial rotation if not done yet and script is available
        if (!hasSetInitialRotation && swarmFollowCameraScript != null)
        {
            SetInitialRotationTowardsSwarm();
        }
    }
    
    private void SetInitialRotationTowardsSwarm()
    {
        // This check is crucial as GetSwarmCentroid depends on it.
        if (swarmFollowCameraScript == null) { 
            // Attempt to find it one last time if called when null
            swarmFollowCameraScript = FindObjectOfType<SwarmFollowCamera>();
            if (swarmFollowCameraScript == null) {
                 if(Time.frameCount % 60 == 0)
                    Debug.LogWarning("movePlayer.SetInitialRotation: SwarmFollowCamera script not found. Cannot set rotation.");
                return;
            }
        }

        // GetSwarmCentroid() should be safe even if swarm list is empty (returns Vector3.zero)
        Vector3 swarmCentroid = swarmFollowCameraScript.GetSwarmCentroid();
        
        // Ensure centroid is valid (not zero, which indicates an issue or empty swarm)
        // and the player is not already at the centroid (to avoid zero direction vector)
        if (swarmCentroid != Vector3.zero)
        {
            Vector3 directionToSwarm = swarmCentroid - transform.position; 
            directionToSwarm.y = 0; // Keep player's rotation horizontal
            
            // Check if direction is significant enough to avoid issues with LookRotation
            if (directionToSwarm.sqrMagnitude > 0.001f) 
            {
                transform.rotation = Quaternion.LookRotation(directionToSwarm.normalized);
                hasSetInitialRotation = true;
                Debug.Log("Player initial rotation set towards swarm centroid.");
            }
            // else: Direction is too small, likely already at centroid or very close.
        }
        // else: Centroid is zero, swarm might not be initialized or is empty.
    }
}
