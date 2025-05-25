using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class movePlayer : MonoBehaviour
{
    // Reference to the camera GameObject that the player should follow.
    public GameObject swarmFollowCamera;
    
    private bool hasSetInitialRotation = false;

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
        
        // Set initial rotation towards swarm center
        SetInitialRotationTowardsSwarm();
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
        
        // Try to set initial rotation if not done yet
        if (!hasSetInitialRotation)
        {
            SetInitialRotationTowardsSwarm();
        }
    }
    
    private void SetInitialRotationTowardsSwarm()
    {
        SwarmFollowCamera swarmCameraScript = FindObjectOfType<SwarmFollowCamera>();
        if (swarmCameraScript != null)
        {
            Vector3 swarmCentroid = swarmCameraScript.GetSwarmCentroid();
            
            if (swarmCentroid != Vector3.zero)
            {
                Vector3 directionToSwarm = swarmCentroid - transform.position;
                directionToSwarm.y = 0; // Only horizontal rotation
                
                if (directionToSwarm.magnitude > 0.1f)
                {
                    transform.rotation = Quaternion.LookRotation(directionToSwarm.normalized);
                    hasSetInitialRotation = true;
                }
            }
        }
    }
}
