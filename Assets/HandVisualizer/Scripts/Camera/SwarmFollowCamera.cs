using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class SwarmFollowCamera : MonoBehaviour
{
    public Vector3 cameraOffset;

    private List<GameObject> swarm;
    private Camera swarmCamera;

    // This method should be called from another script once the swarm is created
    public void SetupCamera(List<GameObject> swarm)
    {
        
        // Set the swarm list
        this.swarm = swarm;
        
        // Create the camera if it doesn't exist yet.
        if (swarmCamera == null)
        {
            GameObject cameraObject = new GameObject("SwarmCamera");
            swarmCamera = cameraObject.AddComponent<Camera>();

            // Set the display to 2
            swarmCamera.targetDisplay = 2;
        }

        // Compute the centroid of the swarm.
        Vector3 centroid = GetSwarmCentroid();

        // Set the camera's position to the centroid plus the offset.
        swarmCamera.transform.position = centroid + cameraOffset;

        // Rotate the camera to look at the swarm centroid.
        swarmCamera.transform.LookAt(centroid);
    }
    
    // Update the position of the camera at the end of every frame
    void LateUpdate()
    {
        // If the camera hasn't been created yet, do nothing.
        if (swarmCamera == null)
        {
            return;
        }

        // Compute the centroid of the swarm.
        Vector3 centroid = GetSwarmCentroid();

        // Set the camera's position to the centroid plus the offset.
        swarmCamera.transform.position = centroid + cameraOffset;

        // Rotate the camera to look at the swarm centroid.
        swarmCamera.transform.LookAt(centroid);
    }
    
    // Get the centroid of the swarm
    public Vector3 GetSwarmCentroid()
    {
        Vector3 centroid = Vector3.zero;

        // Safety check in case the swarm is empty.
        if (swarm.Count == 0)
        {
            Debug.LogWarning("Swarm is empty!");
            return centroid;
        }

        // Sum up all drone positions.
        foreach (GameObject drone in swarm)
        {
            // Get the drone parent
            Transform droneParent = drone.transform.Find("DroneParent");
            centroid += droneParent.position;
        }

        // Average out the positions to get the centroid.
        centroid /= swarm.Count;

        return centroid;
    }
}
