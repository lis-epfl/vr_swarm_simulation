using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class singleFPV : MonoBehaviour
{
    public List<GameObject> swarm;
    public bool frontDrone = false;
    public int frontMostDroneId = -1;
    
    private string droneName;    
    
    void Start()
    {
        droneName = transform.parent.name;
    }

    void FixedUpdate()
    {
        CheckIfFrontDrone();
    }

    private void CheckIfFrontDrone()
    {
        // Get the current drone's position and forward direction
        Vector3 currentPosition = transform.position;
        Vector3 currentForward = transform.forward;

        float maxDotProduct = float.MinValue;
        GameObject frontMostDrone = null;
        int frontMostIndex = -1;

        // Iterate through all drones in the swarm
        for (int i = 0; i < swarm.Count; i++)
        {
            GameObject drone = swarm[i];
            
            // Get the DroneParent child object
            GameObject droneChild = drone.transform.Find("DroneParent").gameObject;
            
            if (droneChild == null)
                continue;

            Vector3 dronePosition = droneChild.transform.position;
            Vector3 droneForward = droneChild.transform.forward;

            // Calculate the dot product between the drone's forward direction and its position relative to origin
            // This represents how far forward the drone is in its looking direction
            float dotProduct = Vector3.Dot(droneForward, dronePosition);

            // Track the drone with the highest dot product
            if (dotProduct > maxDotProduct)
            {
                maxDotProduct = dotProduct;
                frontMostDrone = droneChild;
                frontMostIndex = i;
            }
        }

        // Store the frontmost drone ID
        frontMostDroneId = frontMostIndex;

        // Check if the current drone is the frontmost drone
        if (frontMostDrone == transform.gameObject)
        {
            frontDrone = true;
        }
        else
        {
            frontDrone = false;
        }
    }
}
