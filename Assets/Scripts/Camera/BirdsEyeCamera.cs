using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class BirdsEyeCamera : MonoBehaviour
{

    public GameObject gameManager;
    public int selected_drone = 0;
    public Vector3 offset = new Vector3(0, 20, 0);
    private GameObject droneObject;
    private List<GameObject> swarm_ref;
    private Vector3 dronePosition;


    // Start is called before the first frame update
    void Start()
    {
        if (gameManager != null)
        {
            swarmSpawn swarmSpawn = gameManager.GetComponent<swarmSpawn>();
            if (swarmSpawn != null)
            {
                swarm_ref = swarmSpawn.swarm;
            }
            else
            {
                Debug.LogError("SwarmSpawn is not assigned.");
            }
        } 
        else
        {
            Debug.LogError("GameManager is not assigned.");
        }

    }

    // Update is called once per frame
    void Update()
    {
        if (swarm_ref != null)
        {
            // Select the right drone from the swarm
            if (selected_drone < 0 || selected_drone >= swarm_ref.Count)
            {
                return;
            }
            droneObject = swarm_ref[selected_drone];
            if (droneObject == null)
            {
                return;
            }
            // Get the position of the selected drone's "DroneParent"
            GameObject droneChild = droneObject.transform.Find("DroneParent").gameObject;
            Vector3 position = droneChild.transform.position;

            // Set the camera position above the drone
            transform.position = position + offset;
            // Look at the drone
            transform.LookAt(position);

        }
        
    }
}
