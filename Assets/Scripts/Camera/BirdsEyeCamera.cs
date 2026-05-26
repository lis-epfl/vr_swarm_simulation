using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class BirdsEyeCamera : MonoBehaviour
{

    public GameObject gameManager;
    public int selected_drone = 0;
    public List<GameObject> swarm;
    public Vector3 offset = new Vector3(0, 20, 0);
    private GameObject droneObject;
    private Vector3 dronePosition;


    // Start is called before the first frame update
    void Start()
    {

    }

    // Update is called once per frame
    void Update()
    {
        if (swarm != null)
        {
            // Select the right drone from the swarm
            if (selected_drone < 0 || selected_drone >= swarm.Count)
            {
                return;
            }
            droneObject = swarm[selected_drone];
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
