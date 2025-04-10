using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class BirdsEyeCamera : MonoBehaviour
{

    public GameObject gameManager;
    public Vector3 offset = new Vector3(0, 20, 0);
    public visualiseOlfatiSaber visualiseOlfatiSaber;
    public GameObject droneObject;
    private Vector3 dronePosition;


    // Start is called before the first frame update
    void Start()
    {
        if (gameManager != null)
        {
            visualiseOlfatiSaber = gameManager.GetComponent<visualiseOlfatiSaber>();
        } 
        else
        {
            Debug.LogError("GameManager is not assigned.");
        }

        if (visualiseOlfatiSaber != null)
        {
            droneObject = visualiseOlfatiSaber.droneObject;
        }
        else
        {
            Debug.LogError("VisualiseOlfatiSaber is not assigned.");
        }

    }

    // Update is called once per frame
    void Update()
    {
        // Update the droneObject reference if it has changed
        if (visualiseOlfatiSaber != null && visualiseOlfatiSaber.droneObject != droneObject)
        {
            droneObject = visualiseOlfatiSaber.droneObject;
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
