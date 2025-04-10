using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class visualiseOlfatiSaber : MonoBehaviour
{
    public List<GameObject> swarm;
    public int selectedDrone = 0;
    
    // A scaling factor for the line thickness (adjust in the Inspector)
    public float widthScale = 0.1f; 
    public GameObject droneObject;
    
    private OlfatiSaber olfatiSaber;
    private float cohesionMag;
    private Vector3 cohesion;

    // Dictionary to hold a LineRenderer for each neighbour
    private Dictionary<GameObject, LineRenderer> neighbourLineRenderers = new Dictionary<GameObject, LineRenderer>();

    // Start is called before the first frame update
    void Start()
    {
        // You can leave Start empty or use it for other initialization.
    }

    // Update is called once per frame
    void Update()
    {
        // If the swarm is not yet available, do nothing.
        if (swarm == null || swarm.Count == 0)
            return;

        // Create the line renderers once the swarm is found, if not already created.
        if (neighbourLineRenderers.Count == 0)
        {
            // Validate that the selectedDrone is within the swarm's range.
            if (selectedDrone < 0 || selectedDrone >= swarm.Count)
            {
                Debug.LogError("Selected drone index is out of range.");
                return;
            }
            
            droneObject = swarm[selectedDrone];
            foreach (GameObject drone in swarm)
            {
                // Skip the selected drone.
                if (drone == droneObject)
                    continue;

                // Create a new GameObject to hold the LineRenderer for this neighbour.
                GameObject lineObj = new GameObject("LineTo_" + drone.name);
                lineObj.transform.parent = this.transform;

                // Add and configure the LineRenderer.
                LineRenderer lr = lineObj.AddComponent<LineRenderer>();
                lr.material = new Material(Shader.Find("Sprites/Default"));
                lr.positionCount = 2;
                lr.useWorldSpace = true;
                lr.shadowCastingMode = UnityEngine.Rendering.ShadowCastingMode.Off;
                lr.receiveShadows = false;
                lr.startWidth = 0.01f;
                lr.endWidth = 0.01f;

                // Store this LineRenderer for later updates.
                neighbourLineRenderers.Add(drone, lr);
            }
        }

        // At this point, swarm exists and the line renderers have been created.
        // Validate the selectedDrone.
        if (selectedDrone < 0 || selectedDrone >= swarm.Count)
            return;
        droneObject = swarm[selectedDrone];

        // Find the OlfatiSaber script from the selected drone.
        olfatiSaber = FindOlfatiSaber();
        if (olfatiSaber == null)
            return;

        // Reset the cohesion vector.
        cohesion = Vector3.zero;

        // Get the position of the selected drone's "DroneParent" child.
        Transform droneParentTransform = droneObject.transform.Find("DroneParent");
        if (droneParentTransform == null)
        {
            Debug.LogError("DroneParent not found on the selected drone!");
            return;
        }
        Vector3 position = droneParentTransform.position;
        
        // Calculate and update the cohesion visualization for each neighbour.
        foreach (GameObject neighbour in swarm)
        {
            if (neighbour == droneObject)
                continue;

            Transform neighbourParent = neighbour.transform.Find("DroneParent");
            if (neighbourParent == null)
            {
                Debug.LogError("DroneParent not found for neighbour: " + neighbour.name);
                continue;
            }
            Vector3 neighbourPosition = neighbourParent.position;

            // Compute the relative position and its distance.
            Vector3 relativePosition = neighbourPosition - position;
            float distance = relativePosition.magnitude;

            // Calculate cohesion force based on the distance.
            cohesionMag = olfatiSaber.GetCohesionForce(distance);

            // Calculate the cohesion vector.
            if (distance != 0)
                cohesion = cohesionMag * (relativePosition / distance);
            else
                cohesion = Vector3.zero;

            // Retrieve the corresponding LineRenderer and update its properties.
            if (neighbourLineRenderers.TryGetValue(neighbour, out LineRenderer lr))
            {
                lr.SetPosition(0, position);
                lr.SetPosition(1, neighbourPosition);

                // Set line width based on the absolute magnitude of the cohesion force.
                float lineWidth = Mathf.Abs(cohesionMag) * widthScale;
                lr.startWidth = lineWidth;
                lr.endWidth = lineWidth;

                // Set color: red for positive cohesion, blue for negative, white for zero.
                if (cohesionMag > 0)
                    lr.material.color = Color.red;
                else if (cohesionMag < 0)
                    lr.material.color = Color.blue;
                else
                    lr.material.color = Color.white;
            }
        }
    }

    // Find the OlfatiSaber script from the selected drone's "DroneParent" child.
    private OlfatiSaber FindOlfatiSaber()
    {
        Transform droneParent = droneObject.transform.Find("DroneParent");
        if (droneParent != null)
        {
            return droneParent.GetComponent<OlfatiSaber>();
        }
        else
        {
            Debug.LogError("DroneParent not found for the selected drone!");
            return null;
        }
    }
}
