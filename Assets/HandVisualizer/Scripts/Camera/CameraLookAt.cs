using UnityEngine;
using System.Collections;
using System;

public class CameraLookAt : MonoBehaviour {

//	public Transform Target;
    public GameObject Drone;
    private Quaternion original_rotation;

    public Vector3 offset;
//	private Vector3 rotation_off;

    void Start() {
        if (Drone == null)
        {
            Drone = GameObject.Find("Drone 0"); // Attempt to find by a common name
            if (Drone == null)
            {
                Drone = GameObject.FindGameObjectWithTag("Player"); // Fallback to player tag
            }
            if (Drone == null)
            {
                Debug.LogError("CameraLookAt: Drone target could not be found automatically (tried name 'Drone 0' and tag 'Player'). Please assign it in the Inspector or check GameObject naming/tagging.");
            }
        }
        //		offset = transform.position - Drone.transform.position;
        //		rotation_off = transform.rotation - Drone.transform.rotation;
        original_rotation = transform.rotation;
    }
    // Update is called once per frame
    void Update () {
        //		transform.LookAt (Target);
        float yaw = Drone.transform.localEulerAngles.y;
        Vector3 relative_offset = Quaternion.AngleAxis(yaw, Vector3.up) * offset;
        transform.position = Drone.transform.position + relative_offset;

        transform.rotation = Quaternion.AngleAxis(yaw, Vector3.up) * original_rotation;
	}
}
