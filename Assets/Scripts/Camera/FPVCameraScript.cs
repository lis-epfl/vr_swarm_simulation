using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class FPVCameraScript : MonoBehaviour
{

	public Transform droneTransform;
	public Vector3 offset;
	[Range(0, 1)] public float temp;
	[Range(0, 1)] public float rpLimitRatio;

	// NEW: Add pitch speed control
    [Range(0.01f, 1.0f)] private float pitchSpeed = 0.1f; // How fast the pitch changes

	// NEW: Add pitch limits
    [Header("Pitch Limits")]
    private float minPitch = -90.0f; // Looking down limit
    private float maxPitch = 35.0f;  // Looking up limit

	// NEW: Add pitch control variable (PRIVATE - not visible in Unity Inspector)
	private float additionalPitch = 0.0f; // This will be set by NBV script
	private float currentAdditionalPitch = 0.0f; // Current smoothed pitch


	// Use this for initialization
	void Start()
	{

	}

	// Update is called once per frame
	void Update()
	{
		transform.position = droneTransform.position + droneTransform.rotation * offset;

		// Smoothly interpolate the additional pitch
        currentAdditionalPitch = Mathf.Lerp(currentAdditionalPitch, additionalPitch, pitchSpeed * Time.deltaTime * 10f);


		Vector3 euler = droneTransform.rotation.eulerAngles;
		float x = (euler.x > 180.0f ? euler.x - 360.0f : euler.x) * rpLimitRatio;
		float z = (euler.z > 180.0f ? euler.z - 360.0f : euler.z) * rpLimitRatio;

		float nx = (x > 0 ? x : 360.0f + x);
		float nz = (z > 0 ? z : 360.0f + z);

		//		Debug.Log (nx);
		//		Debug.Log (nz);
		
		// NEW: Clamp the additional pitch to limits
        float clampedPitch = Mathf.Clamp(currentAdditionalPitch, minPitch, maxPitch);

		//	
		// Vector3 newEuler = new Vector3(nx + additionalPitch, euler.y, nz);
		// Use the smoothed pitch instead of the immediate value
		Vector3 newEuler = new Vector3(nx - clampedPitch, euler.y, nz);


		//		Debug.Log (euler);
		//		Debug.Log (newEuler);

		Quaternion target = Quaternion.Euler(newEuler);

		//		transform.position = new Vector3 (droneTransform.position.x, droneTransform.position.y, droneTransform.position.z + 0.45f);
		transform.rotation = Quaternion.Slerp(transform.rotation, target, temp);
		//			Vector3.SmoothDamp(transform.position, drone.transform.TransformPoint(behindPosition) + Vector3.up * Input.GetAxis("Vertical"), ref velocityCameraFollow, .1f);
	}
	
	// NEW: Public method to set the pitch (called from NBV script)
    // NEW: Public method to set the pitch with limits (called from NBV script)
    public void SetAdditionalPitch(float pitch)
    {
        // Clamp the incoming pitch to our limits
        additionalPitch = Mathf.Clamp(pitch, minPitch, maxPitch);
    }
}
