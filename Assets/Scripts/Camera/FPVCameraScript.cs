﻿using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class FPVCameraScript : MonoBehaviour {

	public Transform droneTransform;
	public Vector3 offset;
	[Range(0,1)] public float temp;
	[Range(0,1)] public float rpLimitRatio;
	public float pitchOffset;


	// Use this for initialization
	void Start () {
		
	}
	
	// Update is called once per frame
	/// <summary>
	/// Updates the camera position and rotation to follow the drone with roll and pitch limitations applied.
	/// </summary>
	/// <remarks>
	/// This method performs the following operations each frame:
	/// 1. Positions the camera relative to the drone by applying the drone's rotation to a local offset vector
	/// 2. Extracts the drone's euler angles and converts them to a normalized range (-180 to 180 degrees)
	/// 3. Applies a roll and pitch limit ratio to constrain the camera's rotational movement
	/// 4. Normalizes the constrained angles to positive values (0-360 degrees)
	/// 5. Constructs a new rotation using the limited roll and pitch while preserving the drone's yaw
	/// 6. Smoothly interpolates the camera rotation toward the target rotation using spherical linear interpolation (Slerp)
	/// 
	/// The rpLimitRatio acts as a damping factor to reduce the camera's response to the drone's roll and pitch movements,
	/// while the temp parameter controls the speed of the rotation interpolation.
	/// </remarks>
	void Update () {
		transform.position = droneTransform.position + droneTransform.rotation * offset;

		Vector3 euler = droneTransform.rotation.eulerAngles;
		float x = (euler.x > 180.0f ? euler.x - 360.0f : euler.x) * rpLimitRatio + pitchOffset;
		float z = (euler.z > 180.0f ? euler.z - 360.0f : euler.z) * rpLimitRatio;

		float nx = (x > 0 ? x : 360.0f + x);
		float nz = (z > 0 ? z : 360.0f + z);

//		Debug.Log (nx);
//		Debug.Log (nz);
//	
		Vector3 newEuler = new Vector3 (nx, euler.y, nz);

//		Debug.Log (euler);
//		Debug.Log (newEuler);

		Quaternion target = Quaternion.Euler (newEuler);

//		transform.position = new Vector3 (droneTransform.position.x, droneTransform.position.y, droneTransform.position.z + 0.45f);
		transform.rotation = Quaternion.Slerp (transform.rotation, target, temp);
//			Vector3.SmoothDamp(transform.position, drone.transform.TransformPoint(behindPosition) + Vector3.up * Input.GetAxis("Vertical"), ref velocityCameraFollow, .1f);
	}
}