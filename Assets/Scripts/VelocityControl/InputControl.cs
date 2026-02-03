using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class InputControl : MonoBehaviour {

	public SwarmAlgorithm swarmAlgorithm;
	public AttitudeAlgorithm attitudeAlgorithm;
	public float speedCoeff = 4.0f;
	public float yawRateCoeff = 0.5f;
	public float altCoeff = 0.1f;
	// Use this for initialization
	void Start () {}

	// Update is called once per frame
	void FixedUpdate()
	{
		swarmAlgorithm.desired_vx = Input.GetAxisRaw("Pitch") * speedCoeff;
		swarmAlgorithm.desired_vy = Input.GetAxisRaw("Roll") * speedCoeff;
		attitudeAlgorithm.SetYawRateFromCommand(Input.GetAxisRaw("Yaw") * yawRateCoeff);
		swarmAlgorithm.desired_height = Input.GetAxisRaw("Throttle") * altCoeff;
	}
}
