using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class InputControl : MonoBehaviour {

	public VelocityControl vc;
	public SwarmAlgorithm swarmAlgorithm;

	public SwarmManager.SwarmAlgorithm currentAlgorithm;

	public float speedCoeff = 4.0f;
	public float yawRateCoeff = 0.5f;
	public float altCoeff = 0.1f;

	private float abs_height = 14;

	// Use this for initialization
	void Start () {
		
	}

	// Update is called once per frame
	void FixedUpdate()
	{


		if (currentAlgorithm == SwarmManager.SwarmAlgorithm.REYNOLDS)
		{
			// Default velocity control
			vc.desired_vx = Input.GetAxisRaw("Pitch") * speedCoeff;
			vc.desired_vy = Input.GetAxisRaw("Roll") * speedCoeff;
			vc.desiredYawRate = Input.GetAxisRaw("Yaw") * yawRateCoeff;
			abs_height += Input.GetAxisRaw("Throttle") * altCoeff;
			// vc.desired_height = abs_height;
		}
		else if (currentAlgorithm == SwarmManager.SwarmAlgorithm.OLFATI_SABER)
		{
			// Olfati-Saber velocity control
			swarmAlgorithm.desired_vx = Input.GetAxisRaw("Pitch") * speedCoeff;
			swarmAlgorithm.desired_vy = Input.GetAxisRaw("Roll") * speedCoeff;
			swarmAlgorithm.desiredYawRate = Input.GetAxisRaw("Yaw") * yawRateCoeff;
			vc.swarm_vy = Input.GetAxisRaw("Throttle") * altCoeff;
		}
		
	}
}
