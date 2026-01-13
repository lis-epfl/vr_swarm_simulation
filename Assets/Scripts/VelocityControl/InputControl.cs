using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class InputControl : MonoBehaviour {

	public VelocityControl vc;
	public OlfatiSaber olfatiSaber;

	public SwarmManager.SwarmAlgorithm currentAlgorithm;

	public float speedCoeff = 4.0f;
	public float yawRateCoeff = 0.5f;
	public float altCoeff = 0.1f;

	private float abs_height = 1;

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
			vc.desired_yaw = Input.GetAxisRaw("Yaw") * yawRateCoeff;
			abs_height += Input.GetAxisRaw("Throttle") * altCoeff;
			// vc.desired_height = abs_height;
		}
		else if (currentAlgorithm == SwarmManager.SwarmAlgorithm.OLFATI_SABER)
		{
			// Olfati-Saber velocity control
			olfatiSaber.desired_vx = Input.GetAxisRaw("Pitch") * speedCoeff;
			olfatiSaber.desired_vy = Input.GetAxisRaw("Roll") * speedCoeff;
			olfatiSaber.desired_yaw = Input.GetAxisRaw("Yaw") * yawRateCoeff;
			olfatiSaber.desired_height = abs_height;
		}

		else if (currentAlgorithm != SwarmManager.SwarmAlgorithm.NBV) // NEW ADVAITH NBV
		{
			// If not NBV, do nothing FOR NOW
			// ALL of your existing Reynolds or Olfati-Saber velocity/force
			// calculations should go inside this block.
			// For example:
			// Vector3 separationForce = CalculateSeparation();
			// Vector3 cohesionForce = CalculateCohesion();
			// ApplyForces(separationForce + cohesionForce);
		}
		
	}
}
