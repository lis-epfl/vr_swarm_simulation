﻿using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class InputControl : MonoBehaviour {

	public VelocityControl vc;
	public OlfatiSaber olfatiSaber;

	public SwarmManager.SwarmAlgorithm currentAlgorithm;

	private float abs_height = 1;

	// Use this for initialization
	void Start () {
		
	}
	
	// Update is called once per frame
	void FixedUpdate () {
		
		
		if (currentAlgorithm == SwarmManager.SwarmAlgorithm.REYNOLDS)
		{
			// Default velocity control
			vc.desired_vx = Input.GetAxisRaw ("Pitch")*4.0f;
			vc.desired_vy = Input.GetAxisRaw ("Roll")*4.0f;
			vc.desired_yaw = Input.GetAxisRaw ("Yaw")*0.5f;
			abs_height += Input.GetAxisRaw("Throttle") * 0.1f;
			// vc.desired_height = abs_height;
		}
		else if (currentAlgorithm == SwarmManager.SwarmAlgorithm.OLFATI_SABER)
		{
			// Olfati-Saber velocity control
			olfatiSaber.desired_vx = Input.GetAxisRaw ("Pitch")*4.0f;
			olfatiSaber.desired_vy = Input.GetAxisRaw ("Roll")*4.0f;
			olfatiSaber.desired_yaw = Input.GetAxisRaw ("Yaw")*0.5f;
			olfatiSaber.desired_height = abs_height;
		}
	}
}
