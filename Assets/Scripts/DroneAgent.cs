﻿using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using MLAgents;

public class DroneAgent: Agent {

	public VelocityControl velocityControl;
	[Range(0,100)] public float Scale;

	public bool use_new_state = true;


    private FreeSpaceDetection fsd;

	private Bounds endBounds;

	public float FORWARD_VELOCITY;
	public float YAW_RATE;

    public float DONE_DISTANCE;

    private bool collided = false;

	private bool wait = false;

	private Vector3 initialPos;
    private Quaternion initialRot;

	private float maxX;
	private float minX;
	private float maxZ;
	private float minZ;

	private System.Random rand;

	private bool local_done = false;

	public override void InitializeAgent() {

        fsd = gameObject.GetComponent<FreeSpaceDetection>();
        
		wait = false;

        collided = false;

		local_done = false;

	}


	// gets relative header
	public float normalizedHeader(Vector3 gpsCurr, Vector3 gpsTarg) {
		Vector3 normalized = Vector3.Normalize (gpsTarg - gpsCurr);
		normalized.y = 0.0f;

		Vector3 currentHeading = Quaternion.Euler(new Vector3(0.0f, velocityControl.state.Angles.y, 0.0f)) * Vector3.forward;
		currentHeading.y = 0.0f;

		float angle = Vector3.SignedAngle (currentHeading, normalized, Vector3.up);
		return angle;
			
	}


	// API-3 changes
	public override void CollectObservations()
	{
        //		List<float> state = new List<float>();
        // Debug.Log("CALLED");

        //NEW STATE
        //do this in collect state so we make sure we don't miss it
        local_done = isDone() || collided;


        if (use_new_state)
        {

            //Velocities (v forward, yaw)
            //Debug.Log(velocityControl.state.VelocityVector);
            //Debug.Log(velocityControl.state.AngularVelocityVector);
            //Debug.Log();
            AddVectorObs(velocityControl.state.VelocityVector.z / FORWARD_VELOCITY); // VX scaled -1 to 1
            AddVectorObs(velocityControl.state.AngularVelocityVector.y / YAW_RATE); //Yaw rate scaled -1  to 1
            //collision
            AddVectorObs((collided ? 1.0f : 0.0f));

            AddVectorObs(fsd.batchRaycast());
        }
        else
        {

            //13 elements
            AddVectorObs(velocityControl.state.VelocityVector.z / 8.0f); // VX scaled
            AddVectorObs(velocityControl.state.VelocityVector.x / 8.0f); // VY scaled
            AddVectorObs(velocityControl.state.AngularVelocityVector.y / 360.0f); //Yaw scaled

            AddVectorObs(velocityControl.transform.position.x);
            AddVectorObs(velocityControl.transform.position.y);
            AddVectorObs(velocityControl.transform.position.z);

            AddVectorObs(velocityControl.transform.rotation.x);
            AddVectorObs(velocityControl.transform.rotation.y);
            AddVectorObs(velocityControl.transform.rotation.z);

            AddVectorObs((collided ? 1.0f : 0.0f));
        }
        
        if (collided)
        {
            Debug.Log("COLLISION MSG SENT");
            collided = false;
        }
        

	}

	// 1 element input
	// -> -1 : STOP
	// -> 0 : LEFT + FORWARD
	// -> 1 : STRAIGHT + FORWARD
	// -> 2 : RIGHT + FORWARD
	public override void AgentAction(float[] vectorAction, string textAction)
	{
		//only wait initially if we are a non external player
		if (wait && brain.brainType == BrainType.Player) {
			return;
		}

        if (isDone()) {
            GetComponent<Rigidbody>().velocity = Vector3.zero;
        }
		// add in code logic for drone control
//		basicControl.Controller.InputAction(0, act[0], act[1], act[2]);

		// Debug.Log (act);

        // Debug.Log("ACTION");

//		float angle = normalizedHeader (transform.position, endRegion.transform.position);
//		Debug.Log (angle);

		// pitch forward as long as it isn't –1
        velocityControl.desired_vx = vectorAction[0] >= 0 ? FORWARD_VELOCITY : 0.0f;
		velocityControl.desired_vy = 0.0f;

        if (vectorAction[0] < -1 + 1e-8) {
            //STOP
            velocityControl.desired_yaw = 0.0f;
        } else if (vectorAction[0] < 1e-8) {  // equals 0
			//LEFT
			velocityControl.desired_yaw = -YAW_RATE;
        } else if (vectorAction[0] < 1 + 1e-8) {  // equals 1
			//STRAIGHT
			velocityControl.desired_yaw = 0.0f;
        } else{
            //RIGHT
            velocityControl.desired_yaw = YAW_RATE;
        }


        // no state collections being called coming in
        if (local_done)
        {
            //Debug.Log("STOP");
            Done();
            //HALT ALL MOTION UNTIL RESET
            velocityControl.enabled = false;
            GetComponent<Rigidbody>().isKinematic = true;
        }

	}

	public bool isDone(){
        Vector3 currPos = new Vector3(transform.position.x, endBounds.center.y, transform.position.z);
        //return endBounds.Contains(currPos)
        return Vector3.Magnitude(currPos - endBounds.center) <= DONE_DISTANCE;
    }

	public override void AgentReset()
	{
        Debug.Log("Resetting");
        
        local_done = false;

        //temporary
		velocityControl.enabled = false;
		// randomness
		float startX = ((float) rand.NextDouble()) * (maxX - minX) + minX;
		float startZ = ((float) rand.NextDouble()) * (maxZ - minZ) + minZ;

		transform.position = new Vector3 (startX, initialPos.y, startZ);
        transform.rotation = Quaternion.AngleAxis( (float) (rand.NextDouble()) * 2.0f * 180.0f, Vector3.up );
		//reset, which also re enables

		//StartCoroutine (Waiting (1.0f));
		//while (!wait) {
		//}

        GetComponent<Rigidbody>().isKinematic = false;
		velocityControl.Reset ();
	}

	IEnumerator Waiting(float time) {
		wait = true;
		yield return new WaitForSeconds(time);
		wait = false;
	}

	public override void AgentOnDone()
	{
	}

	// super basic reward function
	//float RewardFunction(){
		//if (collided) {
		//	collided = false;
		//	local_done = true;
		//	return -1000.0f;
		//} else {
		//	//euclidean horizontal plane distance
		//	float dist = Mathf.Pow(endRegion.transform.position.x - velocityControl.transform.position.x, 2) + Mathf.Pow(endRegion.transform.position.z - velocityControl.transform.position.z, 2);
		//	dist = Scale * dist;
		//	return 1.0f / dist;
		//}
			
	//}

	void OnCollisionEnter(Collision other)
	{
		Debug.LogWarning ("-- COLLISION --");
		collided = true;
	}
}
