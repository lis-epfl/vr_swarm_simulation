using System.Collections;
using System.Collections.Generic;
using UnityEngine;

[System.Serializable]
public class StateFinder : MonoBehaviour {
//	public float Pitch; // The current pitch for the given transform in radians
//	public float Roll; // The current roll for the given transform in radians
//	public float Yaw; // The current Yaw for the given transform in radians
	public float Altitude; // The current altitude from the zero position
	public Vector3 Position;
	public Vector3 GroundTruthPosition; // Position before noise is applied
	public Vector3 Angles;
	public Vector3 VelocityVector; // Velocity vector
	public Vector3 AngularVelocityVector; // Angular Velocity
	public Vector3 Acceleration; // Acceleration vector
	public Vector3 AngularAcceleration; // Angular Acceleration vector
	public Vector3 Inertia;
	public float Mass;

	// State estimation noise (simulates IMU/GPS sensor noise)
	public bool enableStateNoise = true;
	public float positionNoiseSigma = 0.03f;
	public float altitudeNoiseSigma = 0.015f;
	public float attitudeNoiseSigma = 0.004f;
	public float velocityNoiseSigma = 0.06f;
	private System.Random _rng;
	private bool _hasSpare = false;
	private float _spare;

	private bool flag = true; // Only get mass and inertia once

	public VelocityControl vc; // linked externally

	// Drone health status
	public bool IsAlive = true;

	void Awake() {
		_rng = new System.Random();
	}

	public void GetState() {

		Vector3 worldDown = vc.transform.InverseTransformDirection (Vector3.down);
		float Pitch = worldDown.z; // Small angle approximation (radians)
		float Roll = -worldDown.x; // Small angle approximation (radians)

		// Normalize yaw to [-180, 180] range to avoid 360 degree wraparound issue
		float rawYaw = vc.transform.eulerAngles.y;
		float Yaw = (rawYaw > 180f) ? rawYaw - 360f : rawYaw; // Now in [-180, 180] degrees
		Yaw *= Mathf.Deg2Rad; // Convert to radians

		Position = vc.transform.position;
		GroundTruthPosition = vc.transform.position; // Store before noise is applied
		Angles = new Vector3 (Pitch, Yaw, Roll);

		Altitude = vc.transform.position.y;

		VelocityVector = vc.transform.GetComponent<Rigidbody> ().velocity;
		VelocityVector = vc.transform.InverseTransformDirection (VelocityVector);

		AngularVelocityVector = vc.transform.GetComponent<Rigidbody> ().angularVelocity;
		AngularVelocityVector = vc.transform.InverseTransformDirection (AngularVelocityVector);

		Acceleration = vc.transform.GetComponent<Rigidbody>().GetAccumulatedForce() / Mass;
		Acceleration = vc.transform.InverseTransformDirection(Acceleration);
		
		AngularAcceleration = vc.transform.GetComponent<Rigidbody>().GetAccumulatedTorque() / Inertia.magnitude; //Approximation
		AngularAcceleration = vc.transform.InverseTransformDirection(AngularAcceleration);


		if (flag) {
			Inertia = vc.transform.GetComponent<Rigidbody> ().inertiaTensor;
			Mass = vc.transform.GetComponent<Rigidbody> ().mass;
			flag = false;
		}

		// Inject sensor noise to simulate realistic state estimation
		if (enableStateNoise) {
			Position.x += NextGaussian() * positionNoiseSigma;
			Position.z += NextGaussian() * positionNoiseSigma;
			Position.y += NextGaussian() * altitudeNoiseSigma;
			Altitude   += NextGaussian() * altitudeNoiseSigma;
			Angles.x   += NextGaussian() * attitudeNoiseSigma;
			Angles.z   += NextGaussian() * attitudeNoiseSigma;
			VelocityVector += new Vector3(
				NextGaussian() * velocityNoiseSigma,
				NextGaussian() * velocityNoiseSigma,
				NextGaussian() * velocityNoiseSigma);
		}

	}

	private float NextGaussian() {
		if (_hasSpare) {
			_hasSpare = false;
			return _spare;
		}
		float u, v, S;
		do {
			u = 2.0f * (float) _rng.NextDouble() - 1.0f;
			v = 2.0f * (float) _rng.NextDouble() - 1.0f;
			S = u * u + v * v;
		} while (S >= 1.0f);
		float fac = Mathf.Sqrt(-2.0f * Mathf.Log(S) / S);
		_spare = v * fac;
		_hasSpare = true;
		return u * fac;
	}

	public void Reset() {
		flag = true;
		Position = Vector3.zero;
		VelocityVector = Vector3.zero;
		AngularVelocityVector = Vector3.zero;
		Angles = Vector3.zero;
		Altitude = 0.0f;

		enabled = true;
	}
}
