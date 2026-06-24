using System.Collections;
using System.Collections.Generic;
using UnityEngine;

/// <summary>
/// Drives the drone FPV camera as a DJI Mini 3 Pro–style stabilised gimbal.
/// </summary>
/// <remarks>
/// The camera follows the drone externally (its position is set each frame from the drone
/// mount point) but its orientation is fully stabilised so the view does not change when the
/// drone tilts to fly:
///
/// - <b>Roll</b> is forced to 0 (the horizon stays level).
/// - <b>Pitch</b> is set to a controllable gimbal angle (<see cref="SharedPitch"/>),
///   independent of the drone's pitch.
/// - <b>Yaw</b> is locked to the drone heading, so the camera always faces the way the drone
///   points (the image-stitching pipeline relies on camera yaw == drone heading).
///
/// The gimbal pitch is a single <i>swarm-wide</i> value shared by every drone, which keeps the
/// stitched panorama coherent (all cameras look the same direction). It is normally driven from
/// <c>SwarmManager.gimbalPitch</c> in the Inspector (the central control point, tunable at
/// runtime); the per-instance <see cref="pitch"/> field is only a fallback when no SwarmManager
/// is present. Either way the value is applied via <see cref="SetPitch"/>.
///
/// The camera also hides the owning drone's own body during its render pass, so the drone never
/// sees its own arms (which can swing into view during aggressive maneuvers) while still seeing
/// the other drones. See <see cref="SetOwnBodyHidden"/>.
/// </remarks>
public class FPVCameraScript : MonoBehaviour {

	public Transform droneTransform;
	public Vector3 offset;

	// The owning drone's renderers, hidden only while this FPV camera renders so the drone
	// doesn't see its own body. Cached in Start (the body hierarchy is fixed per prefab).
	private Renderer[] ownBodyRenderers;

	/// <summary>DJI-style gimbal tilt limits (degrees): straight down to slightly up.</summary>
	public const float MinPitch = -90f;
	public const float MaxPitch = 60f;

	[Header("Gimbal")]
	[Tooltip("Gimbal pitch in degrees (DJI convention): 0 = level horizon, negative = look " +
	         "down (to -90 = straight down), positive = look up (to +60). This is shared by " +
	         "the whole swarm so the stitched panorama stays coherent.")]
	[Range(MinPitch, MaxPitch)] public float pitch = 0f;

	[Tooltip("Gimbal response speed. Higher = snappier; very high is effectively instant. " +
	         "Frame-rate independent.")]
	public float smoothSpeed = 10f;

	// Authoritative swarm-wide gimbal pitch (degrees, DJI convention). Every instance reads
	// this so all drones tilt together.
	private static float sharedPitch = 0f;
	public static float SharedPitch => sharedPitch;

	/// <summary>
	/// Sets the swarm-wide gimbal pitch (degrees, DJI convention: negative = down, positive =
	/// up). Clamped to [<see cref="MinPitch"/>, <see cref="MaxPitch"/>]. Affects every drone.
	/// </summary>
	public static void SetPitch(float degrees) {
		sharedPitch = Mathf.Clamp(degrees, MinPitch, MaxPitch);
	}

	// Use this for initialization
	void Start () {
		// Seed the swarm-wide angle. SwarmManager owns the gimbal pitch when present (central
		// runtime control); fall back to this instance's Inspector field otherwise. Its Awake
		// has already run by now, so Instance is set.
		if (SwarmManager.Instance != null) {
			SetPitch(SwarmManager.Instance.GetGimbalPitch());
		} else {
			SetPitch(pitch);
		}

		// Cache the owning drone's renderers (DroneObj, motors, props, minimap marker). The FPV
		// camera is a sibling of the drone body, so it isn't included; the 3PV child is a Camera,
		// not a Renderer, so it isn't either.
		ownBodyRenderers = droneTransform != null
			? droneTransform.GetComponentsInChildren<Renderer>(true)
			: new Renderer[0];
	}

	// Update is called once per frame
	void Update () {
		if (droneTransform == null) return;

		// Position: follow the gimbal mount point on the drone.
		transform.position = droneTransform.position + droneTransform.rotation * offset;

		// Yaw locked to drone heading. Project the drone forward onto the horizontal plane so
		// the heading stays robust even when the drone pitches/rolls hard (avoids the
		// gimbal-lock artefacts of reading eulerAngles.y directly).
		Vector3 fwd = droneTransform.forward;
		fwd.y = 0f;
		float yaw = fwd.sqrMagnitude > 1e-6f
			? Quaternion.LookRotation(fwd, Vector3.up).eulerAngles.y
			: droneTransform.eulerAngles.y;

		// Stabilised orientation: roll = 0, pitch = swarm gimbal angle (Unity's +X euler tilts
		// the camera down, so negate the DJI-convention pitch), yaw = drone heading.
		Quaternion target = Quaternion.Euler(-sharedPitch, yaw, 0f);

		// Smooth toward the target, frame-rate independent.
		transform.rotation = Quaternion.Slerp(transform.rotation, target,
			1f - Mathf.Exp(-smoothSpeed * Time.deltaTime));
	}

	void OnValidate () {
		pitch = Mathf.Clamp(pitch, MinPitch, MaxPitch);
		if (Application.isPlaying) {
			SetPitch(pitch);
		}
	}

	// Hide the owning drone's body just for this camera's render pass: OnPreCull (before this
	// camera culls) hides it, OnPostRender (after this camera finishes) restores it. Cameras
	// render sequentially, so the body stays visible to every other view (other FPVs, this
	// drone's 3PV, the watching cam, the minimap, and neighbours in the stitched feed).
	void OnPreCull () {
		SetOwnBodyHidden(true);
	}

	void OnPostRender () {
		SetOwnBodyHidden(false);
	}

	// Safety net: never leave the body hidden if the camera is disabled between OnPreCull and
	// OnPostRender.
	void OnDisable () {
		SetOwnBodyHidden(false);
	}

	private void SetOwnBodyHidden (bool hidden) {
		if (ownBodyRenderers == null) return;
		for (int i = 0; i < ownBodyRenderers.Length; i++) {
			if (ownBodyRenderers[i] != null) {
				ownBodyRenderers[i].forceRenderingOff = hidden;
			}
		}
	}
}
