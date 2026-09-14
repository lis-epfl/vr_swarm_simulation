using UnityEngine;

/// <summary>
/// Defines a flight profile with all adjustable drone parameters.
/// Create instances as assets (right-click in Project → Create → Drone → Flight Profile)
/// and assign to VelocityControl, CWLController, or FlightHUD.
///
/// Profiles can be: Soft (easy), Racing (hard), Custom (user-defined), Average (midpoint), etc.
/// </summary>
[CreateAssetMenu(fileName = "FlightProfile", menuName = "Drone/Flight Profile")]
public class FlightProfile : ScriptableObject
{
    // Defaults match VelocityControl's (and so DroneReduced.prefab's): ApplyControlStyle copies every
    // field here over the drone's own, so a profile created from these defaults leaves the tuning
    // exactly as it was instead of silently replacing it.

    [Header("Angle Limits (radians)")]
    [Tooltip("Max pitch angle — ~9° = 0.15, ~15° = 0.26, ~25° = 0.44")]
    public float maxPitch = 0.436332f;

    [Tooltip("Max roll angle")]
    public float maxRoll = 0.436332f;

    [Header("Rate Limits")]
    [Tooltip("Max yaw rate (rad/s). 1.309 = 75 deg/s, the DJI Mini 3 Pro's maximum.")]
    public float maxYawRate = 1.309f;

    [Tooltip("Max horizontal speed (m/s)")]
    public float maxSpeed = 9.31f;

    [Tooltip("Max vertical speed (m/s)")]
    public float maxAltitudeRate = 3.34f;

    [Tooltip("Max angular acceleration (rad/s²). This is the limit actually achieved: " +
             "VelocityControl applies its torque with ForceMode.Force, so the inertia tensor is " +
             "honoured rather than multiplied in twice. Scaled by 0.3893 from the 8.68 the prefab " +
             "used to hold, which delivered 3.38 on pitch/roll and 6.67 on yaw.")]
    public float maxAlpha = 3.38f;

    [Header("Response")]
    [Tooltip("Time constant of velocity → acceleration controller (seconds). " +
             "Larger = softer response = more angle budget for swarm. " +
             "Saturation ≈ g × maxPitch × tau.")]
    public float timeConstantAccel = 0.75f;
}
