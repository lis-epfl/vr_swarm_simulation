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
    [Header("Angle Limits (radians)")]
    [Tooltip("Max pitch angle — ~9° = 0.15, ~15° = 0.26, ~23° = 0.45")]
    public float maxPitch = 0.25f;

    [Tooltip("Max roll angle")]
    public float maxRoll = 0.25f;

    [Header("Rate Limits")]
    [Tooltip("Max yaw rate (rad/s)")]
    public float maxYawRate = 1.0f;

    [Tooltip("Max horizontal speed (m/s)")]
    public float maxSpeed = 5.0f;

    [Tooltip("Max vertical speed (m/s)")]
    public float maxAltitudeRate = 2.0f;

    [Tooltip("Max angular acceleration (rad/s²)")]
    public float maxAlpha = 10.0f;

    [Header("Response")]
    [Tooltip("Time constant of velocity → acceleration controller (seconds). " +
             "Larger = softer response = more angle budget for swarm. " +
             "Saturation ≈ g × maxPitch × tau.")]
    public float timeConstantAccel = 0.5f;
}
