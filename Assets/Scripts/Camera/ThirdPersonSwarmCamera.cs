using System.Collections.Generic;
using UnityEngine;

/// <summary>
/// Follows the swarm centroid from behind, oriented along the average swarm heading.
/// Attach to a Camera GameObject and assign the swarm list (same as BirdsEyeCamera).
/// </summary>
public class ThirdPersonSwarmCamera : MonoBehaviour
{
    public List<GameObject> swarm;

    [Header("Position Offset")]
    [Tooltip("Distance behind the swarm centroid.")]
    public float backDistance = 8f;

    [Tooltip("Height above the swarm centroid.")]
    public float height = 4f;

    [Header("Smoothing")]
    [Tooltip("Higher = snappier. Lower = more cinematic lag.")]
    public float smoothSpeed = 5f;

    void Update()
    {
        if (swarm == null || swarm.Count == 0) return;

        Vector3 center  = Vector3.zero;
        Vector3 heading = Vector3.zero;
        int     count   = 0;

        foreach (GameObject drone in swarm)
        {
            if (drone == null) continue;
            Transform dp = drone.transform.Find("DroneParent");
            if (dp == null) continue;

            center  += dp.position;
            heading += dp.forward;
            count++;
        }

        if (count == 0) return;

        center  /= count;
        heading  = heading.normalized;

        if (heading.sqrMagnitude < 0.001f)
            heading = Vector3.forward;

        // Position: behind and above the swarm centroid
        Vector3 targetPos = center - heading * backDistance + Vector3.up * height;

        transform.position = Vector3.Lerp(transform.position, targetPos, Time.deltaTime * smoothSpeed);
        transform.LookAt(center);
    }
}
