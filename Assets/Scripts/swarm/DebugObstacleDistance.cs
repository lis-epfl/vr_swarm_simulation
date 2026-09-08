using UnityEngine;

// TEMPORARY DEBUG SCRIPT.
// Mirrors the distance calculation inside OlfatiSaber.GetObstacleForce() so the values
// it actually uses can be inspected for a single selected drone. Delete when done.
public class DebugObstacleDistance : MonoBehaviour
{
    [Header("Drone Selection")]
    public int selectedDroneIndex = 0;

    [Header("Logging")]
    [Tooltip("Seconds between console prints (0 = every frame).")]
    public float logInterval = 0.5f;

    private const string k_ObstacleLayerName = "Obstacle";

    private float _nextLogTime;

    void Update()
    {
        if (Time.time < _nextLogTime) return;
        _nextLogTime = Time.time + logInterval;

        // Resolve the selected drone the same way the tuner does.
        GameObject selectedDrone = GameObject.Find($"SwarmParent/Drone {selectedDroneIndex}");
        if (selectedDrone == null)
        {
            Debug.LogWarning($"[ObstacleDebug] Drone {selectedDroneIndex} not found under SwarmParent.");
            return;
        }

        Transform droneParent = selectedDrone.transform.Find("DroneParent");
        if (droneParent == null) return;

        OlfatiSaber olfatiSaber = droneParent.GetComponent<OlfatiSaber>();
        VelocityControl vc = droneParent.GetComponent<VelocityControl>();
        if (olfatiSaber == null || vc == null) return;

        Vector3 dronePosition = vc.State.Position;

        // --- Copied from OlfatiSaber.GetObstacleForce() ---
        float queryRadius = olfatiSaber.r0_obs * olfatiSaber.ScaleFactor;
        Collider[] obstacles = Physics.OverlapSphere(
            dronePosition, queryRadius, LayerMask.GetMask(k_ObstacleLayerName));

        if (obstacles.Length == 0)
        {
            Debug.Log($"[ObstacleDebug] Drone {selectedDroneIndex}: no obstacles within " +
                      $"r0_obs*ScaleFactor = {queryRadius:F2} world units.");
            return;
        }

        foreach (Collider obstacleCollider in obstacles)
        {
            // Ask OlfatiSaber itself rather than re-deriving: the surface distance is now measured
            // to the enclosing cylinder, not to the bounding box.
            OlfatiSaber.ObstacleFrame frame = olfatiSaber.GetObstacleFrame(obstacleCollider, dronePosition);
            if (!frame.valid) continue;
            float distanceToObstacle = frame.distance;                            // value the algorithm uses
            float worldDistance = distanceToObstacle * olfatiSaber.ScaleFactor;

            Transform obstacleParent = obstacleCollider.transform.parent;
            string parentName = obstacleParent != null ? obstacleParent.name : "(none)";

            Debug.Log($"[ObstacleDebug] Drone {selectedDroneIndex} -> '{obstacleCollider.name}' " +
                      $"(parent='{parentName}'): " +
                      $"world dist = {worldDistance:F2}, " +
                      $"scaled dist (used by algorithm) = {distanceToObstacle:F3}  " +
                      $"| obstacle pos = {obstacleCollider.transform.position}, drone pos = {dronePosition}  " +
                      $"(ScaleFactor={olfatiSaber.ScaleFactor}, d_obs={olfatiSaber.d_obs}, r0_obs={olfatiSaber.r0_obs})");
        }
    }
}
