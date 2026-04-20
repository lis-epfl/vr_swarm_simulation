using UnityEngine;

/// <summary>
/// Smoothly follows a target (the demo dot) from behind and above.
/// Intended to be used with a Camera component on the same GameObject, targeting Display 5.
/// </summary>
public class DemoFollowCamera : MonoBehaviour
{
    [SerializeField] private Transform target;          // The DemoDot parent transform
    [SerializeField] private float backDistance = 8f;   // Distance behind the target
    [SerializeField] private float height = 3f;         // Height above the target
    [SerializeField] private float smoothSpeed = 5f;    // Lerp smoothing speed

    private Vector3 _lastForward = Vector3.forward;
    private Vector3 _lastTargetPos = Vector3.zero;

    private void OnValidate()
    {
        if (target == null && GetComponent<Camera>() == null)
            Debug.LogWarning("[DemoFollowCamera] No target assigned and no Camera component found on this GameObject.");
    }

    private void LateUpdate()
    {
        if (target == null)
            return;

        // Estimate the target's forward direction from movement
        Vector3 targetPos = target.position;
        Vector3 movement = targetPos - _lastTargetPos;

        if (movement.sqrMagnitude > 0.001f)
            _lastForward = movement.normalized;

        _lastTargetPos = targetPos;

        // Calculate desired camera position: behind and above the target
        Vector3 desiredPos = targetPos - _lastForward * backDistance + Vector3.up * height;

        // Smoothly interpolate camera position
        transform.position = Vector3.Lerp(transform.position, desiredPos, smoothSpeed * Time.deltaTime);

        // Look at the target
        transform.LookAt(target);
    }
}
