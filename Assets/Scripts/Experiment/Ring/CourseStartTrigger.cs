using UnityEngine;
using UnityEngine.Events;

/// <summary>
/// Invisible trigger plane that starts the course when a drone enters.
/// Fires an event and calls RingGateManager.StartCourse() on first valid trigger.
/// </summary>
[RequireComponent(typeof(BoxCollider))]
public class CourseStartTrigger : MonoBehaviour
{
    // ─────────────────────────────────────────────────────────────────────────
    // Inspector
    // ─────────────────────────────────────────────────────────────────────────

    [Header("References")]
    [Tooltip("The course manager to call StartCourse() on.")]
    public RingGateManager courseManager;

    [SerializeField]
    [Tooltip("Child renderer used to visualize the start zone. Size it independently of the trigger collider.")]
    private Renderer nextGateRenderer;

    [Header("Trigger Settings")]
    [Tooltip("Tag of colliders that can activate the course start.")]
    public string droneTag = "Player";

    [Tooltip("If true, the trigger fires only once per session. " +
             "Call ResetTrigger() or set this to false to re-arm.")]
    public bool singleFirePerSession = true;

    [Header("Events")]
    [Tooltip("Fires once when the first qualifying drone enters the trigger.")]
    public UnityEvent onCourseTriggered;

    // ─────────────────────────────────────────────────────────────────────────
    // Private
    // ─────────────────────────────────────────────────────────────────────────

    private bool _triggered = false;

    // ─────────────────────────────────────────────────────────────────────────
    // Lifecycle
    // ─────────────────────────────────────────────────────────────────────────

    private void Awake()
    {
        var col = GetComponent<BoxCollider>();
        col.isTrigger = true;
    }

    private void OnTriggerEnter(Collider other)
    {
        if (singleFirePerSession && _triggered) return;

        // Check via attached Rigidbody root (matches RingGate.cs convention)
        Rigidbody rb = other.attachedRigidbody;
        GameObject root = rb != null ? rb.gameObject : other.gameObject;

        if (!root.CompareTag(droneTag)) return;

        _triggered = true;

        // Disable the visual indicator when course starts
        if (nextGateRenderer != null)
        {
            nextGateRenderer.enabled = false;
        }

        onCourseTriggered?.Invoke();
        courseManager?.StartCourse();

        Debug.Log($"[CourseStartTrigger] Course triggered by '{root.name}'.");
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Public API
    // ─────────────────────────────────────────────────────────────────────────

    /// <summary>Re-arms the trigger so it can fire again on the next drone entry.</summary>
    public void ResetTrigger()
    {
        _triggered = false;
        // Re-enable the visual indicator
        if (nextGateRenderer != null)
        {
            nextGateRenderer.enabled = true;
        }
    }

    /// <summary>Immediately fires the trigger programmatically (bypasses tag check).</summary>
    public void FireManually()
    {
        if (singleFirePerSession && _triggered) return;
        _triggered = true;

        // Disable the visual indicator when course starts
        if (nextGateRenderer != null)
        {
            nextGateRenderer.enabled = false;
        }

        onCourseTriggered?.Invoke();
        courseManager?.StartCourse();
    }
}
