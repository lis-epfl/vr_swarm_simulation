using UnityEngine;

/// <summary>
/// Placed on an <b>EntryTrigger</b> or <b>ExitTrigger</b> child GameObject.
/// Relays OnTriggerEnter events up to the parent <see cref="RingGate"/>,
/// tagging each hit as coming from the entry (approach) or exit (far) plane
/// so RingGate can enforce directional pass detection.
///
/// Each trigger child needs:
///   • BoxCollider with <c>isTrigger = true</c>
///   • Local scale (500, 500, 0.5) — creates an effectively infinite detection plane
///     perpendicular to the ring axis (local Z).
///   • Local Z position: negative offset for entry plane, positive for exit plane.
///   • No Rigidbody required here; the drone GameObjects should carry a Rigidbody.
/// </summary>
[RequireComponent(typeof(BoxCollider))]
public class GateTriggerRelay : MonoBehaviour
{
    [Header("Plane Role")]
    [Tooltip("True = approach/entry plane (negative local Z).\n" +
             "False = exit plane (positive local Z).\n" +
             "A valid pass is entry → exit in that order.")]
    public bool isEntryPlane = true;

    [Header("Debug Logging")]
    [Tooltip("Log every collider that enters this trigger plane — including non-drone objects.\n" +
             "Useful for verifying the trigger volume is active and correctly positioned.\n" +
             "Inherits the parent RingGate's master debugEnabled switch.")]
    public bool debugLogAllTriggerHits = false;

    private RingGate _gate;

    private void Awake()
    {
        _gate = GetComponentInParent<RingGate>();

        if (_gate == null)
            Debug.LogError($"[GateTriggerRelay] '{gameObject.name}' could not find a " +
                           $"RingGate component in its parent hierarchy. " +
                           $"Make sure this script lives on a child of the RingGate prefab.");

        var col = GetComponent<BoxCollider>();
        if (!col.isTrigger)
        {
            col.isTrigger = true;
            Debug.LogWarning($"[GateTriggerRelay] BoxCollider on '{gameObject.name}' " +
                             $"was not set to isTrigger — fixed automatically.");
        }
    }

    private void OnTriggerEnter(Collider other)
    {
        // Raw hit log — fires for every collider, before RingGate filters by tag.
        // Lets you confirm the trigger volume is working even before drones are tagged.
        if (_gate != null && _gate.debugEnabled && debugLogAllTriggerHits)
        {
            Debug.Log($"[GateTriggerRelay | <color=#aaaaff>{transform.parent?.name ?? gameObject.name}</color>] " +
                      $"Trigger hit by '<color=#ffcc44>{other.gameObject.name}</color>' " +
                      $"(tag='{other.tag}', layer={LayerMask.LayerToName(other.gameObject.layer)})");
        }

        _gate?.RegisterPlaneHit(other, isEntryPlane);
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Editor helper: draw the trigger plane footprint as a thin disc gizmo
    // ─────────────────────────────────────────────────────────────────────────
#if UNITY_EDITOR
    private void OnDrawGizmos()
    {
        var gate = GetComponentInParent<RingGate>();
        if (gate == null) return;

        float r = gate.ringRadius * 2.5f;

        // Entry plane = green, exit plane = red
        Gizmos.color = isEntryPlane
            ? new Color(0.2f, 1f, 0.3f, 0.12f)
            : new Color(1f, 0.3f, 0.2f, 0.12f);

        int   segs = 48;
        float step = Mathf.PI * 2f / segs;

        Vector3 right  = transform.right;
        Vector3 up     = transform.up;
        Vector3 centre = transform.position;
        Vector3 prev   = centre + right * r;

        for (int i = 1; i <= segs; i++)
        {
            float   a    = i * step;
            Vector3 next = centre + (right * Mathf.Cos(a) + up * Mathf.Sin(a)) * r;
            Gizmos.DrawLine(prev, next);
            prev = next;
        }
    }
#endif
}
