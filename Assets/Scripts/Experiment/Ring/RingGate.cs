using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Events;

/// <summary>
/// Core component for a swarm drone racing gate.
/// Tracks how many drones from the swarm pass inside vs outside the ring,
/// exposes metrics per-drone and in aggregate, and fires UnityEvents for
/// downstream systems (CWL logging, adaptive difficulty, UI, etc.).
///
/// Prefab hierarchy expected:
///   RingGate            ← this script + RingMeshGenerator
///   ├── GateTrigger     ← GateTriggerRelay + BoxCollider (isTrigger, scale 500,500,0.5)
///   └── CenterPoint     ← plain Transform, used as spline anchor
/// </summary>
[RequireComponent(typeof(RingMeshGenerator))]
public class RingGate : MonoBehaviour
{
    // ─────────────────────────────────────────────────────────────────────────
    // Inspector
    // ─────────────────────────────────────────────────────────────────────────

    [Header("Ring Geometry")]
    [Tooltip("Clear radius of the aperture (half of inner diameter). Default = 2.75 m for a 9-drone swarm.")]
    public float ringRadius = 2.75f;

    [Tooltip("Cross-section radius of the torus tube.")]
    public float tubeRadius = 0.2f;

    [Header("Swarm Settings")]
    [Tooltip("Total number of drones in the swarm. Used to fire onAllDronesPassed.")]
    public int swarmSize = 9;

    [Tooltip("Tag used to identify drone colliders.")]
    public string droneTag = "Player";

    [Header("References (auto-created if left empty)")]
    [Tooltip("Child transform placed at the geometric centre; used as spline node for path guidance.")]
    public Transform centerPoint;

    // ─────────────────────────────────────────────────────────────────────────
    // Debug
    // ─────────────────────────────────────────────────────────────────────────

    [Header("Debug Logging")]
    [Tooltip("Master switch. When off, no debug output is produced regardless of the options below.")]
    public bool debugEnabled = false;

    [Tooltip("Log each drone pass: name, inside/outside, radial distance, pass index.")]
    public bool debugLogPasses = true;

    [Tooltip("Log when a drone is rejected (wrong tag or duplicate in the same activation window).")]
    public bool debugLogRejections = false;

    [Tooltip("Log the live swarm progress counter after each pass (e.g. '3 / 9 drones cleared').")]
    public bool debugLogSwarmProgress = true;

    [Tooltip("Log when all swarmSize drones have cleared the gate.")]
    public bool debugLogGateCleared = true;

    [Tooltip("Log whenever ResetMetrics() is called.")]
    public bool debugLogReset = true;

    // ─────────────────────────────────────────────────────────────────────────
    // Events
    // ─────────────────────────────────────────────────────────────────────────

    [Header("Directional Detection")]
    [Tooltip("Seconds a drone has to cross the exit plane after the entry plane before its pending state resets.\n" +
             "Increase if drones are slow; decrease to avoid counting U-turns.")]
    public float passTimeoutSeconds = 3f;

    [Header("Events")]
    [Tooltip("Fires whenever any drone passes inside the ring aperture.")]
    public UnityEvent<RingPassData> onDronePassedInside;

    [Tooltip("Fires whenever any drone passes outside the ring aperture.")]
    public UnityEvent<RingPassData> onDronePassedOutside;

    [Tooltip("Fires for every drone pass (inside or outside).")]
    public UnityEvent<RingPassData> onDronePassed;

    [Tooltip("Fires once all swarmSize drones have passed through this gate (in a single gate activation).")]
    public UnityEvent<RingGate> onAllDronesPassed;

    [Tooltip("Fires when a drone crosses exit → entry (wrong direction). Data.wasInsideRing still reflects radial position.")]
    public UnityEvent<RingPassData> onDronePassedWrongDirection;

    // ─────────────────────────────────────────────────────────────────────────
    // Metrics (read from inspector or via properties)
    // ─────────────────────────────────────────────────────────────────────────

    [Header("Metrics — read-only")]
    [SerializeField] private int _totalPasses;
    [SerializeField] private int _insidePasses;
    [SerializeField] private int _outsidePasses;

    /// <summary>Total drone passes recorded since last ResetMetrics().</summary>
    public int TotalPasses  => _totalPasses;
    /// <summary>Passes where the drone was within ringRadius of the centre.</summary>
    public int InsidePasses => _insidePasses;
    /// <summary>Passes where the drone was outside ringRadius.</summary>
    public int OutsidePasses => _outsidePasses;
    /// <summary>Fraction of passes that went through the aperture. 0–1.</summary>
    public float AccuracyRatio => _totalPasses > 0 ? (float)_insidePasses / _totalPasses : 0f;

    // ─────────────────────────────────────────────────────────────────────────
    // Private state
    // ─────────────────────────────────────────────────────────────────────────

    // instanceID → inside count / outside count
    private readonly Dictionary<int, int> _droneInsideCounts  = new();
    private readonly Dictionary<int, int> _droneOutsideCounts = new();

    // Tracks which drones have passed in the current "activation" window
    // (reset when the gate is reset or when all drones have passed).
    private readonly HashSet<int> _passedThisActivation = new();

    // Directional state: droneId → (hitEntryPlane, timestamp of that hit).
    // A drone is "pending" after it crosses one plane; it must cross the other
    // within passTimeoutSeconds to register a pass.
    private readonly Dictionary<int, (bool hitEntry, float time)> _pendingDrones = new();

    // ─────────────────────────────────────────────────────────────────────────
    // Lifecycle
    // ─────────────────────────────────────────────────────────────────────────

    private void Awake()
    {
        EnsureCenterPoint();
        Log($"Initialised — aperture radius {ringRadius} m, swarm size {swarmSize}, tag '{droneTag}'.");
    }

    private void EnsureCenterPoint()
    {
        if (centerPoint != null) return;

        Transform existing = transform.Find("CenterPoint");
        if (existing != null)
        {
            centerPoint = existing;
            Log("CenterPoint found in hierarchy.");
            return;
        }

        var go = new GameObject("CenterPoint");
        go.transform.SetParent(transform, false);
        centerPoint = go.transform;
        Log("CenterPoint not found — created automatically.");
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Detection — called by GateTriggerRelay
    // ─────────────────────────────────────────────────────────────────────────

    /// <summary>
    /// Called by <see cref="GateTriggerRelay"/> when a collider enters one of the two gate planes.
    /// Enforces directional detection: a valid pass requires entry-plane → exit-plane order.
    /// Wrong-direction crossings (exit → entry) fire <see cref="onDronePassedWrongDirection"/>.
    /// </summary>
    public void RegisterPlaneHit(Collider col, bool isEntryPlane)
    {
        Rigidbody rb = col.attachedRigidbody;
        GameObject droneRoot = rb != null ? rb.gameObject : col.gameObject;

        if (!droneRoot.CompareTag(droneTag))
            return;

        int   droneId = droneRoot.GetInstanceID();
        float now     = Time.time;

        // Expire stale pending state
        if (_pendingDrones.TryGetValue(droneId, out var pending) &&
            now - pending.time > passTimeoutSeconds)
        {
            _pendingDrones.Remove(droneId);
            pending = default;
        }

        bool hasPending = _pendingDrones.ContainsKey(droneId);

        if (!hasPending)
        {
            // First plane hit — store which plane it was
            _pendingDrones[droneId] = (isEntryPlane, now);
            Log($"Drone '{droneRoot.name}' hit {(isEntryPlane ? "entry" : "exit")} plane — waiting for {(isEntryPlane ? "exit" : "entry")}.");
            return;
        }

        bool firstWasEntry = _pendingDrones[droneId].hitEntry;
        _pendingDrones.Remove(droneId);

        if (firstWasEntry && !isEntryPlane)
        {
            // Correct direction: entry → exit
            RecordConfirmedPass(droneRoot, col);
        }
        else if (!firstWasEntry && isEntryPlane)
        {
            // Wrong direction: exit → entry
            Log($"Drone '{droneRoot.name}' passed in the WRONG direction.");
            RecordWrongDirectionPass(droneRoot);
        }
        // Same plane twice (e.g. drone bounced back before clearing): ignore
    }

    /// <summary>
    /// Records a direction-confirmed gate pass, updates metrics, and fires events.
    /// </summary>
    private void RecordConfirmedPass(GameObject droneRoot, Collider originCollider)
    {
        int droneId = droneRoot.GetInstanceID();

        if (_passedThisActivation.Contains(droneId))
        {
            LogRejection($"Duplicate pass ignored for '{droneRoot.name}' (already counted in this activation window).");
            return;
        }
        _passedThisActivation.Add(droneId);

        // ── Radial distance check ────────────────────────────────────────────
        Vector3 localPos   = transform.InverseTransformPoint(droneRoot.transform.position);
        float   radialDist = new Vector2(localPos.x, localPos.y).magnitude;
        bool    isInside   = radialDist <= ringRadius;

        // ── Build pass record ────────────────────────────────────────────────
        var data = new RingPassData
        {
            droneObject    = droneRoot,
            droneId        = droneId,
            radialDistance = radialDist,
            wasInsideRing  = isInside,
            ringGate       = this,
            passIndex      = _totalPasses,
            timestamp      = Time.time,
        };

        // ── Update metrics ───────────────────────────────────────────────────
        _totalPasses++;

        if (isInside)
        {
            _insidePasses++;
            _droneInsideCounts.TryAdd(droneId, 0);
            _droneInsideCounts[droneId]++;
            LogPass(data, "INSIDE ");
            onDronePassedInside?.Invoke(data);
        }
        else
        {
            _outsidePasses++;
            _droneOutsideCounts.TryAdd(droneId, 0);
            _droneOutsideCounts[droneId]++;
            LogPass(data, "OUTSIDE");
            onDronePassedOutside?.Invoke(data);
        }

        onDronePassed?.Invoke(data);

        // ── Swarm progress ───────────────────────────────────────────────────
        LogProgress(_passedThisActivation.Count);

        // ── Check if all swarm drones have now passed ────────────────────────
        if (_passedThisActivation.Count >= swarmSize)
        {
            LogGateCleared();
            onAllDronesPassed?.Invoke(this);
            _passedThisActivation.Clear();
        }
    }

    private void RecordWrongDirectionPass(GameObject droneRoot)
    {
        int droneId = droneRoot.GetInstanceID();

        Vector3 localPos   = transform.InverseTransformPoint(droneRoot.transform.position);
        float   radialDist = new Vector2(localPos.x, localPos.y).magnitude;

        var data = new RingPassData
        {
            droneObject    = droneRoot,
            droneId        = droneId,
            radialDistance = radialDist,
            wasInsideRing  = radialDist <= ringRadius,
            ringGate       = this,
            passIndex      = _totalPasses,
            timestamp      = Time.time,
        };

        if (debugEnabled)
            Debug.LogWarning($"{GateLabel} <color=#ff6666>WRONG DIRECTION</color> — drone='{droneRoot.name}'  " +
                             $"radial={radialDist:F3} m  t={data.timestamp:F2}s");

        onDronePassedWrongDirection?.Invoke(data);
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Public API
    // ─────────────────────────────────────────────────────────────────────────

    /// <summary>Reset all accumulated metrics and the current activation window.</summary>
    public void ResetMetrics()
    {
        _totalPasses = _insidePasses = _outsidePasses = 0;
        _droneInsideCounts.Clear();
        _droneOutsideCounts.Clear();
        _passedThisActivation.Clear();
        _pendingDrones.Clear();

        if (debugEnabled && debugLogReset)
            Debug.Log($"[RingGate | <color=#aaaaff>{gameObject.name}</color>] Metrics reset.");
    }

    /// <summary>Number of times a specific drone (by instance ID) passed inside.</summary>
    public int GetInsideCountForDrone(int droneId) =>
        _droneInsideCounts.TryGetValue(droneId, out int c) ? c : 0;

    /// <summary>Number of times a specific drone (by instance ID) passed outside.</summary>
    public int GetOutsideCountForDrone(int droneId) =>
        _droneOutsideCounts.TryGetValue(droneId, out int c) ? c : 0;

    // ─────────────────────────────────────────────────────────────────────────
    // Internal debug helpers
    // ─────────────────────────────────────────────────────────────────────────

    // Colour-coded gate name prefix reused by all log methods.
    private string GateLabel => $"[RingGate | <color=#aaaaff>{gameObject.name}</color>]";

    private void Log(string msg)
    {
        if (debugEnabled) Debug.Log($"{GateLabel} {msg}");
    }

    private void LogRejection(string msg)
    {
        if (debugEnabled && debugLogRejections)
            Debug.LogWarning($"{GateLabel} {msg}");
    }

    private void LogPass(RingPassData d, string label)
    {
        if (!debugEnabled || !debugLogPasses) return;

        string colour  = d.wasInsideRing ? "#00ff99" : "#ff6666";
        string outcome = $"<color={colour}>{label}</color>";
        Debug.Log($"{GateLabel} Pass #{d.passIndex}  {outcome}  " +
                  $"drone='{d.droneObject.name}'  " +
                  $"radial={d.radialDistance:F3} m  " +
                  $"(limit={ringRadius:F2} m)  " +
                  $"t={d.timestamp:F2}s");
    }

    private void LogProgress(int count)
    {
        if (!debugEnabled || !debugLogSwarmProgress) return;
        Debug.Log($"{GateLabel} Swarm progress: {count} / {swarmSize} drones cleared.");
    }

    private void LogGateCleared()
    {
        if (!debugEnabled || !debugLogGateCleared) return;
        Debug.Log($"{GateLabel} <color=#ffdd00>Gate fully cleared!</color>  " +
                  $"inside={_insidePasses}  outside={_outsidePasses}  " +
                  $"accuracy={AccuracyRatio * 100f:F1}%");
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Editor gizmos
    // ─────────────────────────────────────────────────────────────────────────

#if UNITY_EDITOR
    private void OnDrawGizmos()
    {
        // Aperture circle
        Gizmos.color = new Color(0f, 0.9f, 1f, 0.9f);
        DrawGizmoCircle(transform.position, transform.forward, ringRadius, 64);

        // Outer (tube centre) circle
        Gizmos.color = new Color(0f, 0.9f, 1f, 0.3f);
        DrawGizmoCircle(transform.position, transform.forward, ringRadius + tubeRadius, 64);

        // Centre point indicator
        if (centerPoint != null)
        {
            Gizmos.color = Color.yellow;
            Gizmos.DrawSphere(centerPoint.position, 0.12f);
            Gizmos.DrawLine(transform.position, centerPoint.position);
        }

        // Ring normal axis (direction of travel)
        Gizmos.color = Color.green;
        Gizmos.DrawRay(transform.position, transform.forward * (ringRadius * 0.6f));
    }

    private static void DrawGizmoCircle(Vector3 centre, Vector3 normal, float radius, int segments)
    {
        Vector3 right = Vector3.Cross(normal, Vector3.up);
        if (right.sqrMagnitude < 0.001f) right = Vector3.right;
        right.Normalize();
        Vector3 up = Vector3.Cross(right, normal).normalized;

        Vector3 prev = centre + right * radius;
        for (int i = 1; i <= segments; i++)
        {
            float   angle = i * Mathf.PI * 2f / segments;
            Vector3 next  = centre + (right * Mathf.Cos(angle) + up * Mathf.Sin(angle)) * radius;
            Gizmos.DrawLine(prev, next);
            prev = next;
        }
    }
#endif
}

// ─────────────────────────────────────────────────────────────────────────────
// Data transfer object
// ─────────────────────────────────────────────────────────────────────────────

/// <summary>
/// Snapshot of a single drone gate-pass event.
/// Passed to all RingGate UnityEvents so listeners have full context.
/// </summary>
[System.Serializable]
public class RingPassData
{
    /// <summary>The drone GameObject that passed.</summary>
    public GameObject droneObject;

    /// <summary>Instance ID of the drone (stable within a session).</summary>
    public int droneId;

    /// <summary>Radial distance from the ring axis at the moment of crossing.</summary>
    public float radialDistance;

    /// <summary>True if the drone was within ringRadius (i.e. through the aperture).</summary>
    public bool wasInsideRing;

    /// <summary>The gate that recorded this pass.</summary>
    public RingGate ringGate;

    /// <summary>Zero-based index of this pass within the gate's total pass count.</summary>
    public int passIndex;

    /// <summary>Time.time when the pass was registered.</summary>
    public float timestamp;
}
