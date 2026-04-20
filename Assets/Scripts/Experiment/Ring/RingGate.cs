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
///   RingGate            ← this script + RectGateMeshGenerator
///   ├── GateTrigger     ← GateTriggerRelay + BoxCollider (isTrigger, scale 500,500,0.5)
///   └── CenterPoint     ← plain Transform, used as spline anchor
/// </summary>
[RequireComponent(typeof(RectGateMeshGenerator))]
public class RingGate : MonoBehaviour
{

    // ─────────────────────────────────────────────────────────────────────────
    // Inspector
    // ─────────────────────────────────────────────────────────────────────────

    [Header("Gate Geometry")]
    [Tooltip("Full interior opening width (X axis). Default = 5.5 m for a 9-drone swarm.")]
    public float gateWidth = 5.5f;

    [Tooltip("Full interior opening height (Y axis).")]
    public float gateHeight = 5.5f;

    [Tooltip("Cross-section thickness of the frame bars.")]
    public float barThickness = 0.2f;

    [Header("Swarm Settings")]
    [Tooltip("Total number of drones in the swarm. Used as fallback when swarmSpawnRef is null.")]
    public int swarmSize = 9;

    [Tooltip("Reference to the swarm spawner. When set, the gate dynamically queries the alive " +
             "drone count so dead drones are excluded from completion tracking.")]
    public swarmSpawn swarmSpawnRef;

    [Tooltip("Tag used to identify drone colliders.")]
    public string droneTag = "Player";

    [Header("References (auto-created if left empty)")]
    [Tooltip("Child transform placed at the geometric centre; used as spline node for path guidance.")]
    public Transform centerPoint;

    [Tooltip("Gate manager reference; used to check if in demo mode.")]
    public RingGateManager gateManager;

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
    [SerializeField] private CourseGenerator.SegmentType _gateType;

    /// <summary>Total drone passes recorded since last ResetMetrics().</summary>
    public int TotalPasses  => _totalPasses;
    /// <summary>Passes where the drone was within the gate aperture.</summary>
    public int InsidePasses => _insidePasses;
    /// <summary>Passes where the drone was outside the gate aperture.</summary>
    public int OutsidePasses => _outsidePasses;
    /// <summary>Fraction of passes that went through the aperture. 0–1.</summary>
    public float AccuracyRatio => _totalPasses > 0 ? (float)_insidePasses / _totalPasses : 0f;

    public int IsHard => _gateType == CourseGenerator.SegmentType.Hard ? 1 : 0;

    public CourseGenerator.SegmentType Type
    {
        get => _gateType;
        set
        {
            _gateType = value;
        }
    }

    /// <summary>Unix ms timestamp of the first confirmed pass in this activation window. 0 if none yet.</summary>
    public long FirstPassUnixMs { get; private set; }

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
        Log($"Initialised — aperture {gateWidth}×{gateHeight} m, swarm size {swarmSize}, tag '{droneTag}'.");
    }

    /// <summary>
    /// Polls completion while the gate is mid-activation so that a drone crashing
    /// before it crosses the exit plane does not permanently stall the gate.
    /// </summary>
    private void Update()
    {
        // Only relevant when at least one drone has passed but the gate is not yet complete.
        if (_passedThisActivation.Count == 0) return;
        if (gateManager != null && gateManager.DemoMode) return;

        CheckAndFireCompletion();
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
    // Swarm alive count
    // ─────────────────────────────────────────────────────────────────────────

    /// <summary>
    /// Returns the number of drones currently alive in the swarm.
    /// Lazy-discovers <see cref="swarmSpawn"/> from the scene if <see cref="swarmSpawnRef"/>
    /// is not explicitly assigned. Falls back to <see cref="swarmSize"/> only when no
    /// spawner can be found (logs a warning so the misconfiguration is visible).
    /// </summary>
    private int GetAliveSwarmCount()
    {
        // Lazy auto-discover so the gate works without a manual inspector connection.
        if (swarmSpawnRef == null)
            swarmSpawnRef = FindFirstObjectByType<swarmSpawn>();

        if (swarmSpawnRef == null)
        {
            Debug.LogWarning($"[RingGate] {gameObject.name}: swarmSpawnRef not found — " +
                             $"falling back to swarmSize ({swarmSize}). Assign swarmSpawnRef " +
                             $"on the RingGateManager for accurate alive-drone tracking.");
            return swarmSize;
        }

        int count = 0;
        foreach (var droneObj in swarmSpawnRef.swarm)
        {
            if (droneObj == null) continue;
            StateFinder sf = droneObj.GetComponentInChildren<StateFinder>();
            if (sf != null && sf.IsAlive)
                count++;
        }

        if (count == 0)
        {
            // No alive drones found — use total spawned count so the gate does not
            // instantly complete. This is a safety net; it should not occur in normal play.
            int total = swarmSpawnRef.swarm.Count;
            Debug.LogWarning($"[RingGate] {gameObject.name}: GetAliveSwarmCount found 0 alive drones " +
                             $"(total spawned: {total}) — using total spawned count as target.");
            return Mathf.Max(1, total);
        }

        return count;
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
        // In demo mode: just advance gate visual without recording metrics/events
        if (gateManager != null && gateManager.DemoMode)
        {
            gateManager.HandleDemoPass(this);
            return;
        }

        // ── Skip dead drones ─────────────────────────────────────────────────
        VelocityControl vc = droneRoot.GetComponent<VelocityControl>();
        if (vc != null && vc.State != null && !vc.State.IsAlive)
        {
            LogRejection($"Dead drone '{droneRoot.name}' ignored — not counted toward gate pass.");
            return;
        }

        // ── Inside/outside check (rectangular gate) ─────────────────────────
        Vector3 localPos   = transform.InverseTransformPoint(droneRoot.transform.position);
        float   radialDist = new Vector2(localPos.x, localPos.y).magnitude;
        bool    isInside   = Mathf.Abs(localPos.x) <= gateWidth * 0.5f &&
                             Mathf.Abs(localPos.y) <= gateHeight * 0.5f;

        // If first drone, make sure it is inside the gate to avoid displaying plane on next gate
        if (_totalPasses == 0 && !isInside)
        {
            LogRejection($"First drone '{droneRoot.name}' is outside the gate — ignoring pass.");
            return;
        }

        int droneId = droneRoot.GetInstanceID();

        if (_passedThisActivation.Contains(droneId))
        {
            LogRejection($"Duplicate pass ignored for '{droneRoot.name}' (already counted in this activation window).");
            return;
        }
        _passedThisActivation.Add(droneId);

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

        // ── Check if all alive swarm drones have now passed ──────────────────
        CheckAndFireCompletion();
    }

    /// <summary>
    /// Fires <see cref="onAllDronesPassed"/> if every alive drone has now cleared this gate.
    /// Called both from <see cref="RecordConfirmedPass"/> (normal path) and from
    /// <see cref="Update"/> (crash-recovery path — catches the case where the last
    /// expected drone dies before reaching the exit plane).
    /// </summary>
    private void CheckAndFireCompletion()
    {
        int aliveCount = GetAliveSwarmCount();
        LogProgress(_passedThisActivation.Count, aliveCount);

        if (_passedThisActivation.Count >= aliveCount)
        {
            LogGateCleared();
            FirstPassUnixMs = System.DateTimeOffset.UtcNow.ToUnixTimeMilliseconds();
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
            wasInsideRing  = Mathf.Abs(localPos.x) <= gateWidth * 0.5f &&
                             Mathf.Abs(localPos.y) <= gateHeight * 0.5f,
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
        FirstPassUnixMs = 0;

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
                  $"(gate={gateWidth:F2}×{gateHeight:F2} m)  " +
                  $"t={d.timestamp:F2}s");
    }

    private void LogProgress(int count, int aliveCount)
    {
        if (!debugEnabled || !debugLogSwarmProgress) return;
        Debug.Log($"{GateLabel} Swarm progress: {count} / {aliveCount} alive drones cleared.");
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
        float hw = gateWidth * 0.5f;
        float hh = gateHeight * 0.5f;

        // Compute world-space corners of the gate rectangle
        Vector3 tl = transform.TransformPoint(new Vector3(-hw,  hh, 0f));
        Vector3 tr = transform.TransformPoint(new Vector3( hw,  hh, 0f));
        Vector3 bl = transform.TransformPoint(new Vector3(-hw, -hh, 0f));
        Vector3 br = transform.TransformPoint(new Vector3( hw, -hh, 0f));

        // Aperture rectangle
        Gizmos.color = new Color(0f, 0.9f, 1f, 0.9f);
        Gizmos.DrawLine(tl, tr);
        Gizmos.DrawLine(tr, br);
        Gizmos.DrawLine(br, bl);
        Gizmos.DrawLine(bl, tl);

        // Centre point indicator
        if (centerPoint != null)
        {
            Gizmos.color = Color.yellow;
            Gizmos.DrawSphere(centerPoint.position, 0.12f);
            Gizmos.DrawLine(transform.position, centerPoint.position);
        }

        // Gate normal axis (direction of travel)
        Gizmos.color = Color.green;
        Gizmos.DrawRay(transform.position, transform.forward * (Mathf.Max(gateWidth, gateHeight) * 0.3f));
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

    /// <summary>True if the drone was within the gate aperture (inside the rectangle).</summary>
    public bool wasInsideRing;

    /// <summary>The gate that recorded this pass.</summary>
    public RingGate ringGate;

    /// <summary>Zero-based index of this pass within the gate's total pass count.</summary>
    public int passIndex;

    /// <summary>Time.time when the pass was registered.</summary>
    public float timestamp;
}
