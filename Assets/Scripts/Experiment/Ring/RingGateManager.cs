using System.Collections.Generic;
using System.Linq;
using UnityEngine;
using UnityEngine.Events;

/// <summary>
/// Course-level manager. Holds an ordered list of all <see cref="RingGate"/>
/// instances and aggregates metrics across them for CWL / performance logging.
///
/// Gate advancement logic:
///   - The "Next" visual advances to the following gate as soon as the FIRST
///     drone passes through the current gate (not when all drones pass).
///   - A gate that has been advanced past but not fully completed shows as
///     PartialComplete (yellow). It upgrades to Completed (green) when
///     all swarmSize drones have passed through.
///   - The course finishes when every gate has been fully completed AND the
///     cursor has moved past the last gate.
///
/// Drop this on a single "CourseManager" GameObject in the scene and populate
/// the <see cref="gates"/> list (or call <see cref="AutoDiscoverGates"/>).
/// </summary>
public class RingGateManager : MonoBehaviour
{
    // ─────────────────────────────────────────────────────────────────────────
    // Inspector
    // ─────────────────────────────────────────────────────────────────────────

    [Header("Gates (ordered by course position)")]
    public List<RingGate> gates = new();

    [Tooltip("If true, finds all RingGate components in the scene on Awake " +
             "and adds any not already in the list.")]
    public bool autoDiscoverOnAwake = false;

    [Header("Course Generation")]
    [SerializeField] private CourseGenerator courseGenerator;

    [Tooltip("Swarm spawner reference. Injected into every registered gate so they can " +
             "query the alive drone count instead of using a fixed swarm size.")]
    public swarmSpawn swarmSpawnRef;

    [Header("Events")]
    [Tooltip("Fires when every drone has cleared every gate (run complete).")]
    public UnityEvent<CourseRunSummary> onRunComplete;

    [Tooltip("Fires after each individual gate is fully cleared by all drones.")]
    public UnityEvent<int, RingGate> onGateCleared;   // (gateIndex, gate)

    [Tooltip("Fires when the active gate advances. Args: (newGateIndex, newActiveGate).")]
    public UnityEvent<int, RingGate> onNextGateChanged;

    [Header("Visual Overrides")]
    [Tooltip("When false, all gates stay in Idle state (gray) regardless of course progress.")]
    [SerializeField] private bool _enableGateColorStates = true;

    [Tooltip("Number of subsequent gates to show ahead of the current gate. " +
             "Gates beyond this distance are hidden. Default = 4.")]
    [Min(1)] public int visibleGatesAhead = 4;

    [Header("Course State")]
    [SerializeField] private CourseTimer _timer;
    [SerializeField] private bool _showLapTimer = false;
    public bool ShowLapTimer
    {
        get => _showLapTimer;
        set
        {_showLapTimer = value;
            if (_timer != null)
            {
                _timer.ShowText = value;
            }
        }
    }

    /// <summary>Enable/disable demo mode to highlight gates for the CourseDemo.</summary>
    public bool DemoMode
    {
        get => _demoMode;
        set => _demoMode = value;
    }

    /// <summary>
    /// Initializes the course for demo: resets visuals, sets gate 0 as Next,
    /// but does NOT start the timer or mark the course as running.
    /// </summary>
    public void StartDemoMode()
    {
        _demoMode = true;
        ResetAll();
        _currentGateIndex = 0;

        for (int i = 0; i < _gateVisuals.Count; i++)
            if (_gateVisuals[i] != null)
                _gateVisuals[i].SetState(GateVisualState.Idle);

        if (gates.Count > 0 && _gateVisuals[0] != null)
        {
            _gateVisuals[0].SetState(GateVisualState.Next);
            UpdateGateVisibility();
            Debug.Log("[RingGateManager] Demo mode started. Gate 0 set as Next.");
        }
        else
        {
            Debug.LogWarning($"[RingGateManager] StartDemoMode: gates={gates.Count}, visual[0]={(gates.Count > 0 ? _gateVisuals[0]?.ToString() ?? "null" : "no gates")}");
        }
    }

    /// <summary>Set which gate is highlighted as "next" during demo mode.</summary>
    public void SetDemoNextGate(int gateIndex)
    {
        if (!_demoMode || gateIndex < 0 || gateIndex >= gates.Count)
            return;

        // Clear previous "next" gate visuals
        for (int i = 0; i < _gateVisuals.Count; i++)
            if (_gateVisuals[i] != null && _gateVisuals[i].CurrentState == GateVisualState.Next)
                _gateVisuals[i].SetState(GateVisualState.Idle);

        // Set new "next" gate
        if (_gateVisuals[gateIndex] != null)
            _gateVisuals[gateIndex].SetState(GateVisualState.Next);
    }

    /// <summary>Handle a demo pass: advance gate visually without recording metrics.</summary>
    public void HandleDemoPass(RingGate passedGate)
    {
        if (!_demoMode)
            return;

        // Find which gate was passed
        int passedIndex = gates.IndexOf(passedGate);
        if (passedIndex < 0)
            return;

        // Allow advancement from -1 (uninitialized) to 0 (first gate) for demo init
        if (_currentGateIndex >= 0 && passedIndex != _currentGateIndex)
            return;

        // Deactivate current gate visual if it was set
        if (_currentGateIndex >= 0 && _currentGateIndex < _gateVisuals.Count)
            if (_gateVisuals[_currentGateIndex] != null)
                _gateVisuals[_currentGateIndex].SetState(GateVisualState.Idle);

        // Advance to next gate
        _currentGateIndex = passedIndex + 1;
        if (_currentGateIndex < gates.Count && _gateVisuals[_currentGateIndex] != null)
        {
            _gateVisuals[_currentGateIndex].SetState(GateVisualState.Next);
            Debug.Log($"[CourseDemo] Gate {_currentGateIndex} is next");
        }

        UpdateGateVisibility();
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Private
    // ─────────────────────────────────────────────────────────────────────────

    private int _currentGateIndex = -1;  // -1 = not started
    private bool _isCourseRunning = false;
    private bool _demoMode = false;
    private List<RingGateVisual> _gateVisuals = new();

    // Tracking arrays — sized to match gates list
    private bool[] _gateFullyCompleted;  // true when onAllDronesPassed fired
    private bool[] _gateAdvanced;        // true once first drone triggered advancement

    // ─────────────────────────────────────────────────────────────────────────
    // Lifecycle
    // ─────────────────────────────────────────────────────────────────────────

    private void Awake()
    {
        if (autoDiscoverOnAwake) AutoDiscoverGates();
        CacheGateVisuals();
        InitTrackingArrays();
        RegisterGateCallbacks();
        SyncColorToggleToVisuals();
    }

    private void CacheGateVisuals()
    {
        _gateVisuals.Clear();
        foreach (var gate in gates)
        {
            if (gate != null)
            {
                gate.gateManager = this;
                gate.swarmSpawnRef = swarmSpawnRef;
                _gateVisuals.Add(gate.GetComponent<RingGateVisual>());
            }
            else
            {
                _gateVisuals.Add(null);
            }
        }
    }

    private void InitTrackingArrays()
    {
        _gateFullyCompleted = new bool[gates.Count];
        _gateAdvanced = new bool[gates.Count];
    }

    private void RegisterGateCallbacks()
    {
        for (int i = 0; i < gates.Count; i++)
        {
            if (gates[i] == null) continue;
            int capturedIndex = i;
            gates[i].onDronePassed.AddListener(data => HandleFirstDronePassed(capturedIndex, data));
            gates[i].onAllDronesPassed.AddListener(gate => HandleAllDronesPassed(capturedIndex, gate));
        }
    }

    private void SyncColorToggleToVisuals()
    {
        foreach (var v in _gateVisuals)
            if (v != null) v.ColorStatesEnabled = _enableGateColorStates;
    }

    // ─────────────────────────────────────────────────────────────────────────
    // First drone passed — advance visual cursor to next gate
    // ─────────────────────────────────────────────────────────────────────────

    private void HandleFirstDronePassed(int gateIndex, RingPassData data)
    {
        if (!_isCourseRunning) return;
        if (gateIndex != _currentGateIndex) return;
        if (_gateAdvanced[gateIndex]) return;

        _gateAdvanced[gateIndex] = true;

        // First drone passed — mark gate as PartialComplete (yellow) so both
        // Unity visuals and the Python side see state=2 while remaining drones clear.
        if (_gateVisuals[gateIndex] != null)
        {
            _gateVisuals[gateIndex].SetState(GateVisualState.PartialComplete);
        }

        _currentGateIndex++;

        if (_currentGateIndex < gates.Count)
        {
            _gateVisuals[_currentGateIndex]?.SetState(GateVisualState.Next);
            onNextGateChanged?.Invoke(_currentGateIndex, gates[_currentGateIndex]);
            UpdateGateVisibility();
        }

        NotifyGateStatus();
    }

    // ─────────────────────────────────────────────────────────────────────────
    // All drones passed — confirm full completion (may fire after advancement)
    // ─────────────────────────────────────────────────────────────────────────

    private void HandleAllDronesPassed(int gateIndex, RingGate gate)
    {
        if (!_isCourseRunning) return;
        if (_gateFullyCompleted[gateIndex]) return;

        _gateFullyCompleted[gateIndex] = true;

        // Always mark as Completed (state=3) since all alive drones passed.
        // The visual color (green vs yellow) is handled by RingGateVisual
        // based on inside-pass accuracy.
        _gateVisuals[gateIndex]?.SetState(GateVisualState.Completed);
        _timer?.RecordGateSplit();
        onGateCleared?.Invoke(gateIndex, gate);

        // Check course completion: all gates fully complete AND cursor past the end
        if (_gateFullyCompleted.All(x => x) && _currentGateIndex >= gates.Count)
        {
            _isCourseRunning = false;
            _timer?.StopTimer();
            var summary = BuildSummary();
            onRunComplete?.Invoke(summary);
        }

        NotifyGateStatus();
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Gate Visibility
    // ─────────────────────────────────────────────────────────────────────────

    /// <summary>
    /// Updates the visibility of gates based on distance from the current active gate.
    /// Shows all passed gates (index < currentGateIndex) + current gate + visibleGatesAhead future gates.
    /// Hides gates beyond the lookahead window.
    /// </summary>
    private void UpdateGateVisibility()
    {
        for (int i = 0; i < gates.Count; i++)
        {
            // Show: all passed gates + current gate + next visibleGatesAhead gates
            bool shouldBeVisible = i < _currentGateIndex + visibleGatesAhead;
            gates[i].gameObject.SetActive(shouldBeVisible);
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Public API
    // ─────────────────────────────────────────────────────────────────────────

    /// <summary>Reset all gates and the cleared-gate counter. Call between runs.</summary>
    public void ResetAll()
    {
        _currentGateIndex = -1;
        _isCourseRunning = false;

        if (_gateFullyCompleted != null)
            System.Array.Clear(_gateFullyCompleted, 0, _gateFullyCompleted.Length);
        if (_gateAdvanced != null)
            System.Array.Clear(_gateAdvanced, 0, _gateAdvanced.Length);

        foreach (var gate in gates)
        {
            gate?.ResetMetrics();
            gate?.gameObject.SetActive(true);
        }

        for (int i = 0; i < _gateVisuals.Count; i++)
            if (_gateVisuals[i] != null)
                _gateVisuals[i].SetState(GateVisualState.Idle);

        _timer?.Reset();
        NotifyGateStatus();
    }

    /// <summary>
    /// Begins the ordered course run. Called by CourseStartTrigger or directly.
    /// Resets all gate metrics and visuals, then activates the first gate.
    /// Safe to call multiple times — re-arms the course for a new run.
    /// </summary>
    public void StartCourse()
    {
        if (_isCourseRunning)
            Debug.LogWarning("[RingGateManager] StartCourse() called while course is already running. Re-starting.");

        ResetAll();
        _currentGateIndex = 0;
        _isCourseRunning = true;

        Debug.Log($"[RingGateManager] StartCourse: gates.Count={gates.Count}, _gateVisuals.Count={_gateVisuals.Count}");

        for (int i = 0; i < _gateVisuals.Count; i++)
            if (_gateVisuals[i] != null)
                _gateVisuals[i].SetState(GateVisualState.Idle);

        if (gates.Count > 0)
        {
            if (_gateVisuals[0] != null)
            {
                Debug.Log($"[RingGateManager] Setting gate 0 visual to Next state");
                _gateVisuals[0].SetState(GateVisualState.Next);
            }
            else
            {
                Debug.LogWarning("[RingGateManager] Gate 0 visual is null!");
            }
            onNextGateChanged?.Invoke(0, gates[0]);
            UpdateGateVisibility();
        }

        _timer?.StartTimer();
        NotifyGateStatus();

        Debug.Log("[RingGateManager] Course started.");
    }

    /// <summary>True between StartCourse() and the last gate being cleared.</summary>
    public bool IsCourseRunning => _isCourseRunning;

    /// <summary>Zero-based index of the gate currently expected to be cleared next. -1 if not started.</summary>
    public int CurrentGateIndex => _currentGateIndex;

    /// <summary>
    /// Generates a new course by clearing all existing gates and generating a fresh layout.
    /// Updates the shared memory with the new gate configuration.
    /// Call this before starting a new trial or when a fresh course layout is needed.
    /// </summary>
    public void GenerateNewCourse()
    {
        if (courseGenerator == null)
        {
            Debug.LogError("[RingGateManager] CourseGenerator is not assigned. Cannot generate new course.");
            return;
        }

        // Clear any running course state
        if (_isCourseRunning)
        {
            Debug.LogWarning("[RingGateManager] GenerateNewCourse() called during active course. Aborting run.");
            _isCourseRunning = false;
        }

        // Generate the new course (includes clearing gates, instantiating, and registering)
        courseGenerator.GenerateCourse();

        // Update shared memory with new gate status
        NotifyGateStatus();

        Debug.Log("[RingGateManager] New course generated and shared memory updated.");
    }

    /// <summary>
    /// Finds all RingGate components in the scene and appends any not already listed.
    /// Useful during development; disable in production builds.
    /// </summary>
    public void AutoDiscoverGates()
    {
        var found = FindObjectsByType<RingGate>(FindObjectsSortMode.InstanceID);
        foreach (var g in found)
        {
            if (!gates.Contains(g))
                gates.Add(g);
        }
        Debug.Log($"[RingGateManager] Auto-discovered {found.Length} gates " +
                  $"({gates.Count} total in list).");
    }

    /// <summary>Toggle gate color states (Next/Completed) on or off at runtime.</summary>
    public bool EnableGateColorStates
    {
        get => _enableGateColorStates;
        set
        {
            _enableGateColorStates = value;
            SyncColorToggleToVisuals();
        }
    }

    /// <summary>
    /// Destroys all managed gate GameObjects, clears internal lists and callbacks.
    /// Safe to call before procedurally regenerating a course.
    /// </summary>
    public void ClearGeneratedGates()
    {
        if (_isCourseRunning)
        {
            Debug.LogWarning("[RingGateManager] ClearGeneratedGates() called during active run. Aborting run.");
            _isCourseRunning = false;
        }

        // Remove listeners before destroying
        foreach (var gate in gates)
        {
            if (gate != null)
            {
                gate.onAllDronesPassed.RemoveAllListeners();
                gate.onDronePassed.RemoveAllListeners();
            }
        }

        // Destroy GameObjects
        foreach (var gate in gates)
            if (gate != null)
                Destroy(gate.gameObject);

        gates.Clear();
        _gateVisuals.Clear();
        _gateFullyCompleted = System.Array.Empty<bool>();
        _gateAdvanced = System.Array.Empty<bool>();
        _currentGateIndex = -1;
        _timer?.Reset();
    }

    /// <summary>
    /// Registers a single gate into the managed course. Call once per gate after instantiation,
    /// then call CoursePathVisual.RebuildPath() when all gates are registered.
    /// </summary>
    public void RegisterGate(RingGate gate)
    {
        if (gate == null) return;

        int index = gates.Count;
        gates.Add(gate);

        gate.gateManager = this;
        gate.swarmSpawnRef = swarmSpawnRef;

        var visual = gate.GetComponent<RingGateVisual>();
        _gateVisuals.Add(visual);

        if (visual != null)
            visual.ColorStatesEnabled = _enableGateColorStates;

        // Expand tracking arrays
        System.Array.Resize(ref _gateFullyCompleted, gates.Count);
        System.Array.Resize(ref _gateAdvanced, gates.Count);

        gate.onDronePassed.AddListener(data => HandleFirstDronePassed(index, data));
        gate.onAllDronesPassed.AddListener(g => HandleAllDronesPassed(index, g));
    }

    /// <summary>Returns an ordered array of CenterPoint transforms — ready to feed a spline.</summary>
    public Transform[] GetCenterPoints()
    {
        var pts = new Transform[gates.Count];
        for (int i = 0; i < gates.Count; i++)
            pts[i] = gates[i]?.centerPoint;
        return pts;
    }

    /// <summary>Aggregate accuracy across all gates: insidePasses / totalPasses.</summary>
    public float GetOverallAccuracy()
    {
        int totalIn  = 0, totalAll = 0;
        foreach (var g in gates)
        {
            if (g == null) continue;
            totalIn  += g.InsidePasses;
            totalAll += g.TotalPasses;
        }
        return totalAll > 0 ? (float)totalIn / totalAll : 0f;
    }

    public bool IsCourseCompleted => _gateFullyCompleted != null && _gateFullyCompleted.All(x => x);

    // ─────────────────────────────────────────────────────────────────────────
    // Shared memory notification
    // ─────────────────────────────────────────────────────────────────────────

    private void NotifyGateStatus()
    {
        if (PySender.Instance != null)
            PySender.Instance.UpdateGateStatus(this);
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Summary builder
    // ─────────────────────────────────────────────────────────────────────────

    private CourseRunSummary BuildSummary()
    {
        var summary = new CourseRunSummary
        {
            gateCount        = gates.Count,
            overallAccuracy  = GetOverallAccuracy(),
            gateResults      = new List<GateResult>()
        };

        for (int i = 0; i < gates.Count; i++)
        {
            var g = gates[i];
            if (g == null) continue;
            summary.gateResults.Add(new GateResult
            {
                gateIndex    = i,
                gateName     = g.gameObject.name,
                totalPasses  = g.TotalPasses,
                insidePasses = g.InsidePasses,
                outsidePasses = g.OutsidePasses,
                accuracy     = g.AccuracyRatio
            });
        }

        return summary;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Data transfer objects
// ─────────────────────────────────────────────────────────────────────────────

[System.Serializable]
public class CourseRunSummary
{
    public int              gateCount;
    public float            overallAccuracy;
    public List<GateResult> gateResults;
}

[System.Serializable]
public class GateResult
{
    public int    gateIndex;
    public string gateName;
    public int    totalPasses;
    public int    insidePasses;
    public int    outsidePasses;
    public float  accuracy;
}
