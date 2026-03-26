using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Events;

/// <summary>
/// Course-level manager. Holds an ordered list of all <see cref="RingGate"/>
/// instances and aggregates metrics across them for CWL / performance logging.
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

    [Header("Events")]
    [Tooltip("Fires when every drone has cleared every gate (run complete).")]
    public UnityEvent<CourseRunSummary> onRunComplete;

    [Tooltip("Fires after each individual gate is fully cleared by all drones.")]
    public UnityEvent<int, RingGate> onGateCleared;   // (gateIndex, gate)

    // ─────────────────────────────────────────────────────────────────────────
    // Private
    // ─────────────────────────────────────────────────────────────────────────

    private int _clearedGateCount;

    // ─────────────────────────────────────────────────────────────────────────
    // Lifecycle
    // ─────────────────────────────────────────────────────────────────────────

    private void Awake()
    {
        if (autoDiscoverOnAwake) AutoDiscoverGates();
        RegisterGateCallbacks();
    }

    private void RegisterGateCallbacks()
    {
        for (int i = 0; i < gates.Count; i++)
        {
            if (gates[i] == null) continue;
            int capturedIndex = i;  // capture for lambda
            gates[i].onAllDronesPassed.AddListener(gate => HandleGateCleared(capturedIndex, gate));
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Gate cleared callback
    // ─────────────────────────────────────────────────────────────────────────

    private void HandleGateCleared(int gateIndex, RingGate gate)
    {
        _clearedGateCount++;
        onGateCleared?.Invoke(gateIndex, gate);

        if (_clearedGateCount >= gates.Count)
        {
            var summary = BuildSummary();
            onRunComplete?.Invoke(summary);
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Public API
    // ─────────────────────────────────────────────────────────────────────────

    /// <summary>Reset all gates and the cleared-gate counter. Call between runs.</summary>
    public void ResetAll()
    {
        _clearedGateCount = 0;
        foreach (var gate in gates)
            gate?.ResetMetrics();
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
