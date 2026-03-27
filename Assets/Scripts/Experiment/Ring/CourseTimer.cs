using System.Collections.Generic;
using TMPro;
using UnityEngine;

/// <summary>
/// Pure data service for tracking course run times: total elapsed and per-gate splits.
/// Does not subscribe to any events — RingGateManager calls the API directly.
/// Optionally displays live timer to a TextMeshProUGUI element.
/// </summary>
public class CourseTimer : MonoBehaviour
{
    // ─────────────────────────────────────────────────────────────────────────
    // Inspector
    // ─────────────────────────────────────────────────────────────────────────

    [Header("Display (optional)")]
    [Tooltip("If assigned, updates every frame while the timer is running.")]
    public TextMeshProUGUI timerDisplay;

    [Tooltip("Format string. {0} = TimeSpan (e.g. 'MM:SS.ff'). Default: mm\\:ss\\.ff")]
    public string displayFormat = "{0:mm\\:ss\\.ff}";

    // ─────────────────────────────────────────────────────────────────────────
    // Properties
    // ─────────────────────────────────────────────────────────────────────────

    /// <summary>Seconds elapsed since StartTimer() was called.</summary>
    public float TotalElapsed { get; private set; }

    /// <summary>Per-gate split times in seconds, in order of gate clearance.</summary>
    public List<float> GateSplits { get; private set; } = new();

    /// <summary>True between StartTimer() and StopTimer().</summary>
    public bool IsRunning { get; private set; }

    // ─────────────────────────────────────────────────────────────────────────
    // Private
    // ─────────────────────────────────────────────────────────────────────────

    private float _splitStart;  // Time.time at the start of the current gate split

    // ─────────────────────────────────────────────────────────────────────────
    // Lifecycle
    // ─────────────────────────────────────────────────────────────────────────

    private void Update()
    {
        if (!IsRunning) return;
        TotalElapsed += Time.deltaTime;
        UpdateDisplay();
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Public API
    // ─────────────────────────────────────────────────────────────────────────

    /// <summary>Starts or restarts the timer and clears all split data.</summary>
    public void StartTimer()
    {
        TotalElapsed = 0f;
        GateSplits.Clear();
        _splitStart = Time.time;
        IsRunning = true;
        UpdateDisplay();
    }

    /// <summary>
    /// Records the elapsed time since the last split (or since StartTimer if this
    /// is the first gate), and begins timing the next gate.
    /// </summary>
    public void RecordGateSplit()
    {
        float splitTime = Time.time - _splitStart;
        GateSplits.Add(splitTime);
        _splitStart = Time.time;
        Debug.Log($"[CourseTimer] Split {GateSplits.Count}: {splitTime:F3}s  " +
                  $"(total: {TotalElapsed:F3}s)");
    }

    /// <summary>Stops the timer. TotalElapsed remains at its final value.</summary>
    public void StopTimer()
    {
        IsRunning = false;
        UpdateDisplay();
        Debug.Log($"[CourseTimer] Stopped. Total: {TotalElapsed:F3}s  Splits: " +
                  string.Join(", ", GateSplits.ConvertAll(s => s.ToString("F3"))));
    }

    /// <summary>Resets the timer to zero without starting it.</summary>
    public void Reset()
    {
        IsRunning = false;
        TotalElapsed = 0f;
        GateSplits.Clear();
        UpdateDisplay();
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Internal
    // ─────────────────────────────────────────────────────────────────────────

    private void UpdateDisplay()
    {
        if (timerDisplay == null) return;
        var span = System.TimeSpan.FromSeconds(TotalElapsed);
        timerDisplay.text = string.Format(displayFormat, span);
    }
}
