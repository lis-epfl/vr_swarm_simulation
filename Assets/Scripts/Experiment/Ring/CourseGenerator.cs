using System.Collections.Generic;
using UnityEngine;
using Random = UnityEngine.Random;

/// <summary>
/// Procedurally generates a ring gate course composed of alternating Easy and Hard segments.
///
/// Easy segments: gentle arcs with small angle changes between gates.
/// Hard segments: S-shaped paths with sharp direction reversals mid-segment.
///
/// Call <see cref="GenerateCourse"/> at runtime (or from Start) to build a new course.
/// Call <see cref="RegenerateCourse"/> between trials for a fresh random layout.
/// </summary>
public class CourseGenerator : MonoBehaviour
{
    // ─────────────────────────────────────────────────────────────────────────
    // Inspector
    // ─────────────────────────────────────────────────────────────────────────

    [Header("Course Structure")]
    [Tooltip("Number of segments in the course (alternating Easy/Hard).")]
    public int segmentCount = 6;

    [Tooltip("Number of ring gates per segment.")]
    public int ringsPerSegment = 3;

    [Tooltip("If true the first segment is Easy (E-H-E-H...), otherwise Hard first.")]
    public bool startWithEasy = true;

    [Header("Easy Segment Ranges")]
    [Tooltip("Minimum Z-distance progression between consecutive rings (units).")]
    public float easyZDistanceMin = 8f;
    [Tooltip("Maximum Z-distance progression between consecutive rings (units).")]
    public float easyZDistanceMax = 12f;

    [Tooltip("Minimum absolute lateral angle deviation (degrees). Actual range: ±[Min, Max]. Left/right.")]
    public float easyLateralAngleMin = 0f;
    [Tooltip("Maximum absolute lateral angle deviation (degrees). Actual range: ±[Min, Max]. Left/right.")]
    public float easyLateralAngleMax = 5f;

    [Tooltip("Minimum absolute pitch angle deviation (degrees). Actual range: ±[Min, Max]. Up/down.")]
    public float easyPitchAngleMin = 0f;
    [Tooltip("Maximum absolute pitch angle deviation (degrees). Actual range: ±[Min, Max]. Up/down.")]
    public float easyPitchAngleMax = 2f;

    [Header("Hard Segment Ranges")]
    [Tooltip("Minimum Z-distance progression between consecutive rings (units).")]
    public float hardZDistanceMin = 6f;
    [Tooltip("Maximum Z-distance progression between consecutive rings (units).")]
    public float hardZDistanceMax = 10f;

    [Tooltip("Minimum absolute lateral angle deviation (degrees). Actual range: ±[Min, Max]. Left/right. Creates S-curves.")]
    public float hardLateralAngleMin = 15f;
    [Tooltip("Maximum absolute lateral angle deviation (degrees). Actual range: ±[Min, Max]. Left/right. Creates S-curves.")]
    public float hardLateralAngleMax = 30f;

    [Tooltip("Minimum absolute pitch angle deviation (degrees). Actual range: ±[Min, Max]. Up/down.")]
    public float hardPitchAngleMin = 5f;
    [Tooltip("Maximum absolute pitch angle deviation (degrees). Actual range: ±[Min, Max]. Up/down.")]
    public float hardPitchAngleMax = 12f;

    [Header("Constraints")]
    [Tooltip("Minimum height above terrain for any ring gate placement.")]
    public float minHeightAboveTerrain = 3f;
    [Tooltip("Extra forward gap (units) inserted between consecutive segments. " +
             "Increase to create breathing room between the last easy gate and first hard gate, and vice versa.")]
    public float segmentTransitionBuffer = 5f;

    [Header("References")]
    public RingGateManager gateManager;
    public CoursePathVisual pathVisual;
    public GameObject ringGatePrefab;
    [Tooltip("Start position and forward direction for the course.")]
    public Transform courseStartPoint;

    [Header("Auto-generate")]
    [Tooltip("If true, generates a course automatically on Start().")]
    public bool generateOnStart = true;

    [Header("Advanced")]
    [Tooltip("If true, aligns hard-segment gates to the spline tangent (yaw only). Gates remain vertical.")]
    public bool useSplineYawAlignment = true;
    [Tooltip("If true, validates course geometry and logs suggestions for S-shape design.")]
    public bool enableCourseValidation = true;

    // ─────────────────────────────────────────────────────────────────────────
    // Internal types
    // ─────────────────────────────────────────────────────────────────────────

    public enum SegmentType { Easy, Hard }

    private struct PlacedRing
    {
        public Vector3 position;
        public Quaternion rotation;
        public int segmentIndex;
        public bool isHard;
    }

    private struct SegmentState
    {
        public SegmentType type;
        public int yawSign;        // +1 or -1, constant within Easy; reverses at midpoint for Hard
        public int pitchSign;      // +1 or -1, constant within Easy; reverses at midpoint for Hard
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Private state
    // ─────────────────────────────────────────────────────────────────────────

    private readonly List<GameObject> _spawnedGates = new();
    private List<SegmentState> _segmentStates;
    private List<PlacedRing> _lastGeneratedPlacements = new();  // Stored for gizmo visualization
    private List<Vector3> _lastComputedTangents = new();        // Spline tangent directions for each gate

    // ─────────────────────────────────────────────────────────────────────────
    // Lifecycle
    // ─────────────────────────────────────────────────────────────────────────

    private void Start()
    {
        if (generateOnStart)
            GenerateCourse();
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Public API
    // ─────────────────────────────────────────────────────────────────────────

    /// <summary>
    /// Generates course placement data only (for gizmo preview).
    /// Does NOT instantiate actual gate GameObjects.
    /// Call this from the editor to preview the course layout.
    /// </summary>
    public void GenerateCoursePlan()
    {
        if (!ValidateReferences()) return;

        // Clean up previous preview
        _spawnedGates.Clear();

        // Pre-compute per-segment randomisation
        _segmentStates = BuildSegmentStates();

        // Compute ring placements (pure data — no instantiation)
        List<PlacedRing> placements = ComputeRingPlacements();

        // Store for gizmo visualization
        _lastGeneratedPlacements = new List<PlacedRing>(placements);

        // Validate course geometry if enabled
        if (enableCourseValidation)
            ValidateCourseGeometry();

        Debug.Log($"[CourseGenerator] Course plan generated: {placements.Count} gates " +
                  $"across {segmentCount} segments ({ringsPerSegment} rings each). " +
                  $"Call InstantiateCourseFinal() to create actual gates.");
    }

    /// <summary>
    /// Instantiates actual gate GameObjects from the last generated course plan.
    /// Call this when the simulation starts to create the real gates.
    /// </summary>
    public void InstantiateCourseFinal()
    {
        if (!ValidateReferences()) return;

        // Clean up any old gates
        gateManager.ClearGeneratedGates();
        DestroySpawnedGates();

        // Instantiate and register each gate from stored placements
        for (int i = 0; i < _lastGeneratedPlacements.Count; i++)
        {
            PlacedRing ring = _lastGeneratedPlacements[i];
            RingGate gate = InstantiateGate(ring, i);
            gate.Type = ring.isHard ? SegmentType.Hard : SegmentType.Easy;  // Set gate type for metrics and visualization
            gateManager.RegisterGate(gate);
        }

        // Rebuild the spline path visual
        if (pathVisual != null)
            pathVisual.RebuildPath();

        // Write static gate layout to shared memory (Python reads once for live plotting)
        if (PySender.Instance != null)
            PySender.Instance.WriteGateLayout(gateManager.gates);

        Debug.Log($"[CourseGenerator] Instantiated {_lastGeneratedPlacements.Count} gates from course plan.");
    }

    /// <summary>
    /// Generates and instantiates a course in one call.
    /// Combines GenerateCoursePlan() + InstantiateCourseFinal().
    /// </summary>
    public void GenerateCourse()
    {
        GenerateCoursePlan();
        InstantiateCourseFinal();
    }

    /// <summary>Alias for GenerateCourse(). Call between trials for a fresh layout.</summary>
    public void RegenerateCourse()
    {
        GenerateCourse();
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Segment state generation
    // ─────────────────────────────────────────────────────────────────────────

    private List<SegmentState> BuildSegmentStates()
    {
        var states = new List<SegmentState>(segmentCount);
        for (int seg = 0; seg < segmentCount; seg++)
        {
            bool isEven = seg % 2 == 0;
            bool isEasy = startWithEasy ? isEven : !isEven;

            var state = new SegmentState
            {
                type = isEasy ? SegmentType.Easy : SegmentType.Hard,
                yawSign = Random.value > 0.5f ? 1 : -1,
                pitchSign = Random.value > 0.5f ? 1 : -1,
            };

            states.Add(state);
        }
        return states;
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Ring placement computation
    // ─────────────────────────────────────────────────────────────────────────

    private List<PlacedRing> ComputeRingPlacements()
    {
        var placements = new List<PlacedRing>(segmentCount * ringsPerSegment);

        Vector3 currentPos = courseStartPoint.position;

        // Base forward is Z-axis (consistent progression in +Z direction).
        // Horizontal forward is the direction the rings "face" (may deviate due to lateral sway).
        Vector3 baseForward = Vector3.ProjectOnPlane(courseStartPoint.forward, Vector3.up).normalized;
        if (baseForward.sqrMagnitude < 0.001f)
            baseForward = Vector3.forward;

        // For lateral sway calculation, we need a perpendicular axis (left/right).
        Vector3 leftAxis = Vector3.Cross(Vector3.up, baseForward).normalized;

        for (int seg = 0; seg < segmentCount; seg++)
        {
            SegmentState state = _segmentStates[seg];
            bool isHard = state.type == SegmentType.Hard;

            // Insert a forward gap between segments for cleaner easy↔hard transitions
            if (seg > 0)
                currentPos += baseForward * segmentTransitionBuffer;

            // For hard segments with spline yaw alignment, use dedicated spline-based placement
            if (isHard && useSplineYawAlignment)
            {
                var (segmentPlacements, endPos) = ComputeHardSegmentSpline(seg, state, currentPos, baseForward, leftAxis);
                placements.AddRange(segmentPlacements);
                currentPos = endPos;
            }
            else
            {
                // Standard per-ring placement (used for easy segments or when spline yaw is disabled)
                for (int r = 0; r < ringsPerSegment; r++)
                {
                    // Sample Z-distance (forward progression) and angle deviations
                    float zDistance = SampleZDistance(state.type);
                    float lateralAngle, pitchAngle;
                    SampleAngles(state, r, out lateralAngle, out pitchAngle);

                    // Convert angles to displacement components
                    // Z is always the sampled distance (forward progression)
                    // Lateral angle (left/right): tan(angle) * zDistance
                    // Pitch angle (up/down): tan(angle) * zDistance
                    float lateralRad = lateralAngle * Mathf.Deg2Rad;
                    float pitchRad = pitchAngle * Mathf.Deg2Rad;

                    float lateralDist = Mathf.Tan(lateralRad) * zDistance;
                    float verticalDist = Mathf.Tan(pitchRad) * zDistance;

                    // Build displacement in local space (relative to base forward)
                    // baseForward is along Z, leftAxis is along X
                    Vector3 displacement = (baseForward * zDistance) + (leftAxis * lateralDist) + (Vector3.up * verticalDist);

                    // New position
                    Vector3 newPos = currentPos + displacement;

                    // Enforce terrain height constraint (may push Y up)
                    newPos = EnforceMinHeight(newPos);

                    // Ring orientation: use baseForward temporarily (will be updated with spline tangent if enabled)
                    Quaternion ringRotation = Quaternion.LookRotation(baseForward, Vector3.up);

                    placements.Add(new PlacedRing
                    {
                        position = newPos,
                        rotation = ringRotation,
                        segmentIndex = seg,
                        isHard = isHard,
                    });

                    currentPos = newPos;
                }
            }
        }

        // Extract spline tangents if yaw alignment is enabled
        if (useSplineYawAlignment && placements.Count > 1)
        {
            ApplySplineTangentRotations(placements, baseForward);
        }
        else
        {
            // Store dummy tangents (same as baseForward)
            _lastComputedTangents.Clear();
            _lastComputedTangents.Capacity = placements.Count;
            for (int i = 0; i < placements.Count; i++)
                _lastComputedTangents.Add(baseForward);
        }

        return placements;
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Angle sampling
    // ─────────────────────────────────────────────────────────────────────────

    private void SampleAngles(SegmentState state, int ringIndex, out float lateralAngle, out float pitchAngle)
    {
        if (state.type == SegmentType.Easy)
        {
            // Easy: minimal angle deviations, nearly straight path
            lateralAngle = state.yawSign * Random.Range(easyLateralAngleMin, easyLateralAngleMax);
            pitchAngle = state.pitchSign * Random.Range(easyPitchAngleMin, easyPitchAngleMax);
        }
        else
        {
            // Hard: S-shape — lateral angle alternates left-right-left for each ring,
            // with pitch also alternating for vertical variation
            int ringSignMultiplier = (ringIndex % 2 == 0) ? 1 : -1;
            int yawSign = ringSignMultiplier * state.yawSign;
            int pitchSign = ringSignMultiplier * state.pitchSign;

            lateralAngle = yawSign * Random.Range(hardLateralAngleMin, hardLateralAngleMax);
            pitchAngle = pitchSign * Random.Range(hardPitchAngleMin, hardPitchAngleMax);
        }
    }

    private float SampleZDistance(SegmentType type)
    {
        if (type == SegmentType.Easy)
            return Random.Range(easyZDistanceMin, easyZDistanceMax);
        else
            return Random.Range(hardZDistanceMin, hardZDistanceMax);
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Hard segment spline-based placement (even distribution with yaw guidance)
    // ─────────────────────────────────────────────────────────────────────────

    private (List<PlacedRing> placements, Vector3 endPosition) ComputeHardSegmentSpline(
        int segIndex, SegmentState state, Vector3 startPos, Vector3 baseForward, Vector3 leftAxis)
    {
        var placements = new List<PlacedRing>(ringsPerSegment);
        Vector3 endPosition;

        // Analytical S-curve using shape(s) = sin(2πs)·sin(πs), with s ∈ [0,1] across the gates.
        // The sin(πs) envelope pins shape=0 AND shape'=0 at s=0 and s=1, so the first/last gates
        // land on the center line at the segment's base altitude, aligned with baseForward — no
        // lateral, altitude, or yaw jumps at easy↔hard segment boundaries. Middle transition at
        // s=0.5 still carries the configured pitch/lateral angle.

        // Total forward depth of the segment (same budget as ringsPerSegment standard gates)
        float totalZ = Random.Range(hardZDistanceMin, hardZDistanceMax) * ringsPerSegment;

        float lateralAngleRad = Random.Range(hardLateralAngleMin, hardLateralAngleMax) * Mathf.Deg2Rad;
        float pitchAngleRad   = Random.Range(hardPitchAngleMin,   hardPitchAngleMax)   * Mathf.Deg2Rad;

        // Find the max gate-to-gate shape delta. With n rings, gates sample s = i/(n-1).
        // The steepest chord (usually at the middle crossing) carries the configured angle; other
        // chords come out naturally smaller.
        float maxDeltaFactor = 0f;
        int denom = Mathf.Max(1, ringsPerSegment - 1);
        for (int i = 0; i < ringsPerSegment - 1; i++)
        {
            float sa = (float)i / denom;
            float sb = (float)(i + 1) / denom;
            float ya = Mathf.Sin(2f * Mathf.PI * sa) * Mathf.Sin(Mathf.PI * sa);
            float yb = Mathf.Sin(2f * Mathf.PI * sb) * Mathf.Sin(Mathf.PI * sb);
            float d  = Mathf.Abs(yb - ya);
            if (d > maxDeltaFactor) maxDeltaFactor = d;
        }
        float gateSpacingZ = totalZ / (ringsPerSegment + 1);

        float lateralAmplitude = (maxDeltaFactor > 0.001f)
            ? Mathf.Tan(lateralAngleRad) * gateSpacingZ / maxDeltaFactor
            : 0f;
        float pitchAmplitude = (maxDeltaFactor > 0.001f)
            ? Mathf.Tan(pitchAngleRad) * gateSpacingZ / maxDeltaFactor
            : 0f;

        // First pass: compute raw gate positions + per-gate terrain floor deficits.
        // Terrain lift uses a sin(πs) envelope (zero at s=0 and s=1) so the first and last hard
        // gates stay at base altitude — preserving continuity with the neighboring easy segments
        // — while the middle of the segment can rise to clear terrain.
        var rawPositions  = new Vector3[ringsPerSegment];
        var directions    = new Vector3[ringsPerSegment];
        var sParams       = new float[ringsPerSegment];
        var minYRequired  = new float[ringsPerSegment];
        float maxEnvLift  = 0f;  // peak of sin(πs) lift profile (achieved at s=0.5)

        // Shape parameter s ∈ [0,1] (indexed by gate, pins shape/deriv to 0 at boundaries);
        // z-placement parameter t ∈ (0,1) (evenly spaced forward, keeps first/last gates inside segment).
        for (int i = 0; i < ringsPerSegment; i++)
        {
            float s = (float)i / denom;
            float t = (float)(i + 1) / (ringsPerSegment + 1);
            sParams[i] = s;

            float sin2pis = Mathf.Sin(2f * Mathf.PI * s);
            float cos2pis = Mathf.Cos(2f * Mathf.PI * s);
            float sinpis  = Mathf.Sin(Mathf.PI * s);
            float cospis  = Mathf.Cos(Mathf.PI * s);
            float shape      = sin2pis * sinpis;
            float shapeDeriv = 2f * Mathf.PI * cos2pis * sinpis + Mathf.PI * sin2pis * cospis;

            float z = totalZ * t;
            float x = state.yawSign   * lateralAmplitude * shape;
            float y = state.pitchSign * pitchAmplitude   * shape;

            Vector3 pos = startPos + baseForward * z + leftAxis * x + Vector3.up * y;
            rawPositions[i] = pos;

            float terrainH = SampleTerrainHeight(pos);
            float minY     = terrainH + minHeightAboveTerrain;
            minYRequired[i] = minY;

            float deficit = minY - pos.y;
            if (deficit > 0f)
            {
                // Solve lift · sinpis >= deficit → lift >= deficit / sinpis.
                // For s very close to 0 or 1 (sinpis≈0) the envelope cannot help those gates —
                // they fall through to a per-gate safety clamp below. This is fine because the
                // boundary gates are on the centerline and typically near the previous segment's
                // altitude already.
                if (sinpis > 0.05f)
                    maxEnvLift = Mathf.Max(maxEnvLift, deficit / sinpis);
            }

            // Horizontal tangent: dx/di vs dz/di. ds/di = 1/denom, dt/di = 1/(n+1).
            float dxOverDz = (state.yawSign * lateralAmplitude * shapeDeriv * (ringsPerSegment + 1))
                             / (denom * totalZ);
            Vector3 dir = (baseForward + leftAxis * dxOverDz).normalized;
            if (dir.sqrMagnitude < 0.001f) dir = baseForward;
            directions[i] = dir;
        }

        // Second pass: apply envelope lift (zero at boundaries, peak in middle) and emit placements.
        for (int i = 0; i < ringsPerSegment; i++)
        {
            Vector3 pos = rawPositions[i];
            pos.y += maxEnvLift * Mathf.Sin(Mathf.PI * sParams[i]);

            // Safety clamp: catches boundary gates whose envelope factor was too small to fix.
            if (pos.y < minYRequired[i])
                pos.y = minYRequired[i];

            placements.Add(new PlacedRing
            {
                position     = pos,
                rotation     = Quaternion.LookRotation(directions[i], Vector3.up),
                segmentIndex = segIndex,
                isHard       = true,
            });
        }

        if (maxEnvLift > 0.01f)
            Debug.Log($"[CourseGenerator] Hard segment {segIndex}: envelope lift peak {maxEnvLift:F2}m (sin πs, 0 at boundaries).");

        // End position: segment endpoint on center line at base altitude — no lift
        // (sin(π·1)=0), so the next easy segment continues from the original altitude.
        endPosition = startPos + baseForward * totalZ;

        Debug.Log($"[CourseGenerator] Hard segment {segIndex}: analytical S-curve, totalZ={totalZ:F1}, " +
                  $"lateralAmplitude={lateralAmplitude:F1}, {ringsPerSegment} gates placed.");

        return (placements, endPosition);
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Spline tangent alignment
    // ─────────────────────────────────────────────────────────────────────────

    private void ApplySplineTangentRotations(List<PlacedRing> placements, Vector3 baseForward)
    {
        try
        {
            if (placements.Count < 2)
            {
                Debug.LogWarning("[CourseGenerator] Need at least 2 gates for spline tangent extraction. Skipping yaw alignment.");
                return;
            }

            // Compute chord direction for each placement.
            // Hard gates already have analytical S-curve rotation — skip them to preserve it.
            // Easy gates: compute chord only within the same segment, so a segment-end gate
            // doesn't try to point at the first gate of the next (hard) segment.
            _lastComputedTangents.Clear();
            _lastComputedTangents.Capacity = placements.Count;

            for (int i = 0; i < placements.Count; i++)
            {
                if (placements[i].isHard)
                {
                    // Preserve analytical rotation; record its forward for debug gizmos.
                    _lastComputedTangents.Add(placements[i].rotation * Vector3.forward);
                    continue;
                }

                int seg = placements[i].segmentIndex;
                Vector3 chord;

                // Prefer forward chord within the same segment
                if (i + 1 < placements.Count && placements[i + 1].segmentIndex == seg)
                    chord = placements[i + 1].position - placements[i].position;
                // At end of segment: look backward within same segment
                else if (i - 1 >= 0 && placements[i - 1].segmentIndex == seg)
                    chord = placements[i].position - placements[i - 1].position;
                else
                    chord = baseForward;

                Vector3 dir = Vector3.ProjectOnPlane(chord, Vector3.up).normalized;
                if (dir.sqrMagnitude < 0.001f)
                    dir = baseForward;

                _lastComputedTangents.Add(dir);
            }

            // Apply new rotations — only for easy gates; hard gates keep analytical rotation.
            int updated = 0;
            for (int i = 0; i < placements.Count; i++)
            {
                if (placements[i].isHard) continue;

                PlacedRing placement = placements[i];
                placement.rotation = Quaternion.LookRotation(_lastComputedTangents[i], Vector3.up);
                placements[i] = placement;
                updated++;
            }

            Debug.Log($"[CourseGenerator] Applied within-segment chord yaw to {updated} easy gates; {placements.Count - updated} hard gates kept analytical rotation.");
        }
        catch (System.Exception ex)
        {
            Debug.LogWarning($"[CourseGenerator] Spline tangent extraction failed: {ex.Message}. Falling back to constant forward direction.");
            _lastComputedTangents.Clear();
            for (int i = 0; i < placements.Count; i++)
                _lastComputedTangents.Add(baseForward);
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Height constraint enforcement
    // ─────────────────────────────────────────────────────────────────────────

    private Vector3 EnforceMinHeight(Vector3 candidatePos)
    {
        float terrainH = SampleTerrainHeight(candidatePos);
        float minH = terrainH + minHeightAboveTerrain;

        if (candidatePos.y < minH)
        {
            candidatePos.y = minH;
            Debug.Log($"[CourseGenerator] Ring clamped to minimum height {minH:F1}m above terrain.");
        }

        return candidatePos;
    }

    private static float SampleTerrainHeight(Vector3 pos)
    {
        Terrain terrain = Terrain.activeTerrain;
        if (terrain != null)
            return terrain.SampleHeight(pos) + terrain.transform.position.y;

        // Fallback: physics raycast
        if (Physics.Raycast(pos + Vector3.up * 500f, Vector3.down, out RaycastHit hit, 1000f))
            return hit.point.y;

        return 0f;
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Course geometry validation
    // ─────────────────────────────────────────────────────────────────────────

    private void ValidateCourseGeometry()
    {
        // Check segment count
        if (segmentCount < 2)
        {
            Debug.LogError("[CourseGenerator] At least 2 segments required for alternating pattern (Easy-Hard or Hard-Easy).");
            return;
        }

        // Check hard segment lateral angle range
        float hardLateralRange = hardLateralAngleMax - hardLateralAngleMin;
        if (hardLateralRange < 10f)
        {
            Debug.LogWarning($"[CourseGenerator] Hard segment lateral range ({hardLateralRange:F1}°) < 10° may not be perceivable. Consider 20-35° for visible S-curves.");
        }
        if (hardLateralAngleMax > 45f)
        {
            Debug.LogWarning($"[CourseGenerator] Hard segment max lateral angle ({hardLateralAngleMax:F1}°) > 45° may be too sharp. Consider 20-35° for smoother turns.");
        }

        // Check hard segment pitch angle range
        float hardPitchRange = hardPitchAngleMax - hardPitchAngleMin;
        if (hardPitchRange < 5f)
        {
            Debug.Log($"[CourseGenerator] Suggestion: Hard segment pitch range ({hardPitchRange:F1}°) < 5° won't create visible up/down elevation. Try 8-15° for elevation changes.");
        }
        if (hardPitchAngleMin == 0f && hardPitchAngleMax > 0f)
        {
            Debug.Log("[CourseGenerator] Suggestion: Consider setting hardPitchAngleMin > 0 to add gentle elevation throughout the course.");
        }

        // Check easy segment configuration (should be much gentler)
        float easyLateralRange = easyLateralAngleMax - easyLateralAngleMin;
        float hardVsEasyRatio = hardLateralRange > 0.001f ? easyLateralRange / hardLateralRange : 0f;
        if (hardVsEasyRatio > 0.5f)
        {
            Debug.LogWarning($"[CourseGenerator] Easy and hard segments have similar lateral ranges ({easyLateralRange:F1}° vs {hardLateralRange:F1}°). Hard segments should be much sharper for contrast.");
        }

        Debug.Log("[CourseGenerator] Course geometry validation complete.");
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Gate instantiation
    // ─────────────────────────────────────────────────────────────────────────

    private RingGate InstantiateGate(PlacedRing ring, int index)
    {
        GameObject go = Instantiate(ringGatePrefab, ring.position, ring.rotation);
        go.name = $"Gate_{index:D2}_Seg{ring.segmentIndex}_{(ring.isHard ? "H" : "E")}";
        go.transform.SetParent(transform, worldPositionStays: true);

        var gate = go.GetComponent<RingGate>();
        if (gate == null)
            Debug.LogError($"[CourseGenerator] Prefab is missing RingGate component on '{go.name}'.");

        _spawnedGates.Add(go);
        return gate;
    }

    private void DestroySpawnedGates()
    {
        foreach (var go in _spawnedGates)
            if (go != null) Destroy(go);
        _spawnedGates.Clear();
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Validation
    // ─────────────────────────────────────────────────────────────────────────

    private bool ValidateReferences()
    {
        bool ok = true;
        if (gateManager == null) { Debug.LogError("[CourseGenerator] gateManager is not assigned."); ok = false; }
        if (ringGatePrefab == null) { Debug.LogError("[CourseGenerator] ringGatePrefab is not assigned."); ok = false; }
        if (courseStartPoint == null) { Debug.LogError("[CourseGenerator] courseStartPoint is not assigned."); ok = false; }
        return ok;
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Editor gizmos
    // ─────────────────────────────────────────────────────────────────────────

#if UNITY_EDITOR
    private void OnDrawGizmosSelected()
    {
        if (courseStartPoint == null) return;

        // Draw start point
        Gizmos.color = Color.white;
        Gizmos.DrawWireSphere(courseStartPoint.position, 1f);
        Gizmos.DrawRay(courseStartPoint.position, courseStartPoint.forward * 3f);

        // If no course has been generated yet, show a message in the console
        if (_lastGeneratedPlacements.Count == 0)
        {
            return;  // Nothing to draw yet
        }

        // Draw the last generated course (exact same as what's in the simulation)
        Vector3 prevPos = courseStartPoint.position;

        for (int i = 0; i < _lastGeneratedPlacements.Count; i++)
        {
            PlacedRing ring = _lastGeneratedPlacements[i];
            Gizmos.color = ring.isHard
                ? new Color(1f, 0.3f, 0.3f, 0.8f)   // red for Hard
                : new Color(0.3f, 1f, 0.3f, 0.8f);   // green for Easy

            // Draw a wire rectangle for the gate aperture
            float hw = 5.5f * 0.5f;
            float hh = 5.5f * 0.5f;
            Vector3 tl = ring.position + ring.rotation * new Vector3(-hw,  hh, 0f);
            Vector3 tr = ring.position + ring.rotation * new Vector3( hw,  hh, 0f);
            Vector3 bl = ring.position + ring.rotation * new Vector3(-hw, -hh, 0f);
            Vector3 br = ring.position + ring.rotation * new Vector3( hw, -hh, 0f);
            Gizmos.DrawLine(tl, tr);
            Gizmos.DrawLine(tr, br);
            Gizmos.DrawLine(br, bl);
            Gizmos.DrawLine(bl, tl);
            Gizmos.DrawLine(prevPos, ring.position);

            // Draw forward direction
            Gizmos.color = new Color(Gizmos.color.r, Gizmos.color.g, Gizmos.color.b, 0.4f);
            Gizmos.DrawRay(ring.position, ring.rotation * Vector3.forward * 2f);

            prevPos = ring.position;
        }
    }
#endif
}
