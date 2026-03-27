using System.Collections.Generic;
using UnityEngine;

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

    [Header("References")]
    public RingGateManager gateManager;
    public CoursePathVisual pathVisual;
    public GameObject ringGatePrefab;
    [Tooltip("Start position and forward direction for the course.")]
    public Transform courseStartPoint;

    [Header("Auto-generate")]
    [Tooltip("If true, generates a course automatically on Start().")]
    public bool generateOnStart = true;

    // ─────────────────────────────────────────────────────────────────────────
    // Internal types
    // ─────────────────────────────────────────────────────────────────────────

    private enum SegmentType { Easy, Hard }

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

                // Ring orientation: always aligned with XY plane (constant baseForward direction)
                // Independent of path curvature or tangent
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
