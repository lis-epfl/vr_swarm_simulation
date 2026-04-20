using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Splines;
using Unity.Mathematics;

/// <summary>
/// Builds and renders a glowing spline path through all gate centers at runtime.
/// Includes a LineRenderer for the main path and animated dots that flow along it.
/// </summary>
public class CoursePathVisual : MonoBehaviour
{
    // ─────────────────────────────────────────────────────────────────────────
    // Inspector
    // ─────────────────────────────────────────────────────────────────────────

    [Header("References")]
    public RingGateManager gateManager;
    [Tooltip("Optional: Starting point for the spline path. If assigned, path begins here before the first gate.")]
    public Transform courseStartPoint;

    [Header("Visibility")]
    [Tooltip("Toggle the entire path (line + dots) on or off at runtime.")]
    [SerializeField] private bool _pathVisible = true;

    [Header("Line Renderer Settings")]
    [Tooltip("Number of positions sampled along the spline to build the line.")]
    public int splineSampleCount = 200;
    public float lineWidth = 0.15f;

    [Header("Path Color")]
    public Color pathColor = new Color(0f, 1f, 1f, 0.7f);  // cyan with transparency

    [Header("Animation")]
    public float pulseFrequency = 0.8f;
    [Range(0f, 1f)]
    public float pulseAmplitude = 0.3f;  // affects opacity/intensity

    [Header("Moving Dots")]
    [Tooltip("Prefab for dots. If null, a procedural sphere is created.")]
    public GameObject dotPrefab;
    public int dotCount = 12;
    public float dotSpeed = 5f;
    public float dotSpacing = 3f;
    public Color dotColor = new Color(0f, 0.8f, 1f, 1f);  // bright cyan

    // ─────────────────────────────────────────────────────────────────────────
    // Private
    // ─────────────────────────────────────────────────────────────────────────

    private SplineContainer _splineContainer;
    private LineRenderer _lineRenderer;
    private Material _lineMaterial;

    private Transform[] _dotObjects;
    private float _dotOffsetDistance;
    private float _splineLength;

    private static readonly int BaseColorID = Shader.PropertyToID("_Color");

    // ─────────────────────────────────────────────────────────────────────────
    // Public API
    // ─────────────────────────────────────────────────────────────────────────

    /// <summary>Show or hide the entire path (line renderer + animated dots) at runtime.</summary>
    public bool PathVisible
    {
        get => _pathVisible;
        set
        {
            _pathVisible = value;
            ApplyVisibility();
        }
    }

    /// <summary>Access the underlying SplineContainer for external animation or evaluation.</summary>
    public SplineContainer Spline => _splineContainer;

    /// <summary>
    /// Destroys all current visuals and rebuilds the spline, line, and dots
    /// from the current gateManager state. Call after procedurally placing gates.
    /// </summary>
    public void RebuildPath()
    {
        // Destroy existing dots
        if (_dotObjects != null)
        {
            foreach (var dot in _dotObjects)
                if (dot != null) Destroy(dot.gameObject);
            _dotObjects = null;
        }

        // Reset line renderer
        if (_lineRenderer != null)
            _lineRenderer.positionCount = 0;

        // Rebuild everything
        BuildSpline();
        BuildLineRenderer();
        BuildDots();
        ApplyVisibility();
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Lifecycle
    // ─────────────────────────────────────────────────────────────────────────

    private void Awake()
    {
        BuildSpline();
        BuildLineRenderer();
        BuildDots();
        ApplyVisibility();
    }

    private void Update()
    {
        if (!_pathVisible) return;
        if (_lineMaterial == null || _lineRenderer == null) return;

        // Animate line opacity pulse
        float t = Mathf.Sin(Time.time * pulseFrequency * Mathf.PI * 2f);
        float alphaScale = 1f + t * pulseAmplitude;
        Color pulsedColor = pathColor;
        pulsedColor.a *= alphaScale;  // Vary transparency
        _lineMaterial.SetColor(BaseColorID, pulsedColor);

        // Animate dots along spline
        AnimateDots();
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Visibility
    // ─────────────────────────────────────────────────────────────────────────

    private void ApplyVisibility()
    {
        if (_lineRenderer != null)
            _lineRenderer.enabled = _pathVisible;

        if (_dotObjects != null)
            foreach (var dot in _dotObjects)
                if (dot != null)
                    dot.gameObject.SetActive(_pathVisible);
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Spline construction
    // ─────────────────────────────────────────────────────────────────────────

    private void BuildSpline()
    {
        if (gateManager == null)
        {
            Debug.LogError("[CoursePathVisual] gateManager is not assigned.");
            return;
        }

        Transform[] centerPoints = gateManager.GetCenterPoints();

        if (centerPoints == null || centerPoints.Length == 0)
        {
            Debug.LogError("[CoursePathVisual] No center points returned from gateManager. Check that gates list is populated.");
            return;
        }

        int validPoints = 0;
        foreach (var cp in centerPoints)
        {
            if (cp != null) validPoints++;
        }

        if (validPoints == 0)
        {
            Debug.LogError("[CoursePathVisual] All center points are null. Ensure each gate has a centerPoint child.");
            return;
        }

        _splineContainer = gameObject.GetComponent<SplineContainer>()
                        ?? gameObject.AddComponent<SplineContainer>();

        var spline = _splineContainer.Spline;
        spline.Clear();

        // Add start point first (if assigned)
        if (courseStartPoint != null)
        {
            Vector3 localPos = transform.InverseTransformPoint(courseStartPoint.position);
            if (float.IsFinite(localPos.x) && float.IsFinite(localPos.y) && float.IsFinite(localPos.z))
            {
                spline.Add(new BezierKnot(new float3(localPos.x, localPos.y, localPos.z)));
                Debug.Log("[CoursePathVisual] Added start point to spline.");
            }
            else
            {
                Debug.LogWarning("[CoursePathVisual] CourseStartPoint has invalid position.");
            }
        }

        // Add all gate center points
        foreach (var cp in centerPoints)
        {
            if (cp == null)
            {
                Debug.LogWarning("[CoursePathVisual] Skipping null centerPoint.");
                continue;
            }

            // Knot positions are in SplineContainer's local space.
            Vector3 localPos = transform.InverseTransformPoint(cp.position);

            // Validate the position
            if (!float.IsFinite(localPos.x) || !float.IsFinite(localPos.y) || !float.IsFinite(localPos.z))
            {
                Debug.LogError($"[CoursePathVisual] Invalid center point position: {cp.position}");
                continue;
            }

            spline.Add(new BezierKnot(new float3(localPos.x, localPos.y, localPos.z)));
        }

        int minKnots = courseStartPoint != null ? 2 : 2;  // At least start + gate, or 2 gates
        if (spline.Count < minKnots)
        {
            Debug.LogError($"[CoursePathVisual] Spline has {spline.Count} knots. Need at least {minKnots}.");
            return;
        }

        // Apply auto-smooth to produce natural curved path through all gates
        for (int i = 0; i < spline.Count; i++)
            spline.SetTangentMode(i, TangentMode.AutoSmooth);

        _splineLength = _splineContainer.CalculateLength();
        Debug.Log($"[CoursePathVisual] Spline built with {spline.Count} knots, length: {_splineLength:F2}m");
    }

    // ─────────────────────────────────────────────────────────────────────────
    // LineRenderer construction
    // ─────────────────────────────────────────────────────────────────────────

    private void BuildLineRenderer()
    {
        if (_splineContainer == null)
        {
            Debug.LogWarning("[CoursePathVisual] SplineContainer is null. Cannot build LineRenderer.");
            return;
        }

        // Check if spline has any knots
        if (_splineContainer.Spline.Count == 0)
        {
            Debug.LogWarning("[CoursePathVisual] Spline has no knots. Make sure gates have centerPoints and gateManager is assigned.");
            return;
        }

        _lineRenderer = gameObject.GetComponent<LineRenderer>();
        if (_lineRenderer == null)
        {
            _lineRenderer = gameObject.AddComponent<LineRenderer>();
            if (_lineRenderer == null)
            {
                Debug.LogError("[CoursePathVisual] Failed to add LineRenderer component to CoursePath GameObject!");
                return;
            }
        }

        // Configure line renderer
        _lineRenderer.positionCount = splineSampleCount;
        _lineRenderer.useWorldSpace = true;
        _lineRenderer.widthMultiplier = lineWidth;
        _lineRenderer.numCornerVertices = 4;
        _lineRenderer.numCapVertices = 4;
        _lineRenderer.shadowCastingMode = UnityEngine.Rendering.ShadowCastingMode.Off;
        _lineRenderer.receiveShadows = false;

        // Sample positions along the spline
        int validPositions = 0;
        for (int i = 0; i < splineSampleCount; i++)
        {
            float t = (float)i / (splineSampleCount - 1);
            Vector3 pos = _splineContainer.EvaluatePosition(t);

            // Validate position
            if (float.IsFinite(pos.x) && float.IsFinite(pos.y) && float.IsFinite(pos.z))
            {
                _lineRenderer.SetPosition(i, pos);
                validPositions++;
            }
            else
            {
                Debug.LogWarning($"[CoursePathVisual] Invalid position at t={t}: {pos}");
                _lineRenderer.SetPosition(i, Vector3.zero);
            }
        }

        if (validPositions == 0)
        {
            Debug.LogError("[CoursePathVisual] No valid positions found on spline!");
            return;
        }

        // Build the material (one for the line)
        _lineMaterial = CreateLineMaterial(pathColor);
        _lineRenderer.material = _lineMaterial;

        Debug.Log("[CoursePathVisual] LineRenderer created with " + splineSampleCount +
                  " segments, width " + lineWidth + "m. Spline has " + _splineContainer.Spline.Count + " knots.");
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Dots construction and animation
    // ─────────────────────────────────────────────────────────────────────────

    private void BuildDots()
    {
        if (_splineContainer == null) return;

        // Create dot prefab if not assigned
        if (dotPrefab == null)
            dotPrefab = CreateDotPrefab();

        // Instantiate dots
        _dotObjects = new Transform[dotCount];
        for (int i = 0; i < dotCount; i++)
        {
            var dotGo = Instantiate(dotPrefab, transform);
            dotGo.SetActive(true);
            _dotObjects[i] = dotGo.transform;
        }

        _dotOffsetDistance = 0f;
    }

    private void AnimateDots()
    {
        if (_dotObjects == null || _dotObjects.Length == 0) return;
        if (_splineLength <= 0) return;

        _dotOffsetDistance += dotSpeed * Time.deltaTime;
        _dotOffsetDistance %= _splineLength;

        for (int i = 0; i < _dotObjects.Length; i++)
        {
            float distance = (i * dotSpacing + _dotOffsetDistance) % _splineLength;

            float t = _splineContainer.Spline.ConvertIndexUnit(
                distance,
                PathIndexUnit.Distance,
                PathIndexUnit.Normalized);

            Vector3 pos = _splineContainer.EvaluatePosition(t);
            Vector3 tangent = _splineContainer.EvaluateTangent(t);

            _dotObjects[i].position = pos;
            _dotObjects[i].forward = tangent.normalized;
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Material factory
    // ─────────────────────────────────────────────────────────────────────────

    private Material CreateLineMaterial(Color color)
    {
        // Use built-in Unlit shader — works with all render pipelines
        Shader shader = Shader.Find("Unlit/Color");
        if (shader == null)
            shader = Shader.Find("Unlit/Texture");

        var mat = new Material(shader);
        mat.SetColor(BaseColorID, color);

        // Enable transparency
        mat.renderQueue = (int)UnityEngine.Rendering.RenderQueue.Transparent;
        mat.SetOverrideTag("RenderType", "Transparent");

        return mat;
    }

    private GameObject CreateDotPrefab()
    {
        var go = GameObject.CreatePrimitive(PrimitiveType.Sphere);
        go.transform.localScale = Vector3.one * 0.18f;

        var col = go.GetComponent<Collider>();
        if (col != null)
            Destroy(col);

        var mr = go.GetComponent<MeshRenderer>();
        if (mr != null)
        {
            var dotMat = CreateLineMaterial(dotColor);
            mr.material = dotMat;
        }

        // Don't render in scene — it's a prefab for instantiation
        go.SetActive(false);
        go.hideFlags = HideFlags.HideAndDontSave;

        return go;
    }
}
