using UnityEngine;

/// <summary>
/// Visual state for a gate. Used to control material color and indicator plane.
/// </summary>
public enum GateVisualState { Idle, Next, PartialComplete, Completed }

/// <summary>
/// Adds visual feedback to a gate via material color changes and a translucent indicator plane.
/// Works with any render pipeline (no HDR/bloom required).
///
/// Visual feedback:
/// - Idle:            gray frame, no plane
/// - Next:            gray frame + translucent white plane filling the opening
/// - PartialComplete: yellow frame, no plane (some drones passed, not all)
/// - Completed:       green frame, no plane  (all drones passed)
/// </summary>
[RequireComponent(typeof(RingGate))]
public class RingGateVisual : MonoBehaviour
{
    // ─────────────────────────────────────────────────────────────────────────
    // Inspector
    // ─────────────────────────────────────────────────────────────────────────

    [Header("State Colors")]
    [Tooltip("Frame color when idle (semi-transparent gray).")]
    public Color idleColor = new Color(0.5f, 0.5f, 0.5f, 0.6f);

    [Tooltip("Frame color when some drones passed but not all (yellow).")]
    public Color partialCompleteColor = new Color(1f, 0.85f, 0f, 0.6f);

    [Tooltip("Frame color when all drones passed (green).")]
    public Color completedColor = new Color(0f, 1f, 0f, 0.6f);

    [Header("Next Gate Indicator Plane")]
    [Tooltip("Material for the translucent plane shown inside the gate when it is the next target. " +
             "If assigned, this material is used directly. Otherwise a simple transparent material is created.")]
    public Material nextPlaneMaterial;

    [Tooltip("Color of the translucent plane (used only if nextPlaneMaterial is not assigned).")]
    public Color nextPlaneColor = new Color(1f, 1f, 1f, 0.18f);

    [Header("Visual Override")]
    [Tooltip("When false, this gate always renders in Idle state regardless of SetState() calls.")]
    [SerializeField] private bool _colorStatesEnabled = true;

    // ─────────────────────────────────────────────────────────────────────────
    // Private
    // ─────────────────────────────────────────────────────────────────────────

    private GateVisualState _state = GateVisualState.Idle;
    private MeshRenderer[] _barRenderers;
    private RingGate _ringGate;

    private GameObject _planeGameObject;
    private Material _planeMaterial;

    private static readonly int BaseColorID = Shader.PropertyToID("_BaseColor");
    private static readonly int ColorID = Shader.PropertyToID("_Color");

    // ─────────────────────────────────────────────────────────────────────────
    // Lifecycle
    // ─────────────────────────────────────────────────────────────────────────

    private void Awake()
    {
        _ringGate = GetComponent<RingGate>();

        // Cache bar renderers BEFORE creating the plane so the plane renderer
        // is not included in the array (avoids coloring it with frame colors).
        _barRenderers = GetComponentsInChildren<MeshRenderer>();

        CreateIndicatorPlane();
        ApplyState();
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Public API
    // ─────────────────────────────────────────────────────────────────────────

    /// <summary>
    /// When false, the gate always renders in Idle state (gray) regardless of
    /// its logical state. The logical state is preserved so re-enabling restores
    /// the correct visual immediately.
    /// </summary>
    public bool ColorStatesEnabled
    {
        get => _colorStatesEnabled;
        set
        {
            _colorStatesEnabled = value;
            ApplyState();
        }
    }

    /// <summary>Changes the visual state. Immediately updates frame color and plane visibility.</summary>
    public void SetState(GateVisualState state)
    {
        _state = state;
        ApplyState();
    }

    /// <summary>Returns the current visual state.</summary>
    public GateVisualState CurrentState => _state;

    // ─────────────────────────────────────────────────────────────────────────
    // Indicator Plane
    // ─────────────────────────────────────────────────────────────────────────

    private void CreateIndicatorPlane()
    {
        var go = GameObject.CreatePrimitive(PrimitiveType.Quad);
        Destroy(go.GetComponent<Collider>());

        go.name = "NextIndicatorPlane";
        go.transform.SetParent(transform, false);
        go.transform.localPosition = Vector3.zero;
        go.transform.localRotation = Quaternion.identity; // Quad faces +Z = gate through-axis

        float w = _ringGate != null ? _ringGate.gateWidth  : 5.5f;
        float h = _ringGate != null ? _ringGate.gateHeight : 5.5f;
        go.transform.localScale = new Vector3(w, h, 1f);

        // Use assigned material if provided, otherwise try to create one
        if (nextPlaneMaterial != null)
        {
            _planeMaterial = new Material(nextPlaneMaterial);
        }
        else
        {
            Shader shader = Shader.Find("Unlit/Color");
            if (shader == null) shader = Shader.Find("Standard");
            if (shader == null) shader = Shader.Find("Sprites/Default");
            if (shader == null)
            {
                Debug.LogError("[RingGateVisual] Could not find suitable shader for indicator plane. " +
                               "Assign 'nextPlaneMaterial' in the inspector.");
                return;
            }

            _planeMaterial = new Material(shader);
            _planeMaterial.renderQueue = (int)UnityEngine.Rendering.RenderQueue.Transparent;
            _planeMaterial.SetOverrideTag("RenderType", "Transparent");
            _planeMaterial.SetColor("_Color", nextPlaneColor);
        }

        go.GetComponent<MeshRenderer>().material = _planeMaterial;
        go.SetActive(false);

        _planeGameObject = go;
    }

    // ─────────────────────────────────────────────────────────────────────────
    // State Application
    // ─────────────────────────────────────────────────────────────────────────

    private void ApplyState()
    {
        // When color states are disabled, always render as Idle regardless of logical state
        GateVisualState effectiveState = _colorStatesEnabled ? _state : GateVisualState.Idle;

        Color frameColor;
        bool showPlane;

        switch (effectiveState)
        {
            case GateVisualState.Next:
                frameColor = idleColor;   // frame stays gray — plane provides the visual cue
                showPlane  = true;
                break;
            case GateVisualState.PartialComplete:
                frameColor = partialCompleteColor;
                showPlane  = false;
                break;
            case GateVisualState.Completed:
                frameColor = completedColor;
                showPlane  = false;
                break;
            case GateVisualState.Idle:
            default:
                frameColor = idleColor;
                showPlane  = false;
                break;
        }

        SetFrameColor(frameColor);

        if (_planeGameObject != null)
            _planeGameObject.SetActive(showPlane);
    }

    private void SetFrameColor(Color color)
    {
        if (_barRenderers == null) return;

        foreach (var r in _barRenderers)
        {
            if (r == null) continue;
            var mat = r.material;

            // Enable transparency on the material if alpha < 1
            if (color.a < 1f)
            {
                mat.SetFloat("_Surface", 1f);
                mat.renderQueue = (int)UnityEngine.Rendering.RenderQueue.Transparent;
                mat.SetOverrideTag("RenderType", "Transparent");
            }

            if (mat.HasProperty(BaseColorID))
                mat.SetColor(BaseColorID, color);
            else if (mat.HasProperty(ColorID))
                mat.SetColor(ColorID, color);
        }
    }
}
