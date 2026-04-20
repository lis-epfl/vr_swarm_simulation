using UnityEngine;
using UnityEngine.UI;

/// <summary>
/// Real-time HUD overlay displaying Pitch / Roll / Speed / Altitude Rate / Yaw Rate
/// and the horizontal acceleration contributions of the user velocity loop and swarm
/// algorithm, each relative to the maximum angle-budget acceleration (g × maxTilt).
///
/// Usage: attach to any GameObject in the scene, drag the swarmSpawn component
/// into the "spawn" field, then set "droneId" to the index of the drone you
/// want to monitor (0-based, matching "Drone 0", "Drone 1", …).
/// Everything else (Canvas, Panel, labels) is created at runtime.
/// </summary>
public class FlightHUD : MonoBehaviour
{
    [Header("References")]
    public swarmSpawn spawn;
    public Experiment.CWLController cwlController;
    [Tooltip("0-based index of the drone to monitor (matches 'Drone 0', 'Drone 1', …)")]
    public int droneId = 0;

    // Resolved at runtime
    private VelocityControl vc;

    // Racing-profile max values (fixed bar ceilings for CWL display)
    private const float RacingMaxSpeed       = 15.0f;
    private const float RacingMaxYawRate     =  1.5f;
    private const float RacingMaxPitch       =  0.45f;
    private const float RacingMaxRoll        =  0.45f;
    private const float RacingMaxAltRate     =  5.0f;

    [Header("Display")]
    [Tooltip("Screen-space position of the HUD panel (top-left corner, pixels from screen top-left)")]
    public Vector2 panelPosition = new Vector2(20f, 20f);
    public float panelWidth  = 280f;
    public float panelHeight = 380f;
    [Range(0.6f, 1.0f)]
    [Tooltip("Fraction of max at which bars turn yellow/red")]
    public float warningThreshold = 0.75f;

    // Runtime UI references
    private Canvas   _canvas;
    private RowUI    _pitchRow;
    private RowUI    _rollRow;
    private RowUI    _speedRow;
    private RowUI    _altRateRow;
    private RowUI    _yawRateRow;
    private RowUI    _thrustRow;

    // Colors
    private static readonly Color ColNormal  = new Color(0.15f, 0.75f, 0.25f);
    private static readonly Color ColWarning = new Color(1.00f, 0.75f, 0.05f);
    private static readonly Color ColDanger  = new Color(0.90f, 0.15f, 0.10f);
    private static readonly Color ColBg      = new Color(0.05f, 0.05f, 0.05f, 0.72f);
    private static readonly Color ColBar     = new Color(0.15f, 0.15f, 0.15f, 0.85f);
    private static readonly Color ColAccent  = new Color(0.25f, 0.55f, 0.95f); // blue tint for swarm row

    // -----------------------------------------------------------------------

    void Start()
    {
        BuildUI();
        StartCoroutine(ResolveVelocityControl());
    }

    private System.Collections.IEnumerator ResolveVelocityControl()
    {
        // Wait one frame so swarmSpawn.Start() has finished building the swarm list
        yield return null;

        if (spawn == null)
        {
            Debug.LogError("[FlightHUD] 'spawn' reference is not set.");
            yield break;
        }

        if (droneId < 0 || droneId >= spawn.swarm.Count)
        {
            Debug.LogError($"[FlightHUD] droneId {droneId} is out of range (swarm has {spawn.swarm.Count} drones).");
            yield break;
        }

        GameObject drone = spawn.swarm[droneId];
        Transform droneParent = drone.transform.Find("DroneParent");
        if (droneParent == null)
        {
            Debug.LogError($"[FlightHUD] Could not find 'DroneParent' on drone {droneId}.");
            yield break;
        }

        vc = droneParent.GetComponent<VelocityControl>();
        if (vc == null)
            Debug.LogError($"[FlightHUD] No VelocityControl found on DroneParent of drone {droneId}.");
    }

    void Update()
    {
        if (vc == null) return;

        StateFinder st = vc.State;
        Rigidbody   rb = vc.GetComponent<Rigidbody>();

        float pitchDeg    = st.Angles.x * Mathf.Rad2Deg;
        float rollDeg     = st.Angles.z * Mathf.Rad2Deg;

        // Horizontal speed — magnitude only, unipolar
        Vector3 worldVel = rb.velocity;
        float   hSpeed   = new Vector2(worldVel.x, worldVel.z).magnitude;

        // Vertical (altitude) rate — world frame y velocity
        float altRate = worldVel.y;

        // Yaw rate — body frame y angular velocity, converted to deg/s
        float yawRateDeg = st.AngularVelocityVector.y * Mathf.Rad2Deg;

        // CWL limits (fallback to Racing max if no controller assigned)
        float limitSpeed    = cwlController != null ? vc.maxSpeed       : RacingMaxSpeed;
        float limitYawRate  = cwlController != null ? vc.maxYawRate     : RacingMaxYawRate;
        float limitPitch    = cwlController != null ? vc.maxPitch       : RacingMaxPitch;
        float limitAltRate  = cwlController != null ? vc.maxAltitudeRate : RacingMaxAltRate;

        // Limit ratios for bar positioning (0..1 relative to Racing max)
        float limitRatioSpeed    = limitSpeed    / RacingMaxSpeed;
        float limitRatioYawRate  = (limitYawRate * Mathf.Rad2Deg) / (RacingMaxYawRate * Mathf.Rad2Deg);
        float limitRatioPitch    = (limitPitch   * Mathf.Rad2Deg) / (RacingMaxPitch   * Mathf.Rad2Deg);
        float limitRatioAltRate  = limitAltRate  / RacingMaxAltRate;

        // Max angle in degrees using Racing max
        float racingAngleDeg = RacingMaxPitch * Mathf.Rad2Deg;
        float racingYawDeg   = RacingMaxYawRate * Mathf.Rad2Deg;

        // --- update rows (using Racing max for fixed bar scales) ---
        UpdateRowBipolar  (_pitchRow,      "Pitch",       pitchDeg,              racingAngleDeg,      racingAngleDeg,      "°",     limitRatioPitch);
        UpdateRowBipolar  (_rollRow,       "Roll",        rollDeg,               racingAngleDeg,      racingAngleDeg,      "°",     limitRatioPitch);
        UpdateRowBipolar  (_yawRateRow,    "Yaw Rate",    yawRateDeg,            racingYawDeg,        racingYawDeg,        " °/s",  limitRatioYawRate);
        UpdateRowBipolar  (_altRateRow,    "Alt Rate",    altRate,               RacingMaxAltRate,    RacingMaxAltRate,    " m/s",  limitRatioAltRate);
        UpdateRowUnipolar (_speedRow,      "Speed",       hSpeed,                RacingMaxSpeed,                           " m/s",  limitRatio: limitRatioSpeed);
        UpdateRowUnipolar (_thrustRow,     "Thrust",      vc.lastThrustClamped,  2.7f * 9.81f,                             " m/s²");
    }

    /// <summary>Bipolar bar — value can be positive or negative, fills from center.</summary>
    void UpdateRowBipolar(RowUI row, string label, float value, float maxPos, float maxNeg, string unit, float limitRatio = -1f)
    {
        float maxSide  = value >= 0 ? maxPos : maxNeg;
        float ratio    = (maxSide > 0) ? Mathf.Clamp(value / maxSide, -1f, 1f) : 0f;
        float absRatio = Mathf.Abs(ratio);

        string sign    = (value >= 0) ? "+" : "";
        string limSign = (value >= 0) ? "+" : "-";
        row.label.text = label;
        row.value.text = string.Format("{0}{1:F2}{2}  /  {3}{4:F2}{5}", sign, value, unit, limSign, maxSide, unit);

        float fillMin = 0.5f + Mathf.Min(0f, ratio) * 0.5f;
        float fillMax = 0.5f + Mathf.Max(0f, ratio) * 0.5f;
        RectTransform fillRect = row.fill.rectTransform;
        fillRect.anchorMin = new Vector2(fillMin, 0);
        fillRect.anchorMax = new Vector2(fillMax, 1);
        fillRect.offsetMin = Vector2.zero;
        fillRect.offsetMax = Vector2.zero;

        row.fill.color     = RatioColor(absRatio);
        row.centerTick.color = new Color(1f, 1f, 1f, 0.5f); // visible for bipolar bars

        // Position limit tick
        if (limitRatio >= 0f && row.limitTick != null)
        {
            row.limitTick.enabled = true;
            float anchorX = 0.5f + limitRatio * 0.5f;
            RectTransform lt = row.limitTick.rectTransform;
            lt.anchorMin = new Vector2(anchorX, 0);
            lt.anchorMax = new Vector2(anchorX, 1);
        }
        else if (row.limitTick != null)
            row.limitTick.enabled = false;
    }

    /// <summary>Unipolar bar — value is always ≥ 0, fills from left edge.</summary>
    void UpdateRowUnipolar(RowUI row, string label, float value, float max, string unit, Color? baseColor = null, float limitRatio = -1f)
    {
        float ratio = (max > 0) ? Mathf.Clamp01(value / max) : 0f;

        row.label.text = label;
        row.value.text = string.Format("{0:F2}{1}  /  {2:F2}{3}", value, unit, max, unit);

        RectTransform fillRect = row.fill.rectTransform;
        fillRect.anchorMin = new Vector2(0f, 0);
        fillRect.anchorMax = new Vector2(ratio, 1);
        fillRect.offsetMin = Vector2.zero;
        fillRect.offsetMax = Vector2.zero;

        // Use danger/warning override only when near limit; otherwise use the supplied base color
        Color c;
        if      (ratio >= 0.9f)             c = ColDanger;
        else if (ratio >= warningThreshold)  c = ColWarning;
        else                                c = baseColor ?? ColNormal;
        row.fill.color = c;

        row.centerTick.color = Color.clear; // no center tick for unipolar bars

        // Position limit tick
        if (limitRatio >= 0f && row.limitTick != null)
        {
            row.limitTick.enabled = true;
            RectTransform lt = row.limitTick.rectTransform;
            lt.anchorMin = new Vector2(limitRatio, 0);
            lt.anchorMax = new Vector2(limitRatio, 1);
        }
        else if (row.limitTick != null)
            row.limitTick.enabled = false;
    }

    Color RatioColor(float absRatio)
    {
        if      (absRatio >= 0.9f)             return ColDanger;
        else if (absRatio >= warningThreshold)  return ColWarning;
        else                                    return ColNormal;
    }

    // -----------------------------------------------------------------------

    struct RowUI
    {
        public Text  label;
        public Text  value;
        public Image fill;
        public Image centerTick;
        public Image limitTick;
    }

    void BuildUI()
    {
        // --- Canvas ---
        GameObject canvasGO = new GameObject("FlightHUD_Canvas");
        canvasGO.transform.SetParent(transform);
        _canvas = canvasGO.AddComponent<Canvas>();
        _canvas.renderMode = RenderMode.ScreenSpaceOverlay;
        _canvas.sortingOrder = 100;
        canvasGO.AddComponent<CanvasScaler>();
        canvasGO.AddComponent<GraphicRaycaster>();

        // --- Background panel ---
        GameObject panelGO = CreateUIObject("Panel", canvasGO.transform);
        Image panelImg = panelGO.AddComponent<Image>();
        panelImg.color = ColBg;
        RectTransform panelRect = panelGO.GetComponent<RectTransform>();
        panelRect.anchorMin = panelRect.anchorMax = new Vector2(0, 1);
        panelRect.pivot     = new Vector2(0, 1);
        panelRect.anchoredPosition = new Vector2(panelPosition.x, -panelPosition.y);
        panelRect.sizeDelta = new Vector2(panelWidth, panelHeight);

        // --- Title ---
        GameObject titleGO = CreateUIObject("Title", panelGO.transform);
        Text titleTxt = titleGO.AddComponent<Text>();
        titleTxt.text      = "FLIGHT HUD";
        titleTxt.font      = Resources.GetBuiltinResource<Font>("LegacyRuntime.ttf");
        titleTxt.fontSize  = 13;
        titleTxt.fontStyle = FontStyle.Bold;
        titleTxt.color     = Color.white;
        titleTxt.alignment = TextAnchor.MiddleCenter;
        RectTransform titleRect = titleGO.GetComponent<RectTransform>();
        titleRect.anchorMin = new Vector2(0, 1);
        titleRect.anchorMax = new Vector2(1, 1);
        titleRect.pivot     = new Vector2(0.5f, 1);
        titleRect.anchoredPosition = new Vector2(0, -4);
        titleRect.sizeDelta = new Vector2(0, 20);

        // --- Section divider label helper ---
        float rowH    = 36f;
        float startY  = -28f;
        float padding = 6f;

        _pitchRow      = BuildRow(panelGO.transform, startY - 0 * (rowH + padding), rowH);
        _rollRow       = BuildRow(panelGO.transform, startY - 1 * (rowH + padding), rowH);
        _yawRateRow    = BuildRow(panelGO.transform, startY - 2 * (rowH + padding), rowH);
        _altRateRow    = BuildRow(panelGO.transform, startY - 3 * (rowH + padding), rowH);
        _speedRow      = BuildRow(panelGO.transform, startY - 4 * (rowH + padding), rowH);
        _thrustRow     = BuildRow(panelGO.transform, startY - 5 * (rowH + padding), rowH);
    }

    RowUI BuildRow(Transform parent, float yOffset, float rowH)
    {
        float hPad = 8f;

        GameObject rowGO = CreateUIObject("Row", parent);
        RectTransform rowRect = rowGO.GetComponent<RectTransform>();
        rowRect.anchorMin = new Vector2(0, 1);
        rowRect.anchorMax = new Vector2(1, 1);
        rowRect.pivot     = new Vector2(0, 1);
        rowRect.anchoredPosition = new Vector2(hPad, yOffset);
        rowRect.sizeDelta = new Vector2(-(hPad * 2), rowH);

        // Label (top-left)
        GameObject labelGO = CreateUIObject("Label", rowGO.transform);
        Text labelTxt = labelGO.AddComponent<Text>();
        labelTxt.font      = Resources.GetBuiltinResource<Font>("LegacyRuntime.ttf");
        labelTxt.fontSize  = 11;
        labelTxt.fontStyle = FontStyle.Bold;
        labelTxt.color     = new Color(0.85f, 0.85f, 0.85f);
        labelTxt.alignment = TextAnchor.UpperLeft;
        RectTransform labelRect = labelGO.GetComponent<RectTransform>();
        labelRect.anchorMin = new Vector2(0, 0.5f);
        labelRect.anchorMax = new Vector2(0.35f, 1);
        labelRect.offsetMin = labelRect.offsetMax = Vector2.zero;

        // Value text (top-right of label area)
        GameObject valuGO = CreateUIObject("Value", rowGO.transform);
        Text valuTxt = valuGO.AddComponent<Text>();
        valuTxt.font      = Resources.GetBuiltinResource<Font>("LegacyRuntime.ttf");
        valuTxt.fontSize  = 10;
        valuTxt.color     = new Color(0.9f, 0.9f, 0.9f);
        valuTxt.alignment = TextAnchor.UpperRight;
        RectTransform valuRect = valuGO.GetComponent<RectTransform>();
        valuRect.anchorMin = new Vector2(0.35f, 0.5f);
        valuRect.anchorMax = new Vector2(1, 1);
        valuRect.offsetMin = valuRect.offsetMax = Vector2.zero;

        // Bar background
        GameObject barBgGO = CreateUIObject("BarBg", rowGO.transform);
        Image barBgImg = barBgGO.AddComponent<Image>();
        barBgImg.color = ColBar;
        RectTransform barBgRect = barBgGO.GetComponent<RectTransform>();
        barBgRect.anchorMin = new Vector2(0, 0);
        barBgRect.anchorMax = new Vector2(1, 0.5f);
        barBgRect.offsetMin = Vector2.zero;
        barBgRect.offsetMax = Vector2.zero;

        // Bar fill (child of barBg so it clips naturally via anchors)
        GameObject fillGO = CreateUIObject("Fill", barBgGO.transform);
        Image fillImg = fillGO.AddComponent<Image>();
        fillImg.color = ColNormal;
        RectTransform fillRect = fillGO.GetComponent<RectTransform>();
        fillRect.anchorMin = new Vector2(0.5f, 0);
        fillRect.anchorMax = new Vector2(0.5f, 1);
        fillRect.offsetMin = fillRect.offsetMax = Vector2.zero;

        // Center tick — 1 px wide white line at x = 0.5
        GameObject tickGO = CreateUIObject("CenterTick", barBgGO.transform);
        Image tickImg = tickGO.AddComponent<Image>();
        tickImg.color = new Color(1f, 1f, 1f, 0.5f);
        RectTransform tickRect = tickGO.GetComponent<RectTransform>();
        tickRect.anchorMin = new Vector2(0.5f, 0);
        tickRect.anchorMax = new Vector2(0.5f, 1);
        tickRect.offsetMin = new Vector2(-1f, 0);
        tickRect.offsetMax = new Vector2( 1f, 0);

        // Limit tick — 2 px wide red line, positioned dynamically in Update
        GameObject limitTickGO = CreateUIObject("LimitTick", barBgGO.transform);
        Image limitTickImg = limitTickGO.AddComponent<Image>();
        limitTickImg.color = new Color(0.90f, 0.10f, 0.10f, 0.90f);
        RectTransform limitTickRect = limitTickGO.GetComponent<RectTransform>();
        limitTickRect.anchorMin = new Vector2(0f, 0);
        limitTickRect.anchorMax = new Vector2(0f, 1);
        limitTickRect.offsetMin = new Vector2(-1f, 0);
        limitTickRect.offsetMax = new Vector2( 1f, 0);

        return new RowUI { label = labelTxt, value = valuTxt, fill = fillImg, centerTick = tickImg, limitTick = limitTickImg };
    }

    static GameObject CreateUIObject(string name, Transform parent)
    {
        var go = new GameObject(name, typeof(RectTransform));
        go.transform.SetParent(parent, false);
        return go;
    }
}
