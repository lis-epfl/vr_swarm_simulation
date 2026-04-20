using System;
using System.IO;
using System.Globalization;
using UnityEngine;
using UnityEngine.UI;

/// <summary>
/// Post-trial feedback overlay — built entirely at runtime (like FlightHUD).
/// Attach to any GameObject; the full Canvas/Slider/Button hierarchy is created
/// programmatically. Use [ExecuteAlways] so the layout is visible in the Scene
/// view without entering Play mode.
///
/// CSV export: call <see cref="ExportToCsv"/> or subscribe to
/// <see cref="OnFeedbackSubmitted"/> and let the script auto-append each row.
/// All trials for a given experiment session go into the same file.
/// </summary>
[ExecuteAlways]
public class FeedbackOverlayManager : MonoBehaviour
{
    // ── Layout parameters (tweak in Inspector, preview in Scene) ────────
    [Header("Layout")]
    [SerializeField] private float panelWidth  = 1920f;
    [SerializeField] private float panelHeight = 1080f;
    [SerializeField] private Color panelColor  = new Color(0.08f, 0.08f, 0.12f, 0.95f);

    [Header("Title")]
    [SerializeField] private string titleString   = "Time for your feedback!";
    [SerializeField] private int    titleFontSize  = 38;
    [SerializeField] private Color  titleColor     = new Color(1f, 0.85f, 0.3f);

    [Header("Question")]
    [SerializeField] private string questionString   = "How did you perceive the adaptation of the drone controls during this trial?";
    [SerializeField] private int    questionFontSize  = 22;

    [Header("Slider")]
    [SerializeField] private float  sliderWidth      = 1200f;
    [SerializeField] private float  sliderHeight     = 24f;
    [SerializeField] private Color  sliderBgColor    = new Color(0.25f, 0.25f, 0.3f);
    [SerializeField] private Color  sliderFillColor  = new Color(0.35f, 0.65f, 0.95f);
    [SerializeField] private Color  handleColor      = Color.white;
    [SerializeField] private float  handleSize       = 34f;

    [Header("Labels")]
    [SerializeField] private string leftLabel   = "Made it harder\n(not helpful)";
    [SerializeField] private string centerLabel = "No noticeable\nchange";
    [SerializeField] private string rightLabel  = "Smoother & helpful\n(assisted me)";
    [SerializeField] private int    labelFontSize = 16;

    [Header("Submit Button")]
    [SerializeField] private string buttonText     = "Submit";
    [SerializeField] private int    buttonFontSize = 22;
    [SerializeField] private float  buttonWidth    = 220f;
    [SerializeField] private float  buttonHeight   = 50f;
    [SerializeField] private Color  buttonColor    = new Color(0.2f, 0.55f, 0.9f);

    [Header("CSV Export")]
    [Tooltip("Folder path relative to the project root (e.g. ExperimentData). Created automatically.")]
    [SerializeField] private string csvFolder   = "ExperimentData";
    [Tooltip("Base file name. A timestamp is appended on first write to keep sessions separate.")]
    [SerializeField] private string csvBaseName = "feedback";
    [Tooltip("If true, a row is appended automatically on each submit.")]
    [SerializeField] private bool   autoExport  = true;

    // ── Runtime state ───────────────────────────────────────────────────
    private Canvas _canvas;
    private Slider _slider;
    private Button _button;
    private Text   _buttonLabel;

    private bool  _submitted;
    private float _submittedValue;
    private int   _trialNumber;
    private string _csvFilePath;       // resolved on first export
    private bool  _dirty;              // deferred rebuild flag for OnValidate

    /// <summary>True once the user clicks Submit. Reset each time the overlay is shown.</summary>
    public bool IsSubmitted => _submitted;

    /// <summary>Raw slider value at the moment of submission (0 = left, 1 = right).</summary>
    public float SubmittedValue => _submittedValue;

    /// <summary>Fired on submit. Args: (trialNumber, sliderValue).</summary>
    public event Action<int, float> OnFeedbackSubmitted;

    // ── Unity lifecycle ─────────────────────────────────────────────────

    private void OnEnable()
    {
        RebuildUI();

        if (Application.isPlaying)
        {
            ResetState();
            if (_button != null)
                _button.onClick.AddListener(HandleSubmit);
        }
    }

    private void OnDisable()
    {
        if (_button != null)
            _button.onClick.RemoveListener(HandleSubmit);

        DestroyChildren();
    }

    private void OnValidate()
    {
        // OnValidate cannot reliably DestroyImmediate — set a flag and
        // let Update (which runs in edit mode via ExecuteAlways) handle it.
        _dirty = true;
    }

    private void Update()
    {
        if (!Application.isPlaying && _dirty)
        {
            _dirty = false;
            DestroyChildren();
            RebuildUI();
        }
    }

    // ── Public API ──────────────────────────────────────────────────────

    /// <summary>Set the current trial number before showing the overlay.</summary>
    public void SetTrialNumber(int trial) => _trialNumber = trial;

    /// <summary>Normalised score mapped to [-1, +1]. -1 = harmful, 0 = neutral, +1 = helpful.</summary>
    public float GetNormalisedScore() => (_submittedValue - 0.5f) * 2f;

    /// <summary>
    /// Manually export one row to the CSV file (useful if autoExport is off).
    /// </summary>
    public void ExportToCsv(int trialNumber, float sliderValue)
    {
        EnsureCsvFile();
        float normalised = (sliderValue - 0.5f) * 2f;
        string timestamp = DateTimeOffset.UtcNow.ToUnixTimeMilliseconds().ToString();
        string line = string.Format(CultureInfo.InvariantCulture,
            "{0},{1},{2},{3},{4}",
            timestamp,
            trialNumber,
            sliderValue,
            normalised,
            questionString.Replace(",", ";") // escape commas in the question text
        );
        File.AppendAllText(_csvFilePath, line + Environment.NewLine);
        Debug.Log($"[FeedbackOverlay] Exported trial {trialNumber} to {_csvFilePath}");
    }

    /// <summary>
    /// Resets the CSV session so the next export creates a fresh file.
    /// Call this at the start of a new experiment if you want a separate file.
    /// </summary>
    public void ResetCsvSession() => _csvFilePath = null;

    public void ForceSubmit()
    {
        if (!_submitted)
            HandleSubmit();
    }

    // ── Internals ───────────────────────────────────────────────────────

    private void ResetState()
    {
        _submitted = false;
        _submittedValue = 0.5f;
        if (_slider != null) _slider.value = 0.5f;
        if (_button != null) _button.interactable = true;
        if (_buttonLabel != null) _buttonLabel.text = buttonText;
    }

    private void HandleSubmit()
    {
        if (_submitted) return;
        _submitted = true;
        _submittedValue = _slider != null ? _slider.value : 0.5f;

        if (_button != null) _button.interactable = false;
        if (_buttonLabel != null) _buttonLabel.text = "Submitted!";

        Debug.Log($"[FeedbackOverlay] Trial {_trialNumber} — value: {_submittedValue:F2} (normalised: {GetNormalisedScore():F2})");

        if (autoExport)
            ExportToCsv(_trialNumber, _submittedValue);

        OnFeedbackSubmitted?.Invoke(_trialNumber, _submittedValue);
    }

    // ── CSV helpers ─────────────────────────────────────────────────────

    private void EnsureCsvFile()
    {
        if (!string.IsNullOrEmpty(_csvFilePath) && File.Exists(_csvFilePath))
            return;

        string folder = Path.Combine(Application.dataPath, "..", csvFolder);
        Directory.CreateDirectory(folder);

        string stamp = DateTime.Now.ToString("yyyyMMdd_HHmmss");
        _csvFilePath = Path.Combine(folder, $"{csvBaseName}_{stamp}.csv");

        // Write header
        string header = "timestamp,trial,slider_value,normalised_score,question";
        File.WriteAllText(_csvFilePath, header + Environment.NewLine);
        Debug.Log($"[FeedbackOverlay] Created CSV: {_csvFilePath}");
    }

    // ── UI construction ─────────────────────────────────────────────────

    private void DestroyChildren()
    {
        // Destroy the canvas we built (if any)
        for (int i = transform.childCount - 1; i >= 0; i--)
        {
            var child = transform.GetChild(i).gameObject;
            if (Application.isPlaying)
                Destroy(child);
            else
                DestroyImmediate(child);
        }
        _canvas = null;
        _slider = null;
        _button = null;
        _buttonLabel = null;
    }

    private void RebuildUI()
    {
        // Always clean up first — catches orphaned children from domain reloads,
        // play-mode transitions, or duplicate OnEnable calls.
        DestroyChildren();

        // ── Canvas ──────────────────────────────────────────────────────
        var canvasGO = CreateUI("FeedbackCanvas", transform);
        _canvas = canvasGO.AddComponent<Canvas>();
        _canvas.renderMode = RenderMode.ScreenSpaceOverlay;
        _canvas.sortingOrder = 200;
        var scaler = canvasGO.AddComponent<CanvasScaler>();
        scaler.uiScaleMode = CanvasScaler.ScaleMode.ScaleWithScreenSize;
        scaler.referenceResolution = new Vector2(1920, 1080);
        canvasGO.AddComponent<GraphicRaycaster>();

        // ── Panel (sized via Inspector) ────────────────────────────────
        var panelGO = CreateUI("Panel", canvasGO.transform);
        var panelImg = panelGO.AddComponent<Image>();
        panelImg.color = panelColor;
        var panelRT = panelGO.GetComponent<RectTransform>();
        panelRT.anchorMin = panelRT.anchorMax = new Vector2(0.5f, 0.5f);
        panelRT.pivot = new Vector2(0.5f, 0.5f);
        panelRT.sizeDelta = new Vector2(panelWidth, panelHeight);

        // Content is vertically centred by working from the middle of the screen.
        // Total content height: title(50) + gap(20) + question(60) + gap(50) +
        //                       labels(48) + gap(12) + slider(24) + gap(40) + button(50) = ~354
        float contentH  = 500f;
        float y = contentH / 2f;   // start from the top of the content block

        // ── Title ───────────────────────────────────────────────────────
        y = AddText(panelGO.transform, "Title", titleString, titleFontSize, titleColor,
                    FontStyle.Bold, TextAnchor.MiddleCenter, y, 70f);

        y -= 40f;

        // ── Question ────────────────────────────────────────────────────
        y = AddText(panelGO.transform, "Question", questionString, questionFontSize, new Color(0.9f, 0.9f, 0.9f),
                    FontStyle.Normal, TextAnchor.MiddleCenter, y, 60f);

        y -= 50f;

        // ── Centre label (above the slider midpoint) ────────────────────
        float centerLabelH = 48f;
        float labelW = sliderWidth * 0.28f;
        AddTextAnchored(panelGO.transform, "LabelCenter", centerLabel, labelFontSize,
                        new Color(0.75f, 0.75f, 0.75f), FontStyle.Italic, TextAnchor.MiddleCenter,
                        y, centerLabelH, 0f, labelW);

        y -= centerLabelH + 12f;

        // ── Slider with left/right labels at its sides ──────────────────
        float sliderY = y;
        y = BuildSlider(panelGO.transform, sliderY);

        // Left label — to the left of the slider, vertically centred with it
        float halfSlider = sliderWidth / 2f;
        float sideLabelW = (panelWidth - sliderWidth) / 2f - 20f; // fill the gap between panel edge and slider
        float sideLabelH = 48f;
        // Anchor vertically at the slider's centre: sliderY is the top, so centre = sliderY - sliderHeight/2
        float sliderCenterY = sliderY - sliderHeight / 2f;

        AddTextAnchored(panelGO.transform, "LabelLeft", leftLabel, labelFontSize,
                        new Color(0.95f, 0.45f, 0.4f), FontStyle.Normal, TextAnchor.MiddleRight,
                        sliderCenterY + sideLabelH / 2f, sideLabelH,
                        -(halfSlider + sideLabelW / 2f + 10f), sideLabelW);

        // Right label — to the right of the slider, vertically centred with it
        AddTextAnchored(panelGO.transform, "LabelRight", rightLabel, labelFontSize,
                        new Color(0.4f, 0.85f, 0.5f), FontStyle.Normal, TextAnchor.MiddleLeft,
                        sliderCenterY + sideLabelH / 2f, sideLabelH,
                        halfSlider + sideLabelW / 2f + 10f, sideLabelW);

        y -= 100f;

        // ── Submit button ───────────────────────────────────────────────
        BuildButton(panelGO.transform, y);
    }

    /// <summary>Adds a full-width text element anchored from the vertical centre of the panel.
    /// yFromCenter is positive = above centre, negative = below. Returns next Y below this element.</summary>
    private float AddText(Transform parent, string name, string text, int fontSize, Color color,
                          FontStyle style, TextAnchor anchor, float yFromCenter, float height)
    {
        var go = CreateUI(name, parent);
        var t  = go.AddComponent<Text>();
        t.text      = text;
        t.font      = Resources.GetBuiltinResource<Font>("LegacyRuntime.ttf");
        t.fontSize  = fontSize;
        t.fontStyle = style;
        t.color     = color;
        t.alignment = anchor;
        t.horizontalOverflow = HorizontalWrapMode.Wrap;
        t.verticalOverflow   = VerticalWrapMode.Truncate;

        var rt = go.GetComponent<RectTransform>();
        // Anchor at vertical centre, stretch horizontally
        rt.anchorMin = new Vector2(0, 0.5f);
        rt.anchorMax = new Vector2(1, 0.5f);
        rt.pivot     = new Vector2(0.5f, 1);
        rt.anchoredPosition = new Vector2(0, yFromCenter);
        rt.sizeDelta = new Vector2(-80f, height); // 40 px horizontal padding each side

        return yFromCenter - height;
    }

    /// <summary>Adds a label at a horizontal offset from centre (aligned to slider edges).
    /// xOffset: horizontal pixel offset from centre. width: label box width.</summary>
    private void AddTextAnchored(Transform parent, string name, string text, int fontSize, Color color,
                                 FontStyle style, TextAnchor anchor, float yFromCenter, float height,
                                 float xOffset, float width)
    {
        var go = CreateUI(name, parent);
        var t  = go.AddComponent<Text>();
        t.text      = text;
        t.font      = Resources.GetBuiltinResource<Font>("LegacyRuntime.ttf");
        t.fontSize  = fontSize;
        t.fontStyle = style;
        t.color     = color;
        t.alignment = anchor;
        t.horizontalOverflow = HorizontalWrapMode.Wrap;

        var rt = go.GetComponent<RectTransform>();
        rt.anchorMin = rt.anchorMax = new Vector2(0.5f, 0.5f);
        rt.pivot     = new Vector2(0.5f, 1);
        rt.anchoredPosition = new Vector2(xOffset, yFromCenter);
        rt.sizeDelta = new Vector2(width, height);
    }

    private float BuildSlider(Transform parent, float yFromCenter)
    {
        // Container
        var sliderGO = CreateUI("Slider", parent);
        var sliderRT = sliderGO.GetComponent<RectTransform>();
        sliderRT.anchorMin = sliderRT.anchorMax = new Vector2(0.5f, 0.5f);
        sliderRT.pivot     = new Vector2(0.5f, 1);
        sliderRT.anchoredPosition = new Vector2(0, yFromCenter);
        sliderRT.sizeDelta = new Vector2(sliderWidth, sliderHeight);

        // Background
        var bgGO  = CreateUI("Background", sliderGO.transform);
        var bgImg = bgGO.AddComponent<Image>();
        bgImg.color = sliderBgColor;
        Stretch(bgGO);

        // Fill area
        var fillAreaGO = CreateUI("Fill Area", sliderGO.transform);
        Stretch(fillAreaGO);
        var fillAreaRT = fillAreaGO.GetComponent<RectTransform>();
        fillAreaRT.offsetMin = new Vector2(5, 0);
        fillAreaRT.offsetMax = new Vector2(-5, 0);

        var fillGO  = CreateUI("Fill", fillAreaGO.transform);
        var fillImg = fillGO.AddComponent<Image>();
        fillImg.color = sliderFillColor;
        Stretch(fillGO);

        // Centre tick mark
        var tickGO  = CreateUI("CenterTick", sliderGO.transform);
        var tickImg = tickGO.AddComponent<Image>();
        tickImg.color = new Color(1f, 1f, 1f, 0.6f);
        tickImg.raycastTarget = false;
        var tickRT = tickGO.GetComponent<RectTransform>();
        tickRT.anchorMin = new Vector2(0.5f, 0);
        tickRT.anchorMax = new Vector2(0.5f, 1);
        tickRT.sizeDelta = new Vector2(2f, 6f);    // 2 px wide, extends 3 px above and below
        tickRT.anchoredPosition = Vector2.zero;

        // Handle slide area
        var handleAreaGO = CreateUI("Handle Slide Area", sliderGO.transform);
        Stretch(handleAreaGO);
        var handleAreaRT = handleAreaGO.GetComponent<RectTransform>();
        handleAreaRT.offsetMin = new Vector2(10, 0);
        handleAreaRT.offsetMax = new Vector2(-10, 0);

        // Handle
        var handleGO  = CreateUI("Handle", handleAreaGO.transform);
        var handleImg = handleGO.AddComponent<Image>();
        handleImg.color = handleColor;
        var handleRT = handleGO.GetComponent<RectTransform>();
        handleRT.sizeDelta = new Vector2(handleSize, handleSize);

        // Slider component
        _slider = sliderGO.AddComponent<Slider>();
        _slider.fillRect      = fillGO.GetComponent<RectTransform>();
        _slider.handleRect    = handleRT;
        _slider.targetGraphic = handleImg;
        _slider.minValue      = 0f;
        _slider.maxValue      = 1f;
        _slider.value         = 0.5f;
        _slider.direction     = Slider.Direction.LeftToRight;

        return yFromCenter - sliderHeight;
    }

    private void BuildButton(Transform parent, float yFromCenter)
    {
        var btnGO = CreateUI("SubmitButton", parent);
        var btnImg = btnGO.AddComponent<Image>();
        btnImg.color = buttonColor;

        var btnRT = btnGO.GetComponent<RectTransform>();
        btnRT.anchorMin = btnRT.anchorMax = new Vector2(0.5f, 0.5f);
        btnRT.pivot     = new Vector2(0.5f, 1);
        btnRT.anchoredPosition = new Vector2(0, yFromCenter);
        btnRT.sizeDelta = new Vector2(buttonWidth, buttonHeight);

        _button = btnGO.AddComponent<Button>();
        _button.targetGraphic = btnImg;
        var colours = _button.colors;
        colours.normalColor      = buttonColor;
        colours.highlightedColor = buttonColor * 1.15f;
        colours.pressedColor     = buttonColor * 0.85f;
        colours.disabledColor    = new Color(0.3f, 0.3f, 0.3f);
        _button.colors = colours;

        var txtGO = CreateUI("Text", btnGO.transform);
        _buttonLabel = txtGO.AddComponent<Text>();
        _buttonLabel.text      = buttonText;
        _buttonLabel.font      = Resources.GetBuiltinResource<Font>("LegacyRuntime.ttf");
        _buttonLabel.fontSize  = buttonFontSize;
        _buttonLabel.fontStyle = FontStyle.Bold;
        _buttonLabel.color     = Color.white;
        _buttonLabel.alignment = TextAnchor.MiddleCenter;
        Stretch(txtGO);
    }

    // ── Tiny helpers ────────────────────────────────────────────────────

    private static GameObject CreateUI(string name, Transform parent)
    {
        var go = new GameObject(name, typeof(RectTransform));
        go.transform.SetParent(parent, false);
        return go;
    }

    private static void Stretch(GameObject go)
    {
        var rt = go.GetComponent<RectTransform>();
        rt.anchorMin = Vector2.zero;
        rt.anchorMax = Vector2.one;
        rt.offsetMin = rt.offsetMax = Vector2.zero;
    }
}
