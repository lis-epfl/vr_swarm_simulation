using UnityEngine;

/// <summary>
/// In-headset debug visualization of the combined eye gaze, for bring-up / verification.
/// Draws a small cursor a fixed distance along the gaze ray and (optionally) a line from
/// the eyes. Pure consumer of EyeGazeTracker's static API — no coupling to its internals.
///
/// Off by default; toggle with <see cref="toggleKey"/>. Remove or leave disabled for studies.
/// </summary>
public class GazeDebugCursor : MonoBehaviour
{
    [Header("Gaze Debug Cursor")]
    [SerializeField]
    [Tooltip("Key that toggles the gaze cursor on/off (matches the calibrateKey convention).")]
    private KeyCode toggleKey = KeyCode.G;

    [SerializeField]
    [Tooltip("Whether the cursor starts visible. Off by default so it doesn't intrude on normal runs.")]
    private bool visible = false;

    [SerializeField]
    [Tooltip("Distance (metres) along the gaze ray to place the cursor.")]
    private float rayDistance = 5f;

    [SerializeField]
    [Tooltip("Diameter (metres) of the cursor sphere.")]
    private float cursorSize = 0.05f;

    [SerializeField]
    [Tooltip("Also draw a line from the eyes to the cursor.")]
    private bool showRay = true;

    [SerializeField]
    [Tooltip("Colour of the cursor and ray.")]
    private Color color = new Color(0.2f, 0.9f, 1f, 1f);

    private Transform cursor;
    private Renderer cursorRenderer;
    private LineRenderer line;

    private void Start()
    {
        // Unlit, emissive-ish sphere so it reads clearly in VR without depending on scene lighting.
        Material mat = new Material(Shader.Find("Unlit/Color"));
        mat.color = color;

        GameObject sphere = GameObject.CreatePrimitive(PrimitiveType.Sphere);
        sphere.name = "GazeCursor";
        Destroy(sphere.GetComponent<Collider>());
        cursor = sphere.transform;
        cursor.SetParent(transform, false);
        cursor.localScale = Vector3.one * cursorSize;
        cursorRenderer = sphere.GetComponent<Renderer>();
        cursorRenderer.sharedMaterial = mat;

        line = gameObject.AddComponent<LineRenderer>();
        line.material = mat;
        line.widthMultiplier = 0.005f;
        line.positionCount = 2;
        line.useWorldSpace = true;

        SetShown(false);
    }

    private void Update()
    {
        if (Input.GetKeyDown(toggleKey))
        {
            visible = !visible;
        }

        EyeGazeTracker.GazeSample gaze = EyeGazeTracker.Combined;
        bool show = visible && gaze.IsValid;
        SetShown(show);
        if (!show) return;

        Vector3 hit = gaze.Origin + gaze.Direction * rayDistance;
        cursor.position = hit;

        if (showRay)
        {
            line.SetPosition(0, gaze.Origin);
            line.SetPosition(1, hit);
        }
    }

    private void SetShown(bool show)
    {
        if (cursorRenderer != null) cursorRenderer.enabled = show;
        if (line != null) line.enabled = show && showRay;
    }
}
