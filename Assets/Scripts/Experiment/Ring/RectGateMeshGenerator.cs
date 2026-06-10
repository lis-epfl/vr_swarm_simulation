using UnityEngine;

/// <summary>
/// Generates a rectangular gate frame from 4 box-shaped bars as child GameObjects.
/// Each bar has a MeshFilter, MeshRenderer, and BoxCollider on the "Obstacle" layer
/// so drones perform obstacle avoidance via OlfatiSaber.
///
/// The gate lies in the local XY plane; its through-axis faces local +Z,
/// matching the convention used by RingGate and GateTriggerRelay.
/// </summary>
[ExecuteAlways]
public class RectGateMeshGenerator : MonoBehaviour
{
    // ─────────────────────────────────────────────────────────────────────────
    // Inspector
    // ─────────────────────────────────────────────────────────────────────────

    [Header("Gate Geometry")]
    [Tooltip("Full interior opening width (X axis).")]
    [Min(0.5f)] public float width = 5.5f;

    [Tooltip("Full interior opening height (Y axis).")]
    [Min(0.5f)] public float height = 5.5f;

    [Tooltip("Cross-section thickness of each bar.")]
    [Min(0.01f)] public float barThickness = 0.20f;

    [Header("Visuals")]
    [Tooltip("Material applied to all four frame bars.")]
    public Material frameMaterial;

    // ─────────────────────────────────────────────────────────────────────────
    // Private
    // ─────────────────────────────────────────────────────────────────────────

    private GameObject[] _bars;
    private static readonly string[] BarNames = { "Bar_Top", "Bar_Bottom", "Bar_Left", "Bar_Right" };

    // Change-detection cache
    private float _cachedW, _cachedH, _cachedT;

    // ─────────────────────────────────────────────────────────────────────────
    // Lifecycle
    // ─────────────────────────────────────────────────────────────────────────

    private void OnEnable()
    {
        GenerateMesh();
    }

    private void OnValidate()
    {
        bool changed =
            !Mathf.Approximately(_cachedW, width) ||
            !Mathf.Approximately(_cachedH, height) ||
            !Mathf.Approximately(_cachedT, barThickness);

        if (!changed) return;

#if UNITY_EDITOR
        UnityEditor.EditorApplication.delayCall += GenerateMesh;
#endif
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Public API
    // ─────────────────────────────────────────────────────────────────────────

    /// <summary>
    /// Rebuilds the 4 bar children. Safe to call at runtime.
    /// </summary>
    public void GenerateMesh()
    {
        if (this == null) return;

#if UNITY_EDITOR
        if (UnityEditor.EditorUtility.IsPersistent(gameObject)) return;
#endif

        EnsureBars();

        float w = width;
        float h = height;
        float t = barThickness;

        // Top bar: spans full width at the top edge
        UpdateBar(0, new Vector3(0f, (h + t) * 0.5f, 0f), new Vector3(w + 2f * t, t, t));

        // Bottom bar: spans full width at the bottom edge
        UpdateBar(1, new Vector3(0f, -(h + t) * 0.5f, 0f), new Vector3(w + 2f * t, t, t));

        // Left bar: spans inner height between top and bottom bars
        UpdateBar(2, new Vector3(-(w + t) * 0.5f, 0f, 0f), new Vector3(t, h, t));

        // Right bar: spans inner height between top and bottom bars
        UpdateBar(3, new Vector3((w + t) * 0.5f, 0f, 0f), new Vector3(t, h, t));

        _cachedW = width;
        _cachedH = height;
        _cachedT = barThickness;
    }

    /// <summary>Syncs geometry from the sibling RingGate component and regenerates.</summary>
    public void SyncFromGate()
    {
        var gate = GetComponent<RingGate>();
        if (gate == null) return;
        width = gate.gateWidth;
        height = gate.gateHeight;
        barThickness = gate.barThickness;
        GenerateMesh();
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Bar management
    // ─────────────────────────────────────────────────────────────────────────

    private void EnsureBars()
    {
        if (_bars != null && _bars.Length == 4 && _bars[0] != null)
            return;

        _bars = new GameObject[4];
        for (int i = 0; i < 4; i++)
        {
            Transform existing = transform.Find(BarNames[i]);
            if (existing != null)
            {
                _bars[i] = existing.gameObject;
                continue;
            }

            var go = new GameObject(BarNames[i]);
            go.transform.SetParent(transform, false);

            go.AddComponent<MeshFilter>();
            go.AddComponent<MeshRenderer>();
            go.AddComponent<BoxCollider>();

            int obstacleLayer = LayerMask.NameToLayer("Obstacle");
            if (obstacleLayer >= 0)
                go.layer = obstacleLayer;
            else
                Debug.LogWarning($"[RectGateMeshGenerator] 'Obstacle' layer not found. " +
                                 $"Bar '{BarNames[i]}' will use default layer. " +
                                 $"Create the Obstacle layer in Project Settings > Tags and Layers.");

            _bars[i] = go;
        }
    }

    private void UpdateBar(int index, Vector3 localCenter, Vector3 size)
    {
        GameObject bar = _bars[index];
        bar.transform.localPosition = localCenter;
        bar.transform.localRotation = Quaternion.identity;
        bar.transform.localScale = Vector3.one;

        // Mesh — unit cube scaled to size
        var mf = bar.GetComponent<MeshFilter>();
        mf.sharedMesh = BuildBoxMesh(size);

        // Material
        var mr = bar.GetComponent<MeshRenderer>();
        if (frameMaterial != null)
            mr.sharedMaterial = frameMaterial;
        mr.shadowCastingMode = UnityEngine.Rendering.ShadowCastingMode.Off;

        // Collider
        var col = bar.GetComponent<BoxCollider>();
        col.center = Vector3.zero;
        col.size = size;

        // Ensure layer
        int obstacleLayer = LayerMask.NameToLayer("Obstacle");
        if (obstacleLayer >= 0)
            bar.layer = obstacleLayer;
    }

    private static Mesh BuildBoxMesh(Vector3 size)
    {
        float x = size.x * 0.5f;
        float y = size.y * 0.5f;
        float z = size.z * 0.5f;

        var vertices = new Vector3[]
        {
            // Front face (z+)
            new(-x, -y, z), new(x, -y, z), new(x, y, z), new(-x, y, z),
            // Back face (z-)
            new(x, -y, -z), new(-x, -y, -z), new(-x, y, -z), new(x, y, -z),
            // Top face (y+)
            new(-x, y, z), new(x, y, z), new(x, y, -z), new(-x, y, -z),
            // Bottom face (y-)
            new(-x, -y, -z), new(x, -y, -z), new(x, -y, z), new(-x, -y, z),
            // Right face (x+)
            new(x, -y, z), new(x, -y, -z), new(x, y, -z), new(x, y, z),
            // Left face (x-)
            new(-x, -y, -z), new(-x, -y, z), new(-x, y, z), new(-x, y, -z),
        };

        var normals = new Vector3[]
        {
            Vector3.forward, Vector3.forward, Vector3.forward, Vector3.forward,
            Vector3.back, Vector3.back, Vector3.back, Vector3.back,
            Vector3.up, Vector3.up, Vector3.up, Vector3.up,
            Vector3.down, Vector3.down, Vector3.down, Vector3.down,
            Vector3.right, Vector3.right, Vector3.right, Vector3.right,
            Vector3.left, Vector3.left, Vector3.left, Vector3.left,
        };

        var triangles = new int[]
        {
            0,2,1, 0,3,2,
            4,6,5, 4,7,6,
            8,10,9, 8,11,10,
            12,14,13, 12,15,14,
            16,18,17, 16,19,18,
            20,22,21, 20,23,22,
        };

        var mesh = new Mesh { name = "RectGate_Bar" };
        mesh.vertices = vertices;
        mesh.normals = normals;
        mesh.triangles = triangles;
        mesh.RecalculateBounds();
        return mesh;
    }
}
