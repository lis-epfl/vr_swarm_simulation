using System.Collections.Generic;
using UnityEngine;

/// <summary>
/// Generates a procedural torus (ring) mesh and its physics collider.
/// Updates automatically in edit mode when parameters change via OnValidate.
///
/// The torus lies in the local XY plane; its axis / hole faces along local +Z.
/// This matches the convention used by RingGate for radial-distance detection.
///
/// Two collider strategies are available:
///   MeshCollider        — exact shape, one component. Best accuracy.
///   CapsuleApproximation — N CapsuleColliders as children. Convex-friendly,
///                          lower runtime cost, works with all Rigidbody modes.
/// </summary>
[RequireComponent(typeof(MeshFilter), typeof(MeshRenderer))]
[ExecuteAlways]
public class RingMeshGenerator : MonoBehaviour
{
    // ─────────────────────────────────────────────────────────────────────────
    // Types
    // ─────────────────────────────────────────────────────────────────────────

    public enum ColliderMode
    {
        None,
        MeshCollider,
        CapsuleApproximation
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Inspector
    // ─────────────────────────────────────────────────────────────────────────

    [Header("Torus Geometry")]
    [Tooltip("Major radius: distance from the ring centre to the tube centre-line. " +
             "Should match RingGate.ringRadius.")]
    [Min(0.1f)] public float ringRadius  = 2.75f;

    [Tooltip("Minor radius: cross-section radius of the tube.")]
    [Min(0.01f)] public float tubeRadius = 0.20f;

    [Header("Visual Resolution")]
    [Tooltip("Segments around the major (ring) circumference.")]
    [Range(8, 128)] public int ringSegments = 64;

    [Tooltip("Segments around the tube cross-section.")]
    [Range(4, 32)]  public int tubeSegments = 16;

    [Header("Physics Collider")]
    [Tooltip(
        "None                — no physical collider (drones pass through).\n" +
        "MeshCollider        — exact torus shape. Most accurate, slightly higher PhysX cost.\n" +
        "CapsuleApproximation — N CapsuleColliders placed around the ring. Convex-safe,\n" +
        "                       good performance. Recommended for most experiments.")]
    public ColliderMode colliderMode = ColliderMode.CapsuleApproximation;

    [Tooltip("Number of capsules used in CapsuleApproximation mode. 16–24 is visually seamless.")]
    [Range(8, 48)] public int capsuleCount = 20;

    [Tooltip("Physics material applied to the ring collider (optional — e.g. for bounce/friction tuning).")]
    public PhysicMaterial ringPhysicsMaterial;

    // ─────────────────────────────────────────────────────────────────────────
    // Private
    // ─────────────────────────────────────────────────────────────────────────

    private MeshFilter   _meshFilter;
    private MeshCollider _meshCollider;

    // Parent for capsule child objects so they don't clutter the root hierarchy
    private Transform _capsuleParent;
    private readonly List<CapsuleCollider> _capsules = new();

    // Parameter snapshot for change detection in OnValidate
    private float        _cachedRR, _cachedTR;
    private int          _cachedRS, _cachedTS, _cachedCC;
    private ColliderMode _cachedMode;

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
            !Mathf.Approximately(_cachedRR, ringRadius) ||
            !Mathf.Approximately(_cachedTR, tubeRadius) ||
            _cachedRS   != ringSegments ||
            _cachedTS   != tubeSegments ||
            _cachedCC   != capsuleCount ||
            _cachedMode != colliderMode;

        if (!changed) return;

#if UNITY_EDITOR
        UnityEditor.EditorApplication.delayCall += GenerateMesh;
#endif
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Mesh + collider generation
    // ─────────────────────────────────────────────────────────────────────────

    /// <summary>
    /// Rebuilds the torus visual mesh and its physics collider.
    /// Safe to call at runtime — e.g. when resizing the ring between trials.
    /// </summary>
    public void GenerateMesh()
    {
        if (this == null) return;

        if (_meshFilter == null)
            _meshFilter = GetComponent<MeshFilter>();

        Mesh mesh = BuildTorusMesh();
        _meshFilter.sharedMesh = mesh;

        RebuildCollider(mesh);

        _cachedRR   = ringRadius;
        _cachedTR   = tubeRadius;
        _cachedRS   = ringSegments;
        _cachedTS   = tubeSegments;
        _cachedCC   = capsuleCount;
        _cachedMode = colliderMode;
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Collider strategy
    // ─────────────────────────────────────────────────────────────────────────

    private void RebuildCollider(Mesh mesh)
    {
        RemoveExistingColliders();

        switch (colliderMode)
        {
            case ColliderMode.MeshCollider:
                BuildMeshCollider(mesh);
                break;

            case ColliderMode.CapsuleApproximation:
                BuildCapsuleApproximation();
                break;

            case ColliderMode.None:
            default:
                break;
        }
    }

    // ── MeshCollider ──────────────────────────────────────────────────────────

    private void BuildMeshCollider(Mesh mesh)
    {
        _meshCollider = gameObject.GetComponent<MeshCollider>();
        if (_meshCollider == null)
            _meshCollider = gameObject.AddComponent<MeshCollider>();

        // A torus is concave, so convex MUST be off.
        // Non-convex MeshColliders collide correctly with any primitive collider
        // (box, capsule, sphere) on the drone — which is the typical setup.
        _meshCollider.convex         = false;
        _meshCollider.sharedMesh     = mesh;
        _meshCollider.sharedMaterial = ringPhysicsMaterial;
    }

    // ── Capsule approximation ─────────────────────────────────────────────────

    /// <summary>
    /// Places <see cref="capsuleCount"/> CapsuleColliders evenly around the ring
    /// circumference. Each capsule is tangent to the ring at its position and sized
    /// so adjacent capsules overlap, leaving no physics gaps.
    /// </summary>
    private void BuildCapsuleApproximation()
    {
#if UNITY_EDITOR
        // Creating child GameObjects inside a Prefab Asset is not allowed by Unity.
        // The colliders are only needed in scene instances, so skip when editing the asset directly.
        if (UnityEditor.EditorUtility.IsPersistent(gameObject)) return;
#endif

        // Re-use or create the hidden parent object
        if (_capsuleParent == null)
        {
            var existing = transform.Find("__RingColliders");
            _capsuleParent = existing != null
                ? existing
                : new GameObject("__RingColliders").transform;

            _capsuleParent.SetParent(transform, false);
            _capsuleParent.localPosition = Vector3.zero;
            _capsuleParent.localRotation = Quaternion.identity;
            _capsuleParent.localScale    = Vector3.one;

#if UNITY_EDITOR
            _capsuleParent.gameObject.hideFlags = HideFlags.HideInHierarchy;
#endif
        }

        // Clear old capsule children
        _capsules.Clear();
        for (int k = _capsuleParent.childCount - 1; k >= 0; k--)
            DestroyImmediate(_capsuleParent.GetChild(k).gameObject);

        // Capsule height: arc between centres + end-sphere overlap on each side
        float arcStep   = 2f * Mathf.PI * ringRadius / capsuleCount;
        float capHeight = arcStep + tubeRadius * 2f;

        for (int i = 0; i < capsuleCount; i++)
        {
            float theta = i * Mathf.PI * 2f / capsuleCount;

            // Centre of this capsule — on the ring centreline, in the XY plane
            var pos = new Vector3(Mathf.Cos(theta) * ringRadius,
                                  Mathf.Sin(theta) * ringRadius,
                                  0f);

            // Tangent direction: perpendicular to radial, stays in XY plane
            var tangent = new Vector3(-Mathf.Sin(theta), Mathf.Cos(theta), 0f);

            var go = new GameObject($"Cap_{i:00}");
            go.transform.SetParent(_capsuleParent, false);
            go.transform.localPosition = pos;
            // Align capsule axis (local X after direction=0) with the tangent.
            // LookRotation points local Z toward tangent; the extra Euler aligns X instead.
            go.transform.localRotation = Quaternion.LookRotation(Vector3.forward, tangent)
                                         * Quaternion.Euler(0f, 0f, 90f);

            var cap = go.AddComponent<CapsuleCollider>();
            cap.direction      = 0;         // along local X
            cap.radius         = tubeRadius;
            cap.height         = capHeight;
            cap.sharedMaterial = ringPhysicsMaterial;

            _capsules.Add(cap);
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Cleanup
    // ─────────────────────────────────────────────────────────────────────────

    private void RemoveExistingColliders()
    {
        var mc = GetComponent<MeshCollider>();
        if (mc != null) DestroyImmediate(mc);
        _meshCollider = null;

        if (_capsuleParent != null)
        {
            DestroyImmediate(_capsuleParent.gameObject);
            _capsuleParent = null;
        }
        _capsules.Clear();
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Torus mesh builder
    // ─────────────────────────────────────────────────────────────────────────

    private Mesh BuildTorusMesh()
    {
        int vertsPerRow = tubeSegments + 1;
        int totalVerts  = (ringSegments + 1) * vertsPerRow;

        var vertices  = new Vector3[totalVerts];
        var normals   = new Vector3[totalVerts];
        var uvs       = new Vector2[totalVerts];

        for (int i = 0; i <= ringSegments; i++)
        {
            float u     = (float)i / ringSegments;
            float theta = u * Mathf.PI * 2f;

            var tubeCentre = new Vector3(Mathf.Cos(theta) * ringRadius,
                                         Mathf.Sin(theta) * ringRadius,
                                         0f);
            var radialDir  = new Vector3(Mathf.Cos(theta), Mathf.Sin(theta), 0f);

            for (int j = 0; j <= tubeSegments; j++)
            {
                float v      = (float)j / tubeSegments;
                float phi    = v * Mathf.PI * 2f;
                var   normal = radialDir * Mathf.Cos(phi) + Vector3.forward * Mathf.Sin(phi);

                int idx = i * vertsPerRow + j;
                vertices[idx] = tubeCentre + normal * tubeRadius;
                normals[idx]  = normal;
                uvs[idx]      = new Vector2(u, v);
            }
        }

        var triangles = new int[ringSegments * tubeSegments * 6];
        int t = 0;
        for (int i = 0; i < ringSegments; i++)
        {
            for (int j = 0; j < tubeSegments; j++)
            {
                int a = i       * vertsPerRow + j;
                int b = (i + 1) * vertsPerRow + j;

                triangles[t++] = a;
                triangles[t++] = a + 1;
                triangles[t++] = b;

                triangles[t++] = b;
                triangles[t++] = a + 1;
                triangles[t++] = b + 1;
            }
        }

        var mesh = new Mesh { name = "RingGate_Torus" };
        mesh.indexFormat = UnityEngine.Rendering.IndexFormat.UInt32;
        mesh.vertices    = vertices;
        mesh.normals     = normals;
        mesh.uv          = uvs;
        mesh.triangles   = triangles;
        mesh.RecalculateBounds();
        mesh.RecalculateTangents();
        return mesh;
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Convenience
    // ─────────────────────────────────────────────────────────────────────────

    /// <summary>Syncs geometry from the parent RingGate component and regenerates.</summary>
    public void SyncFromGate()
    {
        var gate = GetComponent<RingGate>();
        if (gate == null) return;
        ringRadius = gate.gateWidth * 0.5f;
        tubeRadius = gate.barThickness;
        GenerateMesh();
    }
}
