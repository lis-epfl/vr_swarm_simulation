using System.Collections.Generic;
using UnityEngine;

public class DroneIndicator : MonoBehaviour
{
    [Tooltip("Distance from the arena centre to place each indicator.")]
    [SerializeField] private float radius = 2.1f;

    [Tooltip("Height above the arena origin.")]
    [SerializeField] private float heightOffset = 0.1f;

    [Tooltip("Uniform scale applied to the triangle mesh.")]
    [SerializeField] private float triangleScale = 0.25f;

    [Tooltip("Fixed angular separation between indicators (degrees).")]
    public float separationDegrees = 20f;

    // Arena reference resolved once in SpawnIndicators
    private Transform arenaTransform;

    // One entry per drone
    private readonly List<GameObject> indicators = new List<GameObject>();

    // ------------------------------------------------------------------
    // Public API
    // ------------------------------------------------------------------

    public void SpawnIndicators(int count)
    {
        foreach (GameObject go in indicators)
        {
            if (go != null) Destroy(go);
        }
        indicators.Clear();

        GameObject arenaGO = GameObject.Find("Arena");
        arenaTransform = arenaGO != null ? arenaGO.transform : null;
        Vector3 origin = arenaTransform != null ? arenaTransform.position : Vector3.zero;

        for (int i = 0; i < count; i++)
        {
            GameObject go = CreateTriangleObject(i);

            // Change the order
            int k = i;
            if (i == 1) { k = 2; }
            else if (i == 2) { k = 1; }

            // Fixed position: evenly spaced by separationDegrees around the arena
            float radians = k * separationDegrees * Mathf.Deg2Rad;
            go.transform.position = new Vector3(
                origin.x - radius * Mathf.Cos(radians),
                origin.y + heightOffset,
                origin.z + radius * Mathf.Sin(radians)
            );
            go.SetActive(true);

            indicators.Add(go);
        }

        Debug.Log($"[DroneIndicator] Spawned {count} indicator(s) with {separationDegrees}° separation.");
    }

    public void UpdateYaw(int droneIndex, float yaw)
    {
        if (droneIndex < 0 || droneIndex >= indicators.Count)
        {
            Debug.LogWarning($"[DroneIndicator] UpdateYaw: index {droneIndex} out of range (count={indicators.Count}).");
            return;
        }

        // Only rotation changes — position is fixed at spawn
        indicators[droneIndex].transform.rotation = Quaternion.Euler(0f, yaw + 90f, 0f);
    }

    // ------------------------------------------------------------------
    // Private helpers
    // ------------------------------------------------------------------

    private GameObject CreateTriangleObject(int index)
    {
        GameObject go = new GameObject($"drone_indicator_{index}");

        MeshFilter mf = go.AddComponent<MeshFilter>();
        MeshRenderer mr = go.AddComponent<MeshRenderer>();

        mf.mesh = BuildTriangleMesh();

        // Unlit white material so the triangle is clearly visible regardless of lighting
        Material mat = new Material(Shader.Find("Unlit/Color"));
        mat.color = Color.white;
        mr.material = mat;

        go.transform.localScale = Vector3.one * triangleScale;

        return go;
    }

    /// <summary>
    /// Builds a flat isosceles triangle lying in the XZ plane.
    /// The tip points along local +Z (forward). Both faces are rendered so
    /// the triangle is visible from above and below.
    /// </summary>
    private static Mesh BuildTriangleMesh()
    {
        Mesh mesh = new Mesh();
        mesh.name = "DroneIndicatorTriangle";

        // Tip at +Z, base spanning −Z
        Vector3 tip       = new Vector3( 0.00f, 0f,  0.50f);
        Vector3 baseLeft  = new Vector3(-0.35f, 0f, -0.35f);
        Vector3 baseRight = new Vector3( 0.35f, 0f, -0.35f);

        // Two sets of identical vertices so each face can have its own normal
        mesh.vertices = new Vector3[]
        {
            // Top face (normal = up)
            tip, baseRight, baseLeft,
            // Bottom face (normal = down)
            tip, baseLeft, baseRight,
        };

        mesh.triangles = new int[]
        {
            0, 1, 2,   // top face (clockwise when viewed from +Y)
            3, 4, 5,   // bottom face (clockwise when viewed from −Y)
        };

        mesh.normals = new Vector3[]
        {
            Vector3.up,   Vector3.up,   Vector3.up,
            Vector3.down, Vector3.down, Vector3.down,
        };

        mesh.RecalculateBounds();
        return mesh;
    }
}