using System.Collections.Generic;
using UnityEngine;

/// <summary>
/// A soft dark patch on the ground under every pedestrian, which anchors a small figure to the footpath in a drone's
/// feed. <see cref="ProceduralCityBuilder"/> adds this at Play when its scene assigns a walkerShadowMaterial, so no
/// other scene is touched.
///
/// <para><b>One mesh, not one object per walker.</b> It is rewritten once a frame and costs each FPV render one draw
/// call, however many walkers there are and whichever of them are in view. A renderer per walker would be culled and
/// batched again in every render, and the feeds' frame cost is per render.</para>
///
/// <para><b>Nothing here may be parented under a walker.</b> <see cref="SpecialWalker.HatCentre"/> takes every rigid
/// mesh a walker carries to be its hat, and the recorder scores identifications against that.</para>
/// </summary>
// Before ScreenSpawn's LateUpdate, which renders the feeds, and after every WalkerPatrol has moved its walkers in Update.
[DefaultExecutionOrder(-100)]
public class WalkerGroundShadows : MonoBehaviour
{
    private const float Radius = 0.8f;
    // Above CityDressing's 0.1 m footpath, which the walkers' origins stand under, and clear of it at range.
    private const float GroundHeight = 0.12f;

    private Material material;
    private readonly List<Animator> walkers = new List<Animator>();
    private Mesh mesh;
    private Vector3[] vertices;

    public void Init(Material shadowMaterial)
    {
        material = shadowMaterial;
    }

    // In the first LateUpdate rather than Start: a goal's WalkerPatrols spawn their walkers in their own Start, and
    // this frame's have all run by now. A walker spawned later has no shadow, which no scene using this has.
    private void CollectWalkers()
    {
        foreach (Animator animator in FindObjectsByType<Animator>(FindObjectsSortMode.None))
        {
            if (animator.GetComponentInChildren<SkinnedMeshRenderer>(true) != null) walkers.Add(animator);
        }
        if (walkers.Count == 0)
        {
            enabled = false;   // the pedestrians are off (the headless bench)
            return;
        }

        int n = walkers.Count;
        vertices = new Vector3[n * 4];
        var uv = new Vector2[n * 4];
        var triangles = new int[n * 6];
        for (int i = 0; i < n; i++)
        {
            int v = i * 4, t = i * 6;
            uv[v] = new Vector2(-1f, -1f);
            uv[v + 1] = new Vector2(-1f, 1f);
            uv[v + 2] = new Vector2(1f, 1f);
            uv[v + 3] = new Vector2(1f, -1f);
            triangles[t] = v;
            triangles[t + 1] = v + 1;
            triangles[t + 2] = v + 2;
            triangles[t + 3] = v;
            triangles[t + 4] = v + 2;
            triangles[t + 5] = v + 3;
        }

        mesh = new Mesh { name = "Walker ground shadows", indexFormat = UnityEngine.Rendering.IndexFormat.UInt32 };
        mesh.MarkDynamic();
        mesh.vertices = vertices;
        mesh.uv = uv;
        mesh.triangles = triangles;
        // The goals stand across the city, so one mesh is nearly always in a frustum anyway: skip the bounds upkeep.
        mesh.bounds = new Bounds(Vector3.zero, Vector3.one * 100000f);

        var go = new GameObject("Walker ground shadows");
        go.AddComponent<MeshFilter>().sharedMesh = mesh;
        var meshRenderer = go.AddComponent<MeshRenderer>();
        meshRenderer.sharedMaterial = material;
        meshRenderer.shadowCastingMode = UnityEngine.Rendering.ShadowCastingMode.Off;
        meshRenderer.receiveShadows = false;
        meshRenderer.lightProbeUsage = UnityEngine.Rendering.LightProbeUsage.Off;
        meshRenderer.reflectionProbeUsage = UnityEngine.Rendering.ReflectionProbeUsage.Off;
        meshRenderer.allowOcclusionWhenDynamic = false;
    }

    void LateUpdate()
    {
        if (mesh == null)
        {
            CollectWalkers();
            if (mesh == null) return;
        }

        for (int i = 0; i < walkers.Count; i++)
        {
            int v = i * 4;
            Animator walker = walkers[i];
            // A sleeping patrol's walkers are undrawn (WalkerPatrol.SleepDistance), so their shadows must go with them.
            if (walker == null || !walker.enabled)
            {
                vertices[v] = vertices[v + 1] = vertices[v + 2] = vertices[v + 3] = Vector3.zero;   // a degenerate quad
                continue;
            }
            Vector3 p = walker.transform.position;
            float y = p.y + GroundHeight;
            vertices[v] = new Vector3(p.x - Radius, y, p.z - Radius);
            vertices[v + 1] = new Vector3(p.x - Radius, y, p.z + Radius);
            vertices[v + 2] = new Vector3(p.x + Radius, y, p.z + Radius);
            vertices[v + 3] = new Vector3(p.x + Radius, y, p.z - Radius);
        }
        mesh.vertices = vertices;
    }

    void OnDestroy()
    {
        if (mesh != null) Destroy(mesh);
    }
}
