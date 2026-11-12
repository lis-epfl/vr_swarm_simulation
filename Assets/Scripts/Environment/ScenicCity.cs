using System.Collections.Generic;
using UnityEngine;

/// <summary>
/// On a <see cref="ProceduralCityBuilder"/>'s root: its block-shaped buildings are drawn as the scenic city's
/// (<see cref="ScenicBuilding"/>, in <see cref="facades"/>) rather than as <see cref="PrismBuilding"/>'s flat-coloured
/// prisms. ScenicPatchCityWorld has it; a root without it builds exactly as before.
///
/// <para>The builder calls it at three points. <see cref="Deal"/>, before any building is placed: every building's
/// style, from the layout's seed. <see cref="Dress"/>, on each prism as it is made: the prism stays as the collider and
/// its renderer takes the scenic mesh -- the lite one in a bake, the detailed one at Play. <see cref="Finish"/>, when the
/// city is complete: at Play, a baked city's lite meshes are swapped for detailed ones and the buildings are batched
/// (they share one material); in a bake, the buildings are kept out of Unity's load-time static batching, which would
/// otherwise fix the lite meshes in place.</para>
///
/// <para>It runs in the editor too, only to hand the shader its texture arrays (<see cref="ScenicFacades.Apply"/>), so
/// the Scene view shows the baked city in its materials.</para>
/// </summary>
[ExecuteAlways]
[DisallowMultipleComponent]
[RequireComponent(typeof(ProceduralCityBuilder))]
public class ScenicCity : MonoBehaviour
{
    public ScenicFacades facades;

    private Dictionary<string, ScenicBuilding.Spec> specs;
    private string dealtFor = "";

    /// <summary>Every block-shaped building's style by name, once dealt (null before): for probes and analysis.</summary>
    public IReadOnlyDictionary<string, ScenicBuilding.Spec> Specs => specs;

    private bool Usable => facades != null && facades.material != null && facades.walls != null && facades.walls.Length > 0;

    private void OnEnable()
    {
        if (Usable) facades.Apply();
    }

    /// <summary>Every block-shaped building's style, from the layout's seed. A summary for the builder's log line.</summary>
    public string Deal(ProceduralCityBuilder.LayoutFile file)
    {
        if (!Usable)
        {
            specs = null;
            return "NOT scenic (the ScenicCity has no facades asset, or it has no material or walls)";
        }
        facades.Apply();
        specs = ScenicBuilding.Deal(file.boxes, facades, file.layout.seed, file.layout.p);
        dealtFor = file.layout.id ?? "";
        var families = new int[System.Enum.GetValues(typeof(ScenicBuilding.Family)).Length];
        float tallest = 0f;
        int raised = 0;
        foreach (ScenicBuilding.Spec s in specs.Values)
        {
            families[(int)s.family]++;
            tallest = Mathf.Max(tallest, s.extra);
            raised += s.extra > 2f ? 1 : 0;
        }
        var parts = new List<string>();
        foreach (ScenicBuilding.Family f in System.Enum.GetValues(typeof(ScenicBuilding.Family))) parts.Add($"{families[(int)f]} {f.ToString().ToLowerInvariant()}");
        return $"scenic: {string.Join(", ", parts)}; {raised} rise above their roofs, by up to {tallest:F0} m";
    }

    /// <summary>Every doorway of the city's buildings (<see cref="ScenicBuilding.Doors"/>), for the dressing to keep
    /// clear. Empty if the facades are not set up.</summary>
    public List<(Vector2 a, Vector2 b)> Doors(ProceduralCityBuilder.LayoutFile file)
    {
        var doors = new List<(Vector2, Vector2)>();
        if (!Usable) return doors;
        if (specs == null || dealtFor != (file.layout.id ?? "")) Deal(file);
        foreach (ProceduralCityBuilder.LayoutBox b in file.boxes)
        {
            if (PrismBuilding.IsPrism(b) && specs.TryGetValue(b.name, out ScenicBuilding.Spec spec)) doors.AddRange(ScenicBuilding.Doors(b, spec));
        }
        return doors;
    }

    /// <summary>Gives a prism its scenic mesh and material; its collider keeps the prism.</summary>
    public void Dress(GameObject prism, ProceduralCityBuilder.LayoutBox b)
    {
        if (!Usable || specs == null || !specs.TryGetValue(b.name, out ScenicBuilding.Spec spec)) return;
        MeshFilter filter = prism.GetComponent<MeshFilter>();
        MeshRenderer renderer = prism.GetComponent<MeshRenderer>();
        if (filter == null || renderer == null) return;
        filter.sharedMesh = ScenicBuilding.Build(b, spec, Application.isPlaying);
        renderer.sharedMaterials = new[] { facades.material };
        renderer.shadowCastingMode = UnityEngine.Rendering.ShadowCastingMode.Off;
        renderer.receiveShadows = false;
    }

    /// <summary>When the city is complete. At Play: every building has its detailed mesh, and they are batched.</summary>
    public void Finish(ProceduralCityBuilder.LayoutFile file)
    {
        if (!Usable) return;
        var clock = System.Diagnostics.Stopwatch.StartNew();
        if (specs == null || dealtFor != (file.layout.id ?? "")) Deal(file);
        var boxes = new Dictionary<string, ProceduralCityBuilder.LayoutBox>();
        foreach (ProceduralCityBuilder.LayoutBox b in file.boxes)
        {
            if (PrismBuilding.IsPrism(b)) boxes[b.name] = b;
        }
        var buildings = new List<GameObject>();
        int detailed = 0, fixedInPlace = 0;
        foreach (MeshCollider collider in GetComponentsInChildren<MeshCollider>(true))
        {
            if (!boxes.TryGetValue(collider.name, out ProceduralCityBuilder.LayoutBox b)) continue;
            MeshFilter filter = collider.GetComponent<MeshFilter>();
            MeshRenderer renderer = collider.GetComponent<MeshRenderer>();
            if (filter == null || renderer == null) continue;
#if UNITY_EDITOR
            if (!Application.isPlaying)
            {
                var flags = UnityEditor.GameObjectUtility.GetStaticEditorFlags(collider.gameObject);
                UnityEditor.GameObjectUtility.SetStaticEditorFlags(collider.gameObject, flags & ~UnityEditor.StaticEditorFlags.BatchingStatic);
                continue;
            }
#endif
            if (renderer.isPartOfStaticBatch)
            {
                fixedInPlace++;          // saved batching-static: its mesh is already part of a batch
                continue;
            }
            if (ScenicBuilding.IsLite(filter.sharedMesh) || renderer.sharedMaterial != facades.material)
            {
                Dress(collider.gameObject, b);
                detailed++;
            }
            buildings.Add(collider.gameObject);
        }
        if (!Application.isPlaying) return;
        if (fixedInPlace > 0)
        {
            Debug.LogWarning($"ScenicCity: {fixedInPlace} buildings were saved batching-static, so they are drawn without " +
                             "their relief and roof plant. Bake the city again (the bake keeps them out of static batching).", this);
        }
        long vertices = 0;
        foreach (GameObject b in buildings) vertices += b.GetComponent<MeshFilter>().sharedMesh.vertexCount;
        double meshMs = clock.Elapsed.TotalMilliseconds;
        if (buildings.Count > 0) StaticBatchingUtility.Combine(buildings.ToArray(), gameObject);
        Debug.Log($"ScenicCity: {buildings.Count} buildings in {facades.name} ({detailed} given their relief at Play, " +
                  $"{vertices} vertices, {meshMs:F0} ms), batched ({clock.Elapsed.TotalMilliseconds - meshMs:F0} ms).", this);
    }
}
