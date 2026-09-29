using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Rendering;

/// <summary>
/// Merges a drone's rigid body parts into one renderer per material, at spawn.
/// </summary>
/// <remarks>
/// The DJI Mini 3 Pro model (<c>Assets/Models/DJIMini3Pro.prefab</c>, a glTFast import) is 96
/// MeshRenderers on seven materials, all rigidly attached to <c>DroneParent</c> except the four
/// propellers <c>VelocityControl</c> spins. Every FPV render that has another drone in view draws
/// each of those 96 as its own draw call -- the FPV renders are the whole CPU cost of this project
/// (see <c>ScreenSpawn.feedRenderHz</c>) -- and <c>FPVCameraScript</c> toggles all 96 off and on
/// again around every render of the drone's own camera. Merged, a drone is one renderer carrying a
/// submesh per material plus the four propellers: the same triangles and materials, drawn in about
/// a tenth of the calls.
///
/// Must run before anything caches the drone's renderers (<c>FPVCameraScript</c> and
/// <c>DroneHealthMonitor</c>, both in <c>Start</c>), which is why <c>SwarmSpawn</c> adds it right
/// after instantiating the drone and it merges in <c>Awake</c>. The originals are destroyed, not
/// disabled: <c>DroneHealthMonitor.UnparkDrone</c> re-enables every renderer it cached, which
/// would bring a disabled original back on top of the merged copy. Its <c>r != null</c> check
/// already skips destroyed ones.
///
/// Parts under a mirroring transform (the model has two nodes at scale -10) are merged into a
/// second renderer on a child at scale -1. Baking a mirror into vertices reverses the triangles'
/// winding, and rather than depend on whether <c>CombineMeshes</c> compensates, the mirror is kept
/// in a transform, where the renderer flips culling exactly as it did for the originals.
///
/// Leaves the drone untouched, with one warning per session, if any part's mesh is not readable.
/// </remarks>
[DisallowMultipleComponent]
public class DroneBodyCombiner : MonoBehaviour
{
    private static bool warnedUnreadable;

    private readonly List<Mesh> createdMeshes = new List<Mesh>();

    private struct Part
    {
        public Mesh mesh;
        public int subMesh;
        public Material material;
        public Matrix4x4 toLocal;   // part space -> this transform's space (mirror removed if mirrored)
    }

    private void Awake()
    {
        Combine();
    }

    private void OnDestroy()
    {
        foreach (Mesh m in createdMeshes)
        {
            if (m != null) Destroy(m);
        }
        createdMeshes.Clear();
    }

    private void Combine()
    {
        List<Transform> propellers = Propellers();
        var parts = new List<Part>();
        var mirroredParts = new List<Part>();
        var merged = new List<MeshRenderer>();
        MeshRenderer template = null;
        int bodyLayer = -1;

        Matrix4x4 worldToLocal = transform.worldToLocalMatrix;
        Matrix4x4 mirror = Matrix4x4.Scale(new Vector3(-1f, -1f, -1f));

        foreach (MeshFilter filter in GetComponentsInChildren<MeshFilter>(false))
        {
            MeshRenderer renderer = filter.GetComponent<MeshRenderer>();
            Mesh mesh = filter.sharedMesh;
            if (renderer == null || !renderer.enabled || mesh == null) continue;
            if (IsUnderAny(filter.transform, propellers)) continue;

            // One layer per merged renderer; anything else keeps its own renderer.
            if (bodyLayer < 0) bodyLayer = filter.gameObject.layer;
            if (filter.gameObject.layer != bodyLayer) continue;

            if (!mesh.isReadable)
            {
                if (!warnedUnreadable)
                {
                    warnedUnreadable = true;
                    Debug.LogWarning($"[DroneBodyCombiner] Mesh '{mesh.name}' is not readable, so the drone " +
                                     "bodies are left unmerged (96 renderers each). Enable read/write on the " +
                                     "drone model's meshes to merge them.", this);
                }
                return;
            }

            Matrix4x4 toLocal = worldToLocal * filter.transform.localToWorldMatrix;
            bool mirrored = toLocal.determinant < 0f;
            if (mirrored) toLocal = mirror * toLocal;

            Material[] materials = renderer.sharedMaterials;
            int count = Mathf.Min(mesh.subMeshCount, materials.Length);
            for (int s = 0; s < count; s++)
            {
                if (materials[s] == null) continue;
                var part = new Part { mesh = mesh, subMesh = s, material = materials[s], toLocal = toLocal };
                (mirrored ? mirroredParts : parts).Add(part);
            }

            if (template == null) template = renderer;
            merged.Add(renderer);
        }

        // Nothing worth doing: no body, or a body that is already a single renderer.
        if (merged.Count < 2 || template == null) return;

        if (parts.Count > 0) BuildRenderer("BodyCombined", parts, template, bodyLayer, false);
        if (mirroredParts.Count > 0) BuildRenderer("BodyCombinedMirrored", mirroredParts, template, bodyLayer, true);

        foreach (MeshRenderer renderer in merged)
        {
            Destroy(renderer);
        }
    }

    private void BuildRenderer(string objectName, List<Part> parts, MeshRenderer template, int layer, bool mirrored)
    {
        // Group by material, keeping first-seen order so the submesh order is stable.
        var materials = new List<Material>();
        var groups = new List<List<CombineInstance>>();
        foreach (Part part in parts)
        {
            int g = materials.IndexOf(part.material);
            if (g < 0)
            {
                g = materials.Count;
                materials.Add(part.material);
                groups.Add(new List<CombineInstance>());
            }
            groups[g].Add(new CombineInstance { mesh = part.mesh, subMeshIndex = part.subMesh, transform = part.toLocal });
        }

        // One merged mesh per material, then those as the submeshes of one mesh. UInt32 indices:
        // the whole body is ~139k triangles, far past the 16-bit limit.
        var perMaterial = new CombineInstance[groups.Count];
        for (int g = 0; g < groups.Count; g++)
        {
            var mesh = new Mesh { indexFormat = IndexFormat.UInt32 };
            mesh.CombineMeshes(groups[g].ToArray(), true, true);
            perMaterial[g] = new CombineInstance { mesh = mesh, transform = Matrix4x4.identity };
        }
        var combined = new Mesh { name = objectName, indexFormat = IndexFormat.UInt32 };
        combined.CombineMeshes(perMaterial, false, false);
        foreach (CombineInstance ci in perMaterial) Destroy(ci.mesh);
        createdMeshes.Add(combined);

        var go = new GameObject(objectName) { layer = layer };
        go.transform.SetParent(transform, false);
        if (mirrored) go.transform.localScale = new Vector3(-1f, -1f, -1f);

        go.AddComponent<MeshFilter>().sharedMesh = combined;
        MeshRenderer renderer = go.AddComponent<MeshRenderer>();
        renderer.sharedMaterials = materials.ToArray();
        renderer.shadowCastingMode = template.shadowCastingMode;
        renderer.receiveShadows = template.receiveShadows;
        renderer.lightProbeUsage = template.lightProbeUsage;
        renderer.reflectionProbeUsage = template.reflectionProbeUsage;
        renderer.motionVectorGenerationMode = template.motionVectorGenerationMode;
    }

    private List<Transform> Propellers()
    {
        var propellers = new List<Transform>(4);
        VelocityControl control = GetComponent<VelocityControl>();
        if (control == null) return propellers;
        foreach (GameObject prop in new[] { control.PropFL, control.PropFR, control.PropRR, control.PropRL })
        {
            if (prop != null) propellers.Add(prop.transform);
        }
        return propellers;
    }

    private static bool IsUnderAny(Transform t, List<Transform> roots)
    {
        foreach (Transform root in roots)
        {
            if (t.IsChildOf(root)) return true;
        }
        return false;
    }
}
