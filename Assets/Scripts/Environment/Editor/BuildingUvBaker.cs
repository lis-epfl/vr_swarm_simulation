using System.Collections.Generic;
using System.Linq;
using UnityEngine;
using UnityEditor;

/// <summary>
/// Removes the texture stretching on ScaledCity's buildings by baking corrected UVs into mesh copies.
///
/// <para><b>The problem.</b> <c>Assets/Prefabs/ScaledCity/</c> was forked from the Modular City Pack by
/// rewriting one thing per building: its <c>localScale</c>. The horizontal axes went to exactly
/// <c>original x 0.25</c> on all 220 buildings, and the vertical axis was set per-building so every
/// building is exactly 50 units tall. Transform scaling does not touch mesh UVs, so each facade is drawn
/// with an aspect distortion of (vertical scale / horizontal scale) — median <b>7.8x</b>, up to 18.1x.
/// Windows come out as tall smears.</para>
///
/// <para><b>Why the obvious fixes do not work.</b> The buildings share two atlases
/// (<c>MC_Buildings_n_Props_diffuse</c>, which they share with the props, and
/// <c>MC_Sky_Scrappers_diffuse</c>), and each mesh owns its own sub-rectangle of the sheet. Raising the
/// material's <c>_MainTex</c> tiling, or overriding <c>_MainTex_ST</c> per renderer, is an *absolute*
/// transform: it slides a building onto the neighbouring building's — or a hydrant's — slice of the
/// atlas. Scaling a mesh's UVs outward does the same thing. Only a *rect-relative* change is safe, and
/// the safest rect-relative change of all is one that shrinks.</para>
///
/// <para><b>What this does.</b> Per UV chart, it measures the mapping from object space to UV space,
/// works out how far the rendered texel aspect is from square, and contracts that chart's UVs along
/// whichever direction is over-dense, about the chart's own centroid. Contraction can never leave the
/// region the chart already occupied, so it cannot bleed into a neighbour and it cannot break the handful
/// of pack meshes that deliberately run outside [0,1] and rely on Repeat wrap for window strips
/// (<c>BnP_Large_Building_F</c> reaches u = 4.55, <c>Skyscraper_C</c> reaches u = -1.64) — for those it
/// simply reduces the number of repeats, which is the right answer on a narrower wall.</para>
///
/// <para><b>Why per chart and not a global "scale v".</b> The unwrap orientation is not consistent.
/// Measuring the UV direction that object-up maps to, <c>BnP_Apartment_A_006</c> gives |cos| to +v of
/// 1.000 (v really is vertical) but <c>Skyscraper_D_000</c> gives 0.161 — most of its walls are unwrapped
/// rotated 90 degrees. A global v-scale would fix the apartments and wreck the skyscrapers.</para>
///
/// <para><b>The trade-off, which is a real one.</b> Because only contraction is safe, the result is
/// isotropic but *magnified*: every building lands at 1/f of its authored texel density, where f is its
/// height factor — median 2.0x, worst about 4.5x. Windows read correctly shaped but larger, and each
/// facade shows a centred crop of its texture rather than all of it. For a building narrowed 4x, showing
/// roughly a quarter of the bays is the physically correct outcome. Restoring the authored density as
/// well would need the facade to repeat vertically, and this atlas cannot: an island runs ground floor to
/// cornice, so tiling it would band each building with repeating roofs and pavements. Repetition is
/// therefore deliberately not attempted.</para>
///
/// <para><b>The bake is a snapshot.</b> It is correct for the scales the prefabs hold when it runs.
/// Changing <see cref="BuildingWidthTuner"/>'s width afterwards re-introduces anisotropy by exactly that
/// factor; the fix is to re-run the bake. That is the accepted cost of having no runtime component, and
/// <see cref="Report"/> makes the drift visible at any time.</para>
///
/// <para>The building families are the ones <see cref="BuildingWidthTuner"/> tunes and
/// <see cref="ObstacleLayerAuditor"/> filters on; this is the third copy of that list, so keep all three
/// in step if the city pack gains another family.</para>
/// </summary>
public static class BuildingUvBaker
{
    private const string k_AssetPath = "Assets/Prefabs/ScaledCity/ScaledCityBuildingMeshes.asset";
    private const string k_PatchFolder = "Assets/Prefabs/ScaledCity/Patches";
    private const string k_GoalPatch = "Assets/Prefabs/ScaledCity/goal_patch_Scaled.prefab";

    /// <summary>Above this |dot| with object-up a face is a roof or a floor, not a wall.</summary>
    private const float k_RoofDot = 0.7f;

    /// <summary>A chart is corrected only if this much of its area agrees on being one flat wall.</summary>
    private const float k_ChartAgreement = 0.8f;

    /// <summary>Distortions closer to square than this are left alone; the bake would be a no-op.</summary>
    private const float k_Tolerance = 0.02f;

    // Same four families BuildingWidthTuner scales and ObstacleLayerAuditor promotes.
    private static readonly string[] k_BuildingPrefixes =
    {
        "BnP_Small_Building_",
        "BnP_Large_Building_",
        "BnP_Apartment_",
        "Skyscraper_",
    };

    // ---------------------------------------------------------------- menu items

    [MenuItem("Tools/City/Report building texture distortion")]
    public static void Report()
    {
        List<Instance> instances = CollectInstances(out int prefabCount);
        if (instances.Count == 0)
        {
            Debug.LogError($"BuildingUvBaker: found no buildings under {k_PatchFolder}. Nothing to report.");
            return;
        }

        var byMesh = instances.GroupBy(i => i.sourceMesh).ToList();
        var byVariant = instances.GroupBy(i => i.key).ToList();

        List<float> anis = instances.Select(i => i.anisotropy).OrderBy(a => a).ToList();
        Debug.Log($"BuildingUvBaker: {instances.Count} buildings in {prefabCount} prefabs — " +
                  $"{byMesh.Count} distinct meshes, {byVariant.Count} (mesh, distortion) variants.\n" +
                  $"  anisotropy (vertical/horizontal texel stretch): min {anis.First():F2}, " +
                  $"median {anis[anis.Count / 2]:F2}, max {anis.Last():F2}");

        foreach (var family in instances.GroupBy(i => Family(i.objectName)).OrderBy(g => g.Key))
        {
            List<float> a = family.Select(i => i.anisotropy).OrderBy(x => x).ToList();
            Debug.Log($"  {family.Count(),4} x {family.Key,-24} anisotropy {a.First():F2} .. " +
                      $"{a[a.Count / 2]:F2} .. {a.Last():F2}");
        }

        // The measured half. This reads the meshes rather than the transforms, so after a bake it is what
        // says whether the correction actually landed: every wall chart should come back at ratio ~1.
        int charts = 0, corrected = 0, skipped = 0;
        List<float> ratios = new List<float>();
        foreach (var variant in byVariant)
        {
            Instance any = variant.First();
            MeshAnalysis a = Analyse(any.sourceMesh, any.scale, any.heightAxis);
            charts += a.chartCount;
            corrected += a.wallCharts;
            skipped += a.skippedCharts;
            ratios.AddRange(a.ratios);
        }

        if (ratios.Count > 0)
        {
            ratios.Sort();
            int square = ratios.Count(r => Mathf.Abs(r - 1f) <= 0.05f);
            Debug.Log($"  wall charts: {corrected} correctable, {skipped} skipped (curved or mostly roof), " +
                      $"{charts} charts total\n" +
                      $"  rendered texel aspect per wall chart: min {ratios.First():F3}, " +
                      $"median {ratios[ratios.Count / 2]:F3}, max {ratios.Last():F3} — " +
                      $"{square} of {ratios.Count} within 5% of square\n" +
                      $"  (1.000 everywhere is what a successful bake looks like)");
        }
    }

    [MenuItem("Tools/City/Bake isotropic building UVs")]
    public static void Bake()
    {
        List<Instance> instances = CollectInstances(out int prefabCount);
        if (instances.Count == 0)
        {
            Debug.LogError($"BuildingUvBaker: found no buildings under {k_PatchFolder}. Nothing baked.");
            return;
        }

        if (AssetDatabase.LoadAssetAtPath<BuildingUvBakeManifest>(k_AssetPath) != null)
        {
            Debug.LogError($"BuildingUvBaker: {k_AssetPath} already exists, so this city is already baked. " +
                           "Run Tools/City/Restore original building meshes first if you want to re-bake " +
                           "(for instance after changing BuildingWidthTuner's width).");
            return;
        }

        // One baked mesh per (source mesh, distortion), not per building: 220 buildings resolve to a few
        // dozen variants because a mesh reused across patches almost always carries the same scale.
        var variants = instances.GroupBy(i => i.key).ToList();

        BuildingUvBakeManifest manifest = ScriptableObject.CreateInstance<BuildingUvBakeManifest>();
        AssetDatabase.CreateAsset(manifest, k_AssetPath);

        Dictionary<string, Mesh> baked = new Dictionary<string, Mesh>();
        int correctedCharts = 0, skippedCharts = 0, escapedCharts = 0;

        foreach (var variant in variants)
        {
            Instance any = variant.First();
            Mesh copy = BakeMesh(any.sourceMesh, any.scale, any.heightAxis,
                                 out int corrected, out int skipped, out int escaped);
            correctedCharts += corrected;
            skippedCharts += skipped;
            escapedCharts += escaped;

            copy.name = $"{any.sourceMesh.name}_iso{any.anisotropy:F2}";
            AssetDatabase.AddObjectToAsset(copy, manifest);
            baked[variant.Key] = copy;

            manifest.variants.Add(new BuildingUvBakeManifest.Variant
            {
                bakedMesh = copy,
                sourceMesh = any.sourceMesh,
                anisotropy = any.anisotropy,
                buildingCount = variant.Count(),
                correctedCharts = corrected,
                skippedCharts = skipped,
                escapedCharts = escaped,
            });
        }

        // Persist the meshes before anything points at them: a prefab can only serialise a
        // reference to an object that is already part of an asset on disk.
        AssetDatabase.SaveAssets();

        // Re-point the MeshFilters. Done per prefab asset rather than in the scene so it propagates to
        // every instance with no per-instance overrides, and so a goal patch instantiated at runtime by
        // GoalPatchReplacer comes out already corrected.
        int rewritten = 0;
        foreach (string prefabPath in PrefabPaths())
        {
            GameObject root = PrefabUtility.LoadPrefabContents(prefabPath);
            bool dirty = false;

            foreach (MeshFilter filter in root.GetComponentsInChildren<MeshFilter>(true))
            {
                if (!IsBuilding(filter.gameObject.name) || filter.sharedMesh == null)
                {
                    continue;
                }

                Transform t = filter.transform;
                string key = VariantKey(filter.sharedMesh, t.localScale, HeightAxis(t));
                if (!baked.TryGetValue(key, out Mesh replacement))
                {
                    continue; // collected and baked from the same pass, so this should not happen
                }

                manifest.assignments.Add(new BuildingUvBakeManifest.Assignment
                {
                    prefabPath = prefabPath,
                    objectPath = PathOf(t, root.transform),
                    originalMesh = filter.sharedMesh,
                });

                filter.sharedMesh = replacement;
                dirty = true;
                rewritten++;
            }

            if (dirty)
            {
                PrefabUtility.SaveAsPrefabAsset(root, prefabPath);
            }
            PrefabUtility.UnloadPrefabContents(root);
        }

        EditorUtility.SetDirty(manifest);
        AssetDatabase.SaveAssets();
        AssetDatabase.Refresh();

        int sourceCount = instances.Select(i => i.sourceMesh).Distinct().Count();
        Debug.Log($"BuildingUvBaker: baked {variants.Count} mesh variants from {sourceCount} " +
                  $"source meshes and re-pointed {rewritten} buildings across {prefabCount} prefabs.\n" +
                  $"  {correctedCharts} wall charts corrected, {skippedCharts} skipped (curved, or mostly " +
                  $"roof/floor, or already square).\n" +
                  (escapedCharts > 0
                      ? $"  {escapedCharts} of those were refused because the contraction would have left " +
                        "their own UV box; they keep their original UVs, so nothing samples a neighbouring " +
                        "building's slice of the atlas.\n"
                      : "") +
                  $"  Meshes live in {k_AssetPath}. Run the report again: a simple box building should " +
                  "come back at ~1.000 on every chart, while a mesh whose charts span several face " +
                  "orientations converges only partly, because one anisotropic scale per chart cannot " +
                  "square all of them at once. Offline on the pack this landed between 0.79 and 1.00, " +
                  "against 0.08-0.15 before.");
    }

    [MenuItem("Tools/City/Restore original building meshes")]
    public static void Restore()
    {
        BuildingUvBakeManifest manifest = AssetDatabase.LoadAssetAtPath<BuildingUvBakeManifest>(k_AssetPath);
        if (manifest == null)
        {
            Debug.LogError($"BuildingUvBaker: no {k_AssetPath}, so there is nothing to restore.");
            return;
        }

        int restored = 0;
        foreach (var group in manifest.assignments.GroupBy(a => a.prefabPath))
        {
            GameObject root = PrefabUtility.LoadPrefabContents(group.Key);
            bool dirty = false;

            foreach (var assignment in group)
            {
                Transform t = root.transform.Find(assignment.objectPath);
                if (t == null || !t.TryGetComponent(out MeshFilter filter))
                {
                    Debug.LogWarning($"BuildingUvBaker: {assignment.objectPath} is gone from " +
                                     $"{group.Key}; leaving it alone.");
                    continue;
                }
                filter.sharedMesh = assignment.originalMesh;
                dirty = true;
                restored++;
            }

            if (dirty)
            {
                PrefabUtility.SaveAsPrefabAsset(root, group.Key);
            }
            PrefabUtility.UnloadPrefabContents(root);
        }

        AssetDatabase.DeleteAsset(k_AssetPath);
        AssetDatabase.SaveAssets();
        AssetDatabase.Refresh();

        Debug.Log($"BuildingUvBaker: restored {restored} buildings to their original FBX meshes and " +
                  $"deleted {k_AssetPath}. A git diff on the ScaledCity prefabs should now be empty.");
    }

    // ---------------------------------------------------------------- collection

    /// <summary>One building in one prefab, and the distortion its transform applies to its mesh.</summary>
    private struct Instance
    {
        public string objectName;
        public Mesh sourceMesh;
        public Vector3 scale;
        public int heightAxis;
        public float anisotropy;
        public string key;
    }

    private static IEnumerable<string> PrefabPaths()
    {
        foreach (string guid in AssetDatabase.FindAssets("t:Prefab", new[] { k_PatchFolder }))
        {
            yield return AssetDatabase.GUIDToAssetPath(guid);
        }
        if (AssetDatabase.LoadAssetAtPath<GameObject>(k_GoalPatch) != null)
        {
            yield return k_GoalPatch;
        }
    }

    /// <summary>
    /// Every building in the ScaledCity prefab assets, read straight off the assets rather than the open
    /// scene: the scene carries no mesh or scale overrides, and reading the assets is what lets this run
    /// with no scene loaded.
    /// </summary>
    private static List<Instance> CollectInstances(out int prefabCount)
    {
        List<Instance> found = new List<Instance>();
        prefabCount = 0;

        foreach (string path in PrefabPaths())
        {
            GameObject asset = AssetDatabase.LoadAssetAtPath<GameObject>(path);
            if (asset == null)
            {
                continue;
            }
            prefabCount++;

            foreach (MeshFilter filter in asset.GetComponentsInChildren<MeshFilter>(true))
            {
                if (!IsBuilding(filter.gameObject.name) || filter.sharedMesh == null)
                {
                    continue;
                }

                Transform t = filter.transform;
                int axis = HeightAxis(t);
                Vector3 s = t.localScale;
                float vertical = Mathf.Abs(s[axis]);
                float horizontal = 0.5f * (Mathf.Abs(s[(axis + 1) % 3]) + Mathf.Abs(s[(axis + 2) % 3]));

                found.Add(new Instance
                {
                    objectName = filter.gameObject.name,
                    sourceMesh = filter.sharedMesh,
                    scale = s,
                    heightAxis = axis,
                    anisotropy = horizontal > 1e-6f ? vertical / horizontal : 1f,
                    key = VariantKey(filter.sharedMesh, s, axis),
                });
            }
        }
        return found;
    }

    /// <summary>
    /// Identifies one bake. The correction depends on the scale only up to a uniform factor — a uniform
    /// scale distorts nothing — so the key normalises it, which merges two buildings that differ only in
    /// overall size. The height axis is part of the key because MC_Patch_32 leaves its tile node at
    /// identity while the other 48 use the -90-about-X node, so its buildings are Y-up not Z-up.
    /// </summary>
    private static string VariantKey(Mesh mesh, Vector3 scale, int heightAxis)
    {
        float m = Mathf.Max(Mathf.Abs(scale.x), Mathf.Max(Mathf.Abs(scale.y), Mathf.Abs(scale.z)));
        if (m < 1e-9f)
        {
            m = 1f;
        }
        Vector3 n = scale / m;
        return $"{mesh.GetInstanceID()}|{heightAxis}|{n.x:F4},{n.y:F4},{n.z:F4}";
    }

    /// <summary>
    /// Which local axis of <paramref name="t"/> points at the sky. Measured rather than assumed, exactly
    /// as <see cref="BuildingWidthTuner"/> does: the pack's buildings are local +Z up because of the FBX
    /// Z-up import, but MC_Patch_32 is the exception that makes measuring necessary.
    /// </summary>
    private static int HeightAxis(Transform t)
    {
        float x = Mathf.Abs(Vector3.Dot(t.right, Vector3.up));
        float y = Mathf.Abs(Vector3.Dot(t.up, Vector3.up));
        float z = Mathf.Abs(Vector3.Dot(t.forward, Vector3.up));

        if (x >= y && x >= z) { return 0; }
        return y >= z ? 1 : 2;
    }

    private static bool IsBuilding(string objectName)
    {
        foreach (string prefix in k_BuildingPrefixes)
        {
            if (objectName.StartsWith(prefix))
            {
                return true;
            }
        }
        return false;
    }

    /// <summary>Trailing instance number stripped, so BnP_Apartment_A_006 groups with its peers.</summary>
    private static string Family(string objectName)
    {
        int i = objectName.Length;
        while (i > 0 && (char.IsDigit(objectName[i - 1]) || objectName[i - 1] == '_'))
        {
            i--;
        }
        return i > 0 ? objectName.Substring(0, i) : objectName;
    }

    /// <summary>Slash-separated path from <paramref name="root"/>, for Transform.Find on restore.</summary>
    private static string PathOf(Transform t, Transform root)
    {
        List<string> parts = new List<string>();
        for (Transform c = t; c != null && c != root; c = c.parent)
        {
            parts.Add(c.name);
        }
        parts.Reverse();
        return string.Join("/", parts);
    }

    // ---------------------------------------------------------------- geometry

    private struct MeshAnalysis
    {
        public int chartCount;
        public int wallCharts;
        public int skippedCharts;
        public List<float> ratios; // rendered texel aspect per correctable wall chart
    }

    /// <summary>Measure without modifying — the read-only half of <see cref="BakeMesh"/>.</summary>
    private static MeshAnalysis Analyse(Mesh mesh, Vector3 scale, int heightAxis)
    {
        MeshAnalysis result = new MeshAnalysis { ratios = new List<float>() };
        if (!TryReadMesh(mesh, out Vector3[] verts, out Vector2[] uvs, out int[] tris))
        {
            return result;
        }

        List<List<int>> charts = FindCharts(verts.Length, tris);
        result.chartCount = charts.Count;

        foreach (List<int> chart in charts)
        {
            if (TryMeasureChart(chart, verts, uvs, tris, scale, heightAxis, out float ratio, out _, out _))
            {
                result.wallCharts++;
                result.ratios.Add(ratio);
            }
            else
            {
                result.skippedCharts++;
            }
        }
        return result;
    }

    /// <summary>A copy of <paramref name="mesh"/> with each wall chart's UVs contracted to square.</summary>
    private static Mesh BakeMesh(Mesh mesh, Vector3 scale, int heightAxis,
                                 out int corrected, out int skipped, out int escaped)
    {
        corrected = 0;
        skipped = 0;
        escaped = 0;

        Mesh copy = Object.Instantiate(mesh);

        if (!TryReadMesh(mesh, out Vector3[] verts, out Vector2[] uvs, out int[] tris))
        {
            Debug.LogWarning($"BuildingUvBaker: {mesh.name} has no readable UVs; left unchanged.");
            return copy;
        }

        Vector2[] outUvs = (Vector2[])uvs.Clone();
        Vector2[] staging = new Vector2[uvs.Length];

        foreach (List<int> chart in FindCharts(verts.Length, tris))
        {
            if (!TryMeasureChart(chart, verts, uvs, tris, scale, heightAxis,
                                 out float ratio, out Vector2 denseDir, out HashSet<int> chartVerts))
            {
                skipped++;
                continue;
            }

            // ratio < 1 means the vertical is the *sparse* direction, so the horizontal is over-dense and
            // must be contracted; ratio > 1 is the mirror image. Either way the factor is <= 1, which is
            // what keeps every chart inside the atlas region it already occupied.
            float factor = ratio < 1f ? ratio : 1f / ratio;
            if (!IsFinite(factor) || factor <= 0f || factor > 1f - k_Tolerance)
            {
                skipped++; // already square, or a degenerate measurement; either way leave it alone
                continue;
            }

            // Seeded from the first chart *vertex*, not from chart[0] — a chart holds triangle offsets
            // into `tris`, so indexing `uvs` with one would read an unrelated vertex or run off the end.
            Vector2 centre = Vector2.zero;
            Vector2 min = Vector2.positiveInfinity;
            Vector2 max = Vector2.negativeInfinity;
            foreach (int v in chartVerts)
            {
                centre += uvs[v];
                min = Vector2.Min(min, uvs[v]);
                max = Vector2.Max(max, uvs[v]);
            }
            centre /= chartVerts.Count;

            // Written to a staging buffer and checked before it is committed. Contracting along one
            // direction about the centroid cannot geometrically leave the chart's own bounding box — every
            // point moves toward the centroid line and so stays inside the convex hull — but "cannot
            // happen" is the wrong footing for the one invariant this whole approach rests on. A UV that
            // escaped its box would sample the neighbouring building's, or a hydrant's, slice of the
            // atlas, which is exactly the visible garbage the tiling and MaterialPropertyBlock routes were
            // rejected for. So the box is enforced, and a chart that would breach it keeps its original
            // UVs and is reported as skipped rather than half-corrected.
            const float k_Slack = 1e-5f;
            bool safe = true;
            foreach (int v in chartVerts)
            {
                Vector2 d = uvs[v] - centre;
                Vector2 moved = centre + d + (factor - 1f) * Vector2.Dot(d, denseDir) * denseDir;
                if (!IsFinite(moved.x) || !IsFinite(moved.y) ||
                    moved.x < min.x - k_Slack || moved.x > max.x + k_Slack ||
                    moved.y < min.y - k_Slack || moved.y > max.y + k_Slack)
                {
                    safe = false;
                    break;
                }
                staging[v] = moved;
            }

            if (!safe)
            {
                escaped++;
                skipped++;
                continue;
            }

            foreach (int v in chartVerts)
            {
                outUvs[v] = staging[v];
            }
            corrected++;
        }

        copy.uv = outUvs;
        copy.RecalculateBounds();
        return copy;
    }

    /// <summary>Finite and not NaN. A degenerate chart must cost its own correction, not the mesh.</summary>
    private static bool IsFinite(float f)
    {
        return !float.IsNaN(f) && !float.IsInfinity(f);
    }

    private static bool TryReadMesh(Mesh mesh, out Vector3[] verts, out Vector2[] uvs, out int[] tris)
    {
        verts = null;
        uvs = null;
        tris = null;
        if (mesh == null || !mesh.isReadable)
        {
            return false;
        }
        verts = mesh.vertices;
        uvs = mesh.uv;
        tris = mesh.triangles;
        return uvs != null && uvs.Length == verts.Length && tris.Length >= 3;
    }

    /// <summary>
    /// Group triangles into UV charts by shared vertex index. The FBX imports with <c>weldVertices</c>,
    /// which merges two vertices only when position, normal *and* UV all match — so a UV seam or a hard
    /// crease is already a vertex split, and connectivity through the index buffer is exactly UV
    /// connectivity. That matters for more than tidiness: charts built this way are vertex-disjoint by
    /// construction, so moving one chart's UVs cannot drag a neighbouring chart's with it, and no vertex
    /// splitting is needed.
    /// </summary>
    private static List<List<int>> FindCharts(int vertexCount, int[] tris)
    {
        int[] parent = new int[vertexCount];
        for (int i = 0; i < vertexCount; i++)
        {
            parent[i] = i;
        }

        int Find(int a)
        {
            while (parent[a] != a)
            {
                parent[a] = parent[parent[a]];
                a = parent[a];
            }
            return a;
        }

        void Union(int a, int b)
        {
            int ra = Find(a), rb = Find(b);
            if (ra != rb)
            {
                parent[rb] = ra;
            }
        }

        for (int i = 0; i + 2 < tris.Length; i += 3)
        {
            Union(tris[i], tris[i + 1]);
            Union(tris[i + 1], tris[i + 2]);
        }

        Dictionary<int, List<int>> byRoot = new Dictionary<int, List<int>>();
        for (int i = 0; i + 2 < tris.Length; i += 3)
        {
            int root = Find(tris[i]);
            if (!byRoot.TryGetValue(root, out List<int> list))
            {
                list = new List<int>();
                byRoot[root] = list;
            }
            list.Add(i); // store the triangle's first index
        }
        return byRoot.Values.ToList();
    }

    /// <summary>
    /// Work out how far one chart's rendered texels are from square, and which UV direction is the dense
    /// one. Returns false for a chart this cannot safely correct: a roof or floor, a curved chart whose
    /// faces disagree about which way is up, or a degenerate unwrap.
    ///
    /// <para>The measurement is area-weighted over the chart's triangles. For each one it solves the
    /// object-space-to-UV Jacobian in the face's own (horizontal, vertical) basis, then divides by the
    /// world length each of those basis vectors has *after* the transform's scale — which is what turns a
    /// UV-per-object-unit figure into the UV-per-metre the eye actually sees. Scaling the basis vectors
    /// rather than assuming a single horizontal factor is what keeps the six BnP_Apartment_I buildings
    /// correct, since the pack authored those with x != y.</para>
    /// </summary>
    private static bool TryMeasureChart(List<int> chart, Vector3[] verts, Vector2[] uvs, int[] tris,
                                        Vector3 scale, int heightAxis,
                                        out float ratio, out Vector2 denseDir, out HashSet<int> chartVerts)
    {
        ratio = 1f;
        denseDir = Vector2.right;
        chartVerts = new HashSet<int>();

        Vector3 up = Vector3.zero;
        up[heightAxis] = 1f;

        double wallArea = 0, totalArea = 0;
        double sumH = 0, sumV = 0;       // area-weighted UV-per-metre along horizontal / vertical
        Vector2 sumHDir = Vector2.zero;  // area-weighted UV direction of the horizontal
        Vector2 sumVDir = Vector2.zero;
        Vector2 refH = Vector2.zero, refV = Vector2.zero;

        foreach (int t in chart)
        {
            int i0 = tris[t], i1 = tris[t + 1], i2 = tris[t + 2];
            chartVerts.Add(i0);
            chartVerts.Add(i1);
            chartVerts.Add(i2);

            Vector3 e1 = verts[i1] - verts[i0];
            Vector3 e2 = verts[i2] - verts[i0];
            Vector3 cross = Vector3.Cross(e1, e2);
            double area = cross.magnitude * 0.5;
            if (area < 1e-12)
            {
                continue;
            }
            totalArea += area;

            Vector3 n = cross.normalized;
            if (Mathf.Abs(Vector3.Dot(n, up)) > k_RoofDot)
            {
                continue; // roof or floor: with x == y it is undistorted, and there is no "vertical"
            }

            Vector3 h = Vector3.Cross(n, up).normalized;
            Vector3 v = (up - n * Vector3.Dot(n, up)).normalized;
            if (h.sqrMagnitude < 0.5f || v.sqrMagnitude < 0.5f)
            {
                continue;
            }

            Vector2 d1 = uvs[i1] - uvs[i0];
            Vector2 d2 = uvs[i2] - uvs[i0];
            Vector2 a1 = new Vector2(Vector3.Dot(e1, h), Vector3.Dot(e1, v));
            Vector2 a2 = new Vector2(Vector3.Dot(e2, h), Vector3.Dot(e2, v));
            float det = a1.x * a2.y - a1.y * a2.x;
            if (Mathf.Abs(det) < 1e-12f)
            {
                continue; // this triangle carries no usable UV gradient
            }

            Vector2 duvdh = (d1 * a2.y - d2 * a1.y) / det;
            Vector2 duvdv = (d2 * a1.x - d1 * a2.x) / det;

            // World length of the object-space unit vectors h and v once the transform's scale is applied.
            float sh = new Vector3(h.x * scale.x, h.y * scale.y, h.z * scale.z).magnitude;
            float sv = new Vector3(v.x * scale.x, v.y * scale.y, v.z * scale.z).magnitude;
            if (sh < 1e-9f || sv < 1e-9f || duvdh.magnitude < 1e-12f || duvdv.magnitude < 1e-12f)
            {
                continue;
            }

            wallArea += area;
            sumH += area * (duvdh.magnitude / sh);
            sumV += area * (duvdv.magnitude / sv);

            // Average the UV directions as axes, not vectors: two triangles of the same quad can disagree
            // in sign and would otherwise cancel to zero.
            Vector2 hDir = duvdh.normalized;
            Vector2 vDir = duvdv.normalized;
            if (refH == Vector2.zero) { refH = hDir; refV = vDir; }
            sumHDir += (Vector2.Dot(hDir, refH) < 0f ? -hDir : hDir) * (float)area;
            sumVDir += (Vector2.Dot(vDir, refV) < 0f ? -vDir : vDir) * (float)area;
        }

        if (wallArea <= 0 || totalArea <= 0 || wallArea < k_ChartAgreement * totalArea)
        {
            return false; // mostly roof, or a curved chart the single-axis correction would misrepresent
        }
        if (sumHDir.sqrMagnitude < 1e-12f || sumVDir.sqrMagnitude < 1e-12f)
        {
            return false;
        }

        double densityH = sumH / wallArea;
        double densityV = sumV / wallArea;
        if (densityH < 1e-12 || densityV < 1e-12)
        {
            return false;
        }

        ratio = (float)(densityV / densityH);
        denseDir = (ratio < 1f ? sumHDir : sumVDir).normalized;

        // A non-finite ratio or direction would propagate NaN into the UVs, and a NaN UV samples nothing
        // in particular — the worst kind of atlas escape, because it is invisible in the numbers.
        if (!IsFinite(ratio) || ratio <= 0f || !IsFinite(denseDir.x) || !IsFinite(denseDir.y))
        {
            return false;
        }
        return true;
    }
}
