using System.Collections.Generic;
using UnityEditor;
using UnityEngine;
using UnityEngine.SceneManagement;

/// <summary>
/// The scenic city's street furniture (called by <see cref="ScenicPatchCityWorldBuilder"/> before it bakes the city):
/// the same props in the same places, drawn in this scene's materials. The Modular City Pack draws nearly every prop
/// in one material, a 4096 atlas in Standard (Specular setup) at smoothness 0.95 with saturated paint, which under the
/// scenic sky reads as shiny plastic; and its trees' leaves as flat cut-outs.
///
/// <para>So the dressing (<see cref="CityDressing"/>) is given <see cref="CityDressing.materialSwaps"/>: the pack's
/// atlas drawn in <c>Swarm/Scenic Prop</c> (graded, matte, grimy at the foot) and its bark likewise, and
/// <see cref="CityDressing.cutoutLeaves"/> in <c>Swarm/Scenic Leaves</c> (light through the leaves, a different green
/// tree to tree). They live on the scene's CityDressing, so redressing the scene with the dressing tool (the block-shaped
/// window's <i>Redress the streets</i>, or a bake) keeps the look. The materials only reference the pack's textures:
/// nothing of the pack is copied into the repository. Pack props elsewhere in the scene (the lamps along the road in)
/// are given the same material directly.</para>
/// </summary>
public static class ScenicPropsBuilder
{
    private const string Folder = ScenicPatchCityWorldBuilder.Folder + "/Materials";
    private const string AtlasName = "MC_Building_B_diffuse", BarkName = "MC_Tree_Bark", LeavesName = "MC_Plants_Alpha";

    /// <summary>The scenic props' materials, made if missing and set on every build, and the dressing's swaps. A summary.</summary>
    public static string Props(ProceduralCityBuilder builder, Scene scene)
    {
        CityDressing d = builder.GetComponent<CityDressing>();
        if (d == null) return "no dressing to restyle";

        // The pack's materials, as the props themselves carry them.
        var found = new Dictionary<string, Material>();
        void Collect(GameObject g)
        {
            if (g == null) return;
            foreach (Renderer r in g.GetComponentsInChildren<Renderer>(true))
            {
                foreach (Material m in r.sharedMaterials)
                {
                    if (m != null && !found.ContainsKey(m.name)) found[m.name] = m;
                }
            }
        }
        foreach (GameObject g in new[] { d.streetLight, d.tree, d.borderTree, d.hydrant, d.bench, d.bin, d.busStop, d.trafficSignal,
                                         d.frontageBench, d.bollard, d.phoneBooth, d.hedge, d.parkLamp, d.flowerBed })
        {
            Collect(g);
        }
        foreach (GameObject[] list in new[] { d.cornerProps, d.planters, d.kerbProps }) foreach (GameObject g in list ?? new GameObject[0]) Collect(g);
        foreach (CityDressing.PropModel m in d.lawnTrees ?? new CityDressing.PropModel[0]) foreach (GameObject g in m?.parts ?? new GameObject[0]) Collect(g);
        found.TryGetValue(AtlasName, out Material atlas);
        found.TryGetValue(BarkName, out Material bark);
        found.TryGetValue(LeavesName, out Material packLeaves);

        EnsureFolder(Folder);
        var swaps = new List<CityDressing.MaterialSwap>();
        Material props = null;
        if (atlas != null)
        {
            props = Make("ScenicProps", "Swarm/Scenic Prop");
            props.SetTexture("_MainTex", atlas.mainTexture);
            props.SetTexture("_SpecGlossMap", atlas.HasProperty("_SpecGlossMap") ? atlas.GetTexture("_SpecGlossMap") : null);
            props.SetColor("_Color", Color.white);
            props.SetFloat("_Saturation", 0.62f);
            props.SetFloat("_MaxAlbedo", 0.7f);
            props.SetFloat("_Gloss", 0.45f);
            props.SetFloat("_SpecScale", 0.35f);
            props.SetFloat("_Grime", 0.3f);
            props.SetFloat("_FootDarken", 0.25f);
            props.SetFloat("_GroundHeight", 0.1f);
            EditorUtility.SetDirty(props);
            swaps.Add(new CityDressing.MaterialSwap { from = atlas, to = props });
        }
        if (bark != null)
        {
            Material scenicBark = Make("ScenicBark", "Swarm/Scenic Prop");
            scenicBark.SetTexture("_MainTex", bark.mainTexture);
            scenicBark.SetTexture("_SpecGlossMap", null);
            scenicBark.SetColor("_Color", bark.HasProperty("_Color") ? bark.color : Color.white);
            scenicBark.SetFloat("_Saturation", 0.8f);
            scenicBark.SetFloat("_MaxAlbedo", 0.55f);
            scenicBark.SetFloat("_Gloss", 0f);
            scenicBark.SetFloat("_SpecScale", 0f);
            scenicBark.SetFloat("_Grime", 0.2f);
            scenicBark.SetFloat("_FootDarken", 0.15f);
            scenicBark.SetFloat("_GroundHeight", 0.1f);
            EditorUtility.SetDirty(scenicBark);
            swaps.Add(new CityDressing.MaterialSwap { from = bark, to = scenicBark });
        }
        Material leavesFrom = d.cutoutLeaves != null ? d.cutoutLeaves : packLeaves;
        Material leaves = null;
        if (leavesFrom != null && leavesFrom.mainTexture != null)
        {
            leaves = Make("ScenicLeaves", "Swarm/Scenic Leaves");
            leaves.SetTexture("_MainTex", leavesFrom.mainTexture);
            // VoronoiLeaves' tint, which takes the pack's yellow-green to a summer green, a shade deeper
            leaves.SetColor("_Color", new Color(0.70f, 0.70f, 0.48f));
            leaves.SetFloat("_Cutoff", leavesFrom.HasProperty("_Cutoff") ? leavesFrom.GetFloat("_Cutoff") : 0.4f);
            leaves.SetFloat("_Wrap", 0.45f);
            leaves.SetFloat("_Translucency", 0.6f);
            leaves.SetFloat("_Variation", 0.5f);
            EditorUtility.SetDirty(leaves);
        }

        Undo.RecordObject(d, "Scenic props");
        d.materialSwaps = swaps.ToArray();
        if (leaves != null) d.cutoutLeaves = leaves;
        EditorUtility.SetDirty(d);

        // Pack props elsewhere in the scene: outside the city, which its dressing restyles.
        int other = 0;
        if (props != null)
        {
            foreach (GameObject root in scene.GetRootGameObjects())
            {
                if (root.GetComponentInChildren<ProceduralCityBuilder>(true) != null) continue;
                foreach (MeshRenderer r in root.GetComponentsInChildren<MeshRenderer>(true))
                {
                    Material[] mats = r.sharedMaterials;
                    bool changed = false;
                    for (int k = 0; k < mats.Length; k++)
                    {
                        if (mats[k] != atlas) continue;
                        mats[k] = props;
                        changed = true;
                    }
                    if (!changed) continue;
                    Undo.RecordObject(r, "Scenic props");
                    r.sharedMaterials = mats;
                    other++;
                }
            }
        }
        return $"props restyled ({swaps.Count} material swaps{(leaves != null ? ", scenic leaves" : "")}; {other} pack props outside the city)";
    }

    private static Material Make(string name, string shader)
    {
        string path = $"{Folder}/{name}.mat";
        var m = AssetDatabase.LoadAssetAtPath<Material>(path);
        if (m == null)
        {
            m = new Material(Shader.Find(shader)) { name = name };
            AssetDatabase.CreateAsset(m, path);
        }
        else if (m.shader == null || m.shader.name != shader)
        {
            m.shader = Shader.Find(shader);
        }
        return m;
    }

    private static void EnsureFolder(string folder)
    {
        if (AssetDatabase.IsValidFolder(folder)) return;
        string parent = System.IO.Path.GetDirectoryName(folder).Replace('\\', '/');
        EnsureFolder(parent);
        AssetDatabase.CreateFolder(parent, System.IO.Path.GetFileName(folder));
    }
}
