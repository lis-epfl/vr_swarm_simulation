using System.Collections.Generic;
using UnityEngine;

/// <summary>
/// The record of a <see cref="BuildingUvBaker"/> run: which corrected meshes it created, and which
/// original mesh each building used to point at. It is the main asset of
/// <c>Assets/Prefabs/ScaledCity/ScaledCityBuildingMeshes.asset</c>, with the baked meshes stored inside
/// it as sub-objects — one file rather than several dozen loose ones.
///
/// <para>Its real job is to make the bake reversible without reaching for git:
/// <c>Tools/City/Restore original building meshes</c> walks <see cref="assignments"/> and puts every
/// <c>MeshFilter</c> back. Its presence is also what stops a second bake stacking a correction on top of
/// an already-corrected mesh.</para>
///
/// <para>This lives in the runtime assembly rather than beside the baker in <c>Editor/</c> purely so the
/// class still exists in a player build. The asset itself is never loaded at runtime — but the prefabs
/// reference the meshes stored inside it, and an asset whose main object has a missing script is a
/// needless thing to hand a build.</para>
/// </summary>
public class BuildingUvBakeManifest : ScriptableObject
{
    /// <summary>One corrected mesh, shared by every building that had the same mesh and distortion.</summary>
    [System.Serializable]
    public struct Variant
    {
        public Mesh bakedMesh;
        public Mesh sourceMesh;
        [Tooltip("Vertical/horizontal texel stretch this variant was built to cancel.")]
        public float anisotropy;
        public int buildingCount;
        public int correctedCharts;
        [Tooltip("Charts left alone: roofs and floors, curved charts, and any already square.")]
        public int skippedCharts;
        [Tooltip("Charts refused because the contraction would have left their own UV box. Included in " +
                 "skippedCharts. Should normally be 0 — a non-zero count means those charts kept their " +
                 "original UVs rather than risk sampling a neighbour's slice of the atlas.")]
        public int escapedCharts;
    }

    /// <summary>One building whose <c>MeshFilter</c> was re-pointed, and what it pointed at before.</summary>
    [System.Serializable]
    public struct Assignment
    {
        public string prefabPath;
        [Tooltip("Slash-separated path from the prefab root, for Transform.Find on restore.")]
        public string objectPath;
        public Mesh originalMesh;
    }

    public List<Variant> variants = new List<Variant>();
    public List<Assignment> assignments = new List<Assignment>();
}
