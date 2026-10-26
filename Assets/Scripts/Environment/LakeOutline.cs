using UnityEngine;
using UnityEngine.Splines;

/// <summary>
/// A lake whose shoreline is the closed spline on the same object (a <see cref="SplineContainer"/>): drag the knots in
/// the Scene view, or edit a field here, and the water, its bank and its beaches are rebuilt as you go
/// (<c>LandmarkBuilder.RebuildLake</c>, hooked to <see cref="Spline.Changed"/> and to inspector edits in the editor).
/// Nothing happens at Play; the mesh is an asset in <c>Assets/Landmarks/Generated</c>, so the lake costs a mesh and two
/// materials, no script.
///
/// <para>The mesh is built in this object's own frame, which stands on the ground: the water flat at
/// <see cref="waterLevel"/> above it as submesh 0, and the bank as submesh 1, a narrow damp lip sloping from the water's
/// edge down to <see cref="shoreDepth"/> below the ground, so it meets the grass in a clean line. They take the
/// renderer's first and second materials. <b>The water never lies on the ground</b>: the city tiles under it are flat and
/// already at their lowest height, so no hollow can be dug for it, and water 0.1 m over flat grass lost to it in the
/// depth buffer and showed as grass. The water's UVs are in metres (<see cref="uvMetres"/> per repeat), so the water
/// material tiles the same however the shoreline is drawn.</para>
///
/// <para><b>The beaches are not in the mesh: they are painted into the ground</b> (a Sand terrain layer on the scene's
/// own copies of the city tiles), so they blend into the grass with a soft, ragged edge, and their width comes and goes
/// along the shore — wide in a few stretches (<see cref="beachShare"/> of it), none elsewhere, where the grass runs down
/// to the bank. A sand band in the mesh was a solid colour of one width tracing the shoreline exactly, which no shore
/// looks like. The bank's width wanders too, by about ±35%.</para>
///
/// <para>Part of <see cref="LandmarkSet"/>'s prefab. Edit it in the prefab, or apply the overrides to it, because a
/// scene rebuild puts the prefab's own lake back. The beaches are painted only into a scene's tiles, so editing the lake
/// in the prefab stage repaints them when the scene's lake next rebuilds (<i>Add to open scene</i> does).</para>
/// </summary>
[RequireComponent(typeof(SplineContainer), typeof(MeshFilter), typeof(MeshRenderer))]
public class LakeOutline : MonoBehaviour
{
    [Tooltip("Height of the water above the ground this object stands on (m).")]
    public float waterLevel = 1f;
    [Tooltip("How far below the ground the bank's outer edge goes (m), so the bank meets the grass in a clean line.")]
    public float shoreDepth = 0.5f;
    [Tooltip("Typical width of the damp bank round the water (m); it wanders by about ±35% along the shore.")]
    public float bankWidth = 3f;
    [Tooltip("Width of the widest beach (m), painted into the ground beyond the bank with a soft, ragged edge.")]
    public float beachWidth = 28f;
    [Tooltip("Roughly the share of the shore that has a beach; the rest is grass down to the bank.")]
    [Range(0f, 1f)] public float beachShare = 0.4f;
    [Tooltip("Picks where along the shore the beaches fall.")]
    public int beachSeed = 1;
    [Tooltip("Points the shoreline is sampled at.")]
    [Range(16, 512)] public int samples = 160;
    [Tooltip("Metres per repeat of the water material's texture.")]
    public float uvMetres = 40f;

    /// <summary>The version of <c>LandmarkBuilder</c>'s lake generator this lake was made by (0: before versions), so
    /// <i>Add to open scene</i> can offer to replace a prefab lake the generator has since changed.</summary>
    [HideInInspector] public int generatorVersion;
}
