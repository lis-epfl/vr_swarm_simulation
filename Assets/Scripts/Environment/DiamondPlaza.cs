using UnityEngine;

/// <summary>
/// Marks a plaza in a diamond's park (<c>DiamondParkBuilder</c>): a footpath the size of a city block at the city's
/// block scale, with a screen of buildings across the north–south street it stands in, and no road ring or verge round
/// it — the park's lawn stands in for them. It is not a tile, so <see cref="GoalPatchReplacer"/> finds it by this
/// component and may replace it with a goal like any tile. Being a block's size is what lets a goal, which the city's
/// <see cref="StreetWidthTuner"/> brings to that same scale, stand exactly where it stood.
///
/// <para>The plaza's ground, where a goal replacing it puts its road level, is a little above the diamond's: the
/// park's lawn is a few centimetres above the terrain, and at road level a goal's interior streets would be under the
/// grass (see <see cref="GroundPoint"/>).</para>
/// </summary>
[DisallowMultipleComponent]
public class DiamondPlaza : MonoBehaviour
{
    [Tooltip("The block's centre, the point a goal's kerb is laid over.")]
    [SerializeField] private Transform centre;

    [Tooltip("How far the centre stands above the plaza's ground, in world units.")]
    [SerializeField] private float centreAboveGround;

    /// <summary>The block's centre.</summary>
    public Transform Centre => centre;

    /// <summary>The plaza's ground under its centre, where a goal replacing it puts its own road level.</summary>
    public Vector3 GroundPoint => centre.position - Vector3.up * centreAboveGround;

    /// <summary>Called by the park builder when it lays the plaza out.</summary>
    public void Initialise(Transform blockCentre, float heightAboveGround)
    {
        centre = blockCentre;
        centreAboveGround = heightAboveGround;
    }
}
