using UnityEngine;

/// <summary>
/// Marks a subtree as runtime-spawned content that the city tools must leave alone.
///
/// <para>The tuners identify a tile's block content as "every <see cref="Renderer"/> under the tile",
/// which is right for authored city props — one object, one renderer — and wrong for anything spawned
/// during play. <see cref="WalkerPatrol"/> hangs its walkers under the tile so they stay inside the goal
/// patch (<c>ExperimentRecorder</c> pairs a special walker to its goal through that parentage), and a
/// walker is a composite: its renderers are *children* of the object whose position the patrol drives.
/// Scaling those renderers about the kerb therefore does not move the walker, it pulls the walker apart —
/// the body mesh silently, since a skinned mesh renders from its bones and ignores its own transform, and
/// the hat visibly, by metres.</para>
///
/// <para>Whatever a city tool would do to such a subtree is wrong anyway: these objects are positioned
/// every frame by their own script, so a one-off reposition is either undone or, as here, applied to a
/// part that nothing will correct.</para>
/// </summary>
public class SpawnedContentRoot : MonoBehaviour
{
    /// <summary>True if <paramref name="t"/> is inside a subtree marked as runtime-spawned.</summary>
    public static bool Covers(Transform t)
    {
        return t != null && t.GetComponentInParent<SpawnedContentRoot>(true) != null;
    }
}
