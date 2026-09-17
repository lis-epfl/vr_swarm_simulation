using UnityEngine;

// Runtime marker attached to the single "special" pedestrian in each goal patch
// (see GoalSpecialWalker). Mostly a detection hook: other systems locate the
// findable target with FindObjectsByType<SpecialWalker>() or
// GetComponentInParent<SpecialWalker>(). It holds no behaviour of its own — the
// special walker patrols exactly like a normal one; only its prefab (one of the
// hat-wearing walkers — CapWalker, CowboyWalker, BucketWalker) looks different.
//
// It also names which hat this one is wearing, so ExperimentRecorder can log the
// hat per goal. The id belongs on the prefab rather than in a list beside it: it
// then travels with the asset, and renaming or reordering the prefabs cannot
// silently relabel a session's data. A prefab that leaves it blank is labelled by
// WalkerPatrol with the prefab's own name.
public class SpecialWalker : MonoBehaviour
{
    [Tooltip("Name of this walker's hat as it appears in the experiment logs (e.g. cowboy). " +
             "Leave blank to fall back to the prefab's name.")]
    [SerializeField] private string hatId;

    public string HatId => hatId;

    // Called by WalkerPatrol right after spawning, with the prefab it used.
    public void EnsureHatId(string fallback)
    {
        if (string.IsNullOrEmpty(hatId)) hatId = fallback;
    }
}
