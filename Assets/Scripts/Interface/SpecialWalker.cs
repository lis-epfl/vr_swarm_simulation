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

    // The body is skinned, so the rigid meshes a walker carries are its hat.
    private MeshRenderer[] hatRenderers;

    // Where the hat is right now, read off its own mesh, so nothing has to know how large the walkers are, which hat
    // this is or where the walk cycle has its head.
    public Vector3 HatCentre
    {
        get
        {
            if (hatRenderers == null)
            {
                hatRenderers = GetComponentsInChildren<MeshRenderer>(true);
                if (hatRenderers.Length == 0) Debug.LogError($"SpecialWalker: {name} has no hat mesh.", this);
            }
            if (hatRenderers.Length == 0) return transform.position;

            Bounds bounds = hatRenderers[0].bounds;
            for (int i = 1; i < hatRenderers.Length; i++) bounds.Encapsulate(hatRenderers[i].bounds);
            return bounds.center;
        }
    }

    // Called by WalkerPatrol right after spawning, with the prefab it used.
    public void EnsureHatId(string fallback)
    {
        if (string.IsNullOrEmpty(hatId)) hatId = fallback;
    }
}
