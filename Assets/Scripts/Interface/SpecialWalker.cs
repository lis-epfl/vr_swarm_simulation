using UnityEngine;

// Runtime marker attached to the single "special" pedestrian in each goal patch
// (see GoalSpecialWalker). Purely a detection hook: other systems locate the
// findable target with FindObjectsByType<SpecialWalker>() or
// GetComponentInParent<SpecialWalker>(). It holds no behaviour of its own — the
// special walker patrols exactly like a normal one; only its prefab (ModifedWalker)
// looks different.
public class SpecialWalker : MonoBehaviour
{
}
