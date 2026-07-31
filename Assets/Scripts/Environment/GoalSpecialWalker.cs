using System.Collections.Generic;
using UnityEngine;
#if UNITY_EDITOR
using UnityEditor;
#endif

/// <summary>
/// Ensures each goal patch contains exactly one visually-distinct "special"
/// pedestrian (the <c>ModifedWalker</c> prefab) among its many ordinary walkers.
///
/// A goal patch holds several <see cref="WalkerPatrol"/> components (each spawning
/// its own batch of walkers), so "one per goal" cannot be decided inside a single
/// patrol. This component sits on the goal patch root, picks one of its child
/// patrols at random in <see cref="Awake"/> (before any patrol's <c>Start</c> runs)
/// and asks it to swap one of its walkers for the special prefab. Because
/// <see cref="GoalPatchReplacer"/> instantiates each goal patch independently, every
/// goal ends up with exactly one special pedestrian.
/// </summary>
public class GoalSpecialWalker : MonoBehaviour
{
    [Tooltip("The visually-distinct pedestrian prefab. Auto-discovered by name " +
             "(ModifedWalker) in the editor; assign manually to override.")]
    [SerializeField] private GameObject specialWalkerPrefab;

    // Awake runs before every component's Start, so the chosen patrol sees the
    // request before it spawns its walkers in WalkerPatrol.Start.
    private void Awake()
    {
        if (specialWalkerPrefab == null)
        {
            Debug.LogError("GoalSpecialWalker: specialWalkerPrefab is not assigned; no special pedestrian placed.", this);
            return;
        }

        List<WalkerPatrol> patrols = new List<WalkerPatrol>(GetComponentsInChildren<WalkerPatrol>(true));
        if (patrols.Count == 0)
        {
            Debug.LogWarning("GoalSpecialWalker: no WalkerPatrol found under this goal patch.", this);
            return;
        }

        WalkerPatrol chosen = patrols[Random.Range(0, patrols.Count)];
        chosen.RequestSpecial(specialWalkerPrefab);
    }

#if UNITY_EDITOR
    // Auto-assign the ModifedWalker prefab so it doesn't have to be dragged onto
    // the goal patch by hand; the resolved reference is serialized for builds.
    // Mirrors WalkerPatrol's SimpleWalker auto-assignment.
    private void Reset() { AssignDefaultSpecialPrefab(); }
    private void OnValidate() { AssignDefaultSpecialPrefab(); }

    private void AssignDefaultSpecialPrefab()
    {
        if (specialWalkerPrefab != null) return;
        string[] guids = AssetDatabase.FindAssets("ModifedWalker t:Prefab");
        foreach (string guid in guids)
        {
            string path = AssetDatabase.GUIDToAssetPath(guid);
            if (System.IO.Path.GetFileNameWithoutExtension(path) == "ModifedWalker")
            {
                specialWalkerPrefab = AssetDatabase.LoadAssetAtPath<GameObject>(path);
                break;
            }
        }
    }
#endif
}
