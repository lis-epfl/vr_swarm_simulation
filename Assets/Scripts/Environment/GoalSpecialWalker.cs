using System.Collections.Generic;
using UnityEngine;
#if UNITY_EDITOR
using UnityEditor;
#endif

/// <summary>
/// Ensures each goal patch contains exactly one visually-distinct "special"
/// pedestrian (one of the hat-wearing walker prefabs) among its many ordinary walkers,
/// and that no two goals in a session hand the pilot the same one.
///
/// A goal patch holds several <see cref="WalkerPatrol"/> components (each spawning
/// its own batch of walkers), so "one per goal" cannot be decided inside a single
/// patrol. This component sits on the goal patch root, picks one of its child
/// patrols at random in <see cref="Awake"/> (before any patrol's <c>Start</c> runs)
/// and asks it to swap one of its walkers for the special prefab. Because
/// <see cref="GoalPatchReplacer"/> instantiates each goal patch independently, every
/// goal ends up with exactly one special pedestrian.
///
/// <para>Which prefab — i.e. which hat — a goal gets is dealt from a shuffled deck
/// shared by every goal patch in the session, because each patch decides alone and
/// three independent random picks would collide. The deck is a permutation of
/// <see cref="specialWalkerPrefabs"/>: with three hats and three goals every goal is
/// different, and a fourth goal starts a fresh shuffle that never repeats the hat
/// just dealt. Which hat landed where is recorded per goal by
/// <c>ExperimentRecorder</c>, off the <see cref="SpecialWalker"/> marker.</para>
/// </summary>
public class GoalSpecialWalker : MonoBehaviour
{
    [Tooltip("The visually-distinct pedestrian prefabs, one per hat. Each goal patch takes a " +
             "different one until the set runs out, then the set is reshuffled. An empty list is " +
             "filled in the editor from DefaultPrefabNames; assign explicitly to override.")]
    [SerializeField] private GameObject[] specialWalkerPrefabs;

    // The deal, shared across every goal patch: indices into one patch's prefab list, shuffled.
    // Static state in the editor outlives a play session, so it is cleared on entering play.
    private static readonly List<int> undealt = new List<int>();
    private static int deckSize = -1;
    private static int lastDealt = -1;

    [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.SubsystemRegistration)]
    private static void ResetDeal()
    {
        undealt.Clear();
        deckSize = -1;
        lastDealt = -1;
    }

    // Awake runs before every component's Start, so the chosen patrol sees the
    // request before it spawns its walkers in WalkerPatrol.Start.
    private void Awake()
    {
        List<WalkerPatrol> patrols = new List<WalkerPatrol>(GetComponentsInChildren<WalkerPatrol>(true));
        if (patrols.Count == 0)
        {
            Debug.LogWarning("GoalSpecialWalker: no WalkerPatrol found under this goal patch.", this);
            return;
        }

        // Deal only once a patrol is known to exist, so a patch with nothing to spawn
        // does not silently consume a hat that the next goal would then not be given.
        GameObject prefab = Deal();
        if (prefab == null)
        {
            Debug.LogError("GoalSpecialWalker: specialWalkerPrefabs is empty; no special pedestrian placed.", this);
            return;
        }

        WalkerPatrol chosen = patrols[Random.Range(0, patrols.Count)];
        chosen.RequestSpecial(prefab);
    }

    // Next prefab off the shuffled deck, refilling it when it runs out.
    private GameObject Deal()
    {
        List<int> usable = new List<int>();
        if (specialWalkerPrefabs != null)
        {
            for (int i = 0; i < specialWalkerPrefabs.Length; i++)
            {
                if (specialWalkerPrefabs[i] != null) usable.Add(i);
            }
        }
        if (usable.Count == 0) return null;

        // A patch configured with a different set than the last one re-deals rather than
        // indexing this list with the other's indices.
        if (undealt.Count == 0 || deckSize != usable.Count) Refill(usable.Count);

        int take = undealt[undealt.Count - 1];
        undealt.RemoveAt(undealt.Count - 1);
        lastDealt = take;
        return specialWalkerPrefabs[usable[take]];
    }

    private static void Refill(int count)
    {
        undealt.Clear();
        deckSize = count;
        for (int i = 0; i < count; i++) undealt.Add(i);

        for (int i = count - 1; i > 0; i--)
        {
            int j = Random.Range(0, i + 1);
            (undealt[i], undealt[j]) = (undealt[j], undealt[i]);
        }

        // Cards are taken off the end, so the last entry is dealt first: keep a new round
        // from opening with the hat the previous round closed on.
        if (count > 1 && undealt[count - 1] == lastDealt)
        {
            (undealt[count - 1], undealt[0]) = (undealt[0], undealt[count - 1]);
        }
    }

#if UNITY_EDITOR
    /// <summary>
    /// The hats, named rather than pattern-matched. A name pattern is what there *isn't* a safe one
    /// of: these are named for their hats, so the only shared token is the "Walker" suffix, and that
    /// also matches <c>SimpleWalker</c> — the ordinary pedestrian the other 63 in a patch are built
    /// from. Dealing that as a "special" would put a hatless target in a goal, and the pilot would be
    /// hunting someone indistinguishable from the crowd. Add a hat by adding it here.
    /// </summary>
    private static readonly string[] DefaultPrefabNames = { "CapWalker", "CowboyWalker", "BucketWalker" };

    // Auto-assign the special walker prefabs so they don't have to be dragged onto the goal patch by
    // hand; the resolved references are serialized for builds, and this only fills an empty list, so
    // an explicit assignment is never overwritten. Mirrors WalkerPatrol's SimpleWalker assignment.
    private void Reset() { AssignDefaultSpecialPrefabs(); }
    private void OnValidate() { AssignDefaultSpecialPrefabs(); }

    private void AssignDefaultSpecialPrefabs()
    {
        if (specialWalkerPrefabs != null && specialWalkerPrefabs.Length > 0) return;

        List<GameObject> found = new List<GameObject>();
        foreach (string wanted in DefaultPrefabNames)
        {
            foreach (string guid in AssetDatabase.FindAssets(wanted + " t:Prefab"))
            {
                string path = AssetDatabase.GUIDToAssetPath(guid);
                if (System.IO.Path.GetFileNameWithoutExtension(path) != wanted) continue;

                GameObject prefab = AssetDatabase.LoadAssetAtPath<GameObject>(path);
                if (prefab != null) found.Add(prefab);
                break;
            }
        }

        specialWalkerPrefabs = found.ToArray();
    }
#endif
}
