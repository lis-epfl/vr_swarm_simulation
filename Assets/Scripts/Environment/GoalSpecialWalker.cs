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
///
/// <para>Every draw here comes from the session's goal seed (<see cref="GoalSession"/>): the
/// deck's shuffle, which patrol gets the hat and which of its walkers wears it, and each
/// patrol's own stream for where its walkers start. Keyed by this goal's place in the deal
/// (<see cref="DealOrdinal"/>), which is the order goals are instantiated in and so the
/// recorder's goal index, so a replayed seed gives every goal the same hat and the same
/// target walker. The choices are kept on the component for the recorder to log.</para>
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
    private static System.Random dealRng;   // from the session's goal seed, on the first deal
    private static int dealCount;           // goals dealt so far this session

    [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.SubsystemRegistration)]
    private static void ResetDeal()
    {
        undealt.Clear();
        deckSize = -1;
        lastDealt = -1;
        dealRng = null;
        dealCount = 0;
    }

    /// <summary>This goal's place in the session's deal (0 for the first goal dealt), or -1 if it dealt nothing.</summary>
    public int DealOrdinal { get; private set; } = -1;

    /// <summary>Index, among this patch's patrols in hierarchy order, of the one whose walker wears the hat.</summary>
    public int TargetPatrolIndex { get; private set; } = -1;

    /// <summary>Name of the building that patrol walks round.</summary>
    public string TargetPatrolName { get; private set; } = "";

    /// <summary>Which of that patrol's walkers wears the hat (WalkerPatrol's spawn index).</summary>
    public int TargetWalkerIndex { get; private set; } = -1;

    /// <summary>The hat prefab dealt to this goal.</summary>
    public string DealtPrefab { get; private set; } = "";

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
        DealOrdinal = dealCount++;
        DealtPrefab = prefab.name;

        // Every patrol walks its walkers from its own stream of the session seed, so a replay starts them where
        // they started; the target's patrol and walker come from this goal's stream.
        int seed = GoalSession.EnsureSeed();
        for (int i = 0; i < patrols.Count; i++)
        {
            patrols[i].SetSeed(GoalSession.Derive(seed, $"goal {DealOrdinal} patrol {i}"));
        }
        System.Random rng = GoalSession.Rng($"goal {DealOrdinal} target");
        TargetPatrolIndex = rng.Next(patrols.Count);
        WalkerPatrol chosen = patrols[TargetPatrolIndex];
        TargetPatrolName = chosen.name;
        TargetWalkerIndex = chosen.numWalkers > 0 ? rng.Next(chosen.numWalkers) : -1;
        chosen.RequestSpecial(prefab, TargetWalkerIndex);
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
        if (dealRng == null) dealRng = GoalSession.Rng("deal");
        if (undealt.Count == 0 || deckSize != usable.Count) Refill(usable.Count, dealRng);

        int take = undealt[undealt.Count - 1];
        undealt.RemoveAt(undealt.Count - 1);
        lastDealt = take;
        return specialWalkerPrefabs[usable[take]];
    }

    private static void Refill(int count, System.Random rng)
    {
        undealt.Clear();
        deckSize = count;
        for (int i = 0; i < count; i++) undealt.Add(i);

        for (int i = count - 1; i > 0; i--)
        {
            int j = rng.Next(i + 1);
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
