using UnityEngine;

/// <summary>
/// The one seed a play session's goal randomness is drawn from: which tiles become goals
/// (<see cref="GoalPatchReplacer"/>), which hat each goal is dealt (<see cref="GoalSpecialWalker"/>), which of a
/// goal's walkers wears it, and where every walker starts on its loop (<see cref="WalkerPatrol"/>).
///
/// <para>Before this existed all of those drew from the global <c>UnityEngine.Random</c>, whose sequence depends on
/// everything else that consumes it in the frame, so a session could not be reproduced: Replay put the goals back
/// where they were but dealt the hats and picked the target walker afresh. Now each consumer derives its own
/// <c>System.Random</c> from <see cref="Seed"/> and a stable key (<see cref="Derive"/>), so the same seed reproduces
/// the same layout whatever else ran, and <c>ExperimentRecorder</c> writes the seed into the session JSON, where
/// Replay reads it back.</para>
///
/// <para>Whoever lays out the goals sets the seed before instantiating them (<see cref="Set"/>): the inspector's
/// fixed seed, a replayed session's, or a fresh draw. A goal patch that exists without one — the practice world's,
/// placed in edit mode — draws a fresh seed on first use (<see cref="EnsureSeed"/>), so its hat still changes from
/// session to session.</para>
/// </summary>
public static class GoalSession
{
    /// <summary>The session's goal seed. Valid once <see cref="HasSeed"/>.</summary>
    public static int Seed { get; private set; }

    public static bool HasSeed { get; private set; }

    /// <summary>Where the seed came from: "inspector", "replay", "layout", "fresh" or "unseeded".</summary>
    public static string Source { get; private set; } = "";

    [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.SubsystemRegistration)]
    private static void ResetOnPlay()
    {
        Seed = 0;
        HasSeed = false;
        Source = "";
    }

    public static void Set(int seed, string source)
    {
        Seed = seed;
        HasSeed = true;
        Source = source;
        Debug.Log($"GoalSession: goal seed {seed} ({source}).");
    }

    /// <summary>A seed drawn from the clock and a GUID, independent of <c>UnityEngine.Random</c>'s state.</summary>
    public static int FreshSeed()
    {
        unchecked
        {
            int h = System.Guid.NewGuid().GetHashCode();
            return (h ^ System.Environment.TickCount) & 0x7fffffff;
        }
    }

    /// <summary>The seed, drawing a fresh one (source "unseeded") if nobody set it.</summary>
    public static int EnsureSeed()
    {
        if (!HasSeed)
        {
            Set(FreshSeed(), "unseeded");
        }
        return Seed;
    }

    /// <summary>
    /// A seed for one consumer, from the session seed and a key naming it (e.g. "deal", "goal 2 patrol 5").
    /// FNV-1a over the key folded with the seed: deterministic across runs and platforms, unlike
    /// <c>string.GetHashCode</c>, which .NET randomises per process.
    /// </summary>
    public static int Derive(int seed, string key)
    {
        unchecked
        {
            uint h = 2166136261u ^ (uint)seed;
            h *= 16777619u;
            foreach (char c in key)
            {
                h ^= c;
                h *= 16777619u;
            }
            h ^= h >> 15;
            h *= 2246822519u;
            h ^= h >> 13;
            return (int)(h & 0x7fffffff);
        }
    }

    /// <summary>A generator for one consumer of the session seed (drawing one if none is set).</summary>
    public static System.Random Rng(string key) => new System.Random(Derive(EnsureSeed(), key));

    /// <summary>A uniform float in [min, max) from <paramref name="rng"/>, as UnityEngine.Random.Range would give.</summary>
    public static float Range(System.Random rng, float min, float max) => min + (float)rng.NextDouble() * (max - min);
}
