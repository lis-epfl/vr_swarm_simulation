using System;
using System.Collections.Generic;
using UnityEngine;

/// <summary>
/// The goals of a block-shaped city (<c>ConvexPrism</c> buildings, S4's variant B: BlockShapedCityWorld), drawn at Play
/// from the city's own buildings by <see cref="ProceduralCityBuilder"/>. A goal is one building; its ring is every
/// building across a street from it, and each ring building's footpath carries the goal's pedestrians
/// (<see cref="WalkerPatrol.PathShape.Polygon"/>), the hat among them -- citygen's occluder and patrolled ring, with
/// nothing added: the chosen buildings change role, not place, and the layout file is left as it is.
///
/// <para><b>Which buildings.</b> Two blocks are neighbours when a street names both (<c>streets[].blocks</c>), so a block
/// meeting the goal's only at a junction is in its ring too: 4 to 8 buildings, 6 on average, at p 40 s 18. The goal's
/// block and every block in its ring must be interior and built (no plaza), so every goal has a whole ring of patrolled
/// buildings and none touches the arena's edge. The goals are spread as citygen spreads VoronoiCityWorld's
/// (<c>citygen.choose_goal_centres</c>): at least <see cref="MinSpawnDistance"/> from the spawn, at least
/// <see cref="MinSeparation"/> apart, one from each third of the candidates' distances from the spawn, else any
/// such set; and no two share a ring building, nor stands one in another's ring.</para>
///
/// <para><b>Drawn every Play</b>, from the session's goal seed (<see cref="GoalSession"/>, key <see cref="SeedKey"/>), as
/// the tile cities' GoalPatchReplacer draws its tiles: the goals, the hats and the target walkers change from session
/// to session, and the recorded <c>goalSeed</c> puts them all back. A function of the layout and the seed alone, so the
/// probe can draw it again.</para>
/// </summary>
public static class BlockGoals
{
    /// <summary><see cref="ProceduralCityBuilder.LayoutGoal.kind"/> of a goal drawn here.</summary>
    public const string Kind = "block";
    /// <summary>The key of the session seed's stream the goals are drawn from.</summary>
    public const string SeedKey = "goal blocks";
    /// <summary>m: citysim/layout.py GOAL_MIN_SPAWN_DIST_M.</summary>
    public const float MinSpawnDistance = 100f;
    /// <summary>m: citysim/layout.py GOAL_MIN_SEPARATION_M.</summary>
    public const float MinSeparation = 150f;
    private const int StrataTries = 2000, AnyTries = 5000;        // citygen.choose_goal_centres'

    /// <summary>Whether a layout takes goals drawn here: none of its own, block-shaped buildings, and its blocks list.</summary>
    public static bool Applies(ProceduralCityBuilder.LayoutFile file) =>
        file?.layout != null && (file.layout.goals == null || file.layout.goals.Length == 0) &&
        file.blocks != null && file.blocks.Length > 0 && file.boxes != null && Array.Exists(file.boxes, PrismBuilding.IsPrism);

    /// <summary>
    /// <paramref name="count"/> goals for <paramref name="file"/>, nearest the spawn first, drawn from
    /// <paramref name="rng"/>, with their buildings' roles set in <paramref name="file"/> (the goal's goal_centre, its
    /// ring's goal_ring with a patrol). Null with the reason if the city cannot hold them.
    /// </summary>
    public static ProceduralCityBuilder.LayoutGoal[] Draw(ProceduralCityBuilder.LayoutFile file, System.Random rng, int count,
                                                          out string error)
    {
        error = null;
        if (count <= 0) return new ProceduralCityBuilder.LayoutGoal[0];
        float[] spawn = file.layout.spawn;
        if (spawn == null || spawn.Length < 3)
        {
            error = "the layout has no spawn to keep the goals from.";
            return null;
        }
        var building = new Dictionary<int, int>();            // block id -> its building's index in boxes
        var interior = new HashSet<int>();
        foreach (ProceduralCityBuilder.LayoutBlock b in file.blocks)
        {
            if (b.building >= 0 && b.building < file.boxes.Length && PrismBuilding.IsPrism(file.boxes[b.building]))
            {
                building[b.id] = b.building;
            }
            if (b.interior) interior.Add(b.id);
        }
        var ring = new Dictionary<int, SortedSet<int>>();     // block id -> the blocks across a street from it
        foreach (ProceduralCityBuilder.LayoutStreet s in file.streets ?? new ProceduralCityBuilder.LayoutStreet[0])
        {
            if (s.blocks == null || s.blocks.Length < 2) continue;
            int a = s.blocks[0], b = s.blocks[1];
            if (a < 0 || b < 0 || a == b) continue;
            Ring(ring, a).Add(b);
            Ring(ring, b).Add(a);
        }

        bool Usable(int k) => interior.Contains(k) && building.ContainsKey(k);
        Vector2 Centre(int k)
        {
            float[] c = file.boxes[building[k]].c;
            return new Vector2(c[0], c[2]);
        }
        var home = new Vector2(spawn[0], spawn[2]);
        var candidates = new List<int>();
        foreach (ProceduralCityBuilder.LayoutBlock b in file.blocks)
        {
            if (!Usable(b.id) || !ring.TryGetValue(b.id, out SortedSet<int> r) || r.Count == 0) continue;
            bool whole = true;
            foreach (int k in r) whole &= Usable(k);
            if (whole && (Centre(b.id) - home).magnitude >= MinSpawnDistance) candidates.Add(b.id);
        }
        if (candidates.Count < count)
        {
            error = $"only {candidates.Count} buildings stand in a whole ring of buildings, {MinSpawnDistance:0} m or more " +
                    $"from the spawn; {count} goals need {count}.";
            return null;
        }
        candidates.Sort((a, b) =>
        {
            int c = (Centre(a) - home).sqrMagnitude.CompareTo((Centre(b) - home).sqrMagnitude);
            return c != 0 ? c : a.CompareTo(b);
        });

        bool Apart(int[] pick)
        {
            for (int i = 0; i < pick.Length; i++)
            {
                for (int j = i + 1; j < pick.Length; j++)
                {
                    if ((Centre(pick[i]) - Centre(pick[j])).magnitude < MinSeparation) return false;
                    var a = new HashSet<int>(ring[pick[i]]) { pick[i] };
                    if (a.Contains(pick[j]) || a.Overlaps(ring[pick[j]])) return false;
                }
            }
            return true;
        }

        // One from each third of the spawn distances (numpy's array_split: the first strata take the remainder), then
        // any well-separated set, nearest the spawn first.
        int[] chosen = null;
        var strata = new List<List<int>>();
        for (int i = 0, at = 0; i < count; i++)
        {
            int size = candidates.Count / count + (i < candidates.Count % count ? 1 : 0);
            strata.Add(candidates.GetRange(at, size));
            at += size;
        }
        for (int t = 0; t < StrataTries && chosen == null; t++)
        {
            var pick = new int[count];
            for (int i = 0; i < count; i++) pick[i] = strata[i][rng.Next(strata[i].Count)];
            if (Apart(pick)) chosen = pick;
        }
        for (int t = 0; t < AnyTries && chosen == null; t++)
        {
            var pool = new List<int>(candidates);
            var pick = new int[count];
            for (int i = 0; i < count; i++)
            {
                int k = rng.Next(pool.Count);
                pick[i] = pool[k];
                pool.RemoveAt(k);
            }
            if (!Apart(pick)) continue;
            Array.Sort(pick, (a, b) => candidates.IndexOf(a).CompareTo(candidates.IndexOf(b)));
            chosen = pick;
        }
        if (chosen == null)
        {
            error = $"no {count} goals {MinSeparation:0} m apart with rings of their own.";
            return null;
        }

        var goals = new ProceduralCityBuilder.LayoutGoal[count];
        for (int gi = 0; gi < count; gi++)
        {
            int k = chosen[gi];
            ProceduralCityBuilder.LayoutBox centre = file.boxes[building[k]];
            Vector2 c = Centre(k);
            centre.role = "goal_centre";
            centre.goal = gi;
            centre.patrol = false;
            Vector2[] own = PrismBuilding.Footprint(centre);
            float width = 0f, reach = 0f;
            foreach (Vector2 p in own)
            {
                reach = Mathf.Max(reach, (p - c).magnitude);
                foreach (Vector2 q in own) width = Mathf.Max(width, (p - q).magnitude);
            }
            var names = new List<string>();
            float sum = 0f, nearest = float.PositiveInfinity, patch = 0f;
            foreach (int j in ring[k])
            {
                ProceduralCityBuilder.LayoutBox b = file.boxes[building[j]];
                b.role = "goal_ring";
                b.goal = gi;
                b.patrol = true;
                names.Add(b.name);
                sum += (Centre(j) - c).magnitude;
                Vector2[] fp = PrismBuilding.Footprint(b);
                for (int i = 0; i < fp.Length; i++)
                {
                    nearest = Mathf.Min(nearest, ToSegment(c, fp[i], fp[(i + 1) % fp.Length]));
                    patch = Mathf.Max(patch, (fp[i] - c).magnitude);
                }
            }
            goals[gi] = new ProceduralCityBuilder.LayoutGoal
            {
                index = gi,
                kind = Kind,
                block_id = k,
                centre = new[] { c.x, c.y },
                centre_box = centre.name,
                ring_boxes = names.ToArray(),
                n_b = names.Count,
                w_c = width,                 // the footprint's longest chord
                R_c = reach,                 // its farthest corner from the centre
                r_g = sum / names.Count,     // the ring's centres, on average
                c_g = nearest,               // the nearest ring facade
                patch_radius = patch,        // the disc round the ring's footprints
                clear_radius = 0f,           // nothing is cleared for a goal drawn from the city
                ring_size = 1f,
            };
        }
        return goals;
    }

    private static SortedSet<int> Ring(Dictionary<int, SortedSet<int>> ring, int k)
    {
        if (!ring.TryGetValue(k, out SortedSet<int> r)) ring[k] = r = new SortedSet<int>();
        return r;
    }

    private static float ToSegment(Vector2 p, Vector2 a, Vector2 b)
    {
        Vector2 ab = b - a;
        float t = ab.sqrMagnitude < 1e-9f ? 0f : Mathf.Clamp01(Vector2.Dot(p - a, ab) / ab.sqrMagnitude);
        return (a + ab * t - p).magnitude;
    }
}
