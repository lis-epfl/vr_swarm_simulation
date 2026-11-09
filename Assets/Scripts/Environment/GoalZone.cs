using UnityEngine;

/// <summary>
/// A goal patch's footprint and geometry, for the recorder's auto-scored identify and its session record. A goal root
/// that carries a component implementing this (the procedural city's goals) answers for itself; a tile city's goal
/// patch has none and is the city tile it replaced — a square of half-width <see cref="TileHalf"/> about its root,
/// the rule analyse.py's phases use (<see cref="TileDistance"/>).
/// </summary>
public interface IGoalZone
{
    /// <summary>Horizontal distance (m) from a world point to the goal patch: 0 anywhere over it.</summary>
    float DistanceToPatch(Vector3 world);

    /// <summary>"disc" (procedural goals) or "tile".</summary>
    string ZoneShape { get; }

    /// <summary>The patch's radius (disc goals): ring radius plus a ring building's half-diagonal.</summary>
    float PatchRadius { get; }

    /// <summary>r_g: distance from the goal centre to the ring buildings' centres.</summary>
    float RingRadius { get; }

    /// <summary>w_c: the central occluder's long side.</summary>
    float OccluderWidth { get; }

    /// <summary>c_g: distance from the goal centre to the nearest building other than the occluder.</summary>
    float Clearance { get; }
}

public static class GoalZones
{
    /// <summary>A tile goal's half-width: half of CityTiles.Pitch (analyse.TILE_HALF).</summary>
    public const float TileHalf = CityTiles.Pitch / 2f;

    /// <summary>Distance from a point to a tile goal's square about its root (analyse.tile_distance).</summary>
    public static float TileDistance(Transform goal, Vector3 world)
    {
        Vector3 d = world - goal.position;
        float ex = Mathf.Max(Mathf.Abs(d.x) - TileHalf, 0f);
        float ez = Mathf.Max(Mathf.Abs(d.z) - TileHalf, 0f);
        return Mathf.Sqrt(ex * ex + ez * ez);
    }

    /// <summary>The goal's own zone if its root has one, else the tile rule.</summary>
    public static float DistanceToPatch(Transform goal, Vector3 world)
    {
        IGoalZone zone = goal != null ? goal.GetComponent<IGoalZone>() : null;
        return zone != null ? zone.DistanceToPatch(world) : TileDistance(goal, world);
    }
}
