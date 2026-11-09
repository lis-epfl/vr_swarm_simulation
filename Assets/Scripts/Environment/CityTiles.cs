using System.Collections.Generic;
using UnityEngine;
using UnityEngine.SceneManagement;

/// <summary>
/// What the city-editing components share about the Modular City Pack's tile grid: its dimensions, how a tile
/// and its block are found, and the containers that tie scenery to a tile.
///
/// <para><b>A tile is located by its kerb</b> (<c>Carbs_NN</c>), the object its block's footpath and kerb meshes
/// are authored about — never by the tile's own transform. <c>MC_Patch_32</c> has its whole block baked ~318
/// units off its pivot, so its transform sits at the city centre while its buildings stand at the edge.</para>
///
/// <para><b>Tied scenery hangs under one of three children of its tile, named for where it stands</b>, because
/// that alone decides how it answers <see cref="StreetWidthTuner"/> shrinking the block: see
/// <see cref="BlockShare"/>. Every city tool agrees on these names, and <see cref="GoalPatchReplacer"/> hands all
/// three over to the goal that replaces a tile.</para>
/// </summary>
public static class CityTiles
{
    /// <summary>Name prefix of a tile's kerb. Every patch in the pack has exactly one.</summary>
    public const string KerbPrefix = "Carbs_";

    /// <summary>Tile pitch, and the size of the <c>Road_Structure</c> plate. 3576 in at 0.0254 m/in.</summary>
    public const float Pitch = 90.8304f;

    /// <summary>
    /// How far an authored block's footpath and kerb reach from its centre: 1500 in. Beyond it, out to half the
    /// pitch, is street: the <c>Road_Structure</c> ring's inner edge is on this line, and stays there however the
    /// block is scaled.
    /// </summary>
    public const float BlockHalfSpan = 38.1f;

    /// <summary>Scenery standing on the block, such as a garden or trees on the footpath. Moves with the block.</summary>
    public const string SceneryContainer = "Scenery";

    /// <summary>
    /// Scenery standing in the street: the median trees and verge strips on the line between two tiles. Stays
    /// where it was authored, which is the middle of the street at every block scale.
    /// </summary>
    public const string StreetContainer = "Street";

    /// <summary>
    /// Scenery on the grass verge a shrunk block leaves between its footpath and the road (see
    /// <see cref="VergeTreePlanter"/>). Stays in the middle of the verge at every block scale.
    /// </summary>
    public const string VergeContainer = "Verge";

    /// <summary>Every container that ties scenery to a tile.</summary>
    public static readonly string[] TiedContainers = { SceneryContainer, StreetContainer, VergeContainer };

    /// <summary>
    /// How much of the block's shrink an object in <paramref name="containerName"/> follows. 1 moves with the block;
    /// 0 stays put; ½ keeps to the middle of the verge, whose inner edge is the footpath (moving with the block) and
    /// whose outer edge is the road-edge line (fixed). Anything not in a known container follows the block.
    /// </summary>
    public static float BlockShare(string containerName)
    {
        switch (containerName)
        {
            case StreetContainer: return 0f;
            case VergeContainer: return 0.5f;
            default: return 1f;
        }
    }

    /// <summary>
    /// The factor, about its tile's centre, that an object following <paramref name="share"/> of the block's
    /// shrink is scaled by at <paramref name="blockScale"/>.
    /// </summary>
    public static float ScaleFor(float blockScale, float share)
    {
        return 1f + share * (blockScale - 1f);
    }

    /// <summary>The kerb under <paramref name="root"/>, or null if it has none.</summary>
    public static Transform FindKerb(Transform root)
    {
        foreach (Transform t in root.GetComponentsInChildren<Transform>(true))
        {
            if (t.name.StartsWith(KerbPrefix))
            {
                return t;
            }
        }
        return null;
    }

    /// <summary>
    /// The index of the tile whose square footprint holds <paramref name="point"/>: the nearest centre in the
    /// square (Chebyshev) metric, ties broken by straight-line distance, which only happens outside the city.
    /// Unlike the straight-line nearest centre, this is still the square the point is in once alternate rows
    /// are offset by half a tile. <paramref name="margin"/> is how much further away the runner-up is.
    /// </summary>
    public static int FootprintOwner(IReadOnlyList<Vector3> centres, Vector3 point, out float margin)
    {
        int best = -1;
        float bestSquare = float.MaxValue;
        float bestRound = float.MaxValue;
        float runnerUp = float.MaxValue;
        for (int i = 0; i < centres.Count; i++)
        {
            float dx = Mathf.Abs(point.x - centres[i].x);
            float dz = Mathf.Abs(point.z - centres[i].z);
            float square = Mathf.Max(dx, dz);
            float round = dx * dx + dz * dz;
            if (square < bestSquare || (square == bestSquare && round < bestRound))
            {
                runnerUp = Mathf.Min(runnerUp, bestSquare);
                best = i;
                bestSquare = square;
                bestRound = round;
            }
            else
            {
                runnerUp = Mathf.Min(runnerUp, square);
            }
        }
        margin = runnerUp - bestSquare;
        return best;
    }

    /// <summary>A city: its root, and as its tiles the root's children that hold a kerb.</summary>
    public sealed class City
    {
        public Transform Root;
        public readonly List<Transform> Tiles = new List<Transform>();
        public readonly List<Vector3> Centres = new List<Vector3>(); // each tile's kerb, world space

        /// <summary>Children of the root that are not tiles, i.e. scenery that moving a tile leaves behind.</summary>
        public List<Transform> LooseChildren()
        {
            HashSet<Transform> tiles = new HashSet<Transform>(Tiles);
            List<Transform> loose = new List<Transform>();
            foreach (Transform child in Root)
            {
                if (!tiles.Contains(child))
                {
                    loose.Add(child);
                }
            }
            return loose;
        }
    }

    /// <summary>The city in <paramref name="scene"/>: the lowest node holding every kerb in it.</summary>
    public static City FindCity(Scene scene, out string error)
    {
        List<Transform> kerbs = new List<Transform>();
        foreach (GameObject rootObject in scene.GetRootGameObjects())
        {
            CollectKerbs(rootObject.transform, kerbs);
        }
        return CityFromKerbs(kerbs, scene.name, out error);
    }

    /// <summary>The city under <paramref name="scope"/>: the lowest node below it holding every kerb.</summary>
    public static City FindCity(Transform scope, out string error)
    {
        List<Transform> kerbs = new List<Transform>();
        CollectKerbs(scope, kerbs);
        return CityFromKerbs(kerbs, scope.name, out error);
    }

    private static void CollectKerbs(Transform root, List<Transform> kerbs)
    {
        foreach (Transform t in root.GetComponentsInChildren<Transform>(true))
        {
            if (t.name.StartsWith(KerbPrefix))
            {
                kerbs.Add(t);
            }
        }
    }

    private static City CityFromKerbs(List<Transform> kerbs, string where, out string error)
    {
        if (kerbs.Count < 2)
        {
            error = $"found {kerbs.Count} tile kerbs ({KerbPrefix}NN) in {where}; a city needs at least two.";
            return null;
        }

        Transform root = kerbs[0].parent;
        foreach (Transform kerb in kerbs)
        {
            while (root != null && !kerb.IsChildOf(root))
            {
                root = root.parent;
            }
        }
        if (root == null)
        {
            error = $"the tile kerbs in {where} share no common parent, so there is no city root.";
            return null;
        }

        City city = new City { Root = root };
        foreach (Transform kerb in kerbs)
        {
            Transform tile = kerb;
            while (tile.parent != root)
            {
                tile = tile.parent;
            }
            if (city.Tiles.Contains(tile))
            {
                error = $"{tile.name} holds more than one kerb, so it cannot be told apart from its neighbours.";
                return null;
            }
            city.Tiles.Add(tile);
            city.Centres.Add(kerb.position);
        }

        error = null;
        return city;
    }
}
