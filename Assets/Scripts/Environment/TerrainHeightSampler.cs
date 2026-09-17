using System.Collections.Generic;
using UnityEngine;
using UnityEngine.SceneManagement;

/// <summary>
/// Ground height under a world position, read off the scene's Unity Terrains. The one consumer is
/// the altitude ceiling (<see cref="SwarmManager.maxHeightAboveTerrain"/>, applied in
/// <see cref="VelocityControl"/> and <see cref="SwarmPlaneController"/>), which needs "how high is
/// this drone above the ground" every physics tick for every drone.
///
/// <para><b>The terrain list is cached, because reading it is the expensive part.</b>
/// <c>Terrain.activeTerrains</c> allocates a fresh array on every access, so calling it per drone
/// per FixedUpdate would hand the GC ~500 arrays a second for a value that never changes. The
/// sample itself is a bilinear heightmap lookup and costs next to nothing. The cache is rebuilt
/// only when a scene is loaded or unloaded, or when a cached Terrain has been destroyed — terrain
/// is static scenery in every scene here, and nothing moves or resizes one at runtime.</para>
///
/// <para><b>Off the edge of the terrain the nearest tile answers, rather than nobody.</b> The four
/// 1000x1000 tiles in the city scenes cover x,z in [-1000, 1000] and the city sits well inside
/// that, but failing open past the edge would make "fly beyond the terrain" a way to defeat the
/// ceiling. Clamping the query into the nearest tile's bounds keeps the ceiling defined and
/// continuous everywhere. A scene with no Terrain at all (DJIScene, FactoryScene) genuinely has no
/// ground to measure from, and there the ceiling does not apply.</para>
/// </summary>
public static class TerrainHeightSampler
{
    private struct Tile
    {
        public Terrain terrain;
        public float baseY;
        public float minX, maxX, minZ, maxZ;
    }

    private static readonly List<Tile> tiles = new List<Tile>();
    private static bool cacheValid = false;

    // Domain reload is off in some player/editor configurations, so statics survive a Play-mode
    // stop. Reset here rather than trusting the field initialisers above.
    [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.SubsystemRegistration)]
    private static void Initialise()
    {
        Invalidate();

        SceneManager.sceneLoaded -= OnSceneLoaded;
        SceneManager.sceneLoaded += OnSceneLoaded;
        SceneManager.sceneUnloaded -= OnSceneUnloaded;
        SceneManager.sceneUnloaded += OnSceneUnloaded;
    }

    private static void OnSceneLoaded(Scene scene, LoadSceneMode mode) => Invalidate();
    private static void OnSceneUnloaded(Scene scene) => Invalidate();

    /// <summary>Drops the cached terrain list; the next query rebuilds it.</summary>
    public static void Invalidate()
    {
        tiles.Clear();
        cacheValid = false;
    }

    /// <summary>
    /// Ground height (world y) under <paramref name="worldPosition"/>. False when the scene has no
    /// Terrain at all, which every caller treats as "no ground reference, so no limit".
    /// </summary>
    public static bool TryGetHeight(Vector3 worldPosition, out float height)
    {
        EnsureCache();
        height = 0.0f;
        if (tiles.Count == 0) return false;

        int nearest = -1;
        float nearestDistanceSq = float.MaxValue;

        for (int i = 0; i < tiles.Count; i++)
        {
            Tile tile = tiles[i];

            // Clamping into the tile and measuring the leftover distance answers both questions at
            // once: zero means the point is on this tile, and otherwise it ranks the tiles for the
            // off-the-edge fallback.
            float clampedX = Mathf.Clamp(worldPosition.x, tile.minX, tile.maxX);
            float clampedZ = Mathf.Clamp(worldPosition.z, tile.minZ, tile.maxZ);
            float dx = worldPosition.x - clampedX;
            float dz = worldPosition.z - clampedZ;
            float distanceSq = dx * dx + dz * dz;

            if (distanceSq < nearestDistanceSq)
            {
                nearest = i;
                nearestDistanceSq = distanceSq;
                if (distanceSq <= 0.0f) break;
            }
        }

        if (nearest < 0) return false;

        Tile chosen = tiles[nearest];
        if (chosen.terrain == null)
        {
            // Destroyed between the rebuild above and this line — a scene being torn down. Drop the
            // cache and answer "no ground" for this one tick rather than retrying into itself; the
            // next call rebuilds. Every caller treats that as "no ceiling", which for a scene that
            // is going away is the right answer anyway.
            Invalidate();
            return false;
        }

        Vector3 samplePoint = new Vector3(Mathf.Clamp(worldPosition.x, chosen.minX, chosen.maxX),
                                          worldPosition.y,
                                          Mathf.Clamp(worldPosition.z, chosen.minZ, chosen.maxZ));

        // SampleHeight is relative to the terrain's own transform, so the tile's base y goes back on.
        height = chosen.terrain.SampleHeight(samplePoint) + chosen.baseY;
        return true;
    }

    private static void EnsureCache()
    {
        if (cacheValid)
        {
            for (int i = 0; i < tiles.Count; i++)
            {
                if (tiles[i].terrain == null)
                {
                    cacheValid = false;
                    break;
                }
            }
        }

        if (cacheValid) return;

        tiles.Clear();
        cacheValid = true;

        Terrain[] active = Terrain.activeTerrains;
        if (active == null) return;

        for (int i = 0; i < active.Length; i++)
        {
            Terrain terrain = active[i];
            if (terrain == null || terrain.terrainData == null) continue;

            Vector3 origin = terrain.transform.position;
            Vector3 size = terrain.terrainData.size;

            tiles.Add(new Tile
            {
                terrain = terrain,
                baseY = origin.y,
                minX = origin.x,
                maxX = origin.x + size.x,
                minZ = origin.z,
                maxZ = origin.z + size.z,
            });
        }
    }
}
