using System;
using System.Collections.Generic;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;

/// <summary>
/// Bakes occlusion culling for a generated city's flight volume, with only the buildings and the terrain as occluders,
/// and records which city it was baked for (<see cref="ProceduralCityBuilder.occlusionCitySignature"/>). An FPV render
/// then skips everything a building hides, which in BlockShapedPatchCityWorld halved its cost (1.85 to 0.98 ms of
/// main thread) and held a recorded flight's replay at 72 fps throughout.
///
/// <para><b>Use this, not the Occlusion window.</b> CityDressing marks its props occluder-static, and a tree's
/// alpha-tested leaves as an occluder hide whatever stands behind the gaps between them: a walker behind a canopy would
/// vanish. Here only Obstacle colliders' objects and the Terrains occlude, everything else is only culled. The occlusion
/// area exists only while baking, so the scene keeps no extra object.</para>
///
/// <para><b>Rebake after rebaking the city.</b> Data baked for buildings that have moved hides what stands where they
/// were, so at Play a city that does not match the recorded one flies with occlusion off and a warning. Not for a tile
/// city: GoalPatchReplacer destroys and replaces tiles at runtime (CLAUDE.md, "Do not bake occlusion culling").</para>
/// </summary>
public static class FlightOcclusionBaker
{
    private const float Margin = 150f;          // m round the buildings: the streets beyond the edge row are flown too
    private const float SmallestOccluder = 8f;
    private const float SmallestHole = 0.5f;

    [MenuItem("Tools/Swarm/Bake flight occlusion")]
    public static void BakeOpenScene()
    {
        if (!Bake(out string message))
        {
            EditorUtility.DisplayDialog("Bake flight occlusion", message, "OK");
            return;
        }
        Debug.Log("[FlightOcclusionBaker] " + message);
        EditorUtility.DisplayDialog("Bake flight occlusion", message + "\n\nSave the scene to keep it.", "OK");
    }

    /// <summary>Batchmode: <c>-executeMethod FlightOcclusionBaker.BakeFromCommandLine</c>, scenes in FLIGHT_OCCLUSION_SCENES (';').</summary>
    public static void BakeFromCommandLine()
    {
        string list = Environment.GetEnvironmentVariable("FLIGHT_OCCLUSION_SCENES");
        int failed = 0;
        foreach (string scene in (list ?? "").Split(new[] { ';' }, StringSplitOptions.RemoveEmptyEntries))
        {
            EditorSceneManager.OpenScene(scene.Trim());
            bool ok = Bake(out string message);
            Debug.Log($"[FlightOcclusionBaker] {scene}: {message}");
            if (ok) EditorSceneManager.SaveOpenScenes();
            else failed++;
        }
        AssetDatabase.SaveAssets();
        EditorApplication.Exit(failed == 0 ? 0 : 1);
    }

    /// <summary>For a builder that has just made the open scene: bakes it, leaving the save to the caller.</summary>
    public static bool BakeForOpenScene(out string message) => Bake(out message);

    private static bool Bake(out string message)
    {
        ProceduralCityBuilder city = UnityEngine.Object.FindObjectOfType<ProceduralCityBuilder>();
        if (city == null)
        {
            message = "This scene has no ProceduralCityBuilder: the bake is for generated cities only.";
            return false;
        }

        int obstacle = LayerMask.NameToLayer("Obstacle");
        var occluders = new HashSet<GameObject>();
        Bounds area = default;
        bool any = false;
        foreach (Collider c in UnityEngine.Object.FindObjectsOfType<Collider>())
        {
            if (c.gameObject.layer != obstacle) continue;
            foreach (Renderer r in c.GetComponentsInChildren<Renderer>(true)) occluders.Add(r.gameObject);
            if (any) area.Encapsulate(c.bounds);
            else { area = c.bounds; any = true; }
        }
        if (!any)
        {
            message = "No colliders on the Obstacle layer: no buildings to bake for.";
            return false;
        }
        swarmSpawn spawn = UnityEngine.Object.FindObjectOfType<swarmSpawn>();
        if (spawn != null) area.Encapsulate(new Vector3(spawn.start_x, spawn.start_y, spawn.start_z));
        SwarmManager manager = UnityEngine.Object.FindObjectOfType<SwarmManager>();
        float ceiling = manager != null && manager.maxHeightAboveTerrain > 0f ? manager.maxHeightAboveTerrain : 80f;
        Vector3 min = area.min - new Vector3(Margin, 0f, Margin), max = area.max + new Vector3(Margin, 0f, Margin);
        min.y = -2f;
        max.y = ceiling + 25f;     // the flown ceiling sits ~1.3 m above the number; the rest is margin

        // Occluders: the buildings and the ground, nothing else, for as long as the bake takes.
        var restore = new Dictionary<GameObject, StaticEditorFlags>();
        foreach (Renderer r in UnityEngine.Object.FindObjectsOfType<Renderer>())
        {
            GameObject go = r.gameObject;
            StaticEditorFlags flags = GameObjectUtility.GetStaticEditorFlags(go);
            if ((flags & StaticEditorFlags.OccluderStatic) == 0 || occluders.Contains(go)) continue;
            restore[go] = flags;
            GameObjectUtility.SetStaticEditorFlags(go, flags & ~StaticEditorFlags.OccluderStatic);
        }

        var areaObject = new GameObject("Flight occlusion area (bake only)");
        OcclusionArea occlusionArea = areaObject.AddComponent<OcclusionArea>();
        occlusionArea.center = (min + max) * 0.5f;
        occlusionArea.size = max - min;

        bool ok;
        var clock = System.Diagnostics.Stopwatch.StartNew();
        try
        {
            StaticOcclusionCulling.smallestOccluder = SmallestOccluder;
            StaticOcclusionCulling.smallestHole = SmallestHole;
            StaticOcclusionCulling.backfaceThreshold = 100f;
            ok = StaticOcclusionCulling.Compute();
        }
        finally
        {
            UnityEngine.Object.DestroyImmediate(areaObject);
            foreach (var kv in restore) GameObjectUtility.SetStaticEditorFlags(kv.Key, kv.Value);
        }
        if (!ok)
        {
            message = "Unity's occlusion bake failed; see the console.";
            return false;
        }

        Undo.RecordObject(city, "Bake flight occlusion");
        city.occlusionCitySignature = ProceduralCityBuilder.CitySignature();
        EditorUtility.SetDirty(city);
        EditorSceneManager.MarkSceneDirty(city.gameObject.scene);
        message = $"Baked in {clock.Elapsed.TotalSeconds:0.0} s for {occluders.Count} building renderers over " +
                  $"{(max.x - min.x):0} x {(max.z - min.z):0} m up to {max.y:0} m ({restore.Count} other renderers " +
                  $"left as occludees only); city {city.occlusionCitySignature}.";
        return true;
    }
}
