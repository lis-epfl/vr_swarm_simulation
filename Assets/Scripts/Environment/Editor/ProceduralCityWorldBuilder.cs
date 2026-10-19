using System.Collections.Generic;
using System.IO;
using System.Linq;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;
using UnityEngine.SceneManagement;

/// <summary>
/// <c>Tools/Swarm/Build procedural city world</c>: makes <c>ProceduralCityWorld.unity</c>, where a generated layout
/// (Assets/Scripts/Experiment/layouts, from citysim/citygen.py) is the city, by taking DiamondCityWorld and removing its
/// city — the way <see cref="PracticeWorldBuilder"/> makes the practice world, and for the same reason: the swarm gains,
/// the altitude ceiling, the spawn, the input, the feed screens, PyUniSharingFast, the Arena, the rig, the lighting
/// and the four Terrain tiles the ceiling is measured from all live on scene objects, so copying the scene is what
/// keeps the flight experience identical. Rebuild after changing DiamondCityWorld; hand edits here are lost.
///
/// <para>Taken out: the tile city and the diamonds (<see cref="RemovedRoots"/>; ParkRoad stays as the spawn pad), and
/// the components that only serve the tile city (<see cref="StrippedComponents"/>: the goal replacer and the city
/// tuners and planter). Put in: a <c>ProceduralCity</c> root with a <see cref="ProceduralCityBuilder"/> that builds the
/// layout in Awake (the scene is saved without it built, so a different layout is one field — or the
/// <c>CITY_LAYOUT</c> environment variable — away). The build is run once here, in edit mode, to check the layout
/// builds (every collider asserted onto its box), then cleared before saving.</para>
///
/// <para>Also makes, if missing, the goal root prefab (<see cref="GoalRootPrefabPath"/>: a <see cref="GoalSpecialWalker"/>
/// with its hat prefabs set explicitly — the list is normally filled by the editor, so it has to come from a prefab —
/// and a <see cref="ProceduralGoal"/>) and the street material.</para>
///
/// <para><c>Tools/Swarm/Build procedural city from layout…</c> bakes a chosen layout into the open scene
/// (<see cref="ProceduralCityBuilder.Bake"/>), for looking at it; saving that is safe — Play adopts a bake only while it
/// is of the layout to build. Batch: <c>-executeMethod ProceduralCityWorldBuilder.Run</c> (CITY_LAYOUT names the layout).
/// <see cref="VoronoiCityWorldBuilder"/> makes a second scene the same way, with its layout baked in.</para>
/// </summary>
public static class ProceduralCityWorldBuilder
{
    private const string SourceScenePath = "Assets/Scenes/DiamondCityWorld.unity";
    public const string ScenePath = "Assets/Scenes/ProceduralCityWorld.unity";
    /// <summary>The trial layouts, <c>L&lt;n&gt;_&lt;level&gt;__&lt;id&gt;.json</c> (citygen trial).</summary>
    private const string TrialLayoutPattern = "L?_*.json";
    public const string GoalRootPrefabPath = "Assets/Prefabs/ScaledCity/ProceduralGoalRoot.prefab";
    public const string StreetMaterialPath = "Assets/Materials/ProceduralStreet.mat";
    private const string WalkerPrefabPath = "Assets/Prefabs/SimpleWalker.prefab";
    private static readonly string[] HatPrefabPaths =
    {
        "Assets/Prefabs/CapWalker.prefab", "Assets/Prefabs/CowboyWalker.prefab", "Assets/Prefabs/BucketWalker.prefab",
    };
    private static readonly string[] RemovedRoots = { "City_Pack_01", "Diamonds" };
    private static readonly System.Type[] StrippedComponents =
    {
        typeof(GoalPatchReplacer), typeof(StreetWidthTuner), typeof(BuildingWidthTuner), typeof(VergeTreePlanter),
        typeof(CityRowOffsetter),
    };
    private static readonly Color StreetColour = new Color(0.27f, 0.27f, 0.28f);

    [MenuItem("Tools/Swarm/Build procedural city world")]
    private static void BuildFromMenu()
    {
        if (!EditorSceneManager.SaveCurrentModifiedScenesIfUserWantsTo()) return;
        if (File.Exists(ScenePath) &&
            !EditorUtility.DisplayDialog("Build procedural city world",
                $"Rebuild {ScenePath} from {SourceScenePath}? This replaces the scene, including hand edits.",
                "Rebuild", "Cancel"))
        {
            return;
        }
        string layout = DefaultLayout(TrialLayoutPattern);
        if (!Build(layout, out string summary)) Debug.LogError("ProceduralCityWorldBuilder: " + summary);
    }

    [MenuItem("Tools/Swarm/Build procedural city world", true)]
    private static bool CanBuild() => !EditorApplication.isPlayingOrWillChangePlaymode;

    [MenuItem("Tools/Swarm/Build procedural city from layout…")]
    private static void BuildLayoutInOpenScene()
    {
        ProceduralCityBuilder builder = Object.FindFirstObjectByType<ProceduralCityBuilder>(FindObjectsInactive.Include);
        if (builder == null)
        {
            Debug.LogError("ProceduralCityWorldBuilder: the open scene has no ProceduralCityBuilder (open " + ScenePath + ").");
            return;
        }
        string folder = Path.Combine(Application.dataPath, ProceduralCityBuilder.LayoutFolder);
        string file = EditorUtility.OpenFilePanel("Layout to build", folder, "json");
        if (string.IsNullOrEmpty(file)) return;
        Undo.RecordObject(builder, "Bake procedural city");
        builder.layoutPath = RelativeToLayouts(file);
        if (builder.Bake(out string error))
        {
            EditorSceneManager.MarkSceneDirty(builder.gameObject.scene);
        }
        else
        {
            Debug.LogError("ProceduralCityWorldBuilder: " + error);
        }
    }

    [MenuItem("Tools/Swarm/Build procedural city from layout…", true)]
    private static bool CanBuildLayout() => !EditorApplication.isPlayingOrWillChangePlaymode;

    /// <summary>Batchmode: builds the scene (CITY_LAYOUT, else the first layout in the folder) and exits 0 / 1.</summary>
    public static void Run()
    {
        string layout = System.Environment.GetEnvironmentVariable(ProceduralCityBuilder.LayoutEnv);
        if (string.IsNullOrEmpty(layout)) layout = DefaultLayout(TrialLayoutPattern);
        bool ok = Build(layout, out string summary);
        if (!ok) Debug.LogError("ProceduralCityWorldBuilder: " + summary);
        EditorApplication.Exit(ok ? 0 : 1);
    }

    /// <summary>The first layout in Assets/Scripts/Experiment/layouts matching <paramref name="pattern"/> (sorted),
    /// or "".</summary>
    public static string DefaultLayout(string pattern)
    {
        string folder = Path.Combine(Application.dataPath, ProceduralCityBuilder.LayoutFolder);
        if (!Directory.Exists(folder)) return "";
        string first = Directory.GetFiles(folder, pattern).OrderBy(f => f, System.StringComparer.Ordinal).FirstOrDefault();
        return first == null ? "" : Path.GetFileName(first);
    }

    public static string RelativeToLayouts(string file)
    {
        string folder = Path.GetFullPath(Path.Combine(Application.dataPath, ProceduralCityBuilder.LayoutFolder));
        string full = Path.GetFullPath(file);
        return full.StartsWith(folder) ? full.Substring(folder.Length).TrimStart('\\', '/') : full;
    }

    /// <summary>Rebuilds the scene and leaves it open. False with the reason in <paramref name="summary"/>.</summary>
    public static bool Build(string layout, out string summary) => Build(ScenePath, layout, false, out summary);

    /// <summary>Rebuilds <paramref name="scenePath"/> from DiamondCityWorld around <paramref name="layout"/> and leaves
    /// it open: saved light (built at Awake), or with the city baked in. False with the reason in
    /// <paramref name="summary"/>; the previous scene file is then put back.</summary>
    public static bool Build(string scenePath, string layout, bool bake, out string summary,
                             System.Action<ProceduralCityBuilder> configure = null)
    {
        if (string.IsNullOrEmpty(layout))
        {
            summary = $"no layout for {scenePath} (none found under Assets/{ProceduralCityBuilder.LayoutFolder}).";
            return false;
        }
        // Made (if missing) before any scene is touched, but LOADED only once the scene is open: opening a scene in
        // Single mode unloads every asset no open scene references, which leaves a reference taken earlier dead.
        if (EnsureGoalRootPrefab(out summary) == null) return false;
        EnsureStreetMaterial();
        if (AssetDatabase.LoadAssetAtPath<BuildingCatalogue>(BuildingCatalogueExport.AssetPath) == null)
        {
            summary = $"no {BuildingCatalogueExport.AssetPath}: run Tools/Swarm/Export building catalogue first.";
            return false;
        }
        if (AssetDatabase.LoadAssetAtPath<GameObject>(WalkerPrefabPath) == null)
        {
            summary = $"no walker prefab at {WalkerPrefabPath}.";
            return false;
        }

        byte[] previous = File.Exists(scenePath) ? File.ReadAllBytes(scenePath) : null;
        EditorSceneManager.NewScene(NewSceneSetup.EmptyScene, NewSceneMode.Single);
        File.Copy(SourceScenePath, scenePath, true);
        AssetDatabase.ImportAsset(scenePath, ImportAssetOptions.ForceUpdate);
        Scene scene = EditorSceneManager.OpenScene(scenePath, OpenSceneMode.Single);
        try
        {
            summary = Populate(scene, layout, bake, configure);
            if (summary != null && !EditorSceneManager.SaveScene(scene)) summary = null;
        }
        catch (System.Exception e)
        {
            Debug.LogException(e);
            summary = null;
        }
        if (summary == null)
        {
            EditorSceneManager.NewScene(NewSceneSetup.EmptyScene, NewSceneMode.Single);
            if (previous != null)
            {
                File.WriteAllBytes(scenePath, previous);
                AssetDatabase.ImportAsset(scenePath, ImportAssetOptions.ForceUpdate);
            }
            else
            {
                AssetDatabase.DeleteAsset(scenePath);
            }
            summary = "the build failed (see the log above); the previous scene was put back.";
            return false;
        }
        Debug.Log($"ProceduralCityWorldBuilder: built {scenePath} from {SourceScenePath}: {summary}");
        return true;
    }

    private static string Populate(Scene scene, string layout, bool bake, System.Action<ProceduralCityBuilder> configure)
    {
        BuildingCatalogue catalogue = AssetDatabase.LoadAssetAtPath<BuildingCatalogue>(BuildingCatalogueExport.AssetPath);
        GameObject goalRoot = AssetDatabase.LoadAssetAtPath<GameObject>(GoalRootPrefabPath);
        GameObject walker = AssetDatabase.LoadAssetAtPath<GameObject>(WalkerPrefabPath);
        Material street = AssetDatabase.LoadAssetAtPath<Material>(StreetMaterialPath);
        int removed = 0;
        foreach (GameObject root in scene.GetRootGameObjects())
        {
            if (RemovedRoots.Contains(root.name))
            {
                Object.DestroyImmediate(root);
                removed++;
            }
        }
        int stripped = 0;
        foreach (System.Type type in StrippedComponents)
        {
            foreach (Object c in Object.FindObjectsByType(type, FindObjectsInactive.Include, FindObjectsSortMode.None))
            {
                Object.DestroyImmediate(c);
                stripped++;
            }
        }

        var cityRoot = new GameObject("ProceduralCity");
        SceneManager.MoveGameObjectToScene(cityRoot, scene);
        ProceduralCityBuilder builder = cityRoot.AddComponent<ProceduralCityBuilder>();
        builder.layoutPath = layout;
        builder.catalogue = catalogue;
        builder.goalRootPrefab = goalRoot;
        builder.walkerPrefab = walker;
        builder.streetMaterial = street;
        builder.buildOnAwake = true;
        configure?.Invoke(builder);

        // Build once to check the layout and set the spawn from it, then either clear (the scene is saved light and the
        // city is built from the layout file on Awake) or keep it as a bake (adopted at Play).
        TerrainHeightSampler.Invalidate();
        bool ok = bake ? builder.Bake(out string error) : builder.Build(out error);
        if (!ok)
        {
            Debug.LogError("ProceduralCityWorldBuilder: " + error);
            return null;
        }
        string id = builder.Layout.layout.id;
        int buildings = builder.Layout.boxes.Length, goals = builder.Layout.layout.goals?.Length ?? 0;
        if (!bake) builder.Clear();
        swarmSpawn spawner = Object.FindFirstObjectByType<swarmSpawn>(FindObjectsInactive.Include);
        if (spawner != null) EditorUtility.SetDirty(spawner);
        EditorSceneManager.MarkSceneDirty(scene);
        return $"removed {removed} city root(s) and {stripped} tile-city component(s); layout {id} checked " +
               $"({buildings} buildings, {goals} goals) and " + (bake ? "baked into the scene" : "left to build on Awake");
    }

    /// <summary>The goal root prefab, made if missing: GoalSpecialWalker with the three hats, and ProceduralGoal.</summary>
    private static GameObject EnsureGoalRootPrefab(out string error)
    {
        error = null;
        GameObject existing = AssetDatabase.LoadAssetAtPath<GameObject>(GoalRootPrefabPath);
        if (existing != null) return existing;
        var hats = new List<GameObject>();
        foreach (string p in HatPrefabPaths)
        {
            GameObject h = AssetDatabase.LoadAssetAtPath<GameObject>(p);
            if (h == null)
            {
                error = $"no hat walker prefab at {p}.";
                return null;
            }
            hats.Add(h);
        }
        var go = new GameObject("ProceduralGoalRoot");
        try
        {
            GoalSpecialWalker gsw = go.AddComponent<GoalSpecialWalker>();
            var so = new SerializedObject(gsw);
            SerializedProperty list = so.FindProperty("specialWalkerPrefabs");
            list.arraySize = hats.Count;
            for (int i = 0; i < hats.Count; i++) list.GetArrayElementAtIndex(i).objectReferenceValue = hats[i];
            so.ApplyModifiedPropertiesWithoutUndo();
            go.AddComponent<ProceduralGoal>();
            GameObject prefab = PrefabUtility.SaveAsPrefabAsset(go, GoalRootPrefabPath, out bool ok);
            if (!ok) error = $"could not save {GoalRootPrefabPath}.";
            return ok ? prefab : null;
        }
        finally
        {
            Object.DestroyImmediate(go);
        }
    }

    private static Material EnsureStreetMaterial()
    {
        Material m = AssetDatabase.LoadAssetAtPath<Material>(StreetMaterialPath);
        if (m != null) return m;
        m = new Material(Shader.Find("Standard")) { color = StreetColour };
        m.SetFloat("_Glossiness", 0.1f);
        AssetDatabase.CreateAsset(m, StreetMaterialPath);
        return m;
    }
}
