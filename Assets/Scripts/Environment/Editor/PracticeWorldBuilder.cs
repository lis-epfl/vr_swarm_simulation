using System.Collections.Generic;
using System.IO;
using TMPro;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;
using UnityEngine.SceneManagement;

/// <summary>
/// Builds <c>PracticeWorld</c>, where participants train before ScaledCityWorld: the same swarm, rig and display, the
/// four pedestrians the experiment uses standing in a row under signs that name their hat, and one city tile to
/// practise obstacle avoidance on.
///
/// <para><b>It is ScaledCityWorld with things taken out, not a scene assembled from parts</b>, because what has to
/// carry over is the flight experience, and that lives on scene objects rather than on prefabs: the swarm gains and
/// the altitude ceiling on <see cref="SwarmManager"/>, the spawn on <see cref="swarmSpawn"/>, the input, feed screens
/// and layout on the game manager, the panorama on <see cref="PyUniSharingFast"/>, the Arena and camera rig the pilot
/// sits in, and the lighting. Copying the scene file carries all of it, so <b>rebuilding after ScaledCityWorld changes
/// is how the two stay in step</b> — a hand-made copy would drift the first time the gains were retuned. Rebuilding
/// also discards anything changed in the practice scene by hand; change the constants here instead.</para>
///
/// <para>What is taken out is every city tile but <see cref="KeptTileName"/>, and the components that only serve the
/// experiment or the tuning of the city (<see cref="StrippedComponents"/>). None of those undoes its work when removed
/// — each acts only when its own values change — so the kept tile keeps ScaledCityWorld's street width, verge and
/// trees exactly, and its buildings stay on the <c>Obstacle</c> layer the swarm avoids.</para>
///
/// <para><b>The walkers stand still because nothing moves them</b>: walking is <see cref="WalkerPatrol"/>'s doing,
/// and these are bare prefab instances. The walk <i>cycle</i> is on the prefab, though, so it is swapped for the
/// character pack's idle loop — left alone they would walk on the spot, and with the Animator off they would stand in
/// the model's T-pose. Root motion goes off with it, so a long session cannot drift them out from under their signs.
/// Each hat hangs off its walker's head bone and follows the idle exactly as it follows the walk.</para>
/// </summary>
public static class PracticeWorldBuilder
{
    private const string SourceScenePath = "Assets/Scenes/ScaledCityWorld.unity";
    public const string PracticeScenePath = "Assets/Scenes/PracticeWorld.unity";

    /// <summary>
    /// The tile kept for obstacle practice. Eight buildings of mixed footprint — a slab across the middle, blocks on
    /// the corners and a row of three small towers — so one tile has the whole range of gaps, from ones the swarm
    /// flies through in formation to ones it has to split around. Most tiles have three or four corner blocks and an
    /// open middle.
    /// </summary>
    private const string KeptTileName = "MC_Patch_14";

    private const string IdleControllerPath = "Assets/Kevin Iglesias/Human Animations/Unity Demo Scenes/" +
                                              "Human Basic Motions/AnimatorControllers/HumanM@Idles.controller";

    private const string SignMaterialPath = "Assets/Materials/PracticeSign.mat";

    // The layout runs east from the drone spawn: down ParkRoad, the way ScaledCityWorld's city lies. The swarm starts
    // on the ground and holds whatever height the pilot climbs to, and a level FPV camera at height h sees the ground
    // only beyond ~1.5 h, so at 45 m the walkers stay in view from the spawn through a climb to ~29 m. The tile is far
    // enough past them that the two exercises do not overlap.
    private static readonly Vector3 DownRange = Vector3.right;
    private const float WalkerRowDistance = 45f; // spawn to the row of walkers, metres
    private const float WalkerSpacing = 7f;      // between neighbouring walkers
    private const float TileCentreDistance = 120f; // spawn to the tile's block centre

    // Sign geometry, metres. The board clears the tallest hat (~2.0 m) and hangs directly over its walker, so it
    // cannot come between a drone's camera and the hat: only a view from straight overhead crosses it, and from there
    // it is 5 cm thick.
    private const float BoardWidth = 3f;
    private const float BoardHeight = 1f;
    private const float BoardThickness = 0.05f;
    private const float BoardBottom = 2.4f;
    private const float PostDiameter = 0.08f;
    private const float PostInset = 0.15f;      // from the board's ends
    private const float LabelMargin = 0.1f;     // clear space round the text
    private const float LabelStandoff = 0.02f;  // text off the board face, enough not to z-fight at range
    private static readonly Color BoardColour = new Color(0.95f, 0.95f, 0.95f);

    private readonly struct Station
    {
        public readonly string PrefabPath;
        public readonly string Label;

        public Station(string prefabPath, string label)
        {
            PrefabPath = prefabPath;
            Label = label;
        }
    }

    // Left to right as seen from the spawn. The three hats are GoalSpecialWalker's special walkers; SimpleWalker is
    // everyone else in a goal patch.
    private static readonly Station[] Stations =
    {
        new Station("Assets/Prefabs/CapWalker.prefab", "CAP"),
        new Station("Assets/Prefabs/BucketWalker.prefab", "BUCKET"),
        new Station("Assets/Prefabs/CowboyWalker.prefab", "COWBOY"),
        new Station("Assets/Prefabs/SimpleWalker.prefab", "NO HAT"),
    };

    // Experiment bookkeeping and city tuning. The tuners' work is already baked into the tile's transforms, and with
    // no goal patches spawned there is nothing left for their play-mode passes to match.
    private static readonly System.Type[] StrippedComponents =
    {
        typeof(ExperimentRecorder),
        typeof(GoalPatchReplacer),
        typeof(StreetWidthTuner),
        typeof(BuildingWidthTuner),
        typeof(VergeTreePlanter),
        typeof(CityRowOffsetter),
    };

    // What the build places, loaded and checked before anything on disk is touched.
    private sealed class Parts
    {
        public RuntimeAnimatorController Idle;
        public TMP_FontAsset Font;
        public readonly List<GameObject> Walkers = new List<GameObject>();
    }

    [MenuItem("Tools/Swarm/Build practice world")]
    private static void BuildFromMenu()
    {
        if (!EditorSceneManager.SaveCurrentModifiedScenesIfUserWantsTo())
        {
            return;
        }
        if (File.Exists(PracticeScenePath) &&
            !EditorUtility.DisplayDialog("Build practice world",
                $"Rebuild {PracticeScenePath} from {SourceScenePath}? This replaces the practice scene, " +
                "including anything changed in it by hand.", "Rebuild", "Cancel"))
        {
            return;
        }
        Build();
    }

    [MenuItem("Tools/Swarm/Build practice world", true)]
    private static bool CanBuild()
    {
        return !EditorApplication.isPlayingOrWillChangePlaymode;
    }

    /// <summary>
    /// Rebuilds the practice scene from ScaledCityWorld, saves it and leaves it open. On failure it logs why, puts back
    /// whatever practice scene was there before, and returns false. ScaledCityWorld itself is never opened.
    /// </summary>
    public static bool Build()
    {
        Parts parts = LoadParts(out string error);
        if (parts == null)
        {
            Debug.LogError("PracticeWorldBuilder: " + error);
            return false;
        }

        SceneSetup[] previousSetup = EditorSceneManager.GetSceneManagerSetup();
        byte[] previousScene = File.Exists(PracticeScenePath) ? File.ReadAllBytes(PracticeScenePath) : null;

        // Nothing may hold the practice scene open while its file is replaced underneath it.
        EditorSceneManager.NewScene(NewSceneSetup.EmptyScene, NewSceneMode.Single);
        File.Copy(SourceScenePath, PracticeScenePath, true);
        AssetDatabase.ImportAsset(PracticeScenePath, ImportAssetOptions.ForceUpdate);
        Scene scene = EditorSceneManager.OpenScene(PracticeScenePath, OpenSceneMode.Single);

        string summary;
        try
        {
            summary = Populate(scene, parts, out error);
            if (summary != null && !EditorSceneManager.SaveScene(scene))
            {
                summary = null;
                error = $"could not save {PracticeScenePath}.";
            }
        }
        catch (System.Exception e)
        {
            summary = null;
            error = e.ToString();
        }

        if (summary == null)
        {
            Debug.LogError("PracticeWorldBuilder: " + error + " The practice scene was left as it was.");
            EditorSceneManager.NewScene(NewSceneSetup.EmptyScene, NewSceneMode.Single);
            if (previousScene != null)
            {
                File.WriteAllBytes(PracticeScenePath, previousScene);
                AssetDatabase.ImportAsset(PracticeScenePath, ImportAssetOptions.ForceUpdate);
            }
            else
            {
                AssetDatabase.DeleteAsset(PracticeScenePath);
            }
            // An untitled scene has no path to reopen it from.
            if (previousSetup.Length > 0 &&
                System.Array.TrueForAll(previousSetup, s => !string.IsNullOrEmpty(s.path)))
            {
                EditorSceneManager.RestoreSceneManagerSetup(previousSetup);
            }
            return false;
        }

        Debug.Log($"PracticeWorldBuilder: built {PracticeScenePath} from {SourceScenePath}: {summary}");
        return true;
    }

    private static Parts LoadParts(out string error)
    {
        Parts parts = new Parts
        {
            Idle = AssetDatabase.LoadAssetAtPath<RuntimeAnimatorController>(IdleControllerPath),
            Font = TMP_Settings.defaultFontAsset,
        };
        if (parts.Idle == null)
        {
            error = $"no idle animator controller at {IdleControllerPath}.";
            return null;
        }
        if (parts.Font == null)
        {
            error = "TextMesh Pro has no default font asset; import the TMP Essential Resources.";
            return null;
        }
        foreach (Station station in Stations)
        {
            GameObject prefab = AssetDatabase.LoadAssetAtPath<GameObject>(station.PrefabPath);
            if (prefab == null)
            {
                error = $"no walker prefab at {station.PrefabPath}.";
                return null;
            }
            parts.Walkers.Add(prefab);
        }
        error = null;
        return parts;
    }

    /// <summary>
    /// Turns a fresh copy of ScaledCityWorld into the practice world. Returns a one-line account of what it did, or
    /// null with <paramref name="error"/> set.
    /// </summary>
    private static string Populate(Scene scene, Parts parts, out string error)
    {
        swarmSpawn spawner = Object.FindFirstObjectByType<swarmSpawn>(FindObjectsInactive.Include);
        if (spawner == null)
        {
            error = $"{SourceScenePath} has no swarmSpawn, so there is no spawn to lay the practice area out from.";
            return null;
        }
        Vector3 spawn = new Vector3(spawner.start_x, spawner.start_y, spawner.start_z);

        CityTiles.City city = CityTiles.FindCity(scene, out error);
        if (city == null)
        {
            return null;
        }
        int keep = city.Tiles.FindIndex(t => t.name == KeptTileName);
        if (keep < 0 || city.Tiles.FindLastIndex(t => t.name == KeptTileName) != keep)
        {
            error = $"{SourceScenePath} needs exactly one tile named {KeptTileName}; set KeptTileName to another.";
            return null;
        }

        int stripped = 0;
        foreach (System.Type type in StrippedComponents)
        {
            foreach (Object component in Object.FindObjectsByType(type, FindObjectsInactive.Include,
                                                                  FindObjectsSortMode.None))
            {
                Object.DestroyImmediate(component);
                stripped++;
            }
        }

        // Scenery not tied to any tile belongs to the city as a whole, so it goes with the city. After
        // CitySceneryParenter has run there is none.
        List<Transform> leaving = city.LooseChildren();
        for (int i = 0; i < city.Tiles.Count; i++)
        {
            if (i != keep)
            {
                leaving.Add(city.Tiles[i]);
            }
        }
        foreach (Transform t in leaving)
        {
            Object.DestroyImmediate(t.gameObject);
        }

        // Moved by its kerb, which is where its block is — not by its own transform, which for some tiles is not.
        // The tied scenery hangs under the tile and comes with it.
        Transform tile = city.Tiles[keep];
        Vector3 tileCentre = spawn + DownRange * TileCentreDistance;
        Vector3 shift = tileCentre - city.Centres[keep];
        shift.y = 0f;
        tile.position += shift;

        // The walkers stand on the surface the altitude ceiling is measured from. Its cache follows runtime scene
        // loads, not an editor OpenScene, so it is dropped first.
        TerrainHeightSampler.Invalidate();
        Vector3 rowCentre = OnGround(spawn + DownRange * WalkerRowDistance, spawn.y);

        GameObject row = new GameObject("Practice walkers");
        row.transform.SetPositionAndRotation(rowCentre, Quaternion.LookRotation(Flat(spawn - rowCentre)));

        Material boardMaterial = LoadOrCreateSignMaterial();
        List<TextMeshPro> labels = new List<TextMeshPro>();
        for (int i = 0; i < Stations.Length; i++)
        {
            GameObject station = new GameObject(Stations[i].Label + " station");
            station.transform.SetParent(row.transform, false);

            // The row faces the spawn, so its +X is the spawn's left.
            float across = ((Stations.Length - 1) * 0.5f - i) * WalkerSpacing;
            station.transform.localPosition = new Vector3(across, 0f, 0f);
            station.transform.position = OnGround(station.transform.position, rowCentre.y);

            GameObject walker = (GameObject)PrefabUtility.InstantiatePrefab(parts.Walkers[i], station.transform);
            walker.transform.localPosition = Vector3.zero;
            walker.transform.localRotation = Quaternion.identity;
            foreach (Animator animator in walker.GetComponentsInChildren<Animator>(true))
            {
                animator.runtimeAnimatorController = parts.Idle;
                animator.applyRootMotion = false;
                PrefabUtility.RecordPrefabInstancePropertyModifications(animator);
            }

            labels.AddRange(BuildSign(Stations[i].Label, station.transform, boardMaterial, parts.Font));
        }
        float fontSize = MatchLabelSizes(labels);

        EditorSceneManager.MarkSceneDirty(scene);
        return $"kept {KeptTileName} with its block centred at {tileCentre}, removed {leaving.Count} tiles and loose " +
               $"objects and {stripped} experiment/tuning components, stood {Stations.Length} walkers at {rowCentre} " +
               $"under signs lettered at font size {fontSize:F2}.";
    }

    /// <summary>A board on two posts, lettered on both faces. Returns the two labels.</summary>
    private static TextMeshPro[] BuildSign(string text, Transform station, Material boardMaterial, TMP_FontAsset font)
    {
        GameObject sign = new GameObject("Sign");
        sign.transform.SetParent(station, false);

        float boardCentre = BoardBottom + BoardHeight * 0.5f;
        float top = BoardBottom + BoardHeight;

        GameObject board = Prop(PrimitiveType.Cube, "Board", sign.transform);
        board.transform.localPosition = new Vector3(0f, boardCentre, 0f);
        board.transform.localScale = new Vector3(BoardWidth, BoardHeight, BoardThickness);
        board.GetComponent<MeshRenderer>().sharedMaterial = boardMaterial;

        float postX = BoardWidth * 0.5f - PostInset;
        foreach (float side in new[] { -1f, 1f })
        {
            GameObject post = Prop(PrimitiveType.Cylinder, "Post", sign.transform);
            post.transform.localPosition = new Vector3(side * postX, top * 0.5f, 0f);
            post.transform.localScale = new Vector3(PostDiameter, top * 0.5f, PostDiameter); // the mesh is 2 tall
        }

        // The station faces the spawn along its +Z. A TextMeshPro reads correctly from its own -Z side, so the label
        // on the spawn's face is turned round and the one on the far face is not.
        return new[]
        {
            Label(text, sign.transform, font, boardCentre, "Label (front)", 1f, Quaternion.Euler(0f, 180f, 0f)),
            Label(text, sign.transform, font, boardCentre, "Label (back)", -1f, Quaternion.identity),
        };
    }

    /// <summary>
    /// A primitive without its collider. The signs are labels, not obstacles: a collider off the <c>Obstacle</c> layer
    /// would be a wall the swarm cannot see, and one on it a thing to avoid that the experiment does not have.
    /// </summary>
    private static GameObject Prop(PrimitiveType type, string name, Transform parent)
    {
        GameObject prop = GameObject.CreatePrimitive(type);
        prop.name = name;
        Object.DestroyImmediate(prop.GetComponent<Collider>());
        prop.transform.SetParent(parent, false);
        return prop;
    }

    private static TextMeshPro Label(string text, Transform sign, TMP_FontAsset font, float height, string name,
                                     float face, Quaternion rotation)
    {
        GameObject go = new GameObject(name, typeof(RectTransform));
        go.transform.SetParent(sign, false);
        TextMeshPro label = go.AddComponent<TextMeshPro>();

        RectTransform rect = label.rectTransform;
        rect.localPosition = new Vector3(0f, height, face * (BoardThickness * 0.5f + LabelStandoff));
        rect.localRotation = rotation;
        rect.sizeDelta = new Vector2(BoardWidth - 2f * LabelMargin, BoardHeight - 2f * LabelMargin);

        label.font = font;
        label.text = text;
        label.fontStyle = FontStyles.Bold;
        label.color = Color.black;
        label.alignment = TextAlignmentOptions.Center;
        label.enableWordWrapping = false;
        label.enableAutoSizing = true;
        label.fontSizeMin = 0.1f;
        label.fontSizeMax = 100f;
        return label;
    }

    /// <summary>
    /// Letters every sign at one size, the largest at which the longest word still fits its board: each label is
    /// auto-sized alone, and the smallest result is kept for all. Returns that size.
    /// </summary>
    private static float MatchLabelSizes(List<TextMeshPro> labels)
    {
        float size = float.MaxValue;
        foreach (TextMeshPro label in labels)
        {
            label.ForceMeshUpdate();
            size = Mathf.Min(size, label.fontSize);
        }
        foreach (TextMeshPro label in labels)
        {
            label.enableAutoSizing = false;
            label.fontSize = size;
        }
        return size;
    }

    // Unlit, so a sign reads the same on its sunlit face and its shaded one, from every drone.
    private static Material LoadOrCreateSignMaterial()
    {
        Material material = AssetDatabase.LoadAssetAtPath<Material>(SignMaterialPath);
        if (material == null)
        {
            material = new Material(Shader.Find("Unlit/Color")) { color = BoardColour };
            AssetDatabase.CreateAsset(material, SignMaterialPath);
        }
        return material;
    }

    private static Vector3 OnGround(Vector3 point, float fallbackHeight)
    {
        point.y = TerrainHeightSampler.TryGetHeight(point, out float ground) ? ground : fallbackHeight;
        return point;
    }

    private static Vector3 Flat(Vector3 v)
    {
        v.y = 0f;
        return v;
    }
}
