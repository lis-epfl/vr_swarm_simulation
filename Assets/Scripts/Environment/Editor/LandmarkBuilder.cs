using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using Unity.Mathematics;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;
using UnityEngine.Rendering;
using UnityEngine.SceneManagement;
using UnityEngine.Splines;

/// <summary>
/// <c>Tools/Swarm/Landmarks</c>: the landmarks round VoronoiCityWorld (<see cref="LandmarkSet"/>), so a pilot who sees
/// the world only through the drones' feeds has something fixed to orient by — the Voronoi streets never run straight
/// and every building is 50 m. One per side: a snow-capped mountain range to the south, a lake in front of green hills
/// to the north, eight wind turbines on a ridge to the east, and a red-and-white TV mast with a blinking beacon to the
/// west, in line with ParkRoad behind the spawn, so "fly towards the mast" brings a pilot home.
///
/// <para><b>Everything is sized to what the feeds can show.</b> An FPV camera looks level (gimbal 0), sees ±23° vertically
/// (<see cref="FpvCameraFov"/>) and clips at 2000 m, and the drones fly below the 50 m roofs (ceiling 45 m, flown ~1.3 m
/// above). So a landmark's top has to stand at least ~8–10° up seen from the far side of the city, to clear the roofs,
/// and at most ~22° from the near side, or the frame cuts it off; and its skyline has to stay inside the far clip from
/// every corner of the city. That is why each feature lies on an arc round the centre, 1.25–1.45 km out and 300–400 m
/// high, with its relief falling away behind <see cref="BackStart"/>: laid along straight lines into the corner tiles,
/// ~25° of each side's skyline was past the far clip from the opposite corner and showed sliced. <see cref="Check"/>
/// measures all three after any edit. The mountain is to the south because the sun (Sun_02) stands in the north-west:
/// placed north, it would face the city with its unlit side.</para>
///
/// <para><b>The ground is a ring of twelve 1000 m Terrain tiles</b> round the city scenes' own four (±1000 m), out to
/// ±2000 m, each with its own TerrainData in <see cref="TerrainFolder"/>. The four city tiles are shared with five other
/// scenes and can never be sculpted, and the altitude ceiling is measured from Terrain (<see cref="TerrainHeightSampler"/>):
/// inside ±1000 m the city tiles still answer, so nothing changes over the city, and past it the ceiling follows this
/// ground. The heights come from one world-space function (<see cref="GroundHeight"/>), so neighbouring edges agree, and
/// rise from exactly 0 at the joint with a flat start, so there is no crack. The ring is grouping ID 1, so a Terrain
/// Tools brush never spills onto the shared tiles. The ring keeps its TerrainColliders, on Default like the city's
/// ground: the brushes need them, and a drone that flies 600 m out hits a hillside rather than passing through it.
/// Nothing else in the set has a collider.</para>
///
/// <para><b>Hand edits are never overwritten.</b> TerrainData, terrain layers and materials are made only if missing, so
/// sculpting and painting survive every rebuild; <i>Regenerate ground</i> is the one command that overwrites them, and
/// it asks. Everything lives in <see cref="PrefabPath"/>, which a scene gets an instance of (and VoronoiCityWorldBuilder
/// re-instantiates on a rebuild), so edits belong in the prefab or applied to it. The lake's shoreline is a spline
/// (<see cref="LakeOutline"/>), rebuilt live as its knots are dragged. The mast and the turbines take Asset Store models
/// through <see cref="LandmarkSet"/>'s slots and <see cref="PlaceModels"/>.</para>
///
/// <para><b>The lake is glossy Standard water with RingChallenge's wave normal map, not RingChallenge's AQUAS
/// material.</b> That material (AQUAS-Lite Frontface, on its WaterPlane prefab) fades with the camera depth texture,
/// which RingChallenge has because it renders realtime shadows; this scene has shadows off, so from a drone the water
/// rendered fully transparent, leaving only the shore. Giving it the depth texture means a depth pre-pass per FPV
/// render, the cost the feeds cannot take. <c>LakeWater.mat</c> reflects the sky, reads as water from the city's edge
/// and costs the same as no lake. AQUAS-Lite is git-ignored; without it the water is plain glossy. The lake stands on
/// the city's own tiles, which carry painted terrain trees right out to ±1000 m, so the tiles under it get this scene's
/// own copy with those trees cleared (<see cref="ClearTreesUnderLakes"/>); the other scenes keep the shared ones.</para>
///
/// <para>Batch: <c>-executeMethod LandmarkBuilder.Run</c> adds the set to VoronoiCityWorld, checks it, saves and exits
/// 0 / 1.</para>
/// </summary>
public static class LandmarkBuilder
{
    public const string Folder = "Assets/Landmarks";
    public const string PrefabPath = Folder + "/Landmarks.prefab";
    public const string GeneratedFolder = Folder + "/Generated";
    public const string TerrainFolder = Folder + "/Terrain";
    public const string RootName = "Landmarks";
    private const string AquasWavesPath = "Assets/AQUAS-Lite/Textures/Waves_HiRes.png";
    private const string GrassLayerPath = "Assets/BaseLayer.terrainlayer";
    private const string DronePrefabPath = "Assets/Prefabs/DroneReduced.prefab";
    private const string VoronoiScenePath = "Assets/Scenes/VoronoiCityWorld.unity";
    private const StaticEditorFlags AllStatic = (StaticEditorFlags)~0;

    // ---------------------------------------------------------------- the ring
    private const float TileSize = 1000f;
    /// <summary>The city scenes' own four tiles cover ±Inner.</summary>
    private const float Inner = 1000f;
    private const int HeightRes = 257, AlphaRes = 256, BaseMapRes = 512;
    /// <summary>TerrainData.size.y: room to sculpt well above the generated 400 m.</summary>
    private const float MaxHeight = 800f;
    /// <summary>The ground rises from exactly 0 at the joint over this distance, with a flat start.</summary>
    private const float ApronMetres = 200f;
    /// <summary>The relief falls away behind this radius from the city's centre (over <see cref="BackWidth"/>), so
    /// nothing a far corner of the city sees on the skyline is past the 2000 m far clip: the features lie along arcs
    /// round the centre, not along straight lines into the corner tiles, which is what that clip sliced.</summary>
    private const float BackStart = 1500f, BackWidth = 250f;
    /// <summary>The city tiles' y (ProceduralCityWorldBuilder's scenes keep DiamondCityWorld's).</summary>
    private const float GroundY = -0.017333984f;
    private const float RingPixelError = 8f;
    private const int Seed = 1;

    // ---------------------------------------------------------------- the features (world metres, +Z north)
    // Bearings are degrees clockwise from north (+Z), radii metres from the city's centre (the origin).
    /// <summary>The mountain range's peaks: bearing, radius, height, radial and tangential half-widths. The first is
    /// the main peak.</summary>
    private static readonly (float bearing, float radius, float height, float radial, float tangential)[] Peaks =
    {
        (184f, 1450f, 400f, 520f, 650f),
        (164f, 1420f, 300f, 380f, 420f),
        (203f, 1430f, 270f, 380f, 400f),
        (148f, 1400f, 220f, 340f, 380f),
        (218f, 1400f, 200f, 340f, 380f),
    };
    private const float MountainBearing = 180f, MountainSpan = 34f, MountainSpine = 160f, SnowLine = 240f;
    private const float HillsRadius = 1230f, HillsHeight = 210f, HillsSpan = 34f;
    private const float RidgeRadius = 1250f, RidgeHeight = 190f, RidgeSpan = 28f;
    private static readonly Vector2 MastSpot = new Vector2(-1250f, 0f);
    private const float MastHillHeight = 70f;
    /// <summary>The lake's centre and radii: its near shore 110–200 m beyond the city's edge (z 365), its far shore cut
    /// off along the hills' foot (<see cref="LakeNorthGap"/>), and ~1.4 km across.</summary>
    private static readonly Vector2 LakeCentre = new Vector2(-50f, 760f);
    private static readonly Vector2 LakeRadii = new Vector2(700f, 240f);
    private const int LakeKnots = 20;
    /// <summary>How far short of the hills' foot (the joint with the ring, z = <see cref="Inner"/>) the lake's north
    /// shore runs, at most; it wanders up to 12 m further back. The ground starts rising at the joint and passes the
    /// water's level ~12 m beyond it, so the water laps the foot of the slope.</summary>
    private const float LakeNorthGap = 12f;
    /// <summary>Bumped whenever the generated lake changes, so <see cref="AddTo"/> offers to replace a prefab lake made by
    /// an older generator (<see cref="LakeOutline.generatorVersion"/>). 2: twice as wide, out to the hills' foot.</summary>
    private const int LakeGeneratorVersion = 2;
    /// <summary>Likewise for the ring's ground (<see cref="LandmarkSet.groundVersion"/>). 2: the hills lowered to 210 m.</summary>
    private const int GroundGeneratorVersion = 2;
    /// <summary>How far beyond the shore the city tiles' trees are cleared (m).</summary>
    private const float TreeClearance = 6f;
    private const string CityCopyPrefix = "City_";
    private const int TurbineCount = 8;
    /// <summary>Degrees of bearing between turbines along the ridge's arc (~150 m at its radius).</summary>
    private const float TurbineSpacingDeg = 7f;
    /// <summary>The terrain tree the city's own tiles carry, so the mountain's woods match the city's.</summary>
    private const string TreePrefabPath = "Assets/Tree9/Tree9_3.prefab";
    /// <summary>The mountain's trees are scaled as the city tiles' terrain trees are (0.9–1.1 of the prefab), so the
    /// woods on the mountain match the woods round the city.</summary>
    private const float TreeScaleMin = 0.9f, TreeScaleMax = 1.1f;
    /// <summary>About the city tiles' own density (~100 trees a hectare in their woods).</summary>
    private const float TreeSpacing = 10f;

    // ---------------------------------------------------------------- the generated models
    public const float MastHeight = 300f;
    private const float MastBaseRadius = 3f, MastTopRadius = 1.5f, MastAntenna = 18f;
    private const int MastBands = 7;   // aviation marking: an odd number of bands, red at both ends
    public const float HubHeight = 100f, RotorRadius = 50f;
    public const float TurbineTip = HubHeight + RotorRadius;
    private const float BeaconRadius = 2.2f;

    // ---------------------------------------------------------------- Check
    private const float MinElevationDeg = 8f;
    /// <summary>The height PD holds each drone this far above its setpoint (CLAUDE.md), so the flown ceiling.</summary>
    private const float FlownAboveCeiling = 1.3f;
    private const float DefaultEyeHeight = 46.3f;
    private const float DefaultFarClip = 2000f;
    private const float CheckStep = 25f;

    // ================================================================ menu and batch

    [MenuItem("Tools/Swarm/Landmarks/Add to open scene")]
    private static void AddFromMenu()
    {
        bool ok = AddTo(SceneManager.GetActiveScene(), out string summary);
        if (ok) Debug.Log("LandmarkBuilder: " + summary);
        else Debug.LogError("LandmarkBuilder: " + summary);
    }

    [MenuItem("Tools/Swarm/Landmarks/Place models")]
    private static void PlaceFromMenu()
    {
        LandmarkSet set = FindSet();
        if (set == null) Debug.LogError("LandmarkBuilder: no LandmarkSet selected or in the open scene.");
        else PlaceModelsAndSave(set);
    }

    [MenuItem("Tools/Swarm/Landmarks/Check")]
    private static void CheckFromMenu()
    {
        LandmarkSet set = FindSet();
        if (set == null) Debug.LogError("LandmarkBuilder: no LandmarkSet selected or in the open scene.");
        else CheckAndLog(set);
    }

    [MenuItem("Tools/Swarm/Landmarks/Regenerate ground (overwrites sculpting)…")]
    private static void RegenerateFromMenu()
    {
        LandmarkSet set = FindSet();
        if (set == null || set.ground == null)
        {
            Debug.LogError("LandmarkBuilder: no LandmarkSet with a ground selected or in the open scene.");
            return;
        }
        if (!EditorUtility.DisplayDialog("Regenerate landmark ground",
                "Overwrite the heights, paint and trees of all the ring's Terrain tiles with the generated ones? Every " +
                "bit of sculpting and painting on them is lost.", "Overwrite", "Cancel"))
        {
            return;
        }
        Debug.Log($"LandmarkBuilder: {RegenerateGround(set)}");
    }

    /// <summary>Overwrites every ring tile's heights, paint and trees with the generated ones, with no question asked
    /// (the menu asks). Also brings the generated terrain layers up to date.</summary>
    public static string RegenerateGround(LandmarkSet set)
    {
        if (set == null || set.ground == null) return "no ground to regenerate.";
        TerrainLayer[] layers = EnsureTerrainLayers();
        int n = 0, trees = 0;
        foreach (Terrain t in set.ground.GetComponentsInChildren<Terrain>(true))
        {
            Vector3 p = t.transform.position;
            t.terrainData.terrainLayers = layers;
            trees += Generate(t.terrainData, new Vector2(p.x, p.z));
            EditorUtility.SetDirty(t.terrainData);
            if (!t.drawTreesAndFoliage)
            {
                t.drawTreesAndFoliage = true;
                EditorUtility.SetDirty(t);
            }
            n++;
        }
        set.groundVersion = GroundGeneratorVersion;
        EditorUtility.SetDirty(set);
        AssetDatabase.SaveAssets();
        return $"regenerated {n} ground tile(s), {trees} trees.";
    }

    [MenuItem("Tools/Swarm/Landmarks/Add to open scene", true)]
    [MenuItem("Tools/Swarm/Landmarks/Place models", true)]
    [MenuItem("Tools/Swarm/Landmarks/Check", true)]
    [MenuItem("Tools/Swarm/Landmarks/Regenerate ground (overwrites sculpting)…", true)]
    private static bool CanEdit() => !EditorApplication.isPlayingOrWillChangePlaymode;

    /// <summary>Batchmode: adds the set to VoronoiCityWorld (making the prefab if missing), checks it, saves, exits 0 / 1.</summary>
    public static void Run()
    {
        bool ok = false;
        try
        {
            Scene scene = EditorSceneManager.OpenScene(VoronoiScenePath, OpenSceneMode.Single);
            ok = AddTo(scene, out string summary);
            Debug.Log("LandmarkBuilder: " + summary);
            LandmarkSet set = FindSetIn(scene);
            if (ok && set != null) Debug.Log(Check(set, out _));
            if (ok && !EditorSceneManager.SaveScene(scene))
            {
                Debug.LogError($"LandmarkBuilder: could not save {VoronoiScenePath}.");
                ok = false;
            }
        }
        catch (System.Exception e)
        {
            Debug.LogException(e);
            ok = false;
        }
        EditorApplication.Exit(ok ? 0 : 1);
    }

    private static LandmarkSet FindSet()
    {
        if (Selection.activeGameObject != null)
        {
            LandmarkSet s = Selection.activeGameObject.GetComponentInParent<LandmarkSet>();
            if (s != null) return s;
        }
        PrefabStage stage = PrefabStageUtility.GetCurrentPrefabStage();
        if (stage != null)
        {
            LandmarkSet s = stage.prefabContentsRoot.GetComponentInChildren<LandmarkSet>(true);
            if (s != null) return s;
        }
        return FindSetIn(SceneManager.GetActiveScene());
    }

    private static LandmarkSet FindSetIn(Scene scene)
    {
        foreach (GameObject root in scene.GetRootGameObjects())
        {
            LandmarkSet s = root.GetComponentInChildren<LandmarkSet>(true);
            if (s != null) return s;
        }
        return null;
    }

    // ================================================================ adding the set to a scene

    /// <summary>Puts an instance of <see cref="PrefabPath"/> in <paramref name="scene"/>, making the prefab first if it is
    /// missing (and only then the ground, layers and materials that are missing too), then clears the city tiles' trees
    /// from under the lake (<see cref="ClearTreesUnderLakes"/>). A scene that already has a <see cref="LandmarkSet"/> keeps
    /// it and only has its clearing redone. VoronoiCityWorldBuilder.Configure calls this, so a rebuild keeps them. First
    /// the prefab is offered the generator's newer lake and ground, if it has older ones (<see cref="UpgradePrefab"/>).</summary>
    public static bool AddTo(Scene scene, out string summary)
    {
        string upgraded = UpgradePrefab();
        bool ok = Place(scene, out summary);
        summary += upgraded;
        if (!ok) return false;
        LandmarkSet set = FindSetIn(scene);
        if (set != null)
        {
            foreach (LakeOutline lake in set.GetComponentsInChildren<LakeOutline>(true)) RebuildLake(lake);
        }
        int cleared = ClearTreesUnderLakes(scene);
        AssetDatabase.SaveAssets();
        summary += $" Rebuilt the lake, painted its beaches and cleared {cleared} of the city tiles' trees from round it.";
        return true;
    }

    /// <summary>
    /// Brings the prefab's lake and ground up to the generator, which they otherwise never see: the prefab is the
    /// record, so a scene that deletes the set and adds it back gets the prefab's old lake again, and the ground is only
    /// ever generated when its tiles are missing. A lake made by an older generator
    /// (<see cref="LakeOutline.generatorVersion"/> below <see cref="LakeGeneratorVersion"/>) is replaced by a new one, and
    /// older ground (<see cref="LandmarkSet.groundVersion"/>) regenerated, each only after the editor asks, since both
    /// discard hand edits; the ground goes first, because the lake's beaches read it. Declining stamps the current
    /// version, so the question is not asked again. Batchmode only reports. Returns a note for the summary.
    /// </summary>
    private static string UpgradePrefab()
    {
        var prefab = AssetDatabase.LoadAssetAtPath<GameObject>(PrefabPath);
        LandmarkSet asset = prefab != null ? prefab.GetComponentInChildren<LandmarkSet>(true) : null;
        if (asset == null) return "";
        bool oldGround = asset.ground != null && asset.groundVersion < GroundGeneratorVersion;
        bool oldLake = asset.GetComponentsInChildren<LakeOutline>(true).Any(l => l.generatorVersion < LakeGeneratorVersion);
        if (!oldGround && !oldLake) return "";
        if (Application.isBatchMode)
        {
            return (oldGround ? " The prefab's ground predates the generator's (run Regenerate ground)." : "") +
                   (oldLake ? " The prefab's lake predates the generator's (delete it and run Add to open scene)." : "");
        }

        bool regenerate = oldGround && EditorUtility.DisplayDialog("Landmarks: newer ground",
            "The generated ground has changed since the prefab's was made (the hills are lower). Regenerate the ring's " +
            "Terrain tiles? Every bit of sculpting and painting on them is lost.", "Regenerate", "Keep mine");
        bool replace = oldLake && EditorUtility.DisplayDialog("Landmarks: newer lake",
            "The generated lake has changed since the prefab's was made (it is twice as wide and runs to the hills' " +
            "foot). Replace the prefab's lake with a new one? Its knots and settings are lost.", "Replace", "Keep mine");

        string note = "";
        GameObject contents = PrefabUtility.LoadPrefabContents(PrefabPath);
        try
        {
            LandmarkSet set = contents.GetComponentInChildren<LandmarkSet>(true);
            if (oldGround)
            {
                if (regenerate) note += " Prefab ground: " + RegenerateGround(set);
                set.groundVersion = GroundGeneratorVersion;
            }
            foreach (LakeOutline lake in set.GetComponentsInChildren<LakeOutline>(true))
            {
                if (lake.generatorVersion >= LakeGeneratorVersion) continue;
                if (replace) Object.DestroyImmediate(lake.gameObject);
                else lake.generatorVersion = LakeGeneratorVersion;
            }
            if (replace)
            {
                BuildLake(set.transform, EnsureMaterials());
                note += " The prefab's lake was replaced by the generator's.";
            }
            PrefabUtility.SaveAsPrefabAsset(contents, PrefabPath);
        }
        finally
        {
            PrefabUtility.UnloadPrefabContents(contents);
        }
        return note;
    }

    private static bool Place(Scene scene, out string summary)
    {
        LandmarkSet existing = FindSetIn(scene);
        if (existing != null)
        {
            summary = $"{scene.path} already has landmarks ({existing.name})." + RestoreLake(existing);
            foreach (LakeOutline lake in existing.GetComponentsInChildren<LakeOutline>(true))
            {
                summary += RescueSunkenLake(lake);
            }
            return true;
        }
        var prefab = AssetDatabase.LoadAssetAtPath<GameObject>(PrefabPath);
        if (prefab != null)
        {
            var instance = (GameObject)PrefabUtility.InstantiatePrefab(prefab, scene);
            instance.name = RootName;
            EditorSceneManager.MarkSceneDirty(scene);
            summary = $"added {PrefabPath} to {scene.path}.";
            return true;
        }

        // Built in the scene itself and saved with SaveAsPrefabAssetAndConnect, so the built object becomes the instance.
        Scene previous = SceneManager.GetActiveScene();
        if (scene != previous) SceneManager.SetActiveScene(scene);
        try
        {
            GameObject root = BuildSet(out summary);
            if (root == null) return false;
            PrefabUtility.SaveAsPrefabAssetAndConnect(root, PrefabPath, InteractionMode.AutomatedAction, out bool ok);
            if (!ok)
            {
                Object.DestroyImmediate(root);
                summary = $"could not save {PrefabPath}.";
                return false;
            }
            AssetDatabase.SaveAssets();
            EditorSceneManager.MarkSceneDirty(scene);
            summary = $"made {PrefabPath} ({summary}) and added it to {scene.path}.";
            return true;
        }
        finally
        {
            if (scene != previous && previous.IsValid()) SceneManager.SetActiveScene(previous);
        }
    }

    /// <summary>The whole set, built in the active scene: ground, lake, wind farm and mast. Null with the reason.</summary>
    private static GameObject BuildSet(out string summary)
    {
        EnsureFolder(Folder);
        EnsureFolder(GeneratedFolder);
        EnsureFolder(TerrainFolder);
        Materials mats = EnsureMaterials();

        var root = new GameObject(RootName);
        LandmarkSet set = root.AddComponent<LandmarkSet>();

        Transform ground = Child(root.transform, "Ground");
        ground.position = new Vector3(0f, GroundY, 0f);
        TerrainLayer[] layers = EnsureTerrainLayers();
        var terrainTemplate = AssetDatabase.GetBuiltinExtraResource<Material>("Default-Terrain-Standard.mat");
        int made = 0, kept = 0;
        foreach ((string side, Vector2 origin) in RingTiles())
        {
            string name = $"Ground_{side}_{Coord(origin.x)}_{Coord(origin.y)}";
            TerrainData data = EnsureTerrainData(name, origin, layers, out bool created);
            if (created) made++;
            else kept++;
            MakeTerrain(data, ground, $"Ground {side} ({origin.x:0}, {origin.y:0})", origin, terrainTemplate);
        }
        set.ground = ground;
        if (kept == 0) set.groundVersion = GroundGeneratorVersion;   // kept tiles are whatever generated them

        BuildLake(root.transform, mats);

        Transform farm = Child(root.transform, "Wind farm (E)");
        for (int i = 0; i < TurbineCount; i++)
        {
            float b = (90f + (i - (TurbineCount - 1) * 0.5f) * TurbineSpacingDeg) * Mathf.Deg2Rad;
            Transform spot = Child(farm, $"Turbine {i + 1}");
            spot.position = new Vector3(Mathf.Sin(b) * RidgeRadius, 0f, Mathf.Cos(b) * RidgeRadius);
            // The rotor (+Z) faces the city, so the discs are seen face on.
            Vector3 toCity = new Vector3(set.cityCentre.x, 0f, set.cityCentre.y) - spot.position;
            spot.rotation = Quaternion.LookRotation(new Vector3(toCity.x, 0f, toCity.z).normalized, Vector3.up);
        }
        set.windFarm = farm;

        Transform mast = Child(root.transform, "TV mast (W)");
        mast.position = new Vector3(MastSpot.x, 0f, MastSpot.y);
        set.mast = mast;

        string placed = PlaceModels(set, mats);
        summary = $"ground: {made} tile(s) generated, {kept} kept; {placed}";
        return root;
    }

    private static IEnumerable<(string side, Vector2 origin)> RingTiles()
    {
        for (int ix = -2; ix <= 1; ix++)
        {
            for (int iz = -2; iz <= 1; iz++)
            {
                bool cityX = ix == -1 || ix == 0, cityZ = iz == -1 || iz == 0;
                if (cityX && cityZ) continue;
                string ns = iz == 1 ? "N" : iz == -2 ? "S" : "";
                string ew = ix == 1 ? "E" : ix == -2 ? "W" : "";
                yield return (ns + ew, new Vector2(ix * TileSize, iz * TileSize));
            }
        }
    }

    private static string Coord(float v) => v < 0 ? "m" + (-v).ToString("0") : v.ToString("0");

    private static Transform Child(Transform parent, string name)
    {
        var go = new GameObject(name);
        go.transform.SetParent(parent, false);
        return go.transform;
    }

    // ================================================================ the ground

    /// <summary>The generated ground height (m above the city's ground) at world (x, z): zero over the city tiles,
    /// rising from a flat start at the joint, with the four features on arcs round the centre, each on its side, and
    /// falling away behind <see cref="BackStart"/>.</summary>
    public static float GroundHeight(float x, float z)
    {
        float d = Mathf.Max(Mathf.Abs(x), Mathf.Abs(z)) - Inner;
        if (d <= 0f) return 0f;
        float r = Mathf.Sqrt(x * x + z * z);
        float bearing = Mathf.Atan2(x, z) * Mathf.Rad2Deg;
        float relief = Mathf.Max(Mathf.Max(Mountains(x, z, r, bearing), Hills(x, z, r, bearing)),
                                 Mathf.Max(Ridge(x, z, r, bearing), MastHill(x, z)));
        relief *= 1f - Smooth01((r - BackStart) / BackWidth);
        float rolling = 18f * Fbm(x, z, 500f, 3);
        return Smooth01(d / ApronMetres) * (rolling + relief);
    }

    /// <summary>South: one main peak, four lower ones and a spine along the arc between them, craggy higher up.</summary>
    private static float Mountains(float x, float z, float r, float bearing)
    {
        float off = Mathf.Abs(Mathf.DeltaAngle(MountainBearing, bearing));
        if (off > 70f) return 0f;
        float h = 0f;
        foreach ((float b, float pr, float ph, float radial, float tangential) in Peaks)
        {
            h = Mathf.Max(h, Peak(x, z, b, pr, ph, radial, tangential));
        }
        float spine = MountainSpine * Mathf.Exp(-Sq((r - 1440f) / 200f)) * (1f - Smooth01((off - MountainSpan) / 10f));
        h = Mathf.Max(h, spine);
        if (h <= 0f) return 0f;
        return h * (0.7f + 0.4f * Ridged(x, z, 300f, 5)) + 16f * (Fbm(x, z, 70f, 3) - 0.5f) * Smooth01(h / 80f);
    }

    /// <summary>North: a band of rounded hills behind the lake.</summary>
    private static float Hills(float x, float z, float r, float bearing)
    {
        float off = Mathf.Abs(Mathf.DeltaAngle(0f, bearing));
        if (off > HillsSpan + 15f) return 0f;
        float band = Mathf.Exp(-Sq((r - HillsRadius) / 340f));
        float window = 1f - Smooth01((off - HillsSpan) / 12f);
        float rolls = 0.45f + 0.55f * Smooth01((Fbm(x, z, 650f, 3) - 0.3f) / 0.4f);
        return HillsHeight * band * window * rolls;
    }

    /// <summary>East: a long ridge for the wind farm.</summary>
    private static float Ridge(float x, float z, float r, float bearing)
    {
        float off = Mathf.Abs(Mathf.DeltaAngle(90f, bearing));
        if (off > RidgeSpan + 12f) return 0f;
        float across = Mathf.Exp(-Sq((r - RidgeRadius) / 260f));
        float along = 1f - Smooth01((off - RidgeSpan) / 10f);
        return RidgeHeight * across * along * (0.85f + 0.3f * (Fbm(x, z, 400f, 2) - 0.5f));
    }

    /// <summary>West: a low hill for the mast to stand on.</summary>
    private static float MastHill(float x, float z)
    {
        float dx = (x - MastSpot.x) / 220f, dz = (z - MastSpot.y) / 220f;
        return MastHillHeight * Mathf.Exp(-1.6f * (dx * dx + dz * dz));
    }

    /// <summary>A peak of height <paramref name="h"/> at <paramref name="bearing"/> and <paramref name="radius"/> from the
    /// centre, with elliptical half-widths along and across that bearing: concave at the foot, steepest near the top.</summary>
    private static float Peak(float x, float z, float bearing, float radius, float h, float radial, float tangential)
    {
        float b = bearing * Mathf.Deg2Rad;
        var along = new Vector2(Mathf.Sin(b), Mathf.Cos(b));
        var across = new Vector2(along.y, -along.x);
        Vector2 q = new Vector2(x, z) - along * radius;
        float u = Vector2.Dot(q, along) / radial, v = Vector2.Dot(q, across) / tangential;
        float rr = Mathf.Sqrt(u * u + v * v);
        return rr >= 1f ? 0f : h * Mathf.Pow(1f - rr, 1.4f);
    }

    /// <summary>Grass, forest, rock and snow weights (the order of <see cref="EnsureTerrainLayers"/>).</summary>
    private static Vector4 Paint(float x, float z, float h, float slopeDeg)
    {
        // Rock and snow are the mountain's alone. The hills and the ridge have to rise ~300 m within ~350 m of the joint
        // to clear the roofs from across the city and stay inside the far clip, so their faces are ~40° whatever is
        // done: painted by slope they came out rock, and the hills' crests took snow. Their steep faces are wooded.
        float south = Smooth01((-z - Inner) / 100f);
        float snowLine = SnowLine + 40f * (Fbm(x, z, 180f, 2) - 0.5f);
        float snow = south * Smooth01((h - snowLine + 15f) / 40f) * (1f - Smooth01((slopeDeg - 42f) / 14f));
        // The mountain's woods stay grass under their trees, as the city's do; the forest layer is the hills' and ridge's.
        float rock = south * MountainCover(x, z, h, slopeDeg).rock * (1f - snow);
        float wooded = Mathf.Max(Smooth01((z - Inner) / 100f), Smooth01((x - Inner) / 100f)) * (1f - south);
        float mottle = Mathf.Max(Smooth01((Fbm(x, z, 90f, 3) - 0.38f) / 0.12f), Smooth01((slopeDeg - 28f) / 12f));
        float forest = wooded * Smooth01((h - 25f) / 35f) * mottle;
        forest *= (1f - rock) * (1f - snow);
        float grass = Mathf.Max(0f, 1f - snow - rock - forest);
        var w = new Vector4(grass, forest, rock, snow);
        float sum = w.x + w.y + w.z + w.w;
        return sum > 0f ? w / sum : new Vector4(1f, 0f, 0f, 0f);
    }

    /// <summary>The mountain side's cover at a point: how much is bare rock and how much is wood; the rest is grass.
    /// Rock starts at a height that wanders between ~50 and ~250 m and on slopes from ~20° to ~40°, and meadows break
    /// it up to ~300 m, so the face towards the city is not one sheet of tiled rock. The woods are patches of grass on
    /// the lower, gentler slopes, and <see cref="MountainTrees"/> plants their trees.</summary>
    private static (float rock, float wood) MountainCover(float x, float z, float h, float slopeDeg)
    {
        float rockLine = 150f + 300f * (Fbm(x, z, 260f, 3) - 0.5f);
        float steepStart = 30f + 40f * (Fbm(x + 500f, z, 140f, 2) - 0.5f);
        float rock = Mathf.Max(Smooth01((h - rockLine) / 80f), Smooth01((slopeDeg - steepStart) / 12f));
        float meadow = Smooth01((Fbm(x, z + 700f, 110f, 3) - 0.5f) / 0.08f)
                       * (1f - Smooth01((h - 290f) / 40f)) * (1f - Smooth01((slopeDeg - 46f) / 8f));
        rock *= 1f - meadow;
        float woodLine = 170f + 160f * (Fbm(x - 300f, z, 200f, 2) - 0.5f);
        float wood = Smooth01((woodLine - h) / 40f) * Smooth01((Fbm(x, z - 900f, 140f, 3) - 0.46f) / 0.08f)
                     * (1f - Smooth01((slopeDeg - 40f) / 8f)) * Smooth01((h - 15f) / 25f);
        return (rock * (1f - wood), wood);
    }

    /// <summary>The slope (degrees) of the generated ground at world (x, z).</summary>
    private static float Slope(float x, float z)
    {
        const float e = 4f;
        float gx = (GroundHeight(x + e, z) - GroundHeight(x - e, z)) / (2f * e);
        float gz = (GroundHeight(x, z + e) - GroundHeight(x, z - e)) / (2f * e);
        return Mathf.Atan(Mathf.Sqrt(gx * gx + gz * gz)) * Mathf.Rad2Deg;
    }

    /// <summary>The tile's TerrainData, made and generated only if missing, so sculpting and painting survive.</summary>
    private static TerrainData EnsureTerrainData(string name, Vector2 origin, TerrainLayer[] layers, out bool created)
    {
        string path = $"{TerrainFolder}/{name}.asset";
        var data = AssetDatabase.LoadAssetAtPath<TerrainData>(path);
        created = data == null;
        if (!created) return data;
        data = new TerrainData
        {
            name = name,
            heightmapResolution = HeightRes,
            alphamapResolution = AlphaRes,
            baseMapResolution = BaseMapRes,
        };
        data.SetDetailResolution(32, 16);
        data.size = new Vector3(TileSize, MaxHeight, TileSize);
        data.terrainLayers = layers;
        // An asset first: SetAlphamaps keeps its splat textures only as sub-assets of an existing asset, so paint
        // written before CreateAsset is lost on reload (the ground came back all grass).
        AssetDatabase.CreateAsset(data, path);
        Generate(data, origin);
        EditorUtility.SetDirty(data);
        return data;
    }

    /// <summary>Writes the generated heights, paint and trees into <paramref name="data"/>, a tile with its corner at
    /// world (x, z) = <paramref name="origin"/>. Returns how many trees it planted.</summary>
    private static int Generate(TerrainData data, Vector2 origin)
    {
        int n = data.heightmapResolution;
        float size = data.size.x, top = data.size.y;
        float step = size / (n - 1);
        var heights = new float[n, n];
        for (int j = 0; j < n; j++)
        {
            for (int i = 0; i < n; i++)
            {
                heights[j, i] = Mathf.Clamp01(GroundHeight(origin.x + i * step, origin.y + j * step) / top);
            }
        }
        data.SetHeights(0, 0, heights);

        int a = data.alphamapResolution;
        if (data.terrainLayers.Length >= 4)
        {
            var alpha = new float[a, a, data.terrainLayers.Length];
            float texel = size / a;
            for (int j = 0; j < a; j++)
            {
                for (int i = 0; i < a; i++)
                {
                    float x = origin.x + (i + 0.5f) * texel, z = origin.y + (j + 0.5f) * texel;
                    Vector4 w = Paint(x, z, GroundHeight(x, z), Slope(x, z));
                    for (int k = 0; k < 4; k++) alpha[j, i, k] = w[k];
                }
            }
            data.SetAlphamaps(0, 0, alpha);
        }

        var tree = AssetDatabase.LoadAssetAtPath<GameObject>(TreePrefabPath);
        if (tree == null)
        {
            data.SetTreeInstances(new TreeInstance[0], false);
            return 0;
        }
        data.treePrototypes = new[] { new TreePrototype { prefab = tree } };
        data.RefreshPrototypes();
        List<TreeInstance> trees = MountainTrees(origin, size);
        data.SetTreeInstances(trees.ToArray(), true);
        return trees.Count;
    }

    /// <summary>The mountain's trees in one tile, in its normalised coordinates: on a jittered grid as dense as the city
    /// tiles' woods, each spot kept with the probability of wood there (<see cref="MountainCover"/>), and only where the
    /// ground is painted grass or forest — never on rock or snow. Sized as the city's own terrain trees are.</summary>
    private static List<TreeInstance> MountainTrees(Vector2 origin, float size)
    {
        var trees = new List<TreeInstance>();
        var rng = new System.Random(Seed * 7919 + (int)origin.x * 31 + (int)origin.y);
        for (float zz = TreeSpacing * 0.5f; zz < size; zz += TreeSpacing)
        {
            for (float xx = TreeSpacing * 0.5f; xx < size; xx += TreeSpacing)
            {
                // Every spot draws all its numbers, so changing the cover moves trees in and out, not the whole wood.
                float x = origin.x + xx + ((float)rng.NextDouble() - 0.5f) * TreeSpacing;
                float z = origin.y + zz + ((float)rng.NextDouble() - 0.5f) * TreeSpacing;
                float keep = (float)rng.NextDouble(), size01 = (float)rng.NextDouble();
                float width01 = (float)rng.NextDouble(), turn = (float)rng.NextDouble();
                float south = Smooth01((-z - Inner) / 100f);
                if (south <= 0f) continue;
                float h = GroundHeight(x, z);
                if (h < 10f) continue;
                float slope = Slope(x, z);
                float wood = south * MountainCover(x, z, h, slope).wood;
                if (keep >= 0.95f * Smooth01((wood - 0.2f) / 0.25f)) continue;   // a patch is a full wood, as the city's are
                Vector4 ground = Paint(x, z, h, slope);
                if (ground.z + ground.w > 0.05f) continue;   // rock or snow under it
                float s = Mathf.Lerp(TreeScaleMin, TreeScaleMax, size01);
                trees.Add(new TreeInstance
                {
                    position = new Vector3(Mathf.Clamp01((x - origin.x) / size), 0f, Mathf.Clamp01((z - origin.y) / size)),
                    prototypeIndex = 0,
                    heightScale = s,
                    widthScale = s * (0.85f + 0.3f * width01),
                    rotation = turn * 2f * Mathf.PI,
                    color = Color.white,
                    lightmapColor = Color.white,
                });
            }
        }
        return trees;
    }

    private static Terrain MakeTerrain(TerrainData data, Transform parent, string name, Vector2 origin, Material template)
    {
        var go = new GameObject(name);
        go.transform.SetParent(parent, false);
        go.transform.position = new Vector3(origin.x, GroundY, origin.y);
        var t = go.AddComponent<Terrain>();
        t.terrainData = data;
        go.AddComponent<TerrainCollider>().terrainData = data;
        t.groupingID = 1;
        t.allowAutoConnect = true;
        t.heightmapPixelError = RingPixelError;
        t.basemapDistance = 1000f;
        t.shadowCastingMode = ShadowCastingMode.Off;
        t.drawTreesAndFoliage = true;
        if (template != null) t.materialTemplate = template;
        GameObjectUtility.SetStaticEditorFlags(go, AllStatic);
        return t;
    }

    /// <summary>How a generated terrain layer looks: its colour, how much it is mottled, its tile size, whether it gets
    /// a normal map, and the version of its generated texture.</summary>
    private struct LayerLook
    {
        public string name;
        public Color colour;
        public float variation, tile;
        public int version;
        public bool normal, strata;
    }

    /// <summary>Grass (the city's own layer, so the joint matches), forest, rock and snow. A layer is made if missing; one
    /// still on a texture this builder generated is brought up to the current version (version 2 is a larger, less
    /// repetitive texture, and the rock's has a normal map and strata); one given another texture by hand is left
    /// alone.</summary>
    private static TerrainLayer[] EnsureTerrainLayers()
    {
        TerrainLayer grass = AssetDatabase.LoadAssetAtPath<TerrainLayer>(GrassLayerPath)
                             ?? EnsureLayer(new LayerLook
                             {
                                 name = "Grass", colour = new Color(0.36f, 0.45f, 0.22f), variation = 0.25f, tile = 7f,
                                 version = 1,
                             });
        return new[]
        {
            grass,
            EnsureLayer(new LayerLook
            {
                name = "Forest", colour = new Color(0.17f, 0.27f, 0.13f), variation = 0.4f, tile = 40f, version = 2,
            }),
            EnsureLayer(new LayerLook
            {
                name = "Rock", colour = new Color(0.45f, 0.43f, 0.40f), variation = 0.35f, tile = 70f, version = 2,
                normal = true, strata = true,
            }),
            EnsureLayer(new LayerLook
            {
                name = "Snow", colour = new Color(0.93f, 0.94f, 0.97f), variation = 0.06f, tile = 30f, version = 1,
            }),
        };
    }

    private static TerrainLayer EnsureLayer(LayerLook look)
    {
        string path = $"{TerrainFolder}/{look.name}.terrainlayer";
        string texPath = look.version > 1 ? $"{TerrainFolder}/{look.name}_v{look.version}.png" : $"{TerrainFolder}/{look.name}.png";
        var layer = AssetDatabase.LoadAssetAtPath<TerrainLayer>(path);
        string current = layer != null && layer.diffuseTexture != null ? AssetDatabase.GetAssetPath(layer.diffuseTexture) : "";
        if (layer != null && current == texPath) return layer;                                    // up to date
        if (layer != null && layer.diffuseTexture != null && !IsGenerated(current, look.name)) return layer;   // by hand
        Texture2D diffuse = EnsureNoiseTexture(texPath, look, out Texture2D normal);
        if (layer == null)
        {
            layer = new TerrainLayer { name = look.name };
            AssetDatabase.CreateAsset(layer, path);
        }
        layer.diffuseTexture = diffuse;
        layer.normalMapTexture = normal;
        layer.tileSize = new Vector2(look.tile, look.tile);
        EditorUtility.SetDirty(layer);
        if (!string.IsNullOrEmpty(current) && current != texPath)
        {
            AssetDatabase.DeleteAsset(current);   // the older generated texture, which nothing else uses
        }
        return layer;
    }

    /// <summary>Whether <paramref name="path"/> is a texture this builder generated for the layer
    /// <paramref name="name"/> (any version).</summary>
    private static bool IsGenerated(string path, string name) =>
        path == $"{TerrainFolder}/{name}.png" ||
        (path.StartsWith($"{TerrainFolder}/{name}_v") && path.EndsWith(".png") && !path.EndsWith("_n.png"));

    /// <summary>A tileable texture: the layer's colour mottled by five octaves of periodic value noise (version 1: two),
    /// with strata if asked, and a matching normal map if asked (else null).</summary>
    private static Texture2D EnsureNoiseTexture(string path, LayerLook look, out Texture2D normal)
    {
        string normalPath = path.Replace(".png", "_n.png");
        normal = look.normal ? AssetDatabase.LoadAssetAtPath<Texture2D>(normalPath) : null;
        var existing = AssetDatabase.LoadAssetAtPath<Texture2D>(path);
        if (existing != null && (!look.normal || normal != null)) return existing;

        int n = look.version > 1 ? 256 : 128;
        var rng = new System.Random(look.name.Aggregate(17, (h, ch) => h * 31 + ch) ^ look.version);
        int[] cells = look.version > 1 ? new[] { 4, 8, 16, 32, 64 } : new[] { 8, 32 };
        float[] amps = look.version > 1 ? new[] { 0.32f, 0.25f, 0.18f, 0.14f, 0.11f } : new[] { 0.65f, 0.35f };
        float[][,] grids = cells.Select(c => RandomGrid(rng, c)).ToArray();
        var field = new float[n, n];
        for (int y = 0; y < n; y++)
        {
            for (int x = 0; x < n; x++)
            {
                float v = 0f;
                for (int o = 0; o < grids.Length; o++) v += amps[o] * Periodic(grids[o], x, y, n);
                if (look.strata)
                {
                    // Bedding planes: a few warped horizontal bands, periodic over the tile so it still tiles.
                    float warp = Periodic(grids[0], x, y, n);
                    v += 0.12f * Mathf.Sin(2f * Mathf.PI * (6f * y / n + 0.8f * warp));
                }
                field[y, x] = v;
            }
        }

        var tex = new Texture2D(n, n, TextureFormat.RGB24, false);
        Color dark = new Color(look.colour.r * 0.78f, look.colour.g * 0.76f, look.colour.b * 0.72f);
        for (int y = 0; y < n; y++)
        {
            for (int x = 0; x < n; x++)
            {
                float k = 1f + look.variation * (2f * field[y, x] - 1f);
                Color c = look.version > 1 ? Color.Lerp(dark, look.colour, Periodic(grids[0], x + n / 3, y, n)) : look.colour;
                tex.SetPixel(x, y, new Color(c.r * k, c.g * k, c.b * k));
            }
        }
        WriteTexture(tex, path, false);

        if (look.normal)
        {
            var nm = new Texture2D(n, n, TextureFormat.RGB24, false);
            const float strength = 6f;
            for (int y = 0; y < n; y++)
            {
                for (int x = 0; x < n; x++)
                {
                    float dx = field[y, (x + 1) % n] - field[y, (x + n - 1) % n];
                    float dy = field[(y + 1) % n, x] - field[(y + n - 1) % n, x];
                    Vector3 v = new Vector3(-dx * strength, -dy * strength, 1f).normalized;
                    nm.SetPixel(x, y, new Color(0.5f + 0.5f * v.x, 0.5f + 0.5f * v.y, 0.5f + 0.5f * v.z));
                }
            }
            WriteTexture(nm, normalPath, true);
            normal = AssetDatabase.LoadAssetAtPath<Texture2D>(normalPath);
        }
        return AssetDatabase.LoadAssetAtPath<Texture2D>(path);
    }

    private static void WriteTexture(Texture2D tex, string path, bool isNormal)
    {
        File.WriteAllBytes(path, tex.EncodeToPNG());
        Object.DestroyImmediate(tex);
        AssetDatabase.ImportAsset(path);
        var importer = (TextureImporter)AssetImporter.GetAtPath(path);
        importer.wrapMode = TextureWrapMode.Repeat;
        importer.mipmapEnabled = true;
        if (isNormal) importer.textureType = TextureImporterType.NormalMap;
        else importer.sRGBTexture = true;
        importer.SaveAndReimport();
    }

    private static float[,] RandomGrid(System.Random rng, int cells)
    {
        var g = new float[cells, cells];
        for (int j = 0; j < cells; j++)
        {
            for (int i = 0; i < cells; i++) g[j, i] = (float)rng.NextDouble();
        }
        return g;
    }

    /// <summary>Bilinear (smoothstepped) interpolation of a wrapping grid over an n-pixel tile, so the result tiles.</summary>
    private static float Periodic(float[,] g, int x, int y, int n)
    {
        int cells = g.GetLength(0);
        x = ((x % n) + n) % n;
        float fx = x * cells / (float)n, fy = y * cells / (float)n;
        int x0 = Mathf.FloorToInt(fx), y0 = Mathf.FloorToInt(fy);
        float tx = Smooth01(fx - x0), ty = Smooth01(fy - y0);
        int x1 = (x0 + 1) % cells, y1 = (y0 + 1) % cells;
        x0 %= cells;
        y0 %= cells;
        return Mathf.Lerp(Mathf.Lerp(g[y0, x0], g[y0, x1], tx), Mathf.Lerp(g[y1, x0], g[y1, x1], tx), ty);
    }

    // ---------------------------------------------------------------- noise

    private static float Noise(float x, float z, float scale, int octave) =>
        Mathf.PerlinNoise(x / scale + 137.1f * (octave + 1) + Seed * 31.7f, z / scale + 71.3f * (octave + 1) + Seed * 11.9f);

    /// <summary>Fractal noise, about 0..1.</summary>
    private static float Fbm(float x, float z, float scale, int octaves)
    {
        float sum = 0f, amp = 1f, norm = 0f;
        for (int o = 0; o < octaves; o++)
        {
            sum += amp * Noise(x, z, scale, o);
            norm += amp;
            amp *= 0.5f;
            scale *= 0.5f;
        }
        return sum / norm;
    }

    /// <summary>Ridged fractal noise, 0..1, with sharp crests.</summary>
    private static float Ridged(float x, float z, float scale, int octaves)
    {
        float sum = 0f, amp = 1f, norm = 0f;
        for (int o = 0; o < octaves; o++)
        {
            float r = 1f - Mathf.Abs(2f * Noise(x, z, scale, o) - 1f);
            sum += amp * r * r;
            norm += amp;
            amp *= 0.5f;
            scale *= 0.5f;
        }
        return sum / norm;
    }

    private static float Smooth01(float t)
    {
        t = Mathf.Clamp01(t);
        return t * t * (3f - 2f * t);
    }

    private static float Sq(float v) => v * v;

    // ================================================================ the lake

    private static void BuildLake(Transform parent, Materials mats)
    {
        var go = new GameObject("Lake (N)");
        go.transform.SetParent(parent, false);
        go.transform.position = new Vector3(LakeCentre.x, GroundY, LakeCentre.y);   // on the ground; the mesh lifts the water
        LakeOutline lake = go.AddComponent<LakeOutline>();
        lake.generatorVersion = LakeGeneratorVersion;
        var knots = new List<float3>();
        for (int k = 0; k < LakeKnots; k++)
        {
            float a = 2f * Mathf.PI * k / LakeKnots;
            float wobble = 1f + 0.4f * (Noise(Mathf.Cos(a) * 300f, Mathf.Sin(a) * 300f, 200f, 5) - 0.5f);
            float z = LakeRadii.y * Mathf.Sin(a) * wobble;
            // The north shore follows the hills' foot instead of rounding off short of it.
            float foot = Inner - LakeNorthGap - LakeCentre.y - 12f * Noise(Mathf.Cos(a) * 300f, Mathf.Sin(a) * 300f, 60f, 2);
            knots.Add(new float3(LakeRadii.x * Mathf.Cos(a) * wobble, 0f, Mathf.Min(z, foot)));
        }
        go.GetComponent<SplineContainer>().Spline = new Spline(knots, TangentMode.AutoSmooth, true);
        var r = go.GetComponent<MeshRenderer>();
        r.sharedMaterials = new[] { mats.water, mats.shore };
        r.shadowCastingMode = ShadowCastingMode.Off;
        r.receiveShadows = false;
        GameObjectUtility.SetStaticEditorFlags(go, AllStatic);
        RebuildLake(lake);
    }

    /// <summary>The lake's shoreline sampled from its spline, counter-clockwise from above, in the lake's own frame
    /// (x, z); null if the spline has fewer than three knots.</summary>
    private static List<Vector2> Shoreline(LakeOutline lake)
    {
        var container = lake.GetComponent<SplineContainer>();
        Spline spline = container != null ? container.Spline : null;
        if (spline == null || spline.Count < 3) return null;
        int n = Mathf.Clamp(lake.samples, 16, 512);
        var poly = new List<Vector2>(n);
        for (int i = 0; i < n; i++)
        {
            float3 p = spline.EvaluatePosition(i / (float)n);
            var q = new Vector2(p.x, p.z);
            if (poly.Count == 0 || (q - poly[poly.Count - 1]).sqrMagnitude > 1e-4f) poly.Add(q);
        }
        if (poly.Count >= 2 && (poly[0] - poly[poly.Count - 1]).sqrMagnitude <= 1e-4f) poly.RemoveAt(poly.Count - 1);
        if (poly.Count < 3) return null;
        if (SignedArea(poly) < 0f) poly.Reverse();
        return poly;
    }

    /// <summary>The shoreline with, at each of its points, the bank's width and the beach's width beyond it (m).</summary>
    private sealed class ShoreProfile
    {
        public List<Vector2> poly;
        public float[] bank, beach;
        public Rect bounds;
        public float maxReach;
        public Vector2 noiseOffset;
    }

    /// <summary>
    /// <paramref name="lake"/>'s shoreline and how wide its bank and beaches are along it, in the lake's own frame; null
    /// if it has no shoreline. Both widths come from noise at each shore point, so they follow the shore round and are
    /// the same every time. The beaches are the shore points whose noise is in the top <see cref="LakeOutline.beachShare"/>,
    /// widening smoothly from nothing at the threshold to <see cref="LakeOutline.beachWidth"/> at the highest.
    /// </summary>
    private static ShoreProfile Profile(LakeOutline lake)
    {
        List<Vector2> poly = Shoreline(lake);
        if (poly == null) return null;
        int n = poly.Count;
        var offset = new Vector2(lake.beachSeed * 173.3f, lake.beachSeed * -91.7f);
        var bank = new float[n];
        var raw = new float[n];
        float bankWidth = Mathf.Max(lake.bankWidth, 0.5f);
        for (int i = 0; i < n; i++)
        {
            Vector2 q = poly[i] + offset;
            bank[i] = bankWidth * Mathf.Clamp(0.3f + 1.4f * Fbm(q.x, q.y, 45f, 2), 0.5f, 1.6f);
            raw[i] = Fbm(q.x + 500f, q.y - 500f, 160f, 3);
        }
        var beach = new float[n];
        float share = Mathf.Clamp01(lake.beachShare), width = Mathf.Max(lake.beachWidth, 0f);
        if (share > 0f && width > 0f)
        {
            float[] sorted = raw.OrderBy(v => v).ToArray();
            float threshold = share >= 1f ? sorted[0] - 1e-4f : sorted[Mathf.Clamp((int)((1f - share) * n), 0, n - 1)];
            float span = Mathf.Max(sorted[n - 1] - threshold, 1e-4f);
            // A beach reaches its full width over the upper ~70% of its noise, so most of each is broad, not a sliver.
            for (int i = 0; i < n; i++) beach[i] = width * Smooth01((raw[i] - threshold) / (0.7f * span));
            // No beach where the ground beyond the shore rises (the hills' foot): only the city tiles take paint, so it
            // would end in a straight line at the joint, and a beach does not climb a slope anyway.
            LandmarkSet set = lake.GetComponentInParent<LandmarkSet>();
            if (set != null)
            {
                float ground = lake.transform.position.y;
                for (int i = 0; i < n; i++)
                {
                    if (beach[i] <= 0f) continue;
                    Vector2 dir = Outward(poly, i).normalized;
                    float rise = 0f;
                    for (int k = 1; k <= 3; k++)
                    {
                        Vector2 q = poly[i] + dir * (bank[i] + width * 0.35f * k);
                        Vector3 w = lake.transform.TransformPoint(new Vector3(q.x, 0f, q.y));
                        rise = Mathf.Max(rise, GroundAt(set, w.x, w.z) - ground);
                    }
                    beach[i] *= 1f - Smooth01((rise - 0.3f) / 1.2f);
                }
            }
        }
        return new ShoreProfile
        {
            poly = poly, bank = bank, beach = beach, bounds = Bounds2(poly), noiseOffset = offset,
            maxReach = bank.Max() + BeachReach * beach.Max() + 2f * BeachFeatherMax,
        };
    }

    /// <summary>The farthest a beach's ragged edge reaches past the bank, as a multiple of its width there.</summary>
    private const float BeachReach = 1.25f;
    private const float BeachFeatherMax = 6f;

    /// <summary>
    /// How much sand the ground at <paramref name="q"/> (the lake's frame) shows, 0..1: all of it under the water and
    /// the bank, then out to a beach's edge, which wanders in and out by about a third of its width with fine noise
    /// and fades over a few metres. Where a stretch has no beach the sand ends under the visible part of the bank
    /// (which crosses the ground about two thirds of the way out), so the grass runs down to it.
    /// </summary>
    private static float SandWeight(ShoreProfile s, Vector2 q)
    {
        float d = ShoreDistance(s, q, out float bank, out float beach);
        if (d <= 0f) return 1f;
        Vector2 r = q + s.noiseOffset;
        float edge = 0.3f * bank + beach * (0.9f + 1.4f * (Fbm(r.x, r.y, 18f, 3) - 0.5f));
        float feather = Mathf.Min(0.5f + 0.2f * beach, BeachFeatherMax);
        return 1f - Smooth01((d - edge + feather) / (2f * feather));
    }

    /// <summary>Distance (m) from <paramref name="q"/> to the shoreline, 0 inside the water, with the bank's and the
    /// beach's widths at the nearest shore point.</summary>
    private static float ShoreDistance(ShoreProfile s, Vector2 q, out float bank, out float beach)
    {
        List<Vector2> p = s.poly;
        int n = p.Count, best = 0;
        float bestD2 = float.MaxValue, bestT = 0f;
        for (int i = 0; i < n; i++)
        {
            Vector2 a = p[i], ab = p[(i + 1) % n] - a;
            float t = Mathf.Clamp01(Vector2.Dot(q - a, ab) / Mathf.Max(ab.sqrMagnitude, 1e-6f));
            float d2 = (a + ab * t - q).sqrMagnitude;
            if (d2 < bestD2)
            {
                bestD2 = d2;
                best = i;
                bestT = t;
            }
        }
        int j = (best + 1) % n;
        bank = Mathf.Lerp(s.bank[best], s.bank[j], bestT);
        beach = Mathf.Lerp(s.beach[best], s.beach[j], bestT);
        return Inside(p, q) ? 0f : Mathf.Sqrt(bestD2);
    }

    /// <summary>Rebuilds <paramref name="lake"/>'s mesh from its spline: the water (submesh 0) ear-clipped from the
    /// sampled shoreline, the bank (submesh 1) a narrow band of wandering width offset outward from it. Written into
    /// the mesh asset the lake already uses, or <c>Generated/Lake.asset</c>. The city tiles round it are the caller's
    /// to redo (<see cref="ClearTreesUnderLakes"/>): <see cref="AddTo"/> does it at once, and the editor hook a moment
    /// after the last edit, so a drag does not repaint the tiles every frame.</summary>
    public static void RebuildLake(LakeOutline lake)
    {
        if (lake == null) return;
        var filter = lake.GetComponent<MeshFilter>();
        ShoreProfile shore = Profile(lake);
        if (shore == null || filter == null) return;
        List<Vector2> poly = shore.poly;

        var draft = new MeshDraft(2);
        float uv = Mathf.Max(lake.uvMetres, 1f);
        // The water stands waterLevel above the ground and the bank slopes from just under its edge to shoreDepth below
        // the ground, so neither lies flat on the grass (see LakeOutline).
        float level = Mathf.Max(lake.waterLevel, 0.05f);
        var water = new int[poly.Count];
        for (int i = 0; i < poly.Count; i++)
        {
            water[i] = draft.Add(new Vector3(poly[i].x, level, poly[i].y), Vector3.up, poly[i] / uv);
        }
        foreach ((int a, int b, int c) in EarClip(poly)) draft.TriUp(0, water[a], water[b], water[c]);

        float bankTop = level - 0.03f, bankFoot = -Mathf.Max(lake.shoreDepth, 0.05f);
        var inner = new int[poly.Count];
        var outer = new int[poly.Count];
        for (int i = 0; i < poly.Count; i++)
        {
            Vector2 o = poly[i] + Outward(poly, i) * shore.bank[i];
            inner[i] = draft.Add(new Vector3(poly[i].x, bankTop, poly[i].y), Vector3.up, poly[i] / uv);
            outer[i] = draft.Add(new Vector3(o.x, bankFoot, o.y), Vector3.up, o / uv);
        }
        for (int i = 0; i < poly.Count; i++)
        {
            int j = (i + 1) % poly.Count;
            draft.TriUp(1, inner[i], inner[j], outer[j]);
            draft.TriUp(1, inner[i], outer[j], outer[i]);
        }

        Mesh mesh = filter.sharedMesh;
        string path = mesh != null ? AssetDatabase.GetAssetPath(mesh) : "";
        if (string.IsNullOrEmpty(path) || !path.StartsWith(GeneratedFolder)) mesh = MeshAsset("Lake");
        draft.WriteTo(mesh);
        mesh.RecalculateTangents();
        EditorUtility.SetDirty(mesh);
        if (filter.sharedMesh != mesh)
        {
            filter.sharedMesh = mesh;
            EditorUtility.SetDirty(filter);
        }
    }

    /// <summary>
    /// Clears the city tiles' trees from under and round every lake in <paramref name="scene"/> (its water, its bank,
    /// its beaches' furthest reach and <see cref="TreeClearance"/> beyond) and paints its beaches into them
    /// (<see cref="PaintBeaches"/>), returning how many trees were cleared. The four city tiles carry ~30,000 painted
    /// terrain trees between the city and ±1000 m, and their TerrainData is shared with five other scenes, so a tile a
    /// lake stands on is given this scene's own copy (<c>Landmarks/Terrain/City_&lt;source GUID&gt;.asset</c>, made once)
    /// and the copy's trees and paint are rewritten from the source's every time, so a lake that moves leaves no
    /// clearing or sand behind. A tile no lake reaches any more goes back to its shared data; its copy is left on disk
    /// (deleting it would break the scene as last saved) and reused if a lake comes back, and can be deleted once the
    /// scene is saved. Heights are untouched, so the altitude ceiling is too.
    /// </summary>
    public static int ClearTreesUnderLakes(Scene scene)
    {
        var lakes = new List<LakeInScene>();
        foreach (GameObject root in scene.GetRootGameObjects())
        {
            foreach (LakeOutline lake in root.GetComponentsInChildren<LakeOutline>(true))
            {
                ShoreProfile shore = Profile(lake);
                if (shore == null) continue;
                Rect b = shore.bounds;
                float m = shore.maxReach + TreeClearance;
                var corners = new List<Vector2>
                {
                    new Vector2(b.xMin - m, b.yMin - m), new Vector2(b.xMax + m, b.yMin - m),
                    new Vector2(b.xMin - m, b.yMax + m), new Vector2(b.xMax + m, b.yMax + m),
                };
                lakes.Add(new LakeInScene
                {
                    frame = lake.transform, shore = shore,
                    box = Bounds2(corners.Select(c =>
                    {
                        Vector3 w = lake.transform.TransformPoint(new Vector3(c.x, 0f, c.y));
                        return new Vector2(w.x, w.z);
                    }).ToList()),
                });
            }
        }

        TerrainLayer sand = EnsureSandLayer();
        int cleared = 0;
        foreach (GameObject root in scene.GetRootGameObjects())
        {
            foreach (Terrain t in root.GetComponentsInChildren<Terrain>(true))
            {
                if (t.terrainData == null || t.GetComponentInParent<LandmarkSet>() != null) continue;
                TerrainData source = CitySource(t.terrainData);
                bool isCopy = source != t.terrainData;
                Vector3 o = t.transform.position, s = source.size;
                var rect = new Rect(o.x, o.z, s.x, s.z);
                if (!lakes.Any(l => rect.Overlaps(l.box)))
                {
                    // No lake reaches it (any more): it shows its shared tile, not a 2 MB copy that only duplicates it.
                    if (isCopy) UseTerrainData(t, source, scene);
                    continue;
                }

                TerrainData copy = isCopy ? t.terrainData : CityCopy(source);
                TreeInstance[] all = source.treeInstances;
                var kept = new List<TreeInstance>(all.Length);
                foreach (TreeInstance tree in all)
                {
                    var p = new Vector2(o.x + tree.position.x * s.x, o.z + tree.position.z * s.z);
                    bool wet = false;
                    foreach (LakeInScene l in lakes)
                    {
                        if (!l.box.Contains(p)) continue;
                        float d = ShoreDistance(l.shore, l.Local(p), out float bank, out float beach);
                        wet = d < bank + BeachReach * beach + TreeClearance;
                        if (wet) break;
                    }
                    if (wet) cleared++;
                    else kept.Add(tree);
                }
                bool changed = false;
                TreeInstance[] current = copy.treeInstances;
                if (current.Length != kept.Count || kept.Where((tree, i) => tree.position != current[i].position).Any())
                {
                    copy.SetTreeInstances(kept.ToArray(), false);
                    changed = true;
                }
                changed |= PaintBeaches(source, copy, o, lakes, sand);
                if (changed) EditorUtility.SetDirty(copy);
                UseTerrainData(t, copy, scene);
            }
        }
        return cleared;
    }

    /// <summary>Points <paramref name="t"/> and its collider at <paramref name="data"/>, if they are not already.</summary>
    private static void UseTerrainData(Terrain t, TerrainData data, Scene scene)
    {
        if (t.terrainData == data) return;
        t.terrainData = data;
        var collider = t.GetComponent<TerrainCollider>();
        if (collider != null) collider.terrainData = data;
        EditorUtility.SetDirty(t);
        EditorSceneManager.MarkSceneDirty(scene);
    }

    /// <summary>A lake's shore profile, with its frame and the bounds of its reach in world (x, z).</summary>
    private sealed class LakeInScene
    {
        public Transform frame;
        public ShoreProfile shore;
        public Rect box;

        public Vector2 Local(Vector2 world)
        {
            Vector3 l = frame.InverseTransformPoint(new Vector3(world.x, frame.position.y, world.y));
            return new Vector2(l.x, l.z);
        }
    }

    /// <summary>The beaches' sand: a terrain layer like the others this builder generates, made if missing.</summary>
    private static TerrainLayer EnsureSandLayer() => EnsureLayer(new LayerLook
    {
        name = "Sand", colour = new Color(0.74f, 0.67f, 0.50f), variation = 0.15f, tile = 12f, version = 2,
    });

    /// <summary>
    /// Writes <paramref name="copy"/>'s paint as <paramref name="source"/>'s with the lakes' sand over it
    /// (<see cref="SandWeight"/>), the sand layer appended to the source's layers (with the city's one grass layer that
    /// is still a single splat texture). Left alone when it already holds that, so rebuilding a lake whose shore did not
    /// move (after any undo, say) does not rewrite the tiles. Returns whether it wrote anything.
    /// </summary>
    private static bool PaintBeaches(TerrainData source, TerrainData copy, Vector3 origin, List<LakeInScene> lakes,
                                     TerrainLayer sand)
    {
        TerrainLayer[] sourceLayers = source.terrainLayers;
        int nSource = sourceLayers.Length;
        if (nSource == 0) return false;
        int sandIndex = System.Array.IndexOf(sourceLayers, sand);
        TerrainLayer[] layers = sandIndex >= 0 ? sourceLayers : sourceLayers.Append(sand).ToArray();
        if (sandIndex < 0) sandIndex = nSource;

        int res = source.alphamapResolution;
        Vector3 size = source.size;
        float[,,] paint = source.GetAlphamaps(0, 0, res, res);
        var result = new float[res, res, layers.Length];
        for (int y = 0; y < res; y++)
        {
            for (int x = 0; x < res; x++)
            {
                var p = new Vector2(origin.x + (x + 0.5f) / res * size.x, origin.z + (y + 0.5f) / res * size.z);
                float w = 0f;
                foreach (LakeInScene l in lakes)
                {
                    if (l.box.Contains(p)) w = Mathf.Max(w, SandWeight(l.shore, l.Local(p)));
                }
                for (int k = 0; k < nSource; k++) result[y, x, k] = paint[y, x, k] * (1f - w);
                result[y, x, sandIndex] += w;
            }
        }

        if (copy.terrainLayers.SequenceEqual(layers) && copy.alphamapResolution == res &&
            SamePaint(copy.GetAlphamaps(0, 0, res, res), result))
        {
            return false;
        }
        copy.terrainLayers = layers;
        copy.SetAlphamaps(0, 0, result);
        return true;
    }

    /// <summary>Whether two alphamaps match to within the 8 bits a splat texture stores per channel.</summary>
    private static bool SamePaint(float[,,] a, float[,,] b)
    {
        if (a.GetLength(0) != b.GetLength(0) || a.GetLength(1) != b.GetLength(1) || a.GetLength(2) != b.GetLength(2))
        {
            return false;
        }
        for (int y = 0; y < a.GetLength(0); y++)
        {
            for (int x = 0; x < a.GetLength(1); x++)
            {
                for (int k = 0; k < a.GetLength(2); k++)
                {
                    if (Mathf.Abs(a[y, x, k] - b[y, x, k]) > 0.006f) return false;
                }
            }
        }
        return true;
    }

    /// <summary>Whether any Terrain in <paramref name="scene"/> stands on this scene's own copy of a city tile, i.e.
    /// whether a lake has ever cleared trees there.</summary>
    public static bool HasCityTileCopies(Scene scene)
    {
        if (!scene.IsValid() || !scene.isLoaded) return false;
        string prefix = $"{TerrainFolder}/{CityCopyPrefix}";
        foreach (GameObject root in scene.GetRootGameObjects())
        {
            foreach (Terrain t in root.GetComponentsInChildren<Terrain>(true))
            {
                if (t.terrainData != null && AssetDatabase.GetAssetPath(t.terrainData).StartsWith(prefix)) return true;
            }
        }
        return false;
    }

    /// <summary>The shared TerrainData a city tile's copy was made from (by the GUID in its name), or the data itself.</summary>
    private static TerrainData CitySource(TerrainData data)
    {
        string path = AssetDatabase.GetAssetPath(data);
        string prefix = $"{TerrainFolder}/{CityCopyPrefix}";
        if (!path.StartsWith(prefix)) return data;
        string guid = Path.GetFileNameWithoutExtension(path).Substring(CityCopyPrefix.Length);
        var source = AssetDatabase.LoadAssetAtPath<TerrainData>(AssetDatabase.GUIDToAssetPath(guid));
        return source != null ? source : data;
    }

    /// <summary>This scene's copy of a shared city tile, made if missing.</summary>
    private static TerrainData CityCopy(TerrainData source)
    {
        string sourcePath = AssetDatabase.GetAssetPath(source);
        string path = $"{TerrainFolder}/{CityCopyPrefix}{AssetDatabase.AssetPathToGUID(sourcePath)}.asset";
        var copy = AssetDatabase.LoadAssetAtPath<TerrainData>(path);
        if (copy != null) return copy;
        EnsureFolder(TerrainFolder);
        AssetDatabase.CopyAsset(sourcePath, path);
        return AssetDatabase.LoadAssetAtPath<TerrainData>(path);
    }

    private static Rect Bounds2(List<Vector2> poly)
    {
        float x0 = poly.Min(p => p.x), x1 = poly.Max(p => p.x), z0 = poly.Min(p => p.y), z1 = poly.Max(p => p.y);
        return Rect.MinMaxRect(x0, z0, x1, z1);
    }

    /// <summary>Whether q lies inside the polygon (even-odd rule).</summary>
    private static bool Inside(List<Vector2> poly, Vector2 q)
    {
        bool inside = false;
        for (int i = 0, j = poly.Count - 1; i < poly.Count; j = i++)
        {
            Vector2 a = poly[i], b = poly[j];
            if ((a.y > q.y) != (b.y > q.y) && q.x < (b.x - a.x) * (q.y - a.y) / (b.y - a.y) + a.x) inside = !inside;
        }
        return inside;
    }

    /// <summary>How far the lake's water stands above the ground under its centre (m; negative is under it), from the
    /// scene's own Terrains. 0 when there is no ground to measure.</summary>
    public static float WaterAboveGround(LakeOutline lake)
    {
        Vector3 water = lake.transform.TransformPoint(new Vector3(0f, lake.waterLevel, 0f));
        Scene scene = lake.gameObject.scene;
        foreach (Terrain t in Terrain.activeTerrains)
        {
            if (t.terrainData == null || t.gameObject.scene != scene) continue;
            Vector3 o = t.transform.position, s = t.terrainData.size;
            if (water.x < o.x || water.x > o.x + s.x || water.z < o.z || water.z > o.z + s.z) continue;
            return water.y - (t.SampleHeight(water) + o.y);
        }
        return 0f;
    }

    private static bool HasPositionOverride(Transform t) =>
        PrefabUtility.IsPartOfPrefabInstance(t) &&
        (PrefabUtility.GetPropertyModifications(t) ?? new PropertyModification[0])
            .Any(m => m.target != null && m.target == PrefabUtility.GetCorrespondingObjectFromSource(t) &&
                      m.propertyPath.StartsWith("m_LocalPosition"));

    /// <summary>Puts a deleted lake back as a freshly generated one, so deleting it and running <i>Add to open scene</i>
    /// starts it over, at its generated size and place: a lake deleted from this scene's copy of the prefab has the
    /// deletion reverted and the prefab's lake replaced by a new one (the prefab is the lake's record, and a scene that
    /// merely reverted would bring back the old lake), and one deleted from the prefab itself is built there again.
    /// <see cref="AddTo"/> then re-clears the city tiles' trees round wherever it now stands. Returns a note for the
    /// summary, or "" when the set still has its lake.</summary>
    private static string RestoreLake(LandmarkSet set)
    {
        if (set.GetComponentInChildren<LakeOutline>(true) != null) return "";
        GameObject root = set.gameObject;
        if (!PrefabUtility.IsPartOfPrefabInstance(root))
        {
            BuildLake(set.transform, EnsureMaterials());
            EditorSceneManager.MarkSceneDirty(root.scene);
            return " The lake was missing and was generated again.";
        }
        GameObject instanceRoot = PrefabUtility.GetNearestPrefabInstanceRoot(root);
        bool fromScene = false;
        foreach (RemovedGameObject removed in PrefabUtility.GetRemovedGameObjects(instanceRoot))
        {
            if (removed.assetGameObject != null && removed.assetGameObject.GetComponentInChildren<LakeOutline>(true) != null)
            {
                removed.Revert(InteractionMode.UserAction);
                fromScene = true;
            }
        }
        EditorSceneManager.MarkSceneDirty(root.scene);
        string path = PrefabUtility.GetPrefabAssetPathOfNearestInstanceRoot(root);
        GameObject contents = PrefabUtility.LoadPrefabContents(path);
        try
        {
            LandmarkSet inPrefab = contents.GetComponentInChildren<LandmarkSet>(true);
            foreach (LakeOutline old in inPrefab.GetComponentsInChildren<LakeOutline>(true))
            {
                Object.DestroyImmediate(old.gameObject);
            }
            BuildLake(inPrefab.transform, EnsureMaterials());
            PrefabUtility.SaveAsPrefabAsset(contents, path);
        }
        finally
        {
            PrefabUtility.UnloadPrefabContents(contents);
        }
        return fromScene
            ? " The lake deleted from this scene was generated afresh (in the prefab, which holds it)."
            : " The lake was missing from the prefab and was generated there again.";
    }

    /// <summary>A lake whose water this scene has pushed under the ground by overriding its position — it then shows as
    /// grass — gets that override reverted (the editor asks first; batchmode only reports it). Returns a note for the
    /// summary, or "".</summary>
    private static string RescueSunkenLake(LakeOutline lake)
    {
        float above = WaterAboveGround(lake);
        if (above >= 0.2f || !HasPositionOverride(lake.transform)) return "";
        string where = $"{lake.name}'s water is {-above:0.0} m under the ground: this scene overrides its position";
        if (Application.isBatchMode ||
            !EditorUtility.DisplayDialog("Lake under the ground",
                where + " (y " + lake.transform.localPosition.y.ToString("0.##") + "). Revert that override, so the lake " +
                "stands where the prefab puts it?", "Revert", "Keep"))
        {
            Debug.LogWarning($"LandmarkBuilder: {where}; revert it (Transform ⋮ > Revert) to see the lake.", lake);
            return $" {where}, kept.";
        }
        PrefabUtility.RevertObjectOverride(lake.transform, InteractionMode.UserAction);
        EditorSceneManager.MarkSceneDirty(lake.gameObject.scene);
        return $" {lake.name}'s position override (under the ground) was reverted.";
    }

    /// <summary>Every lake in the open scenes and the prefab stage (or only the one whose spline is
    /// <paramref name="changed"/>), rebuilt; returns them.</summary>
    public static List<LakeOutline> RebuildLakes(Spline changed = null)
    {
        var rebuilt = new List<LakeOutline>();
        var lakes = new List<LakeOutline>(Object.FindObjectsByType<LakeOutline>(FindObjectsInactive.Include,
                                                                                  FindObjectsSortMode.None));
        PrefabStage stage = PrefabStageUtility.GetCurrentPrefabStage();
        if (stage != null) lakes.AddRange(stage.prefabContentsRoot.GetComponentsInChildren<LakeOutline>(true));
        foreach (LakeOutline lake in lakes.Distinct())
        {
            var container = lake.GetComponent<SplineContainer>();
            if (container == null) continue;
            if (changed != null && !container.Splines.Contains(changed)) continue;
            RebuildLake(lake);
            rebuilt.Add(lake);
        }
        return rebuilt;
    }

    private static float SignedArea(IReadOnlyList<Vector2> p)
    {
        float s = 0f;
        for (int i = 0; i < p.Count; i++)
        {
            Vector2 a = p[i], b = p[(i + 1) % p.Count];
            s += a.x * b.y - b.x * a.y;
        }
        return 0.5f * s;
    }

    /// <summary>The outward miter direction at vertex i of a counter-clockwise polygon, scaled so the offset edges stay
    /// parallel (capped at 2x on sharp corners).</summary>
    private static Vector2 Outward(IReadOnlyList<Vector2> p, int i)
    {
        int n = p.Count;
        Vector2 prev = p[(i - 1 + n) % n], cur = p[i], next = p[(i + 1) % n];
        Vector2 e0 = (cur - prev).normalized, e1 = (next - cur).normalized;
        var n0 = new Vector2(e0.y, -e0.x);
        var n1 = new Vector2(e1.y, -e1.x);
        Vector2 m = (n0 + n1).normalized;
        if (m.sqrMagnitude < 1e-6f) m = n1;
        return m / Mathf.Max(Vector2.Dot(m, n1), 0.5f);
    }

    /// <summary>Triangles of a simple counter-clockwise polygon, by ear clipping.</summary>
    private static List<(int, int, int)> EarClip(IReadOnlyList<Vector2> p)
    {
        var tris = new List<(int, int, int)>();
        var idx = Enumerable.Range(0, p.Count).ToList();
        int guard = 0;
        while (idx.Count > 3 && guard++ < 100000)
        {
            bool clipped = false;
            for (int k = 0; k < idx.Count; k++)
            {
                int ia = idx[(k - 1 + idx.Count) % idx.Count], ib = idx[k], ic = idx[(k + 1) % idx.Count];
                Vector2 a = p[ia], b = p[ib], c = p[ic];
                if (Cross(b - a, c - b) <= 0f) continue;   // reflex
                bool empty = true;
                foreach (int iq in idx)
                {
                    if (iq == ia || iq == ib || iq == ic) continue;
                    if (InTriangle(p[iq], a, b, c))
                    {
                        empty = false;
                        break;
                    }
                }
                if (!empty) continue;
                tris.Add((ia, ib, ic));
                idx.RemoveAt(k);
                clipped = true;
                break;
            }
            if (!clipped) break;   // not simple (a self-crossing outline): give up on the rest rather than loop
        }
        if (idx.Count == 3) tris.Add((idx[0], idx[1], idx[2]));
        return tris;
    }

    private static float Cross(Vector2 u, Vector2 v) => u.x * v.y - u.y * v.x;

    private static bool InTriangle(Vector2 q, Vector2 a, Vector2 b, Vector2 c) =>
        Cross(b - a, q - a) >= 0f && Cross(c - b, q - b) >= 0f && Cross(a - c, q - c) >= 0f;

    // ================================================================ the mast and the turbines

    /// <summary>Puts each spot's model in place: the slot's model if set, else the generated one. Every spot is snapped
    /// to the ground first; a model is scaled to the generated height (if <see cref="LandmarkSet.fitModelHeight"/>), has
    /// its colliders removed, its layers set to Default and its shadows turned off. The mast's beacons go on top of
    /// whatever model stands there. Returns a summary with each model's triangle count.</summary>
    public static string PlaceModels(LandmarkSet set) => PlaceModels(set, EnsureMaterials());

    private static string PlaceModels(LandmarkSet set, Materials mats)
    {
        var report = new StringBuilder();
        if (set.windFarm != null)
        {
            Mesh tower = TurbineTowerMesh(), rotor = TurbineRotorMesh();
            int n = 0, tris = 0;
            foreach (Transform spot in set.windFarm)
            {
                SnapToGround(set, spot);
                GameObject model = set.turbineModel != null
                    ? PlaceModel(spot, set.turbineModel, set.turbineModelYaw, set.fitModelHeight ? TurbineTip : 0f)
                    : GeneratedTurbine(spot, tower, rotor, mats);
                tris += Triangles(model);
                n++;
            }
            report.Append($"{n} turbine(s) ({(set.turbineModel != null ? set.turbineModel.name : "generated")}, " +
                          $"{tris:N0} triangles); ");
        }
        if (set.mast != null)
        {
            SnapToGround(set, set.mast);
            GameObject model = set.mastModel != null
                ? PlaceModel(set.mast, set.mastModel, set.mastModelYaw, set.fitModelHeight ? MastHeight : 0f)
                : GeneratedMast(set.mast, mats);
            set.beacons = PlaceBeacons(set.mast, model, mats);
            report.Append($"mast ({(set.mastModel != null ? set.mastModel.name : "generated")}, " +
                          $"{Triangles(model):N0} triangles, {set.beacons.Length} beacons)");
        }
        EditorUtility.SetDirty(set);
        return report.ToString();
    }

    /// <summary>Places the models where they will last: in the prefab. On a scene instance, its overrides are applied
    /// to the prefab first (after asking), so nothing the user changed is lost; in the prefab stage, or on an object
    /// that is not a prefab, it works in place.</summary>
    public static void PlaceModelsAndSave(LandmarkSet target)
    {
        GameObject root = target.gameObject;
        bool inStage = PrefabStageUtility.GetPrefabStage(root) != null;
        // A slot set from a script (not the inspector) is not an override until recorded, and would be lost below.
        if (!inStage && PrefabUtility.IsPartOfPrefabInstance(target))
        {
            PrefabUtility.RecordPrefabInstancePropertyModifications(target);
        }
        string path = "";
        if (!inStage && PrefabUtility.IsPartOfPrefabAsset(root))
        {
            path = AssetDatabase.GetAssetPath(root);
        }
        else if (!inStage && PrefabUtility.IsPartOfPrefabInstance(root))
        {
            GameObject instanceRoot = PrefabUtility.GetNearestPrefabInstanceRoot(root);
            path = PrefabUtility.GetPrefabAssetPathOfNearestInstanceRoot(root);
            if (PrefabUtility.HasPrefabInstanceAnyOverrides(instanceRoot, false))
            {
                if (!Application.isBatchMode &&
                    !EditorUtility.DisplayDialog("Place landmark models",
                        "Place models edits Landmarks.prefab, so that a scene rebuild keeps them. This scene's unapplied " +
                        "changes to the landmarks will be applied to the prefab first.", "Apply and place", "Cancel"))
                {
                    return;
                }
                PrefabUtility.ApplyPrefabInstance(instanceRoot, InteractionMode.UserAction);
            }
        }

        string summary;
        if (string.IsNullOrEmpty(path))
        {
            Undo.RegisterFullObjectHierarchyUndo(root, "Place landmark models");
            summary = PlaceModels(target);
            EditorSceneManager.MarkSceneDirty(root.scene);
        }
        else
        {
            GameObject contents = PrefabUtility.LoadPrefabContents(path);
            try
            {
                LandmarkSet set = contents.GetComponentInChildren<LandmarkSet>(true);
                summary = PlaceModels(set);
                PrefabUtility.SaveAsPrefabAsset(contents, path);
            }
            finally
            {
                PrefabUtility.UnloadPrefabContents(contents);
            }
        }
        AssetDatabase.SaveAssets();
        Debug.Log("LandmarkBuilder: placed " + summary);
    }

    /// <summary>Drops <paramref name="spot"/> onto the ring's ground (0 over the city).</summary>
    private static void SnapToGround(LandmarkSet set, Transform spot)
    {
        Vector3 p = spot.position;
        p.y = GroundAt(set, p.x, p.z);
        spot.position = p;
    }

    /// <summary>World height of the set's ground at (x, z): the ring tile there, else the city's ground.</summary>
    private static float GroundAt(LandmarkSet set, float x, float z)
    {
        if (set.ground != null)
        {
            foreach (Terrain t in set.ground.GetComponentsInChildren<Terrain>(true))
            {
                if (t.terrainData == null) continue;
                Vector3 o = t.transform.position, s = t.terrainData.size;
                if (x < o.x || x > o.x + s.x || z < o.z || z > o.z + s.z) continue;
                return t.SampleHeight(new Vector3(x, 0f, z)) + o.y;
            }
        }
        return GroundY;
    }

    private static GameObject PlaceModel(Transform spot, GameObject source, float yaw, float height)
    {
        RemoveModel(spot);
        GameObject go = PrefabUtility.IsPartOfPrefabAsset(source)
            ? (GameObject)PrefabUtility.InstantiatePrefab(source, spot)
            : Object.Instantiate(source, spot);
        // Unpacked, so its colliders can go and nothing links the set to a pack that is not in git.
        if (PrefabUtility.IsPartOfPrefabInstance(go))
        {
            PrefabUtility.UnpackPrefabInstance(go, PrefabUnpackMode.Completely, InteractionMode.AutomatedAction);
        }
        go.name = LandmarkSet.ModelName;
        go.transform.localPosition = Vector3.zero;
        go.transform.localRotation = Quaternion.Euler(0f, yaw, 0f);
        go.transform.localScale = Vector3.one;
        Bounds b = RendererBounds(go);
        if (height > 0f && b.size.y > 1e-3f) go.transform.localScale = Vector3.one * (height / b.size.y);
        b = RendererBounds(go);
        if (b.size.y > 0f) go.transform.position += Vector3.up * (spot.position.y - b.min.y);   // stand on the ground
        foreach (Collider c in go.GetComponentsInChildren<Collider>(true)) Object.DestroyImmediate(c);
        foreach (Rigidbody rb in go.GetComponentsInChildren<Rigidbody>(true)) Object.DestroyImmediate(rb);
        foreach (Transform t in go.GetComponentsInChildren<Transform>(true)) t.gameObject.layer = 0;
        foreach (Renderer r in go.GetComponentsInChildren<Renderer>(true))
        {
            r.shadowCastingMode = ShadowCastingMode.Off;
            r.receiveShadows = false;
        }
        return go;
    }

    private static void RemoveModel(Transform spot)
    {
        Transform old;
        while ((old = spot.Find(LandmarkSet.ModelName)) != null) Object.DestroyImmediate(old.gameObject);
    }

    private static GameObject GeneratedTurbine(Transform spot, Mesh tower, Mesh rotorMesh, Materials mats)
    {
        RemoveModel(spot);
        GameObject go = MeshObject(LandmarkSet.ModelName, spot, tower, new[] { mats.turbine }, true);
        GameObject rotor = MeshObject("Rotor", go.transform, rotorMesh, new[] { mats.turbine }, false);
        rotor.transform.localPosition = new Vector3(0f, HubHeight, 4.2f);
        return go;
    }

    private static GameObject GeneratedMast(Transform spot, Materials mats)
    {
        RemoveModel(spot);
        return MeshObject(LandmarkSet.ModelName, spot, MastMesh(), new[] { mats.mastRed, mats.mastWhite }, true);
    }

    /// <summary>A beacon on the top of the mast's model and one at 55% of its height, rebuilt every time.</summary>
    private static Renderer[] PlaceBeacons(Transform spot, GameObject model, Materials mats)
    {
        foreach (Transform old in spot.Cast<Transform>().Where(t => t.name.StartsWith("Beacon")).ToList())
        {
            Object.DestroyImmediate(old.gameObject);
        }
        Bounds b = RendererBounds(model);
        float top = b.size.y > 0f ? b.max.y - spot.position.y : MastHeight + MastAntenna;
        Mesh sphere = BeaconMesh();
        var beacons = new List<Renderer>();
        foreach ((string name, float y) in new[] { ("Beacon top", top + BeaconRadius), ("Beacon mid", 0.55f * top + 3.5f) })
        {
            GameObject go = MeshObject(name, spot, sphere, new[] { mats.beacon }, false);
            go.transform.localPosition = new Vector3(0f, y, 0f);
            beacons.Add(go.GetComponent<Renderer>());
        }
        return beacons.ToArray();
    }

    private static GameObject MeshObject(string name, Transform parent, Mesh mesh, Material[] materials, bool isStatic)
    {
        var go = new GameObject(name);
        go.transform.SetParent(parent, false);
        go.AddComponent<MeshFilter>().sharedMesh = mesh;
        var r = go.AddComponent<MeshRenderer>();
        r.sharedMaterials = materials;
        r.shadowCastingMode = ShadowCastingMode.Off;
        r.receiveShadows = false;
        if (isStatic) GameObjectUtility.SetStaticEditorFlags(go, AllStatic);
        return go;
    }

    private static Bounds RendererBounds(GameObject go)
    {
        Renderer[] rs = go.GetComponentsInChildren<Renderer>(true);
        if (rs.Length == 0) return new Bounds(go.transform.position, Vector3.zero);
        Bounds b = rs[0].bounds;
        foreach (Renderer r in rs) b.Encapsulate(r.bounds);
        return b;
    }

    private static int Triangles(GameObject go)
    {
        if (go == null) return 0;
        long t = 0;
        foreach (MeshFilter f in go.GetComponentsInChildren<MeshFilter>(true))
        {
            if (f.sharedMesh != null) t += TriangleCount(f.sharedMesh);
        }
        foreach (SkinnedMeshRenderer s in go.GetComponentsInChildren<SkinnedMeshRenderer>(true))
        {
            if (s.sharedMesh != null) t += TriangleCount(s.sharedMesh);
        }
        return (int)t;
    }

    private static long TriangleCount(Mesh m)
    {
        long t = 0;
        for (int s = 0; s < m.subMeshCount; s++) t += m.GetIndexCount(s) / 3;
        return t;
    }

    // ---------------------------------------------------------------- generated meshes

    /// <summary>Tapered tower and nacelle; the rotor faces +Z.</summary>
    private static Mesh TurbineTowerMesh()
    {
        var d = new MeshDraft(1);
        d.Lathe(0, new[] { new Vector2(2.6f, 0f), new Vector2(1.7f, HubHeight - 1.6f) }, 20, Vector3.zero);
        d.Box(0, new Vector3(0f, HubHeight, -1.5f), new Vector3(1.8f, 1.8f, 5.2f));
        return Save("TurbineTower", d);
    }

    /// <summary>Hub and three pitched, tapered blades about the origin, turning about +Z.</summary>
    private static Mesh TurbineRotorMesh()
    {
        var d = new MeshDraft(1);
        d.Box(0, new Vector3(0f, 0f, 0.6f), new Vector3(1.6f, 1.6f, 1.6f));
        for (int k = 0; k < 3; k++)
        {
            float a = (90f + 120f * k) * Mathf.Deg2Rad;
            var radial = new Vector3(Mathf.Cos(a), Mathf.Sin(a), 0f);
            Quaternion pitch = Quaternion.AngleAxis(12f, radial);
            Vector3 chord = pitch * new Vector3(-Mathf.Sin(a), Mathf.Cos(a), 0f);
            Vector3 thick = pitch * Vector3.forward;
            var corners = new Vector3[8];
            for (int i = 0; i < 8; i++)
            {
                bool tip = (i & 4) != 0;
                float r = tip ? RotorRadius : 1.4f;
                float c = (tip ? 0.9f : 3.2f) * ((i & 1) != 0 ? 0.5f : -0.5f);
                float t = (tip ? 0.25f : 0.9f) * ((i & 2) != 0 ? 0.5f : -0.5f);
                corners[i] = radial * r + chord * c + thick * t + new Vector3(0f, 0f, 0.6f);
            }
            d.Block(0, corners);
        }
        return Save("TurbineRotor", d);
    }

    /// <summary>A six-sided tapered shaft in red and white bands (submeshes 0 and 1), two equipment platforms and a red
    /// antenna on top.</summary>
    private static Mesh MastMesh()
    {
        var d = new MeshDraft(2);
        for (int b = 0; b < MastBands; b++)
        {
            float y0 = MastHeight * b / MastBands, y1 = MastHeight * (b + 1) / MastBands;
            d.Lathe(b % 2 == 0 ? 0 : 1, new[] { new Vector2(MastRadius(y0), y0), new Vector2(MastRadius(y1), y1) }, 6,
                    Vector3.zero);
        }
        foreach (float f in new[] { 0.55f, 0.8f })
        {
            float y = f * MastHeight;
            d.Lathe(1, new[]
            {
                new Vector2(0f, y - 1.5f), new Vector2(7f, y - 1.5f), new Vector2(7f, y + 1.5f), new Vector2(0f, y + 1.5f),
            }, 12, Vector3.zero);
        }
        d.Lathe(0, new[] { new Vector2(0.6f, MastHeight), new Vector2(0.35f, MastHeight + MastAntenna) }, 8, Vector3.zero);
        return Save("Mast", d);
    }

    private static float MastRadius(float y) => Mathf.Lerp(MastBaseRadius, MastTopRadius, y / MastHeight);

    private static Mesh BeaconMesh()
    {
        var profile = new List<Vector2>();
        for (int i = 0; i <= 8; i++)
        {
            float t = Mathf.PI * i / 8f;
            profile.Add(new Vector2(Mathf.Sin(t) * BeaconRadius, -Mathf.Cos(t) * BeaconRadius));
        }
        var d = new MeshDraft(1);
        d.Lathe(0, profile, 12, Vector3.zero);
        return Save("Beacon", d);
    }

    private static Mesh Save(string name, MeshDraft d)
    {
        Mesh mesh = MeshAsset(name);
        d.WriteTo(mesh);
        EditorUtility.SetDirty(mesh);
        return mesh;
    }

    /// <summary>The mesh asset <c>Generated/&lt;name&gt;.asset</c>, made if missing. An existing one is rewritten in
    /// place by the caller, so its GUID and every reference to it survive a rebuild.</summary>
    private static Mesh MeshAsset(string name)
    {
        EnsureFolder(GeneratedFolder);
        string path = $"{GeneratedFolder}/{name}.asset";
        var mesh = AssetDatabase.LoadAssetAtPath<Mesh>(path);
        if (mesh != null) return mesh;
        mesh = new Mesh { name = name };
        AssetDatabase.CreateAsset(mesh, path);
        return mesh;
    }

    /// <summary>Vertices, normals, UVs and per-submesh triangles, written into a mesh at the end.</summary>
    private sealed class MeshDraft
    {
        private readonly List<Vector3> v = new List<Vector3>();
        private readonly List<Vector3> n = new List<Vector3>();
        private readonly List<Vector2> uv = new List<Vector2>();
        private readonly List<int>[] sub;

        public MeshDraft(int submeshes)
        {
            sub = new List<int>[submeshes];
            for (int i = 0; i < submeshes; i++) sub[i] = new List<int>();
        }

        public int Add(Vector3 p, Vector3 normal, Vector2 t)
        {
            v.Add(p);
            n.Add(normal);
            uv.Add(t);
            return v.Count - 1;
        }

        /// <summary>A triangle wound to face up (+Y).</summary>
        public void TriUp(int s, int a, int b, int c)
        {
            if (Vector3.Cross(v[b] - v[a], v[c] - v[a]).y < 0f) (b, c) = (c, b);
            sub[s].Add(a);
            sub[s].Add(b);
            sub[s].Add(c);
        }

        /// <summary>A triangle wound to face along its vertices' normals (Unity's front face is clockwise as seen, i.e.
        /// along Cross(b − a, c − a)).</summary>
        private void TriOut(int s, int a, int b, int c)
        {
            Vector3 face = Vector3.Cross(v[b] - v[a], v[c] - v[a]);
            if (Vector3.Dot(face, n[a] + n[b] + n[c]) < 0f) (b, c) = (c, b);
            sub[s].Add(a);
            sub[s].Add(b);
            sub[s].Add(c);
        }

        /// <summary>A surface of revolution about the vertical through <paramref name="offset"/>: profile points are
        /// (radius, y), bottom to top; smooth around, normals from the profile's slope.</summary>
        public void Lathe(int s, IList<Vector2> profile, int sides, Vector3 offset)
        {
            int rings = profile.Count;
            var start = new int[rings];
            for (int i = 0; i < rings; i++)
            {
                Vector2 below = profile[Mathf.Max(i - 1, 0)], above = profile[Mathf.Min(i + 1, rings - 1)];
                Vector2 tangent = (above - below).normalized;
                var pn = new Vector2(tangent.y, -tangent.x);   // outward in (radius, y)
                start[i] = v.Count;
                for (int j = 0; j <= sides; j++)
                {
                    float phi = 2f * Mathf.PI * j / sides;
                    float c = Mathf.Cos(phi), sn = Mathf.Sin(phi);
                    Vector3 normal = new Vector3(pn.x * c, pn.y, pn.x * sn);
                    if (normal.sqrMagnitude < 1e-8f) normal = new Vector3(c, 0f, sn);
                    Add(offset + new Vector3(profile[i].x * c, profile[i].y, profile[i].x * sn), normal.normalized,
                        new Vector2(j / (float)sides, i / (float)Mathf.Max(rings - 1, 1)));
                }
            }
            for (int i = 0; i + 1 < rings; i++)
            {
                for (int j = 0; j < sides; j++)
                {
                    int a = start[i] + j, b = start[i] + j + 1, c = start[i + 1] + j + 1, d = start[i + 1] + j;
                    if (profile[i].x > 1e-4f || profile[i + 1].x > 1e-4f)
                    {
                        TriOut(s, a, b, c);
                        TriOut(s, a, c, d);
                    }
                }
            }
        }

        public void Box(int s, Vector3 centre, Vector3 half)
        {
            var c = new Vector3[8];
            for (int i = 0; i < 8; i++)
            {
                c[i] = centre + new Vector3((i & 1) != 0 ? half.x : -half.x, (i & 2) != 0 ? half.y : -half.y,
                                            (i & 4) != 0 ? half.z : -half.z);
            }
            Block(s, c);
        }

        /// <summary>A flat-shaded hexahedron; corner i is on the + side of axis k when bit k of i is set.</summary>
        public void Block(int s, Vector3[] c)
        {
            Vector3 centre = Vector3.zero;
            foreach (Vector3 p in c) centre += p / 8f;
            for (int axis = 0; axis < 3; axis++)
            {
                int u = (axis + 1) % 3, w = (axis + 2) % 3;
                for (int side = 0; side < 2; side++)
                {
                    int bit = side << axis;
                    int[] ring =
                    {
                        bit, bit | (1 << u), bit | (1 << u) | (1 << w), bit | (1 << w),
                    };
                    Vector3 a = c[ring[0]], b = c[ring[1]], cc = c[ring[2]], d = c[ring[3]];
                    Vector3 faceCentre = (a + b + cc + d) / 4f;
                    Vector3 normal = Vector3.Cross(b - a, cc - a).normalized;
                    if (Vector3.Dot(normal, faceCentre - centre) < 0f)
                    {
                        (b, d) = (d, b);
                        normal = -normal;
                    }
                    int i0 = Add(a, normal, Vector2.zero), i1 = Add(b, normal, Vector2.right);
                    int i2 = Add(cc, normal, Vector2.one), i3 = Add(d, normal, Vector2.up);
                    sub[s].AddRange(new[] { i0, i1, i2, i0, i2, i3 });
                }
            }
        }

        public void WriteTo(Mesh m)
        {
            m.Clear();
            m.indexFormat = v.Count > 65000 ? IndexFormat.UInt32 : IndexFormat.UInt16;
            m.SetVertices(v);
            m.SetNormals(n);
            m.SetUVs(0, uv);
            m.subMeshCount = sub.Length;
            for (int i = 0; i < sub.Length; i++) m.SetTriangles(sub[i], i);
            m.RecalculateBounds();
        }
    }

    // ================================================================ materials

    private struct Materials
    {
        public Material water, shore, turbine, mastRed, mastWhite, beacon;
    }

    /// <summary>The set's materials, each made only if missing, so a colour tuned by hand survives. The water is
    /// <c>LakeWater.mat</c>, glossy Standard with RingChallenge's wave normal map (see the class summary for why not its
    /// AQUAS material).</summary>
    private static Materials EnsureMaterials()
    {
        EnsureFolder(Folder);
        var waves = AssetDatabase.LoadAssetAtPath<Texture2D>(AquasWavesPath);
        if (waves == null && AssetDatabase.LoadAssetAtPath<Material>($"{Folder}/LakeWater.mat") == null)
        {
            Debug.Log($"LandmarkBuilder: no {AquasWavesPath} (AQUAS-Lite, git-ignored); the lake is plain glossy water.");
        }
        return new Materials
        {
            water = EnsureMaterial("LakeWater", new Color(0.10f, 0.22f, 0.27f), 0.92f, null, waves),
            shore = EnsureMaterial("LakeShore", new Color(0.50f, 0.44f, 0.33f), 0.1f),   // the damp bank
            turbine = EnsureMaterial("TurbineWhite", new Color(0.92f, 0.93f, 0.94f), 0.35f),
            mastRed = EnsureMaterial("MastRed", new Color(0.78f, 0.10f, 0.07f), 0.25f),
            mastWhite = EnsureMaterial("MastWhite", new Color(0.92f, 0.92f, 0.90f), 0.25f),
            beacon = EnsureMaterial("Beacon", new Color(1f, 0.1f, 0.05f), 0.5f, new Color(1f, 0.08f, 0.04f) * 4f),
        };
    }

    private static Material EnsureMaterial(string name, Color colour, float smoothness, Color? emission = null,
                                           Texture2D normalMap = null)
    {
        string path = $"{Folder}/{name}.mat";
        var m = AssetDatabase.LoadAssetAtPath<Material>(path);
        if (m != null) return m;
        m = new Material(Shader.Find("Standard")) { name = name, color = colour, enableInstancing = true };
        m.SetFloat("_Glossiness", smoothness);
        if (emission.HasValue)
        {
            m.EnableKeyword("_EMISSION");
            m.SetColor("_EmissionColor", emission.Value);
            m.globalIlluminationFlags = MaterialGlobalIlluminationFlags.None;
        }
        if (normalMap != null)
        {
            m.SetTexture("_BumpMap", normalMap);
            m.EnableKeyword("_NORMALMAP");
        }
        AssetDatabase.CreateAsset(m, path);
        return m;
    }

    private static void EnsureFolder(string folder)
    {
        if (AssetDatabase.IsValidFolder(folder)) return;
        string parent = Path.GetDirectoryName(folder).Replace('\\', '/');
        EnsureFolder(parent);
        AssetDatabase.CreateFolder(parent, Path.GetFileName(folder));
    }

    // ================================================================ Check

    public static void CheckAndLog(LandmarkSet set)
    {
        string report = Check(set, out int warnings);
        if (warnings > 0) Debug.LogWarning(report, set);
        else Debug.Log(report, set);
    }

    /// <summary>Whether every landmark still shows in the feeds, and the set's hygiene: per side, its top's elevation from
    /// the far and the near edge of the city (against <see cref="MinElevationDeg"/> and the frame's top), how much of its
    /// skyline the far clip cuts off from any city corner, colliders and layers that would make it part of the swarm's
    /// world, shadow casters, triangle counts and cracks at the joint with the city's own tiles.</summary>
    public static string Check(LandmarkSet set, out int warnings)
    {
        var sb = new StringBuilder();
        var warn = new List<string>();
        float eye = EyeHeight();
        float frameTop = 0.5f * FpvCameraFov.VerticalDeg(FpvCameraFov.FeedAspect);
        float farClip = FarClip();
        Vector2 c = set.cityCentre;
        float e = set.cityHalfExtent;
        sb.AppendLine($"Landmarks check: eye {eye:0.0} m, frame top {frameTop:0.0}°, far clip {farClip:0} m, city " +
                      $"±{e:0} m round ({c.x:0}, {c.y:0}).");

        // Points: the ground on a grid, and the tops of the man-made landmarks.
        var points = new List<Vector3>();
        if (set.ground != null)
        {
            foreach (Terrain t in set.ground.GetComponentsInChildren<Terrain>(true))
            {
                if (t.terrainData == null) continue;
                Vector3 o = t.transform.position, s = t.terrainData.size;
                for (float z = o.z + CheckStep * 0.5f; z < o.z + s.z; z += CheckStep)
                {
                    for (float x = o.x + CheckStep * 0.5f; x < o.x + s.x; x += CheckStep)
                    {
                        points.Add(new Vector3(x, t.SampleHeight(new Vector3(x, 0f, z)) + o.y, z));
                    }
                }
            }
        }
        var objects = new List<(string, Vector3)>();
        if (set.windFarm != null)
        {
            foreach (Transform spot in set.windFarm) objects.Add(("turbine", TopOf(spot)));
        }
        if (set.mast != null) objects.Add(("mast", TopOf(set.mast)));
        points.AddRange(objects.Select(o => o.Item2));

        sb.AppendLine("  side  top (m)  far edge  near edge  skyline past the far clip");
        foreach ((string side, Vector2 dir) in new[]
                 {
                     ("S", Vector2.down), ("N", Vector2.up), ("E", Vector2.right), ("W", Vector2.left),
                 })
        {
            List<Vector3> mine = points.Where(p => OnSide(p, dir, c)).ToList();
            if (mine.Count == 0) continue;
            Vector2 near = c + dir * e, far = c - dir * e;
            float top = mine.Max(p => p.y);
            float elevFar = MaxElevation(mine, far, eye), elevNear = MaxElevation(mine, near, eye);
            int clipped = 0;
            Vector2 worst = Vector2.zero;
            foreach (Vector2 corner in new[] { c + new Vector2(e, e), c + new Vector2(-e, e), c + new Vector2(e, -e),
                                               c + new Vector2(-e, -e) })
            {
                int k = ClippedSkyline(mine, corner, eye, farClip);
                if (k > clipped)
                {
                    clipped = k;
                    worst = corner;
                }
            }
            sb.AppendLine($"  {side,-4}  {top,7:0}  {elevFar,7:0.0}°  {elevNear,8:0.0}°  " +
                          (clipped > 0 ? $"{clipped}° from ({worst.x:0}, {worst.y:0})" : "none"));
            if (elevFar < MinElevationDeg)
            {
                warn.Add($"{side}: only {elevFar:0.0}° up from the far edge of the city, so the roofs hide it from " +
                         "there; raise it or bring it closer.");
            }
            if (elevNear > frameTop)
            {
                warn.Add($"{side}: {elevNear:0.0}° up from the near edge, above the frame's {frameTop:0.0}°, so its top " +
                         "is cut off there (fine if intended; lower it or move it out).");
            }
            if (clipped >= 3)
            {
                warn.Add($"{side}: {clipped}° of its skyline lie beyond the {farClip:0} m far clip from " +
                         $"({worst.x:0}, {worst.y:0}); bring the far crests in.");
            }
        }

        // Hygiene.
        foreach (Collider col in set.GetComponentsInChildren<Collider>(true))
        {
            if (!(col is TerrainCollider)) warn.Add($"{PathOf(col.transform, set.transform)} has a {col.GetType().Name}.");
        }
        int shadows = 0;
        foreach (Renderer r in set.GetComponentsInChildren<Renderer>(true))
        {
            int layer = r.gameObject.layer;
            if (layer == 4 || layer == 5 || layer == 6)
            {
                warn.Add($"{PathOf(r.transform, set.transform)} is on layer {LayerMask.LayerToName(layer)} " +
                         (layer == 6 ? "(a swarm obstacle)." : "(the FPV cameras do not draw it)."));
            }
            if (r.shadowCastingMode != ShadowCastingMode.Off) shadows++;
        }
        foreach (Terrain t in set.GetComponentsInChildren<Terrain>(true))
        {
            if (t.gameObject.layer == 6) warn.Add($"{PathOf(t.transform, set.transform)} is on layer Obstacle.");
        }
        if (shadows > 0) warn.Add($"{shadows} renderer(s) cast shadows.");
        foreach (Transform child in set.transform)
        {
            if (child == set.ground) continue;
            sb.AppendLine($"  {child.name}: {Triangles(child.gameObject):N0} triangles");
        }
        float crack = JointCrack(set);
        if (crack > 0.1f) warn.Add($"the ground stands {crack:0.0} m off the city's ground at the joint (a crack).");
        foreach (LakeOutline lake in set.GetComponentsInChildren<LakeOutline>(true))
        {
            float above = WaterAboveGround(lake);
            if (above < 0.2f || above > 5f)
            {
                warn.Add($"{PathOf(lake.transform, set.transform)}'s water stands {Mathf.Abs(above):0.0} m " +
                         (above < 0f ? "under" : "above") + " the ground beneath it" +
                         (HasPositionOverride(lake.transform) ? " because this scene overrides its position: " +
                                                                 "revert that override (Transform ⋮ > Revert)." : "."));
            }
        }

        warnings = warn.Count;
        if (warn.Count == 0) sb.AppendLine("  no warnings.");
        foreach (string w in warn) sb.AppendLine("  warning: " + w);
        return sb.ToString();
    }

    private static Vector3 TopOf(Transform spot)
    {
        Transform model = spot.Find(LandmarkSet.ModelName);
        Bounds b = RendererBounds(model != null ? model.gameObject : spot.gameObject);
        return new Vector3(b.center.x, b.max.y, b.center.z);
    }

    /// <summary>Whether p lies beyond the city's tiles on the side <paramref name="dir"/> points to (the corners count
    /// for both their sides).</summary>
    private static bool OnSide(Vector3 p, Vector2 dir, Vector2 c)
    {
        float along = (p.x - c.x) * dir.x + (p.z - c.y) * dir.y;
        return along > Inner;
    }

    private static float MaxElevation(List<Vector3> pts, Vector2 from, float eye)
    {
        float best = -90f;
        foreach (Vector3 p in pts)
        {
            float d = Vector2.Distance(new Vector2(p.x, p.z), from);
            best = Mathf.Max(best, Mathf.Atan2(p.y - eye, Mathf.Max(d, 1f)) * Mathf.Rad2Deg);
        }
        return best;
    }

    /// <summary>Degrees of azimuth, seen from <paramref name="from"/>, where the skyline's highest point is past the far
    /// clip, at least 1° up, and stands visibly (0.5°, ~8 px) above the best the clip leaves. Radial distance is used, which is conservative:
    /// the far plane is flat, so off-axis it reaches further.</summary>
    private static int ClippedSkyline(List<Vector3> pts, Vector2 from, float eye, float farClip)
    {
        var inside = new float[360];
        var outside = new float[360];
        for (int i = 0; i < 360; i++)
        {
            inside[i] = -90f;
            outside[i] = -90f;
        }
        foreach (Vector3 p in pts)
        {
            float dx = p.x - from.x, dz = p.z - from.y;
            float d = Mathf.Sqrt(dx * dx + dz * dz);
            int bin = (int)Mathf.Repeat(Mathf.Atan2(dx, dz) * Mathf.Rad2Deg, 360f) % 360;
            float elev = Mathf.Atan2(p.y - eye, Mathf.Max(d, 1f)) * Mathf.Rad2Deg;
            if (d <= farClip) inside[bin] = Mathf.Max(inside[bin], elev);
            else outside[bin] = Mathf.Max(outside[bin], elev);
        }
        int clipped = 0;
        for (int i = 0; i < 360; i++)
        {
            if (outside[i] > 1f && outside[i] > inside[i] + 0.5f) clipped++;
        }
        return clipped;
    }

    /// <summary>The largest height difference between the ring and the city's ground along their shared edges.</summary>
    private static float JointCrack(LandmarkSet set)
    {
        if (set.ground == null) return 0f;
        float worst = 0f;
        foreach (Terrain t in set.ground.GetComponentsInChildren<Terrain>(true))
        {
            if (t.terrainData == null) continue;
            Vector3 o = t.transform.position, s = t.terrainData.size;
            for (float u = 0f; u <= s.x; u += 4f)
            {
                foreach (Vector2 q in new[]
                         {
                             new Vector2(o.x + u, o.z), new Vector2(o.x + u, o.z + s.z), new Vector2(o.x, o.z + u),
                             new Vector2(o.x + s.x, o.z + u),
                         })
                {
                    bool onJoint = Mathf.Max(Mathf.Abs(q.x), Mathf.Abs(q.y)) <= Inner + 0.01f;
                    if (!onJoint) continue;
                    float h = t.SampleHeight(new Vector3(q.x, 0f, q.y)) + o.y;
                    worst = Mathf.Max(worst, Mathf.Abs(h - GroundY));
                }
            }
        }
        return worst;
    }

    private static float EyeHeight()
    {
        var manager = Object.FindFirstObjectByType<SwarmManager>(FindObjectsInactive.Include);
        if (manager != null && manager.maxHeightAboveTerrain > 0f)
        {
            return manager.maxHeightAboveTerrain + FlownAboveCeiling;
        }
        return DefaultEyeHeight;
    }

    private static float FarClip()
    {
        var drone = AssetDatabase.LoadAssetAtPath<GameObject>(DronePrefabPath);
        if (drone == null) return DefaultFarClip;
        Camera fpv = drone.GetComponentsInChildren<Camera>(true).FirstOrDefault(cam => cam.name == "FPV");
        return fpv != null ? fpv.farClipPlane : DefaultFarClip;
    }

    private static string PathOf(Transform t, Transform root)
    {
        var parts = new List<string>();
        for (Transform x = t; x != null && x != root; x = x.parent) parts.Add(x.name);
        parts.Reverse();
        return string.Join("/", parts);
    }
}

/// <summary>
/// Keeps every <see cref="LakeOutline"/> and the city tiles round it current in the editor: its mesh is rebuilt at once
/// when its spline changes, a field of it is edited, or after an undo; and the city tiles' trees and beaches
/// (<see cref="LandmarkBuilder.ClearTreesUnderLakes"/>) are redone a moment after the last such edit, after the lake (or
/// anything above it) is moved, and after anything is deleted from a scene whose tiles a lake has touched, so the trees
/// a deleted or moved lake cleared come back. The pause keeps a drag from repainting the tiles every frame.
/// </summary>
[InitializeOnLoad]
internal static class LakeOutlineHook
{
    private const double TilesDelaySeconds = 0.4;
    private static readonly HashSet<Scene> pendingTiles = new HashSet<Scene>();
    private static double tilesDue;

    static LakeOutlineHook()
    {
        Spline.Changed += OnSplineChanged;
        Undo.undoRedoPerformed += OnUndoRedo;
        ObjectChangeEvents.changesPublished += OnChangesPublished;
    }

    private static void OnChangesPublished(ref ObjectChangeEventStream stream)
    {
        if (EditorApplication.isPlayingOrWillChangePlaymode) return;
        for (int i = 0; i < stream.length; i++)
        {
            switch (stream.GetEventType(i))
            {
                case ObjectChangeKind.ChangeGameObjectOrComponentProperties:
                {
                    stream.GetChangeGameObjectOrComponentPropertiesEvent(i, out ChangeGameObjectOrComponentPropertiesEventArgs e);
                    Object changed = EditorUtility.InstanceIDToObject(e.instanceId);
                    if (changed is LakeOutline lake)
                    {
                        LandmarkBuilder.RebuildLake(lake);
                        ScheduleTiles(lake.gameObject.scene);
                    }
                    else if (changed is Transform t && t.GetComponentInChildren<LakeOutline>(true) != null)
                    {
                        ScheduleTiles(t.gameObject.scene);   // moved: the mesh is in the lake's own frame
                    }
                    break;
                }
                case ObjectChangeKind.DestroyGameObjectHierarchy:
                {
                    stream.GetDestroyGameObjectHierarchyEvent(i, out DestroyGameObjectHierarchyEventArgs e);
                    if (LandmarkBuilder.HasCityTileCopies(e.scene)) ScheduleTiles(e.scene);
                    break;
                }
            }
        }
    }

    private static void OnSplineChanged(Spline spline, int knot, SplineModification modification)
    {
        if (EditorApplication.isPlayingOrWillChangePlaymode) return;
        foreach (LakeOutline lake in LandmarkBuilder.RebuildLakes(spline)) ScheduleTiles(lake.gameObject.scene);
    }

    private static void OnUndoRedo()
    {
        if (EditorApplication.isPlayingOrWillChangePlaymode) return;
        foreach (LakeOutline lake in LandmarkBuilder.RebuildLakes()) ScheduleTiles(lake.gameObject.scene);
        for (int i = 0; i < SceneManager.sceneCount; i++)
        {
            Scene scene = SceneManager.GetSceneAt(i);
            if (LandmarkBuilder.HasCityTileCopies(scene)) ScheduleTiles(scene);   // an undone delete or move
        }
    }

    private static void ScheduleTiles(Scene scene)
    {
        PrefabStage stage = PrefabStageUtility.GetCurrentPrefabStage();
        if (!scene.IsValid() || (stage != null && stage.scene == scene)) return;   // a prefab has no city tiles
        pendingTiles.Add(scene);
        tilesDue = EditorApplication.timeSinceStartup + TilesDelaySeconds;
        EditorApplication.update -= FlushTiles;
        EditorApplication.update += FlushTiles;
    }

    private static void FlushTiles()
    {
        if (EditorApplication.timeSinceStartup < tilesDue) return;
        EditorApplication.update -= FlushTiles;
        Scene[] scenes = pendingTiles.ToArray();
        pendingTiles.Clear();
        if (EditorApplication.isPlayingOrWillChangePlaymode) return;
        foreach (Scene scene in scenes)
        {
            if (scene.IsValid() && scene.isLoaded) LandmarkBuilder.ClearTreesUnderLakes(scene);
        }
    }
}
