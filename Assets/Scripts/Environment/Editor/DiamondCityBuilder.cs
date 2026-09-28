using System.Collections.Generic;
using System.IO;
using System.Linq;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;
using UnityEngine.SceneManagement;

/// <summary>
/// Builds <c>DiamondCityWorld</c>: ScaledCityWorld's 48 tiles re-laid on an 8 x 8 grid, with four <b>diamonds</b> in
/// the 16 slots that leaves over. A diamond is a superblock standing across the north–south corridors that the
/// half-tile stagger leaves straight — a stagger can only ever break one axis — to cut the sight lines down them.
///
/// <para><b>The layout is data</b>: <see cref="Layout"/> says what stands in every slot, and the build only carries it
/// out. The slots keep ScaledCityWorld's pitch, columns and stagger (odd columns half a tile north) with one row added
/// at each end, so the spawn still looks down the street between MC_Patch_24 and MC_Patch_32 onto MC_Patch_23. Each
/// tile is moved by its kerb, not its transform (see <see cref="CityTiles"/>), and takes its tied scenery with it. That
/// keeps the medians whole: every tile owns the medians along its east and north edges, so any arrangement of tiles
/// still has exactly one down every street.</para>
///
/// <para><b>A diamond is four blocks</b>: two stacked in one column and one either side, level with the street between
/// the two — the one way four blocks close up round a centre in this stagger, and the same shape whichever column it
/// is centred on. The first build drafts each one from four ScaledCityWorld tiles that stood in exactly that shape
/// (<see cref="Sources"/>) and saves it as a prefab in <see cref="DiamondFolder"/>. <b>Every later build places the
/// prefab as it stands</b>, so restyling a diamond survives a rebuild; delete its prefab to get the draft back. A draft
/// keeps the streets between its four blocks, so it cuts nothing until they are built over.</para>
///
/// <para><b>A diamond is not a tile.</b> The copies' kerbs are renamed <c>Kerb_NN</c> and their content nodes
/// <c>Block_NN</c>, and the diamonds hang under a root of their own beside the city's. The city tools find tiles by
/// exactly those names — the tuners and the verge planter by kerb, the goal replacer and the obstacle export by
/// <c>MC_Patch</c> — so left alone a diamond would be a goal candidate, would be shrunk a second time, and with kerbs
/// under two roots would stop <see cref="CityTiles.FindCity(Scene, out string)"/> finding a city at all. The buildings
/// keep their names, layer and colliders, so they are still obstacles to the swarm.</para>
///
/// <para><see cref="CityRowOffsetter"/> is removed from the built scene. It re-staggers tiles from where they stand and
/// cannot see the diamonds, so a single nudge of it would slide whole columns of tiles into them.</para>
/// </summary>
public static class DiamondCityBuilder
{
    private const string SourceScenePath = "Assets/Scenes/ScaledCityWorld.unity";
    public const string ScenePath = "Assets/Scenes/DiamondCityWorld.unity";
    public const string DiamondFolder = "Assets/Prefabs/ScaledCity/Diamonds";
    public const string DiamondsRootName = "Diamonds";

    private const string TileNamePrefix = "MC_Patch";
    private const string DiamondKerbPrefix = "Kerb_";
    private const string DiamondContentPrefix = "Block";

    private const int Columns = 8;
    private const int Rows = 8;
    private const int SourceRows = 6;
    private const int RowsAddedSouth = 1; // and one on the north edge
    private const float Pitch = CityTiles.Pitch;

    /// <summary>How far any source kerb may sit off the grid fitted through them all before the build refuses.</summary>
    private const float MaxFitResidual = 1f;

    /// <summary>
    /// The city, north row first. Columns run west to east, and odd columns sit half a tile north of even ones, as in
    /// ScaledCityWorld. A number is the MC_Patch tile standing there — with a trailing <c>'</c>, the scene's second copy
    /// of that design, <c>MC_Patch_NN (1)</c>; ScaledCityWorld has two, where the pack put 37 and 08 — and a letter is
    /// one of a diamond's four blocks.
    ///
    /// <para>Found by search over every placement of four diamonds that do not touch: each of the seven north–south
    /// corridors runs into at least one, the spawn's first junction stays an ordinary T, and once the diamonds' inner
    /// streets are built over, the longest straight run left down any corridor is ~515 units (ScaledCityWorld's, in a
    /// city two rows shorter, run ~590). The tiles fill the other slots by least total movement from where they stood,
    /// so most neighbourhoods survive, shifted a row.</para>
    /// </summary>
    private static readonly string[] Layout =
    {
        // c0 c1 c2 c3 c4 c5 c6 c7
        "04 47 46 45 12 43 42 41", // row 7, north edge
        "40 39 B  10' 36 35 D  15",
        "31 B  B  B  28 D  D  D ",
        "32 38 30 29 C  27 26 34",
        "24 23 22 C  C  C  19 25", // the spawn looks east between 24 and 32, onto 23
        "A  A  A  21 13 20 18 17",
        "16 A  48 14 33 11 10 09",
        "35' 07 06 05 44 03 02 01", // row 0, south edge
    };

    private enum Role
    {
        South, // the lower of the two middle blocks
        North, // the upper
        West,
        East,
    }

    private static readonly Role[] Roles = { Role.South, Role.North, Role.West, Role.East };

    private readonly struct DiamondSource
    {
        public readonly char Letter;
        private readonly string[] tiles; // indexed by Role

        public DiamondSource(char letter, string south, string north, string west, string east)
        {
            Letter = letter;
            tiles = new[] { south, north, west, east };
        }

        public string TileName(Role role) => $"{TileNamePrefix}_{tiles[(int)role]}";
    }

    /// <summary>
    /// The tiles each draft copies: four that stood in exactly the diamond's shape in ScaledCityWorld, so each draft is
    /// a piece of the old city lifted out whole. They are matched to diamonds far from where their own tiles now stand
    /// — every copy at least 2.7 tiles from its original — so no draft sets a block beside its twin. D is 21/29/30/28,
    /// the example the design started from.
    /// </summary>
    private static readonly DiamondSource[] Sources =
    {
        new DiamondSource('A', south: "27", north: "35", west: "36", east: "34"),
        new DiamondSource('B', south: "48", north: "23", west: "24", east: "22"),
        new DiamondSource('C', south: "31", north: "39", west: "40", east: "38"),
        new DiamondSource('D', south: "21", north: "29", west: "30", east: "28"),
    };

    private sealed class Diamond
    {
        public DiamondSource Source;
        public int Column;   // the two middle blocks'
        public int SouthRow; // the lower middle block's
        public int SideRow;  // the two side blocks'

        public string PrefabPath => $"{DiamondFolder}/Diamond_{Source.Letter}.prefab";
    }

    private sealed class Plan
    {
        public readonly List<KeyValuePair<string, Vector2Int>> Tiles = new List<KeyValuePair<string, Vector2Int>>();
        public readonly List<Diamond> Diamonds = new List<Diamond>();
    }

    /// <summary>The new grid in the city root's frame: where the block centre of slot (column, row) stands.</summary>
    private readonly struct SlotGrid
    {
        private readonly float x0;
        private readonly float z0;

        public SlotGrid(float x0, float z0)
        {
            this.x0 = x0;
            this.z0 = z0;
        }

        public Vector3 Slot(int column, int row)
        {
            float stagger = (column & 1) == 1 ? 0.5f : 0f;
            return new Vector3(x0 + column * Pitch, 0f, z0 + (row + stagger) * Pitch);
        }
    }

    [MenuItem("Tools/Swarm/Build diamond city")]
    private static void BuildFromMenu()
    {
        if (!EditorSceneManager.SaveCurrentModifiedScenesIfUserWantsTo())
        {
            return;
        }
        if (File.Exists(ScenePath) &&
            !EditorUtility.DisplayDialog("Build diamond city",
                $"Rebuild {ScenePath} from {SourceScenePath}? This replaces the scene, including anything changed in " +
                $"it by hand. The diamond prefabs in {DiamondFolder} are placed as they are, not rebuilt.",
                "Rebuild", "Cancel"))
        {
            return;
        }
        Build();
    }

    [MenuItem("Tools/Swarm/Build diamond city", true)]
    private static bool CanBuild()
    {
        return !EditorApplication.isPlayingOrWillChangePlaymode;
    }

    /// <summary>Batch-mode entry point: <c>-executeMethod DiamondCityBuilder.BuildInBatch</c>. Exits 0 on success.</summary>
    public static void BuildInBatch()
    {
        EditorApplication.Exit(Build() ? 0 : 1);
    }

    /// <summary>
    /// Rebuilds the diamond city from ScaledCityWorld, saves it and leaves it open. On failure it logs why, puts back
    /// whatever scene was there before, and returns false. ScaledCityWorld itself is never opened.
    /// </summary>
    public static bool Build()
    {
        Plan plan = ParseLayout(out string error);
        if (plan == null)
        {
            Debug.LogError("DiamondCityBuilder: " + error);
            return false;
        }

        SceneSetup[] previousSetup = EditorSceneManager.GetSceneManagerSetup();
        byte[] previousScene = File.Exists(ScenePath) ? File.ReadAllBytes(ScenePath) : null;

        // Nothing may hold the scene open while its file is replaced underneath it.
        EditorSceneManager.NewScene(NewSceneSetup.EmptyScene, NewSceneMode.Single);
        File.Copy(SourceScenePath, ScenePath, true);
        AssetDatabase.ImportAsset(ScenePath, ImportAssetOptions.ForceUpdate);
        Scene scene = EditorSceneManager.OpenScene(ScenePath, OpenSceneMode.Single);

        string summary;
        try
        {
            summary = Populate(scene, plan, out error);
            if (summary != null && !EditorSceneManager.SaveScene(scene))
            {
                summary = null;
                error = $"could not save {ScenePath}.";
            }
        }
        catch (System.Exception e)
        {
            summary = null;
            error = e.ToString();
        }

        if (summary == null)
        {
            Debug.LogError("DiamondCityBuilder: " + error + " The scene was left as it was.");
            EditorSceneManager.NewScene(NewSceneSetup.EmptyScene, NewSceneMode.Single);
            if (previousScene != null)
            {
                File.WriteAllBytes(ScenePath, previousScene);
                AssetDatabase.ImportAsset(ScenePath, ImportAssetOptions.ForceUpdate);
            }
            else
            {
                AssetDatabase.DeleteAsset(ScenePath);
            }
            // An untitled scene has no path to reopen it from.
            if (previousSetup.Length > 0 &&
                System.Array.TrueForAll(previousSetup, s => !string.IsNullOrEmpty(s.path)))
            {
                EditorSceneManager.RestoreSceneManagerSetup(previousSetup);
            }
            return false;
        }

        Debug.Log($"DiamondCityBuilder: built {ScenePath} from {SourceScenePath}: {summary}");
        return true;
    }

    /// <summary>
    /// Turns a fresh copy of ScaledCityWorld into the diamond city. Returns a one-line account of what it did, or null
    /// with <paramref name="error"/> set. Everything is checked before anything is created or moved.
    /// </summary>
    private static string Populate(Scene scene, Plan plan, out string error)
    {
        CityTiles.City city = CityTiles.FindCity(scene, out error);
        if (city == null)
        {
            return null;
        }

        Dictionary<string, int> index = new Dictionary<string, int>();
        for (int i = 0; i < city.Tiles.Count; i++)
        {
            if (index.ContainsKey(city.Tiles[i].name))
            {
                error = $"{SourceScenePath} has two tiles named {city.Tiles[i].name}.";
                return null;
            }
            index.Add(city.Tiles[i].name, i);
        }
        List<string> placed = plan.Tiles.Select(t => t.Key).ToList();
        List<string> missing = placed.Where(n => !index.ContainsKey(n)).ToList();
        List<string> unplaced = index.Keys.Where(n => !placed.Contains(n)).ToList();
        if (missing.Count > 0 || unplaced.Count > 0)
        {
            error = "the layout and the city disagree: " +
                    (missing.Count > 0 ? $"no tile in the city for {string.Join(", ", missing)}; " : "") +
                    (unplaced.Count > 0 ? $"no slot in the layout for {string.Join(", ", unplaced)}." : "");
            return null;
        }
        foreach (Diamond diamond in plan.Diamonds)
        {
            foreach (Role role in Roles)
            {
                if (!index.ContainsKey(diamond.Source.TileName(role)))
                {
                    error = $"diamond {diamond.Source.Letter} copies {diamond.Source.TileName(role)}, which the city lacks.";
                    return null;
                }
            }
        }
        if (scene.GetRootGameObjects().Any(go => go.name == DiamondsRootName))
        {
            error = $"{SourceScenePath} already has a root named {DiamondsRootName}.";
            return null;
        }

        SlotGrid? fitted = FitGrid(city, out float residual, out error);
        if (fitted == null)
        {
            return null;
        }
        SlotGrid grid = fitted.Value;

        int removed = 0;
        foreach (CityRowOffsetter offsetter in Object.FindObjectsByType<CityRowOffsetter>(FindObjectsInactive.Include,
                                                                                        FindObjectsSortMode.None))
        {
            Object.DestroyImmediate(offsetter);
            removed++;
        }

        // In the city's own frame, so a slot means the same point under both roots.
        Transform diamondsRoot = new GameObject(DiamondsRootName).transform;
        diamondsRoot.SetParent(city.Root.parent, false);
        diamondsRoot.localPosition = city.Root.localPosition;
        diamondsRoot.localRotation = city.Root.localRotation;
        diamondsRoot.localScale = city.Root.localScale;
        diamondsRoot.SetSiblingIndex(city.Root.GetSiblingIndex() + 1);

        int built = 0;
        int reused = 0;
        int flagsSet = 0;
        foreach (Diamond diamond in plan.Diamonds)
        {
            // Level with the street between the two middle blocks, which is where the side blocks stand.
            Vector3 centre = grid.Slot(diamond.Column, diamond.SouthRow) + new Vector3(0f, 0f, 0.5f * Pitch);

            GameObject prefab = AssetDatabase.LoadAssetAtPath<GameObject>(diamond.PrefabPath);
            GameObject instance;
            if (prefab != null)
            {
                instance = (GameObject)PrefabUtility.InstantiatePrefab(prefab, diamondsRoot);
                reused++;
            }
            else
            {
                instance = BuildDraft(diamond, city, index, diamondsRoot, centre, ref flagsSet, out error);
                if (instance == null)
                {
                    return null;
                }
                built++;
            }
            instance.transform.localPosition = centre;
            instance.transform.localRotation = Quaternion.identity;
            instance.transform.localScale = Vector3.one;
        }

        int moved = 0;
        float furthest = 0f;
        foreach (KeyValuePair<string, Vector2Int> entry in plan.Tiles)
        {
            int i = index[entry.Key];
            Transform tile = city.Tiles[i];
            Vector3 shift = grid.Slot(entry.Value.x, entry.Value.y) - city.Root.InverseTransformPoint(city.Centres[i]);
            shift.y = 0f; // seven tiles carry a baked height offset, which must survive the move
            if (shift.sqrMagnitude < 1e-8f)
            {
                continue;
            }
            tile.localPosition += shift; // a tile is a child of the city root, whose frame the shift is in
            if (PrefabUtility.IsPartOfPrefabInstance(tile))
            {
                PrefabUtility.RecordPrefabInstancePropertyModifications(tile);
            }
            moved++;
            furthest = Mathf.Max(furthest, shift.magnitude);
        }

        EditorSceneManager.MarkSceneDirty(scene);
        return $"re-laid {plan.Tiles.Count} tiles on {Columns} x {Rows} slots ({moved} moved, the furthest " +
               $"{furthest:F1} units; the source kerbs sat within {residual:F2} units of the fitted grid), drafted " +
               $"{built} diamond(s) and placed {reused} existing diamond prefab(s) under {DiamondsRootName}" +
               (flagsSet > 0 ? $" ({flagsSet} copied objects had their static flags restored)" : "") +
               $", removed {removed} CityRowOffsetter(s).";
    }

    /// <summary>
    /// Fits ScaledCityWorld's grid through its kerbs — <see cref="Columns"/> columns of <see cref="SourceRows"/>, odd
    /// columns half a tile north — and returns the new grid, <see cref="RowsAddedSouth"/> row(s) further south. Null,
    /// with <paramref name="error"/> set, if the kerbs are not on such a grid.
    /// </summary>
    private static SlotGrid? FitGrid(CityTiles.City city, out float residual, out string error)
    {
        residual = 0f;
        int count = city.Tiles.Count;
        Vector3[] local = city.Centres.Select(c => city.Root.InverseTransformPoint(c)).ToArray();

        float minX = local.Min(p => p.x);
        int[] column = local.Select(p => Mathf.RoundToInt((p.x - minX) / Pitch)).ToArray();
        float[] unstaggered = new float[count];
        for (int i = 0; i < count; i++)
        {
            unstaggered[i] = local[i].z - ((column[i] & 1) == 1 ? 0.5f * Pitch : 0f);
        }
        float minZ = unstaggered.Min();
        int[] row = unstaggered.Select(z => Mathf.RoundToInt((z - minZ) / Pitch)).ToArray();

        HashSet<Vector2Int> used = new HashSet<Vector2Int>();
        for (int i = 0; i < count; i++)
        {
            if (column[i] >= Columns || row[i] >= SourceRows || !used.Add(new Vector2Int(column[i], row[i])))
            {
                error = $"{city.Tiles[i].name}'s kerb is not on a {Columns} x {SourceRows} grid with odd columns half " +
                        "a tile north, which is the city this layout was written for.";
                return null;
            }
        }

        float x0 = 0f;
        float z0 = 0f;
        for (int i = 0; i < count; i++)
        {
            x0 += local[i].x - column[i] * Pitch;
            z0 += unstaggered[i] - row[i] * Pitch;
        }
        x0 /= count;
        z0 /= count;
        for (int i = 0; i < count; i++)
        {
            residual = Mathf.Max(residual,
                                 Mathf.Abs(local[i].x - (x0 + column[i] * Pitch)),
                                 Mathf.Abs(unstaggered[i] - (z0 + row[i] * Pitch)));
        }
        if (residual > MaxFitResidual)
        {
            error = $"a kerb sits {residual:F2} units off the grid fitted through the city (at most {MaxFitResidual} " +
                    "allowed), so the tiles are not on the staggered grid this layout was written for.";
            return null;
        }

        error = null;
        return new SlotGrid(x0, z0 - RowsAddedSouth * Pitch);
    }

    /// <summary>
    /// Drafts a diamond from copies of its <see cref="Diamond.Source"/> tiles, each moved so its kerb lands on its
    /// block's place, and saves it as the diamond's prefab. Returns the connected instance, or null with
    /// <paramref name="error"/> set.
    /// </summary>
    private static GameObject BuildDraft(Diamond diamond, CityTiles.City city, Dictionary<string, int> index,
                                         Transform diamondsRoot, Vector3 centre, ref int flagsSet, out string error)
    {
        GameObject root = new GameObject(Path.GetFileNameWithoutExtension(diamond.PrefabPath));
        root.transform.SetParent(diamondsRoot, false);
        root.transform.localPosition = centre;

        foreach (Role role in Roles)
        {
            Transform source = city.Tiles[index[diamond.Source.TileName(role)]];

            // Under the diamonds' root exactly as the tile stands under the city's, so the copy starts on the tile. A
            // copy of the tile as it is in the scene, not a fresh prefab instance: its block scale, layers and static
            // flags live in scene overrides, and the diamond is to be restyled freely, so it keeps no prefab link.
            GameObject copy = Object.Instantiate(source.gameObject, diamondsRoot);
            if (PrefabUtility.IsPartOfPrefabInstance(copy))
            {
                PrefabUtility.UnpackPrefabInstance(copy, PrefabUnpackMode.Completely, InteractionMode.AutomatedAction);
            }
            flagsSet += MatchStaticFlags(source, copy.transform);

            Transform kerb = CityTiles.FindKerb(copy.transform);
            if (kerb == null)
            {
                Object.DestroyImmediate(copy);
                Object.DestroyImmediate(root);
                error = $"{source.name} has no {CityTiles.KerbPrefix}NN kerb to place its copy by.";
                return null;
            }
            Vector3 shift = centre + Offset(role) - diamondsRoot.InverseTransformPoint(kerb.position);
            shift.y = 0f;
            copy.transform.localPosition += shift;
            copy.transform.SetParent(root.transform, true);

            copy.name = $"{role} ({source.name})";
            RenameTileMarkers(copy.transform);
        }

        EnsureFolder(DiamondFolder);
        PrefabUtility.SaveAsPrefabAssetAndConnect(root, diamond.PrefabPath, InteractionMode.AutomatedAction,
                                                  out bool saved);
        if (!saved)
        {
            Object.DestroyImmediate(root);
            error = $"could not save {diamond.PrefabPath}.";
            return null;
        }
        error = null;
        return root;
    }

    /// <summary>Where a block's centre stands relative to its diamond's centre, in the city's frame.</summary>
    private static Vector3 Offset(Role role)
    {
        switch (role)
        {
            case Role.South: return new Vector3(0f, 0f, -0.5f * Pitch);
            case Role.North: return new Vector3(0f, 0f, 0.5f * Pitch);
            case Role.West: return new Vector3(-Pitch, 0f, 0f);
            default: return new Vector3(Pitch, 0f, 0f);
        }
    }

    /// <summary>
    /// Renames what the city tools recognise a tile by — its kerb and its <c>MC_Patch</c> content node — so a copied
    /// block is not taken for one. See the class summary for what each tool would otherwise do.
    /// </summary>
    private static void RenameTileMarkers(Transform block)
    {
        foreach (Transform t in block.GetComponentsInChildren<Transform>(true))
        {
            if (t == block)
            {
                continue;
            }
            if (t.name.StartsWith(CityTiles.KerbPrefix))
            {
                t.name = DiamondKerbPrefix + t.name.Substring(CityTiles.KerbPrefix.Length);
            }
            else if (t.name.StartsWith(TileNamePrefix))
            {
                t.name = DiamondContentPrefix + t.name.Substring(TileNamePrefix.Length);
            }
        }
    }

    /// <summary>
    /// Gives every object in <paramref name="copy"/> the static flags of its original, and returns how many needed it.
    /// The city is Batching Static throughout; a block left out of the batch would cost a draw call per material in
    /// every FPV camera that sees it.
    /// </summary>
    private static int MatchStaticFlags(Transform original, Transform copy)
    {
        Transform[] from = original.GetComponentsInChildren<Transform>(true);
        Transform[] to = copy.GetComponentsInChildren<Transform>(true);
        if (from.Length != to.Length)
        {
            throw new System.InvalidOperationException(
                $"the copy of {original.name} has {to.Length} objects where the tile has {from.Length}.");
        }
        int set = 0;
        for (int i = 0; i < from.Length; i++)
        {
            StaticEditorFlags flags = GameObjectUtility.GetStaticEditorFlags(from[i].gameObject);
            if (GameObjectUtility.GetStaticEditorFlags(to[i].gameObject) != flags)
            {
                GameObjectUtility.SetStaticEditorFlags(to[i].gameObject, flags);
                set++;
            }
        }
        return set;
    }

    /// <summary>Reads <see cref="Layout"/> and <see cref="Sources"/>. Null, with <paramref name="error"/> set, if they are malformed.</summary>
    private static Plan ParseLayout(out string error)
    {
        if (Layout.Length != Rows)
        {
            error = $"the layout has {Layout.Length} rows, not {Rows}.";
            return null;
        }

        Plan plan = new Plan();
        HashSet<string> seen = new HashSet<string>();
        Dictionary<char, List<Vector2Int>> diamondCells = new Dictionary<char, List<Vector2Int>>();
        for (int line = 0; line < Rows; line++)
        {
            int row = Rows - 1 - line;
            string[] cells = Layout[line].Split((char[])null, System.StringSplitOptions.RemoveEmptyEntries);
            if (cells.Length != Columns)
            {
                error = $"layout row {row} has {cells.Length} cells, not {Columns}.";
                return null;
            }
            for (int column = 0; column < Columns; column++)
            {
                string cell = cells[column];
                Vector2Int slot = new Vector2Int(column, row);
                if (cell.Length == 1 && char.IsLetter(cell[0]))
                {
                    if (!diamondCells.TryGetValue(cell[0], out List<Vector2Int> list))
                    {
                        diamondCells[cell[0]] = list = new List<Vector2Int>();
                    }
                    list.Add(slot);
                }
                else if (TryTileName(cell, out string name))
                {
                    if (!seen.Add(name))
                    {
                        error = $"{name} appears twice in the layout.";
                        return null;
                    }
                    plan.Tiles.Add(new KeyValuePair<string, Vector2Int>(name, slot));
                }
                else
                {
                    error = $"layout cell '{cell}' (column {column}, row {row}) is neither a tile number nor a diamond letter.";
                    return null;
                }
            }
        }

        foreach (DiamondSource source in Sources)
        {
            if (!diamondCells.TryGetValue(source.Letter, out List<Vector2Int> cells))
            {
                error = $"diamond {source.Letter} has no cells in the layout.";
                return null;
            }
            Diamond diamond = ShapeOf(cells, source, out error);
            if (diamond == null)
            {
                return null;
            }
            plan.Diamonds.Add(diamond);
            diamondCells.Remove(source.Letter);
        }
        if (diamondCells.Count > 0)
        {
            error = $"layout letter {diamondCells.Keys.First()} has no entry in Sources.";
            return null;
        }

        error = null;
        return plan;
    }

    /// <summary>A layout number as a tile name: <c>04</c> is MC_Patch_04, and <c>10'</c> is MC_Patch_10 (1).</summary>
    private static bool TryTileName(string cell, out string name)
    {
        bool secondCopy = cell.EndsWith("'");
        string number = secondCopy ? cell.Substring(0, cell.Length - 1) : cell;
        if (number.Length == 0 || !number.All(char.IsDigit))
        {
            name = null;
            return false;
        }
        name = $"{TileNamePrefix}_{number}" + (secondCopy ? " (1)" : "");
        return true;
    }

    /// <summary>
    /// The diamond four cells make: two stacked in one column, and one in each neighbouring column level with the street
    /// between them — for an odd centre column the row above the lower one, for an even one the same row, since odd
    /// columns sit half a tile north. Null, with <paramref name="error"/> set, for any other shape.
    /// </summary>
    private static Diamond ShapeOf(List<Vector2Int> cells, DiamondSource source, out string error)
    {
        Dictionary<int, List<int>> byColumn = cells.GroupBy(c => c.x)
                                                   .ToDictionary(g => g.Key, g => g.Select(c => c.y).OrderBy(y => y).ToList());
        if (cells.Count == 4 && byColumn.Count == 3)
        {
            foreach (KeyValuePair<int, List<int>> middle in byColumn)
            {
                List<int> rows = middle.Value;
                if (rows.Count != 2 || rows[1] != rows[0] + 1)
                {
                    continue;
                }
                int sideRow = (middle.Key & 1) == 1 ? rows[0] + 1 : rows[0];
                if (byColumn.TryGetValue(middle.Key - 1, out List<int> west) && west.Count == 1 && west[0] == sideRow &&
                    byColumn.TryGetValue(middle.Key + 1, out List<int> east) && east.Count == 1 && east[0] == sideRow)
                {
                    error = null;
                    return new Diamond { Source = source, Column = middle.Key, SouthRow = rows[0], SideRow = sideRow };
                }
            }
        }
        error = $"diamond {source.Letter}'s cells ({string.Join(", ", cells)}) are not two stacked blocks with one on " +
                "either side, level with the street between them.";
        return null;
    }

    private static void EnsureFolder(string folder)
    {
        if (AssetDatabase.IsValidFolder(folder))
        {
            return;
        }
        string parent = Path.GetDirectoryName(folder).Replace('\\', '/');
        EnsureFolder(parent);
        AssetDatabase.CreateFolder(parent, Path.GetFileName(folder));
    }
}
