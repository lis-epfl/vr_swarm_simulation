using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;

/// <summary>
/// <c>Tools/Swarm/Block-shaped city…</c>: change BlockShapedCityWorld's design and see it. Generates a layout with
/// <c>citysim.blockcity one --variant B --scene</c> (run in the stitching env) at the block spacing p, street width s,
/// footpath width, seed placement and seed set here, writes it as <c>layouts/BlockShaped__&lt;id&gt;.json</c> (the id
/// names every setting that is not S4's), and bakes it into the open scene's <see cref="ProceduralCityBuilder"/>, so the
/// city is there in the editor and is what Play flies. Save the scene to keep it.
///
/// <para><b>How big a building a design leaves.</b> A block is about p − s across, and its building that less a footpath
/// each side; a block whose building would be under 8 m across is left a plaza (blockcity.MIN_BUILDING_M). S4's p 40 s 18
/// leaves ~15 m buildings in 90% of the blocks; p 35 s 20 leaves too little for most (32% of interior blocks built on
/// S4's seeds), so narrow the footpath or space the blocks more evenly there.</para>
///
/// <para><b>How many streets round a building.</b> A Voronoi block always has about six sides, but a side shorter than
/// its two junctions' setbacks is not a street: its junctions merge. On S4's seeds at p 40 s 18 a building has 4.3
/// streets round it and 8.2 leaving its corners; seeds on a jittered triangular lattice (<i>Hex</i>) give 5.7 and 6.3.
/// More sides means more corners to go round, more merged junctions more streets to choose from at each. The generator
/// prints both for every layout.</para>
/// </summary>
public class BlockShapedCityWindow : EditorWindow
{
    private enum Seeding
    {
        RSA,
        Hex,
    }

    private const float MinBuilding = 8f;                 // blockcity.MIN_BUILDING_M

    [SerializeField] private float blockSpacing = 40f;
    [SerializeField] private float streetWidth = 18f;
    [SerializeField] private float footpath = 3.5f;
    [SerializeField] private Seeding seeding = Seeding.RSA;
    [SerializeField] private float jitter = 0.15f;
    [SerializeField] private int relax;
    [SerializeField] private int seed = 1;
    [SerializeField] private int layoutIndex;
    [SerializeField] private string report = "";
    private string[] layoutFiles = new string[0];
    private Vector2 scroll;

    [MenuItem("Tools/Swarm/Block-shaped city…")]
    private static void Open() => GetWindow<BlockShapedCityWindow>("Block-shaped city");

    private void OnEnable() => RefreshLayouts();

    private void OnFocus() => RefreshLayouts();

    private void OnGUI()
    {
        ProceduralCityBuilder builder = FindFirstObjectByType<ProceduralCityBuilder>(FindObjectsInactive.Include);
        scroll = EditorGUILayout.BeginScrollView(scroll);
        if (builder == null)
        {
            EditorGUILayout.HelpBox($"The open scene has no ProceduralCityBuilder: open {BlockShapedCityWorldBuilder.ScenePath} " +
                                    "(Tools/Swarm/Build block-shaped city world makes it).", MessageType.Warning);
        }
        else
        {
            EditorGUILayout.LabelField("Scene", builder.gameObject.scene.name);
            EditorGUILayout.LabelField("Baked layout", string.IsNullOrEmpty(builder.BakedLayout) ? "(none)" : builder.BakedLayout,
                                       EditorStyles.wordWrappedLabel);
        }

        EditorGUILayout.Space();
        EditorGUILayout.LabelField("Design", EditorStyles.boldLabel);
        blockSpacing = EditorGUILayout.FloatField(new GUIContent("Block spacing p (m)",
            "One block seed per p x p m: neighbouring blocks' centres are about p apart. S4 flew 40, 50, 60 and 75."), blockSpacing);
        streetWidth = EditorGUILayout.FloatField(new GUIContent("Street width s (m)",
            "Kerb to kerb. S4 flew 12, 18 and 24."), streetWidth);
        footpath = EditorGUILayout.FloatField(new GUIContent("Footpath (m)",
            "Kerb to facade: the building is the kerb inset by this. S4's variant B: 3.5, CityDressing's footpath."), footpath);
        float building = blockSpacing - streetWidth - 2f * footpath;
        EditorGUILayout.LabelField(" ", $"blocks ≈ {blockSpacing - streetWidth:0} m across, buildings ≈ {building:0} m",
                                   EditorStyles.miniLabel);
        if (building < MinBuilding + 4f)
        {
            EditorGUILayout.HelpBox($"A building under {MinBuilding:0} m across is left out (a plaza), and Voronoi blocks vary: " +
                                    "at this size many blocks will be plazas. A narrower footpath or wider spacing keeps them.",
                                    MessageType.Info);
        }
        using (new EditorGUILayout.HorizontalScope())
        {
            GUILayout.Space(EditorGUIUtility.labelWidth);
            if (GUILayout.Button(new GUIContent("S4: p 40 · s 18", "The design S4's Phase 2 flew, with B's 3.5 m footpath.")))
            {
                blockSpacing = 40f;
                streetWidth = 18f;
                footpath = 3.5f;
            }
            if (GUILayout.Button("p 40 · s 24"))
            {
                blockSpacing = 40f;
                streetWidth = 24f;
                footpath = 3.5f;
            }
        }

        EditorGUILayout.Space();
        EditorGUILayout.LabelField("Streets", EditorStyles.boldLabel);
        seeding = (Seeding)EditorGUILayout.EnumPopup(new GUIContent("Block seeds",
            "RSA: S4's (random, no two closer than 0.8 p), so p 40 s 18 seed 1 is S4's city and BlockCityWorld's streets. " +
            "Hex: a triangular lattice, each seed moved by up to the jitter: more of each block's sides are long enough to be " +
            "streets, fewer streets leave each corner."), seeding);
        if (seeding == Seeding.Hex)
        {
            jitter = EditorGUILayout.Slider(new GUIContent("Jitter", "x the lattice spacing. 0: regular hexagons."), jitter, 0f, 0.5f);
        }
        relax = EditorGUILayout.IntSlider(new GUIContent("Relax (Lloyd)",
            "Times every seed is moved to its block's centroid: blocks grow more even, still irregular."), relax, 0, 6);
        using (new EditorGUILayout.HorizontalScope())
        {
            seed = EditorGUILayout.IntField("Seed", seed);
            if (GUILayout.Button("New", GUILayout.Width(50))) seed = UnityEngine.Random.Range(1, 100000);
        }

        EditorGUILayout.Space();
        using (new EditorGUILayout.HorizontalScope())
        {
            string py = EditorGUILayout.TextField(new GUIContent("Python", "The stitching env's python.exe."), VoronoiCityWindow.Python);
            if (py != VoronoiCityWindow.Python) VoronoiCityWindow.Python = py;
            if (GUILayout.Button("…", GUILayout.Width(24)))
            {
                string f = EditorUtility.OpenFilePanel("python.exe of the stitching env",
                                                       Path.GetDirectoryName(VoronoiCityWindow.Python), "exe");
                if (!string.IsNullOrEmpty(f)) VoronoiCityWindow.Python = f;
            }
        }
        using (new EditorGUI.DisabledScope(builder == null || EditorApplication.isPlayingOrWillChangePlaymode))
        {
            if (GUILayout.Button("Generate and bake into the scene", GUILayout.Height(28))) Generate(builder);
            if (builder != null && GUILayout.Button("Rebake the layout in the builder")) Bake(builder, "");
            if (builder != null && GUILayout.Button(new GUIContent("Redress the streets",
                    "Footpaths and street furniture (CityDressing) made again from the baked layout. Every bake dresses too.")))
            {
                BlockShapedCityWorldBuilder.EnsureDressing(builder);
                bool ok = builder.Redress(out string summary);
                report = ok ? $"Redressed — save the scene to keep it.\n{summary}" : "Redress failed: " + summary;
            }
        }

        EditorGUILayout.Space();
        EditorGUILayout.LabelField("Layouts", EditorStyles.boldLabel);
        if (layoutFiles.Length == 0)
        {
            EditorGUILayout.LabelField(" ", "none yet");
        }
        else
        {
            layoutIndex = Mathf.Clamp(layoutIndex, 0, layoutFiles.Length - 1);
            layoutIndex = EditorGUILayout.Popup("Layout", layoutIndex, layoutFiles.Select(f => Label(f, builder)).ToArray());
            using (new EditorGUI.DisabledScope(builder == null || EditorApplication.isPlayingOrWillChangePlaymode))
            using (new EditorGUILayout.HorizontalScope())
            {
                if (GUILayout.Button(new GUIContent("Fly at Play", "Play builds this layout, from its file unless it is the bake.")))
                {
                    Undo.RecordObject(builder, "Block-shaped city layout");
                    builder.layoutPath = layoutFiles[layoutIndex];
                    EditorUtility.SetDirty(builder);
                    EditorSceneManager.MarkSceneDirty(builder.gameObject.scene);
                    report = $"Play flies {builder.layoutPath}" +
                             (string.Equals(builder.layoutPath, builder.BakedLayout, StringComparison.OrdinalIgnoreCase)
                                 ? ", the bake." : ", built from its file at Play.");
                }
                if (GUILayout.Button("Bake")) Bake(builder, layoutFiles[layoutIndex]);
            }
        }
        if (!string.IsNullOrEmpty(report)) EditorGUILayout.HelpBox(report, MessageType.None);
        EditorGUILayout.EndScrollView();
    }

    private void RefreshLayouts()
    {
        string folder = Path.Combine(Application.dataPath, ProceduralCityBuilder.LayoutFolder);
        layoutFiles = Directory.Exists(folder)
            ? Directory.GetFiles(folder, BlockShapedCityWorldBuilder.LayoutPattern).Select(Path.GetFileName)
                       .OrderBy(f => f, StringComparer.Ordinal).ToArray()
            : new string[0];
    }

    private static string Label(string file, ProceduralCityBuilder builder)
    {
        string label = Path.GetFileNameWithoutExtension(file).Replace("BlockShaped__", "");
        if (builder == null) return label;
        if (string.Equals(file, builder.BakedLayout, StringComparison.OrdinalIgnoreCase)) label += "  ·  baked";
        if (string.Equals(file, builder.layoutPath, StringComparison.OrdinalIgnoreCase)) label += "  ·  flown at Play";
        return label;
    }

    private void Generate(ProceduralCityBuilder builder)
    {
        if (!File.Exists(VoronoiCityWindow.Python))
        {
            report = $"No python at {VoronoiCityWindow.Python}: set the stitching env's python.exe above.";
            return;
        }
        string F(float v) => v.ToString("0.#####", CultureInfo.InvariantCulture);
        string args = $"-B -m citysim.blockcity one --variant B --p {F(blockSpacing)} --s {F(streetWidth)} --seed {seed} " +
                      $"--inset {F(footpath)} --seeds {(seeding == Seeding.Hex ? "hex" : "rsa")} --jitter {F(jitter)} " +
                      $"--relax {relax} --scene";
        var lines = new List<string>();
        int code;
        try
        {
            code = VoronoiCityWindow.RunPython(args, Path.Combine(Application.dataPath, "Scripts", "Experiment"), lines,
                                               "Block-shaped city");
        }
        catch (Exception e)
        {
            report = $"Could not run {VoronoiCityWindow.Python}: {e.Message}";
            return;
        }
        string layoutPath = lines.Where(l => l.StartsWith("LAYOUT_PATH=")).Select(l => l.Substring(12).Trim()).FirstOrDefault();
        RefreshLayouts();
        if (layoutPath == null || !File.Exists(layoutPath))
        {
            report = $"The generator failed (exit {code}):\n" + string.Join("\n", lines.Skip(Math.Max(0, lines.Count - 12)));
            Debug.LogError("BlockShapedCityWindow: " + report);
            return;
        }
        string assetPath = "Assets" + Path.GetFullPath(layoutPath).Substring(Path.GetFullPath(Application.dataPath).Length)
                                                                   .Replace('\\', '/');
        AssetDatabase.ImportAsset(assetPath);
        var summary = new StringBuilder();
        foreach (string l in lines.Where(l => l.StartsWith("[blockcity] ")))
        {
            summary.Append(l.Substring("[blockcity] ".Length)).Append('\n');
        }
        Bake(builder, Path.GetFileName(layoutPath), summary.ToString().TrimEnd());
    }

    private void Bake(ProceduralCityBuilder builder, string layoutName, string generated = null)
    {
        if (!string.IsNullOrEmpty(layoutName))
        {
            Undo.RecordObject(builder, "Block-shaped city layout");
            builder.layoutPath = layoutName;
        }
        BlockShapedCityWorldBuilder.EnsureDressing(builder);
        BlockShapedCityWorldBuilder.EnsureStyle(builder);
        if (!builder.Bake(out string error))
        {
            report = (generated != null ? generated + "\n" : "") + "Bake failed: " + error;
            Debug.LogError("BlockShapedCityWindow: " + error);
            return;
        }
        EditorSceneManager.MarkSceneDirty(builder.gameObject.scene);
        report = $"Baked {builder.layoutPath} into {builder.gameObject.scene.name} — save the scene to keep it." +
                 (string.IsNullOrEmpty(generated) ? "" : "\n" + generated);
        Debug.Log("BlockShapedCityWindow: " + report);
    }
}
