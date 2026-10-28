using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text;
using System.Text.RegularExpressions;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;
using Debug = UnityEngine.Debug;

/// <summary>
/// <c>Tools/Swarm/Voronoi city…</c>: tune VoronoiCityWorld's city and see it. Generates a layout with the Python
/// generator (<c>citysim.citygen mixcity</c>, run in the stitching env: DiamondCityWorld's building mix on Voronoi
/// streets) at the coverage, seed, block spacing, street width and facade gap set here, then bakes it into the open
/// scene's <see cref="ProceduralCityBuilder"/> (<see cref="ProceduralCityBuilder.Bake"/>), so the city is there in the
/// editor and is what Play flies. Save the scene to keep it.
///
/// <para>The layout file is still the city's record — analyse.py and the replica read it by the id in the session — so
/// every generation writes its own <c>layouts/VoronoiCity__&lt;id&gt;.json</c> (the id names the parameters and the
/// seed) and nothing is generated at Play. Delete the ones you do not keep; a run recorded on a deleted layout can no
/// longer be analysed.</para>
///
/// <para>Coverage is the share of the 730 m arena under buildings, goals included. DiamondCityWorld's is 0.0227. With
/// the 4 m facade gap the blocks jam near 0.12 (the generator stops and says so): a smaller gap, narrower streets or a
/// smaller block spacing pack more. Past roughly 0.15 more than 56 buildings stand within 50 m of some point, and
/// OlfatiSaber's obstacle query holds 64: the swarm would not see them all. The generator reports it, and so does this
/// window.</para>
///
/// <para><b>The coverage levels are S3 City Roaming's</b> (docs/studies/S3_city_roaming.md): today's 0.0227, 0.04, 0.06
/// and 0.10 on today's 90.83 m blocks, 12 m streets, goals kept 22 m clear. 0.04 splits the step from 0.0227 to 0.06,
/// which was too large to fly as one. One seed at every coverage shares its streets and goals (the generator draws both
/// before the city), so the levels differ only in clutter. The scene holds one bake; Layouts picks the one Play flies,
/// which is built from its file when it is not the bake.</para>
///
/// <para><b>Each goal stands in a block of its own</b> (Goals, <i>Goal in its own block</i>; citygen's
/// <c>goal_cell</c>, since 2026-10-26): the goals are placed before the streets, which then run round each goal's
/// patch instead of through it, and the goal's block is about the size of the others. Off reproduces the layouts made
/// before, whose rings stand up to ~5 m into the streets and whose walkers patrol them; their files carry no
/// <c>_gcell</c> in the id.</para>
///
/// <para>Every bake also dresses the streets (<see cref="CityDressing"/>, which
/// <see cref="VoronoiCityWorldBuilder.EnsureDressing"/> adds): footpaths, medians with trees and street furniture,
/// redrawn from the layout so they follow any change to it. <i>Redress the streets</i> redoes only that, after a change to
/// the dressing's rules in its inspector.</para>
/// </summary>
public class VoronoiCityWindow : EditorWindow
{
    private const string PythonPref = "VoronoiCityWindow.Python";
    private const float DiamondAlpha = 0.0227f;           // DiamondCityWorld's coverage over its tile grid
    private const float ArenaArea = 730f * 730f;
    private const float MeanFootprint = 50.1f;            // m^2, DiamondCityWorld's buildings (catalogue --mix)
    private const int TimeoutSec = 600;
    /// <summary>S3's levels: the swarm finds a hidden target 1.48x, 1.67x, 1.92x and 2.58x as fast as one drone there.</summary>
    private static readonly float[] Levels = { DiamondAlpha, 0.04f, 0.06f, 0.10f };
    private const float MaxAlpha = 0.12f;                 // S3: all 10 cities build at 0.12 on 90.83 m blocks, 2 at 0.14

    [SerializeField] private float alpha = DiamondAlpha;
    [SerializeField] private int seed = 1;
    [SerializeField] private float blockSpacing = 90.83f;
    [SerializeField] private float streetWidth = 12f;
    [SerializeField] private float facadeGap = 4f;
    [SerializeField] private bool even = true;
    [SerializeField] private bool showGoals;
    [SerializeField] private float occluderWidth = 13.06f;
    [SerializeField] private float ringRadius = 30f;
    [SerializeField] private int ringBuildings = 8;
    [SerializeField] private float minClearance = 22f;
    [SerializeField] private bool goalCell = true;
    [SerializeField] private int layoutIndex;
    [SerializeField] private string report = "";
    private string[] layoutFiles = new string[0];
    private Vector2 scroll;

    [MenuItem("Tools/Swarm/Voronoi city…")]
    private static void Open() => GetWindow<VoronoiCityWindow>("Voronoi city");

    private void OnEnable() => RefreshLayouts();

    private void OnFocus() => RefreshLayouts();

    internal static string Python
    {
        get => EditorPrefs.GetString(PythonPref, Path.Combine(
            Environment.GetEnvironmentVariable("LOCALAPPDATA") ?? "", "miniconda3", "envs", "stitching", "python.exe"));
        set => EditorPrefs.SetString(PythonPref, value);
    }

    private void OnGUI()
    {
        ProceduralCityBuilder builder = FindFirstObjectByType<ProceduralCityBuilder>(FindObjectsInactive.Include);
        scroll = EditorGUILayout.BeginScrollView(scroll);
        if (builder == null)
        {
            EditorGUILayout.HelpBox($"The open scene has no ProceduralCityBuilder: open {VoronoiCityWorldBuilder.ScenePath} " +
                                    "(Tools/Swarm/Build Voronoi city world makes it).", MessageType.Warning);
        }
        else
        {
            EditorGUILayout.LabelField("Scene", builder.gameObject.scene.name);
            EditorGUILayout.LabelField("Baked layout", string.IsNullOrEmpty(builder.BakedLayout) ? "(none)" : builder.BakedLayout,
                                       EditorStyles.wordWrappedLabel);
        }

        EditorGUILayout.Space();
        EditorGUILayout.LabelField("Density", EditorStyles.boldLabel);
        alpha = EditorGUILayout.Slider(new GUIContent("Coverage α", "Share of the 730 m arena under buildings."),
                                       alpha, 0.005f, 0.3f);
        EditorGUILayout.LabelField(" ", $"{alpha / DiamondAlpha:0.0}× DiamondCityWorld, ≈ {alpha * ArenaArea / MeanFootprint:0} buildings");
        using (new EditorGUILayout.HorizontalScope())
        {
            GUILayout.Space(EditorGUIUtility.labelWidth);
            foreach (float k in Levels)
            {
                if (GUILayout.Button(k.ToString("0.00##", CultureInfo.InvariantCulture))) alpha = k;
            }
        }
        if (alpha > MaxAlpha + 1e-4f)
        {
            EditorGUILayout.HelpBox($"S3 built every city at {MaxAlpha:0.00} on 90.83 m blocks with 12 m streets, but only " +
                                    "2 of 10 at 0.14: past 0.12 the blocks may jam.", MessageType.Info);
        }
        blockSpacing = EditorGUILayout.FloatField(new GUIContent("Block spacing p (m)",
            "Mean distance between block centres; DiamondCityWorld's tile pitch is 90.83."), blockSpacing);
        streetWidth = EditorGUILayout.FloatField(new GUIContent("Street width s (m)"), streetWidth);
        facadeGap = EditorGUILayout.FloatField(new GUIContent("Facade gap (m)",
            "Least gap between two buildings. The blocks jam near α 0.12 at 4 m."), facadeGap);
        even = EditorGUILayout.Toggle(new GUIContent("Spread evenly",
            "Every block to the same built fraction, its buildings spread over it. Off: the study's random placement."), even);
        using (new EditorGUILayout.HorizontalScope())
        {
            seed = EditorGUILayout.IntField("Seed", seed);
            if (GUILayout.Button("New", GUILayout.Width(50))) seed = UnityEngine.Random.Range(1, 100000);
        }

        showGoals = EditorGUILayout.Foldout(showGoals, "Goals (the density study's ring goal)", true);
        if (showGoals)
        {
            using (new EditorGUI.IndentLevelScope())
            {
                occluderWidth = EditorGUILayout.FloatField("Occluder long side (m)", occluderWidth);
                ringRadius = EditorGUILayout.FloatField("Ring radius (m)", ringRadius);
                ringBuildings = EditorGUILayout.IntField("Ring buildings", ringBuildings);
                minClearance = EditorGUILayout.FloatField(new GUIContent("Least clearance (m)",
                    "Goal centre to the nearest ring face. S1 saw the swarm surround the occluder at 22 m and above, " +
                    "never at 16 m or below. 0: no floor."), minClearance);
                goalCell = EditorGUILayout.Toggle(new GUIContent("Goal in its own block",
                    "Place the goals before the streets, each in a block that holds its whole patch, ring and walkers " +
                    "(citygen goal_cell). Off: a goal on its block's centroid, its ring and walkers across the streets " +
                    "(the layouts made before 2026-10-26)."), goalCell);
            }
        }

        EditorGUILayout.Space();
        using (new EditorGUILayout.HorizontalScope())
        {
            string py = EditorGUILayout.TextField(new GUIContent("Python", "The stitching env's python.exe."), Python);
            if (py != Python) Python = py;
            if (GUILayout.Button("…", GUILayout.Width(24)))
            {
                string f = EditorUtility.OpenFilePanel("python.exe of the stitching env", Path.GetDirectoryName(Python), "exe");
                if (!string.IsNullOrEmpty(f)) Python = f;
            }
        }

        using (new EditorGUI.DisabledScope(builder == null || EditorApplication.isPlayingOrWillChangePlaymode))
        {
            if (GUILayout.Button("Generate and bake into the scene", GUILayout.Height(28))) Generate(builder);
            if (builder != null && GUILayout.Button("Rebake the layout in the builder")) Bake(builder, "");
            if (builder != null && GUILayout.Button(new GUIContent("Redress the streets",
                    "Footpaths, medians and street furniture (CityDressing, on the builder) made again from the baked " +
                    "layout, after a change to their rules. Every bake dresses too.")))
            {
                Redress(builder);
            }
        }
        using (new EditorGUI.DisabledScope(EditorApplication.isPlayingOrWillChangePlaymode))
        {
            if (GUILayout.Button(new GUIContent($"Generate the {Levels.Length} levels (no bake)",
                    $"This seed at coverage {string.Join(", ", Levels.Select(k => k.ToString("0.00##", CultureInfo.InvariantCulture)))}: " +
                    "the same streets and goals, more buildings. Pick one under Layouts.")))
            {
                GenerateLevels();
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
            EditorGUILayout.LabelField(" ", layoutFiles[layoutIndex], EditorStyles.wordWrappedMiniLabel);
            using (new EditorGUI.DisabledScope(builder == null || EditorApplication.isPlayingOrWillChangePlaymode))
            using (new EditorGUILayout.HorizontalScope())
            {
                if (GUILayout.Button(new GUIContent("Fly at Play", "Play builds this layout, from its file unless it is the " +
                                                    "bake. The bake stays as it is.")))
                {
                    Use(builder, layoutFiles[layoutIndex]);
                }
                if (GUILayout.Button(new GUIContent("Bake", "Bake this layout into the scene and fly it at Play.")))
                {
                    Bake(builder, layoutFiles[layoutIndex]);
                }
            }
        }
        if (!string.IsNullOrEmpty(report)) EditorGUILayout.HelpBox(report, MessageType.None);
        EditorGUILayout.EndScrollView();
    }

    private void RefreshLayouts()
    {
        string folder = Path.Combine(Application.dataPath, ProceduralCityBuilder.LayoutFolder);
        layoutFiles = Directory.Exists(folder)
            ? Directory.GetFiles(folder, VoronoiCityWorldBuilder.LayoutPattern).Select(Path.GetFileName)
                       .OrderBy(f => f, StringComparer.Ordinal).ToArray()
            : new string[0];
    }

    /// <summary>A layout's parameters for the picker, and whether it is the bake and the one Play flies.</summary>
    private static string Label(string file, ProceduralCityBuilder builder)
    {
        Match m = Regex.Match(file, @"_p([\d.]+)_s([\d.]+)_a([\d.]+)_.*_seed(\d+)\.json$");
        string label = m.Success
            ? $"α {m.Groups[3].Value}  ·  p {m.Groups[1].Value}  ·  s {m.Groups[2].Value}  ·  seed {m.Groups[4].Value}" +
              (file.Contains("_gcell_") ? "  ·  goals in own blocks" : "")
            : file;
        if (builder == null) return label;
        if (string.Equals(file, builder.BakedLayout, StringComparison.OrdinalIgnoreCase)) label += "  ·  baked";
        if (string.Equals(file, builder.layoutPath, StringComparison.OrdinalIgnoreCase)) label += "  ·  flown at Play";
        return label;
    }

    /// <summary>Makes <paramref name="layoutName"/> the layout Play flies without touching the bake. Unless it is the
    /// bake, Play builds it from its file and the console says the bake was passed over.</summary>
    private void Use(ProceduralCityBuilder builder, string layoutName)
    {
        Undo.RecordObject(builder, "Voronoi city layout");
        builder.layoutPath = layoutName;
        EditorUtility.SetDirty(builder);
        EditorSceneManager.MarkSceneDirty(builder.gameObject.scene);
        report = string.Equals(layoutName, builder.BakedLayout, StringComparison.OrdinalIgnoreCase)
            ? $"Play flies {layoutName}, the bake."
            : $"Play flies {layoutName}, built from its file (the bake is {builder.BakedLayout}).";
    }

    private void Generate(ProceduralCityBuilder builder)
    {
        string name = RunGenerator(alpha, out string text);
        RefreshLayouts();
        if (name == null)
        {
            report = text;
            return;
        }
        Bake(builder, name, text);
    }

    private void GenerateLevels()
    {
        var sb = new StringBuilder();
        foreach (float a in Levels)
        {
            string name = RunGenerator(a, out string text);
            sb.Append(name ?? $"coverage {a:0.00##} failed").Append('\n').Append(text).Append("\n\n");
            if (name == null) break;
        }
        RefreshLayouts();
        report = sb.Append("Pick one under Layouts to fly or bake it.").ToString();
    }

    /// <summary>Runs the generator at coverage <paramref name="a"/> with this window's settings: the new layout's file
    /// name and, in <paramref name="text"/>, its summary and any problems; or null and the reason.</summary>
    private string RunGenerator(float a, out string text)
    {
        if (!File.Exists(Python))
        {
            text = $"No python at {Python}: set the stitching env's python.exe above.";
            return null;
        }
        string F(float v) => v.ToString("0.#####", CultureInfo.InvariantCulture);
        string args = $"-B -m citysim.citygen mixcity --city DiamondCityWorld --seed {seed} --alpha {F(a)} " +
                      $"--p {F(blockSpacing)} --s {F(streetWidth)} --g {F(facadeGap)} --place {(even ? "even" : "rsa")} " +
                      $"--wc {F(occluderWidth)} --rg {F(ringRadius)} --nb {ringBuildings} --c-min {F(minClearance)} " +
                      $"--goal-cell {(goalCell ? 1 : 0)}";
        string cwd = Path.Combine(Application.dataPath, "Scripts", "Experiment");
        var lines = new List<string>();
        int code;
        try
        {
            code = RunPython(args, cwd, lines);
        }
        catch (Exception e)
        {
            text = $"Could not run {Python}: {e.Message}";
            return null;
        }
        string layoutPath = null;
        var problems = new List<string>();
        string summary = "";
        foreach (string l in lines)
        {
            if (l.StartsWith("LAYOUT_PATH=")) layoutPath = l.Substring("LAYOUT_PATH=".Length).Trim();
            else if (l.TrimStart().StartsWith("PROBLEM:")) problems.Add(l.Trim());
            else if (l.Contains(" buildings, alpha ")) summary = l.Substring(l.IndexOf(": ", StringComparison.Ordinal) + 2);
        }
        if (layoutPath == null || !File.Exists(layoutPath))
        {
            text = $"The generator failed (exit {code}):\n" + string.Join("\n", lines.GetRange(Math.Max(0, lines.Count - 12),
                                                                                            Math.Min(12, lines.Count)));
            Debug.LogError("VoronoiCityWindow: " + text);
            return null;
        }
        string assetPath = "Assets" + Path.GetFullPath(layoutPath).Substring(Path.GetFullPath(Application.dataPath).Length)
                                                                   .Replace('\\', '/');
        AssetDatabase.ImportAsset(assetPath);
        var sb = new StringBuilder(summary);
        if (summary.Contains("JAMMED"))
        {
            sb.Append("\nThe blocks are full before the target: lower the facade gap, the street width or the block spacing.");
        }
        foreach (string p in problems) sb.Append('\n').Append(p);
        if (problems.Exists(p => p.Contains("buffer")))
        {
            sb.Append("\nOlfatiSaber queries at most 64 obstacles within 50 m: past that the swarm misses buildings.");
        }
        text = sb.ToString();
        return Path.GetFileName(layoutPath);
    }

    private void Bake(ProceduralCityBuilder builder, string layoutName, string generated = null)
    {
        if (!string.IsNullOrEmpty(layoutName))
        {
            Undo.RecordObject(builder, "Voronoi city layout");
            builder.layoutPath = layoutName;
        }
        VoronoiCityWorldBuilder.EnsureDressing(builder);
        if (!builder.Bake(out string error))
        {
            report = (generated != null ? generated + "\n" : "") + "Bake failed: " + error;
            Debug.LogError("VoronoiCityWindow: " + error);
            return;
        }
        EditorSceneManager.MarkSceneDirty(builder.gameObject.scene);
        report = $"Baked {builder.layoutPath} into {builder.gameObject.scene.name} — save the scene to keep it." +
                 (string.IsNullOrEmpty(generated) ? "" : "\n" + generated);
        Debug.Log("VoronoiCityWindow: " + report);
    }

    private void Redress(ProceduralCityBuilder builder)
    {
        VoronoiCityWorldBuilder.EnsureDressing(builder);
        bool ok = builder.Redress(out string summary);
        report = ok ? $"Redressed the streets of {builder.gameObject.scene.name} — save the scene to keep it.\n{summary}"
                    : "Redress failed: " + summary;
        if (ok) Debug.Log("VoronoiCityWindow: " + report);
        else Debug.LogError("VoronoiCityWindow: " + summary);
    }

    /// <summary>Runs the generator, collecting stdout and stderr; its exit code. Cancelable from the progress bar.</summary>
    internal static int RunPython(string args, string cwd, List<string> lines, string title = "Voronoi city")
    {
        var psi = new ProcessStartInfo(Python, args)
        {
            WorkingDirectory = cwd,
            UseShellExecute = false,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            CreateNoWindow = true,
            StandardOutputEncoding = Encoding.UTF8,
            StandardErrorEncoding = Encoding.UTF8,
        };
        psi.EnvironmentVariables["PYTHONIOENCODING"] = "utf-8";
        using (var p = new Process { StartInfo = psi })
        {
            p.OutputDataReceived += (_, e) => { if (e.Data != null) lock (lines) lines.Add(e.Data); };
            p.ErrorDataReceived += (_, e) => { if (e.Data != null) lock (lines) lines.Add(e.Data); };
            p.Start();
            p.BeginOutputReadLine();
            p.BeginErrorReadLine();
            var clock = Stopwatch.StartNew();
            try
            {
                while (!p.WaitForExit(200))
                {
                    if (EditorUtility.DisplayCancelableProgressBar(title, $"Generating the layout… {clock.Elapsed.TotalSeconds:0} s",
                                                                   (float)(clock.Elapsed.TotalSeconds / 30.0 % 1.0)) ||
                        clock.Elapsed.TotalSeconds > TimeoutSec)
                    {
                        p.Kill();
                        throw new Exception(clock.Elapsed.TotalSeconds > TimeoutSec ? "timed out" : "cancelled");
                    }
                }
                p.WaitForExit();                 // flushes the redirected streams
            }
            finally
            {
                EditorUtility.ClearProgressBar();
            }
            return p.ExitCode;
        }
    }
}
