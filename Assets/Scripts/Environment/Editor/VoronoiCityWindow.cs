using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Text;
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
/// </summary>
public class VoronoiCityWindow : EditorWindow
{
    private const string PythonPref = "VoronoiCityWindow.Python";
    private const float DiamondAlpha = 0.0227f;           // DiamondCityWorld's coverage over its tile grid
    private const float ArenaArea = 730f * 730f;
    private const float MeanFootprint = 50.1f;            // m^2, DiamondCityWorld's buildings (catalogue --mix)
    private const int TimeoutSec = 600;

    [SerializeField] private float alpha = DiamondAlpha;
    [SerializeField] private int seed = 1;
    [SerializeField] private float blockSpacing = 90.83f;
    [SerializeField] private float streetWidth = 15f;
    [SerializeField] private float facadeGap = 4f;
    [SerializeField] private bool even = true;
    [SerializeField] private bool showGoals;
    [SerializeField] private float occluderWidth = 13.06f;
    [SerializeField] private float ringRadius = 30f;
    [SerializeField] private int ringBuildings = 8;
    [SerializeField] private string report = "";
    private Vector2 scroll;

    [MenuItem("Tools/Swarm/Voronoi city…")]
    private static void Open() => GetWindow<VoronoiCityWindow>("Voronoi city");

    private static string Python
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
            foreach (float k in new[] { 1f, 2f, 3f, 5f })
            {
                if (GUILayout.Button($"{k:0}×")) alpha = DiamondAlpha * k;
            }
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
        }
        if (!string.IsNullOrEmpty(report)) EditorGUILayout.HelpBox(report, MessageType.None);
        EditorGUILayout.EndScrollView();
    }

    private void Generate(ProceduralCityBuilder builder)
    {
        if (!File.Exists(Python))
        {
            report = $"No python at {Python}: set the stitching env's python.exe above.";
            return;
        }
        string F(float v) => v.ToString("0.#####", CultureInfo.InvariantCulture);
        string args = $"-B -m citysim.citygen mixcity --city DiamondCityWorld --seed {seed} --alpha {F(alpha)} " +
                      $"--p {F(blockSpacing)} --s {F(streetWidth)} --g {F(facadeGap)} --place {(even ? "even" : "rsa")} " +
                      $"--wc {F(occluderWidth)} --rg {F(ringRadius)} --nb {ringBuildings}";
        string cwd = Path.Combine(Application.dataPath, "Scripts", "Experiment");
        var lines = new List<string>();
        int code;
        try
        {
            code = RunPython(args, cwd, lines);
        }
        catch (Exception e)
        {
            report = $"Could not run {Python}: {e.Message}";
            return;
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
            report = $"The generator failed (exit {code}):\n" + string.Join("\n", lines.GetRange(Math.Max(0, lines.Count - 12),
                                                                                              Math.Min(12, lines.Count)));
            Debug.LogError("VoronoiCityWindow: " + report);
            return;
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
        Bake(builder, Path.GetFileName(layoutPath), sb.ToString());
    }

    private void Bake(ProceduralCityBuilder builder, string layoutName, string generated = null)
    {
        if (!string.IsNullOrEmpty(layoutName))
        {
            Undo.RecordObject(builder, "Voronoi city layout");
            builder.layoutPath = layoutName;
        }
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

    /// <summary>Runs the generator, collecting stdout and stderr; its exit code. Cancelable from the progress bar.</summary>
    private static int RunPython(string args, string cwd, List<string> lines)
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
                    if (EditorUtility.DisplayCancelableProgressBar("Voronoi city", $"Generating the layout… {clock.Elapsed.TotalSeconds:0} s",
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
