#if UNITY_EDITOR
using System;
using System.Collections;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Reflection;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;

/// <summary>
/// Checks that a scene can be played repeatedly without a domain reload, and times what entering Play costs. With
/// Reload Domain off (the project keeps it on) a static keeps its value from one Play to the next unless a
/// <c>SubsystemRegistration</c> hook resets it, and a stale one does not fail loudly: it starts the next trial from the
/// last trial's body yaw. Run on the bench's project copy, never the live project:
/// <c>Unity.exe -batchmode -projectPath &lt;copy&gt; -executeMethod PlayModeRepeatProbe.Run -logFile &lt;log&gt;</c>
/// with PLAY_PROBE_OUT (the report), EXPERIMENT_RECORDER_DIR and STITCH_SHM_SUFFIX set, so nothing lands among the
/// real runs or on the live sections. Inert otherwise.
///
/// <para>Enters Play several times in one process. At the end of the first Play every writable static of value type in
/// Assembly-CSharp is deliberately corrupted, so a static with no reset is caught whether or not this scene happens to
/// write it (the DJI scene's are, here); references and collections are filled by the flight itself. After the resets
/// and before any scene loads, each Play's statics must equal the first Play's; each difference is reported as a LEAK
/// and the process exits 1. PLAY_PROBE_RELOAD=1 reloads the domain every Play instead: the timing baseline, and the
/// control, since a reload clears everything.</para>
/// </summary>
public class PlayModeRepeatProbe : MonoBehaviour
{
    const string OutEnv = "PLAY_PROBE_OUT", SceneEnv = "PLAY_PROBE_SCENE", ReloadEnv = "PLAY_PROBE_RELOAD",
                 PlaysEnv = "PLAY_PROBE_PLAYS", HoldEnv = "PLAY_PROBE_HOLD";
    const string DefaultScene = "Assets/Scenes/BlockShapedPatchCityWorld.unity";
    // SessionState survives the reload of the baseline run; statics would not.
    const string PlayKey = "PlayModeRepeatProbe.Play", ClickKey = "PlayModeRepeatProbe.Click";
    const BindingFlags AnyStatic = BindingFlags.Static | BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.DeclaredOnly;
    const BindingFlags AnyInstance = BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic;
    static readonly CultureInfo Inv = CultureInfo.InvariantCulture;

    static int errors;

    static string OutPath => Environment.GetEnvironmentVariable(OutEnv);
    static int Play => SessionState.GetInt(PlayKey, 0);
    static bool Armed => !string.IsNullOrEmpty(OutPath) && Play > 0;
    static string Env(string name, string fallback)
    {
        string v = Environment.GetEnvironmentVariable(name);
        return string.IsNullOrEmpty(v) ? fallback : v;
    }
    static int Plays => int.Parse(Env(PlaysEnv, "3"), Inv);

    // ---- the editor side

    public static void Run()
    {
        if (string.IsNullOrEmpty(OutPath))
        {
            Debug.LogError($"PlayModeRepeatProbe: set {OutEnv} to the report path.");
            EditorApplication.Exit(2);
            return;
        }
        File.WriteAllText(OutPath, "");
        bool reload = Env(ReloadEnv, "0") == "1";
        EditorSettings.enterPlayModeOptionsEnabled = !reload;
        EditorSettings.enterPlayModeOptions = EnterPlayModeOptions.DisableDomainReload;
        EditorSceneManager.OpenScene(Env(SceneEnv, DefaultScene), OpenSceneMode.Single);
        Line($"domain reload {(reload ? "on" : "off")}, scene {Env(SceneEnv, DefaultScene)}");
        SessionState.SetInt(PlayKey, 1);
        EditorApplication.playModeStateChanged += OnPlayModeState;
        Enter();
    }

    // The baseline's reload drops Run's subscription.
    [InitializeOnLoadMethod]
    static void Resubscribe()
    {
        if (Armed) EditorApplication.playModeStateChanged += OnPlayModeState;
    }

    static void Enter()
    {
        SessionState.SetString(ClickKey, DateTime.UtcNow.Ticks.ToString(Inv));
        EditorApplication.EnterPlaymode();
    }

    static void OnPlayModeState(PlayModeStateChange state)
    {
        if (state != PlayModeStateChange.EnteredEditMode) return;
        Stamp("back_in_edit_mode");
        if (Play >= Plays)
        {
            Finish();
            return;
        }
        SessionState.SetInt(PlayKey, Play + 1);
        EditorApplication.delayCall += Enter;
    }

    static void Finish()
    {
        Dictionary<string, string> first = ReadSnapshot(1);
        int leaks = 0;
        for (int play = 2; play <= Plays; play++)
        {
            Dictionary<string, string> later = ReadSnapshot(play);
            foreach (KeyValuePair<string, string> field in first)
            {
                if (later.TryGetValue(field.Key, out string now) && now == field.Value) continue;
                leaks++;
                Line($"LEAK play={play} {field.Key}: first Play {field.Value}, this Play {now}");
            }
        }
        Line($"result: {leaks} leaks over {first.Count} statics");
        EditorApplication.Exit(leaks == 0 ? 0 : 1);
    }

    // ---- each Play

    [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.SubsystemRegistration)]
    static void OnPlayBegins()
    {
        if (!Armed) return;
        errors = 0;
        Application.logMessageReceived -= CountError;
        Application.logMessageReceived += CountError;
        Stamp("play_begins");
    }

    // After every SubsystemRegistration reset and before any scene: what a Play starts from.
    [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.AfterAssembliesLoaded)]
    static void WriteSnapshot()
    {
        if (!Armed) return;
        File.WriteAllLines(SnapshotPath(Play), Snapshot().Select(f => f.Key + "\t" + f.Value));
    }

    [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.BeforeSceneLoad)]
    static void BeforeScene()
    {
        if (Armed) Stamp("scene_load_begins");
    }

    [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.AfterSceneLoad)]
    static void AfterScene()
    {
        if (!Armed) return;
        Stamp("scene_loaded");
        // The live editor's joystick port (destroyed, as its OnApplicationQuit assumes Start ran); the DJI feed section.
        foreach (var c in FindObjectsOfType<UDPReceiverManager>(true)) DestroyImmediate(c);
        foreach (var c in FindObjectsOfType<ImageSharing>(true)) c.enabled = false;
        var probe = new GameObject(nameof(PlayModeRepeatProbe));
        DontDestroyOnLoad(probe);
        probe.AddComponent<PlayModeRepeatProbe>();
    }

    IEnumerator Start()
    {
        yield return null;   // past every Start and the first frame's render
        Stamp("first_frame_done");
        yield return new WaitForSecondsRealtime(float.Parse(Env(HoldEnv, "20"), Inv));
        Line($"play={Play} drones={GameObject.FindGameObjectsWithTag("DroneBase").Length} errors={errors} goalSeed={GoalSession.Seed}");
        if (Play == 1) Poison();
        Stamp("exit_requested");
        EditorApplication.ExitPlaymode();
    }

    static void CountError(string message, string stack, LogType type)
    {
        if (type != LogType.Error && type != LogType.Exception && type != LogType.Assert) return;
        if (errors++ < 5) Line($"play={Play} {type}: {message}");
    }

    // ---- statics

    static IEnumerable<FieldInfo> Fields()
    {
        foreach (Type t in typeof(PlayModeRepeatProbe).Assembly.GetTypes())
        {
            if (Skip(t)) continue;
            foreach (FieldInfo f in t.GetFields(AnyStatic))
            {
                bool generated = f.Name.Contains("<") && !f.Name.EndsWith("k__BackingField");
                if (!f.IsLiteral && !generated) yield return f;
            }
        }
    }

    // Not in any flight scene: the eye-tracker SDK, the probes themselves (which run once per process).
    static bool Skip(Type t) =>
        t.ContainsGenericParameters || t.Name.Contains("<")
        || t == typeof(PlayModeRepeatProbe) || t == typeof(ProceduralCityProbe) || t == typeof(SwarmBenchRunner)
        || t.FullName.StartsWith("Tobii") || t.FullName.StartsWith("AQUAS")
        || (t.DeclaringType != null && Skip(t.DeclaringType));

    static string Id(FieldInfo f) => f.DeclaringType.FullName + "." + f.Name;

    static Dictionary<string, string> Snapshot() => Fields().ToDictionary(Id, f => Describe(f.GetValue(null)));

    static string SnapshotPath(int play) => OutPath + $".play{play}.txt";

    static Dictionary<string, string> ReadSnapshot(int play) =>
        File.ReadAllLines(SnapshotPath(play)).Select(l => l.Split('\t')).ToDictionary(p => p[0], p => p[1]);

    static string Describe(object v)
    {
        switch (v)
        {
            case null: return "null";
            case UnityEngine.Object o: return o == null ? "destroyed" : "live " + o.GetType().Name;
            case Delegate d: return d.GetInvocationList().Length + " subscribers";
            case string s: return "'" + s + "'";
            case ICollection c: return c.Count + " items";
        }
        Type t = v.GetType();
        if (!t.IsValueType) return "a " + t.Name;
        if (t.IsPrimitive || t.IsEnum) return Convert.ToString(v, Inv);
        return "{" + string.Join(",", t.GetFields(AnyInstance).Select(f => Describe(f.GetValue(v)))) + "}";
    }

    static void Poison()
    {
        int corrupted = 0, total = 0;
        foreach (FieldInfo f in Fields())
        {
            total++;
            if (f.IsInitOnly) continue;
            object before = f.GetValue(null);
            object after = Poisoned(f.FieldType, before);
            if (Describe(before) == Describe(after)) continue;
            f.SetValue(null, after);
            corrupted++;
        }
        Line($"corrupted {corrupted} of {total} statics");
    }

    static object Poisoned(Type t, object v)
    {
        if (t == typeof(bool)) return !(bool)v;
        if (t == typeof(int)) return (int)v + 12345;
        if (t == typeof(long)) return (long)v + 12345L;
        if (t == typeof(float)) return float.IsNaN((float)v) || float.IsInfinity((float)v) ? 12345f : (float)v + 12345f;
        if (t == typeof(double)) return double.IsNaN((double)v) || double.IsInfinity((double)v) ? 12345d : (double)v + 12345d;
        if (t == typeof(string)) return (v as string) + "~";
        if (t.IsEnum) return Enum.ToObject(t, Convert.ToInt64(v) + 1);
        if (!t.IsValueType || t.IsPrimitive) return v;
        foreach (FieldInfo f in t.GetFields(AnyInstance)) f.SetValue(v, Poisoned(f.FieldType, f.GetValue(v)));
        return v;
    }

    // ---- the report

    static void Stamp(string phase)
    {
        long ms = (DateTime.UtcNow.Ticks - long.Parse(SessionState.GetString(ClickKey, "0"), Inv)) / TimeSpan.TicksPerMillisecond;
        Line($"play={Play} {phase} {ms} ms after Enter");
    }

    static void Line(string text)
    {
        File.AppendAllText(OutPath, text + "\n");
        Debug.Log("[PlayProbe] " + text);
    }
}
#endif
