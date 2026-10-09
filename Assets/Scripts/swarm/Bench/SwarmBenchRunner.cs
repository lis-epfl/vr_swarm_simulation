// Headless swarm bench: flies scripted stick inputs through a city scene in batchmode and writes one
// JSON line per flight. Inert unless the SWARM_BENCH_CONFIG environment variable names a config, so it
// costs nothing in the editor or a build. Launched by run_bench.ps1 beside this file; see README.md.
//
// Parameter sets are OVERRIDES on the scene as authored. The scene's SwarmManager is snapshotted once and
// restored before every set, so an empty set means "the scene", a retune can never be silently undone by a
// config written before it, and one set cannot leak into the next.
using System;
using System.Collections;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Reflection;
using System.Text;
using UnityEngine;

public class SwarmBenchRunner : MonoBehaviour
{
    public const string ConfigEnv = "SWARM_BENCH_CONFIG";
    public const string OutEnv = "SWARM_BENCH_OUT";
    public const int ExitConfigError = 2;
    public const int ExitRuntimeError = 3;

    [Serializable]
    public class Scenario
    {
        public string kind;
        public float sx, sz, T;
        public float[] wps;     // flattened (x, z, stickMagnitude, timeoutSeconds)
        public bool record;     // write every drone's XZ every 5 fixed steps to traj_<set>_<index>.csv
        // Spread-stick schedule, flattened (t, d_ref) pairs, piecewise linear and held past the ends.
        // Written to InputStatus["spread"], i.e. through SwarmAlgorithm.SetSwarmSpread like the joystick.
        // Empty = the stick is left alone (-1).
        public float[] spread;
    }

    [Serializable]
    public class ParamSet
    {
        public string label;
        public string[] names;  // SwarmManager field names, overriding the scene's values
        public float[] values;  // bools as 0/1, ints and enums by integer
    }

    [Serializable]
    public class Config
    {
        public string scene;                 // bare name -> Assets/Scenes/<name>.unity
        public string outPath;               // SWARM_BENCH_OUT takes precedence
        public float timeScale = 20f;
        public float settleTime = 15f;
        public float altitude = 25f;
        public float capture = 10f;
        // Inject a pilot body yaw facing the current waypoint, as a pilot who looks where they fly. The
        // look-direction gap fill reads it, so without it the fill is forced off; with it, each flight
        // also reports the look-gap metrics.
        public bool pilotHeading = false;
        public float pilotYawRate = 90f;     // deg/s: PyUniSharingFast's default bodyYawRate
        // A recorded session (stem or path) whose goal layout GoalPatchReplacer replays, so the bench
        // flies that run's city. Empty = no goals: GoalPatchReplacer is disabled, as a random layout per
        // run would make flights incomparable.
        public string goalReplaySession = "";
        public ParamSet[] paramSets;
        public Scenario[] scenarios;
    }

    // ------------------------------------------------------------------ config, shared with the launcher

    public static Config LoadConfig(out string error)
    {
        error = null;
        string path = Environment.GetEnvironmentVariable(ConfigEnv);
        if (string.IsNullOrEmpty(path)) { error = ConfigEnv + " is not set"; return null; }
        if (!File.Exists(path)) { error = $"{ConfigEnv} names {path}, which does not exist"; return null; }

        Config c;
        try { c = JsonUtility.FromJson<Config>(File.ReadAllText(path)); }
        catch (Exception e) { error = $"cannot parse {path}: {e.Message}"; return null; }

        if (c == null) error = $"{path} is empty";
        else if (string.IsNullOrEmpty(c.scene)) error = "the config names no scene";
        else if (c.paramSets == null || c.paramSets.Length == 0)
            error = "the config has no paramSets (one empty set flies the scene as authored)";
        else if (c.scenarios == null || c.scenarios.Length == 0) error = "the config has no scenarios";
        else if (string.IsNullOrEmpty(OutPath(c))) error = $"no output file: set {OutEnv} or outPath";
        else if (c.timeScale <= 0f) error = "timeScale must be positive";
        return error == null ? c : null;
    }

    public static string ScenePath(string scene) =>
        scene.EndsWith(".unity", StringComparison.OrdinalIgnoreCase) ? scene : $"Assets/Scenes/{scene}.unity";

    public static string OutPath(Config c)
    {
        string env = Environment.GetEnvironmentVariable(OutEnv);
        return !string.IsNullOrEmpty(env) ? env : c?.outPath;
    }

    /// <summary>Logs, writes an error line to the results file when one is known, and exits.</summary>
    public static void Fatal(int code, string message, Config c = null)
    {
        Debug.LogError("[SwarmBench] " + message);
        AppendLine(OutPath(c ?? cfg), new Json().Str("event", "error").Int("code", code).Str("message", message).ToString());
        Exit(code);
    }

    static void Exit(int code)
    {
#if UNITY_EDITOR
        UnityEditor.EditorApplication.Exit(code);
#else
        Application.Quit(code);
#endif
    }

    static void AppendLine(string path, string line)
    {
        if (string.IsNullOrEmpty(path)) return;
        string dir = Path.GetDirectoryName(Path.GetFullPath(path));
        if (!string.IsNullOrEmpty(dir)) Directory.CreateDirectory(dir);
        File.AppendAllText(path, line + "\n");
    }

    // ------------------------------------------------------------------ boot

    static Config cfg;
    static string outPath;
    static MethodInfo setBodyYaw, setBodyYawValid;

    [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.AfterSceneLoad)]
    static void Boot()
    {
        if (string.IsNullOrEmpty(Environment.GetEnvironmentVariable(ConfigEnv))) return;
        cfg = LoadConfig(out string error);
        if (cfg == null) { Fatal(ExitConfigError, error); return; }
        outPath = OutPath(cfg);

        // Nothing that opens a socket or a named section: the user's own editor may be running beside us.
        foreach (var c in FindObjectsOfType<PyUniSharingFast>(true)) c.enabled = false;
        foreach (var c in FindObjectsOfType<UDPReceiverManager>(true)) c.enabled = false;
        foreach (var c in FindObjectsOfType<ImageSharing>(true)) c.enabled = false;
        foreach (var c in FindObjectsOfType<ExperimentRecorder>(true)) c.enabled = false;
        if (!ConfigureGoalPatches()) return;
        DisableCameras();

        if (cfg.pilotHeading)
        {
            // The setters are private: PyUniSharingFast owns the body yaw, and it is disabled here.
            PropertyInfo yaw = typeof(PyUniSharingFast).GetProperty("BodyYawDegrees", BindingFlags.Public | BindingFlags.Static);
            PropertyInfo valid = typeof(PyUniSharingFast).GetProperty("BodyYawValid", BindingFlags.Public | BindingFlags.Static);
            setBodyYaw = yaw?.GetSetMethod(true);
            setBodyYawValid = valid?.GetSetMethod(true);
            if (setBodyYaw == null || setBodyYawValid == null)
            {
                Fatal(ExitRuntimeError, "pilotHeading: PyUniSharingFast.BodyYawDegrees/BodyYawValid setters not found");
                return;
            }
        }

        new GameObject("SwarmBench").AddComponent<SwarmBenchRunner>();
        Debug.Log($"[SwarmBench] booted in {cfg.scene}: {cfg.scenarios.Length} scenarios x {cfg.paramSets.Length} parameter sets -> {outPath}");
    }

    static bool ConfigureGoalPatches()
    {
        GoalPatchReplacer[] replacers = FindObjectsOfType<GoalPatchReplacer>(true);
        if (string.IsNullOrEmpty(cfg.goalReplaySession))
        {
            foreach (var r in replacers) r.enabled = false;
            return true;
        }
        if (replacers.Length == 0)
        {
            Fatal(ExitConfigError, $"goalReplaySession is set but {cfg.scene} has no GoalPatchReplacer");
            return false;
        }
        const BindingFlags Private = BindingFlags.NonPublic | BindingFlags.Instance;
        FieldInfo mode = typeof(GoalPatchReplacer).GetField("mode", Private);
        FieldInfo file = typeof(GoalPatchReplacer).GetField("replaySessionFile", Private);
        if (mode == null || file == null)
        {
            Fatal(ExitRuntimeError, "GoalPatchReplacer.mode/replaySessionFile not found");
            return false;
        }
        foreach (var r in replacers)
        {
            mode.SetValue(r, GoalPatchReplacer.PlacementMode.Replay);
            file.SetValue(r, cfg.goalReplaySession);
            r.enabled = true;
        }
        return true;
    }

    // Physics only: nothing needs drawing, and the FPV cameras' offscreen targets are the expensive part.
    // (Disabled rather than run under -nographics, where those offscreen renders segfault.)
    static void DisableCameras()
    {
        foreach (var cam in FindObjectsOfType<Camera>(true)) cam.enabled = false;
    }

    // ------------------------------------------------------------------ per-drone contact probe

    public class ContactProbe : MonoBehaviour
    {
        public SwarmBenchRunner runner;
        int obstacleLayer;
        void Awake() { obstacleLayer = LayerMask.NameToLayer("Obstacle"); }
        void OnCollisionEnter(Collision c)
        {
            if (runner == null || !runner.measuring) return;
            if (c.collider.gameObject.layer == obstacleLayer)
            {
                float normalSpeed = 0f;
                if (c.contactCount > 0)
                    normalSpeed = Mathf.Abs(Vector3.Dot(c.relativeVelocity, c.GetContact(0).normal));
                runner.contacts++;
                if (normalSpeed > 2f) runner.hardContacts++;
                if (normalSpeed > runner.maxImpact) runner.maxImpact = normalSpeed;
            }
            else if (c.collider.GetComponentInParent<VelocityControl>() != null)
            {
                runner.droneBodyContacts++;
            }
            else
            {
                // Not an obstacle to the swarm (OlfatiSaber sees only the Obstacle layer): street
                // furniture, the ground, a goal patch's props. Named, so a pole strike is visible.
                string key = $"{c.collider.name} ({LayerMask.LayerToName(c.collider.gameObject.layer)})";
                runner.otherContacts.TryGetValue(key, out int n);
                runner.otherContacts[key] = n + 1;
            }
        }
    }

    // ------------------------------------------------------------------ state

    [NonSerialized] public bool measuring;
    int contacts, hardContacts, droneBodyContacts;
    float maxImpact;
    readonly Dictionary<string, int> otherContacts = new Dictionary<string, int>();
    readonly Dictionary<string, int> deathReasons = new Dictionary<string, int>();

    swarmSpawn spawner;
    readonly List<Transform> parents = new List<Transform>();
    readonly List<VelocityControl> vcs = new List<VelocityControl>();
    readonly List<AttitudeAlgorithm> atts = new List<AttitudeAlgorithm>();
    readonly List<Rigidbody> rbs = new List<Rigidbody>();

    // Every SwarmManager setting a set may override -- the public instance fields the inspector
    // serialises as numbers -- and their values as the scene authored them.
    readonly List<FieldInfo> settings = new List<FieldInfo>();
    readonly Dictionary<string, object> sceneValues = new Dictionary<string, object>();
    float pilotYaw;

    void OnEnable() { Application.logMessageReceived += OnLog; }
    void OnDisable() { Application.logMessageReceived -= OnLog; }

    void OnLog(string msg, string stack, LogType type)
    {
        if (!measuring) return;
        int k = msg.IndexOf("Reason: ", StringComparison.Ordinal);
        if (msg.StartsWith("[DroneHealthMonitor] Parking", StringComparison.Ordinal) && k >= 0)
        {
            int end = msg.IndexOf('.', k);
            string reason = end > k ? msg.Substring(k + 8, end - k - 8) : msg.Substring(k + 8);
            deathReasons.TryGetValue(reason, out int n);
            deathReasons[reason] = n + 1;
        }
    }

    // Every flight script runs in FixedUpdate, so more fixed steps per rendered frame changes throughput
    // only, never the 0.02 s physics step. Set here and again before every flight, not in Boot:
    // VrFramePacing caps maximumDeltaTime at 0.1 s in its own AfterSceneLoad, which would quietly hold a
    // 20x run to ~5 steps a frame. Start runs after every AfterSceneLoad, and sim_per_wall checks it.
    void ApplyTiming()
    {
        Time.maximumDeltaTime = 1.0f;
        Time.timeScale = cfg.timeScale;
    }

    IEnumerator Start()
    {
        Application.targetFrameRate = -1;
        QualitySettings.vSyncCount = 0;
        ApplyTiming();

        // Let every Start run (the swarm spawns in swarmSpawn.Start).
        for (int i = 0; i < 5; i++) yield return null;
        float waitUntil = Time.realtimeSinceStartup + 60f;
        spawner = FindObjectOfType<swarmSpawn>();
        while (spawner == null || spawner.swarm == null || spawner.swarm.Count == 0)
        {
            if (Time.realtimeSinceStartup > waitUntil)
            {
                Fatal(ExitRuntimeError, $"no swarm spawned in {cfg.scene} within 60 s");
                yield break;
            }
            yield return null;
            spawner = FindObjectOfType<swarmSpawn>();
        }
        for (int i = 0; i < 5; i++) yield return null;
        DisableCameras();   // the drones' FPV cameras did not exist at Boot

        int goalsPlaced = 0;
        if (!string.IsNullOrEmpty(cfg.goalReplaySession))
        {
            foreach (var r in FindObjectsOfType<GoalPatchReplacer>()) goalsPlaced += r.PlacedGoals.Count;
            if (goalsPlaced == 0)
            {
                Fatal(ExitRuntimeError, $"GoalPatchReplacer placed no goals from '{cfg.goalReplaySession}' (see its log line)");
                yield break;
            }
        }

        SwarmManager sm = SwarmManager.Instance;
        if (sm == null || InputManager.Instance == null)
        {
            Fatal(ExitRuntimeError, $"{cfg.scene} has no SwarmManager or InputManager");
            yield break;
        }
        InputManager.Instance.enabled = false;   // the bench drives InputStatus

        foreach (FieldInfo f in typeof(SwarmManager).GetFields(BindingFlags.Public | BindingFlags.Instance))
        {
            Type t = f.FieldType;
            if (t == typeof(float) || t == typeof(int) || t == typeof(bool) || t.IsEnum)
            {
                settings.Add(f);
                sceneValues[f.Name] = f.GetValue(sm);
            }
        }
        string invalid = ValidateSets();
        if (invalid != null)
        {
            Fatal(ExitConfigError, invalid);
            yield break;
        }

        for (int d = 0; d < spawner.swarm.Count; d++)
        {
            Transform p = spawner.swarm[d].transform.Find("DroneParent");
            parents.Add(p);
            vcs.Add(p.GetComponent<VelocityControl>());
            atts.Add(p.GetComponent<AttitudeAlgorithm>());
            rbs.Add(p.GetComponent<Rigidbody>());
            p.gameObject.AddComponent<ContactProbe>().runner = this;
        }

        AppendLine(outPath, new Json()
            .Str("event", "start").Str("scene", cfg.scene).Str("unity", Application.unityVersion)
            .Num("timeScale", cfg.timeScale).Num("settleTime", cfg.settleTime).Num("altitude", cfg.altitude)
            .Num("capture", cfg.capture).Bool("pilotHeading", cfg.pilotHeading)
            .Str("goalReplaySession", cfg.goalReplaySession).Int("goalsPlaced", goalsPlaced)
            .Int("drones", vcs.Count).Int("paramSets", cfg.paramSets.Length).Int("scenarios", cfg.scenarios.Length)
            .ToString());

        foreach (ParamSet ps in cfg.paramSets)
        {
            ApplyParams(ps);
            WriteParamsLine(ps);
            if (StickFrameIsBody())
            {
                Fatal(ExitConfigError, $"set '{ps.label}': the stick is in the Body frame (InputManager.commandFrame with a " +
                                       "non-hull attitude mode), which the bench cannot steer in world terms");
                yield break;
            }
            for (int s = 0; s < cfg.scenarios.Length; s++)
                yield return RunScenario(ps, s, cfg.scenarios[s]);
        }

        AppendLine(outPath, new Json().Str("event", "done").Bool("done", true).ToString());
        Debug.Log("[SwarmBench] finished");
        Exit(0);
    }

    // ------------------------------------------------------------------ parameter sets

    FieldInfo Setting(string name) => settings.Find(f => f.Name == name);

    static bool TryConvert(FieldInfo f, float v, out object value, out string error)
    {
        value = null;
        error = null;
        if (f.FieldType == typeof(float)) { value = v; return true; }
        if (f.FieldType == typeof(bool))
        {
            if (v != 0f && v != 1f) { error = $"{f.Name} is a bool: give 0 or 1, not {v}"; return false; }
            value = v == 1f;
            return true;
        }
        int i = Mathf.RoundToInt(v);
        if (Mathf.Abs(v - i) > 1e-4f) { error = $"{f.Name} takes an integer, not {v}"; return false; }
        if (f.FieldType == typeof(int)) { value = i; return true; }
        if (!Enum.IsDefined(f.FieldType, i)) { error = $"{f.Name}: {i} is not a {f.FieldType.Name} value"; return false; }
        value = Enum.ToObject(f.FieldType, i);
        return true;
    }

    // All sets are checked before anything flies: a typo must cost seconds, not a 15-minute run on the
    // wrong values.
    string ValidateSets()
    {
        var labels = new HashSet<string>();
        foreach (ParamSet ps in cfg.paramSets)
        {
            if (string.IsNullOrEmpty(ps.label)) return "a parameter set has no label";
            if (!labels.Add(ps.label)) return $"two parameter sets are labelled '{ps.label}'";
            int n = ps.names?.Length ?? 0, nv = ps.values?.Length ?? 0;
            if (n != nv) return $"set '{ps.label}' has {n} names but {nv} values";
            for (int i = 0; i < n; i++)
            {
                FieldInfo f = Setting(ps.names[i]);
                if (f == null) return $"set '{ps.label}': SwarmManager has no setting '{ps.names[i]}'";
                if (!TryConvert(f, ps.values[i], out object v, out string err)) return $"set '{ps.label}': {err}";
                if (f.Name == nameof(SwarmManager.fillLookDirectionGap) && (bool)v && !cfg.pilotHeading)
                    return $"set '{ps.label}' turns fillLookDirectionGap on without pilotHeading; the fill reads " +
                           "the pilot's body yaw, so it would silently do nothing";
            }
        }
        return null;
    }

    void ApplyParams(ParamSet ps)
    {
        SwarmManager sm = SwarmManager.Instance;
        foreach (FieldInfo f in settings) f.SetValue(sm, sceneValues[f.Name]);
        for (int i = 0; i < (ps.names?.Length ?? 0); i++)
        {
            FieldInfo f = Setting(ps.names[i]);
            TryConvert(f, ps.values[i], out object v, out _);
            f.SetValue(sm, v);
        }
        // Explicit, because the scenes set it on: with no injected heading the fill is inert anyway
        // (BodyYawValid stays false), and saying so keeps the results honest about what flew.
        if (!cfg.pilotHeading) sm.fillLookDirectionGap = false;
        // OnValidate is what raises swarmParamsChanged for every drone.
        typeof(SwarmManager).GetMethod("OnValidate", BindingFlags.NonPublic | BindingFlags.Instance).Invoke(sm, null);
    }

    void WriteParamsLine(ParamSet ps)
    {
        SwarmManager sm = SwarmManager.Instance;
        var overrides = new Json();
        for (int i = 0; i < (ps.names?.Length ?? 0); i++) overrides.Num(ps.names[i], ps.values[i]);
        var effective = new Json();
        foreach (FieldInfo f in settings) effective.Value(f.Name, f.GetValue(sm));
        AppendLine(outPath, new Json()
            .Str("event", "params").Str("set", ps.label)
            .Raw("overrides", overrides.ToString())
            .Raw("effective", effective.ToString())
            .Bool("lookGapFillForcedOff", !cfg.pilotHeading)
            .ToString());
        Debug.Log($"[SwarmBench] set '{ps.label}': {overrides}");
    }

    // ------------------------------------------------------------------ stick

    // The frame SwarmAlgorithm.readInputs resolves the stick in. The bench steers in world terms and
    // undoes it; a Body frame would make the command depend on each drone's own heading.
    static bool StickFrameIsBody() => ResolveStickFrame() == InputManager.CommandFrame.Body;

    static InputManager.CommandFrame ResolveStickFrame()
    {
        InputManager.CommandFrame frame = InputManager.Instance.ActiveCommandFrame;
        SwarmManager.AttitudeAlgorithm att = SwarmManager.Instance.GetSelectedAttitudeAlgorithm();
        if (att == SwarmManager.AttitudeAlgorithm.LOCAL_CONVEXHULL || att == SwarmManager.AttitudeAlgorithm.GLOBAL_CONVEXHULL)
            frame = InputManager.CommandFrame.VR;
        return frame;
    }

    // World command (cx, cz) -> stick. VelocityControl resolves the stick as Euler(0, psi, 0) * (roll, 0,
    // pitch) with psi the body yaw in the VR frame and 0 in the World frame; this is its inverse.
    static void SetStick(float cx, float cz, float spread)
    {
        float psi = ResolveStickFrame() == InputManager.CommandFrame.VR ? PyUniSharingFast.BodyYawDegrees * Mathf.Deg2Rad : 0f;
        float cs = Mathf.Cos(psi), sn = Mathf.Sin(psi);
        var st = InputManager.Instance.InputStatus;
        st["roll"] = cx * cs - cz * sn;
        st["pitch"] = cx * sn + cz * cs;
        st["throttle"] = 0f;
        st["yaw"] = 0f;
        st["spread"] = spread;
    }

    static void SetPilotYaw(float degrees)
    {
        setBodyYaw.Invoke(null, new object[] { Mathf.Repeat(degrees, 360f) });
        setBodyYawValid.Invoke(null, new object[] { true });
    }

    static float SpreadAt(float[] s, float t)
    {
        if (s == null || s.Length < 2) return -1f;
        int n = s.Length / 2;
        if (t <= s[0]) return s[1];
        for (int q = 1; q < n; q++)
        {
            float t1 = s[2 * q];
            if (t <= t1)
            {
                float t0 = s[2 * (q - 1)], v0 = s[2 * (q - 1) + 1], v1 = s[2 * q + 1];
                return t1 - t0 < 1e-6f ? v1 : v0 + (v1 - v0) * (t - t0) / (t1 - t0);
            }
        }
        return s[2 * (n - 1) + 1];
    }

    // ------------------------------------------------------------------ one flight

    IEnumerator RunScenario(ParamSet ps, int index, Scenario sc)
    {
        float[] wps = sc.wps ?? Array.Empty<float>();

        // ---- reset the whole swarm at the start point, at altitude, and let the formation settle
        measuring = false;
        ApplyParams(ps);   // a previous spread schedule leaves its d_ref mirrored into SwarmManager
        ApplyTiming();
        SetStick(0f, 0f, -1f);
        spawner.ResetToPos(new Vector3(sc.sx, cfg.altitude, sc.sz));
        for (int d = 0; d < rbs.Count; d++)
        {
            rbs[d].velocity = Vector3.zero;
            rbs[d].angularVelocity = Vector3.zero;
        }
        pilotYaw = wps.Length >= 2 ? Mathf.Atan2(wps[0] - sc.sx, wps[1] - sc.sz) * Mathf.Rad2Deg : 0f;
        if (cfg.pilotHeading) SetPilotYaw(pilotYaw);
        float t0 = Time.fixedTime;
        while (Time.fixedTime - t0 < cfg.settleTime) yield return new WaitForFixedUpdate();

        int aliveAtStart = CountAlive();

        // ---- fly
        contacts = 0; hardContacts = 0; droneBodyContacts = 0; maxImpact = 0f;
        otherContacts.Clear();
        deathReasons.Clear();
        LookGapStats look = cfg.pilotHeading ? new LookGapStats() : null;
        measuring = true;
        int wp = 0, wpDone = 0;
        float wpT0 = Time.fixedTime;
        int nWp = wps.Length / 4;
        Vector3 prevC = Centroid(out _);
        float pathLen = 0f, trackSum = 0f, minPair = 1e9f;
        int trackN = 0, nearTicks = 0;
        double hullSum = 0, nnSum = 0, speedSum = 0;
        int hullN = 0, nnN = 0, speedN = 0, splitN = 0, tick = 0;
        StringBuilder traj = sc.record ? new StringBuilder("t,alive_mask," + TrajHeader() + "\n") : null;
        float wall0 = Time.realtimeSinceStartup;
        t0 = Time.fixedTime;
        while (Time.fixedTime - t0 < sc.T)
        {
            Vector3 c = Centroid(out int alive);
            if (alive == 0) break;
            float t = Time.fixedTime - t0;
            if (tick > 0) look?.Sample(this, pilotYaw);   // the step that just ran, under the yaw set before it
            if (tick % 10 == 0 && alive >= 4)
                SampleFormation(ref hullSum, ref hullN, ref nnSum, ref nnN, ref splitN);
            for (int i = 0; i < vcs.Count; i++)
            {
                if (!vcs[i].State.IsAlive) continue;
                Vector3 v = rbs[i].velocity; v.y = 0f;
                speedSum += v.magnitude; speedN++;
            }
            if (traj != null && tick % 5 == 0)
            {
                traj.Append(t.ToString("F2", Inv)).Append(',');
                int mask = 0;
                for (int i = 0; i < vcs.Count; i++) if (vcs[i].State.IsAlive) mask |= 1 << i;
                traj.Append(mask);
                for (int i = 0; i < vcs.Count; i++)
                    traj.Append(',').Append(parents[i].position.x.ToString("F3", Inv))
                        .Append(',').Append(parents[i].position.z.ToString("F3", Inv));
                traj.Append('\n');
            }
            tick++;
            Vector3 step = c - prevC; step.y = 0f;
            pathLen += step.magnitude;

            // ---- steer toward the waypoint in world terms; a pilot heading turns to face it
            float cx = 0f, cz = 0f;
            if (wp < nWp)
            {
                float dx = wps[wp * 4] - c.x, dz = wps[wp * 4 + 1] - c.z;
                float dd = Mathf.Sqrt(dx * dx + dz * dz);
                float timeout = wps[wp * 4 + 3];
                if (dd < cfg.capture || (timeout > 0f && Time.fixedTime - wpT0 > timeout))
                {
                    if (dd < cfg.capture) wpDone++;
                    wp++; wpT0 = Time.fixedTime;
                }
                if (wp < nWp)
                {
                    dx = wps[wp * 4] - c.x; dz = wps[wp * 4 + 1] - c.z;
                    dd = Mathf.Sqrt(dx * dx + dz * dz);
                    float mag = Mathf.Clamp01(wps[wp * 4 + 2]);
                    if (dd > 1e-6f)
                    {
                        cx = dx / dd * mag; cz = dz / dd * mag;
                        float bearing = Mathf.Atan2(dx, dz) * Mathf.Rad2Deg;
                        pilotYaw = Mathf.MoveTowardsAngle(pilotYaw, bearing, cfg.pilotYawRate * Time.fixedDeltaTime);
                    }
                }
            }
            if (cfg.pilotHeading) SetPilotYaw(pilotYaw);
            SetStick(cx, cz, SpreadAt(sc.spread, t));
            float cm = Mathf.Sqrt(cx * cx + cz * cz);
            if (cm > 1e-6f && t > 0f)
            {
                trackSum += (step.x * cx + step.z * cz) / cm / Time.fixedDeltaTime;
                trackN++;
            }
            prevC = c;

            for (int i = 0; i < vcs.Count; i++)
            {
                if (!vcs[i].State.IsAlive) continue;
                for (int j = i + 1; j < vcs.Count; j++)
                {
                    if (!vcs[j].State.IsAlive) continue;
                    float dd = Vector3.Distance(parents[i].position, parents[j].position);
                    if (dd < minPair) minPair = dd;
                    if (dd < 2f) nearTicks++;
                }
            }
            yield return new WaitForFixedUpdate();
        }
        measuring = false;
        SetStick(0f, 0f, -1f);
        float simS = Time.fixedTime - t0, wallS = Time.realtimeSinceStartup - wall0;
        float simPerWall = wallS > 1e-3f ? simS / wallS : float.NaN;
        if (simPerWall < 0.5f * cfg.timeScale)
            Debug.LogWarning($"[SwarmBench] {ps.label} #{index}: {simPerWall:F1}x real time against a timeScale of {cfg.timeScale}; " +
                             $"maximumDeltaTime is {Time.maximumDeltaTime}");
        if (traj != null)
        {
            string dir = Path.GetDirectoryName(Path.GetFullPath(outPath));
            File.WriteAllText(Path.Combine(dir, $"traj_{ps.label}_{index}.csv"), traj.ToString());
        }

        int aliveEnd = CountAlive();
        var others = new Json();
        foreach (var kv in otherContacts) others.Int(kv.Key, kv.Value);
        var deaths = new Json();
        foreach (var kv in deathReasons) deaths.Int(kv.Key, kv.Value);
        int otherTotal = 0;
        foreach (int n in otherContacts.Values) otherTotal += n;

        Json line = new Json()
            .Str("event", "flight").Str("set", ps.label).Int("scenario", index).Str("kind", sc.kind)
            .Int("alive_start", aliveAtStart).Int("alive_end", aliveEnd)
            .Int("contacts", contacts).Int("hard_contacts", hardContacts).Num("max_impact", maxImpact, "F3")
            .Int("drone_body_contacts", droneBodyContacts).Int("other_contacts", otherTotal)
            .Raw("other_names", others.ToString())
            .Num("path_len", pathLen, "F2").Num("track_sum", trackSum, "F2").Int("track_n", trackN)
            .Num("min_pair", minPair, "F3").Int("near_ticks", nearTicks).Int("wp_done", wpDone)
            .Num("hull_frac", hullN > 0 ? hullSum / hullN : double.NaN, "F4")
            .Num("nn_mean", nnN > 0 ? nnSum / nnN : double.NaN, "F3")
            .Num("split_frac", hullN > 0 ? (double)splitN / hullN : double.NaN, "F4")
            .Num("mean_speed", speedN > 0 ? speedSum / speedN : double.NaN, "F3")
            .Num("sim_s", simS, "F2").Num("wall_s", wallS, "F2").Num("sim_per_wall", simPerWall, "F2")
            .Raw("deaths", deaths.ToString());
        look?.Append(line);
        AppendLine(outPath, line.ToString());
        Debug.Log($"[SwarmBench] {ps.label} #{index} {sc.kind}: contacts {contacts} alive {aliveEnd}/{aliveAtStart} " +
                  $"({simPerWall:F1}x real time)");
    }

    // ------------------------------------------------------------------ look-direction metrics

    // Half the FPV camera's horizontal FOV (66.3 deg vertical at 16:9). Past this the look direction is
    // in no shown drone's image at all.
    const float HalfHfovDeg = 49.25f;

    // What the look-direction gap fill is for: how far the pilot's look direction sits from the nearest
    // hull heading (geometric, before and after the fill's shift) and from the nearest drone whose feed
    // is actually shown.
    class LookGapStats
    {
        readonly List<float> gapRaw = new List<float>(), gapPost = new List<float>(), liveShown = new List<float>();
        readonly List<float> gapPostWhereRaw30 = new List<float>(), liveShownWhereRaw30 = new List<float>();
        int fillActiveTicks, shiftTicks, noShownTicks, ticks;
        double yawRateSum, maxGapSum, hullFracSum;
        int yawRateN, maxGapN, hullFracN;

        public void Sample(SwarmBenchRunner r, float pilotYaw)
        {
            ticks++;
            float raw = AttitudeAlgorithm.SharedLookGapRawDeg;
            float post = AttitudeAlgorithm.SharedLookGapDeg;
            if (AttitudeAlgorithm.SharedLookGapFillActive) fillActiveTicks++;
            if (!float.IsNaN(raw))
            {
                gapRaw.Add(raw);
                gapPost.Add(post);
                if (raw - post > 0.01f) shiftTicks++;
            }
            if (AttitudeAlgorithm.SharedMaxGapDeg < 359f) { maxGapSum += AttitudeAlgorithm.SharedMaxGapDeg; maxGapN++; }
            if (AttitudeAlgorithm.SharedAliveCount > 0)
            {
                hullFracSum += AttitudeAlgorithm.SharedHullVertexCount / (double)AttitudeAlgorithm.SharedAliveCount;
                hullFracN++;
            }

            float bestShown = 180f;
            bool anyShown = false;
            for (int i = 0; i < r.vcs.Count; i++)
            {
                if (!r.vcs[i].State.IsAlive) continue;
                if (r.atts[i] != null && r.atts[i].BoundaryFeedReady)
                {
                    float yawDeg = r.vcs[i].State.Angles.y * Mathf.Rad2Deg;
                    bestShown = Mathf.Min(bestShown, Mathf.Abs(Mathf.DeltaAngle(yawDeg, pilotYaw)));
                    anyShown = true;
                }
                yawRateSum += Mathf.Abs(r.rbs[i].angularVelocity.y) * Mathf.Rad2Deg;
                yawRateN++;
            }
            if (anyShown) liveShown.Add(bestShown); else noShownTicks++;
            if (!float.IsNaN(raw) && raw > 30f)
            {
                gapPostWhereRaw30.Add(post);
                if (anyShown) liveShownWhereRaw30.Add(bestShown);
            }
        }

        public void Append(Json j)
        {
            j.Int("ticks", ticks)
             .Num("hull_frac_att", hullFracN > 0 ? hullFracSum / hullFracN : double.NaN, "F4")
             .Num("fill_active_frac", ticks > 0 ? fillActiveTicks / (double)ticks : double.NaN, "F4")
             .Num("shift_frac", ticks > 0 ? shiftTicks / (double)ticks : double.NaN, "F4")
             .Num("gap_raw_mean", Mean(gapRaw), "F4").Num("gap_raw_p90", Pct(gapRaw, 0.9f), "F4")
             .Num("gap_raw_frac30", FracAbove(gapRaw, 30f), "F4").Num("gap_raw_frac40", FracAbove(gapRaw, 40f), "F4")
             .Num("gap_post_mean", Mean(gapPost), "F4").Num("gap_post_p90", Pct(gapPost, 0.9f), "F4")
             .Num("gap_post_frac30", FracAbove(gapPost, 30f), "F4").Num("gap_post_frac40", FracAbove(gapPost, 40f), "F4")
             .Int("n_raw30", gapPostWhereRaw30.Count).Num("gap_post_mean_raw30", Mean(gapPostWhereRaw30), "F4")
             .Num("live_shown_mean", Mean(liveShown), "F4").Num("live_shown_p90", Pct(liveShown, 0.9f), "F4")
             .Num("live_shown_frac30", FracAbove(liveShown, 30f), "F4")
             .Num("live_shown_blind_frac", FracAbove(liveShown, HalfHfovDeg), "F4")
             .Num("live_shown_mean_raw30", Mean(liveShownWhereRaw30), "F4").Int("n_live_raw30", liveShownWhereRaw30.Count)
             .Int("no_shown_ticks", noShownTicks)
             .Num("yaw_rate_mean", yawRateN > 0 ? yawRateSum / yawRateN : double.NaN, "F4")
             .Num("max_gap_mean", maxGapN > 0 ? maxGapSum / maxGapN : double.NaN, "F4");
        }

        static double Mean(List<float> v)
        {
            if (v.Count == 0) return double.NaN;
            double s = 0; foreach (float x in v) s += x;
            return s / v.Count;
        }

        static double Pct(List<float> v, float p)
        {
            if (v.Count == 0) return double.NaN;
            var s = new List<float>(v);
            s.Sort();
            return s[Mathf.Clamp(Mathf.RoundToInt(p * (s.Count - 1)), 0, s.Count - 1)];
        }

        static double FracAbove(List<float> v, float threshold)
        {
            if (v.Count == 0) return double.NaN;
            int n = 0; foreach (float x in v) if (x > threshold) n++;
            return n / (double)v.Count;
        }
    }

    // ------------------------------------------------------------------ formation geometry

    string TrajHeader()
    {
        var sb = new StringBuilder();
        for (int i = 0; i < vcs.Count; i++) { if (i > 0) sb.Append(','); sb.Append('x').Append(i).Append(",z").Append(i); }
        return sb.ToString();
    }

    // Hull vertices / alive, mean nearest-neighbour distance, and whether the swarm is split into more
    // than one component at 25 m linkage. swarm_replica.py measures the same three the same way.
    void SampleFormation(ref double hullSum, ref int hullN, ref double nnSum, ref int nnN, ref int splitN)
    {
        var pts = new List<Vector2>();
        for (int i = 0; i < vcs.Count; i++)
            if (vcs[i].State.IsAlive) pts.Add(new Vector2(parents[i].position.x, parents[i].position.z));
        int n = pts.Count;
        hullSum += HullCount(pts) / (double)n; hullN++;
        for (int i = 0; i < n; i++)
        {
            float best = float.MaxValue;
            for (int j = 0; j < n; j++) if (j != i) best = Mathf.Min(best, Vector2.Distance(pts[i], pts[j]));
            nnSum += best; nnN++;
        }
        var comp = new int[n];
        for (int i = 0; i < n; i++) comp[i] = -1;
        int nc = 0;
        var stack = new Stack<int>();
        for (int i = 0; i < n; i++)
        {
            if (comp[i] >= 0) continue;
            comp[i] = nc; stack.Push(i);
            while (stack.Count > 0)
            {
                int u = stack.Pop();
                for (int j = 0; j < n; j++)
                    if (comp[j] < 0 && Vector2.Distance(pts[u], pts[j]) < 25f) { comp[j] = nc; stack.Push(j); }
            }
            nc++;
        }
        if (nc > 1) splitN++;
    }

    static int HullCount(List<Vector2> p)
    {
        int n = p.Count;
        if (n < 3) return n;
        var s = new List<Vector2>(p);
        s.Sort((u, v) => u.x != v.x ? u.x.CompareTo(v.x) : u.y.CompareTo(v.y));
        var h = new Vector2[2 * n + 2];
        int k = 0;
        for (int i = 0; i < n; i++)
        {
            while (k >= 2 && Cross(h[k - 2], h[k - 1], s[i]) <= 1e-9f) k--;
            h[k++] = s[i];
        }
        for (int i = n - 2, t = k + 1; i >= 0; i--)
        {
            while (k >= t && Cross(h[k - 2], h[k - 1], s[i]) <= 1e-9f) k--;
            h[k++] = s[i];
        }
        return k - 1;
    }

    static float Cross(Vector2 o, Vector2 a, Vector2 b) => (a.x - o.x) * (b.y - o.y) - (a.y - o.y) * (b.x - o.x);

    Vector3 Centroid(out int alive)
    {
        Vector3 sum = Vector3.zero; alive = 0;
        for (int i = 0; i < vcs.Count; i++)
        {
            if (!vcs[i].State.IsAlive) continue;
            sum += parents[i].position; alive++;
        }
        return alive > 0 ? sum / alive : Vector3.zero;
    }

    int CountAlive()
    {
        int n = 0;
        foreach (var vc in vcs) if (vc.State.IsAlive) n++;
        return n;
    }

    // ------------------------------------------------------------------ JSON lines

    static readonly CultureInfo Inv = CultureInfo.InvariantCulture;

    // A minimal JSON object writer. JsonUtility cannot write dictionaries, and every number goes out in
    // the invariant culture, so a decimal-comma locale cannot corrupt a results file.
    public sealed class Json
    {
        readonly StringBuilder sb = new StringBuilder("{");
        bool first = true;

        Json Key(string k)
        {
            if (!first) sb.Append(',');
            first = false;
            sb.Append('"').Append(Escape(k)).Append("\":");
            return this;
        }

        public Json Str(string k, string v) { Key(k); sb.Append('"').Append(Escape(v ?? "")).Append('"'); return this; }
        public Json Int(string k, long v) { Key(k); sb.Append(v.ToString(Inv)); return this; }
        public Json Bool(string k, bool v) { Key(k); sb.Append(v ? "true" : "false"); return this; }
        public Json Raw(string k, string json) { Key(k); sb.Append(json); return this; }

        public Json Num(string k, double v, string format = "G7")
        {
            Key(k);
            sb.Append(double.IsNaN(v) || double.IsInfinity(v) ? "null" : v.ToString(format, Inv));
            return this;
        }

        public Json Value(string k, object v)
        {
            switch (v)
            {
                case float f: return Num(k, f);
                case int i: return Int(k, i);
                case bool b: return Bool(k, b);
                case Enum e: return Int(k, Convert.ToInt32(e, Inv));
                default: return Str(k, v?.ToString());
            }
        }

        public override string ToString() => sb + "}";

        static string Escape(string s) => s.Replace("\\", "\\\\").Replace("\"", "\\\"").Replace("\n", "\\n").Replace("\r", "");
    }
}
