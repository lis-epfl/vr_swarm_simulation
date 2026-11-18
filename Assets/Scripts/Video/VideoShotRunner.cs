using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Text;
using UnityEngine;

/// <summary>
/// Flies scripted shots for the instruction videos headlessly and writes the capture rig's two cameras
/// straight to MP4 through ffmpeg. Started by VideoShotLauncher.Run in batchmode on a copy of the project
/// (Video/README.md), with VIDEO_SHOTS_CONFIG naming a JSON shot list; inert otherwise.
///
/// Game time runs at exactly 1/fps per rendered frame (Time.captureFramerate), so every recorded frame is
/// one video frame and the flight plays back at real speed however slowly the copy renders. The pilot is
/// scripted the way the bench's is (it writes InputManager.InputStatus, InputManager disabled), and it turns
/// the body yaw with the yaw stick through PyUniSharingFast, exactly as a participant's stick does, so the
/// interface camera sees the ring of screens turn. Run PyUniSharingFast with STITCH_SHM_SUFFIX set: its
/// named sections are machine-wide.
/// </summary>
[DefaultExecutionOrder(-200)]
public class VideoShotRunner : MonoBehaviour
{
    public const string ConfigEnv = "VIDEO_SHOTS_CONFIG";

    [Serializable]
    public class Config
    {
        public string scene = "Assets/Scenes/ScenicPracticeBlockWorld.unity";
        public string outDir = "";
        public string ffmpeg = "ffmpeg";
        public int width = 1920, height = 1080, fps = 30;
        public float initialSettle = 10f;
        public int drones = 0;                     // drones to spawn (a circle pattern's count); 0 = as the scene says
        public Shot[] shots = new Shot[0];
    }

    [Serializable]
    public class Shot
    {
        public string name = "shot";
        public float[] start = new float[0];       // (x, z); empty = stay where the swarm is
        public float alt = 15f;
        public float yaw = 999f;                   // body yaw to turn to while settling; 999 = leave it
        public float settle = 6f;
        public float orthoSize = 40f;
        public bool headingUp = true;
        public float upYaw = 0f;                   // with headingUp off, the world heading up the overhead picture (90 = east)
        public bool follow = true;
        public float coneLength = 12f;
        public bool overhead = true;
        public bool iface = true;
        public float[] goal = new float[0];        // (x, z) of the goal, to pick its hat wearer
        public bool front = false;                 // also record the front screen's own feed, and where the hat is in it
        public float[] centre = new float[0];      // (x, z): hold the overhead view here instead of following the swarm
        public bool highlightHats = false;         // ring the hat wearers in the overhead view
        public string[] closeups = new string[0];  // GameObject names to film close up, one after another (<shot>_closeup.mp4)
        public float closeupDur = 3f, closeupDist = 2.6f, closeupHeight = 1.75f, closeupFov = 32f, closeupRise = 0.25f;
        public float hatRange = 80f;
        public View[] views = new View[0];         // still camera positions, one after another (<shot>_views.mp4)
        public bool hideStartLine = false;         // hide the start line's tint (a camera looking through it is tinted blue)               // findHat: the farthest the hat may be (the recorder scores to 80 m)
        public float gimbal = 999f;                // gimbal dial, -1 (straight down) .. 0 (level) .. 1; 999 = leave it
        public float ifaceBack = 0.9f, ifaceUp = 0.45f, ifaceFov = 90f, ifaceLookAhead = 2f, ifaceLookDown = 0.25f;
        public float ifaceSmooth = 1.2f;           // seconds the interface camera takes to follow the head round; 0 = locked
                                                   // to it, so what turns with the pilot (one drone's screen) holds still
        public Seg[] segs = new Seg[0];
    }

    [Serializable]
    public class View
    {
        public float[] from = new float[0];        // camera position (x, y, z)
        public string target = "";                 // look at this object's position...
        public float[] point = new float[0];       // ...or at this point (x, y, z)
        public float pitch = 0f;                   // degrees above the horizontal
        public float fov = 46.2f;                  // vertical, degrees (46.2 = a drone feed's)
        public float dur = 3f;
    }

    [Serializable]
    public class Seg
    {
        public string kind = "hold";               // hold | goto | stick | gimbal | findHat | patrol
        public float dur = 5f;                     // the most it may last
        public float[] target = new float[0];      // (x, z) for goto / hold; hold with none = where the swarm is
        public float speed = 1f;                   // goto: stick magnitude
        public float arrive = 8f;                  // goto: ends within this of the target
        public bool face = true;                   // goto: turn the body yaw towards the way it flies
        public float[] stick = new float[0];       // stick: (right, forward, climb, yaw) in the body frame
        public float after = 3f;                   // findHat: seconds to keep recording once it is in front
        public float ramp = 0.4f;                  // stick: seconds to ease the stick in and out
        public float gimbalTo = 999f;              // gimbal: dial value to ramp to over dur (holding position meanwhile)
        public float[] path = new float[0];        // patrol: waypoints (x0, z0, x1, z1, ...) flown in a loop, facing the way
                                                   // it flies, until a shown camera has the hat; then it stops and turns to it
        public bool untilHat = true;               // patrol: stop for the hat (false: just fly the path for dur)
        public float approach = 0f;                // patrol: once seen, fly towards the hat to this horizontal distance
                                                   // before stopping (0 = stop where it was seen)
    }

    static Config cfg;
    readonly List<GameObject> drones = new List<GameObject>();
    VideoCaptureRig rig;
    VideoShotCapture capture;
    float nextTrace;
    float dial;                                    // the gimbal dial as last set
    float gimbalFrom = float.NaN;
    Camera closeCam;
    Renderer compass;
    float nextCompassScan;
    RenderTexture closeRT;
    swarmSpawn spawner;
    readonly StringBuilder events = new StringBuilder();

    // ---------------------------------------------------------------------------------------------
    [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.AfterSceneLoad)]
    static void Boot()
    {
        string path = Environment.GetEnvironmentVariable(ConfigEnv);
        if (string.IsNullOrEmpty(path)) return;
        try { cfg = JsonUtility.FromJson<Config>(File.ReadAllText(path)); }
        catch (Exception e) { Fatal($"cannot read {path}: {e.Message}"); return; }
        if (string.IsNullOrEmpty(cfg.outDir)) cfg.outDir = Path.GetDirectoryName(path);
        Directory.CreateDirectory(cfg.outDir);
        foreach (var c in FindObjectsOfType<UDPReceiverManager>(true)) c.enabled = false;
        foreach (var c in FindObjectsOfType<ExperimentRecorder>(true)) c.enabled = false;
        if (cfg.drones > 0)
            foreach (var sp in FindObjectsOfType<swarmSpawn>(true))
            {
                if (sp.IsCirclePattern) sp.circleDroneCount = cfg.drones;
                else { sp.dronesAlongX = cfg.drones; sp.dronesAlongZ = 1; }
                Debug.Log($"[VideoShots] spawning {cfg.drones} drone(s)");
            }
        new GameObject("VideoShotRunner").AddComponent<VideoShotRunner>();
        Debug.Log($"[VideoShots] booted: {cfg.shots.Length} shots -> {cfg.outDir}");
    }

    static void Fatal(string msg)
    {
        Debug.LogError("[VideoShots] " + msg);
        Exit(2);
    }

    static void Exit(int code)
    {
#if UNITY_EDITOR
        UnityEditor.EditorApplication.Exit(code);
#else
        Application.Quit(code);
#endif
    }

    // ---------------------------------------------------------------------------------------------
    IEnumerator Start()
    {
        Application.targetFrameRate = -1;
        QualitySettings.vSyncCount = 0;
        Time.captureFramerate = cfg.fps;
        Time.maximumDeltaTime = 1f;

        for (int i = 0; i < 5; i++) yield return null;
        float giveUp = Time.realtimeSinceStartup + 120f;
        while ((spawner = FindObjectOfType<swarmSpawn>()) == null || spawner.swarm == null || spawner.swarm.Count == 0)
        {
            if (Time.realtimeSinceStartup > giveUp) { Fatal("no swarm spawned"); yield break; }
            yield return null;
        }
        if (InputManager.Instance == null) { Fatal("no InputManager"); yield break; }
        InputManager.Instance.enabled = false;
        ZeroSticks();

        rig = new GameObject("VideoCaptureRig").AddComponent<VideoCaptureRig>();
        rig.showPreview = false;
        rig.manualRender = true;   // batchmode renders no camera by itself; VideoShotCapture renders them
        capture = new GameObject("VideoShotCapture").AddComponent<VideoShotCapture>();
        rig.width = cfg.width;
        rig.height = cfg.height;
        yield return null;
        yield return null;

        // The startup recentre (PyUniSharingFast) and the swarm settling out of its spawn lattice.
        yield return Frames(cfg.initialSettle);

        foreach (var shot in cfg.shots)
        {
            Debug.Log($"[VideoShots] shot {shot.name}");
            yield return RunShot(shot);
        }
        File.WriteAllText(Path.Combine(cfg.outDir, "events.txt"), events.ToString());
        Debug.Log("[VideoShots] done");
        Exit(0);
    }

    IEnumerator Frames(float seconds)
    {
        int n = Mathf.CeilToInt(seconds * cfg.fps);
        for (int i = 0; i < n; i++) yield return null;
    }

    // ---------------------------------------------------------------------------------------------
    IEnumerator RunShot(Shot shot)
    {
        rig.orthoSize = shot.orthoSize;
        rig.headingUp = shot.headingUp;
        rig.upYaw = shot.upYaw;
        rig.followSwarm = shot.follow;
        rig.coneLength = shot.coneLength;
        rig.interfaceBack = shot.ifaceBack;
        rig.interfaceUp = shot.ifaceUp;
        rig.interfaceFov = shot.ifaceFov;
        rig.interfaceLookAhead = shot.ifaceLookAhead;
        rig.interfaceLookDown = shot.ifaceLookDown;
        rig.headYawSmoothTime = Mathf.Max(0.0001f, shot.ifaceSmooth);
        if (shot.gimbal < 900f) SetDial(shot.gimbal);
        rig.useFixedCentre = shot.centre != null && shot.centre.Length >= 2;
        if (rig.useFixedCentre) rig.fixedCentre = new Vector2(shot.centre[0], shot.centre[1]);
        rig.highlightHats = shot.highlightHats;
        foreach (var sl in FindObjectsOfType<StartLine>(true)) if (shot.hideStartLine) sl.enabled = false;

        ZeroSticks();
        if (shot.start != null && shot.start.Length >= 2)
        {
            spawner.ResetToPos(new Vector3(shot.start[0], shot.alt, shot.start[1]));
            foreach (var d in Drones())
                if (SwarmRegistry.TryGet(d, out var e) && e.rigidbody != null)
                {
                    e.rigidbody.velocity = Vector3.zero;
                    e.rigidbody.angularVelocity = Vector3.zero;
                }
        }
        // Settle, turning the body yaw to the shot's heading with the stick.
        int settleFrames = Mathf.CeilToInt(shot.settle * cfg.fps);
        for (int i = 0; i < settleFrames; i++)
        {
            float yawStick = shot.yaw >= 900f ? 0f : YawTowards(shot.yaw);
            SetStick(0f, 0f, 0f, yawStick);
            yield return null;
        }

        var writers = new List<VideoShotCapture.Writer>();
        if (shot.overhead && rig.OverheadCamera != null)
            writers.Add(VideoShotCapture.Open(cfg, rig.OverheadCamera, rig.OverheadTexture, $"{shot.name}_overhead.mp4"));
        if (shot.iface && rig.InterfaceCamera != null)
            writers.Add(VideoShotCapture.Open(cfg, rig.InterfaceCamera, rig.InterfaceTexture, $"{shot.name}_interface.mp4"));
        bool closeups = shot.closeups != null && shot.closeups.Length > 0;
        bool views = shot.views != null && shot.views.Length > 0;
        if (closeups || views)
        {
            EnsureCloseCamera(shot);
            writers.Clear();
            writers.Add(VideoShotCapture.Open(cfg, closeCam, closeRT, $"{shot.name}_{(views ? "views" : "closeup")}.mp4"));
        }
        float shotT0 = Time.time;
        capture.BeginLog(Path.Combine(cfg.outDir, $"{shot.name}_log.csv"), LogHeader(), () => LogRow(shotT0));
        if (shot.front)
        {
            var fw = VideoShotCapture.Open(cfg, null, null, $"{shot.name}_front.mp4");
            capture.BeginFront(fw, Path.Combine(cfg.outDir, $"{shot.name}_front_hat.csv"), shotT0,
                               () => HatWearer(shot), FrontCamera);
        }
        capture.Begin(writers);

        if (views)
        {
            foreach (var v in shot.views)
            {
                Vector3 from = new Vector3(v.from[0], v.from[1], v.from[2]);
                Vector3 at = v.point != null && v.point.Length >= 3 ? new Vector3(v.point[0], v.point[1], v.point[2]) : from;
                if (!string.IsNullOrEmpty(v.target))
                {
                    GameObject go = GameObject.Find(v.target);
                    if (go != null) at = go.transform.position;
                    else Debug.LogWarning($"[VideoShots] no {v.target} to look at");
                }
                Vector3 dir = at - from;
                dir.y = 0f;
                closeCam.fieldOfView = v.fov;
                closeCam.transform.position = from;
                closeCam.transform.rotation = Quaternion.LookRotation(dir.sqrMagnitude > 1e-6f ? dir : Vector3.forward)
                                              * Quaternion.Euler(-v.pitch, 0f, 0f);
                events.AppendLine($"{shot.name}\tview\t{v.target}\t{Time.time - shotT0:F2}");
                int n = Mathf.CeilToInt(v.dur * cfg.fps);
                for (int i = 0; i < n; i++) { SetStick(0f, 0f, 0f, 0f); yield return null; }
            }
        }
        else if (closeups)
        {
            foreach (string target in shot.closeups)
            {
                if (!AimCloseCamera(shot, target)) { Debug.LogWarning($"[VideoShots] no {target} to film"); continue; }
                events.AppendLine($"{shot.name}\tcloseup\t{target}\t{Time.time - shotT0:F2}");
                int n = Mathf.CeilToInt(shot.closeupDur * cfg.fps);
                for (int i = 0; i < n; i++) { SetStick(0f, 0f, 0f, 0f); yield return null; }
            }
        }
        else
            foreach (var seg in shot.segs)
                yield return RunSeg(shot, seg, shotT0);

        int frames = capture.End();
        Debug.Log($"[VideoShots] wrote {shot.name} ({Time.time - shotT0:F1} s, {frames} frames)");
    }

    IEnumerator RunSeg(Shot shot, Seg seg, float shotT0)
    {
        float t0 = Time.time;
        gimbalFrom = float.NaN;
        Vector2 holdAt = seg.target != null && seg.target.Length >= 2 ? new Vector2(seg.target[0], seg.target[1]) : Centroid();
        float foundAt = -1f;
        bool seen = false;          // patrol: the hat is in view and the drone has stopped for it
        bool close = false;         // patrol: within `approach` of the hat (or no approach asked for)
        Vector2 stopAt = Vector2.zero;
        int wp = 0;
        int nPath = seg.path != null ? seg.path.Length / 2 : 0;
        while (Time.time - t0 < seg.dur)
        {
            Vector2 c = Centroid();
            if (Time.time >= nextTrace)
            {
                nextTrace = Time.time + 2f;
                Trace(shot, seg, c, shotT0);
            }
            switch (seg.kind)
            {
                case "stick":
                {
                    float[] s = seg.stick;
                    float el = Time.time - t0;
                    float k = seg.ramp > 0f ? Mathf.Clamp01(Mathf.Min(el, seg.dur - el) / seg.ramp) : 1f;
                    k = k * k * (3f - 2f * k);
                    float r = s.Length > 0 ? s[0] : 0f, f = s.Length > 1 ? s[1] : 0f;
                    float climb = s.Length > 2 ? s[2] : 0f, yaw = s.Length > 3 ? s[3] : 0f;
                    var st = InputManager.Instance.InputStatus;
                    st["roll"] = r * k; st["pitch"] = f * k; st["throttle"] = climb * k; st["yaw"] = yaw * k; st["spread"] = -1f;
                    break;
                }
                case "gimbal":
                {
                    if (seg.gimbalTo < 900f)
                    {
                        float el = Time.time - t0;
                        float k = Mathf.Clamp01(el / Mathf.Max(0.01f, seg.dur));
                        k = k * k * (3f - 2f * k);
                        if (float.IsNaN(gimbalFrom)) gimbalFrom = dial;
                        SetDial(Mathf.Lerp(gimbalFrom, seg.gimbalTo, k));
                    }
                    Vector2 hold = Hold(holdAt, c);
                    SetStick(hold.x, hold.y, 0f, 0f);
                    break;
                }
                case "goto":
                {
                    Vector2 e = holdAt - c;
                    if (e.magnitude < seg.arrive) yield break;
                    Vector2 cmd = e.magnitude > 15f ? e.normalized * seg.speed : e / 15f * seg.speed;
                    float yawStick = seg.face ? YawTowards(Mathf.Atan2(e.x, e.y) * Mathf.Rad2Deg) : 0f;
                    SetStick(cmd.x, cmd.y, 0f, yawStick);
                    break;
                }
                case "findHat":
                {
                    Vector2 hold = Hold(holdAt, c);
                    float yawStick = 0f;
                    if (FindHat(shot, out float camYaw, out float dist, out _))
                    {
                        yawStick = YawTowards(camYaw);
                        if (foundAt < 0f && Mathf.Abs(Mathf.DeltaAngle(PyUniSharingFast.BodyYawDegrees, camYaw)) < 4f)
                        {
                            foundAt = Time.time;
                            events.AppendLine($"{shot.name}\that_in_front\t{foundAt - shotT0:F2}\t{dist:F1} m");
                            Debug.Log($"[VideoShots] {shot.name}: hat in front at {foundAt - shotT0:F2} s ({dist:F1} m)");
                        }
                    }
                    else if (foundAt < 0f) yawStick = 0.35f;   // look round for it
                    SetStick(hold.x, hold.y, 0f, foundAt >= 0f ? yawStick * 0.5f : yawStick);
                    if (foundAt >= 0f && Time.time - foundAt > seg.after) yield break;
                    break;
                }
                case "patrol":
                {
                    if (!seen && seg.untilHat && FindHat(shot, out _, out float d0, out _))
                    {
                        seen = true;
                        stopAt = c;
                        events.AppendLine($"{shot.name}\that_seen\t{Time.time - shotT0:F2}\t{d0:F1} m");
                        Debug.Log($"[VideoShots] {shot.name}: hat seen at {Time.time - shotT0:F2} s ({d0:F1} m)");
                    }
                    if (seen)
                    {
                        // Fly closer if asked, then stop and turn the drone towards the hat. Turned by where the hat
                        // is, not by whether it is still in view: a walker who steps round a corner would otherwise
                        // leave the drone waiting for them to come back.
                        Vector2 hold = Hold(stopAt, c);
                        float yawStick = 0f;
                        SpecialWalker hw = HatWearer(shot);
                        if (hw != null)
                        {
                            Vector3 hp = hw.HatCentre;
                            Vector2 to = new Vector2(hp.x, hp.z) - c;
                            float bearing = Mathf.Atan2(to.x, to.y) * Mathf.Rad2Deg;
                            if (seg.approach > 0f && !close)
                            {
                                stopAt = c;
                                if (to.magnitude > seg.approach)
                                {
                                    hold = to.normalized * seg.speed * 0.7f;
                                    SetStick(hold.x, hold.y, 0f, YawTowards(bearing));
                                    break;
                                }
                                close = true;
                                hold = Vector2.zero;
                                events.AppendLine($"{shot.name}\that_close\t{Time.time - shotT0:F2}\t{to.magnitude:F1} m");
                            }
                            yawStick = 0.6f * YawTowards(bearing);
                            if (foundAt < 0f && Mathf.Abs(Mathf.DeltaAngle(PyUniSharingFast.BodyYawDegrees, bearing)) < 4f)
                            {
                                foundAt = Time.time;
                                bool inView = FindHat(shot, out _, out _, out _, central: false);
                                events.AppendLine($"{shot.name}\that_in_front\t{foundAt - shotT0:F2}\t{to.magnitude:F1} m\t{(inView ? "in view" : "hidden")}");
                                Debug.Log($"[VideoShots] {shot.name}: hat in front at {foundAt - shotT0:F2} s ({to.magnitude:F1} m, {(inView ? "in view" : "hidden")})");
                            }
                        }
                        SetStick(hold.x, hold.y, 0f, yawStick);
                        if (foundAt >= 0f && Time.time - foundAt > seg.after) yield break;
                        break;
                    }
                    if (nPath == 0) { SetStick(0f, 0f, 0f, 0f); break; }
                    Vector2 w = new Vector2(seg.path[2 * (wp % nPath)], seg.path[2 * (wp % nPath) + 1]);
                    Vector2 e = w - c;
                    if (e.magnitude < seg.arrive)
                    {
                        wp++;
                        events.AppendLine($"{shot.name}\twaypoint\t{wp}\t{Time.time - shotT0:F2}");
                        w = new Vector2(seg.path[2 * (wp % nPath)], seg.path[2 * (wp % nPath) + 1]);
                        e = w - c;
                    }
                    Vector2 cmd = e.normalized * seg.speed;
                    float yawCmd = seg.face ? YawTowards(Mathf.Atan2(e.x, e.y) * Mathf.Rad2Deg) : 0f;
                    SetStick(cmd.x, cmd.y, 0f, yawCmd);
                    break;
                }
                default:   // hold
                {
                    Vector2 hold = Hold(holdAt, c);
                    SetStick(hold.x, hold.y, 0f, 0f);
                    break;
                }
            }
            yield return null;
        }
        if ((seg.kind == "findHat" || (seg.kind == "patrol" && seg.untilHat)) && foundAt < 0f)
            events.AppendLine($"{shot.name}\that_not_found\t{Time.time - shotT0:F2}");
    }

    // ---------------------------------------------------------------------------------------------
    // ---------------------------------------------------------------------------------------------
    List<GameObject> Drones()
    {
        drones.Clear();
        if (spawner != null && spawner.swarm != null)
            foreach (var d in spawner.swarm)
                if (d != null && SwarmRegistry.TryGet(d, out var e) && e.droneParent != null &&
                    (e.velocityControl == null || e.velocityControl.State == null || e.velocityControl.State.IsAlive))
                    drones.Add(d);
        return drones;
    }

    // Whether a drone's feed is on a screen: a hull drone in a swarm, always for a lone drone (ScreenSpawn shows it
    // whatever the hull says, and the hull of one point is nothing).
    static bool Shown(SwarmRegistry.Entry e)
    {
        if (SwarmRegistry.TryGetLoneDrone(out _, out _)) return true;
        return e.attitude == null || e.attitude.BoundaryEstimate;
    }

    Vector2 Centroid()
    {
        Vector3 s = Vector3.zero;
        int n = 0;
        foreach (var d in Drones())
        {
            SwarmRegistry.TryGet(d, out var e);
            s += e.droneParent.position;
            n++;
        }
        return n > 0 ? new Vector2(s.x / n, s.z / n) : Vector2.zero;
    }

    Vector2 Hold(Vector2 at, Vector2 c)
    {
        float maxSpeed = 8f;
        foreach (var d in Drones())
            if (SwarmRegistry.TryGet(d, out var e) && e.velocityControl != null) { maxSpeed = e.velocityControl.maxSpeed; break; }
        Vector2 cmd = 0.25f * (at - c) / Mathf.Max(1f, maxSpeed);
        return cmd.magnitude > 1f ? cmd.normalized : cmd;
    }

    static float YawTowards(float yawDeg)
    {
        float err = Mathf.DeltaAngle(PyUniSharingFast.BodyYawDegrees, yawDeg);
        return Mathf.Clamp(err / 25f, -1f, 1f);
    }

    static void ZeroSticks()
    {
        var st = InputManager.Instance.InputStatus;
        st["roll"] = 0f; st["pitch"] = 0f; st["throttle"] = 0f; st["yaw"] = 0f; st["spread"] = -1f;
    }

    // World command (cx, cz) -> stick in the VR frame (the bench's SetStick, SwarmBenchRunner.cs).
    static void SetStick(float cx, float cz, float climb, float yaw)
    {
        float psi = PyUniSharingFast.BodyYawDegrees * Mathf.Deg2Rad;
        float cs = Mathf.Cos(psi), sn = Mathf.Sin(psi);
        var st = InputManager.Instance.InputStatus;
        st["roll"] = cx * cs - cz * sn;
        st["pitch"] = cx * sn + cz * cs;
        st["throttle"] = climb;
        st["yaw"] = yaw;
        st["spread"] = -1f;
    }

    void SetDial(float v)
    {
        dial = Mathf.Clamp(v, -1f, 1f);
        if (SwarmManager.Instance != null) SwarmManager.Instance.SetGimbalPitchNormalized(dial);
    }

    void EnsureCloseCamera(Shot shot)
    {
        if (closeCam == null)
        {
            var go = new GameObject("Video Closeup Camera");
            closeCam = go.AddComponent<Camera>();
            closeCam.enabled = false;
            closeCam.stereoTargetEye = StereoTargetEyeMask.None;
            closeCam.nearClipPlane = 0.05f;
            closeCam.farClipPlane = 1500f;
            int ui = LayerMask.NameToLayer("UI");
            closeCam.cullingMask = ui >= 0 ? ~(1 << ui) : ~0;
            closeRT = new RenderTexture(cfg.width, cfg.height, 24, RenderTextureFormat.ARGB32) { antiAliasing = 4 };
            closeRT.Create();
            closeCam.targetTexture = closeRT;
        }
        closeCam.fieldOfView = shot.closeupFov;
    }

    // In front of the named object (along its forward, which for the practice hat stations faces the spawn),
    // looking at head height.
    bool AimCloseCamera(Shot shot, string target)
    {
        GameObject go = GameObject.Find(target);
        if (go == null) return false;
        Vector3 p = go.transform.position;
        Vector3 f = go.transform.forward;
        f.y = 0f;
        f = f.sqrMagnitude > 1e-6f ? f.normalized : Vector3.forward;
        Vector3 look = p + Vector3.up * shot.closeupHeight;
        closeCam.transform.position = look + f * shot.closeupDist + Vector3.up * shot.closeupRise;
        closeCam.transform.rotation = Quaternion.LookRotation(look - closeCam.transform.position);
        return true;
    }

    string LogHeader()
    {
        var sb = new StringBuilder("frame,t,roll,pitch,throttle,yaw,dial,alt,bodyYaw,spread,camX,camZ,camYaw,camSize,arrowU,arrowV");
        int n = spawner != null && spawner.swarm != null ? spawner.swarm.Count : 0;
        for (int i = 0; i < n; i++) sb.Append($",d{i}x,d{i}z,d{i}edge,d{i}alive");
        return sb.ToString();
    }

    string LogRow(float shotT0)
    {
        var st = InputManager.Instance.InputStatus;
        float alt = 0f;
        int na = 0;
        foreach (var d in Drones()) { SwarmRegistry.TryGet(d, out var e); alt += e.droneParent.position.y; na++; }
        alt = na > 0 ? alt / na : 0f;
        float spread = SwarmPlaneController.Instance != null ? SwarmPlaneController.Instance.SpreadScale : 1f;
        Camera oc = rig.OverheadCamera;
        var sb = new StringBuilder();
        sb.Append($"{capture.Frames},{Time.time - shotT0:F3},{st["roll"]:F3},{st["pitch"]:F3},{st["throttle"]:F3},{st["yaw"]:F3},");
        sb.Append($"{dial:F3},{alt:F2},{PyUniSharingFast.BodyYawDegrees:F2},{spread:F3},");
        if (oc != null)
            sb.Append($"{oc.transform.position.x:F2},{oc.transform.position.z:F2},{oc.transform.eulerAngles.y:F2},{oc.orthographicSize:F2}");
        else
            sb.Append(",,,");
        // Where the forward arrow (BodyYawCompass, on the floor in front of the seat) is in the interface camera.
        if (compass == null && Time.time >= nextCompassScan)
        {
            nextCompassScan = Time.time + 2f;
            // BodyYawCompass draws its arrow on a separate object it creates, not a child.
            var arrow = GameObject.Find("BodyYawCompassArrow");
            if (arrow != null) compass = arrow.GetComponent<Renderer>();
        }
        Camera ic = rig.InterfaceCamera;
        if (compass != null && ic != null)
        {
            Vector3 vp = ic.WorldToViewportPoint(compass.bounds.center);
            sb.Append(vp.z > 0f ? $",{vp.x:F4},{vp.y:F4}" : ",,");
        }
        else
            sb.Append(",,");
        if (spawner != null && spawner.swarm != null)
            foreach (var d in spawner.swarm)
            {
                bool ok = d != null && SwarmRegistry.TryGet(d, out var e) && e.droneParent != null;
                if (!ok) { sb.Append(",,,0,0"); continue; }
                SwarmRegistry.TryGet(d, out var en);
                Vector3 p = en.droneParent.position;
                bool alive = en.velocityControl == null || en.velocityControl.State == null || en.velocityControl.State.IsAlive;
                bool edge = Shown(en);
                sb.Append($",{p.x:F2},{p.z:F2},{(edge ? 1 : 0)},{(alive ? 1 : 0)}");
            }
        return sb.ToString();
    }

    // The feed on the screen straight ahead: the edge drone whose camera heading is nearest the body yaw.
    Camera FrontCamera()
    {
        Camera best = null;
        float bestErr = float.MaxValue;
        foreach (var d in Drones())
        {
            if (!SwarmRegistry.TryGet(d, out var e) || !Shown(e)) continue;
            Transform fpv = d.transform.Find("FPV");
            Camera cam = fpv != null ? fpv.GetComponent<Camera>() : null;
            if (cam == null || cam.targetTexture == null) continue;
            float err = Mathf.Abs(Mathf.DeltaAngle(cam.transform.eulerAngles.y, PyUniSharingFast.BodyYawDegrees));
            if (err < bestErr) { bestErr = err; best = cam; }
        }
        return best;
    }

    SpecialWalker HatWearer(Shot shot)
    {
        Vector2 goal = shot.goal != null && shot.goal.Length >= 2 ? new Vector2(shot.goal[0], shot.goal[1]) : Centroid();
        SpecialWalker hat = null;
        float best = float.MaxValue;
        foreach (var w in FindObjectsOfType<SpecialWalker>())
        {
            Vector3 p = w.HatCentre;
            float d = Vector2.Distance(new Vector2(p.x, p.z), goal);
            if (d < best) { best = d; hat = w; }
        }
        return best < 150f ? hat : null;
    }

    void Trace(Shot shot, Seg seg, Vector2 c, float shotT0)
    {
        int alive = 0, edge = 0, ready = 0;
        foreach (var d in Drones())
        {
            alive++;
            if (SwarmRegistry.TryGet(d, out var e) && e.attitude != null)
            {
                if (e.attitude.BoundaryEstimate) edge++;
                if (e.attitude.BoundaryFeedReady) ready++;
            }
        }
        string hat = "none";
        Vector2 goal = shot.goal != null && shot.goal.Length >= 2 ? new Vector2(shot.goal[0], shot.goal[1]) : c;
        float best = float.MaxValue;
        int n = 0;
        foreach (var w in FindObjectsOfType<SpecialWalker>())
        {
            n++;
            Vector3 p = w.HatCentre;
            float dd = Vector2.Distance(new Vector2(p.x, p.z), goal);
            if (dd < best)
            {
                best = dd;
                hat = $"({p.x:F0},{p.y:F1},{p.z:F0}) {dd:F0} m from goal, {Vector2.Distance(new Vector2(p.x, p.z), c):F0} m from swarm";
            }
        }
        Debug.Log($"[VideoShots] t={Time.time - shotT0:F1} {seg.kind} centroid=({c.x:F0},{c.y:F0}) bodyYaw={PyUniSharingFast.BodyYawDegrees:F0} " +
                  $"alive={alive} edge={edge} feedReady={ready} walkers={n} hat={hat}");
    }

    // The goal's hat wearer, if a shown feed has it in its central 80%, within hatRange, with a clear line:
    // the recorder's AutoScore rule. camYaw is that camera's heading, the body yaw that puts its screen in front;
    // bearing is the hat's direction from that camera, the heading that puts it in the middle of the picture.
    // central: false accepts anywhere in the picture (to keep turning towards a hat already found).
    bool FindHat(Shot shot, out float camYaw, out float dist, out float bearing, bool central = true)
    {
        camYaw = 0f;
        dist = 0f;
        bearing = 0f;
        SpecialWalker hat = null;
        float best = float.MaxValue;
        Vector2 goal = shot.goal != null && shot.goal.Length >= 2 ? new Vector2(shot.goal[0], shot.goal[1]) : Centroid();
        foreach (var w in FindObjectsOfType<SpecialWalker>())
        {
            Vector3 p = w.HatCentre;
            float d = Vector2.Distance(new Vector2(p.x, p.z), goal);
            if (d < best) { best = d; hat = w; }
        }
        if (hat == null || best > 150f) return false;
        Vector3 h = hat.HatCentre;
        int obstacles = LayerMask.GetMask("Obstacle");
        float bestOff = float.MaxValue;
        bool found = false;
        foreach (var d in Drones())
        {
            if (!SwarmRegistry.TryGet(d, out var e) || !Shown(e)) continue;
            Transform fpv = d.transform.Find("FPV");
            Camera cam = fpv != null ? fpv.GetComponent<Camera>() : null;
            if (cam == null) continue;
            Vector3 vp = cam.WorldToViewportPoint(h);
            float range = Vector3.Distance(cam.transform.position, h);
            float m = central ? 0.1f : 0f;
            if (vp.z <= 0f || vp.x < m || vp.x > 1f - m || vp.y < m || vp.y > 1f - m || range > shot.hatRange) continue;
            if (Physics.Linecast(cam.transform.position, h, obstacles)) continue;
            float off = Mathf.Abs(vp.x - 0.5f);
            if (off < bestOff)
            {
                bestOff = off;
                camYaw = cam.transform.eulerAngles.y;
                dist = range;
                Vector3 to = h - cam.transform.position;
                bearing = Mathf.Atan2(to.x, to.z) * Mathf.Rad2Deg;
                found = true;
            }
        }
        return found;
    }
}

/// <summary>
/// The end of each frame for VideoShotRunner: renders the capture rig's cameras (batchmode renders no camera
/// by itself, and WaitForEndOfFrame never resumes there) and pipes their pixels to ffmpeg. Runs after the
/// rig's LateUpdate, so the frustums it queued with Graphics.DrawMesh are in the picture.
/// </summary>
[DefaultExecutionOrder(5000)]
public class VideoShotCapture : MonoBehaviour
{
    public class Writer
    {
        public Camera cam;
        public RenderTexture rt;
        public System.Diagnostics.Process proc;
        public Stream stream;
    }

    List<Writer> writers;
    Texture2D tex;
    byte[] buf;
    int frames;

    // The front screen's feed, scaled to the output size, and the hat's place in it each frame.
    Writer front;
    StreamWriter hatCsv;
    float frontT0;
    System.Func<SpecialWalker> hatOf;
    System.Func<Camera> frontOf;
    RenderTexture frontRT;

    public void BeginFront(Writer w, string csv, float t0, System.Func<SpecialWalker> hat, System.Func<Camera> cam)
    {
        front = w;
        frontT0 = t0;
        hatOf = hat;
        frontOf = cam;
        hatCsv = new StreamWriter(csv);
        hatCsv.WriteLine("frame,t,drone,u,v,depth,range,clear");
        if (frontRT == null) frontRT = new RenderTexture(1920, 1080, 0, RenderTextureFormat.ARGB32);
    }

    public static Writer Open(VideoShotRunner.Config cfg, Camera cam, RenderTexture rt, string file)
    {
        string outFile = Path.Combine(cfg.outDir, file);
        int w = rt != null ? rt.width : cfg.width, h = rt != null ? rt.height : cfg.height;
        var psi = new System.Diagnostics.ProcessStartInfo
        {
            FileName = cfg.ffmpeg,
            Arguments = $"-y -loglevel error -f rawvideo -pix_fmt rgba -s {w}x{h} -r {cfg.fps} -i - " +
                        $"-vf vflip -c:v libx264 -preset veryfast -crf 18 -pix_fmt yuv420p \"{outFile}\"",
            UseShellExecute = false,
            RedirectStandardInput = true,
            CreateNoWindow = true,
        };
        var p = System.Diagnostics.Process.Start(psi);
        return new Writer { cam = cam, rt = rt, proc = p, stream = p.StandardInput.BaseStream };
    }

    public int Frames => frames;
    StreamWriter log;
    System.Func<string> logRow;

    public void Begin(List<Writer> w)
    {
        writers = w;
        frames = 0;
    }

    public void BeginLog(string path, string header, System.Func<string> row)
    {
        log = new StreamWriter(path);
        log.WriteLine(header);
        logRow = row;
    }

    public int End()
    {
        if (log != null) { log.Close(); log = null; logRow = null; }
        if (front != null)
        {
            front.stream.Close();
            if (!front.proc.WaitForExit(300000)) front.proc.Kill();
            front = null;
            hatCsv.Close();
            hatCsv = null;
        }
        if (writers != null)
            foreach (var w in writers)
            {
                w.stream.Close();
                if (!w.proc.WaitForExit(300000)) w.proc.Kill();
            }
        writers = null;
        return frames;
    }

    void LateUpdate()
    {
        if (writers == null || writers.Count == 0) return;
        if (log != null) log.WriteLine(logRow());
        if (front != null) WriteFront();
        foreach (var w in writers)
        {
            w.cam.Render();
            if (tex == null || tex.width != w.rt.width || tex.height != w.rt.height)
            {
                tex = new Texture2D(w.rt.width, w.rt.height, TextureFormat.RGBA32, false);
                buf = new byte[w.rt.width * w.rt.height * 4];
            }
            var tmp = RenderTexture.GetTemporary(w.rt.width, w.rt.height, 0, RenderTextureFormat.ARGB32);
            Graphics.Blit(w.rt, tmp);   // resolves the MSAA target
            var prev = RenderTexture.active;
            RenderTexture.active = tmp;
            tex.ReadPixels(new Rect(0, 0, tmp.width, tmp.height), 0, 0, false);
            RenderTexture.active = prev;
            RenderTexture.ReleaseTemporary(tmp);
            tex.GetRawTextureData<byte>().CopyTo(buf);
            w.stream.Write(buf, 0, buf.Length);
        }
        frames++;
    }

    void WriteFront()
    {
        Camera cam = frontOf();
        string name = "";
        if (cam != null)
        {
            var fpv = cam.GetComponent<FPVCameraScript>();
            if (fpv != null) fpv.EnsureRenderedThisFrame();
            else cam.Render();
            Graphics.Blit(cam.targetTexture, frontRT);
            name = cam.transform.parent != null ? cam.transform.parent.name : cam.name;
        }
        Read(frontRT, front.stream);
        string row = $"{frames},{Time.time - frontT0:F3},{name},,,,,";
        SpecialWalker hat = hatOf();
        if (cam != null && hat != null)
        {
            Vector3 h = hat.HatCentre;
            Vector3 vp = cam.WorldToViewportPoint(h);
            float range = Vector3.Distance(cam.transform.position, h);
            bool clear = !Physics.Linecast(cam.transform.position, h, LayerMask.GetMask("Obstacle"));
            row = $"{frames},{Time.time - frontT0:F3},{name},{vp.x:F4},{vp.y:F4},{vp.z:F2},{range:F1},{(clear ? 1 : 0)}";
        }
        hatCsv.WriteLine(row);
    }

    void Read(RenderTexture rt, Stream stream)
    {
        if (tex == null || tex.width != rt.width || tex.height != rt.height)
        {
            tex = new Texture2D(rt.width, rt.height, TextureFormat.RGBA32, false);
            buf = new byte[rt.width * rt.height * 4];
        }
        var prev = RenderTexture.active;
        RenderTexture.active = rt;
        tex.ReadPixels(new Rect(0, 0, rt.width, rt.height), 0, 0, false);
        RenderTexture.active = prev;
        tex.GetRawTextureData<byte>().CopyTo(buf);
        stream.Write(buf, 0, buf.Length);
    }
}
