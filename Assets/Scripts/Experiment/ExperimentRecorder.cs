using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Text;
using UnityEngine;

/// <summary>
/// Captures all per-run data for the city search task (find three goal patches, identify the
/// modified "special" walker in each) so performance can be analysed offline.
///
/// Lives on the scene's <c>gameManager</c> object. Session metadata (participant / trial / condition)
/// is set in the Inspector before pressing Play; the recorder then samples continuous state at a
/// fixed rate and logs an "identify" event for each answer.
///
/// <para><b>Identify, AutoScore (<see cref="identifyMode"/>, the default since 2026-10-12).</b> The pilot presses
/// the RC's C2 button (<c>rcjoy bridge --profile sim</c>, <see cref="UDPReceiverManager.MarkPresses"/>) when they
/// believe the hat is on screen, and the press is scored on the spot. Its goal is the one whose zone — the patch
/// grown by <see cref="identifyZoneM"/>, analyse.py's 15 m zone — contains the ALIVE drones' centroid (not merely
/// the nearest goal); it is <c>correct</c> when that goal's hat is inside the central
/// <see cref="identifyCentralFraction"/> of some shown feed's image, within <see cref="identifyRangeM"/> of its
/// camera, with no building in the way; otherwise a <c>false_alarm</c> (logged, the goal stays open) or, with no
/// zone round the centroid, <c>out_of_zone</c>. Every press is logged with the scoring camera's distance, its feed
/// and the feed the pilot's gaze was on. The experimenter's keys override the latest press's score (an
/// <c>outcome_override</c> event) and never create an identify of their own. The session finishes once every goal
/// is resolved — identified correctly, or overridden — or on the end key. The verbal hat report is a secondary
/// check only: with three hats it is right by chance a third of the time.</para>
///
/// <para><b>Legacy</b> is the protocol before that: a press marks the moment of an answer and the experimenter's
/// key (correct / incorrect / skip) gives its outcome; a key with no press waiting is the identify itself; an
/// identify answers the nearest unanswered goal; the session finishes once every goal has an outcome.</para>
///
/// <para><b>The task clock starts at the <see cref="StartLine"/>, not at Play.</b> Every file's <c>t</c> stays on the
/// session clock (t = 0 at Play; crash matching against the editor log needs that), and the first live drone's
/// crossing is logged as a <c>start_line</c> event and <c>startLineSec</c>, with the alive centroid's crossing beside
/// it (<c>start_line_centroid</c>, <c>startCentroidSec</c>). <c>totalTaskTime</c> keeps its old meaning, Play → last
/// identify; <c>taskTimeFromStartLine</c> is the timed task. analyse.py times from the line when there is one.</para>
///
/// Condition is a recorded label only: this component does not change the drone count — the
/// experimenter still configures <see cref="swarmSpawn"/> as usual.
///
/// Output (one set per run) goes to <c>Application.persistentDataPath/experiment/</c> as ';'-delimited
/// CSVs (long format, so a variable drone count needs no schema change) plus a session JSON summary.
/// Reuses the StreamWriter/CSV/Unix-ms convention established in VelocityControl.
/// </summary>
public class ExperimentRecorder : MonoBehaviour
{
    public enum Condition { Swarm, SingleDrone }

    [Header("Session (set before Play)")]
    [Tooltip("Participant code. Convention is a 4-letter code; trimmed/upper-cased for the filename.")]
    public string participantId = "AAAA";
    public int trialNumber = 1;
    [Tooltip("Recorded label only — does NOT change the drone count. Configure swarmSpawn separately.")]
    public Condition condition = Condition.Swarm;

    [Header("Sampling")]
    [Tooltip("Continuous-state sample rate (Hz) for the drones / head / walkers CSVs.")]
    public float sampleHz = 10f;

    [Header("Identify keys (experimenter)")]
    public KeyCode correctKey = KeyCode.Alpha1;
    public KeyCode incorrectKey = KeyCode.Alpha2;
    public KeyCode skipKey = KeyCode.Alpha3;
    [Tooltip("Manually finish/abort the session (also fires automatically after all goals answered).")]
    public KeyCode endSessionKey = KeyCode.Backspace;

    [Header("Identify from the pilot's RC")]
    [Tooltip("The RC's C2 button (rcjoy bridge --profile sim) marks the moment of an identify; the keys above " +
             "then give its outcome. Off: presses are ignored and the keys alone identify, as before.")]
    public bool rcIdentify = true;
    [Tooltip("An RC press this soon after the previous RC identify is a double tap, logged and ignored.")]
    public float rcIdentifyDebounceSec = 1f;

    public enum IdentifyMode { AutoScore, Legacy }

    [Header("Identify scoring")]
    [Tooltip("AutoScore: an RC press is scored against what the shown feeds show (see the class summary) and the " +
             "keys only override it. Legacy: a press or key answers the nearest unanswered goal and the key gives " +
             "the outcome.")]
    public IdentifyMode identifyMode = IdentifyMode.AutoScore;
    [Tooltip("A press is for a goal when the alive drones' centroid is within this many metres of its patch " +
             "(analyse.py's zone).")]
    public float identifyZoneM = 15f;
    [Tooltip("The hat must be within this of the camera whose feed shows it (analyse.py --range).")]
    public float identifyRangeM = 40f;
    [Tooltip("...and inside this central share of that feed's image, horizontally and vertically.")]
    [Range(0.1f, 1f)]
    public float identifyCentralFraction = 0.8f;

    /// <summary>The hat's height above the walker's origin (analyse.py's HAT_HEIGHT).</summary>
    private const float HatAboveWalker = 1.6f;

    // ---- runtime references (resolved lazily on the first Update, once all spawns have run) ----
    private swarmSpawn spawner;
    private ScreenSpawn screenSpawn;   // which drones' feeds are on screen, and where the quads hang
    private StartLine startLine;       // null in a scene without one (or with it removed)
    private bool startLineLogged, startCentroidLogged;
    private Transform headTransform;   // OVRCameraRig.centerEyeAnchor (fallback Camera.main)
    private int obstacleMask;

    private class GoalInfo
    {
        public Transform goal;
        public Transform special;          // the SpecialWalker's transform (may be null)
        public Vector3 specialSpawnPos;    // captured at resolve time
        public string hat = "";            // which hat this goal's special walker wears
        // The seeded choices that put the hat there (GoalSpecialWalker), so a replay can be checked against them.
        public int dealOrdinal = -1;
        public int targetPatrolIndex = -1;
        public string targetPatrol = "";
        public int targetWalkerIndex = -1;
        // The patch's shape and geometry (IGoalZone on the goal root; a tile goal has none).
        public string zoneShape = "tile";
        public float patchRadius = GoalZones.TileHalf;
        public float ringRadius = -1f, occluderWidth = -1f, clearance = -1f;
        public bool answered;
        // Filled in when the goal is answered:
        public string source = "";         // rc (the pilot's C2) / key (the experimenter's key)
        public string outcome = "";        // correct / incorrect / skip; "" while an RC identify awaits its key
        public float decisionTimeSec = -1f;
        public float swarmToWalkerDist = -1f;
        public Vector3 centroidAtAnswer;
        // AutoScore: presses scored wrong at this goal, and the scoring camera of the correct one.
        public int falseAlarms;
        public int scoredFeed = -1;
        public float scoredCamDist = -1f;
        public int gazeFeedAtAnswer = -1;
    }
    private readonly List<GoalInfo> goals = new List<GoalInfo>();
    private bool referencesResolved = false;

    // ---- output ----
    private StreamWriter dronesWriter, headWriter, walkersWriter, eventsWriter, shapeWriter, stitchWriter, perfWriter;
    private string dirPath, fileStem;

    // ---- frame timing (see WritePerfSample) ----
    // Accumulated every frame and written, then reset, at each sample: frames rendered and the
    // longest frame since the previous row, and the FPV render count at the previous row.
    private int perfFrames;
    private float perfDtSum, perfDtMax;
    private long perfLastRenders = -1;

    // ---- stitching (see UpdateStitchState) ----
    // A panorama on screen with no new frame for this long is frozen, not stitching. Python writes
    // at ~30 Hz and exits after 5 s without a Unity heartbeat, so 1 s is well clear of both.
    private const float PanoramaStaleSec = 1f;
    private bool stitchFunctional = false;
    private float stitchFunctionalSec = 0f;
    private int stitchOnCount = 0;

    // ---- timing / state ----
    private float sessionStartTime;
    private DateTime wallStart;
    private float nextSampleTime;
    private bool sessionOpen = false;   // files open, sampling active
    private bool finalized = false;     // summary written, files closed

    // ---- RC identify (see PollRcIdentify) ----
    private int rcPressesSeen;
    private float lastRcIdentifyTime = float.NegativeInfinity;
    // AutoScore: the latest scored press, which the experimenter's key overrides. -2 none yet, -1 out of zone.
    private int lastPressGoal = -2;
    private float lastPressTime = -1f;
    private Vector3 lastPressCentroid;

    private static readonly CultureInfo Inv = CultureInfo.InvariantCulture;

    void Start()
    {
        try
        {
            OpenSession();
        }
        catch (Exception e)
        {
            Debug.LogError($"ExperimentRecorder: failed to open session — logging disabled. {e}", this);
            CloseWriters();
            sessionOpen = false;
        }
    }

    void Update()
    {
        if (!sessionOpen || finalized) return;

        // Goals and special walkers are spawned during the first frame's Start/Awake pass, so resolve
        // them here (guaranteed after those have run) rather than in our own Start.
        if (!referencesResolved)
        {
            ResolveGoalsAndWalkers();
            startLine = FindObjectOfType<StartLine>();
            referencesResolved = true;
        }

        // Every frame, not at sampleHz, so transitions are timed to the frame.
        UpdateStitchState();
        LogStartLine();

        float dt = Time.unscaledDeltaTime;
        perfFrames++;
        perfDtSum += dt;
        if (dt > perfDtMax) perfDtMax = dt;

        // Fixed-rate continuous sampling (nextSampleTime pattern mirrors PyUniSharingFast.Update).
        if (Time.time >= nextSampleTime)
        {
            float t = Time.time - sessionStartTime;
            long ms = UnixMs();
            WriteDroneSample(t, ms);
            WriteHeadSample(t, ms);
            WriteWalkerSample(t, ms);
            WriteShapeSample(t, ms);
            WriteStitchSample(t, ms);
            WritePerfSample(t, ms);

            float interval = sampleHz > 0f ? 1f / sampleHz : 0.1f;
            // Advance from the scheduled time; if we fell behind, resync to now to avoid a burst.
            nextSampleTime += interval;
            if (nextSampleTime < Time.time) nextSampleTime = Time.time + interval;
        }

        // Identify actions: the pilot's RC press, then the experimenter's key (edge-detected).
        PollRcIdentify();
        if (Input.GetKeyDown(correctKey)) RecordOutcome("correct");
        else if (Input.GetKeyDown(incorrectKey)) RecordOutcome("incorrect");
        else if (Input.GetKeyDown(skipKey)) RecordOutcome("skip");

        if (Input.GetKeyDown(endSessionKey)) Finalize("manual_end");
    }

    void OnApplicationQuit() => Finalize("app_quit");
    void OnDestroy() => Finalize("destroyed");

    // ------------------------------------------------------------------ setup

    private void OpenSession()
    {
        spawner = FindObjectOfType<swarmSpawn>();
        screenSpawn = FindObjectOfType<ScreenSpawn>();
        headTransform = ResolveHeadTransform();
        obstacleMask = LayerMask.GetMask("Obstacle");

        // EXPERIMENT_RECORDER_DIR redirects a headless check (ProceduralCityProbe) away from the experiment folder.
        string redirect = Environment.GetEnvironmentVariable("EXPERIMENT_RECORDER_DIR");
        dirPath = string.IsNullOrEmpty(redirect) ? Path.Combine(Application.persistentDataPath, "experiment") : redirect;
        Directory.CreateDirectory(dirPath);

        string pid = string.IsNullOrWhiteSpace(participantId) ? "XXXX" : participantId.Trim().ToUpperInvariant();
        wallStart = DateTime.Now;
        fileStem = $"{pid}_t{trialNumber}_{condition}_{wallStart:yyyyMMdd_HHmmss}";

        dronesWriter = NewWriter("drones", "t;unixMs;droneId;gtX;gtY;gtZ;yawDeg;alive;hull;shown");
        headWriter = NewWriter("head",
            "t;unixMs;headX;headY;headZ;headYaw;headPitch;headRoll;bodyYaw;inThrottle;inYaw;inPitch;inRoll;inSpread;" +
            "gimbalPitch;gazeX;gazeY;gazeZ;gazeConf;gazeFeed");
        walkersWriter = NewWriter("walkers", "t;unixMs;goalIndex;specialX;specialY;specialZ;hat");
        eventsWriter = NewWriter("events", "t;unixMs;eventType;goalIndex;outcome;swarmToWalkerDist;note");
        shapeWriter = NewWriter("shape",
            "t;unixMs;nAlive;hullVerts;interior;maxGapDeg;meanNNm;ringRadiusM;coreRadiusM;dRef;r0Eff;hollowCore;" +
            "lookGapDeg;lookGapRawDeg;lookGapFill");
        stitchWriter = NewWriter("stitch", "t;unixMs;stitcher;shown;qualityOk;pilotOn;reasonBits;panoAgeSec;functional");
        perfWriter = NewWriter("perf", "t;unixMs;frames;fps;dtMaxMs;fpvRenders");
        WriteSwarmSettings();

        sessionStartTime = Time.time;
        nextSampleTime = Time.time;
        rcPressesSeen = UDPReceiverManager.MarkPresses;   // a press before the session is not an answer
        sessionOpen = true;

        WriteEvent("session_start", -1, "", -1f, $"pid={pid};trial={trialNumber};condition={condition};identify={identifyMode}");
        Debug.Log($"ExperimentRecorder: logging to {dirPath} ({fileStem}_*.csv)", this);
    }

    private StreamWriter NewWriter(string suffix, string header)
    {
        string path = Path.Combine(dirPath, $"{fileStem}_{suffix}.csv");
        var w = new StreamWriter(path, false, new UTF8Encoding(false));
        w.WriteLine(header);
        return w;
    }

    /// <summary>
    /// The swarm's settings as this run started, as <c>&lt;stem&gt;_swarm.json</c>: the scene and every
    /// SwarmManager field, as JsonUtility serialises the component. The CSVs record only the live d_ref and
    /// the core flag, and a scene's gains change between tests (ScaledCityWorld's hollow core was on for
    /// internal_2 and is off now), so without this an offline replay has to be told by hand what a run
    /// flew. Read by <c>Assets/Scripts/swarm/Bench/swarm_params.py</c>. Its d_ref is the starting value;
    /// the spread stick moves it, and shape.csv has it per sample.
    /// </summary>
    private void WriteSwarmSettings()
    {
        if (SwarmManager.Instance == null) return;
        try
        {
            string scene = UnityEngine.SceneManagement.SceneManager.GetActiveScene().name;
            string json = "{\n\"scene\": \"" + scene + "\",\n\"swarmManager\": "
                        + JsonUtility.ToJson(SwarmManager.Instance, true) + "\n}\n";
            File.WriteAllText(Path.Combine(dirPath, $"{fileStem}_swarm.json"), json, new UTF8Encoding(false));
        }
        catch (Exception e)
        {
            // The run's own data matters more than this summary of its settings.
            Debug.LogWarning($"ExperimentRecorder: could not write {fileStem}_swarm.json. {e.Message}", this);
        }
    }

    private Transform ResolveHeadTransform()
    {
        OVRCameraRig rig = FindObjectOfType<OVRCameraRig>();
        if (rig != null && rig.centerEyeAnchor != null) return rig.centerEyeAnchor;
        return Camera.main != null ? Camera.main.transform : null;
    }

    private void ResolveGoalsAndWalkers()
    {
        goals.Clear();

        // Prefer the registries, which keep the goals' order: GoalPatchReplacer's (tile cities), then the procedural
        // city's (layout order). Only with neither does it fall back to scanning for special walkers, whose order
        // FindObjectsByType does not define.
        List<GameObject> goalObjects = new List<GameObject>();
        GoalPatchReplacer replacer = FindObjectOfType<GoalPatchReplacer>();
        ProceduralCityBuilder procedural = FindObjectOfType<ProceduralCityBuilder>();
        if (replacer != null && replacer.PlacedGoals != null && replacer.PlacedGoals.Count > 0)
        {
            foreach (GameObject g in replacer.PlacedGoals)
                if (g != null) goalObjects.Add(g);
        }
        else if (procedural != null && procedural.PlacedGoals.Count > 0)
        {
            foreach (GameObject g in procedural.PlacedGoals)
                if (g != null) goalObjects.Add(g);
        }
        else
        {
            foreach (SpecialWalker sw in FindObjectsByType<SpecialWalker>(FindObjectsSortMode.None))
            {
                GoalSpecialWalker owner = sw.GetComponentInParent<GoalSpecialWalker>();
                if (owner != null && !goalObjects.Contains(owner.gameObject))
                    goalObjects.Add(owner.gameObject);
            }
        }

        SpecialWalker[] specials = FindObjectsByType<SpecialWalker>(FindObjectsSortMode.None);
        foreach (GameObject g in goalObjects)
        {
            GoalSpecialWalker gsw = g.GetComponent<GoalSpecialWalker>();
            Transform special = null;
            string hat = "";
            foreach (SpecialWalker sw in specials)
            {
                if (gsw != null && sw.GetComponentInParent<GoalSpecialWalker>() == gsw)
                {
                    special = sw.transform;
                    hat = sw.HatId;
                    break;
                }
            }
            goals.Add(new GoalInfo
            {
                goal = g.transform,
                special = special,
                specialSpawnPos = special != null ? special.position : Vector3.zero,
                hat = hat,
                dealOrdinal = gsw != null ? gsw.DealOrdinal : -1,
                targetPatrolIndex = gsw != null ? gsw.TargetPatrolIndex : -1,
                targetPatrol = gsw != null ? gsw.TargetPatrolName : "",
                targetWalkerIndex = gsw != null ? gsw.TargetWalkerIndex : -1,
            });
            IGoalZone zone = g.GetComponent<IGoalZone>();
            if (zone != null)
            {
                GoalInfo info = goals[goals.Count - 1];
                info.zoneShape = zone.ZoneShape;
                info.patchRadius = zone.PatchRadius;
                info.ringRadius = zone.RingRadius;
                info.occluderWidth = zone.OccluderWidth;
                info.clearance = zone.Clearance;
            }
        }

        if (goals.Count == 0)
            Debug.LogWarning("ExperimentRecorder: no goal patches found; identify events will be unassigned.", this);
    }

    // ------------------------------------------------------------------ sampling

    private void WriteDroneSample(float t, long ms)
    {
        List<GameObject> swarm = spawner != null ? spawner.swarm : null;
        if (swarm == null) return;

        for (int i = 0; i < swarm.Count; i++)
        {
            GameObject drone = swarm[i];
            if (drone == null) continue;
            Transform dp = drone.transform.Find("DroneParent");
            if (dp == null) continue;

            Vector3 pos = dp.position;
            bool alive = true;
            VelocityControl vc = dp.GetComponent<VelocityControl>();
            if (vc != null && vc.State != null)
            {
                pos = vc.State.GroundTruthPosition;
                alive = vc.State.IsAlive;
            }
            float yaw = dp.eulerAngles.y;
            // hull: AttitudeAlgorithm's debounced boundary flag; shown: the drone's feed quad is on screen (interior
            // drones, a feed held back while it turns, and the stitched triplet under the panorama all are not).
            AttitudeAlgorithm att = dp.GetComponent<AttitudeAlgorithm>();
            bool hull = att != null && att.BoundaryEstimate;
            bool shown = FeedShown(drone);
            dronesWriter.WriteLine(
                $"{F(t)};{ms};{i};{F(pos.x)};{F(pos.y)};{F(pos.z)};{F(yaw)};{(alive ? 1 : 0)};{(hull ? 1 : 0)};{(shown ? 1 : 0)}");
        }
    }

    /// <summary>
    /// Swarm-shape read-outs: how many drones are on the hull (and therefore on screen), how wide the
    /// pilot's largest unobserved sector is, and how far the formation is from the spacing that was
    /// actually commanded. This is what the hollow-core feature is judged on, so it is logged whether
    /// or not the feature is enabled — the disabled runs are the baseline the enabled ones are
    /// compared against.
    ///
    /// The look-gap columns do the same for the look-direction gap fill: how far the pilot's body yaw
    /// is from the nearest heading a hull drone is driven to, before (<c>lookGapRawDeg</c>) and after
    /// (<c>lookGapDeg</c>) the fill, and whether it was acting (<c>lookGapFill</c>). Appended at the end so readers that
    /// index columns by name are unaffected; NaN while there is no body yaw or no usable hull.
    /// </summary>
    private void WriteShapeSample(float t, long ms)
    {
        List<GameObject> swarm = spawner != null ? spawner.swarm : null;
        if (swarm == null) return;

        // Recomputed at most once per physics tick and shared with the drones' own hull pass, so
        // asking for it here costs nothing beyond the first caller in the tick.
        AttitudeAlgorithm.EnsureSharedGlobalHull(swarm);

        SwarmManager manager = SwarmManager.Instance;
        SwarmPlaneController plane = SwarmPlaneController.Instance;

        float dRef = manager != null ? manager.GetDRef() : 0f;
        bool hollow = manager != null && manager.GetHollowSwarmCore();
        float r0Eff = manager != null ? manager.GetEffectiveR0Coh() : 0f;
        float coreR = plane != null ? plane.CoreRadiusMetres : 0f;

        shapeWriter.WriteLine(
            $"{F(t)};{ms};{AttitudeAlgorithm.SharedAliveCount};{AttitudeAlgorithm.SharedHullVertexCount};" +
            $"{AttitudeAlgorithm.SharedInteriorCount};{F(AttitudeAlgorithm.SharedMaxGapDeg)};" +
            $"{F(AttitudeAlgorithm.SharedMeanNearestNeighbourM)};{F(AttitudeAlgorithm.SharedRingRadiusM)};" +
            $"{F(coreR)};{F(dRef)};{F(r0Eff)};{(hollow ? 1 : 0)};" +
            $"{F(AttitudeAlgorithm.SharedLookGapDeg)};{F(AttitudeAlgorithm.SharedLookGapRawDeg)};" +
            $"{(AttitudeAlgorithm.SharedLookGapFillActive ? 1 : 0)}");
    }

    /// <summary>
    /// Head pose, body yaw and the pilot's inputs. <c>gimbalPitch</c> is the swarm-wide camera pitch
    /// the joystick dial commands (<see cref="FPVCameraScript.SharedPitch"/>, degrees, 0 = level,
    /// -90 = straight down); the dial itself is not in <c>InputStatus</c>, and the pitch it sets is a
    /// linear map of it. Appended last, so runs recorded before it simply lack the column.
    /// </summary>
    private void WriteHeadSample(float t, long ms)
    {
        Vector3 hp = Vector3.zero, he = Vector3.zero;
        if (headTransform != null)
        {
            hp = headTransform.position;
            he = headTransform.eulerAngles; // x=pitch, y=yaw, z=roll
        }
        float bodyYaw = PyUniSharingFast.BodyYawDegrees;

        float thr = 0f, yaw = 0f, pit = 0f, rol = 0f, spr = 0f;
        var im = InputManager.Instance;
        if (im != null && im.InputStatus != null)
        {
            var s = im.InputStatus;
            s.TryGetValue("throttle", out thr);
            s.TryGetValue("yaw", out yaw);
            s.TryGetValue("pitch", out pit);
            s.TryGetValue("roll", out rol);
            s.TryGetValue("spread", out spr);
        }

        // Gaze: the combined ray's world direction and confidence, and the feed it lands on (-1 none / no tracking).
        EyeGazeTracker.GazeSample gaze = EyeGazeTracker.Combined;
        bool gazeOk = EyeGazeTracker.EyeTrackingActive && gaze.IsValid;
        Vector3 gd = gazeOk ? gaze.Direction : new Vector3(float.NaN, float.NaN, float.NaN);
        int gazeFeed = gazeOk ? GazeFeed(gaze) : -1;

        headWriter.WriteLine(
            $"{F(t)};{ms};{F(hp.x)};{F(hp.y)};{F(hp.z)};{F(he.y)};{F(he.x)};{F(he.z)};{F(bodyYaw)};" +
            $"{F(thr)};{F(yaw)};{F(pit)};{F(rol)};{F(spr)};{F(FPVCameraScript.SharedPitch)};" +
            $"{F(gd.x)};{F(gd.y)};{F(gd.z)};{F(gazeOk ? gaze.Confidence : 0f)};{gazeFeed}");
    }

    /// <summary>
    /// Whether the stitched panorama is <b>functional and visible</b>: on screen (quality verdict
    /// good and not toggled off by the pilot) and still being refreshed by Python. Runs every frame
    /// and writes a <c>stitch_on</c> / <c>stitch_off</c> event on each change, the off event's note
    /// saying why, so the durations come from frame-timed events rather than the sample grid.
    ///
    /// The freshness test is not redundant with the quality verdict: a Python that dies leaves its
    /// last panorama and quality word in the section, so the screen stays up showing a frozen image
    /// with no transition logged anywhere. Before this existed only PyUniSharingFast's untimed
    /// <c>[Panorama] hidden/restored</c> lines in Editor.log recorded any of it.
    /// </summary>
    private void UpdateStitchState()
    {
        bool shown = PyUniSharingFast.PanoramaDisplayed;
        float age = PanoramaAgeSec();
        bool functional = shown && age <= PanoramaStaleSec;

        if (functional) stitchFunctionalSec += Time.deltaTime;
        if (functional == stitchFunctional) return;
        stitchFunctional = functional;

        if (functional)
        {
            stitchOnCount++;
            WriteEvent("stitch_on", -1, "", -1f, PyUniSharingFast.ActiveStitcher.ToString());
            return;
        }

        string reason;
        if (!PyUniSharingFast.PanoramaPilotEnabled) reason = "pilot_off";
        else if (!shown) reason = "quality: " + PyUniSharingFast.DescribeQualityReason(PyUniSharingFast.PanoramaQualityWord);
        else reason = $"stale: no new panorama for {age.ToString("F1", Inv)} s";
        WriteEvent("stitch_off", -1, "", -1f, reason);
    }

    /// <summary>Seconds since Python last delivered a new panorama; NaN before the first.</summary>
    private static float PanoramaAgeSec()
    {
        float last = PyUniSharingFast.PanoramaLastNewTime;
        return last < 0f ? float.NaN : Time.time - last;
    }

    /// <summary>
    /// The components of <see cref="UpdateStitchState"/> at the sample rate: which stitcher, whether
    /// the panorama is on screen, Python's quality bit and the pilot's toggle (the two things that
    /// hide it), the raw failing-gate bits (<c>PyUniSharingFast.DescribeQualityReason</c> decodes
    /// them), and the age of the newest panorama. <c>functional</c> is the combined verdict.
    /// </summary>
    private void WriteStitchSample(float t, long ms)
    {
        int word = PyUniSharingFast.PanoramaQualityWord;
        stitchWriter.WriteLine(
            $"{F(t)};{ms};{PyUniSharingFast.ActiveStitcher};{(PyUniSharingFast.PanoramaDisplayed ? 1 : 0)};" +
            $"{word & 1};{(PyUniSharingFast.PanoramaPilotEnabled ? 1 : 0)};{word & 0xFFFE};" +
            $"{F(PanoramaAgeSec())};{(stitchFunctional ? 1 : 0)}");
    }

    /// <summary>
    /// Unity's own frame timing since the previous row: frames rendered, the mean rate over them,
    /// the longest single frame, and how many drone FPV cameras rendered
    /// (<see cref="FPVCameraScript.TotalRenders"/>, every schedule). With a headset attached the
    /// frame time is quantised to the refresh (13.9 ms at 72 Hz), so <c>dtMaxMs</c> near 27.8
    /// means a frame was missed, and a run of rows at ~36 fps is the runtime's half-rate lock.
    /// Unscaled time, so a time-scaled session still reports real frame rates.
    /// </summary>
    private void WritePerfSample(float t, long ms)
    {
        long renders = FPVCameraScript.TotalRenders;
        long rendersSince = perfLastRenders >= 0 ? renders - perfLastRenders : 0;
        float fps = perfDtSum > 0f ? perfFrames / perfDtSum : 0f;
        perfWriter.WriteLine($"{F(t)};{ms};{perfFrames};{F(fps)};{F(perfDtMax * 1000f)};{rendersSince}");

        perfLastRenders = renders;
        perfFrames = 0;
        perfDtSum = 0f;
        perfDtMax = 0f;
    }

    private void WriteWalkerSample(float t, long ms)
    {
        for (int i = 0; i < goals.Count; i++)
        {
            Transform sp = goals[i].special;
            if (sp == null) continue;
            Vector3 p = sp.position;
            walkersWriter.WriteLine($"{F(t)};{ms};{i};{F(p.x)};{F(p.y)};{F(p.z)};{goals[i].hat}");
        }
    }

    // ------------------------------------------------------------------ identify

    /// <summary>
    /// The pilot's RC button: an identify timed at the press, its outcome left for the experimenter's
    /// key. Every press is logged — an ignored one as <c>rc_press_ignored</c> — so a mistaken or
    /// doubled press can be found afterwards. Presses arriving in one frame count once.
    /// </summary>
    private void PollRcIdentify()
    {
        int presses = UDPReceiverManager.MarkPresses;
        if (presses == rcPressesSeen) return;
        bool pressed = presses > rcPressesSeen;   // lower: the counter was reset under us
        rcPressesSeen = presses;
        if (!pressed || !rcIdentify) return;

        if (Time.time - lastRcIdentifyTime < rcIdentifyDebounceSec)
        {
            WriteEvent("rc_press_ignored", -1, "", -1f, "debounce");
            return;
        }
        if (identifyMode == IdentifyMode.AutoScore)
        {
            ScorePress();
            lastRcIdentifyTime = Time.time;
            return;
        }
        if (RecordIdentify("", "rc")) lastRcIdentifyTime = Time.time;
        else WriteEvent("rc_press_ignored", -1, "", -1f, "all goals answered");
    }

    /// <summary>
    /// AutoScore: score a press (see the class summary). Correct answers the goal; a false alarm is counted and
    /// leaves it open; out of zone touches no goal. The press is remembered for the experimenter's override.
    /// </summary>
    private void ScorePress()
    {
        Vector3 centroid = AliveCentroid();
        int gazeFeed = EyeGazeTracker.EyeTrackingActive && EyeGazeTracker.Combined.IsValid ? GazeFeed(EyeGazeTracker.Combined) : -1;
        float t = Time.time - sessionStartTime;
        int gi = GoalAtCentroid(centroid, out float zoneDist);
        lastPressGoal = gi;
        lastPressTime = t;
        lastPressCentroid = centroid;
        string where = $"centroid={F(centroid.x)},{F(centroid.y)},{F(centroid.z)} source=rc score=auto gazeFeed={gazeFeed}";
        if (gi < 0)
        {
            WriteEvent("identify", -1, "out_of_zone", -1f, where);
            Debug.Log("ExperimentRecorder: RC press outside every goal's zone (out_of_zone).", this);
            return;
        }
        GoalInfo g = goals[gi];
        float walkerDist = g.special != null ? Vector3.Distance(centroid, g.special.position) : -1f;
        bool seen = HatOnShownFeed(g, out int feed, out float camDist);
        string note = $"{where} feed={feed} camDist={F(camDist)} zoneDist={F(zoneDist)}";
        if (g.answered && g.outcome == "correct")
        {
            WriteEvent("identify", gi, "repeat", walkerDist, note);
            return;
        }
        if (!seen)
        {
            g.falseAlarms++;
            WriteEvent("identify", gi, "false_alarm", walkerDist, note);
            Debug.Log($"ExperimentRecorder: RC press at goal {gi} ({g.hat}) scored a false alarm: the hat is on no " +
                      "shown feed.", this);
            return;
        }
        g.answered = true;
        g.source = "rc";
        g.outcome = "correct";
        g.decisionTimeSec = t;
        g.swarmToWalkerDist = walkerDist;
        g.centroidAtAnswer = centroid;
        g.scoredFeed = feed;
        g.scoredCamDist = camDist;
        g.gazeFeedAtAnswer = gazeFeed;
        WriteEvent("identify", gi, "correct", walkerDist, note);
        Debug.Log($"ExperimentRecorder: RC press at goal {gi} ({g.hat}) scored correct: feed {feed}, {F(camDist)} m.", this);
        FinalizeIfComplete();
    }

    /// <summary>The goal whose zone contains the point (the nearest such patch if zones overlap), or -1.</summary>
    private int GoalAtCentroid(Vector3 point, out float zoneDist)
    {
        int best = -1;
        zoneDist = float.PositiveInfinity;
        for (int i = 0; i < goals.Count; i++)
        {
            if (goals[i].goal == null) continue;
            float d = GoalZones.DistanceToPatch(goals[i].goal, point);
            if (d <= identifyZoneM && d < zoneDist)
            {
                zoneDist = d;
                best = i;
            }
        }
        return best;
    }

    /// <summary>
    /// Whether the goal's hat is inside the central part of an alive drone's shown feed, within range, with no
    /// building between camera and hat. The nearest such camera is reported (index in the swarm list, distance).
    /// </summary>
    private bool HatOnShownFeed(GoalInfo g, out int feed, out float camDist)
    {
        feed = -1;
        camDist = -1f;
        List<GameObject> swarm = spawner != null ? spawner.swarm : null;
        if (g.special == null || swarm == null) return false;
        Vector3 hat = g.special.position + Vector3.up * HatAboveWalker;
        float half = 0.5f * identifyCentralFraction;
        float best = float.PositiveInfinity;
        for (int i = 0; i < swarm.Count; i++)
        {
            GameObject drone = swarm[i];
            if (drone == null || !FeedShown(drone)) continue;
            Transform dp = drone.transform.Find("DroneParent");
            VelocityControl vc = dp != null ? dp.GetComponent<VelocityControl>() : null;
            if (vc != null && vc.State != null && !vc.State.IsAlive) continue;
            Transform fpv = drone.transform.Find("FPV");
            Camera cam = fpv != null ? fpv.GetComponent<Camera>() : null;
            if (cam == null) continue;
            float d = Vector3.Distance(cam.transform.position, hat);
            if (d > identifyRangeM || d >= best) continue;
            Vector3 v = cam.WorldToViewportPoint(hat);
            if (v.z <= 0f || Mathf.Abs(v.x - 0.5f) > half || Mathf.Abs(v.y - 0.5f) > half) continue;
            if (Physics.Linecast(cam.transform.position, hat, obstacleMask, QueryTriggerInteraction.Ignore)) continue;
            best = d;
            feed = i;
            camDist = d;
        }
        return feed >= 0;
    }

    /// <summary>Whether a drone's feed quad is on screen (false when there is no ScreenSpawn).</summary>
    private bool FeedShown(GameObject drone)
    {
        if (screenSpawn == null || drone == null) return false;
        GameObject screen = screenSpawn.ScreenFor(drone);
        return screen != null && screen.activeInHierarchy;
    }

    /// <summary>
    /// The swarm-list index of the drone whose feed quad the gaze ray meets first, or -1. The quads carry no
    /// collider (ScreenSpawn removes them), so this intersects the ray with each shown quad's plane: Unity's Quad
    /// spans [-0.5, 0.5] in its local X and Y.
    /// </summary>
    private int GazeFeed(EyeGazeTracker.GazeSample gaze)
    {
        if (screenSpawn == null || spawner == null || spawner.swarm == null || !gaze.IsValid) return -1;
        float bestT = float.PositiveInfinity;
        GameObject hit = null;
        foreach (GameObject screen in screenSpawn.Screens)
        {
            if (screen == null || !screen.activeInHierarchy) continue;
            Transform s = screen.transform;
            float denom = Vector3.Dot(gaze.Direction, s.forward);
            if (Mathf.Abs(denom) < 1e-6f) continue;
            float t = Vector3.Dot(s.position - gaze.Origin, s.forward) / denom;
            if (t <= 0f || t >= bestT) continue;
            Vector3 local = s.InverseTransformPoint(gaze.Origin + t * gaze.Direction);
            if (Mathf.Abs(local.x) > 0.5f || Mathf.Abs(local.y) > 0.5f) continue;
            bestT = t;
            hit = screen;
        }
        if (hit == null) return -1;
        GameObject drone = screenSpawn.DroneFor(hit);
        return drone != null ? spawner.swarm.IndexOf(drone) : -1;
    }

    /// <summary>
    /// An experimenter key. It gives the outcome of the latest RC identify still waiting for one; with
    /// none waiting it is the identify itself, timed at the key, exactly as before the RC button existed.
    /// </summary>
    private void RecordOutcome(string outcome)
    {
        if (identifyMode == IdentifyMode.AutoScore)
        {
            OverrideLatestPress(outcome);
            return;
        }
        int idx = LatestAwaitingOutcome();
        if (idx < 0)
        {
            RecordIdentify(outcome, "key");
            return;
        }

        GoalInfo g = goals[idx];
        g.outcome = outcome;
        float lag = Time.time - sessionStartTime - g.decisionTimeSec;
        WriteEvent("outcome", idx, outcome, g.swarmToWalkerDist, $"source=key lagSec={F(lag)}");
        FinalizeIfComplete();
    }

    /// <summary>
    /// AutoScore: the experimenter's key overrides the latest press's score. It never makes an identify of its own:
    /// with no scored press, or the latest out of every zone, it is logged and changes nothing.
    /// </summary>
    private void OverrideLatestPress(string outcome)
    {
        if (lastPressGoal < 0)
        {
            WriteEvent("outcome_override", -1, outcome, -1f,
                       lastPressGoal == -1 ? "source=key ignored: the latest press was out of zone"
                                           : "source=key ignored: no press to override");
            return;
        }
        GoalInfo g = goals[lastPressGoal];
        string before = g.outcome;
        g.outcome = outcome;
        g.answered = true;
        g.source = string.IsNullOrEmpty(g.source) ? "key_override" : (g.source.EndsWith("+key") ? g.source : g.source + "+key");
        if (g.decisionTimeSec < 0f)
        {
            g.decisionTimeSec = lastPressTime;
            g.centroidAtAnswer = lastPressCentroid;
        }
        WriteEvent("outcome_override", lastPressGoal, outcome, g.swarmToWalkerDist, $"source=key before={before}");
        FinalizeIfComplete();
    }

    /// <summary>Answers the nearest unanswered goal; false if none remain.</summary>
    private bool RecordIdentify(string outcome, string source)
    {
        int idx = NearestUnansweredGoal(out float dist, out Vector3 centroid);
        if (idx < 0)
        {
            Debug.Log($"ExperimentRecorder: identify ({source}) but all goals already answered (ignored).", this);
            return false;
        }

        GoalInfo g = goals[idx];
        g.answered = true;
        g.source = source;
        g.outcome = outcome;
        g.decisionTimeSec = Time.time - sessionStartTime;
        g.swarmToWalkerDist = dist;
        g.centroidAtAnswer = centroid;
        // No ';' in the note: pandas readers of _events.csv split on it.
        WriteEvent("identify", idx, outcome, dist,
            $"centroid={F(centroid.x)},{F(centroid.y)},{F(centroid.z)} source={source}");
        if (outcome == "")
            Debug.Log($"ExperimentRecorder: RC identify -> goal {idx} ({g.hat}, walker {F(dist)} m away); " +
                      "press the outcome key.", this);

        FinalizeIfComplete();
        return true;
    }

    /// <summary>The most recent RC identify with no outcome yet; -1 if none.</summary>
    private int LatestAwaitingOutcome()
    {
        int best = -1;
        for (int i = 0; i < goals.Count; i++)
        {
            GoalInfo g = goals[i];
            if (g.answered && g.outcome == "" && (best < 0 || g.decisionTimeSec > goals[best].decisionTimeSec))
                best = i;
        }
        return best;
    }

    /// <summary>
    /// Finishes once every goal has an outcome, so an RC identify still waiting for its key holds the
    /// session open. Where no key is pressed (answers noted by hand) it ends manually, as it always has.
    /// </summary>
    private void FinalizeIfComplete()
    {
        if (goals.Count == 0) return;
        foreach (GoalInfo g in goals)
            if (!g.answered || g.outcome == "") return;
        Finalize(identifyMode == IdentifyMode.AutoScore ? "all_goals_identified" : "all_goals_answered");
    }

    /// <summary>Nearest not-yet-answered goal to the swarm centroid; -1 if none remain.</summary>
    private int NearestUnansweredGoal(out float swarmToWalkerDist, out Vector3 centroid)
    {
        centroid = AliveCentroid();
        swarmToWalkerDist = -1f;
        int best = -1;
        float bestSq = float.MaxValue;
        for (int i = 0; i < goals.Count; i++)
        {
            if (goals[i].answered) continue;
            float dSq = (goals[i].goal.position - centroid).sqrMagnitude;
            if (dSq < bestSq) { bestSq = dSq; best = i; }
        }
        if (best >= 0 && goals[best].special != null)
            swarmToWalkerDist = Vector3.Distance(centroid, goals[best].special.position);
        return best;
    }

    /// <summary>
    /// Centroid of the alive drones. A crashed drone stays in the swarm list where it fell (below ground, once
    /// parked), so averaging every drone pulled the centroid down and sideways and could attribute a press to the
    /// wrong goal; every centroid this recorder logs or scores with is now this one.
    /// </summary>
    private Vector3 AliveCentroid()
    {
        List<GameObject> swarm = spawner != null ? spawner.swarm : null;
        if (swarm == null || swarm.Count == 0) return Vector3.zero;
        Vector3 sum = Vector3.zero;
        int n = 0;
        foreach (GameObject drone in swarm)
        {
            if (drone == null) continue;
            Transform dp = drone.transform.Find("DroneParent");
            if (dp == null) continue;
            VelocityControl vc = dp.GetComponent<VelocityControl>();
            if (vc != null && vc.State != null && !vc.State.IsAlive) continue;
            sum += (vc != null && vc.State != null) ? vc.State.GroundTruthPosition : dp.position;
            n++;
        }
        return n > 0 ? sum / n : SwarmCentroid();
    }

    private Vector3 SwarmCentroid()
    {
        List<GameObject> swarm = spawner != null ? spawner.swarm : null;
        if (swarm == null || swarm.Count == 0) return Vector3.zero;

        Vector3 sum = Vector3.zero;
        int n = 0;
        foreach (GameObject drone in swarm)
        {
            if (drone == null) continue;
            Transform dp = drone.transform.Find("DroneParent");
            if (dp == null) continue;
            VelocityControl vc = dp.GetComponent<VelocityControl>();
            sum += (vc != null && vc.State != null) ? vc.State.GroundTruthPosition : dp.position;
            n++;
        }
        return n > 0 ? sum / n : Vector3.zero;
    }

    /// <summary>
    /// Logs the start line's crossings, each at the frame <see cref="StartLine"/> detected it rather than this one's
    /// (the two components' Updates are not ordered). The first live drone's is the clock; the centroid's is kept
    /// beside it because a swarm's front drone leads its centroid by ~1 s, which a single drone does not.
    /// </summary>
    private void LogStartLine()
    {
        if (startLine == null) return;
        if (!startLineLogged && startLine.Crossed)
        {
            startLineLogged = true;
            Vector3 p = startLine.CrossedPosition;
            WriteEventAt(startLine.CrossedTime, "start_line", -1, "", -1f,
                $"drone={startLine.CrossedDrone} x={F(p.x)} z={F(p.z)} centroidBehindM={F(startLine.CentroidBehindAtCross)}");
        }
        if (!startCentroidLogged && startLine.CentroidCrossed)
        {
            startCentroidLogged = true;
            Vector3 p = startLine.CentroidCrossedPosition;
            WriteEventAt(startLine.CentroidCrossedTime, "start_line_centroid", -1, "", -1f, $"x={F(p.x)} z={F(p.z)}");
        }
    }

    // ------------------------------------------------------------------ finalize

    private void Finalize(string reason)
    {
        if (!sessionOpen || finalized) return;
        finalized = true;

        try
        {
            WriteEvent("session_end", -1, "", -1f, reason);
            WriteSessionJson();
        }
        catch (Exception e)
        {
            Debug.LogError($"ExperimentRecorder: error finalizing session. {e}", this);
        }
        finally
        {
            CloseWriters();
            sessionOpen = false;
            Debug.Log($"ExperimentRecorder: session finalized ({reason}).", this);
        }
    }

    private void WriteSessionJson()
    {
        var data = new SessionData
        {
            participantId = participantId,
            trialNumber = trialNumber,
            condition = condition.ToString(),
            droneCount = (spawner != null && spawner.swarm != null) ? spawner.swarm.Count : 0,
            sampleHz = sampleHz,
            wallStartIso = wallStart.ToString("o"),
            wallEndIso = DateTime.Now.ToString("o"),
            durationSec = Time.time - sessionStartTime,
            stitchFunctionalSec = stitchFunctionalSec,
            stitchOnCount = stitchOnCount,
            goalSeed = GoalSession.HasSeed ? GoalSession.Seed : 0,
            goalSeedSource = GoalSession.HasSeed ? GoalSession.Source : "",
            identifyMode = identifyMode.ToString(),
            layout = CityLayoutInfo.Current ?? new CityLayoutInfo.Record(),
        };

        int nCorrect = 0, falseAlarms = 0;
        float lastIdentifyT = 0f;
        for (int i = 0; i < goals.Count; i++)
        {
            GoalInfo g = goals[i];
            if (g.outcome == "correct") nCorrect++;
            falseAlarms += g.falseAlarms;
            if (g.answered && g.decisionTimeSec > lastIdentifyT) lastIdentifyT = g.decisionTimeSec;
            // g.goal may be a destroyed Transform if finalizing during scene teardown.
            Vector3 goalPos = g.goal != null ? g.goal.position : Vector3.zero;
            data.goals.Add(new GoalResult
            {
                goalIndex = i,
                goalX = goalPos.x, goalY = goalPos.y, goalZ = goalPos.z,
                specialSpawnX = g.specialSpawnPos.x, specialSpawnY = g.specialSpawnPos.y, specialSpawnZ = g.specialSpawnPos.z,
                hat = g.hat,
                dealOrdinal = g.dealOrdinal,
                targetPatrolIndex = g.targetPatrolIndex,
                targetPatrol = g.targetPatrol,
                targetWalkerIndex = g.targetWalkerIndex,
                zoneShape = g.zoneShape,
                patchRadius = g.patchRadius,
                r_g = g.ringRadius,
                w_c = g.occluderWidth,
                c_g = g.clearance,
                answered = g.answered,
                identifySource = g.source,
                outcome = g.outcome,
                decisionTimeSec = g.decisionTimeSec,
                swarmToWalkerDist = g.swarmToWalkerDist,
                centroidAtAnswerX = g.centroidAtAnswer.x,
                centroidAtAnswerY = g.centroidAtAnswer.y,
                centroidAtAnswerZ = g.centroidAtAnswer.z,
                falseAlarms = g.falseAlarms,
                scoredFeed = g.scoredFeed,
                scoredCamDist = g.scoredCamDist,
                gazeFeedAtAnswer = g.gazeFeedAtAnswer,
            });
        }
        data.nCorrect = nCorrect;
        data.falseAlarms = falseAlarms;
        data.nGoals = goals.Count;
        // Total task time = start → last identify: the pilot's RC press where there was one, else the
        // key (0 if nothing was answered).
        data.totalTaskTime = lastIdentifyT;

        // The start line, on the session clock; -1 = no line in the scene, or never crossed.
        if (startLine != null && startLine.Placed)
        {
            data.startLineX = startLine.Point.x;
            data.startLineZ = startLine.Point.z;
            data.startLineHeadingDeg = startLine.PlacedHeadingDeg;
            data.startLineDistanceM = startLine.PlacedDistance;
            if (startLine.Crossed)
            {
                data.startLineSec = startLine.CrossedTime - sessionStartTime;
                data.startLineDrone = startLine.CrossedDrone;
                if (lastIdentifyT > 0f) data.taskTimeFromStartLine = lastIdentifyT - data.startLineSec;
            }
            if (startLine.CentroidCrossed) data.startCentroidSec = startLine.CentroidCrossedTime - sessionStartTime;
        }

        string json = JsonUtility.ToJson(data, true);
        File.WriteAllText(Path.Combine(dirPath, $"{fileStem}_session.json"), json, new UTF8Encoding(false));
    }

    // ------------------------------------------------------------------ helpers

    private void WriteEvent(string type, int goalIndex, string outcome, float swarmToWalkerDist, string note) =>
        WriteEventAt(Time.time, type, goalIndex, outcome, swarmToWalkerDist, note);

    /// <summary>An event that happened at <paramref name="time"/> (<c>Time.time</c>), not necessarily this frame.</summary>
    private void WriteEventAt(float time, string type, int goalIndex, string outcome, float swarmToWalkerDist, string note)
    {
        if (eventsWriter == null) return;
        eventsWriter.WriteLine(
            $"{F(time - sessionStartTime)};{UnixMs()};{type};{goalIndex};{outcome};{F(swarmToWalkerDist)};{note}");
        eventsWriter.Flush();
    }

    private void CloseWriters()
    {
        SafeClose(ref dronesWriter);
        SafeClose(ref headWriter);
        SafeClose(ref walkersWriter);
        SafeClose(ref eventsWriter);
        SafeClose(ref shapeWriter);
        SafeClose(ref stitchWriter);
        SafeClose(ref perfWriter);
    }

    private static void SafeClose(ref StreamWriter w)
    {
        if (w == null) return;
        try { w.Flush(); w.Dispose(); } catch { /* already closed */ }
        w = null;
    }

    private static long UnixMs() => DateTimeOffset.UtcNow.ToUnixTimeMilliseconds();
    private static string F(float v) => v.ToString("F4", Inv);

    // ---- JSON DTOs (JsonUtility-serializable) ----
    [Serializable]
    private class SessionData
    {
        public string participantId;
        public int trialNumber;
        public string condition;
        public int droneCount;
        public float sampleHz;
        public string wallStartIso;
        public string wallEndIso;
        public float durationSec;
        public float totalTaskTime;        // Play -> last identify
        // The start line (StartLine): the first live drone's crossing starts the timed task. -1 = none / not crossed.
        public float startLineSec = -1f;
        public int startLineDrone = -1;
        public float startCentroidSec = -1f;   // the alive centroid's crossing (a swarm's ~1 s after its front drone)
        public float taskTimeFromStartLine = -1f;   // start line -> last identify
        public float startLineX, startLineZ, startLineHeadingDeg, startLineDistanceM;
        public float stitchFunctionalSec;  // panorama on screen and live (UpdateStitchState)
        public int stitchOnCount;          // times it came on
        // GoalSession's seed: the goal tiles, hat deal, target walkers and walker starts all follow from it, and
        // GoalPatchReplacer's Replay reads it back. An empty source = no goal seed was set (no goals).
        public int goalSeed;
        public string goalSeedSource;
        public string identifyMode;        // AutoScore or Legacy (absent = Legacy: runs before 2026-10-12)
        public int falseAlarms;            // AutoScore presses scored wrong, over every goal
        public CityLayoutInfo.Record layout;   // the generated layout the city was built from (empty id in tile cities)
        public int nCorrect;
        public int nGoals;
        public List<GoalResult> goals = new List<GoalResult>();
    }

    [Serializable]
    private class GoalResult
    {
        public int goalIndex;
        public float goalX, goalY, goalZ;
        public float specialSpawnX, specialSpawnY, specialSpawnZ;
        public string hat;
        public int dealOrdinal;            // this goal's place in the hat deal (GoalSpecialWalker)
        public int targetPatrolIndex;      // which of the goal's patrols carries the hat, and its building
        public string targetPatrol;
        public int targetWalkerIndex;      // which of that patrol's walkers wears it
        public string zoneShape;           // "tile" (analyse.py's square) or "disc" (a procedural goal)
        public float patchRadius;          // the tile's half-width, or the disc's radius
        public float r_g, w_c, c_g;        // a procedural goal's ring radius, occluder width and clearance (-1 tile)
        public bool answered;
        public string identifySource;   // rc: decisionTimeSec is the pilot's press; key: the experimenter's key
        public string outcome;
        public float decisionTimeSec;
        public float swarmToWalkerDist;
        public float centroidAtAnswerX, centroidAtAnswerY, centroidAtAnswerZ;
        public int falseAlarms;            // AutoScore presses at this goal scored wrong
        public int scoredFeed;             // the swarm-list index of the camera that scored the correct press
        public float scoredCamDist;        // its distance to the hat (m)
        public int gazeFeedAtAnswer;       // the feed the pilot's gaze was on at that press (-1 none)
    }
}
