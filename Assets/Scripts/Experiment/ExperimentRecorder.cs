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
/// fixed rate and logs an "identify" event whenever the experimenter presses a key (correct /
/// incorrect / skip) — the participant reports verbally, the experimenter records the outcome. The
/// identify action also marks that goal complete (there is no separate proximity trigger).
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

    // ---- runtime references (resolved lazily on the first Update, once all spawns have run) ----
    private swarmSpawn spawner;
    private Transform headTransform;   // OVRCameraRig.centerEyeAnchor (fallback Camera.main)

    private class GoalInfo
    {
        public Transform goal;
        public Transform special;          // the SpecialWalker's transform (may be null)
        public Vector3 specialSpawnPos;    // captured at resolve time
        public bool answered;
        // Filled in when the goal is answered:
        public string outcome = "";        // correct / incorrect / skip
        public float decisionTimeSec = -1f;
        public float swarmToWalkerDist = -1f;
        public Vector3 centroidAtAnswer;
    }
    private readonly List<GoalInfo> goals = new List<GoalInfo>();
    private bool referencesResolved = false;

    // ---- output ----
    private StreamWriter dronesWriter, headWriter, walkersWriter, eventsWriter, shapeWriter;
    private string dirPath, fileStem;

    // ---- timing / state ----
    private float sessionStartTime;
    private DateTime wallStart;
    private float nextSampleTime;
    private bool sessionOpen = false;   // files open, sampling active
    private bool finalized = false;     // summary written, files closed

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
            referencesResolved = true;
        }

        // Fixed-rate continuous sampling (nextSampleTime pattern mirrors PyUniSharingFast.Update).
        if (Time.time >= nextSampleTime)
        {
            float t = Time.time - sessionStartTime;
            long ms = UnixMs();
            WriteDroneSample(t, ms);
            WriteHeadSample(t, ms);
            WriteWalkerSample(t, ms);
            WriteShapeSample(t, ms);

            float interval = sampleHz > 0f ? 1f / sampleHz : 0.1f;
            // Advance from the scheduled time; if we fell behind, resync to now to avoid a burst.
            nextSampleTime += interval;
            if (nextSampleTime < Time.time) nextSampleTime = Time.time + interval;
        }

        // Identify actions (edge-detected).
        if (Input.GetKeyDown(correctKey)) RecordIdentify("correct");
        else if (Input.GetKeyDown(incorrectKey)) RecordIdentify("incorrect");
        else if (Input.GetKeyDown(skipKey)) RecordIdentify("skip");

        if (Input.GetKeyDown(endSessionKey)) Finalize("manual_end");
    }

    void OnApplicationQuit() => Finalize("app_quit");
    void OnDestroy() => Finalize("destroyed");

    // ------------------------------------------------------------------ setup

    private void OpenSession()
    {
        spawner = FindObjectOfType<swarmSpawn>();
        headTransform = ResolveHeadTransform();

        dirPath = Path.Combine(Application.persistentDataPath, "experiment");
        Directory.CreateDirectory(dirPath);

        string pid = string.IsNullOrWhiteSpace(participantId) ? "XXXX" : participantId.Trim().ToUpperInvariant();
        wallStart = DateTime.Now;
        fileStem = $"{pid}_t{trialNumber}_{condition}_{wallStart:yyyyMMdd_HHmmss}";

        dronesWriter = NewWriter("drones", "t;unixMs;droneId;gtX;gtY;gtZ;yawDeg;alive");
        headWriter = NewWriter("head",
            "t;unixMs;headX;headY;headZ;headYaw;headPitch;headRoll;bodyYaw;inThrottle;inYaw;inPitch;inRoll;inSpread");
        walkersWriter = NewWriter("walkers", "t;unixMs;goalIndex;specialX;specialY;specialZ");
        eventsWriter = NewWriter("events", "t;unixMs;eventType;goalIndex;outcome;swarmToWalkerDist;note");
        shapeWriter = NewWriter("shape",
            "t;unixMs;nAlive;hullVerts;interior;maxGapDeg;meanNNm;ringRadiusM;coreRadiusM;dRef;r0Eff;hollowCore");

        sessionStartTime = Time.time;
        nextSampleTime = Time.time;
        sessionOpen = true;

        WriteEvent("session_start", -1, "", -1f, $"pid={pid};trial={trialNumber};condition={condition}");
        Debug.Log($"ExperimentRecorder: logging to {dirPath} ({fileStem}_*.csv)", this);
    }

    private StreamWriter NewWriter(string suffix, string header)
    {
        string path = Path.Combine(dirPath, $"{fileStem}_{suffix}.csv");
        var w = new StreamWriter(path, false, new UTF8Encoding(false));
        w.WriteLine(header);
        return w;
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

        // Prefer the registry exposed by GoalPatchReplacer; fall back to name-scanning if empty.
        List<GameObject> goalObjects = new List<GameObject>();
        GoalPatchReplacer replacer = FindObjectOfType<GoalPatchReplacer>();
        if (replacer != null && replacer.PlacedGoals != null && replacer.PlacedGoals.Count > 0)
        {
            foreach (GameObject g in replacer.PlacedGoals)
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
            foreach (SpecialWalker sw in specials)
            {
                if (gsw != null && sw.GetComponentInParent<GoalSpecialWalker>() == gsw)
                {
                    special = sw.transform;
                    break;
                }
            }
            goals.Add(new GoalInfo
            {
                goal = g.transform,
                special = special,
                specialSpawnPos = special != null ? special.position : Vector3.zero,
            });
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
            dronesWriter.WriteLine(
                $"{F(t)};{ms};{i};{F(pos.x)};{F(pos.y)};{F(pos.z)};{F(yaw)};{(alive ? 1 : 0)}");
        }
    }

    /// <summary>
    /// Swarm-shape read-outs: how many drones are on the hull (and therefore on screen), how wide the
    /// pilot's largest unobserved sector is, and how far the formation is from the spacing that was
    /// actually commanded. This is what the hollow-core feature is judged on, so it is logged whether
    /// or not the feature is enabled — the disabled runs are the baseline the enabled ones are
    /// compared against.
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
            $"{F(coreR)};{F(dRef)};{F(r0Eff)};{(hollow ? 1 : 0)}");
    }

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

        headWriter.WriteLine(
            $"{F(t)};{ms};{F(hp.x)};{F(hp.y)};{F(hp.z)};{F(he.y)};{F(he.x)};{F(he.z)};{F(bodyYaw)};" +
            $"{F(thr)};{F(yaw)};{F(pit)};{F(rol)};{F(spr)}");
    }

    private void WriteWalkerSample(float t, long ms)
    {
        for (int i = 0; i < goals.Count; i++)
        {
            Transform sp = goals[i].special;
            if (sp == null) continue;
            Vector3 p = sp.position;
            walkersWriter.WriteLine($"{F(t)};{ms};{i};{F(p.x)};{F(p.y)};{F(p.z)}");
        }
    }

    // ------------------------------------------------------------------ identify

    private void RecordIdentify(string outcome)
    {
        int idx = NearestUnansweredGoal(out float dist, out Vector3 centroid);
        if (idx < 0)
        {
            Debug.Log("ExperimentRecorder: identify pressed but all goals already answered (ignored).", this);
            return;
        }

        GoalInfo g = goals[idx];
        g.answered = true;
        g.outcome = outcome;
        g.decisionTimeSec = Time.time - sessionStartTime;
        g.swarmToWalkerDist = dist;
        g.centroidAtAnswer = centroid;
        WriteEvent("identify", idx, outcome, dist,
            $"centroid={F(centroid.x)},{F(centroid.y)},{F(centroid.z)}");

        int answered = 0;
        foreach (GoalInfo gi in goals) if (gi.answered) answered++;
        if (goals.Count > 0 && answered >= goals.Count) Finalize("all_goals_answered");
    }

    /// <summary>Nearest not-yet-answered goal to the swarm centroid; -1 if none remain.</summary>
    private int NearestUnansweredGoal(out float swarmToWalkerDist, out Vector3 centroid)
    {
        centroid = SwarmCentroid();
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
        };

        int nCorrect = 0;
        float lastIdentifyT = 0f;
        for (int i = 0; i < goals.Count; i++)
        {
            GoalInfo g = goals[i];
            if (g.outcome == "correct") nCorrect++;
            if (g.answered && g.decisionTimeSec > lastIdentifyT) lastIdentifyT = g.decisionTimeSec;
            // g.goal may be a destroyed Transform if finalizing during scene teardown.
            Vector3 goalPos = g.goal != null ? g.goal.position : Vector3.zero;
            data.goals.Add(new GoalResult
            {
                goalIndex = i,
                goalX = goalPos.x, goalY = goalPos.y, goalZ = goalPos.z,
                specialSpawnX = g.specialSpawnPos.x, specialSpawnY = g.specialSpawnPos.y, specialSpawnZ = g.specialSpawnPos.z,
                answered = g.answered,
                outcome = g.outcome,
                decisionTimeSec = g.decisionTimeSec,
                swarmToWalkerDist = g.swarmToWalkerDist,
                centroidAtAnswerX = g.centroidAtAnswer.x,
                centroidAtAnswerY = g.centroidAtAnswer.y,
                centroidAtAnswerZ = g.centroidAtAnswer.z,
            });
        }
        data.nCorrect = nCorrect;
        data.nGoals = goals.Count;
        // Total task time = start → last identify (0 if nothing was answered).
        data.totalTaskTime = lastIdentifyT;

        string json = JsonUtility.ToJson(data, true);
        File.WriteAllText(Path.Combine(dirPath, $"{fileStem}_session.json"), json, new UTF8Encoding(false));
    }

    // ------------------------------------------------------------------ helpers

    private void WriteEvent(string type, int goalIndex, string outcome, float swarmToWalkerDist, string note)
    {
        if (eventsWriter == null) return;
        eventsWriter.WriteLine(
            $"{F(Time.time - sessionStartTime)};{UnixMs()};{type};{goalIndex};{outcome};{F(swarmToWalkerDist)};{note}");
        eventsWriter.Flush();
    }

    private void CloseWriters()
    {
        SafeClose(ref dronesWriter);
        SafeClose(ref headWriter);
        SafeClose(ref walkersWriter);
        SafeClose(ref eventsWriter);
        SafeClose(ref shapeWriter);
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
        public float totalTaskTime;
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
        public bool answered;
        public string outcome;
        public float decisionTimeSec;
        public float swarmToWalkerDist;
        public float centroidAtAnswerX, centroidAtAnswerY, centroidAtAnswerZ;
    }
}
