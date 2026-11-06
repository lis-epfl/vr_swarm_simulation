using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reflection;
using UnityEngine;
using UnityEngine.Rendering;

/// <summary>
/// A headless Play-mode check of the procedural city and the recorder's new fields, run in batchmode on the bench's
/// project copy (<c>ProceduralCityProbeLauncher.Run</c>). Inert unless <see cref="OutEnv"/> names a report file, so it
/// costs nothing in the editor or a build. Like the bench it opens no named section, socket or camera; unlike the bench
/// it keeps <see cref="ExperimentRecorder"/> running, redirected by <c>EXPERIMENT_RECORDER_DIR</c> to a scratch folder.
///
/// <para>Checks, in order: the layout built three goals with three different hats (as many as were drawn, for goals
/// drawn at Play: PracticeBlockWorld draws one), each with every walker its patrols ask for (64 in a citygen goal: 8
/// buildings of 8); goals drawn at Play from a block-shaped city (<see cref="BlockGoals"/>)
/// are spread, have rings of their own and are drawn again the same from the session seed; after the walkers have
/// walked, none stands inside a building and every one is within a footpath's width of one; a swarm reset 70 m up is
/// held under the altitude ceiling; an
/// RC press far from every goal is scored out_of_zone and one beside a goal is scored (correct or false_alarm) with
/// its camera distance and feed; a hat seen only through a feed hidden into the live panorama scores, and not with the
/// panorama down; a colliding pair is logged as two drone_lost events naming each other; and the session record is on
/// disk, unfinished, while the run goes, and ends carrying the layout, the goal seed, the goals' geometry, the commit,
/// the scoring constants, the screen settings and the hull / shown / stitched / gaze columns. The report is one JSON
/// object; the process exits 0 if every check passed.
/// With <see cref="NoRecorderEnv"/> set to 1 a scene without a recorder (the practice scenes) is probed too, and the
/// press and session checks, which need one, are reported as skipped.</para>
/// <para>With <see cref="SphereEnv"/> set to 1 the scene is flown as the SingleSphere condition instead: one drone,
/// <see cref="InterfaceManager.DisplayMode.SPHERE"/>, the matching label. The panorama and colliding-pair checks, which
/// need feeds or two drones, give way to the sphere's: the cube is created, refreshed, counted six renders a refresh
/// and lit on every face; the sphere shows the city the right way round (the view through the sphere against a camera
/// at the drone, at three headings, and against the turns a sign or 90 deg slip would give); a hat straight below the
/// drone scores, and one out of range or behind a building does not; and the run records the display mode, the
/// sphere's settings, its yaw on every head sample and the drone's view as shown.</para>
/// </summary>
public class ProceduralCityProbe : MonoBehaviour
{
    public const string OutEnv = "PROCEDURAL_PROBE_OUT";
    public const string NoRecorderEnv = "PROCEDURAL_PROBE_NO_RECORDER";
    public const string SphereEnv = "PROCEDURAL_PROBE_SPHERE";
    /// <summary>m: CityDressing's footpath, kerb to facade (blockcity.INSET_M); a walker farther off every building is in
    /// the street.</summary>
    const float FootpathM = 3.5f;

    static string outPath;
    static bool sphereRun;
    readonly List<string> lines = new List<string>();
    bool allOk = true;
    int lostA = -1, lostB = -1;   // the swarm-list indices of the pair flown into each other

    [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.AfterSceneLoad)]
    static void Boot()
    {
        outPath = Environment.GetEnvironmentVariable(OutEnv);
        if (string.IsNullOrEmpty(outPath)) return;
        foreach (var c in FindObjectsOfType<PyUniSharingFast>(true)) c.enabled = false;
        foreach (var c in FindObjectsOfType<UDPReceiverManager>(true)) c.enabled = false;
        foreach (var c in FindObjectsOfType<ImageSharing>(true)) c.enabled = false;
        foreach (var cam in FindObjectsOfType<Camera>(true)) cam.enabled = false;

        // After every Awake and before any Start, so the spawn and the display are set before they are read.
        sphereRun = Environment.GetEnvironmentVariable(SphereEnv) == "1";
        if (sphereRun)
        {
            swarmSpawn spawner = FindObjectOfType<swarmSpawn>();
            if (spawner != null)
            {
                // One drone at the start position itself, so ResetToPos puts it exactly where it is sent.
                spawner.spawnPattern = SpawnPattern.CircleWithCentre;
                spawner.circleDroneCount = 1;
            }
            InterfaceManager im = FindObjectOfType<InterfaceManager>();
            if (im != null) im.displayMode = InterfaceManager.DisplayMode.SPHERE;
            ExperimentRecorder rec = FindObjectOfType<ExperimentRecorder>();
            if (rec != null) rec.condition = ExperimentRecorder.Condition.SingleSphere;
        }
        new GameObject("ProceduralCityProbe").AddComponent<ProceduralCityProbe>();
    }

    void Check(string name, bool ok, string detail)
    {
        allOk &= ok;
        lines.Add($"{{\"check\":\"{name}\",\"ok\":{(ok ? "true" : "false")},\"detail\":\"{detail.Replace("\"", "'")}\"}}");
        Debug.Log($"[Probe] {(ok ? "OK  " : "FAIL")} {name}: {detail}");
    }

    IEnumerator Start()
    {
        Time.maximumDeltaTime = 1f;
        Time.timeScale = 4f;
        for (int i = 0; i < 10; i++) yield return null;
        foreach (var cam in FindObjectsOfType<Camera>(true)) cam.enabled = false;   // the drones' FPV cameras
        swarmSpawn spawner = FindObjectOfType<swarmSpawn>();
        ProceduralCityBuilder builder = FindObjectOfType<ProceduralCityBuilder>();
        ExperimentRecorder rec = FindObjectOfType<ExperimentRecorder>();
        bool noRecorder = Environment.GetEnvironmentVariable(NoRecorderEnv) == "1";
        if (spawner == null || builder == null || (rec == null && !noRecorder))
        {
            Check("scene", false, $"spawner {spawner != null}, builder {builder != null}, recorder {rec != null}");
            yield return Finish(rec);
            yield break;
        }

        // ---- goals, hats, walkers
        var goals = builder.PlacedGoals;
        var hats = goals.Select(g => g.GetComponent<GoalSpecialWalker>()?.DealtPrefab ?? "").ToList();
        ProceduralCityBuilder.LayoutGoal[] drawn = builder.Layout.layout.goals ?? new ProceduralCityBuilder.LayoutGoal[0];
        bool atPlay = drawn.Length > 0 && drawn.All(g => g.kind == BlockGoals.Kind);
        int want = atPlay ? drawn.Length : 3;
        Check($"{want} goals, {want} hats", goals.Count == want && hats.Distinct().Count() == want && hats.All(h => h != ""),
              $"{goals.Count} goals, hats [{string.Join(", ", hats)}], seed {GoalSession.Seed} ({GoalSession.Source})");
        if (atPlay) CheckDrawnGoals(builder, drawn);
        yield return new WaitForSeconds(3f);
        if (rec != null) CheckSessionWrittenEarly();
        var walkerCounts = new List<int>();
        var walkers = new List<Transform>();
        foreach (GameObject g in goals)
        {
            int n = 0;
            foreach (Transform c in g.transform)
            {
                if (!c.name.StartsWith("Walkers (")) continue;
                n += c.childCount;
                foreach (Transform w in c) walkers.Add(w);
            }
            walkerCounts.Add(n);
        }
        var asked = goals.Select(g => g.GetComponentsInChildren<WalkerPatrol>(true).Sum(p => p.numWalkers)).ToList();
        Check("walkers per goal", walkerCounts.SequenceEqual(asked) && walkerCounts.All(n => n > 0),
              $"walkers per goal [{string.Join(", ", walkerCounts)}], patrols ask for [{string.Join(", ", asked)}]");
        int obstacle = LayerMask.GetMask("Obstacle");
        int inside = 0, offPath = 0;
        float far = 0f;
        foreach (Transform w in walkers)
        {
            Vector3 p = w.position + Vector3.up;
            float nearest = float.PositiveInfinity;
            foreach (Collider c in Physics.OverlapSphere(p, FootpathM + 1f, obstacle))
            {
                nearest = Mathf.Min(nearest, (c.ClosestPoint(p) - p).magnitude);
            }
            if (nearest < 1e-3f) inside++;
            else if (nearest < 0.3f || nearest > FootpathM - 0.1f) offPath++;
            if (nearest < float.PositiveInfinity) far = Mathf.Max(far, nearest);
        }
        Check("no walker inside a building", inside == 0, $"{inside} of {walkers.Count} walkers inside a collider");
        Check("every walker on a footpath", offPath == 0,
              $"{offPath} of {walkers.Count} walkers under 0.3 m or over {FootpathM - 0.1f:0.0} m from the nearest building " +
              $"(farthest {far:F2} m)");

        // ---- the 360 sphere: its cube, and which way round it shows the city
        if (sphereRun)
        {
            yield return CheckSphereCube();
            yield return CheckSphereOrientation();
        }

        // ---- the altitude ceiling
        Vector3 spawn = new Vector3(spawner.start_x, 0f, spawner.start_z);
        spawner.ResetToPos(spawn + Vector3.up * 70f);
        yield return new WaitForSeconds(25f);
        float top = Alive(spawner).Select(t => t.position.y).DefaultIfEmpty(float.NaN).Max();
        Check("held under the ceiling", top <= 52.5f, $"highest drone at {top:F2} m after a reset to 70 m (ceiling 50, flown ~51.3)");

        // ---- auto-scored presses: one far from every goal, one beside goal 0
        PropertyInfo marks = typeof(UDPReceiverManager).GetProperty("MarkPresses", BindingFlags.Public | BindingFlags.Static);
        MethodInfo setMarks = marks?.GetSetMethod(true);
        if (rec == null)
        {
            lines.Add($"{{\"check\":\"press and session\",\"ok\":true,\"skipped\":true,\"detail\":\"the scene has no " +
                      $"recorder ({NoRecorderEnv})\"}}");
            Debug.Log("[Probe] SKIP press and session: the scene has no recorder");
        }
        else if (setMarks == null)
        {
            Check("press", false, "UDPReceiverManager.MarkPresses has no setter to simulate a press with");
        }
        else
        {
            spawner.ResetToPos(spawn + Vector3.up * 20f);
            yield return new WaitForSeconds(4f);
            setMarks.Invoke(null, new object[] { UDPReceiverManager.MarkPresses + 1 });
            yield return new WaitForSeconds(2f);
            ProceduralGoal g0 = goals[0].GetComponent<ProceduralGoal>();
            Vector3 beside = goals[0].transform.position + new Vector3(0.5f * (g0.clearance + g0.occluderWidth * 0.5f), 12f, 0f);
            spawner.ResetToPos(beside);
            yield return new WaitForSeconds(10f);
            setMarks.Invoke(null, new object[] { UDPReceiverManager.MarkPresses + 1 });
            yield return new WaitForSeconds(1f);
            if (sphereRun)
            {
                yield return CheckSphereScoring(rec, spawner);
                yield return Finish(rec);
                yield break;
            }
            CheckPanoramaScoring(rec, spawner);

            // ---- a collision: last, since it costs the swarm two drones
            List<Transform> pair = Alive(spawner).Take(2).ToList();
            if (pair.Count == 2)
            {
                lostA = spawner.swarm.IndexOf(pair[0].parent.gameObject);
                lostB = spawner.swarm.IndexOf(pair[1].parent.gameObject);
                pair[0].position = pair[1].position + Vector3.right * 0.2f;
                yield return new WaitForSeconds(1f);
            }
        }
        yield return Finish(rec);
    }

    /// <summary>A crash must not cost the session record: it is on disk, unfinished, with the goals in it, from the
    /// first frame.</summary>
    void CheckSessionWrittenEarly()
    {
        string dir = Environment.GetEnvironmentVariable("EXPERIMENT_RECORDER_DIR");
        string session = string.IsNullOrEmpty(dir) ? null :
            Directory.GetFiles(dir, "*_session.json").OrderBy(File.GetLastWriteTimeUtc).LastOrDefault();
        string json = session != null ? File.ReadAllText(session) : "";
        Check("session written early", json.Contains("\"finalized\": false") && json.Contains("\"goalIndex\": 0"),
              session != null ? Path.GetFileName(session) : $"no *_session.json in {dir}");
    }

    /// <summary>A hat seen only through one camera, whose own feed is hidden: it scores while that view is in the live
    /// panorama, and not once the panorama is down. All in one frame, before a layout can bring a screen back; nothing
    /// here runs the bridge, so the panorama's state is set by hand.</summary>
    void CheckPanoramaScoring(ExperimentRecorder rec, swarmSpawn spawner)
    {
        const BindingFlags Private = BindingFlags.NonPublic | BindingFlags.Instance;
        Type type = typeof(ExperimentRecorder);
        FieldInfo functional = type.GetField("stitchFunctional", Private);
        MethodInfo hatOnFeed = type.GetMethod("HatOnShownFeed", Private);
        object goal = ((IList)type.GetField("goals", Private).GetValue(rec))[0];
        var marker = (SpecialWalker)goal.GetType().GetField("marker").GetValue(goal);
        ScreenSpawn screens = FindObjectOfType<ScreenSpawn>();
        GameObject drone = Alive(spawner).First().parent.gameObject;
        Transform fpv = drone.transform.Find("FPV");

        foreach (GameObject s in screens.Screens) s.SetActive(false);
        screens.SetStitchedDronesHidden(new[] { drone });
        fpv.position = marker.HatCentre + new Vector3(0.5f, 10f, 0f);
        fpv.LookAt(marker.HatCentre);

        object[] args = { goal, -1, -1f };
        functional.SetValue(rec, false);
        bool down = (bool)hatOnFeed.Invoke(rec, args);
        functional.SetValue(rec, true);
        bool up = (bool)hatOnFeed.Invoke(rec, args);
        int feed = (int)args[1];
        functional.SetValue(rec, false);
        screens.SetStitchedDronesHidden(null);

        Check("a hat in the live panorama scores", up && !down && feed == spawner.swarm.IndexOf(drone),
              $"panorama up {up} (feed {feed}, {args[2]} m), panorama down {down}");
    }

    /// <summary>SPHERE: the cube exists at its size, refreshes, counts six renders a refresh, and every face has
    /// something on it (a cube that failed to create still "renders", black).</summary>
    IEnumerator CheckSphereCube()
    {
        Sphere360Display s = Sphere360Display.Instance;
        if (!Sphere360Display.Active || s == null || s.Cube == null)
        {
            Check("sphere cube up", false, "Sphere360Display is not active");
            yield break;
        }
        long r0 = FPVCameraScript.TotalRenders, f0 = s.Refreshes;
        yield return new WaitForSecondsRealtime(1f);
        long dr = FPVCameraScript.TotalRenders - r0, df = s.Refreshes - f0;
        bool counted = s.Config.splitFaces ? Math.Abs(dr - 6 * df) <= 3 : dr == 6 * df;

        int w = s.CubeEdge;
        var means = new float[6];
        for (int face = 0; face < 6; face++)
        {
            AsyncGPUReadbackRequest req = AsyncGPUReadback.Request(s.Cube, 0, 0, w, 0, w, face, 1, TextureFormat.RGBA32);
            req.WaitForCompletion();
            if (req.hasError)
            {
                means[face] = -1f;
                continue;
            }
            var data = req.GetData<Color32>();
            double sum = 0;
            for (int i = 0; i < data.Length; i++) sum += data[i].r + data[i].g + data[i].b;
            means[face] = (float)(sum / (3.0 * data.Length));
        }
        bool lit = means.All(m => m > 2f);
        Check("sphere cube up", s.Cube.IsCreated() && w == (int)s.Config.cubeSize && df > 0 && counted && lit,
              $"{w} px, {df} refreshes and {dr} counted renders in 1 s of real time; face means (0-255) " +
              $"[{string.Join(", ", means.Select(m => m.ToString("F1")))}]");
    }

    /// <summary>
    /// SPHERE: the view through the sphere at display yaw phi must be the city at yaw phi - sphereYaw, seen from where
    /// the cube was rendered. Compared with a camera rendering the city directly, at a matched pixel density (30 deg on
    /// 512 px, ~17 px/deg against the cube's ~18), and with the views a sign slip or a 90 deg slip would show instead,
    /// which must all be far worse. Right after a refresh, so the cube and the direct render are one frame apart.
    /// </summary>
    IEnumerator CheckSphereOrientation()
    {
        Sphere360Display s = Sphere360Display.Instance;
        GameObject sphereObj = GameObject.Find("Sphere360");
        GameObject cubeObj = GameObject.Find("Camera360");
        if (s == null || sphereObj == null || cubeObj == null)
        {
            Check("sphere faces the right way", false, $"sphere {sphereObj != null}, cube camera {cubeObj != null}");
            yield break;
        }
        long f0 = s.Refreshes;
        while (s.Refreshes == f0) yield return null;

        const int N = 512;
        const float Fov = 30f;
        var throughSphere = new RenderTexture(N, N, 24);
        var direct = new RenderTexture(N, N, 24);
        Camera view = new GameObject("ProbeSphereView").AddComponent<Camera>();
        view.enabled = false;
        view.cullingMask = 1 << sphereObj.layer;
        view.clearFlags = CameraClearFlags.SolidColor;
        view.backgroundColor = Color.magenta;
        view.allowHDR = false;
        view.allowMSAA = false;
        view.nearClipPlane = 0.1f;
        view.farClipPlane = 1000f;
        view.fieldOfView = Fov;
        view.aspect = 1f;
        view.targetTexture = throughSphere;
        view.transform.position = sphereObj.transform.position;
        Camera city = new GameObject("ProbeCityView").AddComponent<Camera>();
        city.CopyFrom(cubeObj.GetComponent<Camera>());
        city.enabled = false;
        city.fieldOfView = Fov;
        city.aspect = 1f;
        city.targetTexture = direct;
        city.transform.position = s.LastRenderPosition;
        FPVCameraScript body = s.Drone.transform.Find("FPV").GetComponent<FPVCameraScript>();

        float sYaw = Sphere360Display.SphereYawDegrees;
        bool ok = true;
        var detail = new List<string>();
        foreach (float phi in new[] { 0f, 130f, 250f })
        {
            view.transform.rotation = Quaternion.Euler(0f, phi, 0f);
            view.Render();
            float right = 0f, worstWrong = float.PositiveInfinity;
            var errs = new List<string>();
            foreach (var (label, cityYaw) in new[] { ("right", phi - sYaw), ("sign", phi + sYaw),
                                                    ("+90", phi - sYaw + 90f), ("-90", phi - sYaw - 90f) })
            {
                if (label != "right" && Mathf.Abs(Mathf.DeltaAngle(cityYaw, phi - sYaw)) < 20f) continue;
                city.transform.rotation = Quaternion.Euler(0f, cityYaw, 0f);
                body.SetOwnBodyHidden(true);
                city.Render();
                body.SetOwnBodyHidden(false);
                float e = MeanAbsDiff(throughSphere, direct);
                errs.Add($"{label} {e:F1}");
                if (label == "right") right = e;
                else worstWrong = Mathf.Min(worstWrong, e);
            }
            bool here = right < 6f && right < 0.5f * worstWrong;
            ok &= here;
            detail.Add($"phi {phi:F0}: {string.Join(", ", errs)}");
        }
        Destroy(view.gameObject);
        Destroy(city.gameObject);
        throughSphere.Release();
        direct.Release();
        Check("sphere faces the right way", ok,
              $"sphereYaw {sYaw:F1}; mean |diff| (0-255) against the city at phi - sphereYaw and its slips: " +
              string.Join("; ", detail));
    }

    static float MeanAbsDiff(RenderTexture a, RenderTexture b)
    {
        Color32[] pa = ReadBack(a), pb = ReadBack(b);
        double sum = 0;
        for (int i = 0; i < pa.Length; i++)
        {
            sum += Math.Abs(pa[i].r - pb[i].r) + Math.Abs(pa[i].g - pb[i].g) + Math.Abs(pa[i].b - pb[i].b);
        }
        return (float)(sum / (3.0 * pa.Length));
    }

    static Color32[] ReadBack(RenderTexture rt)
    {
        RenderTexture previous = RenderTexture.active;
        RenderTexture.active = rt;
        var tex = new Texture2D(rt.width, rt.height, TextureFormat.RGBA32, false);
        tex.ReadPixels(new Rect(0, 0, rt.width, rt.height), 0, 0);
        tex.Apply();
        RenderTexture.active = previous;
        Color32[] pixels = tex.GetPixels32();
        Destroy(tex);
        return pixels;
    }

    /// <summary>
    /// SPHERE: a hat counts within range with a clear line, at any bearing: straight below the drone, where no level
    /// feed could see it, it scores; from beyond the range, or from behind the building its walker walks round, it
    /// does not. Each case is read after the cube has refreshed from the new position, since the sphere scores from
    /// where its picture was taken.
    /// </summary>
    IEnumerator CheckSphereScoring(ExperimentRecorder rec, swarmSpawn spawner)
    {
        const BindingFlags Private = BindingFlags.NonPublic | BindingFlags.Instance;
        Type type = typeof(ExperimentRecorder);
        MethodInfo hatOnFeed = type.GetMethod("HatOnShownFeed", Private);
        object goal = ((IList)type.GetField("goals", Private).GetValue(rec))[0];
        var marker = (SpecialWalker)goal.GetType().GetField("marker").GetValue(goal);
        Sphere360Display s = Sphere360Display.Instance;
        float range = (float)(type.GetField("identifyMaxRangeM")?.GetValue(rec) ?? type.GetField("identifyRangeM").GetValue(rec));
        int obstacle = LayerMask.GetMask("Obstacle");

        bool Score(out float dist)
        {
            object[] args = { goal, -1, -1f };
            bool seen = (bool)hatOnFeed.Invoke(rec, args);
            dist = Vector3.Distance(s.LastRenderPosition, marker.HatCentre);
            return seen;
        }
        IEnumerator FlyTo(Vector3 p)
        {
            spawner.ResetToPos(p);
            long f = s.Refreshes;
            yield return new WaitForSeconds(0.5f);
            while (s.Refreshes < f + 2) yield return null;
        }

        yield return FlyTo(marker.HatCentre + Vector3.up * 12f);
        bool below = Score(out float dBelow);

        yield return FlyTo(marker.HatCentre + Vector3.up * (range + 15f));
        bool far = Score(out float dFar);

        // Behind the nearest building, low, on the side away from the hat.
        Vector3 hat = marker.HatCentre;
        Collider wall = Physics.OverlapSphere(hat, 15f, obstacle)
            .OrderBy(c => (c.ClosestPoint(hat) - hat).sqrMagnitude).FirstOrDefault();
        bool behind = false, blocked = false;
        float dBehind = -1f;
        if (wall != null)
        {
            Vector3 away = wall.bounds.center - hat;
            away.y = 0f;
            away.Normalize();
            Vector3 p = wall.bounds.center + away * (Mathf.Max(wall.bounds.extents.x, wall.bounds.extents.z) + 6f);
            p.y = hat.y + 3f;
            yield return FlyTo(p);
            behind = Score(out dBehind);
            blocked = Physics.Linecast(s.LastRenderPosition, marker.HatCentre, obstacle, QueryTriggerInteraction.Ignore);
        }

        Check("a hat on the sphere scores at any bearing, in range and in sight",
              below && !far && wall != null && blocked && !behind,
              $"straight below at {dBelow:F1} m: {below}; from {dFar:F1} m (range {range:F0}): {far}; " +
              $"behind {(wall != null ? wall.name : "no building")} at {dBehind:F1} m (line blocked {blocked}): {behind}");
    }

    /// <summary>Goals drawn at Play: citygen's spread, no ring building shared or holding another goal, and the same
    /// buildings drawn again from the session seed on a fresh read of the layout file.</summary>
    void CheckDrawnGoals(ProceduralCityBuilder builder, ProceduralCityBuilder.LayoutGoal[] drawn)
    {
        float[] s = builder.Layout.layout.spawn;
        var home = new Vector2(s[0], s[2]);
        var centres = drawn.Select(g => new Vector2(g.centre[0], g.centre[1])).ToList();
        float apart = float.PositiveInfinity, spawnDist = centres.Min(c => (c - home).magnitude);
        for (int i = 0; i < centres.Count; i++)
        {
            for (int j = i + 1; j < centres.Count; j++) apart = Mathf.Min(apart, (centres[i] - centres[j]).magnitude);
        }
        Check("drawn goals spread", apart >= BlockGoals.MinSeparation && spawnDist >= BlockGoals.MinSpawnDistance,
              $"{(centres.Count < 2 ? "one goal" : $"closest pair {apart:F0} m apart")}, nearest {spawnDist:F0} m from the spawn; " +
              $"spawn distances [{string.Join(", ", centres.Select(c => (c - home).magnitude.ToString("F0")))}]");

        var patches = drawn.Select(g => new HashSet<string>(g.ring_boxes) { g.centre_box }).ToList();
        int shared = 0;
        for (int i = 0; i < patches.Count; i++)
        {
            for (int j = i + 1; j < patches.Count; j++) shared += patches[i].Intersect(patches[j]).Count();
        }
        Check("rings of their own", shared == 0 && drawn.All(g => g.n_b >= 3 && g.n_b == g.ring_boxes.Length),
              $"{shared} buildings shared; ring sizes [{string.Join(", ", drawn.Select(g => g.n_b))}]; goals " +
              $"[{string.Join(", ", drawn.Select(g => g.centre_box))}]");

        var again = JsonUtility.FromJson<ProceduralCityBuilder.LayoutFile>(File.ReadAllText(builder.ResolvedPath));
        ProceduralCityBuilder.LayoutGoal[] redrawn = BlockGoals.Draw(again, GoalSession.Rng(BlockGoals.SeedKey),
                                                                     drawn.Length, out string error);
        bool same = redrawn != null && redrawn.Select(g => g.centre_box).SequenceEqual(drawn.Select(g => g.centre_box)) &&
                    redrawn.Select(g => string.Join(",", g.ring_boxes)).SequenceEqual(drawn.Select(g => string.Join(",", g.ring_boxes)));
        Check("drawn again from the seed", same,
              redrawn == null ? error : $"seed {GoalSession.Seed}: [{string.Join(", ", redrawn.Select(g => g.centre_box))}]");
    }

    /// <summary>SPHERE: the run says what it showed. The label agrees with the one drone and the display; the session
    /// carries the mode and the sphere; every head sample has the sphere's yaw; and the drone's view is logged as shown,
    /// so the analysis's shown basis is the sphere.</summary>
    void CheckSphereRecord(string json, string events, string stem)
    {
        string[] head = File.ReadAllLines(stem + "_head.csv");
        string[] drones = File.ReadAllLines(stem + "_drones.csv");
        bool yawLogged = head.Length > 1 && head.Skip(1).All(l => l.Substring(l.LastIndexOf(';') + 1).Length > 0);
        bool shown = drones.Length > 1 && drones.Skip(1).All(l => l.Split(';')[9] == "1");
        Check("sphere record",
              json.Contains("\"condition\": \"SingleSphere\"") && json.Contains("\"conditionMismatch\": false") &&
              json.Contains("\"displayMode\": \"SPHERE\"") && json.Contains("\"active\": true") &&
              events.Contains("mode=SPHERE") && yawLogged && shown,
              $"{head.Length - 1} head rows with sphereYaw {yawLogged}, {drones.Length - 1} drone rows shown {shown}, " +
              $"display event {events.Contains("mode=SPHERE")}");
    }

    static IEnumerable<Transform> Alive(swarmSpawn spawner)
    {
        foreach (GameObject d in spawner.swarm)
        {
            Transform dp = d != null ? d.transform.Find("DroneParent") : null;
            VelocityControl vc = dp != null ? dp.GetComponent<VelocityControl>() : null;
            if (dp != null && (vc == null || vc.State == null || vc.State.IsAlive)) yield return dp;
        }
    }

    IEnumerator Finish(ExperimentRecorder rec)
    {
        string dir = Environment.GetEnvironmentVariable("EXPERIMENT_RECORDER_DIR");
        if (rec != null)
        {
            // The (string) overload: object.Finalize() would make a by-name lookup ambiguous.
            typeof(ExperimentRecorder).GetMethod("Finalize", BindingFlags.NonPublic | BindingFlags.Instance, null,
                                                 new[] { typeof(string) }, null)?.Invoke(rec, new object[] { "probe" });
            yield return null;
            string session = string.IsNullOrEmpty(dir) ? null :
                Directory.GetFiles(dir, "*_session.json").OrderBy(File.GetLastWriteTimeUtc).LastOrDefault();
            if (session == null)
            {
                Check("session record", false, $"no *_session.json in {dir}");
            }
            else
            {
                string json = File.ReadAllText(session);
                string stem = session.Substring(0, session.Length - "_session.json".Length);
                string drones = File.ReadLines(stem + "_drones.csv").First();
                string head = File.ReadLines(stem + "_head.csv").First();
                string events = File.ReadAllText(stem + "_events.csv");
                // A layout's own goals take its seed; goals drawn at Play a fresh one (or the builder's), and the session
                // names their buildings, which no layout file does.
                ProceduralCityBuilder builder = FindObjectOfType<ProceduralCityBuilder>();
                ProceduralCityBuilder.LayoutGoal[] lg = builder != null ? builder.Layout?.layout.goals : null;
                bool atPlay = lg != null && lg.Length > 0 && lg[0].kind == BlockGoals.Kind;
                string source = !atPlay ? "layout" : builder.useFixedGoalSeed ? "inspector" : "fresh";
                Check("session record", json.Contains("\"layout\"") && json.Contains($"\"goalSeedSource\": \"{source}\"") &&
                      json.Contains("\"zoneShape\": \"disc\"") && json.Contains("\"identifyMode\": \"AutoScore\"") &&
                      (lg == null || lg.All(g => json.Contains($"\"centreBox\": \"{g.centre_box}\""))),
                      $"{Path.GetFileName(session)} (goal seed source {source})");
                Check("csv columns", drones.EndsWith(";hull;shown;stitched") &&
                      head.EndsWith(";gazeX;gazeY;gazeZ;gazeConf;gazeFeed;sphereYaw"),
                      $"drones [{drones}] head ends [{head.Substring(Math.Max(0, head.Length - 40))}]");
                bool outZone = events.Contains(";identify;-1;out_of_zone;");
                bool scored = events.Contains(";identify;0;correct;") || events.Contains(";identify;0;false_alarm;");
                Check("auto-scored presses", outZone && scored && events.Contains("camDist=") && events.Contains("gazeFeed="),
                      $"out_of_zone {outZone}, goal-0 press scored {scored}");

                string[] eventLines = File.ReadAllLines(stem + "_events.csv");
                Check("event columns", eventLines.All(l => l.Count(c => c == ';') == 6),
                      $"{eventLines.Count(l => l.Count(c => c == ';') != 6)} of {eventLines.Length} lines without 7 fields");
                bool Lost(int a, int b, string reason) =>
                    eventLines.Any(l => l.Contains($";drone_lost;") && l.Contains($"drone={a} reason={reason} partner={b} "));
                bool pair = (Lost(lostA, lostB, "CollidingWithAnotherDrone") && Lost(lostB, lostA, "ExternallyKilled")) ||
                            (Lost(lostB, lostA, "CollidingWithAnotherDrone") && Lost(lostA, lostB, "ExternallyKilled"));
                if (sphereRun)
                {
                    lines.Add("{\"check\":\"a colliding pair names each other\",\"ok\":true,\"skipped\":true," +
                              "\"detail\":\"one drone (SPHERE)\"}");
                    Debug.Log("[Probe] SKIP a colliding pair names each other: one drone (SPHERE)");
                    CheckSphereRecord(json, events, stem);
                }
                else
                {
                    Check("a colliding pair names each other", pair && json.Contains("\"dronesLost\": "),
                          $"pair {lostA}, {lostB}: [{string.Join(" | ", eventLines.Where(l => l.Contains(";drone_lost;")).Select(l => l.Substring(l.LastIndexOf(';') + 1)))}]");
                }

                string gitLine = json.Split('\n').FirstOrDefault(l => l.Contains("\"gitCommit\"")) ?? "";
                string gitError = json.Split('\n').FirstOrDefault(l => l.Contains("\"gitError\"")) ?? "";
                string swarm = File.ReadAllText(stem + "_swarm.json");
                Check("what the run flew",
                      json.Contains("\"finalized\": true") && json.Contains("\"endReason\": \"probe\"") &&
                      json.Contains("\"identifyRangeM\"") && json.Contains("\"stitchingEnabled\"") &&
                      json.Contains("\"screenScale\"") && events.Contains(";display;") &&
                      swarm.Contains("\"swarmManager\"") && swarm.Contains("\"interfaceManager\"") &&
                      swarm.Contains("\"screenSpawn\"") && swarm.Contains("\"stitching\""),
                      $"{gitLine.Trim()} {gitError.Trim()}; _swarm.json {swarm.Length} bytes");
            }
        }
        File.WriteAllText(outPath, "{\"ok\":" + (allOk ? "true" : "false") + ",\"checks\":[\n" + string.Join(",\n", lines) + "\n]}\n");
#if UNITY_EDITOR
        UnityEditor.EditorApplication.Exit(allOk ? 0 : 1);
#else
        Application.Quit(allOk ? 0 : 1);
#endif
    }
}
