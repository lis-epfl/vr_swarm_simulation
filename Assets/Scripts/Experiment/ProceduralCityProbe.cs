using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reflection;
using UnityEngine;

/// <summary>
/// A headless Play-mode check of the procedural city and the recorder's new fields, run in batchmode on the bench's
/// project copy (<c>ProceduralCityProbeLauncher.Run</c>). Inert unless <see cref="OutEnv"/> names a report file, so it
/// costs nothing in the editor or a build. Like the bench it opens no named section, socket or camera; unlike the bench
/// it keeps <see cref="ExperimentRecorder"/> running, redirected by <c>EXPERIMENT_RECORDER_DIR</c> to a scratch folder.
///
/// <para>Checks, in order: the layout built three goals with three different hats, each with every walker its patrols
/// ask for (64 in a citygen goal: 8 buildings of 8); goals drawn at Play from a block-shaped city (<see cref="BlockGoals"/>)
/// are spread, have rings of their own and are drawn again the same from the session seed; after the walkers have
/// walked, none stands inside a building and every one is within a footpath's width of one; a swarm reset 70 m up is
/// held under the altitude ceiling; an
/// RC press far from every goal is scored out_of_zone and one beside a goal is scored (correct or false_alarm) with
/// its camera distance and feed; and the session record carries the layout, the goal seed, the goals' geometry and
/// the new hull / shown / gaze columns. The report is one JSON object; the process exits 0 if every check passed.</para>
/// </summary>
public class ProceduralCityProbe : MonoBehaviour
{
    public const string OutEnv = "PROCEDURAL_PROBE_OUT";
    /// <summary>m: CityDressing's footpath, kerb to facade (blockcity.INSET_M); a walker farther off every building is in
    /// the street.</summary>
    const float FootpathM = 3.5f;

    static string outPath;
    readonly List<string> lines = new List<string>();
    bool allOk = true;

    [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.AfterSceneLoad)]
    static void Boot()
    {
        outPath = Environment.GetEnvironmentVariable(OutEnv);
        if (string.IsNullOrEmpty(outPath)) return;
        foreach (var c in FindObjectsOfType<PyUniSharingFast>(true)) c.enabled = false;
        foreach (var c in FindObjectsOfType<UDPReceiverManager>(true)) c.enabled = false;
        foreach (var c in FindObjectsOfType<ImageSharing>(true)) c.enabled = false;
        foreach (var cam in FindObjectsOfType<Camera>(true)) cam.enabled = false;
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
        if (spawner == null || builder == null || rec == null)
        {
            Check("scene", false, $"spawner {spawner != null}, builder {builder != null}, recorder {rec != null}");
            yield return Finish(rec);
            yield break;
        }

        // ---- goals, hats, walkers
        var goals = builder.PlacedGoals;
        var hats = goals.Select(g => g.GetComponent<GoalSpecialWalker>()?.DealtPrefab ?? "").ToList();
        Check("three goals, three hats", goals.Count == 3 && hats.Distinct().Count() == 3 && hats.All(h => h != ""),
              $"{goals.Count} goals, hats [{string.Join(", ", hats)}], seed {GoalSession.Seed} ({GoalSession.Source})");
        ProceduralCityBuilder.LayoutGoal[] drawn = builder.Layout.layout.goals ?? new ProceduralCityBuilder.LayoutGoal[0];
        bool atPlay = drawn.Length > 0 && drawn.All(g => g.kind == BlockGoals.Kind);
        if (atPlay) CheckDrawnGoals(builder, drawn);
        yield return new WaitForSeconds(3f);
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

        // ---- the altitude ceiling
        Vector3 spawn = new Vector3(spawner.start_x, 0f, spawner.start_z);
        spawner.ResetToPos(spawn + Vector3.up * 70f);
        yield return new WaitForSeconds(25f);
        float top = Alive(spawner).Select(t => t.position.y).DefaultIfEmpty(float.NaN).Max();
        Check("held under the ceiling", top <= 52.5f, $"highest drone at {top:F2} m after a reset to 70 m (ceiling 50, flown ~51.3)");

        // ---- auto-scored presses: one far from every goal, one beside goal 0
        PropertyInfo marks = typeof(UDPReceiverManager).GetProperty("MarkPresses", BindingFlags.Public | BindingFlags.Static);
        MethodInfo setMarks = marks?.GetSetMethod(true);
        if (setMarks == null)
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
        }
        yield return Finish(rec);
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
              $"closest pair {apart:F0} m apart, nearest {spawnDist:F0} m from the spawn; " +
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
                Check("csv columns", drones.EndsWith(";hull;shown") && head.EndsWith(";gazeX;gazeY;gazeZ;gazeConf;gazeFeed"),
                      $"drones [{drones}] head ends [{head.Substring(Math.Max(0, head.Length - 40))}]");
                bool outZone = events.Contains(";identify;-1;out_of_zone;");
                bool scored = events.Contains(";identify;0;correct;") || events.Contains(";identify;0;false_alarm;");
                Check("auto-scored presses", outZone && scored && events.Contains("camDist=") && events.Contains("gazeFeed="),
                      $"out_of_zone {outZone}, goal-0 press scored {scored}");
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
