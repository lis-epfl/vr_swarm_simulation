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
/// <para>Checks, in order: the layout built three goals with three different hats and 64 walkers each; after the
/// walkers have walked, none stands inside a building; a swarm reset 70 m up is held under the altitude ceiling; an
/// RC press far from every goal is scored out_of_zone and one beside a goal is scored (correct or false_alarm) with
/// its camera distance and feed; and the session record carries the layout, the goal seed, the goals' geometry and
/// the new hull / shown / gaze columns. The report is one JSON object; the process exits 0 if every check passed.</para>
/// </summary>
public class ProceduralCityProbe : MonoBehaviour
{
    public const string OutEnv = "PROCEDURAL_PROBE_OUT";

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
        Check("64 walkers per goal", walkerCounts.All(n => n == 64), $"walkers per goal [{string.Join(", ", walkerCounts)}]");
        int obstacle = LayerMask.GetMask("Obstacle");
        int inside = 0;
        foreach (Transform w in walkers)
        {
            Vector3 p = w.position + Vector3.up;
            foreach (Collider c in Physics.OverlapSphere(p, 0.05f, obstacle))
            {
                if ((c.ClosestPoint(p) - p).sqrMagnitude < 1e-6f) { inside++; break; }
            }
        }
        Check("no walker inside a building", inside == 0, $"{inside} of {walkers.Count} walkers inside a collider");

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
                Check("session record", json.Contains("\"layout\"") && json.Contains("\"goalSeedSource\": \"layout\"") &&
                      json.Contains("\"zoneShape\": \"disc\"") && json.Contains("\"identifyMode\": \"AutoScore\""),
                      Path.GetFileName(session));
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
