using System;
using System.Collections;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Reflection;
using System.Text;
using UnityEngine;

/// <summary>
/// Records simulated "clips" for the planar stitcher's offline benches: the views a formation of drones would
/// publish, as image sequences with the exact camera pose of every frame. Run headless on a bench copy of the project
/// (<c>PlanarCaptureLauncher.Run</c>); inert unless <see cref="OutEnv"/> names an output folder, so it costs nothing in
/// the editor or a build.
///
/// <para>Why it exists: the recorded real-drone clips test the stitcher against real pose error, but only on the two
/// sites that were flown. The simulator is where the poses are exact, so it is the one place a mosaic's remaining
/// faults can be told apart -- what is left with perfect geometry is the blend and the scene's own depth, and what
/// appears when error is injected afterwards (<c>planar_stream_bench.py --pose-noise</c>) is the estimators'. The city
/// scenes are also the hard case on purpose: storey after storey of identical windows.</para>
///
/// <para>No drone flies. A capture camera copied from a drone's FPV camera -- same culling mask, field of view and, in
/// the scenic scenes, the city lighting -- is placed at each planned pose and rendered, so a formation is exactly the
/// formation asked for and a run is repeatable to the pixel. Three formations are recorded: a wall of six facing the
/// first facade on the city's west edge, drifting along it; six looking straight down over a junction inside the
/// city, translating; and the same over the open ground west of the city.</para>
///
/// <para>Each clip is a folder in the layout <c>tools/planar_stream_bench.py</c> reads for real clips: a
/// <c>session.json</c> with the intrinsics and the true scene plane, and per drone a <c>drone{N}_frames.csv</c> of
/// Unity-world poses beside <c>drone{N}_{frame:0000}.png</c>.</para>
/// </summary>
public class PlanarCaptureProbe : MonoBehaviour
{
    public const string OutEnv = "PLANAR_CAPTURE_OUT";
    /// <summary>Optional "WxH"; the sim scenes' block size by default.</summary>
    public const string SizeEnv = "PLANAR_CAPTURE_SIZE";

    const float Fps = 20f;
    const int Frames = 120;

    static string outRoot;
    int width = 1152, height = 648;
    Camera cam;
    RenderTexture target;
    Texture2D readback;
    readonly StringBuilder log = new StringBuilder();
    // Open ground for the third formation: set from the wall's facade, 90 m out from the city's edge. Not the
    // spawn -- objects tagged as drones are not all where the swarm starts, and the world origin is the pilot's room.
    Vector3? openGround;

    [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.AfterSceneLoad)]
    static void Boot()
    {
        outRoot = Environment.GetEnvironmentVariable(OutEnv);
        if (string.IsNullOrEmpty(outRoot)) return;
        // No named sections, sockets or headset: this runs beside a live editor.
        foreach (var c in FindObjectsOfType<PyUniSharingFast>(true)) c.enabled = false;
        foreach (var c in FindObjectsOfType<UDPReceiverManager>(true)) c.enabled = false;
        foreach (var c in FindObjectsOfType<ImageSharing>(true)) c.enabled = false;
        new GameObject("PlanarCaptureProbe").AddComponent<PlanarCaptureProbe>();
    }

    void Note(string line)
    {
        log.AppendLine(line);
        Debug.Log("[PlanarCapture] " + line);
    }

    IEnumerator Start()
    {
        Time.maximumDeltaTime = 1f;
        string size = Environment.GetEnvironmentVariable(SizeEnv);
        if (!string.IsNullOrEmpty(size))
        {
            string[] wh = size.ToLowerInvariant().Split('x');
            if (wh.Length == 2 && int.TryParse(wh[0], out int w) && int.TryParse(wh[1], out int h) && w > 0 && h > 0)
            {
                width = w;
                height = h;
            }
        }
        // The city is adopted or built, the walkers placed and the lighting baked over the first frames of Play.
        for (int i = 0; i < 60; i++) yield return null;

        bool ok = false;
        string error = null;
        try
        {
            SetUp();
            ok = true;
        }
        catch (Exception e)
        {
            error = e.ToString();
        }
        if (ok)
        {
            string scene = UnityEngine.SceneManagement.SceneManager.GetActiveScene().name;
            yield return CaptureWall(scene);
            yield return CaptureNadir(scene, true);
            yield return CaptureNadir(scene, false);
        }
        else
        {
            Note("setup failed: " + error);
        }
        Directory.CreateDirectory(outRoot);
        File.WriteAllText(Path.Combine(outRoot, "capture_log.txt"), log.ToString());
        File.WriteAllText(Path.Combine(outRoot, ok ? "done.txt" : "failed.txt"), ok ? "ok" : error);
#if UNITY_EDITOR
        UnityEditor.EditorApplication.Exit(ok ? 0 : 3);
#endif
    }

    // ------------------------------------------------------------------ set-up

    void SetUp()
    {
        Camera fpv = null;
        foreach (var script in FindObjectsOfType<FPVCameraScript>(true))
        {
            fpv = script.GetComponent<Camera>();
            if (fpv != null) break;
        }

        var go = new GameObject("PlanarCaptureCamera");
        cam = go.AddComponent<Camera>();
        if (fpv != null) cam.CopyFrom(fpv);
        else cam.cullingMask = 1607;                       // what the FPV cameras draw
        cam.enabled = false;                               // rendered by hand, once per pose
        cam.allowHDR = false;
        cam.stereoTargetEye = StereoTargetEyeMask.None;
        cam.aspect = (float)width / height;
        cam.fieldOfView = FpvCameraFov.VerticalDeg(cam.aspect);

        // sRGB, or every saved frame is far too dark and too contrasty (the descriptor's default is linear).
        var desc = new RenderTextureDescriptor(width, height, RenderTextureFormat.ARGB32, 24) { sRGB = true };
        CityLighting lighting = FindObjectOfType<CityLighting>();
        if (lighting != null) desc.depthStencilFormat = CityLighting.FeedDepthFormat;
        target = new RenderTexture(desc) { name = "PlanarCapture" };
        target.Create();
        cam.targetTexture = target;
        if (lighting != null) lighting.Include(cam);       // shaded as the drone cameras are
        readback = new Texture2D(width, height, TextureFormat.RGB24, false);

        // Nothing of the task in the pictures: the drones sit at the spawn, and the start line tints everything
        // beyond it until a drone crosses.
        foreach (var other in FindObjectsOfType<Camera>(true))
            if (other != cam) other.enabled = false;
        foreach (GameObject drone in GameObject.FindGameObjectsWithTag("DroneBase"))
            foreach (var r in drone.GetComponentsInChildren<Renderer>(true)) r.enabled = false;
        foreach (var line in FindObjectsOfType<StartLine>(true))
        {
            line.enabled = false;
            foreach (string field in new[] { "sheet", "stripe" })
            {
                var f = typeof(StartLine).GetField(field, BindingFlags.NonPublic | BindingFlags.Instance);
                if (f != null && f.GetValue(line) is GameObject visual && visual != null) visual.SetActive(false);
            }
        }
        Note($"camera {width}x{height}, vfov {cam.fieldOfView:F2}, mask {cam.cullingMask}, far {cam.farClipPlane}, " +
             $"city lighting {(lighting != null)}, copied from an FPV camera {(fpv != null)}");
    }

    static int ObstacleMask => 1 << LayerMask.NameToLayer("Obstacle");

    static List<Collider> Buildings()
    {
        int layer = LayerMask.NameToLayer("Obstacle");
        var list = new List<Collider>();
        foreach (var c in FindObjectsOfType<Collider>())
            if (c.gameObject.layer == layer && c.bounds.size.y > 20f) list.Add(c);
        return list;
    }

    /// <summary>Where the swarm stands: the drones themselves, since the spawner's own transform is not the start
    /// position (that is a field of its pattern).</summary>
    static Vector3 SpawnPoint()
    {
        GameObject[] drones = GameObject.FindGameObjectsWithTag("DroneBase");
        if (drones.Length == 0)
        {
            swarmSpawn spawner = FindObjectOfType<swarmSpawn>();
            return spawner != null ? spawner.transform.position : Vector3.zero;
        }
        Vector3 sum = Vector3.zero;
        foreach (GameObject d in drones) sum += d.transform.position;
        return sum / drones.Length;
    }

    static float GroundHeight(Vector3 at)
    {
        // Everything but the buildings: the terrain, or the road and footpath laid on it.
        if (Physics.Raycast(new Vector3(at.x, 400f, at.z), Vector3.down, out RaycastHit hit, 800f, ~ObstacleMask,
                            QueryTriggerInteraction.Ignore))
            return hit.point.y;
        return 0f;
    }

    // ------------------------------------------------------------------ the formations

    /// <summary>Two rows of three, 30 m from the first facade on the city's west edge, drifting along it.</summary>
    IEnumerator CaptureWall(string scene)
    {
        List<Collider> buildings = Buildings();
        float zRef = SpawnPoint().z;
        Collider pick = null;
        foreach (Collider c in buildings)
        {
            if (Mathf.Abs(c.bounds.center.z - zRef) > 160f) continue;
            if (pick == null || c.bounds.center.x < pick.bounds.center.x) pick = c;
        }
        if (pick == null)
        {
            Note("wall: no building found");
            yield break;
        }
        Vector3 from = new Vector3(pick.bounds.min.x - 80f, 16f, pick.bounds.center.z);
        if (!Physics.Raycast(from, Vector3.right, out RaycastHit hit, 200f, ObstacleMask, QueryTriggerInteraction.Ignore))
        {
            Note("wall: the ray from the west missed " + pick.name);
            yield break;
        }
        Vector3 n = hit.normal;
        n.y = 0f;
        n.Normalize();                                     // out of the facade, toward the cameras
        Vector3 along = Vector3.Cross(Vector3.up, n).normalized;
        const float standoff = 30f;
        float ground = GroundHeight(hit.point + n * standoff);
        Quaternion look = Quaternion.LookRotation(-n, Vector3.up);

        var offsets = new List<Vector2>();                 // (along the facade, above the ground)
        foreach (float h in new[] { 12f, 23f })
            foreach (float a in new[] { -9f, 0f, 9f })
                offsets.Add(new Vector2(a, h));

        openGround = hit.point + n * 90f;
        Note($"wall: {pick.name}, facade point {hit.point}, normal {n}, ground {ground:F2}");
        yield return Capture($"{scene}_wall6", offsets.Count, 0f, standoff, n, Vector3.Dot(n, hit.point),
            (drone, t) =>
            {
                Vector2 o = offsets[drone];
                Vector3 p = hit.point + n * standoff + along * (o.x + 1.0f * t - 3f);
                p.y = ground + o.y;
                return (p, look);
            });
    }

    /// <summary>Two rows of three looking straight down, translating: inside the city, or over the open ground.</summary>
    IEnumerator CaptureNadir(string scene, bool inCity)
    {
        Vector3 centre;
        float altitude;
        if (inCity)
        {
            // The most open spot near the middle of the city: a junction or a plaza.
            List<Collider> buildings = Buildings();
            if (buildings.Count == 0)
            {
                Note("nadir: no buildings");
                yield break;
            }
            Vector3 mid = Vector3.zero;
            foreach (Collider c in buildings) mid += c.bounds.center;
            mid /= buildings.Count;
            float best = -1f;
            centre = mid;
            for (float dx = -120f; dx <= 120f; dx += 6f)
            {
                for (float dz = -120f; dz <= 120f; dz += 6f)
                {
                    Vector3 p = new Vector3(mid.x + dx, 20f, mid.z + dz);
                    float clear = 0f;
                    while (clear < 40f && !Physics.CheckSphere(p, clear + 2f, ObstacleMask)) clear += 2f;
                    if (clear > best)
                    {
                        best = clear;
                        centre = p;
                    }
                }
            }
            altitude = 40f;                                // under the 45 m ceiling and the 50 m roofs
            Note($"nadir city: centre {centre}, {best:F0} m clear of buildings");
        }
        else
        {
            if (openGround == null)
            {
                Note("nadir open: no facade was found to place it from");
                yield break;
            }
            centre = openGround.Value;
            altitude = 35f;
            Note($"nadir open: centre {centre}");
        }
        float ground = GroundHeight(centre);

        var offsets = new List<Vector2>();                 // (east, north)
        foreach (float z in new[] { -6f, 6f })
            foreach (float x in new[] { -12f, 0f, 12f })
                offsets.Add(new Vector2(x, z));
        // Headings as a hull would hold them, not all alike: the stitcher has to cope with views rolled against
        // each other, and a detector that is not rotation invariant only can if they are rectified first.
        float[] yaw = { -35f, 0f, 30f, -150f, 180f, 140f };

        yield return Capture(scene + (inCity ? "_nadir_city6" : "_nadir_open6"), offsets.Count, -90f, altitude,
            Vector3.up, ground,
            (drone, t) =>
            {
                Vector2 o = offsets[drone];
                Vector3 p = new Vector3(centre.x + o.x + 2.0f * t - 6f, ground + altitude, centre.z + o.y);
                return (p, Quaternion.Euler(90f, yaw[drone], 0f));
            });
    }

    // ------------------------------------------------------------------ writing a clip

    IEnumerator Capture(string name, int drones, float gimbalPitch, float standoff, Vector3 planeNormal, float planeD,
                        Func<int, float, (Vector3, Quaternion)> pose)
    {
        string dir = Path.Combine(outRoot, name);
        Directory.CreateDirectory(dir);
        var inv = CultureInfo.InvariantCulture;
        var csv = new StringBuilder[drones];
        for (int d = 0; d < drones; d++)
        {
            csv[d] = new StringBuilder();
            csv[d].AppendLine("frame,t_epoch,heading,gimbal_yaw,pos_x,pos_y,pos_z,quat_x,quat_y,quat_z,quat_w,pose_status");
        }

        for (int frame = 0; frame < Frames; frame++)
        {
            float t = frame / Fps;
            for (int d = 0; d < drones; d++)
            {
                (Vector3 p, Quaternion q) = pose(d, t);
                cam.transform.SetPositionAndRotation(p, q);
                cam.Render();
                RenderTexture previous = RenderTexture.active;
                RenderTexture.active = target;
                readback.ReadPixels(new Rect(0, 0, width, height), 0, 0);
                RenderTexture.active = previous;
                File.WriteAllBytes(Path.Combine(dir, $"drone{d + 1}_{frame:0000}.png"), readback.EncodeToPNG());
                // ``frame`` is 1-based, as in the real clips' CSVs; the file index is 0-based.
                csv[d].AppendLine(string.Format(inv, "{0},{1:F4},{2:F3},{2:F3},{3:R},{4:R},{5:R},{6:R},{7:R},{8:R},{9:R},1",
                    frame + 1, 1000.0 + t, q.eulerAngles.y, p.x, p.y, p.z, q.x, q.y, q.z, q.w));
            }
            if (frame % 4 == 3) yield return null;          // let the walkers walk and the log flush
        }
        for (int d = 0; d < drones; d++)
            File.WriteAllText(Path.Combine(dir, $"drone{d + 1}_frames.csv"), csv[d].ToString());

        float fy = 0.5f * height / Mathf.Tan(0.5f * cam.fieldOfView * Mathf.Deg2Rad);
        // Concatenated, not string.Format: a format item directly before a closing brace ("{10:F2}}}") is read as
        // having the brace inside its format string, and prints the format instead of the number.
        string F(float v) => v.ToString("R", inv);
        string json =
            "{\"meta\": {\"sim\": true, \"scene\": \"" + name + "\", \"drones\": " + drones +
            ", \"fps_nominal\": " + F(Fps) + ", \"frames\": " + Frames + ", \"gimbal_pitch\": " + F(gimbalPitch) +
            ", \"wire_size\": [" + width + ", " + height + "], " +
            "\"camera\": {\"width\": " + width + ", \"height\": " + height + ", \"vfov_deg\": " + F(cam.fieldOfView) +
            ", \"fx\": " + F(fy) + ", \"fy\": " + F(fy) + ", \"cx\": " + F(0.5f * width) + ", \"cy\": " + F(0.5f * height) +
            "}, \"scene_plane\": {\"standoff_m\": " + F(standoff) + ", \"plane_n_unity\": [" + F(planeNormal.x) + ", " +
            F(planeNormal.y) + ", " + F(planeNormal.z) + "], \"plane_d_unity\": " + F(planeD) + "}}}";
        File.WriteAllText(Path.Combine(dir, "session.json"), json);
        Note($"wrote {name}: {drones} drones x {Frames} frames");
    }
}
