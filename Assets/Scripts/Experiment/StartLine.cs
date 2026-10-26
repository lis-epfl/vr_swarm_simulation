using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Rendering;
using UnityEngine.SceneManagement;

/// <summary>
/// The trial's start line: the task clock starts when the first live drone crosses it, so the pilot can take off
/// and look around before being timed. It stands <see cref="distanceFromSpawn"/> ahead of the spawn along
/// <see cref="headingDeg"/> and disappears the frame the first drone crosses. <see cref="ExperimentRecorder"/>
/// logs that crossing (<c>start_line</c>) and the alive centroid's (<c>start_line_centroid</c>), and analyse.py
/// times the task from the first.
///
/// <para><b>What the pilot sees is a translucent tint over everything beyond the line, and it has to be that
/// shape.</b> The eye cameras draw only the screens, so the line is visible only through the FPV feeds — which
/// means it is in the stitch input too. A flat-colour sheet whose edges are never in frame cannot cause a seam:
/// for a camera behind it, whether a pixel is tinted depends only on the world point it shows, so the tint is
/// paint on the scenery, which the stitcher aligns like any other. What does break stitching is a layer at a
/// different depth that carries structure, so keep the sheet untextured, unlit and far wider and taller than any
/// view (a finite screen across the road would put its edges in every view, in a different place in each). Its
/// one side effect is that StabStitch's overlap PSNR reads about −20·log10(1 − α) higher until the start, because
/// the tint lowers the contrast of both views equally. A stripe on the ground, just short of the line, marks
/// where it is.</para>
///
/// <para><b>Nothing here has a collider.</b> A collider on <c>Obstacle</c> repels the drones (OlfatiSaber's
/// overlap query hits triggers too), and a solid one anywhere is hit by the scene-plane raycast. Crossing is a
/// signed distance to the line instead, tested every frame for every live drone.</para>
///
/// Added automatically to any scene with a <see cref="ExperimentRecorder"/> and a <see cref="swarmSpawn"/>
/// (PracticeWorld, which has no recorder, gets one from its builder); add one by hand to tune it (the bootstrap
/// then leaves that one alone), or disable it to fly without a line.
/// </summary>
[DefaultExecutionOrder(-50)]   // before ScreenSpawn and PyUniSharingFast render, so the crossing frame has no line
public class StartLine : MonoBehaviour
{
    [Tooltip("Direction the swarm crosses the line in, as a yaw (0 = +Z north, 90 = +X east). Every city " +
             "lies east of its spawn.")]
    public float headingDeg = 90f;

    [Tooltip("Distance from the spawn's centre to the line, in metres. A hovering formation at the widest " +
             "spread reaches ~15 m ahead of the spawn, so this keeps the line clear of it.")]
    public float distanceFromSpawn = 30f;

    [Tooltip("Colour of the sheet over everything beyond the line. Alpha sets how strong the tint is.")]
    public Color tintColor = new Color(0.1f, 0.85f, 1f, 0.3f);

    [Tooltip("Half the sheet's width along the line, in metres. Its edges must stay out of every view.")]
    public float halfWidth = 2000f;

    [Tooltip("Height of the sheet above the ground, in metres. Its top edge must stay above every view.")]
    public float height = 1000f;

    [Tooltip("Colour of the stripe on the ground just short of the line.")]
    public Color stripeColor = new Color(0.1f, 0.85f, 1f, 1f);

    [Tooltip("Depth of the ground stripe along the crossing direction, in metres.")]
    public float stripeDepth = 1.5f;

    [Tooltip("Half the ground stripe's length along the line, in metres.")]
    public float stripeHalfLength = 40f;

    /// <summary>The line has been placed ahead of the spawned swarm.</summary>
    public bool Placed { get; private set; }
    /// <summary>A point on the line, at ground height.</summary>
    public Vector3 Point { get; private set; }
    /// <summary>The heading and distance the line was placed with (the fields may change afterwards).</summary>
    public float PlacedHeadingDeg { get; private set; }
    public float PlacedDistance { get; private set; }

    /// <summary>The first live drone has crossed; the clock has started.</summary>
    public bool Crossed { get; private set; }
    /// <summary><c>Time.time</c> of the first crossing, and which drone (index in <c>swarmSpawn.swarm</c>) made it.</summary>
    public float CrossedTime { get; private set; } = -1f;
    public int CrossedDrone { get; private set; } = -1;
    public Vector3 CrossedPosition { get; private set; }
    /// <summary>The alive drones' centroid was still this far behind the line when the first drone crossed.</summary>
    public float CentroidBehindAtCross { get; private set; }

    /// <summary>The alive drones' centroid has crossed, at <see cref="CentroidCrossedTime"/> (<c>Time.time</c>).</summary>
    public bool CentroidCrossed { get; private set; }
    public float CentroidCrossedTime { get; private set; } = -1f;
    public Vector3 CentroidCrossedPosition { get; private set; }

    // Road surfaces sit a few centimetres above the terrain; a raycast hit higher than this is a prop, not ground.
    private const float MaxGroundAboveTerrain = 1f;
    private const float StripeLift = 0.03f;
    private const float SheetBelowGround = 20f;

    private swarmSpawn spawner;
    private Vector3 normal;
    private GameObject sheet, stripe;
    private readonly List<Material> materials = new List<Material>();
    private readonly List<Mesh> meshes = new List<Mesh>();

    [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.BeforeSceneLoad)]
    private static void RegisterBootstrap()
    {
        // -= first so a disabled domain reload cannot stack the handler.
        SceneManager.sceneLoaded -= OnSceneLoaded;
        SceneManager.sceneLoaded += OnSceneLoaded;
    }

    private static void OnSceneLoaded(Scene scene, LoadSceneMode mode)
    {
        if (FindObjectOfType<StartLine>(true) != null) return;
        if (FindObjectOfType<ExperimentRecorder>() == null || FindObjectOfType<swarmSpawn>() == null) return;
        new GameObject("StartLine").AddComponent<StartLine>();
    }

    private void OnEnable()
    {
        SetVisible(!Crossed);
    }

    private void OnDisable()
    {
        SetVisible(false);
    }

    private void OnDestroy()
    {
        foreach (Mesh m in meshes) Destroy(m);
        foreach (Material m in materials) Destroy(m);
    }

    private void Update()
    {
        if (CentroidCrossed) return;
        if (spawner == null)
        {
            spawner = FindObjectOfType<swarmSpawn>();
            if (spawner == null) return;
        }
        if (!Placed && !TryPlace()) return;

        Vector3 sum = Vector3.zero;
        int alive = 0;
        int first = -1;
        float firstS = 0f;
        Vector3 firstPos = Vector3.zero;
        for (int i = 0; i < spawner.swarm.Count; i++)
        {
            if (!TryGetLiveDrone(spawner.swarm[i], out Vector3 pos)) continue;
            sum += pos;
            alive++;
            float s = SignedDistance(pos);
            if (s >= 0f && (first < 0 || s > firstS))
            {
                first = i;
                firstS = s;
                firstPos = pos;
            }
        }
        if (alive == 0) return;
        Vector3 centroid = sum / alive;

        if (!Crossed && first >= 0)
        {
            Crossed = true;
            CrossedTime = Time.time;
            CrossedDrone = first;
            CrossedPosition = firstPos;
            CentroidBehindAtCross = Mathf.Max(-SignedDistance(centroid), 0f);
            SetVisible(false);
        }
        if (Crossed && SignedDistance(centroid) >= 0f)
        {
            CentroidCrossed = true;
            CentroidCrossedTime = Time.time;
            CentroidCrossedPosition = centroid;
        }
    }

    /// <summary>
    /// Places the line ahead of the spawned swarm's centre, once there is a swarm. Done here rather than in Start
    /// because the spawn is only final once every Start has run (ProceduralCityBuilder writes it).
    /// </summary>
    private bool TryPlace()
    {
        Vector3 sum = Vector3.zero;
        int n = 0;
        foreach (GameObject drone in spawner.swarm)
        {
            if (!TryGetLiveDrone(drone, out Vector3 pos)) continue;
            sum += pos;
            n++;
        }
        if (n == 0) return false;

        PlacedHeadingDeg = headingDeg;
        PlacedDistance = distanceFromSpawn;
        float h = headingDeg * Mathf.Deg2Rad;
        normal = new Vector3(Mathf.Sin(h), 0f, Mathf.Cos(h));
        Vector3 point = sum / n + normal * distanceFromSpawn;
        point.y = GroundHeight(point);
        Point = point;
        Placed = true;

        BuildVisuals();

        // A drone spawned on the far side would start the clock on the first frame without the pilot doing
        // anything; say so loudly rather than quietly moving the line.
        for (int i = 0; i < spawner.swarm.Count; i++)
        {
            if (!TryGetLiveDrone(spawner.swarm[i], out Vector3 pos) || SignedDistance(pos) < 0f) continue;
            Debug.LogError($"[StartLine] Drone {i} spawned past the start line ({SignedDistance(pos):F1} m); " +
                           "the clock starts at once. Increase distanceFromSpawn or check headingDeg.", this);
            break;
        }
        Debug.Log($"[StartLine] Placed {distanceFromSpawn:F0} m from the spawn at ({point.x:F1}, {point.z:F1}), " +
                  $"heading {headingDeg:F0} deg.", this);
        return true;
    }

    private float SignedDistance(Vector3 pos) => Vector3.Dot(pos - Point, normal);

    /// <summary>A drone's position, when it is alive (a crashed one lies where it fell, or is parked below ground).</summary>
    private static bool TryGetLiveDrone(GameObject drone, out Vector3 pos)
    {
        pos = Vector3.zero;
        if (drone == null) return false;
        Transform dp = drone.transform.Find("DroneParent");
        if (dp == null) return false;
        VelocityControl vc = dp.GetComponent<VelocityControl>();
        if (vc != null && vc.State != null && !vc.State.IsAlive) return false;
        pos = dp.position;
        return true;
    }

    /// <summary>
    /// Ground height at the line: the terrain, or a road surface on it. Streets in the generated cities have no
    /// colliders, so the raycast only refines the terrain where there is something to hit.
    /// </summary>
    private float GroundHeight(Vector3 point)
    {
        TerrainHeightSampler.TryGetHeight(point, out float ground);   // 0 with no Terrain
        Vector3 from = new Vector3(point.x, ground + 100f, point.z);
        if (Physics.Raycast(from, Vector3.down, out RaycastHit hit, 200f, ~0, QueryTriggerInteraction.Ignore)
            && Mathf.Abs(hit.point.y - ground) <= MaxGroundAboveTerrain)
        {
            ground = Mathf.Max(ground, hit.point.y);
        }
        return ground;
    }

    private void BuildVisuals()
    {
        Quaternion rotation = Quaternion.LookRotation(normal, Vector3.up);

        // The sheet: local +X along the line, +Y up, facing the crossing direction.
        sheet = MakeQuad("StartLineTint", tintColor,
            new Vector3(-halfWidth, -SheetBelowGround, 0f), new Vector3(halfWidth, -SheetBelowGround, 0f),
            new Vector3(halfWidth, height, 0f), new Vector3(-halfWidth, height, 0f));
        sheet.transform.SetPositionAndRotation(Point, rotation);

        // The stripe: on the ground, ending at the line, so it is never behind the sheet.
        stripe = MakeQuad("StartLineStripe", stripeColor,
            new Vector3(-stripeHalfLength, 0f, -stripeDepth), new Vector3(stripeHalfLength, 0f, -stripeDepth),
            new Vector3(stripeHalfLength, 0f, 0f), new Vector3(-stripeHalfLength, 0f, 0f));
        stripe.transform.SetPositionAndRotation(Point + Vector3.up * StripeLift, rotation);

        SetVisible(enabled && !Crossed);
    }

    private GameObject MakeQuad(string name, Color color, Vector3 a, Vector3 b, Vector3 c, Vector3 d)
    {
        var go = new GameObject(name);
        go.transform.SetParent(transform, false);

        var mesh = new Mesh { name = name };
        mesh.vertices = new[] { a, b, c, d };
        mesh.colors = new[] { Color.white, Color.white, Color.white, Color.white };
        mesh.triangles = new[] { 0, 1, 2, 0, 2, 3 };
        mesh.RecalculateNormals();
        mesh.RecalculateBounds();
        meshes.Add(mesh);
        go.AddComponent<MeshFilter>().sharedMesh = mesh;

        // Sprites/Default: unlit, alpha-blended, double-sided, no fog, and always included in builds. Unlit and
        // flat is the point: any shading would differ between views.
        var material = new Material(Shader.Find("Sprites/Default")) { color = color };
        materials.Add(material);
        MeshRenderer mr = go.AddComponent<MeshRenderer>();
        mr.sharedMaterial = material;
        mr.shadowCastingMode = ShadowCastingMode.Off;
        mr.receiveShadows = false;
        mr.lightProbeUsage = LightProbeUsage.Off;
        mr.reflectionProbeUsage = ReflectionProbeUsage.Off;
        return go;
    }

    private void SetVisible(bool visible)
    {
        if (sheet != null && sheet.activeSelf != visible) sheet.SetActive(visible);
        if (stripe != null && stripe.activeSelf != visible) stripe.SetActive(visible);
    }
}
