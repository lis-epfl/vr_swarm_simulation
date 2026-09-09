using System.Collections.Generic;
using UnityEngine;

/// <summary>
/// Read-out and gizmos for the shape of the swarm: how many drones are on the convex hull (and so
/// have their feed shown to the pilot), how wide the widest unobserved sector is, and where the
/// virtual core stands when the hollow-core feature is on.
///
/// Drop it on any GameObject in a swarm scene. It measures and draws only — nothing here feeds back
/// into the control or display path, so leaving it in a scene cannot change how that scene flies.
///
/// Deliberately no <c>using UnityEditor</c> and no <c>Handles</c>: this compiles into a player build,
/// which is where the interesting flights happen.
/// </summary>
public class SwarmShapeGizmos : MonoBehaviour
{
    [Header("Read-out")]
    [Tooltip("Draw the numeric read-out in the top-left corner at runtime.")]
    public bool showReadout = true;

    [Tooltip("Screen position of the read-out box, in pixels from the top-left.")]
    public Vector2 readoutOrigin = new Vector2(10f, 10f);

    [Header("Gizmos")]
    [Tooltip("Draw the hull polygon, the virtual core and the per-drone headings in the Scene view.")]
    public bool showGizmos = true;

    [Tooltip("Length in metres of the heading ray drawn from each drone.")]
    public float headingRayLength = 3.0f;

    [Tooltip("Draw the widest unobserved sector as a wedge from the swarm centroid.")]
    public bool showLargestGap = true;

    private swarmSpawn spawner;
    private GUIStyle readoutStyle;

    private swarmSpawn Spawner
    {
        get
        {
            // FindObjectOfType is far too slow for a per-frame call, and the spawner never changes.
            if (spawner == null) spawner = FindObjectOfType<swarmSpawn>();
            return spawner;
        }
    }

    /// <summary>
    /// The swarm's roster, or null. Asking for the shape metrics also refreshes them, but at most
    /// once per physics tick and shared with the drones' own hull pass, so it costs nothing beyond
    /// whichever caller gets there first.
    /// </summary>
    private List<GameObject> RefreshShape()
    {
        List<GameObject> swarm = Spawner != null ? Spawner.swarm : null;
        if (swarm != null && swarm.Count > 0)
        {
            AttitudeAlgorithm.EnsureSharedGlobalHull(swarm);
            return swarm;
        }
        return null;
    }

    void OnGUI()
    {
        if (!showReadout || !Application.isPlaying) return;
        if (RefreshShape() == null) return;

        if (readoutStyle == null)
        {
            readoutStyle = new GUIStyle(GUI.skin.box)
            {
                alignment = TextAnchor.UpperLeft,
                fontSize = 12,
                richText = false,
            };
        }

        SwarmManager manager = SwarmManager.Instance;
        SwarmPlaneController plane = SwarmPlaneController.Instance;

        float dRef = manager != null ? manager.GetDRef() : 0f;
        float scale = manager != null ? manager.GetScaleFactor() : 1f;
        bool hollow = manager != null && manager.GetHollowSwarmCore();
        float r0Eff = manager != null ? manager.GetEffectiveR0Coh() : 0f;
        float k = dRef > 0f ? r0Eff / dRef : 0f;

        // Commanded spacing in metres, so meanNN can be read against what was actually asked for
        // rather than against a number in swarm units.
        float commandedSpacingM = dRef * scale;

        string text =
            $"Swarm shape{(hollow ? "   [hollow core ON]" : "")}\n" +
            $"alive        {AttitudeAlgorithm.SharedAliveCount}\n" +
            $"on hull      {AttitudeAlgorithm.SharedHullVertexCount}   (shown to pilot)\n" +
            $"interior     {AttitudeAlgorithm.SharedInteriorCount}   (hidden)\n" +
            $"max gap      {AttitudeAlgorithm.SharedMaxGapDeg:F0} deg\n" +
            $"mean NN      {AttitudeAlgorithm.SharedMeanNearestNeighbourM:F1} m   " +
            $"(commanded {commandedSpacingM:F1})\n" +
            $"ring radius  {AttitudeAlgorithm.SharedRingRadiusM:F1} m\n" +
            $"core radius  {(plane != null ? plane.CoreRadiusMetres : 0f):F1} m\n" +
            $"d_ref        {dRef:F2}   r0 {r0Eff:F2}   k {k:F1}";

        GUI.Label(new Rect(readoutOrigin.x, readoutOrigin.y, 260f, 152f), text, readoutStyle);
    }

    void OnDrawGizmos()
    {
        if (!showGizmos || !Application.isPlaying) return;

        List<GameObject> swarm = RefreshShape();
        if (swarm == null) return;

        SwarmPlaneController plane = SwarmPlaneController.Instance;
        bool planeMode = plane != null && plane.PlaneModeActive;

        // The hull is built in the plane's own axes in vertical-plane mode, and un-projecting that
        // back to world would duplicate ProjectForHull's inverse here. The core is off in plane mode
        // anyway, so the drawing below is the horizontal case and says so rather than drawing a
        // plausible-looking lie.
        if (planeMode)
        {
            return;
        }

        Vector3 centroid = plane != null && plane.HasSwarmAggregates
            ? plane.SwarmCentroid
            : SwarmCentroidFallback(swarm);

        // Hull polygon, at the centroid's altitude.
        IList<Vector2> hull = AttitudeAlgorithm.SharedHull;
        if (hull != null && hull.Count >= 2)
        {
            Gizmos.color = new Color(0.2f, 1.0f, 0.4f, 0.9f);
            for (int i = 0; i < hull.Count; i++)
            {
                Vector2 a = hull[i];
                Vector2 b = hull[(i + 1) % hull.Count];
                Gizmos.DrawLine(new Vector3(a.x, centroid.y, a.y),
                                new Vector3(b.x, centroid.y, b.y));
            }
        }

        // The virtual core, as a horizontal ring with a few verticals so it reads as a cylinder.
        float coreRadius = plane != null ? plane.CoreRadiusMetres : 0f;
        if (coreRadius > 0f)
        {
            Gizmos.color = new Color(1.0f, 0.55f, 0.1f, 0.9f);
            DrawCircle(centroid, coreRadius, 48);
            for (int i = 0; i < 8; i++)
            {
                float a = i * Mathf.PI * 2f / 8f;
                Vector3 p = centroid + new Vector3(Mathf.Cos(a), 0f, Mathf.Sin(a)) * coreRadius;
                Gizmos.DrawLine(p + Vector3.down * 2f, p + Vector3.up * 2f);
            }
        }

        // Per-drone heading rays: green where the feed is shown, grey where the drone is interior
        // and its camera is going to waste.
        foreach (GameObject drone in swarm)
        {
            if (!SwarmRegistry.TryGet(drone, out SwarmRegistry.Entry entry)) continue;

            VelocityControl vc = entry.velocityControl;
            if (vc != null && vc.State != null && !vc.State.IsAlive) continue;

            bool boundary = entry.attitude != null && entry.attitude.BoundaryEstimate;
            Gizmos.color = boundary
                ? new Color(0.2f, 1.0f, 0.4f, 1.0f)
                : new Color(0.55f, 0.55f, 0.55f, 1.0f);

            Vector3 p = entry.droneParent.position;
            Gizmos.DrawLine(p, p + entry.droneParent.forward * headingRayLength);
            Gizmos.DrawWireSphere(p, 0.4f);
        }

        // The widest unobserved sector, as a wedge. Drawn from the centroid because that is where
        // the gap is *about* — it is a range of directions, not a place.
        if (showLargestGap && AttitudeAlgorithm.SharedMaxGapDeg < 359f)
        {
            float radius = Mathf.Max(AttitudeAlgorithm.SharedRingRadiusM * 1.6f, 5f);
            Gizmos.color = new Color(1.0f, 0.25f, 0.25f, 0.8f);
            Gizmos.DrawWireSphere(centroid, 0.6f);
            // Only the width is known here, not where it sits, so draw it as a caption-free arc of
            // the right size centred on the swarm — enough to judge "is that gap big".
            DrawArc(centroid, radius, 0f, AttitudeAlgorithm.SharedMaxGapDeg, 24);
        }
    }

    private static Vector3 SwarmCentroidFallback(List<GameObject> swarm)
    {
        Vector3 sum = Vector3.zero;
        int count = 0;
        foreach (GameObject drone in swarm)
        {
            if (!SwarmRegistry.TryGet(drone, out SwarmRegistry.Entry entry)) continue;
            VelocityControl vc = entry.velocityControl;
            if (vc != null && vc.State != null && !vc.State.IsAlive) continue;
            sum += entry.droneParent.position;
            count++;
        }
        return count > 0 ? sum / count : Vector3.zero;
    }

    private static void DrawCircle(Vector3 centre, float radius, int segments)
    {
        DrawArc(centre, radius, 0f, 360f, segments);
    }

    private static void DrawArc(Vector3 centre, float radius, float startDeg, float sweepDeg, int segments)
    {
        if (segments < 1) return;
        Vector3 previous = centre + AngleToOffset(startDeg) * radius;
        for (int i = 1; i <= segments; i++)
        {
            float deg = startDeg + sweepDeg * i / segments;
            Vector3 next = centre + AngleToOffset(deg) * radius;
            Gizmos.DrawLine(previous, next);
            previous = next;
        }
    }

    // Same yaw convention as StateFinder.Angles.y: forward is (sin yaw, 0, cos yaw).
    private static Vector3 AngleToOffset(float degrees)
    {
        float rad = degrees * Mathf.Deg2Rad;
        return new Vector3(Mathf.Sin(rad), 0f, Mathf.Cos(rad));
    }
}
