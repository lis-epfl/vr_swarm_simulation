using System.Collections.Generic;
using UnityEngine;

/// <summary>
/// S8 Footprint Avoidance: each obstacle collider's horizontal footprint (its convex hull in world XZ) and the
/// smallest circle round it, built once per collider and cached. OlfatiSaber reads it when
/// SwarmManager.obstacleShape is EnclosingCircle or Footprint; under Cylinder nothing here runs.
///
/// <para>The footprint comes from the collider itself, so every scene works without a layout file: a convex
/// MeshCollider (BlockShapedPatchCityWorld's prisms) gives the hull of its sharedMesh -- the collider's mesh, which
/// static batching leaves alone, unlike the MeshFilter's -- a BoxCollider its eight corners, anything else its bounds.
/// Obstacles are taken to be static, as every building in these scenes is; a tile GoalPatchReplacer swaps in is a
/// new collider and gets its own entry.</para>
///
/// <para>Mirrored by swarm_replica.PolyCity (the polygon, closest()) and swarm_replica.enclosing_circle, which every
/// number in S8 was measured on.</para>
/// </summary>
public static class ObstacleFootprint
{
    public sealed class Footprint
    {
        public Vector2[] polygon;       // world (x, z), counter-clockwise
        public Vector2 circleCentre;    // the smallest circle containing the polygon
        public float circleRadius;
        public float centreY;           // the collider's bounds centre height, for frames drawn in 3D
        public float halfHeight;
    }

    private static readonly Dictionary<int, Footprint> cache = new Dictionary<int, Footprint>();

    public static Footprint Get(Collider collider)
    {
        int id = collider.GetInstanceID();
        if (cache.TryGetValue(id, out Footprint fp))
            return fp;
        fp = Build(collider);
        cache[id] = fp;
        return fp;
    }

    /// <summary>Drops every cached footprint (a scene that moves its obstacles at runtime would need this).</summary>
    public static void Clear() => cache.Clear();

    private static Footprint Build(Collider collider)
    {
        var points = new List<Vector2>();
        if (collider is MeshCollider mc && mc.sharedMesh != null && mc.sharedMesh.isReadable)
        {
            Transform t = mc.transform;
            foreach (Vector3 v in mc.sharedMesh.vertices)
            {
                Vector3 w = t.TransformPoint(v);
                points.Add(new Vector2(w.x, w.z));
            }
        }
        else if (collider is BoxCollider box)
        {
            Transform t = box.transform;
            Vector3 h = 0.5f * box.size;
            for (int sx = -1; sx <= 1; sx += 2)
                for (int sy = -1; sy <= 1; sy += 2)
                    for (int sz = -1; sz <= 1; sz += 2)
                    {
                        Vector3 w = t.TransformPoint(box.center + new Vector3(sx * h.x, sy * h.y, sz * h.z));
                        points.Add(new Vector2(w.x, w.z));
                    }
        }
        if (points.Count < 3)
        {
            Bounds b = collider.bounds;
            points.Clear();
            points.Add(new Vector2(b.min.x, b.min.z));
            points.Add(new Vector2(b.max.x, b.min.z));
            points.Add(new Vector2(b.max.x, b.max.z));
            points.Add(new Vector2(b.min.x, b.max.z));
        }

        Bounds bounds = collider.bounds;
        var fp = new Footprint
        {
            polygon = ConvexHull(points),
            centreY = bounds.center.y,
            halfHeight = bounds.extents.y,
        };
        EnclosingCircle(fp.polygon, out fp.circleCentre, out fp.circleRadius);
        return fp;
    }

    private static double Cross(Vector2 o, Vector2 a, Vector2 b)
        => (double)(a.x - o.x) * (b.y - o.y) - (double)(a.y - o.y) * (b.x - o.x);

    /// <summary>Andrew's monotone chain: the convex hull, counter-clockwise in (x, z), collinear points dropped.</summary>
    public static Vector2[] ConvexHull(List<Vector2> points)
    {
        points.Sort((p, q) => p.x != q.x ? p.x.CompareTo(q.x) : p.y.CompareTo(q.y));
        var hull = new List<Vector2>();
        for (int pass = 0; pass < 2; pass++)
        {
            int start = hull.Count;
            for (int k = 0; k < points.Count; k++)
            {
                Vector2 p = pass == 0 ? points[k] : points[points.Count - 1 - k];
                while (hull.Count >= start + 2 && Cross(hull[hull.Count - 2], hull[hull.Count - 1], p) <= 1e-9)
                    hull.RemoveAt(hull.Count - 1);
                hull.Add(p);
            }
            hull.RemoveAt(hull.Count - 1);
        }
        return hull.ToArray();
    }

    /// <summary>
    /// The smallest circle containing the points: the smallest of the circles on two of them as a diameter or
    /// through three of them that contains them all. Brute force and exact, in double precision; a footprint has a
    /// handful of vertices. swarm_replica.enclosing_circle is the same loop in the same order.
    /// </summary>
    public static void EnclosingCircle(Vector2[] p, out Vector2 centre, out float radius)
    {
        int k = p.Length;
        if (k == 1)
        {
            centre = p[0];
            radius = 0.0f;
            return;
        }
        double bestX = 0.0, bestZ = 0.0, bestR = double.PositiveInfinity;
        for (int i = 0; i < k; i++)
        {
            for (int j = i + 1; j < k; j++)
            {
                double cx = 0.5 * ((double)p[i].x + p[j].x), cz = 0.5 * ((double)p[i].y + p[j].y);
                double r = Dist(p[i], cx, cz);
                if (r < bestR && Contains(p, cx, cz, r))
                {
                    bestX = cx; bestZ = cz; bestR = r;
                }
            }
        }
        for (int i = 0; i < k; i++)
        {
            for (int j = i + 1; j < k; j++)
            {
                for (int l = j + 1; l < k; l++)
                {
                    double bx = (double)p[j].x - p[i].x, bz = (double)p[j].y - p[i].y;
                    double qx = (double)p[l].x - p[i].x, qz = (double)p[l].y - p[i].y;
                    double det = 2.0 * (bx * qz - bz * qx);
                    if (System.Math.Abs(det) < 1e-9)
                        continue;
                    double b2 = bx * bx + bz * bz, q2 = qx * qx + qz * qz;
                    double cx = p[i].x + (qz * b2 - bz * q2) / det;
                    double cz = p[i].y + (bx * q2 - qx * b2) / det;
                    double r = Dist(p[i], cx, cz);
                    if (r < bestR && Contains(p, cx, cz, r))
                    {
                        bestX = cx; bestZ = cz; bestR = r;
                    }
                }
            }
        }
        centre = new Vector2((float)bestX, (float)bestZ);
        radius = (float)bestR;
    }

    private static double Dist(Vector2 a, double x, double z)
    {
        double dx = a.x - x, dz = a.y - z;
        return System.Math.Sqrt(dx * dx + dz * dz);
    }

    private static bool Contains(Vector2[] p, double x, double z, double r)
    {
        for (int i = 0; i < p.Length; i++)
            if (Dist(p[i], x, z) > r + 1e-6)
                return false;
        return true;
    }

    /// <summary>
    /// The closest point of the footprint to (x, z) and its distance: the point itself at distance 0 when inside.
    /// swarm_replica.PolyCity.closest.
    /// </summary>
    public static float Closest(Footprint fp, Vector2 point, out Vector2 closest)
    {
        Vector2[] poly = fp.polygon;
        int k = poly.Length;
        bool inside = true;
        float best = float.PositiveInfinity;
        closest = point;
        for (int e = 0; e < k; e++)
        {
            Vector2 a = poly[e], b = poly[(e + 1) % k];
            Vector2 ab = b - a;
            // outward normal of a counter-clockwise edge in (x, z): (dz, -dx)
            if ((ab.y * (point.x - a.x) - ab.x * (point.y - a.y)) > 0.0f)
                inside = false;
            float t = Mathf.Clamp01(Vector2.Dot(point - a, ab) / Mathf.Max(ab.sqrMagnitude, 1e-12f));
            Vector2 q = a + t * ab;
            float d = (point - q).magnitude;
            if (d < best)
            {
                best = d;
                closest = q;
            }
        }
        if (inside)
        {
            closest = point;
            return 0.0f;
        }
        return best;
    }
}
