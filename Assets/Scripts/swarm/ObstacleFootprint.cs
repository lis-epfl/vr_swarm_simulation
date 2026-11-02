using System.Collections.Generic;
using UnityEngine;

/// <summary>
/// S8 Footprint Avoidance: each obstacle collider's horizontal footprint (its convex hull in world XZ) and the
/// smallest circle round it, built once per collider and cached. OlfatiSaber reads it when
/// SwarmManager.obstacleShape is EnclosingCircle, Footprint or Ellipse; under Cylinder nothing here runs.
///
/// <para>The footprint comes from the collider itself, so every scene works without a layout file: a convex
/// MeshCollider (BlockShapedPatchCityWorld's prisms) gives the hull of its sharedMesh -- the collider's mesh, which
/// static batching leaves alone, unlike the MeshFilter's -- a BoxCollider its eight corners, anything else its bounds.
/// Obstacles are taken to be static, as every building in these scenes is; a tile GoalPatchReplacer swaps in is a
/// new collider and gets its own entry.</para>
///
/// <para>Mirrored by swarm_replica.PolyCity (the polygon, closest()) and swarm_replica.enclosing_circle, which every
/// number in S8 was measured on.</para>
///
/// <para>S10 Ellipse Avoidance adds the footprint's ellipse (GetEllipse): the minimum-area ellipse round it, its
/// aspect capped (SwarmManager.ellipseMaxAspect). The axis and aspect come from the footprint alone and are kept; the
/// ellipse is refitted whenever the cap asked for changes, so the bench's parameter sets, flown one after another in
/// one process, each fly their own cap. Mirrored by swarm_replica.mvee_axis, ellipse_fit and ell_closest.</para>
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

        // S10: the minimum-area ellipse's axis and aspect, fitted on first use, and the ellipse under the cap last
        // asked for.
        public bool mveeDone;
        public Vector2 mveeAxis;
        public double mveeAspect;
        public float ellipseCap = -1.0f;
        public Ellipse ellipse;
    }

    /// <summary>S10: a footprint's avoidance ellipse. circle: its capped aspect is under MinEllipseAspect and it is the
    /// enclosing circle exactly (centre and radius copied, axis +x), which the law flies on the circle's own path.</summary>
    public struct Ellipse
    {
        public Vector2 centre;          // world (x, z)
        public Vector2 axis;            // the major axis, unit
        public float a, b;              // semi-axes, a >= b
        public bool circle;
    }

    public const double MinEllipseAspect = 1.02;
    private const int EllipseIterations = 5;

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

    // ---- S10 Ellipse Avoidance ---------------------------------------------------------------------------------

    /// <summary>The footprint's ellipse under aspect cap <paramref name="cap"/>, refitted only when the cap changes.</summary>
    public static Ellipse GetEllipse(Footprint fp, float cap)
    {
        if (fp.ellipseCap == cap)
            return fp.ellipse;
        if (!fp.mveeDone)
        {
            MveeAxis(fp.polygon, out fp.mveeAxis, out fp.mveeAspect);
            fp.mveeDone = true;
        }
        fp.ellipse = EllipseFit(fp.polygon, cap, fp.mveeAxis, fp.mveeAspect, fp.circleCentre, fp.circleRadius);
        fp.ellipseCap = cap;
        return fp.ellipse;
    }

    /// <summary>
    /// The axis and aspect (>= 1) of the minimum-area ellipse containing the points (a footprint's hull): Khachiyan's
    /// algorithm with Todd and Yildirim's away steps, in doubles, the 3x3 inverse by cofactors. The axis is the
    /// covariance's leading eigenvector in closed form, signed so its x (else z) is positive; aspect 1 and axis +x when
    /// the hull has fewer than 3 points or no preferred direction. Only the axis and aspect are used: EllipseFit makes
    /// the ellipse, so the tolerance never decides whether it encloses the footprint. swarm_replica.mvee_axis is the
    /// same loop in the same order.
    /// </summary>
    public static void MveeAxis(Vector2[] p, out Vector2 axis, out double aspect)
    {
        const double tol = 1e-7;
        const int iters = 4000;
        axis = new Vector2(1.0f, 0.0f);
        aspect = 1.0;
        int k = p.Length;
        if (k < 3)
            return;
        var u = new double[k];
        var M = new double[k];
        for (int i = 0; i < k; i++)
            u[i] = 1.0 / k;
        for (int it = 0; it < iters; it++)
        {
            double sxx = 0, sxz = 0, sx = 0, szz = 0, sz = 0, s1 = 0;
            for (int i = 0; i < k; i++)
            {
                double w = u[i], x = p[i].x, z = p[i].y;
                sxx += w * x * x;
                sxz += w * x * z;
                sx += w * x;
                szz += w * z * z;
                sz += w * z;
                s1 += w;
            }
            double c00 = szz * s1 - sz * sz;
            double c01 = sz * sx - sxz * s1;
            double c02 = sxz * sz - szz * sx;
            double c11 = sxx * s1 - sx * sx;
            double c12 = sxz * sx - sxx * sz;
            double c22 = sxx * szz - sxz * sxz;
            double det = sxx * c00 + sxz * c01 + sx * c02;
            if (System.Math.Abs(det) < 1e-300)
                break;
            for (int i = 0; i < k; i++)
            {
                double x = p[i].x, z = p[i].y;
                M[i] = (c00 * x * x + c11 * z * z + c22 + 2.0 * (c01 * x * z + c02 * x + c12 * z)) / det;
            }
            int j = 0;
            for (int i = 1; i < k; i++)
                if (M[i] > M[j]) j = i;
            int away = -1;
            for (int i = 0; i < k; i++)
                if (u[i] > 1e-12 && (away < 0 || M[i] < M[away])) away = i;
            double up = (M[j] - 3.0) / 3.0;
            double down = (3.0 - M[away]) / 3.0;
            if (System.Math.Max(up, down) < tol)
                break;
            if (up >= down)
            {
                double s = (M[j] - 3.0) / (3.0 * (M[j] - 1.0));
                for (int i = 0; i < k; i++) u[i] *= 1.0 - s;
                u[j] += s;
            }
            else
            {
                if (M[away] <= 1.0 || u[away] >= 1.0)
                    break;
                double s = System.Math.Min((3.0 - M[away]) / (3.0 * (M[away] - 1.0)), u[away] / (1.0 - u[away]));
                for (int i = 0; i < k; i++) u[i] *= 1.0 + s;
                u[away] -= s;
            }
        }
        double cx = 0, cz = 0;
        for (int i = 0; i < k; i++)
        {
            cx += u[i] * p[i].x;
            cz += u[i] * p[i].y;
        }
        double A = 0, B = 0, D = 0;
        for (int i = 0; i < k; i++)
        {
            double dx = p[i].x - cx, dz = p[i].y - cz;
            A += u[i] * dx * dx;
            B += u[i] * dx * dz;
            D += u[i] * dz * dz;
        }
        double half = 0.5 * (A - D);
        double r = System.Math.Sqrt(half * half + B * B);
        double lmax = 0.5 * (A + D) + r, lmin = 0.5 * (A + D) - r;
        if (lmin <= 0.0 || r <= 1e-12 * System.Math.Max(A + D, 1e-300))
            return;
        double vx, vz;
        if (half >= 0.0) { vx = half + r; vz = B; }
        else { vx = B; vz = r - half; }
        double n = System.Math.Sqrt(vx * vx + vz * vz);
        vx /= n;
        vz /= n;
        if (vx < 0.0 || (vx == 0.0 && vz < 0.0)) { vx = -vx; vz = -vz; }
        axis = new Vector2((float)vx, (float)vz);
        aspect = System.Math.Sqrt(lmax / lmin);
    }

    /// <summary>
    /// The ellipse a hull is avoided as under aspect cap <paramref name="cap"/>: squash it along the axis by
    /// k = min(cap, aspect), take the smallest circle round the squashed hull (EnclosingCircle) and stretch it back --
    /// an ellipse of aspect k containing the footprint exactly, the minimum-area one while the footprint's own aspect
    /// is within the cap. Under MinEllipseAspect it is the enclosing circle itself. Worked relative to the circle's
    /// centre, so the squashed points stay small in single precision. swarm_replica.ellipse_fit.
    /// </summary>
    public static Ellipse EllipseFit(Vector2[] hull, float cap, Vector2 axis, double aspect, Vector2 circleCentre,
                                     float circleRadius)
    {
        double k = System.Math.Min((double)cap, aspect);
        if (k < MinEllipseAspect)
            return new Ellipse { centre = circleCentre, axis = new Vector2(1.0f, 0.0f), a = circleRadius,
                                 b = circleRadius, circle = true };
        double ex = axis.x, ez = axis.y;
        var sq = new Vector2[hull.Length];
        for (int i = 0; i < hull.Length; i++)
        {
            double dx = hull[i].x - circleCentre.x, dz = hull[i].y - circleCentre.y;
            sq[i] = new Vector2((float)((dx * ex + dz * ez) / k), (float)(-dx * ez + dz * ex));
        }
        EnclosingCircle(sq, out Vector2 cs, out float r);
        double sx = cs.x * k, sy = cs.y;
        var centre = new Vector2((float)(circleCentre.x + sx * ex - sy * ez), (float)(circleCentre.y + sx * ez + sy * ex));
        return new Ellipse { centre = centre, axis = axis, a = (float)(k * r), b = r, circle = false };
    }

    /// <summary>
    /// The closest point (a tx, b ty) of the ellipse x^2/a^2 + y^2/b^2 = 1 to (X, Y) >= 0, inside or out, with
    /// tx^2 + ty^2 = 1: a fixed number of trig-free steps (a point of the ellipse and its centre of curvature define
    /// the circle that osculates it there; step along that circle to where the line to (X, Y) meets it), ~1e-6 m up
    /// to aspect 3.6. swarm_replica.ell_closest.
    /// </summary>
    public static void ClosestOnEllipse(double X, double Y, double a, double b, out double tx, out double ty)
    {
        tx = 0.7071067811865476;
        ty = 0.7071067811865476;
        for (int i = 0; i < EllipseIterations; i++)
        {
            double ex = (a * a - b * b) * tx * tx * tx / a;
            double ey = (b * b - a * a) * ty * ty * ty / b;
            double rx = a * tx - ex, ry = b * ty - ey;
            double qx = X - ex, qy = Y - ey;
            double r = System.Math.Sqrt(rx * rx + ry * ry);
            double q = System.Math.Max(System.Math.Sqrt(qx * qx + qy * qy), 1e-12);
            tx = System.Math.Min(System.Math.Max((qx * r / q + ex) / a, 0.0), 1.0);
            ty = System.Math.Min(System.Math.Max((qy * r / q + ey) / b, 0.0), 1.0);
            double t = System.Math.Max(System.Math.Sqrt(tx * tx + ty * ty), 1e-12);
            tx /= t;
            ty /= t;
        }
    }

    /// <summary>A point's signed distance to the ellipse (negative inside), in metres, horizontally.</summary>
    public static float EllipseSignedDistance(Ellipse el, Vector2 point)
    {
        double dx = point.x - el.centre.x, dz = point.y - el.centre.y;
        double x = dx * el.axis.x + dz * el.axis.y, y = -dx * el.axis.y + dz * el.axis.x;
        double X = System.Math.Abs(x), Y = System.Math.Abs(y);
        ClosestOnEllipse(X, Y, el.a, el.b, out double tx, out double ty);
        double d = System.Math.Sqrt((X - el.a * tx) * (X - el.a * tx) + (Y - el.b * ty) * (Y - el.b * ty));
        return (float)((x / el.a) * (x / el.a) + (y / el.b) * (y / el.b) < 1.0 ? -d : d);
    }

    /// <summary>A horizontal point in the ellipse's unit space, where the ellipse is the unit circle.</summary>
    public static Vector2 EllipseUnit(Ellipse el, Vector2 point)
    {
        float dx = point.x - el.centre.x, dz = point.y - el.centre.y;
        return new Vector2((dx * el.axis.x + dz * el.axis.y) / el.a, (-dx * el.axis.y + dz * el.axis.x) / el.b);
    }

    /// <summary>The distance from the origin to the segment a -> b (the unit circle's centre, in unit space).</summary>
    public static float SegmentOriginDistance(Vector2 a, Vector2 b)
    {
        Vector2 ab = b - a;
        float l2 = ab.sqrMagnitude;
        float t = l2 > 1e-18f ? Mathf.Clamp01(-Vector2.Dot(a, ab) / l2) : 0.0f;
        return (a + t * ab).magnitude;
    }

    /// <summary>The ellipse's half-width across direction w (unit, horizontal): its support along w.</summary>
    public static float EllipseHalfWidth(Ellipse el, Vector2 w)
    {
        float along = w.x * el.axis.x + w.y * el.axis.y, across = -w.x * el.axis.y + w.y * el.axis.x;
        return Mathf.Sqrt(el.a * along * el.a * along + el.b * across * el.b * across);
    }
}
