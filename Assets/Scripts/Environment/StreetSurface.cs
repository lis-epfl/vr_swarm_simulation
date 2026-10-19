using System.Collections.Generic;
using UnityEngine;

/// <summary>
/// A generated layout's streets as road (<see cref="ProceduralCityBuilder.StreetStyle.Textured"/>): one mesh, two
/// submeshes. Submesh 0 is the straight stretches, mapped onto the lane-marked road texture the way ParkRoad's tiles are
/// (MC_Asphalt_Road Texture_M_01: dashes along V, <see cref="Settings.tilesAcross"/> square tiles across a street).
/// Submesh 1 is the junctions and their zebra crossings, both cut from MC_Asphalt_Cross and Corner: a crossing maps its
/// zebra strip (<see cref="ZebraUV"/>), a junction its plain asphalt (<see cref="PlainUV"/>), so no lane marking ever runs
/// through a junction.
///
/// <para><b>Each street stops short of its junction</b> by the setback at which its kerb meets its angular neighbour's,
/// (w/2) / tan(θ/2) for the angle θ between them, the larger of its two neighbours' (capped at
/// <see cref="Settings.maxSetback"/> widths for very acute angles, where the strips then overlap a little). A junction is
/// then the polygon through, in angular order, each street's two end corners and the kerb corner between neighbours, fanned
/// from the junction's point — exact at every angle, with no overlap. At a junction of three or more streets each street
/// ends in a crossing before it.</para>
///
/// <para><b>Voronoi diagrams have very short edges</b> (two vertices a few metres apart). A street with less than
/// <see cref="Settings.minStrip"/> left between its setbacks and crossings is not drawn: its two junctions merge into one,
/// drawn as the convex hull of everything meeting there (the merged street's own rectangle included).</para>
/// </summary>
public static class StreetSurface
{
    public struct Settings
    {
        public float height;           // m above y = 0
        public int tilesAcross;        // road texture tiles across a street
        public float crossingDepth;    // m along the street (3.3 m keeps the zebra strip's texels square at 15 m)
        public float maxSetback;       // in street widths
        public float minStrip;         // m: shorter streets merge their junctions
    }

    /// <summary>The zebra strip of MC_Asphalt_Cross and Corner (u across the street, v along it), measured off the
    /// 2048 px texture: 17 stripes between u 0.0125 and 0.4375, v 0.906 to 0.998.</summary>
    public static readonly Rect ZebraUV = Rect.MinMaxRect(0.0125f, 0.906f, 0.4375f, 0.998f);
    /// <summary>Its plain asphalt (the rest of the content: u, v 0.02 to 0.88; u past 0.9 is black).</summary>
    public static readonly Rect PlainUV = Rect.MinMaxRect(0.02f, 0.02f, 0.88f, 0.88f);
    private const float JunctionMetresPerUV = 60f;    // plain asphalt at 34 px/m, unless a junction is bigger

    private class Edge
    {
        public int a, b;
        public float width;
        public bool absorbed;
        public float[] setback = new float[2];     // at a, at b
        public bool[] crossing = new bool[2];
    }

    private struct Incident
    {
        public int edge, end;                      // end 0: the node is the edge's a; 1: its b
        public Vector2 dir;                        // unit, away from the node
        public float angle;
    }

    public static Mesh Build(IEnumerable<(Vector2 a, Vector2 b, float width)> segments, Settings s)
    {
        // ---- the graph: nodes keyed on position (Voronoi vertices are shared exactly; 1 cm grid)
        var nodes = new List<Vector2>();
        var key = new Dictionary<(long, long), int>();
        int Node(Vector2 p)
        {
            var k = ((long)Mathf.Round(p.x * 100f), (long)Mathf.Round(p.y * 100f));
            if (!key.TryGetValue(k, out int i))
            {
                i = nodes.Count;
                nodes.Add(p);
                key[k] = i;
            }
            return i;
        }
        var edges = new List<Edge>();
        foreach (var (pa, pb, w) in segments)
        {
            if ((pb - pa).sqrMagnitude < 1e-6f) continue;
            edges.Add(new Edge { a = Node(pa), b = Node(pb), width = w });
        }
        var inc = new List<Incident>[nodes.Count];
        for (int i = 0; i < nodes.Count; i++) inc[i] = new List<Incident>();
        for (int e = 0; e < edges.Count; e++)
        {
            Vector2 d = (nodes[edges[e].b] - nodes[edges[e].a]).normalized;
            inc[edges[e].a].Add(new Incident { edge = e, end = 0, dir = d, angle = Mathf.Atan2(d.y, d.x) });
            inc[edges[e].b].Add(new Incident { edge = e, end = 1, dir = -d, angle = Mathf.Atan2(-d.y, -d.x) });
        }

        // ---- setbacks and crossings
        for (int n = 0; n < nodes.Count; n++)
        {
            List<Incident> list = inc[n];
            list.Sort((x, y) => x.angle.CompareTo(y.angle));
            int deg = list.Count;
            for (int k = 0; k < deg; k++)
            {
                Edge e = edges[list[k].edge];
                float t = 0f;
                if (deg >= 2)
                {
                    foreach (int j in new[] { (k + deg - 1) % deg, (k + 1) % deg })
                    {
                        float theta = Mathf.Abs(Mathf.DeltaAngle(list[k].angle * Mathf.Rad2Deg, list[j].angle * Mathf.Rad2Deg))
                                      * Mathf.Deg2Rad;
                        float half = 0.5f * Mathf.Max(e.width, edges[list[j].edge].width);
                        if (theta < Mathf.PI - 1e-3f) t = Mathf.Max(t, half / Mathf.Tan(Mathf.Max(theta, 1e-3f) / 2f));
                    }
                }
                e.setback[list[k].end] = Mathf.Min(t, s.maxSetback * e.width);
                e.crossing[list[k].end] = deg >= 3;
            }
        }

        // ---- streets too short to draw merge their junctions
        var parent = new int[nodes.Count];
        for (int i = 0; i < parent.Length; i++) parent[i] = i;
        int Find(int i)
        {
            while (parent[i] != i) i = parent[i] = parent[parent[i]];
            return i;
        }
        foreach (Edge e in edges)
        {
            float len = (nodes[e.b] - nodes[e.a]).magnitude;
            float used = e.setback[0] + e.setback[1] + (e.crossing[0] ? s.crossingDepth : 0f) +
                         (e.crossing[1] ? s.crossingDepth : 0f);
            if (len - used < s.minStrip)
            {
                e.absorbed = true;
                parent[Find(e.a)] = Find(e.b);
            }
        }
        foreach (Edge e in edges)
        {
            if (Find(e.a) == Find(e.b)) e.absorbed = true;     // both ends in one merged junction
        }

        var verts = new List<Vector3>();
        var uvs = new List<Vector2>();
        var road = new List<int>();
        var junction = new List<int>();

        // ---- the straight stretches and their crossings
        foreach (Edge e in edges)
        {
            if (e.absorbed) continue;
            Vector2 a = nodes[e.a], b = nodes[e.b];
            Vector2 d = (b - a).normalized, left = new Vector2(-d.y, d.x) * (0.5f * e.width);
            float ca = e.crossing[0] ? s.crossingDepth : 0f, cb = e.crossing[1] ? s.crossingDepth : 0f;
            Vector2 p0 = a + d * (e.setback[0] + ca), p1 = b - d * (e.setback[1] + cb);
            float tile = e.width / Mathf.Max(s.tilesAcross, 1);
            float v1 = (p1 - p0).magnitude / tile;
            Quad(verts, uvs, road, s.height, p0 - left, p0 + left, p1 + left, p1 - left,
                 new Vector2(0f, 0f), new Vector2(s.tilesAcross, 0f), new Vector2(s.tilesAcross, v1), new Vector2(0f, v1));
            if (e.crossing[0]) Crossing(verts, uvs, junction, s.height, a + d * e.setback[0], d, left, s.crossingDepth);
            if (e.crossing[1]) Crossing(verts, uvs, junction, s.height, b - d * e.setback[1], -d, -left, s.crossingDepth);
        }

        // ---- junctions
        var members = new Dictionary<int, List<int>>();
        for (int n = 0; n < nodes.Count; n++)
        {
            int r = Find(n);
            if (!members.TryGetValue(r, out List<int> m)) members[r] = m = new List<int>();
            m.Add(n);
        }
        foreach (List<int> group in members.Values)
        {
            if (group.Count == 1)
            {
                Star(verts, uvs, junction, s.height, nodes[group[0]], inc[group[0]], edges);
            }
            else
            {
                Merged(verts, uvs, junction, s.height, group, nodes, inc, edges);
            }
        }

        var mesh = new Mesh { name = "Procedural roads", indexFormat = UnityEngine.Rendering.IndexFormat.UInt32 };
        mesh.SetVertices(verts);
        mesh.SetUVs(0, uvs);
        mesh.subMeshCount = 2;
        mesh.SetTriangles(road, 0);
        mesh.SetTriangles(junction, 1);
        var normals = new Vector3[verts.Count];
        for (int i = 0; i < normals.Length; i++) normals[i] = Vector3.up;
        mesh.normals = normals;
        mesh.RecalculateTangents();
        mesh.RecalculateBounds();
        return mesh;
    }

    /// <summary>One junction: its streets' end corners and the kerb corners between them, in angular order, fanned
    /// from the junction's point (star-shaped from it by construction).</summary>
    private static void Star(List<Vector3> verts, List<Vector2> uvs, List<int> tris, float y, Vector2 c,
                             List<Incident> list, List<Edge> edges)
    {
        int deg = list.Count;
        if (deg == 0) return;
        var ring = new List<Vector2>();
        for (int k = 0; k < deg; k++)
        {
            Incident i = list[k];
            Edge e = edges[i.edge];
            Vector2 left = new Vector2(-i.dir.y, i.dir.x) * (0.5f * e.width);
            Vector2 end = c + i.dir * e.setback[i.end];
            ring.Add(end - left);           // the corner clockwise of the street...
            ring.Add(end + left);           // ...then counter-clockwise
            if (deg < 2) continue;
            // The kerb corner towards the next street counter-clockwise: on the inside of a gap under 180 degrees, if it
            // lies short of both setbacks (else the two end corners already meet there); on the outside of a gap over 180
            // (a bend, e.g. the arena's corners), always, behind the junction's point. A straight run needs none.
            Incident j = list[(k + 1) % deg];
            Edge f = edges[j.edge];
            float gap = Mathf.Repeat(j.angle - i.angle, 2f * Mathf.PI);
            if (gap < 1e-4f) gap = 2f * Mathf.PI;
            Vector2 rightJ = -new Vector2(-j.dir.y, j.dir.x) * (0.5f * f.width);
            if (Mathf.Abs(gap - Mathf.PI) < 1e-3f ||
                !LineCross(c + left, i.dir, c + rightJ, j.dir, out float ti, out float tj)) continue;
            bool inner = gap < Mathf.PI && ti > 1e-3f && tj > 1e-3f &&
                         ti < e.setback[i.end] - 1e-3f && tj < f.setback[j.end] - 1e-3f;
            bool outer = gap > Mathf.PI && ti < 0f && tj < 0f;
            if (inner || outer) ring.Add(c + left + i.dir * ti);
        }
        Fan(verts, uvs, tris, y, c, ring);
    }

    /// <summary>A merged junction: the convex hull of its nodes, the rectangles of the streets merged into it and the
    /// end corners of the streets leaving it.</summary>
    private static void Merged(List<Vector3> verts, List<Vector2> uvs, List<int> tris, float y, List<int> group,
                               List<Vector2> nodes, List<Incident>[] inc, List<Edge> edges)
    {
        var pts = new List<Vector2>();
        foreach (int n in group)
        {
            pts.Add(nodes[n]);
            foreach (Incident i in inc[n])
            {
                Edge e = edges[i.edge];
                Vector2 left = new Vector2(-i.dir.y, i.dir.x) * (0.5f * e.width);
                float reach = e.absorbed ? (nodes[e.b] - nodes[e.a]).magnitude : e.setback[i.end];
                pts.Add(nodes[n] + left);
                pts.Add(nodes[n] - left);
                pts.Add(nodes[n] + i.dir * reach + left);
                pts.Add(nodes[n] + i.dir * reach - left);
            }
        }
        List<Vector2> hull = Hull(pts);
        Vector2 c = Vector2.zero;
        foreach (Vector2 p in hull) c += p;
        Fan(verts, uvs, tris, y, c / hull.Count, hull);
    }

    /// <summary>A crossing across a street: depth along <paramref name="d"/> from <paramref name="start"/>, the zebra
    /// strip's stripes running along the street.</summary>
    private static void Crossing(List<Vector3> verts, List<Vector2> uvs, List<int> tris, float y, Vector2 start,
                                 Vector2 d, Vector2 left, float depth)
    {
        Vector2 end = start + d * depth;
        Quad(verts, uvs, tris, y, start - left, start + left, end + left, end - left,
             new Vector2(ZebraUV.xMin, ZebraUV.yMin), new Vector2(ZebraUV.xMax, ZebraUV.yMin),
             new Vector2(ZebraUV.xMax, ZebraUV.yMax), new Vector2(ZebraUV.xMin, ZebraUV.yMax));
    }

    private static void Fan(List<Vector3> verts, List<Vector2> uvs, List<int> tris, float y, Vector2 c, List<Vector2> ring)
    {
        if (ring.Count < 2) return;
        float radius = 0f;
        foreach (Vector2 p in ring) radius = Mathf.Max(radius, (p - c).magnitude);
        float k = Mathf.Min(1f / JunctionMetresPerUV, 0.5f * PlainUV.width / Mathf.Max(radius, 1e-3f));
        Vector2 Uv(Vector2 p) => PlainUV.center + (p - c) * k;
        int i0 = verts.Count;
        verts.Add(new Vector3(c.x, y, c.y));
        uvs.Add(Uv(c));
        foreach (Vector2 p in ring)
        {
            verts.Add(new Vector3(p.x, y, p.y));
            uvs.Add(Uv(p));
        }
        for (int k2 = 0; k2 < ring.Count; k2++)
        {
            Tri(verts, tris, i0, i0 + 1 + k2, i0 + 1 + (k2 + 1) % ring.Count);
        }
    }

    private static void Quad(List<Vector3> verts, List<Vector2> uvs, List<int> tris, float y,
                             Vector2 p0, Vector2 p1, Vector2 p2, Vector2 p3, Vector2 t0, Vector2 t1, Vector2 t2, Vector2 t3)
    {
        int i = verts.Count;
        verts.Add(new Vector3(p0.x, y, p0.y));
        verts.Add(new Vector3(p1.x, y, p1.y));
        verts.Add(new Vector3(p2.x, y, p2.y));
        verts.Add(new Vector3(p3.x, y, p3.y));
        uvs.Add(t0);
        uvs.Add(t1);
        uvs.Add(t2);
        uvs.Add(t3);
        Tri(verts, tris, i, i + 1, i + 2);
        Tri(verts, tris, i, i + 2, i + 3);
    }

    /// <summary>A triangle wound to face up (Unity: front faces wind clockwise seen from above).</summary>
    private static void Tri(List<Vector3> verts, List<int> tris, int a, int b, int c)
    {
        Vector3 n = Vector3.Cross(verts[b] - verts[a], verts[c] - verts[a]);
        if (n.sqrMagnitude < 1e-10f) return;                     // degenerate (coincident corners)
        tris.Add(a);
        if (n.y > 0f)
        {
            tris.Add(b);
            tris.Add(c);
        }
        else
        {
            tris.Add(c);
            tris.Add(b);
        }
    }

    /// <summary>Where p + t q crosses r + u s (false if parallel).</summary>
    private static bool LineCross(Vector2 p, Vector2 q, Vector2 r, Vector2 s, out float t, out float u)
    {
        float den = q.x * s.y - q.y * s.x;
        t = u = 0f;
        if (Mathf.Abs(den) < 1e-6f) return false;
        Vector2 w = r - p;
        t = (w.x * s.y - w.y * s.x) / den;
        u = (w.x * q.y - w.y * q.x) / den;
        return true;
    }

    /// <summary>Andrew's monotone chain, counter-clockwise.</summary>
    private static List<Vector2> Hull(List<Vector2> pts)
    {
        pts.Sort((a, b) => a.x != b.x ? a.x.CompareTo(b.x) : a.y.CompareTo(b.y));
        var h = new List<Vector2>();
        float Cross(Vector2 o, Vector2 a, Vector2 b) => (a.x - o.x) * (b.y - o.y) - (a.y - o.y) * (b.x - o.x);
        for (int pass = 0; pass < 2; pass++)
        {
            int start = h.Count;
            for (int i = 0; i < pts.Count; i++)
            {
                Vector2 p = pts[pass == 0 ? i : pts.Count - 1 - i];
                while (h.Count >= start + 2 && Cross(h[h.Count - 2], h[h.Count - 1], p) <= 0f) h.RemoveAt(h.Count - 1);
                h.Add(p);
            }
            h.RemoveAt(h.Count - 1);
        }
        return h;
    }
}
