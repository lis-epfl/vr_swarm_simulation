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
///
/// <para><b>No point of the road is drawn twice</b> (<see cref="Lay"/>). Every piece lies at one height, so two that
/// overlap z-fight, and the overlap is not rare: a merged junction's hull spreads over the crossings of the streets
/// leaving it, and streets meeting at an acute angle overlap past <see cref="Settings.maxSetback"/>. The first road of
/// BlockShapedPatchCityWorld drew 9,300 m² twice, at nearly every merged junction, and the zebra stripes flickered
/// through the asphalt there. So each piece is cut down to the ground no earlier piece covers, crossings first, then the
/// stretches, then the junctions, which only fill what the streets leave: a street keeps its whole crossing however far
/// its junction's hull reaches. The cut changes what is drawn, not the analysis: <see cref="Stretches"/> is unchanged.</para>
/// </summary>
public static class StreetSurface
{
    /// <summary>Stamped on the mesh's name, so a road baked into a scene by an older version is known and redrawn
    /// (<see cref="ProceduralCityBuilder"/>). 2: no piece overlaps another.</summary>
    public const int Version = 2;
    public static readonly string MeshName = $"Procedural roads v{Version}";

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
    private const float Eps = 1e-3f;                  // m: pieces this close to a line are on it (they only touch)
    private const float MinArea = 1e-4f;              // m²: a smaller piece is not drawn
    private const float Cell = 16f;                   // m: the grid Lay finds earlier pieces by

    /// <summary>One convex piece of road (counter-clockwise in (x, z)), mapped affinely: a point p has
    /// uv = t0 + ((p - p0)·su) du + ((p - p0)·sr) dr, so a piece cut down keeps its texture where it was.</summary>
    private sealed class Piece
    {
        public List<Vector2> poly;
        public int sub;
        public Vector2 p0, su, sr, t0, du, dr;
        public Rect bounds;

        public Vector2 Uv(Vector2 p)
        {
            Vector2 q = p - p0;
            return t0 + Vector2.Dot(q, su) * du + Vector2.Dot(q, sr) * dr;
        }
    }

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

    /// <summary>One street as the road draws it (<see cref="Stretches"/>): the plain road runs from <see cref="start"/> to
    /// <see cref="end"/> metres along a -> b, past the setback and any crossing at each end; what lies outside that is
    /// junction or crossing.</summary>
    public struct Stretch
    {
        public Vector2 a, b;
        public float width;
        public bool absorbed;          // not drawn: too short (merged into one junction), or of zero length
        public float start, end;
        public bool crossingA, crossingB;
        /// <summary>The junction at each end, after short streets have merged theirs: two ends with one id meet at one
        /// junction. -1 for a segment of zero length.</summary>
        public int junctionA, junctionB;

        public float Length => (b - a).magnitude;
        public Vector2 Dir => (b - a).normalized;
        /// <summary>Unit, to the left of a -> b seen from above (the +side).</summary>
        public Vector2 Left => new Vector2(-Dir.y, Dir.x);
        public Vector2 At(float along, float across) => a + Dir * along + Left * across;
    }

    private sealed class Graph
    {
        public readonly List<Vector2> nodes = new List<Vector2>();
        public readonly List<Edge> edges = new List<Edge>();
        public readonly List<(Vector2 a, Vector2 b, float width)> segments = new List<(Vector2, Vector2, float)>();
        public readonly List<int> edgeOf = new List<int>();      // per segment: its edge, or -1 (zero length)
        public List<Incident>[] inc;
        public int[] parent;

        public int Find(int i)
        {
            while (parent[i] != i) i = parent[i] = parent[parent[i]];
            return i;
        }
    }

    /// <summary>Every segment as the road draws it, in the order given: the same setbacks, crossings and merged junctions
    /// as <see cref="Build"/>, so whatever stands beside a street (CityDressing) keeps off its junctions and crossings.</summary>
    public static List<Stretch> Stretches(IEnumerable<(Vector2 a, Vector2 b, float width)> segments, Settings s)
    {
        Graph g = Analyse(segments, s);
        var list = new List<Stretch>();
        for (int k = 0; k < g.segments.Count; k++)
        {
            var (pa, pb, w) = g.segments[k];
            var st = new Stretch { a = pa, b = pb, width = w, absorbed = true, junctionA = -1, junctionB = -1 };
            if (g.edgeOf[k] >= 0)
            {
                Edge e = g.edges[g.edgeOf[k]];
                st.absorbed = e.absorbed;
                st.crossingA = e.crossing[0];
                st.crossingB = e.crossing[1];
                st.junctionA = g.Find(e.a);
                st.junctionB = g.Find(e.b);
                st.start = e.setback[0] + (e.crossing[0] ? s.crossingDepth : 0f);
                st.end = (pb - pa).magnitude - e.setback[1] - (e.crossing[1] ? s.crossingDepth : 0f);
            }
            list.Add(st);
        }
        return list;
    }

    public static Mesh Build(IEnumerable<(Vector2 a, Vector2 b, float width)> segments, Settings s)
    {
        var verts = new List<Vector3>();
        var uvs = new List<Vector2>();
        var road = new List<int>();
        var junction = new List<int>();
        Draw(segments, s, verts, uvs, road, junction);

        var mesh = new Mesh { name = MeshName, indexFormat = UnityEngine.Rendering.IndexFormat.UInt32 };
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

    /// <summary>The road's vertices, uvs and two submeshes' triangles (<see cref="Build"/> without the Mesh).</summary>
    private static void Draw(IEnumerable<(Vector2 a, Vector2 b, float width)> segments, Settings s, List<Vector3> verts,
                             List<Vector2> uvs, List<int> road, List<int> junction)
    {
        Graph g = Analyse(segments, s);
        List<Vector2> nodes = g.nodes;
        List<Edge> edges = g.edges;
        List<Incident>[] inc = g.inc;
        int Find(int i) => g.Find(i);

        var crossings = new List<Piece>();
        var stretches = new List<Piece>();
        var junctions = new List<Piece>();

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
            Quad(stretches, 0, p0 - left, p0 + left, p1 + left, p1 - left,
                 new Vector2(0f, 0f), new Vector2(s.tilesAcross, 0f), new Vector2(0f, v1));
            if (e.crossing[0]) Crossing(crossings, a + d * e.setback[0], d, left, s.crossingDepth);
            if (e.crossing[1]) Crossing(crossings, b - d * e.setback[1], -d, -left, s.crossingDepth);
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
                Star(junctions, nodes[group[0]], inc[group[0]], edges);
            }
            else
            {
                Merged(junctions, group, nodes, inc, edges);
            }
        }

        var pieces = new List<Piece>(crossings);
        pieces.AddRange(stretches);
        pieces.AddRange(junctions);
        Lay(pieces, s.height, verts, uvs, new[] { road, junction });
    }

    /// <summary>Writes <paramref name="pieces"/> into the mesh in order, each cut down to the ground no earlier piece
    /// covers, so no point is drawn twice (see the class summary). A piece that only touches an earlier one along an
    /// edge is written whole.</summary>
    private static void Lay(List<Piece> pieces, float y, List<Vector3> verts, List<Vector2> uvs, List<int>[] tris)
    {
        var grid = new Dictionary<(int, int), List<int>>();
        var seen = new HashSet<int>();
        var near = new List<int>();
        for (int i = 0; i < pieces.Count; i++)
        {
            Piece p = pieces[i];
            int x0 = Mathf.FloorToInt(p.bounds.xMin / Cell), x1 = Mathf.FloorToInt(p.bounds.xMax / Cell);
            int z0 = Mathf.FloorToInt(p.bounds.yMin / Cell), z1 = Mathf.FloorToInt(p.bounds.yMax / Cell);
            seen.Clear();
            near.Clear();
            for (int x = x0; x <= x1; x++)
            {
                for (int z = z0; z <= z1; z++)
                {
                    if (!grid.TryGetValue((x, z), out List<int> list)) continue;
                    foreach (int j in list)
                    {
                        if (seen.Add(j) && p.bounds.Overlaps(pieces[j].bounds)) near.Add(j);
                    }
                }
            }
            near.Sort();
            var parts = new List<List<Vector2>> { p.poly };
            foreach (int j in near)
            {
                var rest = new List<List<Vector2>>();
                foreach (List<Vector2> part in parts) Subtract(part, pieces[j].poly, rest);
                parts = rest;
                if (parts.Count == 0) break;
            }
            foreach (List<Vector2> part in parts)
            {
                int i0 = verts.Count;
                foreach (Vector2 q in part)
                {
                    verts.Add(new Vector3(q.x, y, q.y));
                    uvs.Add(p.Uv(q));
                }
                for (int k = 1; k + 1 < part.Count; k++) Tri(verts, tris[p.sub], i0, i0 + k, i0 + k + 1);
            }
            for (int x = x0; x <= x1; x++)
            {
                for (int z = z0; z <= z1; z++)
                {
                    if (!grid.TryGetValue((x, z), out List<int> list)) grid[(x, z)] = list = new List<int>();
                    list.Add(i);
                }
            }
        }
    }

    /// <summary>Adds to <paramref name="into"/> the part of convex <paramref name="a"/> outside convex
    /// <paramref name="b"/>, as convex pieces: <paramref name="a"/> itself when the two only touch.</summary>
    private static void Subtract(List<Vector2> a, List<Vector2> b, List<List<Vector2>> into)
    {
        if (Apart(a, b) || Apart(b, a))
        {
            into.Add(a);
            return;
        }
        // Outside b means outside one of its edges: peel off what lies beyond each edge in turn, and keep the rest.
        List<Vector2> rest = a;
        for (int k = 0; k < b.Count; k++)
        {
            Vector2 p = b[k], q = b[(k + 1) % b.Count];
            List<Vector2> beyond = Clip(rest, p, q, -1f);
            if (Area(beyond) > MinArea) into.Add(beyond);
            rest = Clip(rest, p, q, 1f);
            if (Area(rest) <= MinArea) return;
        }
        // what is left lies inside b, which is drawn already
    }

    /// <summary>Whether all of <paramref name="a"/> lies outside one edge of <paramref name="b"/> (to within
    /// <see cref="Eps"/>): a separating line, so the two do not overlap.</summary>
    private static bool Apart(List<Vector2> a, List<Vector2> b)
    {
        for (int k = 0; k < b.Count; k++)
        {
            Vector2 p = b[k], q = b[(k + 1) % b.Count];
            bool outside = true;
            foreach (Vector2 v in a)
            {
                if (Side(p, q, v) > Eps)
                {
                    outside = false;
                    break;
                }
            }
            if (outside) return true;
        }
        return false;
    }

    /// <summary>The part of convex <paramref name="poly"/> left of the line p -> q (<paramref name="keep"/> 1) or right of
    /// it (-1). Corners within <see cref="Eps"/> of the line count as on it.</summary>
    private static List<Vector2> Clip(List<Vector2> poly, Vector2 p, Vector2 q, float keep)
    {
        var o = new List<Vector2>(poly.Count + 1);
        for (int k = 0; k < poly.Count; k++)
        {
            Vector2 u = poly[k], v = poly[(k + 1) % poly.Count];
            float du = keep * Side(p, q, u), dv = keep * Side(p, q, v);
            if (Mathf.Abs(du) < Eps) du = 0f;
            if (Mathf.Abs(dv) < Eps) dv = 0f;
            if (du >= 0f) o.Add(u);
            if ((du > 0f && dv < 0f) || (du < 0f && dv > 0f)) o.Add(u + (v - u) * (du / (du - dv)));
        }
        return o;
    }

    /// <summary>The signed distance of <paramref name="v"/> from the line p -> q, positive on its left.</summary>
    private static float Side(Vector2 p, Vector2 q, Vector2 v)
    {
        Vector2 e = q - p;
        return (e.x * (v.y - p.y) - e.y * (v.x - p.x)) / e.magnitude;
    }

    /// <summary>The polygon's area, positive counter-clockwise.</summary>
    private static float Area(List<Vector2> poly)
    {
        float a = 0f;
        for (int k = 0; k < poly.Count; k++)
        {
            Vector2 u = poly[k], v = poly[(k + 1) % poly.Count];
            a += u.x * v.y - v.x * u.y;
        }
        return 0.5f * a;
    }

    /// <summary>A convex piece from its corners in either order, mapped as given, or nothing if it has no area. Corners a
    /// hair apart are merged, so every edge has a direction.</summary>
    private static void AddPiece(List<Piece> into, int sub, List<Vector2> corners, Vector2 p0, Vector2 su, Vector2 sr,
                            Vector2 t0, Vector2 du, Vector2 dr)
    {
        var poly = new List<Vector2>(corners.Count);
        foreach (Vector2 c in corners)
        {
            if (poly.Count == 0 || (c - poly[poly.Count - 1]).sqrMagnitude > 1e-8f) poly.Add(c);
        }
        while (poly.Count > 1 && (poly[0] - poly[poly.Count - 1]).sqrMagnitude <= 1e-8f) poly.RemoveAt(poly.Count - 1);
        if (poly.Count < 3) return;
        float area = Area(poly);
        if (Mathf.Abs(area) <= MinArea) return;
        if (area < 0f) poly.Reverse();
        Vector2 lo = poly[0], hi = poly[0];
        foreach (Vector2 c in poly)
        {
            lo = Vector2.Min(lo, c);
            hi = Vector2.Max(hi, c);
        }
        into.Add(new Piece
        {
            poly = poly, sub = sub, p0 = p0, su = su, sr = sr, t0 = t0, du = du, dr = dr,
            bounds = Rect.MinMaxRect(lo.x, lo.y, hi.x, hi.y),
        });
    }

    /// <summary>The street graph with every street's setbacks and crossings, and the streets too short to draw merged
    /// into one junction (<see cref="Graph.Find"/> names it).</summary>
    private static Graph Analyse(IEnumerable<(Vector2 a, Vector2 b, float width)> segments, Settings s)
    {
        // ---- the graph: nodes keyed on position (Voronoi vertices are shared exactly; 1 cm grid)
        var g = new Graph();
        List<Vector2> nodes = g.nodes;
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
        List<Edge> edges = g.edges;
        foreach (var (pa, pb, w) in segments)
        {
            g.segments.Add((pa, pb, w));
            if ((pb - pa).sqrMagnitude < 1e-6f)
            {
                g.edgeOf.Add(-1);
                continue;
            }
            g.edgeOf.Add(edges.Count);
            edges.Add(new Edge { a = Node(pa), b = Node(pb), width = w });
        }
        var inc = g.inc = new List<Incident>[nodes.Count];
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
        var parent = g.parent = new int[nodes.Count];
        for (int i = 0; i < parent.Length; i++) parent[i] = i;
        int Find(int i) => g.Find(i);
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
        return g;
    }

    /// <summary>One junction: its streets' end corners and the kerb corners between them, in angular order, fanned
    /// from the junction's point (star-shaped from it by construction).</summary>
    private static void Star(List<Piece> into, Vector2 c, List<Incident> list, List<Edge> edges)
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
        Fan(into, c, ring);
    }

    /// <summary>A merged junction: the convex hull of its nodes, the rectangles of the streets merged into it and the
    /// end corners of the streets leaving it.</summary>
    private static void Merged(List<Piece> into, List<int> group, List<Vector2> nodes, List<Incident>[] inc,
                               List<Edge> edges)
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
        Fan(into, c / hull.Count, hull);
    }

    /// <summary>A crossing across a street: depth along <paramref name="d"/> from <paramref name="start"/>, the zebra
    /// strip's stripes running along the street.</summary>
    private static void Crossing(List<Piece> into, Vector2 start, Vector2 d, Vector2 left, float depth)
    {
        Vector2 end = start + d * depth;
        Quad(into, 1, start - left, start + left, end + left, end - left,
             new Vector2(ZebraUV.xMin, ZebraUV.yMin), new Vector2(ZebraUV.xMax, ZebraUV.yMin),
             new Vector2(ZebraUV.xMin, ZebraUV.yMax));
    }

    /// <summary>A junction's polygon as the triangles fanned from <paramref name="c"/> (one piece each: the ring is
    /// star-shaped from it, not convex), mapped onto the plain asphalt.</summary>
    private static void Fan(List<Piece> into, Vector2 c, List<Vector2> ring)
    {
        if (ring.Count < 2) return;
        float radius = 0f;
        foreach (Vector2 p in ring) radius = Mathf.Max(radius, (p - c).magnitude);
        float k = Mathf.Min(1f / JunctionMetresPerUV, 0.5f * PlainUV.width / Mathf.Max(radius, 1e-3f));
        for (int k2 = 0; k2 < ring.Count; k2++)
        {
            AddPiece(into, 1, new List<Vector2> { c, ring[k2], ring[(k2 + 1) % ring.Count] },
                c, new Vector2(k, 0f), new Vector2(0f, k), PlainUV.center, Vector2.right, Vector2.up);
        }
    }

    /// <summary>A rectangle p0 p1 p2 p3 whose corners p0, p1 and p3 map to t0, t1 and t3 (p2 to t1 + t3 - t0).</summary>
    private static void Quad(List<Piece> into, int sub, Vector2 p0, Vector2 p1, Vector2 p2, Vector2 p3,
                             Vector2 t0, Vector2 t1, Vector2 t3)
    {
        Vector2 u = p1 - p0, r = p3 - p0;
        AddPiece(into, sub, new List<Vector2> { p0, p1, p2, p3 }, p0, u / Mathf.Max(u.sqrMagnitude, 1e-12f),
            r / Mathf.Max(r.sqrMagnitude, 1e-12f), t0, t1 - t0, t3 - t0);
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
