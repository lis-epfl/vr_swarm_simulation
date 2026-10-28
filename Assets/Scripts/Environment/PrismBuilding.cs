using System;
using System.Collections.Generic;
using UnityEngine;

/// <summary>
/// The block-shaped buildings of S4 Split and Surround's variant B (citysim/blockcity.py, <c>type: ConvexPrism</c>):
/// a convex footprint stood up to its height, built as a mesh rather than taken from the building pack, with windows so
/// a pilot can tell one facade from the next and match a building across two feeds. <see cref="ProceduralCityBuilder"/>
/// places them; BlockShapedCityWorldBuilder makes their materials.
///
/// <para><b>The swarm sees them as it sees every building</b>: a collider on <c>Obstacle</c> -- a convex MeshCollider of
/// the prism itself, so contacts, raycasts and the recorder's line of sight meet the true facade -- which OlfatiSaber
/// turns into the circumradius of its axis-aligned bounds. That cylinder encloses the footprint, so the law keeps drones
/// off the facade, but it bulges past it: at p 40 s 18 it closes 8 of the 824 interior streets between two buildings
/// and narrows 71 below 6 m (<c>blockcity.street_stats</c>). S4's replica flew B with a closest-point model instead;
/// that model is not in Unity.</para>
///
/// <para><b>Facades</b>. Each wall runs a whole number of bays (<see cref="Style.bayWidth"/>) from corner to corner, so
/// every face starts and ends on wall with its windows centred; a face shorter than half a bay is plain wall. The ground
/// floor is a band of its own in its own material (shopfronts), the floors above it one window pattern in one colour.
/// Colours are dealt so that no two buildings within <see cref="Style.neighbourSpacings"/> block spacings share one
/// where the palette allows, and the pattern is drawn per building: a building in one feed is then told apart from those
/// beside it, which is what lets two feeds be matched. The deal is drawn from the layout's seed, so a layout always
/// looks the same.</para>
/// </summary>
public static class PrismBuilding
{
    public const string Type = "ConvexPrism";
    /// <summary>The texture column a face too short for a window samples: plain wall in every pattern.</summary>
    private const float BlankU = 0.015f;
    private const float RoofUvMetres = 10f;

    [Serializable]
    public class Style
    {
        [Tooltip("The floors above the ground floor, colour by colour: colours x patterns, a colour's patterns together.")]
        public Material[] facades = new Material[0];
        [Tooltip("Window patterns per colour in facades.")]
        public int patterns = 3;
        [Tooltip("The ground floor's shopfronts, on every building.")]
        public Material groundFloor;
        public Material roof;
        [Tooltip("m: one repeat of the facade texture across (a bay).")]
        public float bayWidth = 3f;
        [Tooltip("m: one repeat of the facade texture up (a floor).")]
        public float floorHeight = 3.5f;
        [Tooltip("m: the ground floor band, in groundFloor (one repeat of its texture up).")]
        public float groundFloorHeight = 4.5f;
        [Tooltip("Buildings whose centres are closer than this many block spacings (the layout's p) are dealt " +
                 "different colours where the palette allows: 1.5 takes in the buildings across every street and most " +
                 "of those across a junction.")]
        public float neighbourSpacings = 1.5f;

        public int Colours => facades == null || patterns < 1 ? 0 : facades.Length / patterns;
    }

    /// <summary>A building's colour (index into the palette) and window pattern.</summary>
    public struct Look
    {
        public int colour, pattern;
    }

    private static Material fallback;

    public static bool IsPrism(ProceduralCityBuilder.LayoutBox b) => b != null && b.type == Type;

    /// <summary>The footprint, (x, z) counter-clockwise seen from above, as the layout gives it.</summary>
    public static Vector2[] Footprint(ProceduralCityBuilder.LayoutBox b)
    {
        int n = b.poly_xz.Length / 2;
        var poly = new Vector2[n];
        for (int i = 0; i < n; i++) poly[i] = new Vector2(b.poly_xz[2 * i], b.poly_xz[2 * i + 1]);
        return poly;
    }

    /// <summary>Where the building stands: the layout's footprint centroid on its base (the layout's centre is at half
    /// height). The built object's transform is put here, and <c>ProceduralCityBuilder.Adopt</c> checks it.</summary>
    public static Vector3 Origin(ProceduralCityBuilder.LayoutBox b) => new Vector3(b.c[0], b.c[1] - 0.5f * b.height, b.c[2]);

    /// <summary>Whether a ConvexPrism record can be built: a centre, a height and a footprint of three or more points.</summary>
    public static bool Valid(ProceduralCityBuilder.LayoutBox b, out string error)
    {
        error = b.c == null || b.c.Length < 3 ? "no centre"
              : !(b.height > 0f) ? "no height"
              : b.poly_xz == null || b.poly_xz.Length < 6 || b.poly_xz.Length % 2 != 0 ? "no footprint (poly_xz)"
              : null;
        if (error != null) error = $"{b.name}: {error}.";
        return error == null;
    }

    /// <summary>Each prism's look, by name. In name order, each building takes the first colour of a shuffled palette
    /// that no building already dealt within the neighbour radius has, or, when every colour is taken, the one whose
    /// nearest use is farthest; <paramref name="clashes"/> counts those.</summary>
    public static Dictionary<string, Look> Deal(IEnumerable<ProceduralCityBuilder.LayoutBox> boxes, Style style, int seed,
                                                float blockSpacing, out int clashes)
    {
        var prisms = new List<ProceduralCityBuilder.LayoutBox>();
        foreach (ProceduralCityBuilder.LayoutBox b in boxes)
        {
            if (IsPrism(b)) prisms.Add(b);
        }
        prisms.Sort((a, b) => string.CompareOrdinal(a.name, b.name));
        int colours = Mathf.Max(style != null ? style.Colours : 0, 1), patterns = Mathf.Max(style != null ? style.patterns : 1, 1);
        float radius = (blockSpacing > 0f ? blockSpacing : 40f) * (style != null ? style.neighbourSpacings : 1.5f);
        float r2 = radius * radius;
        var rng = new System.Random(seed);
        var looks = new Dictionary<string, Look>();
        var centres = new List<Vector2>();
        var dealt = new List<int>();
        var order = new int[colours];
        var nearest = new float[colours];
        clashes = 0;
        foreach (ProceduralCityBuilder.LayoutBox b in prisms)
        {
            var c = new Vector2(b.c[0], b.c[2]);
            for (int k = 0; k < colours; k++)
            {
                order[k] = k;
                nearest[k] = float.PositiveInfinity;
            }
            for (int k = colours - 1; k > 0; k--)
            {
                int j = rng.Next(k + 1);
                (order[k], order[j]) = (order[j], order[k]);
            }
            int pattern = rng.Next(patterns);
            for (int i = 0; i < centres.Count; i++)
            {
                float d2 = (centres[i] - c).sqrMagnitude;
                if (d2 < nearest[dealt[i]]) nearest[dealt[i]] = d2;
            }
            int pick = order[0];
            float best = -1f;
            bool free = false;
            foreach (int k in order)
            {
                if (nearest[k] >= r2)
                {
                    pick = k;
                    free = true;
                    break;
                }
                if (nearest[k] > best)
                {
                    best = nearest[k];
                    pick = k;
                }
            }
            clashes += free ? 0 : 1;
            looks[b.name] = new Look { colour = pick, pattern = pattern };
            centres.Add(c);
            dealt.Add(pick);
        }
        return looks;
    }

    /// <summary>The building, under <paramref name="parent"/> (unrotated and unscaled at the origin, as the builder's
    /// containers are): mesh, renderer and a convex MeshCollider on <c>Obstacle</c>.</summary>
    public static GameObject Create(ProceduralCityBuilder.LayoutBox b, Transform parent, Style style, Look look)
    {
        style = style ?? new Style();
        Vector2[] poly = Footprint(b);
        Vector3 origin = Origin(b);
        var go = new GameObject(b.name);
        go.transform.SetParent(parent, false);
        go.transform.SetPositionAndRotation(origin, Quaternion.identity);
        int layer = LayerMask.NameToLayer("Obstacle");
        if (layer >= 0) go.layer = layer;
        Mesh mesh = BuildMesh(b.name, poly, new Vector2(origin.x, origin.z), b.height, style);
        go.AddComponent<MeshFilter>().sharedMesh = mesh;
        MeshRenderer mr = go.AddComponent<MeshRenderer>();
        int k = look.colour * Mathf.Max(style.patterns, 1) + look.pattern;
        Material facade = style.facades != null && k >= 0 && k < style.facades.Length ? style.facades[k] : null;
        mr.sharedMaterials = new[] { facade ?? Fallback(), style.groundFloor ?? facade ?? Fallback(), style.roof ?? Fallback() };
        MeshCollider mc = go.AddComponent<MeshCollider>();
        mc.sharedMesh = mesh;
        mc.convex = true;
        return go;
    }

    /// <summary>Submesh 0 the floors above the ground floor, 1 the ground floor, 2 the roof; vertices relative to the
    /// footprint's centroid on its base. Texture coordinates count bays across and floors up.</summary>
    private static Mesh BuildMesh(string name, Vector2[] poly, Vector2 o, float height, Style s)
    {
        var v = new List<Vector3>();
        var n = new List<Vector3>();
        var uv = new List<Vector2>();
        var upper = new List<int>();
        var ground = new List<int>();
        var roof = new List<int>();
        float bay = Mathf.Max(s.bayWidth, 0.1f), floor = Mathf.Max(s.floorHeight, 0.1f);
        float g = Mathf.Clamp(s.groundFloorHeight, 0f, height);
        for (int i = 0; i < poly.Length; i++)
        {
            Vector2 a = poly[i] - o, b = poly[(i + 1) % poly.Length] - o, e = b - a;
            float len = e.magnitude;
            if (len < 1e-3f) continue;
            var outward = new Vector3(e.y, 0f, -e.x) / len;           // the right of a -> b: outside, counter-clockwise
            float u0 = 0f, u1 = Mathf.Max(1f, Mathf.Round(len / bay));
            if (len < 0.5f * bay) u0 = u1 = BlankU;
            if (g > 1e-3f) Wall(v, n, uv, ground, a, b, 0f, g, outward, u0, u1, 0f, 1f);
            if (height - g > 1e-3f) Wall(v, n, uv, upper, a, b, g, height, outward, u0, u1, 0f, (height - g) / floor);
        }
        int r0 = v.Count;
        foreach (Vector2 p in poly)
        {
            Vector2 q = p - o;
            v.Add(new Vector3(q.x, height, q.y));
            n.Add(Vector3.up);
            uv.Add(p / RoofUvMetres);
        }
        for (int i = 1; i + 1 < poly.Length; i++) Tri(roof, v, r0, r0 + i, r0 + i + 1, Vector3.up);

        var mesh = new Mesh { name = name };
        mesh.SetVertices(v);
        mesh.SetNormals(n);
        mesh.SetUVs(0, uv);
        mesh.subMeshCount = 3;
        mesh.SetTriangles(upper, 0);
        mesh.SetTriangles(ground, 1);
        mesh.SetTriangles(roof, 2);
        mesh.RecalculateBounds();
        return mesh;
    }

    /// <summary>One wall quad from <paramref name="a"/> to <paramref name="b"/> between two heights, facing
    /// <paramref name="outward"/>, u running a to b (left to right seen from outside).</summary>
    private static void Wall(List<Vector3> v, List<Vector3> n, List<Vector2> uv, List<int> tris, Vector2 a, Vector2 b,
                             float y0, float y1, Vector3 outward, float u0, float u1, float v0, float v1)
    {
        int i = v.Count;
        v.Add(new Vector3(a.x, y0, a.y));
        v.Add(new Vector3(b.x, y0, b.y));
        v.Add(new Vector3(b.x, y1, b.y));
        v.Add(new Vector3(a.x, y1, a.y));
        for (int k = 0; k < 4; k++) n.Add(outward);
        uv.Add(new Vector2(u0, v0));
        uv.Add(new Vector2(u1, v0));
        uv.Add(new Vector2(u1, v1));
        uv.Add(new Vector2(u0, v1));
        Tri(tris, v, i, i + 1, i + 2, outward);
        Tri(tris, v, i, i + 2, i + 3, outward);
    }

    /// <summary>A triangle wound to face <paramref name="facing"/> (Unity's front face: clockwise seen from it).</summary>
    private static void Tri(List<int> tris, List<Vector3> v, int a, int b, int c, Vector3 facing)
    {
        if (Vector3.Dot(Vector3.Cross(v[b] - v[a], v[c] - v[a]), facing) < 0f) (b, c) = (c, b);
        tris.Add(a);
        tris.Add(b);
        tris.Add(c);
    }

    /// <summary>A plain grey material for a style with a slot left empty, so a building is never magenta. A bake saves it
    /// into the scene; BlockShapedCityWorldBuilder always fills the style.</summary>
    private static Material Fallback()
    {
        if (fallback == null)
        {
            Debug.LogWarning("PrismBuilding: the builder's prism style has an empty material slot; those buildings are grey.");
            fallback = new Material(Shader.Find("Standard")) { name = "Prism building (no style)", color = new Color(0.6f, 0.6f, 0.62f) };
        }
        return fallback;
    }
}
