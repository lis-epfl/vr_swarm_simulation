using System;
using System.Collections.Generic;
using UnityEngine;

/// <summary>
/// What a block-shaped building (<see cref="PrismBuilding"/>) looks like in the scenic city: its style, dealt from the
/// layout's seed, and the mesh drawn for it in <c>Swarm/Scenic Facade</c>. <see cref="ScenicCity"/> applies it.
///
/// <para><b>The collider is not touched.</b> A building keeps PrismBuilding's convex prism on <c>Obstacle</c>, footprint
/// and 50 m height as the layout gives them, so the swarm, the contacts, the recorder's line of sight and the analysis
/// see the city they always did. The mesh keeps to that prism below its roof: the walls stand on the footprint, and
/// what stands proud of them (cornices, fins, ledges, balcony fronts) is at most <see cref="MaxProud"/> out, above the
/// ground floor. A pedestrian walks 1.1 m or more off the wall, so none of it can hide one from a drone. Above the
/// roof (50 m) the mesh is free: the drones' ceiling is below it, so extra floors, set-back tops and roof plant change
/// the skyline and nothing else.</para>
///
/// <para><b>Two meshes of one building.</b> The lite one is its walls and roofs: about a hundred vertices, what a bake
/// saves in the scene, and what the editor shows. The detailed one adds the relief and the roof plant, a few thousand
/// vertices, made at Play from the same deal (<see cref="ScenicCity.Finish"/>). Both report the same bounds, so
/// occlusion baked on the lite mesh holds for the detailed one.</para>
///
/// <para><b>Styles.</b> Six families (<see cref="Family"/>), each with its own wall materials, colours, window
/// pattern, ground floor and relief; within a family every building draws its own. The deal (<see cref="Deal"/>) takes,
/// for each building in name order, the one of several candidates least like the buildings already dealt round it, so
/// neighbours differ in more than colour, which is what lets a pilot tell them apart and match one across two feeds.</para>
/// </summary>
public static class ScenicBuilding
{
    /// <summary>m: the most any relief stands out of the footprint below the roof.</summary>
    public const float MaxProud = 0.5f;
    /// <summary>The lightest a ground floor's wall may be (linear luminance of its mean colour): sRGB 0.52.</summary>
    public const float MaxBaseLuminance = 0.23f;
    /// <summary>m above the roof the mesh's bounds allow for plant and masts.</summary>
    private const float PlantAllowance = 18f;
    private const string LiteSuffix = " (lite)";

    public enum Family { Brick, Plaster, Concrete, Stone, Glass, Clad }

    public enum Window { Punched = 0, Ribbon = 1, Curtain = 2, Arched = 3 }

    public enum Ground { Shops = 0, Lobby = 1, Masonry = 2 }

    /// <summary>One building's look. Colours are linear.</summary>
    public class Spec
    {
        public Family family;
        public int wallLayer, baseLayer, trimLayer;
        public Color wallTint = Color.white, baseTint = Color.white, trim = Color.grey, frame = Color.black;
        public Color glass = Color.white;
        public float glassF0 = 0.08f;
        public float bay = 3f, floorH = 3.5f, groundH = 4.5f;
        public Window window;
        public float winW = 0.5f, winH = 0.55f, sill = 0.25f, reveal = 0.2f;
        public int mullions = 1, transoms;
        public int blinds = 1;                  // 0 none, 1 blinds, 2 curtains
        public Ground ground;
        public int shopBays = 2;
        public float extra;                     // m above the collider's roof
        public float setback;                   // m the part above the roof stands in from the footprint; 0: none
        public float parapet = 1.1f;            // m of plain wall above the top windows
        public float cornice, corniceH;         // m out, m high; 0: none
        public float course;                    // m out, the band over the ground floor; 0: none
        public float finDepth, finWidth;        // 0: none
        public int finEvery = 1;
        public float ledge;                     // m out, at every floor; 0: none
        public float balcony;                   // m out; 0: none
        public Color rail = Color.grey;         // the balcony fronts
        public int tank, plant, mast;           // roof: water tanks, plant boxes, masts
        public bool penthouse;
        public float seed;                      // 0..1, the shader's
        public int rng;                         // the mesh's
        /// <summary>0-31: where along each shop (or each eight bays of a lobby or a masonry base) its door stands.</summary>
        public int DoorSeed => rng & 31;
        public Color look;                      // the wall's mean colour, for telling neighbours apart
        public Color paintMean = Color.grey;    // the trim layer's mean colour: a plain surface's tint is (colour / this)

        /// <summary>The tint that gives a plain surface in the trim layer the colour <paramref name="target"/>.</summary>
        public Color Paint(Color target) =>
            new Color(Mathf.Clamp01(target.r / Mathf.Max(paintMean.r, 0.01f)), Mathf.Clamp01(target.g / Mathf.Max(paintMean.g, 0.01f)),
                      Mathf.Clamp01(target.b / Mathf.Max(paintMean.b, 0.01f)), 1f);
    }

    // ------------------------------------------------------------------ the deal

    private static readonly Color[] Paints =
    {
        Rgb(0.86f, 0.85f, 0.81f), Rgb(0.85f, 0.80f, 0.68f), Rgb(0.80f, 0.71f, 0.55f), Rgb(0.86f, 0.77f, 0.52f),
        Rgb(0.83f, 0.65f, 0.52f), Rgb(0.71f, 0.45f, 0.34f), Rgb(0.77f, 0.58f, 0.55f), Rgb(0.62f, 0.68f, 0.56f),
        Rgb(0.62f, 0.70f, 0.76f), Rgb(0.66f, 0.66f, 0.65f), Rgb(0.58f, 0.55f, 0.50f), Rgb(0.75f, 0.60f, 0.34f),
        Rgb(0.50f, 0.53f, 0.39f), Rgb(0.45f, 0.51f, 0.58f), Rgb(0.56f, 0.34f, 0.29f), Rgb(0.31f, 0.32f, 0.34f),
    };
    private static readonly Color[] Panels =
    {
        Rgb(0.76f, 0.76f, 0.75f), Rgb(0.85f, 0.85f, 0.84f), Rgb(0.27f, 0.28f, 0.30f), Rgb(0.56f, 0.31f, 0.21f),
        Rgb(0.46f, 0.49f, 0.37f), Rgb(0.22f, 0.29f, 0.40f), Rgb(0.78f, 0.72f, 0.60f), Rgb(0.40f, 0.42f, 0.44f),
    };
    private static readonly Color[] Frames =
    {
        Rgb(0.86f, 0.86f, 0.84f), Rgb(0.16f, 0.17f, 0.18f), Rgb(0.29f, 0.23f, 0.17f), Rgb(0.13f, 0.23f, 0.18f),
        Rgb(0.62f, 0.63f, 0.64f),
    };
    private static readonly Color[] Glasses =
    {
        Rgb(0.80f, 0.84f, 0.84f), Rgb(0.52f, 0.68f, 0.84f), Rgb(0.56f, 0.80f, 0.74f), Rgb(0.80f, 0.66f, 0.48f),
        Rgb(0.42f, 0.48f, 0.54f), Rgb(0.35f, 0.52f, 0.70f),
    };
    private static readonly Color Limestone = Rgb(0.78f, 0.75f, 0.68f), Concrete = Rgb(0.63f, 0.62f, 0.59f),
                                  WhiteTrim = Rgb(0.84f, 0.84f, 0.82f), DarkStone = Rgb(0.36f, 0.34f, 0.32f);

    private static Color Rgb(float r, float g, float b) => new Color(r, g, b, 1f).linear;

    /// <summary>Every prism's look, by name, from the layout's seed.</summary>
    public static Dictionary<string, Spec> Deal(IEnumerable<ProceduralCityBuilder.LayoutBox> boxes, ScenicFacades lib, int seed,
                                                float blockSpacing)
    {
        var prisms = new List<ProceduralCityBuilder.LayoutBox>();
        foreach (ProceduralCityBuilder.LayoutBox b in boxes)
        {
            if (PrismBuilding.IsPrism(b)) prisms.Add(b);
        }
        prisms.Sort((a, b) => string.CompareOrdinal(a.name, b.name));
        float radius = (blockSpacing > 0f ? blockSpacing : 40f) * 1.6f;
        var specs = new Dictionary<string, Spec>();
        var dealt = new List<(Vector2 at, Spec spec)>();
        var near = new List<Spec>();
        foreach (ProceduralCityBuilder.LayoutBox b in prisms)
        {
            var rng = new System.Random(unchecked(seed * 7919 + Stable(b.name)));
            var at = new Vector2(b.c[0], b.c[2]);
            near.Clear();
            foreach ((Vector2 p, Spec s) in dealt)
            {
                if ((p - at).sqrMagnitude < radius * radius) near.Add(s);
            }
            float inradius = Inradius(PrismBuilding.Footprint(b), at);
            Spec best = null;
            float bestScore = -1f;
            for (int k = 0; k < 7; k++)
            {
                Spec cand = Draw(rng, lib, inradius);
                float score = 10f;
                foreach (Spec s in near) score = Mathf.Min(score, Difference(cand, s));
                if (score > bestScore)
                {
                    bestScore = score;
                    best = cand;
                }
            }
            specs[b.name] = best;
            dealt.Add((at, best));
        }
        return specs;
    }

    /// <summary>A hash of a name that is the same on every run (string.GetHashCode is not promised to be).</summary>
    private static int Stable(string s)
    {
        unchecked
        {
            uint h = 2166136261;
            foreach (char ch in s) h = (h ^ ch) * 16777619;
            return (int)h;
        }
    }

    private static float Difference(Spec a, Spec b)
    {
        float d = a.family != b.family ? 0.5f : 0f;
        d += a.wallLayer != b.wallLayer ? 0.2f : 0f;
        d += a.window != b.window ? 0.15f : 0f;
        Color ca = a.look.gamma, cb = b.look.gamma;
        d += 0.9f * Mathf.Sqrt((ca.r - cb.r) * (ca.r - cb.r) + (ca.g - cb.g) * (ca.g - cb.g) + (ca.b - cb.b) * (ca.b - cb.b));
        return d;
    }

    private static float U(System.Random r, float a, float b) => a + (b - a) * (float)r.NextDouble();

    private static T Pick<T>(System.Random r, params T[] of) => of[r.Next(of.Length)];

    private static bool Chance(System.Random r, float p) => r.NextDouble() < p;

    /// <summary>One candidate look. <paramref name="inradius"/> (m, the footprint's) keeps towers off small plots.</summary>
    private static Spec Draw(System.Random r, ScenicFacades lib, float inradius)
    {
        var s = new Spec { seed = (float)r.NextDouble(), rng = r.Next() };
        double f = r.NextDouble();
        s.family = f < 0.22 ? Family.Brick : f < 0.41 ? Family.Plaster : f < 0.58 ? Family.Concrete : f < 0.72 ? Family.Stone
                 : f < 0.87 ? Family.Glass : Family.Clad;
        if (s.family == Family.Glass && inradius < 7f) s.family = Family.Concrete;
        float extraScale = lib.extraHeight;
        // Trim, rails and roof plant are painted plaster in their own colour.
        s.trimLayer = lib.Layer("painted_plaster_wall");
        s.paintMean = lib.walls[s.trimLayer].mean;
        s.baseLayer = s.trimLayer;
        s.glass = Glasses[0];
        switch (s.family)
        {
            case Family.Brick:
            {
                string wall = Pick(r, "red_brick_03", "brick_wall_006", "brown_brick_02", "painted_brick", "red_brick_03");
                s.wallLayer = lib.Layer(wall);
                s.wallTint = wall == "painted_brick" ? lib.TintFor(s.wallLayer, Pick(r, Paints[0], Paints[1], Paints[7], Paints[9], Paints[13], Paints[14], Paints[15]).gamma)
                                                     : Shade(r, 0.82f, 1f);
                s.bay = U(r, 2.7f, 3.4f);
                s.floorH = U(r, 3.3f, 3.8f);
                s.window = Chance(r, 0.28f) ? Window.Arched : Window.Punched;
                s.winW = U(r, 0.40f, 0.54f);
                s.winH = U(r, 0.52f, 0.62f);
                s.sill = U(r, 0.22f, 0.28f);
                s.reveal = U(r, 0.18f, 0.28f);
                s.mullions = 2;
                s.transoms = Chance(r, 0.6f) ? 1 : 0;
                s.frame = Pick(r, Frames[0], Frames[0], Frames[1], Frames[3]);
                s.trim = Pick(r, Limestone, Limestone, Concrete, WhiteTrim);
                s.blinds = Pick(r, 1, 2, 2);
                s.ground = Chance(r, 0.6f) ? Ground.Shops : Ground.Masonry;
                BaseOf(s, r, lib, Pick(r, "", "", "granite_tile_03", "stone_tile_wall"));
                s.cornice = U(r, 0.3f, 0.45f);
                s.corniceH = U(r, 0.5f, 0.8f);
                s.course = U(r, 0.12f, 0.2f);
                if (Chance(r, 0.25f)) { s.finDepth = U(r, 0.1f, 0.16f); s.finWidth = U(r, 0.5f, 0.7f); s.finEvery = Pick(r, 1, 2); }
                s.extra = Pick(r, 0f, 0f, 0f, 3.5f, 7f) * extraScale;
                s.tank = Chance(r, 0.5f) ? 1 : 0;
                s.plant = r.Next(0, 3);
                s.penthouse = Chance(r, 0.4f);
                break;
            }
            case Family.Plaster:
            {
                string wall = Pick(r, "white_stucco", "painted_plaster_wall");
                s.wallLayer = lib.Layer(wall);
                s.wallTint = lib.TintFor(s.wallLayer, Paints[r.Next(Paints.Length - 1)].gamma);
                s.bay = U(r, 2.8f, 3.6f);
                s.floorH = U(r, 3.0f, 3.4f);
                s.window = Window.Punched;
                s.winW = U(r, 0.44f, 0.62f);
                s.winH = U(r, 0.48f, 0.60f);
                s.sill = U(r, 0.24f, 0.30f);
                s.reveal = U(r, 0.12f, 0.2f);
                s.mullions = Pick(r, 1, 2, 2, 3);
                s.frame = Pick(r, Frames[0], Frames[0], Frames[1], Frames[4]);
                s.trim = Pick(r, WhiteTrim, WhiteTrim, Concrete, Limestone);
                s.blinds = 2;
                s.ground = Chance(r, 0.7f) ? Ground.Shops : Ground.Lobby;
                BaseOf(s, r, lib, Pick(r, "", "concrete_wall_008", "stone_tile_wall", "painted_plaster_wall"));
                if (Chance(r, 0.5f)) { s.cornice = U(r, 0.15f, 0.25f); s.corniceH = U(r, 0.25f, 0.45f); }
                s.course = Chance(r, 0.6f) ? U(r, 0.08f, 0.15f) : 0f;
                if (Chance(r, 0.42f))
                {
                    // balconies: doors down to a low sill behind a rail
                    s.balcony = U(r, 0.38f, MaxProud);
                    s.rail = Pick(r, s.trim, s.trim, Rgb(0.55f, 0.56f, 0.57f), Rgb(0.30f, 0.31f, 0.33f));
                    s.sill = 0.05f;
                    s.winH = U(r, 0.66f, 0.72f);
                    s.winW = U(r, 0.5f, 0.66f);
                }
                else if (Chance(r, 0.3f)) s.ledge = U(r, 0.1f, 0.18f);
                s.extra = Pick(r, 0f, 0f, 3.2f, 6.4f, 9.6f) * extraScale;
                s.plant = r.Next(0, 3);
                s.penthouse = Chance(r, 0.5f);
                s.mast = Chance(r, 0.3f) ? 1 : 0;
                break;
            }
            case Family.Concrete:
            {
                string wall = Pick(r, "concrete_wall_008", "concrete_wall_004", "exterior_wall_cladding_03");
                s.wallLayer = lib.Layer(wall);
                s.wallTint = wall == "exterior_wall_cladding_03" ? lib.TintFor(s.wallLayer, Pick(r, Panels[0], Panels[1], Panels[6]).gamma)
                                                                : Shade(r, 0.85f, 1f);
                s.bay = U(r, 3.0f, 4.2f);
                s.floorH = U(r, 3.5f, 4.0f);
                s.window = Chance(r, 0.6f) ? Window.Ribbon : Window.Punched;
                s.winW = s.window == Window.Ribbon ? 1f : U(r, 0.66f, 0.8f);
                s.winH = U(r, 0.44f, 0.58f);
                s.sill = U(r, 0.26f, 0.30f);
                s.reveal = U(r, 0.1f, 0.22f);
                s.mullions = Pick(r, 2, 3);
                s.frame = Pick(r, Frames[1], Frames[1], Frames[4], Frames[2]);
                s.trim = Shade(r, 0.8f, 1f) * Concrete;
                s.blinds = 1;
                s.ground = Chance(r, 0.65f) ? Ground.Lobby : Ground.Shops;
                BaseOf(s, r, lib, Pick(r, "", "granite_tile_03", "concrete_wall_004"));
                s.parapet = U(r, 0.7f, 1.2f);
                double relief = r.NextDouble();
                if (relief < 0.4) { s.finDepth = U(r, 0.25f, MaxProud); s.finWidth = U(r, 0.2f, 0.4f); s.finEvery = Pick(r, 1, 1, 2); }
                else if (relief < 0.7) s.ledge = U(r, 0.25f, 0.45f);
                s.extra = Pick(r, 0f, 0f, 3.7f, 7.4f, 11.1f, 18.5f) * extraScale;
                s.plant = r.Next(1, 4);
                s.penthouse = true;
                s.mast = Chance(r, 0.4f) ? 1 : 0;
                break;
            }
            case Family.Stone:
            {
                string wall = Pick(r, "stone_tile_wall", "marble_01", "stone_tile_wall");
                s.wallLayer = lib.Layer(wall);
                s.wallTint = Shade(r, 0.86f, 1f);
                s.bay = U(r, 2.3f, 2.9f);
                s.floorH = U(r, 3.7f, 4.3f);
                s.window = Chance(r, 0.2f) ? Window.Arched : Window.Punched;
                s.winW = U(r, 0.42f, 0.52f);
                s.winH = U(r, 0.60f, 0.70f);
                s.sill = U(r, 0.16f, 0.2f);
                s.reveal = U(r, 0.25f, 0.36f);
                s.mullions = 2;
                s.transoms = Pick(r, 1, 1, 2);
                s.frame = Pick(r, Frames[2], Frames[1], Frames[2]);
                s.trim = Shade(r, 0.9f, 1.05f) * Limestone;
                s.blinds = 1;
                s.ground = Pick(r, Ground.Masonry, Ground.Lobby, Ground.Shops);
                BaseOf(s, r, lib, Pick(r, "granite_tile_03", "", "marble_01"));
                s.groundH = U(r, 5.0f, 6.2f);
                s.cornice = U(r, 0.4f, MaxProud);
                s.corniceH = U(r, 0.7f, 1.0f);
                s.course = U(r, 0.18f, 0.28f);
                if (Chance(r, 0.6f)) { s.finDepth = U(r, 0.12f, 0.2f); s.finWidth = U(r, 0.6f, 0.8f); s.finEvery = 2; }
                s.extra = Pick(r, 0f, 0f, 4f, 8f) * extraScale;
                if (s.extra > 3f && inradius > 8f) s.setback = U(r, 1.5f, 2.5f);
                s.plant = r.Next(0, 2);
                s.penthouse = Chance(r, 0.5f);
                break;
            }
            case Family.Glass:
            {
                s.wallLayer = lib.Layer(Pick(r, "exterior_wall_cladding_03", "concrete_tile_facade"));
                s.wallTint = lib.walls[s.wallLayer].paintable ? lib.TintFor(s.wallLayer, Pick(r, Panels[0], Panels[2], Panels[7]).gamma) : Color.white;
                s.bay = U(r, 1.5f, 2.1f);
                s.floorH = U(r, 3.6f, 4.1f);
                s.window = Window.Curtain;
                s.winW = 1f;
                s.winH = 1f;
                s.sill = 0f;
                s.reveal = 0.04f;
                s.mullions = 1;
                s.frame = Pick(r, Frames[4], Frames[1], Frames[1]);
                s.glass = Glasses[1 + r.Next(Glasses.Length - 1)];
                s.glassF0 = U(r, 0.16f, 0.34f);
                s.trim = Concrete;
                s.blinds = Pick(r, 0, 1);
                s.ground = Ground.Lobby;
                BaseOf(s, r, lib, Pick(r, "granite_tile_03", "marble_01", "concrete_tile_facade"));
                s.groundH = U(r, 5.5f, 7.5f);
                s.parapet = U(r, 0.4f, 0.9f);
                s.extra = Pick(r, 7.5f, 11f, 15f, 22f, 30f, 38f) * extraScale;
                if (s.extra > 12f && inradius > 9f && Chance(r, 0.55f)) s.setback = U(r, 2f, 3.5f);
                s.plant = r.Next(1, 3);
                s.penthouse = true;
                s.mast = Chance(r, 0.6f) ? r.Next(1, 3) : 0;
                break;
            }
            default:
            {
                string wall = Pick(r, "concrete_tile_facade", "exterior_wall_cladding_03", "exterior_wall_cladding_03");
                s.wallLayer = lib.Layer(wall);
                s.wallTint = lib.walls[s.wallLayer].paintable ? lib.TintFor(s.wallLayer, Pick(r, Panels[2], Panels[3], Panels[4], Panels[5], Panels[7]).gamma)
                                                               : Shade(r, 0.9f, 1f);
                s.bay = U(r, 2.6f, 3.8f);
                s.floorH = U(r, 3.3f, 3.8f);
                s.window = Chance(r, 0.35f) ? Window.Ribbon : Window.Punched;
                s.winW = s.window == Window.Ribbon ? 1f : U(r, 0.5f, 0.74f);
                s.winH = U(r, 0.5f, 0.68f);
                s.sill = U(r, 0.14f, 0.26f);
                s.reveal = U(r, 0.16f, 0.26f);
                s.mullions = Pick(r, 1, 2);
                s.frame = Pick(r, Frames[2], Frames[1], Frames[4]);
                s.trim = Pick(r, DarkStone, Concrete);
                s.blinds = Pick(r, 1, 2);
                s.ground = Chance(r, 0.5f) ? Ground.Shops : Ground.Lobby;
                BaseOf(s, r, lib, Pick(r, "", "granite_tile_03", "concrete_wall_008"));
                s.parapet = U(r, 0.5f, 1.0f);
                if (Chance(r, 0.3f)) { s.balcony = U(r, 0.35f, MaxProud); s.rail = Pick(r, s.trim, Rgb(0.42f, 0.43f, 0.45f), s.frame); }
                s.extra = Pick(r, 0f, 3.5f, 7f, 10.5f, 14f) * extraScale;
                s.plant = r.Next(1, 4);
                s.penthouse = Chance(r, 0.7f);
                s.mast = Chance(r, 0.3f) ? 1 : 0;
                break;
            }
        }
        if (s.ground == Ground.Shops) s.groundH = Mathf.Max(s.groundH, U(r, 4.3f, 5.0f));
        // The pedestrians are pale figures seen against the ground floor: its wall is kept no lighter than a mid stone
        // (the piers of the plain prisms' shopfronts were), whatever the building above is painted.
        Color baseMean = lib.walls[s.baseLayer].mean;
        float baseLum = 0.2126f * baseMean.r * s.baseTint.r + 0.7152f * baseMean.g * s.baseTint.g + 0.0722f * baseMean.b * s.baseTint.b;
        if (baseLum > MaxBaseLuminance) s.baseTint *= MaxBaseLuminance / baseLum;
        s.shopBays = Mathf.Clamp(Mathf.RoundToInt(U(r, 5f, 9f) / s.bay), 1, 4);
        Color mean = lib.walls[s.wallLayer].mean;
        s.look = s.family == Family.Glass ? s.glass * 0.45f
                                          : new Color(mean.r * s.wallTint.r, mean.g * s.wallTint.g, mean.b * s.wallTint.b, 1f);
        return s;
    }

    /// <summary>A near-white tint, the same on every channel, for a photographed wall.</summary>
    private static Color Shade(System.Random r, float lo, float hi)
    {
        float v = U(r, lo, hi);
        return new Color(v, v, v, 1f);
    }

    /// <summary>The ground floor's wall: <paramref name="wall"/>, or the building's own if it is empty.</summary>
    private static void BaseOf(Spec s, System.Random r, ScenicFacades lib, string wall)
    {
        if (string.IsNullOrEmpty(wall))
        {
            s.baseLayer = s.wallLayer;
            s.baseTint = s.wallTint * 0.92f;
            return;
        }
        s.baseLayer = lib.Layer(wall);
        s.baseTint = lib.walls[s.baseLayer].paintable ? lib.TintFor(s.baseLayer, Pick(r, Paints[9], Paints[10], Paints[15]).gamma) : Shade(r, 0.85f, 1f);
    }

    private static float Inradius(Vector2[] poly, Vector2 c)
    {
        float best = float.MaxValue;
        for (int i = 0; i < poly.Length; i++)
        {
            Vector2 a = poly[i], e = poly[(i + 1) % poly.Length] - a;
            float len = e.magnitude;
            if (len < 1e-4f) continue;
            best = Mathf.Min(best, Mathf.Abs((c.x - a.x) * e.y - (c.y - a.y) * e.x) / len);
        }
        return best;
    }

    // ------------------------------------------------------------------ the doors

    /// <summary>The building's doorways as the facade shader draws them: each a stretch of the footprint's edge, in
    /// world (x, z), from the door's left jamb to its right seen from outside. The same whole-number rule as the shader's
    /// (a door to every shop, or to every eight bays of a lobby or a masonry base), so <see cref="CityDressing"/> can keep
    /// the street furniture out of them.</summary>
    public static List<(Vector2 a, Vector2 b)> Doors(ProceduralCityBuilder.LayoutBox box, Spec s)
    {
        var doors = new List<(Vector2, Vector2)>();
        Vector2[] poly = PrismBuilding.Footprint(box);
        for (int i = 0; i < poly.Length; i++)
        {
            Vector2 a = poly[i], e = poly[(i + 1) % poly.Length] - a;
            float len = e.magnitude;
            if (len < 0.6f * s.bay || len < 1e-3f) continue;
            Vector2 dir = e / len;
            int bays = Mathf.Max(1, Mathf.RoundToInt(len / s.bay));
            float bayW = len / bays;
            int per = s.ground == Ground.Shops ? Mathf.Max(s.shopBays, 1) : 8;
            int step = s.ground == Ground.Shops ? 5 : 3;
            for (int g = 0; g * per < bays; g++)
            {
                int bay = g * per + (g * step + s.DoorSeed) % per;
                if (bay >= bays) continue;
                float lo, hi;
                if (s.ground == Ground.Shops) { lo = 0.22f; hi = bayW - 0.22f; }
                else if (s.ground == Ground.Lobby) { lo = 0.42f; hi = bayW - 0.42f; }
                else
                {
                    float w = Mathf.Min(1.7f, bayW * 0.6f);
                    lo = 0.5f * (bayW - w);
                    hi = 0.5f * (bayW + w);
                }
                doors.Add((a + dir * (bay * bayW + lo), a + dir * (bay * bayW + hi)));
            }
        }
        return doors;
    }

    // ------------------------------------------------------------------ the mesh

    /// <summary>Whether <paramref name="mesh"/> is a lite mesh of <see cref="Build"/>.</summary>
    public static bool IsLite(Mesh mesh) => mesh != null && mesh.name.EndsWith(LiteSuffix, StringComparison.Ordinal);

    /// <summary>The building's mesh round its footprint's centroid on its base (<see cref="PrismBuilding.Origin"/>):
    /// walls and roofs, and with <paramref name="detail"/> the relief and the roof plant.</summary>
    public static Mesh Build(ProceduralCityBuilder.LayoutBox b, Spec s, bool detail)
    {
        Vector2[] world = PrismBuilding.Footprint(b);
        Vector3 origin = PrismBuilding.Origin(b);
        var poly = new List<Vector2>(world.Length);
        var o = new Vector2(origin.x, origin.z);
        foreach (Vector2 p in world)
        {
            Vector2 q = p - o;
            if (poly.Count == 0 || (poly[poly.Count - 1] - q).sqrMagnitude > 1e-6f) poly.Add(q);
        }
        if (poly.Count > 1 && (poly[0] - poly[poly.Count - 1]).sqrMagnitude < 1e-6f) poly.RemoveAt(poly.Count - 1);

        float roof = b.height, top = roof + Mathf.Max(s.extra, 0f);
        List<Vector2> upper = s.setback > 0f && top - roof > 2.5f ? Offset(poly, -s.setback) : null;
        if (upper != null && !Sound(poly, upper)) upper = null;
        float lowTop = upper != null ? roof : top;

        var m = new Buffer(s);
        var rng = new System.Random(s.rng);
        Storey low = Mass(m, s, poly, 0f, lowTop, true, upper != null ? 0.9f : s.parapet);
        Storey high = default;
        m.Roof(poly, lowTop, s);
        if (upper != null)
        {
            high = Mass(m, s, upper, roof, top, false, s.parapet);
            m.Roof(upper, top, s);
        }
        if (detail)
        {
            Relief(m, s, poly, low, true);
            if (upper != null) Relief(m, s, upper, high, false);
            Plant(m, s, upper ?? poly, top, rng);
        }

        Mesh mesh = m.ToMesh(b.name + (detail ? "" : LiteSuffix));
        // One box for both meshes, so occlusion baked on the lite one holds for the detailed one.
        float minX = float.MaxValue, maxX = float.MinValue, minZ = float.MaxValue, maxZ = float.MinValue;
        foreach (Vector2 p in poly)
        {
            minX = Mathf.Min(minX, p.x); maxX = Mathf.Max(maxX, p.x);
            minZ = Mathf.Min(minZ, p.y); maxZ = Mathf.Max(maxZ, p.y);
        }
        var bounds = new Bounds();
        bounds.SetMinMax(new Vector3(minX - MaxProud - 0.1f, 0f, minZ - MaxProud - 0.1f),
                         new Vector3(maxX + MaxProud + 0.1f, top + PlantAllowance, maxZ + MaxProud + 0.1f));
        mesh.bounds = bounds;
        return mesh;
    }

    /// <summary>The floors of one mass: where its windows start and stop and how tall a floor came out.</summary>
    private struct Storey
    {
        public float first, floorH, winTop, top;
        public int floors;
    }

    /// <summary>The walls of one mass from <paramref name="y0"/> to <paramref name="y1"/>, a quad to an edge; the shader
    /// draws the floors. Floors are stretched a little so a whole number fits under the parapet.</summary>
    private static Storey Mass(Buffer m, Spec s, List<Vector2> poly, float y0, float y1, bool withGround, float parapet)
    {
        float first = withGround ? Mathf.Min(s.groundH, y1 - y0 - parapet - 2.5f) : y0;
        float room = Mathf.Max(y1 - parapet - first, 2.5f);
        int floors = Mathf.Max(1, Mathf.RoundToInt(room / s.floorH));
        var st = new Storey { first = first, floors = floors, floorH = room / floors, winTop = first + room, top = y1 };
        for (int i = 0; i < poly.Count; i++)
        {
            Vector2 a = poly[i], b = poly[(i + 1) % poly.Count];
            float len = (b - a).magnitude;
            if (len < 1e-3f) continue;
            int bays = len < 0.6f * s.bay ? 0 : Mathf.Max(1, Mathf.RoundToInt(len / s.bay));
            m.Facade(len, bays, st);
            m.Quad(At(a, y0), At(b, y0), At(b, y1), At(a, y1), 0f, y0);
        }
        return st;
    }

    private static Vector3 At(Vector2 p, float y) => new Vector3(p.x, y, p.y);

    /// <summary>What stands proud of one mass's walls: the band over the ground floor, the cornice, fins, floor ledges,
    /// balcony fronts.</summary>
    private static void Relief(Buffer m, Spec s, List<Vector2> poly, Storey st, bool withGround)
    {
        Color trim = s.Paint(s.trim);
        if (withGround && s.course > 0f) m.Ring(poly, st.first - 0.12f, st.first + 0.16f, s.course, s.trimLayer, trim, true, true);
        if (s.cornice > 0f) m.Ring(poly, st.top - s.corniceH, st.top, s.cornice, s.trimLayer, trim, false, true);
        else m.Ring(poly, st.top - 0.18f, st.top, 0.06f, s.trimLayer, trim * 0.9f, false, true);
        if (s.ledge > 0f)
        {
            for (int f = 1; f <= st.floors; f++)
            {
                float y = st.first + f * st.floorH;
                if (y > st.winTop + 0.05f) break;
                m.Ring(poly, y - 0.16f, y + 0.12f, s.ledge, s.trimLayer, trim, true, true);
            }
        }
        for (int i = 0; i < poly.Count; i++)
        {
            Vector2 a = poly[i], b = poly[(i + 1) % poly.Count], e = b - a;
            float len = e.magnitude;
            if (len < 1e-3f) continue;
            Vector2 dir = e / len, n = new Vector2(dir.y, -dir.x);
            int bays = len < 0.6f * s.bay ? 0 : Mathf.Max(1, Mathf.RoundToInt(len / s.bay));
            if (s.finDepth > 0f && bays > 0)
            {
                float bayW = len / bays, half = 0.5f * s.finWidth;
                Color tint = s.family == Family.Concrete || s.family == Family.Glass ? trim : s.wallTint;
                int layer = s.family == Family.Concrete || s.family == Family.Glass ? s.trimLayer : s.wallLayer;
                for (int j = 0; j <= bays; j += Mathf.Max(s.finEvery, 1))
                {
                    float at = Mathf.Clamp(j * bayW, half, len - half);
                    m.Fin(a + dir * (at - half), dir, n, s.finWidth, s.finDepth, st.first + (withGround ? 0.16f : 0f), st.winTop + 0.3f, layer, tint);
                }
            }
            if (s.balcony > 0f && bays >= 2)
            {
                float inset = 0.5f * (len / bays) * (1f - s.winW) - 0.25f;
                inset = Mathf.Clamp(inset, 0.3f, 0.5f * len - 1f);
                for (int f = withGround ? 0 : 1; f < st.floors; f++)
                {
                    float floorY = st.first + f * st.floorH + s.sill * st.floorH;
                    m.Balcony(a + dir * inset, dir, n, len - 2f * inset, s.balcony, floorY, s.trimLayer, trim, s.Paint(s.rail));
                }
            }
        }
    }

    /// <summary>The roof's plant: a penthouse, boxes of machinery, water tanks, masts. Seen from the street only at a
    /// distance, against the sky.</summary>
    private static void Plant(Buffer m, Spec s, List<Vector2> poly, float y, System.Random r)
    {
        Vector2 c = Vector2.zero;
        foreach (Vector2 p in poly) c += p;
        c /= poly.Count;
        float reach = float.MaxValue;
        int longest = 0;
        float longestLen = 0f;
        for (int i = 0; i < poly.Count; i++)
        {
            Vector2 a = poly[i], e = poly[(i + 1) % poly.Count] - a;
            float len = e.magnitude;
            if (len < 1e-4f) continue;
            reach = Mathf.Min(reach, Mathf.Abs((c.x - a.x) * e.y - (c.y - a.y) * e.x) / len);
            if (len > longestLen) { longestLen = len; longest = i; }
        }
        reach -= 1.6f;                                   // stand back from the roof's edge
        if (reach < 1.2f) return;
        Vector2 along = (poly[(longest + 1) % poly.Count] - poly[longest]).normalized, across = new Vector2(along.y, -along.x);
        Color metal = s.Paint(Rgb(0.60f, 0.61f, 0.62f)), dark = s.Paint(Rgb(0.26f, 0.27f, 0.29f));
        int clad = s.trimLayer;
        if (s.penthouse)
        {
            float hu = Mathf.Clamp(reach * U(r, 0.45f, 0.75f), 1.4f, 5f), hv = Mathf.Clamp(reach * U(r, 0.3f, 0.5f), 1.2f, 3.5f);
            Vector2 at = c + along * U(r, -0.2f, 0.2f) * reach + across * U(r, -0.2f, 0.2f) * reach;
            m.Box(at, along, across, hu, hv, y, y + U(r, 2.6f, 4.2f), s.family == Family.Glass ? clad : s.wallLayer,
                  s.family == Family.Glass ? metal : s.wallTint * 0.95f);
        }
        for (int k = 0; k < s.plant; k++)
        {
            float ang = U(r, 0f, 6.2832f), rad = U(r, 0.35f, 0.85f) * reach;
            Vector2 at = c + new Vector2(Mathf.Cos(ang), Mathf.Sin(ang)) * rad;
            m.Box(at, along, across, U(r, 0.6f, 1.3f), U(r, 0.45f, 0.8f), y, y + U(r, 0.9f, 1.6f), clad, metal * U(r, 0.8f, 1.05f));
        }
        for (int k = 0; k < s.tank; k++)
        {
            float ang = U(r, 0f, 6.2832f), rad = U(r, 0.3f, 0.7f) * reach;
            Vector2 at = c + new Vector2(Mathf.Cos(ang), Mathf.Sin(ang)) * rad;
            float tr = Mathf.Min(U(r, 1.1f, 1.5f), reach * 0.6f);
            m.Box(at, along, across, tr * 0.7f, tr * 0.7f, y, y + 1.3f, clad, dark);
            m.Drum(at, tr, y + 1.3f, y + 1.3f + U(r, 2.2f, 2.9f), clad, s.Paint(Rgb(0.42f, 0.30f, 0.21f)));
        }
        for (int k = 0; k < s.mast; k++)
        {
            float ang = U(r, 0f, 6.2832f), rad = U(r, 0.1f, 0.6f) * reach;
            Vector2 at = c + new Vector2(Mathf.Cos(ang), Mathf.Sin(ang)) * rad;
            float h = s.family == Family.Glass ? U(r, 8f, 15f) : U(r, 4f, 8f);
            m.Box(at, along, across, 0.09f, 0.09f, y, y + h, clad, dark);
        }
    }

    /// <summary>A convex polygon (counter-clockwise seen from above) moved out by <paramref name="d"/> (in, if
    /// negative), its corners mitred.</summary>
    private static List<Vector2> Offset(List<Vector2> poly, float d)
    {
        int n = poly.Count;
        var result = new List<Vector2>(n);
        for (int i = 0; i < n; i++)
        {
            Vector2 prev = poly[(i + n - 1) % n], at = poly[i], next = poly[(i + 1) % n];
            Vector2 e0 = (at - prev).normalized, e1 = (next - at).normalized;
            Vector2 n0 = new Vector2(e0.y, -e0.x), n1 = new Vector2(e1.y, -e1.x);
            float k = Mathf.Max(1f + Vector2.Dot(n0, n1), 0.3f);
            result.Add(at + (n0 + n1) * (d / k));
        }
        return result;
    }

    /// <summary>Whether <paramref name="inner"/>, an inward offset of <paramref name="outer"/>, kept every edge the right
    /// way round and long enough to be a wall.</summary>
    private static bool Sound(List<Vector2> outer, List<Vector2> inner)
    {
        for (int i = 0; i < outer.Count; i++)
        {
            Vector2 e = outer[(i + 1) % outer.Count] - outer[i], f = inner[(i + 1) % inner.Count] - inner[i];
            if (Vector2.Dot(e.normalized, f) < 1.5f) return false;
        }
        return true;
    }

    /// <summary>The mesh as it grows: every vertex carries what the shader needs to know of its face.</summary>
    private sealed class Buffer
    {
        private readonly List<Vector3> v = new List<Vector3>(), n = new List<Vector3>();
        private readonly List<Vector4> t = new List<Vector4>();
        private readonly List<Color32> c = new List<Color32>();
        private readonly List<Vector4>[] uv = new List<Vector4>[8];
        private readonly List<int> tris = new List<int>();
        private readonly Vector4[] face = new Vector4[8];
        private Color32 colour;
        private readonly Spec s;

        public Buffer(Spec spec)
        {
            s = spec;
            for (int k = 0; k < 8; k++) uv[k] = new List<Vector4>();
        }

        private static Color32 Encode(Color tint, bool plain)
        {
            // square roots: eight bits a channel go further that way, and the shader squares them
            return new Color32((byte)Mathf.RoundToInt(255f * Mathf.Sqrt(Mathf.Clamp01(tint.r))),
                               (byte)Mathf.RoundToInt(255f * Mathf.Sqrt(Mathf.Clamp01(tint.g))),
                               (byte)Mathf.RoundToInt(255f * Mathf.Sqrt(Mathf.Clamp01(tint.b))), plain ? (byte)255 : (byte)0);
        }

        /// <summary>The quads that follow are a facade of <paramref name="bays"/> bays along an edge
        /// <paramref name="length"/> long (0 bays: a blank wall).</summary>
        public void Facade(float length, int bays, Storey st)
        {
            colour = Encode(s.wallTint, false);
            face[0] = new Vector4(0f, 0f, length, bays);
            face[1] = new Vector4(st.first, st.floorH, st.winTop, s.reveal);
            face[2] = new Vector4(s.wallLayer, s.seed, (float)s.ground, (float)s.window);
            face[3] = new Vector4(s.winW, s.winH, s.sill, s.mullions + 10 * s.transoms);
            face[4] = new Vector4(s.trim.r, s.trim.g, s.trim.b, s.baseLayer);
            face[5] = new Vector4(s.glass.r, s.glass.g, s.glass.b, s.glassF0);
            face[6] = new Vector4(s.frame.r, s.frame.g, s.frame.b, s.blinds);
            face[7] = new Vector4(s.baseTint.r, s.baseTint.g, s.baseTint.b, s.shopBays + 8 * s.DoorSeed);
        }

        /// <summary>The quads that follow are plain surfaces in wall <paramref name="layer"/>.</summary>
        public void Plain(int layer, Color tint)
        {
            colour = Encode(tint, true);
            for (int k = 0; k < 8; k++) face[k] = Vector4.zero;
            face[2] = new Vector4(layer, s.seed, 0f, 0f);
        }

        /// <summary>A quad p0, p1, p2, p3 with p0 to p1 its texture's u and p0 to p3 its v, facing cross(v, u); the
        /// texture is laid in metres from (<paramref name="s0"/>, <paramref name="y0"/>) at p0.</summary>
        public void Quad(Vector3 p0, Vector3 p1, Vector3 p2, Vector3 p3, float s0, float y0)
        {
            Vector3 u = p1 - p0, w = p3 - p0;
            Vector3 normal = Vector3.Cross(w, u);
            if (normal.sqrMagnitude < 1e-10f) return;
            normal.Normalize();
            Vector3 tangent = u.normalized, bitangent = -Vector3.Cross(normal, tangent);
            int i = v.Count;
            foreach (Vector3 p in new[] { p0, p1, p2, p3 })
            {
                v.Add(p);
                n.Add(normal);
                t.Add(new Vector4(tangent.x, tangent.y, tangent.z, -1f));
                c.Add(colour);
                Vector3 d = p - p0;
                uv[0].Add(new Vector4(s0 + Vector3.Dot(d, tangent), y0 + Vector3.Dot(d, bitangent), face[0].z, face[0].w));
                for (int k = 1; k < 8; k++) uv[k].Add(face[k]);
            }
            Tri(i, i + 1, i + 2, normal);
            Tri(i, i + 2, i + 3, normal);
        }

        /// <summary>A triangle wound to face <paramref name="facing"/> (Unity's front face: clockwise seen from it).</summary>
        private void Tri(int a, int b, int cc, Vector3 facing)
        {
            if (Vector3.Dot(Vector3.Cross(v[b] - v[a], v[cc] - v[a]), facing) < 0f) (b, cc) = (cc, b);
            tris.Add(a);
            tris.Add(b);
            tris.Add(cc);
        }

        /// <summary>A flat roof over <paramref name="poly"/> at <paramref name="y"/>: never seen by a drone, which flies
        /// below it, but it closes the building for the sun's shadow and the sky's.</summary>
        public void Roof(List<Vector2> poly, float y, Spec spec)
        {
            Plain(spec.trimLayer, new Color(0.5f, 0.5f, 0.5f));
            int i = v.Count;
            foreach (Vector2 p in poly)
            {
                v.Add(At(p, y));
                n.Add(Vector3.up);
                t.Add(new Vector4(1f, 0f, 0f, -1f));
                c.Add(colour);
                uv[0].Add(new Vector4(p.x, p.y, 0f, 0f));
                for (int k = 1; k < 8; k++) uv[k].Add(face[k]);
            }
            for (int k = 1; k + 1 < poly.Count; k++) Tri(i, i + k, i + k + 1, Vector3.up);
        }

        /// <summary>A band standing <paramref name="depth"/> out of the walls all round, from <paramref name="y0"/> to
        /// <paramref name="y1"/>: its face, and its underside and top if asked.</summary>
        public void Ring(List<Vector2> poly, float y0, float y1, float depth, int layer, Color tint, bool top, bool bottom)
        {
            Plain(layer, tint);
            List<Vector2> outer = Offset(poly, depth);
            for (int i = 0; i < poly.Count; i++)
            {
                int j = (i + 1) % poly.Count;
                Vector2 a = poly[i], b = poly[j], ao = outer[i], bo = outer[j];
                if ((b - a).sqrMagnitude < 1e-6f) continue;
                Quad(At(ao, y0), At(bo, y0), At(bo, y1), At(ao, y1), 0f, y0);
                if (bottom) Quad(At(bo, y0), At(ao, y0), At(a, y0), At(b, y0), 0f, 0f);
                if (top) Quad(At(ao, y1), At(bo, y1), At(b, y1), At(a, y1), 0f, 0f);
            }
        }

        /// <summary>A fin on a wall: from <paramref name="foot"/> (on the wall line) <paramref name="width"/> along
        /// <paramref name="dir"/> and <paramref name="depth"/> out along <paramref name="outward"/>.</summary>
        public void Fin(Vector2 foot, Vector2 dir, Vector2 outward, float width, float depth, float y0, float y1, int layer, Color tint)
        {
            Plain(layer, tint);
            Vector2 a = foot, b = foot + dir * width, ao = a + outward * depth, bo = b + outward * depth;
            Quad(At(ao, y0), At(bo, y0), At(bo, y1), At(ao, y1), 0f, y0);
            Quad(At(a, y0), At(ao, y0), At(ao, y1), At(a, y1), 0f, y0);
            Quad(At(bo, y0), At(b, y0), At(b, y1), At(bo, y1), 0f, y0);
        }

        /// <summary>A balcony along a wall: a slab at the windows' sill and a rail a metre high round it.</summary>
        public void Balcony(Vector2 foot, Vector2 dir, Vector2 outward, float width, float depth, float y, int layer, Color slab, Color rail)
        {
            Vector2 a = foot, b = foot + dir * width, ao = a + outward * depth, bo = b + outward * depth;
            float y0 = y - 0.16f, y1 = y + 1.0f;
            Plain(layer, slab);
            Quad(At(ao, y0), At(bo, y0), At(bo, y), At(ao, y), 0f, y0);                 // the slab's edge
            Quad(At(bo, y0), At(ao, y0), At(a, y0), At(b, y0), 0f, 0f);                 // its underside
            Quad(At(ao, y), At(bo, y), At(b, y), At(a, y), 0f, 0f);                     // its top
            Quad(At(a, y0), At(ao, y0), At(ao, y), At(a, y), 0f, y0);
            Quad(At(bo, y0), At(b, y0), At(b, y), At(bo, y), 0f, y0);
            Plain(layer, rail);
            Quad(At(ao, y), At(bo, y), At(bo, y1), At(ao, y1), 0f, y);                  // the rail, outside and inside
            Quad(At(bo, y), At(ao, y), At(ao, y1), At(bo, y1), 0f, y);
            Quad(At(a, y), At(ao, y), At(ao, y1), At(a, y1), 0f, y);                    // its ends
            Quad(At(ao, y), At(a, y), At(a, y1), At(ao, y1), 0f, y);
            Quad(At(bo, y), At(b, y), At(b, y1), At(bo, y1), 0f, y);
            Quad(At(b, y), At(bo, y), At(bo, y1), At(b, y1), 0f, y);
        }

        /// <summary>A box on a roof: four sides and a top.</summary>
        public void Box(Vector2 at, Vector2 u, Vector2 w, float halfU, float halfW, float y0, float y1, int layer, Color tint)
        {
            Plain(layer, tint);
            Vector2 p0 = at - u * halfU - w * halfW, p1 = at + u * halfU - w * halfW, p2 = at + u * halfU + w * halfW, p3 = at - u * halfU + w * halfW;
            var ring = new[] { p0, p1, p2, p3 };
            // w is the right of u, so p0, p1, p2, p3 runs clockwise seen from above: each side's outside is its left
            for (int k = 0; k < 4; k++)
            {
                Vector2 a = ring[(k + 1) % 4], b = ring[k];
                Quad(At(a, y0), At(b, y0), At(b, y1), At(a, y1), 0f, y0);
            }
            Quad(At(p0, y1), At(p3, y1), At(p2, y1), At(p1, y1), 0f, 0f);
        }

        /// <summary>A water tank: eight staves and a low cone.</summary>
        public void Drum(Vector2 at, float radius, float y0, float y1, int layer, Color tint)
        {
            Plain(layer, tint);
            const int sides = 8;
            for (int k = 0; k < sides; k++)
            {
                // counter-clockwise seen from above, as a footprint: the outside is on the right of a to b
                float a0 = k * 6.2831853f / sides, a1 = (k + 1) * 6.2831853f / sides;
                Vector2 a = at + new Vector2(Mathf.Cos(a0), Mathf.Sin(a0)) * radius, b = at + new Vector2(Mathf.Cos(a1), Mathf.Sin(a1)) * radius;
                Quad(At(a, y0), At(b, y0), At(b, y1), At(a, y1), k * radius * 0.78f, y0);
                Vector3 apex = At(at, y1 + 0.5f * radius);
                Quad(At(a, y1), At(b, y1), apex, apex, 0f, 0f);
            }
        }

        public Mesh ToMesh(string name)
        {
            var mesh = new Mesh { name = name };
            if (v.Count > 65000) mesh.indexFormat = UnityEngine.Rendering.IndexFormat.UInt32;
            mesh.SetVertices(v);
            mesh.SetNormals(n);
            mesh.SetTangents(t);
            mesh.SetColors(c);
            for (int k = 0; k < 8; k++) mesh.SetUVs(k, uv[k]);
            mesh.SetTriangles(tris, 0);
            return mesh;
        }
    }
}
