using System.Collections;
using System.Collections.Generic;
using UnityEngine;
#if UNITY_EDITOR
using UnityEditor;
#endif

// Moves a set of walker prefabs around the object they're attached to. Three path
// shapes are available:
//   - Rectangle: a rounded rectangle sized from the Collider on the same
//     GameObject, expanded by a random amount in [0, margin], so side lengths
//     range from the collider size up to that plus the margin. Corners are
//     rounded so walkers can smoothly make the turns.
//   - Circle:    the original behaviour — each walker orbits at a random radius
//     in [minRadius, maxRadius] around the centre.
//   - Polygon:   the building's own outline (`polygon`, convex: a block-shaped
//     building's footprint, set by ProceduralCityBuilder), each walker keeping
//     its gap off every face and rounding each corner at that gap, so it follows
//     the footpath round a block of any shape. The gap is drawn as Rectangle's.
public class WalkerPatrol : MonoBehaviour
{
    public enum PathShape { Rectangle, Circle, Polygon }

    [Header("General")]
    public PathShape shape = PathShape.Rectangle;
    public GameObject walkerPrefab;
    [Tooltip("Optional height tweak added on top of the collider base (ground level). " +
             "Usually 0; use it to correct for the walker prefab's pivot.")]
    public float verticalOffset = 0f;
    public int numWalkers = 12;
    public float speedMetersPerSec = 2.0f;

    [Header("Rectangle and Polygon Parameters")]
    [Tooltip("Fixed gap (metres) between the collider edge and the nearest path.")]
    public float baseOffset = 0.5f;
    [Tooltip("Extra distance (metres) added on top of the base offset. " +
             "Each walker gets a random expansion in [0, margin].")]
    public float margin = 3.0f;
    [Tooltip("Radius of the rounded corners (metres). Clamped to fit the rectangle.")]
    public float cornerRadius = 1.0f;

    [Header("Circle Parameters")]
    public float minRadius = 1.5f;
    public float maxRadius = 3.0f;

    [Header("Polygon Parameters")]
    [Tooltip("The outline walked round, world (x, z), convex, either winding. Each corner is rounded at the walker's " +
             "gap, so the path keeps that gap off the whole building.")]
    public Vector2[] polygon;

    // Set via RequestSpecial() by GoalSpecialWalker before Start runs: one of this
    // patrol's walkers is spawned from this prefab instead of walkerPrefab and
    // tagged with a SpecialWalker marker. Null = all walkers are ordinary.
    private GameObject specialWalkerPrefab;
    private int requestedSpecialIndex = -1;   // which walker; -1 = drawn here

    // Set via SetSeed() by GoalSpecialWalker (a stream of the session's goal seed, see
    // GoalSession): every draw in Start then comes from it, so a replayed seed starts
    // each walker where it started. Unseeded patrols draw from UnityEngine.Random.
    private int seed;
    private bool seeded;

    private class WalkerInfo
    {
        public GameObject walker;
        public Vector3 previousPosition;
        public bool clockwise;
        public float speed;          // linear speed (m/s)

        // Rectangle path
        public float expand;         // this walker's gap from the collider edge (metres)
        public float distance;       // arc length travelled along the path
        public float halfW;          // half side length along local X
        public float halfD;          // half side length along local Z
        public float cornerRadius;   // corner radius for this walker's path

        // Circle path
        public float currentAngle;   // degrees
        public float orbitRadius;
        public float angularSpeed;   // degrees per second
    }

    private List<WalkerInfo> walkers = new List<WalkerInfo>();
    private GameObject walkersContainer;
    private Collider footprint;

    // Polygon path: the outline counter-clockwise, each edge's length, direction and outward normal, and the turn
    // from each edge onto the next (the arc a walker rounds that corner by, at its own gap).
    private Vector2[] corners;
    private Vector2[] edgeDirs, edgeNormals;
    private float[] edgeLengths, turns;
    private float outlineLength, totalTurn;

#if UNITY_EDITOR
    // Auto-assign the SimpleWalker prefab so it doesn't have to be dragged onto
    // every instance. Runs when the component is added or edited in the editor;
    // the resolved reference is then serialized normally for builds.
    void Reset() { AssignDefaultWalkerPrefab(); }
    void OnValidate() { AssignDefaultWalkerPrefab(); }

    private void AssignDefaultWalkerPrefab()
    {
        if (walkerPrefab != null) return;
        string[] guids = AssetDatabase.FindAssets("SimpleWalker t:Prefab");
        foreach (string guid in guids)
        {
            string path = AssetDatabase.GUIDToAssetPath(guid);
            if (System.IO.Path.GetFileNameWithoutExtension(path) == "SimpleWalker")
            {
                walkerPrefab = AssetDatabase.LoadAssetAtPath<GameObject>(path);
                break;
            }
        }
    }
#endif

    // Called by GoalSpecialWalker (in its Awake, before this Start) to request that
    // exactly one of this patrol's walkers use the visually-distinct prefab: walker
    // `index` (spawn order), or one drawn here when it is out of range.
    public void RequestSpecial(GameObject prefab, int index = -1)
    {
        specialWalkerPrefab = prefab;
        requestedSpecialIndex = index;
    }

    // Called by GoalSpecialWalker before this Start: draw every random choice below from `value`.
    public void SetSeed(int value)
    {
        seed = value;
        seeded = true;
    }

    void Start()
    {
        if (walkerPrefab == null)
        {
            Debug.LogError("Walker Prefab is not assigned!");
            return;
        }

        System.Random rng = seeded ? new System.Random(seed) : null;
        float Rand(float min, float max) => rng != null ? GoalSession.Range(rng, min, max) : Random.Range(min, max);

        // Pick which walker (if any) is the special one. -1 = none.
        int specialIndex = -1;
        if (specialWalkerPrefab != null && numWalkers > 0)
        {
            specialIndex = requestedSpecialIndex >= 0 && requestedSpecialIndex < numWalkers
                ? requestedSpecialIndex
                : (rng != null ? rng.Next(numWalkers) : Random.Range(0, numWalkers));
        }

        footprint = GetComponent<Collider>();
        if (shape == PathShape.Rectangle && footprint == null)
        {
            Debug.LogError("WalkerPatrol Rectangle mode requires a Collider on the same GameObject to size the rectangle!");
            return;
        }
        if (shape == PathShape.Polygon && !PrepareOutline())
        {
            Debug.LogError("WalkerPatrol Polygon mode requires a polygon of three or more distinct points!");
            return;
        }

        // Base half-extents come from the collider's world-axis-aligned bounds.
        Vector3 baseExtents = footprint != null ? footprint.bounds.extents : Vector3.zero;

        // Create a container for all walkers. It hangs beside the building rather than under it, so
        // resizing the building (BuildingWidthTuner) moves only the path, never the walkers' own scale:
        // under the building, the scale compensation taken here would go stale and stretch them.
        walkersContainer = new GameObject("Walkers (" + name + ")");
        walkersContainer.transform.SetParent(transform.parent, false);
        walkersContainer.transform.position = transform.position;

        // The container hangs inside a city tile, and the tuners treat every Renderer under a tile as
        // block content to be scaled about the kerb. A walker's renderers are children of the object the
        // patrol drives, so that pulls the walker apart rather than moving it: invisibly for the skinned
        // body, which renders from its bones, and by metres for a hat.
        walkersContainer.AddComponent<SpawnedContentRoot>();

        Vector3 center = GetCenter();

        for (int i = 0; i < numWalkers; i++)
        {
            WalkerInfo info = new WalkerInfo();
            info.speed = speedMetersPerSec;
            info.clockwise = i < numWalkers / 2;

            Vector3 position;

            if (shape == PathShape.Rectangle)
            {
                // Start a fixed baseOffset out from the collider edge, then
                // expand outward by a random amount so each side sits in
                // [collider + baseOffset, collider + baseOffset + margin].
                info.expand = baseOffset + Rand(0f, margin) * 0.5f;
                SizeRectangle(info, baseExtents);

                // Random start position along the perimeter.
                info.distance = Rand(0f, Perimeter(info));
                position = center + GetLocalXZ(info, info.distance);
            }
            else if (shape == PathShape.Polygon)
            {
                // As the rectangle: a gap off the facade in [baseOffset, baseOffset + margin / 2], then a start.
                info.expand = baseOffset + Rand(0f, margin) * 0.5f;
                info.distance = Rand(0f, OutlinePerimeter(info.expand));
                position = GetOutlinePoint(info.expand, info.distance, center.y);
            }
            else
            {
                info.orbitRadius = Rand(minRadius, maxRadius);
                info.currentAngle = Rand(0f, 360f);
                info.angularSpeed = (speedMetersPerSec * 360f) / (2f * Mathf.PI * info.orbitRadius);
                position = center + GetCircleXZ(info, info.currentAngle);
            }

            bool isSpecial = i == specialIndex;
            GameObject prefab = isSpecial ? specialWalkerPrefab : walkerPrefab;
            info.walker = Instantiate(prefab, position, Quaternion.identity, walkersContainer.transform);
            info.walker.name = isSpecial ? "Walker_" + i + "_Special" : "Walker_" + i;
            if (isSpecial)
            {
                SpecialWalker marker = info.walker.GetComponent<SpecialWalker>();
                if (marker == null) marker = info.walker.AddComponent<SpecialWalker>();

                // The instance is renamed above, so the prefab is the only place left
                // to read a hat name from once this walker is in the scene.
                marker.EnsureHatId(prefab.name);
            }
            info.previousPosition = position;

            walkers.Add(info);
        }
    }

    // The container is no longer a child, so it has to be cleaned up with the building explicitly.
    void OnDestroy()
    {
        if (walkersContainer != null) Destroy(walkersContainer);
    }

    void Update()
    {
        Vector3 center = GetCenter();

        // Re-read every frame so the path follows a building resized after Start — BuildingWidthTuner
        // sizes goal patches on their first frame, and its goal width can be dragged during Play.
        Vector3 extents = footprint != null ? footprint.bounds.extents : Vector3.zero;

        foreach (WalkerInfo info in walkers)
        {
            if (info.walker == null) continue;

            Vector3 newPosition;

            if (shape == PathShape.Rectangle)
            {
                SizeRectangle(info, extents);

                // Advance along the path (negative direction for clockwise).
                float step = info.speed * Time.deltaTime;
                info.distance += info.clockwise ? -step : step;
                newPosition = center + GetLocalXZ(info, info.distance);
            }
            else if (shape == PathShape.Polygon)
            {
                float step = info.speed * Time.deltaTime;
                info.distance += info.clockwise ? -step : step;
                newPosition = GetOutlinePoint(info.expand, info.distance, center.y);
            }
            else
            {
                float angleChange = info.angularSpeed * Time.deltaTime;
                info.currentAngle += info.clockwise ? -angleChange : angleChange;
                info.currentAngle %= 360f;
                if (info.currentAngle < 0) info.currentAngle += 360f;
                newPosition = center + GetCircleXZ(info, info.currentAngle);
            }

            // Face the direction of travel.
            Vector3 moveDirection = (newPosition - info.previousPosition).normalized;
            if (moveDirection != Vector3.zero)
            {
                info.walker.transform.rotation = Quaternion.LookRotation(moveDirection);
            }

            info.walker.transform.position = newPosition;
            info.previousPosition = newPosition;
        }
    }

    private Vector3 GetCenter()
    {
        // Ground the walkers on the base of the collider (bounds.min.y) rather than
        // its centre, so they sit on the floor regardless of the building's height.
        // verticalOffset is then just an optional tweak (e.g. for the walker pivot).
        if (footprint != null)
        {
            Bounds b = footprint.bounds;
            return new Vector3(b.center.x, b.min.y + verticalOffset, b.center.z);
        }
        Vector3 t = transform.position;
        return new Vector3(t.x, t.y + verticalOffset, t.z);
    }

    // Sides from the collider's world-axis-aligned half-extents plus this walker's gap.
    private void SizeRectangle(WalkerInfo info, Vector3 extents)
    {
        info.halfW = extents.x + info.expand;
        info.halfD = extents.z + info.expand;

        // Corner radius can't exceed half of the shorter side.
        info.cornerRadius = Mathf.Min(cornerRadius, info.halfW, info.halfD);
        info.cornerRadius = Mathf.Max(info.cornerRadius, 0.001f);
    }

    private static Vector3 GetCircleXZ(WalkerInfo info, float angleDeg)
    {
        float radians = -angleDeg * Mathf.Deg2Rad;
        return new Vector3(info.orbitRadius * Mathf.Cos(radians), 0f, info.orbitRadius * Mathf.Sin(radians));
    }

    private static float Perimeter(WalkerInfo info)
    {
        float lengthV = 2f * (info.halfD - info.cornerRadius);
        float lengthH = 2f * (info.halfW - info.cornerRadius);
        float arcs = 2f * Mathf.PI * info.cornerRadius; // four quarter-circles
        return 2f * lengthV + 2f * lengthH + arcs;
    }

    // Returns the local (x, 0, z) offset from the centre for a given arc length
    // along a counter-clockwise rounded rectangle.
    private static Vector3 GetLocalXZ(WalkerInfo info, float s)
    {
        float halfW = info.halfW;
        float halfD = info.halfD;
        float r = info.cornerRadius;

        float lengthV = 2f * (halfD - r);   // vertical (left/right) straight edges
        float lengthH = 2f * (halfW - r);   // horizontal (top/bottom) straight edges
        float arc = Mathf.PI * 0.5f * r;    // quarter-circle length

        float perimeter = 2f * lengthV + 2f * lengthH + 4f * arc;
        s = Mathf.Repeat(s, perimeter);

        float x, z;

        // 1. Right edge, travelling +z
        if (s < lengthV)
        {
            x = halfW;
            z = -(halfD - r) + s;
            return new Vector3(x, 0f, z);
        }
        s -= lengthV;

        // 2. Top-right corner, 0 -> 90 deg
        if (s < arc)
        {
            float a = s / r;
            x = (halfW - r) + r * Mathf.Cos(a);
            z = (halfD - r) + r * Mathf.Sin(a);
            return new Vector3(x, 0f, z);
        }
        s -= arc;

        // 3. Top edge, travelling -x
        if (s < lengthH)
        {
            x = (halfW - r) - s;
            z = halfD;
            return new Vector3(x, 0f, z);
        }
        s -= lengthH;

        // 4. Top-left corner, 90 -> 180 deg
        if (s < arc)
        {
            float a = Mathf.PI * 0.5f + s / r;
            x = -(halfW - r) + r * Mathf.Cos(a);
            z = (halfD - r) + r * Mathf.Sin(a);
            return new Vector3(x, 0f, z);
        }
        s -= arc;

        // 5. Left edge, travelling -z
        if (s < lengthV)
        {
            x = -halfW;
            z = (halfD - r) - s;
            return new Vector3(x, 0f, z);
        }
        s -= lengthV;

        // 6. Bottom-left corner, 180 -> 270 deg
        if (s < arc)
        {
            float a = Mathf.PI + s / r;
            x = -(halfW - r) + r * Mathf.Cos(a);
            z = -(halfD - r) + r * Mathf.Sin(a);
            return new Vector3(x, 0f, z);
        }
        s -= arc;

        // 7. Bottom edge, travelling +x
        if (s < lengthH)
        {
            x = -(halfW - r) + s;
            z = -halfD;
            return new Vector3(x, 0f, z);
        }
        s -= lengthH;

        // 8. Bottom-right corner, 270 -> 360 deg
        {
            float a = Mathf.PI * 1.5f + s / r;
            x = (halfW - r) + r * Mathf.Cos(a);
            z = -(halfD - r) + r * Mathf.Sin(a);
            return new Vector3(x, 0f, z);
        }
    }

    // The polygon without repeated points, made counter-clockwise, and its edges and turns. False if fewer than
    // three distinct points remain. A reflex corner (a turn to the right) is taken as no turn: the outline is
    // meant to be convex, and there the path would have to jump.
    private bool PrepareOutline()
    {
        var pts = new List<Vector2>();
        foreach (Vector2 p in polygon ?? new Vector2[0])
        {
            if (pts.Count == 0 || (p - pts[pts.Count - 1]).sqrMagnitude > 1e-6f) pts.Add(p);
        }
        while (pts.Count > 1 && (pts[0] - pts[pts.Count - 1]).sqrMagnitude <= 1e-6f) pts.RemoveAt(pts.Count - 1);
        if (pts.Count < 3) return false;
        float area = 0f;
        for (int i = 0; i < pts.Count; i++)
        {
            Vector2 a = pts[i], b = pts[(i + 1) % pts.Count];
            area += a.x * b.y - b.x * a.y;
        }
        if (area < 0f) pts.Reverse();

        int n = pts.Count;
        corners = pts.ToArray();
        edgeDirs = new Vector2[n];
        edgeNormals = new Vector2[n];
        edgeLengths = new float[n];
        turns = new float[n];
        outlineLength = 0f;
        for (int i = 0; i < n; i++)
        {
            Vector2 e = corners[(i + 1) % n] - corners[i];
            edgeLengths[i] = e.magnitude;
            edgeDirs[i] = e / edgeLengths[i];
            edgeNormals[i] = new Vector2(edgeDirs[i].y, -edgeDirs[i].x);   // right of the edge: outside, counter-clockwise
            outlineLength += edgeLengths[i];
        }
        totalTurn = 0f;
        for (int i = 0; i < n; i++)
        {
            Vector2 d0 = edgeDirs[i], d1 = edgeDirs[(i + 1) % n];
            turns[i] = Mathf.Max(Mathf.Atan2(d0.x * d1.y - d0.y * d1.x, Vector2.Dot(d0, d1)), 0f);
            totalTurn += turns[i];
        }
        return true;
    }

    // Length of the outline kept `gap` off the polygon: its edges, plus an arc of radius `gap` at every corner
    // (2 pi gap round a convex polygon).
    private float OutlinePerimeter(float gap) => outlineLength + gap * totalTurn;

    // The point at arc length s along the outline kept `gap` off the polygon, counter-clockwise from the start of
    // edge 0, in world (x, z) at height y: each edge moved out along its normal, then the arc round the corner at
    // its end onto the next edge's.
    private Vector3 GetOutlinePoint(float gap, float s, float y)
    {
        int n = corners.Length;
        s = Mathf.Repeat(s, OutlinePerimeter(gap));
        for (int i = 0; i < n; i++)
        {
            if (s < edgeLengths[i])
            {
                Vector2 p = corners[i] + edgeDirs[i] * s + edgeNormals[i] * gap;
                return new Vector3(p.x, y, p.y);
            }
            s -= edgeLengths[i];
            float arc = gap * turns[i];
            if (s < arc)
            {
                float a = Mathf.Atan2(edgeNormals[i].y, edgeNormals[i].x) + s / gap;
                Vector2 p = corners[(i + 1) % n] + new Vector2(Mathf.Cos(a), Mathf.Sin(a)) * gap;
                return new Vector3(p.x, y, p.y);
            }
            s -= arc;
        }
        Vector2 q = corners[0] + edgeNormals[0] * gap;                   // s rounded onto the very end
        return new Vector3(q.x, y, q.y);
    }
}
