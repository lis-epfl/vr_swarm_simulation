using System.Collections;
using System.Collections.Generic;
using UnityEngine;
#if UNITY_EDITOR
using UnityEditor;
#endif

// Moves a set of walker prefabs around the object they're attached to. Two path
// shapes are available:
//   - Rectangle: a rounded rectangle sized from the Collider on the same
//     GameObject, expanded by a random amount in [0, margin], so side lengths
//     range from the collider size up to that plus the margin. Corners are
//     rounded so walkers can smoothly make the turns.
//   - Circle:    the original behaviour — each walker orbits at a random radius
//     in [minRadius, maxRadius] around the centre.
public class WalkerPatrol : MonoBehaviour
{
    public enum PathShape { Rectangle, Circle }

    [Header("General")]
    public PathShape shape = PathShape.Rectangle;
    public GameObject walkerPrefab;
    [Tooltip("Optional height tweak added on top of the collider base (ground level). " +
             "Usually 0; use it to correct for the walker prefab's pivot.")]
    public float verticalOffset = 0f;
    public int numWalkers = 12;
    public float speedMetersPerSec = 2.0f;

    [Header("Rectangle Parameters")]
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

    // Set via RequestSpecial() by GoalSpecialWalker before Start runs: one of this
    // patrol's walkers is spawned from this prefab instead of walkerPrefab and
    // tagged with a SpecialWalker marker. Null = all walkers are ordinary.
    private GameObject specialWalkerPrefab;

    private class WalkerInfo
    {
        public GameObject walker;
        public Vector3 previousPosition;
        public bool clockwise;
        public float speed;          // linear speed (m/s)

        // Rectangle path
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
    // exactly one of this patrol's walkers use the visually-distinct prefab.
    public void RequestSpecial(GameObject prefab)
    {
        specialWalkerPrefab = prefab;
    }

    void Start()
    {
        if (walkerPrefab == null)
        {
            Debug.LogError("Walker Prefab is not assigned!");
            return;
        }

        // Pick which walker (if any) is the special one. -1 = none.
        int specialIndex = (specialWalkerPrefab != null && numWalkers > 0)
            ? Random.Range(0, numWalkers)
            : -1;

        footprint = GetComponent<Collider>();
        if (shape == PathShape.Rectangle && footprint == null)
        {
            Debug.LogError("WalkerPatrol Rectangle mode requires a Collider on the same GameObject to size the rectangle!");
            return;
        }

        // Base half-extents come from the collider's world-axis-aligned bounds.
        Vector3 baseExtents = footprint != null ? footprint.bounds.extents : Vector3.zero;

        // Create a container for all walkers
        walkersContainer = new GameObject("Walkers");
        walkersContainer.transform.parent = transform;
        walkersContainer.transform.localPosition = Vector3.zero;

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
                float expand = baseOffset + Random.Range(0f, margin) * 0.5f;
                info.halfW = baseExtents.x + expand;
                info.halfD = baseExtents.z + expand;

                // Corner radius can't exceed half of the shorter side.
                info.cornerRadius = Mathf.Min(cornerRadius, info.halfW, info.halfD);
                info.cornerRadius = Mathf.Max(info.cornerRadius, 0.001f);

                // Random start position along the perimeter.
                info.distance = Random.Range(0f, Perimeter(info));
                position = center + GetLocalXZ(info, info.distance);
            }
            else
            {
                info.orbitRadius = Random.Range(minRadius, maxRadius);
                info.currentAngle = Random.Range(0f, 360f);
                info.angularSpeed = (speedMetersPerSec * 360f) / (2f * Mathf.PI * info.orbitRadius);
                position = center + GetCircleXZ(info, info.currentAngle);
            }

            bool isSpecial = i == specialIndex;
            GameObject prefab = isSpecial ? specialWalkerPrefab : walkerPrefab;
            info.walker = Instantiate(prefab, position, Quaternion.identity, walkersContainer.transform);
            info.walker.name = isSpecial ? "Walker_" + i + "_Special" : "Walker_" + i;
            if (isSpecial && info.walker.GetComponent<SpecialWalker>() == null)
            {
                info.walker.AddComponent<SpecialWalker>();
            }
            info.previousPosition = position;

            walkers.Add(info);
        }
    }

    void Update()
    {
        Vector3 center = GetCenter();

        foreach (WalkerInfo info in walkers)
        {
            if (info.walker == null) continue;

            Vector3 newPosition;

            if (shape == PathShape.Rectangle)
            {
                // Advance along the path (negative direction for clockwise).
                float step = info.speed * Time.deltaTime;
                info.distance += info.clockwise ? -step : step;
                newPosition = center + GetLocalXZ(info, info.distance);
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
}
