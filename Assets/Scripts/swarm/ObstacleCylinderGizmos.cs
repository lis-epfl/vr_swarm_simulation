using UnityEngine;

// Scene-view visualiser for the cylindrical obstacle model OlfatiSaber builds in
// ComputeObstacleFrame. Drop it on any GameObject in the scene (an empty one is fine).
//
// It exists because the cylinder is an *approximation* of the collider and the error is invisible
// from the numbers alone: the radius is the circumradius of the axis-aligned bounds, so a square
// footprint inflates by root-2 and a long wall becomes an enormous circle. Drawing the cylinder
// against the bounds against the level geometry is the only quick way to see whether that
// approximation is acceptable for a given obstacle.
//
// [ExecuteAlways] plus the static geometry helpers mean this draws in edit mode too, before any
// drone exists -- which is when you want to check the level.
[ExecuteAlways]
public class ObstacleCylinderGizmos : MonoBehaviour
{
    [Header("What to draw")]
    [Tooltip("The cylinder each obstacle is approximated by.")]
    public bool drawCylinders = true;
    [Tooltip("The axis-aligned bounds the radius is derived from. The gap between this and the " +
             "cylinder is the inflation you are paying for.")]
    public bool drawBounds = true;
    [Tooltip("Cylinder at radius + d_obs: the surface where phi_beta switches on.")]
    public bool drawRepulsionShell = true;
    [Tooltip("Per-drone frame: outward/tangent axes, nearest surface point, and the forces.")]
    public bool drawDroneFrames = true;
    [Tooltip("Sphere of radius r0_obs around each drone -- the OverlapSphere query.")]
    public bool drawQueryRange = false;

    [Header("Drones")]
    [Tooltip("Draw every drone in the scene rather than the one named below.")]
    public bool allDrones = false;
    [Tooltip("Resolved as SwarmParent/Drone N, matching the other swarm debug scripts.")]
    public int selectedDroneIndex = 0;

    [Header("Parameters")]
    [Tooltip("Left empty, the first OlfatiSaber in the scene is used. In edit mode there is usually " +
             "none, so the fallbacks below are used instead -- keep them matching the drone prefab.")]
    public OlfatiSaber parameterSource;
    public Vector3 fallbackCylinderAxis = Vector3.up;
    public float fallbackScaleFactor = 10.0f;
    public float fallbackDObs = 5.0f;
    public float fallbackR0Obs = 6.0f;

    [Header("Appearance")]
    [Range(8, 96)] public int circleSegments = 48;
    [Tooltip("Lines joining the two end circles.")]
    [Range(0, 16)] public int spokes = 4;
    [Tooltip("World units drawn per unit of acceleration. Force arrows only.")]
    public float forceArrowScale = 20.0f;
    public bool showLabels = true;

    public Color cylinderColour = new Color(0.2f, 0.9f, 1.0f, 1.0f);
    public Color boundsColour = new Color(0.5f, 0.5f, 0.5f, 0.5f);
    public Color shellColour = new Color(1.0f, 0.8f, 0.2f, 0.35f);
    public Color outwardColour = new Color(1.0f, 0.3f, 0.3f, 1.0f);
    public Color tangentColour = new Color(0.3f, 1.0f, 0.4f, 1.0f);
    public Color repulsionColour = new Color(1.0f, 0.3f, 1.0f, 1.0f);
    public Color velocityMatchColour = new Color(0.4f, 0.6f, 1.0f, 1.0f);
    public Color totalForceColour = Color.white;

    private const string k_ObstacleLayerName = "Obstacle";
    private const float k_ColliderRefreshInterval = 1.0f;

    private Collider[] _obstacles = new Collider[0];
    private float _nextColliderRefresh;

    // Resolved once per draw so every helper agrees on which source won.
    private Vector3 _axis;
    private float _scaleFactor;
    private float _dObs;

    void OnDrawGizmos()
    {
        ResolveParameters();

        if (drawCylinders || drawBounds || drawRepulsionShell)
        {
            RefreshObstacles();
            foreach (Collider obstacle in _obstacles)
            {
                if (obstacle == null) continue;
                DrawObstacle(obstacle);
            }
        }

        if (drawDroneFrames || drawQueryRange)
            DrawDrones();
    }

    // The live component is preferred over the fallbacks for the same reason the debug scripts call
    // GetObstacleFrame rather than re-deriving it: a visualiser drawing different parameters from
    // the ones being flown is worse than no visualiser.
    private void ResolveParameters()
    {
        OlfatiSaber source = parameterSource != null ? parameterSource : FindAnyObjectByType<OlfatiSaber>();

        if (source != null)
        {
            _axis = source.cylinderAxis;
            _scaleFactor = source.ScaleFactor;
            _dObs = source.d_obs;
        }
        else
        {
            _axis = fallbackCylinderAxis;
            _scaleFactor = fallbackScaleFactor;
            _dObs = fallbackDObs;
        }

        if (_axis.sqrMagnitude < 1e-8f) _axis = Vector3.up;
        if (_scaleFactor <= 0.0f) _scaleFactor = 1.0f;
    }

    // FindObjectsByType rather than the algorithm's OverlapSphere, because the point of the global
    // pass is to show obstacles that are *not* currently in range of anything. Cached, since gizmos
    // redraw on every scene-view repaint.
    private void RefreshObstacles()
    {
        if (Time.realtimeSinceStartup < _nextColliderRefresh && _obstacles.Length > 0) return;
        _nextColliderRefresh = Time.realtimeSinceStartup + k_ColliderRefreshInterval;

        int layer = LayerMask.NameToLayer(k_ObstacleLayerName);
        Collider[] all = FindObjectsByType<Collider>(FindObjectsSortMode.None);

        int count = 0;
        for (int i = 0; i < all.Length; i++)
            if (all[i].gameObject.layer == layer) count++;

        _obstacles = new Collider[count];
        int write = 0;
        for (int i = 0; i < all.Length; i++)
            if (all[i].gameObject.layer == layer) _obstacles[write++] = all[i];
    }

    private void DrawObstacle(Collider obstacle)
    {
        OlfatiSaber.GetObstacleCylinder(obstacle, _axis,
                                        out Vector3 centre, out Vector3 axis,
                                        out float radius, out float halfHeight);

        if (drawBounds)
        {
            Gizmos.color = boundsColour;
            Gizmos.DrawWireCube(obstacle.bounds.center, obstacle.bounds.size);
        }

        if (drawCylinders)
        {
            Gizmos.color = cylinderColour;
            DrawCylinder(centre, axis, radius, halfHeight);
        }

        // d_obs is in swarm units; the world radius it corresponds to is d_obs * ScaleFactor.
        if (drawRepulsionShell)
        {
            Gizmos.color = shellColour;
            DrawCylinder(centre, axis, radius + _dObs * _scaleFactor, halfHeight);
        }

#if UNITY_EDITOR
        if (showLabels)
        {
            // Half-width of the bounds perpendicular to the axis, so the inflation the circumradius
            // costs is readable as a number and not just as a gap in the drawing.
            Vector3 extents = obstacle.bounds.extents;
            Vector3 perpExtents = extents - axis * Vector3.Dot(extents, axis);
            float widest = Mathf.Max(Mathf.Abs(perpExtents.x), Mathf.Max(Mathf.Abs(perpExtents.y), Mathf.Abs(perpExtents.z)));

            UnityEditor.Handles.color = cylinderColour;
            UnityEditor.Handles.Label(centre + axis * (halfHeight + 1.0f),
                $"{obstacle.name}\nR = {radius:F2} m (widest half-extent {widest:F2})");
        }
#endif
    }

    private void DrawDrones()
    {
        if (allDrones)
        {
            foreach (OlfatiSaber olfati in FindObjectsByType<OlfatiSaber>(FindObjectsSortMode.None))
                DrawDrone(olfati);
            return;
        }

        // Same resolution as DebugObstacleDistance / TuneOlfatiSaberObstacle.
        GameObject drone = GameObject.Find($"SwarmParent/Drone {selectedDroneIndex}");
        if (drone == null) return;

        Transform droneParent = drone.transform.Find("DroneParent");
        if (droneParent == null) return;

        DrawDrone(droneParent.GetComponent<OlfatiSaber>());
    }

    private void DrawDrone(OlfatiSaber olfati)
    {
        if (olfati == null) return;

        VelocityControl vc = olfati.GetComponent<VelocityControl>();
        if (vc == null) return;

        Vector3 position = vc.State.Position;
        // World-frame velocity, exactly as GetSwarmAcceleration builds it.
        Vector3 velocity = olfati.transform.TransformDirection(vc.State.VelocityVector);

        float queryRadius = olfati.r0_obs * olfati.ScaleFactor;

        if (drawQueryRange)
        {
            Gizmos.color = shellColour;
            Gizmos.DrawWireSphere(position, queryRadius);
        }

        if (!drawDroneFrames) return;

        // The algorithm's own query, so the gizmo shows exactly the obstacle set the force saw.
        Collider[] inRange = Physics.OverlapSphere(position, queryRadius, LayerMask.GetMask(k_ObstacleLayerName));
        foreach (Collider obstacle in inRange)
        {
            OlfatiSaber.ObstacleFrame frame = olfati.GetObstacleFrame(obstacle, position);
            if (!frame.valid) continue;

            // Nearest point on the cylinder surface, on the drone's own radial line.
            Vector3 axisFoot = frame.axisCentre + frame.axis * Vector3.Dot(position - frame.axisCentre, frame.axis);
            Vector3 surfacePoint = axisFoot + frame.outward * frame.radius;

            Gizmos.color = cylinderColour;
            Gizmos.DrawLine(position, surfacePoint);
            Gizmos.DrawLine(axisFoot, surfacePoint);

            // The two frame axes. The tangent is the whole point of the cylinder: it is the
            // direction a drone gets to keep, and on a bounding box it did not exist.
            float axisLength = Mathf.Max(2.0f, frame.radius * 0.4f);
            Gizmos.color = outwardColour;
            DrawArrow(position, position + frame.outward * axisLength);
            Gizmos.color = tangentColour;
            DrawArrow(position, position + frame.tangent * axisLength);

            olfati.GetObstacleContribution(frame, velocity,
                                           out Vector3 repulsion, out Vector3 velocityMatch);

            Gizmos.color = repulsionColour;
            DrawArrow(position, position + olfati.c_obs * repulsion * forceArrowScale);
            Gizmos.color = velocityMatchColour;
            DrawArrow(position, position + olfati.c_vm * velocityMatch * forceArrowScale);

#if UNITY_EDITOR
            if (showLabels)
            {
                UnityEditor.Handles.color = Color.white;
                UnityEditor.Handles.Label(surfacePoint,
                    $"d = {frame.distance:F2}  (d_obs {olfati.d_obs:F1})\n" +
                    $"mu = {frame.mu:F2}\n" +
                    $"phi = {olfati.GetObstacleRepulsion(frame.distance):F3}");
            }
#endif
        }

        // The summed force, drawn once rather than per obstacle. GetObstacleForce is public so this
        // is the real value and not a reimplementation of it.
        Gizmos.color = totalForceColour;
        DrawArrow(position, position + olfati.GetObstacleForce(position, velocity) * forceArrowScale);
    }

    private void DrawCylinder(Vector3 centre, Vector3 axis, float radius, float halfHeight)
    {
        Vector3 top = centre + axis * halfHeight;
        Vector3 bottom = centre - axis * halfHeight;

        DrawCircle(top, axis, radius);
        DrawCircle(bottom, axis, radius);

        if (spokes <= 0) return;

        BasisFor(axis, out Vector3 u, out Vector3 v);
        for (int i = 0; i < spokes; i++)
        {
            float t = (i / (float)spokes) * Mathf.PI * 2.0f;
            Vector3 offset = (u * Mathf.Cos(t) + v * Mathf.Sin(t)) * radius;
            Gizmos.DrawLine(bottom + offset, top + offset);
        }
    }

    private void DrawCircle(Vector3 centre, Vector3 axis, float radius)
    {
        BasisFor(axis, out Vector3 u, out Vector3 v);

        Vector3 previous = centre + u * radius;
        for (int i = 1; i <= circleSegments; i++)
        {
            float t = (i / (float)circleSegments) * Mathf.PI * 2.0f;
            Vector3 next = centre + (u * Mathf.Cos(t) + v * Mathf.Sin(t)) * radius;
            Gizmos.DrawLine(previous, next);
            previous = next;
        }
    }

    // Any orthonormal pair perpendicular to axis. The seed is swapped when axis is near-vertical so
    // the cross product never degenerates.
    private static void BasisFor(Vector3 axis, out Vector3 u, out Vector3 v)
    {
        Vector3 seed = Mathf.Abs(Vector3.Dot(axis.normalized, Vector3.up)) > 0.9f ? Vector3.right : Vector3.up;
        u = Vector3.Cross(axis, seed).normalized;
        v = Vector3.Cross(axis.normalized, u);
    }

    private static void DrawArrow(Vector3 from, Vector3 to)
    {
        Gizmos.DrawLine(from, to);

        Vector3 shaft = to - from;
        float length = shaft.magnitude;
        if (length < 1e-4f) return;

        Vector3 direction = shaft / length;
        BasisFor(direction, out Vector3 u, out Vector3 v);

        float head = Mathf.Min(length * 0.25f, 1.0f);
        Vector3 headBase = to - direction * head;
        Gizmos.DrawLine(to, headBase + u * head * 0.5f);
        Gizmos.DrawLine(to, headBase - u * head * 0.5f);
        Gizmos.DrawLine(to, headBase + v * head * 0.5f);
        Gizmos.DrawLine(to, headBase - v * head * 0.5f);
    }
}
