using UnityEngine;

// Scene-view visualiser for the cylindrical obstacle model OlfatiSaber builds in
// ComputeObstacleFrame. Drop it on any GameObject in the scene (an empty one is fine).
//
// It exists because the cylinder is an *approximation* of the collider and the error is invisible
// from the numbers alone: the radius is the circumradius of the axis-aligned bounds, so a square
// footprint inflates by root-2 and a long wall becomes an enormous circle.
//
// Everything here defaults off except the two things worth seeing at a glance -- the cylinders near
// the drone, and the force they produce. A city scene has hundreds of buildings and the obstacle
// shell is d_obs * ScaleFactor (50 m) across, so drawing every option at once is unreadable. Turn
// on one extra at a time.
//
// [ExecuteAlways] plus the static geometry helpers mean this draws in edit mode too, before any
// drone exists -- which is when you want to check the level.
[ExecuteAlways]
public class ObstacleCylinderGizmos : MonoBehaviour
{
    public enum ObstacleScope
    {
        NearSelectedDrone,  // only what the drone can currently see -- falls back to All with no drone
        AllInScene,
    }

    [Header("Obstacles")]
    public ObstacleScope obstacleScope = ObstacleScope.NearSelectedDrone;
    [Tooltip("The axis-aligned bounds the radius is derived from. The gap between this and the " +
             "cylinder is the inflation you are paying for. Off by default -- it doubles the lines.")]
    public bool drawBounds = false;
    [Tooltip("Cylinder at radius + d_obs, where phi_beta switches on -- 4 m at the city scenes' " +
             "d_obs of 0.4 and ScaleFactor of 10. Note the pilot-command shield acts much further " +
             "out than this (d_shield), so a drone reacts well before this shell is reached.")]
    public bool drawRepulsionShell = false;

    [Header("Drone")]
    [Tooltip("Resolved as SwarmParent/Drone N, matching the other swarm debug scripts.")]
    public int selectedDroneIndex = 0;
    [Tooltip("Draw every drone rather than the one above. Expensive and busy; for spotting which " +
             "drone is stuck, not for reading a single interaction.")]
    public bool allDrones = false;
    [Tooltip("How many obstacles get the per-obstacle drawing, nearest first. The total force " +
             "arrow always sums every obstacle in range regardless.")]
    [Range(0, 8)] public int detailedObstacles = 1;
    [Tooltip("Outward (coral) and circumferential (pistachio) frame axes. The green one is the " +
             "direction the cylinder buys you and the bounding box did not have.")]
    public bool drawFrameAxes = false;
    [Tooltip("Split the force into its repulsion (orchid) and velocity-match (periwinkle) halves.")]
    public bool drawForceSplit = false;
    [Tooltip("Sphere of radius r0_obs around the drone -- the OverlapSphere query.")]
    public bool drawQueryRange = false;

    [Header("Labels")]
    [Tooltip("Name each drawn cylinder, with a leader line from the ring it belongs to.")]
    public bool labelObstacles = false;
    [Tooltip("Cap on labels per frame. Nearest first when scoped to a drone, arbitrary otherwise, " +
             "so a city scene stays readable rather than solid text.")]
    [Range(1, 64)] public int maxLabels = 12;
    [Tooltip("Length of the leader line rising from the ring to the text, in world units.")]
    public float labelLeaderLength = 4.0f;
    [Range(8, 24)] public int labelFontSize = 11;

    [Header("Appearance")]
    [Range(8, 64)] public int circleSegments = 28;
    [Tooltip("Lines joining the two end circles. 0 draws the two rings only.")]
    [Range(0, 8)] public int spokes = 0;
    [Tooltip("World units drawn per unit of acceleration. Force arrows only.")]
    public float forceArrowScale = 20.0f;

    // Pastels, grouped by role rather than picked for contrast alone. The five arrows share an
    // origin, so they take the widest hue spread; the structure lines sit back in teal and a
    // low-alpha warm grey so the forces read on top of them. Saturation is kept mid rather than
    // washed out -- the scene view's skybox is bright, and a true pastel disappears against it.
    //
    // The repulsion term is collinear with a-hat by construction, so those two need separate hues
    // (coral / orchid) rather than shades of one: same direction, different length is exactly the
    // pair that a family resemblance would make unreadable.
    [Header("Palette")]
    [Tooltip("The cylinder outline and the line from the drone to its surface.")]
    public Color cylinderColour = new Color(0.47f, 0.82f, 0.80f, 1.00f);      // soft teal
    [Tooltip("Bounds and repulsion shell -- context, meant to sit behind everything else.")]
    public Color guideColour = new Color(0.76f, 0.74f, 0.70f, 0.35f);         // warm grey
    [Tooltip("a-hat, radially out from the cylinder axis.")]
    public Color outwardColour = new Color(0.96f, 0.60f, 0.53f, 1.00f);       // coral
    [Tooltip("t-hat, circumferential -- the way around the obstacle.")]
    public Color tangentColour = new Color(0.65f, 0.86f, 0.55f, 1.00f);       // pistachio
    [Tooltip("The c_obs repulsion term on its own.")]
    public Color repulsionColour = new Color(0.85f, 0.65f, 0.91f, 1.00f);     // orchid
    [Tooltip("The c2_beta velocity-match term on its own. This is the one that reverses direction " +
             "as a drone rebounds -- that reversal is the damping doing its job.")]
    public Color velocityMatchColour = new Color(0.58f, 0.70f, 0.93f, 1.00f); // periwinkle
    [Tooltip("The summed obstacle force -- the brightest thing drawn, since it is the one that " +
             "is always true no matter how much detail is switched off.")]
    public Color totalForceColour = new Color(0.99f, 0.91f, 0.72f, 1.00f);    // cream

    [Header("Parameters")]
    [Tooltip("Left empty, the first OlfatiSaber in the scene is used. In edit mode there is usually " +
             "none, so the fallbacks below are used instead -- keep them matching the drone prefab.")]
    public OlfatiSaber parameterSource;
    public Vector3 fallbackCylinderAxis = Vector3.up;
    public float fallbackScaleFactor = 10.0f;
    // Matches the city scenes' SwarmManager, not the OlfatiSaber component default -- the component
    // default is overwritten at Start and is not what any scene flies.
    public float fallbackDObs = 0.4f;

    private const string k_ObstacleLayerName = "Obstacle";
    private const float k_ColliderRefreshInterval = 1.0f;

    private Collider[] _obstacles = new Collider[0];
    private float _nextColliderRefresh;

    // Resolved once per draw so every helper agrees on which source won.
    private Vector3 _axis;
    private float _scaleFactor;
    private float _dObs;

    // The drone the labels quote a distance from, and the per-frame label budget.
    private bool _hasReference;
    private Vector3 _referencePosition;
    private int _labelsDrawn;

#if UNITY_EDITOR
    // Handles.color does not tint label text, so the colour has to ride on a style. Cached because
    // OnDrawGizmos runs on every scene-view repaint, and rebuilt only when the colour changes.
    private GUIStyle _labelStyle;
    private Color _labelStyleColour;
#endif

    void OnDrawGizmos()
    {
        ResolveParameters();
        _labelsDrawn = 0;

        // The scoped pass needs the drone anyway, so resolve it once and hand it to both halves.
        OlfatiSaber selected = allDrones ? null : ResolveSelectedDrone();

        // Obstacle labels quote the surface distance the algorithm actually sees, which needs a
        // drone to measure from. Without one they carry the name and radius only.
        //
        // Deliberately a weaker test than IsFlying: this only needs a readable position, so a
        // parked drone still scopes the obstacle pass. With every drone parked that scopes to the
        // parking altitude and draws nothing, which beats falling through to AllInScene and
        // dumping the whole city on screen at the end of a run.
        VelocityControl reference = selected != null ? selected.GetComponent<VelocityControl>() : null;
        _hasReference = reference != null && reference.State != null;
        _referencePosition = _hasReference ? reference.State.Position : Vector3.zero;

        DrawObstacles(selected);

        if (allDrones)
        {
            foreach (OlfatiSaber olfati in FindObjectsByType<OlfatiSaber>(FindObjectsSortMode.None))
                DrawDrone(olfati);
        }
        else
        {
            DrawDrone(selected);
        }
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

    // Parked drones are skipped: DroneHealthMonitor clears StateFinder.IsAlive and drops the drone
    // to parkingY with its renderers off, after a collision, getting stuck on an obstacle, or
    // straying too near the ground. Such a drone is kinematic, at rest, and 100 m under the course
    // where nothing is in range, so its obstacle force is identically zero -- the gizmo would sit
    // on a dead drone drawing nothing and look broken rather than finished.
    //
    // The scan starts at selectedDroneIndex and wraps. It deliberately does NOT write the index
    // back: OnDrawGizmos runs on every scene-view repaint, and assigning a serialized field from
    // there both flags the scene modified and overwrites whatever the operator is typing.
    private OlfatiSaber ResolveSelectedDrone()
    {
        // Same resolution as DebugObstacleDistance / TuneOlfatiSaberObstacle, but bounded by the
        // real fleet size so the wrap terminates.
        GameObject swarmParent = GameObject.Find("SwarmParent");
        if (swarmParent == null) return null;

        int count = swarmParent.transform.childCount;
        if (count <= 0) return null;

        int start = Mathf.Max(0, selectedDroneIndex);
        OlfatiSaber requested = null;

        for (int offset = 0; offset < count; offset++)
        {
            OlfatiSaber candidate = DroneAt(swarmParent.transform, (start + offset) % count);
            if (candidate == null) continue;

            if (offset == 0) requested = candidate;
            if (IsFlying(candidate.GetComponent<VelocityControl>())) return candidate;
        }

        // Whole fleet parked. Hand back the requested drone anyway so the obstacle pass still has a
        // position to scope to; DrawDrone declines to draw it.
        return requested;
    }

    private static OlfatiSaber DroneAt(Transform swarmParent, int index)
    {
        Transform drone = swarmParent.Find($"Drone {index}");
        if (drone == null) return null;

        Transform droneParent = drone.Find("DroneParent");
        return droneParent != null ? droneParent.GetComponent<OlfatiSaber>() : null;
    }

    // Stricter than PyUniSharingFast.IsAlive, which counts a missing StateFinder as alive so a
    // wiring gap never blanks the panorama. A gizmo needs an actual position and velocity to draw
    // anything, so here a missing one is unusable rather than forgiven.
    private static bool IsFlying(VelocityControl vc)
    {
        return vc != null && vc.State != null && vc.State.IsAlive;
    }

    private void DrawObstacles(OlfatiSaber selected)
    {
        // Scoping to the drone is the difference between a handful of cylinders and every building
        // in the city. With no drone to scope to -- edit mode, the usual case for checking the
        // level -- fall back to all of them rather than drawing nothing.
        // selected is non-null whenever _hasReference is, but the query reads r0_obs off it, so
        // say so here rather than leaving the two coupled through OnDrawGizmos.
        if (obstacleScope == ObstacleScope.NearSelectedDrone && !allDrones
            && selected != null && _hasReference)
        {
            Collider[] inRange = Physics.OverlapSphere(_referencePosition,
                                                       selected.r0_obs * selected.ScaleFactor,
                                                       LayerMask.GetMask(k_ObstacleLayerName));
            // Nearest first, so the label budget is spent on the obstacles actually acting.
            SortByDistance(inRange, _referencePosition);
            foreach (Collider obstacle in inRange) DrawObstacle(obstacle);
            return;
        }

        RefreshObstacles();
        foreach (Collider obstacle in _obstacles)
        {
            if (obstacle != null) DrawObstacle(obstacle);
        }
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
            Gizmos.color = guideColour;
            Gizmos.DrawWireCube(obstacle.bounds.center, obstacle.bounds.size);
        }

        Gizmos.color = cylinderColour;
        DrawCylinder(centre, axis, radius, halfHeight);

        if (labelObstacles && _labelsDrawn < maxLabels)
            LabelObstacle(obstacle, centre, axis, radius, halfHeight);

        // d_obs is in swarm units; the world radius it corresponds to is d_obs * ScaleFactor.
        if (drawRepulsionShell)
        {
            Gizmos.color = guideColour;
            DrawCylinder(centre, axis, radius + _dObs * _scaleFactor, halfHeight);
        }
    }

    // The label is anchored ON the top ring and joined to it by a leader line, rather than floating
    // above the cylinder's centre. With several cylinders overlapping in the view -- the normal case
    // in a city -- a name hanging in space belongs to whichever ring the eye happens to pick, which
    // is the ambiguity the labels have to resolve rather than add to. The anchor sits on the side of
    // the ring facing the viewer, so the leader is never drawn away through the building.
    private void LabelObstacle(Collider obstacle, Vector3 centre, Vector3 axis, float radius, float halfHeight)
    {
#if UNITY_EDITOR
        BasisFor(axis, out Vector3 towardViewer, out _);

        // Camera.current is the scene-view camera while gizmos are drawing. The basis direction is
        // the fallback when there is none, so the anchor is deterministic rather than absent.
        Camera view = Camera.current;
        if (view != null)
        {
            Vector3 flat = view.transform.position - centre;
            flat -= axis * Vector3.Dot(flat, axis);
            if (flat.sqrMagnitude > 1e-6f) towardViewer = flat.normalized;
        }

        Vector3 ringPoint = centre + axis * halfHeight + towardViewer * radius;
        Vector3 labelPoint = ringPoint + axis * labelLeaderLength;

        Gizmos.color = cylinderColour;
        Gizmos.DrawLine(ringPoint, labelPoint);

        string text = $"{obstacle.name}\nR {radius:F1} m";

        if (_hasReference)
        {
            // The distance the action functions are fed, in swarm units, with the world value it was
            // divided down from. That ScaleFactor divide is the easiest thing to lose track of when
            // reading d against d_obs.
            OlfatiSaber.ObstacleFrame frame =
                OlfatiSaber.ComputeObstacleFrame(obstacle, _referencePosition, _axis, _scaleFactor);
            if (frame.valid)
                text += $"\nd {frame.distance:F2}  ({frame.distance * _scaleFactor:F1} m)";
        }

        UnityEditor.Handles.Label(labelPoint, text, LabelStyle());
        _labelsDrawn++;
#endif
    }

#if UNITY_EDITOR
    // Built from a bare GUIStyle rather than GUI.skin.label, which is only guaranteed inside an
    // OnGUI -- gizmo drawing is not one.
    private GUIStyle LabelStyle()
    {
        if (_labelStyle == null || _labelStyleColour != cylinderColour || _labelStyle.fontSize != labelFontSize)
        {
            _labelStyleColour = cylinderColour;
            _labelStyle = new GUIStyle
            {
                fontSize = labelFontSize,
                alignment = TextAnchor.LowerLeft,
                normal = { textColor = cylinderColour },
            };
        }
        return _labelStyle;
    }
#endif

    private void DrawDrone(OlfatiSaber olfati)
    {
        if (olfati == null) return;

        // Also the parked check, which is what makes allDrones skip the casualties rather than
        // stack a fan of zero-length arrows at the parking altitude. ResolveSelectedDrone has
        // already moved off a parked drone unless the entire fleet is down.
        VelocityControl vc = olfati.GetComponent<VelocityControl>();
        if (!IsFlying(vc)) return;

        Vector3 position = vc.State.Position;
        // World-frame velocity, exactly as GetSwarmAcceleration builds it.
        Vector3 velocity = olfati.transform.TransformDirection(vc.State.VelocityVector);

        float queryRadius = olfati.r0_obs * olfati.ScaleFactor;

        if (drawQueryRange)
        {
            Gizmos.color = guideColour;
            Gizmos.DrawWireSphere(position, queryRadius);
        }

        // The summed force over every obstacle in range, drawn whatever the detail budget is. It is
        // the one arrow that is always the truth: GetObstacleForce is public so this is the value
        // the drone is actually flying on, not a reimplementation of it.
        Gizmos.color = totalForceColour;
        DrawArrow(position, position + olfati.GetObstacleForce(position, velocity) * forceArrowScale);

        if (detailedObstacles <= 0) return;

        // The algorithm's own query, so the detail shows obstacles from the set the force saw.
        // Nearest first, because the nearest is the one dominating that arrow.
        Collider[] inRange = Physics.OverlapSphere(position, queryRadius, LayerMask.GetMask(k_ObstacleLayerName));
        SortByDistance(inRange, position);

        int drawn = 0;
        foreach (Collider obstacle in inRange)
        {
            if (drawn >= detailedObstacles) break;

            OlfatiSaber.ObstacleFrame frame = olfati.GetObstacleFrame(obstacle, position);
            if (!frame.valid) continue;
            drawn++;

            // Nearest point on the cylinder surface, on the drone's own radial line. This one line
            // says which obstacle is acting and how far away it is, which is most of what the
            // labels used to say.
            Vector3 axisFoot = frame.axisCentre + frame.axis * Vector3.Dot(position - frame.axisCentre, frame.axis);
            Gizmos.color = cylinderColour;
            Gizmos.DrawLine(position, axisFoot + frame.outward * frame.radius);

            if (drawFrameAxes)
            {
                float axisLength = Mathf.Max(2.0f, frame.radius * 0.4f);
                Gizmos.color = outwardColour;
                DrawArrow(position, position + frame.outward * axisLength);
                Gizmos.color = tangentColour;
                DrawArrow(position, position + frame.tangent * axisLength);
            }

            if (drawForceSplit)
            {
                olfati.GetObstacleContribution(frame, velocity,
                                               out Vector3 repulsion, out Vector3 velocityMatch);
                Gizmos.color = repulsionColour;
                DrawArrow(position, position + olfati.c_obs * repulsion * forceArrowScale);
                Gizmos.color = velocityMatchColour;
                DrawArrow(position, position + olfati.c2_beta * velocityMatch * forceArrowScale);
            }
        }
    }

    // Sorted on Bounds.SqrDistance (point to box, 0 inside) rather than distance to the centre, so
    // the order matches the surface distance the frames report -- by centre distance a large near
    // building ranks behind a small far one.
    //
    // Insertion sort: the in-range set is a handful of colliders and this runs per gizmo repaint, so
    // the allocation a comparer-based sort costs matters more than the order of the algorithm.
    private static void SortByDistance(Collider[] colliders, Vector3 from)
    {
        for (int i = 1; i < colliders.Length; i++)
        {
            Collider held = colliders[i];
            float key = held.bounds.SqrDistance(from);

            int j = i - 1;
            while (j >= 0 && colliders[j].bounds.SqrDistance(from) > key)
            {
                colliders[j + 1] = colliders[j];
                j--;
            }
            colliders[j + 1] = held;
        }
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
        Vector3 normalised = axis.normalized;
        Vector3 seed = Mathf.Abs(Vector3.Dot(normalised, Vector3.up)) > 0.9f ? Vector3.right : Vector3.up;
        u = Vector3.Cross(normalised, seed).normalized;
        v = Vector3.Cross(normalised, u);
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
