using UnityEditor;
using UnityEngine;

/// <summary>
/// Scene-view frustum for every drone's FPV camera, drawn whatever is selected: green while the
/// drone is on the hull (<see cref="AttitudeAlgorithm.BoundaryEstimate"/>, the drones whose feed
/// the pilot is shown), red while it is interior.
///
/// A <c>[DrawGizmo]</c> drawer rather than a component, so it reaches every drone in every scene
/// with no scene or prefab edits — the drones are spawned at runtime, and a gizmo that needs the
/// swarm root selected is exactly the problem this solves. Hooked on <see cref="FPVCameraScript"/>,
/// which only the sim drones carry, so it lists under that name in the Scene view's Gizmos dropdown.
///
/// Colour is hull membership only, not whether a screen is actually up: under the NONE/SIMPLE
/// attitude modes nothing computes <c>BoundaryEstimate</c> and every frustum reads red, and a hull
/// drone hidden into the panorama still reads green.
///
/// Settings live in Preferences > Swarm > Drone Frustum Gizmos (per-user, not per-scene), with an
/// on/off toggle under Tools > Swarm as well.
/// </summary>
public static class DroneFrustumGizmos
{
    private const string EnabledKey = "VRSwarm.DroneFrustumGizmos.Enabled";
    private const string LengthKey = "VRSwarm.DroneFrustumGizmos.Length";
    private const string FillAlphaKey = "VRSwarm.DroneFrustumGizmos.FillAlpha";
    private const string MenuPath = "Tools/Swarm/Drone Frustum Gizmos";

    private static readonly Color BoundaryColour = new Color(0.2f, 1.0f, 0.4f, 1.0f);
    private static readonly Color InteriorColour = new Color(1.0f, 0.25f, 0.25f, 1.0f);

    // EditorPrefs reads hit the registry on Windows; OnDrawGizmos-rate calls would be one per drone
    // per repaint, so each value is read once and cached.
    private static bool? enabled;
    private static float? length;
    private static float? fillAlpha;

    private static bool Enabled
    {
        get => enabled ??= EditorPrefs.GetBool(EnabledKey, true);
        set { enabled = value; EditorPrefs.SetBool(EnabledKey, value); }
    }

    private static float Length
    {
        get => length ??= EditorPrefs.GetFloat(LengthKey, 5f);
        set { length = value; EditorPrefs.SetFloat(LengthKey, value); }
    }

    private static float FillAlpha
    {
        get => fillAlpha ??= EditorPrefs.GetFloat(FillAlphaKey, 0.15f);
        set { fillAlpha = value; EditorPrefs.SetFloat(FillAlphaKey, value); }
    }

    // Unit pyramid: apex at the origin, base at z = 1 spanning ±1 in x and y. Scaled per drone by
    // (tan(hfov/2)·L, tan(vfov/2)·L, L), so one mesh serves every FOV, aspect and length.
    private static Mesh pyramid;

    // The camera's far clip (2000 m on the drone prefab) is meaningless as a drawing length, so the
    // frustum is cut at Length instead.
    [DrawGizmo(GizmoType.Selected | GizmoType.NonSelected | GizmoType.InSelectionHierarchy
             | GizmoType.NotInSelectionHierarchy | GizmoType.Active)]
    private static void DrawFrustum(FPVCameraScript fpv, GizmoType gizmoType)
    {
        if (!Enabled || !Application.isPlaying) return;

        Transform droneParent = fpv.droneTransform;
        if (droneParent == null) return;

        VelocityControl vc = droneParent.GetComponent<VelocityControl>();
        if (vc != null && vc.State != null && !vc.State.IsAlive) return;

        Camera cam = fpv.GetComponent<Camera>();
        if (cam == null) return;

        AttitudeAlgorithm attitude = droneParent.GetComponent<AttitudeAlgorithm>();
        Color colour = attitude != null && attitude.BoundaryEstimate ? BoundaryColour : InteriorColour;

        float l = Mathf.Max(Length, 0.1f);
        float aspect = cam.aspect > 0f ? cam.aspect : 16f / 9f;

        Matrix4x4 previousMatrix = Gizmos.matrix;
        Color previousColour = Gizmos.color;

        // Scale-free so the frustum is in metres whatever the drone's hierarchy is scaled by.
        Gizmos.matrix = Matrix4x4.TRS(fpv.transform.position, fpv.transform.rotation, Vector3.one);
        Gizmos.color = colour;
        Gizmos.DrawFrustum(Vector3.zero, cam.fieldOfView, l, 0f, aspect);

        // A 1 px wireframe vanishes once the swarm is more than a few drones across; the tinted
        // fill is what makes the colour readable from a zoomed-out view.
        float alpha = FillAlpha;
        if (alpha > 0f)
        {
            float tanV = Mathf.Tan(cam.fieldOfView * 0.5f * Mathf.Deg2Rad);
            Gizmos.color = new Color(colour.r, colour.g, colour.b, alpha);
            Gizmos.DrawMesh(Pyramid, Vector3.zero, Quaternion.identity,
                            new Vector3(tanV * aspect * l, tanV * l, l));
        }

        Gizmos.matrix = previousMatrix;
        Gizmos.color = previousColour;
    }

    private static Mesh Pyramid
    {
        get
        {
            if (pyramid != null) return pyramid;

            Vector3 apex = Vector3.zero;
            Vector3 bl = new Vector3(-1f, -1f, 1f);
            Vector3 br = new Vector3(1f, -1f, 1f);
            Vector3 tr = new Vector3(1f, 1f, 1f);
            Vector3 tl = new Vector3(-1f, 1f, 1f);

            // Four sides plus the base as two triangles, each face emitted with both windings so it
            // shows from inside and out (the gizmo shader culls back faces). Unshared vertices keep
            // the normals flat.
            Vector3[][] faces =
            {
                new[] { apex, br, bl }, new[] { apex, tr, br }, new[] { apex, tl, tr },
                new[] { apex, bl, tl }, new[] { bl, br, tr }, new[] { bl, tr, tl },
            };

            var vertices = new Vector3[faces.Length * 6];
            var triangles = new int[faces.Length * 6];
            for (int f = 0; f < faces.Length; f++)
            {
                for (int k = 0; k < 3; k++)
                {
                    int front = f * 6 + k;
                    int back = f * 6 + 3 + k;
                    vertices[front] = faces[f][k];
                    vertices[back] = faces[f][2 - k];
                    triangles[front] = front;
                    triangles[back] = back;
                }
            }

            pyramid = new Mesh { name = "DroneFrustumGizmoPyramid", hideFlags = HideFlags.HideAndDontSave };
            pyramid.vertices = vertices;
            pyramid.triangles = triangles;
            pyramid.RecalculateNormals();
            return pyramid;
        }
    }

    [MenuItem(MenuPath)]
    private static void ToggleEnabled()
    {
        Enabled = !Enabled;
        SceneView.RepaintAll();
    }

    [MenuItem(MenuPath, true)]
    private static bool ToggleEnabledValidate()
    {
        Menu.SetChecked(MenuPath, Enabled);
        return true;
    }

    [SettingsProvider]
    private static SettingsProvider CreateSettingsProvider()
    {
        return new SettingsProvider("Preferences/Swarm/Drone Frustum Gizmos", SettingsScope.User)
        {
            keywords = new[] { "drone", "frustum", "gizmo", "fpv", "camera", "boundary", "hull" },
            guiHandler = _ =>
            {
                EditorGUI.BeginChangeCheck();
                bool newEnabled = EditorGUILayout.Toggle(
                    new GUIContent("Draw frustums", "Green = on the hull (feed shown), red = interior."),
                    Enabled);
                float newLength = EditorGUILayout.FloatField(
                    new GUIContent("Length (m)", "How far out from the camera the frustum is drawn."),
                    Length);
                float newAlpha = EditorGUILayout.Slider(
                    new GUIContent("Fill opacity", "0 draws the wireframe only."),
                    FillAlpha, 0f, 1f);
                if (EditorGUI.EndChangeCheck())
                {
                    Enabled = newEnabled;
                    Length = Mathf.Max(newLength, 0.1f);
                    FillAlpha = newAlpha;
                    SceneView.RepaintAll();
                }
            },
        };
    }
}
