using UnityEngine;

/// <summary>
/// Marks a goal of a procedural city (<see cref="ProceduralCityBuilder"/>) and carries its geometry: its index in the
/// layout, the ring radius r_g, the occluder width w_c, the clearance c_g and the patch radius, as the Python
/// generator recorded them (Assets/Scripts/Experiment/citysim). Sits on the goal root, at the goal centre, beside the
/// root's <see cref="GoalSpecialWalker"/>; the occluder and the patrolled ring buildings are the root's direct
/// children.
///
/// <para>Implements <see cref="IGoalZone"/>, so the recorder's auto-scored identify and analyse.py share one zone:
/// a disc of <see cref="patchRadius"/> about the centre, grown by the 15 m zone, where a tile goal is a square.</para>
/// </summary>
public class ProceduralGoal : MonoBehaviour, IGoalZone
{
    public int index;
    [Tooltip("r_g: goal centre to the ring buildings' centres (m).")]
    public float ringRadius;
    [Tooltip("w_c: the central occluder's long side (m).")]
    public float occluderWidth;
    [Tooltip("c_g: goal centre to the nearest building other than the occluder (m).")]
    public float clearance;
    [Tooltip("The patch's radius: r_g plus a ring building's half-diagonal (m).")]
    public float patchRadius;
    [Tooltip("The patch's cleared radius in the layout: no city building within it (m).")]
    public float clearRadius;
    public string centreBox = "";
    public string[] ringBoxes = new string[0];

    public float DistanceToPatch(Vector3 world)
    {
        Vector3 d = world - transform.position;
        d.y = 0f;
        return Mathf.Max(d.magnitude - patchRadius, 0f);
    }

    public string ZoneShape => "disc";
    public float PatchRadius => patchRadius;
    public float RingRadius => ringRadius;
    public float OccluderWidth => occluderWidth;
    public float Clearance => clearance;

    void OnDrawGizmosSelected()
    {
        Gizmos.color = new Color(0.16f, 0.47f, 0.84f, 0.8f);
        DrawCircle(transform.position, patchRadius);
        Gizmos.color = new Color(0.32f, 0.32f, 0.31f, 0.6f);
        DrawCircle(transform.position, clearance);
    }

    private static void DrawCircle(Vector3 c, float r)
    {
        const int n = 64;
        for (int i = 0; i < n; i++)
        {
            float a0 = i * 2f * Mathf.PI / n, a1 = (i + 1) * 2f * Mathf.PI / n;
            Gizmos.DrawLine(c + r * new Vector3(Mathf.Sin(a0), 0f, Mathf.Cos(a0)),
                            c + r * new Vector3(Mathf.Sin(a1), 0f, Mathf.Cos(a1)));
        }
    }
}
