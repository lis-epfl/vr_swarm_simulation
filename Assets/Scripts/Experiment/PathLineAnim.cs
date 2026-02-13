using UnityEngine;
using UnityEngine.Splines;
using System.IO;
using System.Globalization;

public class MovingDotsOnSpline : MonoBehaviour
{
    public SplineContainer spline;
    public GameObject dotPrefab;

    public int dotCount = 15;
    public float speed = 3f;          // units per second
    public float spacing = 2f;        // distance between dots
    public bool exportOnStart = false;
    public float spatialStep = 0.1f;  // meters between samples
    public string fileName = "spline_trajectory.csv";

    private Transform[] dots;
    private float splineLength;
    private float offsetDistance;

    void OnEnable()
    {
        splineLength = spline.CalculateLength();
        dots = new Transform[dotCount];

        for (int i = 0; i < dotCount; i++)
        {
            dots[i] = Instantiate(dotPrefab, transform).transform;
        }
        
        if (exportOnStart)        
        {
            Export();
            exportOnStart = false; // prevent multiple exports if enabled again
        }
    }

    void Update()
    {
        offsetDistance += speed * Time.deltaTime;
        offsetDistance %= splineLength;

        for (int i = 0; i < dotCount; i++)
        {
            float distance = (i * spacing + offsetDistance) % splineLength;

            float t = spline.Spline.ConvertIndexUnit(
                distance,
                PathIndexUnit.Distance,
                PathIndexUnit.Normalized);

            Vector3 pos = spline.EvaluatePosition(t);
            Vector3 tangent = spline.EvaluateTangent(t);

            dots[i].position = pos;
            dots[i].forward = tangent.normalized;
        }
    }
    GameObject CreateDot()
    {
        GameObject dot = GameObject.CreatePrimitive(PrimitiveType.Sphere);
        dot.transform.localScale = Vector3.one * 0.2f;

        Destroy(dot.GetComponent<Collider>());

        return dot;
    }

    public void Export()
    {
        float length = spline.CalculateLength();
        string path = Path.Combine("Assets/Data", fileName);

        using (StreamWriter writer = new StreamWriter(path))
        {
            writer.WriteLine("x,y,z,dx,dy,dz");

            for (float d = 0; d <= length; d += spatialStep)
            {
                float t = spline.Spline.ConvertIndexUnit(
                    d,
                    PathIndexUnit.Distance,
                    PathIndexUnit.Normalized);

                Vector3 pos = spline.EvaluatePosition(t);
                Vector3 tangent = Vector3.Normalize(spline.EvaluateTangent(t));

            writer.WriteLine(
                string.Format(CultureInfo.InvariantCulture,
                "{0},{1},{2},{3},{4},{5}",
                pos.x, pos.y, pos.z,
                tangent.x, tangent.y, tangent.z));
            }
        }

        Debug.Log("Spline exported to: " + path);
    }
}
