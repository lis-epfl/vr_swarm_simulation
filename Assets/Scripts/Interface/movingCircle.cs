using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class movingCircle : MonoBehaviour
{
    [Header("Circle Parameters")]
    public GameObject orbiterSphere;
    public float circleRadius = 2.0f;
    public float speedDegPerSec = 45.0f;
    public Vector3 circleCenter = Vector3.zero;

    private float currentAngle = 0.0f;

    void Start()
    {
        if (orbiterSphere == null)
        {
            Debug.LogError("Orbiter Sphere is not assigned!");
        }
    }
    
    void Update()
    {
        if (orbiterSphere == null) return;

        // Update the angle based on speed and time
        currentAngle += speedDegPerSec * Time.deltaTime;
        
        // Keep angle between 0 and 360 degrees
        currentAngle %= 360.0f;

        // Convert angle to radians
        float radians = -currentAngle * Mathf.Deg2Rad;

        // Calculate new position
        float x = circleCenter.x + circleRadius * Mathf.Cos(radians);
        float z = circleCenter.z + circleRadius * Mathf.Sin(radians);
        
        // Update sphere position
        Vector3 newPosition = new Vector3(x, circleCenter.y, z);
        orbiterSphere.transform.position = newPosition;
    }
}
