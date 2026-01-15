using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class movingCircle : MonoBehaviour
{
    [Header("Circle Parameters")]
    public GameObject walkerPrefab;
    public int numWalkers = 6;
    public float minRadius = 1.5f;
    public float maxRadius = 3.0f;
    public float speedMetersPerSec = 2.0f;
    public Vector3 circleCenter = Vector3.zero;

    private class WalkerInfo
    {
        public GameObject walker;
        public float currentAngle;
        public Vector3 previousPosition;
        public bool clockwise;
        public float radius;
        public float angularSpeed;  // Speed in degrees per second
    }
    private List<WalkerInfo> walkers = new List<WalkerInfo>();

    void Start()
    {
        if (walkerPrefab == null)
        {
            Debug.LogError("Walker Prefab is not assigned!");
            return;
        }

        // Create walkers
        for (int i = 0; i < numWalkers; i++)
        {
            WalkerInfo info = new WalkerInfo();
            
            // Assign random radius and start angle
            info.radius = Random.Range(minRadius, maxRadius);
            info.currentAngle = Random.Range(0f, 360f);
            
            // Calculate angular speed based on desired linear speed
            info.angularSpeed = (speedMetersPerSec * 360f) / (2f * Mathf.PI * info.radius);
            
            // Alternate direction (first half clockwise, second half counterclockwise)
            info.clockwise = i < numWalkers / 2;

            // Calculate initial position
            float radians = -info.currentAngle * Mathf.Deg2Rad;
            float x = circleCenter.x + info.radius * Mathf.Cos(radians);
            float z = circleCenter.z + info.radius * Mathf.Sin(radians);
            Vector3 position = new Vector3(x, circleCenter.y, z);

            // Instantiate walker
            info.walker = Instantiate(walkerPrefab, position, Quaternion.identity);
            info.walker.name = "Walker_" + i;
            info.previousPosition = position;

            walkers.Add(info);
        }
    }
    
    void Update()
    {
        foreach (WalkerInfo info in walkers)
        {
            if (info.walker == null) continue;

            // Update angle using pre-calculated angular speed
            float angleChange = info.angularSpeed * Time.deltaTime;
            info.currentAngle += info.clockwise ? -angleChange : angleChange;
            
            // Keep angle between 0 and 360 degrees
            info.currentAngle %= 360.0f;
            if (info.currentAngle < 0) info.currentAngle += 360.0f;

            // Convert angle to radians
            float radians = -info.currentAngle * Mathf.Deg2Rad;

            // Calculate new position using walker's specific radius
            float x = circleCenter.x + info.radius * Mathf.Cos(radians);
            float z = circleCenter.z + info.radius * Mathf.Sin(radians);
            Vector3 newPosition = new Vector3(x, circleCenter.y, z);

            // Calculate direction for walker to face
            Vector3 moveDirection = (newPosition - info.previousPosition).normalized;
            if (moveDirection != Vector3.zero)
            {
                info.walker.transform.rotation = Quaternion.LookRotation(moveDirection);
            }

            // Update walker position and store for next frame
            info.walker.transform.position = newPosition;
            info.previousPosition = newPosition;
        }
    }
}
