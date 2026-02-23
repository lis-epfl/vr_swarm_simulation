using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class movingCircle : MonoBehaviour
{
    [Header("Circle Parameters")]
    public GameObject walkerPrefab;
    public GameObject obstacle;
    public float verticalOffset = 0f;
    public int numWalkers = 6;
    public float minRadius = 1.5f;
    public float maxRadius = 3.0f;
    public float speedMetersPerSec = 2.0f;

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
    private GameObject walkersContainer;

    void Start()
    {
        if (walkerPrefab == null)
        {
            Debug.LogError("Walker Prefab is not assigned!");
            return;
        }

        if (obstacle == null)
        {
            Debug.LogError("Obstacle is not assigned!");
            return;
        }

        // Create a container for all walkers
        walkersContainer = new GameObject("Walkers");
        walkersContainer.transform.parent = transform;
        walkersContainer.transform.localPosition = Vector3.zero;

        // Print the center position
        Vector3 centerPos = obstacle.transform.position + Vector3.up * verticalOffset;
        Debug.Log("Walker Circle Center Position: " + centerPos);

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

            // Calculate initial position relative to obstacle
            float radians = -info.currentAngle * Mathf.Deg2Rad;
            Vector3 center = obstacle.transform.position + Vector3.up * verticalOffset;
            float x = center.x + info.radius * Mathf.Cos(radians);
            float z = center.z + info.radius * Mathf.Sin(radians);
            Vector3 position = new Vector3(x, center.y, z);

            // Instantiate walker and set as child of the walkers container
            info.walker = Instantiate(walkerPrefab, position, Quaternion.identity, walkersContainer.transform);
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

            // Calculate new position relative to obstacle
            Vector3 centerPos = obstacle.transform.position + Vector3.up * verticalOffset;
            float x = centerPos.x + info.radius * Mathf.Cos(radians);
            float z = centerPos.z + info.radius * Mathf.Sin(radians);
            Vector3 newPosition = new Vector3(x, centerPos.y, z);

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