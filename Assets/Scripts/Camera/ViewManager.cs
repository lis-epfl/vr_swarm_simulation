using System.Collections;
using System.Collections.Generic;
using Unity.VisualScripting;
using UnityEngine;

public class ViewManager : MonoBehaviour
{

    public List<GameObject> swarm;
    public bool visualize = true;

    private struct DroneInfo
    {
        public int droneIdx;
        public float dot;
        public float cross;
    }
    private List<int> views_idx; // front, left, back, right
    private List<DroneInfo> droneInfos;
    // Start is called before the first frame update
    void Start()
    {
        views_idx = new List<int>() { -1, -1, -1, -1 };
        droneInfos = new List<DroneInfo>();
    }

    // Update is called once per frame
    void Update()
    {
        if (swarm == null || swarm.Count == 0)    
            return;
        // Loop through all drones in the swarm and calculate dot and cross products
        // w.r.t swarm los and distancevector from center
        Vector3 los = GetSwarmHeading();
        Vector3 center = GetSwarmCenter();
        for (int i = 0; i < swarm.Count; i++)
        {
            GameObject drone = swarm[i];
            Vector3 toDrone = drone.transform.Find("DroneParent").position - center;
            float dot = Vector3.Dot(los, toDrone);
            Vector3 crossProduct = Vector3.Cross(toDrone, los);
            float cross = crossProduct.y; // Assuming a 2D swarm and y is up

            DroneInfo info = new DroneInfo
            {
                droneIdx = i,
                dot = dot,
                cross = cross
            };
            droneInfos.Add(info);
        }
        // Sort drones based on dot product descending (front to back))
        droneInfos.Sort((a, b) => b.dot >= a.dot ? 1 : -1);
        // Select first (front most) and last (back most) drones
        if (droneInfos.Count >= 2)
        {
            views_idx[0] = droneInfos[0].droneIdx; // Front most
            views_idx[2] = droneInfos[droneInfos.Count - 1].droneIdx; // Back most
        }
        // Sort drones based on cross product descending (right to left)
        droneInfos.Sort((a, b) => b.cross >= a.cross ? 1 : -1);
        // Select first (right most) and last (left most) drones
        if (droneInfos.Count >= 2)
        {
            views_idx[1] = droneInfos[0].droneIdx; // Left most
            views_idx[3] = droneInfos[droneInfos.Count - 1].droneIdx; // Right most
        }
        // Clear the droneInfos list for the next update
        droneInfos.Clear();

        // Apply views to corresponding drone FPV camera
        for (int i = 0; i < views_idx.Count; i++)
        {
            int droneIdx = views_idx[i];
            if (droneIdx != -1)
            {
                GameObject drone = swarm[droneIdx];
                Camera fpvCamera = drone.transform.Find("FPV").GetComponent<Camera>();
                if (fpvCamera != null)
                {
                    fpvCamera.targetDisplay = i;
                }
            }
        }

        if (visualize)
        {
            visualizeViews();
        }
        // Debug.Log("Views idx: " + string.Join(", ", views_idx));
    }

    void Reset()
    {
        views_idx = new List<int>() { -1, -1, -1, -1 };
        droneInfos = new List<DroneInfo>();
    }

    private Vector3 GetSwarmCenter()
    {
        Vector3 center = Vector3.zero;
        foreach (GameObject drone in swarm)
        {
            center += drone.transform.Find("DroneParent").position;
        }
        center /= swarm.Count;
        return center;
    }

    private Vector3 GetSwarmHeading()
    {
        Vector3 heading = Vector3.zero;
        Vector3 test = Vector3.zero;
        Vector3 test_yaw = Vector3.zero;
        foreach (GameObject drone in swarm)
        {
            StateFinder stateFinder = drone.transform.Find("DroneParent").GetComponent<VelocityControl>().State;
            float yaw = stateFinder.Angles.y * Mathf.Deg2Rad;
            heading += new Vector3(Mathf.Cos(yaw), 0, Mathf.Sin(yaw));
            
        }
        return heading.normalized;
    }

    // Visualize with a color coded sphere the detected drones (front, right, back, left)
    private void visualizeViews()
    {
        Color[] colors = new Color[] { Color.green, Color.blue, Color.red, Color.yellow };
        for (int i = 0; i < views_idx.Count; i++)
        {
            int droneIdx = views_idx[i];
            if (droneIdx != -1)
            {
                GameObject drone = swarm[droneIdx];
                Vector3 position = drone.transform.Find("DroneParent").position;
                Debug.DrawLine(position, position + Vector3.up * 2, colors[i], 0, false);
            }
        }
    }
}