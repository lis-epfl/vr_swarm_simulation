using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UIElements;

public class screenSpawn : MonoBehaviour
{

    public List<GameObject> screens = new List<GameObject>();
    public GameObject arena;
    public GameObject screenParent;
    public float circleRadius = 2.0f;
    public int numberOfScreens = 2;


    // Function to spawn screens for the drones in the swarm
    public void SpawnScreens()
    {
        
        // Create an empty GameObject to serve as the parent for all screens
        screenParent = new GameObject("ScreenParent");
        
        for (int i = 0; i < numberOfScreens; i++)
        {

            // Create a screen using a quad
            GameObject screen = GameObject.CreatePrimitive(PrimitiveType.Quad);

            // Name the screen
            screen.name = "screen_" + i;

            // Parent the screen under the screenParent GameObject
            screen.transform.parent = screenParent.transform;

            // Set the tag of the screen to 'Screen'
            screen.tag = "Screen";

            // Add the screen to the screens list
            screens.Add(screen);

            // Create a render texture
            RenderTexture rt = new RenderTexture(640, 360, 24);

            // Name the render texture 'rt_' followed by the drone number
            rt.name = "rt_" + i;

            // Create a new Material object
            Material screenMaterial = new Material(Shader.Find("Standard"));

            // Set the color to black
            screenMaterial.color = Color.black;

            // Set the smoothness to 0
            screenMaterial.SetFloat("_Glossiness", 0f);

            // Name the material 'screenMaterial'
            screenMaterial.name = "screenMaterial" + i;

            // Set the emission map to the render texture
            screenMaterial.SetTexture("_EmissionMap", rt);

            // Set the emission color to white
            screenMaterial.SetColor("_EmissionColor", Color.white);

            // Set the emission global illumination to baked
            screenMaterial.globalIlluminationFlags = MaterialGlobalIlluminationFlags.BakedEmissive;

            // Turn on emission for the material
            screenMaterial.EnableKeyword("_EMISSION");

            // Set the screens material to the screen material
            screen.GetComponent<Renderer>().material = screenMaterial;

            // set the scale to match the aspect ratio of the feed
            screen.transform.localScale = new Vector3(640f/360f, 1f, 1f);
        }

    }

    // Function to update the positions and orientations of the screens
    public void UpdateScreenPosition(int i, float yaw)
    {
        // Find the screen
        GameObject screen = screens.Find(s => s.name == "screen_" + i);

        // Calculate the screen position based on the yaw of the drone
        float yawRads = -yaw * Mathf.Deg2Rad; // Convert yaw to radians

        // Calculate the position on circle relative to the arena
        float x = arena.transform.position.x + circleRadius * Mathf.Cos(yawRads);
        float z = arena.transform.position.z + circleRadius * Mathf.Sin(yawRads);
        float y = arena.transform.position.y;// - 0.3f;

        // Position the quad at the calculated coordinates
        screen.transform.position = new Vector3(x, y, z);

        // Rotate the screen to face the center of the arena
        screen.transform.LookAt(arena.transform.position);
        screen.transform.Rotate(0, 180f, 0); // Adjust rotation to face outward

        // Ensure the screen is active
        screen.SetActive(true); 
    }

}
