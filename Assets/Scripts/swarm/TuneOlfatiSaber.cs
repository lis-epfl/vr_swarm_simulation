using UnityEngine;

public class TuneOlfatiSaber : MonoBehaviour
{
    [Header("Reference to Olfati-Saber")]
    public OlfatiSaber olfatiSaber;

    [Header("Plot Settings")]
    public int textureWidth = 900;
    public int textureHeight = 400;
    public float maxDistance = 30f;  // how far out to sample
    public float minDistance = 0f;   // usually 0

    [Header("Axis Tick Settings")]
    public int xTickCount = 5;       // How many ticks on the distance axis
    public int yTickCount = 5;       // How many ticks on the force axis

    public enum plotChoice
    {
        COHESIONFORCE,
        NEIGHBOURWEIGHT,
        SHAPEFUNCTION,
    }

    [Header("Plot Type")]
    public plotChoice plotType = plotChoice.COHESIONFORCE;

    // Margins for axes
    private int marginLeft = 30;
    private int marginBottom = 20;
    private int marginRight = 10;
    private int marginTop = 10;

    // Force range now spans -forceRange..+forceRange
    public float forceRange = 10f;

    private Texture2D plotTexture;
    private int dataWidth;   // width of the plotting area (excluding margins)
    private int dataHeight;  // height of the plotting area (excluding margins)

    // We'll use this to draw the x-axis (y=0) in the middle
    private float yZero;     // pixel row in the texture where force=0

    void Start()
    {
        olfatiSaber = olfatiSaber ?? FindOlfatiSaber();

        plotTexture = new Texture2D(textureWidth, textureHeight, TextureFormat.RGBA32, false);
        plotTexture.wrapMode = TextureWrapMode.Clamp;

        // The "usable" plotting area after accounting for margins
        dataWidth = textureWidth - marginLeft - marginRight;
        dataHeight = textureHeight - marginBottom - marginTop;

        // yZero is halfway up the data area:
        yZero = marginBottom + (dataHeight * 0.5f);

        ClearTexture(Color.black);
    }

    void Update()
    {
        if (olfatiSaber == null)
        {
            olfatiSaber = FindOlfatiSaber();
            if (olfatiSaber == null) return;
        }

        // Clear with a background color
        ClearTexture(Color.black);

        // 1) Draw the vertical (Y) axis in white
        for (int y = marginBottom; y < marginBottom + dataHeight; y++)
        {
            plotTexture.SetPixel(marginLeft, y, Color.white);
        }

        // 2) Draw the horizontal (X) axis in white at y=0 => yZero
        int yZeroInt = Mathf.RoundToInt(yZero);
        for (int x = marginLeft; x < marginLeft + dataWidth; x++)
        {
            plotTexture.SetPixel(x, yZeroInt, Color.white);
        }

        // 3) Draw tick marks along the X-axis (distance) from minDistance..maxDistance
        for (int i = 0; i <= xTickCount; i++)
        {
            float fraction = i / (float)xTickCount;
            int xPixel = marginLeft + Mathf.RoundToInt(fraction * dataWidth);

            // Small vertical line as the tick
            for (int dy = -2; dy <= 2; dy++)
            {
                int yTickPixel = yZeroInt + dy;
                if (yTickPixel >= 0 && yTickPixel < textureHeight)
                {
                    plotTexture.SetPixel(xPixel, yTickPixel, Color.white);
                }
            }
        }

        // 4) Draw tick marks along the Y-axis (force) from -forceRange..+forceRange
        // We'll loop 0..yTickCount, then map to -forceRange..+forceRange
        for (int i = 0; i <= yTickCount; i++)
        {
            float fraction = i / (float)yTickCount; // 0..1
            float forceVal = -forceRange + fraction * (2f * forceRange); // -forceRange..+forceRange

            // Convert that forceVal to a Y pixel in the texture
            float scaleFactor = dataHeight / (2f * forceRange);
            float yFloat = (forceVal * scaleFactor) + (dataHeight * 0.5f); 
            int yPixel = marginBottom + Mathf.RoundToInt(yFloat);

            // Draw a small horizontal line as the tick
            for (int dx = -2; dx <= 2; dx++)
            {
                int xTickPixel = marginLeft + dx;
                if (xTickPixel >= 0 && xTickPixel < textureWidth)
                {
                    plotTexture.SetPixel(xTickPixel, yPixel, Color.white);
                }
            }
        }

        // 5) Plot the (signed) force function
        float rangeDenominator = 2f * forceRange;  // for scaling -range..range
        float plotScale = dataHeight / rangeDenominator; // used to scale force -> pixel offset
        for (int i = 0; i < dataWidth; i++)
        {
            // Map i -> [minDistance, maxDistance]
            float t = i / (float)(dataWidth - 1);
            float distance = Mathf.Lerp(minDistance, maxDistance, t);

            // Evaluate Olfati-Saber's chosen function (NO abs!)
            float force = 0f;
            if (plotType == plotChoice.COHESIONFORCE)
            {
                force = olfatiSaber.GetCohesionForce(distance);
            }
            else if (plotType == plotChoice.NEIGHBOURWEIGHT)
            {
                force = olfatiSaber.GetNeighbourWeight(distance);
            }
            else if (plotType == plotChoice.SHAPEFUNCTION)
            {
                force = olfatiSaber.GetCohesionIntensity(distance);
            }

            // Convert "force" to a Y pixel in [marginBottom, marginBottom + dataHeight]
            // with 0 mapped to yZero.
            float yFloat = yZero + (force * plotScale);
            int yPixel = Mathf.RoundToInt(yFloat);

            int xPixel = marginLeft + i;

            // Draw a green pixel if it’s within the plot area
            if (yPixel >= marginBottom && yPixel < (marginBottom + dataHeight))
            {
                plotTexture.SetPixel(xPixel, yPixel, Color.green);
            }
        }

        // Apply all pixel changes
        plotTexture.Apply();
    }

    void OnGUI()
    {
        if (plotTexture == null) return;

        // Draw the texture somewhere on-screen.
        GUI.DrawTexture(new Rect(10, 10, textureWidth, textureHeight), plotTexture);

        // X-axis label
        GUI.Label(new Rect(10, 10 + textureHeight + 2, 100, 20), "Distance");

        // Y-axis label
        GUI.Label(new Rect(10, 10 - 20, 100, 20), "Cohesion Force (F)");

        // Draw numeric labels for X ticks
        for (int i = 0; i <= xTickCount; i++)
        {
            float fraction = i / (float)xTickCount;
            float distVal = Mathf.Lerp(minDistance, maxDistance, fraction);

            int xPixel = marginLeft + Mathf.RoundToInt(fraction * dataWidth);
            float xGUI = 10 + xPixel - 10; // shift label left slightly
            float yGUI = 10 + textureHeight - (marginBottom / 2f);

            GUI.Label(new Rect(xGUI, yGUI, 50, 20), distVal.ToString("0.0"));
        }

        // Draw numeric labels for Y ticks (covering -forceRange..+forceRange)
        for (float i = 0; i <= yTickCount; i++)
        {
            float fraction = i / (float)yTickCount; // 0..1
            float forceVal = -forceRange + fraction * (2f * forceRange);

            float scaleFactor = dataHeight / (2f * forceRange);
            float yFloat = (forceVal * scaleFactor) + (dataHeight * 0.5f);
            int yPixel = marginBottom + Mathf.RoundToInt(yFloat);

            float xGUI = 10 + marginLeft - 25;
            float yGUI = 10 + textureHeight - yPixel - 10;
            GUI.Label(new Rect(xGUI, yGUI, 30, 20), forceVal.ToString("0.0"));
        }
    }

    // Fills the entire texture with one color
    private void ClearTexture(Color col)
    {
        Color[] fillColorArray = plotTexture.GetPixels();
        for (int i = 0; i < fillColorArray.Length; i++)
        {
            fillColorArray[i] = col;
        }
        plotTexture.SetPixels(fillColorArray);
        plotTexture.Apply();
    }

    // Find the gameObject with the name "Drone 0" and return its OlfatiSaber component
    private OlfatiSaber FindOlfatiSaber()
    {
        GameObject drone0 = GameObject.Find("Drone 0");
        if (drone0 != null)
        {
            // Return the OlfatiSaber component from the 'DroneParent' child
            Transform droneParent = drone0.transform.Find("DroneParent");
            if (droneParent != null)
            {
                return droneParent.GetComponent<OlfatiSaber>();
            }
            else
            {
                Debug.LogError("DroneParent not found in Drone 0!");
                return null;
            }
        }
        else
        {
            Debug.LogError("Drone 0 not found!");
            return null;
        }
    }
}
