using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class PathLineAnim : MonoBehaviour
{

    public Transform[] pathPoints;
    public LineRenderer line;
    public float scrollSpeed = 1f;


    // Start is called before the first frame update
    void Start()
    {
        // line.positionCount = pathPoints.Length;
        // for (int i = 0; i < pathPoints.Length; i++)
        //     line.SetPosition(i, pathPoints[i].position + Vector3.up * 0.05f);
        line.material.mainTexture = MakeDashedTexture();
    }

    // Update is called once per frame
    void Update()
    {
        line.material.mainTextureOffset += new Vector2(Time.deltaTime * scrollSpeed, 0);
    }

    Texture2D MakeDashedTexture()
    {
        int width = 256;
        int height = 32;

        Texture2D tex = new Texture2D(width, height, TextureFormat.RGBA32, false);

        for (int x = 0; x < width; x++)
        {
            bool dash = (x / 32) % 2 == 0; // dash / gap
            for (int y = 0; y < height; y++)
            {
                tex.SetPixel(x, y, dash ? Color.white : Color.clear);
            }
        }

        tex.Apply();
        tex.wrapMode = TextureWrapMode.Repeat;
        return tex;
    }
}
