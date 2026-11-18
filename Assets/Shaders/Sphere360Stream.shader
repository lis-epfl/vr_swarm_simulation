// The live link of a 360 drone (Sphere360Display, settings.streamWidth): the cube the drone's camera rendered,
// resampled into the equirectangular frame a product sends to its goggles (the Antigravity A1's "2K", 2048 x 1024).
// The sphere then shows this frame, not the cube, so the pilot sees the sphere at the stream's resolution.
//
// Each output pixel is the average of a 4 x 4 grid of directions across its own footprint, each read from the cube's
// mip level that matches the grid's spacing: a box filter over the pixel, as a camera's downscaler gives, rather than
// one sample that would alias the 18-36 px/deg cube down to ~5.7. No codec is modelled.
//
// The mapping must match Sphere360View.shader's: u = 0.5 + atan2(x, z) / 2pi, v = 0.5 + asin(y) / pi.
Shader "Hidden/Swarm/Sphere 360 Stream"
{
    Properties
    {
        _Cube ("Cube", CUBE) = "" {}
    }
    SubShader
    {
        Cull Off
        ZWrite Off
        ZTest Always

        Pass
        {
            CGPROGRAM
            #pragma vertex vert_img
            #pragma fragment frag
            #pragma target 3.5
            #include "UnityCG.cginc"

            samplerCUBE _Cube;
            float _CubeEdge;      // pixels per face edge
            float4 _StreamSize;   // width, height, 1/width, 1/height

            #define TAPS 4

            float3 EquirectToDir(float2 uv)
            {
                float lon = (uv.x - 0.5) * (2.0 * UNITY_PI);
                float lat = (uv.y - 0.5) * UNITY_PI;
                float c = cos(lat);
                return float3(c * sin(lon), sin(lat), c * cos(lon));
            }

            fixed4 frag (v2f_img i) : SV_Target
            {
                float lat = (i.uv.y - 0.5) * UNITY_PI;
                // The grid's spacing in radians: vertically pi / H / TAPS everywhere; horizontally 2pi / W / TAPS
                // times cos(lat), which is smaller off the equator, so the vertical spacing sets the blur.
                float spacing = max(2.0 * UNITY_PI * _StreamSize.z * cos(lat), UNITY_PI * _StreamSize.w) / TAPS;
                float4 sum = 0;
                [unroll] for (int y = 0; y < TAPS; y++)
                {
                    [unroll] for (int x = 0; x < TAPS; x++)
                    {
                        float2 offset = (float2(x, y) + 0.5) / TAPS - 0.5;
                        float3 d = EquirectToDir(i.uv + offset * _StreamSize.zw);
                        // A face's density at d is (edge / 2) / m^2 radially and / m tangentially, m the largest
                        // component of d; their geometric mean picks the level.
                        float m = max(abs(d.x), max(abs(d.y), abs(d.z)));
                        float pxPerRad = 0.5 * _CubeEdge * pow(m, -1.5);
                        float lod = max(0.0, log2(pxPerRad * spacing));
                        sum += texCUBElod(_Cube, float4(d, lod));
                    }
                }
                return sum / (TAPS * TAPS);
            }
            ENDCG
        }
    }
}
