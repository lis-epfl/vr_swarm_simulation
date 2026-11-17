// The stitched panorama arrives from Python as RGB24 bytes. PyUniSharingFast uploads those
// bytes untouched as a single-channel R8 texture three times as wide -- a format D3D11 takes
// as it is, where an RGB24 texture is converted to RGBA32 on the main thread -- and this blit
// unpacks them into the curved screen's sRGB render texture, whose mip chain is generated
// after it. Byte-exact: each channel is read with Load (no filtering) and decoded with the
// exact piecewise sRGB curve, so the target's sRGB encode on write returns the uploaded byte.
Shader "Hidden/Swarm/PanoramaUnpack"
{
    Properties
    {
        _MainTex ("Packed RGB24 as R8, 3W x H", 2D) = "black" {}
    }
    SubShader
    {
        Cull Off ZWrite Off ZTest Always
        Pass
        {
            CGPROGRAM
            #pragma vertex vert_img
            #pragma fragment frag
            #include "UnityCG.cginc"

            Texture2D _MainTex;
            float4 _MainTex_TexelSize;   // (1/w, 1/h, w, h) of the packed texture

            float3 SrgbToLinear(float3 c)
            {
                float3 lo = c / 12.92;
                float3 hi = pow((c + 0.055) / 1.055, 2.4);
                return lerp(hi, lo, step(c, 0.04045));
            }

            fixed4 frag (v2f_img i) : SV_Target
            {
                int w = (int)(_MainTex_TexelSize.z / 3.0 + 0.5);
                int h = (int)(_MainTex_TexelSize.w + 0.5);
                int x = min((int)(i.uv.x * w), w - 1);
                int y = min((int)(i.uv.y * h), h - 1);
                float r = _MainTex.Load(int3(3 * x,     y, 0)).r;
                float g = _MainTex.Load(int3(3 * x + 1, y, 0)).r;
                float b = _MainTex.Load(int3(3 * x + 2, y, 0)).r;
                return fixed4(SrgbToLinear(float3(r, g, b)), 1.0);
            }
            ENDCG
        }
    }
}
