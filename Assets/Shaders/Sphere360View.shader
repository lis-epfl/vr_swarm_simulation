// The 360 view on the sphere round the pilot (Sphere360Display): each pixel shows the cube in the direction of its
// own eye's view ray, turned from the display's frame into the city's by _DisplayToCity.
//
// Looking up by the eye's ray rather than by the vertex's direction from the head is what makes the view mono and
// at infinity, as a 360 video in goggles is: both eyes read the same direction for parallel rays, and leaning the
// head changes nothing. The sphere's radius then only has to keep it inside the eye cameras' clip planes.
//
// Stereo-instanced (the eye cameras render single-pass instanced, OculusSettings' desktop mode 1): without the
// UNITY_* macros below it draws in one eye only. The ray is formed in the vertex function, where the instance setup
// has picked the eye, and is linear across a triangle, so it interpolates exactly.
//
// With SPHERE_STREAM on (settings.streamWidth > 0) the lookup is in the equirectangular frame Sphere360Stream.shader
// resampled the cube into, the live link's resolution, instead of in the cube itself. The frame wraps in u, so the
// u with the smaller screen-space derivative of the two that differ at the seam is used (Tarini's trick); otherwise
// the jump from 1 to 0 picks the smallest mip and draws a line straight down the back of the sphere.
//
// Drawn first (Background+1), writing no depth and testing none: the display clears its cameras to black and hides
// the Arena box, and anything on the UI layer drawn later (BodyYawCompass, the gaze cursor) lands on top.
Shader "Swarm/Sphere 360 View"
{
    Properties
    {
        _Cube ("Cube", CUBE) = "" {}
        _Stream ("Live stream (equirectangular)", 2D) = "black" {}
        _BandHalfDeg ("Band half-height (deg, 0 = whole sphere)", Float) = 0
        _BandCentreDeg ("Band centre elevation (deg)", Float) = 0
        _MaskColor ("Outside the band", Color) = (0, 0, 0, 1)
    }
    SubShader
    {
        Tags { "Queue"="Background+1" "RenderType"="Background" "IgnoreProjector"="True" "PreviewType"="Skybox" }
        Cull Front
        ZWrite Off
        ZTest Always

        Pass
        {
            CGPROGRAM
            #pragma vertex vert
            #pragma fragment frag
            #pragma multi_compile_local __ SPHERE_STREAM
            #include "UnityCG.cginc"

            samplerCUBE _Cube;
            sampler2D _Stream;
            float4x4 _DisplayToCity;
            float _BandHalfDeg;
            float _BandCentreDeg;
            fixed4 _MaskColor;

            struct appdata
            {
                float4 vertex : POSITION;
                UNITY_VERTEX_INPUT_INSTANCE_ID
            };

            struct v2f
            {
                float4 pos : SV_POSITION;
                float3 ray : TEXCOORD0;
                UNITY_VERTEX_OUTPUT_STEREO
            };

            v2f vert (appdata v)
            {
                v2f o;
                UNITY_SETUP_INSTANCE_ID(v);
                UNITY_INITIALIZE_OUTPUT(v2f, o);
                UNITY_INITIALIZE_VERTEX_OUTPUT_STEREO(o);
                float3 world = mul(unity_ObjectToWorld, v.vertex).xyz;
                o.pos = UnityWorldToClipPos(world);
                o.ray = world - _WorldSpaceCameraPos;
                return o;
            }

            fixed4 frag (v2f i) : SV_Target
            {
                float3 d = normalize(mul((float3x3)_DisplayToCity, i.ray));
                // Sampled before the band test: the stream's lookup takes derivatives, which must not sit behind a
                // branch that varies from pixel to pixel.
            #ifdef SPHERE_STREAM
                // Must match Sphere360Stream.shader's mapping.
                float u = 0.5 + atan2(d.x, d.z) / (2.0 * UNITY_PI);
                float v = 0.5 + asin(clamp(d.y, -1, 1)) / UNITY_PI;
                float uSeamFront = frac(u + 0.5) - 0.5;
                u = fwidth(u) <= fwidth(uSeamFront) ? u : uSeamFront;
                fixed4 c = tex2D(_Stream, float2(u, v));
            #else
                fixed4 c = texCUBE(_Cube, d);
            #endif
                if (_BandHalfDeg > 0 && abs(degrees(asin(clamp(d.y, -1, 1))) - _BandCentreDeg) > _BandHalfDeg)
                    c = _MaskColor;
                return c;
            }
            ENDCG
        }
    }
}
