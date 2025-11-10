Shader "Custom/LinearDepth"
{
    SubShader
    {
        Tags { "RenderType"="Opaque" }
        Pass
        {
            CGPROGRAM
            #pragma vertex vert
            #pragma fragment frag
            #include "UnityCG.cginc"

            struct appdata
            {
                float4 vertex : POSITION;
            };

            struct v2f
            {
                float4 pos : SV_POSITION;
                float depth : TEXCOORD0;
            };

            v2f vert(appdata v)
            {
                v2f o;
                o.pos = UnityObjectToClipPos(v.vertex);
                // Calculate linear eye-space depth
                float4 viewPos = mul(UNITY_MATRIX_MV, v.vertex);
                o.depth = -viewPos.z; // Depth in view space (positive forward)
                return o;
            }

            float frag(v2f i) : SV_Target
            {
                // Clamp depth to camera's near/far range (0.1m to 100m)
                // This prevents invalid values from skybox/infinity
                return clamp(i.depth, 0.1, 100.0);
            }
            ENDCG
        }
    }
}
