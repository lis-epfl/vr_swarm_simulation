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
                return i.depth; // Output linear depth directly
            }
            ENDCG
        }
    }
}
