// The soft dark patch under each pedestrian (WalkerGroundShadows). A falloff worked out in the fragment, so no texture
// and no lighting: the scenes it is used in cast no real shadows, and turning on the sun's for the walkers alone costs
// three times as much per FPV render and adds less. Queued just after the opaque ground and before anything alpha-tested,
// with the depth test keeping it under whatever stands on the footpath.
Shader "Swarm/Walker Ground Shadow" {
Properties {
	_Color ("Colour (alpha: strength)", Color) = (0,0,0,1)
}

SubShader {
	Tags {"Queue"="Geometry+10" "IgnoreProjector"="True" "RenderType"="Transparent" "ForceNoShadowCasting"="True"}
	ZWrite Off
	Blend SrcAlpha OneMinusSrcAlpha
	Offset -1, -1

	Pass {
		CGPROGRAM
		#pragma vertex vert
		#pragma fragment frag
		#include "UnityCG.cginc"

		fixed4 _Color;

		struct appdata {
			float4 vertex : POSITION;
			float2 uv : TEXCOORD0;      // -1..1 across the patch
		};

		struct v2f {
			float4 pos : SV_POSITION;
			float2 uv : TEXCOORD0;
		};

		v2f vert (appdata v) {
			v2f o;
			o.pos = UnityObjectToClipPos(v.vertex);
			o.uv = v.uv;
			return o;
		}

		fixed4 frag (v2f i) : SV_Target {
			float fade = saturate(1.0 - length(i.uv));
			return fixed4(_Color.rgb, _Color.a * saturate(1.6 * fade * fade));
		}
		ENDCG
	}
}
}
