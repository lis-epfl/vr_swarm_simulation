// A drone's feed on its screen, as the feed is: unlit, so the pilot sees the camera's own pixels.
//
// ScreenSpawn's screens are otherwise the Standard shader with the feed as emission over a black base. A black
// Standard surface is not black: it still reflects the scene's sky (the Arena box renders sRGB 19/21/26 for the same
// reason) and carries a broad highlight from the sun, so every feed sits under a faint veil, and a different one on
// each screen of the ring, since each faces the sun at its own angle. Here nothing in the scene reaches the screen.
// ScreenSpawn uses this when its feedScreenMaterial is set (the scenic city sets it); other scenes are unchanged.
//
// Stereo-instanced, like Sphere360View: without the UNITY_* macros it would draw in one eye only.
Shader "Swarm/Feed Screen" {
Properties {
	_MainTex ("Feed", 2D) = "black" {}
	_Gain ("Brightness", Range(0, 2)) = 1
}

SubShader {
	Tags { "RenderType"="Opaque" "Queue"="Geometry" "IgnoreProjector"="True" }
	Pass {
		CGPROGRAM
		#pragma vertex vert
		#pragma fragment frag
		#include "UnityCG.cginc"

		sampler2D _MainTex;
		float4 _MainTex_ST;
		float _Gain;

		struct appdata {
			float4 vertex : POSITION;
			float2 uv : TEXCOORD0;
			UNITY_VERTEX_INPUT_INSTANCE_ID
		};

		struct v2f {
			float4 pos : SV_POSITION;
			float2 uv : TEXCOORD0;
			UNITY_VERTEX_OUTPUT_STEREO
		};

		v2f vert (appdata v) {
			v2f o;
			UNITY_SETUP_INSTANCE_ID(v);
			UNITY_INITIALIZE_OUTPUT(v2f, o);
			UNITY_INITIALIZE_VERTEX_OUTPUT_STEREO(o);
			o.pos = UnityObjectToClipPos(v.vertex);
			o.uv = TRANSFORM_TEX(v.uv, _MainTex);
			return o;
		}

		fixed4 frag (v2f i) : SV_Target {
			return fixed4(tex2D(_MainTex, i.uv).rgb * _Gain, 1.0);
		}
		ENDCG
	}
}
Fallback Off
}
