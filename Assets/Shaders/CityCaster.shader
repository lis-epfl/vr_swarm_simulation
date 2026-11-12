// What CityLighting draws the city's static renderers with, once at the start of Play: their depth seen from the sun
// (passes 0 and 1, into a shadow map) and their height seen from above (passes 2 and 3, into a float texture). The odd
// passes clip by the source material's texture alpha, for leaves; CityLighting makes one material per cutout texture.
Shader "Hidden/Swarm/City Caster" {
Properties {
	_MainTex ("Base (alpha: cutout)", 2D) = "white" {}
	_Cutoff ("Alpha cutoff", Range(0, 1)) = 0.4
}

CGINCLUDE
#include "UnityCG.cginc"

sampler2D _MainTex;
float4 _MainTex_ST;
float _Cutoff;
// World to clip, built by CityLighting for the target as the platform addresses it (row 0 first, the depth the way
// the depth test wants it), so the map is read back with the same matrix and nothing rests on how a camera would
// have flipped it.
float4x4 _CityCasterVP;

struct v2f {
	float4 pos : SV_POSITION;
	float2 uv : TEXCOORD0;
	float height : TEXCOORD1;
};

v2f vert (appdata_base v) {
	v2f o;
	float4 world = mul(unity_ObjectToWorld, float4(v.vertex.xyz, 1.0));
	o.pos = mul(_CityCasterVP, world);
	o.uv = TRANSFORM_TEX(v.texcoord, _MainTex);
	o.height = world.y;
	return o;
}

fixed4 fragDepth (v2f i) : SV_Target { return 0; }

fixed4 fragDepthCut (v2f i) : SV_Target {
	clip(tex2D(_MainTex, i.uv).a - _Cutoff);
	return 0;
}

float4 fragHeight (v2f i) : SV_Target { return i.height; }

float4 fragHeightCut (v2f i) : SV_Target {
	clip(tex2D(_MainTex, i.uv).a - _Cutoff);
	return i.height;
}
ENDCG

SubShader {
	Cull Off

	Pass {      // 0: depth from the sun
		ColorMask 0
		Offset 1.5, 2
		CGPROGRAM
		#pragma vertex vert
		#pragma fragment fragDepth
		ENDCG
	}
	Pass {      // 1: depth from the sun, cutout
		ColorMask 0
		Offset 1.5, 2
		CGPROGRAM
		#pragma vertex vert
		#pragma fragment fragDepthCut
		ENDCG
	}
	Pass {      // 2: height from above
		CGPROGRAM
		#pragma vertex vert
		#pragma fragment fragHeight
		ENDCG
	}
	Pass {      // 3: height from above, cutout
		CGPROGRAM
		#pragma vertex vert
		#pragma fragment fragHeightCut
		ENDCG
	}
}
Fallback Off
}
