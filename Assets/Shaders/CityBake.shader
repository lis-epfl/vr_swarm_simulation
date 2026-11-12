// CityLighting's one-off bakes at the start of Play, each a full-screen draw into a texture of its own.
//   0: one horizontal slice of the sky-visibility volume, from the buildings' height map. For a point in the air it
//      marches the height map outwards in 16 directions for the highest horizon angle h in each, and stores the mean of
//      cos^2 h -- the share of a uniform sky's light that still reaches a horizontal surface there. A point inside a
//      building answers for the roof above it.
//   1, 2: the soft patch under the street furniture, from the props' height map: where anything stands, blurred across
//      and then down.
// Texels are addressed by SV_POSITION throughout, so what is written at a texel is read back at the same texel
// whichever way up the platform draws.
Shader "Hidden/Swarm/City Bake" {
SubShader {
	Cull Off ZWrite Off ZTest Always

	CGINCLUDE
	#pragma target 3.5
	#include "UnityCG.cginc"

	sampler2D _CityBakeSource;
	float4 _CityBakeSize;        // target w, h, 1/w, 1/h
	float4 _CityBakeRegion;      // size x, size z in metres; z: slice height; w: reach of the march in metres
	float4 _CityBakeBlur;        // x: radius in texels; y, z: heights between which a prop counts as standing there

	struct v2f {
		float4 pos : SV_POSITION;
	};

	v2f vert (float4 vertex : POSITION) {
		v2f o;
		o.pos = float4(vertex.xy, 0.5, 1.0);
		return o;
	}

	float4 fragSky (v2f i) : SV_Target {
		float2 uv = i.pos.xy * _CityBakeSize.zw;
		float h0 = max(_CityBakeRegion.z, tex2Dlod(_CityBakeSource, float4(uv, 0, 0)).r);
		const int DIRS = 16;
		const int STEPS = 14;
		float growth = pow(_CityBakeRegion.w / 1.5, 1.0 / (STEPS - 1));
		// a quarter-step turn per texel in a 2x2 pattern, so the 16 directions do not line up into spokes
		float turn = (fmod(i.pos.x, 2.0) > 1.0 ? 0.25 : 0.0) + (fmod(i.pos.y, 2.0) > 1.0 ? 0.5 : 0.0);
		float seen = 0.0;
		for (int d = 0; d < DIRS; d++) {
			float a = (d + turn) * (6.2831853 / DIRS);
			float2 dir = float2(cos(a), sin(a));
			float r = 1.5;
			float tanMax = 0.0;
			for (int s = 0; s < STEPS; s++) {
				float h = tex2Dlod(_CityBakeSource, float4(uv + dir * r / _CityBakeRegion.xy, 0, 0)).r;
				tanMax = max(tanMax, (h - h0) / r);
				r *= growth;
			}
			seen += 1.0 / (1.0 + tanMax * tanMax);
		}
		return seen / DIRS;
	}

	static const float Weights[5] = { 0.227027, 0.1945946, 0.1216216, 0.054054, 0.016216 };

	float Standing (float2 uv) {
		float h = tex2Dlod(_CityBakeSource, float4(uv, 0, 0)).r;
		return smoothstep(_CityBakeBlur.y, _CityBakeBlur.z, h);
	}

	float4 fragBlurAcross (v2f i) : SV_Target {
		float2 uv = i.pos.xy * _CityBakeSize.zw;
		float2 step = float2(_CityBakeSize.z * _CityBakeBlur.x * 0.25, 0.0);
		float s = Standing(uv) * Weights[0];
		for (int k = 1; k < 5; k++) s += (Standing(uv + step * k) + Standing(uv - step * k)) * Weights[k];
		return s;
	}

	float4 fragBlurDown (v2f i) : SV_Target {
		float2 uv = i.pos.xy * _CityBakeSize.zw;
		float2 step = float2(0.0, _CityBakeSize.w * _CityBakeBlur.x * 0.25);
		float s = tex2Dlod(_CityBakeSource, float4(uv, 0, 0)).r * Weights[0];
		for (int k = 1; k < 5; k++) {
			s += (tex2Dlod(_CityBakeSource, float4(uv + step * k, 0, 0)).r + tex2Dlod(_CityBakeSource, float4(uv - step * k, 0, 0)).r) * Weights[k];
		}
		return s;
	}
	ENDCG

	Pass {
		CGPROGRAM
		#pragma vertex vert
		#pragma fragment fragSky
		ENDCG
	}
	Pass {
		CGPROGRAM
		#pragma vertex vert
		#pragma fragment fragBlurAcross
		ENDCG
	}
	Pass {
		CGPROGRAM
		#pragma vertex vert
		#pragma fragment fragBlurDown
		ENDCG
	}
}
Fallback Off
}
