// CityLighting's pass over a drone camera's picture, drawn once per render after the opaque geometry: the sun's shadows,
// the sky hidden by the buildings, and the haze, all worked out from the depth buffer the render has just written.
//
// Nothing in the scene is drawn a second time for it. The shadows come from one depth map of the city seen from the
// sun and the sky's visibility from one small volume, both made once at the start of Play (the city never moves), so a
// render pays two full-screen draws on the GPU and nothing on the CPU -- where a real-time shadow pass would cull and
// submit the city again for every drone camera, which is the cost the feeds cannot take (CLAUDE.md).
//
// Pass 0 multiplies the picture. Every material has already lit its surface as albedo * (ambient + sun * N.L), with no
// shadow and the whole sky. What it should have been is albedo * (ambient * sky + sun * N.L * shadow), so the picture is
// multiplied by the ratio of the two, which does not depend on the albedo: the ambient and the sun are known, and the
// normal is rebuilt from the depth buffer. Where a surface faces away from the sun the ratio leaves the sun term alone,
// which is also why a shadow edge cannot show acne there.
// Pass 1 blends the haze over it.
Shader "Hidden/Swarm/City Lighting Composite" {
SubShader {
	Cull Off ZWrite Off ZTest Always

	CGINCLUDE
	#pragma target 4.5
	#include "UnityCG.cginc"

	Texture2D<float> _CityDepthTex;
	float4 _CityTargetSize;          // w, h, 1/w, 1/h
	float4 _CityCamPos;
	float4 _CityRayX, _CityRayY, _CityRayZ;   // the view ray through a pixel, to eye depth 1: Z + ndc.x X + ndc.y Y
	float4 _CityZ;                   // near, far, 1 if the depth buffer is reversed

	float4x4 _CityWorldToShadow;     // world -> (u, v, depth as stored)
	UNITY_DECLARE_SHADOWMAP(_CitySunShadowMap);
	float4 _CityShadowTexel;         // 1/w, 1/h, softness in texels, depth bias

	sampler3D _CitySkyVolume;
	float4 _CityVolumeMin, _CityVolumeInvSize;
	sampler2D _CityCanopyMap;
	float4 _CityRegion;              // min x, min z, 1/size x, 1/size z

	float4 _CitySunDir;              // towards the sun
	float4 _CitySunColor;            // linear, intensity included
	float4 _CitySHAr, _CitySHAg, _CitySHAb, _CitySHBr, _CitySHBg, _CitySHBb, _CitySHC;
	float4 _CityShade;               // sun-shadow strength, sky-occlusion strength, bounce albedo, canopy strength
	float4 _CityOffsets;             // shadow normal offset (m), volume normal offset (m), least sky left, ground height for the canopy term
	float4 _CityHaze;                // density (1/m), height falloff (1/m), base height, most haze
	float4 _CityHazeColor;
	float4 _CityHazeSunColor;        // a: exponent of the glow round the sun

	struct v2f {
		float4 pos : SV_POSITION;
	};

	// The mesh is a triangle given in clip space, covering the target.
	v2f vert (float4 vertex : POSITION) {
		v2f o;
		o.pos = float4(vertex.xy, 0.5, 1.0);
		return o;
	}

	float RawDepth (int2 p) {
		p = clamp(p, int2(0, 0), int2(_CityTargetSize.xy) - 1);
		return _CityDepthTex.Load(int3(p, 0));
	}

	float EyeDepth (float raw) {
		// near * far / (far - d (far - near)) with d the conventional depth; well conditioned at any range
		float d = _CityZ.z > 0.5 ? 1.0 - raw : raw;
		return _CityZ.x * _CityZ.y / (_CityZ.y - d * (_CityZ.y - _CityZ.x));
	}

	float3 WorldAt (float2 pixel, float eye) {
		float2 ndc = pixel * _CityTargetSize.zw * 2.0 - 1.0;
		// A camera drawing into a RenderTexture draws it bottom row first (_ProjectionParams.x < 0).
		ndc.y *= _ProjectionParams.x < 0.0 ? 1.0 : -1.0;
		return _CityCamPos.xyz + (_CityRayZ.xyz + ndc.x * _CityRayX.xyz + ndc.y * _CityRayY.xyz) * eye;
	}

	// The surface under a pixel: its world position and the normal of the triangle it belongs to, taking each
	// difference from the neighbour nearer in depth so a silhouette does not bend it.
	bool Surface (float4 pos, out float3 world, out float3 normal) {
		int2 p = int2(pos.xy);
		float raw = RawDepth(p);
		world = 0;
		normal = float3(0, 1, 0);
		if ((_CityZ.z > 0.5 ? raw : 1.0 - raw) <= 0.0) return false;     // the sky
		float e0 = EyeDepth(raw);
		float el = EyeDepth(RawDepth(p + int2(-1, 0))), er = EyeDepth(RawDepth(p + int2(1, 0)));
		float ed = EyeDepth(RawDepth(p + int2(0, -1))), eu = EyeDepth(RawDepth(p + int2(0, 1)));
		world = WorldAt(pos.xy, e0);
		float3 dx = abs(el - e0) < abs(er - e0) ? world - WorldAt(pos.xy + float2(-1, 0), el) : WorldAt(pos.xy + float2(1, 0), er) - world;
		float3 dy = abs(ed - e0) < abs(eu - e0) ? world - WorldAt(pos.xy + float2(0, -1), ed) : WorldAt(pos.xy + float2(0, 1), eu) - world;
		float3 n = cross(dy, dx);
		float len = length(n);
		n = len > 1e-12 ? n / len : float3(0, 1, 0);
		normal = dot(n, world - _CityCamPos.xyz) > 0.0 ? -n : n;
		return true;
	}

	float3 Ambient (float3 n) {
		float4 v = float4(n, 1.0);
		float3 x1 = float3(dot(_CitySHAr, v), dot(_CitySHAg, v), dot(_CitySHAb, v));
		float4 vB = n.xyzz * n.yzzx;
		float3 x2 = float3(dot(_CitySHBr, vB), dot(_CitySHBg, vB), dot(_CitySHBb, vB));
		float vC = n.x * n.x - n.y * n.y;
		return max(x1 + x2 + _CitySHC.rgb * vC, 1e-3);
	}

	float SunShadow (float3 world, float3 n, float ndl) {
		float3 p = world + n * (_CityOffsets.x * (1.0 + 2.0 * (1.0 - ndl)));
		float3 sc = mul(_CityWorldToShadow, float4(p, 1.0)).xyz;
		if (any(sc.xy < 0.0) || any(sc.xy > 1.0) || sc.z < 0.0 || sc.z > 1.0) return 1.0;
		sc.z += (_CityZ.z > 0.5 ? 1.0 : -1.0) * _CityShadowTexel.w;
		float2 t = _CityShadowTexel.xy * _CityShadowTexel.z;
		float s = 0.0;
		for (int y = -1; y <= 1; y++) {
			for (int x = -1; x <= 1; x++) {
				float w = (2.0 - abs(x)) * (2.0 - abs(y));
				s += w * UNITY_SAMPLE_SHADOW(_CitySunShadowMap, float3(sc.xy + float2(x, y) * t, sc.z));
			}
		}
		return s / 16.0;
	}

	// How much of the sky a surface still sees past the buildings, 1 in the open: the volume holds the share of the sky
	// seen from a point in the air, which beside a lone wall is the half the wall leaves, so it is measured against that.
	float SkySeen (float3 world, float3 n) {
		float3 q = (world + n * _CityOffsets.y - _CityVolumeMin.xyz) * _CityVolumeInvSize.xyz;
		float3 edge = min(q, 1.0 - q);
		float inside = saturate(min(edge.x, edge.z) * 40.0) * step(0.0, q.y);
		// the volume's rows run along world z and its slices up
		float v = tex3Dlod(_CitySkyVolume, float4(saturate(q.xzy), 0.0)).r;
		float seen = saturate(v / max(0.5 + 0.5 * n.y, 0.3));
		seen = seen / (1.0 - _CityShade.z * (1.0 - seen));        // light bounced between the facades fills part of it back in
		seen = max(seen, _CityOffsets.z);
		// the soft dark patch under a tree, a bench, a shelter: on the ground only
		float2 uv = (world.xz - _CityRegion.xy) * _CityRegion.zw;
		float ground = saturate(1.0 - (world.y - _CityOffsets.w) * 1.5) * saturate(n.y * 2.0 - 1.0);
		float canopy = tex2Dlod(_CityCanopyMap, float4(uv, 0.0, 0.0)).r * ground * _CityShade.w;
		seen *= 1.0 - canopy;
		return lerp(1.0, seen, inside * _CityShade.y);
	}

	float4 fragShade (v2f i) : SV_Target {
		float3 world, n;
		if (!Surface(i.pos, world, n)) discard;
		float ndl = saturate(dot(n, _CitySunDir.xyz));
		float3 ambient = Ambient(n);
		float3 sun = _CitySunColor.rgb * ndl;
		float shadow = ndl > 0.0 ? lerp(1.0, SunShadow(world, n, ndl), _CityShade.x) : 1.0;
		float seen = SkySeen(world, n);
		float3 ratio = (ambient * seen + sun * shadow) / (ambient + sun);
		return float4(ratio, 1.0);
	}

	float4 fragHaze (v2f i) : SV_Target {
		float3 world, n;
		if (!Surface(i.pos, world, n)) discard;
		float3 ray = world - _CityCamPos.xyz;
		float dist = length(ray);
		float3 view = ray / max(dist, 1e-4);
		// haze thinning with height: the integral of density * exp(-k (y - base)) along the ray
		float k = _CityHaze.y;
		float dy = view.y * dist;
		float kd = k * dy;
		float along = abs(kd) > 1e-3 ? (1.0 - exp(-kd)) / kd : 1.0 - 0.5 * kd;
		float od = _CityHaze.x * exp(-k * (_CityCamPos.y - _CityHaze.z)) * along * dist;
		float f = min(1.0 - exp(-od), _CityHaze.w);
		float glow = pow(saturate(dot(view, _CitySunDir.xyz)), _CityHazeSunColor.a);
		return float4(lerp(_CityHazeColor.rgb, _CityHazeSunColor.rgb, glow), f);
	}
	ENDCG

	Pass {      // 0: shadows and hidden sky
		Blend DstColor Zero
		CGPROGRAM
		#pragma vertex vert
		#pragma fragment fragShade
		ENDCG
	}
	Pass {      // 1: haze
		Blend SrcAlpha OneMinusSrcAlpha
		CGPROGRAM
		#pragma vertex vert
		#pragma fragment fragHaze
		ENDCG
	}
}
Fallback Off
}
