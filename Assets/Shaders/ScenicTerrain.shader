// The scenic city's Terrain material (ScenicGroundBuilder sets it as every tile's materialTemplate): the four splat
// layers of a tile, laid in world metres rather than in the tile's own UVs, so neighbouring tiles join with no seam
// whatever their layers' tile sizes.
//
// Against the look of a terrain seen from far off, where one photograph repeated every few metres turns into a single
// flat colour or a visible grid:
//   - each layer is sampled twice, at its own size and at 2.9 times it turned, blended by a noise of a few tens of
//     metres, so no repeat lines up;
//   - its colour and lightness drift over hundreds of metres (the noise texture at two scales);
//   - on steep ground (the mountain's faces) it is projected from the sides as well as from above (triplanar), so the
//     rock does not stretch into vertical streaks.
// Lit like the city's other surfaces (ambient probe plus the sun, no shadow of its own): CityLighting lays the shadows
// over the picture where it has them. Four layers only (one pass): every scenic tile is painted with four.
//
// Requires Draw Instanced off on the Terrain (ScenicGroundBuilder turns it off): the instanced path builds the mesh
// in a vertex function this shader does not have.
Shader "Swarm/Scenic Terrain" {
Properties {
	[HideInInspector] _Control ("Control (RGBA)", 2D) = "red" {}
	[HideInInspector] _Splat3 ("Layer 3 (A)", 2D) = "grey" {}
	[HideInInspector] _Splat2 ("Layer 2 (B)", 2D) = "grey" {}
	[HideInInspector] _Splat1 ("Layer 1 (G)", 2D) = "grey" {}
	[HideInInspector] _Splat0 ("Layer 0 (R)", 2D) = "grey" {}
	[HideInInspector] _Normal3 ("Normal 3 (A)", 2D) = "bump" {}
	[HideInInspector] _Normal2 ("Normal 2 (B)", 2D) = "bump" {}
	[HideInInspector] _Normal1 ("Normal 1 (G)", 2D) = "bump" {}
	[HideInInspector] _Normal0 ("Normal 0 (R)", 2D) = "bump" {}
	[HideInInspector] _MainTex ("BaseMap (RGB)", 2D) = "grey" {}
	[HideInInspector] _Color ("Main Color", Color) = (1, 1, 1, 1)
	_TerrainSize ("Tile size (m): the splat layers' ST are relative to it", Float) = 1000
	[NoScaleOffset] _MacroNoise ("Noise (R coarse, G, B fine)", 2D) = "grey" {}
	_MacroMetres ("Colour drift: metres to the noise's coarse repeat", Float) = 700
	_MacroAmount ("Colour drift: how much", Range(0, 1)) = 0.35
	_BombMetres ("Second sample: metres to the blend's repeat", Float) = 90
	_SteepStart ("Triplanar from this slope (cos of the normal's angle with up)", Range(0.5, 1)) = 0.85
	_NormalStrength ("Normal maps", Range(0, 2)) = 1
}

SubShader {
	Tags { "Queue"="Geometry-100" "RenderType"="Opaque" "TerrainCompatible"="True" }
	Pass {
		Tags { "LightMode"="ForwardBase" }
		CGPROGRAM
		#pragma vertex vert
		#pragma fragment frag
		#pragma target 4.5
		#include "UnityCG.cginc"
		#include "Lighting.cginc"

		sampler2D _Control;
		float4 _Control_TexelSize;
		sampler2D _Splat0, _Splat1, _Splat2, _Splat3;
		float4 _Splat0_ST, _Splat1_ST, _Splat2_ST, _Splat3_ST;
		sampler2D _Normal0, _Normal1, _Normal2, _Normal3;
		sampler2D _MacroNoise;
		float _TerrainSize, _MacroMetres, _MacroAmount, _BombMetres, _SteepStart, _NormalStrength;

		struct appdata {
			float4 vertex : POSITION;
			float3 normal : NORMAL;
			float2 uv : TEXCOORD0;
		};

		struct v2f {
			float4 pos : SV_POSITION;
			float2 uv : TEXCOORD0;
			float3 wpos : TEXCOORD1;
			float3 normal : TEXCOORD2;
		};

		v2f vert (appdata v) {
			v2f o;
			o.pos = UnityObjectToClipPos(v.vertex);
			o.uv = v.uv;
			o.wpos = mul(unity_ObjectToWorld, float4(v.vertex.xyz, 1.0)).xyz;
			o.normal = UnityObjectToWorldNormal(v.normal);
			return o;
		}

		// A layer's metres to one repeat, from its ST (the terrain sets size / tileSize).
		float Metres (float4 st) { return _TerrainSize / max(st.x, 1e-3); }

		// One layer's colour at a point seen from one side (uv in metres), the two samples blended by the bomb mask.
		float3 Sample2 (sampler2D tex, float2 m, float metres, float bomb) {
			float2 a = m / metres;
			// the second sample: 2.9 times larger, turned by ~37 degrees and shifted, so its repeat never lines up
			float2 b = float2(0.799 * a.x - 0.602 * a.y, 0.602 * a.x + 0.799 * a.y) * 0.345 + 0.37;
			float3 ca = tex2D(tex, a).rgb, cb = tex2D(tex, b).rgb;
			return lerp(ca, cb, bomb);
		}

		// The same, projected from above and, on steep ground, from the two sides.
		float3 Layer (sampler2D tex, float3 w, float3 tri, float metres, float bomb, bool steep) {
			float3 c = Sample2(tex, w.xz, metres, bomb) * tri.y;
			if (steep) {
				c += Sample2(tex, w.zy, metres, bomb) * tri.x;
				c += Sample2(tex, w.xy, metres, bomb) * tri.z;
			}
			return c;
		}

		float2 NormalXY (sampler2D tex, float2 m, float metres) {
			return UnpackNormal(tex2D(tex, m / metres)).xy;
		}

		fixed4 frag (v2f i) : SV_Target {
			float3 N = normalize(i.normal);
			float3 w = i.wpos;

			// the splat weights, read at the texel centres as Unity's own terrain shader reads them
			float2 splatUV = (i.uv * (_Control_TexelSize.zw - 1.0) + 0.5) * _Control_TexelSize.xy;
			float4 ctrl = tex2D(_Control, splatUV);
			ctrl /= max(dot(ctrl, 1.0), 1e-3);

			float3 tri = pow(abs(N), 6.0);
			tri /= dot(tri, 1.0);
			bool steep = N.y < _SteepStart;
			if (!steep) tri = float3(0, 1, 0);

			float bomb = smoothstep(0.35, 0.65, tex2D(_MacroNoise, w.xz / _BombMetres + 0.13).b);
			float m0 = Metres(_Splat0_ST), m1 = Metres(_Splat1_ST), m2 = Metres(_Splat2_ST), m3 = Metres(_Splat3_ST);

			float3 albedo = 0;
			if (ctrl.r > 0.004) albedo += ctrl.r * Layer(_Splat0, w, tri, m0, bomb, steep);
			if (ctrl.g > 0.004) albedo += ctrl.g * Layer(_Splat1, w, tri, m1, bomb, steep);
			if (ctrl.b > 0.004) albedo += ctrl.b * Layer(_Splat2, w, tri, m2, bomb, steep);
			if (ctrl.a > 0.004) albedo += ctrl.a * Layer(_Splat3, w, tri, m3, bomb, steep);

			// colour and lightness drifting over hundreds of metres
			float3 macro = tex2D(_MacroNoise, w.xz / _MacroMetres).rgb;
			float3 macro2 = tex2D(_MacroNoise, w.xz / (_MacroMetres * 0.23) + 0.61).rgb;
			float lightness = 1.0 + _MacroAmount * (0.55 * (macro.g - 0.5) + 0.45 * (macro2.r - 0.5));
			float3 drift = float3(1.0 + 0.10 * (macro.r - 0.5), 1.0, 1.0 - 0.12 * (macro.r - 0.5));
			albedo *= lightness * lerp(float3(1, 1, 1), drift, _MacroAmount * 2.0);

			// normal: the layers' maps from above, whiteout-blended onto the ground's own normal; none on the steep sides
			float3 n = N;
			if (_NormalStrength > 0.0 && tri.y > 0.05) {
				float2 d = 0;
				if (ctrl.r > 0.004) d += ctrl.r * NormalXY(_Normal0, w.xz, m0);
				if (ctrl.g > 0.004) d += ctrl.g * NormalXY(_Normal1, w.xz, m1);
				if (ctrl.b > 0.004) d += ctrl.b * NormalXY(_Normal2, w.xz, m2);
				if (ctrl.a > 0.004) d += ctrl.a * NormalXY(_Normal3, w.xz, m3);
				n = normalize(N + float3(d.x, 0, d.y) * (_NormalStrength * tri.y));
			}

			float3 L = normalize(_WorldSpaceLightPos0.xyz);
			float3 c = albedo * (max(ShadeSH9(float4(n, 1.0)), 0.0) + _LightColor0.rgb * saturate(dot(n, L)));
			return fixed4(c, 1.0);
		}
		ENDCG
	}
}
Dependency "AddPassShader" = "Hidden/TerrainEngine/Splatmap/Standard-AddPass"
Dependency "BaseMapShader" = "Hidden/TerrainEngine/Splatmap/Standard-Base"
Dependency "BaseMapGenShader" = "Hidden/TerrainEngine/Splatmap/Standard-BaseGen"
Fallback Off
}
