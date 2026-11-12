// The scenic city's ground: its roads, footpaths and lawns. One draw each, as before; what changes is the surface.
//
// A road (Marks on) keeps the mesh and the mapping StreetSurface gives it, and reads the Modular City Pack's road
// textures only for what is painted on them: where the texture is bright there is paint (lane dashes, zebra stripes),
// and its soft darker bands are the tyre tracks. The asphalt itself is a photographed surface laid in world metres, so
// it does not repeat with the lane tiles, with the paint worn through in places.
// A footpath or a lawn (Marks off) is its own photographed surface in the mesh's mapping.
// Both carry stains at two scales from a noise texture, so no stretch of ground is one flat tone.
//
// Lit by the scene's sun and ambient probe with no shadow of its own: CityLighting lays the shadows over the picture.
Shader "Swarm/Scenic Ground" {
Properties {
	_MainTex ("Surface, or the road's painted pattern (UV0)", 2D) = "white" {}
	[NoScaleOffset] _BumpMap ("Surface normal (UV0)", 2D) = "bump" {}
	_Color ("Tint", Color) = (1, 1, 1, 1)
	[Toggle(SCENIC_MARKS)] _Marks ("A road: the main texture gives the paint", Float) = 0
	[NoScaleOffset] _SurfaceTex ("Road surface (world metres)", 2D) = "grey" {}
	[NoScaleOffset] _SurfaceBump ("Road surface normal", 2D) = "bump" {}
	_SurfaceMetres ("Road surface: metres per repeat", Float) = 3
	_PaintColor ("Paint", Color) = (0.62, 0.62, 0.60, 1)
	_PaintWear ("Paint wear", Range(0, 1)) = 0.5
	[NoScaleOffset] _Noise ("Noise (R coarse, G, B fine)", 2D) = "grey" {}
	_StainMetres ("Stains: metres per repeat", Float) = 41
	_Stains ("Stains", Range(0, 1)) = 0.3
	_BumpScale ("Normal strength", Range(0, 2)) = 1
}

SubShader {
	Tags { "RenderType"="Opaque" "Queue"="Geometry" }
	LOD 200

	Pass {
		Tags { "LightMode"="ForwardBase" }
		CGPROGRAM
		#pragma vertex vert
		#pragma fragment frag
		#pragma target 3.0
		#pragma shader_feature_local SCENIC_MARKS
		#include "UnityCG.cginc"
		#include "Lighting.cginc"

		sampler2D _MainTex, _BumpMap, _SurfaceTex, _SurfaceBump, _Noise;
		float4 _MainTex_ST;
		fixed4 _Color, _PaintColor;
		float _SurfaceMetres, _PaintWear, _StainMetres, _Stains, _BumpScale;

		struct v2f {
			float4 pos : SV_POSITION;
			float2 uv : TEXCOORD0;
			float3 wpos : TEXCOORD1;
			float3 normal : TEXCOORD2;
		};

		v2f vert (appdata_base v) {
			v2f o;
			o.pos = UnityObjectToClipPos(v.vertex);
			o.uv = TRANSFORM_TEX(v.texcoord, _MainTex);
			o.wpos = mul(unity_ObjectToWorld, float4(v.vertex.xyz, 1.0)).xyz;
			o.normal = UnityObjectToWorldNormal(v.normal);
			return o;
		}

		fixed4 frag (v2f i) : SV_Target {
			float3 N = normalize(i.normal);
			float2 w = i.wpos.xz;
			float3 n1 = tex2D(_Noise, w / _StainMetres).rgb;
			float3 n2 = tex2D(_Noise, w / (_StainMetres * 0.19) + 0.37).rgb;
			float stain = 1.0 + _Stains * (0.9 * (n1.g - 0.5) + 0.5 * (n2.r - 0.5));
			float3 albedo, nm;
		#ifdef SCENIC_MARKS
			float3 pattern = tex2D(_MainTex, i.uv).rgb;
			float lum = dot(pattern, 0.3333);
			// paint where the pattern is bright, worn through where the fine noise is low
			float paint = smoothstep(0.17, 0.36, lum);
			paint *= lerp(1.0, smoothstep(0.18, 0.55, n2.b * 0.7 + n1.r * 0.5), _PaintWear);
			float tracks = clamp(lum / 0.082, 0.78, 1.1);
			float2 suv = w / _SurfaceMetres;
			albedo = tex2D(_SurfaceTex, suv).rgb * tracks * stain;
			albedo = lerp(albedo, _PaintColor.rgb * (0.82 + 0.3 * n2.g), saturate(paint));
			nm = UnpackNormal(tex2D(_SurfaceBump, suv));
		#else
			albedo = tex2D(_MainTex, i.uv).rgb * stain;
			nm = UnpackNormal(tex2D(_BumpMap, i.uv));
		#endif
			albedo *= _Color.rgb;
			// The mapping runs with world x and z on the flat; a kerb's face keeps its own normal.
			float3 bumped = normalize(float3(nm.x * _BumpScale, nm.z, nm.y * _BumpScale));
			float3 n = N.y > 0.7 ? bumped : N;
			float3 L = normalize(_WorldSpaceLightPos0.xyz);
			float3 c = albedo * (max(ShadeSH9(float4(n, 1.0)), 0.0) + _LightColor0.rgb * saturate(dot(n, L)));
			return fixed4(c, 1.0);
		}
		ENDCG
	}
}
Fallback "VertexLit"
}
