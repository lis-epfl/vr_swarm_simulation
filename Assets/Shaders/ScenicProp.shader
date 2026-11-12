// The scenic city's street furniture: the Modular City Pack's own meshes and texture atlas, drawn as weathered street
// furniture rather than shiny toys. CityDressing puts it in place of the pack's material on every prop it copies
// (its materialSwaps, which ScenicPatchCityWorldBuilder fills); nothing is copied out of the pack.
//
// The pack's material is Standard (Specular setup) at smoothness 0.95 with saturated paint, which under the scenic
// sky reads as plastic. Here the atlas is graded (saturation compressed, the brightest paint toned down, a little
// warmer), the gloss cut to a matte sheen, and dirt laid on in world space: a blotchy grime at a few metres and a
// darkening at the foot, where splash and wear collect. Lit like the city's other surfaces.
Shader "Swarm/Scenic Prop" {
Properties {
	_MainTex ("Atlas", 2D) = "white" {}
	_Color ("Tint", Color) = (1, 1, 1, 1)
	[NoScaleOffset] _SpecGlossMap ("Specular (RGB) and smoothness (A), optional", 2D) = "black" {}
	_Saturation ("Saturation kept", Range(0, 1.5)) = 0.62
	_MaxAlbedo ("Brightest albedo", Range(0.3, 1)) = 0.72
	_Gloss ("Smoothness scale", Range(0, 1)) = 0.45
	_SpecScale ("Specular scale", Range(0, 1)) = 0.35
	_Grime ("Grime", Range(0, 1)) = 0.3
	_FootDarken ("Darker at the foot", Range(0, 1)) = 0.25
	_GroundHeight ("The ground's height (m)", Float) = 0.1
}

SubShader {
	Tags { "RenderType"="Opaque" "Queue"="Geometry" }
	Pass {
		Tags { "LightMode"="ForwardBase" }
		CGPROGRAM
		#pragma vertex vert
		#pragma fragment frag
		#pragma target 3.0
		#include "UnityCG.cginc"
		#include "Lighting.cginc"

		sampler2D _MainTex, _SpecGlossMap, _ScenicNoise;
		float4 _MainTex_ST;
		fixed4 _Color;
		float _Saturation, _MaxAlbedo, _Gloss, _SpecScale, _Grime, _FootDarken, _GroundHeight;

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
			o.uv = TRANSFORM_TEX(v.uv, _MainTex);
			o.wpos = mul(unity_ObjectToWorld, float4(v.vertex.xyz, 1.0)).xyz;
			o.normal = UnityObjectToWorldNormal(v.normal);
			return o;
		}

		fixed4 frag (v2f i, fixed facing : VFACE) : SV_Target {
			float3 N = normalize(i.normal) * (facing > 0 ? 1.0 : -1.0);
			float3 V = normalize(_WorldSpaceCameraPos - i.wpos);
			float3 L = normalize(_WorldSpaceLightPos0.xyz);

			float3 c = tex2D(_MainTex, i.uv).rgb * _Color.rgb;
			float luma = dot(c, float3(0.2126, 0.7152, 0.0722));
			// saturated paint loses more of its saturation than muted paint does
			float chroma = max(c.r, max(c.g, c.b)) - min(c.r, min(c.g, c.b));
			c = lerp(luma.xxx, c, _Saturation * lerp(1.0, 0.75, saturate(chroma * 2.0)));
			c *= float3(1.03, 1.0, 0.95);
			float top = max(c.r, max(c.g, c.b));
			c *= top > _MaxAlbedo ? _MaxAlbedo / top : 1.0;

			// grime, blotchy at a few metres, and darker towards the ground
			float blot = tex2D(_ScenicNoise, i.wpos.xz / 3.1 + i.wpos.y * 0.13).g * 0.6 + tex2D(_ScenicNoise, i.wpos.xy / 1.3 + 0.4).r * 0.4;
			c *= 1.0 - _Grime * saturate((blot - 0.35) * 1.6);
			c *= 1.0 - _FootDarken * saturate(1.0 - (i.wpos.y - _GroundHeight) / 0.8);

			float4 sg = tex2D(_SpecGlossMap, i.uv);
			float smooth = sg.a * _Gloss;
			float3 spec = sg.rgb * _SpecScale;
			float ndl = saturate(dot(N, L));
			float3 H = normalize(L + V);
			float power = exp2(1.0 + 9.0 * smooth);
			float3 highlight = spec * pow(saturate(dot(N, H)), power) * (power + 2.0) / 8.0 * ndl;
			float3 col = c * (max(ShadeSH9(float4(N, 1.0)), 0.0) + _LightColor0.rgb * ndl) + _LightColor0.rgb * highlight;
			return fixed4(col, 1.0);
		}
		ENDCG
	}
}
Fallback "VertexLit"
}
