// The scenic city's leaves: CityDressing's cutoutLeaves in ScenicPatchCityWorld, in place of Swarm/Double-sided Cutout
// (Lambert). Alpha-tested and double-sided as that one is, for the same reason (the dressing's trees are merged at Play,
// so blended leaves would draw out of order), and so CityLighting's caster draws them cut out too
// (RenderType TransparentCutout, _MainTex, _Cutoff).
//
// What changes is the light: a leaf lets light through, so a canopy lit from behind glows rather than going black, and
// the light it gets is wrapped round it rather than cut off at its edge. And no two trees are quite the same green: the
// colour and lightness drift with a noise in world space at about a tree's spacing, since the merged mesh no longer
// knows one tree from the next.
Shader "Swarm/Scenic Leaves" {
Properties {
	_MainTex ("Leaves (RGB) and alpha (A)", 2D) = "white" {}
	_Color ("Tint", Color) = (0.78, 0.74, 0.52, 1)
	_Cutoff ("Alpha cutoff", Range(0, 1)) = 0.4
	_Wrap ("Wrapped light", Range(0, 1)) = 0.45
	_Translucency ("Light through the leaves", Range(0, 2)) = 0.6
	_Variation ("Tree to tree colour variation", Range(0, 1)) = 0.5
}

SubShader {
	Tags { "Queue"="AlphaTest" "IgnoreProjector"="True" "RenderType"="TransparentCutout" }
	Cull Off
	Pass {
		Tags { "LightMode"="ForwardBase" }
		CGPROGRAM
		#pragma vertex vert
		#pragma fragment frag
		#pragma target 3.0
		#include "UnityCG.cginc"
		#include "Lighting.cginc"

		sampler2D _MainTex, _ScenicNoise;
		float4 _MainTex_ST;
		fixed4 _Color;
		float _Cutoff, _Wrap, _Translucency, _Variation;

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
			fixed4 t = tex2D(_MainTex, i.uv);
			clip(t.a - _Cutoff);
			float3 N = normalize(i.normal) * (facing > 0 ? 1.0 : -1.0);
			float3 V = normalize(_WorldSpaceCameraPos - i.wpos);
			float3 L = normalize(_WorldSpaceLightPos0.xyz);

			// tree to tree: lightness and a drift between a yellower and a bluer green
			float3 n = tex2D(_ScenicNoise, i.wpos.xz / 23.0 + 0.71).rgb;
			float3 c = t.rgb * _Color.rgb;
			c *= 1.0 + _Variation * 0.5 * (n.g - 0.5);
			c *= lerp(float3(1, 1, 1), float3(1.0 + 0.25 * (n.r - 0.5), 1.0, 1.0 - 0.35 * (n.r - 0.5)), _Variation * 2.0);
			// the canopy is darker low down and inside: a leaf low on the tree sees less sky
			float shade = saturate(0.75 + 0.25 * N.y);

			float wrap = saturate((dot(N, L) + _Wrap) / (1.0 + _Wrap));
			float through = pow(saturate(dot(V, -L)), 4.0) * _Translucency;
			float3 ambient = max(ShadeSH9(float4(normalize(N + float3(0, 0.6, 0)), 1.0)), 0.0) * shade;
			float3 col = c * (ambient + _LightColor0.rgb * (wrap * 0.85 + through * float3(0.9, 1.0, 0.55)));
			return fixed4(col, 1.0);
		}
		ENDCG
	}
}
Fallback "Legacy Shaders/Transparent/Cutout/VertexLit"
}
