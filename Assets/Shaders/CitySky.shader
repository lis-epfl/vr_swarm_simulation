// The scenic city's sky: the upper hemisphere of a photographed sky panorama (equirectangular, its bottom row the
// horizon), turned about the vertical so its sun stands where the scene's directional light comes from. Below the
// horizon there is no photograph -- a "pure sky" panorama mirrors the sky there -- so it fades into a ground haze
// colour, which is also what a glass facade reflects when it looks down. CityLighting's haze uses the same horizon
// colours, so the far city fades into this sky.
Shader "Swarm/City Sky" {
Properties {
	[NoScaleOffset] _MainTex ("Upper hemisphere (equirect, bottom row = horizon)", 2D) = "grey" {}
	_Rotation ("Turn (fraction of a circle)", Range(0, 1)) = 0
	_Exposure ("Exposure", Range(0, 4)) = 1
	_Tint ("Tint", Color) = (1, 1, 1, 1)
	_GroundColor ("Below the horizon", Color) = (0.30, 0.32, 0.36, 1)
	_HorizonBlend ("Horizon blend (deg)", Range(0.1, 20)) = 4
}

SubShader {
	Tags { "Queue"="Background" "RenderType"="Background" "PreviewType"="Skybox" }
	Cull Off ZWrite Off

	Pass {
		CGPROGRAM
		#pragma vertex vert
		#pragma fragment frag
		#pragma target 3.0
		#include "UnityCG.cginc"

		sampler2D _MainTex;
		float4 _MainTex_TexelSize;
		float _Rotation, _Exposure, _HorizonBlend;
		fixed4 _Tint, _GroundColor;

		struct appdata {
			float4 vertex : POSITION;
			UNITY_VERTEX_INPUT_INSTANCE_ID
		};

		struct v2f {
			float4 pos : SV_POSITION;
			float3 dir : TEXCOORD0;
			UNITY_VERTEX_OUTPUT_STEREO
		};

		v2f vert (appdata v) {
			v2f o;
			UNITY_SETUP_INSTANCE_ID(v);
			UNITY_INITIALIZE_VERTEX_OUTPUT_STEREO(o);
			o.pos = UnityObjectToClipPos(v.vertex);
			o.dir = v.vertex.xyz;
			return o;
		}

		fixed4 frag (v2f i) : SV_Target {
			float3 d = normalize(i.dir);
			float u = atan2(d.x, d.z) * (0.5 / UNITY_PI) + _Rotation;
			float elev = asin(clamp(d.y, -1.0, 1.0)) * (2.0 / UNITY_PI);        // 0 horizon, 1 zenith
			float v = saturate(elev);
			// The seam of atan2 would otherwise pick the smallest mip along one column of the sky.
			float2 uv = float2(u, v);
			float2 dx = ddx(uv), dy = ddy(uv);
			dx.x -= round(dx.x);
			dy.x -= round(dy.x);
			float3 sky = tex2Dgrad(_MainTex, float2(frac(u), clamp(v, 0.5 * _MainTex_TexelSize.y, 1.0)), dx, dy).rgb;
			// Below the horizon the photograph's bottom row would be drawn out into vertical streaks: within half a
			// degree it gives way to that row blurred along the horizon (a small mip), which then fades into the ground.
			float3 band = tex2Dlod(_MainTex, float4(frac(u), 2.0 * _MainTex_TexelSize.y, 0, 7)).rgb;
			sky = lerp(sky, band, saturate(-elev * 90.0 / 0.5));
			float below = saturate(-elev * 90.0 / _HorizonBlend);
			float3 c = lerp(sky, _GroundColor.rgb, smoothstep(0.0, 1.0, below));
			return fixed4(c * _Tint.rgb * _Exposure, 1);
		}
		ENDCG
	}
}
Fallback Off
}
