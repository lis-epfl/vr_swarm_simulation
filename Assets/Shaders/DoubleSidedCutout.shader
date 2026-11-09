// Leaves for CityDressing's trees: the Modular City Pack's Doublesided_IF/Transparent/Diffuse, alpha-tested instead of
// blended. The pack's shader blends without writing depth, so leaves only layer correctly when each tree is its own
// renderer, sorted by Unity; CityDressing merges its trees into a few meshes at Play, where blended leaves would draw in
// index order and overlapping trees would show through each other. A cutout needs no sorting and writes depth.
Shader "Swarm/Double-sided Cutout" {
Properties {
	_Color ("Main Color", Color) = (1,1,1,1)
	_MainTex ("Base (RGB) Trans (A)", 2D) = "white" {}
	_Cutoff ("Alpha cutoff", Range(0,1)) = 0.4
}

SubShader {
	Tags {"Queue"="AlphaTest" "IgnoreProjector"="True" "RenderType"="TransparentCutout"}
	LOD 200
	Cull Off
CGPROGRAM
#pragma surface surf Lambert alphatest:_Cutoff

sampler2D _MainTex;
fixed4 _Color;

struct Input {
	float2 uv_MainTex;
};

void surf (Input IN, inout SurfaceOutput o) {
	fixed4 c = tex2D(_MainTex, IN.uv_MainTex) * _Color;
	o.Albedo = c.rgb;
	o.Alpha = c.a;
}
ENDCG
}

Fallback "Legacy Shaders/Transparent/Cutout/VertexLit"
}
