// The scenic city's buildings (ScenicBuilding makes the meshes): every building in one material, so they batch into a
// few draw calls, with what tells one from the next carried by its vertices -- the wall's photographed material and
// colour, the window pattern, the frames, the glass, the ground floor.
//
// A facade is a flat wall and everything else is worked out per pixel, which costs an FPV render nothing on the CPU:
//   - the wall: one layer of a texture array (colour; normal + roughness), laid in metres, tinted, with stains;
//   - windows: a rectangle in every bay of every floor, set back into the wall by the view ray (so the reveal, the sill
//     and the frame have depth and shift with the viewpoint), the sun's shadow of the reveal on the glass;
//   - behind the glass a room (interior mapping: the ray is followed to the room's walls, floor or ceiling), some lit,
//     some with blinds or curtains; over it the reflection of the sky, or of the street's other side where the
//     reflected ray would meet it, each pane tilted a hair as real glazing is;
//   - the ground floor: shopfronts with a sign band, an office lobby, or a masonry base, with doors.
// Edges are smoothed over a pixel and the detail fades out where a bay is only a few pixels wide, so a distant facade
// does not shimmer in a feed that has no anti-aliasing.
//
// Vertex data (ScenicBuilding.Quad): uv0 (s, y, edge length, bays) in metres along the wall and above the ground;
// uv1 (ground floor height, floor height, top of the windows, reveal); uv2 (wall layer, seed, ground style, window
// kind); uv3 (window width, height, sill as fractions of the bay and floor, mullions across + 10 x transoms);
// uv4 (trim colour, base layer); uv5 (glass tint, reflectance at normal incidence); uv6 (frame colour, blinds);
// uv7 (base tint, bays per shop + 8 x door seed); colour (wall tint as square roots, a: 0 a facade, 1 a plain surface in its layer).
//
// Lit by the scene's sun and ambient probe with no shadow of its own: CityLighting lays the shadows over the picture.
Shader "Swarm/Scenic Facade" {
Properties {
	_InteriorExposure ("Rooms: how bright seen from outside", Range(0, 2)) = 0.3
	_ShopExposure ("Shops: how bright seen from outside", Range(0, 2)) = 0.2
	_ReflectionBoost ("Glass: reflection strength", Range(0, 3)) = 1
	_Weathering ("Stains and streaks", Range(0, 2)) = 1
	_DetailPixels ("Bay width in pixels below which detail fades", Range(1, 30)) = 7
}

SubShader {
	Tags { "RenderType"="Opaque" "Queue"="Geometry" }
	LOD 300

	Pass {
		Tags { "LightMode"="ForwardBase" }
		CGPROGRAM
		#pragma vertex vert
		#pragma fragment frag
		#pragma target 4.5
		#include "UnityCG.cginc"
		#include "Lighting.cginc"

		Texture2DArray _ScenicWallAlbedo;
		SamplerState sampler_ScenicWallAlbedo;
		Texture2DArray _ScenicWallNormal;
		SamplerState sampler_ScenicWallNormal;
		sampler2D _ScenicNoise;
		float _ScenicWallMetres[16];
		float4 _ScenicCanyon;        // roof height, street width, brightness of a reflected building, 0
		float _InteriorExposure, _ShopExposure, _ReflectionBoost, _Weathering, _DetailPixels;

		struct appdata {
			float4 vertex : POSITION;
			float3 normal : NORMAL;
			float4 tangent : TANGENT;
			fixed4 color : COLOR;
			float4 uv0 : TEXCOORD0;
			float4 uv1 : TEXCOORD1;
			float4 uv2 : TEXCOORD2;
			float4 uv3 : TEXCOORD3;
			float4 uv4 : TEXCOORD4;
			float4 uv5 : TEXCOORD5;
			float4 uv6 : TEXCOORD6;
			float4 uv7 : TEXCOORD7;
		};

		// What a face's vertices share is taken from one of them (nointerpolation). Interpolated, the same value comes
		// out a hair different from pixel to pixel, and a hash of it turns that into hatching.
		struct v2f {
			float4 pos : SV_POSITION;
			float2 uv0 : TEXCOORD0;
			nointerpolation float2 face : TEXCOORD11;
			nointerpolation float4 uv1 : TEXCOORD1;
			nointerpolation float4 uv2 : TEXCOORD2;
			nointerpolation float4 uv3 : TEXCOORD3;
			nointerpolation float4 uv4 : TEXCOORD4;
			nointerpolation float4 uv5 : TEXCOORD5;
			nointerpolation float4 uv6 : TEXCOORD6;
			nointerpolation float4 uv7 : TEXCOORD7;
			float3 wpos : TEXCOORD8;
			float3 normal : TEXCOORD9;
			float4 tangent : TEXCOORD10;
			nointerpolation fixed4 color : COLOR;
		};

		v2f vert (appdata v) {
			v2f o;
			o.pos = UnityObjectToClipPos(v.vertex);
			o.wpos = mul(unity_ObjectToWorld, float4(v.vertex.xyz, 1.0)).xyz;
			o.normal = UnityObjectToWorldNormal(v.normal);
			o.tangent = float4(UnityObjectToWorldDir(v.tangent.xyz), v.tangent.w);
			o.uv0 = v.uv0.xy; o.face = v.uv0.zw; o.uv1 = v.uv1; o.uv2 = v.uv2; o.uv3 = v.uv3;
			o.uv4 = v.uv4; o.uv5 = v.uv5; o.uv6 = v.uv6; o.uv7 = v.uv7;
			o.color = v.color;
			return o;
		}

		// ---------------------------------------------------------------- small tools

		float hash1 (float2 p) {
			float3 p3 = frac(float3(p.xyx) * 0.1031);
			p3 += dot(p3, p3.yzx + 33.33);
			return frac((p3.x + p3.y) * p3.z);
		}

		float3 hash3 (float2 p) {
			float3 p3 = frac(float3(p.xyx) * float3(0.1031, 0.1030, 0.0973));
			p3 += dot(p3, p3.yxz + 33.33);
			return frac((p3.xxy + p3.yzz) * p3.zyx);
		}

		// 1 inside [lo, hi], falling to 0 over a pixel (w) at each end
		float band (float x, float lo, float hi, float w) {
			return saturate((x - lo) / w + 0.5) * saturate((hi - x) / w + 0.5);
		}

		// a line of half-width hw at c, smoothed over a pixel
		float stripe (float x, float c, float hw, float w) {
			return saturate((hw - abs(x - c)) / w + 0.5);
		}

		// the room colours a hash picks from
		float3 palette (float t) {
			return 0.5 + 0.5 * cos(6.2831853 * (t + float3(0.0, 0.33, 0.67)));
		}

		// a shop's sign: dark and muted for the most part, now and then a light one
		float3 signColour (float t) {
			static const float3 signs[8] = {
				float3(0.030, 0.033, 0.040), float3(0.020, 0.045, 0.090), float3(0.130, 0.022, 0.025), float3(0.020, 0.070, 0.045),
				float3(0.090, 0.060, 0.030), float3(0.045, 0.045, 0.050), float3(0.430, 0.400, 0.330), float3(0.180, 0.100, 0.030),
			};
			return signs[(int)min(floor(t * 8.0), 7.0)];
		}

		struct Wall {
			float3 albedo;
			float3 normal;       // tangent space
			float rough;
		};

		Wall SampleWall (float layer, float2 metres, float2 dx, float2 dy) {
			float k = 1.0 / _ScenicWallMetres[(int)layer];
			float3 uvw = float3(metres * k, layer);
			Wall w;
			w.albedo = _ScenicWallAlbedo.SampleGrad(sampler_ScenicWallAlbedo, uvw, dx * k, dy * k).rgb;
			float3 n = _ScenicWallNormal.SampleGrad(sampler_ScenicWallNormal, uvw, dx * k, dy * k).rgb;
			float2 xy = n.rg * 2.0 - 1.0;
			w.normal = float3(xy, sqrt(saturate(1.0 - dot(xy, xy))));
			w.rough = n.b;
			return w;
		}

		float3 Lit (float3 albedo, float3 n, float3 sunDir, float sunMask) {
			return albedo * (max(ShadeSH9(float4(n, 1.0)), 0.0) + _LightColor0.rgb * saturate(dot(n, sunDir)) * sunMask);
		}

		// What a pane of glass reflects: the sky, or the far side of the street where the reflected ray would meet it.
		float3 Reflection (float3 r, float3 wpos, float pix) {
			float3 sky = DecodeHDR(UNITY_SAMPLE_TEXCUBE_LOD(unity_SpecCube0, r, 0.0), unity_SpecCube0_HDR);
			float horiz = max(length(r.xz), 1e-3);
			float elev = r.y / horiz;                                    // tangent of the ray's elevation
			float roof = (_ScenicCanyon.x - wpos.y) / _ScenicCanyon.y;   // and of the roofline across the street
			float built = saturate((roof - elev) / 0.12 + 0.5);
			// the facade over there: floors and bays in a colour of its own, the street below it
			float hy = wpos.y + elev * _ScenicCanyon.y;
			float hx = dot(wpos.xz, float2(r.z, -r.x) / horiz);
			float3 tone = lerp(float3(0.34, 0.31, 0.28), palette(hash1(floor(float2(hx / 37.0, 3.0)))) * 0.35 + 0.12, 0.25);
			float fade = saturate(1.6 - pix * 6.0);
			float win = band(frac(hy / 3.5), 0.3, 0.8, 0.1) * band(frac(hx / 3.1), 0.2, 0.8, 0.1) * fade;
			float3 other = lerp(tone, float3(0.10, 0.12, 0.15), win * 0.45);
			other = lerp(float3(0.14, 0.14, 0.14), other, saturate(hy / 1.5 + 0.5));
			return lerp(sky, other * _ScenicCanyon.z, built);
		}

		fixed4 frag (v2f i) : SV_Target {
			float3 N = normalize(i.normal);
			float3 T = normalize(i.tangent.xyz - N * dot(N, i.tangent.xyz));
			float3 B = cross(N, T) * i.tangent.w;
			float3 toEye = _WorldSpaceCameraPos - i.wpos;
			float3 V = normalize(toEye);
			float3 L = normalize(_WorldSpaceLightPos0.xyz);
			float2 sy = i.uv0.xy;
			float2 dx = ddx(sy), dy = ddy(sy);
			float pix = max(max(length(dx), length(dy)), 1e-4);          // metres a pixel covers on the wall
			float3 tint = i.color.rgb * i.color.rgb;
			float seed = i.uv2.y;

			// ---------------------------------------------------------------- a plain surface
			if (i.color.a > 0.5) {
				Wall p = SampleWall(round(i.uv2.x), sy, dx, dy);
				float3 pn = normalize(T * p.normal.x + B * p.normal.y + N * p.normal.z);
				float vary = tex2Dgrad(_ScenicNoise, sy / 23.0 + seed * 7.0, dx / 23.0, dy / 23.0).g;
				return fixed4(Lit(p.albedo * tint * (0.9 + 0.2 * vary), pn, L, 1.0), 1.0);
			}

			// ---------------------------------------------------------------- the facade
			float edge = i.face.x, bays = max(i.face.y, 1.0);
			float groundH = i.uv1.x, floorH = i.uv1.y, winTop = i.uv1.z, reveal = i.uv1.w;
			float groundStyle = round(i.uv2.z), kind = round(i.uv2.w);
			float bayW = edge / bays;
			bool blank = i.face.y < 0.5;                                 // a face too short for a bay: wall only
			bool ground = sy.y < groundH;
			bool windows = !blank && sy.y < winTop;

			float bay = clamp(floor(sy.x / bayW), 0.0, bays - 1.0);
			float cx = sy.x - bay * bayW;
			float flr = ground ? -1.0 : floor((sy.y - groundH) / floorH);
			float cy = ground ? sy.y : sy.y - groundH - flr * floorH;
			float cellH = ground ? groundH : floorH;
			float3 cellHash = hash3(float2(bay + seed * 91.7, flr + seed * 37.3));

			// ---- the opening in this cell, in metres from the cell's corner
			float2 lo, hi;
			float frameW = 0.06, mullAcross = 1.0, transoms = 0.0, arch = 0.0;
			bool openX = false;                                          // glazing that runs on into the next bay
			float roomDepth = 5.5, exposure = _InteriorExposure;
			// uv7.w packs the bays to a shop (1-7) and the building's door seed (x 8): whole numbers, so
			// ScenicBuilding.Doors finds the same doors the pixels show.
			float packed = round(i.uv7.w);
			float doorSeed = floor(packed / 8.0);
			float shopBays = max(packed - doorSeed * 8.0, 1.0);
			float shop = floor(bay / shopBays);
			float3 shopHash = hash3(float2(shop + seed * 13.1, seed * 71.9));
			bool door = false;
			float fascia = 0.0;                                          // the sign band over a shopfront
			if (ground) {
				// one door to a shop, or to every eight bays of a lobby or a masonry base, where the door seed puts it
				float group = floor(bay / 8.0);
				float doorBay = groundStyle < 0.5 ? shop * shopBays + fmod(shop * 5.0 + doorSeed, shopBays)
				                                  : group * 8.0 + fmod(group * 3.0 + doorSeed, 8.0);
				door = abs(bay - doorBay) < 0.5;
				if (groundStyle < 0.5) {            // shopfronts
					float pier = 0.22;
					fascia = 0.85;
					lo = float2(pier, door ? 0.06 : 0.5);
					hi = float2(bayW - pier, groundH - fascia - 0.3);
					mullAcross = door ? 2.0 : (bayW > 3.4 ? 2.0 : 1.0);
					transoms = door ? 1.0 : 0.0;
					roomDepth = 9.0;
					exposure = _ShopExposure;
				} else if (groundStyle < 1.5) {     // a lobby: tall glazing between clad piers
					float pier = 0.42;
					lo = float2(pier, door ? 0.06 : 0.35);
					hi = float2(bayW - pier, groundH - 0.75);
					mullAcross = 2.0;
					transoms = 1.0;
					roomDepth = 11.0;
					exposure = _ShopExposure * 0.8;
				} else {                            // a masonry base: small windows, doors
					float w = door ? min(1.7, bayW * 0.6) : bayW * 0.46;
					lo = float2(0.5 * (bayW - w), door ? 0.06 : 1.15);
					hi = float2(0.5 * (bayW + w), groundH - 0.95);
					mullAcross = 2.0;
					transoms = door ? 1.0 : 0.0;
					arch = kind > 2.5 ? 1.0 : 0.0;
				}
			} else {
				float w = i.uv3.x * bayW, h = i.uv3.y * floorH, sill = i.uv3.z * floorH;
				mullAcross = max(fmod(round(i.uv3.w), 10.0), 1.0);
				transoms = floor(round(i.uv3.w) / 10.0);
				lo = float2(0.5 * (bayW - w), sill);
				hi = float2(0.5 * (bayW + w), min(sill + h, floorH - 0.2));
				if (kind > 0.5 && kind < 2.5) {     // a ribbon, or a curtain wall: glazed from pier to pier
					openX = true;
					float pier = kind > 1.5 ? 0.3 : 0.5;
					lo.x = bay < 0.5 ? pier : -100.0;
					hi.x = bay > bays - 1.5 ? bayW - pier : bayW + 100.0;
					frameW = 0.045;
					if (kind > 1.5) { lo.y = 0.0; hi.y = floorH; }
				}
				arch = kind > 2.5 ? 1.0 : 0.0;
			}

			// ---- the wall
			float wallLayer = ground ? round(i.uv4.w) : round(i.uv2.x);
			float3 wallTint = ground ? i.uv7.rgb : tint;
			Wall wall = SampleWall(wallLayer, sy, dx, dy);
			float3 trim = i.uv4.rgb;
			float3 noise = tex2Dgrad(_ScenicNoise, sy / float2(31.0, 47.0) + seed * 3.0, dx / 31.0, dy / 47.0).rgb;
			float streak = tex2Dgrad(_ScenicNoise, float2(sy.x / 9.0, sy.y / 60.0) + seed * 11.0, dx / 9.0, dy / 60.0).a;
			float3 albedo = wall.albedo * wallTint;
			float wear = _Weathering;
			albedo *= 1.0 + wear * (0.16 * (noise.g - 0.5) + 0.10 * (noise.r - 0.5));
			// rain streaks running down from the top of the wall and from under each window, the splash at its foot
			float underWindow = windows && !ground && (kind < 0.5 || kind > 2.5)
				? band(cx, lo.x - 0.1, hi.x + 0.1, 0.3) * saturate(1.0 - (lo.y - cy) / 1.1) * step(cy, lo.y) : 0.0;
			float fromTop = saturate(1.0 - (i.uv1.z + 3.0 - sy.y) / 6.0);
			albedo *= 1.0 - wear * saturate(underWindow * 0.5 + fromTop * 0.35) * smoothstep(0.35, 0.8, streak) * 0.45;
			albedo *= 1.0 - wear * 0.3 * saturate(1.0 - sy.y / 0.9) * (0.5 + noise.b);
			if (ground) {
				// a plinth; on a masonry base, courses of dressed stone
				albedo *= lerp(1.0, 0.62, band(sy.y, -1.0, 0.32, pix));
				if (groundStyle > 1.5) albedo *= 1.0 - 0.3 * stripe(frac(sy.y / 0.52), 0.5, 0.035 / 0.52, pix / 0.52) * saturate(1.5 - pix * 20.0);
				// the sign band: the shop's colour, with a row of marks for its lettering
				if (fascia > 0.0) {
					float fy0 = groundH - fascia - 0.12, fy1 = groundH - 0.14;
					float sign = band(cy, fy0, fy1, pix) * band(cx, 0.1, bayW - 0.1, pix);
					float3 signC = signColour(shopHash.y);
					float sx = sy.x - shop * shopBays * bayW;                       // along the shop
					float letters = band(cy, fy0 + 0.25, fy1 - 0.25, pix)
						* band(sx, 0.2 * shopBays * bayW, 0.8 * shopBays * bayW, pix)
						* step(0.42, hash1(float2(floor(sx / 0.22), shop + seed * 3.0)))
						* saturate(1.3 - pix * 9.0);
					float3 ink = dot(signC, 0.333) > 0.2 ? float3(0.03, 0.03, 0.03) : float3(0.62, 0.60, 0.55);
					albedo = lerp(albedo, lerp(signC, ink, letters * 0.85), sign);
					wall.normal = lerp(wall.normal, float3(0, 0, 1), sign);
				}
			} else if (kind < 0.5 || kind > 2.5) {
				// a stone sill under a window set in a wall
				float sillBand = band(cy, lo.y - 0.09, lo.y, pix) * band(cx, lo.x - 0.07, hi.x + 0.07, pix) * (windows ? 1.0 : 0.0);
				albedo = lerp(albedo, trim, sillBand);
			} else if (kind < 1.5) {
				// under a ribbon the spandrel takes the trim colour
				albedo = lerp(albedo, trim * (0.9 + 0.2 * noise.g), 0.55 * (windows ? 1.0 : 0.0) * step(cy, lo.y));
			}
			float3 wallN = normalize(T * wall.normal.x + B * wall.normal.y + N * wall.normal.z);
			float3 color = Lit(albedo, wallN, L, 1.0);

			// ---- the opening
			if (windows) {
				float3 Vt = float3(dot(V, T), dot(V, B), max(dot(V, N), 0.18));
				float3 Lt = float3(dot(L, T), dot(L, B), dot(L, N));
				float2 c = float2(cx, cy);
				float inWin = band(c.x, lo.x, hi.x, pix) * band(c.y, lo.y, hi.y, pix);
				float archR = 0.5 * (hi.x - lo.x);
				if (arch > 0.5) {
					// a round head: above the springing only the half-disc is open
					float2 a = c - float2(0.5 * (lo.x + hi.x), hi.y - archR);
					inWin *= a.y < 0.0 ? 1.0 : saturate((archR - length(a)) / pix + 0.5);
				}
				if (inWin > 0.001) {
					// where the view ray meets the glass, set back by the reveal
					float2 slope = Vt.xy / Vt.z;
					float2 q = c - slope * reveal;
					float2 qc = clamp(q, lo, hi);
					float2 over = q - qc;
					float3 open;
					bool side = abs(over.x) + abs(over.y) > 1e-5;
					if (side) {
						// the ray met the reveal first: the jamb, the head or the sill
						float dX = abs(over.x) > 1e-5 ? abs((c.x - (over.x < 0.0 ? lo.x : hi.x)) / slope.x) : 1e9;
						float dY = abs(over.y) > 1e-5 ? abs((c.y - (over.y < 0.0 ? lo.y : hi.y)) / slope.y) : 1e9;
						float3 n = dX < dY ? T * (over.x < 0.0 ? 1.0 : -1.0) : B * (over.y < 0.0 ? 1.0 : -1.0);
						bool isSill = dX >= dY && over.y < 0.0;
						float3 sideAlbedo = isSill ? trim : albedo * 0.92;
						float deep = saturate(min(dX, dY) / max(reveal, 1e-3));
						open = Lit(sideAlbedo, n, L, 1.0) * (1.0 - 0.3 * deep);
					} else {
						// the sun's shadow of the reveal on the frame and the glass
						float2 e = q + Lt.xy / max(Lt.z, 1e-3) * reveal;
						float lit = Lt.z > 0.0 ? band(e.x, lo.x, hi.x, 0.03) * band(e.y, lo.y, hi.y, 0.03) : 0.0;

						// frame, mullions and transoms
						float fw = frameW;
						float wq = max(pix, 0.004);
						float2 span = hi - lo;
						float frame = 1.0 - band(q.x, lo.x + fw, hi.x - fw, wq) * band(q.y, lo.y + fw, hi.y - fw, wq);
						float paneX, panes = mullAcross;
						if (openX) {
							// mullions at every bay line and between them
							float m = bayW / mullAcross;
							paneX = floor(q.x / m);
							frame = max(frame, stripe(frac(q.x / m), 0.0, 0.5 * fw / m, wq / m) + stripe(frac(q.x / m), 1.0, 0.5 * fw / m, wq / m));
						} else {
							float m = span.x / mullAcross;
							paneX = floor((q.x - lo.x) / m);
							for (int k = 1; k < 4; k++) {
								if (k < mullAcross) frame = max(frame, stripe(q.x, lo.x + k * m, 0.4 * fw, wq));
							}
						}
						float paneY = 0.0;
						if (transoms > 0.5) {
							float ty = door ? lo.y + 2.25 : lo.y + span.y * (transoms > 1.5 ? 0.5 : 0.72);
							frame = max(frame, stripe(q.y, ty, 0.4 * fw, wq));
							if (transoms > 1.5) frame = max(frame, stripe(q.y, lo.y + span.y * 0.25, 0.4 * fw, wq) + stripe(q.y, lo.y + span.y * 0.75, 0.4 * fw, wq));
							paneY = step(ty, q.y);
						}
						frame = saturate(frame);
						float detail = saturate((bayW / pix - _DetailPixels * 0.5) / _DetailPixels);
						frame *= detail;

						// ---- the room behind the glass
						float3 d = -float3(Vt.xy, Vt.z) / length(float3(Vt.xy, Vt.z));
						float2 room = float2(openX ? q.x - floor(q.x / bayW) * bayW : q.x, q.y);
						float3 inv = 1.0 / float3(abs(d.x) < 1e-4 ? 1e-4 : d.x, abs(d.y) < 1e-4 ? 1e-4 : d.y, d.z);
						float tx = ((d.x > 0.0 ? bayW : 0.0) - room.x) * inv.x;
						float ty2 = ((d.y > 0.0 ? cellH : 0.0) - room.y) * inv.y;
						float tz = -roomDepth * inv.z;
						float t = min(min(tx, ty2), tz);
						float3 h = float3(room, 0.0) + d * t;
						float depth = saturate(-h.z / roomDepth);
						float3 roomHash = ground ? shopHash : cellHash;
						float3 wallC = lerp(float3(0.70, 0.67, 0.61), palette(roomHash.x) * 0.5 + 0.3, 0.12);
						float3 inside;
						bool litRoom = roomHash.z < (ground ? 0.45 : (kind > 1.5 ? 0.08 : 0.14));
						if (t == tz) {
							// the back wall: a dado of furniture, something hung on it
							float2 uv = h.xy / float2(bayW, cellH);
							inside = wallC * 0.85;
							float shelf = ground && groundStyle < 0.5
								? band(frac(uv.y * 4.0), 0.15, 0.9, 0.05) * step(0.3, hash1(floor(float2(uv.x * 7.0, uv.y * 4.0)) + shop))
								: band(uv.y, 0.0, 0.3, 0.02) * step(0.35, hash1(float2(floor(uv.x * 3.0), bay + flr * 7.0)));
							float3 goods = palette(hash1(floor(float2(uv.x * 7.0, uv.y * 4.0)) + roomHash.xy * 9.0)) * 0.45 + 0.2;
							inside = lerp(inside, ground && groundStyle < 0.5 ? goods : wallC * 0.4, shelf * detail);
						} else if (t == tx) {
							inside = wallC * 0.7;
						} else if (d.y > 0.0) {
							// the ceiling, with its light panels
							float2 uv = float2(h.x / bayW, -h.z / roomDepth);
							float panel = band(frac(uv.x * 2.0), 0.3, 0.7, 0.05) * band(frac(uv.y * 3.0), 0.2, 0.8, 0.05);
							inside = float3(0.80, 0.80, 0.78) + (litRoom ? panel * float3(3.0, 2.9, 2.6) * detail : 0.0);
						} else {
							inside = lerp(float3(0.24, 0.17, 0.11), float3(0.27, 0.28, 0.30), roomHash.y);
						}
						// daylight falls off into the room; a lit room keeps some light at the back
						float day = lerp(1.0, 0.22, depth);
						inside *= (day + (litRoom ? 0.4 : 0.0)) * exposure;
						if (litRoom) inside *= float3(1.08, 1.0, 0.88);
						inside = lerp(dot(wallC, 0.333) * 0.45 * exposure, inside, detail);

						// blinds or curtains, drawn to a different height in every window
						float blinds = round(i.uv6.w);
						float3 winHash = hash3(float2(bay * 3.7 + flr * 17.3 + seed * 29.0, flr - bay * 1.3));
						if (blinds > 0.5 && !ground) {
							float v = (q.y - lo.y) / max(span.y, 1e-3);
							float level = winHash.x < 0.45 ? 0.0 : (winHash.x < 0.88 ? 0.12 + 0.5 * winHash.y : 1.0);
							float3 cloth = lerp(float3(0.70, 0.68, 0.62), float3(0.50, 0.52, 0.54), step(0.6, winHash.z)) * (0.3 + 0.3 * saturate(Lt.z) * lit);
							float cover;
							if (blinds < 1.5) {
								cover = step(1.0 - level, v) * step(0.001, level);
								cloth *= 0.9 + 0.1 * sin(v * span.y * 90.0) * detail;
							} else {
								// curtains drawn in from the sides
								float u = abs((q.x - lo.x) / max(span.x, 1e-3) - 0.5) * 2.0;
								cover = step(1.0 - 0.75 * level, u);
								cloth *= 0.88 + 0.12 * sin(u * 40.0) * detail;
							}
							inside = lerp(inside, cloth, cover * lerp(0.6, 0.92, detail));
						}

						// ---- the glass over it
						float3 paneHash = hash3(float2(bay * 7.3 + paneX * 1.7 + seed * 53.0, flr * 3.1 + paneY * 5.9));
						float2 tilt = (paneHash.xy - 0.5) * 0.03;
						float3 gn = normalize(N + T * tilt.x + B * tilt.y);
						float3 r = reflect(-V, gn);
						float3 refl = Reflection(r, i.wpos, pix) * _ReflectionBoost;
						float f0 = i.uv5.a;
						// a curtain wall's spandrel is backed glass: nothing behind it, a little more mirror
						bool spandrel = !ground && kind > 1.5 && q.y < 0.95;
						if (spandrel) { inside = i.uv5.rgb * 0.12; f0 = min(f0 + 0.08, 0.9); }
						if (door) inside *= 0.8;
						float fres = f0 + (1.0 - f0) * pow(1.0 - saturate(dot(V, gn)), 5.0);
						float3 glass = inside * i.uv5.rgb * (1.0 - fres) + refl * fres * lerp(float3(1, 1, 1), i.uv5.rgb, 0.35);
						float glint = pow(saturate(dot(r, L)), 900.0) * 6.0 * fres * step(0.0, Lt.z);
						glass += _LightColor0.rgb * glint;

						float3 frameC = Lit(i.uv6.rgb, N, L, lit);
						open = lerp(glass, frameC, frame);
					}
					color = lerp(color, open, inWin);
				}
			}
			return fixed4(color, 1.0);
		}
		ENDCG
	}
}
Fallback "VertexLit"
}
