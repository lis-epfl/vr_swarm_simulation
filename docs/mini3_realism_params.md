# DJI Mini 3 — Realistic Simulation Parameters

Source: DJI Mini 3 official specs (provided).
Sim context: `ScaleFactor = 10`, `d_ref = 7` → equilibrium inter-drone spacing ≈ **70 m world-space**.

---

## 1. State Estimation Noise (`StateFinder.cs`)

DJI does not publish raw IMU noise figures. Values below are derived from published **hover accuracy**,
which represents the post-filter residual the flight controller achieves — i.e., the noise the swarm
algorithm effectively "sees".

> **Note:** The Mini 3 has only a **downward** vision system (no forward/backward/lateral cameras).
> Vision positioning is only reliable in hover. In forward flight the drone falls back to GNSS,
> which has significantly worse horizontal accuracy (±1.5 m) than the Mavic 3 (±0.5 m).

| Parameter | Code Field | Current Value | Vision Mode (tight) | GNSS Mode (realistic) | Derivation |
|---|---|---|---|---|---|
| Horizontal position noise σ | `positionNoiseSigma` | 0.03 m | **0.10 m** | **0.50 m** | Vision horizontal: ±0.3 m → σ ≈ 0.15 m. GNSS horizontal: ±1.5 m → σ ≈ 0.50 m (much worse than Mavic 3's ±0.5 m). |
| Altitude noise σ | `altitudeNoiseSigma` | 0.015 m | **0.05 m** | **0.20 m** | Vision vertical: ±0.1 m → σ ≈ 0.05 m. GNSS vertical: ±0.5 m → σ ≈ 0.20 m. |
| Attitude noise σ (pitch/roll) | `attitudeNoiseSigma` | 0.004 rad | **0.002 rad (0.11°)** | **0.005 rad (0.29°)** | Gimbal vibration range ±0.01° (slightly larger than Mavic 3's ±0.007°). DJI Mini-class IMU attitude accuracy ~0.15–0.3° based on vibration range. 0.002–0.005 rad is appropriate. |
| Velocity noise σ | `velocityNoiseSigma` | 0.06 m/s | **0.05 m/s** | **0.15 m/s** | Vision optical flow: ~±0.1 m/s. GNSS Doppler on a light drone: ~±0.2–0.3 m/s (higher than Mavic 3 due to lighter body, more susceptible to vibration). |

### Notes on ScaleFactor and Swarm Stability

With `ScaleFactor = 10` and `d_ref = 7`, the equilibrium distance is 70 m. Position noise is
divided by `ScaleFactor` inside `OlfatiSaber.cs` before entering the potential function:

| Position noise σ (world m) | Normalized (÷10) | As % of d_ref = 7 | Effect on O-S |
|---|---|---|---|
| 0.03 m (current) | 0.003 | 0.04% | Invisible to algorithm |
| 0.10 m (Vision) | 0.010 | 0.14% | Negligible — swarm stable |
| 0.50 m (GNSS horizontal) | 0.050 | 0.71% | Negligible — small formation jitter |
| 1.50 m (GNSS 2σ bound) | 0.150 | 2.14% | Still well within stability margin |
| 3.50 m (upper bound) | 0.350 | 5.0% | Approaching instability threshold |

All values up to ~2 m are safe. **Recommended: use the GNSS column** for visible, realistic jitter in VR.

---

## 2. Flight Control Limits (`VelocityControl.cs`)

> The Mini 3 does **not** publish separate per-axis angular rate limits. The max pitch angle of 40°
> is the highest of any DJI Mini-class drone. Max angular velocity is estimated at ~150°/s based
> on DJI Mini 2/Air 2S class performance (not officially published for Mini 3).

| Parameter | Code Field | Current Value | DJI Mini 3 Real | Recommended for Sim |
|---|---|---|---|---|
| Max pitch angle | `maxPitch` | 0.175 rad (10°) | **0.698 rad (40°)** | **0.436 rad (25°)** — good balance for swarm flight ¹ |
| Max roll angle | `maxRoll` | 0.175 rad (10°) | **0.698 rad (40°)** | **0.436 rad (25°)** |
| Max yaw rate | `maxYawRate` | 1.0 rad/s (57°/s) | **~2.62 rad/s (150°/s)** ² | **1.57 rad/s (90°/s)** |
| Max angular accel | `maxAlpha` | 10.0 rad/s² | ~15–20 rad/s² ³ | **15.0 rad/s²** |
| Max speed | `maxSpeed` | 5.0 m/s | **16 m/s** (Sport mode) | **10.0 m/s** (swarm-safe) |
| Max ascent speed | (via height PID) | — | **5 m/s** | **4.0 m/s** |
| Max descent speed | (via height PID) | — | **3.5 m/s** | **3.0 m/s** |

¹ 25° allows the drone to respond aggressively to wind and swarm forces without looking unnatural.
  The real 40° limit is rarely reached in practice; 25° is a good operational ceiling.

² Not officially published. Estimated from DJI Mini-class hardware (brushless motors, ~250 g body).
  DJI Mini 2 community measurements suggest ~100–150°/s is achievable for yaw.

³ Not published. Estimated: if max angular velocity (150°/s = 2.62 rad/s) is reached in ~0.15–0.2 s,
  angular acceleration ≈ 13–17 rad/s².

---

## 3. Physical Parameters (`SwarmSpawn.cs`)

| Parameter | Current in Sim | DJI Mini 3 Real | Notes |
|---|---|---|---|
| Mass (Rigidbody) | Set by prefab | **0.248 kg** | Verify prefab Rigidbody mass matches this if using SI units. Much lighter than Mavic 3 (0.895 kg) — makes wind effects more pronounced per unit force. |
| Mass variation range | ±8% | ±2–3% (manufacturing tolerance) | ±8% is exaggerated but gives good visual variety. ±3% is physically accurate. |
| Inertia variation range | ±5% | ±1–2% | ±5% is fine for visual variety. |

---

## 4. Wind Resistance Reference (`RandomPulseNoise.cs`)

| Parameter | Code Field | Current | DJI Mini 3 Real | Notes |
|---|---|---|---|---|
| Max sustained wind | — | — | **10.7 m/s (Level 5)** | Lower than Mavic 3's 12 m/s. The Mini 3 is lighter and more susceptible. |
| Wind force calibration | `strength_coef` | 0.0015 | — | With drone mass ≈ 0.248 kg, aerodynamic drag at 10.7 m/s ≈ ½ × 1.2 × 0.5 × 0.02 × 114 ≈ **0.68 N**. At 50 Hz FixedUpdate: force per frame = 0.68 / 50 = 0.0136 N·s. So `strength × strength_coef = 0.0136` → at `strength_mean = 30`, `strength_coef ≈ 0.00045`. The current 0.0015 produces gusts roughly 3× stronger than the Mini 3's wind resistance limit — quite dramatic but visually impactful. Reduce to **0.0005** for physically accurate wind, or keep 0.0015 for dramatic effect. |

---

## 5. Summary: Recommended Parameter Set

```csharp
// StateFinder.cs — GNSS-grade (visible, realistic for Mini 3 outdoor flight)
enableStateNoise       = true
positionNoiseSigma     = 0.50f    // was 0.03 — GNSS horizontal hover accuracy (±1.5 m / 2σ)
altitudeNoiseSigma     = 0.20f    // was 0.015 — GNSS vertical hover accuracy (±0.5 m / 2σ)
attitudeNoiseSigma     = 0.004f   // unchanged — matches DJI Mini-class INS accuracy
velocityNoiseSigma     = 0.15f    // was 0.06 — GNSS Doppler on light airframe

// VelocityControl.cs
maxPitch               = 0.436f   // was 0.175 — 25°, safe swarm limit (real drone: 40°)
maxRoll                = 0.436f   // was 0.175 — same
maxYawRate             = 1.57f    // was 1.0 — ~90°/s (estimated real: ~150°/s)
maxAlpha               = 15.0f    // was 10.0 — closer to real motor response
maxSpeed               = 10.0f    // was 5.0 — real: 16 m/s; 10 m/s for swarm safety

// RandomPulseNoise.cs — physically accurate wind
strength_coef          = 0.0005f  // was 0.0015 — matches Mini 3's ~0.68 N max wind force
// Or keep 0.0015 for more dramatic, cinematic wind gusts
```

---

## Key Differences vs. Mavic 3

| Property | DJI Mini 3 | DJI Mavic 3 | Impact on Simulation |
|---|---|---|---|
| Weight | 248 g | 895 g | Wind forces ~3.6× more impactful per unit force on Mini 3 |
| GNSS horizontal accuracy | ±1.5 m | ±0.5 m | Mini 3 position noise is 3× larger |
| Max pitch angle | 40° | 35° | Mini 3 slightly more agile |
| Max speed | 16 m/s | 21 m/s | Mini 3 is slower |
| Max ascent | 5 m/s | 8 m/s | Mini 3 climbs more slowly |
| Vision system | Downward only | Omnidirectional | Mini 3 GNSS-only in forward flight |
| Angular vibration | ±0.01° | ±0.007° | Mini 3 has slightly more body vibration |
