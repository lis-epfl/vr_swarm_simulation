# Velocity Control & Swarm Integration Refactor

This document explains every change made to the drone velocity control system and swarm algorithm integration, the reasoning behind each decision, and a reference for all tunable parameters.

---

## Table of Contents

1. [Architecture Overview: Before vs After](#architecture-overview-before-vs-after)
2. [Change 1: Height Control Simplified to PD](#change-1-height-control-simplified-to-pd)
3. [Change 2: World-Frame Velocity Mixing](#change-2-world-frame-velocity-mixing)
4. [Change 3: Swarm Output as Feedforward Acceleration](#change-3-swarm-output-as-feedforward-acceleration)
5. [Change 4: Neighbor Velocity Consensus](#change-4-neighbor-velocity-consensus)
6. [Change 5: SwarmAlgorithm Interface Update](#change-5-swarmalgorithm-interface-update)
7. [Why the Height Derivative Filter is Kept](#why-the-height-derivative-filter-is-kept)
8. [Complete Parameter Reference](#complete-parameter-reference)

---

## Architecture Overview: Before vs After

### Before

```
User Input (body frame) ──┐
                           ├─► Sum as VELOCITY TARGET ──► Velocity P-controller ──► desired accel ──► desired angle
Swarm Algorithm Output ────┘
  (acceleration-like)           ↑ second P-controller
                                  on same error = doubled gain
```

The swarm algorithm (Olfati-Saber) outputs quantities like `c_vm * (desiredVel - currentVel)` and cohesion forces. These are **accelerations** (force-like). But VelocityControl treated them as **velocity targets** and ran another proportional controller (`error / timeConstantAcceleration`) on top. Two cascaded proportional controllers on the same error doubled the effective loop gain, causing guaranteed overshoot and oscillation, especially at high pitch/roll angles.

Additionally, the velocity mixing happened in body frame. When the drone was pitched 30 degrees forward, `InverseTransformDirection` rotated horizontal swarm corrections into a tilted body frame, creating cross-coupling between horizontal and vertical axes.

### After

```
User Input (body frame → world frame) ──► Velocity P-controller ──► user accel ──┐
                                                                                   ├─► Sum as ACCELERATION ──► desired angle
Swarm Algorithm Output (acceleration) ──► Light filter ────────────────────────────┘
                                           (no P-controller)
```

The two control paths are now independent:
- **User velocity controller**: a single proportional loop that tracks the pilot's commanded velocity.
- **Swarm feedforward**: the swarm's acceleration output is added directly to the desired acceleration, bypassing the velocity controller entirely.

This eliminates the gain doubling and the body-frame cross-coupling.

---

## Change 1: Height Control Simplified to PD

**Files changed:** `VelocityControl.cs`

### What was removed

| Parameter | Purpose | Why removed |
|-----------|---------|-------------|
| `HeightKi` | Integral gain for height error | In a rigid-body simulation with gravity compensation already applied at startup (`rb.AddForce(gravity * mass)`), there is no persistent steady-state offset. The integral term was accumulating error and causing altitude overshoot after sustained altitude rate commands. |
| `HeightRateCoefficient` | Low-pass filter on the user's altitude rate input | The altitude rate input comes from a joystick — it's already a clean signal. Filtering it added unnecessary lag between stick input and height response. |
| `cumulativeHeightError` | Accumulated integral of height error | No longer needed without Ki. |
| `filteredHeightRate` | Filtered version of userAltitudeRate | No longer needed without HeightRateCoefficient. |

### What remains

```csharp
// Direct altitude rate integration — no filter on user input
desired_height += userAltitudeRate * Time.deltaTime;
desired_height = Mathf.Max(desired_height, MinHeight);

// PD controller on height error
float currentHeightError = desired_height - State.Altitude;
float rawDerivative = (currentHeightError - previousHeightError) / Time.deltaTime;
filteredHeightErrorDerivative = filteredHeightErrorDerivative * (1 - heightDerivFilterCoeff)
                              + rawDerivative * heightDerivFilterCoeff;
float altitudeCommand = HeightKp * currentHeightError + HeightKd * filteredHeightErrorDerivative;
```

The altitude rate from the joystick is now applied directly to `desired_height` each frame — no smoothing delay. The PD controller then tracks this setpoint with `HeightKp` (proportional) and `HeightKd` (derivative).

---

## Change 2: World-Frame Velocity Mixing

**Files changed:** `VelocityControl.cs`

### The problem

Previously, the swarm's world-frame velocity was transformed into body frame before being combined with the user's body-frame command:

```csharp
// OLD: body-frame mixing
Vector3 totalTargetVelocity = transform.InverseTransformDirection(swarmVelocity);
totalTargetVelocity += new Vector3(userVelX, 0.0f, userVelZ);
```

When the drone is pitched forward at a large angle, `InverseTransformDirection` rotates a horizontal swarm correction so that it partially projects onto the drone's vertical body axis. This creates **cross-coupling**: a lateral formation correction produces an unintended altitude change, and vice versa. At large pitch/roll angles (which you need for fast flight), this cross-coupling becomes severe.

### The fix

Everything now happens in world frame:

```csharp
// NEW: world-frame mixing
Vector3 userWorldVel = transform.TransformDirection(new Vector3(userVelX, 0f, userVelZ));
Vector3 worldVelocity = GetComponent<Rigidbody>().velocity;
Vector3 userVelError = worldVelocity - userWorldVel;
Vector3 userAccel = userVelError * -1.0f / timeConstantAcceleration;
```

The user's body-frame command is converted to world frame using `TransformDirection`. The velocity error is computed in world frame against the rigidbody's world velocity. The resulting desired acceleration maps cleanly to pitch/roll angles regardless of current orientation:

```csharp
desiredTheta = new Vector3(desiredAcceleration.z / gravity, 0.0f, -desiredAcceleration.x / gravity);
```

This relationship (horizontal acceleration = g * tan(angle) ≈ g * angle for small angles) is only valid in world frame. Applying it in body frame was physically incorrect.

---

## Change 3: Swarm Output as Feedforward Acceleration

**Files changed:** `VelocityControl.cs`, `SwarmAlgorithm.cs`

### The core problem

Olfati-Saber computes terms like:
- **Cohesion**: a force proportional to inter-drone distance error — this is an acceleration
- **Velocity matching**: `c_vm * (v_desired - v_current)` — this is also an acceleration (proportional to velocity error)
- **Obstacle avoidance**: force-like repulsion — acceleration

All of these are **acceleration-level** quantities. But VelocityControl was treating the sum as a **velocity target** and computing yet another proportional correction:

```csharp
// OLD: swarm accel treated as velocity target
velocityError = currentVelocity - swarmOutput;  // swarmOutput is already an accel!
desiredAccel = velocityError * -1 / timeConstantAcceleration;  // second P-controller
```

This creates a cascaded P-P system. The effective gain is `swarmGain / timeConstantAcceleration`, which amplifies the swarm corrections beyond what the algorithm intended. The result: overshoot on formation corrections, followed by an equally aggressive correction in the opposite direction — oscillation.

### The fix

The swarm output is now added directly as a **feedforward acceleration**, bypassing the velocity controller:

```csharp
// NEW: separate paths
Vector3 userAccel = userVelError * -1.0f / timeConstantAcceleration;  // velocity loop
filteredSwarmAccel = Vector3.Lerp(filteredSwarmAccel, swarmAcceleration, SwarmAccelFilterCoefficient);  // feedforward

Vector3 desiredAcceleration = userAccel + filteredSwarmAccel;  // sum at acceleration level
```

The velocity controller only processes the user's velocity command. The swarm acceleration is filtered lightly (to smooth inter-frame jitter from discrete neighbor calculations) and then added directly. No gain amplification, no extra dynamics.

### Why this eliminates oscillation

Consider a drone that's 2m too far right of its formation position:

**Before:**
1. Cohesion outputs a large leftward "velocity" (really acceleration)
2. Velocity controller sees a large velocity error, commands aggressive left roll
3. Drone moves left, overshoots formation position
4. Cohesion reverses, velocity controller commands aggressive right roll
5. Oscillation

**After:**
1. Cohesion outputs a leftward acceleration proportional to 2m error
2. This directly becomes a left roll angle (no amplification)
3. Drone accelerates left; as it approaches the target, the cohesion force naturally decreases (the potential field diminishes)
4. Smooth deceleration, no overshoot

The cohesion potential field is already designed to produce stable convergence. Putting another proportional controller on top destroyed that stability.

---

## Change 4: Neighbor Velocity Consensus

**Files changed:** `OlfatiSaber.cs`

### Before: global reference tracking

```csharp
// OLD: single velocity matching term against a global reference
velocityMatching = c_vm * (desiredVelocity - velocity);
```

This was originally `desiredVelocity = Vector3.zero` (pure drag), then we changed it to the user's commanded velocity. Either way, it's a **global reference tracking** term — every drone independently tries to match a single target velocity.

Problem: when combined with the velocity controller in VelocityControl (which also tracks the user velocity), this created redundant and conflicting control loops.

### After: neighbor velocity consensus

```csharp
// NEW: accumulate velocity differences with each neighbour
foreach (neighbour in swarm)
{
    Vector3 neighbourVelocity = neighbourChild.transform.TransformDirection(neighbourState.VelocityVector);
    velocityConsensus += c_vm * (neighbourVelocity - velocity);
}
```

This is the standard Olfati-Saber velocity consensus protocol. Each drone compares its velocity to each neighbor's velocity and produces a correction proportional to the difference. Key properties:

- **When all drones move at the same velocity** (e.g., all tracking the user command): consensus force is zero. It doesn't fight the user input.
- **When one drone lags behind** (e.g., recovering from a disturbance): consensus pulls it toward the swarm's average velocity, helping it catch up.
- **When drones diverge** (e.g., after obstacle avoidance): consensus brings them back to velocity agreement.

This is the correct role for velocity matching in a centralized-control experiment: it handles **inter-drone synchronization**, while VelocityControl handles **user command tracking**.

### Method signature change

```csharp
// OLD
public Vector3 GetSwarmVelocityCommand(List<GameObject> swarm, Vector3 desiredVelocity)

// NEW
public Vector3 GetSwarmAcceleration(List<GameObject> swarm)
```

The `desiredVelocity` parameter is removed — it's no longer needed since velocity matching is now peer-to-peer. The method name also changes from `GetSwarmVelocityCommand` to `GetSwarmAcceleration` to correctly reflect that the output is an acceleration, not a velocity.

---

## Change 5: SwarmAlgorithm Interface Update

**Files changed:** `SwarmAlgorithm.cs`

The intermediary `SwarmAlgorithm.FixedUpdate` was updated to match the new interfaces:

```csharp
// OLD
Vector3 userWorldVel = transform.TransformDirection(...);
velocityCommand = olfatiSaberAlgorithm.GetSwarmVelocityCommand(swarm, userWorldVel);
velocityControl.swarm_vx = velocityCommand.x;
velocityControl.swarm_vy = velocityCommand.y;
velocityControl.swarm_vz = velocityCommand.z;

// NEW
swarmAccel = olfatiSaberAlgorithm.GetSwarmAcceleration(swarm);
velocityControl.swarmAcceleration = swarmAccel;
```

The three separate float fields (`swarm_vx/vy/vz`) are replaced by a single `Vector3 swarmAcceleration` field, which is cleaner and correctly named.

---

## Why the Height Derivative Filter is Kept

The `heightDerivFilterCoeff` low-pass filter on the height error derivative is the one filter that **must** stay. Here's why:

### The derivative amplifies noise

The height error derivative is computed as:

```csharp
float rawDerivative = (currentHeightError - previousHeightError) / Time.deltaTime;
```

This is a finite-difference approximation of d(error)/dt. Differentiation amplifies high-frequency content: if the altitude measurement has noise of amplitude `A` at frequency `f`, the derivative has noise of amplitude `A * 2πf`. Your `StateFinder` injects Gaussian noise with `altitudeNoiseSigma = 0.015m`. At the physics timestep (typically 50Hz), the raw derivative noise is:

```
noise_derivative ≈ 0.015 * 2 * 50 = 1.5 m/s  (worst case)
```

Multiplied by `HeightKd = 1.0`, this produces up to 1.5 m/s^2 of random thrust variation — enough to cause visible jitter.

### The filter removes this noise

The exponential moving average filter:

```csharp
filtered = filtered * (1 - coeff) + raw * coeff;
```

With `heightDerivFilterCoeff = 0.2`, this has a cutoff frequency of approximately:

```
f_cutoff = coeff / (2π * dt) ≈ 0.2 / (2π * 0.02) ≈ 1.6 Hz
```

This passes the actual altitude dynamics (which are slow — a drone doesn't change altitude at more than a few Hz) while rejecting the high-frequency sensor noise. Without this filter, the derivative term would inject random thrust oscillations.

### Why other removed filters were different

- **HeightRateCoefficient** filtered the joystick input, which is already clean (no sensor noise). Filtering it only added lag.
- **VelocityFilterCoefficient** filtered the combined velocity target, which added lag to both user commands and swarm corrections. The swarm filter (`SwarmAccelFilterCoefficient`) now only filters the swarm acceleration, leaving user commands unfiltered for crisp response.

---

## Complete Parameter Reference

### VelocityControl.cs — Rates & Limits

Defaults in these tables are the C# initialisers, which match `DroneReduced.prefab`'s serialized
values field for field.

| Parameter | Default | Unit | Description |
|-----------|---------|------|-------------|
| `maxPitch` | 0.436332 | rad | Maximum pitch angle (25°). Limits how aggressively the drone can accelerate forward/backward: max horizontal accel ≈ g·tan(25°) ≈ 4.6 m/s². The tilt map is small-angle, so at this limit the delivered acceleration runs roughly 11% above the command. |
| `maxRoll` | 0.436332 | rad | Maximum roll angle. Same as maxPitch but for lateral acceleration. |
| `maxYawRate` | 1.309 | rad/s | Maximum yaw rotation rate: 75 deg/s, the DJI Mini 3 Pro airframe's limit. The real fleet's PC additionally clamps its commands at 40 deg/s (`MAX_YAW_RATE_DEG_S`), which the sim does not reproduce. Also scales the pilot's yaw stick (`AttitudeAlgorithm.readInputs`). |
| `maxAlpha` | 3.38 | rad/s^2 | Maximum angular acceleration, clamped about the world vertical / tilt split rather than about the body axes. **Rescaled by 0.3893 from the 8.68 the prefab used to hold**: the torque is now applied with `ForceMode.Force`, so this is the limit actually achieved, where before the inertia pre-multiply silently delivered 3.38 on pitch/roll and 6.67 on yaw. Same authority, honestly stated, and isotropic. On yaw it is the **net** acceleration: the Rigidbody's `angularDrag` is compensated on top of it, without which the airframe could hold only 34.9 deg/s. |
| `maxSpeed` | 9.31 | m/s | Scales the normalised joystick input to a velocity command. Full stick = maxSpeed. |
| `maxAltitudeRate` | 3.34 | m/s | Maximum rate at which the throttle stick moves `desired_height`, up or down. |
| `MinHeight` | 0.5 | m | Floor clamp for desired_height to prevent ground collision. |

### VelocityControl.cs — Filters & Coefficients

| Parameter | Default | Unit | Description |
|-----------|---------|------|-------------|
| `HeightKp` | 5.0 | 1/s | Proportional gain for altitude error. Higher = more aggressive altitude correction. A value of 5.0 means 1 m of error produces 5 m/s² of thrust correction. |
| `HeightKd` | 2.0 | — | Derivative gain for altitude error rate. Damps vertical oscillation. Higher = more damping but can amplify noise if filter is too weak. |
| `heightDerivFilterCoeff` | 0.5 | — | Exponential filter coefficient for the height error derivative. Range [0, 1]. Lower = heavier filtering (more lag, less noise). Higher = less filtering (less lag, more noise). 0.5 is a time constant of ~29 ms (cutoff ≈ 5.5 Hz) at the 50 Hz physics rate. **Do not remove** — derivative amplifies sensor noise. |
| `yawFilterCoefficient` | 1.0 (off) | — | Exponential prefilter on the **outer** yaw rate command (stick + `attitude_control_yaw`) only — the heading-hold P term is added after it and is deliberately unlagged. 1 = off, which is the fleet-faithful setting: `joystick_controller` smooths pitch, roll and gimbal pitch but feeds the yaw stick in raw, because the heading hold is what makes smoothing unnecessary. |
| `headingHoldKp` | 5.0 | 1/s | The DJI flight controller's heading-hold gain, which the sim previously had no counterpart for. The yaw rate loop has unity DC gain now that `angularDrag` is compensated, so the closed-loop time constant ≈ `1/headingHoldKp`; 5 reproduces the response the old 8 gave against a loop delivering 0.58 of its command (0.38 s rise, 0.5% overshoot). Useful range 4–6. The setpoint is back-calculated against the `maxYawRate` clamp (it advances only by the rate the clamp let through, never reversing), which is what allows a brisk outer gain without overshoot. |
| `maxHeadingHoldErrorDeg` | 25.0 | deg | Anti-windup on the integrated heading setpoint, the analogue of the fleet's `MAX_TARGET_LEAD_DEG` one level below `SwarmPlaneController.maxTargetLeadDeg`. Now the hard bound behind the back-calculation, which holds a saturated turn's lead near `maxYawRate / headingHoldKp` (15°). Unconditional here because `headingHoldKp × 25°` (2.2 rad/s) is above `maxYawRate` (1.309), so the rate clamp binds first and this costs no authority. |
| `SwarmAccelFilterCoefficient` | 0.3 | — | Exponential filter on the swarm feedforward acceleration. Smooths inter-frame jitter from discrete neighbor calculations. Only affects swarm corrections — user input is unfiltered. Range [0, 1]: lower = smoother formation adjustments but slower reaction; higher = more responsive but may transmit neighbor calculation noise. |

### VelocityControl.cs — Internal Time Constants (not exposed in Inspector)

| Parameter | Value | Unit | Description |
|-----------|-------|------|-------------|
| `timeConstantAcceleration` | 0.75 | s | Time constant for the user velocity controller. Defines how quickly the drone converges to the user's commanded velocity. Lower = snappier stick response. Unsaturated, the drone reaches 63% of commanded velocity in 0.75 s. Only affects user input — swarm feedforward bypasses this. |
| `timeConstantOmegaXYRate` | 0.1 | s | Time constant for the attitude (pitch/roll) rate controller. Determines how fast the drone tracks desired pitch/roll angles. 0.1 s means the attitude loop is 7.5x faster than the velocity loop, which is correct (inner loop must be faster than outer loop). |
| `timeConstantAlphaRate` | 0.1284 | s | Time constant for the angular acceleration controller, **one value for all three body axes**. Replaces the separate `timeConstantAlphaXYRate` / `timeConstantAlphaZRate` (both 0.05): the world-vertical yaw command is projected into the body frame via `upBody`, so it lands on all three axes, and that projection is exact only while the three share a gain — unequal constants rotate the command off `upBody` and leak heading on every tilt change. The value preserves the old pitch/roll response through the `ForceMode.Force` fix (0.05 / 0.3893). Innermost loop — must stay fastest. |

### OlfatiSaber.cs — Swarm Parameters

| Parameter | Default | Unit | Description |
|-----------|---------|------|-------------|
| `c_vm` | 1.0 | — | Velocity consensus gain. Controls how strongly each drone matches its neighbors' velocities. With the new neighbor consensus (not global reference), this only activates when drones have different velocities. Higher = faster velocity synchronization but can cause coupling oscillations between neighbors. **Recommended range: 0.3–1.5** |
| `d_ref` | 7.0 | scaled | Desired inter-drone distance (in scaled units). The cohesion force is zero at this distance, attractive beyond it, repulsive below it. |
| `r0_coh` | 150.0 | scaled | Interaction radius for cohesion. Drones farther than r0_coh apart don't interact. |
| `ScaleFactor` | 10.0 | — | Converts Unity world units to the algorithm's internal distance scale. All distances are divided by this before entering the cohesion/weight functions. |
| `a` | 0.9 | — | Cohesion intensity parameter. Controls the shape of the attraction-repulsion potential. |
| `b` | 1.5 | — | Cohesion intensity parameter. Together with `a`, determines the steepness of the potential well around d_ref. |
| `c` | — | — | Cohesion intensity smoothing parameter. Affects the shape of the potential near d_ref. |
| `delta` | 0.1 | — | Neighbor weight inner cutoff (as fraction of r0_coh). Below delta * r0_coh, neighbor weight is 1.0 (full influence). |
| `c_obs` | 4.3 | — | Obstacle avoidance cohesion gain. Higher = stronger repulsion from obstacles. |
| `d_obs` | 5.0 | scaled | Obstacle detection radius (in scaled units). |
| `c_altitude_2d` | 1.0 | — | 2D mode altitude correction gain. Pulls drones toward the mean swarm altitude. |

---

## Control Loop Diagram (Final Architecture)

```
                    ┌─────────────────────────────────────────────┐
                    │              VelocityControl                │
                    │                                             │
  Joystick ──────►  │  userVelX/Z ──► body→world ──► vel error   │
  (normalised)      │                                  │          │
                    │                         ÷ timeConstantAccel │
                    │                                  │          │
                    │                              userAccel      │
                    │                                  │          │
                    │                                  ├──► SUM ──┤──► desiredTheta ──► attitude loop ──► torque/force
                    │                                  │          │
  SwarmAlgorithm ──►│  swarmAcceleration ──► filter ───┘          │
  (Olfati-Saber)    │  (already acceleration)                     │
                    │                                             │
  Joystick ──────►  │  userAltitudeRate ──► desired_height        │
  (throttle)        │                         │                   │
                    │                    PD controller ──► thrust │
                    │                                             │
                    └─────────────────────────────────────────────┘

                    ┌─────────────────────────────────────────────┐
                    │              OlfatiSaber                     │
                    │                                             │
  For each          │  cohesion(position_i, position_j)           │
  neighbor j:       │  + c_vm * (velocity_j - velocity_i)        │──► swarmAcceleration
                    │  + obstacle_avoidance                       │    (world frame)
                    │                                             │
                    └─────────────────────────────────────────────┘
```
