# Fixed per-drone gimbal-yaw bias in the DJI camera pose

**Status:** diagnosed and measured here; **not fixed at source**. A temporary,
self-calibrating shim in `DJI_Swarm/AOS server/clip_replay.py` makes existing recordings
replay correctly — see *Temporary shim* below, and delete it when the real fix lands.

## The claim

`dji_camera_pose.py:41` documents `gimbal_yaw` as *"compass bearing, 0 = North, increasing
clockwise"*, and `CameraPoseSolver` builds the camera rotation from it directly
(`quat_from_gimbal(telem['gimbal_yaw'], …)`, line 172). On this fleet it is that bearing
**plus a fixed per-aircraft offset**, and the cameras physically point along the aircraft
heading rather than along the reported bearing.

`gimbal_yaw − heading`, median over each clip:

| drone | MED_facade_stationary_1 | MED_facade_stationary_2 | grass_nadir_long |
|---|---|---|---|
| 1 | +2.8° | +3.0° | +3.0° |
| 2 | −15.5° | −15.5° | −15.6° |
| 3 | −6.6° | −6.6° | −6.4° |

Constant to ±0.2° across two sites, two gimbal pitches (level and −90° nadir) and three
separate flights. Between MED_1 and MED_2 the *mean* `gimbal_yaw` moved +7.5° and the mean
`heading` moved +7.4°, so `gimbal_yaw` is an absolute (world) bearing that tracks the body —
it is simply offset per aircraft.

## Why it matters more than everything else

The **differential** part is the damaging one: it points the cameras up to 18.2° apart in the
planar solve while they were physically parallel. At the MED facade's 34.255 m standoff that
is `Z·tan(18.2°)` = **11.3 m of seam** — larger than every other term in
`CLAUDE.md`'s error budget combined. It is a rotation, so no per-view translation can absorb
it, and the pose refiner is structurally unable to help. (The common-mode part merely rotates
the whole mosaic on the plane and costs nothing.)

Predicted `Z·tan(Δgimbal_yaw)` against the measured masked-NCC peak between warped view pairs,
on one frame of MED_1:

| pair | Δgimbal_yaw | predicted | measured |
|---|---|---|---|
| 0-1 | 18.4° | 11.40 m | 11.48 m |
| 0-2 | 9.8° | 5.92 m | 6.24 m |
| 1-2 | −8.6° | −5.18 m | −5.24 m |

Removing the bias, with the calibration derived from a **different clip at a different site**
(`grass_nadir_long`) so it is not a per-clip fit:

| clip | raw pose | calibrated |
|---|---|---|
| MED_facade_stationary_1 | 6.03 m (92 px) | 0.96 m (15 px) |
| MED_facade_stationary_2 | 12.4 m | 0.99 m |

Ruled out along the way: the warp algebra round-trips plane → pixel → plane to **1e-15**, and
the triangulated scene-plane offset is −0.55 m against a measured 34.255 m standoff. The
geometry and the plane were never the problem.

## Two hypotheses, and how to tell them apart

The data above cannot distinguish these, because the gimbals never panned relative to the
body during any recorded clip. Both are consistent with everything measured.

- **(A) Reported-bearing offset.** The gimbal physically points along the body as commanded,
  but its yaw zero (mount or encoder) disagrees with the flight controller's compass by a
  constant. Fix: subtract the constant.
- **(B) `gimbal_yaw` is not the field we think it is** — a joint angle, a setpoint, or a
  different reference frame — and happens to differ from `heading` by a constant here. Fix:
  read the correct field.

### Tests, cheapest first

1. **Bench, no aircraft.** `tools/planar_clip_bench.py --clip <label> --gimbal-yaw-cal off`
   vs `--gimbal-yaw-cal self`. Reproduces the table above in ~20 s and is the regression test
   for any fix.
2. **Ground test, no flight.** Power up all three on the ground, gimbals commanded forward,
   aircraft aligned along a marked line. Read `heading` and `gimbal_yaw`. If the same three
   constants appear, it is static — mount or encoder, not flight-dependent.
3. **Commanded-pan test — this is the one that separates (A) from (B).** Pan one gimbal by a
   known −30° and re-read. If `gimbal_yaw − heading` moves by exactly −30°, the field is a
   valid bearing carrying a constant offset ⇒ **(A)**, and subtracting the per-drone constant
   is correct *and preserves genuine pans*. If it does not track, the field is not the camera
   bearing ⇒ **(B)**, and using `heading` is the safer read.
4. **Optical ground truth.** From one spot, point all three at a distant landmark and compare
   its pixel column across the frames. Column difference × `1/f` gives the true relative
   bearing with no telemetry involved — this is independent of both the compass and the
   gimbal encoder, so it also settles whether the *compass* is the biased sensor rather than
   the gimbal.
5. **Does it follow the airframe or the gimbal?** Re-seat or swap a gimbal and re-measure. If
   the constant moves with the gimbal it is an encoder zero; if it stays with the airframe it
   is a compass calibration. This decides where the calibration should be stored.

## Recommended fix

Prefer **(A)**: a per-drone `gimbal_yaw` calibration constant, subtracted where the pose is
built, so genuine gimbal pans survive. Using `heading` in place of `gimbal_yaw` gives an
identical result on every clip recorded so far, but silently discards any future pan.

Whichever is chosen, the constant should be **stored per aircraft in config, not hardcoded**,
and re-derived after a gimbal re-seat or a fleet change — the bias is stable to ±0.2° but
nothing guarantees it survives maintenance.

Note the fix must reach two places: `dji_camera_pose` for live flights, and the replay path
for recorded clips, because `clip_replay.py` reads the already-solved `quat_*` columns
straight out of `drone{N}_frames.csv` — the bias is baked into every existing recording. The
raw `heading` / `gimbal_yaw` / `gimbal_pitch` columns are all present, so recorded clips can
be corrected or re-solved without re-flying.

## Temporary shim (delete when fixed)

`clip_replay.py` carries `_gimbal_yaw_bias_deg` / `_apply_yaw_fix`, applied in `_pose_of` and
announced on every run:

```
[replay] TEMPORARY gimbal-yaw fix active: d1 +2.8, d2 -15.4, d3 -6.6 deg removed
         (spread 18.2 deg). Remove this shim once dji_camera_pose is fixed;
         --no-gimbal-yaw-fix disables it.
```

It is self-calibrating — the median `gimbal_yaw − heading` over the clip's own rows — so it
needs no fleet constants and preserves any genuine pan *within* a clip. `--no-gimbal-yaw-fix`
reproduces the raw recorded poses.

**Remove the shim and the flag together with the source fix**, or the two will double-correct.

## What is left after the bias is removed

Residual seam drops to 0.6–1.0 m (9–15 px), which is in the regime the existing estimators
were designed for. Two things are then worth re-measuring rather than assuming:

- The pose refiner only engages *now* — at 6 m it reported `idle: all 3 views rejected`,
  because `planarRefineMaxShift` defaults to 3 m and `PlanarStitcher.py:925` rejects rather
  than clamps, and tests the measurement *before* the zero-mean gauge at `:954`. Once under
  the gate it takes 0.96 → 0.50 m on MED_1. That gate ordering is still worth fixing on its
  own merits; it is deferred until the calibration lands so the two changes can be measured
  apart.
- The bench's triangulated plane offset still reads −2.5 m on MED_1 against −0.4 m on MED_2,
  so the standoff may carry ~1–2 m of error — consistent with the clip's own recorded warning
  that a map-traced facade is worth about ±2 m once DJI's absolute GPS is included.
