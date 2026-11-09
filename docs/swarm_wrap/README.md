# The wrap study (S11 Wrap Hold)

The registry entry is [docs/studies/S11_wrap_hold.md](../studies/S11_wrap_hold.md). This manual has the method, every
measure and decision, and how to run it. Code: `citysim/swarmwrap.py` (on S8's `swarmshape.py`, S7's `swarmflow.py`,
S6's `swarmtune.py`), `swarmwrap_bench.py`, `swarmwrap_replays.py`, `swarmwrap_report.py` (+ `.html`); the replica's
`inside_formation`, `wrap_holds` and `spread_target`; in Unity `SwarmManager.cs` and `SwarmPlaneController.cs`, and the
bench's `spread` trajectory column (`SwarmBenchRunner.cs`, read by `tierc.read_traj`). Outputs:
`experiment_results/swarm_wrap/`.

## 1. The question

The user's, looking at the map of ERKS's first swarm trial in the trial scene (internal_4, 2026-11-06): at the second
goal the swarm widened as it first surrounded the building, then narrowed while it was still wrapped round it. The
adaptive spread (S7; 0.85 in the streets with a 3 s filter, BlockShapedPatchCityWorld since S8) keeps the full spread
only while

- the stick's line from the centroid, 40 m long, passes within 0.8 of a building's radius of its centre (in the
  ellipse's unit space under S10's shape), or
- with the stick centred, the centroid itself is that close to a building's centre, or
- no building stands within 40 m of the centroid at all.

So a formation wrapped round a building narrows as soon as its centroid drifts more than 0.8 of the building's radius
off its middle, or the pilot steers so the line ahead misses it. The aim is a rule that keeps the swarm wide while it
is round a building, and the question is what that costs in the streets it then enters, where S9 found narrowing makes
the swarm faster and where the losses cluster.

## 2. The setting: `SwarmManager.spreadWrapRule`

`SpreadWrapRule { Off = 0, Hold = 1, HoldUntilLeaving = 2 }`, under `adaptiveSpread` (values spelled out: Unity
serialises enums by integer, and swarm_params maps the members by position). Off by default, so every scene flies as
before.

- **Wrapped.** A building is wrapped when its centre lies inside the alive drones' convex hull, tested as no gap of
  180° or more in the drones' bearings round it (`SwarmPlaneController.InsideFormation`,
  `swarm_replica.inside_formation`). The centre is the ellipse's under S10's shape and otherwise the circle S7's rules
  aim by, the same one the rule's aim test uses. Fewer than three drones wrap nothing, and a building farther from the
  centroid than the farthest drone is skipped before the test.
- **Hold**: the target is 1 while any building is wrapped, whatever the stick's look-ahead says.
- **HoldUntilLeaving**: the same, except while a stick above 0.1 points away from that building (its centre behind the
  centroid along the stick), so the formation starts narrowing as it leaves rather than in the next street.
- The aggregates pass caches the drones' (x, z) and the farthest from the centroid each tick, so the test costs one
  sort of ten bearings per building near the formation.
- **Off is S7's rule exactly**: the check is guarded on the rule, and 40 of S10's stored confirmation flights (every
  kind, every city) re-flew bit for bit in the replica.

## 3. ERKS's flights, open loop (`swarmwrap human`)

The rule replayed on ERKS's three swarm flights as they flew (the drones' recorded positions, the recorded stick
rotated by the body yaw, the 3 s filter), from the start line to the last find. The rule as flown reproduces the
logged scale to a mean of 0.0005. A building was wrapped 51% of the time.

| rule | wrapped: at full spread (≥ 0.99) | wrapped: narrowed (< 0.95) | not wrapped: narrowed | mean scale |
|---|---|---|---|---|
| Off (as flown) | 36% | 39% | 86% | 0.918 |
| Hold | 65% | 9% | 74% | 0.947 |
| HoldUntilLeaving | 52% | 23% | 84% | 0.932 |

The wrapped moments themselves: the centroid sat a median 0.78 of the wrapped building's radius off its centre (in the
ellipse's unit space; beyond the rule's 0.8 for 48% of the time, 9.4 m median), and the stick was held for 52% of the
time, almost always at full deflection. Wrapped stretches lasted a median 5 s, the long goal searches 30–80 s.

Open loop is only what each rule decides on the same drones. A formation held wider round a building also moves
differently, which is what the flights below are for.

## 4. How it was flown

S8's harness: `ContactSim` on the prism city (`PolyCity`), settled starts, the bench's waypoint pilot, d_ref 2.0, the
scene's city (seed 1) for the screen and seeds 1–4 for the confirmation. Every row names every law field
(`obstacleShape`, `ellipseMaxAspect`, `d_obs`, the slide, route stragglers, the adaptive spread, its factor and filter,
`spreadWrapRule`).

Rows (`screen_cands.json`): **E15_35a** (the scene: Off), **H15_35a** (Hold), **L15_35a** (HoldUntilLeaving) and
**F20** (the adaptive spread off, the spread held at 2.0: S9's row, the end point of holding everywhere).

Flights (route indices: the screen from 19000, the confirmation from 20000, the bench from 21000; S6–S10 used
0–18999): S8's four (`cross`, `pass`, `street`, `face`) and **`search`**, a goal search round one building:

- the street route from a junction 120–260 m away to the nearest corner of a random interior building's block;
- the circuit of the block's corners at full stick, either way round, with a pause after each corner (chance 0.5)
  beside the building on that corner's side, 0.4–1.2 of its enclosing radius off its centre (at most 16 m), for
  12–22 s;
- a hold on its centre (6–12 s), then away by the streets to a junction 150–300 m off.

A pause or hold is a negative stick: the bench's centring pilot on the point, stick = 0.25 × error / maxSpeed, capped
(0.5 for a pause, 0.3 for the hold), never captured (`SwarmBenchRunner`; `swarmwrap.fly` flies it identically). It
was calibrated against ERKS's wrapped moments (§3) in three steps; the version flown, under the scene's rule on 24
searches:

| | ERKS (internal_4) | scripted search |
|---|---|---|
| narrowed while wrapped | 39% | 44% |
| stick held while wrapped | 52% | 68% |
| centroid off the building's centre, median (unit) | 0.78 | 1.06 |
| beyond the rule's 0.8 | 48% | 79% |

So the scripted search sits in ERKS's regime and is a little harder on the rule as flown: it steers more and holds
further off the middle.

## 5. Measures

On every flight, every 0.1 s (`WrapLog`; for Unity the same from the trajectory, the scale from its `spread` column):

- `wrap_s`: time with a building wrapped (§2's test, the same in every row);
- `wrap_full`: the share of `wrap_s` at full spread (scale ≥ 0.99), **the question**; `wrap_sigma` the mean scale then,
  `wrap_ring` the formation's mean radius then;
- `exit_s`, `exit_sigma`: the 10 s after the last building left the formation (none wrapped since), and the scale
  then: where the cost is predicted to fall;
- loss events (instants at which drones were lost: a colliding pair is one) and contacts, each placed wrapped /
  leaving / elsewhere, with each phase's exposure. The phases are geometric, so every row, F20 included, carries the
  same moments (lesson 47);
- S8's: cohesive and street speed, one group, losses and contacts per km, in pieces at the end, hull share,
  surrounding a building met (`cross_wrapped`), the square-on split;
- for `search`: its hold window (the circuit's first waypoint to the hold's end) wrapped and at full spread
  (`hold_full`), the scale over it, the speed it leaves on (`leave_v`) and the scale over the first 10 s of leaving
  (`leave_sigma`).

## 6. The screen

`screen1/` (the scene's city, 936 flights, no errors): per row 40 straight lines, 30 blocks, 60 street routes, 24
square on and 80 searches, about 42 km of formation flights and 6 h of flight.

| row | wrapped: mean scale | at full spread | ring while wrapped (m) | street speed (m/s) | loss events | contacts |
|---|---|---|---|---|---|---|
| E15_35a (the scene, Off) | 0.942 | 31% | 16.21 | 6.23 | 6 | 43 |
| H15_35a (Hold) | 0.983 | 70% | 16.51 | 6.22 | 7 | 42 |
| L15_35a (HoldUntilLeaving) | 0.971 | 55% | 16.41 | 6.23 | 6 | 43 |
| F20 (2.0 held) | 1 | 100% | 16.62 | 5.87 | 6 | 56 |

Both rules did what they were built to do, with no sign of a cost in the streets. Six or seven loss events a row
cannot compare losses (lesson 46); the confirmation is sized for that. The full-spread share is mostly the 3 s
filter's lag on short wraps (a street route wraps a building for ~2.5 s), so the rule's bar is carried by the mean
scale.

## 7. The rule

`rule.json`, written 2026-11-06 20:29:40, after the screen and before the confirmation flew (`swarmwrap rule`):

- **Base**: the scene, E15_35a. F20 is a reference, judged but not a candidate.
- **Safe**, against the base: loss events over every flight (instants with a drone lost: a colliding pair is one) not
  more frequent per hour at one-sided exact p ≥ 0.10; contacts per hour at most 1.25×; ending in pieces at most 2
  points more; street cohesive speed and the speed leaving a search at least 0.97× (the cost the open-loop replay
  predicted); the square-on split at most 1.1×; hull share at least 0.5; buildings met on a straight line surrounded at
  most 5 points less.
- **Delivers** (the question): the shortfall from full spread while wrapped (1 − the mean scale then) at most half the
  base's, and the wrapped time at full spread at least 25 points above it.
- **Pick**: among rows safe and delivering, the nearest full spread while wrapped; within 0.005 of it, the fastest in
  the streets. None: the scene stays as it is.

## 8. The confirmation

`confirm/` (the scene's city and seeds 2–4, 4,352 flights in 66 min, no errors): per city and row 30 straight lines,
30 blocks, 100 street routes, 12 square on and 100 searches; per row about 219 km of formation flights and 30 h of
flight.

| row | wrapped: mean scale | at full spread | narrowed (< 0.95) | ring while wrapped (m) | street speed | leaving a search | loss events (/h, 95%) | contacts /h | rule |
|---|---|---|---|---|---|---|---|---|---|
| E15_35a | 0.941 | 31% | 48% | 15.81 | 6.18 | 6.29 | 45 (1.50, 1.09–2.01) | 8.17 | base |
| **H15_35a** | **0.982** | 69% | 15% | 16.11 | 6.17 | 6.27 | 45 (1.50, 1.09–2.01), p 0.54 | 7.63 | **safe, delivers: the pick** |
| L15_35a | 0.970 | 56% | 25% | 16.03 | 6.18 | 6.29 | 45 (1.50, 1.09–2.01), p 0.54 | 7.80 | safe; short on both bars by a hair |
| F20 | 1 | 100% | 0% | 16.37 | 5.78 | 6.14 | 49 (1.61, 1.19–2.12), p 0.41 | 9.15 | unsafe: street speed 0.935× |

- **Hold delivers what the question asked of the setting**: the shortfall from full spread while wrapped falls from
  0.059 to 0.018, under a third, and the narrowed wrapped time from 48% to 15%. HoldUntilLeaving releases while the
  pilot steers away. It falls just short of both bars (shortfall 0.0302 against the bar's 0.0296; full spread 55.6%
  against 56.4%), delivering a little over half of what Hold does.
- **No measured cost.** Loss events 45 against 45 over 30 h a row, contacts 7% fewer, the street speed 0.1% lower,
  the speed leaving a search 0.3% lower, the split, hull share and surrounding unchanged. Loss events while wrapped:
  4 for the scene, 3 for Hold.
- **The formation barely changes.** The ring round a wrapped building is 0.30 m wider under Hold (1.9%), 0.38 m in the
  searches (15.31 → 15.69 m, 2.5%), and even the spread held at 2.0 is only 0.56 m wider (3.5%), for the 6.5% of
  street speed it costs. Wrapped round a building, the drones sit at the standoff on its avoidance shape, so the
  building sets the formation's size and the spread setting moves it little (lesson 52). What Hold does change is how
  long the swarm stays round the building: wrapped time per search 36.6 → 40.0 s (F20 49.0 s), since a slightly wider
  formation keeps the building's centre inside it a little longer.
- **The searches**: wrapped time at full spread 32% → 75% (HoldUntilLeaving 61%); the hold window (the circuit and
  the hold on the building) at full spread 12% → 32%, its mean scale 0.937 → 0.961.

## 9. Unity

`bench1` (BlockShapedPatchCityWorld headless at 20×, pilot heading on; 142 flights a set: 20 straight lines, 20
blocks, 50 street routes, 12 square on, 40 searches; about 3.3 h a set), with the replica's twins (Unity's starts,
the same routes) gated against it (`swarmwrap_bench gate`).

| set | wrapped: mean scale (twins) | at full spread | narrowed | ring (m) | street speed (twins) | leaving a search | loss events (twins) | contacts (twins) |
|---|---|---|---|---|---|---|---|---|
| E15_35a | 0.938 (0.940) | 29% | 50% | 16.23 | 6.16 (6.20) | 6.29 | 6 (6) | 16 (23) |
| H15_35a | 0.979 (0.980) | 62% | 18% | 16.55 | 6.14 (6.16) | 6.51 | 9 (8) | 19 (28) |
| L15_35a | 0.965 (0.968) | 49% | 28% | 16.48 | 6.14 (6.21) | 6.24 | 5 (6) | 26 (20) |
| F20 | 1 (1) | 100% | 0% | 16.86 | 5.75 (5.86) | 6.10 | 5 (4) | 24 (28) |

- **Unity reproduces the replica on everything the rule acts on**: the scale while wrapped to 0.003, the street speed
  to 1%, the ring to 0.1 m.
- **Hold's 9 loss events against the scene's 6 are not the rule's** (one-sided p 0.30). Every one is located
  (`swarmwrap_bench losses`, `bench1/bench/losses.json`). At each of the 9 the spread scale was within 0.01 of what the
  scene's own rule gives at that moment, so Hold was not acting, and 7 came before the flight had wrapped any building
  at all, where Hold and Off are the same law. The twins lose 8 against 6 on the same starts, and the replica's 30 h a
  row 45 against 45. It is the run-to-run divergence of chaotic flights (lesson 53).

## 10. What it means

- **Hold makes the setting do what the user expected**, in the replica, in Unity and on ERKS's own flights (narrowed
  while wrapped 39% → 9% open loop): a swarm wrapped round a building is no longer told to narrow while it is still
  round it. It costs nothing measurable, in the streets, leaving a building, in losses or in contacts.
- **What the pilot would see changes little.** The formation round a building is about 2% wider under Hold, because
  the building, not the spread, sets its size while it is wrapped. ERKS's map coloured the track by the spread
  *setting*, and the ring itself moves far less than the setting. Expect Hold to remove a pulse in the setting more than to change
  what the cameras see.
- **HoldUntilLeaving** sits between the two and offers nothing Hold does not: it releases while the pilot steers away,
  which the streets did not need.
- **The spread held at 2.0** (S9's F20) is what holding everywhere costs: 6.5% of street speed for 0.6 m of ring.
- Hold is the rule's pick, and the user adopted it on 2026-11-09 (below).

## Adopting it (done 2026-11-09)

BlockShapedPatchCityWorld and PracticeBlockWorld fly Hold since 2026-11-09, at the user's word. From
`Assets/Scripts/Experiment` in the `stitching` env, once per scene:

```
python -B -m citysim.swarmtune apply --cand ../../../experiment_results/swarm_wrap/confirm_cands.json \
    --label "H15_35a the scene with the wrap rule Hold: full spread while a building lies inside the formation" \
    --scene BlockShapedPatchCityWorld        # then again with --scene PracticeBlockWorld
```

This inserted `spreadWrapRule: 1` into each scene's SwarmManager after `spreadFilterTime`; every other field already
held the row's value. PracticeBlockWorld is a copy of the trial scene, and its SwarmManager was identical to the trial
scene's field for field before the change, so it took the same line rather than a rebuild (which would also have
redone its city and its occlusion bake for one field). `--scene` was added to `swarmtune apply` for this.

**Checked:** ten stored confirmation flights of H15_35a (one of each kind in the scene's city and in held-out seed 2)
re-flew identical with an empty override, and the same ten of E15_35a with `PRE_S11` alone; on those routes the two
rules' stored rows differ in 9 of 10, so the check can tell them apart. **An empty harness override now means Hold**:
`swarmwrap.PRE_S11` (`{"spreadWrapRule": 0}`) restores the scene as S11 found it. S11's own candidates all name the
rule, so its stored runs do not depend on the scene.

The editor held BlockShapedPatchCityWorld open when the file was written, so it has to reload the scene (it asks) before
Play flies Hold; saving the scene from the editor without reloading would write the old in-memory value back.

## Running it

From `Assets/Scripts/Experiment` in the `stitching` env. Until the law is committed and `FORCE_LAW_COMMIT` bumped, set
`SWARM_REPLICA_ALLOW_STALE=1`.

```
python -B -m citysim.swarmwrap human                                    # §3, human/summary.json
python -B -m citysim.swarmwrap_calib                                    # §4, calibration.json
python -B -m citysim.swarmwrap screen --name screen1 --cand ../../../experiment_results/swarm_wrap/screen_cands.json \
    --street 60 --search 80
python -B -m citysim.swarmwrap table --name screen1
python -B -m citysim.swarmwrap rule                                      # rule.json (refuses to overwrite)
python -B -m citysim.swarmwrap confirm --name confirm --cand .../confirm_cands.json --cross 30 --street 100 \
    --face 12 --search 100
python -B -m citysim.swarmwrap choose --name confirm                     # verdicts.json
python -B -m citysim.swarmwrap_bench plan --name bench1 --cand .../confirm_cands.json
#   then run_bench.ps1 as the plan prints (tag swarmwrap_bench1), and
python -B -m citysim.swarmwrap_bench gate --name bench1 --runs D:\claude_obstacle_bench\runs\swarmwrap_bench1
python -B -m citysim.swarmwrap_bench losses --name bench1 --run D:\claude_obstacle_bench\runs\swarmwrap_bench1
python -B -m citysim.swarmwrap_replays                                   # the page's replays, each twin checked
python -B -m citysim.swarmwrap_report                                    # the page; publish swarm_wrap_bare.html
python -B Assets/Scripts/Experiment/citysim/tests/run_tests.py -k swarmwrap      # from the repo root
```

The page: https://claude.ai/artifact/658t8HQXHxpyqU6T7VX7CD (publish `swarm_wrap_bare.html` with the files
`swarm_wrap_files.json` lists, root `experiment_results/swarm_wrap`).
