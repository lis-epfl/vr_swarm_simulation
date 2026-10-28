# Swarm testing offline: the headless Unity bench and the 2D replica

Two tools for judging a change to the swarm before it reaches a headset. They answer different questions:

| | **Headless Unity bench** (`run_bench.ps1`) | **2D replica** (`swarm_replica.py`) |
|---|---|---|
| What flies | The real scene and code, PhysX and all, in batchmode Unity | A numpy model of the horizontal force law and control loop |
| Driven by | Scripted flights (`scenarios/*.json`): transits, tours, rams, zigzags, spread steps, goal flights | A test's **recorded** pilot inputs (stick, body yaw, spread), or a bench run's flights (`validate`) |
| Good for | Building contacts, drone losses by Unity's own rules, step response, anything physical | Fast A/B on how real pilots flew: drone-drone kills at tight spread, spacing, and attributing a crash to force terms |
| Cost | ~3 min startup, then ~15 min per 120-flight set at 20x | ~1 min for a whole test x 3 seeds |
| Blind to | Real pilot behaviour (flights are scripted) | Heading, altitude, street furniture, PhysX contact, DroneHealthMonitor (see below) |

Use the replica to narrow a design down and the bench to confirm it. The September obstacle retune and the
damper were found that way. A replica result about building contacts needs the bench to confirm it.

Run Python with the `stitching` miniconda env:
`%LOCALAPPDATA%\miniconda3\envs\stitching\python.exe`.

## The bench

```powershell
# 1. a config: flight set(s) x parameter set(s)
python bench.py config --scene DiamondCityWorld --scenarios scenarios/DiamondCityWorld_heldout.json `
       --set scene --set "no damper:c_damp=0" --out D:\claude_obstacle_bench\cfg_damper.json
# 2. fly it (syncs the copy first, then batchmode; prints the report at the end)
.\run_bench.ps1 -Config D:\claude_obstacle_bench\cfg_damper.json -Tag diamond_damper
# 3. compare runs later
python bench.py report D:\claude_obstacle_bench\runs\diamond_damper D:\claude_obstacle_bench\runs\other --kinds
```

- **The copy.** The live editor holds the project open, so batchmode runs on a copy
  (`D:\claude_obstacle_bench\vr_swarm_simulation`). **Every run mirrors `Assets`, `Packages` and
  `ProjectSettings` from this working tree into it first**, uncommitted edits included, so the bench
  never flies stale code. The copy is not a git checkout: the large packs, such as `Modular City Pack`, are gitignored. Its
  `Library` is kept between runs, so only what changed reimports. If the copy has no Library, pass
  `-LibrarySeed` with another D: copy's Library. `-CompileOnly` checks the copy compiles; do this after any C#
  change, because the live editor compiles these runners too. `-SyncOnly` only mirrors.
- **Parameter sets are overrides on the scene as authored.** `--set scene`, with no fields, flies the scene
  exactly as saved. The runner snapshots the scene's `SwarmManager` and restores it before every set, so a
  retune is never undone by an old config and one set cannot leak into the next. Field names and values are
  checked against `SwarmManager.cs` by `bench.py config` and again by the runner before anything flies. A
  typo costs seconds (exit 2), not a run. Enums go by integer.
- **The look-direction gap fill is off unless you ask for a pilot.** It reads the pilot's body yaw, which
  nothing sets headless, so the scenes' `fillLookDirectionGap: 1` would be a claim that never flew. Without
  `--pilot-heading` it is written off explicitly and forced off. `--pilot-heading` injects a heading that faces the
  current waypoint (as a pilot who looks where they fly) and adds the look-gap metrics to each flight.
- **The city.** `GoalPatchReplacer` is disabled by default, so flights are comparable run to run.
  `--goal-replay <session stem or path>` replays a recorded run's goal layout instead, through the
  replacer's own Replay mode. That puts in its goal patches, street lights included.
  `--scene ProceduralCityWorld --layout <id or file>` flies a **generated layout** (the density study's
  irregular-block cities, `Assets/Scripts/Experiment/layouts/`, sweep layouts under `layouts/sweep~/` included):
  the runner builds it with the scene's `ProceduralCityBuilder` before anything spawns (a `CITY_LAYOUT` left in
  the shell is cleared, so the config is what flies), and `run_bench.ps1` copies the layout file beside the
  config as `layout.json`, so a run records exactly which city it flew. The goals' pedestrians are off unless
  `--walkers` (they cost frames and the swarm ignores them).
- **Drone count.** `--drones N` overrides the scene's `swarmSpawn` count (0 = the scene's); `--drones 1` is the
  single-drone condition. The formation metrics degrade gracefully at one drone (no hull, no pairs). A **lone
  drone's heading is steered by the scripted pilot** -- turned to face the commanded direction, and scanning at
  the scenario's `holdYawRate` during a hold -- because its view is its heading (AttitudeAlgorithm hands a lone
  drone the yaw stick). A swarm's headings stay the hull rule's and its yaw stick stays zero, as before.
- **Timing.** `timeScale` 20 runs 20x real time without changing the 0.02 s physics step. The runner sets
  `Time.maximumDeltaTime = 1` in `Start` and before every flight. `VrFramePacing` caps it at 0.1 s at scene
  load, which would quietly hold a 20x run to ~5 steps per frame. Each flight records `sim_per_wall`, and
  the report warns when it falls below half the time scale.
- Disabled in every run: `PyUniSharingFast`, `UDPReceiverManager`, `ImageSharing`, `ExperimentRecorder`,
  and every camera. A batchmode run beside the user's editor must open no named section, port or recording.

### Config (JSON, written by `bench.py config`)

`scene`, `timeScale`, `settleTime`, `altitude`, `capture` (m from a waypoint that counts as reaching it),
`pilotHeading`, `pilotYawRate`, `goalReplaySession`, `layout`, `droneCount`, `walkers`,
`paramSets[] {label, names[], values[]}`, `scenarios[]`.
A scenario is `{kind, sx, sz, T, wps: [x, z, stick, timeout]*, record?, spread?: [t, d_ref]*}`. The swarm is
reset to `(sx, altitude, sz)`, settles, then flies toward each waypoint at `stick` (0–1), moving on when
within `capture` or after `timeout` s (0 = none). `spread` drives the spread stick piecewise-linearly, and
`record` writes the trajectory to `traj_<set>_<index>.csv`.

**Holds** (the goal flights): a waypoint with a **negative** stick is a hold -- a centring pilot on its point
(stick = `holdGain` × error / `maxSpeed`, clipped to |stick|: citysim's `formation.PILOT_GAIN` / `PILOT_MAX`)
for its timeout, never captured, the spread stick at `holdSpread` meanwhile (0 = the schedule's). The
waypoint before a hold is captured within `arrive` m (0 = `capture`), and `endOnDone` ends the flight when the
last waypoint is done, so `T` is only a cap. `swarm_replica.py validate` flies the same rule.

**Spread per waypoint** (S4's manoeuvres, `citysim/split_bench.py`): a positive `wpSpread[i]` sets the spread stick
while waypoint `i` is being flown, over `holdSpread` and the schedule; zero or a short array leaves those in charge.
`citysim/split_probe.py`'s `Flight` is the replica side of the same pilot, so every S4 replica trial has a bench twin.

### Flight sets (`scenarios/`)

- `ScaledCityWorld_heldout.json` / `_spread.json` are **frozen** from the September obstacle retune's configs:
  120 held-out flights (30 transit, 10 tour, 50 ram, 30 zigzag) and 16 spread steps. Their generator was
  lost; keeping these exact keeps new numbers comparable with the CLAUDE.md figures (16 drones lost,
  0 contacts, 95% clean on the retuned gains, hollow core on).
- `DiamondCityWorld_*.json` were generated by `bench.py scenarios --city DiamondCityWorld --seed 1`. The
  generator rebuilds that structure from a city's `city_obstacles_<city>.json` (or a generated layout's file):
  transits edge to edge, rams 90 m either side of a building, and so on. Regenerating with another seed makes
  a different test, so results flown on the old file stop being comparable.
- `bench.py scenarios --city <layout> --goals` writes the **goal flights** the density study's Tier C flies
  (`citysim/tierc.py`): from the layout's spawn along the street route at full stick to every goal, in the
  layout's order and in reverse, at transit spreads 0.4 and 1.0, arriving within 25 m and holding 60 s at
  spread 1.6. It is citysim's Tier B' program (`citysim/dynamic.py`), so the two tiers fly the same thing.

### Output (`<bench root>\runs\<tag>\`)

`config.json`, `meta.json` (live commit and uncommitted-file count at launch), `layout.json` (a layout run),
`unity.log`, `traj_*.csv`, and `results.jsonl`: one `start` line (with the layout id, its coverage and goal
count, the drone count and the gimbal pitch), one `params` line per set (overrides **and every effective
SwarmManager value**), one `flight` line per flight, then `done`. Errors write an `error` line and exit 2
(config) or 3 (runtime).

`traj_<set>_<index>.csv`, every 5 physics steps: `t, alive_mask, hull_mask, shown_mask` (bit i is drone i:
alive; on the hull, `AttitudeAlgorithm.BoundaryEstimate`; feed shown, `BoundaryFeedReady`, the gate
`OUTER_CIRCLE` applies, and a lone drone's feed always), `x0, z0, ...`, `yaw0, ...` (deg), `y0, ...` (m),
`wp` (the waypoint being flown, holds included) and `bodyYaw` (the scripted pilot's heading). Readers address
columns by name; runs from before 2026-10-12 stop after the yaws.

The flight fields are `alive_start/end`, `contacts` (Obstacle layer), `hard_contacts` (> 2 m/s normal) and
`drone_body_contacts`. `other_contacts`/`other_names` are colliders off the Obstacle layer: poles, props.
`deaths` is DroneHealthMonitor's reasons. Then `path_len`, `track_sum/n` (progress speed), `min_pair`,
`near_ticks`, `hull_frac`, `nn_mean`, `split_frac`, `mean_speed`, `sim_per_wall`, and the `gap_*`/`live_shown_*`
look-gap metrics with a pilot heading. Their field of view is `FpvCameraFov`'s, the one constant the scenes,
`analyse.py` and citysim share.

### Reference runs (2026-10-09, held-out + spread sets, 136 flights each)

| run | set | lost | building contacts | clean | notes |
|---|---|---|---|---|---|
| `runs\baseline_scaled_20261009` | `scene` (core off, damper on) | 20 | 1 | 94.9% | every loss in a zigzag reversal |
| | `sep_equiv` (core on, absolute, no damper) | 24 | 1 | 92.6% | September's conditions; it recorded 16 / 0 / 95% on the 120 |
| `runs\baseline_diamond_20261009` | `scene` (core relative + damper) | 11 | 1 | 94.9% | |

Losses come in drone-drone pairs, so 20 lost is about 10 events and differences of a few events are noise.
Spread steps in all three: expansion t90 2.3–3.5 s, contraction t90 3.8–5.6 s, overshoot ≤ 11%.

## The replica

```
python swarm_replica.py sim    --test internal_3 --variant "as flown" --variant "core off" --seeds 3
python swarm_replica.py sim    --test internal_2 --variant "mine:c_damp=3,d_damp=0.4"
python swarm_replica.py forces --test internal_3                # every drone-drone crash in results/crashes.csv
python swarm_replica.py step   --scene ScaledCityWorld --variant scene --variant "no damper"
python swarm_replica.py params --test internal_3 --run BBCC_t2
python swarm_replica.py validate D:\claude_obstacle_bench\runs\<tag> [more ...] --out <dir>   # the gate
```

- **`sim`** replays each swarm run's pilot inputs on a simulated swarm, starting from that run's take-off
  state. It flies through that run's city: the edit-mode buildings with its goal patches swapped in
  (`analyse.scene_boxes`). The swap matters. On internal_2's inputs with the damper and relative core, the
  replica predicts 0.46 drone-drone kills/min at d_ref 0.4 with the swaps and 0.15 without, because the
  tight-spread minutes are flown searching among the goal patches' narrow pillars. internal_3 measured 0.58
  in Unity. Output is `<test>/results/replica/sim_<time>.csv`.
- **`forces`** evaluates the force law on the recorded states before each crash and attributes the pair's
  closing acceleration to each term (positive pushes the pair together). Run `analyse.py run <test>` first for
  `crashes.csv`, or name crashes with `--crash RUN@T:i,j`.
- **`step`** is the spread-response requirement: d_ref steps at hover in open sky (expand, contract,
  re-expand), with t50/t90/overshoot, then a cruise/stop/reversal block at 0.4.
- **Parameters are read, not typed in** (`swarm_params.py`). The order is the SwarmManager.cs initialisers,
  then the scene's saved values, then the test's `test.json` `"swarmParams"`, then the run's
  `<stem>_swarm.json` (ExperimentRecorder writes one since 2026-10-09), then `--set`/`--variant`. The airframe
  (`maxSpeed`, tilt, time constants, drags, noise) comes from `DroneReduced.prefab` and `VelocityControl.cs`.
  `params` shows where every value came from. Tests flown before the per-run record need `swarmParams` in
  their `test.json`, because the scene may have changed since. internal_2 and internal_3 have them. `sim`
  warns when the resolved core flag disagrees with the run's own `shape.csv`.
- Variants (`sim`, `step`): `as flown` / `scene` (no overrides), `core off`, `core absolute`, `core relative`,
  `no damper`, `damper`, `core relative + damper`, `core absolute, no damper`, or `LABEL:field=value,...`.
  Presets that mean "as shipped" read the C# defaults, so they follow a retune.
- **`validate`** is the replica's gate against Unity. It flies every recorded flight of the given bench runs
  again -- same start, same waypoint pilot (holds included), same city (the run's `layout.json`, else the
  scene's city file) -- and reports per run the centroid track error as a fraction of the path, the
  nearest-neighbour spacing error, losses and building-contact events beside the bench's own, and the spread
  steps' t90. Limits: track ≤ 10% (median), spacing ≤ 15% (median), losses within 2 per 10 flights, t90 within
  1.5 s. It flies **`ContactSim`** (below) unless `--buildings kill`. Fly the runs with
  `bench.py config --record-all` so every flight has a trajectory.
- **`sim`** and **`forces`** take each run's city from its session: a ProceduralCityWorld run's own layout
  (`City.for_run(goals, city, layout)`), else the test's tile city with its goal patches swapped in.

### It goes stale: `FORCE_LAW_COMMIT`

The replica reads every *number* from the project, but the *structure* of the force law is written out in
`swarm_replica.py`. It mirrors `OlfatiSaber.cs`, `SwarmAlgorithm.cs`, `SwarmPlaneController.cs` (core
state), `VelocityControl.cs` and `StateFinder.cs` **as of `FORCE_LAW_COMMIT` (fe3a448b)**. `sim`, `forces`
and `step` check git for commits or uncommitted edits to those files since then (`params` reports it), and
if they find any they **refuse (exit 3)** unless `--allow-stale`. Review the diff, then bring the replica into line with any change to the
horizontal law and bump the constant. A change that leaves the law alone (yaw, altitude, plane mode) needs
only the bump.

### What the replica does not model

- **Heading/yaw.** There is no hull heading rule, no look-direction gap fill and no FPV view. Heading
  does not feed translation, so kills and spacing are unaffected, but nothing about what the pilot sees.
- **Altitude.** There is no climb, ceiling, MinHeight floor or plane mode. In `sim` every drone is within
  every building's height. `forces` uses recorded 3D positions, so its obstacle query does see altitude.
- **Street furniture.** Obstacles are the Obstacle-layer buildings plus each goal patch's nine buildings.
  Lamps and props on Default are invisible to the swarm in Unity too, but here they cannot be hit either.
  internal_3's street-light crash cannot happen in the replica.
- **PhysX contact, in `sim`, `forces` and `step`.** There (`Sim`) a drone dies on entering a building
  footprint (+0.25 m) or coming within 0.5 m of another drone. **`ContactSim`**, which `validate` and the density
  study (`citysim`) fly, turns a footprint entry into a contact as PhysX resolves it: the drone is put back on
  the surface with its inward velocity removed and Coulomb friction (μ 0.6, PhysX's default material, which
  every collider here has) taken off its slide; contacts are counted as events, as the bench's
  `OnCollisionEnter` counts them; and a drone whose nearest live neighbour is beyond the cohesion cutoff
  (`r0_coh × S`) is lost, as `DroneHealthMonitor.IsTooFarFromSwarm` loses it. Measured on the 2026-10-12
  validation runs (17 flights each in DiamondCityWorld and two generated cities,
  `experiment_results/density_study/replica_validation/`): with buildings as kills the dense city (coverage
  0.17) failed the gate, the replica losing 18 drones to Unity's 5 with spacing 20% off; as contacts without
  friction the losses matched (5 and 5) but spacing was 15.1% off; with friction every run passed (losses 3
  against 5, contacts 16 against 23, spacing 10% off, track 3% of the path). `sim` keeps `Sim`, so its
  internal_2/internal_3 figures above are unchanged.
- **DroneHealthMonitor.** There are no `Crashed` (flip) or stuck rules, and `Sim` has no `TooFarFromSwarm`.
- **Shield timing.** The shield uses this tick's obstacle frames; Unity's may be a tick stale.

## Files

- `SwarmBenchRunner.cs` is the runtime harness. It is inert unless `SWARM_BENCH_CONFIG` is set, so it costs
  nothing in the editor or a build.
- `Editor/SwarmBenchLauncher.cs` holds the batchmode entry points: `Run` opens the config's scene and
  enters play mode; `CompileCheck`.
- `run_bench.ps1`: sync, Library seeding, ilpp.pid hygiene, one-Unity-per-copy guard, batchmode with
  timeout and retry, report.
- `bench.py` has the `scenarios` (`--goals`: a layout's goal flights), `config` and `report` subcommands.
- `swarm_params.py` resolves parameters from the C#, scene, prefab and experiment records. Run it directly
  to print a scene's values.
- `swarm_replica.py` has the `sim`, `forces`, `step`, `validate` and `params` subcommands, and `ContactSim`.
- `scenarios/` holds the flight sets.
