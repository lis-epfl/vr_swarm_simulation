# Swarm testing offline: the headless Unity bench and the 2D replica

Two tools for judging a change to the swarm before it reaches a headset. They answer different questions:

| | **Headless Unity bench** (`run_bench.ps1`) | **2D replica** (`swarm_replica.py`) |
|---|---|---|
| What flies | The real scene and code, PhysX and all, in batchmode Unity | A numpy model of the horizontal force law and control loop |
| Driven by | Scripted flights (`scenarios/*.json`): transits, tours, rams, zigzags, spread steps | A test's **recorded** pilot inputs (stick, body yaw, spread) |
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
- **Timing.** `timeScale` 20 runs 20x real time without changing the 0.02 s physics step. The runner sets
  `Time.maximumDeltaTime = 1` in `Start` and before every flight. `VrFramePacing` caps it at 0.1 s at scene
  load, which would quietly hold a 20x run to ~5 steps per frame. Each flight records `sim_per_wall`, and
  the report warns when it falls below half the time scale.
- Disabled in every run: `PyUniSharingFast`, `UDPReceiverManager`, `ImageSharing`, `ExperimentRecorder`,
  and every camera. A batchmode run beside the user's editor must open no named section, port or recording.

### Config (JSON, written by `bench.py config`)

`scene`, `timeScale`, `settleTime`, `altitude`, `capture` (m from a waypoint that counts as reaching it),
`pilotHeading`, `pilotYawRate`, `goalReplaySession`, `paramSets[] {label, names[], values[]}`, `scenarios[]`.
A scenario is `{kind, sx, sz, T, wps: [x, z, stick, timeout]*, record?, spread?: [t, d_ref]*}`. The swarm is
reset to `(sx, altitude, sz)`, settles, then flies toward each waypoint at `stick` (0–1), moving on when
within `capture` or after `timeout` s (0 = none). `spread` drives the spread stick piecewise-linearly, and
`record` writes every drone's XZ to `traj_<set>_<index>.csv`.

### Flight sets (`scenarios/`)

- `ScaledCityWorld_heldout.json` / `_spread.json` are **frozen** from the September obstacle retune's configs:
  120 held-out flights (30 transit, 10 tour, 50 ram, 30 zigzag) and 16 spread steps. Their generator was
  lost; keeping these exact keeps new numbers comparable with the CLAUDE.md figures (16 drones lost,
  0 contacts, 95% clean on the retuned gains, hollow core on).
- `DiamondCityWorld_*.json` were generated by `bench.py scenarios --city DiamondCityWorld --seed 1`. The
  generator rebuilds that structure from a city's `city_obstacles_<city>.json`: transits edge to edge, rams
  90 m either side of a building, and so on. Regenerating with another seed makes a different test, so
  results flown on the old file stop being comparable.

### Output (`<bench root>\runs\<tag>\`)

`config.json`, `meta.json` (live commit and uncommitted-file count at launch), `unity.log`, `traj_*.csv`, and
`results.jsonl`: one `start` line, one `params` line per set (overrides **and every effective SwarmManager
value**), one `flight` line per flight, then `done`. Errors write an `error` line and exit 2 (config) or 3
(runtime).

The flight fields are `alive_start/end`, `contacts` (Obstacle layer), `hard_contacts` (> 2 m/s normal) and
`drone_body_contacts`. `other_contacts`/`other_names` are colliders off the Obstacle layer: poles, props.
`deaths` is DroneHealthMonitor's reasons. Then `path_len`, `track_sum/n` (progress speed), `min_pair`,
`near_ticks`, `hull_frac`, `nn_mean`, `split_frac`, `mean_speed`, `sim_per_wall`, and the `gap_*`/`live_shown_*`
look-gap metrics with a pilot heading.

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
- **PhysX contact.** A drone dies on entering a building footprint (+0.25 m) or coming within 0.5 m of
  another drone. Bounces and friction pinning against a facade are not modelled; in Unity that pinning
  often ends as `TooFarFromSwarm`.
- **DroneHealthMonitor.** There are no `TooFarFromSwarm`, `Crashed` or stuck rules.
- **Shield timing.** The shield uses this tick's obstacle frames; Unity's may be a tick stale.

## Files

- `SwarmBenchRunner.cs` is the runtime harness. It is inert unless `SWARM_BENCH_CONFIG` is set, so it costs
  nothing in the editor or a build.
- `Editor/SwarmBenchLauncher.cs` holds the batchmode entry points: `Run` opens the config's scene and
  enters play mode; `CompileCheck`.
- `run_bench.ps1`: sync, Library seeding, ilpp.pid hygiene, one-Unity-per-copy guard, batchmode with
  timeout and retry, report.
- `bench.py` has the `scenarios`, `config` and `report` subcommands.
- `swarm_params.py` resolves parameters from the C#, scene, prefab and experiment records. Run it directly
  to print a scene's values.
- `swarm_replica.py` has the `sim`, `forces`, `step` and `params` subcommands.
- `scenarios/` holds the flight sets.
