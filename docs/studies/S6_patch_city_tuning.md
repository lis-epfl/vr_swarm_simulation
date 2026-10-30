# S6 Patch City Tuning

**Code name:** the patch-city tuning · **Dates:** 2026-10-30 · **Status:** done; applied to the scene ·
**Builds on:** S5 (the replica on prisms, the spread findings), BlockShapedPatchCityWorld (2026-10-29)

**Question.** In BlockShapedPatchCityWorld, which spread (d_ref) suits the buildings, and which obstacle-avoidance and
swarm gains let the formation surround a building it meets and re-form beyond it quickly, without fragmenting? d_ref is
held constant through a flight. The user asked for the metrics as well as the values.

**Design.** The replica on the scene's city (Unity's avoidance of prisms, PhysX-like contacts) under the bench's
waypoint pilot at full stick: straight lines across the city (the formation meets building after building), S4's pass
round a block, and street routes. Measures: progress speed, the share of the flight as one group (S4's 30 m free-space
link), their product the cohesive speed (the headline), fragment episodes (time to re-form), buildings surrounded,
per-building pass and split times, losses, contacts, hull share, spread steps. Screens on the scene's city (a baseline
over d_ref, one-at-a-time sweeps, three grids), then a confirmation on fresh routes in it and three held-out cities of
the same design (~104 km per candidate) by a rule fixed before the sweep's results were read, then the Unity bench.
Manual: [docs/swarm_tune/README.md](../swarm_tune/README.md).

**Findings.**
- **The scene as it was lost drones at its own spread**: at d_ref 1.08, 1.86 drones per km in the replica and 1.72 in
  Unity (32 in 64 flights), drone-drone collisions in streets the avoidance circles narrow, and it surrounded only
  half the buildings it met. Losses stop from about 1.4.
- **The formation's ring at d_ref 2.0 (31 m) is as wide as a typical building's avoidance cylinder (31 m)**: it closes
  round a building without stretching the lattice, surrounding 94–98% of the buildings met.
- **Speed through the city was capped by the shield, not the swarm**: flown straight at a building a drone stalls (one
  drone 4.0 m/s, 43% of top speed, its longest stall 9–10 s a flight). A shorter shield (`d_shield` 1.6 instead of 2.5) is the
  speed lever; below ~1.6 drones arrive too fast and collide. A thinner obstacle field (`d_obs` 0.16 instead of 0.4)
  keeps the formation together round buildings and in narrowed streets. Cohesion (`a`) trades fragmentation against
  contacts and stays 0.8. Nothing else moved the measures.
- **The pick (T3: d_ref 2.0, `d_shield` 1.6, `d_obs` 0.16)**, by the rule's fallback since no candidate met every safety
  bar (all exceeded 0.1 contacts per km). Confirmation, against the scene's gains at 2.0: cohesive speed 4.03 against
  3.34 m/s (+21%), progress 4.77 against 3.81 (+25%), a building passed in 12.5 s instead of 15.6, 96% surrounded,
  losses 0.02 against 0.06 per km, contacts 0.37 against 0.15; ahead in all four cities. **Unity** (64 flights per set):
  3.94 against 3.10 (+27%), 4.74 against 3.76, 98% surrounded, losses equal (one pair), contacts 0.54 against 0.22 per
  km (none cost a drone). The replica's twins matched Unity on every measure.
- **The obstacle gains are shared with the single-drone condition and help it more**: 4.03 → 4.96 m/s, its longest stall
  in a flight 10.3 → 2.9 s (Unity: 3.96 → 4.93, no contacts). The swarm and one drone are now about equally quick on these
  routes.
- **The first screens were biased** by starting each flight from an unsettled formation (lesson 39); they are kept as a
  record and not used.

**For the trials.** The scene now carries T3. Launch with `.\launch_sim_scripts.ps1 -SpreadRange 1.4,2.6` so the RC's
spread dial (which overrides `SwarmManager.d_ref` every frame) is centred on 2.0. If building contacts matter, T4
(`a` 0.72) brings them back to the old rate for ~4% of cohesive speed (not flown in Unity).

**d_ref note.** A spread narrowing to ~1.6 in the streets and widening to ~2.0 at buildings would beat any constant one
(street cohesive speed 5.1–5.3 against 3.4–4.4 m/s), but at 1.6 drones still collide in the streets (0.43 per km);
untested.

**Limits and refinements.** Scripted straight-line and street pilots, not people; the replica is pessimistic about
collisions at small spreads in this city (Unity lost fewer); one Unity run of 64 flights per set in the scene's own city.
The remaining contacts are stragglers pulled through a facade by cohesion, which the force law does not shield
(`RemoveInwardComponent` covers the stick and the damper only): a force-law change, untested. Lessons 39, 11.

**Where.**
- Code: `citysim/swarmtune.py` (the flights, the measures, the screens, the confirmation, spread steps, the bench plan and
  gate, and `apply`, which writes a candidate into the scene); `launch_sim_scripts.ps1 -SpreadRange`.
- Outputs: `experiment_results/swarm_tune/` (`rule.json`, `baseline2/`, `sweep2/`, `grid2/`–`grid4/`, `confirm/` with
  `confirm_cands.json`, `steps.csv`, `chosen.json`, `bench_base/`, `bench_tuned/`, `bench_single/`; the biased first
  screens `baseline/`, `sweep1/`, `search1/`, `grid1/`). Unity runs: `D:\claude_obstacle_bench\runs\swarmtune_*`.
- Page: [S6 Patch City Tuning](https://claude.ai/artifact/EXW3359CjhLNfHg6SpaBvF) (`python -B -m citysim.swarmtune_replays` (the Cities tab's replays: five bench routes flown five ways, Unity's flights and the replica's twins, each twin checked identical to the gate) and `python -B -m citysim.swarmtune_report`, then publish `experiment_results/swarm_tune/swarm_tune_bare.html` with the files `swarm_tune_files.json` lists). Answer first (five questions, one chart each, my reading), then the cities (the scene's and the three held out) with a drawing of what the settings mean on one building, five routes replayed every way in Unity and the replica, every table, the terms.
