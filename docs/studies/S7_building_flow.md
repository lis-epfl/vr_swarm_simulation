# S7 Building Flow

**Code name:** the flow study · **Dates:** 2026-10-30 · **Status:** done; three switches on `SwarmManager`, all off in
the scenes, the slide recommended · **Builds on:** S6 (T3, its harness, measures and Unity gate), S5 (prism buildings in
the replica)

**Question.** After S6, three limits remained in how the Olfati-Saber law treats BlockShapedPatchCityWorld's buildings: a
drone flown at a building stalls against it; a drone left behind a building is dragged through its facade by cohesion;
and one spread cannot suit both the narrow streets and the buildings the swarm should surround. The user chose three
fixes from the follow-up recommendations, asked for each behind its own Inspector switch, faithful to Olfati-Saber (no
path finding), tuned and tested in this scene and its replicas: **slide round buildings**, **route stragglers round
buildings**, **adaptive spread**.

**Design.** The three in the C# and mirrored in the replica, each off = the law as before (24 stored S6 flights re-flown
bit for bit). S6's harness, flights and measures (cohesive speed the headline), on settled starts. Six screens in the
scene's city (52–104 flights per variant) shaped the rules; a rule fixed before the confirmation (`rule.json`: safe =
losses, contacts and ending in pieces no worse than T3's within margins; helps = cohesive speed +5%, streets ≥ 98%,
surrounding within 10 points); the confirmation flew all eight on/off combinations and one drone with and without the
slide, 360 fresh flights (~100 km) each, in the scene's city and S6's three held-out ones; then the Unity bench, 64
flights per combination, scored beside replica twins. A short tuning of the adaptive spread followed on fresh routes.
Manual: [docs/swarm_flow/README.md](../swarm_flow/README.md).

**Findings.**
- **Slide round buildings is the rule's pick.** Turning the part of the stick the shield removes along the building
  (head-on approaches only), with the formation splitting in halves round a building the swarm flies squarely at and
  each drone going its own way elsewhere: cohesive speed 4.98 against 4.10 m/s (+21%; ahead in all four cities by
  20–22%), progress +14%, a building passed in 10.6 s instead of 12.6, one group 92% of the time instead of 86%, no
  flight ending in pieces (4% for T3), contacts −15%, losses unchanged (0.04 per km); it surrounds 90% of the buildings
  flown at instead of 98%. One drone shares the shield and slides too: 5.70 against 4.87 m/s. In Unity (64 flights): 23% faster while together (4.83 against 3.94 m/s), 63% fewer contacts, 2 colliding pairs against 1; one drone 5.62 against 4.80 m/s.
- **The way round decided everything.** Splitting the formation in halves everywhere (the suggestion as written)
  shattered the street flights (one group 37%): at a corner the trailing half went round the far side of the corner
  building. Always going round on the swarm's side held it together and stopped surrounding (54%).
- **Route stragglers cuts contacts and nothing else**: −28% alone (0.28 against 0.39 per km), speed, cohesion and
  surrounding unchanged; with the slide, contacts fall to 0.14 per km (−64%). It needs two limits: attraction only (turning
  cohesion's whole push into the building also turned the drones' repulsion of each other, and they collided), and only
  the excess over 0.7 of the obstacle field's ceiling (turning all of the hidden pull left nothing holding a surround's
  two halves together). With the slide its losses rose to 0.12 per km, past the rule's bar (6 pairs against 2 in
  ~100 km; not significant).
- **The adaptive spread helps the streets and costs drones.** Narrowing to 0.8 of d_ref among buildings: the streets'
  cohesive speed +14% alone, +23% with the slide (5.38 against 4.38 m/s), but 0.18 drones lost per km against 0.04 (pairs
  squeezed together by two buildings' fields in a tight gap, mostly while the formation narrows or widens; a stronger
  damper did not help) and it reaches some buildings still narrow (95%, with the slide 90%). In Unity it lost 8 drones in 64 flights against T3's 2 (with route 10; with the slide 3, all three 2). A gentler setting (0.85 of d_ref, a 3 s filter), tuned on fresh routes afterwards, kept the street gain and lost 4 drones alone in Unity, so the switch now ships that way; with the slide the replica still lost about twice as many as the slide alone.

**For the trials.** Tick **Slide Round Buildings** on BlockShapedPatchCityWorld's SwarmManager. It applies to both
conditions (the single drone shares the shield), which keeps the comparison fair: both get faster. Route Stragglers is
an optional extra if building contacts matter. Leave Adaptive Spread off until the avoidance shape is tightened (the
circles round each building's bounds squeeze the streets; first set's 3).

**Limits and refinements.** Scripted pilots at full stick, not people; losses are rare events and differences of a
tenth of a drone per km sit near the noise of ~100 km per setting; S7 measures movement and surrounding, not what the
screens show. The replica's force-law guard (`FORCE_LAW_COMMIT`) needs bumping once the C# is committed. Lessons 40–42.

**Where.**
- Unity: `swarm/SwarmManager.cs` (the switches), `OlfatiSaber.cs` (`SlideRoundObstacles`, `FirstBlockingFrame`,
  `RouteBlockedPull`, `EffectiveDRef`), `SwarmPlaneController.cs` (`UpdateSpreadScale`), `SwarmAlgorithm.cs` (pushes,
  `SharedWorldStick`), `ExperimentRecorder.cs` (`spreadScale` in shape.csv).
- Replica: `swarm/Bench/swarm_replica.py` (`slide_round`, `route_round`, `aim_miss`, `spread_target`).
- Code: `citysim/swarmflow.py` (screens, confirmation, rule, bench plan and gate), `swarmflow_replays.py`,
  `swarmflow_report.py` (+ `.html`).
- Outputs: `experiment_results/swarm_flow/` (`rule.json`, `screen1/`–`screen6/`, `confirm/` with `verdicts.json`,
  `tune2/`, `bench_flow/`, `bench_single/`, the candidate files). Unity runs: `D:\claude_obstacle_bench\runs\swarmflow_*`.
- Page: [S7 Building Flow](https://claude.ai/artifact/UkwGfhkWqnBAqF4iuBiz9a) (`python -B -m citysim.swarmflow_replays`, then
  `python -B -m citysim.swarmflow_report`, then publish `experiment_results/swarm_flow/swarm_flow_bare.html` with the
  files `swarm_flow_files.json` lists): the answer first, the cities and replayed flights (five routes, seven ways, Unity
  beside the replica), every table, the terms.
