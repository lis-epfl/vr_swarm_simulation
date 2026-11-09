# S9 Street Spread

**Code name:** the spread study · **Dates:** 2026-11-02 (a short study, ~70 minutes, a follow-up and a settling run in
Unity, all the same day) · **Status:** done; the scene unchanged (C35a), confirmed in Unity. Narrower in the streets is
faster, and in Unity it costs no drones (the replica's own doubling is not its law); the adaptive spread's changes cost
nothing · **Builds on:** S8 (its harness, its law and its two settings C35 / C35a, pooled), S7 (the adaptive spread), S6
(the flights and measures)

**Question.** BlockShapedPatchCityWorld flies S8's pick, C35a: the enclosing circle, the slide, d_obs 0.35 and S7's
adaptive spread at 0.85 (d_ref 2.0 narrowed to 1.7 in the streets). It was the fastest setting in S8's streets and lost
more drones than the same law without it. The user asked whether the adaptive spread simply narrows too much and should
stay a bit wider, and for the fixed-d_ref checks the S7/S8 evaluation suggested: the narrowed spread held fixed (is the
cost the narrow spread or the narrowing?), and the pilot's dial at its ends (`EffectiveDRef` has no floor). Then, after
the first answer: is any of it significant, do the losses really happen while d_ref changes, and why do the replays
look alike? And finally: does the narrower street spread cost drones in Unity?

**Design.** No change to the law or the scene. **First run** (`run1`): thirteen rows on the scene's law: d_ref 2.0
with the adaptive spread off (F20, S8's C35) or at street factors 0.85 / 0.90 / 0.95 (A85 = the scene, A90, A95); 1.9 /
1.8 / 1.7 held (F19, F18, F17); the dial at 1.4 held, with 0.85 and 0.95 (F14, L85, L95), 1.2 held (F12); at 2.6 held and
with 0.85 (F26, H85); 336 flights per row (~117 km); Unity six rows × 56. **Follow-up** (`run2`): F20, A85, A90, A95,
F18, F17, 1,100 flights per row (431 km), every loss timed against the adaptive spread's *shadow* (what it would be
doing, logged in every row, so held rows mark the same moments); Unity F20 / A85 / F17 × 100 street routes, with twins.
**Settling run** (`bench_spread3`): Unity F20 and A85 × 300 fresh street routes, with twins, judged by a rule written
before it flew (`settle_rule.json`: the 95% interval of the loss ratio over all three benches' street routes, against
1 and against the replica's 2). Losses counted as events (a colliding pair is one), exact intervals and tests; speeds
paired route by route. Manual: [docs/swarm_spread/README.md](../swarm_spread/README.md).

**Findings.**
- **Narrower is faster, beyond doubt**: route by route over 1,000 replica street routes, 0.95 / 0.90 / 0.85 are 2.7 /
  5.3 / 8.0% faster in the streets than 2.0 held, 1.8 held 8.0% and 1.7 held 11.4%, in every city (p < 1e-15); in Unity,
  over 300 fresh street routes, 0.85 is 8.1% faster (faster on 84%).
- **In Unity the narrower street spread costs no drones.** The replica's own flights lose about twice as many at the
  narrower spread, held or adaptive (24 loss events at 2.0 held, 36 / 35 / 48 at the three factors, 43 at 1.8 held, 48
  at 1.7 held; p = 0.003 for 0.85 and 1.7). The settling run found 18 against 18 in Unity; over all three benches 30
  against 28, ×1.07 (95%: 0.62–1.86): the doubling is ruled out, a rise up to ~×1.9 is not.
- **The replica's twins side with Unity** (17 against 16 flights with losses, flown from Unity's starts), so the
  doubling lies in the replica's own flights, not the law: its 2.0-held flights lose fewer than its twins of Unity's
  (p = 0.12), in the scene's city as in the others. Unity loses ~1.75× its twins in both rows alike.
- **In Unity the narrower spread touches buildings less**: 73 contacts against 119 (×0.61, p = 0.001).
- **The adaptive spread's changes cost nothing.** Losses cluster as the swarm enters a street in every row, including
  2.0 held where the spread never changes (replica: 2.05 an hour then against 0.40 held narrow), and the adaptive
  spread's excess is the same in those moments as elsewhere (replica ×1.96 against ×2.18, p = 0.83; Unity ×1.25 against
  ×1.02, p = 0.71, with 57% of its losses in the first 4 s in a street, 36% of the time). The adaptive 0.85 flies like
  1.8 held (the same street speed).
- **The first run's two strong readings were wrong**: "held 1.7–2.0 lose alike" was noise at ~10 events a row, and
  "33 of 63 losses during a change" was the place, not the change. Its dial-end rows (1.4 lost the most, surrounding
  81%; 2.6 slow in the streets) were not re-flown.
- **The replays look alike because the effects are small per flight**: a few percent of speed, and a loss in 2–4% of
  replica flights (one Unity flight in 16).

**For the trials.** Keep the scene as it is (C35a, the adaptive spread at 0.85): in Unity it is 8% faster in the streets
than 2.0 held, touches buildings 39% less and loses no more drones. 1.7 held is about 3% faster again in the replica but
was flown only 156 times in Unity, too few to clear it. The follow-up's caution (2.0 held for the fewest losses) is
withdrawn with the replica's doubling, as the first write-up's advice to hold 1.7–1.8 was before it. The single-drone
condition has no spread.

**Limits and refinements.** Scripted pilots at full stick with the dial fixed; a loss rise smaller than ×1.9 is not
excluded in Unity; why the replica's own 2.0-held flights lose fewer than its twins (the start, most likely) was not
tested; the dial ends only in the first run. Lessons 45–48.

**Where.**
- Code: `citysim/swarmspread.py` (the runs, the rule, `SpreadEventLog` / `ShadowLog`, the analysis, the bench),
  `swarmspread_temporal.py` (the timing: phases, exposure, the tests; Unity rebuilt), `swarmspread_settle.py` (the
  settling run's rule and verdict), `swarmspread_replays.py`, `swarmspread_report.py` (+ `.html`).
- Outputs: `experiment_results/swarm_spread/` (`rule.json`, `settle_rule.json`, `run_cands.json`, `run2_cands.json`,
  `bench3_cands.json`, `run1/` and `run2/` with `analysis.json`, `run2/temporal.json`, `unity_temporal.json`,
  `settle.json`, `bench_spread/`, `bench_spread2/`, `bench_spread3/`, `replay/`). Unity runs:
  `D:\claude_obstacle_bench\runs\swarmspread_bench_spread`, `swarmspread_bench_spread2`, `swarmspread_bench_spread3`.
- Page: [S9 Street Spread](https://claude.ai/artifact/3tbJxGdDjKzmRVxqEyXmNx) (version 2, before the settling run; the
  settling run's version 3 is at [M6fGKgM1WvMVw3mRSusMwK](https://claude.ai/artifact/M6fGKgM1WvMVw3mRSusMwK), published from a second claude.ai
  account until the original can be republished). Rebuild: `python -B -m citysim.swarmspread_replays`, then
  `python -B -m citysim.swarmspread_report`, then publish `experiment_results/swarm_spread/swarm_spread_bare.html` with
  the files `swarm_spread_files.json` lists, root `experiment_results/swarm_spread`.
