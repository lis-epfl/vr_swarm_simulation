# S9 Street Spread

**Code name:** the spread study · **Dates:** 2026-11-02 (a short study, ~70 minutes, and a follow-up the same day) ·
**Status:** done; the scene unchanged (C35a). The adaptive spread's changes cost nothing; the street spread is the
choice (2.0 held for the fewest losses in the replica, 1.7–1.8 for 8–11% more street speed); Unity's losses unsettled ·
**Builds on:** S8 (its harness, its law and its two settings C35 / C35a, pooled), S7 (the adaptive spread), S6 (the
flights and measures)

**Question.** BlockShapedPatchCityWorld flies S8's pick, C35a: the enclosing circle, the slide, d_obs 0.35 and S7's
adaptive spread at 0.85 (d_ref 2.0 narrowed to 1.7 in the streets). It was the fastest setting in S8's streets and lost
more drones than the same law without it. The user asked whether the adaptive spread simply narrows too much and should
stay a bit wider, and for the fixed-d_ref checks the S7/S8 evaluation suggested: the narrowed spread held fixed (is the
cost the narrow spread or the narrowing?), and the pilot's dial at its ends (`EffectiveDRef` has no floor). Then, after
the first answer: is any of it significant, do the losses really happen while d_ref changes, and why do the replays
look alike?

**Design.** No change to the law or the scene. **First run** (`run1`): thirteen rows on the scene's law: d_ref 2.0
with the adaptive spread off (F20, S8's C35) or at street factors 0.85 / 0.90 / 0.95 (A85 = the scene, A90, A95); 1.9 /
1.8 / 1.7 held (F19, F18, F17); the dial at 1.4 held, with 0.85 and 0.95 (F14, L85, L95), 1.2 held (F12); at 2.6 held and
with 0.85 (F26, H85); 336 flights per row (~117 km); Unity six rows × 56. **Follow-up** (`run2`): F20, A85, A90, A95,
F18, F17, 1,100 flights per row (431 km), every loss timed against the adaptive spread's *shadow* (what it would be
doing, logged in every row, so held rows mark the same moments); Unity F20 / A85 / F17 × 100 street routes, with twins.
Losses counted as events (a colliding pair is one), exact intervals and tests; speeds paired route by route. A rule
fixed before the first run picked among the three factors. Manual: [docs/swarm_spread/README.md](../swarm_spread/README.md).

**Findings.**
- **Narrower is faster, beyond doubt**: route by route over 1,000 street routes, 0.95 / 0.90 / 0.85 are 2.7 / 5.3 / 8.0%
  faster in the streets than 2.0 held, 1.8 held 8.0% and 1.7 held 11.4%, in every city (p < 1e-15).
- **In the replica, narrower costs drones, held or adaptive alike**: 24 loss events at 2.0 held, 36 / 35 / 48 at the
  three factors, 43 at 1.8 held, 48 at 1.7 held (p = 0.003 for 0.85 and 1.7). The rule picked 0.85 on the first run's
  ~10 events a row; applied to the follow-up it finds no factor safe.
- **Unity does not reproduce it**: 12 / 9 / 10 flights with losses at 0.85 / 1.7 held / 2.0 held in 156 each, and it
  loses about twice what the replica's twins do on the same routes (which match its speeds and contacts). Unsettled.
- **The adaptive spread's changes cost nothing.** Losses cluster as the swarm leaves a building into a street in every
  row, including 2.0 held where the spread never changes (2.05 an hour then against 0.40 held narrow), and the adaptive
  spread's excess over 2.0 held is the same in those moments as elsewhere (×1.96 against ×2.18, p = 0.83). The adaptive
  0.85 flies like 1.8 held (the same street speed, 48 against 43 loss events); 1.7 held is 3% faster for the same losses.
- **The first run's two strong readings were wrong**: "held 1.7–2.0 lose alike" was noise at ~10 events a row, and
  "33 of 63 losses during a change" was the place, not the change. Its dial-end rows (1.4 lost the most, surrounding
  81%; 2.6 slow in the streets) were not re-flown.
- **The replays look alike because the effects are small per flight**: a few percent of speed, and a loss in 2–4% of
  replica flights (one Unity flight in 16).

**For the trials.** The choice is the street spread, not adaptive against held: 2.0 held (adaptive off) if the
replica's losses are real, about 8% slower in the streets; the scene as it is (≈ 1.8 held) or 1.7 held for the speed.
The first write-up's advice to hold 1.7–1.8 "to avoid the narrowing" is withdrawn. A Unity run of 2.0 held against the
scene's setting, ~300 street routes each, would settle the losses. The single-drone condition has no spread.

**Limits and refinements.** Scripted pilots at full stick with the dial fixed; the replica under-counts Unity's
collisions and the two disagree on which flights lose drones; the dial ends only in the first run. Lessons 45–47.

**Where.**
- Code: `citysim/swarmspread.py` (the runs, the rule, `SpreadEventLog` / `ShadowLog`, the analysis, the bench),
  `swarmspread_temporal.py` (the timing: phases, exposure, the tests; Unity rebuilt), `swarmspread_replays.py`,
  `swarmspread_report.py` (+ `.html`).
- Outputs: `experiment_results/swarm_spread/` (`rule.json`, `run_cands.json`, `run2_cands.json`, `run1/` and `run2/`
  with `analysis.json`, `run2/temporal.json`, `unity_temporal.json`, `bench_spread/`, `bench_spread2/`, `replay/`).
  Unity runs: `D:\claude_obstacle_bench\runs\swarmspread_bench_spread`, `swarmspread_bench_spread2`.
- Page: [S9 Street Spread](https://claude.ai/artifact/3tbJxGdDjKzmRVxqEyXmNx), with an "Is it significant?" tab
  (`python -B -m citysim.swarmspread_replays`, then `python -B -m citysim.swarmspread_report`, then publish
  `experiment_results/swarm_spread/swarm_spread_bare.html` with the files `swarm_spread_files.json` lists, root
  `experiment_results/swarm_spread`).
