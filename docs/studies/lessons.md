# Lessons carried forward

Refinements the studies identified, for whoever designs the next one. Each says what went wrong or what to watch,
what to do instead, and the study it came from (S1 Goal Visibility, S2 Hidden Goal Search, S3 City Roaming). Add new
ones with the study's ID; mark one **Applied** once the code does it by default.

## Measures

1. **Check every measure for a ceiling before comparing conditions with it.** S3's sweep width only counted ground
   within 42 m of the track (`transit.sweep_width` bins to the range plus one bin), so it cannot exceed 84 m; the swarm
   reached 82.5 m at spread 1.0 and the measure could not rank 1.0 against 1.6, and understated the swarm 6–20% against
   the single drone. A wider formation's outer cameras see past that window. **Applied in S3's code**
   (`clutter.SWEEP_WINDOW_M` 80 m via `transit.score(sweep_window=…)`); S1's capability numbers still use 42 m.
   Ground per metre has no ceiling and is the better spread comparison. (S3, also affects S1)
2. **A probability within a fixed horizon saturates; compare rates or times instead.** "Found within 5 minutes" ran to
   50% (single) and 64% (swarm) in open cities, compressing the ratio to 1.27× against 1.48× for the find rate. Use
   time-to-event (Kaplan-Meier, restricted mean time) with a horizon long enough to resolve the curve. (S3; S2 used
   restricted mean times)
3. **A target anywhere in a small city is mostly found by the view from the start.** At 200 m in a 730 m arena the
   swarm saw a third of the open ground in its first minute; the implied find rate halved over five minutes and the
   speed-up fell from 1.9× to 1.5×. Place targets out of sight of the start (S2 put starts ≥ 260 m from the goal) or
   condition on that, and measure time directly rather than extrapolating a constant rate. (S3)
4. **Range and attention assumptions set the effect size; report them as dimensions, not footnotes.** S3's speed-up
   was 1.88× at 40 m, 1.48× at 200 m and 1.19× counting only the screens straight ahead; S2's shrank to 0.81–1.03 in
   the fitted head's view; S1's discovery was 0.58–0.69 on shown feeds and 0.71–0.84 on the headset basis. Measure at
   40, 80 (the trials' discovery range) and 200 m, on any-feed, head-ahead and internal_3's fitted head
   (`pilots.HeadProcess("fit")`). (S1, S2, S3)
5. **A single glimpse is a generous detection.** Every study counts a target seen the first time it is in a feed's
   view. That ignores whether neighbouring screens connect (S3's overlap limit) and how long a target stays on
   screen. Consider a dwell requirement (e.g. ≥ 1 s on screen) or a detection probability. (S3)
6. **Separate the goal from the transit.** At a goal, its geometry decides who sees more (S1: clutter barely moves
   it); in transit and discovery, building coverage does (S3). The two can move in opposite directions, so model and
   pre-register them as separate outcomes. (S1, S3)
7. **Block size matters far less than coverage** for the swarm-vs-single comparison (−4.7% per doubling against
   +7.9% per 0.01 of coverage): spend a study's budget on coverage, replicates and longer runs, not on block sizes.
   Small blocks at high coverage do penalise wide spreads. (S3)

## Models and validity

8. **Scripted pilots are not people.** In internal_3's own city people found goals 2.3× (swarm) and 10× (single
   drone) faster than S2's fitted pilot, and real single-drone pilots saw the target 2.7× longer than S1's scripted
   policies. Ratios from scripted pilots describe undirected behaviour; expect the human swarm-vs-single difference to
   be smaller, and calibrate against a pilot block. (S1, S2)
9. **Fly pilot paths, not the street route.** The street route made transits 1.35–1.64× the straight line, where
   internal_3's pilots flew 1.06–1.12×, and hid the swarm's transit advantage. **Applied** (`citysim.routing`). (S1)
10. **Spread 0.4 loses drones in every model** (replica and Unity: 5–7 per flight in dense cities) and is worse on
    every visibility measure (S3: 6–13%). It drove S1's L3 failure. Model the losses, report per spread, and consider
    flooring the trials' spread dial near 1.0. (S1, S2, S3)
11. **Validate the replica in each new kind of city before trusting absolute numbers.** It passed Unity's gate in
    S1's cities (with contact friction), failed on spacing and losses in packed pilot-path cities, and has not been
    flown in S3's mixed-building Voronoi cities. Run a Unity spot check of the cities a decision rests on. (S1, S3)
12. **The replica is 2-D**: every building blocks every sight line and nothing is lower than the drones; there is no
    street furniture, wind or attention. (S1, S2, S3)

## Statistics

13. **Cities are the independent units.** Walks or searches in one city share its buildings; tests over walks
    overstate significance (S3's Wilcoxon over walks against its t test over city means). Average per city, then test
    over cities; with few cities per design, lean on trends fitted over all cities. (S3; S2 bootstrapped over goals)
14. **Pair everything.** Every condition flies the same intent chain from the same start and seed, so each comparison
    is within-scenario. (S2, S3)
15. **Fix the selection rule before the data**, record near-misses (S1's L3 knife-edge between G6 and G7), and rerun
    the rule rather than hand-editing its output. (S1)

## Running studies on this machine

16. **The CPU faults** (i9-13900K, microcode 0x113; `docs/cpu_instability_13900k.md`): pool workers die silently and
    single-threaded numpy at full boost throws impossible errors. Use `citysim.parallel.run_jobs` (stall detection,
    reruns), retry each flight on an exception, re-fly a sample bit for bit at the end, and pin single-threaded steps
    to the E-cores. (S1, S2, S3)
17. **Make long runs resumable and stoppable**: append each result as it arrives, order jobs replicate by replicate so
    a time budget stops with every condition equally covered, and keep the system awake for the process's life.
    (S3: `clutter.cmd_clutter`)
18. **Never edit modules a running pool imports**: every new pool re-imports them from disk. Keep statistics and pages
    in modules only the parent imports. (S3)
19. **Take subsets of scenarios by index, never by drawing fewer**: scenario start points, headings and seeds come off
    one random stream. (S2)
20. **Whole-city grids**: build them footprint by footprint (`clutter.arena_cells`); `transit.Cells`' all-pairs test
    needs gigabytes per worker over a whole arena. Goal-free cities: `citygen` `goals=0`. (S3)
