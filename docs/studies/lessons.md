# Lessons carried forward

Refinements the studies identified, for whoever designs the next one. Each says what went wrong or what to watch,
what to do instead, and the study it came from (S1 Goal Visibility, S2 Hidden Goal Search, S3 City Roaming, S4 Split
and Surround). Add new
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
33. **Score a view by the decision it informs.** S4 counted a street as previewed once a feed had seen 60% of its
    first 80 m, which a feed at the corner can do from across the junction. A searcher at a corner needs to know
    whether the goal is at the building down the street. Scored on that (half the footpath in front of the building at
    the street's end seen), the street-bound swarm knew 41% of a block's exits at 40 m blocks, where the old measure
    said 71%, and 9% at 75 m blocks (26%). The surround's lead over a pause of the same length at the corner was 13
    points at Phase 2's own setting, where H1 had measured 6 on the old measure, and 42–49 points for a 12–20 s
    surround at a wider spread. Define the measure from the decision, keep the weaker ones beside it as levels, and
    check the old one against it before trusting a result built on it. (S4 Phase 3)
34. **Score each condition's view at its own decision.** S4's pass round is aimed by its decision, so its pilot
    decides on arrival at the corner, as the street-bound swarm's does. Phases 1 and 2 scored its preview at its aim
    point, after it had gone round the block, and credited it with 98% of the block's streets for no time. Scored where
    it decides, it knows what the street-bound swarm knows. By the next junction it has looked down about what that
    swarm has too (83% against 80% at 40 m blocks). (S4 Phase 3)
37. **Choose the sight range for what is being seen, and record it with the run.** Every S4 measure counted a feed as
    seeing within 80 m, a distance inherited from recognising a goal's hat, and it was then used for seeing down a street
    to the building at its end. On the smallest blocks half those buildings were over 89 m from the corner where the
    pilot chooses (154 m on the largest), so the range, not the formation, set most of the numbers, and the user judged
    it too short to mean anything. The study was flown again at 250 m (a recorded `sight_m` / `look_range` per run, the
    old runs reproducing bit for bit). The answer held, but every share changed by a factor of two to twenty. Before a
    study, say what each range stands for (recognising a person, seeing a building, a street's extent) and pick it for
    that; give the runs the range as a setting rather than a constant, so it can be revisited without forking the code.
    (S4)

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
    flown in S3's mixed-building Voronoi cities.
    - It passed in S4's block cities: variant R, 12 bench runs, 480 manoeuvres, track within 1%, spacing within 2%.
    - S4's Phase 2 walks flown in Unity reproduced H1 (+6.6 points against +6.8) and every time-to-find ratio within
      0.3%. Unity lost three times the drones (29 against 9 in 128 swarm walks): agreement on what is seen does not
      carry over to the costs.

    Run a Unity spot check of the cities a decision rests on, and read its costs separately. (S1, S3, S4)
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
21. **Stopping a background shell may not stop the script it runs.** In S4 the inner bash outlived the stop, and when
    its wait ended it read the rewritten script at its old byte offset (a syntax error, luckily). Never rewrite a script
    that a running or stopped process may still be reading; write a new file. (S4)

## Manoeuvres and tuning

22. **A stronger long-range attraction can raise building contacts.** S4 raised cohesion's cutoff (`r0_coh` 20 → 40)
    so far-apart drones would stop repelling each other. The stronger pull at 30–60 m dragged a drone on one side of a
    block through the building toward the drones on the other: up to 0.3 contacts per manoeuvre for R and 1.7 for B,
    against 0.04 and 0.3 at 20. 30 kept 40's reform with 20's contacts. Tune long-range terms against contacts, not
    only against cohesion. (S4)
23. **Size a pass/fail run to its bar, then confirm fresh.** A Wilson 95% upper bound of 0.06 on a share with no
    events needs at least 61 trials, so S4's 50-trial screens could not pass safety by construction. Work out the
    trials each bar needs before screening, and confirm the chosen profile on fresh trials in fresh cities (S4's
    confirmation reproduced the screens exactly). (S4)
24. **Look past a censored recovery measure before reading it.** S4's reform (one cluster before the next junction)
    failed in 38–92% of passes at larger blocks. Flown on past the junction, almost every failure was one drone left
    round the far side, rejoining in under a second at p 50 and in 3–12 s or more at p 75. The bar was right; the
    diagnosis needed the uncensored time. (S4)
25. **A manoeuvre's timing scales with the geometry it works on.** The surround that previews a block's streets grew
    from about 5 s (p 40) to 9–14 s (p 50) and 20 s or more (p 60–75), while the time price capped it near 6–8 s. One
    length for every design passes only the small blocks. When the rules fix a timing globally, report that as a
    limit, and consider coupling it to the geometry the way the spread is. (S4)
26. **Screen a mitigation against the failure it might cause.** S4's face bend (meant to keep the closest-point
    obstacle model sliding along flat faces) raised contacts 2–5× and barely helped the head-on case it was for.
    (S4)
35. **Treat the separation (d_ref) as a factor, and size it to the task, not to enclosure.** S4 matched the spread to
    the block by envelopment (d* ≈ the ring's perimeter ÷ 100). Looking down a block's exits paid for more: with 18 m
    streets every block size gained from spreads past d*, up to where the drones start touching the building they
    surround (1.8 at 40–60 m blocks, 2.2 at 75 m; 84 of 85 located contacts at 2.6 were with that building, not the
    buildings across the street), and every one of those is wider than the trials' 0.4–1.6 dial. At 75 m blocks no safe spread
    enclosed the block reliably (67% at 2.2): ten drones are too few for that ring. Sweep d_ref with the other factors,
    record it on every flight (round the block and in the streets), and say so when the useful spread is beyond what a
    pilot can dial. (S4 Phase 3)

## Pilots and outcomes

27. **Seeing more is not finding sooner until a pilot uses what it sees.** S4's surround previewed 7.1 points more of
    a block's streets than a pause of the same length (H1, p ≈ 10⁻³⁴), yet an informed pilot acting on it found
    targets no sooner than the swarm kept in the streets (×1.02). Its knowledge saved 10% of the blocks searched, and
    the time each block took cost 21%. Measure the outcome a view is for, with a pilot that acts on what it has seen,
    and decompose the result into how many decisions and how long each took. Then a null tells you which side moved.
    (S4)
    - Phase 3 repeated it on the stricter measure. A surround looked down 71% of a block's exits at the decision
      against 41% in the streets, and the target stood at a building, where that should pay most. The surround still
      took ×1.48 the time: 3–6% fewer decisions, each 27–74% longer. A lead at the decision is worth little when the
      other condition gets the same view one block later by flying the exit it chose. (S4 Phase 3)
28. **Read the full run, not the first rounds.** S4's H2 ratio read ×1.12–1.39 on the first third of the targets and
    ×1.02 on all 288. Rounds are replicates, so an early read is a small sample, not a trend; report it, if at all,
    with its interval. (S4)
29. **Check an adaptive pilot for dead ends, and a walk for early ends, before the run.**
    - S4's informed pilot stopped early in 12 of its first 40 routes when its exit rule left no move (mostly where half
      the blocks were plazas). Each would have been scored as unfound at the cap. Give such a pilot an escape (any
      street, then back the way it came), and count the flights that end before their cap without a find.
    - Likewise, a walk that runs out of chain is charged for the rest of the horizon, since a flight that has ended
      sees nothing more. In S4 12% of walks ended at the arena's edge, which moved the ratios by at most 0.3%. Check
      end times against the horizon, or lay chains that cannot run out. (S4)

## Reporting

30. **Every study's page has a Cities tab: what the cities look like, and flights you can replay.** This is a
    standing requirement for every simulation study from S4 on, at the user's request (2026-10-28). Until then the
    pages showed what flights measured but not the cities or the flights, so a reader could not check that a
    manoeuvre or a pilot did what the text said. The tab has two parts.
    - **The cities.** Every design at one scale, any city whole, and one block (or goal) drawn to scale with what the
      design codes mean.
    - **The flights.** Representative flights of every condition, and Unity's of the same flights where the bench flew
      them. Each replica flight is re-flown from its job with a tape (`citysim.playback.Tape`) and kept only if its
      re-flown scores are identical to the run's, so what plays is what was scored.

    `citysim/playback.py` (recording and encoding) and `citysim/replay_view.js` (the player) are generic, and S4's
    `split_replays.py` and `split_report`'s Cities tab are the worked example. Choose the flights when the study is
    designed: a typical one per condition, and one of each failure the findings describe. Check each caption against
    its flight. **Applied** in S4's page. (S4)
31. **Define every code in plain words where a reader first meets it.** S4's page used `p 40 · s 18`, `R` and `B`
    throughout without saying what they meant, and the reviewer had to ask. Put a key to the codes under the page's
    masthead and a hover definition on each code wherever it appears, and keep a Terms tab with every term. In prose,
    give the plain form before the code ("40 m blocks with 18 m streets"). **Applied** in S4's page. (S4)
32. **Check a claim about why something happens against the events it is about.** S4 reported that variant B's drones
    touched the buildings across the street, a claim written from aggregate counts and the street's geometry. When the
    replays located every contact, 87–97% were with the building being passed or surrounded. Before a mechanism goes
    into the findings, locate its events (where, when, what was hit) in a replay or a table. (S4)
    - **The same applies to a counter.** In S4 Phase 3 the search pilot's `blind_decisions` (no exit with unseen
      footpath nearby) ran at about two a search for the surrounds against 0.25 for the street-bound swarm. That read
      as a pilot choosing at random once only small slivers were unseen, so the frontier rule was changed mid-run.
      The counter also counts the pilot's moves towards the nearest unexplored junction. Re-flying those searches with
      every decision logged showed no random choice at all, and the rule was changed back. Know what a counter counts
      before diagnosing with it, and log the decisions themselves when the diagnosis would change the design.
      (S4 Phase 3)
36. **A study's page opens on the question it serves and its answer, with the detail behind it.** Every study exists to
    design the human trials (swarm against one drone on a visual search), but S4's page had grown to twelve tabs in
    the order the work was done, each with its own codes, and the reader lost sight of why the study was run
    (the user's review, 2026-10-28). Its redesign is the pattern:
    - **The masthead** states what the study tests for the trials and gives a three-line answer.
    - **An Answer tab, shown first,** argues it in a few numbered questions in the order the argument needs them, each
      with one chart, a verdict and plain words (block sizes rather than design codes), then the author's reading:
      what the study settles, what it cannot, and what to take into the trial design. Every number is read from the
      runs (`split_report.answer_data`), never typed in.
    - **The comparison the purpose names** (here, against one drone) is on the page directly, even when the study's
      own hypotheses compared something else. S4's tested the surround against the swarm in the streets, and Phase 3
      had never flown one drone at all: `lookdown_A1` added it.
    - **The detail tabs move under one "All results" tab**, grouped by the question each phase asked, with the key to
      the codes beside them (lesson 31's key belongs where the codes are used). The Cities tab and Terms stay.
    **Applied** in S4's page. (S4)
38. **Check what a rule's fallback would pick before the run, and when nothing qualifies, look for the missing
    factor.** S5's step 1 picked the spread and surround by a rule fixed in advance, with a fallback ("the least unsafe")
    for when nothing met the safety bar. Nothing did, because the swarm lost drones in the streets themselves, and the
    fallback picked a surround that never enclosed the block. The cause was a factor the step held fixed (the street
    spread, at S4's 1.0): flown with one spread for streets and block, losses fell to none from 1.6. Write the fallback so
    it names a setting the later steps can use, and when the first run finds nothing, keep it and its pick on record, name
    the factor it missed, and fix the rule for the next run before reading that run. (S5)
