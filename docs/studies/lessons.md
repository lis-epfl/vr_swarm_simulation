# Lessons carried forward

Refinements the studies identified, for whoever designs the next one. Each says what went wrong or what to watch,
what to do instead, and the study it came from (S1 Goal Visibility, S2 Hidden Goal Search, S3 City Roaming, S4 Split
and Surround, S5 Block-Shaped Surround, S6 Patch City Tuning, S7 Building Flow, S8 Footprint Avoidance, S9 Street
Spread, S10 Ellipse Avoidance). Add new ones with the study's ID; mark one **Applied** once the code does it by
default.

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
49. **Measure the question in its own terms, with a probe built for it, before reaching for the study's usual
    flights.** S10 asked whether a shape round each building keeps a drone at an even distance from every wall. The
    formation flights could not see it: 71% of their closest approaches to a facade were within half a metre of a
    corner, where every enclosing shape touches the footprint. A one-drone probe flown down a street parallel to one
    wall, 6, 9 or 12.5 m off it, answered it directly (the standoff's spread across walls ×0.55, Unity ×0.56) and showed
    the mechanism (the push off a wall correlates 0.96–0.99 with the shape's distance from that wall). When the question
    is about one interaction, isolate it in a probe, then fly the usual flights for what it costs. (S10)

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
    - In BlockShapedPatchCityWorld (S6), twins flown from Unity's settled starts matched Unity's speed, cohesion,
      surrounding and hull share within a few percent, and T3's contacts and losses exactly; at d_ref 1.2–1.6 under
      the old gains the replica lost more drones than Unity (22 against 10 in 52 flights). The direction of the
      collision error differs from city to city.
    - In the same scene under S8's law (S9), the twins matched Unity's street speed within 2% and its contacts, Unity
      lost about 1.75 times the twins' drones in both rows alike, and the twins did not reproduce the replica's own
      doubling (lesson 48).

    Run a Unity spot check of the cities a decision rests on, and read its costs separately. (S1, S3, S4, S6, S9)
12. **The replica is 2-D**: every building blocks every sight line and nothing is lower than the drones; there is no
    street furniture, wind or attention. (S1, S2, S3)
39. **Start a replica flight from a settled formation, as the bench does.** S6's first screens placed the open-sky
    hover formation straight onto a city junction. Drones overlapping a building were pushed out to its far side, and
    cohesion dragged them back through the facade in the first seconds: about 70% of the contacts at d_ref 1.8
    (0.87 → 0.25 per km once fixed), more the bigger the formation. So those screens favoured narrow spreads and any
    gain that shrinks the formation, and an apparent effect of the core's size was entirely the artifact. The bench
    settles 15 s at zero stick and counts from there; `swarmtune.fly` now does too, from a formation shrunk to fit the
    junction's streets. Before trusting a harness's contact or loss counts, locate the events in time (lesson 32): a
    cluster in the first seconds is the start, not the flight. (S6)
48. **When the replica and Unity disagree on a cost, ask the replica's twins before flying more replica.** A twin is the
    replica flown from Unity's own start on Unity's route, so it shares the force law with Unity and the start with
    Unity, and tells the law apart from the replica's own flights. S9's replica lost about twice the drones at the
    narrower street spread (48 against 24 loss events in 431 km); Unity did not (30 against 28 over 440 street routes a
    row, ×1.07, 95% 0.62–1.86); and the twins sided with Unity (17 against 16), as they already had on the first two
    benches (4 against 6). The doubling lay in the replica's own flights (its 2.0-held flights lost fewer than its twins
    of Unity's), and the follow-up's caution rested on it. Settle a cost that decides a scene setting in Unity, with a
    rule written before the run and the run sized so its interval can exclude the replica's effect: ~300 street routes a
    row for a doubling at ~15 loss events per 100 km, with the earlier benches pooled. (S9)

## Statistics

13. **Cities are the independent units.** Walks or searches in one city share its buildings; tests over walks
    overstate significance (S3's Wilcoxon over walks against its t test over city means). Average per city, then test
    over cities; with few cities per design, lean on trends fitted over all cities. (S3; S2 bootstrapped over goals)
14. **Pair everything.** Every condition flies the same intent chain from the same start and seed, so each comparison
    is within-scenario. (S2, S3)
15. **Fix the selection rule before the data**, record near-misses (S1's L3 knife-edge between G6 and G7), and rerun
    the rule rather than hand-editing its output. (S1)
46. **Count rare events with their exact intervals, and pool every run of the same setting before calling it safe.**
    Collisions run at about ten per ~100 km row and fall on different routes in every row, so pairing the routes buys no
    power for them and each count's 95% interval spans a factor of three. S9's rule, on its first run's counts, found the
    adaptive spread at 0.85 safe (14 loss events against 9, p = 0.20); the follow-up at four times the distance had 48
    against 24 (p = 0.003), and every replica run of the two settings together 68 against 34. Write the pooling of earlier
    runs of the same setting into the rule, count events (a colliding pair is one), and test a trend across ordered levels
    rather than one level against the base. **Pool within a platform, not across**: Unity's benches had the same two
    settings at 15 against 10 and lose about twice the replica's twins on the same routes, so a pooled replica-and-Unity
    count hides the disagreement that matters. (S9)

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
50. **A route filter's clearance test must leave out the thing the route is flown at, and a filter's knob should move
    its output the way you expect.** S10's angled run-ins took S8's clearance test (lines across the corridor clear of
    every footprint) and kept the target building in it. Square on that never matters; at an angle the lines beside the
    aim point run into the target's own face, so no 60° run-in qualified at all and 468 flights failed. The tell was that
    a narrower corridor qualified fewer faces than a wider one. Before running a filter, try its knob both ways and
    check the count moves the right way. (S10)

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
    - **Read what a censored measure counts before it decides anything.** S10's split time round a building flown at 60°
      put every ellipse row ~2.4 s behind the circle: one route in twelve never split within 40 s. Watched, the formation
      went round one side whole, one group throughout and past the building as fast as the circle, which split it. The
      measure counted keeping together as failing to split, so the rule judged the split on square-on flights, as S8's
      did, and the angled flights on time past the building. (S10)
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
40. **A rule that moves the formation as a whole must be gated on where the formation is going.** S7's slide splits the
    formation in halves round a building so it surrounds it. Applied to every building a drone's stick pointed at, it
    sent the trailing half of the formation round the far side of the corner building whenever a street route turned,
    because the new stick points across the corner from where they are, and the street flights fell apart (one group
    37%). Gated on the formation's own track (the centroid's stick line within 0.6 of the building's radius of its
    centre), with each drone taking its own, shorter side elsewhere, it kept the surround and the streets. Screen a
    formation-level rule on turning routes, not only on straight lines. (S7)
41. **Redirect only the part of a force that causes the failure, and find that part by attributing forces at the
    failure.** S7's first straggler routing turned all of cohesion's push into a building along it, which also turned
    the drones' repulsion of each other: neighbours beside a facade collided (2 drones lost per km). The force attribution
    at each collision (cohesion with and without routing, −4.5 m/s² apart becoming +2 together) showed it at once.
    Turning all of the pull towards hidden neighbours then removed what holds a surround's two halves together. Attraction
    to hidden neighbours only, and only beyond a share of the obstacle field's ceiling, kept the contacts down without
    either failure. (S7)
42. **A spread that changes in flight inherits the collisions of the narrowest spread it reaches, and adds its own in
    the transitions.** S7's adaptive spread narrowed 2.0 to 1.6 in the streets: +14–23% there, but 0.18 drones lost per
    km against 0.04 in the replica and 8 against 2 in 64 Unity flights, pairs squeezed by two buildings' fields in tight
    gaps, five of nine while the formation was narrowing or widening. Before a dynamic spread, check the static spread it
    narrows to in the same city, and locate its losses against the spread's changes. (S7)
43. **A tighter obstacle shape must keep its curvature.** S8 tried the building's footprint itself, Olfati-Saber's wall on
    each face, to give the streets back. Without S7's slide, a formation flown square at a wall split round it in 22 s
    against 6 (half never split within 40 s): the stall the user's first city had with bounding boxes, and the reason the
    cylinder was chosen. With the slide it moved, but drones sliding along a flat face settled in single file 1–2 m off
    it and ran into the one ahead, pinned in a pocket between two faces or slowed at a corner: 1.55 drones lost per km
    against 0.23, 21 times the contacts. A larger standoff only moved the lane out. The smallest circle round each
    footprint gave back most of the street (20 m median of 25 against 14) and kept the turning normal that spreads
    drones round a building. When a law's shape is chosen for one property (here, a sideways push), test the
    replacement for that property before anything else: S8 did so with a square-on flight from open ground. (S8)
    - **Carrying a rule into a new shape's coordinates can bring the stall back.** S10 aimed S7's rules by the ellipse in
      its unit space, where it is the unit circle, which is exact for the split, hidden and blocking tests. The slide's
      head-on gate, measured there as a line, fell to 0.52 at cap 1.5 and 0.25 at cap 2 where the stick meets a flat side
      square on: the stall again. Kept local (1 − F |t·u| / slideHeadOnFraction, exactly S7's gate on a circle), the
      ellipse split round a building as fast as the circle square on, and faster at d_ref 1.2. (S10)
51. **Pick by the measured effect, not by the size of the knob.** S10's plan picked "the closest fit, the largest cap".
    In the screen cap 2 fitted no closer than 1.5 (standoff spread ×0.45 against ×0.44 of the circle's; per building
    3.2 against 3.3 m) and has flatter sides. The rule, changed before it was written, picks the rows within 5% of the
    evenest measured standoff, then the roundest. A parameter that should improve an effect monotonically often
    saturates; let the rule read the effect. (S10)
44. **Bound a safety measure by the configuration the scene flies, and say so before the run.** S8's rule capped building
    contacts at the scene's law as it flies (S7's T3), not at S7's slide, which had already cut them; bounding by the
    slide would have ruled out any shape that lets drones nearer the facades, however much else it gains. The enclosing
    circle passed the speed and loss bars by a distance (a tenth of the losses) and failed only on contacts unless the
    standoff was raised; state the bound in rule.json and report every bar a candidate misses, so the reader can move the
    bar knowingly. (S8)
45. **Before tuning a setting that moves between values in flight, fly the values it moves between, held, and fly them
    far enough to compare their costs.** S7 and S8 tuned the adaptive spread's street factor without ever holding the
    spread it narrows to. S9 did, and the held rows answered the question the factor could not: 1.8 held flies exactly
    like the adaptive spread at 0.85 (the same street speed, p = 0.68; 43 against 48 loss events over 431 km), 1.7 held is
    3% faster for the same losses, and both lose about twice what 2.0 held does. So the cost was the narrower spread, not
    the dynamics, and a dynamic setting has to beat the best static value in its range. S9's first run (~10 loss events a
    row) read the held rows as losing alike and recommended them on that basis; the follow-up at four times the distance
    reversed it. Size the run to the comparison before recommending from it (lesson 23). (S9)
47. **A timing claim needs a placebo of the same moments.** S9's first run found the adaptive spread's losses clustered
    while the spread changed (33 of 63 against 23 expected from the time spent changing) and read it as the narrowing's
    doing. But the spread only changes near buildings, so the comparison was confounded with the place. The follow-up
    logged the adaptive spread's *shadow* in every row (what it would be doing, from the swarm's own position, stick and
    buildings): 2.0 held, whose spread never changes, lost five times as often in the shadow's "narrowing" moments as when
    it was held narrow, and the adaptive rows' excess over the held rows was the same in every phase (equal-ratio test
    p = 0.69). When a loss is timed against an event, time the control rows against the same event, computed but not
    applied, and count exposure per phase; a cluster that appears in every row is where, not what. (S9)

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
