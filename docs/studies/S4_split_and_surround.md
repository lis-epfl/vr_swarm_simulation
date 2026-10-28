# S4 Split and Surround

**Code name:** the split study · **Dates:** 2026-10-27 to 28 · **Status:** done (Phase 1 and Phase 2 in the replica,
both checked on the Unity bench); Phase 3, a short pilot on looking down the streets and on d_ref, in the replica, for
the user to review before another full study ·
**Builds on:** S1 (envelopment at a goal), S2 (morphing versus steering), S3 (coverage), the view-correspondence
report ([correspondence_metrics.md](correspondence_metrics.md))

**Question.** In a city of small Voronoi blocks, each holding one large building, can the swarm surround a block (S1's
goal behaviour: round the central building, looking outward), let the pilot see down every street round it, and
reform down the one chosen quickly enough to make block-by-block hopping an efficient way to explore? Does that find
things faster than the same swarm kept in the streets, or one drone, and what does it cost in drones and in views that
no longer relate?

**Design.** Two phases.

- **Phase 1, the initial test.**
  - **Cities:** 12 designs, block pitch p 40 / 50 / 60 / 75 m × street width s 12 / 18 / 24 m, written `p 40 · s 18`.
    Seed points are scattered one per p × p m (never closer than 0.8 p) and each seed's Voronoi cell is a block, so
    neighbouring blocks' centres are about p apart (41 m at p 40, 81 m at p 75). s is the street width, kerb to kerb,
    so a block is about p − s across. Half the interior blocks at p 40 · s 24 are too small for a building and are
    left as plazas.
  - **Two building variants on the same streets:**
    - *R*: one catalogue building per block, scaled so its avoidance cylinder fits inside the block. Today's force
      law, unchanged.
    - *B*: a block-shaped building, the kerb polygon inset 3.5 m, with a closest-point obstacle model.
  - **Manoeuvres,** flown in the replica and then on the Unity bench:
    - pass round (M3);
    - surround and exit down a side street (M3b);
    - surround hold (M4);
    - street run, spread steps, and a head-on face.
  - **Tuning:** the coupling of block size, street width and separation (d_ref), with cohesion's range raised
    (`r0_coh` 20 → 40, so far-apart neighbours stop pushing each other away).
  - Criteria fixed in advance pick the variant and the S4 profile.
- **Phase 2, the comparison.**
  - **Conditions:** `single`, `single_look`, `swarm_street`, `swarm_hold`, `swarm_pass` and `swarm_hop`.
  - **Pilots:** a paired block-hop chain and an informed frontier searcher.
  - **Primary outcomes:**
    - H1: the share of a block's streets previewed when the pilot commits, `swarm_hop` against the time-matched
      `swarm_hold`;
    - H2: informed-search time, `swarm_hop` against `swarm_street`.

The full design is in the manual ([docs/split_study/README.md](../split_study/README.md)).

**Findings, Phase 1** (the replica; 90,120 manoeuvres; [results page](https://claude.ai/artifact/C1PtAjZybB7JdAkoJy2xXD)):
- **Decision: variant R, with Phase 2 on p 40 with 18 and 24 m streets**, the only designs where both manoeuvres pass
  every criterion fixed in advance (100 fresh trials per design on fresh cities). R is safe at all 12 designs; B
  passes none.
- **The pass round is the efficient manoeuvre.** Aimed through the block at full stick, the formation flows round both
  sides and previews 88–99% of the block's streets with no hold. At p ≥ 50 with 12–18 m streets it usually leaves one
  drone round the far side, still apart when the next junction comes a few seconds later.
- **A brief surround pays only at p 40.** A 6 s surround (aim through, then turn off down the side street) previews
  80% of a block's streets in 85–90% of trials there. At p 50 that takes 9–14 s and at p 60–75 20 s or more, while
  flying round the block by the streets allows about 6–8 s before the surround costs more than 1.5× as long.
- **The coupling holds:** the separation that surrounds a block is its ring perimeter ÷ 100, within one 0.2 step for R
  at every design.
- **Raising cohesion's range trades contacts for reform** (`r0_coh` 20 → 40): the stronger pull across a block drags
  drones into the building, while at 20 a straggler rejoins more slowly. R's best is 30.
- **Block-shaped buildings (B) are touched**: nearly all of B's contacts (97% in the screens, 87% at its profile) are
  with the building the formation is passing or surrounding, pressed into its flat facade, and its longer ring is
  slower to surround. A face bend meant to help it slide made it worse. (Corrected 2026-10-28, when the replays located
  every contact: the first write-up said they were with the buildings across the street.)

**Findings, Phase 2** (the replica; 48 cities at p 40 with 18 and 24 m streets; 384 paired block-hop walks and 288
informed searches; manual section 7):
- **H1 holds: surrounding shows the pilot more.** A 6 s surround previews 95.5% of a block's streets when the pilot
  commits, 7.1 points more than a pause of the same length at the corner (t over 48 cities 33.0, p ≈ 2×10⁻³⁴).
- **H2 is not supported: it finds nothing sooner.** An informed pilot (the exit with the most unseen street within two
  edges per metre) finds a hidden target in the same time as the swarm kept in the streets: ×1.02 (0.92–1.12),
  Holm p 0.62. Equivalence within ±10% is not shown either.
  - The surround's knowledge saves 10% of the blocks searched, and each block takes 21% longer.
  - It beats spending the same time paused at the corner (×0.88).
- **The swarm's cameras are the effect, as in S2.** Kept in the streets, the swarm finds a target in 62% of one drone's
  time, and 11% sooner on random walks. One drone that stops to look down each street gains nothing.
- **The pass round gives the preview for free.** Flowing round the block on the way through previews 98% of its
  streets at no cost in time, but adds no speed to the search (×0.97).
- **The surround is the costly way to take a block.**
  - It loses about ten times the drones (0.29 against 0.03 per 10-minute walk, nearly all drone-drone collisions).
  - It touches buildings eight times as often.
  - A quarter of its screens have no neighbour sharing content, against 7% in the streets. Every view-correspondence
    measure ranks the conditions hold, street, pass, surround.
- **Checked in Unity.**
  - 160 of the walks, flown on the headless bench, reproduce H1 (+6.6 points against the replica's +6.8) and every
    time-to-find ratio within 0.3%.
  - 40 search routes re-flown there found their targets within 5 s of the replica.
  - Unity loses more drones than the replica (29 against 9 in 128 swarm walks), so the costs above are understated.

**Findings, Phase 3: looking down the streets** (a short pilot after the user's review, 2026-10-28; manual section 9).
The preview Phases 1 and 2 scored counts a street once a feed has seen the start of it. Phase 3 asks whether the
formation saw *down* each exit, to the footpath in front of the building at its end, and at which spread d_ref.
- **Part A** (one block per trial, every way of taking it at five block spreads, 4 designs at s 18, 6,600 flights):
  - **The old preview overstated what the street-bound swarm knows.** At the decision it had looked down 41% of a
    block's exits at p 40 (the old measure: 71%), 28% at p 50, 21% at p 60 and 9% at p 75.
  - **One drone looks down almost none** (added 2026-10-28 as `lookdown_A1`, the lone drone over Part A's 120
    blocks): at 80 m, 6 / 5 / 3 / 1% at the decision (p 40 / 50 / 60 / 75), and 26 / 11 / 6 / 1% if it stops to look
    down each street at the corner. Most of the swarm's lead over one drone in what it sees before choosing is
    therefore there without surrounding (41% against 6% at p 40).
  - **Surrounding is what looks down the exits.** A 12 s aim-and-turn surround at d_ref 1.8 looks down 93 / 79 / 50 /
    17% (p 40 / 50 / 60 / 75). A pause of the same length at the corner looks down 51 / 35 / 25 / 9%, and holding
    longer than 6 s there adds nothing.
  - **d_ref matters.** The wider the block spread, the more the surround looks down: +5 to +13 points from 1.0 to 1.8
    for a 12 s surround, up to +25 at 2.6. Above 1.8 (2.2 at p 75) the drones touch the building they surround (84 of 85 located
    contacts), beyond the safety bar.
    The useful spread is therefore wider than Phase 1's d* at p 40–60 and wider than the trials' dial (0.4–1.6).
  - **At p 75 nothing safe surrounds the block** (the user's observation): even at 2.2 a 20 s surround encloses it in
    67% of trials and looks down 38% of its exits.
  - **The cost is time.** From the block to the next junction a 12 s surround takes 24–30 s, against 15–22 s for the
    street-bound swarm.
- **Part B** (a search for a goal standing at a building, p 40 and p 60 at s 18, 8 cities × 4 targets each, with the
  surround Part A's rule picks):
  - **The surround still finds the goal more slowly than the swarm kept in the streets**: ×1.27 at p 40, ×1.74 at
    p 60, ×1.48 (1.18–1.87) pooled.
  - **The decomposition is Phase 2's.** It needs 3–6% fewer decisions, and each takes 27–74% longer. What it sees
    down an exit before choosing, the street-bound swarm sees one block later by flying the exit it chose.
  - It beats a corner pause of the same length at p 40 (×0.83). The rule's wider, longer surround does no better
    than Phase 2's (×1.06).
  - The swarm in the streets finds the goal in 44% of one drone's time; the surround in 65% (0.48–0.88), and at p 60
    alone 78% (0.54–1.13), not clearly sooner than one drone.
- **At a 250 m sight range** (2026-10-28, the user's call: 80 m, inherited from the goal's recognition distance, was too
  short to mean anything for looking down a street; manual 9.7). Phase 3 (Parts A and B, the lone drone) and Phase 2's
  informed search were flown again with every condition counting a sighting within 250 m (runs `_r250`; the 80 m runs
  reproduce bit for bit and stay as the record). The answer is the same:
  - everyone sees far more: one drone looks down 33 / 33 / 27 / 23% of a block's exits at the decision, the swarm in
    the streets 87 / 75 / 63 / 51% and a 12 s surround 99 / 97 / 92 / 81%, and the spread barely changes what a
    surround sees (it still sets whether the formation encloses the block);
  - time to find, as a share of one drone's: swarm in the streets 51% (goal at a building) and 53% (target anywhere),
    surrounding 65% and 58%; surrounding ÷ streets ×1.26 (1.01–1.58) and ×1.08 (0.97–1.21);
  - the decomposition holds: one drone needs 2.0–2.3× the decisions; surrounding saves 14–15% of them and makes each
    23–60% longer.
  The page's answer and Look down tab now use the 250 m runs; Phase 2's own tabs are its 80 m record.
- **Against one drone, the comparison the trials need** (the redesigned page's Answer tab, at 80 m; the 250 m figures
  are above): every way of flying the
  swarm finds a target sooner than one drone (44–72% of its time across Phase 2 and Part B), and flying the streets is
  the quickest. One drone needs 1.7–2.5 times the swarm's decisions, which is where the swarm's edge comes from;
  surrounding saves a further 3–10% of decisions but makes each 21–74% longer.
- **For the user to decide:** whether another full study is worth running, and on what task. Phase 3 suggests a full
  study would need:
  - the look-down measure;
  - d_ref as a factor beyond the dial;
  - surrounds long enough for their blocks;
  - more cities;
  - a task where seeing down every exit before choosing matters more than it does in a search.

**Limits and refinements.**
- **Only small buildings.** Phase 2 flew the two designs Phase 1 passed: 40 m blocks with buildings about 13 m
  across. Larger blocks, where surrounding might be needed, failed Phase 1 on time and reform.
- **Scripted pilots.** The informed pilot uses every feed and a fixed rule. On head-ahead knowledge alone the surround
  leaned slower (×1.16, 0.99–1.35). People would use less of the preview (lesson 8).
- **One surround** (6 s, aim and turn) and the 2-D replica (lesson 12).
- Refinements identified: lessons 22–29.

**Open decisions.**
- Which view-correspondence measure stands for "the pilot can relate the screens". The results page gives them all,
  as the user asked; none is chosen.
- Whether a surround matched to the block's size deserves a study of its own. Phase 3 flew one matched in spread and
  length. It looked down far more of each block's exits and still found a goal at a building more slowly than the
  swarm kept in the streets (×1.48). A further study would need a task where the decision itself is the cost.

**Alternatives for a later study.**
- **A city of tessellated regular hexagons.** This is the user's note of 2026-10-27; Voronoi cells were kept for
  S4.
  - **For it:**
    - Every block would be the same, so the matched separation d* would be one number per design. Voronoi blocks
      spread it by 15–20% within a city.
    - Every junction would be three-way at 120°.
    - A hexagon's circumcircle bulges only 0.077·W past its faces. Block-shaped buildings with a per-building
      cylinder radius would therefore keep a wide gap across every street: 13.6 m at p 50 with 12 m streets. Today's
      cylinder law would need only that radius, not a new obstacle model. Today's bounding-box rule still closes
      those streets, with a 9.2 m bulge per face.
  - **Against it:** an artificial grid, unlike S1–S3's cities and VoronoiCityWorld, with every block and junction
    alike.
  - **Cost to add:** small. Put the block seeds on a triangular lattice in place of RSA, as an option in
    `citysim/blockcity.py`.

**Where.**
- Manual: `docs/split_study/README.md`.
- Code: `citysim/blockcity.py`, `polyfoot.py` and `streetnet.py`; `split_probe.py`, `split_tables.py` and
  `split_bench.py` (Phase 1); `split.py` and `split_stats.py` (Phase 2); `lookdown.py`, `lookdown_search.py` and
  `lookdown_contacts.py` (Phase 3); `correspondence.py`; `split_report.py` + `.html` (the page).
- Outputs: `experiment_results/split_study/` (`phase1/<run>/`, Phase 2's `split/` and `split_search/`, the Unity
  spot checks `p2_bench/` and `p2_routes/`, and Phase 3's `lookdown_A/` and `lookdown_B/`; `lookdown_B_v2/` is a
  stopped run with a different pilot rule, kept but not analysed).
- Scene: `Assets/Scenes/BlockCityWorld.unity`, R at p 40 s 18, for flying the manoeuvres.
- Scene: `Assets/Scenes/BlockShapedCityWorld.unity` (2026-10-28), B at p 40 s 18 on the same streets: simple windowed
  prisms behind a 3.5 m footpath. Its design is changed in `Tools/Swarm/Block-shaped city…` (p, s, footpath, seed
  placement). Unity flies it with today's cylinder law, not the replica's closest-point model (CLAUDE.md).
- Page: [S4 Split and Surround](https://claude.ai/artifact/C1PtAjZybB7JdAkoJy2xXD) (`python -B -m citysim.split_replays`,
  then `python -B -m citysim.split_report phase1`). **Redesigned 2026-10-28 at the user's request** (the twelve tabs
  had lost sight of why the study was run; lesson 36). It opens on the question for the human trials ("does
  surrounding a block give the swarm its edge over one drone?") with a three-line answer, and its first tab, **The
  answer**, argues it in five steps (sees more? can it surround? finds sooner? why not? what it costs?), each one chart
  from `split_report.answer_data`, then the reading: what S4 settles, what it cannot, and what to take into the trial
  design. The detail tabs below are unchanged, under **All results**, grouped by phase with the key to the codes.
  Its **Cities and flights** tab shows every design's city and replays 75 flights:
  - 7 of Phase 1's manoeuvres;
  - two block-hop walks and two informed searches in all six conditions;
  - Unity's flights of the same walks and routes;
  - Phase 3's: one block each at p 40 and p 75, flown six ways (one drone first), with the buildings at the ends of its streets
    marked as they are looked down; and one search for a goal at a building each at p 40 and p 60.

  Its **Look down** tab has Phase 3: the measure, every way of taking a block against d_ref, d_ref at a glance, the
  rule's pick and Part B. Every flight's summary gives both its spreads.

  Every replica flight was re-flown from its job and is identical to the run's (`experiment_results/split_study/
  replay/`). A key under the masthead and a **Terms** tab define every code.
