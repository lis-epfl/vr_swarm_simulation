# The clutter study

**S3 City Roaming** in the study registry ([docs/studies/](../studies/README.md)); "the clutter study" is its code name.

**Question.** How much more does the ten-drone swarm see than one drone, standing anywhere in a city and roaming it
with no task, and how does that change with the city's design: how big its blocks are and how much of the ground is
built on? And where does spreading the formation stop helping, because its cameras end up pressed against buildings
and neighbouring feeds no longer overlap?

It continues the task-independent half of the density study (the explorer's *What each sees* tab, `citysim.capability`)
with the search study's random walker (`citysim.pilots`, intent `rw`), on many more cities: a grid of Voronoi city
designs built the way `VoronoiCityWorld` is. Nothing here involves goals or walkers, so goal geometry, which dominated
the density study, plays no part.

Code: `citysim/clutter.py` (cities, walks, standing points, the run), `citysim/clutter_stats.py` (statistics),
`citysim/clutter_report.py` + `clutter_report.html` (figures and the page), spec `citysim/specs/clutter.json`, tests
`citysim/tests/test_clutter.py`. Outputs: `experiment_results/clutter_study/<name>/`. Run everything in the
`stitching` env from `Assets/Scripts/Experiment`. The results page is a private artifact,
https://claude.ai/artifact/3e13hWAnvaHGyvJoADAN5J (built from `clutter/clutter_study_bare.html`).

## Results (run `clutter`, 2026-10-20/21)

63 of 72 designs buildable at 12 m streets, 627 cities, 14 five-minute walks per city (8,778 walks; the 12.5 h budget
stopped the run after round 14 of 16), 6 standing points per city; no job failed, 176 of 176 re-flown walks identical.
Cities are the units of every test. Ratios are swarm ÷ single drone at spread 1.0 unless named.

| | result |
|---|---|
| Swarm ahead | on all 11 ratio measures, all 63 designs, all three spreads, q < 0.05 (2,079 of 2,079) |
| Per metre: sweep width / headset sweep | 1.93× (1.87–2.08) / 1.44× (1.34–1.49); +0.6% per 0.01 coverage: flat |
| Standing: on screen / headset / within 1 s | 6.3× / 1.8× / 2.05×, nearly flat |
| Finding a hidden target (200 m, any feed) | 1.48× at coverage 0.02, 1.92× at 0.06, 2.58× at 0.10, 3.5× at 0.14; +7.9% per 0.01 coverage, −4.7% per doubling of block spacing (both p < 0.0001, R² 0.91) |
| … headset view only | 1.19×, 1.30×, 1.51×, 1.96× (overall 1.36×, SD of city ratios 0.26) |
| Why it grows | speed: single drone 6.2 → 1.8 m/s, swarm 6.1 → 2.4–3.2 m/s as coverage rises (swarm ÷ single speed 0.98 → 1.62) |
| Mean time to find (random walker) | single 7.6 → 35 min, swarm 5.2 → 12.6 min, coverage 0.02 → 0.10 |
| Spread 1.6 ÷ 1.0 | ground per second +11% (62 of 63 designs better), finding +4% (51 better, 3 worse), headset sweep +5% |
| Spread 0.4 ÷ 1.0 | worse on every measure in all 63 designs (−6 to −13%); 4.8 drones lost per walk (1.0: 0.88, 1.6: 0.26); in the densest designs over half its walks end under four drones |
| Neighbouring feeds sharing ground | 49% / 42% / 26% at 0.4 / 1.0 / 1.6; at 1.0 65% → 30%, at 1.6 44% → 13% as coverage rises; unbroken ring 11% / 3% / 0.1% |
| Overlap's cost | lost overlap from 1.0 to 1.6 predicts a smaller gain from widening (partial r −0.42, p < 0.0001); 1.6's sweep width is 3–9% below 1.0's in small dense blocks (40–75 m, coverage 0.08–0.12) |

**Sweep width saturates for the swarm, so do not compare spreads on it.** `transit.sweep_width` bins lateral offsets
only out to the range plus one bin (±42 m), so W cannot exceed 84 m, and the swarm at 1.0 already scores 82.5 m in
open cities. A wider formation's outer cameras see past 42 m, and that is not counted: re-measured over ±82 m on the
same walks (3 designs × 6 walks, 2026-10-21), spread 1.6 sweeps 97.5 vs 88.1 m (open, p 90.83 / α 0.02), 95.1 vs
84.0 m (α 0.06) and 83.6 vs 78.7 m (p 60 / α 0.10) -- ahead everywhere, although in the dense design it sees less
within 20 m of the track (85% vs 92%) and more beyond 42 m (46% vs 19%). Ground per metre (uncapped) gives the same
answer from the full run: 1.6 is 8.5% ahead of 1.0. Swarm ÷ single sweep ratios are understated at every spread,
by about 6–7% at 1.0 and 11–20% at 1.6 in those designs (the single drone's camera reaches 36 m, inside the window,
so its W is not). **Fixed in code on 2026-10-21, not re-flown**: `clutter.SWEEP_WINDOW_M` (80 m, the spec's
`sweep_window_m`) is passed to `transit.score(sweep_window=…)`, so future runs bin to ±82 m (a 164 m maximum); the
range, and every other measure, is unchanged, and the capability and Tier C paths keep 42 m byte for byte. A quick
run on the new code gave single 39.6 m, swarm 74.2 / 83.4 / 88.8 m at 0.4 / 1.0 / 1.6. The figures above are from
the old window; correcting them needs the walks re-flown (the per-strip sightings were not saved): about 12 h for
the full grid, about 3.5 h for every design at 4 walks per city -- best folded into a refined S3 run
([../studies/S3_city_roaming.md](../studies/S3_city_roaming.md)). The results page says which window its run used.

**For the human trials** (protocol: [../density_study/protocol.md](../density_study/protocol.md)):

- **Vary coverage, not block size.** Block size moves the discovery advantage by about 5% per doubling, coverage by
  about 8% per 0.01; a level set at today's pitch (90.83 m, VoronoiCityWorld's) loses nothing.
- **The discovery co-primary rises with density; the search-time co-primary falls at a cramped goal.** The density
  study's L3 reversal is goal geometry (the formation cannot surround a cramped occluder). In transit, density slows the
  single drone more than the swarm. Pre-register the two interactions in opposite directions.
- **At today's density, the discovery effect depends on where pilots look.** Any feed: a time ratio of about 0.68.
  Headset view with the head ahead: about 0.84, barely past the smallest effect of interest (0.8). A denser level
  (coverage 0.08–0.10: 0.66–0.74 on the headset basis) gives an effect that survives the pilots' attention.
- **The tight spread should not be free.** 0.4 is worse on every measure and loses drones at 0.5–2.8 per minute; it is
  also what drives L3's G3 failure (7.0 drones per flight at 0.4 against 1.0 at 1.0). A dial floor near 1.0, or
  coaching, removes the loss problem; 1.6 is best except in small dense blocks.
- **Expect the panorama and screen-to-screen tracking to degrade with density and spread** (neighbouring feeds share
  ground 30% of the time at coverage 0.12 at 1.0). Log the stitcher's quality flag with panorama on-time.
- **Check in Unity before fixing levels**: these are replica results in cities the bench has not flown. The growing
  advantage rests on the single drone's speed in dense cities and on losses; fly VoronoiCityWorld at two or three
  coverages on the bench, as the search study's spot check did.

## Design

**Cities.** `citygen` blocks mode, as VoronoiCityWorld: DiamondCityWorld's building mix at size 1
(`catalogue.city_mix`), every block filled evenly (`place="even"`), 12 m streets, 4 m facade gaps, and **no goal
patches** (`goals=0`, new; a goal-free city is named `..._nogoals_...` and draws no random number for goals, so every
goal city's layout is unchanged). Two factors:

| factor | levels | meaning |
|---|---|---|
| block spacing p | 40, 50, 60, 75, 90.83, 110, 135, 165, 200 m | the mean pitch of the Voronoi cells (90.83 = today's tile pitch) |
| coverage α | 0.02, 0.03, 0.04, 0.06, 0.08, 0.10, 0.12, 0.14 | the built share of the 730 m arena (DiamondCityWorld is 0.023) |

Ten cities (seeds) per design. A city whose blocks fill before α is reached (it *jams*), or that puts more than 56
buildings within 50 m of a point (OlfatiSaber's collider query), is not used, and a design is flown only when at least
70 % of its cities could be built: small blocks cannot take dense coverage (p 40 jams above 0.06, p 60 above 0.10),
and the frontier is itself a result. 58 of the 72 designs are flown.

**Conditions.** `single`: one drone on the path rule that plans round every building (routing "single"), its view
slaved to its heading. `swarm` at spreads (d_ref) 0.4, 1.0, 1.6: ten drones on the rule that plans round walls only
and lets the formation morph round the rest (routing "swarm"), its feeds from `display.FeedLatch` (boundary debounce,
heading low-pass, 75 deg/s yaw, the ready latch) -- exactly the search study's `single` and `swarm` arms.

**Walks.** Every condition of a walk flies the same intent chain (run and tumble: 20 m + Exp, mean 120 m, a uniform
new heading per run, targets snapped to open ground) from the same start and heading for 300 s, or until a swarm is
down to fewer than four drones. The chain is laid target to target and never reads the flight, so the conditions are
paired. Starts are free points with as much clearance as the city allows (12, 8, 5, 3 m relaxed in turn).

**Standing points.** `capability.settle_at`: the formation flown in from the most open side of a random free point and
held by the centring pilot for 60 s, at each spread; the single drone on its centroid facing the same way; four body
yaws; `capability.static_metrics` (ground on screen, in the headset's view weighted by internal_3's head yaws,
within 1 s, parallax escape, useful feeds).

## Measures

All at the hat's recognition range, 40 m slant, through clear 2-D sight lines, in the camera's 74.4 x 46.2 deg field
of view at internal_3's band (walks: the swarm's transit median, 19.5 m and -15 deg, for both interfaces --
"matched", the comparison; the single drone also at its own band, "own"; standing: the search-phase median).

| measure | column | what it is |
|---|---|---|
| sweep width | `sweep_any_r40_m` | **primary.** width of city cleared per metre flown: for ground at each distance from the track, the share ever seen, summed across the track (`transit.sweep_width`) |
| clearance rate | `area_any_r40_per_s` | new ground seen per second (sweep width x speed: a slower interface pays) |
| per metre | `area_any_r40_per_m` | new ground seen per metre flown |
| headset sweep | `sweep_head_r40_m` | sweep width on the feeds inside the HMD's +-50 deg straight ahead |
| on screen | `inst_any_r40_m2` / `inst_head_r40_m2` | ground on the shown feeds at an instant, averaged over the walk |
| standing | `A_any_r40`, `A_head_r40`, `V_swarm_1_r40` vs `A_single_r40`, `V_single_1_r40` | the explorer's standing measures |
| look-behind | `behind_corner_r40_frac` | swarm ground seen first while a camera at the formation's centre could not (round a corner) |
| time to find | `find_any_r200`, `find_any_r200_curve`, `find_any_r200_rate_per_h`, `find_any_r200_time_min` (and `_r40`) | a target hidden at a uniformly random spot of the arena's open ground (a point at hat height: no goal patch, no walkers) is found the first time a shown feed has it within 200 m with a clear line; P(found by t) is the share of the arena's free cells first seen by t (the curve, every 15 s), `find_` is P by the end of the walk, `_rmst_s` the mean time capped at the walk's length, and `_rate_per_h` / `_time_min` the random-search (Koopman) fit P(t) = 1 - exp(-lambda t) at the walk's end: lambda and the mean time 1 / lambda |
| how it flies | `speed_ms`, `path_ratio`, `adj_per_100m`, `morph_per_km`, `street_path_frac` | path ratio = flown over the net progress toward the intent targets |
| what it costs | `lost`, `lost_per_km`, `split_frac`, `contacts` | drones lost, time split |

**Why no goal.** A goal patch brings its own geometry (occluder, ring, clearance), which is what dominated the density
study; the search study measured time to find a goal patch with walkers. Here the target is a point hidden at random,
so its time to find is a property of the city and the interface alone, and every walk gives the whole curve: the
walks are scored on one 2 m grid over the **whole arena** (`clutter.arena_cells`, built footprint by footprint --
transit.Cells' all-pairs test would need gigabytes per worker -- and identical to it cell for cell), with
`transit.score(near_only=True)` measuring lateral offsets only for cells within reach of the track (exactly the same
sweep width, asserted in the tests). In a 730 m city most targets are still hidden after 5 minutes, which is why the
random-search fit is reported beside the share found.

**The headset.** Screens sit on a ring 2 m round the pilot, each at its drone's heading relative to the body yaw (a
74.4 deg feed fills 47.9 deg of the ring); the HMD sees +-50 deg. Flying (`*_head` walk measures), the head faces the
direction of travel -- internal_3's swarm pilots looked within 25 deg of straight ahead 87 % of the time in transit --
and the body yaw turns to the bearing flown at 90 deg/s. Standing (`A_head`), the head direction is weighted by
internal_3's recorded search-phase head directions (`operator_band.head_weights`). Within 1 s (`V_*_1`): the head
turns at 180 deg/s up to 90 deg, then the body yaw at 90 deg/s, which reaches every screen; the single drone yaws at
75 deg/s. The search study's fitted head process (`pilots.HeadProcess("fit")`) is not used here.

**Streets** are 12 m in every city (the user's choice on 2026-10-20, down from today's tile cities' 15 m; the settled
formation is about 6, 14 and 24 m across at 0.4, 1.0 and 1.6, so only 0.4 fits a street without squeezing), held
fixed until VoronoiCityWorld's design settles.

**Discovery at 200 m.** The hidden target is found within `find_range_m` (200 m, the user's choice; internal_3's turns
onto goals matched best at about 200 m in the density study's check) -- the `find_*_r200` columns, from
`clutter.discovery_first`: per camera, rays every 0.5 deg across its field of view to the first facade
(`Footprints.first_hit`), a cell of a 4 m arena grid visible when it lies in the field of view and the vertical band and
nearer than its ray's hit, checked every second. An exact sight line per cell, as at 40 m, would cost about a hundred
times more at 200 m. `find_*_r40` (exact, 2 m grid) are kept beside them. Every other measure is at the hat's 40 m.

**The ring of feeds** (`clutter.RingTally`), the limit of a wide formation among buildings. The shown feeds in display
order; each adjacent pair (the ring wraps) shares the ground both see within 40 m:

- `ring_conn_frac`: pairs sharing at least 20 m^2 (five 2 m cells) through clear sight lines, of all pairs;
- `ring_geo_conn_frac`: the same for the frustums alone, buildings ignored -- what the spread and the hull rule allow,
  so `geo - conn` is what the buildings take;
- `ring_whole_frac`: share of the time every pair is connected (an unbroken ring);
- `ring_overlap_m2`, `ring_geo_overlap_m2`: the shared ground per pair;
- `wall_feed_frac`: shown feeds more than half filled by a facade nearer than 5 m (`display.WALL_HIT_M`);
- `uncovered_deg`: azimuth round the ring no feed's field of view covers;
- `cam_facade_m`: the shown cameras' median distance to the nearest facade.

## Statistics (`clutter_stats`)

Each walk (and each standing point, averaged over its body yaws) pairs the swarm at a spread with the single drone:
`lr = log(swarm / single)`. The walks of one city share its streets and buildings, so **the cities are the
independent units**: `lr` is averaged per city, and per design and spread

- `ratio` = exp(mean of the city means), the geometric-mean ratio, with its 95 % t interval over the cities;
- `sd_city_ratio` (SD of the city ratios), `sd_log_city`, `sd_log_unit` (SD of the walk log ratios);
- `sw_mean ± sw_sd`, `sg_mean ± sg_sd`: each interface's measure over the walks;
- `p` = one-sample t test of the city means against 0; `q` = Benjamini-Hochberg over a map's designs (one measure,
  one spread); `d_z` the standardised effect; `swarm_better_units` the share of walks the swarm won;
- `p_wilcoxon_units`, a signed-rank test over the walks, as a sensitivity only: it treats walks in one city as
  independent and overstates significance (the unit test shows it).

`spread_contrasts.csv`: spread 1.6 and 0.4 against 1.0 on the same walks, the same way. `trends.csv`: per measure and
spread, least squares of the city-mean `lr` on centred α, centred log2 p and their product, cities as points: the
coefficient as % change of the ratio per +0.01 coverage and per doubling of the block spacing, with SE and p.
`level_cells.csv`: every absolute measure per condition and design (mean, SD over units, SD and 95 % interval over
cities). `city_summary.csv` + `links.csv` put the wide formation's limit to the test: per city, the spread contrasts
(1.6 / 1.0, 0.4 / 1.0) and swarm / single ratios against its ring and cost measures (neighbour overlap at 1.0 and 1.6
and its drop, wall-filled feeds, losses), as Pearson r over cities and as partial r with coverage and log2 block
spacing held fixed -- the design drives both sides, so only the partial r says the overlap itself matters.

## Running it

```bat
cd Assets\Scripts\Experiment
python -B -m citysim.vissim clutter-feasibility            :: which designs can be built (generates the cities)
python -B -m citysim.vissim clutter --quick                :: four designs, three cities, one short walk: ~1 min
python -B -m citysim.vissim clutter --jobs 30              :: the full run, to the spec's time budget (11.5 h)
python -B -m citysim.vissim clutter --resume --jobs 30     :: after a crash or a reboot: skips every finished job
python -B -m citysim.vissim clutter-tables --name clutter  :: statistics, figures and the page from the rows again
python -B -m citysim.clutter_report --name clutter         :: just the figures and the page
```

A spec may set `layout_set` (the `layouts/sweep~/` folder to use, default the run's name) and `judge_cities` (how
many seeds decide whether a design is flown, default `cities_per_cell`): a preview run with `cities_per_cell` 3,
`judge_cities` 10 and `layout_set` "clutter" flies three cities of exactly the designs the full run flies, on the
full run's layouts.

Invariants of the run:

- **Rounds are replicates.** Round k holds every flown city's k-th walk and k-th standing point, shuffled. The time
  budget (`--hours`, less a 30 min reserve) stops before a round that would overrun it, so a stopped run covers
  every design equally; `--resume` carries on from the rounds already done.
- **Every result is appended to `results.jsonl` as it arrives**; a new run (without `--resume`) moves an old file
  aside. A line cut short by a crash is skipped on reading; a later failure never hides an earlier success.
- **This machine's CPU faults** (docs/cpu_instability_13900k.md): a worker that dies loses its job silently.
  `citysim.parallel` notices a pool that has produced nothing for 10 min and re-runs what is outstanding; a pool that
  gives up costs its jobs, not the run; failed and lost jobs are retried once at the end; each flight and each
  scoring retries twice on an exception. Then **2 % of the walks are flown again and compared field by field**
  (`recheck.csv`): every job is deterministic in its seed, so a difference is corruption, not chance.
- **Do not edit `clutter.py` or anything it imports while a run is going**: every round starts a fresh pool, and its
  workers import the modules from disk. `clutter_stats.py`, `clutter_report.py` and the page template are imported
  only by the parent, at the end, so they can be revised during a run (and rerun with `clutter-tables`).
- The run asks Windows not to sleep while it is going (`SetThreadExecutionState`, released when it exits; no power
  setting changes). A reboot (Windows Update) ends it: `--resume`.

Outputs in `experiment_results/clutter_study/<name>/`: `cities.csv` (every city, buildable or why not, its
descriptors and porosity), `walks.csv` (one row per condition and band of a walk), `compare.csv` (who saw shared
ground first), `static.csv`, `units.csv` (the paired log ratios), `ratio_cells.csv`, `spread_contrasts.csv`,
`level_cells.csv`, `trends.csv`, `find_curves.csv`, `city_summary.csv`, `links.csv`, `cell_descriptors.csv`,
`recheck.csv`,
`errors.csv`, `figures/*.png`,
`clutter_study.html` (open it as a file) and `clutter_study_bare.html` (what the artifact is published from),
`manifest.json`, `run.log`. A `conclusions.html` placed beside them is set into the page's findings by
`clutter_report`. **`results.jsonl`, `walks.csv`, `units.csv`, `static.csv`, `compare.csv` and
`clutter_study_bare.html` are gitignored** (results.jsonl alone is 235 MB, past GitHub's 100 MB limit): they exist
only on the machine that ran the study, so the tables, figures and page are committed but rebuilding them, or the
page, needs those files.

## What it cannot tell

- The replica is 2-D: every building is taller than the drones and blocks every sight line; there is no street
  furniture, wind, or attention. DiamondCityWorld has low buildings a real camera would see over.
- The walker is a task-free model of moving through a city. Pilots fly differently (the search study fitted
  internal_3's), so absolute numbers are the model's; the paired design makes the ratios comparable.
- The swarm's feed timing is modelled (FeedLatch) but not the panorama, and "seen" is geometry at 40 m, not
  recognition.
