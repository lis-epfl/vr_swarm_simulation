# S3 City Roaming

**Formerly / code name:** the clutter study · **Dates:** 2026-10-20 → 10-21 · **Status:** done; a refined re-run is
proposed (below) · **Builds on:** S1 (what each sees), S2 (the random walker), VoronoiCityWorld

**Question.** On task-free walks through a city, how much more does the swarm see than one drone, how does that change
with the city's block size and building coverage, and where does spreading the formation stop helping?

**Design.** 63 buildable Voronoi city designs (block spacing 40–200 m × coverage 0.02–0.14, 12 m streets,
DiamondCityWorld's building mix filled evenly, no goals), 10 cities each (627). Per city 14 five-minute random walks,
each flown on one intent chain by the single drone and the swarm at spreads 0.4, 1.0 and 1.6 (8,778 walks), plus 6
standing points. Measures at the hat's 40 m: sweep width, ground per metre and per second, ground on screen (any
feed / headset view), look-behind, path measures, losses, and the overlap of neighbouring feeds; a hidden point
target anywhere in the city, found within 200 m. Cities are the statistical units (t tests over city means of the
paired log ratio, Benjamini-Hochberg over designs, trends by least squares). 12.0 h run, 0 failed jobs, 176 of 176
re-flown walks identical.

**Findings.**
- **The swarm is ahead on every measure, in every design, at every spread** (q < 0.05 in 2,079 of 2,079 squares).
- **Per metre, what each sees hardly depends on the city**: ground per metre 2.02× at spread 1.0, on screen 5.9×,
  in the headset's view 1.62×.
- **Its lead in finding a hidden target grows with coverage**: 1.48× at coverage 0.02, 1.92× at 0.06, 2.58× at
  0.10, 3.5× at 0.14 (spread 1.0, any screen); 1.19× to 1.96× on the screens straight ahead. +7.9% per 0.01 of
  coverage, −4.7% per doubling of block spacing (both p < 0.0001, R² 0.91). About half of it is speed: the single
  drone slows from 6.2 to 1.8 m/s as coverage rises, the swarm only to 2.4–3.2.
- **Spread 1.6 beats 1.0, which beats 0.4**: 1.6 covers ground 11% faster than 1.0 (62 of 63 designs) and loses
  0.3 drones per walk; 0.4 is 6–13% worse on every measure and loses 4.8.
- **The overlap limit is real**: neighbouring screens share ground 49% / 42% / 26% of the time at 0.4 / 1.0 / 1.6,
  falling with coverage; where widening lost more overlap it gained less (partial r −0.42); in small dense blocks
  1.6 falls 3–9% behind 1.0 on sweep width.
- **For VoronoiCityWorld and the trials** (report linked below): vary coverage (about 0.02, 0.06, 0.10) at today's
  block pitch; keep goal clearance ≥ 22 m (S1); floor the spread dial near 1.0; log panorama quality.

**Limits and refinements** (in [lessons.md](lessons.md)):
- **Sweep width saturates for the swarm** at the ±42 m window (max 84 m; the swarm reaches 82.5 m), so it cannot rank
  spreads and understates the swarm (6–7% at 1.0, 11–20% at 1.6). Re-measured to ±82 m, 1.6 sweeps 6–13% more than
  1.0. **Fixed in code on 2026-10-21** (`clutter.SWEEP_WINDOW_M` = 80 m, `transit.score(sweep_window=…)`); the
  published numbers are from the old window and need re-flying to correct.
- **The hidden-target measure needs refining**: the 5-minute chance saturates (single 50%, swarm 64% in open
  cities); at 200 m in a 730 m city it measures the view from the start (the speed-up falls from 1.9× at 1 min to 1.5×
  at 5); range and attention set its size (1.88× at 40 m, 1.48× at 200 m, 1.19× head ahead); a single glimpse counts.
  Proposed: targets out of sight of the start, discovery at 40/80/200 m, internal_3's head movements, a 1 s on-screen
  requirement, 10-minute walks, fewer block sizes.
- Replica only, not yet flown in Unity in these cities; random walker, not people.

**Open decisions.** Whether to run the refined S3 (about 12 h, which would also correct the sweep widths), and the
Unity spot check of VoronoiCityWorld at coverage 0.02 / 0.04 / 0.06 / 0.10.

**Applied to VoronoiCityWorld (2026-10-21).** Three layouts of one seed at coverage 0.0227 / 0.06 / 0.10 (same
streets and goals, 246 / 688 / 1,296 buildings, at most 13 / 22 / 38 within 50 m), 90.83 m blocks, S3's 12 m streets
(the scene had 15 m), every goal's ring at least 22 m from its centre (24.5 / 23.4 / 22.2 m), altitude ceiling 45 m
under the 50 m roofs. The spread dial stays 0.4–1.6 for now, by the user's choice. Panorama quality was already logged
(`_stitch.csv`). Next: the user flies the levels; the bench spot check is still open.

**A fourth level, 0.04 (2026-10-26).** The user found the step from 0.0227 to 0.06 too large to fly as one. 0.04 is an
S3 grid level (1.67×, mean over block spacings; 1.66× at 90.83 m) and splits it: 246 / 438 / 688 / 1,296 buildings, at
most 13 / 16 / 22 / 38 within 50 m, same seed, streets and goals (goal rings 24.5 / 23.4 / 22.2 m clear).

**Where.** Manual [docs/clutter_study/README.md](../clutter_study/README.md); code `citysim/clutter.py`,
`clutter_stats.py`, `clutter_report.py` (+ `.html`); spec `specs/clutter.json`; outputs
`experiment_results/clutter_study/clutter/` (the raw per-walk files, 360 MB, are gitignored and only on the PC that ran
it; the tables, figures and page are committed); results page https://claude.ai/artifact/3e13hWAnvaHGyvJoADAN5J; report
for VoronoiCityWorld's design https://claude.ai/artifact/RTdZRe192eeGPoJaQ26A3s.
