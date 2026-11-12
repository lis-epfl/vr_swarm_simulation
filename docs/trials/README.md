# User trial registry

Every series of user trials (pilots in the headset, the city search task, the swarm against one drone), with one
results page each and an index of them, the **User Trial Index**. The simulation studies that shaped each series are
in [docs/studies/](../studies/README.md); the Swarm Study Index links here.

| Series | Dates | Pilots | City | What it was |
|---|---|---|---|---|
| [internal_1](internal_1.md) | 2026-07-06 → 07-07 | ERIC, ERIP | CityWorld | The first trials: no answers recorded (times taken by hand, not on this machine), the buildings invisible to the swarm's avoidance |
| [internal_2](internal_2.md) | 2026-09-23 | AABB, AABC | ScaledCityWorld | Answers entered afterwards; swarm collisions at the tightest spread |
| [internal_3](internal_3.md) | 2026-10-01 | BBAA, BBCC | DiamondCityWorld | RC presses (outcomes not keyed); the core and damper fix |
| [internal_4](internal_4.md) | 2026-11-06, 11-10 | ERKS, ALEX | BlockShapedPatchCityWorld | Presses scored on the spot, the start line, no stitching, S6–S11's swarm law |

The page addresses are [pages.json](pages.json) (`index` and one per series): private claude.ai artifacts; open them
with the Artifact tool, never by web fetch, and update them by their address so the links keep working.

## How a page is made

`Assets/Scripts/Experiment/trials_report.py` (the `stitching` env), from `Assets/Scripts/Experiment`:

    python -B trials_report.py all                 # every series below, then the index (~5 min)
    python -B trials_report.py page internal_4     # one series: analysis, tables, replays, page
    python -B trials_report.py render internal_4   # the page again from its data.json (template or prose changed)
    python -B trials_report.py index               # the index, from the series' data.json

Each writes `experiment_results/trials/<series>/`: `<series>.html` (everything inlined, to open as a file),
`<series>_bare.html` (what the artifact is published from), `<series>_files.json` (the replay files to publish beside
it, under `replay/`), `data.json` (everything the page shows) and `tables/` (runs, legs, goals, crashes as the page
counts them). The index is `experiment_results/trials/trials_index_bare.html`, with no files beside it.

- **The numbers are analyse.py's,** run in memory on the series folder in the experiment folder (its test.json: the
  practice list, the city, the identify range, the overrides), with one change: **every run is timed from the start
  line** (`--start line`: the recorder's crossing where it exists, the same crossing derived from the drone logs
  before 26 October 2026), so every series is on one clock. internal_2's and internal_3's times are therefore ~25 s
  shorter than their own `results/` tables, which were made from Play.
- **The comparison** is per pilot: the ratio of the geometric means of the swarm's and the single drone's runs, pooled
  as the geometric mean over the pilots, with an exact permutation p (the conditions relabelled within each pilot).
  With two pilots and two runs each there are 36 relabellings, so p cannot go below 0.056; the page says so.
- **The phases** (at a goal, between goals) are compared over runs that reached every goal; practice runs are drawn
  hollow and counted in nothing.
- **The prose is the series' `.md` here**, read by fixed headings: `## Summary` (the lead), `## Changes since …` (or
  `## The starting setup`) with a `###` per area and bullets `- **Lead:** text (`commit`)`, `## Within the series`,
  `## Setup` (a table), `## Notable` (`### <card>: <title>` and its caption; `setups`, `legs`, `panorama`, `restarts`
  add a card, `identify`, `smoothness`, `losses` caption a card every page has), `## Reading` (its first paragraph is
  the masthead's "My reading") and `## Data notes`. Keep run results out of the prose: the page computes them, and a
  number typed here goes stale the first time a definition changes.
- **The replays** are the recorded runs (`_drones.csv` every 0.2 s, the feeds shown from `citysim.replay.shown_mask`,
  the goals, the presses and the start line as the flight's `marks`), played by `citysim/replay_view.js`. A tile
  city's map is its `city_obstacles_<scene>.json` with the run's goal patches swapped in; CityWorld has no export, so
  internal_1's maps draw only the tile grid.

## Publishing

Publish each series' `<series>_bare.html` with the files its `<series>_files.json` lists (published path → source),
then the index's bare page. On a first publish put the new address in `pages.json` and run `render` for every series
and `index` again, then republish them, so the masthead links and the index cards point at the pages.

## Adding a series

1. Group its runs (`python analyse.py group internal_N ...`, see CLAUDE.md "Analysing experiment runs") and check its
   test.json: practice, city, range, overrides.
2. Write `docs/trials/internal_N.md` from the commits since the series before (`git log` between their dates; only what
   a pilot would meet) and from what the runs' own files say (`_session.json`, `_swarm.json`).
3. `python -B trials_report.py page internal_N`, read the page, write `## Reading` from what it shows, `render`, then
   `index`, and publish both (the series before's page gets a link to the new one on its next `render`).
4. Add the row above, the address to `pages.json`, and a line to the series' memory note.
