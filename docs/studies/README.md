# Study registry

Every simulation study run for the swarm-vs-single-drone trials, under a permanent ID, with what it asked, what it
found, what it got wrong, and where everything is. **Start here** before analysing, citing or designing a study, and
**add an entry here** for every new one. In another chat, refer to a study by its ID and name ("see S3 City
Roaming"); this folder is where an agent finds it.

| ID | Name | Formerly | Dates | Question | Status |
|---|---|---|---|---|---|
| [S1](S1_goal_visibility.md) | **Goal Visibility** | density study (+ its capability study) | 2026-10-12 → 10-19 | What each interface sees at a goal, by goal geometry and clutter; picks the trial levels | Done; trial level L3 on hold |
| [S2](S2_hidden_goal_search.md) | **Hidden Goal Search** | search study | 2026-10-15 → 10-16 | How soon each finds a goal patch it does not know the position of | Done |
| [S3](S3_city_roaming.md) | **City Roaming** | clutter study | 2026-10-20 → 10-21 | What each sees on task-free walks across 63 Voronoi city designs, at three spreads | Done; refined re-run proposed |
| [S4](S4_split_and_surround.md) | **Split and Surround** | split study | 2026-10-27 | Can the swarm surround each block of a city of large buildings and reform quickly enough to explore it block by block, and does that pay? | Done: surrounding shows more but finds nothing sooner; Phase 3 pilot (looking down, d_ref): it looks down far more, still finds a goal more slowly |
| [S5](S5_block_shaped_surround.md) | **Block-Shaped Surround** | block-shaped study | 2026-10-28 → 10-29 | S4's question in BlockShapedCityWorld's city (buildings filling their blocks): the spread that gets round a block, footpaths, block size, street width and block layouts | Done |

**[lessons.md](lessons.md)** lists every refinement these studies identified (measures that saturate, assumptions
that set effect sizes, validation gaps, engineering traps), each with the study it came from. A new study should
work through it before its design is fixed.

Pages for people (private claude.ai artifacts; open them with the Artifact tool, never by web fetch):

| | page |
|---|---|
| Hub: all studies ("Swarm Study Index") | https://claude.ai/artifact/8FbWjQ3bF7C65khZYJ3aN5 |
| S1 explorer | https://claude.ai/artifact/NLNeGtKvo5vYW11Rj563qc |
| S2 results dashboard (light rebuild, 2026-10-21) | https://claude.ai/artifact/9KcgYMcAH4RRWWSKdFbU59 |
| S3 results page | https://claude.ai/artifact/3e13hWAnvaHGyvJoADAN5J |
| S3 report (with S1's goal findings, for VoronoiCityWorld's design) | https://claude.ai/artifact/RTdZRe192eeGPoJaQ26A3s |
| S4 results page (Phases 1–3) | https://claude.ai/artifact/C1PtAjZybB7JdAkoJy2xXD (a copy published from another account on 2026-10-28, HG2M1DLjWKUdaMWzqutY5q, is not kept up to date) |
| S5 results page | https://claude.ai/artifact/PrHZmczN5CnhAvZLQTCkb9 |

Do not cite https://claude.ai/artifact/6QyLDyzucn1rN3RqauCkie ("Swarm Density Study"): it is a stale 2026-10-13 build of
S1's explorer, from the street-route results, left up because it was shared.

The S2 dashboard is **less in depth than the first one** (2026-10-16, MjEnrECiRZA1Wgx4W7EiuF), which was found gone
on 2026-10-21: it has the ratios, every condition's searches and the checks, not the city maps, search replays or
pilots' target chains ([search_study/README.md §5](../search_study/README.md#5-running-it)).

Pages are built from the repo, so a page can always be rebuilt: S1 `python -B -m citysim.explorer --bare --out <file>`,
S2 `python -B -m citysim.search_dashboard` (its data files) + `citysim/search_dashboard.html` (its page), written to
the dashboard's store as its type describes, S3 `python -B -m citysim.clutter_report --name clutter --no-figures`
(then publish `<run>/clutter_study_bare.html`), S4 `python -B -m citysim.split_replays` (the Cities tab's replays,
~10 min) and `python -B -m citysim.split_report phase1`, then publish
`experiment_results/split_study/phase1/split_phase1_bare.html` with the files `split_phase1_files.json` lists beside
it (the Cities tab fetches them), S5 `python -B -m citysim.shaped_replays` and `python -B -m citysim.shaped_report`, then publish `experiment_results/shaped_study/shaped_study_bare.html` with the files `shaped_study_files.json` lists, the hub from `docs/studies/hub.html` as it stands.

## Names in code

The code, specs and output folders keep the names the studies had when they ran, so nothing that reads them broke:

| ID | Manual | Code | Spec | Outputs |
|---|---|---|---|---|
| S1 | [docs/density_study/](../density_study/README.md) (+ [protocol.md](../density_study/protocol.md)) | `citysim/vissim.py` tiers, `capability.py`, `transit.py`, `passby.py`, `explorer.py` | `specs/tierB*.json`, `capability.json` | `experiment_results/density_study/` |
| S2 | [docs/search_study/](../search_study/README.md) | `citysim/search.py`, `pilots.py`, `pilotfit.py`, `search_dashboard.py` (+ `.html`) | `specs/search.json` | `experiment_results/search_study/` |
| S3 | [docs/clutter_study/](../clutter_study/README.md) | `citysim/clutter.py`, `clutter_stats.py`, `clutter_report.py` | `specs/clutter.json` | `experiment_results/clutter_study/clutter/` |
| S4 | [docs/split_study/](../split_study/README.md) | `citysim/blockcity.py`, `polyfoot.py`, `streetnet.py`, `split_probe.py`, `split_tables.py`, `split_bench.py`, `split.py`, `split_stats.py`, `correspondence.py`, `split_replays.py`, `split_report.py` (+ `.html`); the replays' `playback.py` and `replay_view.js` are shared | `specs/split.json` (Phase 1's settings are command-line, recorded per run) | `experiment_results/split_study/` |
| S5 | [docs/shaped_study/](../shaped_study/README.md) | `citysim/shaped.py`, `shaped_replays.py`, `shaped_report.py` (+ `.html`); S4's `lookdown.py`, `lookdown_search.py` and `split_replays`' workers fly its designs | none: the designs and rules are in `shaped.py`, recorded per run | `experiment_results/shaped_study/` |

All Python runs in the `stitching` env from `Assets/Scripts/Experiment` (`python -B -m citysim.vissim <command>`).

## Starting a new study (S4 onwards)

1. Read [lessons.md](lessons.md) and the entries of the studies it builds on; say in the design which lessons apply
   and how each is handled.
2. Take the next ID. Copy the template below to `docs/studies/S<n>_<slug>.md` and add a row to the table above.
3. Keep its manual (how to run, every measure, every decision) in `docs/<code_name>/README.md` as S1–S3 do, and its
   outputs in `experiment_results/<code_name>/`. The registry entry is the summary; the manual is the detail.
4. **Give its page a Cities tab and a key to its codes** (lessons 30–31, a standing requirement). The cities: every
   design at one scale, any city whole, one block or goal annotated with what the codes mean. The flights: replays of
   representative flights of every condition, plus Unity's where the bench flew them, each re-flown from its job with
   `citysim.playback`'s tape and checked identical to the run. They are played by `citysim/replay_view.js`, which the
   page inlines. Choose the flights in the design. S4's `split_replays.py` and `split_report`'s Cities and Terms tabs
   are the worked example.
   **Open the page on the study's purpose and its answer** (lesson 36): the masthead says what it tests for the
   human trials and answers in three lines, the first tab argues the answer in a few charts against the comparison
   the trials care about (swarm against one drone), and the detail tabs sit behind one "All results" tab. S4's
   Answer tab (`split_report.answer_data`) is the worked example.
5. **Put a link back to the hub at the top of its page** (the user's requirement, 2026-10-28): every study page
   opens with `← Swarm Study Index` above its eyebrow (the `.back` link in `split_report.html`'s masthead; `.s2-back`
   in the S2 dashboard; a first-line link in the S3 report doc).
6. When it finishes, fill in Findings and Refinements, add any new lesson to lessons.md with the study's ID, and
   update the hub page (`docs/studies/hub.html`, republished to the hub URL above).

```markdown
# S<n> <Name>

**Formerly / code name:** … · **Dates:** … · **Status:** … · **Builds on:** S…

**Question.** One or two sentences.

**Design.** Cities, conditions, pilots, measures, statistics, run size.

**Findings.** The answers, with numbers and their uncertainty.

**Limits and refinements.** What it cannot tell; what the next study should do differently (also in lessons.md).

**Open decisions.** Anything waiting on the experimenter.

**Where.** Manual, code, spec, outputs, pages (with their Cities tab: the cities and the replayed flights).
```
