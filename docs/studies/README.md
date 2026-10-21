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

Do not cite https://claude.ai/artifact/6QyLDyzucn1rN3RqauCkie ("Swarm Density Study"): it is a stale 2026-10-13 build of
S1's explorer, from the street-route results, left up because it was shared.

The S2 dashboard is **less in depth than the first one** (2026-10-16, MjEnrECiRZA1Wgx4W7EiuF), which was found gone
on 2026-10-21: it has the ratios, every condition's searches and the checks, not the city maps, search replays or
pilots' target chains ([search_study/README.md §5](../search_study/README.md#5-running-it)).

Pages are built from the repo, so a page can always be rebuilt: S1 `python -B -m citysim.explorer --bare --out <file>`,
S2 `python -B -m citysim.search_dashboard` (its data files) + `citysim/search_dashboard.html` (its page), written to
the dashboard's store as its type describes, S3 `python -B -m citysim.clutter_report --name clutter --no-figures`
(then publish `<run>/clutter_study_bare.html`), the hub from `docs/studies/hub.html` as it stands.

## Names in code

The code, specs and output folders keep the names the studies had when they ran, so nothing that reads them broke:

| ID | Manual | Code | Spec | Outputs |
|---|---|---|---|---|
| S1 | [docs/density_study/](../density_study/README.md) (+ [protocol.md](../density_study/protocol.md)) | `citysim/vissim.py` tiers, `capability.py`, `transit.py`, `passby.py`, `explorer.py` | `specs/tierB*.json`, `capability.json` | `experiment_results/density_study/` |
| S2 | [docs/search_study/](../search_study/README.md) | `citysim/search.py`, `pilots.py`, `pilotfit.py`, `search_dashboard.py` (+ `.html`) | `specs/search.json` | `experiment_results/search_study/` |
| S3 | [docs/clutter_study/](../clutter_study/README.md) | `citysim/clutter.py`, `clutter_stats.py`, `clutter_report.py` | `specs/clutter.json` | `experiment_results/clutter_study/clutter/` |

All Python runs in the `stitching` env from `Assets/Scripts/Experiment` (`python -B -m citysim.vissim <command>`).

## Starting a new study (S4 onwards)

1. Read [lessons.md](lessons.md) and the entries of the studies it builds on; say in the design which lessons apply
   and how each is handled.
2. Take the next ID. Copy the template below to `docs/studies/S<n>_<slug>.md` and add a row to the table above.
3. Keep its manual (how to run, every measure, every decision) in `docs/<code_name>/README.md` as S1–S3 do, and its
   outputs in `experiment_results/<code_name>/`. The registry entry is the summary; the manual is the detail.
4. When it finishes, fill in Findings and Refinements, add any new lesson to lessons.md with the study's ID, and
   update the hub page (`docs/studies/hub.html`, republished to the hub URL above).

```markdown
# S<n> <Name>

**Formerly / code name:** … · **Dates:** … · **Status:** … · **Builds on:** S…

**Question.** One or two sentences.

**Design.** Cities, conditions, pilots, measures, statistics, run size.

**Findings.** The answers, with numbers and their uncertainty.

**Limits and refinements.** What it cannot tell; what the next study should do differently (also in lessons.md).

**Open decisions.** Anything waiting on the experimenter.

**Where.** Manual, code, spec, outputs, pages.
```
