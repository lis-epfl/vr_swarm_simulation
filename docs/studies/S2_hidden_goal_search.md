# S2 Hidden Goal Search

**Formerly / code name:** the search study · **Dates:** 2026-10-15 → 10-16 · **Status:** done · **Builds on:** S1
(its capability cities, discovery definition and replica)

**Question.** A pilot who does not know where the goal is searches the city for it: does the swarm find it sooner than
one drone, and is that its cameras or its formation morphing round buildings?

**Design.** S1's 15 capability cities (C0, C1p, C2p, C1, C2 × 3 seeds), one hidden goal patch (today's G0) per
scenario, 360 scenarios (3 goals × 8 starts per city, starts at least 260 m from the goal). Pilots: a random walker
(run and tumble, runs Exp(120 m); 60 and 240 m sensitivities) and internal_3's searching pilots fitted
(`pilotfit`). Arms: single, single_scan, swarm (plans round walls, morphs round the rest), swarm_steer (the swarm on
the single drone's path rule: the morphing counterfactual). The swarm's feeds from `display.FeedLatch`. The clock stops
at first sight (S1's discovery definition, 80 m; 200 m beside it), capped at 25 min; a swarm under four drones has not
found it. Restricted mean time ratios with a cluster bootstrap over goals; Unity spot check.

**Findings.**
- **The swarm finds a hidden goal 6–18% sooner** (random walker, swarm ÷ single restricted mean time 0.82–0.94 at
  spread 1.0; fitted pilot 0.77–0.92). At 200 m it is 0.72–0.85.
- **It is the cameras, not the morphing**: swarm ÷ swarm_steer is about 1 everywhere except the packed C2 (0.84),
  where steering ten drones through the single drone's gaps splits the formation and costs drones.
- **Its reach is the mechanism**: 73–97% of its first sights were out of range or band of a camera at the formation's
  centre; 3–23% came round a corner.
- **Spread 0.4 is slower than the single drone in dense cities** (1.12–1.34), losing 5.7–7.1 drones a search; half its
  searches end under four drones.
- The Unity spot check agrees (swarm ÷ single 0.91 against the replica's 0.90).

**Limits and refinements** (in [lessons.md](lessons.md)):
- People found goals 2.3× (swarm) and 10× (single drone) faster than the fitted pilot in internal_3's own city, so the
  ratios describe undirected search and the human advantage may well be smaller.
- In the head's view (fitted head) the advantage shrinks (0.81–1.03): where pilots look sets the effect size.
- Subsets of searches must be taken by start index, never by drawing fewer starts (the random stream changes).
- Every run stranded 1–4 first-batch jobs (this PC's CPU); fly them separately with the same seeds and merge.

**Open decisions.** None recorded.

**Where.** Manual [docs/search_study/README.md](../search_study/README.md); code `citysim/search.py`, `pilots.py`,
`pilotfit.py`; spec `specs/search.json`; outputs `experiment_results/search_study/` (`search/`, `search_streets/`,
`check/`, `bench/spot/`, `search/dashboard/`); results dashboard https://claude.ai/artifact/9KcgYMcAH4RRWWSKdFbU59.
**The dashboard is a light rebuild (2026-10-21)**, less in depth than the first (2026-10-16), which was lost: ratios
with intervals, every condition's searches and the two checks, without the city maps, search replays and pilots'
target chains. Its data files come from `citysim/search_dashboard.py` and its page is `citysim/search_dashboard.html`,
so unlike the first it can be rebuilt (manual §5).
