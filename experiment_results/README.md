# Experiment results

Snapshots of analysed test series, copied out of the experiment folder
(`%USERPROFILE%\AppData\LocalLow\UAVS@BERKELEY\DroneSim\experiment`) so the numbers and figures are
versioned. One folder per test:

- `test.json` — the test definition (runs' practice list, city, swarm parameters).
- `results/` — the tables `analyse.py run <test>` writes (`runs.csv`, `legs.csv`, `goals.csv`,
  `crashes.csv`) and `analysis_config.json`, which records when they were generated and with which settings.
- `plots/` — the figures from the same invocation (`plots/<test>/` in the experiment folder).

The raw per-run recordings are **not** here; they stay in the experiment folder. To regenerate, run
`python Assets/Scripts/Experiment/analyse.py run <test>` (in the `stitching` env) and copy the outputs back.
