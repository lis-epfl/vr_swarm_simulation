"""citysim: parameterised cities and the visibility-scaling study that grounds the human trials.

The package behind docs/density_study/ (see its README for the study, its tiers and how to run them):

    camera      the FPV camera and display constants (one place, mirrored by FpvCameraFov.cs)
    geometry    convex polygons, rectangles, blue-noise sampling (numpy only)
    catalogue   the ScaledCity building designs (building_catalogue.json) and the study design
    coverage    coverage alpha, building intensity, the Boolean free-sight length ell(theta)
    layout      the layout schema: city_obstacles boxes + streets + generator record; verify()
    citygen     the irregular-block generator (and the scatter mode for the Boolean comparison)
    los         line of sight: the 3-D slab test (== analyse.segment_blocked) and 2-D isovists
    walkers     goal pedestrians on WalkerPatrol's loops
    formation   swarm templates from the 2-D replica: hover, envelopment, and the envelopment map
    display     the hull / heading / feed-ring display rule, head visibility, the reveal model
    single      single-drone policies: static best vantage and the orbit / scan
    operator    the operator band from internal_3 (altitude, pitch, spread, head yaw)
    theory      Tier A: the Boolean blockage model
    vissim      the tiers' command line: Tier A/B sweeps, B' dynamic, Tier C scoring, validate, gate1
    report      figures

Everything here is numpy (+ scipy/pandas where noted) and never imports Unity. Run with the `stitching`
env. Nothing under Assets/ may grow a __pycache__, so the bytecode switch is thrown before any submodule
is imported -- entry scripts throw it again before importing this package.
"""
import sys

sys.dont_write_bytecode = True

from pathlib import Path  # noqa: E402

PACKAGE_DIR = Path(__file__).resolve().parent
EXPERIMENT_DIR = PACKAGE_DIR.parent                     # Assets/Scripts/Experiment
REPO = PACKAGE_DIR.parents[3]
BENCH_DIR = REPO / "Assets" / "Scripts" / "swarm" / "Bench"
LAYOUT_DIR = EXPERIMENT_DIR / "layouts"
STUDY_DIR = REPO / "experiment_results" / "density_study"
DOCS_DIR = REPO / "docs" / "density_study"
DEFAULT_ROOT = Path.home() / "AppData" / "LocalLow" / "UAVS@BERKELEY" / "DroneSim" / "experiment"


def use_bench_modules():
    """Put the swarm bench (swarm_params, swarm_replica, bench) and analyse.py on sys.path."""
    for p in (str(EXPERIMENT_DIR), str(BENCH_DIR)):
        if p not in sys.path:
            sys.path.insert(0, p)
