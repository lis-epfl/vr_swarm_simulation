"""Flight sets, configs and reports for the headless Unity swarm bench (SwarmBenchRunner.cs, run_bench.ps1).

    python bench.py scenarios --city DiamondCityWorld --seed 1          # writes scenarios/<city>_{heldout,spread}.json
    python bench.py config --scene DiamondCityWorld --scenarios scenarios/DiamondCityWorld_heldout.json \\
                           --set scene --set "no damper:c_damp=0" --out my_config.json
    python bench.py report D:/claude_obstacle_bench/runs/<tag> [more runs ...]

A config's parameter sets are OVERRIDES on the scene as authored -- `--set scene` (no fields) flies the scene
exactly as it is -- and every field name and value is checked against SwarmManager.cs before anything is
written. With no --pilot-heading the look-direction gap fill is written explicitly off in every set: it reads
the pilot's body yaw, which nothing sets in a headless run, so leaving the scene's `on` in a config would only
claim something that never flew.

Run with the `stitching` env.
"""
import sys

sys.dont_write_bytecode = True   # nothing under Assets/ may grow a __pycache__ for Unity to import

import argparse  # noqa: E402
import glob  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import os  # noqa: E402
from collections import Counter, OrderedDict  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import swarm_params as sp  # noqa: E402

sys.path.insert(0, str(sp.EXPERIMENT_DIR))
import analyse  # noqa: E402  (load_city: the edit-mode city's buildings)

HERE = Path(__file__).resolve().parent
SCENARIO_DIR = HERE / "scenarios"

# The held-out set's structure, read off the 120 flights frozen from the September tuning
# (scenarios/ScaledCityWorld_heldout.json), so a generated set for another city is the same kind of test.
EDGE_MARGIN = 73.0        # m beyond the outermost building: where transits start and end
TOUR_INSET = 10.0         # m inside that edge: where tours start
HELDOUT_COUNTS = OrderedDict(transit=30, tour=10, ram=50, zigzag=30)
TOUR_WAYPOINTS, TOUR_TIMEOUT, TOUR_T = 6, 45.0, 240.0
RAM_RUNUP, RAM_TIMEOUTS, RAM_T = 90.0, (20.0, 40.0), 60.0
ZIGZAG_LEGS, ZIGZAG_LEG, ZIGZAG_TIMEOUT, ZIGZAG_T = 24, (15.0, 40.0), (2.1, 5.7), 150.0
START_CLEARANCE = 10.0    # m from any building footprint, for a start or a zigzag waypoint
SPREAD_STEPS = [(1.0, 1.6), (1.6, 1.0), (1.0, 0.4), (0.4, 1.0), (0.4, 1.6), (1.6, 0.4), (1.08, 0.5), (1.08, 1.5)]
SPREAD_STEP_AT, SPREAD_T, SPREAD_OFFSET = 30.0, 75.0, 280.0   # step time, flight length, m west of the city


def transit_time(length):
    """The September transits' budget: 30 s plus the length at 6.5 m/s."""
    return 30.0 + length / 6.5


# ============================================================================= scenarios

class Footprints:
    """The city's buildings as horizontal oriented rectangles, for keeping starts out of them."""

    def __init__(self, city):
        boxes, _ = analyse.load_city(city)
        self.c = np.array([b[1] for b in boxes])[:, [0, 2]]
        axes = []
        for _, _, ax in boxes:
            flat = [v[[0, 2]] for v in ax if abs(v[1]) < 0.5 * np.linalg.norm(v)]
            axes.append(flat[:2] if len(flat) >= 2 else [np.array([1e-3, 0.0]), np.array([0.0, 1e-3])])
        self.ax = np.array(axes)                              # (B, 2, 2) half-axis vectors
        ext = np.abs(self.ax).sum(axis=1)
        self.x0, self.x1 = (self.c[:, 0] - ext[:, 0]).min(), (self.c[:, 0] + ext[:, 0]).max()
        self.z0, self.z1 = (self.c[:, 1] - ext[:, 1]).min(), (self.c[:, 1] + ext[:, 1]).max()

    def clear(self, p, margin):
        rel = np.asarray(p) - self.c
        L = np.linalg.norm(self.ax, axis=-1)
        proj = np.abs(np.einsum("bk,bjk->bj", rel, self.ax / L[..., None]))
        return not ((proj < L + margin).all(-1)).any()


def generate(city, seed):
    """(heldout, spread) scenario lists for `city`, the September sets' structure, deterministic in seed."""
    rng = np.random.default_rng(seed)
    fp = Footprints(city)
    X0, X1, Z0, Z1 = fp.x0 - EDGE_MARGIN, fp.x1 + EDGE_MARGIN, fp.z0 - EDGE_MARGIN, fp.z1 + EDGE_MARGIN

    def edge_point(edge):
        if edge in ("W", "E"):
            return (X0 if edge == "W" else X1), rng.uniform(Z0, Z1)
        return rng.uniform(X0, X1), (Z0 if edge == "S" else Z1)

    def inside():
        while True:
            p = (rng.uniform(fp.x0, fp.x1), rng.uniform(fp.z0, fp.z1))
            if fp.clear(p, START_CLEARANCE):
                return p

    heldout = []
    for _ in range(HELDOUT_COUNTS["transit"]):
        a = rng.choice(["W", "E", "S", "N"])
        b = {"W": "E", "E": "W", "S": "N", "N": "S"}[a]
        (sx, sz), (ex, ez) = edge_point(a), edge_point(b)
        heldout.append(dict(kind="transit", sx=sx, sz=sz, T=transit_time(math.hypot(ex - sx, ez - sz)),
                            wps=[ex, ez, 1.0, 0.0]))
    for _ in range(HELDOUT_COUNTS["tour"]):
        wps = []
        for _ in range(TOUR_WAYPOINTS):
            wps += [*inside(), 1.0, TOUR_TIMEOUT]
        heldout.append(dict(kind="tour", sx=X0 + TOUR_INSET, sz=rng.uniform(fp.z0, fp.z1), T=TOUR_T, wps=wps))
    n_ram = 0
    while n_ram < HELDOUT_COUNTS["ram"]:
        c = fp.c[rng.integers(len(fp.c))]
        u = rng.uniform(0, 2 * np.pi)
        d = np.array([np.sin(u), np.cos(u)])
        start, end = c - d * RAM_RUNUP, c + d * RAM_RUNUP
        if not (fp.x0 - EDGE_MARGIN <= start[0] <= fp.x1 + EDGE_MARGIN and
                fp.z0 - EDGE_MARGIN <= start[1] <= fp.z1 + EDGE_MARGIN) or not fp.clear(start, START_CLEARANCE):
            continue
        heldout.append(dict(kind="ram", sx=start[0], sz=start[1], T=RAM_T,
                            wps=[c[0], c[1], 1.0, RAM_TIMEOUTS[0], end[0], end[1], 1.0, RAM_TIMEOUTS[1]]))
        n_ram += 1
    for _ in range(HELDOUT_COUNTS["zigzag"]):
        p = np.array(inside())
        start, wps = p.copy(), []
        while len(wps) < 4 * ZIGZAG_LEGS:
            u = rng.uniform(0, 2 * np.pi)
            q = p + rng.uniform(*ZIGZAG_LEG) * np.array([np.sin(u), np.cos(u)])
            if fp.x0 <= q[0] <= fp.x1 and fp.z0 <= q[1] <= fp.z1:
                wps += [q[0], q[1], 1.0, rng.uniform(*ZIGZAG_TIMEOUT)]
                p = q
        heldout.append(dict(kind="zigzag", sx=start[0], sz=start[1], T=ZIGZAG_T, wps=wps))

    spread = []
    for k, (a, b) in enumerate(SPREAD_STEPS):
        for rep in range(2):
            spread.append(dict(kind=f"step_{a}_{b}", sx=fp.x0 - SPREAD_OFFSET - 60.0 * rep, sz=fp.z0 + 70.0 * k,
                               T=SPREAD_T, wps=[], record=True,
                               spread=[0.0, a, SPREAD_STEP_AT, a, SPREAD_STEP_AT, b]))
    return heldout, spread


def write_scenarios(path, city, source, seed, scenarios):
    """One scenario per line, so a diff of two sets reads."""
    def r(x):
        return [round(v, 3) for v in x] if isinstance(x, list) else (round(x, 3) if isinstance(x, float) else x)
    head = json.dumps(OrderedDict(city=city, source=source, seed=seed))[:-1]
    lines = [json.dumps({k: r(v) for k, v in s.items()}) for s in scenarios]
    Path(path).write_text(head + ',\n"scenarios": [\n' + ",\n".join(lines) + "\n]}\n", encoding="utf-8", newline="\n")


def load_scenarios(path):
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    return d.get("city"), d["scenarios"]


def cmd_scenarios(a):
    heldout, spread = generate(a.city, a.seed)
    SCENARIO_DIR.mkdir(exist_ok=True)
    src = f"bench.py scenarios --city {a.city} --seed {a.seed}"
    for name, sc in (("heldout", heldout), ("spread", spread)):
        path = Path(a.out_dir or SCENARIO_DIR) / f"{a.city}_{name}.json"
        if path.exists() and not a.force:
            sys.exit(f"[bench] {path} exists; --force to overwrite (results flown on it stop being comparable)")
        write_scenarios(path, a.city, src, a.seed, sc)
        print(f"[bench] {path}: {dict(Counter(s['kind'] for s in sc))}")


# ============================================================================= configs

def parse_set(spec, fields, pilot_heading):
    """'label' or 'label:field=value,...' -> {label, names, values}, validated against SwarmManager.cs."""
    label, _, assigns = spec.partition(":")
    label = label.strip()
    if not label:
        raise ValueError(f"set {spec!r} has no label")
    over = sp.parse_assignments([assigns]) if assigns else {}
    names, values = [], []
    for n, v in over.items():
        if n not in fields:
            raise ValueError(f"set '{label}': SwarmManager has no setting {n!r}")
        val = fields[n].convert(v, f"set '{label}'")
        if n == "fillLookDirectionGap" and val and not pilot_heading:
            raise ValueError(f"set '{label}' turns fillLookDirectionGap on without --pilot-heading: the fill reads "
                             "the pilot's body yaw, so it would silently do nothing")
        names.append(n)
        values.append(float(val))
    if not pilot_heading and "fillLookDirectionGap" not in names:
        names.append("fillLookDirectionGap")
        values.append(0.0)
    return OrderedDict(label=label, names=names, values=values)


def pick(scenarios, spec):
    """'transit:2,ram:1' -> the first 2 transits and the first ram, in file order."""
    if not spec:
        return scenarios
    want = {k: int(n) for k, n in (p.split(":") for p in spec.split(","))}
    taken, out = Counter(), []
    for s in scenarios:
        if taken[s["kind"]] < want.get(s["kind"], 0):
            out.append(s)
            taken[s["kind"]] += 1
    missing = {k: n - taken[k] for k, n in want.items() if taken[k] < n}
    if missing:
        raise ValueError(f"not enough scenarios of kind {missing}")
    return out


def cmd_config(a):
    sp.scene_path(a.scene).stat()   # a typo'd scene fails here, not after Unity starts
    fields = sp.swarm_manager_fields()
    scenarios = []
    for f in a.scenarios:
        city, sc = load_scenarios(f)
        if city and city != Path(a.scene).stem and not a.any_city:
            raise ValueError(f"{f} was generated for {city}, not {a.scene} (--any-city to fly it anyway)")
        scenarios += sc
    scenarios = pick(scenarios, a.pick)
    sets = [parse_set(s, fields, a.pilot_heading) for s in (a.set or ["scene"])]
    labels = [s["label"] for s in sets]
    if len(set(labels)) != len(labels):
        raise ValueError(f"duplicate set labels in {labels}")
    cfg = OrderedDict(scene=a.scene, timeScale=a.time_scale, settleTime=a.settle, altitude=a.altitude,
                      capture=a.capture, pilotHeading=a.pilot_heading, pilotYawRate=a.pilot_yaw_rate,
                      goalReplaySession=a.goal_replay or "", paramSets=sets, scenarios=scenarios)
    Path(a.out).write_text(json.dumps(cfg, indent=1) + "\n", encoding="utf-8", newline="\n")
    flights = len(scenarios) * len(sets)
    est = sum(s["T"] + a.settle for s in scenarios) * len(sets) / a.time_scale / 60
    print(f"[bench] {a.out}: {a.scene}, {len(sets)} set(s) {labels} x {len(scenarios)} scenarios = {flights} flights, "
          f"~{est:.0f} min at {a.time_scale:g}x plus ~3 min startup")


# ============================================================================= reports

def read_results(path):
    """(start, {set: params line}, [flight dicts], errors, done) from a run dir or a results.jsonl."""
    p = Path(path)
    jsonl = p / "results.jsonl" if p.is_dir() else p
    start, params, flights, errors, done = None, OrderedDict(), [], [], False
    for line in jsonl.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        ev = d.get("event")
        if ev == "start":
            start = d
        elif ev == "params":
            params[d["set"]] = d
        elif ev == "error":
            errors.append(d)
        elif ev == "done" or d.get("done"):
            done = True
        elif "scenario" in d:          # a flight; September lines carry no "event"
            flights.append(d)
    return jsonl, start, params, flights, errors, done


def traj_path(jsonl, label, index):
    new = jsonl.parent / f"traj_{label}_{index}.csv"
    old = Path(f"{jsonl}.traj_{label}_{index}.csv")
    return new if new.exists() else (old if old.exists() else None)


def step_response(traj, step_at):
    """Settled ring radius, t50, t90, overshoot % of a spread step, from a trajectory CSV (mean distance of
    the alive drones from their centroid -- the measure swarm_replica.py step uses)."""
    df = pd.read_csv(traj)
    t = df.t.values
    n = (len(df.columns) - 2) // 2
    X = df[[f"x{i}" for i in range(n)]].values
    Z = df[[f"z{i}" for i in range(n)]].values
    alive = ((df.alive_mask.values[:, None] >> np.arange(n)) & 1).astype(bool)
    cx = np.where(alive, X, np.nan)
    cz = np.where(alive, Z, np.nan)
    R = np.nanmean(np.hypot(cx - np.nanmean(cx, 1, keepdims=True), cz - np.nanmean(cz, 1, keepdims=True)), 1)
    r0 = R[(t > step_at - 3) & (t <= step_at)].mean()
    r1 = R[t > t[-1] - 5].mean()
    w = t > step_at
    if abs(r1 - r0) < 1e-6:
        return r1, np.nan, np.nan, np.nan
    frac = (R[w] - r0) / (r1 - r0)
    tt = t[w] - step_at
    reach = lambda f: tt[np.argmax(frac >= f)] if (frac >= f).any() else np.nan  # noqa: E731
    return r1, reach(0.5), reach(0.9), max(0.0, frac.max() - 1.0) * 100


def summarise_set(fl):
    df = pd.DataFrame(fl)
    lost = (df.alive_start - df.alive_end).clip(lower=0)
    km = df.path_len.sum() / 1000.0
    deaths = Counter()
    for d in df.deaths:
        deaths.update(d or {})
    track = df.track_sum.sum() / max(df.track_n.sum(), 1)
    row = OrderedDict(
        flights=len(df), lost=int(lost.sum()), contacts=int(df.contacts.sum()), hard=int(df.hard_contacts.sum()),
        other=int(df.other_contacts.sum()) if "other_contacts" in df else 0,
        clean_pct=round(100 * ((lost == 0) & (df.contacts == 0)).mean(), 1),
        contacts_per_km=round(df.contacts.sum() / km, 3) if km > 0 else np.nan,
        lost_per_km=round(lost.sum() / km, 3) if km > 0 else np.nan,
        km=round(km, 1), speed=round(df.mean_speed.mean(), 2), progress=round(track, 2),
        hull=round(df.hull_frac.mean(), 3), nn=round(df.nn_mean.mean(), 2), min_pair=round(df.min_pair.min(), 2),
        split_pct=round(100 * df.split_frac.mean(), 1))
    if "sim_per_wall" in df:
        row["x_real"] = round(df.sim_per_wall.min(), 1)
    return row, deaths, df


def cmd_report(a):
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    rows, kinds = [], []
    for path in a.runs:
        jsonl, start, params, flights, errors, done = read_results(path)
        run = Path(path).name if Path(path).is_dir() else jsonl.stem
        head = f"== {run}"
        if start:
            head += (f": {start['scene']}, {start['drones']} drones, {start['timeScale']:g}x, "
                     f"pilotHeading {start['pilotHeading']}" +
                     (f", goals from {start['goalReplaySession']}" if start.get("goalReplaySession") else ""))
        print(head + ("" if done else "   [INCOMPLETE: no done line]"))
        for e in errors:
            print(f"   ERROR (exit {e.get('code')}): {e.get('message')}")
        for label, pl in params.items():
            over = pl.get("overrides", {})
            print(f"   set '{label}': " + (", ".join(f"{k}={v}" for k, v in over.items()) or "the scene as authored"))
        timescale = start["timeScale"] if start else None
        cfg_path = jsonl.parent / "config.json"
        step_at = SPREAD_STEP_AT
        if cfg_path.exists():
            c = json.loads(cfg_path.read_text(encoding="utf-8"))
            sched = next((s["spread"] for s in c.get("scenarios", []) if s.get("spread")), None)
            if sched and len(sched) >= 4:
                step_at = sched[2]
        for label in dict.fromkeys(f["set"] for f in flights):
            fl = [f for f in flights if f["set"] == label]
            row, deaths, df = summarise_set(fl)
            rows.append(OrderedDict(run=run, set=label, **row))
            if deaths:
                print(f"   {label}: losses by reason {dict(deaths)}")
            others = Counter()
            for o in df.get("other_names", pd.Series(dtype=object)).dropna():
                others.update(o)
            if others:
                print(f"   {label}: non-obstacle contacts {dict(others.most_common(6))}")
            if timescale and "sim_per_wall" in df and df.sim_per_wall.min() < 0.5 * timescale:
                print(f"   {label}: WARNING slowest flight ran at {df.sim_per_wall.min():.1f}x against {timescale:g}x "
                      "(maximumDeltaTime overridden?)")
            for kind, g in df.groupby("kind"):
                k = OrderedDict(run=run, set=label, kind=kind, flights=len(g),
                                lost=int((g.alive_start - g.alive_end).clip(lower=0).sum()),
                                contacts=int(g.contacts.sum()), speed=round(g.mean_speed.mean(), 2))
                if kind.startswith(("step_", "spread_")):
                    resp = [step_response(tp, step_at) for tp in
                            (traj_path(jsonl, label, int(i)) for i in g.scenario) if tp is not None]
                    if resp:
                        m = np.nanmean(np.array(resp), 0)
                        k.update(R=round(m[0], 2), t50=round(m[1], 1), t90=round(m[2], 1), over_pct=round(m[3], 1))
                kinds.append(k)
    if rows:
        print()
        print(pd.DataFrame(rows).to_string(index=False))
    if kinds and a.kinds:
        print()
        print(pd.DataFrame(kinds).to_string(index=False))
    elif kinds:
        steps = [k for k in kinds if "t90" in k]
        if steps:
            print()
            print(pd.DataFrame(steps).to_string(index=False))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("scenarios", help="generate a held-out flight set and a spread-step set for a city")
    p.add_argument("--city", required=True, help="a scene with a city_obstacles_<city>.json beside analyse.py")
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--out-dir", help=f"default {SCENARIO_DIR}")
    p.add_argument("--force", action="store_true", help="overwrite existing files")
    p.set_defaults(func=cmd_scenarios)

    p = sub.add_parser("config", help="build a validated bench config")
    p.add_argument("--scene", required=True, help="scene name (Assets/Scenes/<scene>.unity) or path")
    p.add_argument("--scenarios", nargs="+", required=True, help="scenario files (scenarios/*.json)")
    p.add_argument("--pick", help="e.g. transit:2,ram:1 -- the first N of each kind (a smoke test)")
    p.add_argument("--set", action="append", metavar="LABEL[:FIELD=VALUE,...]",
                   help="a parameter set, repeatable; overrides on the scene (default: one set 'scene')")
    p.add_argument("--pilot-heading", action="store_true", help="inject a pilot heading; enables look-gap metrics")
    p.add_argument("--pilot-yaw-rate", type=float, default=90.0)
    p.add_argument("--goal-replay", metavar="SESSION", help="fly a recorded session's goal layout (stem or path)")
    p.add_argument("--time-scale", type=float, default=20.0)
    p.add_argument("--settle", type=float, default=15.0)
    p.add_argument("--altitude", type=float, default=25.0)
    p.add_argument("--capture", type=float, default=10.0, help="m from a waypoint that counts as reaching it")
    p.add_argument("--any-city", action="store_true", help="allow scenarios generated for another city")
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_config)

    p = sub.add_parser("report", help="summarise one or more bench runs")
    p.add_argument("runs", nargs="+", help="run folders (runs/<tag>) or results .jsonl files")
    p.add_argument("--kinds", action="store_true", help="also break every set down by scenario kind")
    p.set_defaults(func=cmd_report)

    a = ap.parse_args()
    try:
        a.func(a)
    except (ValueError, FileNotFoundError) as e:
        sys.exit(f"[bench] {e}")


if __name__ == "__main__":
    main()
