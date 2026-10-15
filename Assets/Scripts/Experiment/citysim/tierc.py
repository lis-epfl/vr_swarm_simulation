"""Tier C: the headless Unity bench flying the goal flights through the candidate levels' layouts -- the ground truth
for losses (PhysX contacts, DroneHealthMonitor's rules), the real hull / heading / look-gap code, and camera poses at
real altitudes. It answers G3 (the loss rate a level costs) and how many drones are left at a goal, which Tier B
assumed was all ten.

    python -B -m citysim.vissim tierC-plan [--name tierC]       # scenarios, configs and a runner on the bench root
    powershell -File <bench root>/cfg/tierC/<name>/run_tierC.ps1  # fly them (resumable: a finished tag is skipped)
    python -B -m citysim.vissim tierC [--name tierC]            # score -> experiment_results/density_study/tierC/<name>/

The flights (bench.py scenarios --goals) are Tier B''s program, so the two can be compared flight for flight
(swarm_replica.py validate replays them): from the layout's spawn along the street route at full stick to every goal,
in the layout's order and in reverse, at transit spreads 0.4 and 1.0; at each goal, arrival within 25 m and a 60 s
centring hold at spread 1.6 (a lone drone scanning at its 75 deg/s). Pilot heading on (the look-gap fill as the
scenes have it), 12.4 m (internal_3's median search altitude; flown ~1.3 m higher). Ten drones on every planned
level, one drone on the three picked ones.

Scoring, per goal hold (the hold waypoint's rows of traj_*.csv), on the samples every DT s:
  losses       per flight: drones lost (the flight is one 'run' -- three goals from the spawn, as a trial is), deaths by
               DroneHealthMonitor reason, building contact events, waypoints given up (stuck)
  formation    at the hold: alive, hull drones, shown feeds (BoundaryFeedReady), the envelopment test of Tier B' on the
               positions averaged over 20-30 s into the hold (B'-comparable) and over its last 10 s
  exposure     every walker of the goal (walkers.GoalWalkers, seeded as Tier B seeds it) recognised by a shown feed --
               camera 0.4 m ahead of the drone at its recorded height and yaw, slant range R, the gimbal's vertical
               band at each of the operator band's pitches, the 3-D slab test -- E_any, E_head (head-yaw weights about
               the recorded body yaw), first sight, as Tier B summarises them
"""
if __name__ == "__main__" and __package__ in (None, ""):
    import os.path as _p
    import sys as _s
    _s.path[0] = _p.dirname(_p.dirname(_p.abspath(__file__)))

import sys

sys.dont_write_bytecode = True

import json  # noqa: E402
import math  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from citysim import (BENCH_DIR, LAYOUT_DIR, STUDY_DIR, camera, exposure, formation, layout as lay,  # noqa: E402
                     operator_band, walkers)
from citysim.los import Boxes  # noqa: E402

DEFAULT_BENCH_ROOT = Path("D:/claude_obstacle_bench")
ALTITUDE = 12.4
TRIAL_SEEDS = (101, 102, 103)
SWEEP_SEEDS = (0, 1, 2)
DT = 0.5
ENV_EARLY = (20.0, 30.0)          # s into a hold: Tier B' measured its 30 s hover's last 10 s
ENV_LATE_S = 10.0
RUN_TIMEOUT_MIN = 30              # run_bench.ps1 -TimeoutMin per Tier C run (its default, 180, is for long sets)
# what each routing's runs fly: (scenario set name, 10-drone pick, 1-drone pick, capture radius)
ROUTINGS = {
    "street": ("goals", None, "goals_1.0:2", 10.0),          # one drone: the spread means nothing, so two orders only
    "pilot": ("goals_pilot", "goals_pilot_0.4:2,goals_pilot_1.0:2", "goals_single:2", 6.0),
}
TRANSIT_PITCH = camera.DIAL_CENTRE_PITCH_DEG      # internal_3's transit median (operator_band, Swarm transit p50)


def spread_key(kind):
    """'0.4' / '1.0' for a swarm goal flight (goals_<d> or goals_pilot_<d>), 'single' for goals_single."""
    tail = str(kind).rsplit("_", 1)[-1]
    try:
        float(tail)
        return tail
    except ValueError:
        return tail


# ============================================================================= plan

def default_levels(levels_json):
    """The three picks and each one's neighbours: in clutter (the next level up and down the sweep spec's order,
    same goal geometry) and in goal geometry (the other goal level nearest in (w_c, r_g), same clutter). That is
    the plan's "the candidate levels and their neighbours": where a pick's G3 might read differently one step
    away, and where the choice between a pick and its neighbour is close (C2-G6 / C2-G7 for L3)."""
    from citysim import vissim
    d = json.loads(Path(levels_json).read_text(encoding="utf-8"))
    spec = vissim.load_spec(vissim.SPEC_DIR / f"{d['sweep']}.json")
    pick = [v for v in d["pick"].values() if v]
    clutters = list(spec["clutter"])
    goals = {g: (float(v["w_c"]), float(v["r_g"])) for g, v in spec["goal"].items()}
    out = list(pick)
    for p in pick:
        c, g = p.split("-")
        i = clutters.index(c)
        near = [f"{clutters[j]}-{g}" for j in (i - 1, i + 1) if 0 <= j < len(clutters)]
        others = [h for h in goals if h != g]
        if others:
            h = min(others, key=lambda h: math.dist(goals[h], goals[g]))
            near.append(f"{c}-{h}")
        out += [lv for lv in near if lv not in out]
    return out, d["pick"]


def trial_layout(level, seed):
    """layouts/<L>_<level>__<id>.json for a picked level's trial seed, or None."""
    hits = sorted(LAYOUT_DIR.glob(f"L?_{level}__*_seed{seed}.json"))
    return hits[0] if hits else None


def sweep_layout(sweep, level, seed):
    lays = pd.read_csv(STUDY_DIR / "tierB" / sweep / "layouts.csv")
    r = lays[(lays.level == level) & (lays.seed == seed)]
    return Path(r.path.iloc[0]) if len(r) else None


def cmd_plan(a):
    bench_root = Path(a.bench_root)
    levels, pick = default_levels(STUDY_DIR / "levels.json")
    if a.levels:
        levels = a.levels
    picked = [v for v in pick.values() if v]
    cfg_dir = bench_root / "cfg" / "tierC" / a.name
    scen_dir = bench_root / "scenarios_tierC"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    scen_dir.mkdir(parents=True, exist_ok=True)
    py = sys.executable
    bench = str(BENCH_DIR / "bench.py")
    set_name, pick10, pick1, capture = ROUTINGS[a.routing]
    runs = []
    for level in levels:
        is_pick = level in picked
        for k, seed in enumerate(TRIAL_SEEDS if is_pick else SWEEP_SEEDS):
            path = trial_layout(level, seed) if is_pick else sweep_layout(a.sweep, level, seed)
            if path is None or not path.exists():
                print(f"[tierC] no layout for {level} seed {seed}; skipped")
                continue
            L = lay.load(path)
            scen = scen_dir / f"{L.id}_{set_name}.json"
            if not scen.exists():
                subprocess.run([py, "-B", bench, "scenarios", "--city", str(path), "--goals", "--routing", a.routing,
                                "--out-dir", str(scen_dir)], check=True, cwd=str(BENCH_DIR))
            for drones in ((10, 1) if is_pick else (10,)):
                tag = f"tierC_{a.name}_{level}_s{seed}_n{drones}"
                cfg = cfg_dir / f"{tag}.json"
                cmd = [py, "-B", bench, "config", "--scene", "ProceduralCityWorld", "--layout", str(path),
                       "--drones", str(drones), "--pilot-heading", "--altitude", str(a.altitude),
                       "--time-scale", str(a.time_scale), "--scenarios", str(scen), "--capture", str(capture),
                       "--out", str(cfg)]
                pk = pick1 if drones == 1 else pick10
                if pk:
                    cmd += ["--pick", pk]
                if getattr(a, "record_all", False):
                    cmd += ["--record-all"]
                subprocess.run(cmd, check=True, cwd=str(BENCH_DIR), stdout=subprocess.DEVNULL)
                runs.append(dict(tag=tag, config=str(cfg), level=level, seed=seed, drones=drones, layout=L.id,
                                 layout_path=str(path), picked=is_pick))
    out = STUDY_DIR / "tierC" / a.name
    out.mkdir(parents=True, exist_ok=True)
    plan = dict(generated=time.strftime("%Y-%m-%dT%H:%M:%S"), name=a.name, sweep=a.sweep, bench_root=str(bench_root),
                pick=pick, levels=levels, altitude=a.altitude, time_scale=a.time_scale, routing=a.routing,
                capture=capture, runs=runs)
    (out / "plan.json").write_text(json.dumps(plan, indent=1) + "\n", encoding="utf-8")
    write_runner(cfg_dir, bench_root, runs)
    print(f"[tierC] {len(runs)} runs over {len(levels)} levels -> {out / 'plan.json'}\n"
          f"  fly them: powershell -NoProfile -ExecutionPolicy Bypass -File {cfg_dir / 'run_tierC.ps1'}")


def write_runner(cfg_dir, bench_root, runs):
    """A PowerShell loop over the configs: one Unity at a time on the bench copy, synced once, finished tags skipped,
    a log line per run."""
    lines = [
        "# Generated by citysim tierC-plan: fly every Tier C config on the bench copy, one at a time. Re-run to resume.",
        f"$bench = '{BENCH_DIR / 'run_bench.ps1'}'",
        f"$runs = '{bench_root / 'runs'}'",
        f"$log = '{cfg_dir / 'run_tierC.log'}'",
        "Add-Content -Path $log -Value \"start $(Get-Date -Format o)\" -Encoding utf8",
        "$synced = $false",
        "$todo = @(",
    ]
    lines += [f"    @('{r['tag']}', '{r['config']}')," for r in runs]
    lines[-1] = lines[-1].rstrip(",")
    lines += [
        ")",
        "foreach ($r in $todo) {",
        "    $tag, $cfg = $r[0], $r[1]",
        "    if (Test-Path (Join-Path $runs \"$tag\\results.jsonl\")) {",
        "        if (Select-String -Path (Join-Path $runs \"$tag\\results.jsonl\") -Pattern '\"event\": *\"done\"' -Quiet) {",
        "            Add-Content -Path $log -Value \"skip $tag (done)\" -Encoding utf8; continue",
        "        }",
        "        Rename-Item (Join-Path $runs $tag) \"$tag.partial_$(Get-Date -Format yyyyMMdd_HHmmss)\"",
        "    }",
        "    # A run takes 2-7 min; a Unity that hangs at startup (it has, after an OpenXR exception) is killed at",
        f"    # {RUN_TIMEOUT_MIN} min and the run tried once more, its partial folder set aside.",
        "    for ($attempt = 1; $attempt -le 2; $attempt++) {",
        f"        if ($synced) {{ $out = & $bench -NoSync -TimeoutMin {RUN_TIMEOUT_MIN} -Config $cfg -Tag $tag 2>&1 }}",
        f"        else {{ $out = & $bench -TimeoutMin {RUN_TIMEOUT_MIN} -Config $cfg -Tag $tag 2>&1 }}",
        "        $code = $LASTEXITCODE",
        "        $synced = $true",
        "        Add-Content -Path $log -Value \"=== $tag exit $code attempt $attempt $(Get-Date -Format o)\" -Encoding utf8",
        "        Add-Content -Path $log -Value ($out | Select-Object -Last 4 | Out-String) -Encoding utf8",
        "        if ($code -eq 0) { break }",
        "        if (Test-Path (Join-Path $runs \"$tag\\results.jsonl\")) {",
        "            Rename-Item (Join-Path $runs $tag) \"$tag.failed_$(Get-Date -Format yyyyMMdd_HHmmss)\"",
        "        }",
        "    }",
        "}",
        "Add-Content -Path $log -Value \"done $(Get-Date -Format o)\" -Encoding utf8",
    ]
    (cfg_dir / "run_tierC.ps1").write_text("\n".join(lines) + "\n", encoding="utf-8-sig")


# ============================================================================= scoring

def read_run(run_dir):
    run_dir = Path(run_dir)
    cfg = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    start, flights = None, []
    for line in (run_dir / "results.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        if d.get("event") == "start":
            start = d
        elif d.get("event") == "flight":
            flights.append(d)
    lpath = run_dir / "layout.json"
    L = lay.load(lpath if lpath.exists() else lay.resolve(cfg["layout"]))
    return cfg, start, flights, L


def read_traj(path):
    df = pd.read_csv(path)
    n = sum(1 for c in df.columns if c.startswith("x") and c[1:].isdigit())
    bits = np.arange(n)

    def mask(col):
        return ((df[col].values[:, None].astype(np.int64) >> bits) & 1).astype(bool)

    return dict(t=df.t.values, X=df[[f"x{i}" for i in range(n)]].values, Z=df[[f"z{i}" for i in range(n)]].values,
                Y=df[[f"y{i}" for i in range(n)]].values, yaw=df[[f"yaw{i}" for i in range(n)]].values,
                alive=mask("alive_mask"), hull=mask("hull_mask"), shown=mask("shown_mask"),
                wp=df.wp.values.astype(int), n=n,
                body=df.bodyYaw.values if "bodyYaw" in df else np.full(len(df), np.nan))   # pre-column runs: NaN


def hold_goals(sc, L):
    """[(hold waypoint index, goal dict)] in flight order: each hold waypoint matched to its goal centre."""
    wps = sc.get("wps") or []
    goals = list(L.goals())
    centres = np.array([g["centre"] for g in goals], float)
    out = []
    for k in range(len(wps) // 4):
        if wps[4 * k + 2] < 0:
            gi = int(np.argmin(np.linalg.norm(centres - [wps[4 * k], wps[4 * k + 1]], axis=1)))
            out.append((k, goals[gi]))
    return out


def popcount_mean(m, rows):
    return float(m[rows].sum(1).mean()) if len(rows) else math.nan


def score_hold(tr, rows, g, L, B3, pitches, head_yaws, head_w, single):
    """One goal hold's formation and exposure rows."""
    from citysim import vissim
    info = vissim.goal_info(L, g)
    t_rows = tr["t"][rows]
    t_rel = t_rows - t_rows[0]
    alive = tr["alive"][rows]
    P = np.stack([tr["X"][rows], tr["Z"][rows]], -1)                              # (T, n, 2)
    centre = info["centre"]

    def env_at(lo, hi):
        sel = (t_rel >= lo) & (t_rel <= hi)
        if not sel.any():
            return dict(surrounds=False, max_gap_deg=360.0, inside_frac=math.nan, enveloped=False, n_alive=0)
        live = alive[sel].all(0)
        Q = np.nanmean(P[sel][:, live], axis=0) if live.any() else np.zeros((0, 2))
        e = formation.envelopment(Q, centre) if len(Q) >= 3 else dict(surrounds=False, max_gap_deg=360.0)
        r = np.linalg.norm(Q - centre, axis=1) if len(Q) else np.zeros(0)
        inside = float((r <= info["c_g"]).mean()) if len(r) else math.nan
        ok = bool(e["surrounds"] and e["max_gap_deg"] <= formation.MAX_GAP_DEG and inside >= formation.INSIDE_FRAC)
        return dict(surrounds=bool(e["surrounds"]), max_gap_deg=round(float(e["max_gap_deg"]), 1),
                    inside_frac=round(inside, 3), enveloped=ok, n_alive=int(live.sum()))

    e30 = env_at(*ENV_EARLY)
    eend = env_at(t_rel[-1] - ENV_LATE_S, t_rel[-1])
    base = dict(goal=info["index"], w_c=info["w_c"], r_g=info["r_g"], c_g=info["c_g"], hold_s=round(float(t_rel[-1]), 1),
                alive=popcount_mean(tr["alive"], rows), hull=popcount_mean(tr["hull"] & tr["alive"], rows),
                shown=popcount_mean(tr["shown"] & tr["alive"], rows),
                enveloped_30=e30["enveloped"], max_gap_30=e30["max_gap_deg"], inside_30=e30["inside_frac"],
                enveloped_end=eend["enveloped"], max_gap_end=eend["max_gap_deg"], inside_end=eend["inside_frac"],
                centroid_offset_m=round(float(np.linalg.norm(np.nanmean(np.where(alive[..., None], P, np.nan),
                                                                         axis=(0, 1)) - centre)), 2))
    # ---- exposure on the DT clock
    step = max(1, int(round(DT / max(np.median(np.diff(tr["t"])), 1e-6))))
    rs = rows[::step]
    t_s = tr["t"][rs] - tr["t"][rs[0]]
    yaw = tr["yaw"][rs]
    y = np.radians(yaw)
    pos = np.stack([tr["X"][rs], tr["Z"][rs]], -1) + camera.CAMERA_FORWARD_M * np.stack([np.sin(y), np.cos(y)], -1)
    valid = tr["alive"][rs] & (tr["shown"][rs] | single)
    body = tr["body"][rs]
    if not np.isfinite(body).all():      # a run from before the bodyYaw column: the pilot faced the goal on arrival
        c0 = np.nanmean(np.where(tr["alive"][rows[0]][:, None], np.c_[tr["X"][rows[0]], tr["Z"][rows[0]]], np.nan), 0)
        body = np.full(len(rs), math.degrees(math.atan2(centre[0] - c0[0], centre[1] - c0[1])))
    rng = np.random.default_rng(10_000 * int(L.layout["seed"]) + 17 * info["index"] + 3)
    W = walkers.GoalWalkers(info["ring_rects"], rng)
    pts = W.positions(t_s)
    near = B3.subset(B3.near(centre, camera.R_ID_M + 120.0))
    rows_out = []
    for pit in pitches:
        track = exposure.Track(pos=pos, yaw=yaw, h=tr["Y"][rs], valid=valid, pitch=np.full(len(rs), float(pit)),
                               body_yaw=body, dt=float(np.median(np.diff(t_s))) if len(t_s) > 1 else DT,
                               single=single)
        vis, off = exposure.recognised_3d(near, track, pts)
        m = exposure.summarise(vis, off, track, None if single else head_yaws, None if single else head_w)
        rows_out.append(dict(base, pitch=pit, **{k: v for k, v in m.items() if not isinstance(v, np.ndarray)}))
    return rows_out


def score_transit(tr, sc, L, holds, single, height=None):
    """citysim.transit's per-flight measures of a bench flight: its legs between goal holds, the feeds the bench showed
    (BoundaryFeedReady), the camera at the flight's median altitude (or `height`, which the replica's transit gate
    passes so both sides are scored at one) and the transit-phase gimbal pitch, and the waypoint pilot's stick rebuilt
    from its waypoints (the bench does not log it)."""
    from citysim import transit, vissim
    from citysim.los import Footprints
    wps = sc.get("wps") or []
    n_wp = len(wps) // 4
    hold_idx = [k for k, _ in holds]
    goal_of, prev = {}, -1
    for k, g in holds:
        for w in range(prev + 1, k + 1):
            goal_of[w] = int(g["index"])
        prev = k
    wp = tr["wp"]
    leg = np.array([goal_of.get(int(w), -1) for w in wp])
    in_transit = ~np.isin(wp, hold_idx) & (wp < n_wp) & (leg >= 0)
    X, Z, A = tr["X"], tr["Z"], tr["alive"]
    with np.errstate(all="ignore"):
        cx = np.nanmean(np.where(A, X, np.nan), axis=1)
        cz = np.nanmean(np.where(A, Z, np.nan), axis=1)
    # the scripted stick: at the current waypoint from the centroid, its magnitude; the heading the pilot faced
    tx = np.array([wps[4 * int(w)] if w < n_wp else np.nan for w in wp])
    tz = np.array([wps[4 * int(w) + 1] if w < n_wp else np.nan for w in wp])
    mag = np.array([abs(wps[4 * int(w) + 2]) if w < n_wp else 0.0 for w in wp])
    d = np.c_[tx - cx, tz - cz]
    nd = np.maximum(np.linalg.norm(d, axis=1), 1e-9)
    stick = d / nd[:, None] * np.minimum(mag, 1.0)[:, None]
    heading = tr["body"] if np.isfinite(tr["body"]).all() else np.degrees(np.arctan2(stick[:, 0], stick[:, 1]))
    cmd = np.c_[tr["t"], stick, heading, np.where(in_transit, leg, -1 - np.maximum(leg, 0))]
    cmd = cmd[np.isfinite(cmd).all(1)]
    legs = []
    for k, g in holds:
        rows = np.flatnonzero(in_transit & (leg == int(g["index"])))
        hold_rows = np.flatnonzero(wp == k)
        if not len(rows) or not len(hold_rows):
            continue
        t0, t1 = float(tr["t"][rows[0]]), float(tr["t"][hold_rows[0]])
        legs.append(dict(leg=int(g["index"]), t0=t0, t1=t1, start=np.array([cx[rows[0]], cz[rows[0]]]),
                         goal=np.array(g["centre"], float), arrive_m=float(sc.get("arrive") or 25.0)))
    step = max(1, int(round(DT / max(float(np.median(np.diff(tr["t"]))), 1e-6))))
    keep = np.arange(0, len(tr["t"]), step)
    thin = {k: (v[keep] if isinstance(v, np.ndarray) and v.ndim >= 1 and len(v) == len(tr["t"]) else v)
            for k, v in tr.items()}
    thin["leg"] = leg[keep]
    key = spread_key(sc.get("kind", ""))
    d_ref = float(key) if key.replace(".", "", 1).isdigit() else 1.0
    rec = transit.FlightRecord.from_bench(thin, dict(interface="single" if single else "swarm", d_ref=d_ref), single,
                                          in_transit[keep], legs)
    rec.cmd = cmd
    fp = Footprints(L.boxes_as_tuples())
    rows = rec.transit_rows()
    if not len(rows):
        return {}
    flown_h = float(np.nanmedian(np.where(A[keep][rows], tr["Y"][keep][rows], np.nan)))
    height = flown_h if height is None else float(height)
    cells = transit.Cells.around(fp, [rec.centroid[rows]], max(transit.RANGES) + 4.0)
    s = transit.score(rec, fp, cells, height, TRANSIT_PITCH, layout=L.id, goals=vissim.goal_loops(L))
    s.pop("firsts", None)
    s.pop("interface", None)
    s["transit_height_m"] = round(flown_h, 2)
    return s


def score_flight(job):
    run_dir, meta, fi, flight, sc, L_path, pitches, head_yaws, head_w = job
    L = lay.load(L_path)
    traj = Path(run_dir) / f"traj_{flight['set']}_{int(flight['scenario'])}.csv"
    frow = dict(meta, flight=fi, scenario=int(flight["scenario"]), kind=flight["kind"], order=sc.get("order", ""),
                alive_start=flight["alive_start"], alive_end=flight["alive_end"],
                lost=int(flight["alive_start"]) - int(flight["alive_end"]), contacts=flight.get("contacts", 0),
                hard_contacts=flight.get("hard_contacts", 0), path_km=round(flight.get("path_len", 0.0) / 1000, 3),
                sim_s=flight.get("sim_s"), wp_total=len(sc.get("wps") or []) // 4, wp_done=flight.get("wp_done"),
                min_pair=flight.get("min_pair"), nn_mean=flight.get("nn_mean"), hull_frac=flight.get("hull_frac"),
                mean_speed=flight.get("mean_speed"), sim_per_wall=flight.get("sim_per_wall"),
                deaths=";".join(f"{k}={v}" for k, v in sorted((flight.get("deaths") or {}).items())))
    for k, v in (flight.get("deaths") or {}).items():
        frow[f"death_{k}"] = v
    frow["stuck_wps"] = frow["wp_total"] - (frow["wp_done"] or 0)
    frow["spread_key"] = spread_key(flight["kind"])
    if not traj.exists():
        return frow, []
    tr = read_traj(traj)
    single = tr["n"] == 1
    holds = hold_goals(sc, L)
    hold_set = {k for k, _ in holds}
    transit = np.flatnonzero(~np.isin(tr["wp"], list(hold_set)) & (tr["wp"] < frow["wp_total"]))
    frow["alive_transit"] = popcount_mean(tr["alive"], transit)
    frow["hull_transit"] = popcount_mean(tr["hull"] & tr["alive"], transit)
    frow["shown_transit"] = popcount_mean(tr["shown"] & tr["alive"], transit)
    try:
        frow.update(score_transit(tr, sc, L, holds, single))
    except Exception as e:  # noqa: BLE001 -- the transit scores are added value; the flight's row stands without them
        frow["transit_error"] = f"{type(e).__name__}: {e}"
    B3 = Boxes(L.boxes_as_tuples())
    grows = []
    for order_k, (k, g) in enumerate(holds):
        rows = np.flatnonzero(tr["wp"] == k)
        if len(rows) < 4:
            grows.append(dict(meta, flight=fi, kind=flight["kind"], order=sc.get("order", ""), visit=order_k,
                              goal=g["index"], reached=False))
            continue
        for r in score_hold(tr, rows, g, L, B3, pitches, head_yaws, head_w, single):
            grows.append(dict(meta, flight=fi, kind=flight["kind"], order=sc.get("order", ""), visit=order_k,
                              reached=True, **r))
    return frow, grows


def cmd_score(a):
    out = STUDY_DIR / "tierC" / a.name
    plan = json.loads((out / "plan.json").read_text(encoding="utf-8"))
    band = operator_band.load("internal_3")
    head_yaws, head_w = operator_band.head_weights(band)
    pitches = operator_band.grid(band)["pitch"]
    runs_root = Path(plan["bench_root"]) / "runs"
    jobs, missing = [], []
    for r in plan["runs"]:
        run_dir = runs_root / r["tag"]
        if not (run_dir / "results.jsonl").exists():
            missing.append(r["tag"])
            continue
        cfg, start, flights, L = read_run(run_dir)
        lpath = run_dir / "layout.json"
        meta = dict(level=r["level"], seed=r["seed"], drones=r["drones"], layout=r["layout"], picked=r["picked"],
                    run=r["tag"])
        for fi, f in enumerate(flights):
            jobs.append((str(run_dir), meta, fi, f, cfg["scenarios"][int(f["scenario"])],
                         str(lpath if lpath.exists() else r["layout_path"]), pitches, head_yaws, head_w))
    if missing:
        print(f"[tierC] {len(missing)} planned run(s) not flown yet: {', '.join(missing[:6])}"
              f"{' ...' if len(missing) > 6 else ''}")
    if not jobs:
        raise SystemExit("[tierC] nothing to score")
    t0 = time.time()
    # Scoring a flight's transit takes minutes, so each flight's result is cached against its trajectory file
    # (size and mtime) and the scoring code; a plan scored again re-scores only what was re-flown.
    import hashlib
    import pickle
    from citysim import discovery, transit
    cache = out / "cache"
    cache.mkdir(exist_ok=True)
    code = hashlib.sha1(b"".join(Path(m.__file__).read_bytes()
                                 for m in (sys.modules[__name__], transit, discovery, exposure)))
    res, todo, keys = [None] * len(jobs), [], []
    for i, j in enumerate(jobs):
        traj = Path(j[0]) / f"traj_{j[3]['set']}_{int(j[3]['scenario'])}.csv"
        st = traj.stat() if traj.exists() else None
        key = hashlib.sha1(repr((j[0], sorted(j[1].items()), j[2], st.st_size if st else 0,
                                 st.st_mtime_ns if st else 0, j[6], code.hexdigest())).encode()).hexdigest()
        f = cache / f"{key}.pkl"
        if f.exists():
            res[i] = pickle.loads(f.read_bytes())
        else:
            todo.append(i)
        keys.append(f)
    if todo:
        from citysim.parallel import run_jobs
        print(f"[tierC] scoring {len(todo)} flight(s), {len(jobs) - len(todo)} from the cache", flush=True)
        for i, r in zip(todo, run_jobs(score_flight, [jobs[i] for i in todo], "tierC", a.jobs)):
            res[i] = r
            keys[i].write_bytes(pickle.dumps(r))
    flights = pd.DataFrame([r[0] for r in res])
    goals = pd.DataFrame([g for r in res for g in r[1]])
    flights.to_csv(out / "tierC_flights.csv", index=False)
    goals.to_csv(out / "tierC_goals.csv", index=False)
    weights = transit_weights(band)
    lev = summarise_levels(flights, goals, weights)
    lev.to_csv(out / "tierC_levels.csv", index=False)
    (out / "manifest.json").write_text(json.dumps(dict(scored=time.strftime("%Y-%m-%dT%H:%M:%S"), flights=len(flights),
                                                       runs=len(plan["runs"]) - len(missing), missing=missing,
                                                       pitches=pitches, dt=DT, transit_weights=weights,
                                                       seconds=round(time.time() - t0)),
                                                  indent=1) + "\n", encoding="utf-8")
    pd.set_option("display.width", 220)
    print(lev.round(3).to_string(index=False))
    print(f"[tierC] {len(flights)} flights scored in {time.time() - t0:.0f} s -> {out}")


def transit_weights(band):
    """How much of a run's transit internal_3's swarm pilots flew nearest each spread Tier C flies (0.4, 1.0): the
    band's transit spread shares, with the little time nearest 1.6 counted at 1.0 (the nearer flown spread)."""
    sh = operator_band.spread_share(band, "Swarm", "transit")
    if not sh:
        return {"0.4": 0.5, "1.0": 0.5}
    w04 = float(sh.get("0.4", 0.0))
    return {"0.4": round(w04, 4), "1.0": round(1.0 - w04, 4)}


def per_run_losses(sw, weights):
    """Drones lost per run the way pilots fly: each transit spread's mean loss per goal flight, weighted by the share
    of transit time internal_3's pilots spent nearest it. The goal flights are the trial's run (spawn, three goals);
    G3's bar, 2.5 per run, is internal_3's rate under the same spread use, so this is the like-for-like figure."""
    keys = sw.spread_key if "spread_key" in sw else sw.kind.map(spread_key)
    rates = {k: sw[keys == k].lost.mean() for k in weights}
    if any(not np.isfinite(v) for v in rates.values()):
        return float(sw.lost.mean()) if len(sw) else math.nan
    return float(sum(weights[k] * rates[k] for k in weights))


def summarise_levels(flights, goals, weights=None):
    """One row per level (ten drones), with the single drone's columns beside it where it flew. loss_rate is what
    select_levels' G3 reads: drones lost per run at internal_3's transit-spread use (per_run_losses), its SD over
    layout seeds beside it; loss_rate_flown is the plain mean over the flights as flown (half at 0.4, half at 1.0),
    and loss_rate_04 / _10 the two spreads apart. Also what a Tier B re-run needs: alive and hull drones at a goal."""
    weights = weights or {"0.4": 0.5, "1.0": 0.5}
    pit = camera.DIAL_CENTRE_PITCH_DEG
    rows = []
    for level, f in flights.groupby("level"):
        sw = f[f.drones > 1]
        sg = f[f.drones == 1]
        g = goals[(goals.level == level) & goals.get("reached", True).astype(bool)] if len(goals) else goals
        gs = g[(g.drones > 1) & np.isclose(g.pitch, pit)] if len(g) else g
        g1 = g[(g.drones == 1) & np.isclose(g.pitch, pit)] if len(g) else g
        by_seed = [per_run_losses(x, weights) for _, x in sw.groupby("seed")]
        keys = sw.kind.map(spread_key) if len(sw) else pd.Series(dtype=object)
        row = dict(level=level, flights=len(sw), runs=sw.run.nunique() if len(sw) else 0,
                   loss_rate=per_run_losses(sw, weights) if len(sw) else math.nan,
                   loss_rate_sd=float(np.std(by_seed, ddof=1)) if len(by_seed) > 1 else math.nan,
                   loss_rate_flown=float(sw.lost.mean()) if len(sw) else math.nan,
                   loss_rate_04=float(sw[keys == "0.4"].lost.mean()) if len(sw) else math.nan,
                   loss_rate_10=float(sw[keys == "1.0"].lost.mean()) if len(sw) else math.nan,
                   w_04=weights["0.4"], w_10=weights["1.0"],
                   contacts_per_flight=float(sw.contacts.mean()) if len(sw) else math.nan,
                   contacts_per_km=float(sw.contacts.sum() / max(sw.path_km.sum(), 1e-9)) if len(sw) else math.nan,
                   stuck_wps=float(sw.stuck_wps.mean()) if len(sw) else math.nan,
                   flight_min=float(sw.sim_s.mean() / 60) if len(sw) else math.nan,
                   min_pair_m=float(sw.min_pair.min()) if len(sw) else math.nan,
                   nn_m=float(sw.nn_mean.mean()) if len(sw) else math.nan,
                   alive_goal=float(gs.alive.mean()) if len(gs) else math.nan,
                   hull_goal=float(gs.hull.mean()) if len(gs) else math.nan,
                   shown_goal=float(gs.shown.mean()) if len(gs) else math.nan,
                   hull_transit=float(sw.hull_transit.mean()) if "hull_transit" in sw and len(sw) else math.nan,
                   enveloped_30=float(gs.enveloped_30.mean()) if len(gs) else math.nan,
                   enveloped_end=float(gs.enveloped_end.mean()) if len(gs) else math.nan,
                   swarm_E_any_C=float(gs.E_any.mean()) if len(gs) else math.nan,
                   swarm_E_head_C=float(gs.E_head.mean()) if len(gs) else math.nan,
                   single_flights=len(sg),
                   single_loss_rate=float(sg.lost.mean()) if len(sg) else math.nan,
                   single_contacts_per_flight=float(sg.contacts.mean()) if len(sg) else math.nan,
                   single_E_any_C=float(g1.E_any.mean()) if len(g1) else math.nan)
        deaths = [c for c in sw.columns if c.startswith("death_")]
        for c in deaths:
            row[c] = float(sw[c].fillna(0).mean())
        # transit (citysim.transit on the bench flights): the swarm at spread 1.0 and the single drone
        sw10 = sw[keys == "1.0"] if len(sw) else sw
        for who, df_ in (("swarm", sw10), ("single", sg)):
            for c in TRANSIT_SUMMARY:
                if c in df_ and len(df_):
                    row[f"{who}_{c}_C"] = float(pd.to_numeric(df_[c], errors="coerce").mean())
            disc = [c for c in df_.columns if c.startswith("disc_g") and c.endswith("_any_r80_s")] if len(df_) else []
            if disc:
                row[f"{who}_discovery_s_C"] = float(np.nanmean(df_[disc].to_numpy(float)))
        rows.append(row)
    return pd.DataFrame(rows)


TRANSIT_SUMMARY = ("path_ratio", "street_frac", "adj_per_100m", "cmd_straightness", "morph_per_km",
                   "sweep_any_r40_m", "sweep_head_r40_m", "area_any_r40_per_m", "behind_corner_r40_frac",
                   "progress_ms")


# ============================================================================= Tier B at Tier C's alive count

def cmd_alive(a):
    """Tier B's swarm exposure and advantage re-read at the number of drones Tier C found alive at a goal: the
    sweep (vissim sweep --levels <picks> --drones 8 9 10) flies the same goals with fewer drones, and each level's
    value is interpolated linearly in the drone count at Tier C's alive_goal. The single drone is unchanged."""
    from citysim import checks
    exp = pd.read_csv(STUDY_DIR / "tierB" / a.sweep / "exposure.csv")
    lev = pd.read_csv(STUDY_DIR / "tierC" / a.name / "tierC_levels.csv").set_index("level")
    singles = exp[exp.interface == "single"]
    ns = sorted(int(n) for n in exp.n_drones.dropna().unique())
    rows = []
    for level in sorted(set(exp.level) & set(lev.index)):
        alive = float(lev.loc[level, "alive_goal"])
        if not np.isfinite(alive):
            continue
        per_n = {}
        for n in ns:
            sw = exp[(exp.interface == "swarm") & (exp.level == level) & (exp.n_drones == n)]
            if len(sw):
                per_n[n] = checks.level_deltas(pd.concat([sw, singles[singles.level == level]]), "pilot")
        if not per_n:
            continue
        x = min(max(alive, min(per_n)), max(per_n))
        lo = max(n for n in per_n if n <= x)
        hi = min(n for n in per_n if n >= x)
        f = 0.0 if hi == lo else (x - lo) / (hi - lo)
        a_, b_ = per_n[lo].set_index("seed"), per_n[hi].set_index("seed")
        delta = (1 - f) * a_.delta + f * b_.delta
        swarm = (1 - f) * a_.swarm + f * b_.swarm
        full = per_n.get(10)
        rows.append(dict(level=level, alive_goal=round(alive, 2), n_lo=lo, n_hi=hi,
                         swarm_E_head_alive=float(swarm.mean()), delta_alive=float(delta.mean()),
                         delta_alive_sd=float(delta.std()) if len(delta) > 1 else math.nan,
                         delta_10=float(full.delta.mean()) if full is not None else math.nan,
                         **{f"delta_n{n}": float(d.delta.mean()) for n, d in per_n.items()}))
    out = pd.DataFrame(rows)
    path = STUDY_DIR / "tierC" / a.name / "tierB_at_alive.csv"
    out.to_csv(path, index=False)
    print(out.round(4).to_string(index=False))
    print(f"[tierC] written {path}")

