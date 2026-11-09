"""Swarm formations from the 2-D replica (Assets/Scripts/swarm/Bench/swarm_replica.py): where the drones actually
sit, as the force law puts them, rather than an assumed ring.

    python -B -m citysim.formation diameters                       # hover ring diameter per d_ref -> data/
    python -B -m citysim.formation envelope --out envmap.csv       # the envelopment map over (w_c, r_g, d_ref, n)
    python -B -m citysim.vissim gates                              # the gate probe -> data/w_pass.json (routing)

hover_template   the pattern run_step uses: random start in open sky, stick (0, 0), d_ref held, 40 s to settle.
envelop          a goal patch (occluder + ring, citygen.goal_buildings) or a layout's goal, flown with a centring
                 pilot (a P-controller on the centroid, stick <= PILOT_MAX) holding the formation over the goal
                 centre. start="arrive": a settled hover template starts ARRIVE_M out on the spawn side and is
                 brought in (the envelopment map); start="placed": the template starts on the goal centre, any drone
                 inside a footprint pushed radially clear of it (the Tier B templates: where the force law holds the
                 formation once the pilot has it there, without the transit through the city). The settled
                 positions are the swarm's template at that goal. Enveloped = no drone lost, the goal centre inside
                 the drones' convex hull, no bearing gap round it wider than MAX_GAP_DEG and, given the patch's
                 clear radius, at least INSIDE_FRAC of the drones inside it -- between the occluder and the ring,
                 looking out at the ring; surrounding the whole patch from outside the ring is not envelopment.
gate_map         what the formation flows through, for the pilots of citysim.routing: a wall with one gap of a set
                 width between avoidance cylinders, and a lone building straight ahead ("ram"), flown at full stick
                 at each spread and by a lone drone. w_pass is the narrowest gap passed with nothing lost.
Contacts         the replica kills a drone that enters a building footprint; in Unity that is a bump (only a
                 drone-drone approach under 0.5 m, a flip, the ground or isolation kill). swarm_replica.ContactSim
                 keeps the replica's pair kills but turns a footprint entry into a CONTACT -- the drone is put back
                 on the footprint's surface with the inward velocity removed, and counted -- and loses a drone whose
                 nearest neighbour is beyond the cohesion cutoff, as DroneHealthMonitor does.
The replica's force law is checked against git (swarm_replica.check_force_law) before any formation is flown.
"""
if __name__ == "__main__" and __package__ in (None, ""):
    import os.path as _p
    import sys as _s
    _s.path[0] = _p.dirname(_p.dirname(_p.abspath(__file__)))

import sys

sys.dont_write_bytecode = True

import argparse  # noqa: E402
import itertools  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import os  # noqa: E402
from dataclasses import replace  # noqa: E402
from multiprocessing import Pool  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

from citysim import PACKAGE_DIR, use_bench_modules  # noqa: E402

use_bench_modules()
import swarm_params as sp  # noqa: E402
import swarm_replica as rep  # noqa: E402

SCENE = "DiamondCityWorld"          # whose SwarmManager values (the internal_3 city: core + damper on)
SETTLE_S = 40.0
ARRIVE_M = 60.0
PILOT_GAIN = 0.25                   # 1/s: commanded speed per metre of centroid error
PILOT_MAX = 0.5                     # stick magnitude
ENVELOP_S = 75.0                    # arrival + settling
MEASURE_S = 10.0                    # averaged at the end
MAX_GAP_DEG = 150.0
INSIDE_FRAC = 0.7
DREFS = (0.4, 0.55, 0.7, 0.85, 1.0, 1.15, 1.3, 1.45, 1.6)
DATA_DIR = PACKAGE_DIR / "data"
_CHECKED = False


def check(allow_stale=False):
    global _CHECKED
    if not _CHECKED:
        rep.check_force_law(allow_stale)
        _CHECKED = True


# swarm_replica.ContactSim: footprint entries as contacts (and isolation losses), what every tier here flies. It
# lived here until the replica's validate gate needed the same model.
ContactSim = rep.ContactSim
push_out = rep.push_out


def model(scene=SCENE, overrides=None):
    return rep.Model(dict(sp.resolve_swarm(scene, overrides=overrides or {})), dict(sp.resolve_airframe()))


def hover_template(d_ref, n=10, seed=0, scene=SCENE, settle=SETTLE_S, overrides=None):
    """(positions (k, 2) about the centroid, alive mask (n,)) after settling at hover in open sky."""
    check()
    m = model(scene, overrides)
    rng = np.random.default_rng(1000 + seed)
    sim = rep.Sim(m, rep.City([]), rng.normal(0, 6.0, (n, 2)), np.zeros((n, 2)), seed)
    while sim.t < settle:
        sim.step(np.zeros(2), 0.0, d_ref)
    P = sim.P[sim.alive]
    return P - P.mean(0), sim.alive.copy()


def ring_diameter(P):
    """2 x the mean distance of the hull drones from the centroid."""
    from citysim.display import convex_hull
    if len(P) < 3:
        return 2 * float(np.linalg.norm(P - P.mean(0), axis=1).max()) if len(P) else 0.0
    h = convex_hull(P)
    return 2 * float(np.linalg.norm(P[h] - P.mean(0), axis=1).mean())


def write_diameters(n=10, seeds=3, scene=SCENE):
    rows = {}
    for d in DREFS:
        ds = [ring_diameter(hover_template(d, n, s, scene)[0]) for s in range(seeds)]
        rows[f"{d:g}"] = round(float(np.mean(ds)), 3)
    DATA_DIR.mkdir(exist_ok=True)
    out = DATA_DIR / "formation_diameters.json"
    out.write_text(json.dumps(dict(scene=scene, drones=n, seeds=seeds, settle_s=SETTLE_S,
                                   force_law=rep.FORCE_LAW_COMMIT, diameter_m=rows), indent=1) + "\n",
                   encoding="utf-8", newline="\n")
    return out, rows


# ============================================================================= envelopment

def goal_city(centre, buildings):
    """A replica City of a goal patch's buildings [(rect, yaw, size, role)] (citygen.goal_buildings)."""
    from citysim import catalogue, layout as lay
    design = catalogue.study_design()
    boxes = []
    for rect, yaw, size, role in buildings:
        b = lay.make_box("g", "goal", rect[0], rect[1], design, size, yaw, role=role)
        boxes.append((b["tile"], np.array(b["c"], float), np.array(b["ax"], float)))
    return boxes


def envelop(boxes, centre, d_ref, n=10, seed=0, approach_deg=270.0, scene=SCENE, T=ENVELOP_S, overrides=None,
            record=False, start="placed", inner_radius=None):
    """Fly (start="arrive") or place (start="placed") the formation on a goal and let it settle. `boxes` are every
    building the replica should see (analyse.load_city tuples), `approach_deg` the heading FROM the goal TO the
    arrival's start (270 = from the west).

    Returns a dict: settled positions P (k, 2) and alive mask, kills (drone-drone), contacts, whether it enveloped
    the centre, the bearing gap round it, the closest approach to a building, and (record) the trajectory."""
    check()
    m = model(scene, overrides)
    city = rep.City(boxes)
    P0, _ = hover_template(d_ref, n, seed, scene, overrides=overrides)
    centre = np.asarray(centre, float)
    if start == "arrive":
        a = math.radians(approach_deg)
        P0 = P0 + centre + ARRIVE_M * np.array([math.sin(a), math.cos(a)])
    else:
        P0 = clear_of_buildings(city, P0 + centre, centre)
    sim = ContactSim(m, city, P0, np.zeros((n, 2)), seed)
    tail, traj = [], []
    while sim.t < T and sim.alive.sum() > 1:
        c = sim.P[sim.alive].mean(0)
        v = PILOT_GAIN * (np.asarray(centre) - c) / m.max_speed
        s = np.linalg.norm(v)
        stick = v if s <= PILOT_MAX else v * PILOT_MAX / s
        sim.step(stick, 0.0, d_ref)
        if sim.t >= T - MEASURE_S and sim.tick % 5 == 0:
            tail.append(np.where(sim.alive[:, None], sim.P, np.nan))
        if record and sim.tick % 25 == 0:
            traj.append((sim.t, np.where(sim.alive[:, None], sim.P, np.nan)))
    alive = sim.alive.copy()
    P = np.nanmean(np.array(tail)[:, alive], axis=0) if tail else sim.P[alive]
    out = dict(P=P, alive=alive, kills=len(sim.kills), kill_kinds=[k[1] for k in sim.kills], contacts=sim.contacts,
               n_alive=int(alive.sum()), d_ref=d_ref, seed=seed, start=start)
    out.update(envelopment(P, centre))
    out["min_building_m"] = float(city.min_distance(P, None).min()) if city.n and len(P) else math.inf
    r = np.linalg.norm(P - centre, axis=1) if len(P) else np.zeros(0)
    out["inside_frac"] = float((r <= inner_radius).mean()) if inner_radius is not None and len(P) else math.nan
    out["mean_radius_m"] = float(r.mean()) if len(P) else math.nan
    out["enveloped"] = bool(out["kills"] == 0 and out["surrounds"] and out["max_gap_deg"] <= MAX_GAP_DEG and
                            (inner_radius is None or out["inside_frac"] >= INSIDE_FRAC))
    out["diameter_m"] = ring_diameter(P) if len(P) >= 2 else 0.0
    out["centroid_offset_m"] = float(np.linalg.norm(P.mean(0) - np.asarray(centre))) if len(P) else math.nan
    if record:
        out["traj"] = traj
    return out


def clear_of_buildings(city, P, centre, margin=1.5):
    """Each point inside a footprint (grown by margin) moved radially away from `centre` until it is clear."""
    P = np.array(P, float)
    if city.n == 0:
        return P
    allb = np.arange(city.n)
    for i in range(len(P)):
        d = P[i] - centre
        u = d / np.linalg.norm(d) if np.linalg.norm(d) > 1e-6 else np.array([1.0, 0.0])
        r = float(np.linalg.norm(d))
        while city.footprint_hit(P[i][None], allb, margin)[0] and r < 200:
            r += 0.5
            P[i] = centre + r * u
    return P


def envelopment(P, centre):
    """Whether the points surround `centre`: inside their convex hull, and the widest bearing gap round it."""
    from citysim.display import convex_hull
    P = np.asarray(P, float)
    if len(P) < 3:
        return dict(surrounds=False, max_gap_deg=360.0)
    b = np.sort(np.mod(np.degrees(np.arctan2(P[:, 0] - centre[0], P[:, 1] - centre[1])), 360.0))
    gaps = np.diff(np.r_[b, b[0] + 360.0])
    h = P[convex_hull(P)]
    inside = all(((h[(i + 1) % len(h)] - h[i])[0] * (centre[1] - h[i][1]) -
                  (h[(i + 1) % len(h)] - h[i])[1] * (centre[0] - h[i][0])) > 0 for i in range(len(h)))
    return dict(surrounds=bool(inside), max_gap_deg=float(gaps.max()))


def _envmap_job(job):
    w_c, r_g, n_b, d_ref, n, seed, size = job
    from citysim import catalogue, citygen
    P = citygen.Params(w_c=w_c, r_g=r_g, n_b=n_b, size=size)
    rng = np.random.default_rng(seed)
    try:
        bld = citygen.goal_buildings(P, catalogue.study_design(), (0.0, 0.0), rng)
    except ValueError as e:
        return dict(w_c=w_c, r_g=r_g, n_b=n_b, d_ref=d_ref, n=n, seed=seed, size=size, feasible_geometry=False,
                    error=str(e))
    boxes = goal_city((0.0, 0.0), bld)
    rects = np.array([b[0] for b in bld])
    others = rects[1:]
    from citysim import geometry as geo
    c_g = float(geo.rect_point_distance(others, [[0.0, 0.0]]).min())
    r = envelop(boxes, (0.0, 0.0), d_ref, n, seed, start="arrive", inner_radius=c_g)
    return dict(w_c=w_c, r_g=r_g, n_b=n_b, d_ref=d_ref, n=n, seed=seed, size=size, feasible_geometry=True,
                R_c=round(float(np.hypot(rects[0, 2], rects[0, 3])), 3), c_g=round(c_g, 3),
                enveloped=r["enveloped"], surrounds=r["surrounds"], max_gap_deg=round(r["max_gap_deg"], 1),
                kills=r["kills"], contacts=r["contacts"], n_alive=r["n_alive"], diameter_m=round(r["diameter_m"], 2),
                min_building_m=round(r["min_building_m"], 2), centroid_offset_m=round(r["centroid_offset_m"], 2))


def envelope_map(w_cs, r_gs, n_bs, drefs, ns, seeds, size=2.0, jobs=None):
    import pandas as pd
    check()
    grid = list(itertools.product(w_cs, r_gs, n_bs, drefs, ns, range(seeds), [size]))
    with Pool(min(jobs or max(1, (os.cpu_count() or 2) - 2), len(grid))) as pool:
        rows = pool.map(_envmap_job, grid)
    return pd.DataFrame(rows)


# ============================================================================= the gate probe: what the formation flows through

GATE_GAPS = (0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 20.0)   # m between avoidance cylinders
GATE_DREFS = (0.4, 1.0, 1.6)
RAM_OFFSETS = (0.0, 3.0)          # m the lone building sits off the flight line
GATE_START_X, GATE_AIM_X, GATE_PAST_X = -90.0, 150.0, 40.0
GATE_T = 75.0
GATE_WALL_EACH_SIDE = 6           # buildings either side of the gap, cylinders overlapping: no way round in time
GATE_OVERLAP = 0.25               # of a cylinder radius, between neighbours in the wall
GATE_STALL_OK_S = 5.0             # longest stretch the centroid may make < 1 m/s and still "flow through": a longer
#                                   one (a lone drone parked head-on against a building until the noise slides it
#                                   off) is where a pilot would step in


def gate_city(kind, gap=0.0, offset=0.0, size=2.0):
    """Replica boxes of a probe: "gate" -- a wall of the study design across the flight line (x = 0) with one gap of
    `gap` m between the two central avoidance cylinders; "ram" -- one building on the line, `offset` m off it."""
    from citysim import catalogue, layout as lay
    design = catalogue.study_design()
    hx, hz = design.footprint(size, 0)
    R = math.hypot(hx, hz)
    if kind == "ram":
        centres = [(0.0, offset)]
    else:
        z = R + gap / 2.0
        centres = []
        for k in range(GATE_WALL_EACH_SIDE):
            centres += [(0.0, z + k * (2 - GATE_OVERLAP) * R), (0.0, -z - k * (2 - GATE_OVERLAP) * R)]
    boxes = []
    for k, (x, zc) in enumerate(centres):
        b = lay.make_box(f"probe_{k}", "probe", x, zc, design, size, 0)
        boxes.append((b["tile"], np.array(b["c"], float), np.array(b["ax"], float)))
    return boxes


def gate_trial(job):
    """One probe flight: the formation (or a lone drone) settled at GATE_START_X and flown at full stick at a point
    GATE_AIM_X down the line. Passed (it FLOWS through) = every surviving drone beyond GATE_PAST_X within GATE_T, none
    left behind until it was lost as isolated, and no stall longer than GATE_STALL_OK_S on the way. Drone-drone pair
    losses do not fail it: at the tightest spread they come in open sky too (the bench's known d_ref 0.4 problem), and
    the transit flights count them as losses; `clean` is the stricter nothing-lost-at-all."""
    kind, gap, offset, d_ref, n, seed = job
    check(allow_stale=False)
    m = model()
    boxes = gate_city(kind, gap, offset)
    if n > 1:
        P0, _ = hover_template(d_ref, n, seed)
    else:
        P0 = np.zeros((1, 2))
    P0 = P0 + np.array([GATE_START_X, 0.0])
    sim = ContactSim(m, rep.City(boxes), P0, np.zeros((len(P0), 2)), seed)
    aim = np.array([GATE_AIM_X, 0.0])
    t_cross, stall, slow = math.nan, 0.0, 0.0
    every = max(1, int(round(0.5 / m.dt)))
    prev_x = None
    while sim.t < GATE_T and sim.alive.sum() > 0:
        P = sim.P[sim.alive]
        c = P.mean(0)
        if (P[:, 0] > GATE_PAST_X).all():
            t_cross = sim.t
            break
        d = aim - c
        sim.step(d / max(float(np.linalg.norm(d)), 1e-9), 0.0, d_ref)
        if sim.tick % every == 0:
            x = float(c[0])
            if prev_x is not None and x < GATE_PAST_X:
                slow = slow + 0.5 if (x - prev_x) / 0.5 < 1.0 else 0.0
                stall = max(stall, slow)
            prev_x = x
    alive = sim.alive
    crossed = int((sim.P[alive][:, 0] > GATE_PAST_X).sum()) if alive.any() else 0
    lost = int(n - alive.sum())
    kinds = [k[1] for k in sim.kills]
    isolated = sum(1 for k in kinds if k == "isolated")
    flowed = bool(alive.any() and crossed == int(alive.sum()) and math.isfinite(t_cross) and isolated == 0 and
                  stall <= GATE_STALL_OK_S)
    return dict(kind=kind, gap_m=gap, offset_m=offset, d_ref=d_ref, n=n, seed=seed, lost=lost,
                kills=kinds, pair_kills=sum(1 for k in kinds if k == "drone"), isolated=isolated,
                contacts=int(sim.contacts), crossed=crossed,
                t_cross_s=round(t_cross, 2) if math.isfinite(t_cross) else math.nan,
                longest_stall_s=round(stall, 1), passed=flowed, clean=bool(flowed and lost == 0))


def gate_map(seeds=3, gaps=GATE_GAPS, drefs=GATE_DREFS, jobs=None):
    """Every probe flight (a DataFrame) and the summary routing reads: w_pass per d_ref (the narrowest probed gap
    passed in every seed, with every wider one passed too) and whether a lone building is flowed round (ram_ok); the
    same for one drone."""
    import pandas as pd
    check()
    grid = [("gate", g, 0.0, d, 10, s) for g in gaps for d in drefs for s in range(seeds)]
    grid += [("gate", g, 0.0, 1.0, 1, s) for g in gaps for s in range(seeds)]
    grid += [("ram", 0.0, o, d, 10, s) for o in RAM_OFFSETS for d in drefs for s in range(seeds)]
    grid += [("ram", 0.0, o, 1.0, 1, s) for o in RAM_OFFSETS for s in range(seeds)]
    from citysim.parallel import run_jobs
    df = pd.DataFrame(run_jobs(gate_trial, grid, "gate probe", processes=jobs, stall_s=120.0))

    def narrowest(sub):
        ok = sub.groupby("gap_m").passed.all().sort_index()
        good = None
        for g in sorted(ok.index, reverse=True):
            if not ok[g]:
                break
            good = g
        return None if good is None else float(good)

    gates, rams = df[df.kind == "gate"], df[df.kind == "ram"]
    summary = dict(
        scene=SCENE, force_law=rep.FORCE_LAW_COMMIT, seeds=seeds, gaps_m=list(gaps), passed_rule=(
            "every surviving drone beyond the wall within %g s, none lost as isolated, no stall over %g s (drone-drone "
            "pair losses are counted, not failed); w_pass = the narrowest gap passed in every seed with every wider "
            "probed gap passed too" % (GATE_T, GATE_STALL_OK_S)),
        w_pass_m={f"{d:g}": narrowest(gates[(gates.n == 10) & np.isclose(gates.d_ref, d)]) for d in drefs},
        ram_ok={f"{d:g}": bool(rams[(rams.n == 10) & np.isclose(rams.d_ref, d)].passed.all()) for d in drefs},
        single_w_pass_m=narrowest(gates[gates.n == 1]), single_ram_ok=bool(rams[rams.n == 1].passed.all()))
    return df, summary


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("diameters")
    p.add_argument("--seeds", type=int, default=3)
    p = sub.add_parser("envelope")
    p.add_argument("--wc", type=float, nargs="+", default=[7.0, 10.0, 13.06, 16.0, 20.0])
    p.add_argument("--rg", type=float, nargs="+", default=[18.0, 22.0, 26.0, 30.0, 36.0])
    p.add_argument("--nb", type=int, nargs="+", default=[8])
    p.add_argument("--dref", type=float, nargs="+", default=[0.4, 0.7, 1.0, 1.3, 1.6])
    p.add_argument("--n", type=int, nargs="+", default=[10, 8, 6])
    p.add_argument("--seeds", type=int, default=2)
    p.add_argument("--size", type=float, default=2.0)
    p.add_argument("--out", required=True)
    p.add_argument("--allow-stale", action="store_true")
    a = ap.parse_args()
    check(getattr(a, "allow_stale", False))
    if a.cmd == "diameters":
        out, rows = write_diameters(seeds=a.seeds)
        print(f"[formation] {out}: {rows}")
    else:
        df = envelope_map(a.wc, a.rg, a.nb, a.dref, a.n, a.seeds, a.size)
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(a.out, index=False)
        ok = df[df.feasible_geometry]
        print(ok.groupby(["w_c", "r_g", "d_ref"]).enveloped.mean().unstack().round(2).to_string())
        print(f"[formation] {a.out}: {len(df)} settles")


if __name__ == "__main__":
    main()
