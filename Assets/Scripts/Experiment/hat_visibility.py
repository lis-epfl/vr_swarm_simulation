"""How long each goal's hat walker was observable in the drone feeds.

    python hat_visibility.py [--dir DIR] [--pid AABB AABC] [--range 40] [--out hat_visibility.csv]

Being over the goal tile is the wrong test (hats were often identified on the approach, from outside it),
and "within a neighbouring tile" is far too broad. The target of the task is the special walker, so the
metric asks whether any drone's camera could actually show that walker's hat. A goal is *in view* at a
10 Hz sample when at least one alive drone passes all three tests:

  1. range   -- camera-to-hat distance <= --range (default 40 m). Calibrated on the data: five hats were
                identified although they never came closer than 27-34 m, so the true limit is >= ~35 m;
                at 40 m the walker is ~34 px tall in the 1152x648 feed and the hat ~6 px across.
  2. heading -- the hat is inside the camera's horizontal field of view (74.4 deg: the DJI Mini 3 Pro's
                82.1 deg diagonal at 16:9, as ScreenSpawn sets it).
  3. sight   -- the line from the camera to the hat misses every building: the city's building colliders
                (city_obstacles_ScaledCityWorld.json, from Tools/Swarm/Export city obstacles) minus the three tiles the
                goals replaced, plus each goal patch's own nine buildings.

The gimbal pitch is not logged, so the vertical field of view is not tested (the camera is assumed pitched
to where the walker is); `levelCamPct` is the share of the in-view time a level gimbal would also have
covered. Head direction is not used either: this is time the hat was on a feed, not time on the pilot's eye.

Per goal: `inViewSec` (the headline: summed in-view time, independent of any episode rule),
`firstSightSec`, and *observation episodes* -- in-view samples joined across gaps <= GAP_S -- with the
longest one's span. `occludedSec` is time a drone had the hat in range and heading but a building in the
way (and no other drone saw it); `tileDwellSec` is the old centroid-over-the-tile time, for comparison.
"""
import argparse
import glob
import json
import math
import os

import numpy as np
import pandas as pd

DEFAULT_DIR = os.path.expandvars(r"%USERPROFILE%\AppData\LocalLow\UAVS@BERKELEY\DroneSim\experiment")
HERE = os.path.dirname(os.path.abspath(__file__))
OBSTACLES = os.path.join(HERE, "city_obstacles_ScaledCityWorld.json")

# Camera: DJI Mini 3 Pro 82.1 deg diagonal at 16:9 (ScreenSpawn) -> 46.4 deg vertical, 74.4 deg horizontal.
ASPECT = 16 / 9
VFOV_HALF = math.atan(math.tan(math.radians(82.1 / 2)) / math.sqrt(ASPECT ** 2 + 1))
HFOV_HALF = math.atan(math.tan(VFOV_HALF) * ASPECT)

R_ID = 40.0          # m, identification range (see the docstring)
GAP_S = 5.0          # s, gaps up to this long do not split an episode
HAT_HEIGHT = 1.6     # m above the walker's logged position (0.2 m up) -> hat centre ~1.8 m above ground

# Goal patch buildings (goal_patch_Scaled.prefab, goal-root frame): x0, x1, z0, z1; all 0.2..50.2 tall.
# StreetWidthTuner (blockScale 0.8) moves them about the kerb at runtime; sizes are unchanged.
GOAL_PILLARS = [
    (2.0, 4.7, 27.2, 30.2), (-31.2, -29.2, 26.8, 29.8), (2.0, 4.7, -31.4, -28.4), (29.8, 31.8, 26.8, 29.8),
    (-31.3, -28.3, 0.2, 2.9), (-31.2, -29.2, -31.8, -28.8), (29.8, 31.8, -31.8, -28.8), (28.9, 31.9, 0.2, 2.9),
    (-1.4, 2.1, -1.5, 1.3)]
GOAL_KERB_X = 0.36
BLOCK_SCALE = 0.8
BUILDING_TOP = 50.2


def load_city():
    d = json.load(open(OBSTACLES, encoding="utf-8"))
    boxes = []
    for b in d["boxes"]:
        ax = np.array(b["ax"], dtype=float)            # three half-axis vectors (world)
        boxes.append((b["tile"], np.array(b["c"]), ax))
    kerbs = {t["name"]: np.array(t["kerb"]) for t in d["tiles"] if t["kerb"]}
    return boxes, kerbs


def pillar_boxes(gx, gz):
    out = []
    for x0, x1, z0, z1 in GOAL_PILLARS:
        cx, cz = (x0 + x1) / 2, (z0 + z1) / 2
        cx = GOAL_KERB_X + BLOCK_SCALE * (cx - GOAL_KERB_X)
        cz = BLOCK_SCALE * cz
        c = np.array([gx + cx, (0.2 + BUILDING_TOP) / 2, gz + cz])
        ax = np.diag([(x1 - x0) / 2, (BUILDING_TOP - 0.2) / 2, (z1 - z0) / 2])
        out.append(("goal", c, ax))
    return out


def scene_boxes(goals, city, kerbs):
    """City buildings minus the tiles the goals replaced, plus each goal patch's own buildings."""
    replaced = set()
    for g in goals:
        name, dist = min(((n, math.hypot(k[0] - GOAL_KERB_X - g["goalX"], k[2] - g["goalZ"])) for n, k in kerbs.items()),
                         key=lambda x: x[1])
        if dist > 2.0:
            raise SystemExit(f"goal at ({g['goalX']:.1f}, {g['goalZ']:.1f}) matches no tile (nearest {name} {dist:.1f} m)")
        replaced.add(name)
    boxes = [b for b in city if b[0] not in replaced]
    for g in goals:
        boxes += pillar_boxes(g["goalX"], g["goalZ"])
    return boxes


def segment_blocked(p0, p1, boxes):
    """p0, p1: (N, 3). True where the segment p0->p1 passes through any oriented box."""
    blocked = np.zeros(len(p0), dtype=bool)
    d = p1 - p0
    for _, c, ax in boxes:
        lens = np.linalg.norm(ax, axis=1)
        u = ax / lens[:, None]                      # box axes (unit), rows
        o = (p0 - c) @ u.T                          # segment start in box frame
        v = d @ u.T
        t0 = np.zeros(len(p0)); t1 = np.ones(len(p0))
        with np.errstate(divide="ignore", invalid="ignore"):
            for k in range(3):
                a = (-lens[k] - o[:, k]) / v[:, k]
                b = (lens[k] - o[:, k]) / v[:, k]
                lo = np.minimum(a, b); hi = np.maximum(a, b)
                par = v[:, k] == 0
                inside = np.abs(o[:, k]) <= lens[k]
                lo = np.where(par, np.where(inside, -np.inf, np.inf), lo)
                hi = np.where(par, np.where(inside, np.inf, -np.inf), hi)
                t0 = np.maximum(t0, lo); t1 = np.minimum(t1, hi)
        blocked |= t0 <= t1
    return blocked


def episodes(t, flag, dt):
    """[(start, end, inViewSec)] joining in-view samples across gaps <= GAP_S."""
    out = []
    idx = np.flatnonzero(flag)
    if len(idx) == 0:
        return out
    s = idx[0]; prev = idx[0]; acc = dt[idx[0]]
    for i in idx[1:]:
        if t[i] - t[prev] - dt[prev] > GAP_S:
            out.append((t[s], t[prev] + dt[prev], acc))
            s = i; acc = 0.0
        acc += dt[i]; prev = i
    out.append((t[s], t[prev] + dt[prev], acc))
    return out


def analyse(d, stem, city, kerbs, r_id=R_ID):
    j = json.load(open(os.path.join(d, stem + "_session.json"), encoding="utf-8"))
    dr = pd.read_csv(os.path.join(d, stem + "_drones.csv"), sep=";")
    wk = pd.read_csv(os.path.join(d, stem + "_walkers.csv"), sep=";")
    dr = dr[dr.alive == 1]
    ts = np.sort(dr.t.unique())
    dt = np.minimum(np.diff(np.append(ts, j["durationSec"])), 0.5)
    boxes = scene_boxes(j["goals"], city, kerbs)
    rows = []
    for g in j["goals"]:
        w = wk[wk.goalIndex == g["goalIndex"]][["t", "specialX", "specialY", "specialZ"]]
        m = dr.merge(w, on="t")
        cam = m[["gtX", "gtY", "gtZ"]].to_numpy()
        hat = m[["specialX", "specialY", "specialZ"]].to_numpy() + [0, HAT_HEIGHT, 0]
        rel = hat - cam
        dist = np.linalg.norm(rel, axis=1)
        hd = np.hypot(rel[:, 0], rel[:, 2])
        off = (np.degrees(np.arctan2(rel[:, 0], rel[:, 2])) - m.yawDeg.to_numpy() + 180) % 360 - 180
        dep = np.degrees(np.arctan2(-rel[:, 1], hd))
        cand = (dist <= r_id) & (np.abs(off) <= math.degrees(HFOV_HALF))
        vis = cand.copy()
        if cand.any():
            near = [b for b in boxes if np.linalg.norm(b[1][[0, 2]] - hat[cand][:, [0, 2]].mean(0)) < r_id + 80]
            vis[cand] = ~segment_blocked(cam[cand], hat[cand], near)
        m = m.assign(vis=vis, level=vis & (np.abs(dep) <= math.degrees(VFOV_HALF)),
                     dmin=np.where(vis, dist, np.nan), blocked=cand & ~vis)
        per_t = m.groupby("t").agg(vis=("vis", "any"), level=("level", "any"), n=("vis", "sum"),
                                   blocked=("blocked", "any"), dmin=("dmin", "min"))
        per_t = per_t.reindex(ts).fillna({"vis": False, "level": False, "n": 0, "blocked": False})
        flag = per_t.vis.to_numpy(bool)
        eps = episodes(ts, flag, dt)
        in_view = float(dt[flag].sum())
        blocked_only = float(dt[per_t.blocked.to_numpy(bool) & ~flag].sum())
        level = float(dt[per_t.level.to_numpy(bool)].sum())
        longest = max(eps, key=lambda e: e[1] - e[0]) if eps else None
        rows.append(dict(
            run=stem, pid=stem.split("_")[0], trial=int(stem.split("_")[1][1:]), condition=j["condition"],
            hat=g["hat"].replace("Walker", ""), visitOrder=g.get("visitOrder", -1),
            firstSightSec=round(eps[0][0], 1) if eps else None,
            inViewSec=round(in_view, 1),
            nEpisodes=len(eps),
            episodeSec=round(sum(e[1] - e[0] for e in eps), 1),
            longestStartSec=round(longest[0], 1) if longest else None,
            longestEndSec=round(longest[1], 1) if longest else None,
            longestSec=round(longest[1] - longest[0], 1) if longest else None,
            minDistM=round(float(np.nanmin(per_t.dmin.to_numpy(float))), 1) if flag.any() else None,
            meanDronesInView=round(float(per_t.n[flag].mean()), 1) if flag.any() else 0,
            levelCamPct=round(100 * level / in_view) if in_view else None,
            occludedSec=round(blocked_only, 1),
            tileDwellSec=round(g["visitExitSec"] - g["visitEnterSec"], 1) if not g.get("visitApprox") else 0.0,
            sessionSec=round(j["durationSec"], 1),
            episodes=" | ".join(f"{a:.0f}-{b:.0f} ({v:.0f}s)" for a, b, v in eps)))
    return rows


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dir", default=DEFAULT_DIR)
    p.add_argument("--pid", nargs="*", default=[])
    p.add_argument("--range", type=float, default=R_ID)
    p.add_argument("--out", default="")
    a = p.parse_args()
    city, kerbs = load_city()
    rows = []
    for s in sorted(glob.glob(os.path.join(a.dir, "*_session.json"))):
        stem = os.path.basename(s)[: -len("_session.json")]
        if a.pid and stem.split("_")[0].upper() not in {x.upper() for x in a.pid}:
            continue
        rows += analyse(a.dir, stem, city, kerbs, a.range)
    df = pd.DataFrame(rows)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30); pd.set_option("display.max_colwidth", 80)
    print(df.drop(columns=["run"]).to_string(index=False))
    if a.out:
        df.to_csv(a.out, sep=";", index=False)
        print("wrote", a.out)
