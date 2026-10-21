"""The clutter study: how much more the swarm sees than one drone, standing anywhere in a city and roaming it with no
task, across city designs that vary the block size and the building density. docs/clutter_study/README.md is the
manual.

    python -B -m citysim.vissim clutter-feasibility                      # which cells of the grid can be built
    python -B -m citysim.vissim clutter [--quick] [--hours H] [--resume]  # -> experiment_results/clutter_study/<name>/
    python -B -m citysim.vissim clutter-tables [--name N]                # statistics, figures and the page again

CITIES. citygen's blocks mode built the way VoronoiCityWorld is (DiamondCityWorld's building mix at size 1, every
block filled evenly, 15 m streets, 4 m facade gaps), with no goal patches, over a grid of block spacing p (the mean
pitch of the Voronoi cells) x coverage alpha (the built share of the 730 m arena). A city the generator jams in (its
blocks full before alpha is reached) is reported, not flown; a cell needs MIN_CITIES buildable cities. The cities are
the study's independent units.

CONDITIONS. single: one drone on the path rule that plans round every building, its view slaved to its heading.
swarm at d_ref 0.4, 1.0 and 1.6: ten drones on the rule that plans round walls only and lets the formation morph round
the rest, its feeds from display.FeedLatch. Every condition of a walk flies the same random-walk intent chain
(citysim.pilots "rw": run and tumble, runs MIN_STEP_M + Exp, mean RW_RUN_M) from the same start and heading for T_S
seconds, or until a swarm is down to fewer than search.MIN_ALIVE_SWARM drones.

ROAMING (walks.csv, one row per condition of a walk): citysim.transit.score on one 2 m grid shared by the walk's
conditions, at the hat's 40 m: sweep width, ground cleared per second and per metre (any shown feed / the feeds in the
HMD's +-50 deg straight ahead), look-behind, persistence, path ratio (flown over the net progress made toward the
intent targets), stick adjustments per 100 m, morph events, losses, splits, speed; inst_* the ground on screen at an
instant, averaged over the walk; street shares (search.street_measures); and the RING measures. The grid is the whole
city (arena_cells), so the walk also gives TIME TO FIND a target hidden at a random spot of the city's open ground:
the chance it has been seen by t is the share of the arena's free cells first seen by t (find_curve_*, every FIND_STEP_S;
find_* = by the end of the walk; find_rmst_* = the mean time to find, counting an unfound target as the walk's length).
The target is DISCOVERED at FIND_RANGE_M (200 m, the user's choice: what internal_3's turns onto goals suggest), not at
the hat's 40 m: find_*_r200 come from discovery_first, a ray-cast isovist per camera on a FIND_CELL_M grid of the arena
(an exact sight line per cell, as at 40 m, would cost a hundred times more at that range); find_*_r40 are kept.
The single drone is scored at the swarm's altitude and pitch ("matched", the comparison) and at its own internal_3
band ("own").
STANDING (static.csv): capability's settles at free points (the formation flown in from the most open side and held
by the centring pilot, the single drone on its centroid), capability.static_metrics at four body yaws, and RING.
RING: the limit a wide formation meets among buildings. The shown feeds in display order; each adjacent pair's
ground within 40 m that both see through clear sight lines ("los") and that their frustums alone would share
("geo"): a pair is connected when it shares at least OVERLAP_MIN_M2. ring_conn_frac = connected pairs / pairs,
ring_whole_frac = samples whose every pair is connected, ring_geo_conn_frac the frustums alone (what the spread and
the hull rule allow) -- so the gap between los and geo is what the buildings take; wall_feed_frac = shown feeds more
than half filled by a facade nearer than display.WALL_HIT_M; uncovered_deg = azimuth round the ring no feed's field
of view covers; cam_facade_m = the shown cameras' median distance to the nearest facade.

STATISTICS (citysim.clutter_stats). Per paired unit (a walk; a standing point averaged over its body yaws) the log of swarm /
single; averaged per city; per cell and spread the geometric-mean ratio, its 95 % t interval over cities, the SD of
the city ratios, the SD of the unit log ratios, the one-sample t-test p over cities and the Benjamini-Hochberg q over
a metric's cells (per spread); the spreads against each other the same way; a city-level least-squares trend of the
log ratio on alpha and log2 p.
"""
if __name__ == "__main__" and __package__ in (None, ""):
    import os.path as _p
    import sys as _s
    _s.path[0] = _p.dirname(_p.dirname(_p.abspath(__file__)))

import sys

sys.dont_write_bytecode = True

import json  # noqa: E402
import math  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
from dataclasses import replace  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from citysim import LAYOUT_DIR, PACKAGE_DIR, REPO, camera, citygen, display, dynamic, layout as lay  # noqa: E402
from citysim import pilots, search, transit  # noqa: E402
from citysim.los import Footprints, vertical_band, wrap_deg  # noqa: E402

CLUTTER_DIR = REPO / "experiment_results" / "clutter_study"
SPEC = PACKAGE_DIR / "specs" / "clutter.json"
RANGE_M = camera.R_ID_M                 # the hat's recognition range: every measure here is at it
OVERLAP_MIN_M2 = 20.0                   # five 2 m cells: a pair of feeds shares ground at recognisable range
WALL_RAY_DEG = 5.0                      # rays across a feed's field of view for its wall share
ALPHA_TOL = 0.002                       # a city whose coverage misses its target by more has jammed
MIN_CITIES = 3                          # buildable cities a cell needs to be flown (a t interval over cities) ...
MIN_CITY_SHARE = 0.7                    # ... and this share of its cities: a cell the generator mostly jams in would
#                                         be flown on its loosest few cities only
TRACK_EVERY_S = 2.0                     # the first walk of each city keeps its tracks at this step, for the page
START_CLEAR_M = (12.0, 8.0, 5.0, 3.0)   # a walk's start: clearance from every footprint, relaxed in this order
ARM_RETRIES = 2
RECHECK_SHARE = 0.02                    # of the walks flown again at the end, bit for bit (citysim.parallel's CPU)
RESERVE_S = 1800.0                      # of the time budget kept for the retries, the recheck and the tables
STALL_S = 600.0                         # no result from a round's pool for this long: a worker died with its job
#                                         (a walk takes 1-3 min under load; the timer restarts with every result)
SWEEP_WINDOW_M = 80.0                   # the sweep width counts ground this far either side of the track: the default
#                                         (the range, 42 m with its bin) capped the swarm at 84 m, which it reached
#                                         at spread 1.0 and so could not show spread 1.6's wider reach (2026-10-21)
FIND_STEP_S = 15.0                      # the time-to-find curve's step
FIND_RANGE_M = 200.0                    # a hidden target is discovered within this slant range ...
FIND_CELL_M = 4.0                       # ... on a grid of the arena at this step,
FIND_EVERY_S = 1.0                      # ... checked this often,
FIND_RAY_DEG = 0.5                      # ... through rays this far apart to the first facade
QUICK = dict(p=[60.0, 110.0], alpha=[0.04, 0.10], cities_per_cell=3, walks_per_city=1, points_per_city=1, T_s=90.0)


# ============================================================================= spec and cities

def load_spec(path=SPEC):
    spec = json.loads(Path(path).read_text(encoding="utf-8"))
    spec.setdefault("name", Path(path).stem)
    return spec


def cell_id(p, alpha):
    return f"p{p:g}_a{alpha:g}"


def city_params(spec, p, alpha):
    kw = {k: (tuple(v) if isinstance(v, list) else v) for k, v in spec["city"].items()}
    return replace(citygen.Params(), **kw, p=float(p), alpha=float(alpha))


def _gen_job(job):
    P, seed, out_dir, meta = job
    path = Path(out_dir) / f"{P.ident(seed)}.json"
    try:
        if path.exists():
            L = lay.load(path)
        else:
            L = citygen.generate(P, seed)
            L.save(path)
    except Exception as e:                       # noqa: BLE001 -- an unbuildable city costs its own row
        return dict(meta, path=str(path), buildable=False, why=f"{type(e).__name__}: {e}")
    r = L.layout
    problems = L.verify()
    jam = bool(r["jammed"]) or abs(r["alpha_achieved"] - r["alpha_target"]) > ALPHA_TOL
    why = "jammed" if jam else ("; ".join(problems) if problems else "")
    out = dict(meta, path=str(path), layout=L.id, buildable=not why, why=why, alpha_achieved=r["alpha_achieved"],
               n_buildings=r["n_buildings"], n_blocks=r["n_blocks"], block_fill=r["f"], ell_m=r["ell_mean"],
               max_within_50m=r["max_within_50m"], min_street_width=r["min_street_width"])
    if not why:
        from citysim import capability, routing
        out.update(capability.porosity(L, Footprints(L.boxes_as_tuples()), np.random.default_rng(5),
                                       routing.w_pass(1.0)))
    return out


def cities(spec, name, jobs, quick=False):
    """Every city of the grid, generated or reused: a DataFrame, one row per (p, alpha, seed), with `buildable`,
    `cell_flown` (the design has enough buildable cities among the spec's judge_cities seeds, default its
    cities_per_cell) and `fly` (one of the first cities_per_cell seeds: a short preview run judges the grid as the full
    run does but flies fewer cities). Layouts go to layouts/sweep~/<layout_set, default the run's name>."""
    from citysim.parallel import run_jobs
    ps, alphas, n = spec["p"], spec["alpha"], spec["cities_per_cell"]
    if quick:
        ps, alphas, n = QUICK["p"], QUICK["alpha"], QUICK["cities_per_cell"]
    judge = max(n, int(spec.get("judge_cities", n)) if not quick else n)
    out_dir = LAYOUT_DIR / "sweep~" / spec.get("layout_set", name)
    out_dir.mkdir(parents=True, exist_ok=True)
    todo = [(city_params(spec, p, a), s, str(out_dir), dict(cell=cell_id(p, a), p=float(p), alpha=float(a), seed=s))
            for p in ps for a in alphas for s in range(judge)]
    rows = run_jobs(_gen_job, todo, "cities", jobs, stall_s=900)
    df = pd.DataFrame(rows)
    ok = df[df.buildable].groupby("cell").size()
    df["cell_flown"] = df.cell.map(lambda c: int(ok.get(c, 0)) >= max(MIN_CITIES, math.ceil(MIN_CITY_SHARE * judge)))
    df["fly"] = df.seed < n
    return df


# ============================================================================= scenarios

def walk_starts(L, mask, fp, rng, k):
    """k walk starts: free points the intent chain may stand on, with as much clearance from the buildings as the city
    allows (START_CLEAR_M relaxed in turn), and a random heading and seed each."""
    x0, x1, z0, z1 = L.layout["arena"]
    m = pilots.TARGET_MARGIN_M
    pts = []
    for clear in START_CLEAR_M:
        for _ in range(20000):
            if len(pts) >= k:
                break
            p = np.array([rng.uniform(x0 + m, x1 - m), rng.uniform(z0 + m, z1 - m)])
            if not mask.contains(p) or (fp.n and fp.distance([p])[0] < clear):
                continue
            pts.append(p)
        if len(pts) >= k:
            break
    return [dict(start=[float(p[0]), float(p[1])], heading0=float(rng.uniform(-180.0, 180.0)),
                 seed=int(rng.integers(1 << 30))) for p in pts]


def _scen_job(job):
    row, n_walks, n_points = job
    return scenarios(row, n_walks, n_points)


def scenarios(row, n_walks, n_points):
    """(walks, points) of one city: deterministic in the city."""
    from citysim import capability
    L = lay.load(row["path"])
    boxes = L.boxes_as_tuples()
    fp = Footprints(boxes)
    C = search.city_context(boxes, L.layout["arena"])
    base = 9_300_000 + 1000 * int(row["seed"]) + int(round(row["p"] * 10)) * 7 + int(round(row["alpha"] * 1000)) * 13
    walks = walk_starts(L, C["mask"], fp, np.random.default_rng(base), n_walks)
    pts = capability.free_points(L, fp, np.random.default_rng(base + 1), n_points, min_sep=60.0)
    return walks, [[float(p[0]), float(p[1])] for p in pts]


# ============================================================================= a walk

class RoamPilot(search.IntentPilot):
    """The search study's intent pilot, recording each segment (start, target, times) for the path measures."""

    def _segment(self, a, b, t):
        fl = self.fl
        c = fl.centroid().copy() if fl.sim.alive.any() else np.asarray(a, float)
        if fl.segments:
            fl.segments[-1].update(t1=t, end=c)
        fl.segments.append(dict(leg=len(fl.segments), t0=t, start=c, goal=np.asarray(b, float).copy()))
        super()._segment(a, b, t)


class RoamFlight(search.SearchFlight):
    """One condition of one walk: the replica flown along an intent chain for T seconds, with nothing to find."""

    def __init__(self, sim, m, fp, arm, d_ref, intent, planner, band, rate, T):
        dynamic.Flight.__init__(self, sim, m, fp, None, None, search.ARMS[arm]["interface"], "pilot", d_ref,
                                feeds=False, city_bounds=planner.arena)
        self.arm, self.spec, self.intent, self.planner = arm, search.ARMS[arm], intent, planner
        self.height, self.pitch = band
        self.heads, self.rate, self.cap, self.ranges = {}, rate, float(T), ()
        self.min_alive = 1 if self.single else search.MIN_ALIVE_SWARM
        self.latch = None if self.single else display.FeedLatch(len(sim.P), pitch_deg=self.pitch)
        self.scan_phase = 0.0
        self.bases, self.found, self.mech, self.samples, self.segments = [], {}, {}, [], []
        self.seg = dict(reached=0, timeout=0, stall=0, paused=0, pause_s=0.0)

    def _check(self, feeds, body):
        pass

    def fly(self):
        self._loop(RoamPilot(self), stop_when_found=False)
        c = self.centroid().copy() if self.sim.alive.any() else self.prev_c.copy()
        if self.segments:
            self.segments[-1].update(t1=self.sim.t, end=c)
        return self

    def record(self, **meta):
        """A transit.FlightRecord of the walk: every sample a transit sample, every intent segment a leg (start =
        the centroid when it began, arrive_m = how far short of its target it ended, so transit.score's straight
        distance is the net progress toward the targets)."""
        S = self.samples
        T, n = len(S), len(self.sim.P)
        K = max([len(s[6]) for s in S] + [1])
        pos, yaw, valid = np.zeros((T, K, 2)), np.zeros((T, K)), np.zeros((T, K), bool)
        for i, s in enumerate(S):
            d, y = np.asarray(s[6], int), np.asarray(s[7], float)
            if len(d):
                pos[i, :len(d)] = s[3][d] + camera.CAMERA_FORWARD_M * np.stack([np.sin(np.radians(y)),
                                                                                np.cos(np.radians(y))], -1)
                yaw[i, :len(d)], valid[i, :len(d)] = y, True
        t = np.array([s[0] for s in S], float)
        legs = []
        for g in self.segments:
            if "t1" not in g:
                continue
            legs.append(dict(leg=g["leg"], t0=float(g["t0"]), t1=float(g["t1"]), start=g["start"], goal=g["goal"],
                             arrive_m=float(np.linalg.norm(g["goal"] - g["end"]))))
        leg = np.zeros(T, int)
        for g in legs:
            leg[(t >= g["t0"]) & (t <= g["t1"])] = g["leg"]
        return transit.FlightRecord(
            t=t, phase=np.full(T, transit.TRANSIT), leg=leg,
            centroid=np.array([[s[1], s[2]] for s in S], float).reshape(T, 2),
            P=np.array([s[3] for s in S], float).reshape(T, n, 2), body_yaw=np.array([s[5] for s in S], float),
            cam_pos=pos, cam_yaw=yaw, cam_valid=valid, cmd=np.array(self.cmds, float).reshape(-1, 5), legs=legs,
            meta=dict(meta, interface=self.spec["interface"], d_ref=self.d_ref, n=n,
                      lost=int(n - self.sim.alive.sum()), contacts=int(getattr(self.sim, "contacts", 0))))


# ============================================================================= the ring of feeds

def _wall_share(fp, pos, yaw):
    rel = np.arange(-camera.HFOV_DEG / 2, camera.HFOV_DEG / 2 + 1e-9, WALL_RAY_DEG)
    return float((fp.first_hit(pos, yaw + rel, display.WALL_HIT_M) < display.WALL_HIT_M).mean())


class RingTally:
    """The RING measures, sample by sample: add() takes the shown cameras (positions, yaws), the body yaw and each
    camera's cells in clear sight within range (with their slant ranges and bearings off the axis)."""

    def __init__(self, fp, cells, height, pitch, r=RANGE_M, min_m2=OVERLAP_MIN_M2):
        self.fp, self.cells, self.height, self.pitch, self.r, self.min_m2 = fp, cells, height, pitch, r, min_m2
        self.pairs = self.conn = self.geo_conn = self.whole = self.ring_samples = 0
        self.overlap, self.geo_overlap, self.uncovered, self.facade = [], [], [], []
        self.feeds = self.wall_feeds = 0
        self.inst_any, self.inst_head, self.shown = [], [], []

    def add(self, pos, yaw, body, cams, single=False):
        """pos (K, 2), yaw (K,), cams [(idx, slant, off)] in the same order."""
        a = self.cells.area
        K = len(yaw)
        self.shown.append(K)
        any_ = [idx[slant <= self.r] for idx, slant, _ in cams]
        self.inst_any.append(len(np.unique(np.concatenate(any_))) * a if K else 0.0)
        if single:
            self.inst_head.append(self.inst_any[-1])
        else:
            hv = []
            for (idx, slant, off), y in zip(cams, yaw):
                az = wrap_deg(y - body + off * camera.DISPLAY_SCALE)
                hv.append(idx[(slant <= self.r) & (np.abs(az) <= camera.HMD_HALF_FOV_DEG)])
            self.inst_head.append(len(np.unique(np.concatenate(hv))) * a if K else 0.0)
        if K:
            self.feeds += K
            self.wall_feeds += sum(_wall_share(self.fp, pos[k], yaw[k]) >= 0.5 for k in range(K))
            self.facade.extend(self.fp.distance(pos).tolist() if self.fp.n else [math.inf] * K)
            ys = np.sort(np.mod(np.asarray(yaw, float), 360.0))
            gaps = np.diff(np.r_[ys, ys[0] + 360.0])
            self.uncovered.append(float(np.maximum(gaps - camera.HFOV_DEG, 0.0).sum()))
        if single or K < 2:
            return
        order = np.argsort(wrap_deg(np.asarray(yaw, float) - body))
        pairs = [(order[i], order[(i + 1) % K]) for i in range(K)] if K >= 3 else [(order[0], order[1])]
        geo = [transit.seen_from(self.fp, self.cells, pos[k], yaw[k], self.height, self.pitch, self.r, los=False)[0]
               for k in range(K)]
        whole = True
        for i, j in pairs:
            shared = len(np.intersect1d(any_[i], any_[j], assume_unique=True)) * a
            g = len(np.intersect1d(geo[i], geo[j], assume_unique=True)) * a
            self.pairs += 1
            self.overlap.append(shared)
            self.geo_overlap.append(g)
            ok = shared >= self.min_m2
            self.conn += ok
            self.geo_conn += g >= self.min_m2
            whole &= ok
        self.ring_samples += 1
        self.whole += whole

    def hook(self, rec):
        """visibility()'s on_sample for a FlightRecord."""
        def f(ti, cams):
            ks = [k for k, _, _, _ in cams]
            self.add(rec.cam_pos[ti, ks], rec.cam_yaw[ti, ks], float(rec.body_yaw[ti]),
                     [(idx, slant, off) for _, idx, slant, off in cams], single=rec.single)
        return f

    def summary(self):
        nan = math.nan
        return dict(
            shown_feeds=round(float(np.mean(self.shown)), 3) if self.shown else nan,
            inst_any_r40_m2=round(float(np.mean(self.inst_any)), 1) if self.inst_any else nan,
            inst_head_r40_m2=round(float(np.mean(self.inst_head)), 1) if self.inst_head else nan,
            ring_pairs=self.pairs, ring_conn_frac=round(self.conn / self.pairs, 4) if self.pairs else nan,
            ring_geo_conn_frac=round(self.geo_conn / self.pairs, 4) if self.pairs else nan,
            ring_whole_frac=round(self.whole / self.ring_samples, 4) if self.ring_samples else nan,
            ring_overlap_m2=round(float(np.mean(self.overlap)), 1) if self.overlap else nan,
            ring_geo_overlap_m2=round(float(np.mean(self.geo_overlap)), 1) if self.geo_overlap else nan,
            wall_feed_frac=round(self.wall_feeds / self.feeds, 4) if self.feeds else nan,
            uncovered_deg=round(float(np.mean(self.uncovered)), 2) if self.uncovered else nan,
            cam_facade_m=round(float(np.median(self.facade)), 2) if self.facade else nan)


def _cams_with_off(fp, cells, pos, yaw, height, pitch, r):
    return [transit.seen_from(fp, cells, pos[k], yaw[k], height, pitch, r) for k in range(len(yaw))]


# ============================================================================= jobs

def arena_cells(fp, bounds, step=transit.CELL_M):
    """transit.Cells over a whole city -- the same cells in the same order -- with the free ground found one footprint
    at a time: Cells' own test holds every point against every footprint at once, gigabytes over a whole city."""
    c = transit.Cells.__new__(transit.Cells)
    x0, x1, z0, z1 = bounds
    c.step, c.x0, c.z0 = step, x0, z0
    c.nx, c.nz = int(math.ceil((x1 - x0) / step)), int(math.ceil((z1 - z0) / step))
    occ = np.zeros((c.nx, c.nz), bool)
    for k in range(fp.n):
        r = fp.R[k]
        i0, i1 = max(int((fp.c[k, 0] - r - x0) // step), 0), min(int((fp.c[k, 0] + r - x0) // step) + 1, c.nx)
        j0, j1 = max(int((fp.c[k, 1] - r - z0) // step), 0), min(int((fp.c[k, 1] + r - z0) // step) + 1, c.nz)
        if i0 >= i1 or j0 >= j1:
            continue
        ii, jj = np.meshgrid(np.arange(i0, i1), np.arange(j0, j1), indexing="ij")
        pts = np.stack([x0 + (ii + 0.5) * step, z0 + (jj + 0.5) * step], -1).reshape(-1, 2)
        occ[i0:i1, j0:j1] |= fp.subset([k]).inside(pts).reshape(ii.shape)
    free = ~occ.ravel()
    ii, jj = np.meshgrid(np.arange(c.nx), np.arange(c.nz), indexing="ij")
    pts = np.stack([x0 + (ii + 0.5) * step, z0 + (jj + 0.5) * step], -1).reshape(-1, 2)
    c.index = -np.ones(c.nx * c.nz, np.int64)
    c.index[np.flatnonzero(free)] = np.arange(int(free.sum()))
    c.index = c.index.reshape(c.nx, c.nz)
    c.pts, c.area = pts[free], step * step
    c.n = len(c.pts)
    return c


_CITY = {}


def _city(path):
    """Per-worker cache of one city: its context, layout, street grid, and the 2 m grid over the whole arena (plus the
    range, so the ground the feeds see past its edge counts as before) with the mask of the arena's own cells -- where
    a hidden target may lie. The previous city's are dropped."""
    if path not in _CITY:
        _CITY.clear()
        import citysim
        citysim.use_bench_modules()
        import analyse
        analyse._STREETS.clear()
        L = lay.load(path)
        C = dict(search.city_context(L.boxes_as_tuples(), L.layout["arena"]), L=L,
                 streets=search.street_grid(L.id) if L.streets else None)
        x0, x1, z0, z1 = C["arena"]
        m = RANGE_M + 4.0
        C["cells"] = arena_cells(C["fp"], (x0 - m, x1 + m, z0 - m, z1 + m))
        C["find_grid"] = arena_cells(C["fp"], (x0, x1, z0, z1), step=FIND_CELL_M)
        q = C["cells"].pts
        C["in_arena"] = (q[:, 0] >= x0) & (q[:, 0] <= x1) & (q[:, 1] >= z0) & (q[:, 1] <= z1)
        _CITY[path] = C
    return _CITY[path]


def discovery_first(rec, fp, grid, height, pitch, r=FIND_RANGE_M, every_s=FIND_EVERY_S, ray_deg=FIND_RAY_DEG):
    """(first_any, first_head): the first time each cell of `grid` is within r slant of a shown feed, inside its field
    of view and vertical band, with a clear line -- a 2-D isovist per camera, rays every ray_deg across the field of
    view to the first facade (Footprints.first_hit), a cell visible when it is nearer than its ray's hit. Sampled every
    every_s. head: the feeds in the headset's view with the head facing travel (the single drone's feed always is)."""
    first_any = np.full(grid.n, np.inf)
    first_head = np.full(grid.n, np.inf)
    hmax = math.sqrt(max(r ** 2 - (height - camera.HAT_HEIGHT_M) ** 2, 0.0))
    half = camera.HFOV_DEG / 2
    rel = np.arange(-half, half + 1e-9, ray_deg)
    step = max(1, int(round(every_s / transit.SAMPLE_S)))
    for ti in range(0, len(rec.t), step):
        t = float(rec.t[ti])
        for k in np.flatnonzero(rec.cam_valid[ti]):
            cam, yaw = rec.cam_pos[ti, k], float(rec.cam_yaw[ti, k])
            idx = grid.within(cam, hmax)
            if not len(idx):
                continue
            d = grid.pts[idx] - cam
            hd = np.hypot(d[:, 0], d[:, 1])
            off = wrap_deg(np.degrees(np.arctan2(d[:, 0], d[:, 1])) - yaw)
            ok = (np.abs(off) <= half) & vertical_band(height, pitch, hd)
            if not ok.any():
                continue
            idx, hd, off = idx[ok], hd[ok], off[ok]
            hit = fp.first_hit(cam, yaw + rel, hmax) if fp.n else np.full(len(rel), np.inf)
            j = np.clip(np.round((off + half) / ray_deg).astype(int), 0, len(rel) - 1)
            vis = hd <= hit[j]
            idx, off = idx[vis], off[vis]
            first_any[idx[np.isinf(first_any[idx])]] = t
            if rec.single:
                hv = idx
            else:
                hv = idx[np.abs(wrap_deg(yaw - rec.body_yaw[ti] + off * camera.DISPLAY_SCALE)) <=
                         camera.HMD_HALF_FOV_DEG]
            first_head[hv[np.isinf(first_head[hv])]] = t
    return first_any, first_head


def find_measures(first, in_arena, T, prefix):
    """TIME TO FIND a target hidden at a random free spot of the arena, from the walk's first-sight time of every
    cell: the chance it has been seen by each FIND_STEP_S (the curve) and by the end (P); the mean time to find it with
    an unfound target counted at T (the restricted mean, which the walk's length caps); and the random-search
    (Koopman) estimate, P(t) = 1 - exp(-lambda t) fitted at T: the find rate lambda per hour and the mean time
    1 / lambda in minutes -- what the walk's 5 minutes say about a search that goes on."""
    f = np.asarray(first, float)[in_arena]
    if not len(f):
        return {}
    ts = np.arange(0.0, T + 1e-9, FIND_STEP_S)
    srt = np.sort(f)
    curve = np.searchsorted(srt, ts, side="right") / len(f)
    P = float(np.mean(f <= T))
    lam = -math.log(max(1.0 - P, 1e-12)) / T * 3600.0
    return {f"{prefix}_curve": [round(float(v), 5) for v in curve], f"{prefix}": round(P, 5),
            f"{prefix}_rmst_s": round(float(np.mean(np.minimum(f, T))), 2),
            f"{prefix}_rate_per_h": round(lam, 4), f"{prefix}_time_min": round(60.0 / lam, 2) if lam > 0 else math.nan}


def _retry(fn):
    """fn() again on an exception, up to ARM_RETRIES times: a flight is deterministic in its seed, and this machine's
    CPU throws the odd impossible numpy error under load (docs/cpu_instability_13900k.md)."""
    for attempt in range(ARM_RETRIES + 1):
        try:
            return fn(), attempt + 1
        except Exception:                        # noqa: BLE001 -- retried, then raised
            if attempt == ARM_RETRIES:
                raise


def fly_walk(C, sc, opts):
    """Every condition of one walk: [(arm, d_ref, RoamFlight, wall seconds, attempts)]."""
    out, cache = [], {}
    for arm, d_ref in opts["conditions"]:
        iface = search.ARMS[arm]["interface"]
        band = (opts["band"]["moving_h"], opts["band"]["moving_p"])

        def one():
            intent = pilots.Intent("rw", sc["start"], sc["heading0"], C["mask"], sc["seed"],
                                   **pilots.intent_params("rw", iface, None, opts["run_m"]))
            planner = search.Planner(C["cyl"], search.ARMS[arm]["plan"], d_ref, C["arena"], cache)
            sim, m = dynamic.make_sim(C["boxes"], intent.target(0), iface, d_ref, 10, sc["seed"] % 100000)
            fl = RoamFlight(sim, m, C["fp"], arm, d_ref, intent, planner, band, pilots.heading_rate("rw", iface),
                            opts["T_s"])
            fl.streets = C.get("streets")
            return fl.fly()
        t0 = time.time()
        fl, attempts = _retry(one)
        out.append((arm, d_ref, fl, round(time.time() - t0, 1), attempts))
    return out


def _walk_job(job):
    row, k, sc, opts = job
    meta = dict(job_id=f"{row['layout']}|walk|{k}", kind="walk", cell=row["cell"], p=row["p"], alpha=row["alpha"],
                city_seed=int(row["seed"]), layout=row["layout"], unit=k, heading0=round(sc["heading0"], 2),
                scen_seed=sc["seed"], sx=round(sc["start"][0], 1), sz=round(sc["start"][1], 1))
    try:
        C = _city(row["path"])
        flights = fly_walk(C, sc, opts)
        recs = [fl.record(arm=arm) for arm, _, fl, _, _ in flights]
        cells = C["cells"]
        b = opts["band"]
        rows, firsts = [], {}
        for (arm, d_ref, fl, wall_s, attempts), rec in zip(flights, recs):
            base = dict(meta, arm=arm, interface=search.ARMS[arm]["interface"], d_ref=d_ref, wall_s=wall_s,
                        attempts=attempts, **{k2: v for k2, v in fl.row().items()
                                             if k2 not in ("arm", "interface", "plan", "d_ref")})
            n_alive = len(fl.sim.P)
            base["lost_per_km"] = round(base["lost"] / max(base["path_m"] / 1000.0, 1e-9), 3)
            base["min_alive_frac"] = round(float(fl.sim.alive.sum()) / n_alive, 3)
            bands = [("matched", b["moving_h"], b["moving_p"])]
            if rec.single:
                bands.append(("own", b["own_h"], b["own_p"]))
            for which, h, p in bands:
                def sc_one():
                    ring = RingTally(C["fp"], cells, h, p)
                    s = transit.score(rec, C["fp"], cells, h, p, layout=None, ranges=(RANGE_M,),
                                      on_sample=ring.hook(rec), near_only=True,
                                      sweep_window=opts.get("sweep_window_m", SWEEP_WINDOW_M))
                    return s, ring
                (s, ring), _ = _retry(sc_one)
                f = s.pop("firsts", {})
                if which == "matched":
                    firsts[(arm, d_ref)] = dict(firsts=f)
                for basis in ("any", "head"):
                    key = f"{basis}{int(RANGE_M)}"
                    if key in f:
                        s.update(find_measures(f[key], C["in_arena"], opts["T_s"], f"find_{basis}_r{int(RANGE_M)}"))
                rf = float(opts.get("find_range_m", FIND_RANGE_M))
                (fa, fh), _ = _retry(lambda: discovery_first(rec, C["fp"], C["find_grid"], h, p, rf))
                inside = np.ones(C["find_grid"].n, bool)
                for basis, first in (("any", fa), ("head", fh)):
                    s.update(find_measures(first, inside, opts["T_s"], f"find_{basis}_r{int(rf)}"))
                s = {k2: v for k2, v in s.items() if k2 not in ("arm", "interface", "d_ref", "n", "lost",
                                                                 "contacts", "seed", "replans", "stall_events")}
                rows.append({**base, "band": which, **s, **ring.summary()})
        comp = []
        sg = firsts.get(("single", 1.0))
        for (arm, d_ref), sw in firsts.items():
            if arm != "single" and sg is not None:
                c = transit.compare(sw, sg, "sw")
                comp.append(dict(meta, d_ref=d_ref, **{k2: v for k2, v in c.items() if "r40" in k2 or "40" in k2}))
        res = dict(job_id=meta["job_id"], kind="walk", rows=rows, compare=comp)
        if k == 0:
            step = max(1, int(round(TRACK_EVERY_S / transit.SAMPLE_S)))
            res["tracks"] = {f"{arm}_{d_ref:g}": np.round(rec.centroid[::step], 1).tolist()
                             for (arm, d_ref, *_), rec in zip(flights, recs)}
        return res
    except Exception as e:                       # noqa: BLE001 -- one walk's failure costs that walk
        return dict(job_id=meta["job_id"], kind="walk", error=f"{type(e).__name__}: {e}",
                    trace=traceback.format_exc()[-1200:])


def _static_job(job):
    """One standing point of a city at every spread: capability's settle and static metrics, plus RING."""
    from citysim import capability
    row, k, point, opts = job
    meta = dict(job_id=f"{row['layout']}|static|{k}", kind="static", cell=row["cell"], p=row["p"],
                alpha=row["alpha"], city_seed=int(row["seed"]), layout=row["layout"], unit=k,
                x=round(point[0], 1), z=round(point[1], 1))
    try:
        C = _city(row["path"])
        fp, boxes = C["fp"], C["boxes"]
        point = np.asarray(point, float)
        near = fp.near(point, 250.0)
        b = opts["band"]
        h, pch = b["static_h"], b["static_p"]
        rows = []
        for d_ref in opts["spreads"]:
            seed = int(row["seed"]) * 1000 + k * 7 + int(round(d_ref * 10))
            (P, lost, contacts), attempts = _retry(lambda: capability.settle_at([boxes[i] for i in near],
                                                                                fp.subset(near), point, d_ref, seed))
            if len(P) < 3:
                rows.append(dict(meta, d_ref=d_ref, lost=lost, contacts=contacts, failed=True))
                continue
            o = P.mean(0)
            sub = fp.subset(fp.near(o, 200.0))
            cells = transit.Cells(sub, (o[0] - RANGE_M - 4, o[0] + RANGE_M + 4, o[1] - RANGE_M - 4,
                                        o[1] + RANGE_M + 4))
            for by in capability.GEO_BODY_YAWS:
                def one():
                    m = capability.static_metrics(sub, P, by, h, pch, b["head_yaws"], b["head_w"])
                    feeds = display.swarm_feeds(P, by)
                    ring = RingTally(sub, cells, h, pch)
                    ring.add(feeds.pos, feeds.yaw, by, _cams_with_off(sub, cells, feeds.pos, feeds.yaw, h, pch,
                                                                      RANGE_M))
                    return m, ring
                (m, ring), _ = _retry(one)
                rs = ring.summary()
                rows.append(dict(meta, d_ref=d_ref, body_yaw=by, lost=lost, contacts=contacts, n_alive=len(P),
                                 attempts=attempts, offset_m=round(float(np.linalg.norm(o - point)), 2), failed=False,
                                 **m, **{k2: v for k2, v in rs.items() if k2.startswith(("ring_", "wall_", "uncov",
                                                                                         "cam_"))}))
        return dict(job_id=meta["job_id"], kind="static", rows=rows)
    except Exception as e:                       # noqa: BLE001
        return dict(job_id=meta["job_id"], kind="static", error=f"{type(e).__name__}: {e}",
                    trace=traceback.format_exc()[-1200:])


def _job(job):
    return _walk_job(job[1]) if job[0] == "walk" else _static_job(job[1])


# ============================================================================= the run

def plan_jobs(cdf, spec, opts, jobs, quick=False):
    """Rounds of jobs, replicate by replicate: round r holds every flown city's r-th walk and r-th standing point, so
    a run stopped after any round covers every cell equally. [[(kind, job)]]."""
    from citysim.parallel import run_jobs
    n_walks = QUICK["walks_per_city"] if quick else spec["walks_per_city"]
    n_points = QUICK["points_per_city"] if quick else spec["points_per_city"]
    rounds = [[] for _ in range(max(n_walks, n_points))]
    flown = [row.to_dict() for _, row in cdf[cdf.buildable & cdf.cell_flown & cdf.fly].iterrows()]
    scen = run_jobs(_scen_job, [(r, n_walks, n_points) for r in flown], "scenarios", jobs, stall_s=900)
    for r, (walks, points) in zip(flown, scen):
        for k, sc in enumerate(walks):
            rounds[k].append(("walk", (r, k, sc, opts)))
        for k, pt in enumerate(points):
            rounds[k].append(("static", (r, k, pt, opts)))
    rng = np.random.default_rng(11)
    for rd in rounds:                           # dense and open cities mixed, so no stretch of a round is all slow
        rng.shuffle(rd)
    return [rd for rd in rounds if rd]


def _done_ids(path):
    ids = set()
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue                         # a line cut short by a crash
            if "error" not in r:
                ids.add(r["job_id"])
    return ids


def _append(path, res):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(res, default=_jsonable) + "\n")


def _jsonable(v):
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, (np.bool_,)):
        return bool(v)
    return str(v)


def read_results(out):
    """(walks, compare, static, errors) DataFrames from the run's results.jsonl (the latest result of each job)."""
    by_id = {}
    path = out / "results.jsonl"
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r["job_id"] in by_id and "error" not in by_id[r["job_id"]] and "error" in r:
                continue                         # a later failure never hides an earlier success
            by_id[r["job_id"]] = r
    walks = [x for r in by_id.values() if r.get("kind") == "walk" and "rows" in r for x in r["rows"]]
    comp = [x for r in by_id.values() if r.get("kind") == "walk" and "compare" in r for x in r["compare"]]
    static = [x for r in by_id.values() if r.get("kind") == "static" and "rows" in r for x in r["rows"]]
    errs = [dict(job_id=r["job_id"], error=r["error"], trace=r.get("trace", "")) for r in by_id.values()
            if "error" in r]
    return pd.DataFrame(walks), pd.DataFrame(comp), pd.DataFrame(static), pd.DataFrame(errs)


def log(out, msg):
    line = f"[clutter {time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(out / "run.log", "a", encoding="utf-8") as f:
        f.write(line + "\n")


def keep_awake():
    """Ask Windows not to sleep while this process runs (released when it exits; no power setting is changed)."""
    if sys.platform == "win32":
        import ctypes
        ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)


def cmd_clutter(a):
    from citysim import capability
    from citysim.parallel import run_jobs
    spec = load_spec(a.spec)
    name = a.name or spec["name"] + ("_quick" if a.quick else "")
    out = CLUTTER_DIR / name
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    hours = float(a.hours if a.hours is not None else spec.get("hours", 11.0))
    res_path = out / "results.jsonl"
    if not a.resume and res_path.exists():
        res_path.rename(out / f"results.{time.strftime('%Y%m%d_%H%M%S')}.jsonl")
    log(out, f"start: spec {a.spec}, {'quick' if a.quick else 'full'}, budget {hours:g} h, {a.jobs} workers")
    keep_awake()
    cdf = cities(spec, name, a.jobs, a.quick)
    cdf.to_csv(out / "cities.csv", index=False)
    flown = cdf[cdf.buildable & cdf.cell_flown & cdf.fly]
    log(out, f"cities: {len(cdf)} asked, {int(cdf.buildable.sum())} buildable, {flown.cell.nunique()} of "
             f"{cdf.cell.nunique()} cells flown ({time.time() - t0:.0f} s)")
    opts = dict(band=capability.band_settings(spec.get("band", "internal_3")),
                find_range_m=float(spec.get("find_range_m", FIND_RANGE_M)),
                sweep_window_m=float(spec.get("sweep_window_m", SWEEP_WINDOW_M)),
                T_s=float(QUICK["T_s"] if a.quick else spec["T_s"]), run_m=float(spec.get("run_m", pilots.RW_RUN_M)),
                spreads=[float(d) for d in spec["spreads"]],
                conditions=[("single", 1.0)] + [("swarm", float(d)) for d in spec["spreads"]])
    rounds = plan_jobs(cdf, spec, opts, a.jobs, a.quick)
    done = _done_ids(res_path)
    log(out, f"{sum(len(r) for r in rounds)} jobs in {len(rounds)} rounds ({len(done)} already done)")
    durations, ran = [], []
    stopped = None
    for ri, rd in enumerate(rounds):
        todo = [j for j in rd if _key(j) not in done]
        if not todo:
            continue
        elapsed = time.time() - t0
        est = float(np.mean(durations)) if durations else 0.0
        if durations and elapsed + est > hours * 3600.0 - RESERVE_S:
            stopped = ri
            log(out, f"budget: round {ri} would end at {(elapsed + est) / 3600:.2f} h of {hours:g} h; stopping here")
            break
        r0 = time.time()
        ran.append(ri)
        n_err = [0]

        def tick(k, total, res):
            _append(res_path, res)
            n_err[0] += "error" in res
            if k == total or k % max(1, total // 5) == 0:
                log(out, f"round {ri}: {k}/{total} at {(time.time() - t0) / 3600:.2f} h ({n_err[0]} failed)")
        try:
            run_jobs(_job, todo, f"round {ri}", a.jobs, stall_s=STALL_S, progress=tick)
        except SystemExit as e:                  # jobs a dead pool never finished: left for the retry or --resume
            log(out, f"round {ri}: {e}")
        durations.append(time.time() - r0)
        log(out, f"round {ri} done in {durations[-1] / 60:.1f} min")
    # the failed jobs once more, then a bit-for-bit re-fly of a sample of the walks
    done = _done_ids(res_path)
    failed = [j for ri in ran for j in rounds[ri] if _key(j) not in done]      # failed, or lost with a dead worker
    if failed:
        log(out, f"retrying {len(failed)} failed or lost job(s)")
        try:
            run_jobs(_job, failed, "retry", a.jobs, stall_s=STALL_S, progress=lambda k, n, r: _append(res_path, r))
        except SystemExit as e:
            log(out, f"retry: {e}")
    recheck(out, rounds, res_path, a.jobs, a.quick)
    w, c, s, e = read_results(out)
    log(out, f"results: {len(w)} walk rows, {len(s)} standing rows, {len(e)} failed job(s)")
    (out / "spec.json").write_text(json.dumps(spec, indent=1) + "\n", encoding="utf-8")
    (out / "manifest.json").write_text(json.dumps(dict(
        generated=time.strftime("%Y-%m-%dT%H:%M:%S"), spec=str(a.spec), quick=bool(a.quick), hours_budget=hours,
        rounds_planned=len(rounds), rounds_stopped_at=stopped, round_minutes=[round(d / 60, 1) for d in durations],
        cities=len(cdf), cities_buildable=int(cdf.buildable.sum()), cells_flown=int(flown.cell.nunique()),
        walks=int(w.job_id.nunique()) if len(w) else 0, points=int(s.job_id.nunique()) if len(s) else 0,
        failed=int(len(e)), opts={k: v for k, v in opts.items() if k != "conditions"},
        conditions=opts["conditions"], seconds=round(time.time() - t0)), indent=1, default=_jsonable) + "\n",
        encoding="utf-8")
    tables(out)
    log(out, f"finished in {(time.time() - t0) / 3600:.2f} h")


def _key(j):
    kind, job = j
    return f"{job[0]['layout']}|{kind}|{job[1]}"


def recheck(out, rounds, res_path, jobs, quick):
    """Fly RECHECK_SHARE of the walks again and compare their rows with the first flight's, field by field: every
    job is deterministic in its seed, so a difference is silent corruption (the CPU), not chance."""
    from citysim.parallel import run_jobs
    walks = [j for rd in rounds for j in rd if j[0] == "walk"]
    if not walks:
        return
    rng = np.random.default_rng(3)
    k = max(1 if quick else 4, int(round(RECHECK_SHARE * len(walks))))
    pick = [walks[i] for i in rng.choice(len(walks), size=min(k, len(walks)), replace=False)]
    first = {}
    for line in res_path.read_text(encoding="utf-8").splitlines():
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "rows" in r:
            first[r["job_id"]] = r
    pick = [j for j in pick if _key(j) in first]
    try:
        again = run_jobs(_job, pick, "recheck", jobs, stall_s=STALL_S)
    except SystemExit as e:
        log(out, f"recheck: {e}")
        return
    rows = []
    for j, r in zip(pick, again):
        f = first[_key(j)]
        diff = []
        if "error" in r:
            diff = ["error: " + r["error"]]
        else:
            for x, y in zip(f["rows"], r["rows"]):
                for col in x:
                    if col in ("wall_s", "attempts"):
                        continue
                    a_, b_ = x.get(col), y.get(col)
                    same = (a_ == b_) or (isinstance(a_, float) and isinstance(b_, float) and math.isnan(a_)
                                          and math.isnan(b_))
                    if not same:
                        diff.append(f"{x.get('arm')}{x.get('d_ref')}/{x.get('band')}:{col} {a_} != {b_}")
        rows.append(dict(job_id=_key(j), identical=not diff, differences="; ".join(diff[:5])))
    df = pd.DataFrame(rows)
    df.to_csv(out / "recheck.csv", index=False)
    log(out, f"recheck: {int(df.identical.sum())} of {len(df)} re-flown walks identical")


# ============================================================================= tables

def pin_to_ecores(mask=0xFFFF0000):
    """Pin this process to the i9-13900K's E-cores (logical processors 16-31): single-threaded numpy at full boost is
    what faults on this machine (docs/cpu_instability_13900k.md). A pool's workers inherit it, so call it after them."""
    if sys.platform != "win32" or (__import__("os").cpu_count() or 0) != 32:
        return False
    import ctypes
    k = ctypes.windll.kernel32
    k.GetCurrentProcess.restype = ctypes.c_void_p
    k.SetProcessAffinityMask.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    return bool(k.SetProcessAffinityMask(k.GetCurrentProcess(), mask))


def tables(out, tries=3):
    """Every table, the figures and the report page from a run's rows (citysim.clutter_stats; imported here, not at
    the top, so the statistics can be revised while a run's workers fly with this module unchanged). Pinned to the
    E-cores and retried: the step is deterministic, and on 2026-10-20 it died once with an impossible TypeError that
    a rerun did not repeat."""
    from citysim import clutter_stats
    pin_to_ecores()
    for k in range(tries):
        try:
            return clutter_stats.tables(out)
        except Exception as e:                   # noqa: BLE001 -- retried, then raised
            log(out, f"tables failed ({k + 1} of {tries}): {type(e).__name__}: {e}\n{traceback.format_exc()[-1500:]}")
            if k == tries - 1:
                raise


def cmd_tables(a):
    tables(CLUTTER_DIR / a.name)


def cmd_feasibility(a):
    spec = load_spec(a.spec)
    cdf = cities(spec, a.name or spec["name"], a.jobs)
    tab = cdf.pivot_table(index="p", columns="alpha", values="buildable", aggfunc="sum")
    print(f"buildable cities per cell (of {spec['cities_per_cell']}); a cell is flown with {MIN_CITIES} or more:")
    print(tab.to_string())
    print(cdf[~cdf.buildable].groupby("why").size().to_string())
