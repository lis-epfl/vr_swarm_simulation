"""S12 Distributed Views: what internal_4's pilots tell the simulation (the views study's human half).

Three things the simulated task needs and the trials can measure, each from the recorded runs of internal_4 (ERKS,
ALEX, SIMI; BlockShapedPatchCityWorld and its scenic copy; the swarm, one drone and the 360 sphere):

  recognition   how readily a pilot recognises the hat as a function of its distance from the camera showing it:
                a hazard (presses per second of exposure) by distance, fitted by maximum likelihood to every moment
                the hat was on screen before each press (`exposure`, `hazard_table`, `fit_hazard`)
  discovery     how far away a goal's crowd is noticed: on the legs where the pilot turned onto the goal for the final
                approach, the range at which the walkers had first been on screen (`discovery_ranges`)
  behaviour     how each interface was flown between goals and at them: speed, pauses, course turns, the yaw stick,
                the head (`behaviour`)

    python -B -m citysim.views_human all          # every table -> experiment_results/views_study/human/
    python -B -m citysim.views_human recognition  # the hat's exposure and the hazard fit only

A sample is 0.1 s of the run's own logs (_drones.csv, _walkers.csv, _head.csv). The hat counts as ON SCREEN at a
sample when, for some alive drone whose feed was shown (the recorded `shown` column; the sphere's one drone always):
it is within the camera's image as the recorder's AutoScore tests it (the central identifyCentralFraction of the
viewport, both axes, at the recorded gimbal pitch; the sphere has no frame), with a clear 3-D line to the hat through
the city's prisms. Its distance is the nearest such camera's. ON HEAD adds the HMD: for screens, the hat's display
azimuth (the feed's yaw less the body yaw, plus its bearing off the camera axis times the screens' angular scale)
within HMD_HALF_H of the head's recorded yaw; for the sphere, the hat's direction within HMD_HALF_H in yaw and
HMD_HALF_V in pitch of where the head faced in the city. Nothing here is a model: it is what was on the pilot's screens.
"""
if __name__ == "__main__" and __package__ in (None, ""):
    import os.path as _p
    import sys as _s
    _s.path[0] = _p.dirname(_p.dirname(_p.abspath(__file__)))

import sys

sys.dont_write_bytecode = True

import argparse  # noqa: E402
import contextlib  # noqa: E402
import io  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import re  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from citysim import REPO, camera, use_bench_modules  # noqa: E402
from citysim.los import Boxes, wrap_deg  # noqa: E402

TEST = "internal_4"
OUT = REPO / "experiment_results" / "views_study" / "human"
CENTRAL = 1.0                    # the share of the image a hat counts in: the whole frame, as the recorder scores
                                 # since 2026-11-18 (0.8, identifyCentralFraction, before; S12's first fits used it)
HMD_PX_PER_DEG = 22.0            # Quest Pro, what the headset resolves (CLAUDE.md: ~22 px/deg)
PIXELS = True                    # distances in pixel-equivalent metres (see `resolution`); False: plain metres
SCREEN_RADIUS_M = 2.0
HMD_HALF_H = 50.0                # Quest Pro, about the head's forward (camera.HMD_HALF_FOV_DEG)
HMD_HALF_V = 45.0
ZONE_M = 15.0                    # analyse.py's default phase zone
LOOK_BACK_S = 5.0                # ExperimentRecorder.identifyMaxLookbackSec (since 2026-11-10)
LATENCY_S = (0.5, 1.0, 2.0)      # recognition -> press, primary 1.0
BINS = (0, 10, 15, 20, 25, 30, 40, 50, 60, 80, 100, 130)
R_MAX = 130.0                    # nothing past this is evaluated


def runs(test=TEST):
    """(manifest, [analyse.Run]) with the start line set, quietly."""
    use_bench_modules()
    import analyse
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        _, manifest, rr = analyse.load_test(analyse.DEFAULT_ROOT, test)
    for r in rr:
        r.set_start("line")
    return manifest, rr


def presses(run):
    """Every RC identify of a run: [dict(t, goal, outcome, camDist, passAgo)], with test.json's overrides applied (an
    overridden press counts correct for its goal, as analyse does)."""
    out = []
    with open(run.path("events.csv"), encoding="utf-8") as f:
        lines = f.read().splitlines()[1:]
    for line in lines:
        p = line.split(";", 6)
        if len(p) < 5 or p[2] != "identify":
            continue
        d = dict(re.findall(r"(\w+)=([-\w.,]+)", p[6] if len(p) > 6 else ""))
        out.append(dict(t=float(p[0]), goal=int(p[3]), outcome=p[4], camDist=float(d.get("camDist", "nan")),
                        passAgo=float(d.get("passAgo", "nan")), source=d.get("source", "")))
    for g in run.goals:                     # analyse.apply_identify_overrides wrote the override into the goal
        if g.get("scoredAs") and g.get("decisionTimeSec") is not None:
            t = float(g["decisionTimeSec"])
            for e in out:
                if abs(e["t"] - t) < 1e-3:
                    e.update(goal=int(g["goalIndex"]), outcome="correct", override=True)
    return out


def _screen_scale(run):
    """Degrees on the screen per degree in the camera: the quad's width at the session's scale over the camera's HFOV."""
    s = float(run.session.get("screenScale", 1.0) or 1.0)
    r = float(run.session.get("screenRadius", SCREEN_RADIUS_M) or SCREEN_RADIUS_M)
    quad = math.degrees(2 * math.atan(s * camera.FEED_ASPECT / 2 / r))
    return quad / camera.HFOV_DEG


def resolution(feed_w=1920, screen_scale=1.3, screen_radius=SCREEN_RADIUS_M, sphere=False, cube=2048, stream=0):
    """Pixels per WORLD degree the pilot's eye receives, the lesser of what the image holds and what the headset shows
    of it: a feed holds feed_w / HFOV and its screen shows HFOV in `quad` degrees of the headset's 22 px/deg; the
    sphere is shown at 1:1, from the cube (cube / 90 px/deg at a face's centre) or, since 2026-11-18, from the 2K live
    stream (stream / 360). 1920 feeds at scale 1.3: 17.7; the cube: 22.0; the 2048 stream: 5.7."""
    if sphere:
        src = (stream / 360.0) if stream else (cube / 90.0)
        return min(src, HMD_PX_PER_DEG)
    quad = math.degrees(2 * math.atan(screen_scale * camera.FEED_ASPECT / 2 / screen_radius))
    return min(feed_w / camera.HFOV_DEG, HMD_PX_PER_DEG * quad / camera.HFOV_DEG)


R_REF = resolution()                # the trials' feeds (1920 x 1080 at screen scale 1.3): the reference


def px_factor(run):
    """The distance multiplier that turns a run's metres into pixel-equivalent metres at R_REF: a hat d metres away
    covers as many pixels of what the eye receives as one d * factor metres away on the reference feeds. Recognition
    of a 21 cm hat is limited by the pixels across it (Johnson's criteria; range at a fixed probability is proportional
    to resolution), so a hazard fitted in these units carries to a display of another resolution by its factor."""
    if not PIXELS:
        return 1.0
    s = run.session or {}
    sp = s.get("sphere360") or {}
    r = resolution(float(s.get("feedWidth", 1920) or 1920), float(s.get("screenScale", 1.3) or 1.3),
                   float(s.get("screenRadius", SCREEN_RADIUS_M) or SCREEN_RADIUS_M), sphere=_sphere(run),
                   cube=float(sp.get("cubeSize", 2048) or 2048), stream=float(sp.get("streamWidth", 0) or 0))
    return R_REF / r


def _sphere(run):
    from citysim import replay
    return replay.sphere_view(run)[0]


def exposure(run, boxes=None):
    """Per goal, a DataFrame on the drone-log clock: t, dt, the hat's nearest distance ON SCREEN (`d_screen`) and ON
    HEAD (`d_head`) -- NaN when not --, `d_full` (the whole image rather than its central part, on screen), and the
    phase (`zone`: the centroid within ZONE_M of the goal's patch). {goalIndex: df}."""
    use_bench_modules()
    import analyse
    if boxes is None:
        boxes = analyse.scene_boxes(run.goals, run.city, run.goal_width)
    B = Boxes(boxes)
    dr = run.drones
    T, dtc = run.ct, run.cdt
    ids = sorted(dr.droneId.unique())
    piv = {k: dr.pivot_table(index="t", columns="droneId", values=k).reindex(T)[ids].values
           for k in ("gtX", "gtY", "gtZ", "yawDeg", "alive")}
    from citysim import replay
    A = np.nan_to_num(piv["alive"]) > 0.5
    shown = replay.shown_mask(run) & A
    h = run.head
    hi = np.clip(np.searchsorted(h.t.values, T), 0, len(h) - 1)
    body = h.bodyYaw.values[hi]
    pitch = np.nan_to_num(h.gimbalPitch.values[hi]) if "gimbalPitch" in h else np.zeros(len(T))
    sphere = _sphere(run)
    scale = _screen_scale(run)
    if sphere:
        head_city = wrap_deg(h.headYaw.values[hi] - h.sphereYaw.values[hi])
        head_pitch = -h.headPitch.values[hi]          # Unity's x rotation is positive down
    else:
        head_rel = wrap_deg(h.headYaw.values[hi] - body - 90.0)
    cam = np.stack([piv["gtX"], piv["gtY"], piv["gtZ"]], -1)
    psi = np.radians(piv["yawDeg"])
    th = np.radians(pitch)[:, None]
    fwd = np.stack([np.sin(psi) * np.cos(th), np.broadcast_to(np.sin(th), psi.shape), np.cos(psi) * np.cos(th)], -1)
    right = np.stack([np.cos(psi), np.zeros_like(psi), -np.sin(psi)], -1)
    up = np.stack([-np.sin(psi) * np.sin(th), np.broadcast_to(np.cos(th), psi.shape), -np.cos(psi) * np.sin(th)], -1)
    tx, ty = math.tan(math.radians(camera.HFOV_DEG / 2)), math.tan(math.radians(camera.VFOV_DEG / 2))
    out = {}
    for gi, g in enumerate(run.goals):
        gidx = g.get("goalIndex", gi)
        w = run.walkers[run.walkers.goalIndex == gidx].set_index("t")[["hatX", "hatY", "hatZ"]]
        hat = w[~w.index.duplicated()].reindex(T).values
        have = np.isfinite(hat).all(1)
        rel = hat[:, None, :] - cam
        dist = np.linalg.norm(rel, axis=2)
        base = A & shown & have[:, None] & (dist <= R_MAX)
        if sphere:
            in80 = infull = base
        else:
            z = (rel * fwd).sum(-1)
            x = (rel * right).sum(-1) / np.maximum(z, 1e-6)
            y = (rel * up).sum(-1) / np.maximum(z, 1e-6)
            infull = base & (z > 0) & (np.abs(x) <= tx) & (np.abs(y) <= ty)
            in80 = infull & (np.abs(x) <= CENTRAL * tx) & (np.abs(y) <= CENTRAL * ty)
        vis = infull.copy()
        if infull.any():
            ti, di = np.nonzero(infull)
            centre = np.nanmean(hat[have][:, [0, 2]], axis=0) if have.any() else np.zeros(2)
            sub = B.subset(B.near(centre, R_MAX + 80))
            vis[ti, di] = ~sub.blocked(cam[ti, di], hat[ti])
        bearing = np.degrees(np.arctan2(rel[..., 0], rel[..., 2]))
        if sphere:
            elev = np.degrees(np.arctan2(rel[..., 1], np.hypot(rel[..., 0], rel[..., 2])))
            on_head = (np.abs(wrap_deg(bearing - head_city[:, None])) <= HMD_HALF_H) & \
                      (np.abs(elev - head_pitch[:, None]) <= HMD_HALF_V)
        else:
            off = wrap_deg(bearing - piv["yawDeg"])
            az = wrap_deg(piv["yawDeg"] - body[:, None] + off * scale)
            on_head = np.abs(wrap_deg(az - head_rel[:, None])) <= HMD_HALF_H
            if run.condition != "Swarm":
                on_head[:] = True
        D = np.where(dist > 0, dist, np.nan)
        d_screen = np.nanmin(np.where(vis & in80, D, np.inf), axis=1)
        d_head = np.nanmin(np.where(vis & in80 & on_head, D, np.inf), axis=1)
        d_full = np.nanmin(np.where(vis, D, np.inf), axis=1)
        n_screen = (vis & in80).sum(1)
        zone = analyse.tile_distance(run.cx, run.cz, g) <= ZONE_M
        df = pd.DataFrame(dict(t=T, dt=dtc, d_screen=d_screen, d_head=d_head, d_full=d_full, n_screen=n_screen,
                               zone=zone))
        for c in ("d_screen", "d_head", "d_full"):
            df.loc[~np.isfinite(df[c]), c] = np.nan
        out[gidx] = df
    return out


def recognition_events(rr):
    """One row per goal of every run: its correct press (time, the recorded camDist), and the windows the hazard is
    counted over -- from the start of the task to the press less the latency -- with the exposure tables kept for
    the fit. ([row], {(run, goal): exposure df})."""
    rows, ex = [], {}
    for r in rr:
        e = exposure(r)
        pp = presses(r)
        for gi, g in enumerate(r.goals):
            gidx = g.get("goalIndex", gi)
            ok = [p for p in pp if p["goal"] == gidx and p["outcome"] == "correct"]
            fa = [p for p in pp if p["goal"] == gidx and p["outcome"] == "false_alarm"]
            press = ok[0]["t"] if ok else math.nan
            df = e[gidx]
            ex[(r.stem, gidx)] = df
            rows.append(dict(run=r.stem, pid=r.pid, condition=r.condition, goal=gidx, hat=str(g.get("hat", "")),
                             press_t=press, camDist=ok[0]["camDist"] if ok else math.nan,
                             override=bool(ok and ok[0].get("override")), false_alarms=len(fa),
                             start=r.start if r.start is not None else 0.0,
                             px=px_factor(r),
                             first_screen_t=_first(df, "d_screen"), first_head_t=_first(df, "d_head"),
                             min_screen_before=_min_before(df, "d_screen", press),
                             min_head_before=_min_before(df, "d_head", press),
                             screen_s_before=_secs_before(df, "d_screen", press, 1e9),
                             screen30_s_before=_secs_before(df, "d_screen", press, 30.0),
                             head30_s_before=_secs_before(df, "d_head", press, 30.0)))
    return rows, ex


def _first(df, col):
    k = np.flatnonzero(np.isfinite(df[col].to_numpy()))
    return float(df.t.iloc[k[0]]) if len(k) else math.nan


def _min_before(df, col, t):
    if not math.isfinite(t):
        return math.nan
    s = df[col][df.t <= t]
    return float(s.min()) if s.notna().any() else math.nan


def _secs_before(df, col, t, dmax):
    if not math.isfinite(t):
        t = float(df.t.iloc[-1])
    m = (df.t <= t) & (df[col] <= dmax)
    return float(df.dt[m].sum())


# ============================================================================= the hazard

def _event_distance(df, col, t_press, latency):
    """The distance credited with the recognition behind a press at t_press: the nearest the hat was ON the basis in
    the LOOK_BACK_S before t_press - latency + ... : the last sample on screen at or before t_press - latency, within
    the look-back; NaN if the hat was not on that basis anywhere in [t_press - latency - LOOK_BACK_S, t_press]."""
    t = df.t.to_numpy()
    d = df[col].to_numpy()
    m = (t <= t_press - latency) & (t >= t_press - latency - LOOK_BACK_S) & np.isfinite(d)
    if m.any():
        return float(d[np.flatnonzero(m)[-1]]), float(t[np.flatnonzero(m)[-1]])
    m = (t <= t_press) & (t >= t_press - latency - LOOK_BACK_S) & np.isfinite(d)
    if m.any():                         # on screen only within the latency: credited, flagged short
        return float(d[np.flatnonzero(m)[0]]), float(t[np.flatnonzero(m)[0]])
    return math.nan, math.nan


def exposure_events(rows, ex, col="d_head", latency=1.0, conditions=None, t_from="start"):
    """(d, dt) of every exposed sample in the hazard's windows, and the event distances: (exp_d, exp_dt, ev_d,
    unexplained). A goal's window runs from `t_from` (the task start, or `zone` the first zone entry) to the
    credited sample of its press; a goal never pressed contributes its whole run as exposure with no event."""
    exp_d, exp_dt, ev_d, unexplained = [], [], [], 0
    for row in rows:
        if conditions and row["condition"] not in conditions:
            continue
        df = ex[(row["run"], row["goal"])]
        t0 = row["start"]
        if t_from == "zone":
            z = np.flatnonzero(df.zone.to_numpy())
            t0 = float(df.t.iloc[z[0]]) if len(z) else math.inf
        strict = False
        if math.isfinite(row["press_t"]):
            de, te = _event_distance(df, col, row["press_t"], latency)
            if not math.isfinite(de):
                unexplained += 1
                t1 = row["press_t"] - latency
            else:
                ev_d.append(de)
                t1, strict = te, True        # the credited sample is the event, not exposure before it
        else:
            t1 = float(df.t.iloc[-1])
        m = (df.t >= t0) & ((df.t < t1) if strict else (df.t <= t1)) & df[col].notna()
        if strict:
            ev_d[-1] *= row.get("px", 1.0)
        exp_d.append(df[col][m].to_numpy() * row.get("px", 1.0))
        exp_dt.append(df.dt[m].to_numpy())
    return (np.concatenate(exp_d) if exp_d else np.zeros(0), np.concatenate(exp_dt) if exp_dt else np.zeros(0),
            np.array(ev_d), unexplained)


VERIFY_GAP_S = 2.0


def exposure_events_start(rows, ex, col="d_head", conditions=None):
    """exposure_events with each recognition credited to the START of the final continuous sighting before its press
    (gaps up to VERIFY_GAP_S joined): the earliest the pilot can have recognised the hat. Pilots sometimes held the hat
    in view and flew closer before pressing (11 of 41 presses closed in by 10 m or more over 3 s or more), which the
    press-time credit reads as recognition up close and the approach as unsuccessful exposure. The two credits bound the
    truth. Returns (exp_d, exp_dt, ev_d)."""
    exp_d, exp_dt, ev_d = [], [], []
    for r in rows:
        if conditions and r["condition"] not in conditions:
            continue
        df = ex[(r["run"], r["goal"])]
        t, d, dt = df.t.to_numpy(), df[col].to_numpy(), df.dt.to_numpy()
        t1 = r["press_t"] if math.isfinite(r["press_t"]) else t[-1]
        on = np.flatnonzero(np.isfinite(d) & (t <= t1))
        if math.isfinite(r["press_t"]) and len(on) and r["press_t"] - t[on[-1]] <= LATENCY_S[-1] + LOOK_BACK_S - 1.0:
            j = len(on) - 1
            while j > 0 and t[on[j]] - t[on[j - 1]] <= VERIFY_GAP_S:
                j -= 1
            ev_d.append(d[on[j]] * r.get("px", 1.0))
            m = (t >= r["start"]) & (t < t[on[j]]) & np.isfinite(d)
        else:
            m = (t >= r["start"]) & (t <= t1) & np.isfinite(d)
        exp_d.append(d[m] * r.get("px", 1.0))
        exp_dt.append(dt[m])
    return np.concatenate(exp_d), np.concatenate(exp_dt), np.array(ev_d)


def hazard_table(exp_d, exp_dt, ev_d, bins=BINS):
    """Per distance bin: exposure seconds, events, the hazard (events per second) with its exact Poisson 95%
    interval."""
    from scipy import stats
    out = []
    for a, b in zip(bins[:-1], bins[1:]):
        s = float(exp_dt[(exp_d >= a) & (exp_d < b)].sum())
        k = int(((ev_d >= a) & (ev_d < b)).sum())
        lo = stats.chi2.ppf(0.025, 2 * k) / 2 if k else 0.0
        hi = stats.chi2.ppf(0.975, 2 * k + 2) / 2
        out.append(dict(bin=f"{a}-{b}", d_lo=a, d_hi=b, exposure_s=round(s, 1), events=k,
                        hazard=round(k / s, 4) if s > 0 else math.nan,
                        lo=round(lo / s, 4) if s > 0 else math.nan, hi=round(hi / s, 4) if s > 0 else math.nan))
    return pd.DataFrame(out)


def hill(d, lam0, d50, k):
    """The recognition hazard at distance d: lam0 / (1 + (d / d50)^k), per second on screen."""
    return lam0 / (1.0 + (np.asarray(d, float) / d50) ** k)


def fit_hazard(exp_d, exp_dt, ev_d):
    """Maximum likelihood (an inhomogeneous Poisson process in the time on screen) of hill(): (lam0, d50, k), the
    log-likelihood, and a profile 95% interval on d50."""
    from scipy import optimize

    def nll(p):
        lam0, d50, k = math.exp(p[0]), math.exp(p[1]), math.exp(p[2])
        return -(np.log(hill(ev_d, lam0, d50, k)).sum() - (hill(exp_d, lam0, d50, k) * exp_dt).sum())

    best = None
    for l0 in (0.05, 0.2, 1.0):
        for d0 in (15.0, 30.0, 60.0):
            for k0 in (1.5, 3.0, 6.0):
                r = optimize.minimize(nll, [math.log(l0), math.log(d0), math.log(k0)], method="Nelder-Mead",
                                      options=dict(maxiter=4000, xatol=1e-6, fatol=1e-8))
                if best is None or r.fun < best.fun:
                    best = r
    lam0, d50, k = (math.exp(v) for v in best.x)
    ll = -best.fun

    def prof(d):
        r = optimize.minimize(lambda q: nll([q[0], math.log(d), q[1]]), [best.x[0], best.x[2]], method="Nelder-Mead")
        return -r.fun

    grid = np.exp(np.linspace(math.log(max(d50 / 4, 3.0)), math.log(d50 * 4), 81))
    pl = np.array([prof(d) for d in grid])
    inside = grid[pl >= ll - 1.92]
    return dict(lam0=lam0, d50=d50, k=k, loglik=ll, d50_lo=float(inside.min()), d50_hi=float(inside.max()),
                n_events=int(len(ev_d)), exposure_s=float(exp_dt.sum()))


def fit_hazard_ph(parts):
    """A proportional-hazards version of fit_hazard: one shape (d50, k) shared by every condition, lam0 per condition.
    parts = {condition: (exp_d, exp_dt, ev_d)}. Returns the fit (lam0 per condition) with its log-likelihood, and the
    likelihood-ratio test against one lam0 for all (df = conditions - 1)."""
    from scipy import optimize, stats
    conds = sorted(parts)

    def nll(p, common=False):
        d50, k = math.exp(p[0]), math.exp(p[1])
        ll = 0.0
        for i, c in enumerate(conds):
            lam0 = math.exp(p[2] if common else p[2 + i])
            ed, edt, ev = parts[c]
            with np.errstate(over="ignore", divide="ignore"):
                ll += np.log(hill(ev, lam0, d50, k)).sum() - (hill(ed, lam0, d50, k) * edt).sum()
        return -ll

    best, best0 = None, None
    for d0 in (15.0, 30.0, 60.0):
        for k0 in (2.0, 5.0):
            for l0 in (0.1, 0.5):
                x = [math.log(d0), math.log(k0)] + [math.log(l0)] * len(conds)
                r = optimize.minimize(nll, x, method="Nelder-Mead", options=dict(maxiter=20000, xatol=1e-7,
                                                                                    fatol=1e-9))
                if best is None or r.fun < best.fun:
                    best = r
                r0 = optimize.minimize(lambda q: nll(q, True), x[:3], method="Nelder-Mead",
                                       options=dict(maxiter=20000))
                if best0 is None or r0.fun < best0.fun:
                    best0 = r0
    lr = 2.0 * (best0.fun - best.fun)
    return dict(d50=math.exp(best.x[0]), k=math.exp(best.x[1]),
                lam0={c: math.exp(best.x[2 + i]) for i, c in enumerate(conds)}, loglik=-best.fun,
                loglik_common=-best0.fun, lr=lr, df=len(conds) - 1, p=float(stats.chi2.sf(lr, len(conds) - 1)),
                n_events={c: int(len(parts[c][2])) for c in conds})


def pass_probability(fit, d, seconds):
    """P(recognised) for a hat held on screen at distance d for `seconds`."""
    return 1.0 - math.exp(-hill(d, fit["lam0"], fit["d50"], fit["k"]) * seconds)


# ============================================================================= discovery: the crowd

DISC_R_MAX = 350.0
DISC_EVERY_S = 0.5


def crowd_distance(run, g, boxes, t0, t1, every_s=DISC_EVERY_S, r_max=DISC_R_MAX):
    """For goal g of a run, every `every_s` in [t0, t1]: the distance within which at least one of its walkers is
    expected on screen (citysim.discovery's expected walkers, ON SCREEN: a shown feed's full image at the recorded
    pitch, a clear 3-D line), on the screens (`d_screen`) and on the head (`d_head`, as `exposure` tests it). NaN
    when none within r_max. DataFrame(t, d_screen, d_head, cx, cz)."""
    use_bench_modules()
    import analyse
    from citysim import discovery, replay
    pts, w = analyse.goal_loop_points(run, g)
    dr = run.drones
    T = run.ct
    ids = sorted(dr.droneId.unique())
    piv = {k: dr.pivot_table(index="t", columns="droneId", values=k).reindex(T)[ids].values
           for k in ("gtX", "gtY", "gtZ", "yawDeg", "alive")}
    A = np.nan_to_num(piv["alive"]) > 0.5
    shown = replay.shown_mask(run) & A
    h = run.head
    hi = np.clip(np.searchsorted(h.t.values, T), 0, len(h) - 1)
    body = h.bodyYaw.values[hi]
    sphere = _sphere(run)
    scale = _screen_scale(run)
    if sphere:
        head_city = wrap_deg(h.headYaw.values[hi] - h.sphereYaw.values[hi])
    head_rel = wrap_deg(h.headYaw.values[hi] - body - 90.0)
    pitch = h.gimbalPitch.values[hi] if "gimbalPitch" in h else np.zeros(len(T))
    single = run.condition != "Swarm"
    sel = np.flatnonzero((T >= t0) & (T <= t1))
    if not len(sel):
        return pd.DataFrame(columns=["t", "d_screen", "d_head", "cx", "cz"])
    keep = sel[np.r_[0, np.flatnonzero(np.diff(np.floor(T[sel] / every_s)) > 0) + 1]]
    B = Boxes(boxes)
    centre = pts.mean(0)
    reach = float(np.max(np.linalg.norm(pts - centre, axis=1))) + r_max
    near = B.subset(B.near(centre, reach + 10.0))
    rows = []
    for k in keep:
        live = np.flatnonzero(shown[k])
        a = np.flatnonzero(A[k])
        cx = float(np.nanmean(piv["gtX"][k, a])) if len(a) else math.nan
        cz = float(np.nanmean(piv["gtZ"][k, a])) if len(a) else math.nan
        best_s = np.full(len(pts), np.inf)
        best_h = np.full(len(pts), np.inf)
        for d_i in live:
            c = np.array([piv["gtX"][k, d_i], piv["gtY"][k, d_i], piv["gtZ"][k, d_i]])
            rel = np.c_[pts[:, 0] - c[0], np.full(len(pts), discovery.TARGET_H - c[1]), pts[:, 1] - c[2]]
            dist = np.linalg.norm(rel, axis=1)
            hd = np.hypot(rel[:, 0], rel[:, 2])
            bearing = np.degrees(np.arctan2(rel[:, 0], rel[:, 2]))
            off = wrap_deg(bearing - piv["yawDeg"][k, d_i])
            ok = dist <= r_max
            if not sphere:
                ok &= np.abs(off) <= camera.HFOV_DEG / 2
                if np.isfinite(pitch[k]):
                    from citysim.los import vertical_band
                    ok &= vertical_band(c[1], pitch[k], hd)
            idx = np.flatnonzero(ok & (dist < best_s))
            if not len(idx):
                continue
            tgt = np.c_[pts[idx, 0], np.full(len(idx), discovery.TARGET_H), pts[idx, 1]]
            clear = ~near.blocked(np.broadcast_to(c, (len(idx), 3)), tgt) if near.n else np.ones(len(idx), bool)
            v = idx[clear]
            best_s[v] = np.minimum(best_s[v], dist[v])
            if sphere:
                hv = np.abs(wrap_deg(bearing[v] - head_city[k])) <= HMD_HALF_H
            elif single:
                hv = np.ones(len(v), bool)
            else:
                az = wrap_deg(piv["yawDeg"][k, d_i] - body[k] + off[v] * scale)
                hv = np.abs(wrap_deg(az - head_rel[k])) <= HMD_HALF_H
            best_h[v[hv]] = np.minimum(best_h[v[hv]], dist[v[hv]])
        rows.append(dict(t=float(T[k]), d_screen=_d_one(best_s, w), d_head=_d_one(best_h, w), cx=cx, cz=cz))
    return pd.DataFrame(rows)


def _d_one(best, w):
    """The distance within which the walkers expected on screen first reach one (discovery.SEEN_WALKERS)."""
    o = np.argsort(best)
    cw = np.cumsum(w[o])
    k = np.searchsorted(cw, 1.0 - 1e-9)
    return float(best[o][k]) if k < len(o) and np.isfinite(best[o][k]) else math.nan


def discovery_ranges(rr):
    """Per transit leg (analyse.goal_legs at ZONE_M, from the start line): the leg's start, zone entry and commit (the
    start of the final approach: analyse.commit_time), and the crowd's distance series over the whole run up to the
    zone entry. Per leg: `closest_by_commit` -- the nearest the crowd had been on the head basis by the commit, a lower
    bound on the distance at which it was noticed (it cannot have been noticed closer than it ever was) -- and the
    same on the screens, plus the distance at the commit itself. ([row], {(run, leg): series})."""
    use_bench_modules()
    import analyse
    rows, series = [], {}
    for r in rr:
        boxes = analyse.scene_boxes(r.goals, r.city, r.goal_width)
        legs = analyse.goal_legs(r, ZONE_M, rule="zone")
        for L in legs:
            g = L["goal"]
            c = analyse.commit_time(r, L)
            s = crowd_distance(r, g, boxes, (r.start or 0.0), L["enter"])
            series[(r.stem, L["leg"])] = s
            before = s[s.t <= (c if math.isfinite(c) else L["enter"])]
            at = s[(s.t >= c - 2.0) & (s.t <= c)] if math.isfinite(c) else s.iloc[0:0]
            leg_s = s[s.t >= L["t0"]]
            rows.append(dict(run=r.stem, pid=r.pid, condition=r.condition, leg=L["leg"], t0=L["t0"], px=px_factor(r),
                             enter=L["enter"], commit=c, commit_s=c - L["t0"] if math.isfinite(c) else math.nan,
                             transit_s=L["enter"] - L["t0"],
                             closest_head_by_commit=float(before.d_head.min()) if before.d_head.notna().any()
                             else math.nan,
                             closest_screen_by_commit=float(before.d_screen.min()) if before.d_screen.notna().any()
                             else math.nan,
                             head_at_commit=float(at.d_head.min()) if at.d_head.notna().any() else math.nan,
                             seen_head_at_leg_start=float(leg_s.d_head.iloc[0]) if len(leg_s) else math.nan,
                             centroid_goal_m_at_commit=_goal_dist(r, g, c)))
    return rows, series


def _goal_dist(run, g, t):
    if not math.isfinite(t):
        return math.nan
    i = min(int(np.searchsorted(run.ct, t)), len(run.ct) - 1)
    return float(math.hypot(run.cx[i] - g["goalX"], run.cz[i] - g["goalZ"]))


DISC_LATENCY_S = 2.0             # noticing a crowd -> the turn onto it


NOTICE_K = 8.0                   # the noticing hazard's fixed steepness (see fit_notice)


def fit_notice(legs, series, col="d_head", latency=DISC_LATENCY_S, bins=(0, 50, 100, 150, 200, 250, 300, 350),
               k_fixed=NOTICE_K):
    """The crowd-noticing hazard mu(d) = hill(d; mu0, d50, k), by maximum likelihood with interval censoring: a leg
    whose final approach began more than 1 s into it is an event at its commit less `latency` (credited the crowd's
    distance then, or the nearest sample on screen within LOOK_BACK_S before it), after the exposure from the task
    start; a leg that began on its final approach had noticed the crowd at some unknown time before the leg began,
    so it contributes P(noticed by then) = 1 - exp(-Lambda(t0)). Each sample is DISC_EVERY_S. Returns the fit and a
    binned table of the events with their exposure (the events only, so it understates the hazard by the censored
    legs).

    k is fixed (k_fixed; None frees it). With YEGO's legs the likelihood is flat in the shape: k from 1 to 20 changes
    it by under 0.3 (the rate is about flat over the 16-206 m the events span, and nothing says how it ends), and the
    free fit wanders to a power law at other latencies. A near-step (k 8) is the plain reading: noticed at mu0 while
    within about d50."""
    from scipy import optimize
    ev, ev_exp, cens = [], [], []
    for _, L in legs.iterrows():
        s = series[(L.run, L.leg)]
        if not len(s):
            continue
        t, d = s.t.to_numpy(), s[col].to_numpy() * (float(L.px) if PIXELS and "px" in L and np.isfinite(L.px) else 1.0)
        if not math.isfinite(L.commit):
            continue
        if L.commit_s > 1.0:
            tc = L.commit - latency
            m = (t <= tc) & (t >= tc - LOOK_BACK_S) & np.isfinite(d)
            if not m.any():
                continue
            k = np.flatnonzero(m)[-1]
            exp_m = (t < t[k]) & np.isfinite(d)
            ev.append(float(d[k]))
            ev_exp.append(d[exp_m])
        else:
            exp_m = (t <= L.t0) & np.isfinite(d)
            if exp_m.any():
                cens.append(d[exp_m])
    dt = DISC_EVERY_S
    ev = np.array(ev)

    def nll(p):
        mu0, d50 = math.exp(p[0]), math.exp(p[1])
        k = k_fixed if k_fixed else math.exp(p[2])
        ll = np.log(hill(ev, mu0, d50, k)).sum() - sum((hill(x, mu0, d50, k) * dt).sum() for x in ev_exp)
        for x in cens:
            lam = float((hill(x, mu0, d50, k) * dt).sum())
            ll += math.log(max(-math.expm1(-lam), 1e-300))
        return -ll

    best = None
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        for m0 in (0.02, 0.1, 0.5):
            for d0 in (60.0, 120.0, 200.0):
                for k0 in (1.5, 3.0, 6.0):
                    r = optimize.minimize(nll, [math.log(m0), math.log(d0), math.log(k0)], method="Nelder-Mead",
                                          options=dict(maxiter=4000))
                    if best is None or r.fun < best.fun:
                        best = r
        mu0, d50, k = (math.exp(v) for v in best.x)
        if k_fixed:
            k = k_fixed
        grid = np.exp(np.linspace(math.log(20.0), math.log(800.0), 61))
        pl = []
        for d in grid:
            r = optimize.minimize(lambda q: nll([q[0], math.log(d), q[1]]), [best.x[0], best.x[2]],
                                  method="Nelder-Mead")
            pl.append(-r.fun)
    pl = np.array(pl)
    inside = grid[pl >= -best.fun - 1.92]
    allx = np.concatenate(ev_exp) if ev_exp else np.zeros(0)
    tab = []
    for a, b in zip(bins[:-1], bins[1:]):
        s = float(((allx >= a) & (allx < b)).sum() * dt)
        n = int(((ev >= a) & (ev < b)).sum())
        tab.append(dict(bin=f"{a}-{b}", exposure_s=round(s, 1), events=n, rate=round(n / s, 4) if s else math.nan))
    return dict(mu0=mu0, d50=d50, k=k, loglik=-best.fun, d50_lo=float(inside.min()), d50_hi=float(inside.max()),
                d50_hi_open=bool(inside.max() >= grid[-1] - 1e-6), k_fixed=k_fixed,
                n_events=int(len(ev)), n_censored=len(cens), latency=latency), pd.DataFrame(tab)


def cmd_discovery(a):
    OUT.mkdir(parents=True, exist_ok=True)
    _, rr = runs(a.test)
    rows, series = discovery_ranges(rr)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "discovery_legs.csv", index=False)
    with open(OUT / "discovery_series.json", "w", encoding="utf-8") as f:
        json.dump({f"{k[0]}|{k[1]}": v.round(2).to_dict("list") for k, v in series.items()}, f)
    pd.set_option("display.width", 220)
    print(df.drop(columns=["t0", "enter", "commit"]).round(1).to_string())
    fits = {}
    for col in ("d_head", "d_screen"):
        for lat in (1.0, 2.0, 4.0):
            fit, tab = fit_notice(df, series, col, lat)
            fits[f"{col}|lat{lat}"] = dict(fit=fit, table=tab.to_dict("records"))
            fits[f"free|{col}|lat{lat}"] = dict(fit=fit_notice(df, series, col, lat, k_fixed=None)[0])
            print(f"\nnotice {col} latency {lat}: mu0 {fit['mu0']:.3f}/s d50 {fit['d50']:.0f} m "
                  f"[{fit['d50_lo']:.0f}, {fit['d50_hi']:.0f}] k {fit['k']:.2f}; {fit['n_events']} events, "
                  f"{fit['n_censored']} censored")
            print(tab.to_string(index=False))
    with open(OUT / "discovery_fits.json", "w", encoding="utf-8") as f:
        json.dump(fits, f, indent=1)


def cmd_discovery_fit(a):
    """The noticing fit again from the saved series (no re-measuring)."""
    df = pd.read_csv(OUT / "discovery_legs.csv")
    with open(OUT / "discovery_series.json", encoding="utf-8") as f:
        raw = json.load(f)
    series = {(k.split("|")[0], int(k.split("|")[1])): pd.DataFrame(v) for k, v in raw.items()}
    fits = {}
    for col in ("d_head", "d_screen"):
        for lat in (1.0, 2.0, 4.0):
            fit, tab = fit_notice(df, series, col, lat)
            fits[f"{col}|lat{lat}"] = dict(fit=fit, table=tab.to_dict("records"))
            fits[f"free|{col}|lat{lat}"] = dict(fit=fit_notice(df, series, col, lat, k_fixed=None)[0])
            print(f"\nnotice {col} latency {lat}: mu0 {fit['mu0']:.3f}/s d50 {fit['d50']:.0f} m "
                  f"[{fit['d50_lo']:.0f}, {fit['d50_hi']:.0f}] k {fit['k']:.2f}; {fit['n_events']} events, "
                  f"{fit['n_censored']} censored")
            print(tab.to_string(index=False))
    with open(OUT / "discovery_fits.json", "w", encoding="utf-8") as f:
        json.dump(fits, f, indent=1)


# ============================================================================= behaviour

STOP_MS = 1.0
YAWING_DEG_S = 10.0


def _signals(run):
    """Per drone-log sample: centroid velocity (m/s), the view's world yaw (the single drone's heading; the swarm's
    body yaw as the rig turns it; the sphere drone's heading), the head's world yaw (where the pilot looked in the
    city: for screens the display azimuth turned back to world through the body yaw, for the sphere headYaw -
    sphereYaw), and the stick."""
    T = run.ct
    h = run.head
    hi = np.clip(np.searchsorted(h.t.values, T), 0, len(h) - 1)
    body = h.bodyYaw.values[hi]
    dt = np.gradient(T)
    vx, vz = np.gradient(run.cx) / dt, np.gradient(run.cz) / dt
    one = run.drones[run.drones.droneId == run.drones.droneId.min()].set_index("t").reindex(T)
    heading = one.yawDeg.to_numpy()
    sphere = _sphere(run)
    if run.condition == "Swarm":
        view = body
        head_world = wrap_deg(body + wrap_deg(h.headYaw.values[hi] - body - 90.0))   # display az ~ world bearing
    elif sphere:
        view = heading
        head_world = wrap_deg(h.headYaw.values[hi] - h.sphereYaw.values[hi])
    else:
        view = heading
        head_world = wrap_deg(heading + wrap_deg(h.headYaw.values[hi] - body - 90.0))
    view_rate = np.gradient(np.unwrap(np.radians(view))) / dt
    stick = np.hypot(h.inPitch.values[hi], h.inRoll.values[hi])
    return dict(t=T, dt=run.cdt, vx=vx, vz=vz, speed=np.hypot(vx, vz), view=view, view_rate=np.degrees(view_rate),
                head_rel=wrap_deg(head_world - view), yaw_stick=np.abs(h.inYaw.values[hi]), stick=stick,
                roll=np.abs(h.inRoll.values[hi]), pitch_stick=np.abs(h.inPitch.values[hi]))


def behaviour(rr):
    """Per leg and phase (transit: leg start -> zone entry; search: zone entry -> exit), how it was flown: speed,
    share stopped, the view's yaw rate and the share yawing, the head's offset from the view, distance from the goal
    centre, the share of the search inside the goal's ring, and the course's angle to the view (flying sideways)."""
    use_bench_modules()
    import analyse
    rows = []
    for r in rr:
        s = _signals(r)
        legs = analyse.goal_legs(r, ZONE_M, rule="zone")
        for L in legs:
            g = L["goal"]
            dist = np.hypot(r.cx - g["goalX"], r.cz - g["goalZ"])
            rg = float(g.get("r_g") or g.get("ringRadius") or np.nan)
            for phase, a, b in (("transit", L["t0"], L["enter"]), ("search", L["enter"], L["exit"])):
                m = (s["t"] >= a) & (s["t"] < b)
                if m.sum() < 5:
                    continue
                dt = s["dt"][m]
                mv = m & (s["speed"] >= STOP_MS)
                course = np.degrees(np.arctan2(s["vx"], s["vz"]))
                side = np.abs(wrap_deg(course - s["view"]))
                rows.append(dict(run=r.stem, pid=r.pid, condition=r.condition, leg=L["leg"], phase=phase,
                                 dur_s=float(dt.sum()), speed=float((s["speed"][m] * dt).sum() / dt.sum()),
                                 speed_moving=float(np.median(s["speed"][mv])) if mv.any() else math.nan,
                                 stopped=float(dt[s["speed"][m] < STOP_MS].sum() / dt.sum()),
                                 yaw_rate_abs=float((np.abs(s["view_rate"][m]) * dt).sum() / dt.sum()),
                                 yawing=float(dt[np.abs(s["view_rate"][m]) > YAWING_DEG_S].sum() / dt.sum()),
                                 head_abs=float(np.median(np.abs(s["head_rel"][m]))),
                                 head_p90=float(np.percentile(np.abs(s["head_rel"][m]), 90)),
                                 sideways=float(np.median(side[mv])) if mv.any() else math.nan,
                                 sideways_gt45=float(dt[(side > 45)[m] & (s["speed"][m] >= STOP_MS)].sum() /
                                                     max(dt[s["speed"][m] >= STOP_MS].sum(), 1e-9)),
                                 goal_dist_med=float(np.median(dist[m])), goal_dist_p10=float(np.percentile(dist[m], 10)),
                                 inside_ring=float(dt[dist[m] < rg].sum() / dt.sum()) if math.isfinite(rg) else math.nan,
                                 r_g=rg))
    return pd.DataFrame(rows)


DWELL_TOL_DEG = 7.5


def head_dwells(rr):
    """The head's offset from the view (where it looked relative to the single drone's or the sphere drone's heading,
    or the swarm's body yaw), cut into dwells: maximal runs that stay within DWELL_TOL_DEG of their first sample.
    {condition: {phase: [(mean offset deg, duration s)]}}, the simulation's head process resampling these pairs as
    S2's fitted head did."""
    use_bench_modules()
    import analyse
    out = {}
    for r in rr:
        s = _signals(r)
        legs = analyse.goal_legs(r, ZONE_M, rule="zone")
        for L in legs:
            for phase, a, b in (("transit", L["t0"], L["enter"]), ("search", L["enter"], L["exit"])):
                m = np.flatnonzero((s["t"] >= a) & (s["t"] < b))
                if len(m) < 5:
                    continue
                x, dt = s["head_rel"][m], s["dt"][m]
                i = 0
                while i < len(x):
                    j = i
                    while j + 1 < len(x) and abs(wrap_deg(x[j + 1] - x[i])) <= DWELL_TOL_DEG:
                        j += 1
                    out.setdefault(r.condition, {}).setdefault(phase, []).append(
                        (round(float(np.degrees(np.angle(np.exp(1j * np.radians(x[i:j + 1])).mean()))), 1),
                         round(float(dt[i:j + 1].sum()), 2)))
                    i = j + 1
    return out


def head_series(rr):
    """The head's offset from the view (as head_dwells), per leg and phase, at 0.1 s: {condition: {phase: [[deg]]}}.
    The simulation's head process plays these traces from a random point (a block bootstrap), which keeps the head's
    timing as it was flown, where resampled dwells would not."""
    use_bench_modules()
    import analyse
    out = {}
    for r in rr:
        s = _signals(r)
        for L in analyse.goal_legs(r, ZONE_M, rule="zone"):
            for phase, a, b in (("transit", L["t0"], L["enter"]), ("search", L["enter"], L["exit"])):
                m = (s["t"] >= a) & (s["t"] < b)
                if m.sum() < 20:
                    continue
                t = s["t"][m]
                grid = np.arange(t[0], t[-1], 0.1)
                x = np.degrees(np.unwrap(np.radians(s["head_rel"][m])))
                y = wrap_deg(np.interp(grid, t, x))
                out.setdefault(r.condition, {}).setdefault(phase, []).append([round(float(v), 1) for v in y])
    return out


def cmd_head(a):
    OUT.mkdir(parents=True, exist_ok=True)
    _, rr = runs(a.test)
    with open(OUT / "head_series.json", "w", encoding="utf-8") as f:
        json.dump(head_series(rr), f)
    hd = head_dwells(rr)
    with open(OUT / "head_dwells.json", "w", encoding="utf-8") as f:
        json.dump(hd, f)
    for c, ph in hd.items():
        for p, v in ph.items():
            ang = np.array([x[0] for x in v])
            dur = np.array([x[1] for x in v])
            print(f"{c:12s} {p:8s} dwells {len(v):5d}  time-weighted |offset| median "
                  f"{np.median(np.repeat(np.abs(ang), np.maximum((dur * 10).astype(int), 1))):5.1f}  "
                  f"share >50: {dur[np.abs(ang) > 50].sum() / dur.sum():.2f}  dwell median {np.median(dur):.2f} s "
                  f"mean {dur.mean():.2f} s")


def goal_times(rr):
    """Per goal, in the order found, on the start-line clock: BETWEEN (the previous press, or the start line, to the
    goal's zone entry) and AT (zone entry to its correct press, overrides included), and the task time."""
    use_bench_modules()
    import analyse
    rows = []
    for r in rr:
        legs = {L["gi"]: L for L in analyse.goal_legs(r, ZONE_M, rule="zone")}
        pp = presses(r)
        start = r.start or 0.0
        found = []
        for gi, g in enumerate(r.goals):
            gidx = g.get("goalIndex", gi)
            ok = [p for p in pp if p["goal"] == gidx and p["outcome"] == "correct"]
            found.append((ok[0]["t"] if ok else math.nan, gi))
        found.sort()
        prev = start
        for t_press, gi in found:
            L = legs.get(gi)
            z = L["enter"] if L else math.nan
            # a press before the centroid ever entered the zone (seen from outside it): no at-goal time
            at = t_press - z if (math.isfinite(z) and t_press >= z) else (0.0 if math.isfinite(t_press) else math.nan)
            between = (min(z, t_press) if math.isfinite(z) else t_press) - prev
            rows.append(dict(run=r.stem, pid=r.pid, condition=r.condition, goal=gi, press=t_press - start,
                             between_s=max(between, 0.0), at_s=at, task_s=float(r.session.get("totalTaskTime", 0) or 0)
                             - start))
            prev = t_press
    return pd.DataFrame(rows)


def cmd_behaviour(a):
    OUT.mkdir(parents=True, exist_ok=True)
    _, rr = runs(a.test)
    gt = goal_times(rr)
    gt.to_csv(OUT / "goal_times.csv", index=False)
    print(gt.groupby("condition")[["between_s", "at_s"]].describe().round(1).to_string())
    df = behaviour(rr)
    df.to_csv(OUT / "behaviour_legs.csv", index=False)
    pd.set_option("display.width", 250)
    num = [c for c in df.columns if c not in ("run", "pid", "condition", "leg", "phase")]
    w = df.assign(**{c: df[c] * df.dur_s for c in num if c != "dur_s"})
    agg = w.groupby(["phase", "condition"])[num].sum()
    for c in num:
        if c != "dur_s":
            agg[c] = agg[c] / agg["dur_s"]
    print(agg.round(2).to_string())
    print(df.groupby(["phase", "condition", "pid"])[["dur_s", "speed", "stopped", "yaw_rate_abs", "yawing",
                                                      "head_abs", "sideways_gt45", "goal_dist_med",
                                                      "inside_ring"]].median().round(2).to_string())


# ============================================================================= main

def cmd_recognition(a):
    OUT.mkdir(parents=True, exist_ok=True)
    _, rr = runs(a.test)
    rows, ex = recognition_events(rr)
    goals = pd.DataFrame(rows)
    goals.to_csv(OUT / "recognition_goals.csv", index=False)
    pd.set_option("display.width", 220)
    print(goals.drop(columns=["start"]).to_string())
    res = {}
    for col in ("d_head", "d_screen"):
        for lat in LATENCY_S:
            for conds in (None, ("Swarm",), ("SingleDrone",), ("SingleSphere",)):
                key = f"{col}|lat{lat}|{'all' if not conds else conds[0]}"
                d, s, e, un = exposure_events(rows, ex, col, lat, conds)
                if not len(e):
                    continue
                tab = hazard_table(d, s, e)
                fit = fit_hazard(d, s, e)
                fit["unexplained"] = un
                res[key] = dict(fit=fit, table=tab.to_dict("records"))
                if lat == 1.0:
                    print(f"\n{key}: {len(e)} events, {un} unexplained, {s.sum():.0f} s exposed; "
                          f"lam0 {fit['lam0']:.3f}/s d50 {fit['d50']:.1f} m [{fit['d50_lo']:.1f}, {fit['d50_hi']:.1f}]"
                          f" k {fit['k']:.2f}")
                    print(tab.to_string(index=False))
    for pid in sorted({r["pid"] for r in rows}):                 # each pilot's own (one shape per pilot)
        d, s, e, un = exposure_events([r for r in rows if r["pid"] == pid], ex, "d_head", 1.0)
        fit = fit_hazard(d, s, e)
        res[f"pilot|{pid}"] = dict(fit=fit, n=int(len(e)), exposed_s=float(s.sum()),
                                   median_event_m=float(np.median(e)) if len(e) else None)
        print(f"pilot {pid}: {len(e)} events, lam0 {fit['lam0']:.2f}/s d50 {fit['d50']:.1f} m k {fit['k']:.2f}")
    for col in ("d_head", "d_screen"):
        d, s, e = exposure_events_start(rows, ex, col)
        fit = fit_hazard(d, s, e)
        res[f"start|{col}|all"] = dict(fit=fit, table=hazard_table(d, s, e).to_dict("records"))
        print(f"\nstart|{col}: {len(e)} events, {s.sum():.0f} s exposed; lam0 {fit['lam0']:.3f}/s d50 {fit['d50']:.1f} m "
              f"[{fit['d50_lo']:.1f}, {fit['d50_hi']:.1f}] k {fit['k']:.2f}")
    for col in ("d_head", "d_screen"):
        for lat in LATENCY_S:
            parts = {}
            for c in ("Swarm", "SingleDrone", "SingleSphere"):
                d, s, e, _ = exposure_events(rows, ex, col, lat, (c,))
                parts[c] = (d, s, e)
            ph = fit_hazard_ph(parts)
            res[f"ph|{col}|lat{lat}"] = dict(fit=ph)
            if lat == 1.0:
                print(f"\nph {col}: d50 {ph['d50']:.1f} k {ph['k']:.2f} lam0 " +
                      ", ".join(f"{c} {v:.3f}" for c, v in ph["lam0"].items()) +
                      f"; LR {ph['lr']:.2f} on {ph['df']} df, p = {ph['p']:.3f}")
    with open(OUT / "recognition_fits.json", "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("recognition")
    s.add_argument("--test", default=TEST)
    s.set_defaults(fn=cmd_recognition)
    s = sub.add_parser("discovery")
    s.add_argument("--test", default=TEST)
    s.set_defaults(fn=cmd_discovery)
    s = sub.add_parser("discovery-fit")
    s.set_defaults(fn=cmd_discovery_fit)
    s = sub.add_parser("behaviour")
    s.add_argument("--test", default=TEST)
    s.set_defaults(fn=cmd_behaviour)
    s = sub.add_parser("head")
    s.add_argument("--test", default=TEST)
    s.set_defaults(fn=cmd_head)
    a = p.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
