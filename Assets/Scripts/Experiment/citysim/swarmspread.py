"""S9 Street Spread (the spread study): how far S7's adaptive spread should narrow the formation in
BlockShapedPatchCityWorld's streets, and what the same spreads do held fixed.

BlockShapedPatchCityWorld flies S8's pick, C35a: the enclosing circle, the slide, d_obs 0.35 and the adaptive spread at
streetSpreadFactor 0.85 (d_ref 2.0 narrowed to 1.7 in the streets). In S8 it lost drones in 7 flights of ~104 km where
C35 (the same, adaptive off) lost them in 1, and Unity 3 pairs against 0. Two questions, flown in the replica on S8's
harness (swarmshape / swarmflow / swarmtune: ContactSim on the prism cities, settled starts, the bench's waypoint pilot
at full stick, S6's measures), with route indices from K_RUN:

  1. Does it narrow too much? streetSpreadFactor 0.85 / 0.90 / 0.95 against the adaptive spread off, at d_ref 2.0.
  2. The fixed spreads (lesson 42's check): d_ref 1.7 / 1.8 / 1.9 / 2.0 held, so a narrowed value that loses drones
     held fixed is told apart from the narrowing and widening themselves; and the pilot's dial ends (1.4, 2.6) with and
     without the adaptive spread, since EffectiveDRef has no floor (1.4 x 0.85 = 1.19 in the streets).

The flights lean on the streets, where the adaptive spread acts and S8's losses were (12 straight lines, 12 blocks and
60 street routes per city). Each loss is logged with the spread in force and its change over the preceding TRANS_S
(SpreadEventLog), so a loss can be placed in a narrowing, a widening or a steady spread.

Commands (from Assets/Scripts/Experiment, the stitching env):
  rule                        rule.json, written before the run
  run --name run1 --cand ...  the replica run (resumable)
  analyse --name run1         loss events with exact intervals, the tests the rule names, the spread at each loss,
                              -> <name>/analysis.json
  bench-plan / bench-gate     Unity (swarmshape's, into this study's folder)
"""
if __name__ == "__main__" and __package__ in (None, ""):
    import os.path as _p
    import sys as _s
    _s.path[0] = _p.dirname(_p.dirname(_p.abspath(__file__)))

import sys

sys.dont_write_bytecode = True

import argparse  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402

import numpy as np  # noqa: E402

from citysim import REPO  # noqa: E402
from citysim import swarmflow as F  # noqa: E402
from citysim import swarmshape as SS  # noqa: E402
from citysim import swarmtune as T  # noqa: E402

OUT = REPO / "experiment_results" / "swarm_spread"
K_RUN, K_BENCH = 10000, 11000          # S6-S8 used 0-9999
FLIGHTS = dict(cross=12, pass_=12, street=60)
TRANS_S = 3.0          # a loss is in a transition when the spread scale moved by TRANS_DS or more over this window
TRANS_DS = 0.02

# Fixed before the run (rule.json). base: the adaptive spread off (C35). A factor is safe when its loss events are not
# more than the base's at one-sided exact p < safe_p (conditional binomial on the two counts, equal exposure: the same
# routes), and its contacts are at most contact_ratio x the base's; it helps when the streets' cohesive speed is at
# least street_gain above the base's and it surrounds within wrap_margin of it. Pick: the narrowest safe and helping
# factor (the most street speed); none -> leave the adaptive spread off.
RULE = dict(base="F20", factors=["A85", "A90", "A95"], safe_p=0.10, contact_ratio=1.25, street_gain=0.03,
            wrap_margin=0.05)


class SpreadEventLog:
    """A read-only fly() hook: the spread scale at every step, and every loss with the scale then and TRANS_S earlier."""

    def __init__(self):
        self.t, self.sig, self.events, self.alive = [], [], [], None

    def start(self, sim):
        pass

    def step(self, sim, t, cmd, wp):
        s = 1.0 if sim.sigma is None else float(sim.sigma)
        if self.alive is not None:
            gone = np.flatnonzero(self.alive & ~sim.alive)
            if len(gone):
                self.events.append(dict(t=round(float(t), 2), n=int(len(gone)), sig=round(s, 4),
                                        sig_before=round(self.sig_at(t - TRANS_S), 4)))
        self.t.append(float(t))
        self.sig.append(s)
        self.alive = sim.alive.copy()

    def sig_at(self, t):
        if not self.t:
            return 1.0
        i = int(np.searchsorted(self.t, t))
        return self.sig[min(max(i, 0), len(self.sig) - 1)]

    def transition_share(self):
        """The share of steps in a transition (|scale change over TRANS_S| >= TRANS_DS)."""
        if len(self.t) < 2:
            return 0.0
        t, s = np.array(self.t), np.array(self.sig)
        j = np.clip(np.searchsorted(t, t - TRANS_S), 0, len(t) - 1)
        return float((np.abs(s - s[j]) >= TRANS_DS).mean())


class ShadowLog(SpreadEventLog):
    """SpreadEventLog plus the adaptive spread's *shadow*, for every row (run2, the timing test): the target the adaptive
    spread would set at this moment (swarm_replica.spread_target on the swarm's own centroid, stick and buildings, the
    state "in the streets" when it is under 1), low-passed at the scene's filter with the scene's factor
    (SHADOW_FACTOR). In an adaptive row it is the spread itself; in a fixed row it marks the same moments -- the swarm
    leaving a building into a street, or turning towards one -- without the spread changing. So a loss can be placed
    in the phase of the geometry (PHASE_EDGES: seconds since the target last switched) in every row, and the rows
    compared phase by phase: if the adaptive rows' extra losses are the narrowing's, they appear in the narrowing phase
    of the adaptive rows only. Exposure (seconds per phase) is accumulated beside the losses, and N_CONTROL moments per
    flight drawn at random give the baseline for the spread profile around a loss."""

    def __init__(self, m, rng):
        super().__init__()
        self.m, self.rng = m, rng
        self.sh, self.state, self.t_switch, self.dtick = None, None, 0.0, float(m.dt)
        self.shadow, self.key, self.exposure, self.ev2 = [], [], {}, []

    def step(self, sim, t, cmd, wp):
        m, rep = self.m, T.rep
        P = sim.P[sim.alive]
        target = 1.0
        if len(P):
            cen = P.mean(0)
            # the buildings Sim.step will use this tick: it re-queries on QUERY_EVERY ticks before using them
            cand = sim.city.near(P, m.r0_obs * m.S + rep.QUERY_MARGIN_M) if (
                sim.tick % rep.QUERY_EVERY == 0 or getattr(sim, "cand", None) is None) else sim.cand
            cmdv = m.stick_to_world(cmd, 0.0)
            miss = rep.aim_miss(m, sim.city, cand, cen, cmdv)
            target = rep.spread_target(m, sim.city, cand, cen, miss)
        street = target < 1.0
        sh_t = SHADOW_FACTOR if street else 1.0
        if self.sh is None:
            # seeded as the spread itself is: in an adaptive row from the spread the settle left (rescaled to
            # SHADOW_FACTOR), otherwise from the first target
            f = float(m.street_factor)
            self.sh = (1 - (1 - SHADOW_FACTOR) * (1 - sim.sigma) / (1 - f)) if (
                m.adaptive and sim.sigma is not None and f < 1) else sh_t
        if m.spread_filter > 0:
            self.sh += (1 - np.exp(-m.dt / m.spread_filter)) * (sh_t - self.sh)
        else:
            self.sh = sh_t
        if street != self.state:
            self.state, self.t_switch = street, float(t)
        key = phase_key(street, float(t) - self.t_switch)
        self.exposure[key] = self.exposure.get(key, 0.0) + self.dtick
        n_before = len(self.events)
        super().step(sim, t, cmd, wp)
        for e in self.events[n_before:]:
            e.update(sh=round(self.sh, 4), sh_before=round(self.sh_at(t - TRANS_S), 4), street=bool(street),
                     since=round(float(t) - self.t_switch, 2), phase=key)
        self.shadow.append(self.sh)
        self.key.append(key)

    def sh_at(self, t):
        if not self.t:
            return self.sh if self.sh is not None else 1.0
        i = int(np.searchsorted(self.t, t))
        return self.shadow[min(max(i, 0), len(self.shadow) - 1)]

    def windows(self):
        """Each loss's spread and shadow from WIN_BEFORE before it to WIN_AFTER after, every WIN_DT; and the same round
        N_CONTROL random moments of the flight (the baseline)."""
        if len(self.t) < 2:
            return
        t, s, h = np.array(self.t), np.array(self.sig), np.array(self.shadow)
        rel = np.arange(-WIN_BEFORE, WIN_AFTER + 1e-9, WIN_DT)

        def win(tc):
            i = np.clip(np.searchsorted(t, tc + rel), 0, len(t) - 1)
            return [round(float(x), 4) for x in s[i]], [round(float(x), 4) for x in h[i]]

        for e in self.events:
            e["win_sig"], e["win_sh"] = win(e["t"])
        ctl = []
        for tc in self.rng.uniform(t[0], t[-1], N_CONTROL):
            ws, wh = win(float(tc))
            i = int(np.clip(np.searchsorted(t, tc), 0, len(t) - 1))
            ctl.append(dict(t=round(float(tc), 2), phase=self.key[i], win_sig=ws, win_sh=wh))
        return ctl

    def shadow_change_share(self):
        if len(self.t) < 2:
            return 0.0
        t, h = np.array(self.t), np.array(self.shadow)
        j = np.clip(np.searchsorted(t, t - TRANS_S), 0, len(t) - 1)
        return float((np.abs(h - h[j]) >= TRANS_DS).mean())

    def shadow_phases(self):
        """Seconds of the flight in each phase of the shadow (sh_phase), and each loss's phase."""
        out = {k: 0.0 for k in SH_PHASES}
        if len(self.t) < 2:
            return out
        t, h = np.array(self.t), np.array(self.shadow)
        j = np.clip(np.searchsorted(t, t - TRANS_S), 0, len(t) - 1)
        for k in SH_PHASES:
            out[k] = 0.0
        for ph in (sh_phase(a, b) for a, b in zip(h, h[j])):
            out[ph] += self.dtick
        for e in self.events:
            e["sh_phase"] = sh_phase(e["sh"], e["sh_before"])
        return {k: round(v, 2) for k, v in out.items()}


SH_PHASES = ("narrowing", "widening", "narrow", "wide")


def sh_phase(sh, sh_before):
    """The shadow's phase at a moment: narrowing / widening (it moved TRANS_DS or more over the TRANS_S before), else
    narrow (held under the midpoint of SHADOW_FACTOR and 1) or wide."""
    d = sh - sh_before
    if d <= -TRANS_DS:
        return "narrowing"
    if d >= TRANS_DS:
        return "widening"
    return "narrow" if sh < (1 + SHADOW_FACTOR) / 2 else "wide"


SHADOW_FACTOR = 0.85
PHASE_EDGES = (0.0, 2.0, 4.0, 6.0, 9.0, 14.0, 1e9)       # seconds since the shadow target last switched
WIN_BEFORE, WIN_AFTER, WIN_DT = 15.0, 3.0, 0.5
N_CONTROL = 3


def phase_key(street, since):
    """'S<i>' in the streets (the target under 1) or 'B<i>' at a building or in the open, i the PHASE_EDGES bin of the
    seconds since the target last switched to that state."""
    i = int(np.searchsorted(PHASE_EDGES, since, side="right")) - 1
    return f"{'S' if street else 'B'}{min(max(i, 0), len(PHASE_EDGES) - 2)}"


def run_job(j):
    """One flight: swarmflow.run_job's, with each loss placed against the spread scale (and, with j['shadow'], against
    the adaptive spread's shadow and the phase of the geometry: ShadowLog)."""
    t0 = time.time()
    base = dict(job_id=T.job_id(j), cand=j["cand"], kind=j["kind"], seed=j["seed"], k=j["k"], d=j["d"])
    try:
        C = T.city(j["seed"])
        rng = np.random.default_rng(T.route_seed(j["kind"], j["seed"], j["k"]))
        fseed = int(rng.integers(1 << 30))
        start, wps, L, poly = T.route(C, j["kind"], j["k"], rng)
        over = dict(j["over"])
        m = T.model(over)
        P0 = T.start_positions(C, j["d"], fseed, over, start)
        shadow = bool(j.get("shadow"))
        log = ShadowLog(m, np.random.default_rng(fseed + 7)) if shadow else SpreadEventLog()
        sim, samples, done = T.fly(m, C, P0, wps, j["d"], fseed, hook=log)
        sig = np.array(log.sig) if log.sig else np.ones(1)
        extra = {}
        if shadow:
            extra = dict(controls=log.windows(), exposure={k: round(v, 2) for k, v in log.exposure.items()},
                         sh_exposure=log.shadow_phases(), shadow_change_frac=round(log.shadow_change_share(), 4))
        return dict(base, **T.score_flight(C, start, wps, poly, L, sim, samples, done, sim.n0),
                    sigma_mean=round(float(sig.mean()), 3), narrow_frac=round(float((sig < 0.95).mean()), 3),
                    trans_frac=round(log.transition_share(), 4), flight_s=round(len(log.t) * m.dt, 1),
                    loss_events=log.events, wall_s=round(time.time() - t0, 1), **extra)
    except Exception as e:
        return dict(base, error=repr(e), trace=traceback.format_exc()[-2000:], wall_s=round(time.time() - t0, 1))


# ============================================================================= statistics

def poisson_ci(k, conf=0.95):
    """Exact (Garwood) interval on a Poisson count."""
    from scipy.stats import chi2
    a = 1 - conf
    lo = 0.0 if k == 0 else chi2.ppf(a / 2, 2 * k) / 2
    hi = chi2.ppf(1 - a / 2, 2 * k + 2) / 2
    return lo, hi


def worse_p(k, k0, e=1.0, e0=1.0):
    """One-sided exact p that k events (exposure e) are no more frequent than k0 (exposure e0): the conditional
    binomial on k given k + k0."""
    from scipy.stats import binom
    n = k + k0
    if n == 0:
        return 1.0
    return float(binom.sf(k - 1, n, e / (e + e0)))


def code(label):
    return label.split()[0]


def row_events(g):
    """Loss events of one candidate's flights: flights that lost drones, distinct loss instants, drones, and the
    instants placed against the spread (narrowing, widening, steady)."""
    ev = [e for evs in g.loss_events for e in (evs or [])]
    place = {"narrowing": 0, "widening": 0, "steady": 0}
    for e in ev:
        ds = e["sig"] - e["sig_before"]
        place["narrowing" if ds <= -TRANS_DS else "widening" if ds >= TRANS_DS else "steady"] += 1
    flight_s = float(g.flight_s.sum())
    return dict(flights_lost=int((g.lost > 0).sum()), events=len(ev), drones=int(g.lost.sum()),
                events_street=sum(len(x or []) for x, k in zip(g.loss_events, g.kind) if k == "street"),
                trans_time=float((g.trans_frac * g.flight_s).sum() / max(flight_s, 1e-9)),
                sig_at_loss=[e["sig"] for e in ev], **{f"ev_{k}": v for k, v in place.items()})


def analyse(name):
    import pandas as pd
    df, cands, errs = T.read(name, OUT)
    tab = SS.table(name, root=OUT)
    by = {}
    for cid, g in df.groupby("cand"):
        lab = cands[cid]["label"]
        r = dict(label=lab, code=code(lab), d=float(g.d.iloc[0]), km=float(g.L.sum() / 1000),
                 km_street=float(g[g.kind == "street"].L.sum() / 1000), **row_events(g))
        r["ci"] = poisson_ci(r["events"])
        t = tab[tab.cand == cid].iloc[0].to_dict()
        for k in ("v_coh", "v_prog", "one", "lost_km", "contact_km", "open", "cross_wrapped", "cross_bld_pass",
                  "street_v_coh", "street_v_prog", "street_one", "street_lost_km", "street_contact_km", "sigma"):
            r[k] = float(t.get(k, math.nan))
        r["contacts"] = int(g.contacts.sum())
        by[r["code"]] = r
    rule = json.loads((OUT / "rule.json").read_text(encoding="utf-8"))["rule"]
    b = by[rule["base"]]
    verdicts = {}
    for c, r in by.items():
        if c == rule["base"]:
            continue
        p = worse_p(r["events"], b["events"], r["km"], b["km"])
        verdicts[c] = dict(p_worse=p, safe=bool(p >= rule["safe_p"] and r["contact_km"] <= rule["contact_ratio"] *
                                                b["contact_km"]),
                           street_gain=r["street_v_coh"] / b["street_v_coh"] - 1,
                           wrap_change=r["cross_wrapped"] - b["cross_wrapped"])
        v = verdicts[c]
        v["helps"] = bool(v["street_gain"] >= rule["street_gain"] and v["wrap_change"] >= -rule["wrap_margin"])
    ok = [c for c in rule["factors"] if c in verdicts and verdicts[c]["safe"] and verdicts[c]["helps"]]
    pick = max(ok, key=lambda c: by[c]["street_v_coh"]) if ok else None
    # adaptive against the same narrowed value held fixed (the transitions' own cost), and the dial ends
    pairs = [("A85", "F17"), ("A90", "F18"), ("A95", "F19"), ("L85", "F12"), ("H85", "F26"), ("L85", "F14"),
             ("L95", "F14"), ("H85", "F20")]
    pair_p = {f"{a}>{b_}": worse_p(by[a]["events"], by[b_]["events"], by[a]["km"], by[b_]["km"])
              for a, b_ in pairs if a in by and b_ in by}
    out = dict(rule=rule, rows=by, verdicts=verdicts, pick=pick, pair_p=pair_p, errors=len(errs),
               trans=dict(window_s=TRANS_S, ds=TRANS_DS))
    (OUT / name / "analysis.json").write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
    pd.set_option("display.width", 300)
    rows = []
    for c, r in by.items():
        v = verdicts.get(c, {})
        rows.append(dict(code=c, d=r["d"], km=round(r["km"]), events=r["events"], ci=f"{r['ci'][0]:.1f}-{r['ci'][1]:.1f}",
                         drones=r["drones"], narrow=r["ev_narrowing"], widen=r["ev_widening"], steady=r["ev_steady"],
                         trans_time=round(r["trans_time"], 3), v_coh=round(r["v_coh"], 2),
                         street=round(r["street_v_coh"], 2), wrapped=round(r["cross_wrapped"], 3),
                         contact_km=round(r["contact_km"], 3), sigma=round(r["sigma"], 3),
                         p_worse=round(v.get("p_worse", math.nan), 3), safe=v.get("safe"), helps=v.get("helps")))
    print(pd.DataFrame(rows).to_string(index=False))
    print("pairs (one-sided p, first loses more):", {k: round(v, 3) for k, v in pair_p.items()})
    print(f"pick: {pick or 'none: leave the adaptive spread off'}   ({len(errs)} errors)")
    return out


# ============================================================================= commands

def cmd_rule(a):
    path = OUT / "rule.json"
    if path.exists() and not a.force:
        raise SystemExit(f"[swarmspread] {path} exists (fixed before the run); --force to rewrite")
    OUT.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(rule=RULE, written=time.strftime("%Y-%m-%d %H:%M:%S")), indent=1),
                    encoding="utf-8")
    print(json.dumps(RULE, indent=1))


def cmd_run(a):
    spec = F.load_cands(a.cand)
    fl = dict(cross=a.cross, pass_=a.pass_, street=a.street)
    seeds = (T.TUNE_SEED,) + T.HELD_SEEDS
    cands, jobs = F.jobs_for(spec, seeds, fl, a.k0, False)
    # street routes first: they carry the losses, so a stopped run has the most of what it is for
    jobs.sort(key=lambda j: (j["kind"] != "street", j["k"], j["seed"]))
    if a.limit:
        jobs = jobs[:a.limit]
    if a.shadow:
        for j in jobs:
            j["shadow"] = True
    T.run(a.name, cands, jobs, a.jobs, f"S9 {a.name}", root=OUT, fn=run_job)
    analyse(a.name)


def cmd_analyse(a):
    analyse(a.name)


def cmd_bench_plan(a):
    SS.OUT = OUT
    SS.cmd_bench_plan(a)


def cmd_bench_gate(a):
    SS.OUT = OUT
    SS.cmd_bench_gate(a)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("rule")
    p.add_argument("--force", action="store_true")
    p = sub.add_parser("run")
    p.add_argument("--name", required=True)
    p.add_argument("--cand", required=True)
    p.add_argument("--cross", type=int, default=FLIGHTS["cross"])
    p.add_argument("--pass", dest="pass_", type=int, default=FLIGHTS["pass_"])
    p.add_argument("--street", type=int, default=FLIGHTS["street"])
    p.add_argument("--k0", type=int, default=K_RUN)
    p.add_argument("--limit", type=int, default=0, help="fly only the first N jobs (a smoke test)")
    p.add_argument("--shadow", action="store_true", help="log the adaptive spread's shadow and the phase (ShadowLog)")
    p.add_argument("--jobs", type=int, default=22)
    p = sub.add_parser("analyse")
    p.add_argument("--name", required=True)
    p = sub.add_parser("bench-plan")
    p.add_argument("--name", required=True)
    p.add_argument("--cand", required=True)
    p.add_argument("--cross", type=int, default=8)
    p.add_argument("--pass", dest="pass_", type=int, default=8)
    p.add_argument("--street", type=int, default=40)
    p.add_argument("--face", type=int, default=0)
    p.add_argument("--k0", type=int, default=K_BENCH)
    p.add_argument("--drones", type=int, default=T.N_DRONES)
    p.add_argument("--timeout", type=int, default=180)
    p = sub.add_parser("bench-gate")
    p.add_argument("--name", required=True)
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--jobs", type=int, default=16)
    a = ap.parse_args()
    dict(rule=cmd_rule, run=cmd_run, analyse=cmd_analyse, **{"bench-plan": cmd_bench_plan,
                                                            "bench-gate": cmd_bench_gate})[a.cmd](a)


if __name__ == "__main__":
    main()
