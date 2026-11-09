"""The operator band: how pilots flew in internal_3, as bounds the simulator sweeps (not a fitted model -- two
participants, one of whom never touched the spread or pitch dials, cannot define a joint distribution).

    python -B -m citysim.operator_band calibrate internal_3     # -> experiment_results/internal_3/results/operator_internal_3.json

(Named operator_band, not operator: a module called operator.py in this folder would shadow the standard library's
when a citysim script is run directly.)

Per condition and phase (transit / search, analyse.goal_legs with the 15 m zone), from the recorded files:
  altitude    the alive-drone centroid's height (_drones.csv gtY)
  pitch       the gimbal (_head.csv gimbalPitch)
  spread      the commanded d_ref (_head.csv inSpread, as analyse.spread_signal fills it)
  speed       the centroid's horizontal speed
  head        the head's yaw relative to straight ahead: headYaw - bodyYaw - 90 (the display frame puts the screen
              of a drone facing the body yaw at heading bodyYaw + 90, where every single-drone pilot looks)
reported as the 10th / 50th / 90th percentiles, plus the head's histogram in 10-degree bins (E_head's weights).
grid(phase) is the sweep: three altitudes, three pitches (the dial centre -15 always among them), three spreads.
"""
if __name__ == "__main__" and __package__ in (None, ""):
    import os.path as _p
    import sys as _s
    _s.path[0] = _p.dirname(_p.dirname(_p.abspath(__file__)))

import sys

sys.dont_write_bytecode = True

import argparse  # noqa: E402
import json  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

from citysim import DEFAULT_ROOT, REPO, camera, use_bench_modules  # noqa: E402

HEAD_BINS = np.arange(-180.0, 181.0, 10.0)


def results_dir(test):
    return REPO / "experiment_results" / test / "results"


def band_path(test="internal_3"):
    return results_dir(test) / f"operator_{test}.json"


SPREAD_MARKS = (0.4, 1.0, 1.6)     # the spread stick's ends and centre: the levels the tiers fly


def spread_share(band, condition="Swarm", phase="transit"):
    """{"0.4": share, "1.0": share, "1.6": share} of the phase's time nearest each spread mark."""
    return band["conditions"][condition][phase].get("spread_share") or {}


def calibrate(test, root=DEFAULT_ROOT, zone=15.0):
    use_bench_modules()
    import analyse
    _, _, runs = analyse.load_test(str(root), test)
    pools = {}
    for r in runs:
        legs = analyse.goal_legs(r, zone)
        h = r.head
        t = h.t.values
        search = np.zeros(len(t), bool)
        for L in legs:
            search |= (t >= L["enter"]) & (t < L["exit"])
        inside = (t >= legs[0]["t0"]) & (t < legs[-1]["exit"]) if legs else np.ones(len(t), bool)
        alt = np.interp(t, r.ct, r.cy)
        vx = np.gradient(r.cx, r.ct)
        vz = np.gradient(r.cz, r.ct)
        speed = np.interp(t, r.ct, np.hypot(vx, vz))
        sig = dict(altitude=alt, pitch=h.gimbalPitch.values.astype(float) if "gimbalPitch" in h else np.full(len(t), np.nan),
                   spread=analyse.spread_signal(h, r.shape), speed=speed,
                   head=((h.headYaw - h.bodyYaw - 90.0 + 180.0) % 360.0 - 180.0).values)
        for phase, mask in (("search", search), ("transit", inside & ~search)):
            key = (r.condition, phase)
            for k, v in sig.items():
                pools.setdefault(key, {}).setdefault(k, []).append(v[mask])
    out = dict(test=test, zone_m=zone, runs=[r.stem for r in runs], conditions={})
    for (cond, phase), d in sorted(pools.items()):
        entry = {}
        for k, parts in d.items():
            v = np.concatenate(parts)
            v = v[np.isfinite(v)]
            entry[k] = dict(p10=round(float(np.percentile(v, 10)), 2), p50=round(float(np.percentile(v, 50)), 2),
                            p90=round(float(np.percentile(v, 90)), 2), n=int(len(v))) if len(v) else None
            if k == "head" and len(v):
                hist, _ = np.histogram(v, HEAD_BINS)
                entry["head_hist"] = dict(edges=HEAD_BINS.tolist(), weights=(hist / hist.sum()).round(5).tolist())
            if k == "spread" and len(v):
                # share of the phase's time nearest each of the stick's three marks: what weights a per-spread
                # result (Tier C's losses, flown at 0.4 and 1.0) into a per-run figure the way pilots flew
                near = np.argmin(np.abs(v[:, None] - np.array(SPREAD_MARKS)[None]), axis=1)
                entry["spread_share"] = {f"{m:.1f}": round(float((near == i).mean()), 4)
                                         for i, m in enumerate(SPREAD_MARKS)}
        out["conditions"].setdefault(cond, {})[phase] = entry
    return out


def load(test="internal_3"):
    p = band_path(test)
    if not p.exists():
        raise FileNotFoundError(f"{p} does not exist: run `python -B -m citysim.operator_band calibrate {test}`")
    return json.loads(p.read_text(encoding="utf-8"))


def head_weights(band, condition="Swarm", phase="search"):
    """(head yaws deg, weights) for E_head: the bin centres of the recorded head-yaw histogram."""
    hh = band["conditions"][condition][phase]["head_hist"]
    e = np.array(hh["edges"])
    return 0.5 * (e[:-1] + e[1:]), np.array(hh["weights"])


def grid(band, phase="search", condition="Swarm"):
    """The explicit sweep: altitude, pitch and spread levels from the band (pitch always includes the dial
    centre, spread the stick's ends and middle)."""
    c = band["conditions"][condition][phase]
    alt = sorted({round(max(c["altitude"][k], 2.0), 1) for k in ("p10", "p50", "p90")})
    pit = sorted({camera.DIAL_CENTRE_PITCH_DEG, round(c["pitch"]["p10"], 0), round(c["pitch"]["p90"], 0)})
    spr = [0.4, 1.0, 1.6]
    return dict(altitude=alt, pitch=pit, spread=spr)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("calibrate")
    p.add_argument("test", nargs="?", default="internal_3")
    p.add_argument("--root", default=str(DEFAULT_ROOT))
    a = ap.parse_args()
    band = calibrate(a.test, a.root)
    path = band_path(a.test)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(band, indent=1) + "\n", encoding="utf-8", newline="\n")
    for cond, phases in band["conditions"].items():
        for phase, e in phases.items():
            print(f"{cond:11s} {phase:7s} " + "  ".join(f"{k} {v['p10']}/{v['p50']}/{v['p90']}"
                                                         for k, v in e.items() if isinstance(v, dict) and "p50" in v))
    print("search grid (Swarm):", grid(band))
    print(f"written {path}")


if __name__ == "__main__":
    main()
