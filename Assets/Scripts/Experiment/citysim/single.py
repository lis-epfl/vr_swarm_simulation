"""Single-drone policies at a goal, as camera tracks for the exposure engine. A human single-drone pilot orbits and
scans; a perfectly placed one parks at the best spot. Together they bracket skill:

  orbit_in    round the occluder inside the ring, midway across the free annulus (4 m off the occluder's avoidance
              cylinder and off the ring), camera facing out at the ring buildings, at ORBIT_SPEED
  orbit_out   just outside the ring (its buildings' half-diagonal + 3 m), camera facing in, at ORBIT_SPEED
  scan        hovering in the annulus on the approach side, yawing at maxYawRate (a full turn in 4.8 s)
  static      the fixed position and heading (on the two orbits, every 22.5 deg; headings every 30 deg) that
              recognises the most walkers on average -- the best vantage for a target whose building is unknown
The orbit's phase is a parameter; callers average over a few.
"""
import sys

sys.dont_write_bytecode = True

import math  # noqa: E402

import numpy as np  # noqa: E402

from citysim import camera, exposure  # noqa: E402

ORBIT_SPEED = 3.0          # m/s: the single-drone pilots' median search speed in internal_3 was 3.2
STANDOFF = 4.0             # d_obs * scaleFactor: the drone is repelled within 4 m of an avoidance cylinder
OUTSIDE_MARGIN = 3.0


def _cam(pos, yaw):
    y = np.radians(yaw)
    return pos + camera.CAMERA_FORWARD_M * np.stack([np.sin(y), np.cos(y)], -1)


def _track(pos, yaw, T, dt, height, pitch):
    pos = np.asarray(pos, float).reshape(T, 1, 2)
    yaw = np.asarray(yaw, float).reshape(T, 1)
    return exposure.Track(pos=_cam(pos, yaw), yaw=yaw, h=np.full((T, 1), float(height)), valid=np.ones((T, 1), bool),
                          pitch=np.full(T, float(pitch)), body_yaw=yaw[:, 0].copy(), dt=dt, single=True)


def annulus_radius(R_c, c_g):
    """Mid-radius of the free annulus between the occluder's cylinder and the ring, or None if there is none."""
    lo, hi = R_c + STANDOFF, c_g - STANDOFF
    return 0.5 * (lo + hi) if hi >= lo else None


def orbit(centre, radius, T, dt, height, pitch, facing="out", phase_deg=0.0, speed=ORBIT_SPEED, clockwise=True):
    t = np.arange(T) * dt
    phi = phase_deg + (1 if clockwise else -1) * np.degrees(speed / radius) * t
    pos = np.asarray(centre, float) + radius * np.stack([np.sin(np.radians(phi)), np.cos(np.radians(phi))], -1)
    yaw = phi if facing == "out" else phi + 180.0
    return _track(pos, np.mod(yaw, 360.0), T, dt, height, pitch)


def scan(position, T, dt, height, pitch, yaw0=0.0, rate=camera.DRONE_MAX_YAW_RATE_DEG):
    t = np.arange(T) * dt
    pos = np.broadcast_to(np.asarray(position, float), (T, 2))
    return _track(pos, np.mod(yaw0 + rate * t, 360.0), T, dt, height, pitch)


def fixed(position, yaw, T, dt, height, pitch):
    return _track(np.broadcast_to(np.asarray(position, float), (T, 2)), np.full(T, float(yaw)), T, dt, height, pitch)


def policies(fp, goal, walkers, t, height, pitch, approach_deg, phases=(0.0, 120.0, 240.0), coarse=4, band=True):
    """{name: [Track, ...]} for one goal (dict with centre, R_c, c_g, r_g and ring half-diagonal R_ring). The static
    vantage is chosen at this height and pitch, or in plan view only (band=False) to serve every altitude."""
    c = np.asarray(goal["centre"], float)
    T, dt = len(t), float(t[1] - t[0]) if len(t) > 1 else 0.5
    out = {}
    r_in = annulus_radius(goal["R_c"], goal["c_g"])
    r_out = goal["r_g"] + goal["R_ring"] + OUTSIDE_MARGIN
    if r_in is not None:
        out["orbit_in"] = [orbit(c, r_in, T, dt, height, pitch, "out", ph) for ph in phases]
        a = math.radians(approach_deg)
        spot = c + r_in * np.array([math.sin(a), math.cos(a)])
        out["scan"] = [scan(spot, T, dt, height, pitch, yaw0=ph) for ph in phases]
    out["orbit_out"] = [orbit(c, r_out, T, dt, height, pitch, "in", ph) for ph in phases]
    # static best: score candidates on a coarse clock, keep the best
    tc = t[::coarse]
    pts = walkers.positions(tc)
    best, best_e = None, -1.0
    radii = [r for r in (r_in, r_out) if r is not None]
    for r in radii:
        for ang in np.arange(0.0, 360.0, 22.5):
            p = c + r * np.array([math.sin(math.radians(ang)), math.cos(math.radians(ang))])
            if fp.inside([p], margin=1.0)[0]:
                continue
            for yaw in np.arange(0.0, 360.0, 30.0):
                tr = fixed(p, yaw, len(tc), dt * coarse, height, pitch)
                vis, _ = exposure.recognised(fp, tr, pts, band=band)
                e = float(vis.any(axis=2).mean())
                if e > best_e:
                    best_e, best = e, (p, yaw)
    if best is not None:
        out["static"] = [fixed(best[0], best[1], T, dt, height, pitch)]
    return out
