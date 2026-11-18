"""Generated pictures for the participant instruction video.

Every function here takes a vidkit.Ctx and returns a vidkit.Frame; script.yaml names them under
`visual:` (and `fallback:` for an inset). They are schematics: each one stands in for, or sits
beside, a Unity recording, and the city maps are drawn from the real layout files.

World coordinates are Unity's (x, z) in metres; yaw is Unity's (0 = +z "north", 90 = +x "east").
"""
from __future__ import annotations

import functools
import json
import math
import os

import cv2
import numpy as np
from matplotlib.path import Path as MplPath

import vidkit as vk
from vidkit import (ACCENT, ASPHALT, BG, BUILDING, BUILDING_EDGE, CYAN, DRONE_HUES, FAINT, FOOTPATH, GREEN,
                    GROUND, HIDDEN, INK, MUTED, PANEL, RED, View, clamp01, ease, lerp, smoothstep, yaw_vec)

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
# Where the recordings are. Another video that reuses these pictures (single_drone_instructions/visuals.py)
# points it at its own footage/.
FOOTAGE_DIR = os.path.join(HERE, "footage")
LAYOUTS = os.path.join(REPO, "Assets", "Scripts", "Experiment", "layouts")
PRACTICE_FILE = "PracticeBlock__bbB_p40_s18_patch150_j0.15_sight487_seed2_cut_b28.json"
TRIAL_FILE = "BlockShaped__bbB_p40_s18_patch150_j0.15_sight487_seed1.json"

FOV_H = 74.4          # FPV horizontal field of view, degrees (FpvCameraFov.cs / citysim/camera.py)
N_DRONES = 10
RING_R = 15.0         # hover ring radius at d_ref 2.0 (~31 m across, CLAUDE.md S6)
RING_PHASE = 9.0      # no drone exactly on the travel axis, so a split always knows its side
HAT = (255, 214, 64)
EDGE_GREEN = (60, 230, 110)   # DroneFrustumGizmos: on the hull (its feed is on a screen)
INSIDE_RED = (240, 70, 70)    # DroneFrustumGizmos: inside the swarm (feed hidden)
UP = 90.0                     # maps are drawn east-up, so the start (west of the city) is below it


# =============================================================================================
# Swarm helpers
# =============================================================================================
def ring_offsets(r=RING_R, n=N_DRONES, phase=RING_PHASE):
    a = np.radians(phase + 360.0 * np.arange(n) / n)
    return np.c_[r * np.sin(a), r * np.cos(a)]


def hull_mask(P):
    P = np.asarray(P, np.float32)
    m = np.zeros(len(P), bool)
    if len(P) < 3:
        m[:] = True
        return m
    idx = cv2.convexHull(P, returnPoints=False, clockwise=False).ravel()
    m[idx] = True
    return m


def headings(P, mask=None):
    """Outward bisector at each hull vertex (the convex-hull heading rule); interior drones face
    away from the centroid."""
    P = np.asarray(P, float)
    c = P.mean(axis=0)
    yaw = np.array([vk.vec_yaw(p - c) for p in P])
    if len(P) < 3:
        return yaw
    idx = cv2.convexHull(P.astype(np.float32), returnPoints=False, clockwise=False).ravel()
    n = len(idx)
    for j, i in enumerate(idx):
        a, b = P[idx[j - 1]], P[idx[(j + 1) % n]]
        v = P[i]
        e1, e2 = a - v, b - v
        e1 /= np.linalg.norm(e1) + 1e-9
        e2 /= np.linalg.norm(e2) + 1e-9
        bis = -(e1 + e2)
        if np.linalg.norm(bis) < 1e-3:
            bis = v - c
        if np.dot(bis, v - c) < 0:
            bis = -bis
        yaw[i] = vk.vec_yaw(bis)
    return yaw


def boundary_alpha(pos_fn, t, window=0.5, samples=6):
    """How much of the last `window` seconds each drone spent on the hull: a screen fades
    rather than blinks."""
    acc = np.zeros(N_DRONES)
    for j in range(samples):
        acc += hull_mask(pos_fn(max(0.0, t - window * j / (samples - 1))))
    return acc / samples


def draw_drone(fr, view, p, yaw, color, alpha=1.0, cone=0.0, cone_alpha=0.16, r=9.0):
    sp = view(p)
    ang = view.screen_angle(yaw)
    if cone > 0:
        cc = vk.mix(INSIDE_RED, EDGE_GREEN, alpha)
        fr.wedge(sp, cone * view.scale, ang - FOV_H / 2, ang + FOV_H / 2, cc, alpha=cone_alpha * 1.4)
    col = vk.mix(INSIDE_RED, EDGE_GREEN, alpha)
    a = math.radians(ang)
    u = np.array([math.cos(a), math.sin(a)])
    n = np.array([-u[1], u[0]])
    tip = sp + u * r * 2.4
    fr.poly([tip, sp + n * r * 0.9 + u * r * 0.6, sp - n * r * 0.9 + u * r * 0.6], col)
    fr.circle(sp, r + 2.5, (10, 12, 16))
    fr.circle(sp, r, (255, 255, 255))


def draw_pilot(fr, c, s=1.0, forward=True):
    """A head and shoulders seen from above, facing up the screen."""
    x, y = c
    fr.poly(vk.rounded_rect(x - 46 * s, y - 6 * s, 92 * s, 34 * s, 16 * s), (70, 78, 96))
    fr.circle((x, y), 24 * s, (210, 214, 224))
    fr.poly([(x, y - 40 * s), (x - 9 * s, y - 22 * s), (x + 9 * s, y - 22 * s)], (210, 214, 224))
    if forward:
        fr.arrow((x, y - 52 * s), (x, y - 110 * s), MUTED, 3, 14)
        fr.text("forward", (x, y - 150 * s), 22, MUTED, anchor="ma")


def draw_screen_ring(fr, c, r, items, title=True, width_deg=27.0):
    """The ring of feed screens round the pilot, from above. items: (rel_yaw_deg, colour, alpha)
    where rel_yaw is the drone's heading relative to the pilot's forward."""
    fr.circle(c, r, FAINT, width=1.5, alpha=0.6)
    for rel, col, a in items:
        if a <= 0.01:
            continue
        ang = rel - 90.0
        fr.arc(c, r, ang - width_deg / 2, ang + width_deg / 2, col, width=16, alpha=a)
    draw_pilot(fr, c, 1.0)
    if title:
        fr.text("Your screens, seen from above", (c[0], c[1] + r + 46), 30, MUTED, "semibold", anchor="ma")


def hue(i):
    return DRONE_HUES[i % len(DRONE_HUES)]


# =============================================================================================
# Cities (from the layout files)
# =============================================================================================
class City:
    def __init__(self, fname):
        with open(os.path.join(LAYOUTS, fname), encoding="utf-8") as f:
            d = json.load(f)
        self.layout = d["layout"]
        self.buildings = {}
        for b in d["boxes"]:
            if b.get("type") == "ConvexPrism":
                bid = int(b["name"].rsplit("B", 1)[-1])
                self.buildings[bid] = np.array(b["poly_xz"], float).reshape(-1, 2)
        self.kerbs = {blk["id"]: np.array(blk["kerb_xz"], float).reshape(-1, 2) for blk in d["blocks"]}
        self.interior = {blk["id"]: blk["interior"] for blk in d["blocks"]}
        self.streets = [(np.array(s["polyline"], float), s["width"], s["blocks"]) for s in d["streets"]]
        self.neigh = {}
        for _, _, bl in self.streets:
            for a in bl:
                for b in bl:
                    if a != b and a >= 0 and b >= 0:
                        self.neigh.setdefault(a, set()).add(b)
        segs = []
        for poly in self.buildings.values():
            segs.append(np.c_[poly, np.roll(poly, -1, axis=0)])
        self.segs = np.vstack(segs)
        sp = self.layout["spawn"]
        self.spawn = np.array([sp[0], sp[2]])
        ex = self.layout["extent"]
        self.extent = ex  # [xmin, xmax, zmin, zmax]
        self.paths = {bid: MplPath(p) for bid, p in self.buildings.items()}

    def centre(self, bid):
        return self.buildings[bid].mean(axis=0)

    def radius(self, bid):
        p = self.buildings[bid]
        return float(np.max(np.hypot(*(p - p.mean(axis=0)).T)))

    def visible(self, o, yaw, fov=FOV_H, rng=120.0, n=56):
        a = np.radians(yaw + np.linspace(-fov / 2, fov / 2, n))
        D = np.c_[np.sin(a), np.cos(a)]
        A, B = self.segs[:, :2], self.segs[:, 2:]
        E = B - A
        AO = A - np.asarray(o, float)
        den = D[:, 0, None] * E[None, :, 1] - D[:, 1, None] * E[None, :, 0]
        with np.errstate(divide="ignore", invalid="ignore"):
            t = (AO[None, :, 0] * E[None, :, 1] - AO[None, :, 1] * E[None, :, 0]) / den
            s = (AO[None, :, 0] * D[:, 1, None] - AO[None, :, 1] * D[:, 0, None]) / den
        ok = (np.abs(den) > 1e-9) & (t > 0.05) & (s >= 0) & (s <= 1)
        t = np.where(ok, t, np.inf).min(axis=1)
        t = np.minimum(t, rng)
        return np.vstack([[o], np.asarray(o) + D * t[:, None]])

    def push_out(self, P, margin=3.0):
        """Move points that are inside (or within margin of) a building out to the margin."""
        P = np.array(P, float)
        for i, p in enumerate(P):
            for bid, poly in self.buildings.items():
                c = poly.mean(axis=0)
                if np.hypot(*(p - c)) > self.radius(bid) + margin:
                    continue
                d, q, nrm = _poly_signed_distance(poly, p)
                if d < margin:
                    P[i] = q + nrm * margin
                    p = P[i]
        return P


def _poly_signed_distance(poly, p):
    """Signed distance from p to a convex polygon (negative inside), nearest boundary point and
    the outward normal there."""
    best = (np.inf, None, None)
    inside = MplPath(poly).contains_point(p)
    for a, b in zip(poly, np.roll(poly, -1, axis=0)):
        e = b - a
        L2 = e @ e
        t = clamp01(((p - a) @ e) / L2) if L2 > 0 else 0
        q = a + e * t
        dist = np.hypot(*(p - q))
        if dist < best[0]:
            best = (dist, q, e)
    dist, q, e = best
    c = poly.mean(axis=0)
    nrm = np.array([e[1], -e[0]]) / (np.hypot(*e) + 1e-9)
    if np.dot(nrm, q - c) < 0:
        nrm = -nrm
    return (-dist if inside else dist), q, nrm


@functools.lru_cache(maxsize=None)
def practice_city():
    return City(PRACTICE_FILE)


@functools.lru_cache(maxsize=None)
def trial_city():
    return City(TRIAL_FILE)


def draw_city(fr, city, view, goal=None, ring=(), goal_alpha=1.0, road_in=True):
    s = view.scale
    if road_in:
        ex = city.extent
        fr.polyline(view([city.spawn + [-30, 0], [ex[0] + 10, 0]]), ASPHALT, width=15 * s)
    for pl, w, _ in city.streets:
        fr.polyline(view(pl), ASPHALT, width=w * s)
    for bid, kerb in city.kerbs.items():
        fr.poly(view(kerb), FOOTPATH)
    for bid, poly in city.buildings.items():
        col = BUILDING
        if bid in ring:
            col = vk.mix(BUILDING, (150, 160, 190), goal_alpha)
        fr.poly(view(poly), col, outline=BUILDING_EDGE, width=1.2)
    if goal is not None and goal_alpha > 0.01:
        fr.poly(view(city.buildings[goal]), ACCENT, alpha=0.30 * goal_alpha, outline=None)
        fr.dashed(view(city.buildings[goal]), ACCENT, width=3, closed=True, alpha=goal_alpha)


def ground(fr):
    fr.img[:] = GROUND


# =============================================================================================
# Pedestrians on the goal's ring (WalkerPatrol's polygon loop, 16 per building, ~1.3 m out)
# =============================================================================================
def walk_loop(poly, gap=1.3, n_arc=6):
    area = 0.5 * np.sum(poly[:, 0] * np.roll(poly[:, 1], -1) - np.roll(poly[:, 0], -1) * poly[:, 1])
    if area < 0:
        poly = poly[::-1]
    pts = []
    m = len(poly)
    for i in range(m):
        v = poly[i]
        e_in = v - poly[i - 1]
        e_out = poly[(i + 1) % m] - v
        n_in = np.array([e_in[1], -e_in[0]]) / np.hypot(*e_in)
        n_out = np.array([e_out[1], -e_out[0]]) / np.hypot(*e_out)
        a0, a1 = math.atan2(n_in[1], n_in[0]), math.atan2(n_out[1], n_out[0])
        while a1 > a0:
            a1 -= 2 * math.pi
        for a in np.linspace(a0, a1, n_arc):
            pts.append(v + gap * np.array([math.cos(a), math.sin(a)]))
    pts = np.array(pts)
    seg = np.diff(np.vstack([pts, pts[:1]]), axis=0)
    cum = np.concatenate([[0], np.cumsum(np.hypot(seg[:, 0], seg[:, 1]))])
    return np.vstack([pts, pts[:1]]), cum


class Peds:
    PER_BUILDING = 16

    def __init__(self, city, ring, hat_building, hat_index=3, seed=4):
        rng = np.random.default_rng(seed)
        self.loops = []
        self.hat = None
        for bid in ring:
            loop, cum = walk_loop(city.buildings[bid])
            L = cum[-1]
            for j in range(self.PER_BUILDING):
                ph = (j + rng.uniform(-0.3, 0.3)) / self.PER_BUILDING * L
                sp = rng.uniform(1.15, 1.45) * (1 if j % 2 == 0 else -1)
                if bid == hat_building and j == hat_index:
                    self.hat = len(self.loops)
                self.loops.append((loop, cum, ph, sp))

    def positions(self, t):
        out = np.empty((len(self.loops), 2))
        for i, (loop, cum, ph, sp) in enumerate(self.loops):
            s = (ph + sp * t) % cum[-1]
            j = int(np.searchsorted(cum, s, side="right") - 1)
            j = min(j, len(loop) - 2)
            f = (s - cum[j]) / max(cum[j + 1] - cum[j], 1e-9)
            out[i] = loop[j] + (loop[j + 1] - loop[j]) * f
        return out


def practice_goal():
    c = practice_city()
    pr = c.layout.get("practice", {})
    goal = pr.get("goal_block", 28)
    ring = pr.get("ring_blocks", sorted(c.neigh.get(goal, [])))
    return c, goal, ring


@functools.lru_cache(maxsize=None)
def practice_peds():
    c, goal, ring = practice_goal()
    gc = c.centre(goal)
    # The hat sits on the ring building furthest from the way in (east of the goal), so in the
    # search schematic it is the swarm's surround, not its approach, that finds it.
    hb = max(ring, key=lambda b: c.centre(b)[0] - gc[0])
    return Peds(c, ring, hb)


def draw_peds(fr, view, P, hat_idx=None, lit=None, hat_reveal=0.0, t=0.0):
    r = max(2.2, 0.55 * view.scale)
    for i, p in enumerate(P):
        on = True if lit is None else lit[i]
        fr.circle(view(p), r, (240, 240, 245) if on else (150, 154, 162))
    if hat_idx is not None and hat_reveal > 0:
        sp = view(P[hat_idx])
        pulse = 1 + 0.25 * math.sin(t * 5)
        fr.circle(sp, r * 1.25, HAT, alpha=hat_reveal)
        fr.circle(sp, (r + 10) * pulse, HAT, width=3, alpha=hat_reveal)


# =============================================================================================
# Hats (placeholders for close-up renders of the real ones)
# =============================================================================================
def draw_hat(fr, kind, c, s, col=(196, 198, 204), edge=(40, 42, 48)):
    x, y = c
    if kind == "cap":
        a = np.radians(np.linspace(180, 360, 30))
        crown = np.c_[x - 0.12 * s + 0.42 * s * np.cos(a), y + 0.5 * s * np.sin(a) * 0.95]
        brim = np.array([[x + 0.22 * s, y - 0.02 * s], [x + 0.78 * s, y + 0.06 * s], [x + 0.74 * s, y + 0.13 * s],
                         [x + 0.1 * s, y + 0.07 * s]])
        fr.poly(brim, col, outline=edge, width=3)
        fr.poly(np.vstack([crown, [[x + 0.3 * s, y + 0.04 * s], [x - 0.54 * s, y + 0.04 * s]]]), col, outline=edge, width=3)
        fr.circle((x - 0.12 * s, y - 0.47 * s), 0.05 * s, edge)
    elif kind == "bucket":
        crown = np.array([[x - 0.3 * s, y - 0.42 * s], [x + 0.3 * s, y - 0.42 * s], [x + 0.38 * s, y],
                          [x - 0.38 * s, y]])
        brim = np.array([[x - 0.38 * s, y - 0.02 * s], [x + 0.38 * s, y - 0.02 * s], [x + 0.62 * s, y + 0.2 * s],
                         [x - 0.62 * s, y + 0.2 * s]])
        fr.poly(brim, col, outline=edge, width=3)
        fr.poly(crown, col, outline=edge, width=3)
        fr.line((x - 0.36 * s, y - 0.1 * s), (x + 0.36 * s, y - 0.1 * s), edge, 3)
    else:  # cowboy
        a = np.linspace(0, 1, 30)
        bx = x + (a - 0.5) * 1.5 * s
        by = y + 0.1 * s - 0.3 * s * ((a - 0.5) * 2) ** 4 + 0.06 * s
        top = np.c_[bx, by - 0.08 * s]
        bot = np.c_[bx[::-1], by[::-1] + 0.06 * s]
        fr.poly(np.vstack([top, bot]), col, outline=edge, width=3)
        crown = np.array([[x - 0.36 * s, y + 0.06 * s], [x - 0.32 * s, y - 0.44 * s], [x - 0.12 * s, y - 0.5 * s],
                          [x, y - 0.42 * s], [x + 0.12 * s, y - 0.5 * s], [x + 0.32 * s, y - 0.44 * s],
                          [x + 0.36 * s, y + 0.06 * s]])
        fr.poly(crown, col, outline=edge, width=3)
        fr.line((x - 0.35 * s, y - 0.04 * s), (x + 0.35 * s, y - 0.04 * s), edge, 4)


HAT_NAMES = {"cap": "Cap", "bucket": "Bucket hat", "cowboy": "Cowboy hat"}


# =============================================================================================
# The controller (DJI RC Pro, drawn; not a photo)
# =============================================================================================
def rc_front(fr, x, y, w, hl=(), stick_l=(0.0, 0.0), stick_r=(0.0, 0.0), dial_l=0.0, t=0.0, c2_ghost=False,
             labels=True):
    """Front view. hl: any of 'left', 'right', 'dial', 'c2'. Sticks are (x, y) in -1..1, y up."""
    h = 0.5 * w
    body = (44, 48, 56)
    edge = (100, 106, 118)
    pulse = 0.65 + 0.35 * math.sin(t * 5)
    # antennas, folded up behind the top edge
    for ax in (0.31, 0.62):
        fr.rect(x + ax * w, y - 0.11 * w, 0.075 * w, 0.14 * w, (36, 39, 46), radius=0.02 * w, outline=edge, width=1.5)
    # top dials
    for dx, key in ((0.13, "dial"), (0.78, None)):
        col = (70, 74, 84)
        fr.rect(x + dx * w, y - 0.022 * w, 0.09 * w, 0.04 * w, col, radius=0.012 * w, outline=edge, width=1.5)
        for k in range(7):
            rx = x + (dx + 0.012 + k * 0.011) * w + (dial_l * 0.006 * w if key else 0)
            fr.line((rx, y - 0.016 * w), (rx, y + 0.008 * w), (110, 116, 128), 1.5)
        if key and key in hl:
            fr.rect(x + (dx - 0.012) * w, y - 0.034 * w, 0.114 * w, 0.064 * w, None, outline=ACCENT, width=4,
                    radius=0.02 * w, alpha=pulse)
    fr.rect(x, y, w, h, body, radius=0.11 * w, outline=edge, width=2)
    # grips
    for gx in (0.0, 0.79):
        fr.rect(x + gx * w + 0.02 * w, y + 0.24 * w, 0.17 * w, 0.24 * w, (38, 41, 48), radius=0.07 * w)
    # screen
    sx, sy, sw, sh = x + 0.215 * w, y + 0.07 * w, 0.57 * w, 0.33 * w
    fr.rect(sx - 0.01 * w, sy - 0.01 * w, sw + 0.02 * w, sh + 0.02 * w, (20, 22, 26), radius=0.02 * w)
    fr.rect(sx, sy, sw, sh, (14, 16, 20), radius=0.01 * w)
    fr.poly([(sx + sw * 0.55, sy), (sx + sw * 0.75, sy), (sx + sw * 0.45, sy + sh), (sx + sw * 0.25, sy + sh)],
            (255, 255, 255), alpha=0.035)
    # buttons under the sticks
    for bx in (0.07, 0.15, 0.85, 0.93):
        fr.circle((x + bx * w, y + 0.32 * w), 0.016 * w, (66, 70, 80), )
    # sticks
    for side, (cx, st) in (("left", (0.105, stick_l)), ("right", (0.895, stick_r))):
        px, py = x + cx * w, y + 0.17 * w
        fr.circle((px, py), 0.062 * w, (30, 32, 38))
        fr.circle((px, py), 0.062 * w, edge, width=2)
        kx, ky = px + st[0] * 0.03 * w, py - st[1] * 0.03 * w
        fr.circle((kx, ky), 0.032 * w, (88, 94, 106))
        fr.circle((kx, ky), 0.02 * w, (120, 126, 138))
        if side in hl:
            fr.circle((px, py), 0.08 * w, ACCENT, width=4, alpha=pulse)
    if c2_ghost or "c2" in hl:
        gx, gy = x + 0.935 * w, y + 0.012 * w
        a = pulse if "c2" in hl else 0.6
        fr.rect(gx - 0.045 * w, gy - 0.018 * w, 0.09 * w, 0.036 * w, ACCENT if "c2" in hl else None,
                alpha=0.35 * a, radius=0.015 * w)
        fr.dashed(vk.rounded_rect(gx - 0.045 * w, gy - 0.018 * w, 0.09 * w, 0.036 * w, 0.015 * w),
                  ACCENT if "c2" in hl else MUTED, width=3, dash=8, gap=6, closed=True, alpha=a)
        fr.text("C2", (gx, gy - 0.016 * w), 0.028 * w, ACCENT if "c2" in hl else MUTED, "bold", anchor="ma", alpha=a)
        if labels:
            fr.text("(on the back)", (gx, gy + 0.024 * w), 0.02 * w, ACCENT if "c2" in hl else MUTED, anchor="ma",
                    alpha=a)
    if labels:
        lab = MUTED
        fr.text("left stick", (x + 0.105 * w, y + 0.255 * w), 0.024 * w, lab, anchor="ma")
        fr.text("right stick", (x + 0.895 * w, y + 0.255 * w), 0.024 * w, lab, anchor="ma")


# The photo (assets/rc_pro_front.jpg, 1280x1280, white background) and where its parts are, in its pixels.
RC_PHOTO = os.path.join(HERE, "assets", "rc_pro_front.jpg")
RC_CROP = (150, 250, 1130, 1010)            # x0, y0, x1, y1: the controller with a margin
RC_PARTS = {
    "left": (350, 478),                     # stick centres
    "right": (928, 478),
    "stick_r": 80,                          # stick well radius
    "dial": (300, 342),                     # the gimbal dial, on the top edge, left shoulder
    "c2": (845, 452),                       # C2 is on the back, behind the right stick, a little inboard
}


@functools.lru_cache(maxsize=None)
def _rc_photo():
    im = cv2.imread(RC_PHOTO)
    if im is None:
        return None
    x0, y0, x1, y1 = RC_CROP
    return cv2.cvtColor(im[y0:y1, x0:x1], cv2.COLOR_BGR2RGB)


def rc_photo(fr, x, y, w, hl=(), stick_l=(0.0, 0.0), stick_r=(0.0, 0.0), dial_l=0.0, t=0.0, c2=False,
             labels=True):
    """The controller photo on a white card with the control in question marked. hl: any of 'left',
    'right', 'dial', 'c2'. Sticks are (x, y) in -1..1, y up; the photo can't move, so an amber knob
    shows where the stick is pushed."""
    img = _rc_photo()
    if img is None:
        return rc_front(fr, x, y, w, hl, stick_l, stick_r, dial_l, t, c2_ghost=c2, labels=labels)
    x0, y0, x1, y1 = RC_CROP
    k = w / (x1 - x0)
    h = (y1 - y0) * k
    fr.rect(x - 10, y - 10, w + 20, h + 20, (255, 255, 255), radius=28)
    fr.image(img, x, y, w, h, mode="contain")

    def P(px, py):
        return np.array([x + (px - x0) * k, y + (py - y0) * k])
    pulse = 0.65 + 0.35 * math.sin(t * 5)
    R = RC_PARTS["stick_r"] * k
    for side, st in (("left", stick_l), ("right", stick_r)):
        if side not in hl:
            continue
        c = P(*RC_PARTS[side])
        fr.circle(c, R * 1.08, ACCENT, width=5, alpha=pulse)
        axis = np.array([1.0, 0.0]) if abs(st[0]) > abs(st[1]) else np.array([0.0, -1.0])
        if abs(st[0]) < 0.02 and abs(st[1]) < 0.02:
            axis = np.array([0.0, -1.0])
        fr.arrow(c - axis * R * 1.55, c + axis * R * 1.55, ACCENT, 5, 18 * k / 0.7, double=True, alpha=0.8)
        knob = c + np.array([st[0], -st[1]]) * R * 0.75
        fr.circle(knob, R * 0.32, ACCENT, alpha=0.85)
        fr.circle(knob, R * 0.32, (255, 255, 255), width=2.5)
    if "dial" in hl:
        c = P(*RC_PARTS["dial"])
        fr.rect(c[0] - 70 * k, c[1] - 22 * k, 140 * k, 44 * k, None, outline=ACCENT, width=5, radius=14 * k,
                alpha=pulse)
        rot = dial_l * 40
        fr.arc(c - [0, 70 * k], 60 * k, 200 + rot, 340 + rot, ACCENT, width=5)
        if labels:
            fr.text("dial, on the top edge", (c[0] + 120 * k, c[1] - 120 * k), 30 * k / 0.7, ACCENT, "semibold")
    if c2 or "c2" in hl:
        c = P(*RC_PARTS["c2"])
        a = pulse if "c2" in hl else 0.7
        col = ACCENT if "c2" in hl else MUTED
        box = vk.rounded_rect(c[0] - 46 * k, c[1] - 22 * k, 92 * k, 44 * k, 16 * k)
        fr.poly(box, ACCENT, alpha=0.35 * a)
        fr.dashed(box, col, width=4, dash=10, gap=7, closed=True, alpha=a)
        fr.text("C2", (c[0], c[1] - 18 * k), 30 * k / 0.7, col, "bold", anchor="ma", alpha=a)
        if labels:
            fr.text("on the back", (c[0], c[1] + 28 * k), 24 * k / 0.7, col, "semibold", anchor="ma", alpha=a)
    return h


def stick_schedule(u, segs):
    """segs: [(u0, u1, value)] -> eased stick position at u (0 outside every segment)."""
    for (a, b, v) in segs:
        if a <= u <= b:
            ramp = min(1.0, (u - a) / 0.06, (b - u) / 0.06)
            return v * ease(ramp)
    return 0.0


def integrate(fn, u_end, dur, n=400):
    """∫ fn(u) dt from 0 to u_end (seconds), fn returns a velocity."""
    if u_end <= 0:
        return 0.0
    us = np.linspace(0, u_end, n)
    v = np.array([fn(u) for u in us])
    return float(np.trapz(v, us) * dur)


# =============================================================================================
# Visuals
# =============================================================================================
def v_screens_around_you(ctx):
    """The pilot in the dim cockpit room, ringed by feed screens (schematic of the headset view)."""
    if "base" not in ctx.state:
        fr = vk.Frame(ctx.k, bg=(14, 16, 21))
        # perspective floor grid, as CockpitRoom's FloorAndCeiling
        hor = 420
        for i in range(-12, 13):
            fr.line((960 + i * 30, hor), (960 + i * 260, 1080), (40, 46, 56), 1.5)
        for j in range(1, 14):
            yy = hor + (1080 - hor) * (j / 14) ** 2
            fr.line((0, yy), (1920, yy), (40, 46, 56), 1.5)
        for j in range(1, 8):
            yy = hor - (hor - 0) * (j / 8) ** 2
            fr.line((0, yy), (1920, yy), (30, 34, 42), 1.2)
        # screens on a ring round the eye: project rectangles
        rng = np.random.default_rng(3)
        eye = np.array([0.0, 0.0, -0.35])
        f = 760.0
        quads = []
        for k in range(12):
            az = math.radians(-115 + k * (230 / 11))
            cx, cz = 2.0 * math.sin(az), 2.0 * math.cos(az)
            if k in (2, 9):
                continue  # an interior drone: its screen is hidden
            tx, tz = math.cos(az), -math.sin(az)
            hw, hh = 0.5, 0.29
            corners = [(cx - tx * hw, hh, cz - tz * hw), (cx + tx * hw, hh, cz + tz * hw),
                       (cx + tx * hw, -hh, cz + tz * hw), (cx - tx * hw, -hh, cz - tz * hw)]
            pts = []
            depth = 0
            for (X, Y, Z) in corners:
                Z2 = Z - eye[2]
                if Z2 < 0.25:
                    break
                pts.append((960 + f * X / Z2, hor - 40 - f * (Y + 0.1) / Z2))
                depth += Z2
            if len(pts) == 4:
                quads.append((depth, pts, k))
        for depth, pts, k in sorted(quads, key=lambda q: -q[0]):
            img = _fake_feed(rng, k)
            _warp_into(fr, img, pts)
            fr.polyline(np.vstack([pts, pts[:1]]), (90, 96, 110), 2)
        # the pilot, from behind
        fr.poly(vk.rounded_rect(760, 900, 400, 260, 130), (24, 26, 32))
        fr.circle((960, 860), 88, (30, 33, 40))
        fr.rect(870, 820, 180, 70, (22, 24, 30), radius=24)
        ctx.state["base"] = fr.img.copy()
    fr = vk.Frame(ctx.k, img=ctx.state["base"].copy())
    return fr


def _fake_feed(rng, k):
    w, h = 320, 180
    img = np.zeros((h, w, 3), np.uint8)
    sky = np.array([150, 186, 220]) * (0.85 + 0.15 * (k % 3) / 2)
    img[:] = sky.astype(np.uint8)
    img[int(h * 0.62):] = (70, 74, 80)
    x = 0
    while x < w:
        bw = int(rng.integers(25, 70))
        bh = int(rng.integers(50, 140))
        col = tuple(int(c) for c in rng.integers(90, 200, 3))
        cv2.rectangle(img, (x, int(h * 0.62) - bh), (x + bw - 4, int(h * 0.62)), col, -1)
        for wy in range(int(h * 0.62) - bh + 8, int(h * 0.62) - 10, 14):
            for wx in range(x + 5, x + bw - 10, 12):
                cv2.rectangle(img, (wx, wy), (wx + 5, wy + 7), (60, 70, 84), -1)
        x += bw
    cv2.rectangle(img, (0, int(h * 0.62)), (w, int(h * 0.7)), (110, 112, 118), -1)
    return img


def _warp_into(fr, img, pts):
    k = fr.k
    dst = (np.asarray(pts, np.float32) * k)
    src = np.float32([[0, 0], [img.shape[1], 0], [img.shape[1], img.shape[0]], [0, img.shape[0]]])
    M = cv2.getPerspectiveTransform(src, dst)
    x0, y0 = np.floor(dst.min(0)).astype(int)
    x1, y1 = np.ceil(dst.max(0)).astype(int)
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(fr.W, x1), min(fr.H, y1)
    if x1 <= x0 or y1 <= y0:
        return
    T = np.array([[1, 0, -x0], [0, 1, -y0], [0, 0, 1]], float) @ M
    warped = cv2.warpPerspective(img, T, (x1 - x0, y1 - y0), flags=cv2.INTER_LINEAR)
    mask = cv2.warpPerspective(np.full(img.shape[:2], 255, np.uint8), T, (x1 - x0, y1 - y0))
    sub = fr.img[y0:y1, x0:x1]
    m = (mask > 127)
    sub[m] = warped[m]


def v_interface_ring(ctx):
    """Left: the swarm from above, edge drones facing out. Right: the same colours on the screens."""
    fr = ctx.frame()
    view = View((0, 0), 12.0, (520, 560))
    t = ctx.t
    base = ring_offsets()
    breathe = 1 + 0.03 * math.sin(t * 0.9)
    P = base * breathe + np.c_[np.sin(t * 0.7 + np.arange(N_DRONES)), np.cos(t * 0.6 + np.arange(N_DRONES) * 1.7)] * 0.4
    yaw = headings(P)
    mask = hull_mask(P)
    order = np.argsort((yaw + 360 - 0) % 360)
    fr.text("Seen from above", (520, 130), 34, MUTED, "semibold", anchor="ma")
    fr.circle(view((0, 0)), RING_R * 12 * 1.25, PANEL, alpha=0.6)
    items = []
    for rank, i in enumerate(order):
        a0 = 0.12 + 0.55 * rank / N_DRONES
        a = ease((ctx.u - a0) / 0.06) if ctx.params.get("reveal", True) else 1.0
        a *= 1.0 if mask[i] else 0.0
        draw_drone(fr, view, P[i], yaw[i], hue(i), alpha=max(a, 0.35), cone=12 * max(a, 0.0 if mask[i] else 1.0), cone_alpha=0.18)
        items.append((yaw[i], hue(i), a))
    draw_screen_ring(fr, (1420, 540), 280, items)
    fr.arrow((520, 560 + 300), (520, 560 + 250), MUTED, 3, 12)
    fr.text("forward", (520, 870), 22, MUTED, anchor="ma")
    a = ease((ctx.u - 0.75) / 0.1)
    fr.text("Each edge drone faces outwards. Its camera is shown on the screen in the same direction.",
            (960, 940), 32, INK, anchor="ma", alpha=a, max_width=1500)
    return fr


def _gap_world():
    A = np.array([[-48, -16], [-11, -16], [-11, 16], [-48, 16]], float)
    B = A * [-1, 1]
    return A, B


def _gap_positions(t, dur, span):
    def pos(tt):
        u = clamp01(tt / dur)
        u = span[0] + (span[1] - span[0]) * u
        # approach, then slow through the gap (the squeeze), then out
        zc = float(np.interp(u, [0.0, 0.35, 0.75, 1.0], [-62.0, -8.0, 8.0, 62.0]))
        s = 1 - smoothstep(12, 42, abs(zc))
        base = ring_offsets()
        ang = np.radians(RING_PHASE + 36 * np.arange(N_DRONES))
        ax, az = lerp(RING_R, 7.0, s), lerp(RING_R, 24.0, s)
        P = np.c_[ax * np.sin(ang), az * np.cos(ang)]
        side = {2: np.array([2.4, 3.0]), 7: np.array([-2.4, -3.0])}
        for i, q in side.items():
            P[i] = lerp(base[i], q, s)
        return P + [0.0, zc]
    return pos


def v_deform_gap(ctx):
    """The swarm squeezes between two buildings; two drones end up inside and their screens hide."""
    fr = ctx.frame()
    span = ctx.params.get("span", [0, 1])
    pos = _gap_positions(ctx.t, ctx.dur, span)
    view = View((0, 0), 6.6, (520, 560))
    fr.rect(520 - 430, 120, 860, 880, GROUND, radius=18)
    fr.rect(520 - 11 * 6.6, 120, 22 * 6.6, 880, ASPHALT)
    for poly in _gap_world():
        fr.poly(view(poly), BUILDING, outline=BUILDING_EDGE, width=1.5)
    P = pos(ctx.t)
    yaw = headings(P)
    alpha = boundary_alpha(pos, ctx.t)
    items = []
    for i in range(N_DRONES):
        draw_drone(fr, view, P[i], yaw[i], hue(i), alpha=alpha[i], cone=9, cone_alpha=0.16)
        items.append((yaw[i], hue(i), alpha[i]))
    fr.text("Seen from above", (520, 82), 30, MUTED, "semibold", anchor="ma")
    draw_screen_ring(fr, (1420, 520), 270, items)
    hidden = [i for i in range(N_DRONES) if alpha[i] < 0.5]
    if hidden and ctx.params.get("annotate", True):
        a = clamp01(1.0 - min(alpha[i] for i in hidden) * 2)
        sp = view(P[hidden[0]])
        fr.line(sp + [14, -6], (820, 330), ACCENT, 2, alpha=a)
        fr.text("pushed inside:\nits screen is hidden", (830, 300), 28, ACCENT, "semibold", alpha=a)
    return fr


def _split_positions(dur, span, building_r, z0=-78, z1=78):
    r_av = building_r + 5.0
    L = r_av + 14.0

    def pos(tt):
        u = clamp01(tt / dur)
        u = span[0] + (span[1] - span[0]) * u
        zc = lerp(z0, z1, ease(u) * 0.15 + u * 0.85)
        P = ring_offsets() + [0.0, zc]
        out = P.copy()
        for i, (x, z) in enumerate(P):
            q = 1 - (z / L) ** 2
            W = r_av * q ** 1.5 if q > 0 else 0.0
            out[i, 0] = math.copysign(math.sqrt(x * x + W * W), x)
        return out
    return pos


_SPLIT_BUILDING = np.array([[-12, -9], [-2, -15], [11, -11], [14, 2], [6, 13], [-9, 12], [-15, 2]], float)


def v_split_building(ctx):
    """Steer straight into a building: the swarm splits round it and joins up again."""
    fr = ctx.frame(bg=GROUND)
    view = View((0, 0), 5.6, (960, 545))
    poly = _SPLIT_BUILDING
    rb = float(np.max(np.hypot(*poly.T)))
    span = ctx.params.get("span", [0, 1])
    pos = _split_positions(ctx.dur, span, rb)
    # streets either side, so the picture reads as a city corner
    for x in (-34, 34):
        fr.rect(*view((x - 9, 95)), 18 * 5.6, 190 * 5.6, ASPHALT)
    fr.poly(view(poly * 1.25), FOOTPATH)
    fr.poly(view(poly), BUILDING, outline=BUILDING_EDGE, width=2)
    P = pos(ctx.t)
    c = P.mean(axis=0)
    # the pilot's command: straight ahead, through the building
    cmd_a = 1 - smoothstep(0.55, 0.7, ctx.u)
    fr.dashed(view([[0, c[1] + 18], [0, 70]]), ACCENT, 3, alpha=0.9 * cmd_a, phase=ctx.t * 30)
    fr.arrow(view([0, c[1] + 4]), view([0, c[1] + 26]), ACCENT, 6, 22, alpha=cmd_a)
    yaw = headings(P)
    alpha = boundary_alpha(pos, ctx.t)
    for i in range(N_DRONES):
        draw_drone(fr, view, P[i], yaw[i], hue(i), alpha=alpha[i], cone=7, cone_alpha=0.14, r=8)
    # captions in the picture
    zc = c[1]
    msgs = [(-999, -24, "Keep pushing straight ahead, into the building"),
            (-24, 14, "Obstacle avoidance splits the swarm around it"),
            (14, 999, "...and it joins back up on the other side")]
    for a0, a1, m in msgs:
        a = min(smoothstep(a0, a0 + 6, zc), 1 - smoothstep(a1 - 6, a1, zc)) if a0 > -999 else 1 - smoothstep(a1 - 6, a1, zc)
        if a1 == 999:
            a = smoothstep(a0, a0 + 6, zc)
        if a > 0.01:
            fr.text(m, (960, 110), 40, INK, "semibold", anchor="ma", alpha=a)
    # mini stick: pushed forward the whole time
    rc_stick_icon(fr, (1660, 760), 90, (0, 1.0 * cmd_a + 0.0), "right stick: forward", ctx.t)
    return fr


def rc_stick_icon(fr, c, r, st, label, t):
    fr.circle(c, r, (30, 32, 38))
    fr.circle(c, r, (100, 106, 118), width=2)
    k = (c[0] + st[0] * r * 0.45, c[1] - st[1] * r * 0.45)
    fr.circle(k, r * 0.5, (88, 94, 106))
    fr.circle(k, r * 0.32, (120, 126, 138))
    fr.text(label, (c[0], c[1] + r + 18), 24, MUTED, anchor="ma")


def _surround_ring_targets(city, bid, P, margin=7.0):
    """Where each drone sits once wrapped round building bid: evenly round it, in the drones'
    current angular order, at the footprint's support distance plus a margin."""
    c = city.centre(bid)
    poly = city.buildings[bid] - c
    ang = np.arctan2(P[:, 0] - c[0], P[:, 1] - c[1])
    order = np.argsort(ang)
    a0 = ang[order[0]]
    targets = np.empty_like(P)
    for rank, i in enumerate(order):
        a = a0 + 2 * math.pi * rank / len(P)
        d = np.array([math.sin(a), math.cos(a)])
        support = np.max(poly @ d)
        targets[i] = c + d * (support + margin)
    return targets


def _polar_lerp(P0, P1, c, s):
    out = np.empty_like(P0)
    for i in range(len(P0)):
        a0 = math.atan2(*(P0[i] - c))
        a1 = math.atan2(*(P1[i] - c))
        da = (a1 - a0 + math.pi) % (2 * math.pi) - math.pi
        r0, r1 = np.hypot(*(P0[i] - c)), np.hypot(*(P1[i] - c))
        a, r = a0 + da * s, lerp(r0, r1, s)
        out[i] = c + r * np.array([math.sin(a), math.cos(a)])
    return out


def v_look_down_streets(ctx):
    """One drone in a street sees along it; the swarm wrapped round a building sees every
    street round it. Pedestrians in view are counted live."""
    city, goal, ring = practice_goal()
    peds = practice_peds()
    gc = city.centre(goal)
    fr = ctx.frame(bg=GROUND)
    view = View(gc + [-4, 0], 4.6, (960, 560), up_yaw=UP)
    draw_city(fr, city, view, goal=goal, ring=ring, goal_alpha=0.6, road_in=False)
    P_ped = peds.positions(ctx.t + 20)
    phase_b = smoothstep(0.42, 0.52, ctx.u)
    if phase_b < 0.5:
        # the swarm squeezed into the street beside the goal: it sees along that street only
        o, syaw = _street_beside(city, goal)
        h, r = yaw_vec(syaw), yaw_vec(syaw + 90)
        off = ring_offsets()
        P = o + h * 14 + np.outer(off[:, 1] * 1.0, h) + np.outer(off[:, 0] * 0.33, r)
        yaw = headings(P)
        mask = hull_mask(P)
        polys = [city.visible(P[i], yaw[i], rng=140) for i in range(N_DRONES) if mask[i]]
        drones = [(P[i], yaw[i], hue(i)) for i in range(N_DRONES)]
        title = "Kept to a street: the buildings either side block the view"
    else:
        r = city.radius(goal)
        P = gc + ring_offsets(r + 8)
        P = _surround_ring_targets(city, goal, P, 7)
        yaw = headings(P)
        polys = [city.visible(P[i], yaw[i], rng=140) for i in range(N_DRONES)]
        drones = [(P[i], yaw[i], hue(i)) for i in range(N_DRONES)]
        title = "Wrapped round a building: you look down every street around it"
    a = min(ease((ctx.u - (0 if phase_b < 0.5 else 0.5)) / 0.06), 1.0)
    lit = np.zeros(len(P_ped), bool)
    for poly in polys:
        fr.poly(view(poly), EDGE_GREEN, alpha=0.2 * a)
        lit |= MplPath(poly).contains_points(P_ped)
    draw_peds(fr, view, P_ped, lit=lit)
    for o, yw, col in drones:
        draw_drone(fr, view, o, yw, col, r=7)
    fr.rect(560, 60, 800, 74, (10, 12, 16), alpha=0.7, radius=12)
    fr.text(title, (960, 76), 32, INK, "semibold", anchor="ma", alpha=a)
    fr.rect(70, 880, 420, 70, (10, 12, 16), alpha=0.7, radius=12)
    fr.text(f"pedestrians in view: {int(lit.sum())} of {len(lit)}", (90, 896), 30, EDGE_GREEN, "semibold", alpha=a)
    return fr


def _street_beside(city, goal):
    """A point on the street between the goal and its westernmost neighbour, facing along it."""
    best = None
    for pl, w, bl in city.streets:
        if goal in bl and -1 not in bl:
            mid = pl.mean(axis=0)
            if best is None or mid[0] < best[0][0]:
                d = pl[-1] - pl[0]
                best = (mid, d)
    mid, d = best
    yaw = vk.vec_yaw(d)
    return mid - yaw_vec(yaw) * 14, yaw


def v_adaptive_spread(ctx):
    """Narrow in a street, wide round a building: the swarm's size is set for you."""
    fr = ctx.frame(bg=GROUND)
    view = View((0, -18), 5.0, (760, 560))
    blocks = [np.array([[x0, z0], [x1, z0], [x1, z1], [x0, z1]], float)
              for (x0, x1) in ((-60, -10), (10, 60)) for (z0, z1) in ((-110, -66), (-56, -12))]
    fr.rect(*view((-10, 90)), 20 * 5.0, 210 * 5.0, ASPHALT)
    fr.rect(*view((-75, 0)), 150 * 5.0, 70 * 5.0, ASPHALT)
    target = np.array([[-14, 26], [2, 20], [15, 28], [14, 44], [0, 50], [-14, 42]], float)
    fr.rect(*view((-42, 78)), 84 * 5.0, 66 * 5.0, ASPHALT, radius=60)
    for b in blocks:
        fr.poly(view(b), BUILDING, outline=BUILDING_EDGE, width=1.5)
    fr.poly(view(target), BUILDING, outline=BUILDING_EDGE, width=1.5)
    tc = target.mean(axis=0)

    def pos(tt):
        u = clamp01(tt / ctx.dur)
        zc = lerp(-100, tc[1], ease(min(1, u / 0.8)))
        s = smoothstep(-14, tc[1] - 4, zc)
        narrow = np.c_[0.42 * ring_offsets()[:, 0], 1.1 * ring_offsets()[:, 1]] + [0, zc]
        city = _OneBuilding(target)
        wrap = _surround_ring_targets(city, 0, narrow, 7)
        return _polar_lerp(narrow, wrap, tc, s), s

    P, s = pos(ctx.t)
    yaw = headings(P)
    alpha = boundary_alpha(lambda tt: pos(tt)[0], ctx.t)
    for i in range(N_DRONES):
        draw_drone(fr, view, P[i], yaw[i], hue(i), alpha=alpha[i], cone=6, r=7)
    # gauge
    gx, gy = 1400, 380
    fr.text("Swarm size (automatic)", (gx, gy - 60), 32, INK, "semibold")
    fr.rect(gx, gy, 420, 26, PANEL, radius=13)
    fr.rect(gx, gy, 420 * lerp(0.42, 1.0, s), 26, CYAN, radius=13)
    fr.text("narrow", (gx, gy + 40), 24, MUTED)
    fr.text("wide", (gx + 420, gy + 40), 24, MUTED, anchor="ra")
    fr.text("narrows to fit a street" if s < 0.5 else "widens to surround a building",
            (gx, gy + 110), 30, CYAN, "semibold", max_width=480)
    return fr


class _OneBuilding:
    def __init__(self, poly):
        self.buildings = {0: poly}

    def centre(self, bid):
        return self.buildings[bid].mean(axis=0)


# -- controls -----------------------------------------------------------------------------------
CONTROL_TEXT = {
    "intro": ("", ""),
    "forward": ("Right stick: up / down", "fly forward / backward"),
    "sideways": ("Right stick: left / right", "fly left / right"),
    "updown": ("Left stick: up / down", "climb / descend"),
    "yaw": ("Left stick: left / right", "turn"),
    "pitch": ("Top-left dial", "tilt the cameras"),
    "c2": ("C2 button (on the back)", "I've found the hat"),
}


def v_controls(ctx):
    """The controller on the left with the control in question moving; its effect on the right."""
    p = ctx.params
    which = p.get("control", "intro")
    fr = ctx.frame()
    u, t = ctx.u, ctx.t
    segs_a = [(0.08, 0.42, 1.0), (0.55, 0.88, -1.0)]
    stick_l, stick_r, dial = (0.0, 0.0), (0.0, 0.0), 0.0
    hl = ()
    if which == "forward":
        stick_r = (0.0, stick_schedule(u, segs_a)); hl = ("right",)
    elif which == "sideways":
        stick_r = (stick_schedule(u, segs_a), 0.0); hl = ("right",)
    elif which == "updown":
        stick_l = (0.0, stick_schedule(u, [(0.06, 0.5, 1.0), (0.6, 0.9, -1.0)])); hl = ("left",)
    elif which == "yaw":
        stick_l = (stick_schedule(u, [(0.1, 0.36, 1.0), (0.6, 0.86, -1.0)]), 0.0); hl = ("left",)
    elif which == "pitch":
        dial = stick_schedule(u, [(0.08, 0.42, -1.0), (0.55, 0.85, 1.0)]); hl = ("dial",)
    elif which == "c2":
        hl = ("c2",)
    # With a flight log, the photo's sticks and dial follow what the recorded pilot actually did.
    log = flight_log(p["log"]) if p.get("log") else None
    if log is not None:
        src = footage_time(ctx.cue, t)
        ax = {"forward": "pitch", "sideways": "roll", "updown": "throttle", "yaw": "yaw"}.get(which)
        if which in ("forward", "sideways"):
            stick_r = (log_at(log, "roll", src), log_at(log, "pitch", src))
        elif which in ("updown", "yaw"):
            stick_l = (log_at(log, "yaw", src), log_at(log, "throttle", src))
        elif which == "pitch":
            d_now, d_prev = log_at(log, "dial", src), log_at(log, "dial", src - 0.2)
            dial = float(np.clip((d_now - d_prev) * 12.0, -1.0, 1.0))   # the dial turns while the gimbal moves
    if which == "intro":
        rc_photo(fr, 520, 190, 880, hl=(), t=t, c2=True)
        return fr
    rc_photo(fr, 90, 330, 780, hl=hl, stick_l=stick_l, stick_r=stick_r, dial_l=dial, t=t, c2=(which == "c2"))
    head, sub = CONTROL_TEXT[which]
    head, sub = p.get("head", head), p.get("sub", sub)   # e.g. "tilt the camera" for one drone
    fr.text(head, (480, 150), 46, INK, "semibold", anchor="ma")
    fr.text(sub, (480, 216), 36, ACCENT, anchor="ma")
    if log is not None or p.get("panel") is False:
        return fr   # the recording sits beside it (the cue's footage rect)
    demo = {"forward": _demo_translate, "sideways": _demo_translate, "updown": _demo_updown, "yaw": _demo_yaw,
            "pitch": _demo_pitch, "c2": _demo_c2}[which]
    demo(fr, ctx, which, stick_l, stick_r, dial)
    return fr


def _demo_translate(fr, ctx, which, sl, sr, dial):
    cx, cy = 1420, 560
    segs = [(0.08, 0.42, 1.0), (0.55, 0.88, -1.0)]
    off = integrate(lambda uu: 8.0 * stick_schedule(uu, segs), ctx.u, ctx.dur)
    fwd = which == "forward"
    fr.rect(cx - 380, cy - 380, 760, 760, GROUND, radius=20)
    view = View((0, 0), 5.0, (cx, cy))
    pitch = 34.0
    shift = np.array([0.0, off]) if fwd else np.array([off, 0.0])
    for i in range(-5, 6):
        for j in range(-5, 6):
            c = np.array([(i + 0.5) * pitch, (j + 0.5) * pitch]) - (shift % pitch)
            sq = np.array([[-11, -11], [11, -11], [11, 11], [-11, 11]], float) + c
            pts = view(sq)
            if np.all(np.abs(pts - [cx, cy]) < 370):
                fr.poly(pts, BUILDING, outline=BUILDING_EDGE, width=1)
    P = ring_offsets(9.0) * [1, 1]
    yaw = headings(P)
    for i in range(N_DRONES):
        draw_drone(fr, view, P[i], yaw[i], hue(i), cone=5, r=7)
    v = stick_schedule(ctx.u, segs)
    if abs(v) > 0.05:
        d = np.array([0, -1]) if fwd else np.array([1, 0])
        fr.arrow((cx, cy), (cx + d[0] * 150 * v, cy + d[1] * 150 * v), ACCENT, 8, 26)
    fr.text("you are facing up the picture", (cx, cy + 400), 24, MUTED, anchor="ma")


def _demo_updown(fr, ctx, which, sl, sr, dial):
    x0, y0, w, h = 1050, 200, 760, 700
    fr.rect(x0, y0, w, h, (24, 30, 40), radius=20)
    ground_y = y0 + h - 70
    m = 9.0
    fr.rect(x0, ground_y, w, 70, GROUND)
    for bx, bw in ((x0 + 40, 140), (x0 + 330, 130), (x0 + 600, 130)):
        fr.rect(bx, ground_y - 50 * m, bw, 50 * m, BUILDING, outline=BUILDING_EDGE, width=1.5)
    ceil_y = ground_y - 45 * m
    fr.dashed([(x0 + 10, ceil_y), (x0 + w - 10, ceil_y)], RED, 3)
    fr.text("height limit (below the rooftops)", (x0 + w - 20, ceil_y - 40), 24, RED, anchor="ra")
    segs = [(0.06, 0.5, 1.0), (0.6, 0.9, -1.0)]
    alt = 12.0
    us = np.linspace(0, ctx.u, 300)
    for a, b in zip(us[:-1], us[1:]):
        alt = min(45.0, max(4.0, alt + 4.0 * stick_schedule(a, segs) * (b - a) * ctx.dur))
    ys = ground_y - alt * m
    for k in range(6):
        fr.circle((x0 + 220 + k * 34, ys + (k % 2) * 6), 8, hue(k))
    fr.text(f"{alt:4.0f} m", (x0 + 470, ys - 18), 28, INK, "semibold")


def _demo_yaw(fr, ctx, which, sl, sr, dial):
    segs = [(0.1, 0.36, 1.0), (0.6, 0.86, -1.0)]
    psi = integrate(lambda uu: 40.0 * stick_schedule(uu, segs), ctx.u, ctx.dur)
    P = ring_offsets()
    yaw = headings(P)
    items = [((yaw[i] - psi), hue(i), 1.0) for i in range(N_DRONES)]
    draw_screen_ring(fr, (1300, 520), 230, items, title=False)
    fr.text("the screens turn round you", (1300, 800), 26, MUTED, anchor="ma")
    view = View((0, 0), 5.0, (1730, 300))
    fr.circle((1730, 300), 115, PANEL)
    for i in range(N_DRONES):
        draw_drone(fr, view, P[i], yaw[i], hue(i), r=5)
    d = yaw_vec(psi)
    fr.arrow((1730, 300), (1730 + d[0] * 95, 300 - d[1] * 95), ACCENT, 6, 20)
    fr.text("the swarm (from above):\nthe drones keep their places,\nyour forward turns", (1730, 440), 22, MUTED,
            anchor="ma", max_width=360)


def _demo_pitch(fr, ctx, which, sl, sr, dial):
    x0, y0, w, h = 1050, 200, 760, 700
    fr.rect(x0, y0, w, h, (24, 30, 40), radius=20)
    gy = y0 + h - 80
    fr.rect(x0, gy, w, 80, GROUND)
    fr.rect(x0 + w - 170, gy - 420, 150, 420, BUILDING, outline=BUILDING_EDGE, width=1.5)
    segs = [(0.08, 0.42, -1.0), (0.55, 0.85, 1.0)]
    knob = integrate(lambda uu: 0.7 * stick_schedule(uu, segs), ctx.u, ctx.dur)
    knob = max(-1.0, min(1.0, knob))
    pitch = 90.0 * knob if knob < 0 else 60.0 * knob
    d = (x0 + 200, gy - 330)
    fr.wedge(d, 320, -pitch - 23, -pitch + 23, CYAN, alpha=0.18)
    fr.rect(d[0] - 34, d[1] - 10, 68, 20, (190, 196, 210), radius=8)
    fr.circle((d[0] + 30, d[1] + 12), 12, (60, 64, 74))
    word = "level" if abs(pitch) < 8 else ("looking down" if pitch < 0 else "looking up")
    fr.text(f"camera: {word} ({pitch:+.0f}°)", (x0 + 40, y0 + 30), 30, INK, "semibold")
    fr.text("every drone's camera tilts together; let go and it stays", (x0 + 40, y0 + 76), 22, MUTED)


def _demo_c2(fr, ctx, which, sl, sr, dial):
    x, y = 1150, 330
    a = ease((ctx.u - 0.25) / 0.1)
    fr.rect(x, y, 600, 170, PANEL, radius=40, alpha=a)
    fr.poly([(x + 90, y + 170), (x + 60, y + 240), (x + 160, y + 170)], PANEL, alpha=a)
    fr.text("“It's a bucket hat!”", (x + 300, y + 58), 44, INK, "semibold", anchor="ma", alpha=a)
    b = ease((ctx.u - 0.55) / 0.1)
    fr.text("press C2, then say which hat", (x + 300, y + 330), 30, ACCENT, "semibold", anchor="ma", alpha=b)
    fr.text("the moderator verifies it", (x + 300, y + 380), 28, MUTED, anchor="ma", alpha=b)
    draw_hat(fr, "bucket", (x + 300, y + 520), 160)


# -- the task -------------------------------------------------------------------------------------
def v_practice_map(ctx):
    """The practice city from above, zooming to the goal: pedestrians round the ring buildings,
    none round the centre one, and (optionally) the one with the hat."""
    p = ctx.params
    city, goal, ring = practice_goal()
    peds = practice_peds()
    ex = city.extent
    ov_c = np.array([(ex[0] + ex[1]) / 2 - 55, (ex[2] + ex[3]) / 2])
    ov_s = 2.6
    gc = city.centre(goal)
    zoom = p.get("zoom", [0.55, 0.8])
    s = ease((ctx.u - zoom[0]) / max(zoom[1] - zoom[0], 1e-6)) if zoom else 0.0
    if p.get("start_zoomed"):
        s = 1.0
    scale = math.exp(lerp(math.log(ov_s), math.log(p.get("zoom_scale", 7.2)), s))
    centre = lerp(ov_c, gc, s)
    view = View(tuple(centre), scale, (960, 560), up_yaw=UP)
    fr = ctx.frame(bg=GROUND)
    ga = ease((ctx.u - p.get("goal_at", 0.3)) / 0.08)
    draw_city(fr, city, view, goal=goal, ring=ring, goal_alpha=ga)
    if s < 0.99:
        a = 1 - s
        sp = view(city.spawn)
        fr.circle(sp, 14, GREEN, alpha=a)
        fr.text("you start here", (sp[0] + 24, sp[1] - 18), 28, GREEN, "semibold", alpha=a)
        line_x = city.spawn[0] + 65
        fr.line(view([line_x, -12]), view([line_x, 12]), CYAN, 4, alpha=a)
        fr.text("start line", view([line_x, -14]) + [10, -14], 22, CYAN, alpha=a)
        gp = view(gc)
        fr.text("the goal", (gp[0], gp[1] - 70), 30, ACCENT, "semibold", anchor="ma", alpha=a * ga)
    pa = ease((ctx.u - p.get("peds_at", 0.0)) / 0.1) if p.get("peds", True) else 0.0
    if pa > 0:
        P = peds.positions(ctx.t + 20)
        hr = ease((ctx.u - p.get("hat_at", 2.0)) / 0.08)
        draw_peds(fr, view, P, hat_idx=peds.hat, hat_reveal=hr, t=ctx.t)
        if s > 0.9:
            gp = view(gc)
            fr.text("centre building: no pedestrians", (gp[0], gp[1] - 0.22 * scale * 40 - 40), 26, ACCENT,
                    "semibold", anchor="ma", alpha=ga * (s - 0.9) * 10)
            if hr > 0:
                hp = view(P[peds.hat])
                fr.text("one has a hat", (hp[0] + 30, hp[1] - 48), 28, HAT, "semibold", alpha=hr)
    return fr


def _search_positions(city, goal, dur):
    gc = city.centre(goal)
    start = np.array([city.extent[0] - 40, gc[1] + 4])

    def pos(tt):
        u = clamp01(tt / dur)
        a = ease(clamp01(u / 0.42))
        c = lerp(start, gc - [30, 0], a)
        P = c + ring_offsets(13.0)
        P = city.push_out(P, 3.0)
        s = ease(clamp01((u - 0.4) / 0.2))
        if s > 0:
            P1 = _surround_ring_targets(city, goal, P, 8.0)
            P = _polar_lerp(P, P1, gc, s)
        return P
    return pos


def v_surround_search(ctx):
    """The swarm flies in, wraps round the centre building and scans the ring; the hat is
    spotted when it falls in a camera's view."""
    city, goal, ring = practice_goal()
    peds = practice_peds()
    gc = city.centre(goal)
    fr = ctx.frame(bg=GROUND)
    view = View(tuple(gc + [-22, 0]), 5.4, (960, 560), up_yaw=UP)
    draw_city(fr, city, view, goal=goal, ring=ring, goal_alpha=0.8, road_in=False)
    pos = _search_positions(city, goal, ctx.dur)
    P = pos(ctx.t)
    yaw = headings(P)
    alpha = boundary_alpha(pos, ctx.t)
    Pp = peds.positions(ctx.t + 20)
    lit = np.zeros(len(Pp), bool)
    hat_seen = False
    for i in range(N_DRONES):
        if alpha[i] < 0.5:
            continue
        poly = city.visible(P[i], yaw[i], rng=80)
        fr.poly(view(poly), hue(i), alpha=0.12)
        inside = MplPath(poly).contains_points(Pp)
        lit |= inside
        hat_seen |= bool(inside[peds.hat]) and ctx.u > 0.5
    if hat_seen and "spotted" not in ctx.state:
        ctx.state["spotted"] = ctx.t
    t_sp = ctx.state.get("spotted")
    if t_sp is not None and ctx.t < t_sp:
        del ctx.state["spotted"]
        t_sp = None
    draw_peds(fr, view, Pp, hat_idx=peds.hat, lit=lit, hat_reveal=1.0 if t_sp is not None else 0.0, t=ctx.t)
    for i in range(N_DRONES):
        draw_drone(fr, view, P[i], yaw[i], hue(i), alpha=alpha[i], r=7)
    if t_sp is not None and ctx.params.get("callout", True):
        a = ease((ctx.t - t_sp - 0.6) / 0.4)
        hp = view(Pp[peds.hat])
        fr.text("the hat is on a screen!", (hp[0] + 30, hp[1] - 50), 30, HAT, "semibold", alpha=a)
        b = ease((ctx.t - t_sp - 1.8) / 0.4)
        o_rc_c2(fr, {"pos": [1640, 820], "scale": 1.0, "text": "press C2 and say the hat"}, ctx.t, b)
    return fr


def v_clip_grid(ctx):
    """Several segments of one recording side by side, each revealed at its own time with its label.
    params: {file, tiles: [{start, len, label, at}], rect: [x, y, w, h] of the grid, cols, aspect}"""
    p = ctx.params
    fr = ctx.frame()
    key = p["file"]
    if "feeds" not in ctx.state:
        path = os.path.join(FOOTAGE_DIR, key)
        ctx.state["feeds"] = [vk.Footage(path) if os.path.exists(path) else None for _ in p["tiles"]]
    tiles = p["tiles"]
    cols = p.get("cols", len(tiles))
    rows = math.ceil(len(tiles) / cols)
    gx, gy, gw, gh = p.get("rect", [60, 150, 1800, 760])
    gap = p.get("gap", 30)
    tw = (gw - gap * (cols - 1)) / cols
    th = (gh - gap * (rows - 1)) / rows
    if p.get("title"):
        fr.text(p["title"], (960, 70), 46, INK, "semibold", anchor="ma")
    # intro: {hold, move, height}: each tile first appears large in the centre, then shrinks into its slot.
    intro = p.get("intro")
    order = list(range(len(tiles)))
    if intro:
        # the tile still moving is drawn last, over the ones already in place
        order.sort(key=lambda i: tiles[i].get("at", 0.0))
    for i in order:
        tile = tiles[i]
        a = ease((ctx.t - tile.get("at", 0.0)) / 0.35)
        if a <= 0.01:
            continue
        r_, c_ = divmod(i, cols)
        x, y = gx + c_ * (tw + gap), gy + r_ * (th + gap)
        w_, h_ = tw, th
        if intro:
            local = ctx.t - tile.get("at", 0.0)
            big_h = intro.get("height", 820)
            big_w = tw * big_h / th
            bx, by = 960 - big_w / 2, 540 - big_h / 2 - 20
            k = ease((local - intro.get("hold", 0.9)) / intro.get("move", 0.55))
            x, y, w_, h_ = lerp(bx, x, k), lerp(by, y, k), lerp(big_w, tw, k), lerp(big_h, th, k)
        feed = ctx.state["feeds"][i]
        if feed is not None:
            local = max(0.0, ctx.t - tile.get("at", 0.0))
            ln = tile.get("len", 3.0)
            src = tile["start"] + min(local, ln - 0.05)
            img = feed.at(src)
            if img is not None:
                fr.rect(x - 4, y - 4, w_ + 8, h_ + 8, INK, alpha=0.9 * a, radius=8)
                fr.image(img, x, y, w_, h_, alpha=a, mode="cover")
        if tile.get("label"):
            lh = 64 * h_ / th
            fr.rect(x, y + h_ - lh, w_, lh, (0, 0, 0), alpha=0.6 * a)
            fr.text(tile["label"], (x + w_ / 2, y + h_ - lh * 0.88), 40 * h_ / th, INK, "semibold", anchor="ma", alpha=a)
    return fr


def v_hats(ctx):
    fr = ctx.frame()
    kinds = ["cap", "bucket", "cowboy"]
    for j, k in enumerate(kinds):
        a = ease((ctx.u - 0.1 - 0.22 * j) / 0.08)
        x = 420 + j * 540
        fr.rect(x - 220, 300, 440, 440, PANEL, radius=30, alpha=a)
        if a > 0.01:
            draw_hat(fr, k, (x, 540), 240)
        fr.text(HAT_NAMES[k], (x, 780), 46, INK, "semibold", anchor="ma", alpha=a)
    fr.text("Three kinds of hat", (960, 150), 56, INK, "semibold", anchor="ma")
    return fr


@functools.lru_cache(maxsize=None)
def trial_goals(seed=11):
    """Three illustrative goals in the trial city, picked as BlockGoals picks them: interior,
    every neighbour built, >= 100 m from the spawn and >= 150 m apart, one per third of the
    spawn distances. The real ones are drawn afresh every trial."""
    c = trial_city()
    rng = np.random.default_rng(seed)
    cand = []
    for bid in c.buildings:
        nb = c.neigh.get(bid, set())
        if not c.interior.get(bid) or not nb:
            continue
        if not all(n in c.buildings and c.interior.get(n) for n in nb):
            continue
        d = np.hypot(*(c.centre(bid) - c.spawn))
        if d >= 100:
            cand.append((d, bid))
    cand.sort()
    thirds = np.array_split(np.array(cand), 3)
    for _ in range(500):
        pick = [int(th[rng.integers(len(th))][1]) for th in thirds]
        cs = [c.centre(b) for b in pick]
        rings = [c.neigh[b] for b in pick]
        if all(np.hypot(*(cs[i] - cs[j])) >= 150 for i in range(3) for j in range(i)) and \
                not (rings[0] & rings[1] or rings[0] & rings[2] or rings[1] & rings[2]):
            return pick
    return [int(th[0][1]) for th in thirds]


def v_trial_map(ctx):
    """The trial city at the practice city's scale, with three goals placed at random."""
    tc = trial_city()
    pc, _, _ = practice_goal()
    fr = ctx.frame(bg=GROUND)
    s_tr = 1.12
    view = View((-50, 0), s_tr, (900, 520), up_yaw=UP)
    goals = trial_goals()
    hats = ctx.params.get("hats", ["bucket", "cap", "cowboy"])
    ring = set()
    for g in goals:
        ring |= tc.neigh.get(g, set())
    ga = ease((ctx.u - 0.35) / 0.1)
    draw_city(fr, tc, view, ring=ring if ga > 0 else (), goal_alpha=ga)
    sp = view(tc.spawn)
    fr.circle(sp, 10, GREEN)
    fr.text("start", (sp[0] + 18, sp[1] - 14), 24, GREEN, "semibold")
    for j, g in enumerate(goals):
        a = ease((ctx.u - 0.35 - 0.12 * j) / 0.08)
        if a <= 0.01:
            continue
        gp = view(tc.centre(g))
        fr.circle(gp, 34, ACCENT, width=4, alpha=a)
        fr.rect(gp[0] + 30, gp[1] - 92, 130, 90, PANEL, radius=14, alpha=0.9 * a)
        if a > 0.5:
            draw_hat(fr, hats[j % 3], (gp[0] + 95, gp[1] - 42), 70)
        fr.text(f"goal {j + 1}", (gp[0], gp[1] + 40), 24, ACCENT, "semibold", anchor="ma", alpha=a)
    # the practice city, to scale, for comparison
    pa = 1 - ease((ctx.u - 0.2) / 0.1)
    if ctx.params.get("compare", True):
        pv = View(tuple(np.array([(pc.extent[0] + pc.extent[1]) / 2, (pc.extent[2] + pc.extent[3]) / 2])), s_tr,
                  (1660, 270), up_yaw=UP)
        fr.rect(1480, 100, 360, 400, PANEL, radius=16, alpha=0.85)
        draw_city(fr, pc, pv, road_in=False)
        fr.text("practice city\n(same scale)", (1660, 420), 24, MUTED, "semibold", anchor="ma")
    fr.rect(420, 62, 960, 56, (10, 12, 16), alpha=0.6 * ga, radius=12)
    fr.text(f"{len(tc.buildings)} buildings: three goals, placed at random every trial", (900, 72), 30, INK,
            "semibold", anchor="ma", alpha=ga)
    return fr


# =============================================================================================
# Overlays
# =============================================================================================
def o_rc_c2(fr, ov, t, a):
    """A small controller with C2 lit, e.g. over the search recording when the hat is found."""
    if a <= 0.01:
        return
    x, y = ov.get("pos", (1640, 820))
    s = ov.get("scale", 1.0)
    w = 380 * s
    h = w * (RC_CROP[3] - RC_CROP[1]) / (RC_CROP[2] - RC_CROP[0])
    fr.rect(x - w / 2 - 24, y - h / 2 - 24, w + 48, h + 100, (10, 12, 16), alpha=0.75 * a, radius=18)
    rc_photo(fr, x - w / 2, y - h / 2, w, hl=("c2",), t=t, labels=False)
    if ov.get("text"):
        fr.text(ov["text"], (x, y + h / 2 + 20), 28 * s, ACCENT, "semibold", anchor="ma", alpha=a)


_closeup_cache = {}


def _closeup_source(name):
    """The front feed recording and its per-frame hat coordinates (VideoShotRunner's <shot>_front_hat.csv)."""
    if name not in _closeup_cache:
        base = os.path.join(FOOTAGE_DIR, name)
        feed = vk.Footage(base + ".mp4") if os.path.exists(base + ".mp4") else None
        rows = []
        if os.path.exists(base + "_hat.csv"):
            import csv
            with open(base + "_hat.csv", encoding="utf-8") as f:
                for r in csv.DictReader(f):
                    try:
                        rows.append((float(r["t"]), float(r["u"]), float(r["v"]), float(r["depth"]),
                                     float(r["range"]), r["clear"] == "1"))
                    except (ValueError, KeyError):
                        rows.append((float(r.get("t") or 0), None, None, None, None, False))
        _closeup_cache[name] = (feed, rows)
    return _closeup_cache[name]


def o_hat_closeup(fr, ov, t, a):
    """The front screen's own feed, enlarged, with a ring on the hat where it is in view.
    ov: {name: S07_front (footage/<name>.mp4 + <name>_hat.csv), src_at: source seconds at `at`,
         speed, rect: [x, y, w, h], label, from_rect (grow out of the screen there), mark, grow,
         zoom (then close in on the hat this many times, over zoom_time seconds; 1 = the whole feed)}"""
    feed, rows = _closeup_source(ov.get("name", "S07_front"))
    if feed is None or a <= 0.01:
        return
    src = ov.get("src_at", 0.0) + t * ov.get("speed", 1.0)
    img = feed.at(src)
    if img is None:
        return
    x, y, w, h = ov.get("rect", [980, 330, 880, 495])
    grown = 1.0
    t_grown = t
    if ov.get("from_rect"):
        # Outline the screen in the recording first, then grow the close-up out of it.
        fx, fy, fw, fh = ov["from_rect"]
        mark, grow = ov.get("mark", 0.6), ov.get("grow", 0.8)
        if t < mark:
            pulse = 0.6 + 0.4 * math.sin(t * 10)
            fr.rect(fx - 6, fy - 6, fw + 12, fh + 12, None, outline=ACCENT, width=6, radius=6, alpha=pulse)
            return
        grown = ease((t - mark) / grow)
        t_grown = t - mark - grow
        x, y, w, h = (lerp(fx, x, grown), lerp(fy, y, grown), lerp(fw, w, grown), lerp(fh, h, grown))
    # Where the hat is in the feed now.
    hat = None
    if rows:
        i = int(np.argmin([abs(r[0] - src) for r in rows]))
        _, u, v, depth, rng, clear = rows[i]
        if u is not None and depth > 0 and 0 <= u <= 1 and 0 <= v <= 1 and clear:
            hat = (u, 1 - v)
    # Close in on it: crop the feed round the hat, kept inside the picture.
    H, W = img.shape[:2]
    box = (0.0, 0.0, 1.0, 1.0)            # the visible part of the feed, as fractions (x0, y0, w, h)
    zoom = ov.get("zoom", 1.0)
    if zoom > 1.0 and grown >= 0.999 and hat is not None:
        k = 1.0 + (zoom - 1.0) * ease(t_grown / ov.get("zoom_time", 0.8))
        bw, bh = 1.0 / k, 1.0 / k
        bx = min(max(hat[0] - bw / 2, 0.0), 1.0 - bw)
        by = min(max(hat[1] - bh / 2, 0.0), 1.0 - bh)
        box = (bx, by, bw, bh)
        img = img[int(by * H):int((by + bh) * H), int(bx * W):int((bx + bw) * W)]
    fr.rect(x - 6, y - 6, w + 12, h + 12, ACCENT, alpha=0.9 * a, radius=8)
    fr.image(img, x, y, w, h, alpha=a, mode="cover")
    if grown < 0.999:
        return
    if hat is not None:
        cx = x + (hat[0] - box[0]) / box[2] * w
        cy = y + (hat[1] - box[1]) / box[3] * h
        r = 34 * (1.0 if box[2] > 0.99 else min(2.2, 0.55 / box[2]))
        pulse = 1 + 0.12 * math.sin(t * 6)
        fr.circle((cx, cy), r * pulse, HAT, width=5, alpha=a)
        fr.text(ov.get("ring_label", "the hat"), (cx, cy - r * pulse - 44), 28, HAT, "semibold", anchor="ma",
                alpha=a)
    if ov.get("label"):
        fr.rect(x, y + h - 46, w, 46, (0, 0, 0), alpha=0.55 * a)
        fr.text(ov["label"], (x + 16, y + h - 40), 26, INK, "semibold", alpha=a)


@functools.lru_cache(maxsize=None)
def flight_log(name):
    """VideoShotRunner's per-frame log, footage/<name>_log.csv: {column: array}, or None."""
    path = os.path.join(FOOTAGE_DIR, name + "_log.csv")
    if not os.path.exists(path):
        return None
    import csv
    with open(path, encoding="utf-8") as f:
        rows = list(csv.reader(f))
    head, body = rows[0], rows[1:]
    cols = {}
    for j, h in enumerate(head):
        vals = []
        for r in body:
            try:
                vals.append(float(r[j]))
            except (ValueError, IndexError):
                vals.append(np.nan)
        cols[h] = np.array(vals)
    return cols


def log_at(log, col, t):
    ts = log["t"]
    return float(np.interp(t, ts, log[col]))


def footage_time(cue, t):
    fo = cue.get("footage") or {}
    return fo.get("start", 0.0) + t * fo.get("speed", 1.0)


def o_altitude(fr, ov, t, a):
    """Height readout from the flight log, with the height limit marked."""
    log = flight_log(ov["name"])
    if log is None:
        return
    src = ov.get("src_at", 0.0) + t * ov.get("speed", 1.0)
    alt = log_at(log, "alt", src)
    x, y = ov.get("pos", (1680, 300))
    h = 380
    top = ov.get("max", 50.0)
    fr.rect(x - 70, y - 60, 230, h + 130, (8, 10, 14), alpha=0.7 * a, radius=14)
    fr.text("height", (x + 45, y - 48), 26, MUTED, "semibold", anchor="ma", alpha=a)
    fr.rect(x, y, 26, h, PANEL, alpha=a, radius=8)
    frac = clamp01(alt / top)
    fr.rect(x, y + h * (1 - frac), 26, h * frac, CYAN, alpha=a, radius=8)
    lim = ov.get("limit", 45.0)
    ly = y + h * (1 - lim / top)
    fr.line((x - 14, ly), (x + 40, ly), RED, 3, alpha=a)
    fr.text("limit", (x + 48, ly - 16), 22, RED, alpha=a)
    fr.text(f"{alt:.0f} m", (x + 45, y + h + 12), 34, INK, "semibold", anchor="ma", alpha=a)


def o_drone_mark(fr, ov, t, a):
    """Ring one drone in full-frame overhead footage, placed from the flight log (the overhead camera's pose
    and the drone's position, frame by frame)."""
    log = flight_log(ov["name"])
    if log is None:
        return
    src = ov.get("src_at", 0.0) + t * ov.get("speed", 1.0)
    k = ov["drone"]
    if f"d{k}x" not in log:
        return
    cx, cz = log_at(log, "camX", src), log_at(log, "camZ", src)
    yaw, size = math.radians(log_at(log, "camYaw", src)), log_at(log, "camSize", src)
    dx, dz = log_at(log, f"d{k}x", src) - cx, log_at(log, f"d{k}z", src) - cz
    sc = 540.0 / size
    px = 960 + (dx * math.cos(yaw) - dz * math.sin(yaw)) * sc
    py = 540 - (dx * math.sin(yaw) + dz * math.cos(yaw)) * sc
    pulse = 1 + 0.1 * math.sin(t * 6)
    col = tuple(ov.get("color", HAT))
    fr.circle((px, py), 30 * pulse, col, width=5, alpha=a)
    if ov.get("label"):
        lx, ly = px + ov.get("dx", 50), py + ov.get("dy", -70)
        fr.line((px + 22, py - 22), (lx - 6, ly + 30), col, 3, alpha=a)
        w, hh = fr.text_size(ov["label"], 32, "semibold", max_width=520)
        fr.rect(lx - 12, ly - 6, w + 24, hh + 12, (8, 10, 14), alpha=0.7 * a, radius=10)
        fr.text(ov["label"], (lx, ly), 32, col, "semibold", max_width=520, alpha=a)


def overhead_project(log, src, pts):
    """World (x, z) points -> reference pixels in full-frame overhead footage, from the logged camera pose."""
    cx, cz = log_at(log, "camX", src), log_at(log, "camZ", src)
    yaw, size = math.radians(log_at(log, "camYaw", src)), log_at(log, "camSize", src)
    sc = 540.0 / size
    d = np.asarray(pts, float) - [cx, cz]
    px = 960 + (d[..., 0] * math.cos(yaw) - d[..., 1] * math.sin(yaw)) * sc
    py = 540 - (d[..., 0] * math.sin(yaw) + d[..., 1] * math.cos(yaw)) * sc
    return np.stack([px, py], axis=-1)


def o_building_mark(fr, ov, t, a):
    """Tint practice-city buildings in overhead footage, with a label beside the group.
    ov: {name (log), buildings: [ids], color, label, src_at, speed, dx, dy}"""
    log = flight_log(ov["name"])
    if log is None:
        return
    src = ov.get("src_at", 0.0) + t * ov.get("speed", 1.0)
    city = practice_city()
    col = tuple(ov.get("color", RED))
    cents = []
    for bid in ov["buildings"]:
        poly = overhead_project(log, src, city.buildings[bid])
        fr.poly(poly, col, alpha=0.38 * a, outline=None)
        fr.polyline(poly, col, width=5, closed=True, alpha=a)
        cents.append(poly.mean(axis=0))
    if ov.get("label") and cents:
        c = np.mean(cents, axis=0)
        lx, ly = c[0] + ov.get("dx", 0), c[1] + ov.get("dy", -120)
        w_, h_ = fr.text_size(ov["label"], 32, "semibold", max_width=520)
        fr.rect(lx - w_ / 2 - 14, ly - 8, w_ + 28, h_ + 14, (8, 10, 14), alpha=0.78 * a, radius=10)
        fr.text(ov["label"], (lx, ly), 32, col, "semibold", anchor="ma", max_width=520, alpha=a)


def o_spread_gauge(fr, ov, t, a):
    """The adaptive spread's scale from the flight log, as a narrow-to-wide bar."""
    log = flight_log(ov["name"])
    if log is None or "spread" not in log:
        return
    src = ov.get("src_at", 0.0) + t * ov.get("speed", 1.0)
    sc = log_at(log, "spread", src)
    lo, hi = ov.get("range", [0.6, 1.0])
    frac = clamp01((sc - lo) / (hi - lo))
    x, y = ov.get("pos", (1440, 860))
    w = 400
    fr.rect(x - 24, y - 70, w + 48, 150, (8, 10, 14), alpha=0.72 * a, radius=14)
    fr.text("swarm size (automatic)", (x, y - 58), 28, INK, "semibold", alpha=a)
    fr.rect(x, y, w, 24, PANEL, alpha=a, radius=12)
    fr.rect(x, y, max(24, w * frac), 24, CYAN, alpha=a, radius=12)
    fr.text("narrow", (x, y + 34), 22, MUTED, alpha=a)
    fr.text("wide", (x + w, y + 34), 22, MUTED, anchor="ra", alpha=a)


def o_log_point(fr, ov, t, a):
    """Ring a point logged in viewport coordinates (e.g. arrowU/arrowV, the forward arrow in the interface camera)
    inside a recording drawn at rect (same aspect as the camera)."""
    log = flight_log(ov["name"])
    if log is None:
        return
    cu, cv = ov.get("cols", ["arrowU", "arrowV"])
    if cu not in log or np.all(np.isnan(log[cu])):
        return
    src = ov.get("src_at", 0.0) + t * ov.get("speed", 1.0)
    u, v = log_at(log, cu, src), log_at(log, cv, src)
    if not (0 <= u <= 1 and 0 <= v <= 1):
        return
    x, y, w, h = ov.get("rect", [0, 0, 1920, 1080])
    px, py = x + u * w, y + (1 - v) * h
    pulse = 1 + 0.12 * math.sin(t * 6)
    fr.circle((px, py), ov.get("radius", 34) * pulse, HAT, width=5, alpha=a)
    if ov.get("label"):
        lx, ly = px + ov.get("dx", 60), py + ov.get("dy", 30)
        w_, h_ = fr.text_size(ov["label"], 30, "semibold", max_width=460)
        fr.rect(lx - 12, ly - 6, w_ + 24, h_ + 12, (8, 10, 14), alpha=0.75 * a, radius=10)
        fr.text(ov["label"], (lx, ly), 30, HAT, "semibold", max_width=460, alpha=a)


def o_badge(fr, ov, t, a):
    """A large, steady tag, e.g. '2x SPEED'."""
    x, y = ov.get("pos", (1700, 980))
    size = ov.get("size", 56)
    w, h = fr.text_size(ov["text"], size, "bold")
    fr.rect(x - w / 2 - 30, y - h / 2 - 14, w + 60, h + 28, (8, 10, 14), alpha=0.8 * a, radius=16,
            outline=ACCENT, width=4)
    fr.text(ov["text"], (x, y - h / 2 - 4), size, ACCENT, "bold", anchor="ma", alpha=a)


OVERLAYS = {"rc_c2": o_rc_c2, "hat_closeup": o_hat_closeup, "altitude": o_altitude, "drone_mark": o_drone_mark,
            "spread_gauge": o_spread_gauge, "log_point": o_log_point, "building_mark": o_building_mark,
            "badge": o_badge}

VISUALS = {
    "screens_around_you": v_screens_around_you,
    "interface_ring": v_interface_ring,
    "deform_gap": v_deform_gap,
    "split_building": v_split_building,
    "look_down_streets": v_look_down_streets,
    "adaptive_spread": v_adaptive_spread,
    "controls": v_controls,
    "practice_map": v_practice_map,
    "surround_search": v_surround_search,
    "hats": v_hats,
    "clip_grid": v_clip_grid,
    "trial_map": v_trial_map,
}
