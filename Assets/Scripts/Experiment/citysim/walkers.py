"""Goal pedestrians on WalkerPatrol's loops (Assets/Scripts/Interface/WalkerPatrol.cs, Rectangle shape).

Each patrolled building spawns numWalkers walkers on rounded rectangles round its collider's world AABB: walker i
is expand = baseOffset + U(0, margin) * 0.5 off the facade (0.5-2.0 m with the prefab's 0.5 / 3), corner radius
min(cornerRadius, halfW, halfD), a random start along the loop, speed 2 m/s, the first half clockwise (arc length
decreasing). One walker of one patrolled building per goal wears the hat. Positions are the walker's origin on the
ground; the hat is at camera.HAT_HEIGHT_M.
"""
import sys

sys.dont_write_bytecode = True

import math  # noqa: E402

import numpy as np  # noqa: E402

NUM_WALKERS = 8
SPEED = 2.0
BASE_OFFSET = 0.5
MARGIN = 3.0
CORNER_RADIUS = 1.0


class Loop:
    """One walker's path round one building's AABB (cx, cz, hx, hz)."""

    def __init__(self, rect, expand, s0, clockwise, speed=SPEED, corner=CORNER_RADIUS):
        self.cx, self.cz = rect[0], rect[1]
        self.hw, self.hd = rect[2] + expand, rect[3] + expand
        self.r = max(min(corner, self.hw, self.hd), 0.001)
        self.lv, self.lh = 2 * (self.hd - self.r), 2 * (self.hw - self.r)
        self.arc = 0.5 * math.pi * self.r
        self.perimeter = 2 * self.lv + 2 * self.lh + 4 * self.arc
        self.s0, self.sign, self.speed = s0, (-1.0 if clockwise else 1.0), speed

    @property
    def lap_s(self):
        return self.perimeter / self.speed

    def local_xz(self, s):
        """WalkerPatrol.GetLocalXZ, vectorised: offset from the centre at arc length s (counter-clockwise)."""
        s = np.mod(np.asarray(s, float), self.perimeter)
        hw, hd, r, lv, lh, arc = self.hw, self.hd, self.r, self.lv, self.lh, self.arc
        x, z = np.empty_like(s), np.empty_like(s)
        bounds = np.cumsum([0, lv, arc, lh, arc, lv, arc, lh, arc])
        seg = np.clip(np.searchsorted(bounds, s, side="right") - 1, 0, 7)
        u = s - bounds[seg]
        a = u / r
        # 0 right edge +z, 1 top-right corner, 2 top edge -x, 3 top-left, 4 left edge -z, 5 bottom-left,
        # 6 bottom edge +x, 7 bottom-right
        m = seg == 0
        x[m], z[m] = hw, -(hd - r) + u[m]
        m = seg == 1
        x[m], z[m] = (hw - r) + r * np.cos(a[m]), (hd - r) + r * np.sin(a[m])
        m = seg == 2
        x[m], z[m] = (hw - r) - u[m], hd
        m = seg == 3
        x[m], z[m] = -(hw - r) + r * np.cos(0.5 * math.pi + a[m]), (hd - r) + r * np.sin(0.5 * math.pi + a[m])
        m = seg == 4
        x[m], z[m] = -hw, (hd - r) - u[m]
        m = seg == 5
        x[m], z[m] = -(hw - r) + r * np.cos(math.pi + a[m]), -(hd - r) + r * np.sin(math.pi + a[m])
        m = seg == 6
        x[m], z[m] = -(hw - r) + u[m], -hd
        m = seg == 7
        x[m], z[m] = (hw - r) + r * np.cos(1.5 * math.pi + a[m]), -(hd - r) + r * np.sin(1.5 * math.pi + a[m])
        return np.stack([x, z], -1)

    def position(self, t):
        """(len(t), 2) world (x, z) at times t (s)."""
        return self.local_xz(self.s0 + self.sign * self.speed * np.asarray(t, float)) + [self.cx, self.cz]


def patrol(rect, rng, n=NUM_WALKERS, speed=SPEED, base_offset=BASE_OFFSET, margin=MARGIN):
    """WalkerPatrol.Start for one building: its n loops (draw order as in the C#)."""
    loops = []
    for i in range(n):
        expand = base_offset + rng.uniform(0.0, margin) * 0.5
        probe = Loop(rect, expand, 0.0, i < n // 2, speed)
        loops.append(Loop(rect, expand, rng.uniform(0.0, probe.perimeter), i < n // 2, speed))
    return loops


class GoalWalkers:
    """Every walker of one goal and which one is the target (the hat)."""

    def __init__(self, ring_rects, rng, n=NUM_WALKERS, target=None):
        self.loops, self.building = [], []
        for b, rect in enumerate(ring_rects):
            for lp in patrol(rect, rng, n):
                self.loops.append(lp)
                self.building.append(b)
        self.building = np.array(self.building)
        self.target = int(rng.integers(len(self.loops))) if target is None else int(target)

    def positions(self, t):
        """(len(loops), len(t), 2)."""
        return np.stack([lp.position(t) for lp in self.loops])

    def target_positions(self, t):
        return self.loops[self.target].position(t)
