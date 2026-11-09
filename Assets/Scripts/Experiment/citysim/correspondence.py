"""Can the pilot relate the swarm's screens? The view-correspondence measures of the options report
(docs/studies/correspondence_metrics.md), sample by sample, for the split study (S4) and a refined S3.

Under the user's constraints (2026-10-23): every screen is treated alike (no anchor feed), the panorama is out of
scope, shared content counts open ground AND facades, and "a larger swarm" means a wider spread.

  A1  shared content: a pair of ring neighbours is linked when both see >= MIN_M2 of the same open ground (2 m cells,
      within range, clear sight lines) or >= MIN_FACADE_M of the same facade (the same face of the same building in
      FACADE_BIN_M stretches)
  A2  shared building: both see some face of the same building (a looser, landmark link)
  A3  fragmentation from the A1 links round the ring: islands = max(1, unlinked neighbour pairs), orphan screens (no
      linked neighbour), the largest island as a share of the screens shown
  A4  overlap reach: per neighbour pair, the distance at which the two fields of view start to overlap,
      (baseline / 2) / tan((HFOV - heading gap) / 2), against the free distance from their midpoint along their mean
      heading; reachable when the overlap starts first
  B1  viewpoint difference on a shared building: the angle between the two cameras' lines of sight to the parts of
      it each sees
  B2  placement error: for what a screen shows, the angle between its direction from the camera and from the
      formation's centroid; and the order test: the share of pairs of things on different screens whose left-to-right
      order on the ring (display azimuth: the screen's yaw relative to the body yaw, plus the off-axis angle x
      DISPLAY_SCALE) is the reverse of their true order seen from the centroid
  C1  linked coverage: the open ground seen on screens with at least one A1-linked neighbour, beside all of it

Facades are matched by horizontal stretch (the 2-D model: every building taller than the drones), and "shared" means
shared geometry, not recognisably the same: section 9 of the report.
"""
import sys

sys.dont_write_bytecode = True

import math  # noqa: E402

import numpy as np  # noqa: E402

from citysim import camera  # noqa: E402
from citysim.los import wrap_deg  # noqa: E402
from citysim.polyfoot import ConvexFootprints  # noqa: E402
from citysim.transit import seen_from  # noqa: E402

RANGE_M = 40.0
MIN_M2 = 20.0
MIN_FACADE_M = 4.0
FACADE_BIN_M = 2.0
RAY_DEG = 1.0
ORDER_PAIRS = 400            # content pairs sampled per instant for the order test


class Ring:
    """Accumulates the A-C measures over a flight. `fp` must be a ConvexFootprints (its first_hit reports faces); a
    city of rectangles converts with ConvexFootprints.from_rects or blockcity's building polygons."""

    def __init__(self, fp, cells, height, pitch, r=RANGE_M, seed=0):
        if not isinstance(fp, ConvexFootprints):
            raise TypeError("correspondence.Ring needs ConvexFootprints (faces)")
        self.fp, self.cells, self.h, self.pitch, self.r = fp, cells, height, pitch, r
        self.rng = np.random.default_rng(seed)
        self.hmax = math.sqrt(max(r ** 2 - (height - camera.HAT_HEIGHT_M) ** 2, 0.0))
        self.rel = np.arange(-camera.HFOV_DEG / 2, camera.HFOV_DEG / 2 + 1e-9, RAY_DEG)
        self.rows = []
        self.cells_any, self.cells_linked = set(), set()

    def _facades(self, cam, yaw):
        """{(building, edge, bin)} and [(building, hit point)] one camera sees within range."""
        d, b, e = self.fp.first_hit(cam, yaw + self.rel, self.hmax, return_face=True)
        ok = np.isfinite(d) & (b >= 0) & (e >= 0)
        th = np.radians(yaw + self.rel[ok])
        pts = np.asarray(cam, float) + d[ok][:, None] * np.stack([np.sin(th), np.cos(th)], -1)
        stretches, hits = set(), []
        for bi, ei, q in zip(b[ok], e[ok], pts):
            v0, v1 = self.fp.V[bi, ei], self.fp.W[bi, ei]
            along = float(np.dot(q - v0, v1 - v0) / max(np.linalg.norm(v1 - v0), 1e-9))
            stretches.add((int(bi), int(ei), int(along // FACADE_BIN_M)))
            hits.append((int(bi), q))
        return stretches, hits

    def add(self, t, cams, yaws, body_yaw, centroid):
        """One instant: the shown cameras' positions (k, 2) and yaws (k,) deg, the body yaw, the formation centroid."""
        k = len(yaws)
        if k < 2:
            return
        az = wrap_deg(np.asarray(yaws, float) - body_yaw)
        order = np.argsort(az)
        cams = np.asarray(cams, float)[order]
        yaws = np.asarray(yaws, float)[order]
        ground, facs, hits = [], [], []
        for c, y in zip(cams, yaws):
            idx, _, off = seen_from(self.fp, self.cells, c, y, self.h, self.pitch, self.r)
            ground.append((set(idx.tolist()), idx, off))
            f, hh = self._facades(c, y)
            facs.append(f)
            hits.append(hh)
        a1, a2, reach, b1 = [], [], [], []
        cell_m2 = self.cells.step ** 2 if hasattr(self.cells, "step") else 4.0
        for i in range(k):
            j = (i + 1) % k
            g = len(ground[i][0] & ground[j][0]) * cell_m2
            f = len(facs[i] & facs[j]) * FACADE_BIN_M
            a1.append(g >= MIN_M2 or f >= MIN_FACADE_M)
            bi = {b for b, _ in hits[i]}
            bj = {b for b, _ in hits[j]}
            shared = bi & bj
            a2.append(bool(shared))
            if shared:
                ang = []
                for b in shared:
                    qa = np.mean([q for bb, q in hits[i] if bb == b], axis=0) - cams[i]
                    qb = np.mean([q for bb, q in hits[j] if bb == b], axis=0) - cams[j]
                    cosang = float(np.dot(qa, qb) / max(np.linalg.norm(qa) * np.linalg.norm(qb), 1e-9))
                    ang.append(math.degrees(math.acos(max(-1.0, min(1.0, cosang)))))
                b1.append(float(np.mean(ang)))
            gap = abs(float(wrap_deg(yaws[j] - yaws[i])))
            base = float(np.linalg.norm(cams[j] - cams[i]))
            if gap >= camera.HFOV_DEG:
                reach.append(False)
            else:
                d_ov = (base / 2) / math.tan(math.radians((camera.HFOV_DEG - gap) / 2))
                mid = (cams[i] + cams[j]) / 2
                mean_yaw = float(yaws[i] + wrap_deg(yaws[j] - yaws[i]) / 2)
                d_free = float(self.fp.first_hit(mid, [mean_yaw], 500.0)[0]) if self.fp.n else math.inf
                reach.append(d_ov < d_free)
        unlinked = sum(1 for x in a1 if not x)
        islands = max(1, unlinked)
        linked_screen = [a1[i] or a1[(i - 1) % k] for i in range(k)]
        runs, cur = [], 0
        if all(a1):
            largest = k
        else:
            start = a1.index(False) + 1
            for s in range(k):
                i = (start + s) % k
                cur += 1
                if not a1[i]:
                    runs.append(cur)
                    cur = 0
            if cur:
                runs.append(cur)
            largest = max(runs) if runs else 1
        # B2: placement error and the order test, on the ground each screen shows
        pts_all, disp_all, true_all, err = [], [], [], []
        c0 = np.asarray(centroid, float)
        for s, (gset, idx, off) in enumerate(ground):
            if not len(idx):
                continue
            pts = self.cells.pts[idx]
            b_cam = np.degrees(np.arctan2(*(pts - cams[s]).T))
            b_cen = np.degrees(np.arctan2(*(pts - c0).T))
            err.append(np.abs(wrap_deg(b_cen - b_cam)))
            disp = wrap_deg(yaws[s] - body_yaw + off * camera.DISPLAY_SCALE)
            pts_all.append(np.full(len(idx), s))
            disp_all.append(disp)
            true_all.append(wrap_deg(b_cen - body_yaw))
        order_rev = math.nan
        if len(pts_all) >= 2:
            scr, disp, true = np.concatenate(pts_all), np.concatenate(disp_all), np.concatenate(true_all)
            n = len(scr)
            ii = self.rng.integers(n, size=ORDER_PAIRS)
            jj = self.rng.integers(n, size=ORDER_PAIRS)
            keep = scr[ii] != scr[jj]
            if keep.any():
                dd = wrap_deg(disp[ii[keep]] - disp[jj[keep]])
                dt = wrap_deg(true[ii[keep]] - true[jj[keep]])
                order_rev = float((np.sign(dd) != np.sign(dt)).mean())
        for s in range(k):
            self.cells_any |= ground[s][0]
            if linked_screen[s]:
                self.cells_linked |= ground[s][0]
        self.rows.append(dict(
            t=t, screens=k, a1_linked=float(np.mean(a1)), a2_linked=float(np.mean(a2)), whole_ring=bool(all(a1)),
            islands=islands, orphan_share=float(np.mean([not x for x in linked_screen])), largest_island=largest / k,
            a4_reach=float(np.mean(reach)), b1_deg=float(np.mean(b1)) if b1 else math.nan,
            b2_err_deg=float(np.mean(np.concatenate(err))) if err else math.nan, b2_order_rev=order_rev))

    def summary(self, prefix="corr_"):
        if not self.rows:
            return {}
        keys = ("screens", "a1_linked", "a2_linked", "whole_ring", "islands", "orphan_share", "largest_island",
                "a4_reach", "b1_deg", "b2_err_deg", "b2_order_rev")
        out = {}
        for k in keys:
            v = np.array([float(r[k]) for r in self.rows], float)
            out[prefix + k] = round(float(np.nanmean(v)), 4) if np.isfinite(v).any() else None
        out[prefix + "c1_linked_share"] = round(len(self.cells_linked) / max(len(self.cells_any), 1), 4)
        out[prefix + "c1_cells_any"] = len(self.cells_any)
        out[prefix + "c1_cells_linked"] = len(self.cells_linked)
        return out
