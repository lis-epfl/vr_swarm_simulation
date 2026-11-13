"""
Carry an alignment forward with the images instead of the telemetry.

``planar_align`` solves where every view belongs at the instant of its snapshot, to a
pixel or two.  The mosaic is drawn many times between two such solves, and until this
module existed each of those frames was placed by that frame's *published pose* plus
the correction -- which assumes the pose published with a frame describes that frame.
On the real fleet it does not, and not by a constant:

- telemetry is sampled at ~5 Hz and held (``planar_pose_filter`` deals with that part);
- video latency differs per aircraft and wanders with link quality.  In the recorded
  clips one drone's picture ran about a second behind the others' whenever its link was
  weak, so while the formation translated at 1.5 m/s its view sat 1.5 m from where its
  pose said, and returned when the formation stopped.  No bias that lasts seconds can
  follow that, and the seam opened to 10 px for as long as the drones moved.

A static scene does not care *when* a picture was taken, only what it shows.  So each
view's own frame-to-frame image motion is measured (pyramidal Lucas-Kanade, a robust
affine fit), and a view is drawn at

    G(now) = A(anchor -> now) . G(anchor)

where ``G(anchor)`` is the aligner's mapping for the very frame it solved on and ``A``
is the product of the inter-frame motions since.  The telemetry's timing drops out
entirely between anchors; drift is bounded by the anchor period (a few frames) and is
reset by every solve.

This is the same structure MViS optimises offline -- inter-view transforms tie the views
to each other, temporal transforms tie each view to its own past -- with the temporal
half measured directly rather than fitted.

An affine, not a homography, between consecutive frames: over 50 ms the perspective
terms of a camera's motion are far below what a hundred tracked points can measure, and
estimating them anyway feeds noise into a product of ten matrices.

It is the *plane's* motion that has to be measured, not the picture's.  When a camera
translates, everything nearer than the plane moves faster across the image and
everything behind it slower; a robust fit to all the tracked points follows whichever
there are most of, and over a street the building sides outnumber the road.  The poses
are good enough to say what direction and roughly how far a point on the plane should
have moved between two frames.  Each tracked point is therefore expressed as a multiple
``lambda`` of its own predicted motion: points on the plane share one value (not
necessarily 1 -- the telemetry's timing is what cannot be trusted), points off it are
spread out by their depth, and the plane is the sharpest peak of that distribution.
Only the points in it are fitted.  A pure rotation moves every depth alike, puts
everything in one peak, and loses nothing.
"""

import threading
from collections import deque
from concurrent.futures import ThreadPoolExecutor

import numpy as np

try:
    import cv2
except ImportError:                                     # pragma: no cover
    cv2 = None

# Tracking runs on a reduced copy of each frame: the motion between consecutive frames is
# a few pixels, and half resolution measures it to a tenth of one at a quarter of the cost.
TRACK_SCALE = 0.5
MAX_CORNERS = 160
MIN_TRACKED = 24
MIN_INLIERS = 16
MIN_INLIER_RATIO = 0.4
RANSAC_PX = 1.0                # at the tracking scale
# Plane selection (see the module docstring). Below this much predicted motion, in pixels
# at the tracking scale, depth cannot be told apart and every point is kept.
SELECT_MIN_MOTION_PX = 0.4
SELECT_BIN = 0.1               # histogram bin of lambda
SELECT_BAND = 0.12             # relative half-width kept around the peak
# Frames of history kept per drone: enough to bridge the aligner's own latency (a pass
# takes 0.2-0.5 s) several times over.
HISTORY = 96


def available():
    return cv2 is not None


class ViewTracker:
    """Per-drone inter-frame image motion, and chains of it between two capture times."""

    def __init__(self, workers=6):
        self._lock = threading.Lock()
        self._state = {}       # drone_id -> [last time, last grey, deque[(time, F, ok)]]
        self._pool = ThreadPoolExecutor(max_workers=workers) if cv2 is not None else None
        scale = TRACK_SCALE
        # cv2.resize maps full-resolution pixel x to scale * x + (scale - 1) / 2.
        self._to_small = np.array([[scale, 0.0, (scale - 1.0) / 2.0],
                                   [0.0, scale, (scale - 1.0) / 2.0],
                                   [0.0, 0.0, 1.0]])
        self._to_full = np.linalg.inv(self._to_small)

    # -- per frame -------------------------------------------------------------------

    def step(self, views, predicted=None):
        """
        Measure the motion of every view that has a new frame.  Render thread.

        ``predicted`` maps a drone id to the 3x3 the poses say takes a plane point's
        pixel in that drone's previous frame to its pixel in this one, where known.
        """
        if cv2 is None:
            return
        jobs = []
        for v in views:
            t = float(v.get("capture_time", 0.0) or 0.0)
            if t <= 0.0 or v.get("image") is None:
                continue
            state = self._state.get(v["drone_id"])
            if state is not None and t <= state[0]:
                continue                                   # the same frame re-served
            jobs.append((v["drone_id"], t, v["image"], state,
                         None if predicted is None else predicted.get(v["drone_id"])))
        if not jobs:
            return
        results = list(self._pool.map(self._track_one, jobs)) if len(jobs) > 1 \
            else [self._track_one(jobs[0])]
        with self._lock:
            for (did, t, _img, state, _pred), (grey, F, ok) in zip(jobs, results):
                if state is None or t - state[0] > 1.0:
                    # First frame, or a stream that stalled: nothing to chain through.
                    ring = deque(maxlen=HISTORY)
                    ring.append((t, np.eye(3), False))
                    self._state[did] = [t, grey, ring]
                else:
                    state[0], state[1] = t, grey
                    state[2].append((t, F, ok))

    def _on_plane(self, p0, p1, predicted):
        """Mask of tracked points moving as the plane does, or ``None`` to keep all."""
        small = self._to_small @ predicted @ self._to_full
        q = np.column_stack([p0, np.ones(len(p0))]) @ small.T
        u = q[:, :2] / q[:, 2:3] - p0                      # predicted motion per point
        f = p1 - p0
        mag2 = (u * u).sum(axis=1)
        if float(np.median(np.sqrt(mag2))) < SELECT_MIN_MOTION_PX:
            return None
        lam = (f * u).sum(axis=1) / np.maximum(mag2, 1e-9)
        across = np.linalg.norm(f - lam[:, None] * u, axis=1)
        along = across < 0.6 + 0.25 * np.abs(lam) * np.sqrt(mag2)
        if int(along.sum()) < MIN_TRACKED:
            return None
        hist, edges = np.histogram(lam[along], bins=np.arange(-1.0, 5.0 + SELECT_BIN, SELECT_BIN))
        smooth = np.convolve(hist, [1.0, 2.0, 1.0], mode="same")
        k = int(np.argmax(smooth))
        peak = 0.5 * (edges[k] + edges[k + 1])
        keep = along & (np.abs(lam - peak) <= SELECT_BAND * max(abs(peak), 0.5) + 1.5 * SELECT_BIN)
        return keep if int(keep.sum()) >= MIN_TRACKED else None

    def _track_one(self, job):
        _did, _t, image, state, predicted = job
        grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        grey = cv2.resize(grey, None, fx=TRACK_SCALE, fy=TRACK_SCALE,
                          interpolation=cv2.INTER_AREA)
        if state is None or state[1] is None or state[1].shape != grey.shape:
            return grey, np.eye(3), False
        prev = state[1]
        p0 = cv2.goodFeaturesToTrack(prev, MAX_CORNERS, 0.01, 8)
        if p0 is None or len(p0) < MIN_TRACKED:
            return grey, np.eye(3), False
        p1, status, _err = cv2.calcOpticalFlowPyrLK(prev, grey, p0, None,
                                                    winSize=(21, 21), maxLevel=3)
        good = status.ravel() == 1
        if int(good.sum()) < MIN_TRACKED:
            return grey, np.eye(3), False
        a0, a1 = p0.reshape(-1, 2)[good], p1.reshape(-1, 2)[good]
        if predicted is not None:
            keep = self._on_plane(a0, a1, predicted)
            if keep is not None:
                a0, a1 = a0[keep], a1[keep]
        a, inl = cv2.estimateAffine2D(a0, a1, method=cv2.RANSAC,
                                      ransacReprojThreshold=RANSAC_PX)
        if a is None or inl is None:
            return grey, np.eye(3), False
        n_inl = int(inl.sum())
        if n_inl < MIN_INLIERS or n_inl < MIN_INLIER_RATIO * len(a0):
            return grey, np.eye(3), False
        small = np.vstack([a, [0.0, 0.0, 1.0]])
        return grey, self._to_full @ small @ self._to_small, True

    # -- chains ----------------------------------------------------------------------

    def chain(self, drone_id, t_from):
        """
        The map from pixels of the frame captured at ``t_from`` to pixels of the newest
        frame of that drone, as a 3x3, or ``None`` when it cannot be vouched for: the
        frame is no longer in the history, or a step in between failed to track.
        Either thread.
        """
        with self._lock:
            state = self._state.get(drone_id)
            if state is None:
                return None
            ring = list(state[2])
        if not ring or t_from > ring[-1][0] + 1e-6:
            return None
        A = np.eye(3)
        found = False
        for t, F, ok in ring:
            if abs(t - t_from) <= 1e-6:
                found = True
                A = np.eye(3)
                continue
            if found:
                if not ok:
                    return None
                A = F @ A
        return A if found else None

    def between(self, drone_id, t_from, t_to):
        """As :meth:`chain`, but ending at the frame captured at ``t_to``."""
        with self._lock:
            state = self._state.get(drone_id)
            if state is None:
                return None
            ring = list(state[2])
        if t_to < t_from - 1e-6:
            return None
        A = np.eye(3)
        found = False
        for t, F, ok in ring:
            if abs(t - t_from) <= 1e-6:
                found = True
                A = np.eye(3)
                if abs(t - t_to) <= 1e-6:
                    return A
                continue
            if found:
                if not ok:
                    return None
                A = F @ A
                if abs(t - t_to) <= 1e-6:
                    return A
        return None

    def newest(self, drone_id):
        with self._lock:
            state = self._state.get(drone_id)
            return None if state is None else state[0]
