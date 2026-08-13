"""
Plane-supported correspondence front-end for the PLANAR stitcher.

PLANAR only has to align *the plane* -- the ground in NADIR, the building face in
FACADE.  Sky, cranes and background buildings may be misaligned across seams; they are
not part of the objective.  Nothing in the pose-only backbone or in the existing
estimators encodes that, and on a real facade it is the difference between converging
and converging onto the wrong surface.

WHY DEPTH IS THE DISCRIMINATOR, AND TEXTURE IS NOT
--------------------------------------------------
Warped by the plane homography, a point at depth ``Z`` is displaced between two views by
``B * |1 - Z_p/Z|``.  On the MED facade clip (``B = 10.85 m``, ``Z_p = 34.255 m``,
``f = 525 px``) that is 10.85 m of apparent displacement -- 217 canvas pixels at
0.05 m/px, 166 source pixels -- for a point at infinity, and about 7 m for the crane.
The pose error being corrected is 1-5 m.  So off-plane content is not a perturbation on
the measurement: it is a large, *internally self-consistent* rival solution.

Two things follow, and the second is the whole reason this module exists:

- Weighting the estimator by gradient energy or saliency makes it worse.  The crane and
  the roofline are the highest-contrast things in frame; empty sky is the harmless case,
  because it only dilutes.  Texture weighting promotes exactly the content that must be
  ignored.
- RANSAC over a *canvas-translation* model can lock onto the background, because RANSAC
  keeps the largest consistent set and the background is consistent with itself.

So the consensus is taken **in depth** -- over "which surface is this?" -- and only then
is anything inferred about pose.  A single 3-drone baseline is enough to separate the
facade from everything behind it, because the separation is metres and the required
resolution is decimetres.

WHAT COMES OUT
--------------
``plane_support`` returns, from one pass of matching and triangulation:

- ``offset``   the dominant surface's perpendicular offset from the assumed plane, in
               metres.  This *is* the plane correction (the sweep's ``_correction
               ["plane"]`` quantity), measured directly rather than scanned for.
- ``inliers``  the matches lying on that surface, which are the only ones any pose solve
               should see.
- ``residual`` the per-match disagreement between the two views, in metres on the plane
               -- the seam error, restricted to plane-supported content.

Triangulated depth is good to roughly ``Z^2 * sigma_px / (f * B)``: about 0.2 m at 30 m
for these baselines, which is well inside the plane sweep's fine-scan basin.  That is
what lets the offset seed the sweep instead of ACQUIRE having to find it photometrically.

The band that defines "on the plane" is derived, not tuned.  A depth deviation ``dZ``
shows up as ``B * dZ / Z`` of plane displacement, so tolerating ``tol`` metres of seam
means keeping ``|dZ| <= tol * Z / B`` -- the same ``Z/B`` that sets the sweep's basin.
On the MED clip that is +/-3.2 m for +/-1 m, which keeps the louvre panels (they project
0.5-1 m and *are* the facade) and drops everything behind the roofline.
"""

import numpy as np

import planar_geometry as pg

try:
    import cv2
except ImportError:                                    # pragma: no cover
    cv2 = None


# ORB rather than SuperPoint, deliberately. MGRAPH's front-end is ORB + BF-Hamming +
# ratio filter + RANSAC and this is the same job; SuperPoint costs a model download,
# which would put a network dependency on tools/planar_selftest.py and on the bench.
# BaseStitcher's SuperPoint is still the better detector on low-contrast facades and
# slots in behind `detector=` without anything else changing.
ORB_FEATURES = 3000

# Mutual nearest neighbour + Lowe ratio. Both, not either: on a periodic facade the
# ratio test alone rejects almost everything (every louvre panel has a near-identical
# rival), and crossCheck alone accepts confident nonsense between two different panels.
LOWE_RATIO = 0.80

# A match displaced further than this from its pose prediction is not a measurement of
# pose error, it is a mismatch. Generous, because the whole point is to survive an error
# the dense refiner's 3 m gate cannot: this bounds the *search*, while the depth
# consensus below does the actual rejecting.
DEFAULT_SEARCH_RADIUS_M = 8.0

# Below this many inliers the surface estimate is not a consensus, it is a coincidence.
# SkyEye Fig. 5 reports 38 -> 24 -> 18 inliers at 0 / 10 / 20 deg of pose error on a
# comparable setup, so a floor in the mid-teens is where a real measurement still lives.
MIN_SUPPORT = 16


def _as_gray_u8(img):
    """Canvas warp (float tensor, numpy, colour or not) -> contiguous uint8 grey."""
    a = img.detach().cpu().numpy() if hasattr(img, "detach") else np.asarray(img)
    a = np.squeeze(a)
    if a.ndim == 3:
        a = a.mean(axis=2) if a.shape[2] <= 4 else a.mean(axis=0)
    if a.dtype != np.uint8:
        finite = np.isfinite(a)
        hi = float(a[finite].max()) if finite.any() else 0.0
        a = np.clip(a * (255.0 if hi <= 1.001 else 1.0), 0.0, 255.0)
    return np.ascontiguousarray(a.astype(np.uint8))


def _as_mask_u8(mask, shape):
    if mask is None:
        return np.full(shape, 255, np.uint8)
    m = mask.detach().cpu().numpy() if hasattr(mask, "detach") else np.asarray(mask)
    return np.ascontiguousarray((np.squeeze(m) > 0).astype(np.uint8) * 255)


def match_canvas_pair(warp_a, warp_b, mask_a=None, mask_b=None,
                      search_radius_px=np.inf, detector=None):
    """
    Correspondences between two views already warped into the *same* canvas frame.

    Matching after the warp rather than between raw frames is what makes a cheap
    detector sufficient: the pose prior has removed the viewpoint change, so residual
    disparity is tens of pixels rather than hundreds and a plain radius gate does most
    of the outlier rejection for free.  That is MGRAPH's "GPS prunes the candidates",
    reduced to a search radius because we have metric pose rather than only position.

    Returns ``(pts_a, pts_b)`` as float arrays of canvas pixel coordinates, ``[M, 2]``.
    """
    if cv2 is None:
        return np.zeros((0, 2)), np.zeros((0, 2))

    ga, gb = _as_gray_u8(warp_a), _as_gray_u8(warp_b)
    ma, mb = _as_mask_u8(mask_a, ga.shape), _as_mask_u8(mask_b, gb.shape)

    det = detector if detector is not None else cv2.ORB_create(nfeatures=ORB_FEATURES)
    ka, da = det.detectAndCompute(ga, ma)
    kb, db = det.detectAndCompute(gb, mb)
    if da is None or db is None or len(ka) < 2 or len(kb) < 2:
        return np.zeros((0, 2)), np.zeros((0, 2))

    norm = cv2.NORM_HAMMING if da.dtype == np.uint8 else cv2.NORM_L2
    pairs = cv2.BFMatcher(norm).knnMatch(da, db, k=2)

    pa, pb = [], []
    for group in pairs:
        if len(group) < 2:
            continue
        best, rival = group[0], group[1]
        if best.distance > LOWE_RATIO * rival.distance:
            continue
        x_a = np.asarray(ka[best.queryIdx].pt, dtype=np.float64)
        x_b = np.asarray(kb[best.trainIdx].pt, dtype=np.float64)
        if np.linalg.norm(x_a - x_b) > search_radius_px:
            continue
        pa.append(x_a)
        pb.append(x_b)

    if not pa:
        return np.zeros((0, 2)), np.zeros((0, 2))
    return np.asarray(pa), np.asarray(pb)


def canvas_to_source(pts_xy, G, M):
    """Canvas pixels -> source-image pixels through ``H = G M``, dropping points behind."""
    H = pg.homography_canvas_to_image(G, M)
    xy1 = np.column_stack([pts_xy, np.ones(len(pts_xy))])
    uvw = xy1 @ H.T
    w = uvw[:, 2]
    ok = w > 1e-6
    uv = np.full((len(pts_xy), 2), np.nan)
    uv[ok] = uvw[ok, :2] / w[ok, None]
    return uv, ok


def pixel_rays(uv, K, R_cv, C):
    """
    Unit world rays through source pixels.

    Rows of a world->camera rotation are the camera axes in world, so ``R.T`` takes a
    camera-space direction back to world -- the same convention ``_build_geometry`` uses
    when it reads ``R[2]`` as the forward axis.
    """
    K_inv = np.linalg.inv(np.asarray(K, dtype=np.float64))
    d_cam = np.column_stack([uv, np.ones(len(uv))]) @ K_inv.T
    d_world = d_cam @ np.asarray(R_cv, dtype=np.float64)      # == (R.T @ d_cam.T).T
    n = np.linalg.norm(d_world, axis=1, keepdims=True)
    return np.divide(d_world, np.maximum(n, 1e-12)), np.asarray(C, dtype=np.float64)


def triangulate(C1, d1, C2, d2):
    """
    Midpoint of the common perpendicular of two ray bundles.  ``[M, 3]``.

    Near-parallel rays give a badly conditioned midpoint, so they are returned as NaN
    rather than as a large number: a NaN is dropped by the consensus below, whereas a
    finite garbage depth would be voted on.
    """
    w0 = np.asarray(C1, dtype=np.float64) - np.asarray(C2, dtype=np.float64)
    b = np.einsum("ij,ij->i", d1, d2)
    d = d1 @ w0
    e = d2 @ w0
    denom = 1.0 - b * b                                  # d1, d2 are unit length
    out = np.full((len(d1), 3), np.nan)
    ok = np.abs(denom) > 1e-8
    if not ok.any():
        return out
    t1 = (b[ok] * e[ok] - d[ok]) / denom[ok]
    t2 = (e[ok] - b[ok] * d[ok]) / denom[ok]
    p1 = C1 + t1[:, None] * d1[ok]
    p2 = C2 + t2[:, None] * d2[ok]
    # Both parameters must be in front of their camera; a negative one means the rays
    # cross behind a lens, which is a mismatch rather than a distant surface.
    good = (t1 > 0.0) & (t2 > 0.0)
    idx = np.nonzero(ok)[0][good]
    out[idx] = 0.5 * (p1[good] + p2[good])
    return out


def dominant_surface(offsets, band_m):
    """
    The largest set of triangulated points agreeing on one perpendicular offset.

    A sliding window of width ``2 * band_m`` over the sorted offsets, which is the exact
    maximum-consensus answer in one dimension -- deterministic, O(n log n), and with no
    sample count to tune.  RANSAC would compute the same thing by guessing.

    Returns ``(offset, inlier_mask)``; ``(nan, all-False)`` when nothing is left.
    """
    offsets = np.asarray(offsets, dtype=np.float64)
    finite = np.isfinite(offsets)
    mask = np.zeros(len(offsets), dtype=bool)
    if finite.sum() == 0:
        return float("nan"), mask

    idx = np.nonzero(finite)[0]
    vals = offsets[idx]
    order = np.argsort(vals)
    s = vals[order]

    width = 2.0 * float(band_m)
    hi = np.searchsorted(s, s + width, side="right")
    counts = hi - np.arange(len(s))
    start = int(np.argmax(counts))
    win = order[start:hi[start]]

    # Median of the winning window, not its centre: the window is a capture region, and
    # the surface sits wherever its points actually are inside it.
    centre = float(np.median(vals[win]))
    mask[idx[np.abs(vals - centre) <= band_m]] = True
    return centre, mask


def plane_support(cams, frame, M, warps, valids, K, metres_per_pixel,
                  seam_tolerance_m=1.0, search_radius_m=DEFAULT_SEARCH_RADIUS_M,
                  detector=None):
    """
    Segment plane from non-plane across every overlapping view pair, and measure both.

    ``cams``   the list ``PlanarStitcher._build_geometry`` returns (needs ``G``/``R``/``C``)
    ``warps``  each view rendered into the canvas, same order as ``cams``
    ``valids`` per-view coverage masks, or ``None``

    Returns a dict with ``offset`` (metres, the dominant surface's perpendicular
    displacement from the assumed plane), ``support`` / ``total`` (inlier and match
    counts), ``residual_m`` (plane-supported seam error per match) and ``pairs`` (the
    per-pair records a solve consumes).
    """
    n_views = len(cams)
    band_m = _support_band(cams, seam_tolerance_m)
    radius_px = float(search_radius_m) / max(float(metres_per_pixel), 1e-9)

    pairs, all_off, all_res = [], [], []
    for i in range(n_views):
        for j in range(i + 1, n_views):
            rec = _pair_support(cams[i], cams[j], i, j, frame, M, K,
                                warps[i], warps[j],
                                None if valids is None else valids[i],
                                None if valids is None else valids[j],
                                metres_per_pixel, radius_px, detector)
            if rec is None:
                continue
            pairs.append(rec)
            all_off.append(rec["offset_m"])
            all_res.append(rec["residual_m"])

    if not pairs:
        return {"offset": float("nan"), "support": 0, "total": 0,
                "residual_m": np.zeros(0), "pairs": [], "band_m": band_m}

    offsets = np.concatenate(all_off)
    # One consensus over every pair at once, not a vote of per-pair answers: the surface
    # is global, and pooling keeps a pair with thin overlap from carrying a whole vote.
    offset, inliers = dominant_surface(offsets, band_m)

    at = 0
    for rec in pairs:
        k = len(rec["offset_m"])
        rec["inliers"] = inliers[at:at + k]
        at += k

    residual = np.concatenate(all_res)
    return {"offset": offset,
            "support": int(inliers.sum()),
            "total": int(len(offsets)),
            "residual_m": residual[inliers],
            "residual_off_plane_m": residual[~inliers],
            "pairs": pairs,
            "band_m": band_m}


def _support_band(cams, seam_tolerance_m):
    """``dZ = tol * Z / B`` from the live formation -- see the module docstring."""
    if len(cams) < 2:
        return max(float(seam_tolerance_m), 1e-3)
    ab = np.asarray([c["plane_ab"] for c in cams], dtype=np.float64)
    baseline = float(np.max(np.linalg.norm(ab[:, None, :] - ab[None, :, :], axis=-1)))
    standoff = float(np.median([c["plane_h"] for c in cams]))
    if baseline < 1e-6:
        return max(float(seam_tolerance_m), 1e-3)
    return max(float(seam_tolerance_m) * standoff / baseline, 1e-3)


def _pair_support(cam_i, cam_j, i, j, frame, M, K, warp_i, warp_j,
                  valid_i, valid_j, metres_per_pixel, radius_px, detector):
    pts_i, pts_j = match_canvas_pair(warp_i, warp_j, valid_i, valid_j,
                                     radius_px, detector)
    if len(pts_i) < 3:
        return None

    uv_i, ok_i = canvas_to_source(pts_i, cam_i["G"], M)
    uv_j, ok_j = canvas_to_source(pts_j, cam_j["G"], M)
    keep = ok_i & ok_j
    if keep.sum() < 3:
        return None
    pts_i, pts_j, uv_i, uv_j = pts_i[keep], pts_j[keep], uv_i[keep], uv_j[keep]

    d_i, C_i = pixel_rays(uv_i, K, cam_i["R"], cam_i["C"])
    d_j, C_j = pixel_rays(uv_j, K, cam_j["R"], cam_j["C"])
    X = triangulate(C_i, d_i, C_j, d_j)

    # Perpendicular displacement from the assumed plane, signed. On-plane content sits
    # at whatever the plane error is; everything behind the facade sits further out.
    # This is already the plane correction's units and sign convention, so the consensus
    # value can be handed straight to the sweep.
    offset = X @ frame.n - frame.d

    # The seam: how far apart the two views put the same physical point, on the plane.
    residual = np.linalg.norm(pts_i - pts_j, axis=1) * float(metres_per_pixel)

    return {"i": i, "j": j,
            "drone_i": cam_i["view"]["drone_id"], "drone_j": cam_j["view"]["drone_id"],
            "pts_i": pts_i, "pts_j": pts_j, "uv_i": uv_i, "uv_j": uv_j,
            "world": X, "offset_m": offset, "residual_m": residual,
            "inliers": np.ones(len(offset), dtype=bool)}
