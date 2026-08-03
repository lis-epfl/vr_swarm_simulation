"""
Pose-driven planar stitching geometry.

Pure numpy: no Unity, no torch, no shared memory.  Everything here is a plain
function of numbers so it can be unit-tested offline (``tools/planar_selftest.py``),
which is deliberate -- coordinate-frame errors are the dominant failure mode of a
pose-initialised stitcher and they are far cheaper to catch here than in the headset.

The idea
--------
For a *planar* scene a single homography per view is geometrically **exact**, and it
can be computed in closed form from camera intrinsics + camera pose + the plane, with
no image content at all.  Parameterise the plane by an orthonormal frame ``(e1, e2, n)``
with ``n . O == d``; any plane point is ``X = O + a*e1 + b*e2``, so for camera *i*::

    p ~ K R (X - C) = K R (a*e1 + b*e2 + (O - C))
                    = K R [ e1 | e2 | (O - C) ] (a, b, 1)^T
                    =: G (a, b, 1)^T

That ``G`` is the whole backbone.  It is preferred over the textbook
``K (R - t n^T / d) K^-1`` plane-induced form because it needs no reference view, has
no ``d`` in a denominator, and inverts nothing.

A useful consequence: because ``K``'s last row is ``[0, 0, 1]``, the third component
of ``G (a, b, 1)^T`` is the **camera-space depth in metres**.  So behind-camera
rejection, max-range clipping and the sampling coordinate all fall out of one matmul
(see :func:`homography_canvas_to_image`).

Coordinate frames -- read this before changing anything
-------------------------------------------------------
Unity is **left-handed**: +X right, +Y up, +Z forward.  Computer vision is
**right-handed**: +X right, +Y **down**, +Z forward, with image row 0 at the top.

Rather than conjugating a quaternion by a handedness flip (which is where sign errors
are born), :func:`unity_pose_to_cv` builds the world->camera matrix directly from the
camera's three world-space basis vectors.  Every world quantity that reaches the
geometry -- the plane normal, the plane origin, the in-plane basis -- must be passed
through :func:`unity_dir_to_rh` / :func:`unity_point_to_rh` first.  Those functions and
the explicit row assembly in :func:`unity_pose_to_cv` are the *only* places handedness
appears anywhere in the planar path; everything downstream is textbook CV.
"""

import numpy as np

__all__ = [
    "S_UNITY_TO_RH",
    "unity_dir_to_rh",
    "unity_point_to_rh",
    "quat_to_matrix",
    "unity_pose_to_cv",
    "intrinsics_matrix",
    "build_plane_frame",
    "ray_plane_intersect",
    "build_G",
    "canvas_to_plane_matrix",
    "homography_canvas_to_image",
    "project_points",
    "view_footprint",
    "homography_jacobian",
    "homography_anisotropy",
    "PlaneFrame",
]

# Unity world (LH) -> right-handed world.  Negating Z flips handedness exactly once.
# It is an involution (S == S^-1 == S^T), which is why it can be applied to points,
# directions and plane normals alike without bookkeeping.
S_UNITY_TO_RH = np.diag([1.0, 1.0, -1.0])

_EPS = 1e-9


def unity_dir_to_rh(v):
    """Unity-world direction -> right-handed world direction."""
    return S_UNITY_TO_RH @ np.asarray(v, dtype=np.float64)


def unity_point_to_rh(p):
    """Unity-world point -> right-handed world point.  Same map as directions."""
    return S_UNITY_TO_RH @ np.asarray(p, dtype=np.float64)


def quat_to_matrix(quat_xyzw):
    """
    Unity ``Transform.rotation`` (x, y, z, w) -> 3x3 camera->world rotation, still in
    Unity's left-handed basis.  Columns are the camera's right / up / forward axes.
    """
    x, y, z, w = (float(c) for c in quat_xyzw)
    n = x * x + y * y + z * z + w * w
    if n < _EPS:
        raise ValueError("degenerate quaternion")
    s = 2.0 / n
    xs, ys, zs = x * s, y * s, z * s
    wx, wy, wz = w * xs, w * ys, w * zs
    xx, xy, xz = x * xs, x * ys, x * zs
    yy, yz, zz = y * ys, y * zs, z * zs
    return np.array([
        [1.0 - (yy + zz), xy - wz,         xz + wy],
        [xy + wz,         1.0 - (xx + zz), yz - wx],
        [xz - wy,         yz + wx,         1.0 - (xx + yy)],
    ], dtype=np.float64)


def unity_pose_to_cv(pos_xyz, quat_xyzw):
    """
    Unity camera pose -> ``(R_cv, C)``: the world->camera rotation in the CV convention
    (+X right, +Y down, +Z forward) and the camera centre, both in right-handed world.

    Built explicitly from the camera's world-space basis vectors so the convention is
    readable rather than inferred:  the rows of a world->camera rotation *are* the
    camera axes expressed in world coordinates.
    """
    R_u = quat_to_matrix(quat_xyzw)
    right = unity_dir_to_rh(R_u[:, 0])
    up = unity_dir_to_rh(R_u[:, 1])
    forward = unity_dir_to_rh(R_u[:, 2])

    # CV camera axes: x = right, y = down = -up, z = forward.  As rows -> world->camera.
    R_cv = np.stack([right, -up, forward], axis=0)
    C = unity_point_to_rh(pos_xyz)
    return R_cv, C


def intrinsics_matrix(fx, fy, cx, cy):
    """Pinhole K.  Row 0 is x/right, row 1 is y/down (image row 0 = top)."""
    return np.array([[fx, 0.0, cx],
                     [0.0, fy, cy],
                     [0.0, 0.0, 1.0]], dtype=np.float64)


class PlaneFrame:
    """
    An orthonormal frame on the scene plane: origin ``O`` and in-plane axes
    ``e1`` (canvas +x) and ``e2`` (canvas +y "up"), with ``cross(e1, e2) == n``.
    All vectors are right-handed world.
    """

    __slots__ = ("n", "d", "O", "e1", "e2")

    def __init__(self, n, d, O, e1, e2):
        self.n, self.d, self.O, self.e1, self.e2 = n, d, O, e1, e2

    def to_plane_coords(self, X):
        """World point(s) -> in-plane ``(a, b)``.  Accepts (3,) or (..., 3)."""
        rel = np.asarray(X, dtype=np.float64) - self.O
        return np.stack([rel @ self.e1, rel @ self.e2], axis=-1)

    def to_world(self, ab):
        """In-plane ``(a, b)`` -> world point."""
        ab = np.asarray(ab, dtype=np.float64)
        return self.O + ab[..., 0, None] * self.e1 + ab[..., 1, None] * self.e2

    def __repr__(self):
        return (f"PlaneFrame(n={np.round(self.n, 4)}, d={self.d:.4f}, "
                f"O={np.round(self.O, 3)})")


def _normalize(v):
    v = np.asarray(v, dtype=np.float64)
    m = np.linalg.norm(v)
    if m < _EPS:
        raise ValueError("cannot normalize a zero-length vector")
    return v / m


def build_plane_frame(n, d, O, ref_right, ref_up=None):
    """
    Build the canvas frame on the plane ``{X : n.X == d}``.

    ``ref_right`` is the reference camera's world right axis; its in-plane component
    becomes ``e1``, so "right in the panorama" == "right in that camera's view".  This
    is the centre-drone-locked canvas: the mosaic stays aligned with the view the pilot
    is nominally looking through.

    ``e2 = cross(n, e1)`` is right-handed by construction (``cross(e1, e2) == n``), so
    no sign fix is needed.  ``ref_up``, when given, is only checked -- a negative dot
    means the reference camera is upside down relative to the plane, which is worth
    knowing about but is not corrected here.
    """
    n = _normalize(n)
    O = np.asarray(O, dtype=np.float64)
    ref_right = np.asarray(ref_right, dtype=np.float64)

    e1 = ref_right - np.dot(ref_right, n) * n
    if np.linalg.norm(e1) < 1e-6:
        # Reference right axis is parallel to the normal (camera rolled onto its side
        # relative to the plane).  Any in-plane direction is as good as any other.
        fallback = np.array([1.0, 0.0, 0.0])
        if abs(np.dot(fallback, n)) > 0.9:
            fallback = np.array([0.0, 1.0, 0.0])
        e1 = fallback - np.dot(fallback, n) * n
    e1 = _normalize(e1)
    e2 = np.cross(n, e1)

    if ref_up is not None:
        up_in_plane = np.asarray(ref_up, dtype=np.float64)
        up_in_plane = up_in_plane - np.dot(up_in_plane, n) * n
        if np.linalg.norm(up_in_plane) > 1e-6 and np.dot(e2, up_in_plane) < 0.0:
            # Not corrected on purpose: flipping e2 alone would make (e1, e2, n)
            # left-handed and mirror the panorama.  Callers that care should pass a
            # reference frame that is the right way up.
            pass

    return PlaneFrame(n=n, d=float(d), O=O, e1=e1, e2=e2)


def ray_plane_intersect(origin, direction, n, d, max_range=np.inf):
    """
    Intersect a ray with ``{X : n.X == d}``.

    Returns ``(point, t)`` where ``t`` is the distance along ``direction`` (which must
    be unit length for ``t`` to be metres).  Returns ``(None, nan)`` when the ray is
    parallel to the plane, points away from it, or hits beyond ``max_range`` -- the
    caller is expected to drop the view rather than clamp, since a clamped intersection
    is geometrically meaningless.
    """
    origin = np.asarray(origin, dtype=np.float64)
    direction = np.asarray(direction, dtype=np.float64)
    denom = float(np.dot(n, direction))
    if abs(denom) < 1e-9:
        return None, float("nan")
    t = (float(d) - float(np.dot(n, origin))) / denom
    if t <= 0.0 or t > max_range:
        return None, float("nan")
    return origin + t * direction, t


def build_G(K, R_cv, C, frame):
    """
    ``G`` maps plane coordinates ``(a, b, 1)`` to image pixels (homogeneous).

    ``G = K R [ e1 | e2 | (O - C) ]``.  The third output component is camera-space
    depth in metres.
    """
    B = np.column_stack([frame.e1, frame.e2, frame.O - np.asarray(C, dtype=np.float64)])
    return K @ R_cv @ B


def canvas_to_plane_matrix(metres_per_pixel, canvas_cx, canvas_cy):
    """
    ``M`` maps canvas pixels ``(x, y, 1)`` to plane coordinates ``(a, b, 1)``.

    The ``-s`` on the y row is not cosmetic: canvas rows grow downward while ``e2``
    points "up" on the plane, so without it the panorama comes out vertically mirrored.
    """
    s = float(metres_per_pixel)
    return np.array([[s,  0.0, -float(canvas_cx) * s],
                     [0.0, -s,  float(canvas_cy) * s],
                     [0.0, 0.0, 1.0]], dtype=np.float64)


def homography_canvas_to_image(G, M):
    """
    ``H = G M`` maps canvas pixels straight to source-image pixels -- the *inverse*
    warp, which is exactly the direction a resampler wants.  Third component remains
    camera depth in metres, so one matmul yields the sampling coordinate and every
    rejection test at once.
    """
    return np.asarray(G, dtype=np.float64) @ np.asarray(M, dtype=np.float64)


def project_points(K, R_cv, C, X_world):
    """
    Project right-handed world point(s) to pixels.  Returns ``(uv, depth)`` with
    ``uv`` shaped ``(..., 2)`` and ``depth`` in metres (negative == behind camera).
    """
    X = np.asarray(X_world, dtype=np.float64)
    cam = (X - np.asarray(C, dtype=np.float64)) @ np.asarray(R_cv, dtype=np.float64).T
    p = cam @ np.asarray(K, dtype=np.float64).T
    depth = p[..., 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        uv = p[..., :2] / depth[..., None]
    return uv, depth


def view_footprint(K, R_cv, C, frame, width, height, max_range=np.inf):
    """
    Back-project the image outline onto the plane and return its plane coordinates.

    Samples the 4 corners *and* the 4 edge midpoints: for an oblique view the corners
    can fall behind the camera or beyond the horizon while the centre of the frame
    still lands on the plane, and corners-only would discard a perfectly usable view.
    Points that miss the plane are dropped, so the result has between 0 and 8 rows.
    """
    K = np.asarray(K, dtype=np.float64)
    R_cv = np.asarray(R_cv, dtype=np.float64)
    C = np.asarray(C, dtype=np.float64)
    w, h = float(width) - 1.0, float(height) - 1.0

    pixels = [(0.0, 0.0), (w, 0.0), (w, h), (0.0, h),
              (w * 0.5, 0.0), (w, h * 0.5), (w * 0.5, h), (0.0, h * 0.5)]

    K_inv = np.linalg.inv(K)
    R_wc = R_cv.T  # camera -> world
    out = []
    for u, v in pixels:
        ray_cam = K_inv @ np.array([u, v, 1.0])
        direction = R_wc @ ray_cam
        norm = np.linalg.norm(direction)
        if norm < _EPS:
            continue
        # ray_cam has unit z, so normalising the direction makes t a true distance;
        # the depth cap is applied on the *along-axis* depth, not the slant range.
        point, t = ray_plane_intersect(C, direction / norm, frame.n, frame.d)
        if point is None:
            continue
        depth = t * (ray_cam[2] / norm)
        if depth <= 0.0 or depth > max_range:
            continue
        out.append(frame.to_plane_coords(point))

    return np.array(out, dtype=np.float64) if out else np.zeros((0, 2), dtype=np.float64)


def homography_jacobian(H, x, y):
    """
    Local 2x2 Jacobian of the projective map at canvas pixel ``(x, y)``.

    For ``uv = (H p)_xy / (H p)_w`` the derivative is
    ``(A - uv (h20, h21)) / w``, where ``A`` is the upper-left 2x2.  Returns
    ``(J, w)``; ``w`` is the camera depth, so ``w <= 0`` means the point is behind
    the camera and ``J`` is meaningless.
    """
    H = np.asarray(H, dtype=np.float64)
    p = np.array([float(x), float(y), 1.0])
    num = H[:2] @ p
    w = float(H[2] @ p)
    if abs(w) < 1e-12:
        return np.full((2, 2), np.inf), w
    uv = num / w
    return (H[:2, :2] - np.outer(uv, H[2, :2])) / w, w


def homography_anisotropy(H, width, height, samples=None):
    """
    Worst local stretch ratio (``sigma_max / sigma_min`` of the Jacobian) over the canvas.

    This is the projective-sanity gate.  A homography cannot fold the way a TPS mesh
    can, so mesh-distortion metrics are meaningless here -- they would always pass and
    give false confidence.  The pathology that *does* occur is extreme keystone and
    near-horizon stretching, which is exactly anisotropic scaling.

    Deliberately not the SVD condition number of the raw 3x3: that is dominated by the
    metres-per-pixel unit scaling baked into the canvas matrix, so it reads in the tens
    of thousands even for a perfectly benign nadir view.  A local Jacobian is
    dimensionless and reads ~1.0 for any similarity.

    Returns ``inf`` if any sampled point falls behind the camera.
    """
    if samples is None:
        w, h = float(width) - 1.0, float(height) - 1.0
        samples = [(0.0, 0.0), (w, 0.0), (0.0, h), (w, h), (w * 0.5, h * 0.5)]

    worst = 0.0
    for x, y in samples:
        J, depth = homography_jacobian(H, x, y)
        if depth <= 0.0 or not np.all(np.isfinite(J)):
            return float("inf")
        sv = np.linalg.svd(J, compute_uv=False)
        if sv[-1] < 1e-12:
            return float("inf")
        worst = max(worst, float(sv[0] / sv[-1]))
    return worst
