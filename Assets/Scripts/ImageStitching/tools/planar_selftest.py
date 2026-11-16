"""
Offline self-test for the planar stitching geometry.

No Unity, no shared memory, no GPU.  Run it directly::

    cd Assets/Scripts/ImageStitching && python tools/planar_selftest.py

Why this exists
---------------
Coordinate-frame errors are the dominant failure mode of a pose-initialised stitcher,
and they are nearly invisible in a symmetric test scene: a mirrored panorama looks
plausible until you notice the text is backwards.  Catching them here costs seconds;
catching them in the headset costs an afternoon.

The important property of this file is that :func:`unity_reference_project` is an
**independent** implementation.  It derives the projection from Unity's documented
``Camera.worldToCameraMatrix`` (``Matrix4x4.Scale(1,1,-1) * transform.worldToLocalMatrix``)
and ``Camera.projectionMatrix``, and never calls into ``planar_geometry``.  Test 1
cross-checks the two, so a shared sign error cannot cancel out and pass.

The test plane carries an **asymmetric** marker on purpose.  A checkerboard alone is
invariant under exactly the mirrors and 180-degree rotations we are trying to detect.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import planar_geometry as pg  # noqa: E402

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None


# --------------------------------------------------------------------------------------
# Independent reference: Unity's own projection, built from the documented matrices.
# --------------------------------------------------------------------------------------

def unity_projection_matrix(vfov_deg, aspect, near=0.3, far=1000.0):
    """Unity's ``Camera.projectionMatrix`` for a standard perspective camera."""
    t = 1.0 / np.tan(np.radians(vfov_deg) * 0.5)
    return np.array([
        [t / aspect, 0.0, 0.0, 0.0],
        [0.0, t, 0.0, 0.0],
        [0.0, 0.0, -(far + near) / (far - near), -2.0 * far * near / (far - near)],
        [0.0, 0.0, -1.0, 0.0],
    ], dtype=np.float64)


def unity_world_to_camera(pos_xyz, quat_xyzw):
    """
    Unity's ``Camera.worldToCameraMatrix``:
    ``Matrix4x4.Scale(new Vector3(1, 1, -1)) * transform.worldToLocalMatrix``.
    Produces OpenGL-style camera space (camera looks down -Z).
    """
    R = pg.quat_to_matrix(quat_xyzw)  # camera->world, Unity LH basis
    p = np.asarray(pos_xyz, dtype=np.float64)
    world_to_local = np.eye(4)
    world_to_local[:3, :3] = R.T
    world_to_local[:3, 3] = -R.T @ p
    return np.diag([1.0, 1.0, -1.0, 1.0]) @ world_to_local


def unity_reference_project(pos_xyz, quat_xyzw, vfov_deg, width, height, X_unity):
    """
    Project a Unity-world point exactly the way Unity's ``WorldToScreenPoint`` does.

    Returns ``(u_from_left, v_from_TOP, depth_metres)``.  Note the flip: Unity's screen
    origin is bottom-left, ours is top-left, and getting this backwards is the single
    most likely bug in the whole pipeline.
    """
    aspect = float(width) / float(height)
    V = unity_world_to_camera(pos_xyz, quat_xyzw)
    P = unity_projection_matrix(vfov_deg, aspect)

    cam = V @ np.append(np.asarray(X_unity, dtype=np.float64), 1.0)
    clip = P @ cam
    w = clip[3]                       # == -cam.z
    ndc = clip[:3] / w
    screen_x = (ndc[0] * 0.5 + 0.5) * width
    screen_y_from_bottom = (ndc[1] * 0.5 + 0.5) * height
    return screen_x, height - screen_y_from_bottom, w


def intrinsics_from_unity(vfov_deg, width, height):
    """The C#-side derivation from ``Camera.projectionMatrix`` (plan section 4)."""
    aspect = float(width) / float(height)
    P = unity_projection_matrix(vfov_deg, aspect)
    fx = P[0, 0] * width * 0.5
    fy = P[1, 1] * height * 0.5
    cx = (1.0 + P[0, 2]) * width * 0.5
    cy = (1.0 - P[1, 2]) * height * 0.5
    return pg.intrinsics_matrix(fx, fy, cx, cy)


# --------------------------------------------------------------------------------------
# Scene helpers
# --------------------------------------------------------------------------------------

def euler_to_quat(pitch_deg, yaw_deg, roll_deg):
    """Unity's ZXY intrinsic Euler order -> (x, y, z, w), matching Quaternion.Euler."""
    cp, sp = np.cos(np.radians(pitch_deg) / 2), np.sin(np.radians(pitch_deg) / 2)
    cy, sy = np.cos(np.radians(yaw_deg) / 2), np.sin(np.radians(yaw_deg) / 2)
    cr, sr = np.cos(np.radians(roll_deg) / 2), np.sin(np.radians(roll_deg) / 2)
    qx_p = np.array([sp, 0.0, 0.0, cp])
    qy_y = np.array([0.0, sy, 0.0, cy])
    qz_r = np.array([0.0, 0.0, sr, cr])

    def mul(a, b):
        ax, ay, az, aw = a
        bx, by, bz, bw = b
        return np.array([
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ])

    return mul(mul(qy_y, qx_p), qz_r)  # Unity applies Z, then X, then Y


def make_test_texture(h=600, w=900, square=50):
    """Checkerboard plus an asymmetric L-shaped marker, so mirrors are detectable."""
    yy, xx = np.mgrid[0:h, 0:w]
    board = (((yy // square) + (xx // square)) % 2).astype(np.uint8) * 200 + 30
    img = np.stack([board] * 3, axis=-1)
    img[60:80, 60:220] = (255, 40, 40)     # long arm, +x
    img[60:200, 60:80] = (40, 255, 40)     # short arm, +y  -> L, chiral
    img[h - 120:h - 60, w - 220:w - 60] = (40, 40, 255)
    return img


class Camera:
    def __init__(self, pos, quat, vfov_deg, width, height):
        self.pos = np.asarray(pos, dtype=np.float64)
        self.quat = np.asarray(quat, dtype=np.float64)
        self.vfov_deg = vfov_deg
        self.width = width
        self.height = height
        self.K = intrinsics_from_unity(vfov_deg, width, height)
        self.R_cv, self.C = pg.unity_pose_to_cv(self.pos, self.quat)

    @property
    def right_rh(self):
        return pg.unity_dir_to_rh(pg.quat_to_matrix(self.quat)[:, 0])

    @property
    def up_rh(self):
        return pg.unity_dir_to_rh(pg.quat_to_matrix(self.quat)[:, 1])

    @property
    def forward_rh(self):
        return pg.unity_dir_to_rh(pg.quat_to_matrix(self.quat)[:, 2])


# --------------------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------------------

_failures = []


def make_bare_stitcher(ps_mod, torch):
    """
    A PlanarStitcher with no BaseStitcher constructor run.

    ``__new__`` rather than ``__init__`` on purpose: BaseStitcher's constructor downloads
    and loads a SuperPoint model, which none of these tests need and none should require
    a network for.  The cost is that every attribute the render or estimator paths touch
    has to be populated here -- so it lives in one helper rather than being copied into
    each test, where the copies would silently drift out of date the next time
    ``__init__`` gains a field.
    """
    s = ps_mod.PlanarStitcher.__new__(ps_mod.PlanarStitcher)
    s.render_device = torch.device("cpu")   # deterministic; the fp16 CUDA path differs
    # The learned aligner's attributes (_aligner, _align_stats, _clock, ...) are class-level
    # defaults on PlanarStitcher for exactly this helper, so they need no line here.
    s._correction = {"dpose": None, "plane": None}
    # The render path's caches (_lattice_cache, _src_cache, _out_ring, ...) are class-level
    # defaults as well.
    s._est_grid_key = None
    s._est_grid = None
    s._frame_snapshot = None
    s._snapshot_time = 0.0
    s._live_config = None
    s._plane_offset = 0.0
    s._pose_shift = {}
    s._sweep_stats = {}
    s._refine_stats = {}
    s._last_geometry = None
    s._sweep_mode = "acquire"
    s._sweep_lost = 0
    s._last_acquire = 0.0
    s._plane_source = None
    s._fit_mpp = None
    s._fit_step = 0
    s._fit_centre_world = None
    s._fit_changed_at = 0.0
    s._fit_extent = None
    s._plane_invalid_since = None
    s._unposed = 0
    s._stale = 0
    s._last_stats = {}
    s._last_log = 1e18                      # suppress the periodic log lines
    s._last_est_log = 1e18
    return s


def check(name, ok, detail=""):
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {name}" + (f"  --  {detail}" if detail else ""))
    if not ok:
        _failures.append(name)
    return ok


def test_projection_agreement():
    """
    planar_geometry's projection must match Unity's own, including the v-flip.
    This is the test that catches handedness and image-origin errors.
    """
    print("\n1. Unity <-> planar_geometry projection agreement")
    W, H, vfov = 800, 450, 46.4

    cases = [
        ("nadir",  (0.0, 40.0, 0.0),  euler_to_quat(90.0, 0.0, 0.0)),
        ("nadir yawed", (5.0, 35.0, -3.0), euler_to_quat(90.0, 37.0, 0.0)),
        ("facade", (0.0, 12.0, -30.0), euler_to_quat(0.0, 0.0, 0.0)),
        ("oblique", (8.0, 25.0, -14.0), euler_to_quat(35.0, -22.0, 0.0)),
        ("rolled",  (-4.0, 18.0, -9.0), euler_to_quat(20.0, 15.0, 11.0)),
    ]
    targets_unity = [
        (0.0, 0.0, 0.0), (7.0, 0.0, 3.0), (-5.0, 2.0, 6.0),
        (3.0, -1.0, -4.0), (0.0, 5.0, 12.0),
    ]

    worst = 0.0
    for label, pos, quat in cases:
        cam = Camera(pos, quat, vfov, W, H)
        for X_u in targets_unity:
            u_ref, v_ref, depth_ref = unity_reference_project(pos, quat, vfov, W, H, X_u)
            if depth_ref <= 0.1:
                continue
            uv, depth = pg.project_points(cam.K, cam.R_cv, cam.C, pg.unity_point_to_rh(X_u))
            worst = max(worst, abs(uv[0] - u_ref), abs(uv[1] - v_ref),
                        abs(depth - depth_ref))
    check("projection matches Unity reference", worst < 1e-6, f"max err {worst:.3e} px")


def test_plane_homography():
    """G must reproduce a direct projection of the same plane point."""
    print("\n2. Plane homography G vs direct projection")
    W, H, vfov = 800, 450, 46.4
    cam = Camera((6.0, 30.0, -4.0), euler_to_quat(90.0, 25.0, 0.0), vfov, W, H)

    n = pg.unity_dir_to_rh((0.0, 1.0, 0.0))   # ground plane, normal toward the cameras
    d = 0.0
    O = pg.unity_point_to_rh((0.0, 0.0, 0.0))
    frame = pg.build_plane_frame(n, d, O, cam.right_rh, cam.up_rh)

    check("frame is orthonormal",
          abs(np.dot(frame.e1, frame.e2)) < 1e-12
          and abs(np.linalg.norm(frame.e1) - 1) < 1e-12
          and abs(np.linalg.norm(frame.e2) - 1) < 1e-12)
    check("frame is right-handed (e1 x e2 == n)",
          np.allclose(np.cross(frame.e1, frame.e2), frame.n, atol=1e-12))

    G = pg.build_G(cam.K, cam.R_cv, cam.C, frame)
    worst = 0.0
    for a, b in [(0, 0), (5, 3), (-7, 2), (2, -9), (-4, -6)]:
        p = G @ np.array([a, b, 1.0])
        uv_G, depth_G = p[:2] / p[2], p[2]
        uv_d, depth_d = pg.project_points(cam.K, cam.R_cv, cam.C, frame.to_world((a, b)))
        worst = max(worst, np.max(np.abs(uv_G - uv_d)), abs(depth_G - depth_d))
    check("G matches direct projection", worst < 1e-9, f"max err {worst:.3e}")

    # Third component of G must be metric depth: a nadir camera at 30 m over d=0.
    depth_at_origin = (G @ np.array([0.0, 0.0, 1.0]))[2]
    check("third component is depth in metres", abs(depth_at_origin - 30.0) < 1e-9,
          f"got {depth_at_origin:.6f} m, expected 30.0")


def test_against_textbook_form():
    """Inter-view homography must match the plane-induced K(R + t n^T/d)K^-1 form."""
    print("\n3. Inter-view homography vs textbook plane-induced form")
    W, H, vfov = 800, 450, 46.4
    cam_i = Camera((0.0, 28.0, 0.0), euler_to_quat(90.0, 10.0, 0.0), vfov, W, H)
    cam_j = Camera((6.0, 31.0, 4.0), euler_to_quat(90.0, 18.0, 0.0), vfov, W, H)

    n = pg.unity_dir_to_rh((0.0, 1.0, 0.0))
    O = pg.unity_point_to_rh((0.0, 0.0, 0.0))
    frame = pg.build_plane_frame(n, 0.0, O, cam_i.right_rh, cam_i.up_rh)

    G_i = pg.build_G(cam_i.K, cam_i.R_cv, cam_i.C, frame)
    G_j = pg.build_G(cam_j.K, cam_j.R_cv, cam_j.C, frame)
    H_ij = G_j @ np.linalg.inv(G_i)          # image i -> image j, via the plane

    # Textbook: express the plane in camera i's frame as (n_i, d_i), then
    # H = K_j (R_ij + t_ij n_i^T / d_i) K_i^-1.
    R_ij = cam_j.R_cv @ cam_i.R_cv.T
    t_ij = cam_j.R_cv @ (cam_i.C - cam_j.C)
    n_i = cam_i.R_cv @ n
    d_i = float(np.dot(n, frame.O - cam_i.C))   # signed distance along n, in cam i frame
    H_ref = cam_j.K @ (R_ij + np.outer(t_ij, n_i) / d_i) @ np.linalg.inv(cam_i.K)

    H_ij = H_ij / H_ij[2, 2]
    H_ref = H_ref / H_ref[2, 2]
    err = float(np.max(np.abs(H_ij - H_ref)))
    check("matches plane-induced form", err < 1e-9, f"max elem err {err:.3e}")


def test_canvas_orientation():
    """
    Canvas y must grow downward while e2 points up, or the panorama is mirrored.
    """
    print("\n4. Canvas mapping orientation")
    s, cx_c, cy_c = 0.05, 600.0, 400.0
    M = pg.canvas_to_plane_matrix(s, cx_c, cy_c)

    centre = M @ np.array([cx_c, cy_c, 1.0])
    check("canvas centre maps to plane origin", np.allclose(centre[:2], (0.0, 0.0)))

    right = M @ np.array([cx_c + 100.0, cy_c, 1.0])
    check("canvas +x -> plane +a", right[0] > 0 and abs(right[0] - 5.0) < 1e-12,
          f"a={right[0]:.4f} m")

    down = M @ np.array([cx_c, cy_c + 100.0, 1.0])
    check("canvas +y (down) -> plane -b", down[1] < 0 and abs(down[1] + 5.0) < 1e-12,
          f"b={down[1]:.4f} m")


def test_rejection_cases():
    """Degenerate configurations must be rejected, not silently produce garbage."""
    print("\n5. Rejection / clipping cases")
    W, H, vfov = 800, 450, 46.4
    n = pg.unity_dir_to_rh((0.0, 1.0, 0.0))
    O = pg.unity_point_to_rh((0.0, 0.0, 0.0))

    # (a) camera looking away from the plane
    up_cam = Camera((0.0, 20.0, 0.0), euler_to_quat(-90.0, 0.0, 0.0), vfov, W, H)
    frame = pg.build_plane_frame(n, 0.0, O, up_cam.right_rh, up_cam.up_rh)
    fp = pg.view_footprint(up_cam.K, up_cam.R_cv, up_cam.C, frame, W, H)
    check("camera facing away -> empty footprint", len(fp) == 0, f"{len(fp)} points")

    # (b) max-range clipping on a near-horizon view
    graze = Camera((0.0, 30.0, 0.0), euler_to_quat(6.0, 0.0, 0.0), vfov, W, H)
    frame_g = pg.build_plane_frame(n, 0.0, O, graze.right_rh, graze.up_rh)
    fp_far = pg.view_footprint(graze.K, graze.R_cv, graze.C, frame_g, W, H)
    fp_near = pg.view_footprint(graze.K, graze.R_cv, graze.C, frame_g, W, H, max_range=200.0)
    check("max_range clips the far footprint", len(fp_near) < len(fp_far),
          f"{len(fp_far)} -> {len(fp_near)} points")

    # (c) projective-sanity gate: a nadir view is an isotropic similarity (~1.0), a
    #     near-horizon view stretches hard.  The gate must separate them by a wide
    #     margin, or a threshold cannot be chosen.
    nadir = Camera((0.0, 30.0, 0.0), euler_to_quat(90.0, 0.0, 0.0), vfov, W, H)
    frame_n = pg.build_plane_frame(n, 0.0, O, nadir.right_rh, nadir.up_rh)
    M = pg.canvas_to_plane_matrix(0.05, 600.0, 400.0)
    aniso_nadir = pg.homography_anisotropy(
        pg.homography_canvas_to_image(pg.build_G(nadir.K, nadir.R_cv, nadir.C, frame_n), M),
        1200, 800)
    aniso_graze = pg.homography_anisotropy(
        pg.homography_canvas_to_image(pg.build_G(graze.K, graze.R_cv, graze.C, frame_g), M),
        1200, 800)
    check("nadir view is isotropic", abs(aniso_nadir - 1.0) < 1e-6,
          f"anisotropy {aniso_nadir:.6f}")
    check("grazing view trips the anisotropy gate", aniso_graze > 20.0 * aniso_nadir,
          f"nadir {aniso_nadir:.3f} vs grazing {aniso_graze:.1f}")

    # (d) plane behind the camera
    below = Camera((0.0, -20.0, 0.0), euler_to_quat(90.0, 0.0, 0.0), vfov, W, H)
    frame_b = pg.build_plane_frame(n, 0.0, O, below.right_rh, below.up_rh)
    fp_b = pg.view_footprint(below.K, below.R_cv, below.C, frame_b, W, H)
    check("plane behind camera -> empty footprint", len(fp_b) == 0, f"{len(fp_b)} points")


def test_mosaic_roundtrip():
    """
    End-to-end: synthesise views of a textured plane through the analytic homography,
    stitch them back, and compare against the original texture.  With exact pose and
    exact intrinsics there is nothing being estimated, so this must be near-exact --
    any real error here is a bug, not a limitation.
    """
    print("\n6. Mosaic round-trip (synthesise -> stitch -> compare)")
    if cv2 is None:
        check("cv2 available", False, "opencv not installed; skipped")
        return

    W, H, vfov = 800, 450, 46.4
    texture = make_test_texture(square=60)
    th, tw = texture.shape[:2]

    n = pg.unity_dir_to_rh((0.0, 1.0, 0.0))
    O = pg.unity_point_to_rh((0.0, 0.0, 0.0))
    altitude = 26.0

    # Match the texture scale to the cameras' ground sampling distance, so one texture
    # pixel is about one image pixel.  Otherwise the synthesised view *downsamples* the
    # texture and the round-trip cannot reconstruct it however perfect the geometry is
    # -- the residual would be aliasing masquerading as misalignment.
    fy = float(intrinsics_from_unity(vfov, W, H)[1, 1])
    s_tex = altitude / fy

    poses = [
        ((-3.0, altitude, 0.0), euler_to_quat(90.0, 0.0, 0.0)),
        ((0.0, altitude, 0.0),  euler_to_quat(90.0, 0.0, 0.0)),
        ((3.0, altitude, 0.0),  euler_to_quat(90.0, 0.0, 0.0)),
        ((0.0, altitude, 3.0),  euler_to_quat(90.0, 0.0, 0.0)),
    ]
    cams = [Camera(p, q, vfov, W, H) for p, q in poses]
    frame = pg.build_plane_frame(n, 0.0, O, cams[1].right_rh, cams[1].up_rh)

    M_tex = pg.canvas_to_plane_matrix(s_tex, tw / 2.0, th / 2.0)

    J_centre, _ = pg.homography_jacobian(pg.homography_canvas_to_image(
        pg.build_G(cams[1].K, cams[1].R_cv, cams[1].C, frame), M_tex), tw / 2.0, th / 2.0)
    print("       sampling: 1 texture px -> "
          f"{np.linalg.svd(J_centre, compute_uv=False)[0]:.3f} image px")

    acc = np.zeros((th, tw, 3), np.float64)
    wsum = np.zeros((th, tw, 1), np.float64)
    warped_views = []
    for cam in cams:
        G = pg.build_G(cam.K, cam.R_cv, cam.C, frame)
        H_tex_to_img = pg.homography_canvas_to_image(G, M_tex)

        # Synthesise the view: warp the texture into the camera image.
        view = cv2.warpPerspective(texture, H_tex_to_img, (W, H), flags=cv2.INTER_LINEAR)
        valid_src = cv2.warpPerspective(np.ones((th, tw), np.float32), H_tex_to_img,
                                        (W, H), flags=cv2.INTER_NEAREST)

        # Stitch it back: the same homography, inverted.
        back = cv2.warpPerspective(view, H_tex_to_img, (tw, th),
                                   flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP)
        mask = cv2.warpPerspective(valid_src, H_tex_to_img, (tw, th),
                                   flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP)
        mask = (mask > 0.5).astype(np.float64)[..., None]
        warped_views.append((back.astype(np.float64), mask[..., 0] > 0.5))
        acc += back.astype(np.float64) * mask
        wsum += mask

    covered = wsum[..., 0] > 0
    mosaic = np.zeros_like(acc)
    np.divide(acc, np.maximum(wsum, 1e-6), out=mosaic)

    coverage = covered.mean()
    check("views cover most of the texture", coverage > 0.5, f"{coverage * 100:.1f}%")

    # Interior only: near a view's border the resampling stencil runs off the edge, so
    # those pixels carry interpolation error that says nothing about the geometry.
    er = cv2.erode(covered.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
    blur_t = cv2.GaussianBlur(texture, (0, 0), 1.5).astype(np.float64)
    blur_m = cv2.GaussianBlur(np.clip(mosaic, 0, 255).astype(np.uint8),
                              (0, 0), 1.5).astype(np.float64)
    diff = np.abs(blur_m[er] - blur_t[er])
    rmse = float(np.sqrt((diff ** 2).mean()))
    check("mosaic reconstructs the texture", rmse < 4.0,
          f"RMSE {rmse:.3f} (0-255), p99 {np.percentile(diff, 99):.1f}")

    # The decisive geometric test: measure the residual *registration shift* between
    # every pair of independently-warped views by phase correlation.
    #
    # Deliberately not an overlap-PSNR threshold here.  PSNR on this target is dominated
    # by resampling phase, not alignment: each view resamples the checkerboard's hard
    # edges at a different sub-pixel offset, which costs ~26 dB even at literally zero
    # misregistration (and two symmetrically-placed cameras score 65 dB purely because
    # they happen to share a phase).  A shift in pixels is what we actually care about,
    # is directly interpretable, and is not confounded by texture contrast.
    worst_shift, worst_pair = 0.0, ""
    for i in range(len(warped_views)):
        for j in range(i + 1, len(warped_views)):
            (a, ma), (b, mb) = warped_views[i], warped_views[j]
            both = ma & mb & er
            if both.sum() < 10000:
                continue
            ys, xs = np.where(both)
            box = (slice(ys.min(), ys.max()), slice(xs.min(), xs.max()))
            ga = cv2.cvtColor(a[box].astype(np.float32), cv2.COLOR_BGR2GRAY)
            gb = cv2.cvtColor(b[box].astype(np.float32), cv2.COLOR_BGR2GRAY)
            window = cv2.createHanningWindow((ga.shape[1], ga.shape[0]), cv2.CV_32F)
            (dx, dy), _ = cv2.phaseCorrelate(ga, gb, window)
            shift = float(np.hypot(dx, dy))
            if shift > worst_shift:
                worst_shift, worst_pair = shift, f"views {i}-{j}"
    check("pairwise overlaps are sub-pixel registered", worst_shift < 0.1,
          f"worst shift {worst_shift:.4f} px ({worst_pair})")

    # Chirality: the asymmetric marker must land where it started, not mirrored.
    # Channel 0 is blue under OpenCV's BGR ordering -- this tracks the L's long arm.
    marker = (texture[..., 0] > 200) & (texture[..., 1] < 80) & (texture[..., 2] < 80)
    marker_m = (mosaic[..., 0] > 180) & (mosaic[..., 1] < 100) & (mosaic[..., 2] < 100)
    if marker.any() and marker_m.any():
        cy_t, cx_t = np.argwhere(marker).mean(axis=0)
        cy_m, cx_m = np.argwhere(marker_m).mean(axis=0)
        shift = float(np.hypot(cx_m - cx_t, cy_m - cy_t))
        check("asymmetric marker is not mirrored/rotated", shift < 6.0,
              f"centroid shift {shift:.2f} px")
    else:
        check("asymmetric marker found", False, "marker missing from mosaic")

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "selftest_mosaic.jpg")
    cv2.imwrite(out, np.concatenate(
        [texture, np.clip(mosaic, 0, 255).astype(np.uint8)], axis=0))
    print(f"       wrote {out}  (top = source texture, bottom = stitched)")


def test_planar_stitcher_end_to_end():
    """
    Drive the real PlanarStitcher.planar_pano over synthetic views, exercising the
    torch render path (batched homography, validity masks, border feather, N-view
    weight normalisation) against the numpy geometry the earlier tests pinned down.

    PlanarStitcher is built with __new__ rather than __init__ on purpose: BaseStitcher's
    constructor downloads and loads a SuperPoint model, which this test neither needs
    nor should depend on a network for. Only the attributes the render path touches are
    populated.
    """
    print("\n7. PlanarStitcher end-to-end (torch render path)")
    if cv2 is None:
        check("cv2 available", False, "opencv not installed; skipped")
        return
    try:
        import torch  # noqa: F401
        import PlanarStitcher as ps_mod
    except ImportError as e:
        check("torch + PlanarStitcher importable", False, f"{e}")
        return

    stitcher = make_bare_stitcher(ps_mod, torch)

    W, H, vfov = 800, 450, 46.4
    altitude = 26.0
    texture = make_test_texture(square=60)
    th, tw = texture.shape[:2]
    K = intrinsics_from_unity(vfov, W, H)
    s_tex = altitude / float(K[1, 1])

    n_unity = (0.0, 1.0, 0.0)
    frame = pg.build_plane_frame(
        pg.unity_dir_to_rh(n_unity), 0.0, pg.unity_point_to_rh((0.0, 0.0, 0.0)),
        Camera((0.0, altitude, 0.0), euler_to_quat(90, 0, 0), vfov, W, H).right_rh)
    M_tex = pg.canvas_to_plane_matrix(s_tex, tw / 2.0, th / 2.0)

    # Five nadir views on a cross pattern, overlapping heavily. Drone 0 sits over the
    # plane origin and is nominated as the centre below, while the *median* of the
    # id-sorted list is drone 2 at (0, 4) -- deliberately a different drone. The canvas
    # is framed on the reference view, so the RMSE check further down passes only if
    # _build_geometry honours the published centre rather than taking the median.
    offsets = [(0.0, 0.0), (-4.0, 0.0), (0.0, 4.0), (0.0, -4.0), (4.0, 0.0)]
    views = []
    for i, (dx, dz) in enumerate(offsets):
        pos = (dx, altitude, dz)
        quat = euler_to_quat(90.0, 0.0, 0.0)
        cam = Camera(pos, quat, vfov, W, H)
        H_tex_to_img = pg.homography_canvas_to_image(
            pg.build_G(cam.K, cam.R_cv, cam.C, frame), M_tex)
        img = cv2.warpPerspective(texture, H_tex_to_img, (W, H), flags=cv2.INTER_LINEAR)
        views.append({
            'slot': i, 'drone_id': i, 'heading': 0.0, 'image': img,
            'pos': pos, 'quat': tuple(float(c) for c in quat),
            'capture_time': 0.0, 'pose_status': 3, 'cached': False,
        })

    canvas_w, canvas_h = 1000, 700
    config = {
        "canvas": (canvas_w, canvas_h),
        "metres_per_pixel": s_tex,
        "max_range": 200.0,
        "feather_px": 40,
        "aniso_max": 12.0,
        "min_coverage": 0.2,
        "pose_source": 0,
        "psnr_gate": False,
    }
    plane = {"plane_normal": n_unity, "plane_d": 0.0,
             "plane_valid": True, "plane_mode": 1, "gimbal_pitch": -90.0,
             "centre_drone_id": 0}

    pano, ok, reason = stitcher.planar_pano(
        views, (K[0, 0], K[1, 1], K[0, 2], K[1, 2]), plane, config)

    if not check("planar_pano returned a panorama", ok and pano is not None,
                 f"ok={ok} reason={reason}"):
        return
    check("panorama has the requested canvas shape",
          pano.shape == (canvas_h, canvas_w, 3), f"{pano.shape}")

    stats = stitcher._last_stats
    check("all five views were kept", stats.get("views") == 5,
          f"kept {stats.get('views')}, dropped {stats.get('dropped')}")
    check("nadir views are isotropic", abs(stats.get("max_aniso", 9) - 1.0) < 1e-3,
          f"max anisotropy {stats.get('max_aniso'):.4f}")
    check("coverage is reported", stats.get("coverage", 0) > 0.3,
          f"{stats.get('coverage', 0):.0%}")

    # The decisive check: the panorama must reproduce the plane texture. Compare against
    # the texture resampled through the *canvas* mapping, so only geometry is under test.
    # ref_H maps canvas pixels -> texture pixels, which is the inverse of the direction
    # cv2.warpPerspective applies by default, hence WARP_INVERSE_MAP.
    covered = pano.any(axis=2)
    M_canvas = pg.canvas_to_plane_matrix(
        config["metres_per_pixel"], canvas_w * 0.5, canvas_h * 0.5)
    ref_H = np.linalg.inv(M_tex) @ M_canvas
    ref = cv2.warpPerspective(texture, ref_H, (canvas_w, canvas_h),
                              flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP)

    er = cv2.erode(covered.astype(np.uint8), np.ones((15, 15), np.uint8)) > 0
    if er.sum() < 5000:
        check("enough interior pixels to compare", False, f"{er.sum()}")
        return
    blur_a = cv2.GaussianBlur(pano, (0, 0), 1.5).astype(np.float64)
    blur_b = cv2.GaussianBlur(ref, (0, 0), 1.5).astype(np.float64)
    rmse = float(np.sqrt(((blur_a[er] - blur_b[er]) ** 2).mean()))
    check("panorama matches the plane texture", rmse < 6.0, f"RMSE {rmse:.3f} (0-255)")

    # Sanity-check that the PSNR diagnostic is being computed and reported, not that it
    # is high: on this checkerboard the metric is floored around 26-29 dB by resampling
    # phase even at zero misregistration (see test 6), so a tight threshold here would
    # be measuring interpolation rather than alignment. The RMSE check above is the
    # precise geometric assertion; in production this number matters as a *relative*
    # signal that degrades as pose error grows.
    psnr = stats.get("overlap_psnr")
    check("overlap PSNR is reported and sane",
          psnr is not None and (not np.isfinite(psnr) or psnr > 25.0),
          f"{psnr:.1f} dB (resampling-limited)" if psnr is not None and np.isfinite(psnr)
          else str(psnr))

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "selftest_planar_pano.jpg")
    cv2.imwrite(out, np.concatenate([ref, pano], axis=0))
    print(f"       wrote {out}  (top = expected, bottom = PlanarStitcher output)")

    # Prove the check above has teeth. Unity nominates the centre drone because the canvas
    # origin and axes are built from it; re-deriving it here (the old median-of-selection
    # rule) frames the mosaic on whichever drone happens to sit at the median id, and the
    # mosaic translates when the selection changes. Drone 0 is at the plane origin and
    # drone 2 is 4 m away, so dropping the published centre must visibly move the canvas.
    # If this came out equal, the RMSE assertion would be passing for the wrong reason.
    plane_no_centre = dict(plane)
    plane_no_centre["centre_drone_id"] = -1
    stitcher._plane_invalid_since = None
    pano_median, ok_median, _ = stitcher.planar_pano(
        views, (K[0, 0], K[1, 1], K[0, 2], K[1, 2]), plane_no_centre, config)
    if ok_median and pano_median is not None:
        shift_rmse = float(np.sqrt(((pano_median.astype(np.float64)
                                     - pano.astype(np.float64)) ** 2).mean()))
        check("published centre drone actually frames the canvas", shift_rmse > 20.0,
              f"median-fallback canvas differs by RMSE {shift_rmse:.1f}")
    else:
        check("median-fallback canvas rendered for comparison", False,
              f"ok={ok_median}")

    test_blend_modes(stitcher, views, K, plane, config)
    test_unwritten_blocks(stitcher, views, K, plane, config, pano)


def test_unwritten_blocks(stitcher, views, K, plane, config, good_pano):
    """
    A block slot Unity has never written must cost its own view, not the frame.

    Such a slot reads back zero-filled: flag 0 ("ready"), droneId 0 -- a *legal* id, not
    the -1 sentinel a retired slot carries -- and an all-zero quaternion.  That used to
    reach quat_to_matrix and raise ValueError("degenerate quaternion") out of
    planar_pano, so one unwritten slot killed the whole panorama and the traceback
    pointed at the geometry rather than at the wire.  The zeroed slot is placed *first*
    and given the same drone id as the published centre, which is the worst case: it is
    what both the centre lookup and the median fallback would otherwise land on.
    """
    print("\n9. Unwritten / degenerate block slots")
    import PlanarStitcher as ps_mod
    intr = (K[0, 0], K[1, 1], K[0, 2], K[1, 2])
    zero_slot = {
        'slot': 9, 'drone_id': plane["centre_drone_id"], 'heading': 0.0,
        'image': np.zeros_like(views[0]['image']),
        'pos': (0.0, 0.0, 0.0), 'quat': (0.0, 0.0, 0.0, 0.0),
        'capture_time': 0.0, 'pose_status': 0, 'cached': False,
    }

    stitcher._plane_invalid_since = None
    pano, ok, reason = stitcher.planar_pano([zero_slot] + list(views), intr, plane, config)
    if not check("a zeroed block does not abort the frame", ok and pano is not None,
                 f"ok={ok} reason={reason}"):
        return
    check("the zeroed block is counted as unposed",
          stitcher._last_stats.get("unposed") == 1,
          f"unposed={stitcher._last_stats.get('unposed')}, "
          f"views={stitcher._last_stats.get('views')}")
    check("the surviving views mosaic identically",
          np.array_equal(pano, good_pano),
          "byte-identical" if np.array_equal(pano, good_pano)
          else f"max diff {int(np.abs(pano.astype(int) - good_pano.astype(int)).max())}")

    # All slots unwritten (Python mapped the section before Unity filled any of it): a
    # clean quality reason, still no exception. PLANE_INVALID is the bit Unity prints as
    # "no usable scene plane / pose".
    stitcher._plane_invalid_since = None
    pano_none, ok_none, reason_none = stitcher.planar_pano(
        [dict(zero_slot, slot=i, drone_id=i) for i in range(3)], intr, plane, config)
    check("all-unwritten reports a pose failure, not a crash",
          (not ok_none) and pano_none is None
          and reason_none == ps_mod.REASON_PLANE_INVALID,
          f"ok={ok_none} reason={reason_none}")


def test_blend_modes(stitcher, views, K, plane, config):
    """
    BLEND_NEAREST must not average overlapping views, BLEND_FEATHER must.

    Measured on deliberately *wrong* poses, because that is the only regime where the
    two modes differ: with exact poses every view agrees pixel for pixel and averaging
    them is harmless.  Alternate drones are displaced 0.3 m in the plane -- about 6
    canvas pixels here -- which is GNSS-scale error.  Feathering then superimposes two
    offset copies of the texture, and the giveaway is lost high-frequency energy: a
    ghosted checkerboard is a blurred checkerboard.  Winner-take-all cannot blur,
    because no pixel ever has two contributors.
    """
    print("\n8. PlanarStitcher blend modes (ghosting under pose error)")
    import PlanarStitcher as ps_mod
    intr = (K[0, 0], K[1, 1], K[0, 2], K[1, 2])

    noisy = []
    for i, v in enumerate(views):
        w = dict(v)
        w["pos"] = (v["pos"][0] + (0.3 if i % 2 else -0.3), v["pos"][1], v["pos"][2])
        noisy.append(w)

    panos = {}
    for name, mode in (("feather", ps_mod.BLEND_FEATHER),
                       ("nearest", ps_mod.BLEND_NEAREST)):
        cfg = dict(config)
        cfg["blend_mode"] = mode
        stitcher._plane_invalid_since = None
        pano, ok, reason = stitcher.planar_pano(noisy, intr, plane, cfg)
        if not check(f"{name} mode rendered", ok and pano is not None,
                     f"ok={ok} reason={reason}"):
            return
        panos[name] = pano

    cov = {k: (p.any(axis=2)) for k, p in panos.items()}
    check("winner-take-all loses no coverage",
          abs(cov["nearest"].sum() - cov["feather"].sum())
          <= 0.02 * max(1, cov["feather"].sum()),
          f"nearest {cov['nearest'].sum()} px vs feather {cov['feather'].sum()} px")

    interior = cv2.erode((cov["nearest"] & cov["feather"]).astype(np.uint8),
                         np.ones((21, 21), np.uint8)) > 0
    if interior.sum() < 5000:
        check("enough interior pixels to compare", False, f"{interior.sum()}")
        return

    # Squared gradient energy, not |gradient|: blurring an edge across w pixels leaves
    # the sum of |gradient| unchanged (it is the step height either way) and divides the
    # sum of gradient^2 by w. Only the squared form can tell a sharp seam from a smear.
    energy = {}
    for name, p in panos.items():
        g = cv2.cvtColor(p, cv2.COLOR_BGR2GRAY).astype(np.float64)
        gx = cv2.Sobel(g, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(g, cv2.CV_64F, 0, 1, ksize=3)
        energy[name] = float((gx * gx + gy * gy)[interior].mean())

    check("feathering visibly ghosts under pose error",
          energy["nearest"] > 1.25 * energy["feather"],
          f"edge energy: nearest {energy['nearest']:.0f} vs "
          f"feather {energy['feather']:.0f} "
          f"(ratio {energy['nearest'] / max(energy['feather'], 1e-9):.2f}x)")

    # The overlap PSNR diagnostic is derived from geometric coverage, not from the blend
    # weights -- under winner-take-all the weights are one-hot, so a weight-based test
    # would report "no overlap anywhere" and quietly retire the metric that quantifies
    # pose error, exactly when pose error is what is being looked at.
    psnr = stitcher._last_stats.get("overlap_psnr")
    check("overlap PSNR survives winner-take-all",
          psnr is not None and np.isfinite(psnr),
          f"{psnr}")

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "selftest_planar_blend.jpg")
    cv2.imwrite(out, np.concatenate([panos["feather"], panos["nearest"]], axis=0))
    print(f"       wrote {out}  (top = feather/ghosted, bottom = nearest)")

    # Debug overlay. FLAT under NEAREST is the strongest assertion available here: the
    # panorama must consist of exactly the palette colours of the drones in the
    # selection and nothing else. Anything blended, mis-indexed or off-by-one in the
    # BGR/RGB conversion shows up as a colour that is not in the palette.
    cfg = dict(config)
    cfg["blend_mode"] = ps_mod.BLEND_NEAREST
    cfg["debug_view"] = ps_mod.DEBUG_FLAT
    stitcher._plane_invalid_since = None
    flat, ok, reason = stitcher.planar_pano(noisy, intr, plane, cfg)
    if not check("debug FLAT rendered", ok and flat is not None,
                 f"ok={ok} reason={reason}"):
        return

    expected = {ps_mod.debug_colour(v["drone_id"])[1] for v in noisy}
    seen = {tuple(int(c) for c in px)
            for px in np.unique(flat[cov["nearest"]].reshape(-1, 3), axis=0)}
    check("flat overlay uses only the drones' palette colours",
          seen <= expected, f"unexpected {sorted(seen - expected)[:4]}")
    check("every selected drone owns some of the canvas",
          len(seen) == len(expected), f"{len(seen)} of {len(expected)} drones visible")

    cfg["debug_view"] = ps_mod.DEBUG_TINT
    stitcher._plane_invalid_since = None
    tint, ok, reason = stitcher.planar_pano(noisy, intr, plane, cfg)
    if not check("debug TINT rendered", ok and tint is not None,
                 f"ok={ok} reason={reason}"):
        return
    # The point of TINT over FLAT is that the imagery survives the wash, so assert the
    # contrast does: a tint strength high enough to flatten the scene is a tint nobody
    # can navigate by.
    contrast = float(cv2.cvtColor(tint, cv2.COLOR_BGR2GRAY)[interior].std())
    check("debug TINT keeps the imagery legible", contrast > 40.0,
          f"interior contrast {contrast:.1f} (flat would be ~0)")

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "selftest_planar_debug.jpg")
    cv2.imwrite(out, np.concatenate([flat, tint], axis=0))
    print(f"       wrote {out}  (top = flat, bottom = tint)")


# Textures for the estimator tests are deliberately larger than the swarm's combined
# footprint (~50 x 32 m here, i.e. ~1000 x 650 texture px). A texture the views can see
# the edge of hands the correlator a unique, unambiguous cue for free -- which quietly
# turns the periodic-texture check into a test of the border rather than of the pattern.
def make_unique_texture(h=1400, w=2000, seed=7):
    """
    Low-pass filtered noise: locally unique, so cross-correlation has one clear peak.

    Deliberately NOT the checkerboard ``make_test_texture`` builds.  A periodic pattern
    is the case the refiner is supposed to *refuse*, and it gets its own check below --
    using it here would test the rejection path while claiming to test the accuracy one.
    """
    rng = np.random.default_rng(seed)
    noise = rng.random((h, w)).astype(np.float32)
    smooth = cv2.GaussianBlur(noise, (0, 0), 3.0)
    smooth = (smooth - smooth.min()) / max(1e-9, smooth.ptp())
    grey = (smooth * 235 + 10).astype(np.uint8)
    return np.stack([grey] * 3, axis=-1)


def make_periodic_texture(h=1400, w=2000, square=40):
    """
    A pure checkerboard: no markers, no border features, genuinely ambiguous.

    Distinct from ``make_test_texture``, which adds a chiral L and a corner block
    precisely so that mirrors and rotations are detectable -- those markers also make it
    globally solvable, so it is the wrong tool for testing that ambiguity is refused.
    This is the stand-in for a facade of identical windows.
    """
    yy, xx = np.mgrid[0:h, 0:w]
    board = (((yy // square) + (xx // square)) % 2).astype(np.uint8) * 200 + 30
    return np.stack([board] * 3, axis=-1)


def _nadir_scene(texture, altitude=26.0, W=800, H=450, vfov=46.4,
                 offsets=((0.0, 0.0), (-5.0, 0.0), (0.0, 5.0),
                          (0.0, -5.0), (5.0, 0.0))):
    """
    Five nadir views over a textured ground plane at y = 0.

    Returns ``(views, K, s_tex, plane)``.  Images are rendered from the TRUE poses; the
    caller is free to publish something else in the view dicts, which is exactly how the
    estimators get something to find.
    """
    K = intrinsics_from_unity(vfov, W, H)
    th, tw = texture.shape[:2]
    s_tex = altitude / float(K[1, 1])
    n_unity = (0.0, 1.0, 0.0)
    frame = pg.build_plane_frame(
        pg.unity_dir_to_rh(n_unity), 0.0, pg.unity_point_to_rh((0.0, 0.0, 0.0)),
        Camera((0.0, altitude, 0.0), euler_to_quat(90, 0, 0), vfov, W, H).right_rh)
    M_tex = pg.canvas_to_plane_matrix(s_tex, tw / 2.0, th / 2.0)

    views = []
    for i, (dx, dz) in enumerate(offsets):
        pos = (dx, altitude, dz)
        quat = euler_to_quat(90.0, 0.0, 0.0)
        cam = Camera(pos, quat, vfov, W, H)
        H_tex = pg.homography_canvas_to_image(
            pg.build_G(cam.K, cam.R_cv, cam.C, frame), M_tex)
        views.append({
            'slot': i, 'drone_id': i, 'heading': 0.0,
            'image': cv2.warpPerspective(texture, H_tex, (W, H), flags=cv2.INTER_LINEAR),
            'pos': pos, 'quat': tuple(float(c) for c in quat),
            'capture_time': 0.0, 'pose_status': 3, 'cached': False,
        })

    plane = {"plane_normal": n_unity, "plane_d": 0.0, "plane_valid": True,
             "plane_mode": 1, "gimbal_pitch": -90.0, "centre_drone_id": 0}
    return views, K, s_tex, plane


def _run_estimator(stitcher, views, intr, plane, config, passes):
    """
    planar_pano (publishes the frame) then compute_warps, ``passes`` times.

    The config is pushed separately via ``set_live_config``, mirroring production: the
    render path supplies the frame and the metadata loop supplies the switches.  Passing
    it only to ``planar_pano`` would leave ``compute_warps`` with nothing to read.
    """
    stitcher.set_live_config(config)
    for _ in range(passes):
        stitcher._plane_invalid_since = None
        stitcher.planar_pano(views, intr, plane, config)
        stitcher.compute_warps()


def test_estimators():
    """
    The two warp-thread estimators, each against an error of known size.

    Both are checked for what they fix *and* for what they must leave alone: the flags
    are independent, so "sweep on" must not quietly produce a pose correction and vice
    versa.  The accuracy checks use a locally-unique texture; the rejection check uses a
    periodic one, which is the failure mode that matters on a real facade of identical
    windows.
    """
    print("\n10. Warp-thread estimators, photometric (plane sweep + pose refiner)")
    if cv2 is None:
        check("cv2 available", False, "opencv not installed; skipped")
        return
    try:
        import torch  # noqa: F401
        import PlanarStitcher as ps_mod
    except ImportError as e:
        check("torch + PlanarStitcher importable", False, f"{e}")
        return

    # These are the photometric estimators' own tests: what each can and cannot recover,
    # the gates, the reset paths. The two switches drive planar_align by default now
    # (section 13), so this section selects the older pair the way the environment
    # variable does, and puts the module back however it leaves.
    was_legacy = ps_mod.LEGACY_ESTIMATORS
    ps_mod.LEGACY_ESTIMATORS = True
    try:
        _test_photometric_estimators(ps_mod, torch)
    finally:
        ps_mod.LEGACY_ESTIMATORS = was_legacy


def _test_photometric_estimators(ps_mod, torch):

    texture = make_unique_texture()
    views, K, s_tex, plane = _nadir_scene(texture)
    intr = (K[0, 0], K[1, 1], K[0, 2], K[1, 2])
    base_config = {
        "canvas": (1000, 700), "metres_per_pixel": s_tex, "max_range": 200.0,
        "feather_px": 40, "aniso_max": 12.0, "min_coverage": 0.2,
        "pose_source": 0, "psnr_gate": False,
        "blend_mode": ps_mod.BLEND_NEAREST, "debug_view": ps_mod.DEBUG_OFF,
        "plane_sweep": False, "pose_refine": False,
        "sweep_range": 4.0, "sweep_steps": 9,
        "refine_rate": 1.0, "refine_max_shift": 3.0,
    }

    # --- both flags off: the estimators must not run at all ---------------------------
    s = make_bare_stitcher(ps_mod, torch)
    _run_estimator(s, views, intr, plane, base_config, passes=2)
    check("both flags off leaves the correction identity",
          s._correction.get("dpose") is None and s._correction.get("plane") is None
          and s._correction.get("plane_fix") is None,
          f"correction={s._correction}")

    # --- switching an estimator OFF must actually turn it off --------------------------
    # This is the check that matters, and the one this suite previously did not make. The
    # old version started from a fresh stitcher with both flags already off, so the frame
    # snapshot was never published and compute_warps returned at its first line -- it
    # asserted nothing about the reset path. Meanwhile the real code read the switches
    # from the snapshot, which was itself only published while a switch was ON, so the
    # both-off branch was unreachable by construction and an operator turning the sweep
    # off got a frozen frame with the sweep still running on it.
    #
    # A TRANSITION is therefore the only meaningful form of this test: converge with the
    # estimator on, then turn it off and require the correction to actually clear.
    s_off = make_bare_stitcher(ps_mod, torch)
    _run_estimator(s_off, views, intr, dict(plane, plane_d=-1.35),
                   dict(base_config, plane_sweep=True), passes=3)
    converged = s_off._correction.get("plane")
    _run_estimator(s_off, views, intr, dict(plane, plane_d=-1.35), base_config, passes=2)
    check("turning the sweep off clears its correction",
          converged is not None and abs(converged) > 0.1
          and s_off._correction.get("plane") is None and s_off._plane_offset == 0.0,
          f"{converged} -> {s_off._correction.get('plane')}")

    s_off2 = make_bare_stitcher(ps_mod, torch)
    _run_estimator(s_off2, views, intr, plane, dict(base_config, pose_refine=True),
                   passes=3)
    had_dpose = s_off2._correction.get("dpose") is not None
    _run_estimator(s_off2, views, intr, plane, base_config, passes=2)
    check("turning the refiner off clears its correction",
          had_dpose and s_off2._correction.get("dpose") is None
          and s_off2._pose_shift == {},
          f"dpose set={had_dpose} -> {s_off2._correction.get('dpose')}")

    # --- a frozen frame must not keep driving the estimators ---------------------------
    # Every planar_pano path that skips the publish also returns a blank panorama, so a
    # stale snapshot means the render is already down. Continuing to estimate on it
    # converges the correction onto a dead frame and then applies that answer to live
    # geometry the moment the render recovers.
    s_stale = make_bare_stitcher(ps_mod, torch)
    _run_estimator(s_stale, views, intr, dict(plane, plane_d=-1.35),
                   dict(base_config, plane_sweep=True), passes=2)
    frozen = s_stale._plane_offset
    s_stale._snapshot_time -= 10.0 * ps_mod.PlanarStitcher.SNAPSHOT_MAX_AGE_S
    for _ in range(3):
        s_stale.compute_warps()
    check("a stale frame snapshot stops the estimators",
          s_stale._plane_offset == frozen
          and "stale" in s_stale._sweep_stats.get("skipped", ""),
          f"offset {frozen:+.3f} -> {s_stale._plane_offset:+.3f}, "
          f"stats={s_stale._sweep_stats}")

    # --- plane sweep: recover a known plane-distance error -----------------------------
    # Publish the ground 1.35 m too low. Not a whole number of sweep steps (range 4,
    # 9 steps = 1.0 m apart), so this only passes if the parabolic sub-step fit works.
    true_error = 1.35
    wrong_plane = dict(plane, plane_d=-true_error)

    s = make_bare_stitcher(ps_mod, torch)
    cfg = dict(base_config, plane_sweep=True)
    _run_estimator(s, views, intr, wrong_plane, cfg, passes=3)
    found = s._correction.get("plane")
    check("plane sweep recovers the plane-distance error",
          found is not None and abs(found - true_error) < 0.2,
          f"recovered {found:+.3f} m, injected {true_error:+.3f} m"
          if found is not None else "no offset produced")
    check("plane sweep alone produces no pose correction",
          s._correction.get("dpose") is None,
          f"dpose={s._correction.get('dpose')}")

    # --- pose refiner: recover known per-drone position errors -------------------------
    # In-plane only. The refiner has two DoF per view by construction, so a component
    # along the plane normal is not observable to it and asserting on one would be
    # testing a capability it does not claim.
    # Chosen to sum to zero across the formation. The refiner removes the mean correction
    # on purpose (a shift common to every view is unobservable, and leaving it in lets the
    # mosaic wander), so a set of errors with a non-zero mean would leave the corrected
    # mosaic rigidly translated from the reference -- and the image comparison further
    # down would then be measuring that gauge rather than the alignment.
    injected = {1: (0.6, 0.0, 0.0), 4: (-0.6, 0.0, 0.0),
                2: (0.0, 0.0, 0.45), 3: (0.0, 0.0, -0.45)}
    drifted = []
    for v in views:
        w = dict(v)
        e = injected.get(v["drone_id"])
        if e:
            w["pos"] = (v["pos"][0] + e[0], v["pos"][1] + e[1], v["pos"][2] + e[2])
        drifted.append(w)

    s = make_bare_stitcher(ps_mod, torch)
    cfg = dict(base_config, pose_refine=True)
    _run_estimator(s, drifted, intr, plane, cfg, passes=6)
    dpose = s._correction.get("dpose")

    if not check("pose refiner produced corrections", dpose is not None,
                 f"stats={s._refine_stats}"):
        return
    check("pose refiner alone produces no plane offset",
          s._correction.get("plane") is None,
          f"plane={s._correction.get('plane')}")

    # Expected correction is minus the injected error, in right-handed world. Both sides
    # are zero-meaned before comparing: the refiner deliberately removes the mean (a
    # shift common to every view is unobservable and would let the mosaic wander), so an
    # absolute comparison would be testing the gauge rather than the estimate.
    want = {v["drone_id"]: -pg.unity_dir_to_rh(injected.get(v["drone_id"], (0.0, 0.0, 0.0)))
            for v in views}
    common = [d for d in want if d in dpose]
    want_mean = sum(want[d] for d in common) / len(common)
    got_mean = sum(dpose[d][1] for d in common) / len(common)
    worst = max(float(np.linalg.norm((dpose[d][1] - got_mean) - (want[d] - want_mean)))
                for d in common)
    check("pose refiner recovers the per-drone error",
          worst < 0.12,
          f"worst residual {worst:.3f} m over {len(common)} views "
          f"(injected up to {max(np.linalg.norm(v) for v in injected.values()):.2f} m)")
    check("pose refiner leaves rotation alone (translation-only by design)",
          all(dpose[d][0] is None for d in dpose), "dR is None for every view")

    # --- the correction must actually improve the mosaic -------------------------------
    # The numeric checks above could pass while the render path applied the correction
    # with the wrong sign or in the wrong frame; only rendering catches that.
    # The reference and the uncorrected mosaic must both come from stitchers with NO
    # correction: `s` is carrying the one it just estimated, and rendering the true poses
    # through it would apply the drift correction to undrifted views.
    ref_pano, ok_ref, _ = _mosaic(make_bare_stitcher(ps_mod, torch),
                                  views, intr, plane, base_config)
    bad_pano, ok_bad, _ = _mosaic(make_bare_stitcher(ps_mod, torch),
                                  drifted, intr, plane, base_config)
    fixed_pano, ok_fix, _ = _mosaic(s, drifted, intr, plane, base_config)  # keeps _correction

    if check("all three comparison mosaics rendered",
             ok_ref and ok_bad and ok_fix, f"{ok_ref}/{ok_bad}/{ok_fix}"):
        cover = (ref_pano.any(axis=2) & bad_pano.any(axis=2) & fixed_pano.any(axis=2))
        er = cv2.erode(cover.astype(np.uint8), np.ones((21, 21), np.uint8)) > 0
        if er.sum() > 5000:
            def rmse(a, b):
                return float(np.sqrt(((a[er].astype(np.float64)
                                       - b[er].astype(np.float64)) ** 2).mean()))
            before, after = rmse(bad_pano, ref_pano), rmse(fixed_pano, ref_pano)
            check("applying the correction improves the mosaic",
                  after < 0.6 * before,
                  f"RMSE vs truth: {before:.1f} -> {after:.1f} (0-255)")
        else:
            check("enough shared coverage to compare", False, f"{er.sum()} px")

    # --- the confidence gate must actually fire ----------------------------------------
    # On the first pass the consensus is still a blur of misaligned views, so most
    # measurements have no dominant peak. That is exactly what the gate is for, and a
    # gate that never rejects anything is not a gate.
    s_gate = make_bare_stitcher(ps_mod, torch)
    _run_estimator(s_gate, drifted, intr, plane, dict(base_config, pose_refine=True),
                   passes=1)
    check("confidence gate rejects measurements with no dominant peak",
          s_gate._refine_stats.get("rejected", 0) > 0,
          f"pass 1: {s_gate._refine_stats.get('accepted')} accepted, "
          f"{s_gate._refine_stats.get('rejected')} rejected")

    # --- a periodic texture must not be made worse -------------------------------------
    # Deliberately NOT "must be rejected": run to convergence a repetitive scene can
    # settle into a self-consistent alignment that every view agrees with, and no
    # per-measurement confidence test can detect that (see PlanarStitcher's docstring).
    # The property that actually matters operationally is that the refiner does not
    # degrade a mosaic it cannot improve.
    checker = make_periodic_texture()
    checker_views, K2, s_tex2, plane2 = _nadir_scene(checker)
    intr2 = (K2[0, 0], K2[1, 1], K2[0, 2], K2[1, 2])
    drifted2 = []
    for v in checker_views:
        w = dict(v)
        e = injected.get(v["drone_id"])
        if e:
            w["pos"] = (v["pos"][0] + e[0], v["pos"][1] + e[1], v["pos"][2] + e[2])
        drifted2.append(w)

    cfg2 = dict(base_config, metres_per_pixel=s_tex2)
    s2 = make_bare_stitcher(ps_mod, torch)
    _run_estimator(s2, drifted2, intr2, plane2, dict(cfg2, pose_refine=True), passes=6)

    ref2, ok_r2, _ = _mosaic(make_bare_stitcher(ps_mod, torch),
                             checker_views, intr2, plane2, cfg2)
    bad2, ok_b2, _ = _mosaic(make_bare_stitcher(ps_mod, torch),
                             drifted2, intr2, plane2, cfg2)
    fixed2, ok_f2, _ = _mosaic(s2, drifted2, intr2, plane2, cfg2)
    if check("periodic-texture mosaics rendered", ok_r2 and ok_b2 and ok_f2,
             f"{ok_r2}/{ok_b2}/{ok_f2}"):
        cov2 = ref2.any(axis=2) & bad2.any(axis=2) & fixed2.any(axis=2)
        er2 = cv2.erode(cov2.astype(np.uint8), np.ones((21, 21), np.uint8)) > 0
        if er2.sum() > 5000:
            def rmse2(a, b):
                return float(np.sqrt(((a[er2].astype(np.float64)
                                       - b[er2].astype(np.float64)) ** 2).mean()))
            before2, after2 = rmse2(bad2, ref2), rmse2(fixed2, ref2)
            check("a periodic texture is not made worse by the refiner",
                  after2 <= before2 * 1.05,
                  f"RMSE vs truth: {before2:.1f} -> {after2:.1f} "
                  f"({s2._refine_stats.get('accepted')} accepted, "
                  f"{s2._refine_stats.get('rejected')} rejected)")
        else:
            check("enough shared coverage on the checkerboard", False, f"{er2.sum()} px")

    # --- both flags on -----------------------------------------------------------------
    # The two estimators are separable only when the per-drone error has no dilation
    # component. A plane-depth error scales each view's content about that view's own
    # footprint, and averaged over the overlap that is a convergent translation field --
    # so a convergent set of position errors is genuinely NOT distinguishable from a
    # depth error by any amount of image evidence. Both cases are worth pinning: a
    # separable one, where each estimator must land on its own quantity, and a degenerate
    # one, where the split is arbitrary but the mosaic must still come out right.
    def drift(base_views, errors):
        out = []
        for v in base_views:
            w = dict(v)
            e = errors.get(v["drone_id"])
            if e:
                w["pos"] = (v["pos"][0] + e[0], v["pos"][1] + e[1], v["pos"][2] + e[2])
            out.append(w)
        return out

    # Shear: drones on the x arm displaced along z, in opposite directions. Zero-mean and
    # pure rotation/shear, so it has no overlap with the dilation a depth error produces.
    shear = {1: (0.0, 0.0, 0.5), 4: (0.0, 0.0, -0.5)}
    sheared = drift(views, shear)
    wrong = dict(plane, plane_d=-true_error)

    s_both = make_bare_stitcher(ps_mod, torch)
    _run_estimator(s_both, sheared, intr, wrong,
                   dict(base_config, plane_sweep=True, pose_refine=True), passes=8)
    off_both = s_both._correction.get("plane")
    dp_both = s_both._correction.get("dpose")
    check("with both on, the sweep still recovers the plane error",
          off_both is not None and abs(off_both - true_error) < 0.35,
          f"recovered {off_both:+.3f} m of {true_error:+.3f} m" if off_both is not None
          else "no offset")
    if dp_both is not None:
        want_b = {v["drone_id"]: -pg.unity_dir_to_rh(shear.get(v["drone_id"],
                                                               (0.0, 0.0, 0.0)))
                  for v in views}
        common_b = [d for d in want_b if d in dp_both]
        wb_mean = sum(want_b[d] for d in common_b) / len(common_b)
        gb_mean = sum(dp_both[d][1] for d in common_b) / len(common_b)
        worst_b = max(float(np.linalg.norm((dp_both[d][1] - gb_mean)
                                           - (want_b[d] - wb_mean)))
                      for d in common_b)
        check("with both on, the refiner still recovers the per-drone error",
              worst_b < 0.25, f"worst {worst_b:.3f} m over {len(common_b)} views")
    else:
        check("with both on, the refiner still produces corrections", False,
              f"stats={s_both._refine_stats}")

    # Degenerate case: convergent position errors *plus* a depth error.
    #
    # Scored on overlap PSNR (how well the views agree with each *other*), not on RMSE
    # against the true mosaic. The estimators can only ever drive the views into mutual
    # agreement; when the error is degenerate they reach a self-consistent solution whose
    # absolute scale differs slightly from truth. That is a clean mosaic at a marginally
    # wrong scale -- a success for a pilot's situational-awareness view -- and an
    # RMSE-against-truth test would score it as a failure while the seams it is supposed
    # to be measuring had in fact improved.
    degen = drift(views, injected)

    def consistency(stitcher, vs, pl, cfg_extra=None, passes=0):
        if passes:
            _run_estimator(stitcher, vs, intr, pl, dict(base_config, **cfg_extra), passes)
        stitcher._plane_invalid_since = None
        stitcher.planar_pano(vs, intr, pl, dict(base_config))
        return stitcher._last_stats.get("overlap_psnr", float("nan"))

    psnr_bad = consistency(make_bare_stitcher(ps_mod, torch), degen, wrong)
    s_deg = make_bare_stitcher(ps_mod, torch)
    psnr_both = consistency(s_deg, degen, wrong,
                            {"plane_sweep": True, "pose_refine": True}, passes=8)
    check("plane + pose error together: the views are driven into agreement",
          np.isfinite(psnr_bad) and psnr_both > psnr_bad + 5.0,
          f"overlap PSNR {psnr_bad:.1f} -> {psnr_both:.1f} dB "
          f"(plane {s_deg._correction.get('plane'):+.2f} m; the split between plane and "
          f"pose is not identifiable here)")

    # The known limitation, asserted so it cannot be "fixed" by accident: a convergent
    # per-drone position error is indistinguishable from a depth error, so the sweep on
    # its own moves the plane and buys nothing. The refiner is what rescues this case,
    # which is the practical reason the two flags are separate.
    s_sweep_only = make_bare_stitcher(ps_mod, torch)
    psnr_sweep = consistency(s_sweep_only, degen, wrong, {"plane_sweep": True}, passes=8)
    check("the sweep alone cannot fix a convergent position error (known limitation)",
          psnr_sweep < psnr_bad + 2.0,
          f"overlap PSNR {psnr_bad:.1f} -> {psnr_sweep:.1f} dB after moving the plane "
          f"{s_sweep_only._correction.get('plane'):+.2f} m")

    # --- the low-pass rate must damp a TRACKING update -------------------------------
    # Acquisition deliberately snaps: it is the initial lock, not a refinement of one, and
    # low-passing it would leave the estimate outside the basin it just found, where the
    # fine scan has no signal to follow. So the rate is tested where it applies -- on a
    # tracking update, after a lock exists.
    s3 = make_bare_stitcher(ps_mod, torch)
    _run_estimator(s3, views, intr, dict(plane, plane_d=-true_error),
                   dict(base_config, plane_sweep=True, refine_rate=0.2), passes=1)
    locked = s3._correction.get("plane")
    if check("one pass acquires a lock", locked is not None and s3._sweep_mode == "track",
             f"offset {locked} mode {s3._sweep_mode}"):
        moved = 0.30                                  # inside the fine window
        _run_estimator(s3, views, intr, dict(plane, plane_d=-(true_error + moved)),
                       dict(base_config, plane_sweep=True, refine_rate=0.2), passes=1)
        damped = s3._correction.get("plane")
        check("refine rate damps a tracking update",
              damped is not None and locked < damped < locked + 0.6 * moved,
              f"plane moved {moved:.2f} m; one pass at rate 0.2 took the offset "
              f"{locked:+.3f} -> {damped:+.3f} m" if damped is not None else "no offset")

    # --- the scan step must resolve the basin it is searching --------------------------
    # THE invariant behind the original bug, measured against the real cost curve rather
    # than asserted from theory. To be sure of landing a candidate inside a minimum of
    # width W the step must be at most W/2; the defaults that shipped scanned +/-4 m with
    # 9 candidates, i.e. 1.0 m apart, against a basin of about a metre. The minimum was
    # therefore never resolved: the argmin was noise, the low-pass walked the plane a
    # metre per pass in an arbitrary direction, and toggling the estimator only re-rolled
    # the dice.
    #
    # Measured on BOTH textures because the basin is roughly L*Z/B and L is a property of
    # the scene. Note this synthetic nadir scene is a generous case -- smooth texture,
    # 26 m standoff, 5 m baselines. On the brick facade CLAUDE.md sizes (30 m behind a
    # 15 m wall) the basin is several times narrower, so the margin here is the best case,
    # not the typical one.
    K_probe = pg.intrinsics_matrix(*intr)
    grid_m = np.arange(-2.0, 2.0001, 0.05)
    old_m = 2.0 * base_config["sweep_range"] / (base_config["sweep_steps"] - 1)

    for tex_name, tex_views, tex_intr, tex_plane, tex_mpp in (
            ("smooth", views, intr, plane, s_tex),
            ("periodic", checker_views, intr2, plane2, s_tex2)):
        s_probe = make_bare_stitcher(ps_mod, torch)
        probe_cfg = dict(base_config, plane_sweep=True, metres_per_pixel=tex_mpp)
        K_p = pg.intrinsics_matrix(*tex_intr)
        curve = np.array([s_probe._photometric_cost(tex_views, K_p, tex_plane,
                                                    probe_cfg, float(o))[0]
                          for o in grid_m])
        finite = np.isfinite(curve)
        if not check(f"[{tex_name}] sweep cost curve evaluated", finite.sum() > 20,
                     f"{finite.sum()} samples"):
            continue

        # Width of the contiguous run around the argmin that stays below half way from
        # the minimum to the median -- i.e. the region a scan must land in to be pulled
        # toward the right answer rather than a noise sample.
        g, c = grid_m[finite], curve[finite]
        half = float(c.min()) + 0.5 * (float(np.median(c)) - float(c.min()))
        i0 = int(np.argmin(c))
        lo = hi = i0
        while lo > 0 and c[lo - 1] <= half:
            lo -= 1
        while hi < len(c) - 1 and c[hi + 1] <= half:
            hi += 1
        basin_m = float(g[hi] - g[lo])

        # The fine step in metres AT THIS SCENE's standoff and baseline -- the whole point
        # of sampling in disparity being that one constant in pixels is the right step at
        # every range, and no constant in metres is at any two.
        _, cams_p = s_probe._build_geometry(tex_views, K_p, tex_plane, probe_cfg,
                                            plane_offset=0.0, apply_dpose=False)
        Z = float(np.median([cc["plane_h"] for cc in cams_p]))
        ab = np.array([cc["plane_ab"] for cc in cams_p])
        d2 = ((ab[:, None, :] - ab[None, :, :]) ** 2).sum(axis=2)
        np.fill_diagonal(d2, np.inf)
        B = float(np.median(np.sqrt(d2.min(axis=1))))
        fine_m = Z * Z * ps_mod.PlanarStitcher.SWEEP_FINE_STEP_PX / (float(tex_intr[0]) * B)

        check(f"[{tex_name}] the fine scan step resolves the basin",
              0.0 < fine_m <= 0.5 * basin_m,
              f"basin {basin_m:.2f} m, so the step must be <= {0.5 * basin_m:.2f} m; "
              f"fine step {fine_m:.2f} m "
              f"({ps_mod.PlanarStitcher.SWEEP_FINE_STEP_PX} px at Z={Z:.0f} m, B={B:.0f} m)")
        check(f"[{tex_name}] the old metres-uniform step did not",
              old_m > 0.5 * basin_m,
              f"basin {basin_m:.2f} m, so the step must be <= {0.5 * basin_m:.2f} m; "
              f"old step {old_m:.2f} m "
              f"(+/-{base_config['sweep_range']} m over {base_config['sweep_steps']})")

    # --- a lock lost outside the fine window must be re-acquired ------------------------
    # The tracking scan spans a few pixels of disparity by design, so it cannot see an
    # error metres away. That is not a regression from the old wide-and-coarse scan, it
    # is the trade: the fine scan resolves the minimum, and the mode machine is what
    # supplies the capture range. Seeding TRACK at a wrong offset exercises the path an
    # operator previously had to trigger by hand.
    s_lost = make_bare_stitcher(ps_mod, torch)
    s_lost._sweep_mode = "track"
    s_lost._plane_offset = -2.5                       # far outside the fine window
    _run_estimator(s_lost, views, intr, dict(plane, plane_d=-true_error),
                   dict(base_config, plane_sweep=True, refine_rate=1.0),
                   passes=ps_mod.PlanarStitcher.SWEEP_LOST_PASSES + 6)
    recovered = s_lost._correction.get("plane")
    check("a lock seeded outside the fine window is re-acquired",
          recovered is not None and abs(recovered - true_error) < 0.4,
          f"seeded -2.50 m, recovered {recovered:+.3f} m of {true_error:+.3f} m "
          f"(mode now {s_lost._sweep_mode})" if recovered is not None else "no offset")

    # --- a gated-out measurement must not delete the correction it already earned -------
    # Rebuilding _pose_shift from this pass's accepted views only meant one weak
    # correlation peak deleted that drone's whole history and snapped its patch back to
    # the raw pose. Backwards, too: rejecting ALL views hit an early return that preserved
    # everything, so only a PARTIAL rejection destroyed anything.
    s_keep = make_bare_stitcher(ps_mod, torch)
    _run_estimator(s_keep, drifted, intr, plane, dict(base_config, pose_refine=True),
                   passes=6)
    before_shift = dict(s_keep._pose_shift)
    if check("refiner converged before the rejection pass", len(before_shift) >= 2,
             f"{len(before_shift)} views carrying a correction"):
        # PARTIAL rejection specifically: one view still measures, the rest are gated out.
        # Rejecting every view hits an early return that always preserved the set, so a
        # test that gates all of them exercises the path that was never broken.
        real_shift = ps_mod.PlanarStitcher._phase_shift
        calls = {"n": 0}

        def partial(a, b):
            calls["n"] += 1
            return real_shift(a, b) if calls["n"] == 1 else (0.0, 0.0, 1.0)

        s_keep._phase_shift = partial     # instance attribute shadows the staticmethod
        _run_estimator(s_keep, drifted, intr, plane, dict(base_config, pose_refine=True),
                       passes=1)
        after = s_keep._pose_shift

        # Compared as PAIRWISE DIFFERENCES: the accumulated set is re-gauged to zero mean
        # every pass, and that common translation is unobservable by construction (it is
        # removed on purpose so the mosaic cannot wander). What must survive a rejection
        # is each view's correction *relative to the others*.
        ids = sorted(before_shift)
        same = set(after) == set(before_shift)
        if same and len(ids) >= 2:
            d_before = [before_shift[d] - before_shift[ids[0]] for d in ids]
            d_after = [after[d] - after[ids[0]] for d in ids]
            worst_drift = max(float(np.linalg.norm(a - b))
                              for a, b in zip(d_after, d_before))
        else:
            worst_drift = float("inf")
        check("a gate-rejected view keeps its accumulated correction",
              same and worst_drift < 1e-6
              and s_keep._refine_stats.get("rejected", 0) > 0
              and s_keep._refine_stats.get("held", 0) > 0,
              f"{len(before_shift)} -> {len(after)} views, worst relative drift "
              f"{worst_drift:.2e} m, stats={s_keep._refine_stats}")

    # --- a view whose frame has stopped advancing must be dropped ----------------------
    # read_block_memory re-serves the previous frame for a busy block indefinitely, so
    # one view's pixels can freeze while the formation keeps moving. Both halves of the
    # evidence were already on the wire and were being decoded and thrown away.
    fresh = [dict(v, capture_time=100.0) for v in views]
    fresh[2]["capture_time"] = 100.0 - 10.0 * ps_mod.PlanarStitcher.MAX_CAPTURE_SKEW_S
    s_skew = make_bare_stitcher(ps_mod, torch)
    _mosaic(s_skew, fresh, intr, plane, base_config)
    check("a stale view is dropped from the solve",
          s_skew._last_stats.get("stale") == 1
          and s_skew._last_stats.get("views") == len(views) - 1,
          f"stale={s_skew._last_stats.get('stale')}, "
          f"views={s_skew._last_stats.get('views')} of {len(views)}")

    s_sync = make_bare_stitcher(ps_mod, torch)
    _mosaic(s_sync, [dict(v, capture_time=0.0) for v in views], intr, plane, base_config)
    check("a producer publishing no capture time drops nothing",
          s_sync._last_stats.get("stale") == 0
          and s_sync._last_stats.get("views") == len(views),
          f"stale={s_sync._last_stats.get('stale')}, "
          f"views={s_sync._last_stats.get('views')}")


def _pose_view(drone_id, pos, quat):
    """Minimal view record: _plane_from_formation reads only pos/quat/pose_status."""
    return {'slot': drone_id, 'drone_id': drone_id, 'heading': 0.0, 'image': None,
            'pos': pos, 'quat': tuple(float(c) for c in quat),
            'capture_time': 0.0, 'pose_status': 3, 'cached': False}


def test_formation_plane():
    """
    The FormationRelative plane: derived from the poses, with no raycast and no collider.

    This is the only plane source that can work on real drones, so it is checked against
    cases that actually distinguish its two rules rather than just the easy one.
    """
    print("\n11. FormationRelative plane (derived from poses, no raycast)")
    try:
        import torch  # noqa: F401
        import PlanarStitcher as ps_mod
    except ImportError as e:
        check("torch + PlanarStitcher importable", False, str(e))
        return

    s = ps_mod.PlanarStitcher.__new__(ps_mod.PlanarStitcher)
    s._plane_source = None

    # --- nadir cross, cameras straight down over ground at y = 0 ----------------------
    alt = 26.0
    nadir = [_pose_view(i, (dx, alt, dz), euler_to_quat(90.0, 0.0, 0.0))
             for i, (dx, dz) in enumerate(
                 [(0.0, 0.0), (-5.0, 0.0), (0.0, 5.0), (0.0, -5.0), (5.0, 0.0)])]
    n, d = s._plane_from_formation(nadir, alt)
    check("nadir: normal is world up",
          n is not None and np.allclose(n, [0.0, 1.0, 0.0], atol=1e-6),
          f"n = {np.round(n, 6) if n is not None else None}")
    check("nadir: standoff places the plane at the ground", abs(d - 0.0) < 1e-6,
          f"d = {d:.6f} (true plane d = 0)")
    check("nadir: the position fit is what resolved it", s._plane_source == "positions",
          f"source = {s._plane_source}")

    # --- vertical wall with the gimbal pitched DOWN 15 deg ----------------------------
    # The discriminating case. Fitting the camera positions is immune to gimbal pitch;
    # the mean-forward rule would tilt the plane by the full 15 deg. A facade is vertical
    # regardless of where the gimbal happens to be aimed, so "positions" must win here.
    wall = []
    k = 0
    for dx in (-8.0, -4.0, 0.0, 4.0, 8.0):
        for dy in (12.0, 18.0):
            wall.append(_pose_view(k, (dx, dy, 0.0), euler_to_quat(15.0, 0.0, 0.0)))
            k += 1
    n_w, d_w = s._plane_from_formation(wall, 30.0)
    check("wall + pitched gimbal: normal stays perpendicular to the facade",
          n_w is not None and np.allclose(n_w, [0.0, 0.0, 1.0], atol=1e-6),
          f"n = {np.round(n_w, 6) if n_w is not None else None} "
          "(the mean-forward rule would give ~[0, 0.26, 0.97])")
    check("wall + pitched gimbal: plane sits one standoff in front",
          abs(d_w - (-30.0)) < 1e-6, f"d = {d_w:.6f}, expected -30")

    # --- a single row of drones is degenerate and must fall back ----------------------
    row = [_pose_view(i, (dx, 15.0, 0.0), euler_to_quat(0.0, 0.0, 0.0))
           for i, dx in enumerate((-8.0, -4.0, 0.0, 4.0, 8.0))]
    n_r, d_r = s._plane_from_formation(row, 30.0)
    check("single row: falls back to the mean-forward rule",
          n_r is not None and s._plane_source == "forward",
          f"source = {s._plane_source}")
    check("single row: forward rule still gives a sane facade normal",
          n_r is not None and np.allclose(n_r, [0.0, 0.0, 1.0], atol=1e-6),
          f"n = {np.round(n_r, 6) if n_r is not None else None}")

    # --- refusals ---------------------------------------------------------------------
    check("no standoff is refused rather than guessed",
          s._plane_from_formation(nadir, 0.0)[0] is None, "returns None")
    check("no views is refused", s._plane_from_formation([], 30.0)[0] is None, "returns None")
    opposed = [_pose_view(0, (0.0, 20.0, 0.0), euler_to_quat(0.0, 0.0, 0.0)),
               _pose_view(1, (0.0, 20.0, 0.0), euler_to_quat(0.0, 180.0, 0.0))]
    check("cameras facing opposite ways are refused",
          s._plane_from_formation(opposed, 30.0)[0] is None,
          "no single surface they are all looking at")

    # --- end to end: the derived plane must mosaic identically to the published one ----
    if cv2 is None:
        check("cv2 available for the render comparison", False, "opencv not installed")
        return
    views, K, s_tex, plane = _nadir_scene(make_unique_texture())
    intr = (K[0, 0], K[1, 1], K[0, 2], K[1, 2])
    cfg = {"canvas": (1000, 700), "metres_per_pixel": s_tex, "max_range": 200.0,
           "feather_px": 40, "aniso_max": 12.0, "min_coverage": 0.2,
           "pose_source": 0, "psnr_gate": False,
           "blend_mode": ps_mod.BLEND_NEAREST, "debug_view": ps_mod.DEBUG_OFF,
           "plane_sweep": False, "pose_refine": False,
           "sweep_range": 4.0, "sweep_steps": 9,
           "refine_rate": 0.25, "refine_max_shift": 3.0,
           "standoff": 26.0}

    published, ok_p, _ = _mosaic(make_bare_stitcher(ps_mod, torch), views, intr, plane, cfg)
    derived, ok_d, reason_d = _mosaic(
        make_bare_stitcher(ps_mod, torch), views, intr,
        dict(plane, plane_mode=ps_mod.PLANE_MODE_FORMATION_RELATIVE), cfg)

    if check("both mosaics rendered", ok_p and ok_d, f"published={ok_p} derived={ok_d} "
             f"reason={reason_d}"):
        check("derived plane reproduces the published-plane mosaic exactly",
              np.array_equal(published, derived),
              "byte-identical" if np.array_equal(published, derived)
              else f"max diff {int(np.abs(published.astype(int) - derived.astype(int)).max())}")


def _mosaic(stitcher, views, intr, plane, config):
    """Render once with whatever ``stitcher._correction`` currently holds."""
    stitcher._plane_invalid_since = None
    return stitcher.planar_pano(views, intr, plane, dict(config))


def test_canvas_modes():
    """
    Canvas framing: FIXED must not have moved, AUTOFIT must frame and must not breathe,
    and zoom/pan must be a *viewing* transform that the estimators never see.

    The load-bearing check is the first one.  The whole premise of adding a canvas mode is
    that the operator can put the dropdown back on Fixed and have exactly what they had
    before, so that is asserted rather than asserted-to-be-obvious -- against a mosaic
    rendered through an independently-built centred canvas matrix, so it stays true
    without a stored golden image to rot.
    """
    print("\n12. Canvas modes (fixed / auto-fit) and the zoom-pan viewing transform")
    if cv2 is None:
        check("cv2 available", False, "opencv not installed; skipped")
        return
    try:
        import torch  # noqa: F401
        import PlanarStitcher as ps_mod
    except ImportError as e:
        check("torch + PlanarStitcher importable", False, f"{e}")
        return

    texture = make_unique_texture()
    views, K, s_tex, plane = _nadir_scene(texture)
    intr = (K[0, 0], K[1, 1], K[0, 2], K[1, 2])
    cw, ch = 1000, 700
    base = {
        "canvas": (cw, ch), "metres_per_pixel": s_tex, "max_range": 200.0,
        "feather_px": 40, "aniso_max": 12.0, "min_coverage": 0.2,
        "pose_source": 0, "psnr_gate": False,
        "blend_mode": ps_mod.BLEND_NEAREST, "debug_view": ps_mod.DEBUG_OFF,
        "plane_sweep": False, "pose_refine": False,
        "sweep_range": 4.0, "sweep_steps": 9,
        "refine_rate": 1.0, "refine_max_shift": 3.0,
    }

    def render(cfg):
        s = make_bare_stitcher(ps_mod, torch)
        pano, ok, reason = s.planar_pano(views, intr, plane, dict(base, **cfg))
        return s, pano, ok, reason

    # --- FIXED is exactly the historical framing ---------------------------------------
    # Two ways round, because they fail differently. A config carrying none of the new
    # keys is what an older caller (and every other test in this file) passes, so it must
    # still work; and spelling the defaults out must change nothing.
    _, pano_default, ok_d, reason_d = render({})
    _, pano_fixed, ok_f, _ = render({"canvas_mode": ps_mod.CANVAS_MODE_FIXED,
                                     "zoom": 1.0, "pan": (0.0, 0.0)})
    if not check("fixed-mode mosaic rendered", ok_d and ok_f, f"reason={reason_d}"):
        return
    check("a config with no canvas keys at all still renders the old framing",
          np.array_equal(pano_default, pano_fixed), "byte-identical")

    # The independent half: the mosaic must still land where a plainly-centred canvas
    # matrix says it should. This is what would catch a stray centre offset or a
    # reciprocal applied the wrong way round, which the comparison above cannot see
    # because both sides would move together.
    th, tw = texture.shape[:2]
    M_tex = pg.canvas_to_plane_matrix(s_tex, tw / 2.0, th / 2.0)

    def texture_rmse(pano, M_expected):
        covered = pano.any(axis=2)
        ref = cv2.warpPerspective(texture, np.linalg.inv(M_tex) @ M_expected, (cw, ch),
                                  flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP)
        er = cv2.erode(covered.astype(np.uint8), np.ones((15, 15), np.uint8)) > 0
        if er.sum() < 5000:
            return None
        a = cv2.GaussianBlur(pano, (0, 0), 1.5).astype(np.float64)
        b = cv2.GaussianBlur(ref, (0, 0), 1.5).astype(np.float64)
        return float(np.sqrt(((a[er] - b[er]) ** 2).mean()))

    rmse_fixed = texture_rmse(pano_fixed,
                              pg.canvas_to_plane_matrix(s_tex, cw * 0.5, ch * 0.5))
    check("fixed mode still frames on the plain centred canvas",
          rmse_fixed is not None and rmse_fixed < 6.0,
          f"RMSE {rmse_fixed:.3f} (0-255) vs an independently-built M")

    # --- zoom is a predictable viewing transform ---------------------------------------
    # Zoom must divide metres-per-pixel and nothing else: the same plane content, sampled
    # finer. Checked against the texture warped through the *predicted* matrix, so a zoom
    # that also translated (an easy sign error in the centre term) fails here.
    for z in (2.0, 4.0):
        _, pano_z, ok_z, _ = render({"zoom": z})
        rmse_z = None if not ok_z else texture_rmse(
            pano_z, pg.canvas_to_plane_matrix(s_tex / z, cw * 0.5, ch * 0.5))
        check(f"zoom {z:g}x samples the same plane finer, without translating",
              ok_z and rmse_z is not None and rmse_z < 6.0,
              f"RMSE {rmse_z:.3f}" if rmse_z is not None else "no mosaic")

    # --- pan moves the canvas centre by the amount claimed ------------------------------
    # Pan is a fraction of the canvas, so at zoom 2 a pan of 0.1 is 0.1 * cw * (mpp/2)
    # metres along e1. Small enough here to stay well inside the footprint, so the clamp
    # is not what is under test.
    z, pan_a = 2.0, 0.1
    mpp_v = s_tex / z
    ca = pan_a * cw * mpp_v
    _, pano_p, ok_p, _ = render({"zoom": z, "pan": (pan_a, 0.0)})
    rmse_p = None if not ok_p else texture_rmse(
        pano_p, pg.canvas_to_plane_matrix(mpp_v, cw * 0.5 - ca / mpp_v, ch * 0.5))
    check("pan shifts the canvas centre by exactly the fraction requested",
          ok_p and rmse_p is not None and rmse_p < 6.0,
          f"RMSE {rmse_p:.3f} at pan {pan_a} = {ca:.2f} m along e1")

    # --- zoom must NOT reach the estimators ---------------------------------------------
    # The one that protects the sweep's lock. The refiner is used because it produces a
    # per-view vector that is easy to compare exactly; the argument is identical for the
    # sweep, and both take their geometry from the same _estimator_geometry.
    err = {1: np.array([0.55, 0.0, 0.35]), 3: np.array([-0.4, 0.0, 0.5])}
    noisy = [dict(v, pos=tuple(np.asarray(v["pos"]) + err.get(v["drone_id"], 0.0)))
             for v in views]
    shifts = {}
    for z in (1.0, 4.0):
        s = make_bare_stitcher(ps_mod, torch)
        # The photometric refiner specifically: it is the estimator that measures on a
        # canvas, so it is the one a zoom could reach. planar_align never sees the canvas.
        was_legacy = ps_mod.LEGACY_ESTIMATORS
        ps_mod.LEGACY_ESTIMATORS = True
        try:
            _run_estimator(s, noisy, intr, plane,
                           dict(base, pose_refine=True, zoom=z), passes=3)
        finally:
            ps_mod.LEGACY_ESTIMATORS = was_legacy
        shifts[z] = dict(s._pose_shift)
    common = set(shifts[1.0]) & set(shifts[4.0])
    worst = max((float(np.linalg.norm(shifts[1.0][d] - shifts[4.0][d])) for d in common),
                default=float("inf"))
    check("the pose refiner measures the same correction at 1x and 4x zoom",
          len(common) >= 4 and worst < 1e-6,
          f"{len(common)} views, worst difference {worst:.2e} m")

    # --- auto-fit actually frames -------------------------------------------------------
    # Start deliberately mis-scaled: at 3x the correct metres-per-pixel the formation
    # occupies a ninth of the canvas, which is the "wrong standoff" case in miniature.
    coarse = dict(base, metres_per_pixel=s_tex * 3.0)
    s_fx = make_bare_stitcher(ps_mod, torch)
    s_fx.planar_pano(views, intr, plane, dict(coarse, canvas_mode=ps_mod.CANVAS_MODE_FIXED))
    cov_fixed = s_fx._last_stats.get("coverage", 0.0)

    s_af = make_bare_stitcher(ps_mod, torch)
    s_af.planar_pano(views, intr, plane, dict(coarse, canvas_mode=ps_mod.CANVAS_MODE_AUTOFIT))
    cov_auto = s_af._last_stats.get("coverage", 0.0)
    check("auto-fit fills a canvas the fixed scale leaves mostly empty",
          cov_auto > cov_fixed * 1.5,
          f"coverage {cov_fixed:.0%} -> {cov_auto:.0%} "
          f"({coarse['metres_per_pixel']*100:.1f} -> {s_af._fit_mpp*100:.1f} cm/px)")

    # --- auto-fit does not breathe ------------------------------------------------------
    # A steadily growing formation must produce a monotone staircase, never a dither. The
    # dwell is disabled so this exercises the quantiser and hysteresis rather than just
    # measuring how long the test took to run.
    s_seq = make_bare_stitcher(ps_mod, torch)
    s_seq.AUTOFIT_DWELL_S = 0.0
    cfg_af = dict(base, canvas_mode=ps_mod.CANVAS_MODE_AUTOFIT)
    scales = []
    for i in range(40):
        grow = 1.0 + 0.06 * i
        spread = [dict(v, pos=(v["pos"][0] * grow, v["pos"][1], v["pos"][2] * grow))
                  for v in views]
        s_seq._plane_invalid_since = None
        s_seq.planar_pano(spread, intr, plane, cfg_af)
        scales.append(s_seq._fit_mpp)
    steps = [b for a, b in zip(scales, scales[1:]) if b != a]
    deltas = np.diff([scales[0]] + steps)
    check("auto-fit steps monotonically under a growing formation, never dithering",
          len(steps) >= 1 and bool((deltas > 0).all()),
          f"{len(steps)} step(s) over 40 frames: "
          + " -> ".join(f"{v*100:.2f}" for v in [scales[0]] + steps) + " cm/px")

    # A quantiser that never holds is not a quantiser: the formation grows by 2.3x here,
    # so a continuous fit would produce ~40 distinct scales rather than a handful.
    check("auto-fit holds a scale rather than tracking continuously",
          len(steps) <= 6, f"{len(steps)} changes across 40 frames of steady growth")


# --------------------------------------------------------------------------------------
# 13. Learned alignment (planar_align)
# --------------------------------------------------------------------------------------

def make_panel_texture(h=1400, w=2000, panel=(81, 61), seed=3):
    """
    A facade of near-identical louvre panels with a few irregular windows.

    The stand-in for the MED building: every panel carries the same slat pattern, so any
    alignment a whole number of panels out matches almost everything, and only the
    windows (about one panel in nine, placed at random) say which of them is right.
    """
    rng = np.random.default_rng(seed)
    pw, ph = panel
    yy, xx = np.mgrid[0:h, 0:w]
    slat = ((yy % ph) // 4 % 2).astype(np.float32)                 # horizontal slats
    frame = ((xx % pw) < 4) | ((yy % ph) < 3)                      # panel frames
    img = 150.0 + 45.0 * slat
    img[frame] = 70.0
    for r in range(h // ph):
        for c in range(w // pw):
            if rng.random() < 0.11:
                img[r * ph + 6:(r + 1) * ph - 3, c * pw + 8:(c + 1) * pw - 4] = 35.0
    img += rng.normal(0.0, 3.0, img.shape)
    grey = np.clip(img, 0, 255).astype(np.uint8)
    return np.stack([grey] * 3, axis=-1)


def _true_seam_error(stitcher, published, truth, K, plane, config, span=14.0):
    """
    How far apart the stitcher's geometry puts the *same* ground point as seen by two
    views, in metres on the plane, over a grid of points -- ``(median, worst)``.

    The correspondences come from the true poses, so this needs no matching and has no
    gauge: a mosaic that is rigidly shifted or scaled as a whole still scores zero, which
    is right, because nothing in the images can tell and nothing on screen shows it.
    """
    frame, cams = stitcher._build_geometry(published, K, plane, config)
    if frame is None:
        return float("nan"), float("nan")
    G = {c["view"]["drone_id"]: c["G"] for c in cams}
    xs = np.linspace(-span, span, 15)
    X = np.array([pg.unity_point_to_rh((x, 0.0, z)) for x in xs for z in xs])
    h, w = truth[0]["image"].shape[:2]
    placed = {}
    for v in truth:
        if v["drone_id"] not in G:
            continue
        R, C = pg.unity_pose_to_cv(v["pos"], v["quat"])
        uv, depth = pg.project_points(K, R, C, X)
        inside = (depth > 0) & (uv[:, 0] > 8) & (uv[:, 0] < w - 9) \
            & (uv[:, 1] > 8) & (uv[:, 1] < h - 9)
        q = np.linalg.solve(G[v["drone_id"]], np.column_stack([uv, np.ones(len(uv))]).T)
        placed[v["drone_id"]] = ((q[:2] / q[2]).T, inside)
    errs = []
    ids = sorted(placed)
    for a in range(len(ids)):
        for b in range(a + 1, len(ids)):
            pa, ia = placed[ids[a]]
            pb_, ib = placed[ids[b]]
            both = ia & ib
            if both.any():
                errs.append(np.linalg.norm(pa[both] - pb_[both], axis=1))
    if not errs:
        return float("nan"), float("nan")
    errs = np.concatenate(errs)
    return float(np.median(errs)), float(errs.max())


def test_learned_alignment():
    """
    ``planar_align``: the solver and the shift graph on synthetic data with a known
    answer, the render path's use of the result, and -- when the networks can be loaded
    -- the whole estimator on rendered views, including a facade of repeating panels.
    """
    print("\n13. Learned alignment (planar_align)")
    try:
        import torch
        import planar_align as pa
        import PlanarStitcher as ps_mod
    except ImportError as e:
        check("torch + planar_align importable", False, f"{e}")
        return

    rng = np.random.default_rng(11)

    # ---- the bundle adjustment, on matches with a known answer ------------------------
    # A wall of five cameras 30 m from the plane z = 0, looking along -z.
    K = np.array([[525.0, 0.0, 400.0], [0.0, 525.0, 225.0], [0.0, 0.0, 1.0]])
    look = np.diag([1.0, -1.0, -1.0])                    # world -> camera, facing -z
    C_true = [np.array([x, y, 30.0]) for x, y in
              ((-9.0, 0.0), (0.0, 0.5), (9.0, 0.0), (-4.0, 7.0), (5.0, 7.5))]
    R_true = [pa.rotvec_to_matrix(rng.normal(0.0, 0.004, 3)) @ look for _ in C_true]
    n_true, d_true = np.array([0.0, 0.0, 1.0]), 0.0

    def project(R, C, X):
        pc = (X - C) @ R.T
        pr = pc @ K.T
        return pr[:, :2] / pr[:, 2:3]

    def inside(uv):
        return (uv[:, 0] > 0) & (uv[:, 0] < 799) & (uv[:, 1] > 0) & (uv[:, 1] < 449)

    pairs, clean = [], []
    N = len(C_true)
    for i in range(N):
        for j in range(i + 1, N):
            X = np.column_stack([rng.uniform(-30, 30, 400), rng.uniform(-14, 22, 400),
                                 np.zeros(400)])
            ui, uj = project(R_true[i], C_true[i], X), project(R_true[j], C_true[j], X)
            keep = inside(ui) & inside(uj)
            ui, uj = ui[keep], uj[keep]
            if len(ui) < 20:
                continue
            clean.append((i, j, ui.copy(), uj.copy()))
            noisy_i = ui + rng.normal(0.0, 0.5, ui.shape)
            noisy_j = uj + rng.normal(0.0, 0.5, uj.shape)
            # A quarter of the matches are wrong: half of those are content 12 m behind
            # the plane (self-consistent, the dangerous kind), half are plain mismatches.
            k = len(ui) // 4
            Xb = np.column_stack([rng.uniform(-25, 25, k), rng.uniform(-10, 18, k),
                                  np.full(k, -12.0)])
            bi, bj = project(R_true[i], C_true[i], Xb), project(R_true[j], C_true[j], Xb)
            noisy_i[:len(bi) // 2] = bi[:len(bi) // 2]
            noisy_j[:len(bi) // 2] = bj[:len(bi) // 2]
            noisy_j[len(bi) // 2:k] += rng.uniform(-60, 60, (k - len(bi) // 2, 2))
            pairs.append((i, j, noisy_i, noisy_j))

    # What is published: every pose metres and degrees out, and a plane that is tilted
    # and at the wrong distance -- the real-drone situation.
    w_err = [np.deg2rad(rng.normal(0.0, [1.0, 3.0, 1.0])) for _ in range(N)]
    t_err = [rng.normal(0.0, [1.2, 1.0, 1.2]) for _ in range(N)]
    R_pub = [pa.rotvec_to_matrix(w) @ R for w, R in zip(w_err, R_true)]
    C_pub = [C + t for C, t in zip(C_true, t_err)]
    tilt = pa.rotvec_to_matrix(np.deg2rad([0.0, 8.0, 0.0]))
    n_pub = tilt @ n_true
    d_pub = d_true + 3.0

    def transfer_error(ba, x, matches):
        probe = pa.PlanarBA(K, R_pub, C_pub, n_pub, d_pub, np.zeros(3), matches)
        r, _ = probe.residual(torch.as_tensor(np.asarray(x), dtype=pa.DT))
        return float(np.median(np.linalg.norm(r.numpy(), axis=1)))

    ba = pa.PlanarBA(K, R_pub, C_pub, n_pub, d_pub, np.zeros(3), pairs)
    x_any = torch.as_tensor(rng.normal(0.0, 0.03, ba.P), dtype=pa.DT)
    worst = max(float((a - b).abs().max()) / max(1.0, float(b.abs().max()))
                for a, b in zip(ba.jacobian_blocks(x_any), ba.jacobian_autograd(x_any)))
    check("analytic Jacobian matches forward-mode autograd", worst < 1e-9,
          f"worst relative difference {worst:.1e}")

    prior = ba.prior_precision(5.0, 2.0, 2.5, 2.0, 12.0, 6.0)
    before = transfer_error(ba, np.zeros(ba.P), clean)
    x, err, ok = ba.solve([(prior, np.zeros(ba.P))])
    after = transfer_error(ba, x.numpy(), clean)
    check("bundle adjustment aligns the views through 25% wrong matches",
          before > 15.0 and after < 0.5,
          f"true-correspondence transfer error {before:.1f} px -> {after:.2f} px")

    _om, _tt, n_est, d_est = ba.unpack(x)
    # The plane is only defined up to the similarity the priors pin, so compare its
    # orientation to the *corrected cameras'* mean forward rather than to world axes.
    tilt_left = float(np.rad2deg(np.arccos(np.clip(abs(n_est @ n_pub), -1, 1))))
    check("the plane's orientation is solved, not just its distance",
          tilt_left > 4.0, f"moved {tilt_left:.1f} deg from the published normal "
                           f"(it was published 8 deg out)")

    ba_pose = pa.PlanarBA(K, R_pub, C_pub, n_pub, d_pub, np.zeros(3), pairs,
                          solve_plane=False)
    xp, _, _ = ba_pose.solve([(prior, np.zeros(ba.P))])
    ba_plane = pa.PlanarBA(K, R_pub, C_pub, n_pub, d_pub, np.zeros(3), pairs,
                           solve_pose=False)
    xq, _, _ = ba_plane.solve([(prior, np.zeros(ba.P))])
    check("each switch leaves the other's parameters untouched",
          float(xp[6 * N:].abs().max()) == 0.0 and float(xq[:6 * N].abs().max()) == 0.0,
          f"plane moved {float(xp[6 * N:].abs().max()):.2e} with the plane off, "
          f"poses moved {float(xq[:6 * N].abs().max()):.2e} with the poses off")

    empty = pa.PlanarBA(K, R_pub, C_pub, n_pub, d_pub, np.zeros(3), [])
    xe, _, _ = empty.solve([(prior, np.zeros(empty.P))])
    check("with no matches the priors hold everything at the published geometry",
          float(xe.abs().max()) < 1e-9, f"largest correction {float(xe.abs().max()):.1e}")

    # ---- the shift graph: loops resolve what no pair can ------------------------------
    # Four views of a periodic facade. Every pair's surface has peaks one period (8
    # cells) apart; on two of the six edges the WRONG peak is the highest, as it is on
    # the real building when the overlap happens to hold more panels than windows.
    Rn = 12
    true_shift = np.array([[0, 0], [-4, 1], [5, -1], [-3, -2]])       # (dx, dy) cells

    def surface(d_true, wrong_first):
        R = 2 * Rn
        S = np.zeros((2 * R + 1, 2 * R + 1))
        for k, height in ((0, 0.30), (-8, 0.24), (8, 0.22)):
            if wrong_first and k == 8:
                height = 0.34
            y, x = R + d_true[1], R + d_true[0] + k
            if 0 <= x <= 2 * R:
                S[y, x] = height
        return cv2.GaussianBlur(S, (0, 0), 1.0) * 6.0 if cv2 is not None else S

    edges = []
    wrong = {(0, 2), (1, 3)}
    for i in range(4):
        for j in range(i + 1, 4):
            edges.append((i, j, surface(true_shift[i] - true_shift[j], (i, j) in wrong)))
    naive_wrong = 0
    for i, j, S in edges:
        R = (S.shape[0] - 1) // 2
        y, x = np.unravel_index(int(np.argmax(S)), S.shape)
        naive_wrong += int((x - R, y - R) != tuple(true_shift[i] - true_shift[j]))
    sh, _score = pa.solve_shift_graph(4, edges, Rn, 1e-4)
    rel = sh - sh[0]
    check("the view graph picks the consistent alignment where single pairs are a "
          "period out",
          naive_wrong == 2 and np.array_equal(rel, true_shift - true_shift[0]),
          f"{naive_wrong} of {len(edges)} pairs peak on the wrong panel; graph shifts "
          f"{rel.tolist()} vs truth {(true_shift - true_shift[0]).tolist()}")

    # ---- period verification: equal evidence goes to where GNSS put the view ----------
    # Two rows of two views looking at identical storeys (8 cells). Within a row a pair
    # scores one alignment only; between the rows every storey scores the same. The
    # solve has left the top row a storey from where it was published (`centres`), so
    # the answer is to move that row back, together -- not to leave it, and not to move
    # one view of it.
    def flat(d_true, periods):
        R = 2 * Rn
        S = np.zeros((2 * R + 1, 2 * R + 1))
        for k in periods:
            S[R + d_true[1] + k, R + d_true[0]] = 1.0
        return S

    rows = [0, 0, 1, 1]
    edges = [(i, j, flat((0, 0), (0,) if rows[i] == rows[j] else (-16, -8, 0, 8, 16)))
             for i in range(4) for j in range(i + 1, 4)]
    centres = np.array([[0, 0], [0, 0], [0, 8], [0, 8]], dtype=float)
    sh, _score = pa.solve_shift_graph(4, edges, Rn, 1e-3, prior_centres=centres)
    check("where the images cannot tell storeys apart, the row goes back to where GNSS "
          "put it",
          np.array_equal(sh - sh[0], [[0, 0], [0, 0], [0, 8], [0, 8]]),
          f"shifts {sh.tolist()}")

    # ---- the render path's use of a correction ----------------------------------------
    s = make_bare_stitcher(ps_mod, torch)
    clock = {"t": 0.0}
    s._clock = lambda: clock["t"]
    old = {"dpose": {0: (np.eye(3), np.zeros(3))}, "rotvec": {0: np.zeros(3)},
           "plane_fix": ("world", np.array([0.0, 0.0, 1.0]), 0.0)}
    new_w = np.array([0.0, 0.04, 0.0])
    s._correction = {"dpose": {0: (pa.rotvec_to_matrix(new_w), np.array([1.0, 0.0, 0.0]))},
                     "rotvec": {0: new_w}, "plane": None,
                     "plane_fix": ("world", np.array([0.0, 0.0, 1.0]), 2.0),
                     "blend": (0.0, 0.4, old)}
    at = {}
    for t in (0.0, 0.2, 0.4):
        clock["t"] = t
        c = s._live_correction()
        at[t] = (float(c["dpose"][0][1][0]), float(c["plane_fix"][2]))
    check("a new alignment is eased in, not switched to",
          at[0.0] == (0.0, 0.0) and abs(at[0.2][0] - 0.5) < 1e-9
          and abs(at[0.2][1] - 1.0) < 1e-9 and at[0.4] == (1.0, 2.0),
          f"shift, plane offset at t = 0 / half / end: {at[0.0]} {at[0.2]} {at[0.4]}")

    if cv2 is None:
        check("cv2 available for the rendered checks", False, "opencv not installed")
        return

    # Exposure: the same ground from two aircraft, one of them exposing 25% darker.
    texture = make_unique_texture()
    views, K_n, s_tex, plane = _nadir_scene(texture, offsets=((-6.0, 0.0), (6.0, 0.0)))
    intr = (K_n[0, 0], K_n[1, 1], K_n[0, 2], K_n[1, 2])
    dark = [dict(views[0]),
            dict(views[1], image=(views[1]["image"].astype(np.float32) * 0.75)
                 .astype(np.uint8))]
    cfg = {"canvas": (1000, 700), "metres_per_pixel": s_tex, "max_range": 200.0,
           "feather_px": 40, "aniso_max": 12.0, "min_coverage": 0.05, "pose_source": 0,
           "psnr_gate": False, "blend_mode": ps_mod.BLEND_NEAREST,
           "debug_view": ps_mod.DEBUG_OFF, "plane_sweep": False, "pose_refine": False}

    def seam_step(pano):
        # The partition's seam is the line midway between the two cameras. The canvas is
        # centred under the reference view (the first, at x = -6), so that line is 6 m
        # to the right of the canvas centre. Means over a band either side: the texture
        # is noise, and a narrow strip would measure the texture.
        g = pano.astype(np.float32).mean(axis=2)
        mid = pano.shape[1] // 2 + int(round(6.0 / s_tex))
        rows = slice(pano.shape[0] // 2 - 150, pano.shape[0] // 2 + 150)
        return abs(float(g[rows, mid - 40:mid - 3].mean() - g[rows, mid + 3:mid + 40].mean()))

    s_n = make_bare_stitcher(ps_mod, torch)
    hard = _mosaic(s_n, dark, intr, plane, cfg)[0]
    s_m = make_bare_stitcher(ps_mod, torch)
    soft = None
    for _ in range(40):                                  # let the gain low-pass settle
        soft = _mosaic(s_m, dark, intr, plane,
                       dict(cfg, blend_mode=ps_mod.BLEND_MULTIBAND))[0]
    step_hard, step_soft = seam_step(hard), seam_step(soft)
    check("multiband removes the exposure step a hard seam draws",
          step_hard > 15.0 and step_soft < 0.25 * step_hard,
          f"brightness step across the seam {step_hard:.1f} -> {step_soft:.1f} grey levels")

    # Where the geometry is exact and the exposure equal, the two-band composite must
    # give back what winner-take-all gives: it is a re-weighting of low frequencies, and
    # with nothing to reconcile there is nothing for it to change.
    s_a, s_b = make_bare_stitcher(ps_mod, torch), make_bare_stitcher(ps_mod, torch)
    ref = _mosaic(s_a, views, intr, plane, cfg)[0]
    for _ in range(3):
        mb = _mosaic(s_b, views, intr, plane,
                     dict(cfg, blend_mode=ps_mod.BLEND_MULTIBAND))[0]
    covered = ref.max(axis=2) > 0
    diff = np.abs(ref.astype(np.float32) - mb.astype(np.float32))[covered]
    check("multiband changes nothing where the views already agree",
          float(diff.mean()) < 1.0, f"mean |difference| {float(diff.mean()):.2f} grey levels")

    # ---- the whole estimator, on rendered views ----------------------------------------
    aligner = pa.PlanarAligner()
    if not aligner.available():
        print(f"  [SKIP] end-to-end alignment: {aligner.models.error}")
        return
    if aligner.models.lg is None:
        print(f"  [note] {aligner.models.error}")

    was_legacy = ps_mod.LEGACY_ESTIMATORS
    ps_mod.LEGACY_ESTIMATORS = False
    try:
        offsets = ((0.0, 0.0), (-9.0, 0.0), (9.0, 0.0), (-4.5, 7.0), (4.5, 7.0), (0.0, -7.0))
        errors = [((1.4, 0.0, -0.9), 2.5), ((-1.1, 0.0, 1.3), -3.0), ((0.6, 0.0, 1.6), 1.5),
                  ((-1.5, 0.0, -0.7), 0.0), ((0.9, 0.0, 0.8), -2.0), ((0.2, 0.0, -1.2), 3.5)]

        def publish(truth):
            out = []
            for v, (dp, yaw) in zip(truth, errors):
                pos = tuple(float(a + b) for a, b in zip(v["pos"], dp))
                out.append(dict(v, pos=pos,
                                quat=tuple(float(c) for c in euler_to_quat(90.0, yaw, 0.0))))
            return out

        est_cfg = dict(cfg, min_coverage=0.2, plane_sweep=True, pose_refine=True,
                       sweep_range=4.0, sweep_steps=9, refine_rate=1.0,
                       refine_max_shift=3.0, standoff=29.0)

        # (a) unique texture, formation-relative plane with the standoff typed 3 m out.
        truth, K_n, s_tex, plane = _nadir_scene(make_unique_texture(), offsets=offsets)
        intr = (K_n[0, 0], K_n[1, 1], K_n[0, 2], K_n[1, 2])
        formation = dict(plane, plane_mode=ps_mod.PLANE_MODE_FORMATION_RELATIVE)
        pub = publish(truth)
        s1 = make_bare_stitcher(ps_mod, torch)
        s1.ALIGN_MIN_PERIOD_S = 0.0
        s1.ALIGN_BLEND_S = 0.0
        med0, worst0 = _true_seam_error(s1, pub, truth, K_n, formation, est_cfg)
        _run_estimator(s1, pub, intr, formation, est_cfg, passes=3)
        med1, worst1 = _true_seam_error(s1, pub, truth, K_n, formation, est_cfg)
        check("learned alignment closes metre-scale pose error and a wrong standoff",
              med0 > 1.0 and med1 < 0.10 and worst1 < 0.35,
              f"true seam {med0:.2f} m (worst {worst0:.2f}) -> {med1:.3f} m "
              f"(worst {worst1:.3f}); {s1._align_stats.get('inliers')} inliers")
        check("after its first lock the aligner tracks rather than re-acquires",
              s1._align_stats.get("mode") == "track",
              f"mode {s1._align_stats.get('mode')}: {s1._align_stats.get('held', '')}")

        _run_estimator(s1, pub, intr, formation,
                       dict(est_cfg, plane_sweep=False, pose_refine=False), passes=1)
        check("switching both estimators off releases the learned correction",
              s1._correction.get("dpose") is None
              and s1._correction.get("plane_fix") is None and not s1._aligner.pose,
              f"correction {sorted(k for k, v in s1._correction.items() if v is not None)}")

        # (b) exact poses must be left alone: a refiner that "improves" a perfect mosaic
        #     is one that will also invent corrections in the simulator.
        s2 = make_bare_stitcher(ps_mod, torch)
        s2.ALIGN_MIN_PERIOD_S = 0.0
        s2.ALIGN_BLEND_S = 0.0
        _run_estimator(s2, truth, intr, plane, est_cfg, passes=2)
        med2, worst2 = _true_seam_error(s2, truth, truth, K_n, plane, est_cfg)
        check("exact poses stay exact under the aligner",
              worst2 < 0.08, f"true seam {med2:.3f} m (worst {worst2:.3f})")

        # (c) the repeating facade: per-drone errors larger than half a panel, so the
        #     nearest consistent-looking alignment for some pairs is a panel out.
        if aligner.models.lg is None:
            print("  [SKIP] repeating facade: needs LightGlue")
            return
        truth, K_n, s_tex, plane = _nadir_scene(make_panel_texture(), offsets=offsets)
        formation = dict(plane, plane_mode=ps_mod.PLANE_MODE_FORMATION_RELATIVE)
        big = [((2.3, 0.0, -0.9), 1.0), ((-2.1, 0.0, 1.0), -1.5), ((1.9, 0.0, 1.2), 1.0),
               ((-2.4, 0.0, -0.6), 0.0), ((2.2, 0.0, 0.7), -1.0), ((-1.8, 0.0, -1.1), 1.5)]
        errors = big
        pub = publish(truth)
        s3 = make_bare_stitcher(ps_mod, torch)
        s3.ALIGN_MIN_PERIOD_S = 0.0
        s3.ALIGN_BLEND_S = 0.0
        med0, worst0 = _true_seam_error(s3, pub, truth, K_n, formation, est_cfg)
        _run_estimator(s3, pub, intr, formation, est_cfg, passes=3)
        med3, worst3 = _true_seam_error(s3, pub, truth, K_n, formation, est_cfg)
        check("a facade of repeating panels is aligned, not locked a panel out",
              med0 > 2.0 and med3 < 0.15 and worst3 < 0.6,
              f"true seam {med0:.2f} m (worst {worst0:.2f}) -> {med3:.3f} m "
              f"(worst {worst3:.3f}), panel period {81 * s_tex:.1f} m; "
              f"{s3._align_stats.get('note', s3._align_stats.get('held', ''))}")

        # (d) a picture that lags its telemetry. One drone's video arrives half a second
        #     late, as a weak link's does, so while the formation moves its published
        #     pose is a metre ahead of what its frame shows -- and not at all while it
        #     hovers. A bias cannot follow that; the images' own motion can.
        _test_lagging_video(ps_mod, torch, est_cfg)
    finally:
        ps_mod.LEGACY_ESTIMATORS = was_legacy


def _test_lagging_video(ps_mod, torch, est_cfg):
    texture = make_unique_texture()
    base = ((0.0, 0.0), (-9.0, 0.0), (9.0, 0.0))
    dt, lag_frames, speed = 0.05, 10, 2.0

    def travelled(k):
        # At rest for five frames, then a constant 2 m/s.
        return speed * dt * max(0, k - 5)

    def frame(k):
        """(published views, the same with the pose each picture was really taken at)."""
        now = tuple((x + travelled(k), z) for x, z in base)
        then = tuple((x + travelled(k - lag_frames), z) for x, z in base)
        v_now, K_n, s_tex, plane = _nadir_scene(texture, offsets=now)
        v_then, _, _, _ = _nadir_scene(texture, offsets=then)
        published, truth = [], []
        for i in range(3):
            late = i == 2
            image = (v_then if late else v_now)[i]["image"]
            published.append(dict(v_now[i], image=image, capture_time=1.0 + k * dt))
            truth.append(dict((v_then if late else v_now)[i], image=image))
        return published, truth, K_n, plane

    worst = {}
    for visual in (False, True):
        s = make_bare_stitcher(ps_mod, torch)
        s.ALIGN_MIN_PERIOD_S = 0.0
        s.ALIGN_BLEND_S = 0.0
        s.VISUAL_TRACK = visual
        s.POSE_FILTER = False
        cfg = dict(est_cfg, refine_rate=0.25)
        s.set_live_config(cfg)
        errs = []
        for k in range(0, 30):
            published, truth, K_n, plane = frame(k)
            intr = (K_n[0, 0], K_n[1, 1], K_n[0, 2], K_n[1, 2])
            s._plane_invalid_since = None
            s.planar_pano(published, intr, plane, cfg)
            if k % 3 == 0:
                s.compute_warps()
                s._plane_invalid_since = None
                s.planar_pano(published, intr, plane, cfg)
            if k >= 12:                                   # moving, and the lag has opened
                errs.append(_true_seam_error(s, published, truth, K_n, plane, cfg)[1])
        worst[visual] = max(errs) if errs else float("nan")
    check("a drone whose picture lags its telemetry stays aligned while the formation moves",
          worst[True] < 0.2 and worst[False] > 3.0 * worst[True],
          f"worst true seam while moving: {worst[False]:.2f} m placing frames by pose, "
          f"{worst[True]:.3f} m carrying the alignment with the images "
          f"(the lag is worth {speed * dt * lag_frames:.1f} m)")


# --------------------------------------------------------------------------------------
# 14. Output raster and the one-view-per-pixel render
# --------------------------------------------------------------------------------------

def _oblique_scene(texture, W=800, H=450, vfov=46.4):
    """
    Six views of a ground plane from mixed heights, headings and pitches.

    Their footprints are trapezoids of different sizes at different angles, which is what
    the per-view bounding boxes and the winner-take-all scoring have to get right -- and
    what five identical nadir views never ask of them.  Returns ``(views, K, s_tex, plane)``.
    """
    K = intrinsics_from_unity(vfov, W, H)
    th, tw = texture.shape[:2]
    s_tex = 26.0 / float(K[1, 1])
    n_unity = (0.0, 1.0, 0.0)
    frame = pg.build_plane_frame(
        pg.unity_dir_to_rh(n_unity), 0.0, pg.unity_point_to_rh((0.0, 0.0, 0.0)),
        Camera((0.0, 26.0, 0.0), euler_to_quat(90, 0, 0), vfov, W, H).right_rh)
    M_tex = pg.canvas_to_plane_matrix(s_tex, tw / 2.0, th / 2.0)

    poses = [((0.0, 26.0, 0.0), 90.0, 0.0), ((-7.0, 24.0, 1.0), 82.0, 20.0),
             ((6.0, 28.0, -2.0), 97.0, -15.0), ((1.0, 30.0, 6.0), 90.0, 35.0),
             ((-2.0, 22.0, -6.0), 78.0, -30.0), ((8.0, 25.0, 5.0), 85.0, 60.0)]
    views = []
    for i, (pos, pitch, yaw) in enumerate(poses):
        quat = euler_to_quat(pitch, yaw, 0.0)
        cam = Camera(pos, quat, vfov, W, H)
        H_tex = pg.homography_canvas_to_image(
            pg.build_G(cam.K, cam.R_cv, cam.C, frame), M_tex)
        views.append({
            'slot': i, 'drone_id': i, 'heading': 0.0,
            'image': cv2.warpPerspective(texture, H_tex, (W, H), flags=cv2.INTER_LINEAR),
            'pos': pos, 'quat': tuple(float(c) for c in quat),
            'capture_time': 0.0, 'pose_status': 3, 'cached': False,
        })
    plane = {"plane_normal": n_unity, "plane_d": 0.0, "plane_valid": True,
             "plane_mode": 1, "gimbal_pitch": -90.0, "centre_drone_id": 0}
    return views, K, s_tex, plane


def _raster_to_canvas(canvas_w, canvas_h, out_w, out_h, bottom_up):
    """
    Output pixel -> canvas pixel, written out independently of the stitcher: pixel
    *areas* line up, so output pixel X is centred on canvas ``(X + 0.5) * cw / W - 0.5``,
    and a bottom-up raster's row r is the picture's row ``H - 1 - r``.
    """
    sx, sy = canvas_w / out_w, canvas_h / out_h
    S = np.array([[sx, 0.0, 0.5 * sx - 0.5], [0.0, sy, 0.5 * sy - 0.5], [0.0, 0.0, 1.0]])
    if bottom_up:
        S = S @ np.array([[1.0, 0.0, 0.0], [0.0, -1.0, out_h - 1.0], [0.0, 0.0, 1.0]])
    return S


def _reference_mosaic(cams, M, S, out_w, out_h, feather_px, max_range, blend):
    """
    The mosaic computed the slow, obvious way, in float64 numpy: every view over every
    pixel, no bounding boxes, no stacked texture, nothing shared with the stitcher's
    render but the formulas in its docstrings.

    Returns ``(float BGR [H, W, 3], margin [H, W])``; ``margin`` is how far the winning
    score is ahead of the runner-up, relative to it, so the caller can set aside the
    pixels where float32 may legitimately pick the other side of a seam.
    """
    X, Y = np.meshgrid(np.arange(out_w, dtype=np.float64), np.arange(out_h, dtype=np.float64))
    canvas = np.tensordot(S, np.stack([X, Y, np.ones_like(X)]), axes=1)
    plane = np.tensordot(np.asarray(M, dtype=np.float64), canvas, axes=1)
    weights, scores, colours = [], [], []
    for cam in cams:
        img = cam["view"]["image"].astype(np.float64)
        h, w = img.shape[:2]
        uvw = np.tensordot(np.asarray(cam["H"], dtype=np.float64), canvas, axes=1)
        ok = (uvw[2] > 1e-6) & (uvw[2] < max_range)
        depth = np.where(ok, uvw[2], 1.0)
        u, v = uvw[0] / depth, uvw[1] / depth
        border = np.minimum(np.minimum(u, w - 1 - u), np.minimum(v, h - 1 - v))
        inside = ok & (border > 0)
        fade = np.clip(border / feather_px, 0.0, 1.0)
        u0 = np.clip(np.floor(u), 0, w - 2).astype(int)
        v0 = np.clip(np.floor(v), 0, h - 2).astype(int)
        fu = np.clip(u - u0, 0.0, 1.0)[..., None]
        fv = np.clip(v - v0, 0.0, 1.0)[..., None]
        sample = ((img[v0, u0] * (1 - fu) + img[v0, u0 + 1] * fu) * (1 - fv)
                  + (img[v0 + 1, u0] * (1 - fu) + img[v0 + 1, u0 + 1] * fu) * fv)
        da, db = plane[0] - cam["plane_ab"][0], plane[1] - cam["plane_ab"][1]
        cos = cam["plane_h"] / np.sqrt(da * da + db * db + cam["plane_h"] ** 2)
        weights.append(np.where(inside, fade, 0.0))
        scores.append(np.where(inside, cos * fade, 0.0))
        colours.append(sample)
    weights, scores, colours = np.stack(weights), np.stack(scores), np.stack(colours)

    # A pixel nothing covers has no winner to be unsure of.
    ranked = np.sort(scores, axis=0)
    margin = np.where(ranked[-1] > 0.0,
                      (ranked[-1] - ranked[-2]) / np.maximum(ranked[-1], 1e-12), 1.0)
    if blend == "nearest":
        out = np.take_along_axis(colours, scores.argmax(axis=0)[None, ..., None], axis=0)[0]
        out[ranked[-1] <= 0] = 0.0
        return out, margin
    total = weights.sum(axis=0)
    out = (colours * weights[..., None]).sum(axis=0) / np.maximum(total, 1e-6)[..., None]
    out[total <= 1e-6] = 0.0
    return out, np.ones_like(total)


def test_output_raster():
    """
    The render draws one view per pixel, over per-view bounding boxes, at an output size
    that is not the canvas, straight into the wire's layout.  Each of those is a way to be
    subtly wrong that a mosaic which merely *looks* right would not show, so each is
    pinned against something that does not share its code.
    """
    print("\n14. Output raster and the one-view-per-pixel render")
    if cv2 is None:
        check("cv2 available", False, "opencv not installed; skipped")
        return
    try:
        import torch
        import PlanarStitcher as ps_mod
        from wire_panorama import WireReadyPanorama
    except ImportError as e:
        check("torch + PlanarStitcher importable", False, f"{e}")
        return

    texture = make_unique_texture()
    views, K, s_tex, plane = _oblique_scene(texture)
    intr = (K[0, 0], K[1, 1], K[0, 2], K[1, 2])
    cw, ch = 1000, 700
    base = {"canvas": (cw, ch), "metres_per_pixel": s_tex, "max_range": 200.0,
            "feather_px": 40, "aniso_max": 12.0, "min_coverage": 0.05, "pose_source": 0,
            "psnr_gate": False, "blend_mode": ps_mod.BLEND_NEAREST,
            "debug_view": ps_mod.DEBUG_OFF, "plane_sweep": False, "pose_refine": False,
            # Bilinear here, because that is what the reference below implements; the
            # default kernel has its own check at the end.
            "sampling": "bilinear"}

    def render(stitcher, out=None, **cfg):
        stitcher._plane_invalid_since = None
        pano, ok, reason = stitcher.planar_pano(
            views, intr, plane, dict(base, **cfg),
            out_size=None if out is None else (out[1], out[0]))
        return (np.array(pano) if ok else None), pano

    def to_picture(wire):
        """A wire-ready panorama (RGB, bottom-up) as the BGR top-down picture it shows."""
        return cv2.cvtColor(cv2.flip(wire, 0), cv2.COLOR_RGB2BGR)

    def agreement(a, b, keep=None):
        diff = np.abs(a.astype(np.float64) - b.astype(np.float64)).max(axis=2)
        if keep is not None:
            diff = diff[keep]
        return float((diff <= 1.0).mean()), float(diff.mean())

    s = make_bare_stitcher(ps_mod, torch)
    pano, _ = render(s)
    if not check("oblique scene rendered", pano is not None):
        return
    _frame, cams, M, _cw, _ch, _mpp = s._last_geometry
    check("all six oblique views were kept", len(cams) == 6, f"{len(cams)} kept")

    # --- against the slow, obvious implementation ---------------------------------------
    # Winner-take-all first. Pixels where the two best scores are within 1e-4 of each
    # other are set aside: there float32 may pick the other view, the two views differ by
    # their own resampling, and the comparison would be measuring that.
    ref, margin = _reference_mosaic(cams, M, np.eye(3), cw, ch, 40.0, 200.0, "nearest")
    clear = margin > 1e-4
    same, mean = agreement(pano, np.floor(ref + 0.5), clear)
    check("one-view-per-pixel render matches a float64 all-views reference",
          same > 0.9999 and mean < 0.05 and clear.mean() > 0.995,
          f"{same:.5%} of pixels within one grey level, mean difference {mean:.4f} "
          f"({1 - clear.mean():.3%} set aside as seam ties)")

    feather, _ = render(s, blend_mode=ps_mod.BLEND_FEATHER)
    ref_f, _ = _reference_mosaic(cams, M, np.eye(3), cw, ch, 40.0, 200.0, "feather")
    same, mean = agreement(feather, np.floor(ref_f + 0.5))
    check("feathered render matches the reference too",
          same > 0.9999 and mean < 0.05,
          f"{same:.5%} of pixels within one grey level, mean difference {mean:.4f}")

    # --- the wire layout -----------------------------------------------------------------
    wire, raw = render(s, out=(cw, ch))
    check("an output size returns a wire-ready panorama",
          isinstance(raw, WireReadyPanorama) and wire.shape == (ch, cw, 3),
          f"{type(raw).__name__} {None if wire is None else wire.shape}")
    same, mean = agreement(to_picture(wire), pano)
    check("at the canvas size the wire panorama is the canvas, flipped and in RGB",
          same > 0.9999 and mean < 0.01,
          f"{same:.5%} within one grey level, mean difference {mean:.4f}")

    # --- another size, another aspect ----------------------------------------------------
    # The reference is evaluated through a raster matrix written out in this file, so the
    # pixel-area convention, the row reversal and the channel order are all under test.
    for name, (ow, oh) in (("twice the canvas", (2 * cw, 2 * ch)),
                           ("another aspect (the old 8:3 panorama)", (1600, 600))):
        wire, _ = render(s, out=(ow, oh))
        ref, margin = _reference_mosaic(cams, M, _raster_to_canvas(cw, ch, ow, oh, True),
                                        ow, oh, 40.0, 200.0, "nearest")
        clear = margin > 1e-4
        same, mean = agreement(wire[..., ::-1], np.floor(ref + 0.5), clear)
        check(f"{name}: drawn where the canvas resized to it would be",
              wire.shape == (oh, ow, 3) and same > 0.9999 and mean < 0.05,
              f"{ow}x{oh}: {same:.5%} within one grey level, mean difference {mean:.4f}")

    # ... and that convention is cv2.resize's: the 2x mosaic averaged back down must sit
    # on the canvas mosaic, not half a pixel off it. A convention that anchored pixel
    # *centres* instead would pass every check above against its own matrix and fail here.
    wire2, _ = render(s, out=(2 * cw, 2 * ch))
    big = to_picture(wire2).astype(np.float64)
    inner = cv2.erode((pano.max(axis=2) > 0).astype(np.uint8), np.ones((25, 25), np.uint8)) > 0

    def back_down(shift):
        moved = np.roll(big, shift, axis=1)
        small = cv2.resize(moved, (cw, ch), interpolation=cv2.INTER_AREA)
        return float(np.sqrt(((small - pano.astype(np.float64)) ** 2).mean(axis=2)[inner].mean()))

    here, left, right = back_down(0), back_down(-1), back_down(1)
    check("the 2x mosaic averages back onto the canvas mosaic, not half a pixel off",
          here < 0.7 * min(left, right),
          f"RMSE {here:.2f} in place against {left:.2f} / {right:.2f} half a canvas pixel "
          f"either way")

    # --- the bounding boxes hide nothing --------------------------------------------------
    H_out = np.stack([c["H"] for c in cams]) @ _raster_to_canvas(cw, ch, 2 * cw, 2 * ch, True)
    rois = s._view_rois(H_out, 800, 450, 2 * cw, 2 * ch)
    area = np.mean([(y1 - y0) * (x1 - x0) for y0, y1, x0, x1 in rois]) / (4.0 * cw * ch)
    whole = make_bare_stitcher(ps_mod, torch)
    whole._view_rois = lambda H, sw, sh, ow, oh, pad=2: [(0, oh, 0, ow)] * len(H)
    identical = []
    for mode in (ps_mod.BLEND_NEAREST, ps_mod.BLEND_MULTIBAND, ps_mod.BLEND_FEATHER):
        boxed, _ = render(make_bare_stitcher(ps_mod, torch), out=(2 * cw, 2 * ch), blend_mode=mode)
        full, _ = render(whole, out=(2 * cw, 2 * ch), blend_mode=mode)
        whole._gain_state = None
        identical.append(bool(np.array_equal(boxed, full)))
    check("per-view bounding boxes change nothing but the work",
          all(identical) and area < 0.8,
          f"byte-identical in nearest / multiband / feather: {identical}; a view's box "
          f"is {area:.0%} of the raster on average")

    # --- the point of it: detail ----------------------------------------------------------
    # A canvas half as fine as the feeds, which is the case this exists for. Drawn at 2x
    # the mosaic must be closer to the ground truth than the canvas-sized one enlarged.
    nadir, K_n, s_n, plane_n = _nadir_scene(texture)
    intr_n = (K_n[0, 0], K_n[1, 1], K_n[0, 2], K_n[1, 2])
    coarse = dict(base, metres_per_pixel=2.0 * s_n)
    s_d = make_bare_stitcher(ps_mod, torch)
    low = s_d.planar_pano(nadir, intr_n, plane_n, coarse)[0]
    s_d._plane_invalid_since = None
    high = to_picture(np.array(s_d.planar_pano(nadir, intr_n, plane_n, coarse,
                                               out_size=(2 * ch, 2 * cw))[0]))
    th, tw = texture.shape[:2]
    M_tex = pg.canvas_to_plane_matrix(s_n, tw / 2.0, th / 2.0)
    M_fine = pg.canvas_to_plane_matrix(2.0 * s_n, cw * 0.5, ch * 0.5) \
        @ _raster_to_canvas(cw, ch, 2 * cw, 2 * ch, False)
    truth = cv2.warpPerspective(texture, np.linalg.inv(M_tex) @ M_fine, (2 * cw, 2 * ch),
                                flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP).astype(np.float64)
    seen = cv2.erode((high.max(axis=2) > 0).astype(np.uint8), np.ones((31, 31), np.uint8)) > 0
    enlarged = cv2.resize(low, (2 * cw, 2 * ch), interpolation=cv2.INTER_LINEAR)

    def rmse(img):
        return float(np.sqrt(((img.astype(np.float64) - truth) ** 2).mean(axis=2)[seen].mean()))

    check("drawn at 2x, a coarse canvas keeps the detail it used to discard",
          rmse(high) < 0.6 * rmse(enlarged),
          f"RMSE against the texture {rmse(enlarged):.2f} (canvas, enlarged) -> "
          f"{rmse(high):.2f} (drawn at 2x)")
    stats = s_d._last_stats
    check("the stats report the output size and its scale",
          stats.get("out_size") == (2 * cw, 2 * ch)
          and abs(stats.get("mpp_out", 0.0) - s_n) < 1e-9,
          f"out {stats.get('out_size')}, {stats.get('mpp_out', 0) * 100:.2f} cm/px "
          f"(canvas {stats.get('mpp_view', 0) * 100:.2f})")

    # --- multiband at an output size -------------------------------------------------------
    pair, K_p, s_p, plane_p = _nadir_scene(texture, offsets=((-6.0, 0.0), (6.0, 0.0)))
    intr_p = (K_p[0, 0], K_p[1, 1], K_p[0, 2], K_p[1, 2])
    dark = [dict(pair[0]), dict(pair[1], image=(pair[1]["image"].astype(np.float32) * 0.75)
                                .astype(np.uint8))]
    cfg_p = dict(base, metres_per_pixel=s_p)

    def seam_step(picture, scale):
        g = picture.astype(np.float32).mean(axis=2)
        mid = picture.shape[1] // 2 + int(round(scale * 6.0 / s_p))
        rows = slice(picture.shape[0] // 2 - 150 * scale, picture.shape[0] // 2 + 150 * scale)
        return abs(float(g[rows, mid - 40 * scale:mid - 3 * scale].mean()
                         - g[rows, mid + 3 * scale:mid + 40 * scale].mean()))

    def mosaic(stitcher, vs, mode, frames=1):
        out = None
        for _ in range(frames):
            stitcher._plane_invalid_since = None
            out = np.array(stitcher.planar_pano(vs, intr_p, plane_p, dict(cfg_p, blend_mode=mode),
                                                out_size=(2 * ch, 2 * cw))[0])
        return to_picture(out)

    hard = mosaic(make_bare_stitcher(ps_mod, torch), dark, ps_mod.BLEND_NEAREST)
    soft = mosaic(make_bare_stitcher(ps_mod, torch), dark, ps_mod.BLEND_MULTIBAND, frames=40)
    check("multiband still removes the exposure step at 2x",
          seam_step(hard, 2) > 15.0 and seam_step(soft, 2) < 0.25 * seam_step(hard, 2),
          f"brightness step across the seam {seam_step(hard, 2):.1f} -> "
          f"{seam_step(soft, 2):.1f} grey levels")
    plain = mosaic(make_bare_stitcher(ps_mod, torch), pair, ps_mod.BLEND_NEAREST)
    banded = mosaic(make_bare_stitcher(ps_mod, torch), pair, ps_mod.BLEND_MULTIBAND, frames=3)
    diff = np.abs(plain.astype(np.float32) - banded.astype(np.float32))[plain.max(axis=2) > 0]
    check("multiband at 2x changes nothing where the views already agree",
          float(diff.mean()) < 1.0, f"mean |difference| {float(diff.mean()):.2f} grey levels")

    # --- and the kernel: the feed has to survive the trip ---------------------------------
    # The 2x mosaic is resampled back into one drone's own frame and compared with the
    # frame that drone sent, over a patch it owns outright. This is the number the output
    # size and the bicubic kernel exist for: how much of a feed is still there.
    def feed_psnr(sampling):
        st_k = make_bare_stitcher(ps_mod, torch)
        wire = np.array(st_k.planar_pano(pair, intr_p, plane_p, dict(cfg_p, sampling=sampling),
                                         out_size=(2 * ch, 2 * cw))[0])
        cam = next(c for c in st_k._last_geometry[1] if c["view"]["drone_id"] == 0)
        src = cam["view"]["image"]
        h, w = src.shape[:2]
        U, V = np.meshgrid(np.arange(w, dtype=np.float64), np.arange(h, dtype=np.float64))
        q = np.tensordot(np.linalg.inv(cam["H"]), np.stack([U, V, np.ones_like(U)]), axes=1)
        back = cv2.remap(to_picture(wire),
                         ((q[0] / q[2] + 0.5) * 2.0 - 0.5).astype(np.float32),
                         ((q[1] / q[2] + 0.5) * 2.0 - 0.5).astype(np.float32),
                         cv2.INTER_LANCZOS4)
        mid = (slice(h // 2 - 60, h // 2 + 60), slice(w // 2 - 100, w // 2 + 100))
        err = (back[mid].astype(np.float64) - src[mid].astype(np.float64)) ** 2
        return 10.0 * np.log10(255.0 ** 2 / max(float(err.mean()), 1e-9))

    linear, cubic = feed_psnr("bilinear"), feed_psnr("bicubic")
    check("the default (bicubic) kernel returns the feed nearly intact at 2x",
          ps_mod.SAMPLING == "bicubic" and cubic > 48.0 and cubic > linear + 5.0,
          f"mosaic resampled back into the drone's frame: {linear:.1f} dB bilinear, "
          f"{cubic:.1f} dB bicubic (default kernel: {ps_mod.SAMPLING})")

    # --- the overlap PSNR is throttled only when asked ------------------------------------
    s_t = make_bare_stitcher(ps_mod, torch)
    calls = []
    measure = s_t._overlap_psnr
    s_t._overlap_psnr = lambda *a, **k: calls.append(1) or measure(*a, **k)
    for _ in range(3):
        render(s_t)
    every = len(calls)
    for _ in range(3):
        render(s_t, psnr_period_s=60.0)
    throttled = len(calls) - every
    for _ in range(2):
        render(s_t, psnr_period_s=60.0, psnr_gate=True)
    gated = len(calls) - every - throttled
    check("overlap PSNR: every call by default, on a period when asked, always when gating",
          every == 3 and throttled == 0 and gated == 2
          and np.isfinite(s_t._last_stats.get("overlap_psnr", float("nan"))),
          f"{every} / {throttled} / {gated} measurements over 3 / 3 / 2 renders; "
          f"reported {s_t._last_stats.get('overlap_psnr'):.1f} dB")

    # --- the same on the device the live stitcher uses ------------------------------------
    if not torch.cuda.is_available():
        print("  [SKIP] CUDA render: no device")
        return
    g = make_bare_stitcher(ps_mod, torch)
    g.render_device = torch.device("cuda")
    cpu_wire, _ = render(make_bare_stitcher(ps_mod, torch), out=(2 * cw, 2 * ch))
    gpu_wire, _ = render(g, out=(2 * cw, 2 * ch))
    same, mean = agreement(gpu_wire, cpu_wire)
    check("the CUDA render (pinned uploads, one stacked texture) matches the CPU one",
          same > 0.999 and mean < 0.05,
          f"{same:.4%} within one grey level, mean difference {mean:.4f}")

    # A frame rewritten in place is the same array, and must still be uploaded again.
    before = views[2]["image"].copy()
    views[2]["image"][:] = 255 - views[2]["image"]
    changed, _ = render(g, out=(2 * cw, 2 * ch))
    views[2]["image"][:] = before
    restored, _ = render(g, out=(2 * cw, 2 * ch))
    check("a frame rewritten in place is uploaded again",
          not np.array_equal(changed, gpu_wire) and np.array_equal(restored, gpu_wire),
          f"{float((changed != gpu_wire).any(axis=2).mean()):.0%} of the mosaic changed "
          f"with the frame, and came back with it")


def main():
    print("=" * 74)
    print("planar_geometry self-test")
    print("=" * 74)
    test_projection_agreement()
    test_plane_homography()
    test_against_textbook_form()
    test_canvas_orientation()
    test_rejection_cases()
    test_mosaic_roundtrip()
    test_planar_stitcher_end_to_end()
    test_estimators()
    test_formation_plane()
    test_canvas_modes()
    test_learned_alignment()
    test_output_raster()

    print("\n" + "=" * 74)
    if _failures:
        print(f"FAILED ({len(_failures)}): " + ", ".join(_failures))
        return 1
    print("All checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
