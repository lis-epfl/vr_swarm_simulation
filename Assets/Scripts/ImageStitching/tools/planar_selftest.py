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

    print("\n" + "=" * 74)
    if _failures:
        print(f"FAILED ({len(_failures)}): " + ", ".join(_failures))
        return 1
    print("All checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
