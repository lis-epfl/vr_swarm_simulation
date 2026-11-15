"""
Learned, graph-consistent alignment for the PLANAR stitcher.

The pose-only backbone (:mod:`PlanarStitcher`) is exact when the poses and the plane are.
On real aircraft neither is: GNSS is metres off, the compass degrees, and the plane the
formation "defines" is a fit through three to ten noisy positions.  Measured on the
recorded clips that leaves 10-40 source pixels of seam *on the plane itself*.

This module estimates what is wrong with the published geometry from the images, on the
warp thread, and hands back slowly-varying corrections the render path applies to every
frame's fresh pose:

``omega_i, t_i``  a rotation and a position correction per drone (6 DoF)
``plane``         the scene plane itself -- orientation as well as distance

The pipeline, and why each stage is there:

1. **Rectify** every view onto the prior plane (GPU).  The pose prior removes viewpoint,
   rotation and scale, so what is left between two views is a displacement of a few
   metres -- which is what makes the rest tractable, and what lets nadir views with
   arbitrary headings be matched by a detector that is not rotation invariant.

2. **SuperPoint** once per view: keypoints, descriptors, and the *dense* descriptor map
   from the same encoder pass.

3. **Shift graph** (acquisition only).  A facade of identical panels defeats any matcher
   that reasons about one pair at a time: with partial overlap the best assignment is a
   whole number of panels out, and it is self-consistent.  So each overlapping pair gets
   a correlation surface of the dense descriptors over plane shifts -- every candidate
   alignment scored on *all* the content, not only on the points a matcher chose -- and
   one shift per drone is then picked on the view graph so that the pairs agree with each
   other and with the GNSS prior.  The nodes are the drones, the unary term is GNSS, the
   edges are image evidence.  Period ambiguities that no single pair can resolve are
   resolved by the loops.

4. **LightGlue** per pair, on the pair's common window after the shift.  Its attention
   is given two crops of the same patch of plane, so the near-identity assignment is also
   the globally consistent one.

5. **Planar bundle adjustment** over all pairs at once: transfer error through the plane,
   Cauchy kernel with an annealed scale, Gaussian priors from the published poses.  Off-
   plane content (background, cranes, sky) is rejected by the kernel, not by a mask.

After a successful acquisition the estimator *tracks*: it rectifies with the corrected
geometry, skips stage 3, and gates matches tightly, so a locked alignment cannot hop a
panel.  A run of failed updates returns it to acquisition.

Everything is in right-handed world coordinates (``planar_geometry``'s convention).
"""

import time

import numpy as np
import torch
import torch.nn.functional as F

DT = torch.float64

# ----------------------------------------------------------------------------- tunables

# Longest side of a rectified view, in pixels. SuperPoint's cost is linear in pixels and
# the views arrive at 800-1280 wide, so this keeps one extraction near 10 ms.
MAX_RECT_SIDE = 960
# A grazing view's footprint runs to the horizon. Rays that meet the plane further than
# this multiple of the camera's own standoff are left out of its rectified patch.
MAX_RANGE_FACTOR = 3.0
MAX_KEYPOINTS = 1536

# Shift graph. Node shifts are searched within +/- NODE_SHIFT_M of the published pose; a
# pair's relative shift can then reach twice that, which sizes the correlation surfaces.
NODE_SHIFT_M = 6.0
# A pair needs this many overlapping descriptor cells before its surface means anything.
MIN_OVERLAP_CELLS = 150
# Distinctiveness weighting of the correlation (see _distinctiveness): the power a
# similarity is raised to before it counts as a duplicate, and the most one cell may
# outweigh the average, so a single odd patch cannot carry a whole pair.
DISTINCT_SHARPNESS = 8.0
DISTINCT_MAX_WEIGHT = 12.0
# Weighted overlap, in cells, at which a pair counts in full; thinner pairs count less.
EVIDENCE_UNIT = 1000.0
# Fraction of a pair's best overlap beyond which more overlap earns no more credit.
SATURATE_OVERLAP = 0.5
# Shifts within this fraction of the best alignment's contrast score as its equals.
ALIKE_BAND = 0.15
# Weight of the GNSS prior against image evidence, as score per (shift / sigma)^2. Small:
# it is a tie-break between alignments the images rate about equally, never a veto.
SHIFT_PRIOR_WEIGHT = 0.05
# Charge per (plane-distance sigma)^2 of formation-wide dilation in the shift graph.
DILATION_WEIGHT = 0.5
SHIFT_PRIOR_SIGMA_M = 3.0

# Far background, by depth. Placed on the plane by two views a baseline B apart, a point
# at depth Z lands B * (1 - Zp / Z) apart, along the baseline, in a known direction; for
# the sky and the horizon that is the whole baseline. A pair that shares mostly such
# content can "align" it perfectly by pretending the two aircraft are metres from where
# GNSS put them, and every match will agree -- it is a self-consistent rival solution,
# and no robust kernel can see it. The published baseline can: a displacement, measured
# against the PUBLISHED geometry, of more than this fraction of the baseline in that
# direction is content at more than 2.5x the plane's distance, whatever else it is. The
# floor keeps the test from rejecting plain GNSS error on a short baseline, where the
# parallax of the far field is no larger than the error anyway.
FAR_PARALLAX_FRACTION = 0.6
FAR_PARALLAX_MIN_M = 4.0

# Matches further than this from the (shifted) prior are not measurements of pose error.
GATE_ACQUIRE_M = 3.0
GATE_TRACK_M = 1.5
GATED_NN_MIN_SIMILARITY = 0.7   # descriptor cosine a gated nearest neighbour must reach
MAX_MATCHES_PER_PAIR = 220

# Bundle adjustment. Kernel scale in source pixels, annealed from the first to the last.
KERNEL_ACQUIRE_PX = 30.0
KERNEL_TRACK_PX = 16.0
KERNEL_FINAL_PX = 2.5
MATCH_SIGMA_PX = 1.5
BA_MAX_ITERS = 25

# Acceptance.
MIN_INLIERS = 40            # across all pairs
MIN_VIEW_INLIERS = 10       # a view with fewer keeps what it had
INLIER_PX = 4.0
MIN_INLIER_RATIO = 0.3
TRACK_LOST_UPDATES = 6
# Consecutive solves a single view may go without support before a wide search is spent
# on getting it back.
RELOCK_AFTER = 3
# ...and how many it must then hold support for before it can ask for another.
RELOCK_REARM = 8
# Share of the matcher's raw matches the published geometry must explain, through the
# tight gate, to be accepted without consulting the shift graph.
PRIOR_FIRST_SHARE = 0.5
# ...and how many times more the wide search must explain to overrule it otherwise.
WIDE_MUST_BEAT = 1.3

# Period verification (see PlanarAligner._hops). Cosine units of the dense descriptors'
# mean similarity over a pair's overlap: alignments within VERIFY_TIE of the pair's best
# count as equally good, and the credit falls to nothing over a further VERIFY_RAMP.
VERIFY_TIE = 0.08
VERIFY_RAMP = 0.08
VERIFY_PRIOR_PER_M2 = 0.02      # what moving one view costs, in pair-scores per metre^2
VERIFY_MIN_HOP_M = 1.0          # a smaller move is refinement, which the solver does better
VERIFY_EVERY = 10               # tracking updates between checks of a held lock
VERIFY_KEEP_INLIERS = 0.7       # a hop's re-solve must keep this share of the inliers
VERIFY_TIES_TO_GNSS = True      # at acquisition: equal image evidence -> nearest to GNSS
VERIFY_DECISIVE = 0.5           # held lock: pair-scores a move must gain per view moved

DEFAULT_PRIORS = {
    # Published-pose uncertainty. Compass yaw is the weak rotation axis; the gimbal's own
    # IMU holds pitch and roll much better. Differential GNSS is ~1-2 m horizontally and
    # barometric altitude differs between aircraft by a similar amount.
    "yaw_deg": 5.0, "tilt_deg": 2.0, "horizontal_m": 2.5, "vertical_m": 2.0,
    # Plane uncertainty when it is derived from the formation: a fit through a handful
    # of GNSS positions is good to ~10 degrees, and a typed standoff to a few metres.
    "plane_tilt_deg": 12.0, "plane_offset_m": 6.0,
    # How far a correction may move between consecutive updates while tracking.
    "track_rot_deg": 0.5, "track_pos_m": 0.3, "track_plane_tilt_deg": 1.0,
    "track_plane_offset_m": 0.5,
}


# ----------------------------------------------------------------------------- SO(3)

def _skew(w):
    z = torch.zeros_like(w[..., 0])
    return torch.stack([torch.stack([z, -w[..., 2], w[..., 1]], -1),
                        torch.stack([w[..., 2], z, -w[..., 0]], -1),
                        torch.stack([-w[..., 1], w[..., 0], z], -1)], -2)


def so3_exp(w):
    """Rotation vector(s) ``[..., 3]`` -> matrices.  Smooth through zero for autograd."""
    th2 = (w * w).sum(-1, keepdim=True) + 1e-24
    th = th2.sqrt()
    a = (torch.sin(th) / th)[..., None]
    b = ((1.0 - torch.cos(th)) / th2)[..., None]
    wx = _skew(w)
    eye = torch.eye(3, dtype=w.dtype, device=w.device).expand(wx.shape)
    return eye + a * wx + b * (wx @ wx)


def rotvec_to_matrix(w):
    """Rodrigues, in numpy: this one runs per drone per rendered frame during a blend."""
    w = np.asarray(w, dtype=np.float64)
    th = float(np.linalg.norm(w))
    wx = np.array([[0.0, -w[2], w[1]], [w[2], 0.0, -w[0]], [-w[1], w[0], 0.0]])
    if th < 1e-9:
        return np.eye(3) + wx
    return np.eye(3) + (np.sin(th) / th) * wx + ((1.0 - np.cos(th)) / (th * th)) * (wx @ wx)


def matrix_to_rotvec(R):
    """Log map: the rotation vector of a (near-)rotation matrix."""
    R = np.asarray(R, dtype=np.float64)
    w = 0.5 * np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    s = float(np.linalg.norm(w))
    c = 0.5 * (float(np.trace(R)) - 1.0)
    if s < 1e-12:
        return w
    return w / s * np.arctan2(s, c)


def rotvec_between(a, b):
    """Rotation vector of the smallest rotation taking unit vector ``a`` onto ``b``."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    v = np.cross(a, b)
    s = float(np.linalg.norm(v))
    if s < 1e-12:
        return np.zeros(3)
    return v / s * np.arctan2(s, float(np.dot(a, b)))


def plane_basis(n, hint=None):
    """Orthonormal in-plane ``(e1, e2)`` with ``cross(e1, e2) == n``."""
    n = np.asarray(n, dtype=np.float64)
    cands = [] if hint is None else [np.asarray(hint, dtype=np.float64)]
    cands += [np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0])]
    for c in cands:
        e1 = c - (c @ n) * n
        if np.linalg.norm(e1) > 0.2:
            e1 = e1 / np.linalg.norm(e1)
            return e1, np.cross(n, e1)
    raise ValueError("degenerate plane normal")


# ----------------------------------------------------------------------------- the solve

class PlanarBA:
    """
    Bundle adjustment of N cameras looking at one plane, from pairwise pixel matches.

    Parameters, all small corrections on the supplied base geometry::

        omega_i  camera-frame rotation    R_i = Exp(omega_i) R0_i
        t_i      world position           C_i = C0_i + t_i
        plane    two tilts and an offset about a pivot on the base plane

    The residual is the transfer error in pixels: a pixel of view ``i`` is cast onto the
    plane and projected into view ``j``.  Each residual touches 15 parameters, so the
    Jacobian is taken per match (forward mode, vmapped) and scattered, rather than by
    differentiating the stacked residual against every parameter.
    """

    def __init__(self, K, R0, C0, n0, d0, pivot, pairs, solve_pose=True, solve_plane=True):
        self.N = len(R0)
        self.K = torch.as_tensor(np.asarray(K), dtype=DT)
        self.Kinv = torch.linalg.inv(self.K)
        self.R0 = torch.as_tensor(np.stack(R0), dtype=DT)
        self.C0 = torch.as_tensor(np.stack(C0), dtype=DT)
        n0 = np.asarray(n0, dtype=np.float64)
        self.n0 = torch.as_tensor(n0, dtype=DT)
        pivot = np.asarray(pivot, dtype=np.float64)
        # Slide the pivot onto the base plane so the offset parameter is exactly "metres
        # along the normal" whatever point the caller passed.
        pivot = pivot + (float(d0) - float(n0 @ pivot)) * n0
        self.pivot = torch.as_tensor(pivot, dtype=DT)
        a1, a2 = plane_basis(n0)
        self.a1 = torch.as_tensor(a1, dtype=DT)
        self.a2 = torch.as_tensor(a2, dtype=DT)
        self.solve_pose, self.solve_plane = bool(solve_pose), bool(solve_plane)

        ii, jj, ui, uj, pid = [], [], [], [], []
        for k, (i, j, a, b) in enumerate(pairs):
            # Both directions: the transfer error treats its source pixel as exact, and
            # using each match once each way removes the asymmetry.
            ii += [np.full(len(a), i), np.full(len(a), j)]
            jj += [np.full(len(a), j), np.full(len(a), i)]
            ui += [a, b]
            uj += [b, a]
            pid += [np.full(2 * len(a), k)]
        if ii:
            self.ii = torch.as_tensor(np.concatenate(ii), dtype=torch.long)
            self.jj = torch.as_tensor(np.concatenate(jj), dtype=torch.long)
            ui = torch.as_tensor(np.concatenate(ui), dtype=DT)
            self.uj = torch.as_tensor(np.concatenate(uj), dtype=DT)
            self.pair_id = np.concatenate(pid)
        else:
            self.ii = self.jj = torch.zeros(0, dtype=torch.long)
            ui = self.uj = torch.zeros(0, 2, dtype=DT)
            self.pair_id = np.zeros(0, dtype=int)
        self.ray = torch.cat([ui, torch.ones(len(ui), 1, dtype=DT)], 1) @ self.Kinv.T
        self.M = len(ui)
        self.P = 6 * self.N + 3

        # Parameters that are switched off are simply never updated.
        free = np.ones(self.P, dtype=bool)
        if not self.solve_pose:
            free[:6 * self.N] = False
        if not self.solve_plane:
            free[6 * self.N:] = False
        self.free = torch.as_tensor(free)

    # -- geometry ------------------------------------------------------------------

    def plane(self, pl):
        n = so3_exp((pl[0] * self.a1 + pl[1] * self.a2)[None])[0] @ self.n0
        d = (n * (self.pivot + pl[2] * self.n0)).sum()
        return n, d

    def plane_params_of(self, n, d):
        """
        The three plane parameters that would make this problem's plane ``n . X == d``:
        the inverse of :meth:`plane`, for expressing another plane (the published one)
        in the coordinates of the one being solved about.
        """
        n = np.asarray(n, dtype=np.float64)
        n0, a1, a2 = self.n0.numpy(), self.a1.numpy(), self.a2.numpy()
        w = rotvec_between(n0, n / np.linalg.norm(n))
        pivot = self.pivot.numpy()
        along = float(n @ n0)
        offset = (float(d) - float(n @ pivot)) / along if abs(along) > 1e-9 else 0.0
        return np.array([float(w @ a1), float(w @ a2), offset])

    def unpack(self, x):
        """``x`` -> numpy ``(omega [N,3], t [N,3], n [3], d)``."""
        x = torch.as_tensor(x, dtype=DT)
        N = self.N
        n, d = self.plane(x[6 * N:])
        return (x[:3 * N].view(N, 3).numpy().copy(), x[3 * N:6 * N].view(N, 3).numpy().copy(),
                n.numpy().copy(), float(d))

    def _one(self, th_i, th_j, pl, R0i, C0i, R0j, C0j, ray):
        Ri = so3_exp(th_i[:3][None])[0] @ R0i
        Ci = C0i + th_i[3:]
        Rj = so3_exp(th_j[:3][None])[0] @ R0j
        Cj = C0j + th_j[3:]
        n, d = self.plane(pl)
        rw = Ri.T @ ray
        lam = (d - (Ci * n).sum()) / (rw * n).sum()
        pc = Rj @ (Ci + lam * rw - Cj)
        pr = self.K @ pc
        return pr[:2] / pr[2], torch.stack([lam, pc[2]])

    def _theta(self, x):
        N = self.N
        th = torch.cat([x[:3 * N].view(N, 3), x[3 * N:6 * N].view(N, 3)], 1)
        return th[self.ii], th[self.jj], x[6 * N:]

    def residual(self, x):
        """``(r [M,2] pixels, in_front [M])``."""
        if self.M == 0:
            return torch.zeros(0, 2, dtype=DT), torch.zeros(0, dtype=torch.bool)
        N = self.N
        R = so3_exp(x[:3 * N].view(N, 3)) @ self.R0
        C = self.C0 + x[3 * N:6 * N].view(N, 3)
        n, d = self.plane(x[6 * N:])
        Ci = C[self.ii]
        rw = torch.einsum("mji,mj->mi", R[self.ii], self.ray)
        lam = (d - Ci @ n) / (rw @ n)
        pc = torch.einsum("mij,mj->mi", R[self.jj], Ci + lam[:, None] * rw - C[self.jj])
        pr = pc @ self.K.T
        return pr[:, :2] / pr[:, 2:3] - self.uj, (lam > 0) & (pc[:, 2] > 0)

    @staticmethod
    def _left_jacobian(w):
        """SO(3) left Jacobian, ``d Exp(w) v = -[Exp(w) v]x J_l(w) dw``.  ``[..., 3]``."""
        th2 = (w * w).sum(-1, keepdim=True) + 1e-24
        th = th2.sqrt()
        small = th2 < 1e-10
        b = torch.where(small, 0.5 - th2 / 24.0, (1.0 - torch.cos(th)) / th2)[..., None]
        c = torch.where(small, 1.0 / 6.0 - th2 / 120.0, (th - torch.sin(th)) / (th2 * th))[..., None]
        wx = _skew(w)
        eye = torch.eye(3, dtype=w.dtype).expand(wx.shape)
        return eye + b * wx + c * (wx @ wx)

    def jacobian_blocks(self, x):
        """
        Analytic per-match blocks ``(d r/d theta_i [M,2,6], d r/d theta_j [M,2,6],
        d r/d plane [M,2,3])``, with ``theta = (omega, t)``.

        Hand-derived because it runs every iteration of every update: forward-mode
        autograd over the same expression costs ~25 ms at 4500 matches, this ~2 ms.
        ``jacobian_autograd`` is the reference it is asserted against in the self-test.
        """
        N = self.N
        om = x[:3 * N].view(N, 3)
        tt = x[3 * N:6 * N].view(N, 3)
        pl = x[6 * N:]
        R = so3_exp(om) @ self.R0
        C = self.C0 + tt
        Jl = self._left_jacobian(om)

        phi = pl[0] * self.a1 + pl[1] * self.a2
        n = so3_exp(phi[None])[0] @ self.n0
        pv = self.pivot + pl[2] * self.n0
        d = (n * pv).sum()
        # d n / d (alpha, beta): n = Exp(phi) n0, phi = alpha a1 + beta a2.
        dn_dphi = -_skew(n[None])[0] @ self._left_jacobian(phi[None])[0]
        dn = torch.stack([dn_dphi @ self.a1, dn_dphi @ self.a2], 1)          # [3, 2]

        Ri, Ci, Rj, Cj = R[self.ii], C[self.ii], R[self.jj], C[self.jj]
        ray = self.ray
        rw = torch.einsum("mji,mj->mi", Ri, ray)
        s = rw @ n
        lam = (d - Ci @ n) / s
        X = Ci + lam[:, None] * rw
        pc = torch.einsum("mij,mj->mi", Rj, X - Cj)
        pr = pc @ self.K.T
        uv = pr[:, :2] / pr[:, 2:3]
        # d uv / d pc
        A = (self.K[:2][None] - uv[:, :, None] * self.K[2][None, None]) / pr[:, 2, None, None]
        ARj = A @ Rj                                                            # [M,2,3]

        # Oblique projector onto the plane along the ray: dX/dC_i.
        Pm = torch.eye(3, dtype=DT)[None] - rw[:, :, None] * n[None, None, :] / s[:, None, None]
        drw = torch.einsum("mji,mjk->mik", Ri, _skew(ray)) @ Jl[self.ii]        # d rw / d omega_i
        dX_dom = lam[:, None, None] * (Pm @ drw)

        Ji = torch.cat([ARj @ dX_dom, ARj @ Pm], 2)
        Jj = torch.cat([-(A @ _skew(pc)) @ Jl[self.jj], -ARj], 2)

        # Plane: dX = (rw / s) * ((pv - X) . dn)  for the tilts, (rw / s) * (n . n0) for
        # the offset.
        g = (pv[None] - X) @ dn                                                 # [M,2]
        scale = (ARj @ rw[:, :, None])[:, :, 0] / s[:, None]                    # [M,2]
        Jp = torch.cat([scale[:, :, None] * g[:, None, :],
                        (scale * (n @ self.n0))[:, :, None]], 2)
        return Ji, Jj, Jp

    def jacobian_autograd(self, x):
        """The same blocks by forward-mode autograd.  Reference only."""
        thi, thj, pl = self._theta(x)
        f = lambda a, b, c, *rest: self._one(a, b, c, *rest)[0]
        return torch.func.vmap(
            torch.func.jacfwd(f, argnums=(0, 1, 2)),
            in_dims=(0, 0, None, 0, 0, 0, 0, 0))(
                thi, thj, pl, self.R0[self.ii], self.C0[self.ii],
                self.R0[self.jj], self.C0[self.jj], self.ray)

    def jacobian(self, x, blocks=None):
        """Dense ``[2M, P]`` Jacobian, scattered from the per-match 15-column blocks."""
        Ji, Jj, Jp = self.jacobian_blocks(x) if blocks is None else blocks
        N, M = self.N, self.M
        J = torch.zeros(M, 2, self.P, dtype=DT)
        rows = torch.arange(M)
        for src, idx in ((Ji, self.ii), (Jj, self.jj)):
            for k in range(3):
                # A match never has i == j, so each (row, column) is written once.
                J[rows, :, 3 * idx + k] = src[:, :, k]
                J[rows, :, 3 * N + 3 * idx + k] = src[:, :, 3 + k]
        J[:, :, 6 * N:] = Jp
        return J.reshape(2 * M, self.P)

    # -- priors --------------------------------------------------------------------

    def prior_precision(self, yaw_deg, tilt_deg, horizontal_m, vertical_m,
                        plane_tilt_deg, plane_offset_m):
        """
        Block-diagonal Gaussian precision for the corrections.

        Rotation sigmas are stated in the *world* frame -- yaw about the vertical is the
        compass, the other two axes are the gimbal -- and rotated into each camera's own
        frame, which is where ``omega`` lives.  World up is +Y in both Unity and the
        right-handed frame (only Z is negated between them).
        """
        N = self.N
        L = torch.zeros(self.P, self.P, dtype=DT)
        rot = torch.diag(torch.tensor([np.deg2rad(tilt_deg) ** -2, np.deg2rad(yaw_deg) ** -2,
                                       np.deg2rad(tilt_deg) ** -2], dtype=DT))
        pos = torch.diag(torch.tensor([horizontal_m ** -2, vertical_m ** -2,
                                       horizontal_m ** -2], dtype=DT))
        for i in range(N):
            R0 = self.R0[i]
            L[3 * i:3 * i + 3, 3 * i:3 * i + 3] = R0 @ rot @ R0.T
            s = 3 * N + 3 * i
            L[s:s + 3, s:s + 3] = pos
        L[6 * N, 6 * N] = L[6 * N + 1, 6 * N + 1] = np.deg2rad(plane_tilt_deg) ** -2
        L[6 * N + 2, 6 * N + 2] = plane_offset_m ** -2
        return L.numpy()

    # -- optimiser -----------------------------------------------------------------

    def solve(self, prior_terms, x0=None, kernel0=KERNEL_ACQUIRE_PX,
              kernel_final=KERNEL_FINAL_PX, sigma_px=MATCH_SIGMA_PX, max_iters=BA_MAX_ITERS):
        """
        IRLS Levenberg-Marquardt with a Cauchy kernel whose scale is annealed.

        ``prior_terms`` is a list of ``(precision [P,P], centre [P])``.  Starting the
        kernel wide and tightening it is graduated non-convexity: early iterations see
        every gated match and move toward the bulk; late ones see only what agrees to a
        couple of pixels, so off-plane content loses its vote without being segmented.

        Returns ``(x, residual_norm [M] px, in_front [M])``.
        """
        P = self.P
        x = torch.zeros(P, dtype=DT) if x0 is None else torch.as_tensor(x0, dtype=DT).clone()
        Lp = np.zeros((P, P))
        gp = np.zeros(P)
        for prec, centre in prior_terms:
            prec = np.asarray(prec, dtype=np.float64)
            Lp = Lp + prec
            gp = gp + prec @ np.asarray(centre, dtype=np.float64)
        free = self.free.numpy()

        # The residual is split along and across each match's epipolar direction -- the
        # way its transfer moves when the surface under it is nearer or further, which
        # is the plane-offset column of its own Jacobian.
        #
        # Across it, a match tests the cameras' relative geometry whatever it lies on: a
        # point on a wall, a roof or a tree is as good as one on the plane. Along it, it
        # tests whether the point is ON the plane, and for everything that is not, that
        # component is parallax and says nothing about pose. So the two get different
        # kernels: Cauchy across (keep listening), Geman-McClure along (stop listening
        # once it is clearly off the plane). One isotropic kernel instead lets a street
        # of building sides drag a ground mosaic by several pixels from exact poses.
        def split(r_np, epi):
            return (r_np * epi).sum(1), r_np[:, 0] * (-epi[:, 1]) + r_np[:, 1] * epi[:, 0]

        def cost(x_, c_, epi):
            r, ok = self.residual(x_)
            okn = ok.numpy()
            par, perp = split(r.numpy(), epi)
            par = np.where(okn, par, 1e3)
            perp = np.where(okn, perp, 1e3)
            rho = c_ * c_ * np.log1p(perp * perp / (c_ * c_)) \
                + c_ * c_ * par * par / (c_ * c_ + par * par)
            xn = x_.numpy()
            return float(rho.sum()) / sigma_px ** 2 + float(xn @ Lp @ xn - 2.0 * (xn @ gp))

        c = float(kernel0)
        damp = 1e-3
        epi = np.tile(np.array([1.0, 0.0]), (self.M, 1))
        for _ in range(max_iters):
            r, ok = self.residual(x)
            xn = x.numpy()
            if self.M:
                blocks = self.jacobian_blocks(x)
                d_off = blocks[2][:, :, 2].numpy()
                norm = np.linalg.norm(d_off, axis=1, keepdims=True)
                epi = np.where(norm > 1e-9, d_off / np.maximum(norm, 1e-12), [[1.0, 0.0]])
                r_np, okn = r.numpy(), ok.numpy().astype(np.float64)
                par, perp = split(r_np, epi)
                w_par = okn / (1.0 + par * par / (c * c)) ** 2 / sigma_px ** 2
                w_perp = okn / (1.0 + perp * perp / (c * c)) / sigma_px ** 2
                # The normal equations go through numpy on purpose: torch's CPU matmul
                # fans this 39-column product out over every core and takes 7 ms where
                # one BLAS call takes 1.
                Jn = self.jacobian(x, blocks).numpy().reshape(self.M, 2, P)
                J_par = Jn[:, 0] * epi[:, :1] + Jn[:, 1] * epi[:, 1:]
                J_perp = Jn[:, 0] * (-epi[:, 1:]) + Jn[:, 1] * epi[:, :1]
                H = (J_par.T * w_par) @ J_par + (J_perp.T * w_perp) @ J_perp + Lp
                g = J_par.T @ (w_par * par) + J_perp.T @ (w_perp * perp) + Lp @ xn - gp
            else:
                H, g = Lp.copy(), Lp @ xn - gp
            H = H[free][:, free]
            g = g[free]
            c_now = cost(x, c, epi)
            step = None
            for _try in range(6):
                try:
                    dx = np.linalg.solve(H + damp * np.diag(np.diag(H) + 1e-9), -g)
                except np.linalg.LinAlgError:
                    damp *= 10.0
                    continue
                cand = xn.copy()
                cand[free] += dx
                cand = torch.from_numpy(cand)
                if cost(cand, c, epi) < c_now:
                    x, step = cand, float(np.linalg.norm(dx))
                    damp = max(damp * 0.3, 1e-9)
                    break
                damp *= 10.0
            # Converged once the kernel has reached its final scale and a step no longer
            # moves anything by a hundredth of a pixel's worth (1e-4 rad ~ 0.05 px,
            # 1e-3 m ~ 0.02 px at these ranges).
            if c <= kernel_final * 1.0001 and (step is None or step < 1e-3):
                break
            c = max(float(kernel_final), c * 0.6)

        r, ok = self.residual(x)
        return x, (r * r).sum(-1).sqrt().numpy(), ok.numpy()


# ----------------------------------------------------------------------------- shift graph

def solve_shift_graph(n_nodes, edges, node_radius, prior_weight_per_cell2,
                      positions=None, dilation_limits=None, dilation_weight=0.0,
                      prior_centres=None):
    """
    One integer plane shift per node, maximising the pairs' agreement.

    ``positions`` (``[n, 2]``, the cameras' in-plane positions in cells, same axes as the
    shifts) switches on the one constraint that is about the formation as a whole.  A
    set of shifts proportional to the cameras' own positions, ``s_i = kappa * c_i``, is
    not a set of pose errors: it is what it looks like when the views are being aligned
    on a *different surface*, ``Z = Zp / (1 + kappa)`` away instead of ``Zp`` -- the
    building behind, or the one in front.  GNSS error is independent from drone to drone
    and almost never takes that shape; a wrong plane always does.  So ``kappa`` (the
    least-squares dilation of the shifts) is confined to ``dilation_limits``, which the
    caller derives from how wrong the plane's distance may be, and charged
    ``dilation_weight * kappa^2`` inside them.  Without it, a neighbouring facade with
    more distinctive texture than the one being stitched simply wins.

    ``edges`` is a list of ``(i, j, S)`` where ``S[R + dy, R + dx]`` scores the relative
    displacement ``d = shift_i - shift_j`` (in cells) and ``R = 2 * node_radius``.

    Iterated conditional modes from several starts: the all-zero prior, and every edge's
    strongest few peaks propagated outward.  The second family is what escapes the case
    ICM alone cannot -- two views that have to move together to reach the better optimum.
    Small by construction (a handful of nodes, a few hundred states each), so it is
    exhaustive enough to be trusted and costs well under a millisecond per sweep.

    Returns ``(shifts [n,2] int (dx, dy), score)``.
    """
    Rn = int(node_radius)
    side = 2 * Rn + 1
    ys, xs = np.mgrid[-Rn:Rn + 1, -Rn:Rn + 1]
    # ``prior_centres`` ([n, 2] cells) moves each node's quadratic prior off zero: the
    # shift it charges for is then measured from there (where GNSS put the view) and not
    # from wherever this pass happened to start.
    if prior_centres is None:
        priors = [-prior_weight_per_cell2 * (xs * xs + ys * ys)] * n_nodes
    else:
        priors = [-prior_weight_per_cell2 * ((xs - c[0]) ** 2 + (ys - c[1]) ** 2)
                  for c in np.asarray(prior_centres, dtype=np.float64)]
    nbrs = {k: [] for k in range(n_nodes)}
    for e, (i, j, S) in enumerate(edges):
        nbrs[i].append((e, j, +1))
        nbrs[j].append((e, i, -1))

    centred = spread = None
    if positions is not None and n_nodes >= 2:
        centred = np.asarray(positions, dtype=np.float64)
        centred = centred - centred.mean(axis=0)
        spread = float((centred * centred).sum())
        if spread < 1e-6:
            centred = None
    k_lo, k_hi = dilation_limits if dilation_limits is not None else (-np.inf, np.inf)

    def dilation(sh):
        return 0.0 if centred is None else float((sh * centred).sum()) / spread

    def dilation_term(sh, k):
        """Score of every candidate shift of node ``k`` from the dilation it would give."""
        if centred is None:
            return 0.0
        others = float((sh * centred).sum() - sh[k] @ centred[k])
        kappa = (others + xs * centred[k][0] + ys * centred[k][1]) / spread
        term = -dilation_weight * kappa * kappa
        return np.where((kappa < k_lo) | (kappa > k_hi), -np.inf, term)

    def window(e, other_shift, sign):
        """Edge ``e``'s score for every candidate shift of this node, the other fixed."""
        S = edges[e][2]
        R = (S.shape[0] - 1) // 2
        ox, oy = other_shift
        if sign > 0:            # this node is i: d = s - s_j
            return S[R - oy - Rn:R - oy + Rn + 1, R - ox - Rn:R - ox + Rn + 1]
        # this node is j: d = s_i - s  -> index reversed in s
        return S[R + oy - Rn:R + oy + Rn + 1, R + ox - Rn:R + ox + Rn + 1][::-1, ::-1]

    def total(sh):
        kappa = dilation(sh)
        if kappa < k_lo or kappa > k_hi:
            return -np.inf
        t = float(sum(priors[k][sh[k][1] + Rn, sh[k][0] + Rn] for k in range(n_nodes)))
        t -= dilation_weight * kappa * kappa
        for i, j, S in edges:
            R = (S.shape[0] - 1) // 2
            d = sh[i] - sh[j]
            t += float(S[R + d[1], R + d[0]])
        return t

    def icm(sh, frozen=()):
        sh = sh.copy()
        for _sweep in range(12):
            changed = False
            for k in range(n_nodes):
                if k in frozen or not nbrs[k]:
                    continue
                acc = priors[k] + dilation_term(sh, k)
                for e, other, sign in nbrs[k]:
                    acc = acc + window(e, sh[other], sign)
                if not np.isfinite(acc.max()):
                    continue                 # no shift of this node alone is admissible
                iy, ix = np.unravel_index(int(np.argmax(acc)), acc.shape)
                new = np.array([ix - Rn, iy - Rn])
                if np.any(new != sh[k]):
                    sh[k] = new
                    changed = True
            if not changed:
                break
        return sh

    def peaks(S, count=3, sep=2):
        out = []
        for f in np.argsort(S.ravel())[::-1][:200]:
            y, x = divmod(int(f), S.shape[1])
            if all(abs(y - py) > sep or abs(x - px) > sep for py, px in out):
                out.append((y, x))
                if len(out) == count:
                    break
        return out

    zero = np.zeros((n_nodes, 2), dtype=int)
    best = icm(zero)
    best_score = total(best)
    if prior_centres is not None:
        # "Everything back where GNSS put it" is a start of its own: it is the answer
        # whenever the images have nothing to say, and ICM cannot reach it from zero when
        # that takes several views moving together.
        start = np.clip(np.round(np.asarray(prior_centres, dtype=np.float64)),
                        -Rn, Rn).astype(int)
        cand = icm(start)
        sc = total(cand)
        if sc > best_score + 1e-9:
            best, best_score = cand, sc
    for i, j, S in edges:
        R = (S.shape[0] - 1) // 2
        for py, px in peaks(S):
            d = np.array([px - R, py - R])            # s_i - s_j
            if np.any(np.abs(d) > 2 * Rn):
                continue
            # Split the displacement between the two ends so both stay inside the node
            # range, then let everything else settle around the imposed pair before
            # releasing it.
            si = np.clip(np.round(d / 2.0).astype(int), -Rn, Rn)
            sj = np.clip(si - d, -Rn, Rn)
            start = zero.copy()
            start[i], start[j] = si, sj
            cand = icm(icm(start, frozen=(i, j)))
            sc = total(cand)
            if sc > best_score + 1e-9:
                best, best_score = cand, sc
    return best, best_score


# ----------------------------------------------------------------------------- front end

def _baseline(views, ki, kj, e1, e2):
    """Published in-plane baseline i -> j: ``(unit vector, far threshold)``."""
    B = np.asarray(views[kj]["C"], float) - np.asarray(views[ki]["C"], float)
    b = np.array([B @ e1, B @ e2])
    length = float(np.linalg.norm(b))
    if length < 1e-6:
        return np.zeros(2), np.inf
    return b / length, max(FAR_PARALLAX_FRACTION * length, FAR_PARALLAX_MIN_M)


class _Models:
    """SuperPoint + LightGlue, loaded on first use and never at import."""

    def __init__(self, device):
        self.device = device
        self.sp = None
        self.lg = None
        self.error = None
        self.tried = False

    def load(self):
        if self.tried:
            return self.sp is not None
        self.tried = True
        try:
            from transformers import SuperPointForKeypointDetection
            self.sp = SuperPointForKeypointDetection.from_pretrained(
                "magic-leap-community/superpoint").eval().to(self.device)
        except Exception as exc:                         # no weights, no network
            self.error = "SuperPoint unavailable: %s" % exc
            self.sp = None
            return False
        try:
            import kornia.feature as KF
            # flash=False is load-bearing. With it on, kornia runs the attention in half
            # precision through a path that spends ~1.1 s on every keypoint count it has
            # not seen before -- and every pair of every update has a new count. Off, a
            # pair costs a steady 11-20 ms whatever its size.
            self.lg = KF.LightGlue(features="superpoint", depth_confidence=-1,
                                   width_confidence=-1, filter_threshold=0.1,
                                   flash=False).eval().to(self.device)
        except Exception as exc:
            # Degrade, do not fail: radius-gated mutual nearest neighbours on the same
            # descriptors still align everything except strongly periodic scenes.
            self.error = "LightGlue unavailable (%s); using gated nearest-neighbour matching" % exc
            self.lg = None
        return True


class PlanarAligner:
    """
    Stateful estimator of per-drone pose corrections and the scene plane.

    ``update`` is called from the warp thread with one synchronous set of views; the
    corrections it returns are biases, valid for later frames' poses until the next
    update replaces them.
    """

    def __init__(self, device=None):
        self.device = torch.device(device) if device is not None else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.models = _Models(self.device)
        self.priors = dict(DEFAULT_PRIORS)
        self.reset()

    def reset(self):
        self.mode = "acquire"
        self.pose = {}                   # drone_id -> (omega [3] camera frame, t [3] world)
        self.plane = None                # (n, d) world, the estimate itself
        self.plane_delta = None          # (rotvec, offset) relative to the published plane
        self.lost = 0
        self.stats = {}
        self.updates = 0
        self.unsupported_streak = {}     # drone_id -> consecutive solves without support
        self.supported_streak = {}       # ...and with it
        self.gave_up = set()             # drones a re-acquisition has already been spent on

    def available(self):
        return self.models.load()

    @torch.no_grad()
    def warm_up(self):
        """
        Run both networks once on dummy input.

        The first call of each pays for the device's kernel selection and allocations --
        seconds, against tens of milliseconds afterwards -- and without this that cost
        lands inside the first alignment, which is the one the pilot is waiting for.
        """
        if not self.models.load() or getattr(self, "_warm", False):
            return
        self._warm = True
        try:
            rect = torch.rand(1, 1, 448, 800, device=self.device)
            valid = torch.ones(448, 800, dtype=torch.bool, device=self.device)
            kps, desc, _dense, _cells, _w = self._extract(rect, valid)
            if self.models.lg is not None and len(kps) >= 16:
                size = torch.tensor([[800.0, 448.0]], device=self.device)
                side = {"keypoints": kps[None], "descriptors": desc[None], "image_size": size}
                self.models.lg({"image0": side, "image1": dict(side)})
        except Exception:                                # warming up must never be fatal
            pass

    # -- rectification -------------------------------------------------------------

    @staticmethod
    def _footprint(G, width, height, max_range):
        us = np.linspace(0.0, width - 1.0, 13)
        vs = np.linspace(0.0, height - 1.0, 9)
        uu, vv = np.meshgrid(us, vs)
        pix = np.stack([uu.ravel(), vv.ravel(), np.ones(uu.size)])
        try:
            p = np.linalg.solve(G, pix)
        except np.linalg.LinAlgError:
            return None
        ok = p[2] > 1.0 / max_range
        if ok.mean() < 0.25:
            return None
        ab = (p[:2, ok] / p[2, ok]).T
        return ab.min(0), ab.max(0)

    def _rectify(self, image, G, lo, hi, gsd):
        """Grey rectified patch ``[1,1,h,w]`` and validity mask, on the plane lattice."""
        dev = self.device
        h_src, w_src = image.shape[:2]
        w = int(round((hi[0] - lo[0]) / gsd))
        h = int(round((hi[1] - lo[1]) / gsd))
        xs = torch.arange(w, device=dev, dtype=torch.float32)
        ys = torch.arange(h, device=dev, dtype=torch.float32)
        a = lo[0] + xs * gsd
        b = hi[1] - ys * gsd
        Gt = torch.as_tensor(G, dtype=torch.float32, device=dev)
        aa = a[None, :].expand(h, w)
        bb = b[:, None].expand(h, w)
        uvw = (Gt[:, 0, None, None] * aa + Gt[:, 1, None, None] * bb + Gt[:, 2, None, None])
        depth = uvw[2]
        safe = torch.where(depth > 1e-6, depth, torch.ones_like(depth))
        u, v = uvw[0] / safe, uvw[1] / safe
        valid = (depth > 1e-6) & (u >= 0) & (u <= w_src - 1) & (v >= 0) & (v <= h_src - 1)
        grid = torch.stack([2.0 * u / (w_src - 1) - 1.0, 2.0 * v / (h_src - 1) - 1.0], -1)
        img = torch.from_numpy(image).to(dev)
        grey = img.float().mean(-1)[None, None] / 255.0
        rect = F.grid_sample(grey, grid[None], mode="bilinear", padding_mode="zeros",
                             align_corners=True)
        return rect, valid

    @torch.no_grad()
    def _extract(self, rect, valid):
        """SuperPoint on one rectified patch: keypoints, descriptors, dense map + mask."""
        sp = self.models.sp
        enc = sp.encoder(rect, return_dict=True).last_hidden_state
        kps, scores = sp.keypoint_decoder(enc)
        dd = sp.descriptor_decoder
        dense = F.normalize(dd.conv_descriptor_b(dd.relu(dd.conv_descriptor_a(enc))), dim=1)

        # A keypoint on the edge of the warped image is the edge, not the scene.
        inner = -F.max_pool2d(-valid.float()[None, None], 13, stride=1, padding=6)[0, 0] > 0.5
        if len(kps):
            ki = kps.long()
            keep = inner[ki[:, 1], ki[:, 0]]
            kps, scores = kps[keep], scores[keep]
        if len(kps) > MAX_KEYPOINTS:
            top = torch.topk(scores, MAX_KEYPOINTS).indices
            kps, scores = kps[top], scores[top]
        if len(kps):
            desc = dd._sample_descriptors(kps[None].clone(), dense, 8)[0].T.contiguous()
        else:
            desc = torch.zeros(0, dense.shape[1], device=rect.device)

        cell_ok = F.avg_pool2d(inner.float()[None, None], 8)[0, 0] > 0.99
        weight = self._distinctiveness(dense[0], cell_ok)
        return kps, desc, dense[0] * weight, cell_ok.float(), weight

    @staticmethod
    def _distinctiveness(dense, cell_ok):
        """
        How much each descriptor cell is worth as evidence of *where* a view is, ``[h,w]``.

        On a facade of identical panels nearly every cell has hundreds of look-alikes in
        the same image, and those cells score the same at the true alignment as at one a
        whole panel out -- they vote for every period equally and, being most of the
        image, drown the few cells that can tell the periods apart (a window, the
        roofline, the end of the building).  So a cell is weighted by the inverse of how
        common its descriptor is within its own view: the sum of its (sharpened)
        similarities to every other cell.  It is inverse document frequency, with the
        view as the corpus.  On unique texture every cell is rare and the weights come
        out flat, so nothing changes where there was nothing to fix.
        """
        flat = dense.flatten(1).T                                  # [h*w, C], unit rows
        ok = cell_ok.flatten()
        weight = torch.zeros(flat.shape[0], device=dense.device)
        idx = torch.nonzero(ok)[:, 0]
        if len(idx) < 4:
            return weight.view(cell_ok.shape)
        d = flat[idx]
        # Sharpened so that only near-duplicates count as "the same thing again".
        commonness = (d @ d.T).clamp_min(0.0).pow(DISTINCT_SHARPNESS).sum(dim=1)
        w = 1.0 / commonness
        w = (w / w.mean()).clamp_max(DISTINCT_MAX_WEIGHT)
        weight[idx] = w / w.mean()
        return weight.view(cell_ok.shape)

    # -- matching ------------------------------------------------------------------

    @staticmethod
    def _spectrum(v, what):
        """One view's map of ``what``, transformed once and kept: ``[C, H, W/2+1]``."""
        cache = v.setdefault("fft", {})
        if what not in cache:
            if what == "dense":
                m = v["dense"]
            elif what == "plain":                  # the descriptors without their weights
                m = v["dense"] / v["weight"].clamp_min(1e-6) * v["cells"]
            else:
                m = v[what][None]
            cache[what] = torch.fft.rfft2(m, s=v["fft_size"])
        return cache[what]

    def _xcorr(self, vi, vj, what, R):
        """
        Correlation of two views' maps over relative lattice shifts, ``[2R+1, 2R+1]``.

        ``out[R + dy, R + dx]`` is the sum over view i's cells of ``i[c] * j[c + (dy, dx)]``
        with both maps on the plane's own lattice -- what sliding view j's map by
        ``(-dx, -dy)`` cells over view i's would total. One transform per view and a
        product per pair: the same numbers a convolution with a view-sized kernel gives,
        in milliseconds where that took a tenth of a second a pair.
        """
        Hs, Ws = vi["fft_size"]
        full = torch.fft.irfft2((self._spectrum(vi, what).conj()
                                 * self._spectrum(vj, what)).sum(0), s=(Hs, Ws))
        # The maps sit at different places on the lattice; a relative shift of (dy, dx)
        # on the lattice is one of (dy - off_r, dx - off_c) between the two arrays.
        off_c, off_r = vj["col0"] - vi["col0"], vj["row0"] - vi["row0"]
        dev = full.device
        dy = torch.arange(-R, R + 1, device=dev) - off_r
        dx = torch.arange(-R, R + 1, device=dev) - off_c
        Hi, Wi = vi["cells"].shape
        Hj, Wj = vj["cells"].shape
        oky = (dy > -Hi) & (dy < Hj)
        okx = (dx > -Wi) & (dx < Wj)
        out = full[dy % Hs][:, dx % Ws]
        return out * (oky[:, None] & okx[None, :])

    @torch.no_grad()
    def _surface(self, vi, vj, R):
        """Dense-descriptor correlation over relative lattice shifts, ``[2R+1, 2R+1]``."""
        # Two planes of bookkeeping beside the descriptors: plain coverage, which says
        # whether a shift leaves enough overlap to be scored at all, and the
        # distinctiveness weights, which are what the score is normalised by. The
        # descriptors already carry their weights, so num is sum(w_i w_j <d_i, d_j>).
        num = self._xcorr(vi, vj, "dense", R)
        cnt = self._xcorr(vi, vj, "cells", R).round()
        wsum = self._xcorr(vi, vj, "weight", R).clamp_min(0.0)
        if float(cnt.max()) < MIN_OVERLAP_CELLS:
            return None
        S = num / wsum.clamp_min(1e-3)
        ok = (cnt >= 0.25 * cnt.max()) & (wsum > 0.05 * wsum.max())
        if int(ok.sum()) < 9:
            return None
        # Turn similarity into a score a graph can add up. Three things it must not do,
        # each of which a simpler score did on a real facade:
        #
        # - reward overlap for its own sake. A sum of similarity grows with the shared
        #   area, and on storeys of identical windows a shift by one storey shares more
        #   area and loses almost no similarity -- so the sum prefers the wrong storey
        #   even from exact poses. Credit for overlap therefore saturates: half the
        #   pair's best overlap is as good as all of it.
        # - trust a thin strip. A mean alone lets two views that share a sliver outvote
        #   two that share half the wall. Below the saturation point credit is
        #   proportional to the overlap, and the whole pair is scaled by how much plane
        #   it shares at best.
        # - split hairs between look-alikes. Shifts whose similarity is within a small
        #   fraction of the best substantial alignment's all score alike, so that what
        #   decides between them is the other pairs and, failing those, the GNSS prior
        #   -- "do not move a panel unless the images clearly say so".
        wide = ok & (wsum >= SATURATE_OVERLAP * wsum.max())
        floor = S[ok].median()
        best = S[wide].max() if bool(wide.any()) else S[ok].max()
        contrast = (best - floor).clamp_min(1e-6)
        alike = (((S - floor) / contrast - (1.0 - ALIKE_BAND)) / ALIKE_BAND).clamp(0.0, 1.0)
        credit = (wsum / (SATURATE_OVERLAP * wsum.max())).clamp(0.0, 1.0)
        pair = min(1.0, float(wsum.max()) / EVIDENCE_UNIT) ** 0.5
        S = torch.where(ok, alike * credit * pair, torch.zeros_like(S))
        return S.cpu().numpy()

    @torch.no_grad()
    def _surface_plain(self, vi, vj, R):
        """
        Mean descriptor similarity over the overlap, for every relative lattice shift.

        Returns ``(S, cells)`` as ``[2R+1, 2R+1]`` arrays indexed like ``_surface``'s, or
        ``None`` when the views do not overlap enough as they stand. No distinctiveness
        weights and no credit for overlap: this is for views that are already registered
        to a pixel or two, where the question is only whether a *different* registration,
        a whole period away, is plainly better.
        """
        num = self._xcorr(vi, vj, "plain", R)
        cnt = self._xcorr(vi, vj, "cells", R).round()
        if float(cnt[R, R]) < MIN_OVERLAP_CELLS:
            return None
        return (num / cnt.clamp_min(1.0)).cpu().numpy(), cnt.cpu().numpy()

    def _hops(self, prep, to_gnss=True):
        """
        Whole-period corrections to an alignment that is already pixel-tight.

        A repeating wall has a lattice of alignments that all agree to a pixel, and the
        solve lands on whichever its matches were gated around. They are not equally
        good: with the views registered, the wall's *non*-repeating content -- the end of
        the building, a doorway, one window unlike the rest -- lines up at exactly one of
        them, and the mean descriptor similarity over a pair's overlap shows it plainly
        (0.67 against 0.50-0.55 between two drones of one row on the simulated wall). So
        each pair scores every relative shift, and one shift per view is chosen for the
        whole formation.

        Where the images cannot tell -- two rows of a formation looking at identical
        storeys score the same to within a few hundredths at 0, 1 and 2 storeys out --
        shifts within ``VERIFY_TIE`` of a pair's best count alike, and what decides is
        how far each view would stand from where GNSS put it.

        ``to_gnss`` is that last rule, and it is for an acquisition, where the solve has
        just chosen among look-alikes and GNSS is the only other witness. A lock that has
        been held is checked without it: there a view moves only when the images gain
        ``VERIFY_DECISIVE`` pair-scores for each view moved, because GNSS that is itself
        a panel out (the MED facade, on a bad day) would otherwise pull a good lock apart.

        Returns ``{view index: shift (a, b) metres}`` for the views that should move, or
        ``None`` when the alignment stands.
        """
        usable, feats, cell = prep["usable"], prep["feats"], prep["cell"]
        Rn = max(2, int(round(NODE_SHIFT_M / cell)))
        edges = []
        for a in range(len(usable)):
            for b in range(a + 1, len(usable)):
                out = self._surface_plain(feats[usable[a]], feats[usable[b]], 2 * Rn)
                if out is None:
                    continue
                S, cnt = out
                R = (S.shape[0] - 1) // 2
                ok = cnt >= 0.5 * cnt[R, R]
                best = float(S[ok].max())
                score = np.clip(1.0 - (best - S - VERIFY_TIE) / VERIFY_RAMP, 0.0, 1.0)
                weight = min(1.0, float(cnt[R, R]) / EVIDENCE_UNIT) ** 0.5
                edges.append((a, b, np.where(ok, score * weight, 0.0)))
        if not edges:
            return None
        centres = np.array([[-feats[k]["moved"][0] / cell, feats[k]["moved"][1] / cell]
                            for k in usable])
        sh, _score = solve_shift_graph(len(usable), edges, Rn, VERIFY_PRIOR_PER_M2 * cell * cell,
                                       prior_centres=centres if to_gnss else None)
        # Only relative moves mean anything: take out the shift most views share.
        sh = sh - np.round(np.median(sh, axis=0)).astype(int)
        if not to_gnss:
            def images(shifts):
                total = 0.0
                for i, j, S in edges:
                    R = (S.shape[0] - 1) // 2
                    d = shifts[i] - shifts[j]
                    total += float(S[R + d[1], R + d[0]])
                return total
            n_moved = int(sum(np.linalg.norm(v) * cell >= VERIFY_MIN_HOP_M for v in sh))
            if images(sh) - images(np.zeros_like(sh)) < VERIFY_DECISIVE * max(n_moved, 1):
                return None
        hops = {k: np.array([sh[a][0] * cell, -sh[a][1] * cell]) for a, k in enumerate(usable)}
        if max(float(np.linalg.norm(v)) for v in hops.values()) < VERIFY_MIN_HOP_M:
            return None
        return hops

    @torch.no_grad()
    def _match(self, vi, vj, shift_i, shift_j, gsd, margin, gate=None, use_lg=True):
        """
        Matches between two views on their common window. Indices into each view.

        With ``gate`` (metres) the result is LightGlue's matches plus the mutual nearest
        neighbours within that distance of the hypothesised alignment.
        """
        ab_i = vi["ab"] + shift_i
        ab_j = vj["ab"] + shift_j
        lo = np.maximum(vi["lo"] + shift_i, vj["lo"] + shift_j) - margin
        hi = np.minimum(vi["hi"] + shift_i, vj["hi"] + shift_j) + margin
        if np.any(hi - lo < 8.0 * gsd * 4):
            return None
        in_i = np.nonzero(np.all((ab_i >= lo) & (ab_i <= hi), axis=1))[0]
        in_j = np.nonzero(np.all((ab_j >= lo) & (ab_j <= hi), axis=1))[0]
        if len(in_i) < 12 or len(in_j) < 12:
            return None
        dev = self.device

        def window_xy(ab):
            return torch.as_tensor(np.column_stack([(ab[:, 0] - lo[0]) / gsd,
                                                    (hi[1] - ab[:, 1]) / gsd]),
                                   dtype=torch.float32, device=dev)

        xi, xj = window_xy(ab_i[in_i]), window_xy(ab_j[in_j])
        di = vi["desc"][torch.as_tensor(in_i, device=dev)]
        dj = vj["desc"][torch.as_tensor(in_j, device=dev)]

        # Mutual nearest neighbours among partners within the gate. This is the matcher
        # that answers the question actually being asked -- "given this alignment, which
        # points correspond?" -- and on a repeating wall it is the only one that does:
        # with the gate under half a period each point has one candidate, the right one.
        sim = di @ dj.T
        far = torch.cdist(xi, xj) > ((gate if gate else margin) / gsd)
        sim = sim.masked_fill(far, -1.0)
        bi = sim.argmax(1)
        bj = sim.argmax(0)
        idx = torch.arange(len(xi), device=dev)
        keep = (bj[bi] == idx) & (sim[idx, bi] > GATED_NN_MIN_SIMILARITY)
        nn_i, nn_j = idx[keep].cpu().numpy(), bi[keep].cpu().numpy()
        if self.models.lg is None or not use_lg:
            return in_i[nn_i], in_j[nn_j]

        # LightGlue has no notion of the hypothesis: it returns the one assignment it
        # likes best over the whole window, and on storeys of identical windows that is
        # whichever shares the most wall -- two storeys out between two rows of a
        # simulated formation with EXACT poses, every match of every such pair. Its
        # matches are kept for what they are good at (context on unique texture, where
        # they outnumber nearest neighbours several to one) and the gated ones are added,
        # so a pair whose LightGlue answer is an alias still has its true matches.
        size = torch.tensor([[(hi[0] - lo[0]) / gsd, (hi[1] - lo[1]) / gsd]],
                            device=dev, dtype=torch.float32)
        out = self.models.lg({
            "image0": {"keypoints": xi[None], "descriptors": di[None], "image_size": size},
            "image1": {"keypoints": xj[None], "descriptors": dj[None], "image_size": size}})
        m = out["matches"][0].cpu().numpy()
        if not gate:
            return in_i[m[:, 0]], in_j[m[:, 1]]
        both = np.unique(np.concatenate([m, np.column_stack([nn_i, nn_j])]), axis=0)
        return in_i[both[:, 0]], in_j[both[:, 1]]

    def _prepare(self, views, K, x_pose, n_b, d_b, n_pub, d_pub):
        """
        Rectify every view onto the plane ``(n_b, d_b)`` from the published poses moved by
        ``x_pose`` (``[6N]`` or longer: rotations then translations), and extract features.

        Returns a dict, or a string saying why there is nothing to work with.
        """
        N = len(views)
        R_cur = [rotvec_to_matrix(x_pose[3 * k:3 * k + 3]) @ np.asarray(v["R"], float)
                 for k, v in enumerate(views)]
        C_cur = [np.asarray(v["C"], float) + x_pose[3 * N + 3 * k:3 * N + 3 * k + 3]
                 for k, v in enumerate(views)]

        # Plane frame for rectification: origin under the formation, e1 along the first
        # camera's right axis so the patches come out the way that camera sees them.
        centroid = np.mean(C_cur, axis=0)
        origin = centroid - (n_b @ centroid - d_b) * n_b
        e1, e2 = plane_basis(n_b, hint=R_cur[0][0])
        standoff = float(np.median([abs(n_b @ C - d_b) for C in C_cur]))
        if not np.isfinite(standoff) or standoff < 0.5:
            return "cameras on the plane"

        Kinv = np.linalg.inv(K)
        h_src, w_src = views[0]["image"].shape[:2]
        Gs, boxes = [], []
        for k in range(N):
            B = np.column_stack([e1, e2, origin - C_cur[k]])
            G = K @ R_cur[k] @ B
            Gs.append(G)
            boxes.append(self._footprint(G, w_src, h_src, MAX_RANGE_FACTOR * standoff))

        usable = [k for k in range(N) if boxes[k] is not None]
        if len(usable) < 2:
            return "fewer than 2 views see the plane"
        extent = max(float(np.max(boxes[k][1] - boxes[k][0])) for k in usable)
        gsd = max(standoff / float(K[1, 1]), extent / MAX_RECT_SIDE)
        cell = 8.0 * gsd

        o_ab = np.array([origin @ e1, origin @ e2])
        feats = {}
        for k in usable:
            lo = np.floor(boxes[k][0] / cell) * cell
            hi = np.ceil(boxes[k][1] / cell) * cell
            rect, valid = self._rectify(views[k]["image"], Gs[k], lo, hi, gsd)
            kps, desc, dense, cells, weight = self._extract(rect, valid)
            xy = kps.cpu().numpy().astype(np.float64)
            ab = np.column_stack([lo[0] + xy[:, 0] * gsd, hi[1] - xy[:, 1] * gsd])
            # Source pixels of the keypoints, through the same G the patch was warped
            # with -- so they are exact measurements however wrong that G was.
            q = np.column_stack([ab, np.ones(len(ab))]) @ Gs[k].T
            uv = q[:, :2] / q[:, 2:3]
            # ...and where the PUBLISHED pose and plane would have put them, in the same
            # in-plane axes: the far-background test is made against GNSS, not against
            # whatever the corrections have since done (see FAR_PARALLAX_FRACTION).
            ray = np.column_stack([uv, np.ones(len(uv))]) @ Kinv.T @ np.asarray(
                views[k]["R"], float)
            C_pub = np.asarray(views[k]["C"], float)
            lam = (float(d_pub) - n_pub @ C_pub) / np.where(
                np.abs(ray @ n_pub) > 1e-9, ray @ n_pub, 1e-9)
            X_pub = C_pub + lam[:, None] * ray
            ab_pub = np.column_stack([X_pub @ e1, X_pub @ e2])
            # How far this pass's geometry has moved the view from where it was published,
            # in the plane: what a prior on "do not move a panel" is measured from.
            moved = (np.median(ab + o_ab - ab_pub, axis=0) if len(ab) else np.zeros(2))
            feats[k] = {"ab": ab, "uv": uv, "desc": desc, "ab_pub": ab_pub, "moved": moved,
                        "dense": dense, "cells": cells, "weight": weight,
                        "lo": lo, "hi": hi,
                        "col0": int(round(lo[0] / cell)), "row0": int(round(-hi[1] / cell))}
        # One transform size for every view of this pass: twice the largest map, so no
        # relative shift two maps can overlap at wraps round onto another.
        # Rounded up to a multiple of 32, which keeps the sizes few: the device plans each
        # new transform size once, and that costs more than the transform.
        size = (-(-2 * max(f["cells"].shape[0] for f in feats.values()) // 32) * 32,
                -(-2 * max(f["cells"].shape[1] for f in feats.values()) // 32) * 32)
        for f in feats.values():
            f["fft_size"] = size
        return {"C_cur": C_cur, "centroid": centroid, "origin": origin, "e1": e1, "e2": e2,
                "standoff": standoff, "gsd": gsd, "cell": cell, "feats": feats,
                "usable": usable}

    def _gated_pairs(self, views, prep, shifts, gate, use_lg=True):
        """
        Matches of every pair of views about the alignment ``shifts`` describes.

        Returns ``(pairs, n_raw, n_far, n_in_gate, raw_per_view)``; ``pairs`` is what
        ``PlanarBA`` takes.
        """
        N = len(views)
        usable, feats, gsd = prep["usable"], prep["feats"], prep["gsd"]
        e1, e2 = prep["e1"], prep["e2"]
        pairs, n_raw, n_far, n_in_gate = [], 0, 0, 0
        raw_per_view = np.zeros(N, dtype=int)
        rng = np.random.default_rng(self.updates)
        for a in range(len(usable)):
            for b in range(a + 1, len(usable)):
                ki, kj = usable[a], usable[b]
                m = self._match(feats[ki], feats[kj], shifts[ki], shifts[kj], gsd, gate,
                                gate=gate, use_lg=use_lg)
                if m is None or len(m[0]) < 6:
                    continue
                mi, mj = m
                n_raw += len(mi)
                raw_per_view[ki] += len(mi)
                raw_per_view[kj] += len(mi)
                disp = ((feats[kj]["ab"][mj] + shifts[kj])
                        - (feats[ki]["ab"][mi] + shifts[ki]))
                keep = np.linalg.norm(disp, axis=1) < gate
                b_hat, far = _baseline(views, ki, kj, e1, e2)
                if np.isfinite(far):
                    pub = feats[kj]["ab_pub"][mj] - feats[ki]["ab_pub"][mi]
                    behind = (pub @ b_hat) > far
                    n_far += int((keep & behind).sum())
                    keep &= ~behind
                mi, mj = mi[keep], mj[keep]
                n_in_gate += len(mi)
                if len(mi) < 6:
                    continue
                if len(mi) > MAX_MATCHES_PER_PAIR:
                    sel = rng.choice(len(mi), MAX_MATCHES_PER_PAIR, replace=False)
                    mi, mj = mi[sel], mj[sel]
                pairs.append((ki, kj, feats[ki]["uv"][mi], feats[kj]["uv"][mj]))
        return pairs, n_raw, n_far, n_in_gate, raw_per_view

    def _resolve_hops(self, views, K, prep, hops, x_pose, n_b, d_b, n_pub, d_pub,
                      sigmas, solve_plane, bound_t, before):
        """
        Solve again from an alignment ``_hops`` moved by whole periods.

        ``x_pose`` / ``(n_b, d_b)`` are the solution ``prep`` was rectified with. Returns
        a result in the shape ``update`` chooses between, or ``None`` when the moved
        alignment does not hold up: it has to keep most of the inliers the solution it
        replaces had, since a true period is as good a fit as the one it started from.
        """
        N = len(views)
        usable = prep["usable"]
        shifts = {k: hops.get(k, np.zeros(2)) for k in usable}
        # The views are registered to a pixel or two and the hypothesis is the whole
        # point, so this is the gated neighbours' job alone.
        pairs, n_raw, n_far, n_in_gate, raw_per_view = self._gated_pairs(
            views, prep, shifts, GATE_TRACK_M, use_lg=False)
        n_gated = sum(len(p[2]) for p in pairs)
        if n_gated < MIN_INLIERS:
            return None
        ba = PlanarBA(K, [np.asarray(v["R"], float) for v in views],
                      [np.asarray(v["C"], float) for v in views],
                      n_b, d_b, prep["origin"], pairs, solve_pose=True, solve_plane=solve_plane)
        pr = self.priors
        absolute = ba.prior_precision(pr["yaw_deg"], pr["tilt_deg"], pr["horizontal_m"],
                                      pr["vertical_m"], sigmas[0], sigmas[1])
        pose_only = absolute.copy()
        pose_only[6 * N:, 6 * N:] = 0.0
        anchor = np.zeros(ba.P)
        anchor[6 * N:] = ba.plane_params_of(n_pub, float(d_pub))
        x0 = np.zeros(ba.P)
        x0[:6 * N] = x_pose[:6 * N]
        for k in usable:
            x0[3 * N + 3 * k:3 * N + 3 * k + 3] += (shifts[k][0] * prep["e1"]
                                                    + shifts[k][1] * prep["e2"])
        x, err, ok = ba.solve([(pose_only, np.zeros(ba.P)), (absolute - pose_only, anchor)],
                              x0=x0, kernel0=KERNEL_TRACK_PX)
        inl = ok & (err < INLIER_PX)
        n_inl = int(inl.sum()) // 2
        ratio = float(inl.mean()) if len(inl) else 0.0
        om, tt, n_new, d_new = ba.unpack(x)
        per_view = np.zeros(N, dtype=int)
        if len(inl):
            np.add.at(per_view, ba.ii.numpy()[inl], 1)
        worst_t = float(np.max(np.linalg.norm(tt, axis=1)))
        worst_r = float(np.rad2deg(np.max(np.linalg.norm(om, axis=1))))
        tilt = float(np.rad2deg(np.arccos(np.clip(n_new @ n_pub, -1.0, 1.0))))
        if n_inl < max(MIN_INLIERS, VERIFY_KEEP_INLIERS * before["n_inl"]) \
                or ratio < MIN_INLIER_RATIO:
            return None
        if worst_t > bound_t or worst_r > 4.0 * pr["yaw_deg"] or tilt > 45.0:
            return None
        if any(per_view[k] < MIN_VIEW_INLIERS <= before["per_view"][k] for k in range(N)):
            return None                                   # it cost a view its support
        moved = ", ".join("drone %s by %.1f m" % (views[k]["drone_id"], np.linalg.norm(v))
                          for k, v in sorted(hops.items()) if np.linalg.norm(v) >= VERIFY_MIN_HOP_M)
        now = time.perf_counter()
        return dict(ba=ba, x=x, err=err, inl=inl, n_inl=n_inl, ratio=ratio, om=om, tt=tt,
                    n_new=n_new, d_new=d_new, per_view=per_view, pairs=pairs, n_gated=n_gated,
                    n_far=n_far, note="a whole period out: moved %s" % moved,
                    explained=n_in_gate * ratio, n_raw=n_raw, left_out=[], worst_t=worst_t,
                    worst_r=worst_r, tilt=tilt, attempt="period", times=(now, now, now))

    # -- the update ----------------------------------------------------------------

    def update(self, views, K, n_pub, d_pub, solve_pose=True, solve_plane=True,
               plane_is_prior_only=True, max_shift_m=0.0, rate=1.0, plane_sigma_m=None):
        """
        One estimation pass.

        ``views``  list of dicts with ``drone_id``, ``image`` (BGR uint8), ``R`` (3x3
                   world->camera, CV convention) and ``C`` (world), all *as published*
        ``n_pub, d_pub``  the published (or formation-derived) plane, ``n . X == d``
        ``plane_is_prior_only``  True when the published plane is a guess to be replaced
                   (formation-relative); False when it is a measurement to be nudged

        Returns ``True`` when the corrections were updated.  The results are left on
        ``self.pose`` / ``self.plane`` / ``self.plane_delta`` and ``self.stats``.
        """
        t_start = time.perf_counter()
        self.updates += 1
        if not self.models.load():
            self.stats = {"skipped": self.models.error or "models unavailable"}
            return False
        if len(views) < 2:
            self.stats = {"skipped": "fewer than 2 usable views"}
            return False

        K = np.asarray(K, dtype=np.float64)
        n_pub = np.asarray(n_pub, dtype=np.float64)
        n_pub = n_pub / np.linalg.norm(n_pub)
        N = len(views)
        ids = [v["drone_id"] for v in views]
        tracking = self.mode == "track"
        pr0 = self.priors

        # Base geometry for this pass: published poses, plus whatever has been earned.
        x_base = np.zeros(6 * N + 3)
        for k, did in enumerate(ids):
            if did in self.pose:
                x_base[3 * k:3 * k + 3] = self.pose[did][0]
                x_base[3 * N + 3 * k:3 * N + 3 * k + 3] = self.pose[did][1]
        n_base, d_base = self._effective_plane(n_pub, float(d_pub), views)

        prep = self._prepare(views, K, x_base, n_base, d_base, n_pub, float(d_pub))
        if isinstance(prep, str):
            self.stats = {"skipped": prep}
            return False
        C_cur, centroid, origin = prep["C_cur"], prep["centroid"], prep["origin"]
        e1, e2, standoff = prep["e1"], prep["e2"], prep["standoff"]
        gsd, cell, feats, usable = prep["gsd"], prep["cell"], prep["feats"], prep["usable"]
        t_feat = time.perf_counter()

        def baseline(ki, kj):
            return _baseline(views, ki, kj, e1, e2)

        # What the standing corrections already move each view by, in the plane: a shift
        # the graph proposes is on top of this, and the test is on the total.
        held = {k: np.array([x_base[3 * N + 3 * k:3 * N + 3 * k + 3] @ e1,
                             x_base[3 * N + 3 * k:3 * N + 3 * k + 3] @ e2]) for k in usable}

        # The stages below run once while tracking. While acquiring they run up to
        # twice: first from the published geometry as it stands, through the tight gate
        # ("prior") -- most of the time it is within a metre or so and simply needs
        # refining, and a wide search can only do harm there; on a wall of identical
        # windows it finds a dozen alignments as good as the right one. Only when the
        # published geometry does not explain the matches is the shift graph consulted
        # ("graph") and the gate opened.
        attempts = ["track"] if tracking else (["prior", "graph"] if solve_pose else ["prior"])
        failure = "no attempt made"
        why_wide = ""
        chosen = fallback = None
        if plane_is_prior_only:
            plane_sigmas = [pr0["plane_tilt_deg"], pr0["plane_offset_m"]]
        else:
            # A raycast plane is a measurement; let the images move it, but not far.
            plane_sigmas = [2.0, 1.0]
        if plane_sigma_m is not None and plane_sigma_m > 0.0:
            plane_sigmas[1] = float(plane_sigma_m)
        bound_shift = max(4.0 * pr0["horizontal_m"], float(max_shift_m or 0.0))

        # A held lock is checked now and then for a whole-period error (see _hops): the
        # views are rectified with the standing corrections, which is exactly the
        # registered state that check wants. If it finds one, this update is the re-solve
        # and snaps like an acquisition; a lock that could only defend itself -- its own
        # matches are, by construction, within the gate -- would otherwise be for good.
        if tracking and solve_pose and self.updates % VERIFY_EVERY == 0:
            hops = self._hops(prep, to_gnss=False)
            if hops is not None:
                per_now = np.array([MIN_VIEW_INLIERS if self.unsupported_streak.get(d, 0) == 0
                                    else 0 for d in ids])
                chosen = self._resolve_hops(
                    views, K, prep, hops, x_base, n_base, d_base, n_pub, d_pub, plane_sigmas,
                    solve_plane, bound_shift,
                    {"n_inl": self.stats.get("inliers", MIN_INLIERS), "per_view": per_now})
                if chosen is not None:
                    attempts = []
                    tracking = False
        for attempt in attempts:
            # ---- stage 3: shift graph (acquisition only) ------------------------------
            shifts = {k: np.zeros(2) for k in usable}
            graph_note = ""
            if attempt == "graph":
                Rn = max(2, int(round(NODE_SHIFT_M / cell)))
                edges, where = [], {k: i for i, k in enumerate(usable)}
                for a in range(len(usable)):
                    for b in range(a + 1, len(usable)):
                        ki, kj = usable[a], usable[b]
                        S = self._surface(feats[ki], feats[kj], 2 * Rn)
                        if S is None:
                            continue
                        # Strike out the relative shifts that would be aligning the far
                        # field. S scores d = a_j - a_i in this pass's frame; against the
                        # published geometry that is d - (held_j - held_i).
                        b_hat, far = baseline(ki, kj)
                        if np.isfinite(far):
                            R = (S.shape[0] - 1) // 2
                            dy, dx = np.mgrid[-R:R + 1, -R:R + 1]
                            rel = held[kj] - held[ki]
                            along = ((dx * cell - rel[0]) * b_hat[0]
                                     + (-dy * cell - rel[1]) * b_hat[1])
                            S = np.where(along > far, 0.0, S)
                        if float(S.max()) > 0.0:
                            edges.append((a, b, S))
                if edges:
                    w_prior = SHIFT_PRIOR_WEIGHT * (cell / SHIFT_PRIOR_SIGMA_M) ** 2
                    # How far the plane may be from where it was published, as a dilation of
                    # the shifts (see solve_shift_graph). Two sigma of the same uncertainty
                    # the bundle adjustment's plane prior uses.
                    if plane_is_prior_only:
                        plane_sig = float(plane_sigma_m) if plane_sigma_m else pr0["plane_offset_m"]
                    else:
                        plane_sig = 1.0
                    reach = min(2.0 * plane_sig, 0.8 * standoff)
                    cam_cells = np.array([[(C_cur[k] - centroid) @ e1 / cell,
                                           -((C_cur[k] - centroid) @ e2) / cell] for k in usable])
                    sh, _score = solve_shift_graph(
                        len(usable), edges, Rn, w_prior, positions=cam_cells,
                        dilation_limits=(-reach / (standoff + reach), reach / (standoff - reach)),
                        dilation_weight=DILATION_WEIGHT * (standoff / plane_sig) ** 2)
                    # S is indexed by rows growing downward; plane b grows upward.
                    for a, k in enumerate(usable):
                        shifts[k] = np.array([sh[a][0] * cell, -sh[a][1] * cell])
                    # The correction is the opposite of the displacement the surfaces score:
                    # d = a_j - a_i is undone by moving i by +d/2 and j by -d/2, i.e. the
                    # node shift convention d = s_i - s_j applies directly.
                    moved = max(float(np.linalg.norm(s)) for s in shifts.values())
                    graph_note = "graph %d edges, max shift %.1f m" % (len(edges), moved)
            t_graph = time.perf_counter()

            # ---- stage 4: matches on common windows -----------------------------------
            gate = GATE_ACQUIRE_M if attempt == "graph" else GATE_TRACK_M
            pairs, n_raw, n_far, n_in_gate, raw_per_view = self._gated_pairs(
                views, prep, shifts, gate)
            t_match = time.perf_counter()

            n_gated = sum(len(p[2]) for p in pairs)
            if n_gated < MIN_INLIERS:
                failure = ("only %d matches within %.1f m of the prior (%d raw)%s"
                           % (n_gated, gate, n_raw, "; " + graph_note if graph_note else ""))
                why_wide = why_wide or failure
                continue

            # ---- stage 5: bundle adjustment -------------------------------------------
            ba = PlanarBA(K, [np.asarray(v["R"], float) for v in views],
                          [np.asarray(v["C"], float) for v in views],
                          n_base, d_base, origin, pairs,
                          solve_pose=solve_pose, solve_plane=solve_plane)
            pr = self.priors
            sig_tilt, sig_off = plane_sigmas
            absolute = ba.prior_precision(pr["yaw_deg"], pr["tilt_deg"], pr["horizontal_m"],
                                          pr["vertical_m"], sig_tilt, sig_off)
            x0 = x_base.copy()
            terms = []
            if tracking:
                # The plane parameters are relative to the base plane, which already is the
                # previous estimate -- so "stay near last time" is a prior centred on zero
                # for the plane and on the previous correction for the poses. The absolute
                # prior stays in for the poses at reduced weight: it is what stops the whole
                # formation drifting along the gauge directions the images cannot see.
                temporal = ba.prior_precision(pr["track_rot_deg"], pr["track_rot_deg"],
                                              pr["track_pos_m"], pr["track_pos_m"],
                                              pr["track_plane_tilt_deg"],
                                              pr["track_plane_offset_m"])
                centre = x_base.copy()
                centre[6 * N:] = 0.0
                x0[6 * N:] = 0.0
                # The plane keeps its absolute prior at full weight, centred on the
                # PUBLISHED plane as seen from the one being tracked. Step by step the
                # temporal prior lets it go anywhere; this is what says how far from the
                # published surface it may end up. Without it a formation climbing past
                # the roofline walks the plane, a degree an update, onto whatever is
                # behind the building -- by the time the facade has left the frame the
                # estimate is twelve degrees and a metre and a half away, describing
                # nothing.
                anchor = np.zeros(ba.P)
                anchor[6 * N:] = ba.plane_params_of(n_pub, float(d_pub))
                pose_only = absolute.copy()
                pose_only[6 * N:, 6 * N:] = 0.0
                plane_only = absolute - pose_only
                terms = [(temporal, centre), (pose_only * 0.25, np.zeros(ba.P)),
                         (plane_only, anchor)]
            else:
                x0[6 * N:] = 0.0
                # Start the poses from the graph's shifts: that is the basin it chose.
                for k in usable:
                    t_world = shifts[k][0] * e1 + shifts[k][1] * e2
                    x0[3 * N + 3 * k:3 * N + 3 * k + 3] += t_world
                terms = [(absolute, np.zeros(ba.P))]

            x, err, ok = ba.solve(terms, x0=x0,
                                  kernel0=KERNEL_ACQUIRE_PX if attempt == "graph"
                                  else KERNEL_TRACK_PX)
            t_solve = time.perf_counter()

            inl = ok & (err < INLIER_PX)
            n_inl = int(inl.sum()) // 2
            ratio = float(inl.mean()) if len(inl) else 0.0
            om, tt, n_new, d_new = ba.unpack(x)

            per_view = np.zeros(N, dtype=int)
            if len(inl):
                ii = ba.ii.numpy()
                np.add.at(per_view, ii[inl], 1)

            worst_t = float(np.max(np.linalg.norm(tt, axis=1)))
            worst_r = float(np.rad2deg(np.max(np.linalg.norm(om, axis=1))))
            tilt = float(np.rad2deg(np.arccos(np.clip(n_new @ n_pub, -1.0, 1.0))))
            # The inspector's clamp can widen this, never narrow it: 3 m was sized for a
            # translation-only refiner, and on a bad GNSS day the true error is larger. A
            # correction that is clamped is a view left visibly out of place; this bound is
            # only here to throw away solutions no pose error could explain.
            bound_t = max(4.0 * pr["horizontal_m"], float(max_shift_m or 0.0))
            if n_inl < MIN_INLIERS or ratio < MIN_INLIER_RATIO:
                failure = ("no consensus: %d inliers of %d (%.0f%%)%s"
                           % (n_inl, n_gated, 100.0 * ratio,
                              "; " + graph_note if graph_note else ""))
                why_wide = why_wide or failure
                continue
            if worst_t > bound_t or worst_r > 4.0 * pr["yaw_deg"] or tilt > 45.0:
                failure = ("implausible solution: %.1f m, %.1f deg, plane tilt %.1f deg"
                           % (worst_t, worst_r, tilt))
                continue
            # What this attempt explains of everything the matcher returned -- not of the
            # capped sample the solve ran on: the share inside the gate, times the share
            # of those the solution agrees with.
            explained = n_in_gate * ratio
            left_out = [ids[k] for k in range(N)
                        if raw_per_view[k] >= 40 and per_view[k] < MIN_VIEW_INLIERS]
            result = dict(ba=ba, x=x, err=err, inl=inl, n_inl=n_inl, ratio=ratio, om=om, tt=tt,
                          n_new=n_new, d_new=d_new, per_view=per_view, pairs=pairs,
                          n_gated=n_gated, n_far=n_far, note=graph_note, explained=explained,
                          n_raw=n_raw, left_out=left_out, worst_t=worst_t, worst_r=worst_r,
                          tilt=tilt, attempt=attempt,
                          times=(t_graph, t_match, t_solve))
            if attempt == "prior" and "graph" in attempts:
                if explained >= PRIOR_FIRST_SHARE * n_raw and not left_out:
                    # It explains most of what was matched and leaves no view out:
                    # there is nothing for a wide search to find.
                    result["note"] = "published geometry confirmed"
                    chosen = result
                    break
                fallback = result
                why_wide = ("the published geometry explains %.0f of %d matches%s"
                            % (explained, n_raw, "" if not left_out
                               else ", none for drones %s" % left_out))
                failure = why_wide
                continue
            chosen = result
            break
        if chosen is None and fallback is None:
            return self._fail(failure)
        if chosen is not None and chosen["attempt"] == "period":
            pass
        elif chosen is not None and chosen["attempt"] == "graph" and fallback is not None \
                and chosen["explained"] < WIDE_MUST_BEAT * fallback["explained"]:
            # Two alignments, each explaining a comparable share of the matches: that is
            # what a wall of identical windows looks like, and the images cannot say
            # which is right. The published geometry can -- it is the one that needed no
            # panel moved -- so the wide search has to be decisively better to overrule it.
            fallback["note"] = ("kept the published geometry: the wide search explained "
                                "%.0f matches against its %.0f"
                                % (chosen["explained"], fallback["explained"]))
            chosen = fallback
        elif chosen is None:
            fallback["note"] = "kept the published geometry: the wide search failed (%s)" % failure
            chosen = fallback
        elif chosen["attempt"] == "graph" and why_wide:
            chosen["note"] = chosen["note"] + " (searched wide: %s)" % why_wide

        # ---- verify: is any view a whole period out? ---------------------------------
        if attempts and not tracking and solve_pose:
            for _round in range(2):
                x_sol = np.concatenate([chosen["om"].ravel(), chosen["tt"].ravel()])
                n_sol, d_sol = chosen["n_new"], float(chosen["d_new"])
                prep2 = self._prepare(views, K, x_sol, n_sol, d_sol, n_pub, float(d_pub))
                if isinstance(prep2, str):
                    break
                hops = self._hops(prep2, to_gnss=VERIFY_TIES_TO_GNSS)
                if hops is None:
                    break
                again = self._resolve_hops(views, K, prep2, hops, x_sol, n_sol, d_sol, n_pub,
                                           d_pub, plane_sigmas, solve_plane, bound_shift, chosen)
                if again is None:
                    break
                again["note"] = "%s; after: %s" % (again["note"], chosen["note"] or chosen["attempt"])
                again["times"] = chosen["times"]
                chosen = again

        ba, x, err, inl = chosen["ba"], chosen["x"], chosen["err"], chosen["inl"]
        n_inl, ratio, om, tt = chosen["n_inl"], chosen["ratio"], chosen["om"], chosen["tt"]
        n_new, d_new, per_view, pairs = (chosen["n_new"], chosen["d_new"],
                                         chosen["per_view"], chosen["pairs"])
        n_gated, n_far, graph_note = chosen["n_gated"], chosen["n_far"], chosen["note"]
        worst_t, worst_r, tilt = chosen["worst_t"], chosen["worst_r"], chosen["tilt"]
        t_graph, t_match, t_solve = chosen["times"]

        # ---- accept ---------------------------------------------------------------
        # Acquisition snaps (it is the lock); tracking low-passes, so a converged
        # alignment does not shimmer with each update's choice of matches.
        a = 1.0 if not tracking else float(np.clip(rate, 0.05, 1.0))
        if solve_pose:
            new_pose = {}
            for k, did in enumerate(ids):
                prev = self.pose.get(did)
                if per_view[k] < MIN_VIEW_INLIERS:
                    if prev is not None:
                        new_pose[did] = prev          # unmeasured this pass: keep it
                    continue
                if prev is None or not tracking:
                    new_pose[did] = (om[k], tt[k])
                else:
                    new_pose[did] = (prev[0] + a * (om[k] - prev[0]),
                                     prev[1] + a * (tt[k] - prev[1]))
            self.pose = new_pose
        if solve_plane:
            if tracking and self.plane is not None:
                n_mix = n_base + a * (n_new - n_base)
                n_new = n_mix / np.linalg.norm(n_mix)
                d_new = d_base + a * (d_new - d_base)
            self.plane = (n_new, float(d_new))
            # Also as a delta on the published plane, for sources where that plane is a
            # live measurement the estimate should ride on rather than replace.
            foot = centroid - (n_pub @ centroid - float(d_pub)) * n_pub
            self.plane_delta = (rotvec_between(n_pub, n_new),
                                float((d_new - n_new @ foot) / max(n_new @ n_pub, 1e-6)))

        self.mode = "track"
        self.lost = 0
        # A view that had support and has lost it is not tracking any more, whatever the
        # others are doing: its matches have left the gate, and held, its correction is
        # simply wrong from then on (a drone whose picture lags its telemetry does this
        # every time the formation stops). One wide search per episode; a view that
        # still finds nothing after it -- one looking over the roof at the sky -- is left
        # alone until it earns support by itself, or it would cost a search every second.
        for k, did in enumerate(ids):
            if per_view[k] >= MIN_VIEW_INLIERS:
                self.unsupported_streak[did] = 0
                held = self.supported_streak.get(did, 0) + 1
                self.supported_streak[did] = held
                if held >= RELOCK_REARM:
                    self.gave_up.discard(did)
                continue
            self.supported_streak[did] = 0
            streak = self.unsupported_streak.get(did, 0) + 1
            self.unsupported_streak[did] = streak
            if streak >= RELOCK_AFTER and did not in self.gave_up and solve_pose:
                self.gave_up.add(did)
                self.mode = "acquire"
        med = float(np.median(err[inl])) if inl.any() else float("nan")
        self.stats = {
            "mode": "track" if tracking else "acquire", "attempt": chosen["attempt"],
            "views": N, "pairs": len(pairs), "matches": n_gated, "inliers": n_inl,
            "ratio": ratio, "median_px": med,
            "worst_shift": worst_t, "worst_rot_deg": worst_r,
            "plane_tilt_deg": tilt,
            "plane_offset": float(d_new - n_new @ (centroid - (n_pub @ centroid
                                                               - float(d_pub)) * n_pub)),
            "unsupported": [ids[k] for k in range(N) if per_view[k] < MIN_VIEW_INLIERS],
            "note": graph_note, "far": n_far,
            "matcher": "lightglue" if self.models.lg is not None else "nearest",
            "ms": {"features": 1e3 * (t_feat - t_start), "graph": 1e3 * (t_graph - t_feat),
                   "match": 1e3 * (t_match - t_graph), "solve": 1e3 * (t_solve - t_match),
                   "total": 1e3 * (time.perf_counter() - t_start)},
        }
        return True

    def _fail(self, why):
        if self.mode == "track":
            self.lost += 1
            if self.lost >= TRACK_LOST_UPDATES:
                # Hold the correction, drop the lock: the next pass searches wide again.
                self.mode = "acquire"
                self.lost = 0
        self.stats = {"held": why, "mode": self.mode, "lost": self.lost}
        return False

    def _effective_plane(self, n_pub, d_pub, views):
        """The plane to work from: the standing estimate if there is one."""
        if self.mode == "track" and self.plane is not None:
            return self.plane[0], self.plane[1]
        return n_pub, d_pub

    # -- what the render path applies -----------------------------------------------

    def dpose(self):
        """``{drone_id: (dR, dC)}`` in the shape ``PlanarStitcher._apply_correction`` takes."""
        if not self.pose:
            return None
        return {did: (rotvec_to_matrix(om), np.asarray(t, dtype=np.float64))
                for did, (om, t) in self.pose.items()}

    def rotvecs(self):
        """``{drone_id: omega}``: the same rotations as vectors, which is what blends."""
        return {did: np.asarray(om, dtype=np.float64) for did, (om, _t) in self.pose.items()}
