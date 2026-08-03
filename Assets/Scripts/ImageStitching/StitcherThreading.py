import numpy as np
import cv2
import glob
import os
import sys
import torch
import time
import traceback
from transformers import SuperPointForKeypointDetection
# from torch.quantization import quantize_dynamic
from numba import jit
import queue
import threading
import mmap
import struct
import networkx as nx
import random
from PIL import Image

from collections import deque

from BaseStitcher import *


# Camera horizontal FOV is ~83 deg (DJI Mini 3 Pro, 82.1 deg diagonal at 16:9 -- see
# screenSpawn.cs). Two cameras share field-of-view only if their yaw headings are closer
# than the FOV; require a margin so there is enough common region to stitch. If adjacent
# selected cameras are farther apart than this, the perspectives can't overlap, so we skip
# the warp entirely and fall back to the individual feeds.
MAX_STITCH_YAW_SEPARATION_DEG = 75.0   # FOV (~83 deg) minus a small overlap margin
# Failing-gate reasons. These are the *pre-shift* values: write_panorama_memory packs
# them one bit up, above the good/bad flag, so REASON_X here becomes C#'s REASON_X.
# i.e. 8 -> C# REASON_NO_OVERLAP (1<<4), 16 -> C# REASON_TOO_FEW_IMAGES (1<<5).
# Adding a reason here means widening the pack mask in write_panorama_memory too.
REASON_NO_OVERLAP = 8                  # keep in sync with C# REASON_NO_OVERLAP
REASON_TOO_FEW_IMAGES = 16             # fewer than 3 selected feeds -> can't form L/C/R
REASON_PLANE_INVALID = 32              # planar: no usable scene plane / pose on the wire

# How often the debug panorama snapshot is written to disk, in seconds. The JPEG
# encode + write is a stall inside the shared-memory read/write loop, so it is
# throttled rather than run on every frame.
DEBUG_PANO_WRITE_PERIOD = 1.0

# Minimum period between render-thread wake-ups, in seconds. Unity publishes new
# feeds at 20 Hz (sendInterval = 0.05 in PyUniSharingFast), so signalling faster
# than that only re-renders identical pixels -- and because the render and warp
# threads share one GPU and one GIL, that wasted work directly slows the warp
# update. Set just above the publish rate so the render never aliases below 20 Hz.
RENDER_MIN_PERIOD = 0.045

# A proper left/centre/right panorama needs at least this many distinct feeds.
# With fewer, get_subsets_from_order wraps around and would stitch an image with
# itself, so we skip the warp entirely and fall back to the individual feeds.
MIN_STITCH_IMAGES = 3

# A planar mosaic only needs two overlapping views to be worth showing.
MIN_PLANAR_IMAGES = 2

# ---------------------------------------------------------------------------------
# Shared-memory layout (wire v2). Mirrors the meta*/block* constants in
# PyUniSharingFast.cs -- change one, change the other.
#
# Metadata is a contiguous prefix of v1 fields (0..252, read sequentially by
# readMetadataMemory) followed by a v2 tail addressed by absolute offset. Every tail
# offset is 4-byte aligned so the seqlock counter can be written atomically.
# ---------------------------------------------------------------------------------
METADATA_SIZE = 412                  # must equal PyUniSharingFast.metadataSize
PLANAR_WIRE_VERSION = 2

# The v1 prefix ends at 253; the tail starts at 256 so every 4-byte field is aligned.
META_BLOCK_HEADER_SIZE_OFFSET = 256  # int32
META_WIRE_VERSION_OFFSET = 260       # int32
META_FX_OFFSET = 264                 # float32 fx, fy, cx, cy
META_PLANAR_CANVAS_OFFSET = 280      # int32 width, height
META_PLANAR_MPP_OFFSET = 288         # float32 metres per pixel, then max range
META_PLANAR_FEATHER_OFFSET = 296     # int32
META_PLANAR_ANISO_OFFSET = 300       # float32 aniso max, then min coverage
META_PLANAR_POSE_SOURCE_OFFSET = 308  # uint8, then uint8 psnr gate
# 310-311 padding
META_DYN_SEQ_OFFSET = 312            # int32 seqlock counter
META_PLANE_N_OFFSET = 316            # float32 nx, ny, nz, d
META_PLANE_VALID_OFFSET = 332        # uint8, then uint8 mode
# 334-335 padding
META_GIMBAL_PITCH_OFFSET = 336       # float32
META_CENTRE_DRONE_OFFSET = 340       # int32, -1 = none. Inside the seqlock on purpose:
                                     # the canvas is framed on this drone, so pairing it
                                     # with another frame's plane tears the same way a
                                     # torn normal does.

# Per-drone block header. v1 is flag|droneId|heading; v2 appends the camera pose that
# was snapshotted with the image. Unity advertises which one it is writing in
# metadata's block_header_size, so both producers can coexist (ImageSharing.cs in the
# DJI scene stays on v1 -- real drones publish no position).
BLOCK_HEADER_SIZE_V1 = 12
BLOCK_HEADER_SIZE_V2 = 48
BLOCK_CAM_POS_OFFSET = 12            # float32 x, y, z  (Unity world, LEFT-handed)
BLOCK_CAM_ROT_OFFSET = 24            # float32 x, y, z, w (Unity Transform.rotation)
BLOCK_CAPTURE_TIME_OFFSET = 40       # float32
BLOCK_POSE_STATUS_OFFSET = 44        # int32 bitfield
POSE_VALID = 1 << 0
POSE_GROUND_TRUTH = 1 << 1
POSE_NOISE_INJECTED = 1 << 2


class RateMeter:
    """
    Tracks how often an event fires, averaged over a trailing time window.

    Call ``tick()`` once per loop iteration; ``hz`` returns the average
    frequency (completions per second) over the last ``window`` seconds.
    """

    def __init__(self, window=5.0):
        self.window = window
        self._times = deque()

    def tick(self):
        now = time.perf_counter()
        self._times.append(now)
        cutoff = now - self.window
        while self._times and self._times[0] < cutoff:
            self._times.popleft()

    @property
    def hz(self):
        if len(self._times) < 2:
            return 0.0
        span = self._times[-1] - self._times[0]
        if span <= 0:
            return 0.0
        # (n - 1) intervals over the elapsed span = average rate.
        return (len(self._times) - 1) / span

# --- UDIS ---
HAS_UDIS = False
try:
    sys.path.append(os.path.abspath("UDIS2_main\Warp\Codes"))
    import UDIS2_main.Warp.Codes.utils_udis as udis_utils
    import UDIS2_main.Warp.Codes.utils_udis.torch_DLT as torch_DLT
    import UDIS2_main.Warp.Codes.grid_res as grid_res
    from UDIS2_main.Warp.Codes.network import build_output_model, get_stitched_result, Network, build_new_ft_model
    from UDIS2_main.Warp.Codes.loss import cal_lp_loss2
    from UDISStitcher import *
    HAS_UDIS = True
except ImportError as e:
    print("UDIS modules could not be imported. UDIS stitcher will not be available.")
    print(e)

# --- NIS ---
HAS_NIS = False
try:
    sys.path.append(os.path.abspath("Neural_Image_Stitching_main"))
    import Neural_Image_Stitching_main.srwarp
    import Neural_Image_Stitching_main.utils as nis_utils
    from Neural_Image_Stitching_main.models.ihn import *
    from Neural_Image_Stitching_main.models import *
    from Neural_Image_Stitching_main import stitch
    import Neural_Image_Stitching_main.pretrained
    from NISStitcher import *
    HAS_NIS = True
except ImportError as e:
    print("NIS modules could not be imported. NIS stitcher will not be available.")
    print(e)

# --- REWARP ---
HAS_REWARP = False
try:
    sys.path.append(os.path.abspath("Residual_Elastic_Warp_main"))
    import Residual_Elastic_Warp_main.models
    import Residual_Elastic_Warp_main.utils
    from REStitcher import *
    HAS_REWARP = True
except ImportError as e:
    print("REWARP modules could not be imported. REWARP stitcher will not be available.")
    print(e)

# --- STABSTITCH ---
HAS_STABSTITCH = False
try:
    from StabStitcher import StabStitcher
    HAS_STABSTITCH = True
except ImportError as e:
    print("StabStitch modules could not be imported. STABSTITCH stitcher will not be available.")
    print(e)

# --- PLANAR ---
HAS_PLANAR = False
try:
    from PlanarStitcher import PlanarStitcher
    HAS_PLANAR = True
except ImportError as e:
    print("Planar modules could not be imported. PLANAR stitcher will not be available.")
    print(e)

# Activate environnement
# cmd
# cd Assets\Scripts\ImageStitching
# python StitcherThreading.py

class StitcherManager:
    def __init__(self, device ="cpu"):

        self.stitchers = {
            "CLASSIC": BaseStitcher(algorithm=1, trees=5, checks=50, ratio_thresh=0.7, score_threshold=0.05, device=device),
            "UDIS": UDISStitcher() if HAS_UDIS else None,
            "NIS": NISStitcher() if HAS_NIS else None,
            "REWARP": REStitcher() if HAS_REWARP else None,
            "STABSTITCH": StabStitcher() if HAS_STABSTITCH else None,
            "PLANAR": PlanarStitcher() if HAS_PLANAR else None,
        }
        

        # Manually remove models of NIS from GPU because they load them directly on GPU
        if HAS_NIS:
            self.stitchers["NIS"].model.cpu(), self.stitchers["NIS"].H_model.cpu()

        self.active_stitcher = self.stitchers["STABSTITCH"]
        self.active_stitcher_type = "STABSTITCH"
        self.active_matcher_type = "BF"
        self.device = device
        
        self.switching_lock1 = threading.Lock()
        self.switching_lock2 = threading.Lock()
        self.info_lock = threading.Lock()
        self.new_images_event = threading.Event()  # signalled by first_thread

        self.stitcherTypes = [k for k, v in self.stitchers.items() if v is not None]
        self.cylidnricalWarp = False
        self.isRANSAC = False
        self.headAngle = 0
        self.print_rate = True  # gates the per-loop stitch/warp Hz prints (driven by Unity metadata)
        self.shared_images = None
        self.shared_drone_ids = None
        self.shared_headings = None
        self.known_order = None  # Store the known order of images

        # Full view records (image + pose + timing), published under info_lock
        # alongside the legacy parallel lists above. Used by the PLANAR path.
        self.shared_views = None

        # Planar geometry inputs, refreshed from metadata each loop.
        self.intrinsics = None       # (fx, fy, cx, cy)
        self.scene_plane = None      # dict from read_dynamic_state
        self.planar_config = {}
        self.wire_version = 0
        self._planar_wire_warned = False

        self.processedImageWidth = None
        self.processedImageHeight = None
        self.batchImageWidth = None
        self.batchImageHeight = None

        self.panoram_queue = queue.Queue(1)

        # Set the stitcher to setup the device properly
        self.set_stitcher(self.active_stitcher_type, onlyIHN=False)

    def update_planar_metadata(self, output):
        """Refresh the planar geometry inputs from a metadata read."""
        self.wire_version = output.get("wire_version", 0)
        self.intrinsics = output.get("intrinsics")
        self.planar_config = {
            "canvas": output.get("planar_canvas", (0, 0)),
            "metres_per_pixel": output.get("planar_metres_per_pixel", 0.0),
            "max_range": output.get("planar_max_range", 0.0),
            "feather_px": output.get("planar_feather_px", 0),
            "aniso_max": output.get("planar_aniso_max", 0.0),
            "min_coverage": output.get("planar_min_coverage", 0.0),
            "pose_source": output.get("planar_pose_source", 0),
            "psnr_gate": output.get("planar_psnr_gate", False),
        }

    def planar_inputs_ready(self):
        """
        True when Unity is publishing everything the planar backbone needs.

        Warns once rather than per frame: a v1 producer (the DJI scene, or an older
        Unity build) publishes no pose at all, and the useful thing is to say so and
        fall back to the feeds, not to flood the console.
        """
        reasons = []
        if self.wire_version != PLANAR_WIRE_VERSION:
            reasons.append(f"wire version {self.wire_version} != {PLANAR_WIRE_VERSION}")
        if not self.intrinsics or self.intrinsics[0] <= 0.0:
            reasons.append("no camera intrinsics")
        if self.scene_plane is None:
            reasons.append("no scene plane")

        if reasons:
            if not self._planar_wire_warned:
                self._planar_wire_warned = True
                print("[PLANAR] unavailable: " + "; ".join(reasons) +
                      ". Falling back to individual feeds.")
            return False

        self._planar_wire_warned = False
        return True

    def set_stitcher(self, stitcher_type, onlyIHN):
        """
        Safely switch the active stitcher.
        Waits for both threads to finish their current work before switching.
        """
        
        # Remove devices from GPU
        if self.active_stitcher_type == "CLASSIC":
            self.active_stitcher.superpoint_model.cpu()
        elif self.active_stitcher_type == "UDIS":
            self.active_stitcher.net.cpu()
        elif self.active_stitcher_type == "NIS":
            self.active_stitcher.model.cpu(), self.active_stitcher.H_model.cpu()
        elif self.active_stitcher_type == "REWARP":
            self.active_stitcher.model.cpu(), self.active_stitcher.H_model.cpu()
        elif self.active_stitcher_type == "STABSTITCH":
            self.active_stitcher.spatial_net.cpu()
            self.active_stitcher.temporal_net.cpu()
            self.active_stitcher.smooth_net.cpu()
        # PLANAR has no networks to evict: its warp is closed-form from pose.

        torch.cuda.empty_cache()
        if self.stitchers.get(stitcher_type) is None:
            print(f"[StitcherManager] '{stitcher_type}' is unavailable "
                  f"(module missing); keeping {self.active_stitcher_type}.")
            return
        self.active_stitcher = self.stitchers[stitcher_type]
        self.active_stitcher_type = stitcher_type
        print(f"Switched to {self.active_stitcher.__class__.__name__}")
        if self.active_stitcher_type == "CLASSIC":
            self.active_stitcher.superpoint_model.to(self.device)
        elif self.active_stitcher_type == "UDIS":
            self.active_stitcher.net.to(self.device)
        elif self.active_stitcher_type == "NIS":
            self.active_stitcher.model.to(self.device), self.active_stitcher.H_model.to(self.device)
            self.active_stitcher.onlyIHN = onlyIHN
        elif self.active_stitcher_type == "REWARP":
            self.active_stitcher.model.to(self.device), self.active_stitcher.H_model.to(self.device)
        elif self.active_stitcher_type == "STABSTITCH":
            self.active_stitcher.spatial_net.to(self.device)
            self.active_stitcher.temporal_net.to(self.device)
            self.active_stitcher.smooth_net.to(self.device)

    def set_fusion_mode(self, fusion_mode: str):
        """
        Update fusion_mode on the active stitcher.
        Raises NotImplementedError if the stitcher does not support it.
        """
        if hasattr(self.active_stitcher, 'fusion_mode'):
            if self.active_stitcher.fusion_mode != fusion_mode:
                self.active_stitcher.fusion_mode = fusion_mode
                print(f"[StitcherManager] fusion_mode set to '{fusion_mode}' on {self.active_stitcher_type}")
        else:
            raise NotImplementedError(
                f"Stitcher '{self.active_stitcher_type}' does not implement fusion_mode. "
                f"Switch to UDIS or STABSTITCH to use fusion modes."
            )

    def checkHyperparaChanges(self, output : dict):
        """
        Checks for changes in stitching type or hyperparameters and updates them if necessary.

        Parameters:
        - output (dict): A dictionary containing stitching settings and hyperparameters
        """
        
        typeOfStitcher, isCylindrical, matcherType, isRANSAC  = output["typeOfStitcher"], output["isCylindrical"], output["matcherType"], output["isRANSAC"]
        checks, ratio_thresh, score_threshold, focal, onlyIHN = output["checks"], output["ratio_thresh"], output["score_threshold"], output["focal"], output["onlyIHN"]
        fusion_mode = output.get("fusion_mode", "REFERENCE")
        blur_kernel_size = output.get("blur_kernel_size", 41)
        blur_sigma = output.get("blur_sigma", 15.0)
        border_size = output.get("border_size", 60)
        quality_enabled = output.get("quality_enabled", True)
        quality_threshold = output.get("quality_threshold", 18.0)
        batchImageWidth, batchImageHeight = output["Sizes"][:2]

        def has_stitcher_changes():
            return (
                self.active_stitcher_type != typeOfStitcher and typeOfStitcher in self.stitcherTypes
            ) or (
                self.active_stitcher.cylindricalWarp != isCylindrical
            )

        def has_hyperparameter_changes():
            return (
                self.active_stitcher.active_matcher_type != matcherType or
                self.active_stitcher.isRANSAC != isRANSAC or
                self.active_stitcher.checks != checks or
                self.active_stitcher.ratio_thresh != ratio_thresh or
                self.active_stitcher.score_threshold != score_threshold or
                self.active_stitcher.focal != focal or
                self.batchImageWidth != batchImageWidth or
                self.batchImageHeight != batchImageHeight
            )

        
        if has_stitcher_changes():
            with self.switching_lock1:
                with self.switching_lock2:
                    self.set_stitcher(typeOfStitcher, onlyIHN)
                    self.changeCylindrical(isCylindrical)
                    self.changeCalculationsHyperpara(output)
                    pass
        if has_hyperparameter_changes():
            with self.switching_lock1:
                self.changeCalculationsHyperpara(output)
        
        if self.active_stitcher_type == "NIS" and self.active_stitcher.onlyIHN != onlyIHN:
            with self.switching_lock2:
                self.active_stitcher.onlyIHN = onlyIHN

        if getattr(self.active_stitcher, 'fusion_mode', '__unset__') != fusion_mode:
            with self.switching_lock1:
                self.set_fusion_mode(fusion_mode)

        if getattr(self.active_stitcher, 'blur_kernel_size', None) != blur_kernel_size:
            with self.switching_lock1:
                self.active_stitcher.blur_kernel_size = blur_kernel_size
        if getattr(self.active_stitcher, 'blur_sigma', None) != blur_sigma:
            with self.switching_lock1:
                self.active_stitcher.blur_sigma = blur_sigma
        if getattr(self.active_stitcher, 'border_size', None) != border_size:
            with self.switching_lock1:
                self.active_stitcher.border_size = border_size

        # StabStitch panorama-quality fallback toggle + threshold
        if getattr(self.active_stitcher, 'quality_enabled', None) != quality_enabled:
            with self.switching_lock1:
                self.active_stitcher.quality_enabled = quality_enabled
        if getattr(self.active_stitcher, 'quality_threshold', None) != quality_threshold:
            with self.switching_lock1:
                self.active_stitcher.quality_threshold = quality_threshold

    def process_stitching(self, images, num_pano_img=3, views=None):
        """
        Simplified stitching process using known order from drone IDs.
        No need for homography computation - just stitch based on known order.

        For STABSTITCH the fast render path (stab_pano) only uses cached
        warp parameters and standalone TPS functions — it never touches the
        neural networks — so it can run without ``switching_lock2``.  The
        slow warp computation is handled by ``warp_computation_thread``
        which does acquire ``switching_lock2``.
        """
        if self.known_order is None or len(self.known_order) != len(images):
            print(f"[WARNING] Known order not set or length mismatch. Expected {len(images)} images.")
            return

        if self.active_stitcher_type == "PLANAR":
            # Planar takes its views straight from Unity's selection -- it does not use
            # the heading-sorted ring order or the left/centre/right subsets, which are
            # concepts belonging to the radially-outward configuration.
            if views is None or len(views) < MIN_PLANAR_IMAGES:
                if self.panoram_queue.empty():
                    self.panoram_queue.put((None, False, REASON_TOO_FEW_IMAGES))
                return
            if not self.planar_inputs_ready():
                if self.panoram_queue.empty():
                    self.panoram_queue.put((None, False, REASON_PLANE_INVALID))
                return

            # Lock-free like the STABSTITCH arm: the planar solve touches no networks.
            pano, quality_ok, quality_reason = self.active_stitcher.planar_pano(
                views, self.intrinsics, self.scene_plane, self.planar_config)
            if self.panoram_queue.empty():
                self.panoram_queue.put((pano, quality_ok, quality_reason))
            return

        # Need at least 3 distinct feeds to form a left/centre/right panorama.
        # With fewer, get_subsets_from_order wraps around and stitches an image
        # with itself (poor pano) -- skip stitching and fall back to feeds.
        if len(images) < MIN_STITCH_IMAGES:
            if self.panoram_queue.empty():
                self.panoram_queue.put((None, False, REASON_TOO_FEW_IMAGES))
            return

        if self.active_stitcher_type == "STABSTITCH":
            # Fast path: no switching_lock2 needed — render only uses cached
            # warp params + standalone TPS warp (no neural network access).
            order = np.array(self.known_order)
            subset1, subset2 = self.get_subsets_from_order(order, len(images))

            # Cheap geometric pre-check: only attempt the warp if adjacent selected
            # cameras' yaw headings are close enough to actually share field-of-view.
            # If either pair (left-centre or centre-right) is wider than the camera FOV,
            # the perspectives don't overlap -- skip stitching and fall back to feeds.
            def _yaw_gap(a_idx, b_idx):
                d = abs(self.shared_headings[a_idx] - self.shared_headings[b_idx])
                return 360 - d if d > 180 else d   # circular distance, matches get_subsets_from_order
            gap_lc = _yaw_gap(subset1[0], subset1[1])
            gap_cr = _yaw_gap(subset2[0], subset2[1])
            if gap_lc > MAX_STITCH_YAW_SEPARATION_DEG or gap_cr > MAX_STITCH_YAW_SEPARATION_DEG:
                if self.panoram_queue.empty():
                    self.panoram_queue.put((None, False, REASON_NO_OVERLAP))
                return

            pano, quality_ok, quality_reason = self.active_stitcher.stab_pano(images, subset1, subset2)

            # Always queue (pano, quality_ok, quality_reason): when quality_ok is
            # False the pano is None and only the quality flag + failing-gate
            # reason are forwarded to Unity so it can switch to the individual
            # feeds and log why.
            if self.panoram_queue.empty():
                self.panoram_queue.put((pano, quality_ok, quality_reason))
            return

        # All other stitchers: original behaviour with switching_lock2
        with self.switching_lock2:
            order = np.array(self.known_order)

            if self.active_stitcher_type == "CLASSIC":
                pano = self.stitch_with_known_order(images, order, num_pano_img)
            elif self.active_stitcher_type == "UDIS":
                subset1, subset2 = self.get_subsets_from_order(order, len(images))
                pano = self.active_stitcher.UDIS_pano(images, subset1, subset2)
            elif self.active_stitcher_type == "NIS":
                subset1, subset2 = self.get_subsets_from_order(order, len(images))
                pano = None  # Placeholder
            elif self.active_stitcher_type == "REWARP":
                subset1, subset2 = self.get_subsets_from_order(order, len(images))
                pano = None  # Placeholder
            else:
                pano = None

            # Non-STABSTITCH stitchers have no quality estimate: always good.
            if pano is not None and self.panoram_queue.empty():
                self.panoram_queue.put((pano, True, 0))

    def get_subsets_from_order(self, order, num_images):
        """
        Get image subsets based on head angle and known order.
        Selects 3 images centered around the head direction (left, center, right).
        Reference image is the one with heading closest to headAngle.
        """
        # Find the drone with heading closest to headAngle
        min_diff = float('inf')
        closest_drone_idx = 0
        
        for i, heading in enumerate(self.shared_headings):
            # Calculate circular distance (handling wraparound)
            diff = abs(heading - self.headAngle)
            if diff > 180:
                diff = 360 - diff
            
            if diff < min_diff:
                min_diff = diff
                closest_drone_idx = i
        
        # Find the position of this drone in the order array
        ref_idx = int(np.where(order == closest_drone_idx)[0][0])
        
        # Get indices for left, center, and right
        left_idx = (ref_idx - 1) % num_images
        right_idx = (ref_idx + 1) % num_images
        
        # Create subsets: left subset is [left, center], right subset is [center, right]
        subset1 = np.array([order[left_idx], order[ref_idx]])
        subset2 = np.array([order[ref_idx], order[right_idx]])
        

        return subset1, subset2

    def stitch_with_known_order(self, images, order, num_pano_img):
        """
        Simplified stitching for when order is known.
        Just arranges images side by side without complex homography computation.
        """
        subset1, subset2 = self.get_subsets_from_order(order, len(images))
        
        # Simple horizontal concatenation
        selected_indices = np.concatenate([subset1[::-1], subset2[1:]])  # Avoid duplicate ref image
        selected_images = [images[i] for i in selected_indices]
        
        if len(selected_images) == 0:
            return images[0]
        
        # Resize all images to same height
        target_height = self.processedImageHeight if self.processedImageHeight else images[0].shape[0]
        resized_images = []
        for img in selected_images:
            h, w = img.shape[:2]
            new_width = int(w * target_height / h)
            resized_images.append(cv2.resize(img, (new_width, target_height)))
        
        # Concatenate horizontally
        pano = np.hstack(resized_images)
        
        # Resize to target width if needed
        if self.processedImageWidth and pano.shape[1] != self.processedImageWidth:
            pano = cv2.resize(pano, (self.processedImageWidth, target_height))
        
        return pano

    def changeCylindrical(self, isCylindrical):
        self.active_stitcher.cylindricalWarp = isCylindrical
        self.active_stitcher.points_remap = None
        pass

    def changeCalculationsHyperpara(self, output):
        self.active_stitcher.active_matcher_type = output["matcherType"]
        self.active_stitcher.isRANSAC = output["isRANSAC"]
        self.active_stitcher.checks = output["checks"]
        self.active_stitcher.search_params = dict(checks=output["checks"])
        self.active_stitcher.ratio_thresh = output["ratio_thresh"]
        self.active_stitcher.score_threshold = output["score_threshold"]
        focal = output["focal"]
        self.active_stitcher.focal = focal
        self.active_stitcher.camera_matrix = np.array([[focal,0, 150], [0,focal, 150], [0,0, 1]])
        self.batchImageWidth, self.batchImageHeight = output["Sizes"][:2]
        self.active_stitcher.points_remap = None
        pass

def get_drone_order(drone_ids, headings):
    """
    Calculates the order of drones based on their heading angles.
    Arranges drones by heading in ascending order, handling wrap-around at ±180°.
    
    Parameters:
    - drone_ids: list of drone IDs
    - headings: list of heading angles (between -180 and 180)
    
    Returns:
    - list of indices representing the sorted order of drones by heading
    """
    if len(drone_ids) != len(headings):
        raise ValueError("drone_ids and headings must have the same length")
    
    # Normalize headings: convert negative angles to their positive equivalents
    normalized_headings = [h if h >= 0 else h + 360 for h in headings]
    
    # Create tuples of (index, normalized_heading) and sort
    indexed_headings = [(i, normalized_headings[i]) for i in range(len(headings))]
    sorted_indexed_headings = sorted(indexed_headings, key=lambda x: x[1])
    
    # Extract just the indices in sorted order
    sorted_indices = [idx for idx, _ in sorted_indexed_headings]
    return sorted_indices

def first_thread(manager: StitcherManager, debug=False, enable_debug_logging=False):
    """
    This method reads images from the block-based shared memory structure.
    Each block contains: flag (4 bytes), droneId (4 bytes), heading (4 bytes), image data
    """
    
    # Read metadata first to get image dimensions
    metadataMMF = mmap.mmap(-1, METADATA_SIZE, "MetadataSharedMemory")

    output = readMetadataMemory(metadataMMF)
    batchImageWidth, batchImageHeight, imageCount, manager.processedImageWidth, manager.processedImageHeight = output["Sizes"]

    # Resolution is driven entirely by Unity's metadata (blockImageWidth/Height +
    # panoramaImageWidth/Height in PyUniSharingFast's inspector); scale resolution
    # there. The neural-net warp runs at a fixed NET_W x NET_H regardless, so only
    # the render + memory-bridge costs grow with resolution.
    #
    # Wait until Unity has published real (non-zero) sizes before sizing the block
    # mapping — a pre-Start read yields zeros, which would make the mmap fail. The
    # block header size has to be published too, since it sets the block stride.
    # imageCount is waited on too: it is the block-map slot count, and a producer that
    # publishes 0 (a Unity build that only assigns it when it owns the section) otherwise
    # sails past here and dies silently inside _ensure_block_map's num_blocks <= 0 guard.
    waited = False
    while (batchImageWidth <= 0 or batchImageHeight <= 0
           or imageCount <= 0 or output["block_header_size"] <= 0):
        if not waited:
            # Unconditional, unlike the debug chatter below: without it a producer that
            # never publishes one of these fields looks like a silent hang at startup.
            waited = True
            print(f"[first_thread] Waiting for Unity metadata: "
                  f"size {batchImageWidth}x{batchImageHeight}, {imageCount} blocks, "
                  f"header {output['block_header_size']} B")
        time.sleep(0.1)
        output = readMetadataMemory(metadataMMF)
        batchImageWidth, batchImageHeight, imageCount, manager.processedImageWidth, manager.processedImageHeight = output["Sizes"]

    # The block map is sized from Unity's metadata rather than from a fixed count, so
    # PLANAR can publish more than three views. Unity keeps the section a fixed size
    # for a given camera count (spare slots are marked droneId = -1), so this normally
    # opens once; it re-opens only if the count, resolution or header version changes.
    block_map = None
    block_cache = {}

    first_loop = True
    last_debug_write = 0.0
    last_render_signal = 0.0
    # Panorama output mapping is opened once on the first write (below) and reused,
    # rather than being re-created every frame as it was previously.
    panoramaMMF = None

    while True:
        # Update metadata
        output = readMetadataMemory(metadataMMF)
        batchImageWidth, batchImageHeight, imageCount, manager.processedImageWidth, manager.processedImageHeight = output["Sizes"]
        # Live headset yaw drives which views are selected as centre/left/right.
        manager.headAngle = output["head_angle"]
        manager.print_rate = output["print_rate"]
        try:
            manager.checkHyperparaChanges(output)
        except NotImplementedError as e:
            print(f"[first_thread] fusion_mode error: {e}")

        # Planar geometry inputs: static config from metadata, live plane from the
        # seqlock block. A failed seqlock read leaves the previous plane in place.
        manager.update_planar_metadata(output)
        dynamic = read_dynamic_state(metadataMMF)
        if dynamic is not None:
            manager.scene_plane = dynamic

        block_map, changed = _ensure_block_map(
            block_map, imageCount, output["block_header_size"],
            batchImageWidth, batchImageHeight, enable_debug_logging)
        if changed:
            block_cache.clear()
        if block_map is None:
            time.sleep(0.05)
            continue

        # Read images from block-based memory, falling back to cached frames for busy blocks
        try:
            views = read_block_memory(
                block_map["mmf"],
                block_map["num_blocks"],
                block_map["block_size"],
                block_map["header_size"],
                block_map["image_size"],
                batchImageWidth,
                batchImageHeight,
                enable_debug_logging,
                cache=block_cache,
            )
        except Exception as e:
            if enable_debug_logging:
                print(f"[first_thread] Error reading block memory: {e}")
            time.sleep(0.05)
            continue

        # A planar mosaic tolerates a variable view count (drones die, the selection
        # shrinks); the left/centre/right stitchers need their exact triple, so they
        # keep the original all-or-nothing rule and hold the previous frame otherwise.
        if manager.active_stitcher_type == "PLANAR":
            have_enough = len(views) >= MIN_PLANAR_IMAGES
        else:
            have_enough = len(views) == block_map["num_blocks"]

        if have_enough:
            # Sort by drone ID to get a stable known order
            views = sorted(views, key=lambda v: v['drone_id'])
            sorted_images, sorted_drone_ids, sorted_headings = views_to_legacy(views)

            # DEBUG: dump frames read from BlockSharedMemory so the producer
            # format (size / BGR order / orientation) can be eyeballed. Gated behind
            # enable_debug_logging — writing a JPEG per drone per frame is a disk-I/O
            # stall that otherwise throttles this read/write loop.
            if enable_debug_logging:
                for di, img in zip(sorted_drone_ids, sorted_images):
                    cv2.imwrite(f"debug_input_drone_{di}.jpg", img)

            # Store the images and metadata. shared_views is published under the same
            # lock as the legacy lists so a consumer can never pair one frame's images
            # with another frame's poses.
            with manager.info_lock:
                manager.shared_views = views
                manager.shared_images = sorted_images
                manager.shared_drone_ids = sorted_drone_ids
                manager.shared_headings = sorted_headings
                if manager.active_stitcher_type == "PLANAR":
                    # Planar receives the selection from Unity rather than re-deriving
                    # it; the heading-sorted ring order is a left/centre/right concept.
                    manager.known_order = list(range(len(views)))
                else:
                    manager.known_order = get_drone_order(sorted_drone_ids, sorted_headings)
                # headAngle comes from the live headset yaw (set above from metadata),
                # not from the drone headings.

            # Wake the stitching thread — new images are available. Rate-limited
            # to RENDER_MIN_PERIOD: the shared memory is polled far faster than
            # Unity refills it, and re-rendering an unchanged frame just steals
            # GPU time from the warp thread.
            now = time.perf_counter()
            if now - last_render_signal >= RENDER_MIN_PERIOD:
                last_render_signal = now
                manager.new_images_event.set()

            if enable_debug_logging:
                print(f"[first_thread] Read {len(views)} views, sorted drone IDs: {sorted_drone_ids}, headings: {sorted_headings}")

        # Write panorama / quality flag if available
        if not manager.panoram_queue.empty():
            panorama, quality_ok, quality_reason = manager.panoram_queue.get()
            quality_int = 1 if quality_ok else 0
            image_size = manager.processedImageWidth * manager.processedImageHeight * 3

            if panoramaMMF is None:
                try:
                    panoramaMMF = mmap.mmap(-1, image_size + 4 + 4, "PanoramaSharedMemory")
                except Exception as e:
                    if enable_debug_logging:
                        print(f"[first_thread] Error opening panorama memory: {e}")
                    continue

            if panorama is None:
                # Fallback: panorama is bad — only update the quality flag so
                # Unity switches to the individual feeds. Leave image bytes stale.
                try:
                    write_panorama_memory(panoramaMMF, quality_int, quality_reason, image_size, None)
                except Exception as e:
                    if enable_debug_logging:
                        print(f"[first_thread] Error writing quality flag: {e}")
                    continue
            else:
                H, W, _ = panorama.shape
                if H != manager.processedImageHeight or W != manager.processedImageWidth:
                    try:
                        panorama = cv2.resize(panorama, (manager.processedImageWidth, manager.processedImageHeight))
                    except:
                        continue

                # Debug snapshot of the panorama. Throttled to ~1 Hz: a JPEG
                # encode + disk write of the full panorama every frame was a
                # ~10 ms stall inside this read/write loop, which caps the
                # end-to-end rate. Once a second is plenty to eyeball quality.
                now = time.perf_counter()
                if now - last_debug_write >= DEBUG_PANO_WRITE_PERIOD:
                    last_debug_write = now
                    cv2.imwrite("debug_panorama.jpg", panorama)

                try:
                    # Flip the panorama because unity texture starts bottom left,
                    # and convert cv2's BGR to RGB so Unity can upload the bytes
                    # straight into its RGB24 texture (LoadRawTextureData) with no
                    # per-pixel channel swap on the render thread.
                    panorama = cv2.cvtColor(cv2.flip(panorama, 0), cv2.COLOR_BGR2RGB)
                    write_panorama_memory(panoramaMMF, quality_int, quality_reason, image_size, panorama)
                    del panorama
                except Exception as e:
                    if enable_debug_logging:
                        print(f"[first_thread] Error writing panorama to memory: {e}")
                    continue
        
        # I/O poll period. End-to-end fps is the min of this, Unity's sendInterval /
        # readInterval, and the render throughput — lower all of them together to
        # raise fps. Unity publishes at 20 Hz (sendInterval = 0.05), so poll at a
        # few times that rate: with a 0.02 s sleep the loop's own ~0.01-0.03 s of
        # work pushed the period past 50 ms and capped the pipeline below 20 Hz.
        time.sleep(0.005)

        if first_loop:
            first_loop = False
            time.sleep(1.)
        
        if debug:
            break

def read_block_memory(processedMMF, num_blocks, blockSize, metadataSize, imageSize, imageWidth, imageHeight, enable_debug=False, cache=None):
    """
    Reads images (and, on wire v2, camera poses) from block-based shared memory.

    Block layout for each image:
        - int flag (4 bytes)
        - int droneId (4 bytes)        -- negative means the slot carries no view
        - float heading (4 bytes)
        - [v2 only] float camPos[3], camRot[4] (xyzw), captureTime, int poseStatus
        - image data (imageSize bytes)

    ``metadataSize`` is the block *header* size and selects between the two layouts;
    Unity advertises it in the metadata block.

    Parameters:
        - cache: optional dict {block_idx: view} used to substitute the previous frame
                 when a block is busy being written. Updated in-place. The pose travels
                 inside the view, so a re-served frame keeps *its* pose rather than
                 silently borrowing the current one.

    Returns:
        - views: list of dicts, one per populated slot:
              {'slot', 'drone_id', 'heading', 'image', 'pos', 'quat',
               'capture_time', 'pose_status', 'cached'}
          'pos'/'quat' are None on wire v1.
    """
    views = []
    has_pose = metadataSize >= BLOCK_HEADER_SIZE_V2

    if enable_debug:
        print(f"Reading {num_blocks} blocks from mmmf... blockSize={blockSize}, imageSize={imageSize}, imageWidth={imageWidth}, imageHeight={imageHeight}, header={metadataSize}")

    for block_idx in range(num_blocks):
        blockOffset = block_idx * blockSize

        # Read flag
        processedMMF.seek(blockOffset)
        flag_bytes = processedMMF.read(4)
        if len(flag_bytes) != 4:
            continue
        flag = struct.unpack('i', flag_bytes)[0]

        if enable_debug:
            print(f"[read_block_memory] Block {block_idx}: flag={flag}, offset={blockOffset}")

        if flag == 0:
            # Block is ready — set flag to 1 (busy reading)
            processedMMF.seek(blockOffset)
            processedMMF.write(struct.pack('i', 1))

            # Read droneId
            processedMMF.seek(blockOffset + 4)
            droneId = struct.unpack('i', processedMMF.read(4))[0]

            # Read heading
            processedMMF.seek(blockOffset + 8)
            heading = struct.unpack('f', processedMMF.read(4))[0]

            pos = quat = None
            capture_time = 0.0
            pose_status = 0
            if has_pose:
                processedMMF.seek(blockOffset + BLOCK_CAM_POS_OFFSET)
                pos = struct.unpack('<fff', processedMMF.read(12))
                quat = struct.unpack('<ffff', processedMMF.read(16))
                capture_time = struct.unpack('<f', processedMMF.read(4))[0]
                pose_status = struct.unpack('<i', processedMMF.read(4))[0]

            # Read image data
            processedMMF.seek(blockOffset + metadataSize)
            image_data = processedMMF.read(imageSize)

            # Reset flag to 0 (ready for next write)
            processedMMF.seek(blockOffset)
            processedMMF.write(struct.pack('i', 0))

            # A negative droneId marks a slot Unity deliberately left empty this frame
            # (the selection was shorter than the map). Drop it from the cache too, or
            # a drone that leaves the selection lingers in the mosaic forever.
            if droneId < 0:
                if cache is not None:
                    cache.pop(block_idx, None)
                continue

            if len(image_data) == imageSize:
                image = np.frombuffer(image_data, dtype=np.uint8).reshape((imageHeight, imageWidth, 3)).copy()
                view = {
                    'slot': block_idx,
                    'drone_id': droneId,
                    'heading': heading,
                    'image': image,
                    'pos': pos,
                    'quat': quat,
                    'capture_time': capture_time,
                    'pose_status': pose_status,
                    'cached': False,
                }
                views.append(view)

                if cache is not None:
                    cache[block_idx] = view

                if enable_debug:
                    print(f"[read_block_memory] Successfully read block {block_idx}: droneId={droneId}, heading={heading:.2f}")

        elif cache is not None and block_idx in cache:
            # Block is busy being written — reuse the previous frame for this drone,
            # pose included, since the two belong together.
            cached = dict(cache[block_idx])
            cached['cached'] = True
            views.append(cached)

            if enable_debug:
                print(f"[read_block_memory] Block {block_idx}: busy, using cached frame for droneId={cached['drone_id']}")

    return views


def _ensure_block_map(block_map, num_blocks, header_size, image_w, image_h, debug=False):
    """
    Open (or re-open) BlockSharedMemory whenever its geometry changes.

    Unity's CreateBlockMap destroys and recreates the named section when the block
    count changes, so a mapping held across that is stale -- it silently keeps reading
    a dead section. Unity avoids churning it by sizing from the camera count rather
    than the per-frame selection, but a stitcher-mode switch (3 <-> N views) or a
    resolution change still resizes it, and this is what notices.

    Returns ``(block_map, changed)``; ``block_map`` is None if the mapping failed.
    """
    image_size = image_w * image_h * 3
    block_size = header_size + image_size
    key = (num_blocks, header_size, image_size)

    if block_map is not None and block_map["key"] == key:
        return block_map, False

    if num_blocks <= 0 or header_size <= 0 or image_size <= 0:
        return None, block_map is not None

    try:
        mmf = mmap.mmap(-1, num_blocks * block_size, "BlockSharedMemory")
    except Exception as e:
        print(f"[first_thread] Could not map BlockSharedMemory "
              f"({num_blocks} x {block_size} B): {e}")
        return None, block_map is not None

    if block_map is not None:
        try:
            block_map["mmf"].close()
        except Exception:
            pass

    wire = "v2 (pose)" if header_size >= BLOCK_HEADER_SIZE_V2 else "v1 (no pose)"
    print(f"[first_thread] Block map: {num_blocks} blocks x {block_size} B "
          f"({image_w}x{image_h}, header {header_size} B, {wire})")

    return {
        "mmf": mmf,
        "key": key,
        "num_blocks": num_blocks,
        "block_size": block_size,
        "header_size": header_size,
        "image_size": image_size,
    }, True


def views_to_legacy(views):
    """
    Unpack view records into the three parallel lists the pre-v2 stitchers expect,
    so the STABSTITCH/CLASSIC paths are untouched by the N-view plumbing.
    """
    return ([v['image'] for v in views],
            [v['drone_id'] for v in views],
            [v['heading'] for v in views])

def write_memory(processedMMF, processedFlagPosition, processedDataPosition, processedImageSize, image_data):
    """
    Write an image to shared memory with Unity.

    Inputs:
        - processedMMF: mmap object for the shared memory.
        - processedFlagPosition: position of the flag in the memory
        - processedDataPosition: position to start writing the image data.
        - processedImageSize: expected size of the image data.
        - image_data: numpy array of the image to write.
    """
    while True:
        # Read the flag to check if Unity is ready for new data
        processedMMF.seek(processedFlagPosition)
        flag = struct.unpack('i', processedMMF.read(4))[0]

        if flag == 0:  # Unity isn't writing new images
            # Set flag to 1, indicating we're writing
            processedMMF.seek(processedFlagPosition)
            processedMMF.write(struct.pack('i', 1))

            # Convert image to byte array and check size
            image_bytes = image_data.tobytes()
            if len(image_bytes) != processedImageSize:
                raise ValueError(f"Image size mismatch: expected {processedImageSize}, got {len(image_bytes)}")

            # Write the image bytes to shared memory
            processedMMF.seek(processedDataPosition)
            processedMMF.write(image_bytes)

            # Reset flag to 0, indicating we've written the image
            processedMMF.seek(processedFlagPosition)
            processedMMF.write(struct.pack('i', 0))
            break

def write_panorama_memory(panoramaMMF, quality_int, quality_reason, image_size, image_data=None):
    """
    Write the panorama (and its quality word) to shared memory with Unity.

    Panorama shared-memory layout:
        [0:4]  write-flag (int)   handshake with Unity (0 = free, 1 = writing)
        [4:8]  quality word (int) packed:
                   bit 0 : panorama good (1) / bad (0) -> show panorama vs feeds
                   bit 1 : canvas gate failed
                   bit 2 : distortion gate failed
                   bit 3 : photometric (PSNR) gate failed
                   bit 4 : no overlap (camera yaw gap exceeds FOV)
                   bit 5 : too few feeds
               (bits 1+ = the failing-gate reason; only set when bit 0 == 0)
        [8:  ] RGB24 image data

    ``quality_reason`` is the failing-gate mask (canvas=1, distortion=2,
    photometric=4, no-overlap=8, too-few-images=16) and is shifted up one bit to
    sit above the good/bad flag.  Keep the mask below wide enough for every
    REASON_* constant: it previously truncated at 0xF, which silently discarded
    REASON_TOO_FEW_IMAGES (16) and made Unity log "unspecified" instead.  When
    ``image_data`` is None (quality fallback) only the quality word is updated;
    the stale image bytes are left in place because Unity ignores them while
    showing the feeds.
    """
    flag_position = 0
    quality_position = 4
    data_position = 8

    # Pack the good/bad flag (bit 0) with the failing-gate reason (bits 1+).
    quality_word = (quality_int & 1) | ((quality_reason & 0xFF) << 1)

    while True:
        # Read the flag to check if Unity is ready for new data
        panoramaMMF.seek(flag_position)
        flag = struct.unpack('i', panoramaMMF.read(4))[0]

        if flag == 0:  # Unity isn't reading
            # Set flag to 1, indicating we're writing
            panoramaMMF.seek(flag_position)
            panoramaMMF.write(struct.pack('i', 1))

            # Write the packed quality word (good/bad flag + failing-gate reason)
            panoramaMMF.seek(quality_position)
            panoramaMMF.write(struct.pack('i', quality_word))

            if image_data is not None:
                image_bytes = image_data.tobytes()
                if len(image_bytes) != image_size:
                    # Reset flag before raising so Unity isn't left blocked
                    panoramaMMF.seek(flag_position)
                    panoramaMMF.write(struct.pack('i', 0))
                    raise ValueError(f"Image size mismatch: expected {image_size}, got {len(image_bytes)}")

                panoramaMMF.seek(data_position)
                panoramaMMF.write(image_bytes)

            # Reset flag to 0, indicating we've finished writing
            panoramaMMF.seek(flag_position)
            panoramaMMF.write(struct.pack('i', 0))
            break

def stitching_thread(manager: StitcherManager, num_pano_img=3, verbose=False, debug=False):
    """
    Simplified stitching thread that uses known order from drone IDs.

    For STABSTITCH this is the fast render loop (~15 fps).  Instead of
    busy-looping with a sleep, it blocks on ``new_images_event`` until
    ``first_thread`` signals that fresh images have arrived.  This
    naturally matches the ~15 fps image arrival rate without wasting GPU
    cycles re-rendering the same frame.

    The ``timeout=0.1`` ensures non-STABSTITCH stitchers still loop even
    if the event is never explicitly signalled.
    """
    rate = RateMeter(window=5.0)
    while True:
        # Block until first_thread signals new images (or timeout)
        manager.new_images_event.wait(timeout=0.1)
        manager.new_images_event.clear()

        if manager.shared_images is None or manager.known_order is None:
            continue

        # Snapshot images and views together: they must come from the same frame, or a
        # planar solve would warp one frame's pixels with another frame's poses.
        with manager.info_lock:
            images = manager.shared_images
            views = manager.shared_views

        t = time.time()

        try:
            manager.process_stitching(images, num_pano_img=num_pano_img, views=views)
        except Exception:
            print("[stitching_thread] Error during stitching:")
            traceback.print_exc()

        rate.tick()

        if verbose and manager.print_rate:
            print(f"[stitching_thread] Loop time: {time.time()-t:.3f}s | {rate.hz:.1f} Hz (5s avg)")

        if debug:
            break

    print("[stitching_thread] Quitting stitching thread")

def warp_computation_thread(manager: StitcherManager, verbose=False, debug=False):
    """
    Dedicated thread for STABSTITCH neural-net warp computation (~3 Hz).

    Continuously snapshots the rolling frame buffer, runs
    SpatialNet + TemporalNet + SmoothNet, and updates the cached warp
    parameters that ``stab_pano`` uses for fast rendering.

    Acquires ``switching_lock2`` while computing to prevent stitcher
    switching from moving models off-GPU mid-computation.
    """
    rate = RateMeter(window=5.0)
    while True:
        if manager.shared_images is None or manager.known_order is None:
            time.sleep(0.4)
            continue

        # Capability check rather than a name check: any stitcher exposing compute_warps
        # gets the slow lane. PLANAR provides one as the slot a future refiner will fill
        # (its geometric solve is microseconds and runs inline, every frame, because the
        # poses change every frame -- only the *corrections* are slowly varying).
        if not hasattr(manager.active_stitcher, 'compute_warps'):
            time.sleep(0.1)
            continue

        t = time.perf_counter()
        try:
            with manager.switching_lock2:
                manager.active_stitcher.compute_warps()
        except Exception:
            print("[warp_thread] Error during warp computation:")
            traceback.print_exc()

        rate.tick()

        if verbose and manager.print_rate:
            elapsed = time.perf_counter() - t
            print(f"[warp_thread] Warp update: {elapsed:.3f}s | {rate.hz:.1f} Hz (5s avg)")

        if debug:
            break

    print("[warp_thread] Quitting warp computation thread")


def readMetadataMemory(metadataMMF :mmap )->dict:
    """
    Reads metadata from a memory-mapped file and returns it as a dictionary.
    """
    
    # Read the integers for image sizes
    metadataMMF.seek(0)
    int_values = struct.unpack('iiiii', metadataMMF.read(20))  # 5 integers

    # Read the string for Stitcher Type
    raw_string = metadataMMF.read(64)
    metadata_string = raw_string.decode('utf-8').rstrip('\x00')

    # Read the boolean isCylindrical
    raw_bool = metadataMMF.read(1)
    metadata_bool = bool(struct.unpack('B', raw_bool)[0])

    # Read the string for BF or FLANN
    raw_string = metadataMMF.read(64)
    matcherType = raw_string.decode('utf-8').rstrip('\x00')

    # Read the boolean for RANSAC or not
    raw_bool = metadataMMF.read(1)
    isRANSAC = bool(struct.unpack('B', raw_bool)[0])

    # Read the integer for check sizes
    checks = struct.unpack('i', metadataMMF.read(4))[0]

    # Read the float for ratio and score
    floats = struct.unpack('ff', metadataMMF.read(8))

    # Read the integer for focal
    focal = struct.unpack('i', metadataMMF.read(4))[0]

    # Read the boolean for onlyIHN
    raw_bool = metadataMMF.read(1)
    onlyIHN = bool(struct.unpack('B', raw_bool)[0])

    # Read the string for fusion mode
    raw_string = metadataMMF.read(64)
    fusion_mode = raw_string.decode('utf-8').rstrip('\x00') or "REFERENCE"

    # Read blur parameters for REFERENCE_BLEND
    blur_kernel_size = struct.unpack('i', metadataMMF.read(4))[0]
    blur_sigma = struct.unpack('f', metadataMMF.read(4))[0]
    border_size = struct.unpack('i', metadataMMF.read(4))[0]

    # Read StabStitch panorama-quality fallback parameters
    raw_bool = metadataMMF.read(1)
    quality_enabled = bool(struct.unpack('B', raw_bool)[0])
    quality_threshold = struct.unpack('f', metadataMMF.read(4))[0]

    # Read the live headset yaw (head look direction) used to pick stitched views
    head_angle = struct.unpack('f', metadataMMF.read(4))[0]

    # Console-verbosity toggle: when False, suppress the per-loop stitch/warp rate prints
    raw_bool = metadataMMF.read(1)
    print_rate = bool(struct.unpack('B', raw_bool)[0])

    # ---- Wire v2 static tail -------------------------------------------------------
    # Read by absolute offset rather than sequentially: the dynamic block that follows
    # is rewritten every frame by Unity's WriteDynamicState, so both sides address these
    # by constant. A v1 producer leaves this region zeroed, which reads as
    # wire_version 0 -- checked by require_planar_wire() before PLANAR is allowed.
    metadataMMF.seek(META_BLOCK_HEADER_SIZE_OFFSET)
    block_header_size, wire_version = struct.unpack('<ii', metadataMMF.read(8))
    fx, fy, cx, cy = struct.unpack('<ffff', metadataMMF.read(16))
    canvas_w, canvas_h = struct.unpack('<ii', metadataMMF.read(8))
    metres_per_pixel, max_range = struct.unpack('<ff', metadataMMF.read(8))
    feather_px = struct.unpack('<i', metadataMMF.read(4))[0]
    aniso_max, min_coverage = struct.unpack('<ff', metadataMMF.read(8))
    pose_source, psnr_gate = struct.unpack('<BB', metadataMMF.read(2))

    return {
        "Sizes": int_values,
        "typeOfStitcher": metadata_string,
        "isCylindrical": metadata_bool,
        "matcherType" : matcherType,
        "isRANSAC" : isRANSAC,
        "checks" : checks,
        "ratio_thresh" : floats[0],
        "score_threshold" : floats[1],
        "focal" : focal,
        "onlyIHN" : onlyIHN,
        "fusion_mode" : fusion_mode,
        "blur_kernel_size" : blur_kernel_size,
        "blur_sigma" : blur_sigma,
        "border_size" : border_size,
        "quality_enabled" : quality_enabled,
        "quality_threshold" : quality_threshold,
        "head_angle" : head_angle,
        "print_rate" : print_rate,
        # wire v2
        "block_header_size" : block_header_size,
        "wire_version" : wire_version,
        "intrinsics" : (fx, fy, cx, cy),
        "planar_canvas" : (canvas_w, canvas_h),
        "planar_metres_per_pixel" : metres_per_pixel,
        "planar_max_range" : max_range,
        "planar_feather_px" : feather_px,
        "planar_aniso_max" : aniso_max,
        "planar_min_coverage" : min_coverage,
        "planar_pose_source" : pose_source,
        "planar_psnr_gate" : bool(psnr_gate),
    }


def read_dynamic_state(metadataMMF):
    """
    Read the seqlock-protected dynamic block: the scene plane and gimbal pitch.

    Unity bumps the sequence counter to an odd value before writing the payload and to
    the next even value after, so an odd counter -- or a counter that changed across the
    read -- means the fields were in flux.  Retry a few times, then give up and let the
    caller keep its previous plane; a torn normal is not unit length and not
    perpendicular to anything, and would produce one frame of garbage geometry.

    Returns a dict, or None if no stable read was obtained.
    """
    for _ in range(4):
        metadataMMF.seek(META_DYN_SEQ_OFFSET)
        seq0 = struct.unpack('<i', metadataMMF.read(4))[0]
        if seq0 & 1:
            continue
        nx, ny, nz, d = struct.unpack('<ffff', metadataMMF.read(16))
        valid, mode = struct.unpack('<BB', metadataMMF.read(2))
        metadataMMF.seek(META_GIMBAL_PITCH_OFFSET)
        gimbal_pitch, centre_drone_id = struct.unpack('<fi', metadataMMF.read(8))

        metadataMMF.seek(META_DYN_SEQ_OFFSET)
        if struct.unpack('<i', metadataMMF.read(4))[0] == seq0:
            return {
                "plane_normal": (nx, ny, nz),
                "plane_d": d,
                "plane_valid": bool(valid),
                "plane_mode": mode,
                "gimbal_pitch": gimbal_pitch,
                "centre_drone_id": centre_drone_id,
            }
    return None

def main():
    """
    Activates the threads and initializes the StitcherManager.
    """

    manager = StitcherManager("cuda")
    verbose_stitching_thread = True
    debug = False
    enable_debug_logging = False  # Set to True for debugging

    # Print the keys of available stitchers
    print("Available stitchers:")
    for key in manager.stitchers.keys():
        print(f"- {key}")

    # The block count is read from Unity's metadata (blockImageCount), not fixed here:
    # PLANAR publishes as many views as the formation offers, the others publish 3.
    num_pano_img = 3  # Number of images in the panorama

    first_t = threading.Thread(target=first_thread, args=(manager, debug, enable_debug_logging))
    first_t.daemon = True
    first_t.start()

    stitch_t = threading.Thread(target=stitching_thread, args=(manager, num_pano_img, verbose_stitching_thread, debug))
    stitch_t.daemon = True
    stitch_t.start()

    # Warp computation thread: runs neural nets at ~3 Hz for STABSTITCH,
    # updating the cached warp params that stab_pano renders with at ~15 fps.
    warp_t = threading.Thread(target=warp_computation_thread, args=(manager, verbose_stitching_thread, debug))
    warp_t.daemon = True
    warp_t.start()

    while True:
        time.sleep(100)
    

if __name__ == '__main__':
    main()