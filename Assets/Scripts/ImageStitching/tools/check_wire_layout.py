"""
Cross-check the shared-memory layout constants in PyUniSharingFast.cs against the ones
in StitcherThreading.py, by parsing both files.

The two processes address the same bytes from independent constant tables, so a
mismatch is silent: it does not fail to compile, it just reads a float out of the
middle of another field. Run this after touching either layout::

    cd Assets/Scripts/ImageStitching && python tools/check_wire_layout.py
"""

import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CS = os.path.join(ROOT, "PyUniSharingFast.cs")
PY = os.path.join(ROOT, "StitcherThreading.py")
# The real-drone producer, in the DJI scene. Sits outside ImageStitching/.
SHARING_CS = os.path.join(os.path.dirname(ROOT), "dji", "ImageSharing.cs")
# The other end of the feed map, in a sibling repo. Checked when present, skipped when not
# -- this file has to keep working for anyone who only has the sim checked out.
DJI_SHARING_PY = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(ROOT)))),
    "DJI_Swarm", "AOS server", "utils", "imageSharingUtil.py")


# C# constant name -> Python constant name
PAIRS = [
    ("metadataSize",                  "METADATA_SIZE"),
    ("metaWireVersion",               "PLANAR_WIRE_VERSION"),
    ("metaBlockHeaderSizeOffset",     "META_BLOCK_HEADER_SIZE_OFFSET"),
    ("metaWireVersionOffset",         "META_WIRE_VERSION_OFFSET"),
    ("metaFxOffset",                  "META_FX_OFFSET"),
    ("metaPlanarCanvasWidthOffset",   "META_PLANAR_CANVAS_OFFSET"),
    ("metaPlanarMetresPerPixelOffset", "META_PLANAR_MPP_OFFSET"),
    ("metaPlanarFeatherPxOffset",     "META_PLANAR_FEATHER_OFFSET"),
    ("metaPlanarAnisoMaxOffset",      "META_PLANAR_ANISO_OFFSET"),
    ("metaPlanarPoseSourceOffset",    "META_PLANAR_POSE_SOURCE_OFFSET"),
    ("metaPlanarBlendModeOffset",     "META_PLANAR_BLEND_MODE_OFFSET"),
    ("metaPlanarDebugViewOffset",     "META_PLANAR_DEBUG_VIEW_OFFSET"),
    ("metaDynSeqOffset",              "META_DYN_SEQ_OFFSET"),
    ("metaPlaneNxOffset",             "META_PLANE_N_OFFSET"),
    ("metaPlaneValidOffset",          "META_PLANE_VALID_OFFSET"),
    ("metaGimbalPitchOffset",         "META_GIMBAL_PITCH_OFFSET"),
    ("metaCentreDroneIdOffset",       "META_CENTRE_DRONE_OFFSET"),
    ("metaPlanarPlaneSweepOffset",    "META_PLANAR_SWEEP_ENABLED_OFFSET"),
    ("metaPlanarSweepRangeOffset",    "META_PLANAR_SWEEP_RANGE_OFFSET"),
    ("metaPlanarSweepStepsOffset",    "META_PLANAR_SWEEP_STEPS_OFFSET"),
    ("metaPlanarRefineRateOffset",    "META_PLANAR_REFINE_RATE_OFFSET"),
    ("metaPlanarRefineMaxShiftOffset", "META_PLANAR_REFINE_MAX_SHIFT_OFFSET"),
    ("metaPlanarStandoffOffset",       "META_PLANAR_STANDOFF_OFFSET"),
    ("metaPlanarZoomOffset",           "META_PLANAR_ZOOM_OFFSET"),
    ("metaPlanarCanvasModeOffset",     "META_PLANAR_CANVAS_MODE_OFFSET"),
    ("metaHeartbeatOffset",            "META_HEARTBEAT_OFFSET"),
    ("metaBlockSlotCapacityOffset",    "META_BLOCK_SLOT_CAPACITY_OFFSET"),
    ("metaBlockSlotStrideOffset",      "META_BLOCK_SLOT_STRIDE_OFFSET"),
    ("metaBlockSectionBytesOffset",    "META_BLOCK_SECTION_BYTES_OFFSET"),
    ("metaPanoramaSectionBytesOffset", "META_PANORAMA_SECTION_BYTES_OFFSET"),
    # Section geometry. Not offsets: these are the physical shape of BlockSharedMemory and
    # PanoramaSharedMemory, held as constants on both sides and never negotiated at
    # runtime. A named Windows section cannot be resized, so a disagreement here is not a
    # misread field, it is one process being denied the mapping outright.
    ("blockSlotCapacity",              "BLOCK_SLOT_CAPACITY"),
    ("maxBlockWidth",                  "BLOCK_MAX_WIDTH"),
    ("maxBlockHeight",                 "BLOCK_MAX_HEIGHT"),
    ("blockSlotStride",                "BLOCK_SLOT_STRIDE"),
    ("blockSectionBytes",              "BLOCK_SECTION_BYTES"),
    ("maxPanoramaWidth",               "PANORAMA_MAX_WIDTH"),
    ("maxPanoramaHeight",              "PANORAMA_MAX_HEIGHT"),
    ("panoramaDataPosition",           "PANORAMA_HEADER_BYTES"),
    ("panoramaSectionBytes",           "PANORAMA_SECTION_BYTES"),
    # Not an offset: Python switches on the plane-mode byte by number, so the enum value
    # itself is part of the contract. C# mirrors it as a const because this parser reads
    # `const int` and cannot evaluate an enum member.
    ("planeModeFormationRelative",     "PLANE_MODE_FORMATION_RELATIVE"),
    ("blockLegacyHeaderSize",         "BLOCK_HEADER_SIZE_V1"),
    ("blockPoseHeaderSize",           "BLOCK_HEADER_SIZE_V2"),
    ("blockCamPosOffset",             "BLOCK_CAM_POS_OFFSET"),
    ("blockCamRotOffset",             "BLOCK_CAM_ROT_OFFSET"),
    ("blockCaptureTimeOffset",        "BLOCK_CAPTURE_TIME_OFFSET"),
    ("blockPoseStatusOffset",         "BLOCK_POSE_STATUS_OFFSET"),
    ("POSE_VALID",                    "POSE_VALID"),
    ("POSE_GROUND_TRUTH",             "POSE_GROUND_TRUTH"),
    ("POSE_NOISE_INJECTED",           "POSE_NOISE_INJECTED"),
]

# Quality-reason bits: Python holds the pre-shift value, C# the post-shift bit.
REASON_PAIRS = [
    ("REASON_NO_OVERLAP",     "REASON_NO_OVERLAP"),
    ("REASON_TOO_FEW_IMAGES", "REASON_TOO_FEW_IMAGES"),
    ("REASON_PLANE_INVALID",  "REASON_PLANE_INVALID"),
]


def parse_cs(path):
    src = open(path, encoding="utf-8-sig").read()
    out = {}
    for m in re.finditer(
            r"const\s+int\s+(\w+)\s*=\s*([^;]+);", src):
        name, expr = m.group(1), m.group(2).strip()
        expr = re.sub(r"//.*", "", expr).strip()
        try:
            out[name] = int(eval(expr, {"__builtins__": {}}, dict(out)))
        except Exception:
            pass
    return out


def parse_py(path):
    src = open(path, encoding="utf-8").read()
    out = {}
    for m in re.finditer(r"^([A-Z][A-Z0-9_]*)\s*=\s*([^\n#]+)", src, re.M):
        name, expr = m.group(1), m.group(2).strip()
        try:
            out[name] = int(eval(expr, {"__builtins__": {}}, dict(out)))
        except Exception:
            pass
    return out


def main():
    cs, py = parse_cs(CS), parse_py(PY)
    failures = []

    # The section-size constants run to eight digits, so the value columns are wide
    # enough for them to stay separated rather than running together.
    print(f"{'C# constant':<34}{'Python constant':<36}{'C#':>10}{'Py':>10}  ok")
    print("-" * 100)
    for cs_name, py_name in PAIRS:
        a, b = cs.get(cs_name), py.get(py_name)
        ok = a is not None and a == b
        print(f"{cs_name:<34}{py_name:<36}{str(a):>10}{str(b):>10}  {'OK' if ok else 'MISMATCH'}")
        if not ok:
            failures.append(f"{cs_name} ({a}) != {py_name} ({b})")

    print()
    print("Quality reasons (Python value is pre-shift; C# is the bit after <<1)")
    print("-" * 100)
    for cs_name, py_name in REASON_PAIRS:
        a, b = cs.get(cs_name), py.get(py_name)
        ok = a is not None and b is not None and a == (b << 1)
        print(f"{cs_name:<34}{py_name:<36}{str(a):>10}{str(b):>10}  {'OK' if ok else 'MISMATCH'}")
        if not ok:
            failures.append(f"{cs_name} ({a}) != {py_name}<<1 ({b if b is None else b << 1})")

    # Layout arithmetic: the tail must be contiguous, 4-byte aligned and end where
    # metadataSize says it does.
    print()
    print("Layout arithmetic")
    print("-" * 90)
    fields = [
        ("blockHeaderSize", "metaBlockHeaderSizeOffset", 4),
        ("wireVersion", "metaWireVersionOffset", 4),
        ("fx", "metaFxOffset", 4), ("fy", "metaFyOffset", 4),
        ("cx", "metaCxOffset", 4), ("cy", "metaCyOffset", 4),
        ("canvasW", "metaPlanarCanvasWidthOffset", 4),
        ("canvasH", "metaPlanarCanvasHeightOffset", 4),
        ("metresPerPixel", "metaPlanarMetresPerPixelOffset", 4),
        ("maxRange", "metaPlanarMaxRangeOffset", 4),
        ("featherPx", "metaPlanarFeatherPxOffset", 4),
        ("anisoMax", "metaPlanarAnisoMaxOffset", 4),
        ("minCoverage", "metaPlanarMinCoverageOffset", 4),
        ("poseSource", "metaPlanarPoseSourceOffset", 1),
        ("psnrGate", "metaPlanarPsnrGateOffset", 1),
        ("blendMode", "metaPlanarBlendModeOffset", 1),
        ("debugView", "metaPlanarDebugViewOffset", 1),
        ("dynSeq", "metaDynSeqOffset", 4),
        ("planeNx", "metaPlaneNxOffset", 4), ("planeNy", "metaPlaneNyOffset", 4),
        ("planeNz", "metaPlaneNzOffset", 4), ("planeD", "metaPlaneDOffset", 4),
        ("planeValid", "metaPlaneValidOffset", 1),
        ("planeMode", "metaPlaneModeOffset", 1),
        ("gimbalPitch", "metaGimbalPitchOffset", 4),
        ("centreDroneId", "metaCentreDroneIdOffset", 4),
        # Estimator settings: static, but after the dynamic block (the static tail's
        # padding slot was full), so they are checked as a continuation of the cursor.
        ("planeSweep", "metaPlanarPlaneSweepOffset", 1),
        ("poseRefine", "metaPlanarPoseRefineOffset", 1),
        ("sweepRange", "metaPlanarSweepRangeOffset", 4),
        ("sweepSteps", "metaPlanarSweepStepsOffset", 4),
        ("refineRate", "metaPlanarRefineRateOffset", 4),
        ("refineMaxShift", "metaPlanarRefineMaxShiftOffset", 4),
        ("standoff", "metaPlanarStandoffOffset", 4),
        # Operator viewing transform. Dynamic (written under the seqlock above) but
        # physically out here, because the dynamic block's padding was already spent --
        # so like the estimator settings it continues the same cursor walk.
        ("zoom", "metaPlanarZoomOffset", 4),
        ("panA", "metaPlanarPanAOffset", 4),
        ("panB", "metaPlanarPanBOffset", 4),
        ("canvasMode", "metaPlanarCanvasModeOffset", 1),
        # Producer heartbeat and the published section geometry, taken out of what used to
        # be metadataReservedGap. metadataSize is unchanged, which is the point: an
        # already-running Python must not be stranded by a different section size.
        ("heartbeat", "metaHeartbeatOffset", 4),
        ("blockSlotCapacity", "metaBlockSlotCapacityOffset", 4),
        ("blockSlotStride", "metaBlockSlotStrideOffset", 4),
    ]
    cursor = 253
    for label, const, size in fields:
        off = cs.get(const)
        if off is None:
            failures.append(f"missing C# constant {const}")
            continue
        if off < cursor:
            failures.append(f"{label} at {off} overlaps previous field (ends {cursor})")
        if size == 4 and off % 4 != 0:
            failures.append(f"{label} at {off} is not 4-byte aligned")
        cursor = off + size
    print(f"  v1 prefix ends at 253, tail ends at {cursor} "
          f"(C# metadataTailEnd = {cs.get('metadataTailEnd')})")
    if cs.get("metadataTailEnd") != cursor:
        # Padding after the last field is fine, but it must not be negative.
        if cs.get("metadataTailEnd", 0) < cursor:
            failures.append(f"metadataTailEnd {cs.get('metadataTailEnd')} < actual end {cursor}")

    # The reserved gap shrinks as the tail grows so metadataSize stays fixed; read it from
    # C# rather than hardcoding, or this check silently drifts the next time a field lands.
    gap = cs.get("metadataReservedGap")
    if gap is None:
        failures.append("missing C# constant metadataReservedGap")
        gap = 0
    expected_size = cs.get("metadataTailEnd", 0) + gap + 8
    print(f"  metadataTailEnd + {gap} reserved + 8 = {expected_size} "
          f"(C# metadataSize = {cs.get('metadataSize')})")
    if expected_size != cs.get("metadataSize"):
        failures.append(f"metadataSize {cs.get('metadataSize')} != {expected_size}")

    # The two trailing size fields are what that "+ 8" is. Pin them to the end of the gap,
    # or a future field taken out of the gap can silently land on top of them.
    sizes_at = cs.get("metadataTailEnd", 0) + gap
    print(f"  trailing size fields at {cs.get('metaBlockSectionBytesOffset')}, "
          f"{cs.get('metaPanoramaSectionBytesOffset')} (expected {sizes_at}, {sizes_at + 4})")
    if cs.get("metaBlockSectionBytesOffset") != sizes_at:
        failures.append(f"metaBlockSectionBytesOffset {cs.get('metaBlockSectionBytesOffset')} "
                        f"!= metadataTailEnd + reserved gap ({sizes_at})")
    if cs.get("metaPanoramaSectionBytesOffset") != sizes_at + 4:
        failures.append(f"metaPanoramaSectionBytesOffset "
                        f"{cs.get('metaPanoramaSectionBytesOffset')} != {sizes_at + 4}")
    if gap < 0:
        failures.append(f"metadataReservedGap is negative ({gap}): the tail has outgrown "
                        "metadataSize, which must be raised on both sides together")

    dyn_end = cs.get("metaCentreDroneIdOffset", 0) + 4
    print(f"  dynamic block spans {cs.get('metaDynSeqOffset')}..{dyn_end}")

    # Real-drone path: ImageSharing.cs CREATES BlockSharedMemory in the DJI scene, but
    # PyUniSharingFast DESCRIBES it in metadata (Python sizes its mapping from that). The
    # two files never reference each other, so a mismatch is silent on both sides.
    print()
    print("Real-drone path (ImageSharing.cs creates BlockSharedMemory, "
          "PyUniSharingFast describes it)")
    print("-" * 90)
    if not os.path.exists(SHARING_CS):
        print(f"  ImageSharing.cs not found at {SHARING_CS}; skipped")
    else:
        sh = parse_cs(SHARING_CS)
        # Both producers write the pose-carrying v2 header now, so ImageSharing's block
        # header must equal blockPoseHeaderSize, not the legacy size.
        #
        # The section geometry below matters twice over. Once because Python addresses
        # slots by these numbers while only PyUniSharingFast publishes them; and once
        # because BOTH components request the same named section, and Windows opens an
        # existing section rather than resizing it — so unequal sizes mean whichever
        # component starts second is denied outright. There is no runtime negotiation left
        # that could paper over a mismatch, which is why every one of these is asserted
        # rather than reported.
        for label, sh_name, cs_name in [
                ("block header", "MetadataSize", "blockPoseHeaderSize"),
                ("lrc views", "STITCH_COUNT_LRC", "STITCH_COUNT_LRC"),
                ("slot capacity", "StitchSlotCapacity", "blockSlotCapacity"),
                ("slot stride", "StitchSlotStride", "blockSlotStride"),
                ("section bytes", "StitchSectionBytes", "blockSectionBytes"),
                ("envelope width", "StitchMaxImageWidth", "maxBlockWidth"),
                ("envelope height", "StitchMaxImageHeight", "maxBlockHeight"),
        ]:
            a, b = sh.get(sh_name), cs.get(cs_name)
            ok = a is not None and a == b
            print(f"  {label:<15} ImageSharing.{sh_name} = {a}, "
                  f"PyUniSharingFast.{cs_name} = {b}  {'OK' if ok else 'MISMATCH'}")
            if not ok:
                failures.append(f"ImageSharing.{sh_name} ({a}) != {cs_name} ({b})")

        # The other repo's copies. Reported when DJI_Swarm is not checked out beside this
        # one, asserted when it is: nothing else guards these, and a mismatch is silent on
        # both sides -- it reads image bytes as a header.
        if not os.path.exists(DJI_SHARING_PY):
            print(f"  DJI_Swarm not found at {DJI_SHARING_PY}; cross-repo checks skipped")
        else:
            dji = parse_py(DJI_SHARING_PY)
            # The feed header, which is what that repo writes into DroneFeedSharedMemory.
            # Its own section geometry is independent of the stitcher's and is not checked
            # here -- MAX_DRONES vs MaxFeedBlocks is reported below instead.
            #
            # The STITCH_* trio is a different matter. In the normal architecture nothing
            # in DJI_Swarm creates BlockSharedMemory at all; but its two legacy debug
            # tools (image_stream.py, image_replay.py) bypass Unity and write it directly,
            # and a named Windows section cannot be resized -- so if their geometry drifts
            # from the sim's, whichever process starts first either denies the other its
            # mapping or silently hands it a partial view.
            for label, dji_name, cs_name in [
                    ("feed header", "BLOCK_HEADER_V2_BYTES", "blockPoseHeaderSize"),
                    ("stitch capacity", "STITCH_SLOT_CAPACITY", "blockSlotCapacity"),
                    ("stitch stride", "STITCH_SLOT_STRIDE", "blockSlotStride"),
                    ("stitch section", "STITCH_SECTION_BYTES", "blockSectionBytes"),
                    ("stitch env w", "STITCH_MAX_IMAGE_WIDTH", "maxBlockWidth"),
                    ("stitch env h", "STITCH_MAX_IMAGE_HEIGHT", "maxBlockHeight"),
            ]:
                a, b = dji.get(dji_name), cs.get(cs_name)
                ok = a is not None and a == b
                print(f"  {label:<15} imageSharingUtil.{dji_name} = {a}, "
                      f"PyUniSharingFast.{cs_name} = {b}  {'OK' if ok else 'MISMATCH'}")
                if not ok:
                    failures.append(f"imageSharingUtil.{dji_name} ({a}) != {cs_name} ({b})")

        # image_stream_feed.py writes this map in the DJI_Swarm repo; its MAX_DRONES must
        # equal MaxFeedBlocks. Reported rather than asserted -- that repo is not here.
        print(f"  feed capacity  ImageSharing.MaxFeedBlocks = {sh.get('MaxFeedBlocks')} "
              f"(must equal MAX_DRONES in DJI_Swarm/AOS server/image_stream_feed.py)")
        print(f"  feed image     {sh.get('ImageWidth')}x{sh.get('ImageHeight')} "
              f"(must equal that file's width/height)")

    print()
    if failures:
        print(f"FAILED ({len(failures)}):")
        for f in failures:
            print("  -", f)
        return 1
    print("Wire layout is consistent across C# and Python.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
