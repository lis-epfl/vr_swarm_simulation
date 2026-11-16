"""
The panorama's layout on the wire, shared by every stitcher that renders straight to it.

``PanoramaSharedMemory`` carries RGB24, bottom-up rows, at the size Unity publishes in the
metadata block.  A stitcher that hands ``first_thread`` a plain BGR canvas pays for a
``cv2.resize`` + flip + BGR->RGB pass on the host for every panorama; one that renders
straight into the wire layout returns a :class:`WireReadyPanorama` instead and that pass
is skipped.

Kept in its own module so the planar stitcher can return one without importing
``StabStitcher`` -- which pulls in the vendored StabStitch++ networks and is allowed to
fail to import.
"""

import numpy as np

# Bits 9-15 of the panorama quality word: a tag of the panorama size the image bytes were
# written at.  Mirrors panoramaSizeTagShift / panoramaSizeTagMask in PyUniSharingFast.cs
# (tools/check_wire_layout.py asserts the pair).
#
# The section has no size field -- the live panorama is a prefix whose length both sides
# read out of the metadata block -- and that was enough while the size only changed from
# the inspector.  It now changes with the stitcher: PLANAR publishes its own canvas as the
# panorama, so every STABSTITCH <-> PLANAR switch (the V key, mid-flight) moves it.  A
# panorama rendered for the old size and written after Unity adopted the new one would be
# uploaded with the wrong stride, i.e. a frame of sheared noise.  Unity uploads only a
# panorama whose tag matches the size it is reading at; 0 is a producer without the tag
# and is uploaded as before, so either side can be updated first.
PANORAMA_SIZE_TAG_SHIFT = 9
PANORAMA_SIZE_TAG_MASK = 0x7F


def panorama_size_tag(width, height):
    """
    7-bit tag of a panorama size, never 0.

    Not a checksum of the pixels, only of the two numbers, and with 127 values two
    unrelated sizes share a tag one time in 127.  That is acceptable because a session
    switches between a handful of sizes and ``tools/check_wire_layout.py`` asserts the
    ones in use are told apart -- which is not a formality: the first pair of multipliers
    tried here (31 and 17) sent *every* 3:2 size to the same tag, since
    ``31 * 3 + 17 * 2 == 127``.  Mirrors ``PanoramaSizeTag`` in PyUniSharingFast.cs.
    """
    return (int(width) * 131 + int(height) * 71) % PANORAMA_SIZE_TAG_MASK + 1


class WireReadyPanorama(np.ndarray):
    """
    A panorama already in the PanoramaSharedMemory layout: RGB, bottom-up rows, at the
    panorama size Unity asked for.  ``stab_pano`` and ``planar_pano`` return one of these
    when they rendered straight to the wire; ``first_thread`` then writes it without the
    resize / flip / BGR->RGB pass it applies to a plain BGR canvas.

    An ndarray subclass rather than a wrapper so every existing ``.shape`` / ``imwrite``
    use keeps working; the backing memory is a pinned buffer owned by the stitcher and is
    reused a few renders later, so consumers must not hold on to it.
    """
