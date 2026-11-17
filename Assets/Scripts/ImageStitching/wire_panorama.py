"""
The panorama's layout on the wire, shared by every stitcher that renders straight to it.

``PanoramaSharedMemory`` carries a 16-byte header -- write flag, quality word, width,
height -- and then RGB24, bottom-up rows.  A stitcher that hands ``first_thread`` a plain
BGR canvas pays for a ``cv2.resize`` + flip + BGR->RGB pass on the host for every
panorama; one that renders straight into the wire layout returns a
:class:`WireReadyPanorama` instead and that pass is skipped.

The width and height travel with every panorama because its size is no longer one number
per session.  ``PLANAR`` draws its canvas with as many pixels as the feeds in it need, up
to the size Unity publishes as a ceiling, so the size moves with the formation, the zoom
and the stitcher.  The section itself never moves -- it is the constant maximum -- and
Unity reads each panorama at the size written beside it, which is also what keeps a
panorama rendered for one size from being uploaded at another's stride.

Kept in its own module so the planar stitcher can return one without importing
``StabStitcher`` -- which pulls in the vendored StabStitch++ networks and is allowed to
fail to import.
"""

import numpy as np


class WireReadyPanorama(np.ndarray):
    """
    A panorama already in the PanoramaSharedMemory layout: RGB, bottom-up rows.
    ``stab_pano`` and ``planar_pano`` return one of these when they rendered straight to
    the wire; ``first_thread`` then writes it, at its own size, without the resize / flip
    / BGR->RGB pass it applies to a plain BGR canvas.

    An ndarray subclass rather than a wrapper so every existing ``.shape`` / ``imwrite``
    use keeps working; the backing memory is a pinned buffer owned by the stitcher and is
    reused a few renders later, so consumers must not hold on to it.
    """
