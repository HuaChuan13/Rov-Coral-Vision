"""Fast display-only completion; metric measurements are never filled here."""
import cv2
import numpy as np

from guided_depth_gpu import complete_disparity_for_display


def complete_depth_for_display(estimated, disparity_small, valid_small, geometry, near, far):
    """Complete holes on the matching grid, then return a native-size metre map.

    ``disparity_small`` is in MATCHING-grid pixels, so ``geometry.fB`` and
    ``geometry.cx_delta`` apply directly without a native-resolution multiplier.
    Its nearest-support completion is only a display hypothesis. Known native
    ``estimated`` depths are copied back unchanged; neither input is modified.
    Callers must keep using their original strict measured array for XYZ.

    Return None when fewer than 32 finite in-range matching samples are available.
    """
    estimated = np.asarray(estimated)
    disparity_small = np.asarray(disparity_small)
    valid_small = np.asarray(valid_small, dtype=bool)
    if (estimated.ndim != 2 or disparity_small.ndim != 2 or
            valid_small.shape != disparity_small.shape):
        raise ValueError('Expected a 2D depth map and matching disparity/validity grids.')
    if disparity_small.shape != tuple(reversed(geometry.stereo_size)):
        raise ValueError('Disparity must use the calibrated matching grid and its pixel units.')
    height, width = estimated.shape
    small_height, small_width = disparity_small.shape
    if (not small_height or not small_width or small_height > height or small_width > width or
            height * small_width != width * small_height):
        raise ValueError('Matching and display grids must have the same aspect ratio.')
    if not np.isfinite((near, far, geometry.fB, geometry.cx_delta)).all() or not 0 < near < far or geometry.fB <= 0:
        raise ValueError('Expected finite metre bounds and positive focal-length times baseline.')

    denominator = disparity_small - geometry.cx_delta
    anchors = (valid_small & np.isfinite(disparity_small) & (denominator > .1) &
               (denominator >= geometry.fB / far) & (denominator <= geometry.fB / near))
    completed = complete_disparity_for_display(disparity_small, anchors)
    if completed is None:
        return None
    # Convert exactly once on the small grid. Upsampling metre values does not
    # require rescaling disparity or focal length. Nearest interpolation avoids
    # manufacturing intermediate surfaces across the completed support boundary.
    small_depth = np.asarray(geometry.fB / (completed - geometry.cx_delta), dtype=np.float32)
    result = cv2.resize(small_depth, (width, height), interpolation=cv2.INTER_NEAREST)
    np.copyto(result, estimated, where=np.isfinite(estimated))
    return result
