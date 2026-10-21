"""Unicode-safe image I/O and conservative chessboard enhancement."""
import json
from pathlib import Path

import cv2
import numpy as np


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    tmp.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def save_image(path, image):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(path.suffix or '.png', image)
    if not ok:
        raise RuntimeError('Image encoding failed: ' + str(path))
    encoded.tofile(str(path))


def read_image(path):
    image = cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError('Cannot read ' + str(path))
    return image


def gray(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image


def enhance(image, clip=2.0, sharpen=0.25):
    """Only photometric operations; no resize, warp, generative restoration or fill."""
    if image.ndim == 3:
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        luminance = lab[:, :, 0]
    else:
        luminance = image
    denoised = cv2.bilateralFilter(luminance, 5, 15, 15)
    adjusted = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8)).apply(denoised)
    if sharpen:
        adjusted = cv2.addWeighted(adjusted, 1 + sharpen,
                                  cv2.GaussianBlur(adjusted, (0, 0), .8), -sharpen, 0)
    if image.ndim == 3:
        lab[:, :, 0] = adjusted
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return adjusted


def board_points(cols=11, rows=8, square_m=.025):
    points = np.zeros((cols * rows, 3), np.float32)
    points[:, :2] = np.mgrid[:cols, :rows].T.reshape(-1, 2) * square_m
    return points


def normalize_order(points):
    # Modest tilt, keep the board upright. A 180-degree ordering ambiguity is
    # resolved consistently in the two images; no >80 degree roll during capture.
    points = points.astype(np.float32).reshape(-1, 1, 2)
    delta = points[-1, 0] - points[0, 0]
    if delta[1] < -1 or (abs(delta[1]) <= 1 and delta[0] < 0):
        points = points[::-1].copy()
    return points


def detect(image, pattern=(11, 8)):
    original = gray(image)
    flags = cv2.CALIB_CB_NORMALIZE_IMAGE
    used = 'raw'
    ok, points = cv2.findChessboardCornersSB(original, pattern, flags=flags)
    if not ok:
        ok, points = cv2.findChessboardCornersSB(gray(enhance(image, sharpen=0)), pattern, flags=flags)
        used = 'enhanced'
    if not ok:
        return None, 'not_found'
    points = normalize_order(points)
    initial = points.copy()
    refined = cv2.cornerSubPix(original, points.copy(), (3, 3), (-1, -1),
                              (cv2.TERM_CRITERIA_COUNT + cv2.TERM_CRITERIA_EPS, 40, .005))
    # Do not let local refinement jump to a different corner on a distorted grid.
    movement = np.linalg.norm(refined - initial, axis=2).ravel()
    points[movement <= 1.5] = refined[movement <= 1.5]
    return points, used


def grid_roughness(points, pattern):
    """High-frequency row/column error after smooth cubic curvature removal."""
    cols, rows = pattern
    grid = np.asarray(points, float).reshape(rows, cols, 2)
    residuals = []
    for line in list(grid) + list(grid.transpose(1, 0, 2)):
        centered = line-line.mean(axis=0)
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        axes = centered @ vt.T
        x = axes[:, 0]
        x = (x-x.mean()) / max(np.ptp(x), 1e-9)
        poly = np.polyfit(x, axes[:, 1], min(3, len(x)-2))
        residuals.extend(axes[:, 1]-np.polyval(poly, x))
    return float(np.sqrt(np.mean(np.square(residuals))))


def inspect(image, pattern=(11, 8), min_sharpness=18., margin=4.,
            max_edge_width=3., max_grid_roughness=.15):
    points, method = detect(image, pattern)
    result = {'ok': False, 'reason': 'show_all_88_inner_corners', 'method': method}
    if points is None:
        result['reason'] = 'show_all_%d_inner_corners' % (pattern[0] * pattern[1])
        return result
    result['corners'] = points
    g = gray(image)
    height, width = g.shape
    xy = points[:, 0]
    cols, rows = pattern
    grid = xy.reshape(rows, cols, 2)
    dx = np.linalg.norm(np.diff(grid, axis=1), axis=2)
    dy = np.linalg.norm(np.diff(grid, axis=0), axis=2)
    if min(np.percentile(dx, 10), np.percentile(dy, 10)) < 6:
        result['reason'] = 'board_too_small_move_closer'
        return result
    # A homography is ONLY a conservative boundary estimate. Its residual is not
    # a rejection gate: refraction/lens distortion can legitimately bend the grid.
    H, _ = cv2.findHomography(board_points(cols, rows, 1)[:, :2], xy, 0)
    if H is None:
        result['reason'] = 'boundary_not_estimated'
        return result
    perimeter = []
    for gx in np.linspace(-1, cols, cols * 3):
        perimeter.extend([[gx, -1], [gx, rows]])
    for gy in np.linspace(-1, rows, rows * 3):
        perimeter.extend([[-1, gy], [cols, gy]])
    outer = cv2.perspectiveTransform(np.float32([perimeter]), H)[0]
    result['outer'] = outer
    if (not np.isfinite(outer).all() or (outer[:, 0] < margin).any() or
            (outer[:, 0] > width - 1 - margin).any() or (outer[:, 1] < margin).any() or
            (outer[:, 1] > height - 1 - margin).any()):
        result['reason'] = 'outer_squares_near_edge_move_board_inward'
        return result
    hull = cv2.convexHull(np.rint(xy).astype(np.int32))
    mask = np.zeros_like(g)
    cv2.fillConvexPoly(mask, hull, 255)
    roi = g[mask > 0]
    sharpness = float(cv2.Laplacian(g, cv2.CV_32F)[mask > 0].var())
    contrast = float(np.percentile(roi, 90) - np.percentile(roi, 10))
    result.update(sharpness=sharpness, contrast=contrast)
    if sharpness < min_sharpness or contrast < 25:
        result['reason'] = 'blur_or_low_contrast_hold_still_adjust_light'
        return result
    # Measure edge spread on RAW intensities in both directions; enhancement
    # must not make a motion-blurred frame appear qualified for calibration.
    try:
        widths = [float(cv2.estimateChessboardSharpness(g, pattern, points, .8, vertical)[0][0])
                  for vertical in (False, True)]
    except cv2.error:
        result['reason'] = 'raw_edge_sharpness_not_measurable'
        return result
    result['edge_width_horizontal_vertical_px'] = widths
    if any(not np.isfinite(w) or w <= 0 or w > max_edge_width for w in widths):
        result['reason'] = 'raw_edges_blurred_need_more_light_shorter_exposure'
        return result
    roughness = grid_roughness(points, pattern)
    result['grid_rough_rms_px'] = roughness
    if not np.isfinite(roughness) or roughness > max_grid_roughness:
        result['reason'] = 'grid_jitter_hold_board_flat_and_still'
        return result
    result.update(ok=True, reason='complete_clear_board')
    return result


def serializable(result):
    return {key: val.tolist() if isinstance(val, np.ndarray) else val for key, val in result.items()}


def corner_distance(a, b, size):
    return float(np.mean(np.linalg.norm(np.asarray(a).reshape(-1, 2) -
                                        np.asarray(b).reshape(-1, 2), axis=1))) / min(size)
