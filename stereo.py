"""Water calibration, stereo matching and display-only depth completion."""

import json
import math
import time
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np


def pixel_transform(scale):
    """OpenCV resize pixel-center convention; applies to K and projection P."""
    return np.array([[scale, 0, (scale - 1) / 2], [0, scale, (scale - 1) / 2], [0, 0, 1.0]], dtype=np.float64)


class WaterGeometry:
    def __init__(self, path, stereo_width=320):
        self.source = Path(path).resolve()
        self.data = json.loads(self.source.read_text(encoding="utf-8-sig"))
        data = self.data
        self.size = tuple(data["size"])
        if self.size != (1280, 720):
            raise ValueError("Expected per-eye 1280x720 USB stereo calibration.")
        if data.get("medium") != "water" or data.get("model") != "effective_water_pinhole":
            raise ValueError(
                "Use an effective_water_pinhole water calibration; physical ray models need a different engine."
            )
        R = np.array(data["R_lr"], dtype=np.float64)
        T = np.array(data["T_lr_m"], dtype=np.float64).reshape(3)
        if (
            not np.isfinite(R).all()
            or not np.allclose(R.T @ R, np.eye(3), atol=1e-4)
            or np.linalg.det(R) < 0.99
        ):
            raise ValueError("Invalid stereo rotation.")
        self.baseline = float(np.linalg.norm(T))
        if not np.isfinite(T).all() or not 0.03 < self.baseline < 0.15 or T[0] >= 0:
            raise ValueError(
                "Expected horizontal left-to-right T in METERS, negative Tx, baseline 30..150 mm."
            )
        Ks, Ds = [], []
        for side in ("left", "right"):
            K = np.array(data[side]["K"], dtype=np.float64)
            D = np.array(data[side]["D"], dtype=np.float64)
            if (
                K.shape != (3, 3)
                or not np.isfinite(K).all()
                or not np.isfinite(D).all()
                or min(K[0, 0], K[1, 1]) <= 0
            ):
                raise ValueError("Invalid intrinsic matrix/distortion.")
            Ks.append(K)
            Ds.append(D)
        # Recalculate compatible R/P/Q from the same measured K,D,R,T; never
        # multiply focal lengths by 1.33 or apply Snell again to an effective K.
        self.R1, self.R2, self.P1, self.P2, self.Q, _, _ = cv2.stereoRectify(
            Ks[0], Ds[0], Ks[1], Ds[1], self.size, R, T, flags=cv2.CALIB_ZERO_DISPARITY, alpha=0
        )
        if abs(self.P2[1, 3]) > 1e-8 or self.P2[0, 3] >= 0:
            raise ValueError("Horizontal positive-disparity stereo required.")
        self.color_size = self.size
        self.stereo_size = (stereo_width, stereo_width * self.size[1] // self.size[0])
        if stereo_width not in (320, 640, 1280) or stereo_width > self.size[0]:
            raise ValueError("Stereo width must be 320, 640 or 1280 and cannot exceed the sensor width.")
        A = pixel_transform(1.0)
        self.maps = [
            cv2.initUndistortRectifyMap(K, D, r, A @ p, self.color_size, cv2.CV_16SC2)
            for K, D, r, p in zip(Ks, Ds, (self.R1, self.R2), (self.P1, self.P2))
        ]
        S = pixel_transform(stereo_width / self.size[0])
        self.right_small_map = cv2.initUndistortRectifyMap(
            Ks[1], Ds[1], self.R2, S @ self.P2, self.stereo_size, cv2.CV_16SC2
        )
        self.P_left = S @ self.P1
        self.P_right = S @ self.P2
        self.fx, self.fy = self.P_left[0, 0], self.P_left[1, 1]
        self.cx, self.cy = self.P_left[0, 2], self.P_left[1, 2]
        self.cx_delta = self.P_left[0, 2] - self.P_right[0, 2]
        self.fB = float(-self.P_right[0, 3])
        self.verified = bool(data.get("distance_accuracy_verified", False))
        self.quality_passed = data.get("quality_passed")

    def rectify_left(self, left):
        if left.shape != (self.size[1], self.size[0], 3):
            raise ValueError("Left image must match the calibration size %s." % (self.size,))
        return cv2.remap(left, *self.maps[0], cv2.INTER_LINEAR)

    def matching_images(self, color, right):
        if right.shape != (self.size[1], self.size[0], 3) or color.shape != right.shape:
            raise ValueError("Stereo images must match the calibration size %s." % (self.size,))
        if self.stereo_size == self.color_size:
            return [color, cv2.remap(right, *self.maps[1], cv2.INTER_LINEAR)]
        # Same subpixel calibration mapping on both matching grids. Direct small
        # maps avoid rectifying an unused full-resolution right image each frame.
        left_small = cv2.resize(color, self.stereo_size, interpolation=cv2.INTER_AREA)
        # Directly decimating the right image while area-filtering the left
        # introduced an avoidable texture/aliasing mismatch. Prefilter the raw
        # right grayscale before the small rectification map (no extra full RGB
        # remap). The 3x3 support is deliberately small to retain coral branches.
        right_gray = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)
        right_gray = cv2.GaussianBlur(right_gray, (3, 3), 0.7)
        right_small = cv2.remap(right_gray, *self.right_small_map, cv2.INTER_LINEAR)
        return [left_small, right_small]

    def depth(self, disparity, valid, output_scale=1.0):
        denominator = disparity - self.cx_delta * output_scale
        result = np.full(disparity.shape, np.nan, dtype=np.float32)
        good = valid & np.isfinite(denominator) & (denominator > 0.1)
        result[good] = self.fB * output_scale / denominator[good]
        return result

    def metadata(self):
        return {
            "calibration_file": str(self.source),
            "source_eye_size": self.size,
            "SBS_size": [2 * self.size[0], self.size[1]],
            "stereo_size": self.stereo_size,
            "rectified_color_size": self.color_size,
            "baseline_m": self.baseline,
            "depth_output_size": self.color_size,
            "depth_reconstruction": "native"
            if self.stereo_size == self.color_size
            else "full-color-guided WLS upsampling, NOT native full-resolution matching",
            "P_left_processing": self.P_left.tolist(),
            "P_right_processing": self.P_right.tolist(),
            "P_left_native": self.P1.tolist(),
            "R1": self.R1.tolist(),
            "depth_formula": "Z=-P2[0,3]/(d-(cx1-cx2)); units m",
            "target_coordinates": {
                "frame": "rectified_left",
                "origin": "left camera optical center",
                "axes": {"X": "right", "Y": "down", "Z": "forward along rectified optical axis"},
                "formula": "X=(u-cx)*Z/fx; Y=(v-cy)*Z/fy",
                "pixels": "u,v and fx,fy,cx,cy use the SAME rectified pixel grid, derived from P1",
                "estimator": "component-wise median of backprojected, MAD-filtered valid points in central 60% of bbox",
                "interpretation": "visible target-region position estimate; not the object physical centroid",
                "log_units": "m",
                "label_units": "cm",
                "straight_line_distance_formula": "sqrt(X*X+Y*Y+Z*Z); Z alone is axial depth",
            },
            "distance_accuracy_verified": self.verified,
            "calibration_quality_passed": self.quality_passed,
            "note": "Effective underwater K,D already absorb refractive effects; no extra 1.33 multiplier.",
        }


class StereoDepth:
    def __init__(self, geometry, args):
        if not hasattr(cv2, "ximgproc"):
            raise RuntimeError("opencv-contrib-python is required for WLS.")
        self.geometry, self.args = geometry, args
        required = geometry.fB / args.near + geometry.cx_delta + 4
        self.nd = args.num_disparities or max(16, int(math.ceil(required / 16)) * 16)
        w, h = geometry.stereo_size
        if self.nd % 16 or not 16 <= self.nd < w - args.block_size:
            raise ValueError("Invalid disparity range for chosen processing resolution/near distance.")
        if self.nd < required:
            raise ValueError("Disparity search does not cover --near. Increase --num-disparities.")
        b = args.block_size
        self.left = cv2.StereoSGBM_create(
            minDisparity=0,
            numDisparities=self.nd,
            blockSize=b,
            P1=8 * b * b,
            P2=32 * b * b,
            disp12MaxDiff=1,
            preFilterCap=31,
            uniquenessRatio=10,
            speckleWindowSize=100,
            speckleRange=2,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
        )
        self.right = cv2.ximgproc.createRightMatcher(self.left)
        self.wls = cv2.ximgproc.createDisparityWLSFilter(self.left)
        self.wls.setLambda(args.wls_lambda)
        self.wls.setSigmaColor(args.wls_sigma)
        self.wls.setLRCthresh(24)
        # SGBM's default ROI discards the entire maximum-disparity strip even
        # where a smaller real disparity has a valid in-image correspondence.
        # Equal padding allows that search. It is NEVER measurement evidence:
        # recovered pixels must pass real-coordinate and left/right checks.
        self.conservative_x = cv2.getValidDisparityROI((0, 0, w, h), (0, 0, w, h), 0, self.nd, b)[0]
        self.padding = self.nd + b
        margin = max(1, b // 2)
        self.roi = (margin, margin, w - 2 * margin, h - 2 * margin)
        self.roi_mask = np.zeros((h, w), dtype=bool)
        x, y, rw, rh = self.roi
        self.roi_mask[y : y + rh, x : x + rw] = True
        self.full_roi_mask = cv2.resize(
            self.roi_mask.astype(np.uint8), geometry.color_size, interpolation=cv2.INTER_NEAREST
        ).astype(bool)
        self.yy, self.xx = np.indices((h, w))
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)) if args.enhance_stereo else None
        self.gpu_reconstruction = None
        self.colors = None
        self.measurement_grid = None
        self.display_depth = None
        self.measurement_geometry = geometry

    def enable_gpu_reconstruction(self, device):
        if device != "cpu" and self.geometry.stereo_size != self.geometry.color_size:
            from gpu_depth import CudaGuidedDepth

            self.gpu_reconstruction = CudaGuidedDepth(
                device,
                self.args.near,
                self.args.far,
                keep_on_gpu=True,
                complete_display=self.args.invalid_display == "complete",
            )
            stride = self.geometry.color_size[0] // self.geometry.stereo_size[0]
            # The measurement grid samples native pixels [0, stride, ...], not
            # resized pixel centers; use exactly that coordinate convention.
            p = self.geometry.P1
            self.measurement_geometry = SimpleNamespace(
                color_size=self.geometry.color_size,
                stereo_size=self.geometry.stereo_size,
                fx=p[0, 0] / stride,
                fy=p[1, 1] / stride,
                cx=p[0, 2] / stride,
                cy=p[1, 2] / stride,
            )

    def compute(self, images, guide=None):
        start = time.perf_counter()
        gray = [cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) if im.ndim == 3 else im for im in images]
        if self.clahe:
            gray = [self.clahe.apply(im) for im in gray]
        pad = self.padding
        padded = [cv2.copyMakeBorder(g, 0, 0, pad, pad, cv2.BORDER_CONSTANT, value=0) for g in gray]
        # Each matcher needs padding on only its search side; avoid computing
        # a second unused padded strip (WLS still receives aligned full arrays).
        dl_padded = np.full_like(padded[0], -16, dtype=np.int16)
        dr_padded = np.full_like(padded[0], -16 * self.nd, dtype=np.int16)
        dl_padded[:, :-pad] = self.left.compute(padded[0][:, :-pad], padded[1][:, :-pad])
        dr_padded[:, pad:] = self.right.compute(padded[1][:, pad:], padded[0][:, pad:])
        region = (slice(None), slice(pad, pad + gray[0].shape[1]))
        dl, dr = dl_padded[region], dr_padded[region]
        guide = images[0] if guide is None else guide
        output_size = (guide.shape[1], guide.shape[0])
        output_scale = output_size[0] / self.geometry.stereo_size[0]
        # Optimize on the matching grid first; high-resolution WLS itself was
        # measured too expensive on this CPU. Then use a fast, normalized,
        # color-edge-guided reconstruction, never plain color-map enlargement.
        filtered = (
            self.wls.filter(dl_padded, padded[0], disparity_map_right=dr_padded)[region].astype(np.float32)
            / 16.0
        )
        confidence = self.wls.getConfidenceMap()[region]
        low_valid = self.roi_mask & np.isfinite(filtered) & (filtered > 0.1) & (filtered < self.nd - 1)
        raw = dl.astype(np.float32) / 16.0
        xr = np.rint(self.xx - raw).astype(np.int32)
        sample = dr[self.yy, np.clip(xr, 0, dr.shape[1] - 1)].astype(np.float32) / 16.0
        consistent = (
            (raw > 0.1)
            & (xr >= 0)
            & (xr < dr.shape[1])
            & (sample < 0)
            & (np.abs(raw + sample) <= 1.5)
            & self.roi_mask
        )
        real_right_x = self.xx - filtered
        low_valid &= (real_right_x >= 0) & (real_right_x <= dr.shape[1] - 1)
        # Recover only independently supported matches in the former dead strip;
        # do not let WLS propagate invented depths from constant padding.
        low_valid &= (self.xx >= self.conservative_x) | consistent
        # Keep original matching-grid support for display-only hole completion.
        # These arrays are not mutated by the full-resolution reconstruction.
        display_disparity_small, display_valid_small = filtered, low_valid
        if output_scale != 1 and self.gpu_reconstruction is not None:
            low_strict = (
                low_valid
                & consistent
                & (confidence >= self.args.wls_confidence)
                & (np.abs(filtered - raw) <= 2.0)
            )
            estimated, measured, self.colors = self.gpu_reconstruction.reconstruct(
                guide, filtered, raw, low_valid, low_strict, self.geometry
            )
            self.display_depth = self.gpu_reconstruction.display_depth
            step = int(output_scale)
            self.measurement_grid = measured[::step, ::step].contiguous().cpu().numpy()
            return estimated, measured, 1000 * (time.perf_counter() - start)
        self.colors = None
        if output_scale != 1:
            source = np.stack((np.where(low_valid, filtered, 0), low_valid.astype(np.float32)), axis=-1)
            source = cv2.resize(source, output_size, interpolation=cv2.INTER_LINEAR)
            guidance = cv2.cvtColor(guide, cv2.COLOR_BGR2GRAY)
            normalized = cv2.ximgproc.guidedFilter(guidance, source, 8, 64.0, scale=1 / output_scale)
            support = normalized[..., 1] > 0.25
            filtered = normalized[..., 0] / np.maximum(normalized[..., 1], 0.001) * output_scale
            confidence = cv2.resize(confidence, output_size, interpolation=cv2.INTER_NEAREST)
        else:
            support = low_valid
        roi = self.roi_mask if output_scale == 1 else self.full_roi_mask
        if output_scale != 1:
            consistent = cv2.resize(
                consistent.astype(np.uint8), output_size, interpolation=cv2.INTER_NEAREST
            ).astype(bool)
            raw = cv2.resize(raw, output_size, interpolation=cv2.INTER_LINEAR) * output_scale
        low_support = (
            low_valid
            if output_scale == 1
            else cv2.resize(low_valid.astype(np.uint8), output_size, interpolation=cv2.INTER_NEAREST).astype(
                bool
            )
        )
        display_valid = (
            roi
            & support
            & low_support
            & np.isfinite(filtered)
            & (filtered > 0.1 * output_scale)
            & (filtered < (self.nd - 1) * output_scale)
        )
        real_right_x = np.arange(output_size[0], dtype=np.float32)[None, :] - filtered
        display_valid &= (real_right_x >= 0) & (real_right_x <= output_size[0] - 1)
        strict = (
            display_valid
            & consistent
            & (np.abs(filtered - raw) <= 2.0 * output_scale)
            & (confidence >= self.args.wls_confidence)
        )
        estimated = self.geometry.depth(filtered, display_valid, output_scale)
        measured = self.geometry.depth(filtered, strict, output_scale)
        for z in (estimated, measured):
            z[(z < self.args.near) | (z > self.args.far)] = np.nan
        self.measurement_grid = measured
        self.display_depth = estimated
        if self.args.invalid_display == "complete":
            completed = complete_depth_for_display(
                estimated,
                display_disparity_small,
                display_valid_small,
                self.geometry,
                self.args.near,
                self.args.far,
            )
            if completed is not None:
                self.display_depth = completed
            self.colors = depth_colors(self.display_depth, self.args.near, self.args.far)
        return estimated, measured, 1000 * (time.perf_counter() - start)

    def compute_frame(self, color, right):
        start = time.perf_counter()
        estimated, measured, _ = self.compute(self.geometry.matching_images(color, right), color)
        colors = self.colors
        if colors is None:
            colors = depth_colors(estimated, self.args.near, self.args.far)
        return {
            "estimated": estimated,
            "measured": measured,
            "colors": colors,
            "display_depth": self.display_depth,
            "grid": self.measurement_grid,
            "geometry": self.measurement_geometry,
            "stereo_ms": 1000 * (time.perf_counter() - start),
        }


def observations(boxes, names, confidence=0.6, depth=None, geometry=None, min_points=30):
    """Attach XYZ only when supplied depth belongs to these exact boxes/frame."""
    found = []
    for row in boxes:
        if not np.isfinite(row).all() or row[4] < max(0.6, confidence):
            continue
        box, score, cls = row[:4], float(row[4]), int(row[5])
        if box[2] <= box[0] or box[3] <= box[1]:
            continue
        pos, count, status = (
            (None, 0, "depth_not_updated")
            if depth is None
            else estimate_target(depth, box, geometry, min_points)
        )
        found.append((box, cls, score, str(names[cls]), pos, count, status))
    return found


def estimate_target(depth, box, geometry, min_points):
    """Return median XYZ (meters) in the rectified left camera frame.

    Pixel u,v and intrinsics must describe the same measurement grid. Each
    retained pixel is backprojected with X=(u-cx)*Z/fx, Y=(v-cy)*Z/fy before
    taking component-wise medians. The bbox core estimates the visible target
    region; it is not instance segmentation or the object's physical centroid.
    """
    scale = depth.shape[1] / geometry.color_size[0]
    # Upsampling does not create independent observations. Count at most one
    # depth sample per original matching-grid cell when accepting a target.
    stride = max(1, int(round(depth.shape[1] / geometry.stereo_size[0])))
    x1, y1, x2, y2 = np.asarray(box, float) * scale
    dx, dy = 0.2 * (x2 - x1), 0.2 * (y2 - y1)
    x1, y1 = max(0, int(math.ceil(x1 + dx))), max(0, int(math.ceil(y1 + dy)))
    x2, y2 = (
        min(depth.shape[1], int(math.floor(x2 - dx)) + 1),
        min(depth.shape[0], int(math.floor(y2 - dy)) + 1),
    )
    patch = depth[y1:y2:stride, x1:x2:stride]
    valid = np.isfinite(patch)
    yy, xx = np.where(valid)
    z = patch[valid]
    if len(z) < min_points or len(z) < 0.10 * patch.size:
        return None, int(len(z)), "insufficient_depth"
    median = np.median(z)
    mad = np.median(np.abs(z - median))
    keep = np.abs(z - median) <= max(0.003, 3 * 1.4826 * mad)
    z, xx, yy = z[keep], xx[keep] * stride + x1, yy[keep] * stride + y1
    if len(z) < min_points:
        return None, int(len(z)), "insufficient_inliers"
    if np.percentile(z, 90) - np.percentile(z, 10) > max(0.03, 0.12 * np.median(z)):
        return None, int(len(z)), "mixed_depth"
    output_scale = depth.shape[1] / geometry.stereo_size[0]
    cx, cy = (geometry.cx + 0.5) * output_scale - 0.5, (geometry.cy + 0.5) * output_scale - 0.5
    points = np.stack(
        ((xx - cx) * z / (geometry.fx * output_scale), (yy - cy) * z / (geometry.fy * output_scale), z),
        axis=1,
    )
    position = np.median(points, axis=0)
    return position, int(len(z)), "region_estimate"


def depth_colors(depth, near, far):
    levels = np.zeros(depth.shape, dtype=np.uint8)
    valid = np.isfinite(depth) & (depth >= near) & (depth <= far)
    levels[valid] = np.clip(10 + 245 * (1 / depth[valid] - 1 / far) / (1 / near - 1 / far), 10, 255).astype(
        np.uint8
    )
    return cv2.applyColorMap(levels, cv2.COLORMAP_TURBO)


def _nearest_support_indices(disparity, valid):
    """Indices of real support used as seeds, never measurement validity."""
    valid = valid & np.isfinite(disparity) & (disparity > 0)
    if np.count_nonzero(valid) < 32:
        return None
    _, labels = cv2.distanceTransformWithLabels(
        (~valid).astype(np.uint8), cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL
    )
    positions = np.zeros(int(labels.max()) + 1, np.int32)
    positions[labels[valid]] = np.flatnonzero(valid).astype(np.int32)
    return positions[labels]


def complete_disparity_for_display(disparity, valid):
    """Nearest supported disparity seed map, DISPLAY ONLY.

    No correspondence is claimed for extrapolated pixels, especially occlusions.
    With no useful anchors, refuse completion rather than invent a scene depth.
    """
    indices = _nearest_support_indices(disparity, valid)
    if indices is None:
        return None
    # Do not median-filter across different surfaces before edge reconstruction.
    return disparity.ravel()[indices]


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
    if estimated.ndim != 2 or disparity_small.ndim != 2 or valid_small.shape != disparity_small.shape:
        raise ValueError("Expected a 2D depth map and matching disparity/validity grids.")
    if disparity_small.shape != tuple(reversed(geometry.stereo_size)):
        raise ValueError("Disparity must use the calibrated matching grid and its pixel units.")
    height, width = estimated.shape
    small_height, small_width = disparity_small.shape
    if (
        not small_height
        or not small_width
        or small_height > height
        or small_width > width
        or height * small_width != width * small_height
    ):
        raise ValueError("Matching and display grids must have the same aspect ratio.")
    if (
        not np.isfinite((near, far, geometry.fB, geometry.cx_delta)).all()
        or not 0 < near < far
        or geometry.fB <= 0
    ):
        raise ValueError("Expected finite metre bounds and positive focal-length times baseline.")

    denominator = disparity_small - geometry.cx_delta
    anchors = (
        valid_small
        & np.isfinite(disparity_small)
        & (denominator > 0.1)
        & (denominator >= geometry.fB / far)
        & (denominator <= geometry.fB / near)
    )
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
