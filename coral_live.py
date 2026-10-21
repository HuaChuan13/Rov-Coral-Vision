"""SBS / D435i IR coral detection + calibrated SGBM/WLS. Opens with --live.

Q starts recording the complete visualization; W exits and finalizes the video.
All depth/YOLO results in a visualization belong to the SAME captured stereo pair.
"""
import argparse
import ctypes
import csv
import hashlib
import json
import math
import os
import queue
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = ROOT / 'models/best.pt'
DEFAULT_CALIBRATION = ROOT / 'config/stereo_opencv.yaml'
WINDOW = 'Underwater coral detection and calibrated depth | Q record | W save and exit'


def read_image(path):
    image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError('Cannot decode image: ' + str(path))
    return image


def save_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def host_array(value):
    return value.detach().cpu().numpy() if hasattr(value, 'detach') else value


def pixel_transform(scale):
    """OpenCV resize pixel-center convention; applies to K and projection P."""
    return np.array([[scale, 0, (scale - 1) / 2],
                     [0, scale, (scale - 1) / 2], [0, 0, 1.]], dtype=np.float64)


class WaterGeometry:
    def __init__(self, path, stereo_width=320):
        self.source = Path(path).resolve()
        if self.source.suffix.lower() in ('.yaml', '.yml'):
            # Read through memory so OpenCV also supports Unicode Windows paths.
            fs = cv2.FileStorage(self.source.read_text(encoding='utf-8-sig'),
                                 cv2.FILE_STORAGE_READ | cv2.FILE_STORAGE_MEMORY)
            try:
                if not fs.isOpened():
                    raise ValueError('Cannot read OpenCV calibration YAML.')
                self.data = {
                    'medium': 'water', 'model': fs.getNode('model').string(),
                    'size': [int(fs.getNode('image_width').real()), int(fs.getNode('image_height').real())],
                    'left': {'K': fs.getNode('K1').mat().tolist(), 'D': fs.getNode('D1').mat().reshape(-1).tolist()},
                    'right': {'K': fs.getNode('K2').mat().tolist(), 'D': fs.getNode('D2').mat().reshape(-1).tolist()},
                    'R_lr': fs.getNode('R').mat().tolist(),
                    'T_lr_m': fs.getNode('T').mat().reshape(-1).tolist(),
                    'quality_passed': bool(fs.getNode('quality_passed').real()),
                    'distance_accuracy_verified': False,
                }
            finally:
                fs.release()
        else:
            self.data = json.loads(self.source.read_text(encoding='utf-8-sig'))
        data = self.data
        self.size = tuple(data['size'])
        if self.size not in ((640, 480), (1280, 720)):
            raise ValueError('Expected per-eye 640x480 or 1280x720 calibration.')
        if data.get('medium') != 'water' or data.get('model') != 'effective_water_pinhole':
            raise ValueError('Use an effective_water_pinhole water calibration; physical ray models need a different engine.')
        R = np.array(data['R_lr'], dtype=np.float64)
        T = np.array(data['T_lr_m'], dtype=np.float64).reshape(3)
        if not np.isfinite(R).all() or not np.allclose(R.T @ R, np.eye(3), atol=1e-4) or np.linalg.det(R) < .99:
            raise ValueError('Invalid stereo rotation.')
        self.baseline = float(np.linalg.norm(T))
        if not np.isfinite(T).all() or not .03 < self.baseline < .15 or T[0] >= 0:
            raise ValueError('Expected horizontal left-to-right T in METERS, negative Tx, baseline 30..150 mm.')
        Ks, Ds = [], []
        for side in ('left', 'right'):
            K = np.array(data[side]['K'], dtype=np.float64)
            D = np.array(data[side]['D'], dtype=np.float64)
            if K.shape != (3, 3) or not np.isfinite(K).all() or not np.isfinite(D).all() or min(K[0, 0], K[1, 1]) <= 0:
                raise ValueError('Invalid intrinsic matrix/distortion.')
            Ks.append(K)
            Ds.append(D)
        # Recalculate compatible R/P/Q from the same measured K,D,R,T; never
        # multiply focal lengths by 1.33 or apply Snell again to an effective K.
        self.R1, self.R2, self.P1, self.P2, self.Q, _, _ = cv2.stereoRectify(
            Ks[0], Ds[0], Ks[1], Ds[1], self.size, R, T,
            flags=cv2.CALIB_ZERO_DISPARITY, alpha=0)
        if abs(self.P2[1, 3]) > 1e-8 or self.P2[0, 3] >= 0:
            raise ValueError('Horizontal positive-disparity stereo required.')
        self.color_size = self.size
        self.stereo_size = (stereo_width, stereo_width * self.size[1] // self.size[0])
        if stereo_width not in (320, 640, 1280) or stereo_width > self.size[0]:
            raise ValueError('Stereo width must be 320, 640 or 1280 and cannot exceed the sensor width.')
        A = pixel_transform(1.)
        self.maps = [cv2.initUndistortRectifyMap(K, D, r, A @ p, self.color_size, cv2.CV_16SC2)
                     for K, D, r, p in zip(Ks, Ds, (self.R1, self.R2), (self.P1, self.P2))]
        S = pixel_transform(stereo_width / self.size[0])
        self.right_small_map = cv2.initUndistortRectifyMap(Ks[1], Ds[1], self.R2, S @ self.P2,
                                                          self.stereo_size, cv2.CV_16SC2)
        self.P_left = S @ self.P1
        self.P_right = S @ self.P2
        self.fx, self.fy = self.P_left[0, 0], self.P_left[1, 1]
        self.cx, self.cy = self.P_left[0, 2], self.P_left[1, 2]
        self.cx_delta = self.P_left[0, 2] - self.P_right[0, 2]
        self.fB = float(-self.P_right[0, 3])
        self.verified = bool(data.get('distance_accuracy_verified', False))
        self.quality_passed = data.get('quality_passed')

    def rectify(self, left, right):
        if left.shape != (self.size[1], self.size[0], 3) or right.shape != left.shape:
            raise ValueError('Camera images must match the per-eye calibration size %s.' % (self.size,))
        # Preserve the calibrated native sensor grid for detection and display.
        color = cv2.remap(left, *self.maps[0], cv2.INTER_LINEAR)
        if self.stereo_size == self.color_size:
            return color, [color, cv2.remap(right, *self.maps[1], cv2.INTER_LINEAR)]
        # Same subpixel calibration mapping on both matching grids. Direct small
        # maps avoid rectifying an unused full-resolution right image each frame.
        left_small = cv2.resize(color, self.stereo_size, interpolation=cv2.INTER_AREA)
        # Directly decimating the right image while area-filtering the left
        # introduced an avoidable texture/aliasing mismatch. Prefilter the raw
        # right grayscale before the small rectification map (no extra full RGB
        # remap). The 3x3 support is deliberately small to retain coral branches.
        right_gray = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)
        right_gray = cv2.GaussianBlur(right_gray, (3, 3), .7)
        right_small = cv2.remap(right_gray, *self.right_small_map, cv2.INTER_LINEAR)
        return color, [left_small, right_small]

    def depth(self, disparity, valid, output_scale=1.):
        denominator = disparity - self.cx_delta * output_scale
        result = np.full(disparity.shape, np.nan, dtype=np.float32)
        good = valid & np.isfinite(denominator) & (denominator > .1)
        result[good] = self.fB * output_scale / denominator[good]
        return result

    def metadata(self):
        return {'calibration_file': str(self.source), 'source_eye_size': self.size,
                'SBS_size': [2*self.size[0], self.size[1]], 'stereo_size': self.stereo_size,
                'rectified_color_size': self.color_size, 'baseline_m': self.baseline,
                'depth_output_size': self.color_size,
                'depth_reconstruction': 'native' if self.stereo_size == self.color_size else 'full-color-guided WLS upsampling, NOT native full-resolution matching',
                'P_left_processing': self.P_left.tolist(), 'P_right_processing': self.P_right.tolist(),
                'P_left_native': self.P1.tolist(),
                'R1': self.R1.tolist(), 'depth_formula': 'Z=-P2[0,3]/(d-(cx1-cx2)); units m',
                'target_coordinates': {
                    'frame': 'rectified_left', 'origin': 'left camera optical center',
                    'axes': {'X': 'right', 'Y': 'down', 'Z': 'forward along rectified optical axis'},
                    'formula': 'X=(u-cx)*Z/fx; Y=(v-cy)*Z/fy',
                    'pixels': 'u,v and fx,fy,cx,cy use the SAME rectified pixel grid, derived from P1',
                    'estimator': 'component-wise median of backprojected, MAD-filtered valid points in central 60% of bbox',
                    'interpretation': 'visible target-region position estimate; not the object physical centroid',
                    'log_units': 'm', 'label_units': 'cm',
                    'straight_line_distance_formula': 'sqrt(X*X+Y*Y+Z*Z); Z alone is axial depth'},
                'distance_accuracy_verified': self.verified,
                'calibration_quality_passed': self.quality_passed,
                'note': 'Effective underwater K,D already absorb refractive effects; no extra 1.33 multiplier.'}


class StereoDepth:
    def __init__(self, geometry, args):
        if not hasattr(cv2, 'ximgproc'):
            raise RuntimeError('opencv-contrib-python is required for WLS.')
        self.geometry, self.args = geometry, args
        required = geometry.fB / args.near + geometry.cx_delta + 4
        self.nd = args.num_disparities or max(16, int(math.ceil(required / 16)) * 16)
        w, h = geometry.stereo_size
        if self.nd % 16 or not 16 <= self.nd < w - args.block_size:
            raise ValueError('Invalid disparity range for chosen processing resolution/near distance.')
        if self.nd < required:
            raise ValueError('Disparity search does not cover --near. Increase --num-disparities.')
        b = args.block_size
        self.left = cv2.StereoSGBM_create(
            minDisparity=0, numDisparities=self.nd, blockSize=b, P1=8*b*b, P2=32*b*b,
            disp12MaxDiff=1, preFilterCap=31, uniquenessRatio=10,
            speckleWindowSize=100, speckleRange=2, mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)
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
        margin = max(1, b//2)
        self.roi = (margin, margin, w-2*margin, h-2*margin)
        self.roi_mask = np.zeros((h, w), dtype=bool)
        x, y, rw, rh = self.roi
        self.roi_mask[y:y+rh, x:x+rw] = True
        self.full_roi_mask = cv2.resize(self.roi_mask.astype(np.uint8), geometry.color_size,
                                       interpolation=cv2.INTER_NEAREST).astype(bool)
        self.yy, self.xx = np.indices((h, w))
        self.clahe = cv2.createCLAHE(clipLimit=2., tileGridSize=(8, 8)) if args.enhance_stereo else None
        self.gpu_reconstruction = None
        self.colors = None
        self.measurement_grid = None
        self.display_depth = None
        self.measurement_geometry = geometry

    def enable_gpu_reconstruction(self, device):
        if device != 'cpu' and self.geometry.stereo_size != self.geometry.color_size:
            from guided_depth_gpu import CudaGuidedDepth
            self.gpu_reconstruction = CudaGuidedDepth(device, self.args.near, self.args.far,
                keep_on_gpu=True, complete_display=self.args.invalid_display == 'complete')
            stride = self.geometry.color_size[0]//self.geometry.stereo_size[0]
            # The measurement grid samples native pixels [0, stride, ...], not
            # resized pixel centers; use exactly that coordinate convention.
            p = self.geometry.P1
            self.measurement_geometry = SimpleNamespace(
                color_size=self.geometry.color_size, stereo_size=self.geometry.stereo_size,
                fx=p[0, 0]/stride, fy=p[1, 1]/stride, cx=p[0, 2]/stride, cy=p[1, 2]/stride)

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
        dr_padded = np.full_like(padded[0], -16*self.nd, dtype=np.int16)
        dl_padded[:, :-pad] = self.left.compute(padded[0][:, :-pad], padded[1][:, :-pad])
        dr_padded[:, pad:] = self.right.compute(padded[1][:, pad:], padded[0][:, pad:])
        region = (slice(None), slice(pad, pad+gray[0].shape[1]))
        dl, dr = dl_padded[region], dr_padded[region]
        guide = images[0] if guide is None else guide
        output_size = (guide.shape[1], guide.shape[0])
        output_scale = output_size[0] / self.geometry.stereo_size[0]
        # Optimize on the matching grid first; high-resolution WLS itself was
        # measured too expensive on this CPU. Then use a fast, normalized,
        # color-edge-guided reconstruction, never plain color-map enlargement.
        filtered = self.wls.filter(dl_padded, padded[0], disparity_map_right=dr_padded)[region].astype(np.float32) / 16.
        confidence = self.wls.getConfidenceMap()[region]
        low_valid = self.roi_mask & np.isfinite(filtered) & (filtered > .1) & (filtered < self.nd-1)
        raw = dl.astype(np.float32) / 16.
        xr = np.rint(self.xx - raw).astype(np.int32)
        sample = dr[self.yy, np.clip(xr, 0, dr.shape[1]-1)].astype(np.float32) / 16.
        consistent = ((raw > .1) & (xr >= 0) & (xr < dr.shape[1]) & (sample < 0)
                      & (np.abs(raw+sample) <= 1.5) & self.roi_mask)
        real_right_x = self.xx - filtered
        low_valid &= (real_right_x >= 0) & (real_right_x <= dr.shape[1]-1)
        # Recover only independently supported matches in the former dead strip;
        # do not let WLS propagate invented depths from constant padding.
        low_valid &= (self.xx >= self.conservative_x) | consistent
        # Keep original matching-grid support for display-only hole completion.
        # These arrays are not mutated by the full-resolution reconstruction.
        display_disparity_small, display_valid_small = filtered, low_valid
        if output_scale != 1 and self.gpu_reconstruction is not None:
            low_strict = low_valid & consistent & (confidence >= self.args.wls_confidence) & (np.abs(filtered-raw) <= 2.)
            estimated, measured, self.colors = self.gpu_reconstruction.reconstruct(
                guide, filtered, raw, low_valid, low_strict, self.geometry)
            self.display_depth = self.gpu_reconstruction.display_depth
            step = int(output_scale)
            self.measurement_grid = measured[::step, ::step].contiguous().cpu().numpy()
            return estimated, measured, 1000*(time.perf_counter()-start)
        self.colors = None
        if output_scale != 1:
            source = np.stack((np.where(low_valid, filtered, 0), low_valid.astype(np.float32)), axis=-1)
            source = cv2.resize(source, output_size, interpolation=cv2.INTER_LINEAR)
            guidance = cv2.cvtColor(guide, cv2.COLOR_BGR2GRAY)
            normalized = cv2.ximgproc.guidedFilter(guidance, source, 8, 64., scale=1/output_scale)
            support = normalized[..., 1] > .25
            filtered = normalized[..., 0] / np.maximum(normalized[..., 1], .001) * output_scale
            confidence = cv2.resize(confidence, output_size, interpolation=cv2.INTER_NEAREST)
        else:
            support = low_valid
        roi = self.roi_mask if output_scale == 1 else self.full_roi_mask
        if output_scale != 1:
            consistent = cv2.resize(consistent.astype(np.uint8), output_size, interpolation=cv2.INTER_NEAREST).astype(bool)
            raw = cv2.resize(raw, output_size, interpolation=cv2.INTER_LINEAR) * output_scale
        low_support = low_valid if output_scale == 1 else cv2.resize(low_valid.astype(np.uint8), output_size, interpolation=cv2.INTER_NEAREST).astype(bool)
        display_valid = roi & support & low_support & np.isfinite(filtered) & (filtered > .1*output_scale) & (filtered < (self.nd-1)*output_scale)
        real_right_x = np.arange(output_size[0], dtype=np.float32)[None, :] - filtered
        display_valid &= (real_right_x >= 0) & (real_right_x <= output_size[0]-1)
        strict = (display_valid & consistent & (np.abs(filtered-raw) <= 2.*output_scale)
                  & (confidence >= self.args.wls_confidence))
        estimated = self.geometry.depth(filtered, display_valid, output_scale)
        measured = self.geometry.depth(filtered, strict, output_scale)
        for z in (estimated, measured):
            z[(z < self.args.near) | (z > self.args.far)] = np.nan
        self.measurement_grid = measured
        self.display_depth = estimated
        if self.args.invalid_display == 'complete':
            from cpu_depth_display import complete_depth_for_display
            completed = complete_depth_for_display(
                estimated, display_disparity_small, display_valid_small,
                self.geometry, self.args.near, self.args.far)
            if completed is not None:
                self.display_depth = completed
            self.colors = depth_colors(self.display_depth, self.args.near, self.args.far)
        return estimated, measured, 1000*(time.perf_counter()-start)


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
    dx, dy = .2*(x2-x1), .2*(y2-y1)
    x1, y1 = max(0, int(math.ceil(x1+dx))), max(0, int(math.ceil(y1+dy)))
    x2, y2 = min(depth.shape[1], int(math.floor(x2-dx))+1), min(depth.shape[0], int(math.floor(y2-dy))+1)
    patch = depth[y1:y2:stride, x1:x2:stride]
    valid = np.isfinite(patch)
    yy, xx = np.where(valid)
    z = patch[valid]
    if len(z) < min_points or len(z) < .10 * patch.size:
        return None, int(len(z)), 'insufficient_depth'
    median = np.median(z)
    mad = np.median(np.abs(z-median))
    keep = np.abs(z-median) <= max(.003, 3*1.4826*mad)
    z, xx, yy = z[keep], xx[keep]*stride+x1, yy[keep]*stride+y1
    if len(z) < min_points:
        return None, int(len(z)), 'insufficient_inliers'
    if np.percentile(z, 90)-np.percentile(z, 10) > max(.03, .12*np.median(z)):
        return None, int(len(z)), 'mixed_depth'
    output_scale = depth.shape[1] / geometry.stereo_size[0]
    cx, cy = (geometry.cx+.5)*output_scale-.5, (geometry.cy+.5)*output_scale-.5
    points = np.stack(((xx-cx)*z/(geometry.fx*output_scale), (yy-cy)*z/(geometry.fy*output_scale), z), axis=1)
    position = np.median(points, axis=0)
    return position, int(len(z)), 'region_estimate'


def text(image, label, xy, color=(235, 235, 235), scale=.48):
    cv2.putText(image, label, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def depth_colors(depth, near, far):
    levels = np.zeros(depth.shape, dtype=np.uint8)
    valid = np.isfinite(depth) & (depth >= near) & (depth <= far)
    levels[valid] = np.clip(10+245*(1/depth[valid]-1/far)/(1/near-1/far), 10, 255).astype(np.uint8)
    return cv2.applyColorMap(levels, cv2.COLORMAP_TURBO)


def visible_detections(detections, threshold=.6):
    """Display/log safety gate, independent of YOLO's own confidence filter."""
    cutoff = max(.6, float(threshold))
    return [d for d in detections if np.isfinite(d[2]) and float(d[2]) >= cutoff]


def draw_annotations(image, detections, show_distance=True):
    """Bold class/confidence and XYZ labels, with all coordinates in cm."""
    palette = ((70, 255, 80), (255, 90, 235), (255, 220, 30))  # BGR
    occupied = []
    h, w = image.shape[:2]
    font, scale, thickness = cv2.FONT_HERSHEY_SIMPLEX, .70, 2
    ordered = sorted(detections, key=lambda d: float(d[2]), reverse=True)
    for box, cls, score, name, pos, count, status in ordered:
        x1, y1, x2, y2 = np.rint(box).astype(int)
        x1, x2 = np.clip([x1, x2], 0, w-1)
        y1, y2 = np.clip([y1, y2], 0, h-1)
        accent = palette[int(cls) % len(palette)]
        cv2.rectangle(image, (x1, y1), (x2, y2), (10, 10, 10), 5, cv2.LINE_AA)
        cv2.rectangle(image, (x1, y1), (x2, y2), accent, 3, cv2.LINE_AA)
        center = ((x1+x2)//2, (y1+y2)//2)
        cv2.circle(image, center, 4, (10, 10, 10), -1, cv2.LINE_AA)
        cv2.circle(image, center, 2, (255, 255, 255), -1, cv2.LINE_AA)
    for box, cls, score, name, pos, count, status in ordered:
        x1, y1, x2, y2 = np.rint(box).astype(int)
        accent = palette[int(cls) % len(palette)]
        label = '%s  %.1f%%' % (name, 100*score)
        xy = 'X: N/A | Y: N/A' if pos is None else 'X~ %+.1f cm | Y~ %+.1f cm' % (100*pos[0], 100*pos[1])
        distance = 'Z: N/A' if pos is None else 'Z~ %.1f cm | n=%d' % (100*pos[2], count)
        lines = (label, xy, distance) if show_distance else (label,)
        cw = min(w-8, max(cv2.getTextSize(s, font, scale, thickness)[0][0] for s in lines)+16)
        ch = 86 if show_distance else 34
        candidates = [(x1, y1-ch-5), (x1, y1+6), (x1, y2-ch-6),
                      (x2-cw, y1-ch-5), (x2-cw, y2-ch-6), (x1, y2+6)]
        rectangles = []
        for x, y in candidates:
            x, y = int(np.clip(x, 4, w-cw-4)), int(np.clip(y, 4, h-ch-4))
            rectangles.append((x, y, x+cw, y+ch))
        def overlap(rect):
            return sum(max(0, min(rect[2], r[2])-max(rect[0], r[0])) *
                       max(0, min(rect[3], r[3])-max(rect[1], r[1])) for r in occupied)
        rect = min(rectangles, key=overlap)
        occupied.append(rect)
        x, y, _, _ = rect
        cv2.rectangle(image, (rect[0], rect[1]), (rect[2], rect[3]), (12, 12, 12), -1)
        cv2.rectangle(image, (rect[0], rect[1]), (rect[2], rect[3]), accent, 1, cv2.LINE_AA)
        cv2.putText(image, label, (x+8, y+23), font, scale, accent, thickness, cv2.LINE_AA)
        if show_distance:
            cv2.putText(image, xy, (x+8, y+49), font, scale, (0, 245, 255), thickness, cv2.LINE_AA)
            cv2.putText(image, distance, (x+8, y+75), font, scale, (0, 245, 255), thickness, cv2.LINE_AA)
    return image


def compose_view(color, estimated, detections, geometry, roi, args, fps, capture_fps, age_ms, recording,
                 rendered=None, rgb_preview=None):
    w, h = geometry.color_size
    if color.shape != (h, w, 3) or estimated.shape != (h, w):
        raise ValueError('Both visualization panels must match the calibrated sensor grid.')
    detections = visible_detections(detections, args.confidence)
    if rgb_preview is not None:
        if rgb_preview.shape != (h, w, 3):
            raise ValueError('RGB preview must match the panel size; expected %dx%d.' % (w, h))
        view = draw_annotations(rgb_preview.copy(), detections) if detections else rgb_preview
    else:
        view = draw_annotations(color.copy(), detections) if detections else color
    measurements = []
    for box, cls, score, name, pos, count, status in detections:
        record = {'class_id': int(cls), 'class': name, 'confidence': float(score),
                  'image_size': [w, h], 'status': status, 'independent_samples': count}
        if rgb_preview is not None:
            record.update(box_rgb_xyxy=[float(v) for v in box], coordinate_frame='raw_rgb',
                          XYZ_color_camera_m=None if pos is None else pos.tolist(),
                          registration_approximate=True)
        else:
            record.update(box_rectified_xyxy=[float(v) for v in box],
                          coordinate_frame='rectified_left_ir' if getattr(args, 'source', 'sbs') == 'realsense' else 'rectified_left',
                          XYZ_rectified_left_m=None if pos is None else pos.tolist())
        measurements.append(record)
    # Pure depth colors only. Completed display pixels NEVER enter measurements.
    colors = depth_colors(estimated, args.near, args.far) if rendered is None else rendered.copy()
    if rgb_preview is None and not getattr(args, 'depth_only_panel', False):
        draw_annotations(colors, detections)
    canvas = np.full((h+120, 2*w+128, 3), 25, np.uint8)
    canvas[72:72+h, :w] = view
    canvas[72:72+h, w:2*w] = colors
    def fitted(label, xy, max_width, color=(235, 235, 235), scale=.8):
        pixels = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)[0][0]
        text(canvas, label, xy, color, min(scale, scale*max_width/max(1, pixels)))
    prefix = 'RGB + YOLO' if rgb_preview is not None else ('LEFT IR + YOLO' if getattr(args, 'source', 'sbs') == 'realsense' else 'LEFT + YOLO')
    if getattr(args, 'input_enhancement', 'off') != 'off':
        prefix += ' + ENH'
    if not args.live:
        prefix = 'OFFLINE ' + prefix
    fitted('%s %dx%d | fresh %.1f FPS | capture %.1f FPS' % (prefix, w, h, fps, capture_fps), (15, 28), w-30)
    method = 'native' if geometry.stereo_size == geometry.color_size else 'guided reconstruction'
    if args.invalid_display == 'complete':
        method = 'COMPLETED / display estimate'
    depth_prefix = 'IR DEPTH' if rgb_preview is not None else 'DEPTH'
    fitted('%s %dx%d | %s | SGBM %dx%d' % (depth_prefix, w, h, method, *geometry.stereo_size), (w+15, 28), w-30)
    verified = ('WATER STEREO | RGB registration approximate | XYZ~ estimates' if rgb_preview is not None
                else ('RANGE VALIDATED' if geometry.verified else 'CALIBRATION UNVERIFIED: estimated XYZ, check known distances'))
    if getattr(geometry, 'quality_passed', None) is False:
        verified = 'CALIBRATION QUALITY NOT PASSED | XYZ~ estimates; check known distances'
    fitted(verified + ' | same-pair age %.0f ms' % age_ms, (15, 57), 2*w-30, (0, 200, 255), .7)
    text(canvas, 'REC' if recording else 'READY', (canvas.shape[1]-110, 57), (30, 30, 255) if recording else (0, 220, 100), .7)
    legend_x = 2*w+12
    text(canvas, 'Z (m)', (legend_x, 102), scale=.75)
    bar_height = h-130
    gradient = np.rint(np.linspace(255, 10, bar_height)).astype(np.uint8).reshape(-1, 1)
    bar = cv2.applyColorMap(np.repeat(gradient, 28, axis=1), cv2.COLORMAP_TURBO)
    canvas[132:132+bar_height, legend_x:legend_x+28] = bar
    for fraction in np.linspace(0, 1, 6):
        z = 1 / (1/args.near-fraction*(1/args.near-1/args.far))
        yy = 132+int(fraction*(bar_height-1))
        text(canvas, '%.2f' % z, (legend_x+36, yy+7), scale=.65)
    canvas[h+33:h+57, legend_x:legend_x+28] = cv2.applyColorMap(np.zeros((24, 28), np.uint8), cv2.COLORMAP_TURBO)
    text(canvas, 'N/A', (legend_x+36, h+53), scale=.65)
    unavailable = ('Completed pixels: DISPLAY ONLY; XYZ uses checked stereo' if args.invalid_display == 'complete'
                   else 'Dark purple = NO DEPTH')
    view_note = ('RGB camera XYZ (cm): +X right, +Y down, +Z forward' if rgb_preview is not None
                 else 'Rectified left XYZ (cm): +X right, +Y down, +Z forward')
    fitted('Q: record | W: save & exit | '+unavailable+' | '+view_note,
           (15, h+98), canvas.shape[1]-30, scale=.75)
    return canvas, measurements


class VideoRecorder:
    """Bounded asynchronous writer. CFR repeats preserve wall-time, not fresh FPS.

    Queue overload drops recording updates ONLY; never freezes camera/inference.
    Real acquisition frame rate is recorded independently in frame_metrics.csv.
    """
    def __init__(self, folder, fps=30.):
        self.folder, self.fps = Path(folder), float(fps)
        self.active = False
        self.error = None
        self.path = None
        self.dropped = 0
        self.written = 0
        self.repeated = 0
        self.thread = None

    def start(self, image, stamp):
        if self.active or self.thread is not None:
            return
        self.folder.mkdir(parents=True, exist_ok=True)
        self.size = (image.shape[1], image.shape[0])
        self.start_time = float(stamp)
        for suffix, codec in (('.mp4', 'mp4v'), ('.avi', 'MJPG')):
            self.path = self.folder / ('visualization' + suffix)
            self.writer = cv2.VideoWriter(str(self.path), cv2.VideoWriter_fourcc(*codec), self.fps, self.size)
            if self.writer.isOpened():
                self.codec = codec
                break
            self.writer.release()
        else:
            raise RuntimeError('No working MP4/AVI video encoder; check output folder.')
        self.queue = queue.Queue(maxsize=12)
        self.active = True
        self.thread = threading.Thread(target=self._write, name='video-writer', daemon=True)
        self.thread.start()
        self.submit(image, stamp)

    def submit(self, image, stamp):
        if self.error:
            raise RuntimeError('Recording failed: ' + self.error)
        if not self.active:
            return
        if image.shape[:2][::-1] != self.size:
            raise ValueError('Recording canvas dimensions changed.')
        try:
            self.queue.put_nowait((float(stamp), image.copy()))
        except queue.Full:
            self.dropped += 1

    def _write(self):
        last, next_index = None, 0
        try:
            while True:
                stamp, image = self.queue.get()
                target = max(0, int(math.ceil((stamp-self.start_time)*self.fps-1e-8)))
                if last is not None:
                    for _ in range(next_index, target):
                        self.writer.write(last)
                        self.written += 1
                        if _ > next_index:
                            self.repeated += 1
                    next_index = max(next_index, target)
                if image is None:
                    if self.written == 0 and last is not None:
                        self.writer.write(last)
                        self.written += 1
                    break
                last = image
        except Exception as exc:
            self.error = repr(exc)
        finally:
            self.writer.release()

    def close(self, stamp=None):
        if not self.active:
            return
        stamp = time.perf_counter() if stamp is None else stamp
        self.active = False
        while self.thread.is_alive():
            try:
                self.queue.put((float(stamp), None), timeout=.1)
                break
            except queue.Full:
                continue
        self.thread.join()
        save_json(self.folder / 'recording.json', {
            'video': str(self.path), 'codec': self.codec, 'size': self.size, 'fps_container': self.fps,
            'wall_duration_s': stamp-self.start_time, 'encoded_frames': self.written,
            'encoded_duration_s': self.written/self.fps, 'extra_held_frames': self.repeated,
            'queue_dropped_updates': self.dropped, 'writer_error': self.error,
            'timing_policy': 'constant-rate encoding with previous-frame hold; NOT proof of fresh camera/inference FPS'})
        print('Saved visualization:', self.path, flush=True)


def key_action(key):
    if key in (ord('q'), ord('Q')):
        return 'record'
    if key in (ord('w'), ord('W'), 27):
        return 'exit'
    return None


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument('--live', action='store_true')
    mode.add_argument('--offline', type=Path, help='Dataset folder, reads original/*_left.png; NEVER opens a camera')
    p.add_argument('--calibration', type=Path, default=DEFAULT_CALIBRATION)
    p.add_argument('--model', type=Path, default=DEFAULT_MODEL)
    p.add_argument('--camera', default='1', help='UVC device index; verify identity before live run')
    p.add_argument('--source', choices=('sbs', 'realsense'), default='sbs')
    p.add_argument('--serial', default='', help='Optional RealSense serial number')
    p.add_argument('--color-preview', action='store_true', help='RGB detections + approximately projected stereo Z on left; pure IR depth on right')
    p.add_argument('--depth-only-panel', action='store_true', help='Keep the right depth panel free of detection overlays')
    p.add_argument('--camera-metadata', type=Path, help='Saved camera.json for offline RGB projection; never opens a camera')
    p.add_argument('--backend', choices=('dshow', 'msmf', 'any'), default='dshow')
    p.add_argument('--swap', action='store_true')
    p.add_argument('--device', default='auto', help='YOLO GPU index, cpu or auto')
    p.add_argument('--half', action='store_true')
    p.add_argument('--cuda-convolution', choices=('native', 'cudnn'), default='native',
                   help='Native CUDA avoids cuDNN allocation failures on this 4GB GPU; CUDA inference remains enabled')
    p.add_argument('--input-enhancement', choices=('mild', 'off'), default='mild',
                   help='Photometric enhancement for detection/display only, preserving stereo measurement inputs')
    p.add_argument('--imgsz', type=int, default=416)
    p.add_argument('--confidence', type=float, default=.6)
    p.add_argument('--stereo-width', type=int, choices=(320, 640, 1280), default=320)
    p.add_argument('--num-disparities', type=int, default=0)
    p.add_argument('--block-size', type=int, default=3)
    p.add_argument('--wls-lambda', type=float, default=1600.)
    p.add_argument('--wls-sigma', type=float, default=.8)
    p.add_argument('--wls-confidence', type=float, default=128.)
    p.add_argument('--near', type=float, default=.20)
    p.add_argument('--far', type=float, default=2.)
    p.add_argument('--min-points', type=int, default=30)
    p.add_argument('--cv-threads', type=int, default=2)
    p.add_argument('--enhance-stereo', action='store_true', help='Optional paired CLAHE for grayscale matching only')
    p.add_argument('--invalid-display', choices=('complete', 'purple'), default='complete',
                   help='Pure-color display completion or purple holes; never fills metric measurements')
    p.add_argument('--frames', type=int, default=0, help='0: live unlimited, offline 120 frames')
    p.add_argument('--show-offline', action='store_true')
    p.add_argument('--record-offline', action='store_true', help='Test encoding without camera')
    p.add_argument('--output', type=Path, default=ROOT / 'coral_runs')
    p.add_argument('--stop-file', type=Path, default=None, help='Graceful external stop, also finalizes a recording')
    args = p.parse_args(argv)
    if not 0 < args.near < args.far or not 0 < args.confidence <= 1:
        p.error('Invalid distance range/confidence.')
    args.confidence = max(.6, args.confidence)
    if args.frames < 0 or args.cv_threads < 1 or args.min_points < 1 or args.block_size < 1 or args.block_size % 2 != 1:
        p.error('Invalid frame count/thread count/block size/minimum points.')
    if args.num_disparities < 0 or args.num_disparities % 16 or args.imgsz < 160 or args.imgsz % 32:
        p.error('Invalid disparities or inference image size (multiple of 32).')
    if args.wls_lambda <= 0 or args.wls_sigma <= 0 or not 0 <= args.wls_confidence <= 255:
        p.error('Invalid WLS parameters.')
    if args.live and (args.show_offline or args.record_offline):
        p.error('Offline flags cannot be used with --live.')
    if args.swap:
        p.error('The saved calibration is LEFT|RIGHT. Do not swap eyes with this calibration; verify camera orientation first.')
    if args.color_preview and args.source != 'realsense':
        p.error('--color-preview requires --source realsense.')
    return args


def main(argv=None):
    args = parse_args(argv)
    if not args.model.is_file():
        raise FileNotFoundError('Trained coral model not found: ' + str(args.model))
    geometry = WaterGeometry(args.calibration, args.stereo_width)
    matcher = StereoDepth(geometry, args)
    # Do this before importing Torch. No persistent Windows/system setting is changed.
    os.environ.setdefault('CUDA_MODULE_LOADING', 'LAZY')
    import torch
    from ultralytics import YOLO
    from image_enhancement import UnderwaterEnhancer
    torch.set_num_threads(1)
    torch.backends.cudnn.enabled = args.cuda_convolution == 'cudnn'
    torch.backends.cudnn.benchmark = False
    enhancer = UnderwaterEnhancer(args.input_enhancement)
    device = ('0' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else args.device
    model = YOLO(str(args.model))
    predict = dict(conf=args.confidence, verbose=False, device=device, half=args.half and device != 'cpu', imgsz=args.imgsz)
    for _ in range(5):
        warmup = np.zeros((geometry.size[1], geometry.size[0], 3), np.uint8)
        model.predict(warmup, **predict)
    matcher.enable_gpu_reconstruction(device)
    # Ultralytics changes OpenCV global threading on import; set AFTER warm-up.
    cv2.setNumThreads(args.cv_threads)
    run = args.output / datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    run.mkdir(parents=True, exist_ok=False)
    meta = {'mode': 'live' if args.live else 'offline_benchmark_no_camera',
            'args': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
            'model_sha256': hashlib.sha256(args.model.read_bytes()).hexdigest(),
            'classes': model.names, 'device': device, 'torch_version': torch.__version__,
            'opencv_version': cv2.__version__, 'geometry': geometry.metadata(),
            'input_enhancement': enhancer.metadata(),
            'enhancement_scope': 'detector and left display only; stereo images and reconstruction guide remain original',
            'cuda_convolution': args.cuda_convolution, 'inference_half': predict['half'],
            'num_disparities': matcher.nd,
            'depth_display': '%dx%d full-FOV TURBO; unavailable=%s' % (*geometry.size, args.invalid_display),
            'border_policy': 'equal search padding; recovered border requires real-image correspondence + LR consistency',
            'measurement': 'strict WLS confidence + raw LR check; central bbox MAD-filtered median XYZ; rectified-left frame, meters (labels cm)',
            'image_source': ('RGB YOLO; water-calibrated IR1/IR2 stereo' if args.color_preview else
                             ('left IR copied to 3 channels for YOLO; stereo IR1/IR2' if args.source == 'realsense' else 'color SBS')),
            'rgb_display': 'RGB detections with approximate projected stereo Z; pure IR depth panel' if args.color_preview else None,
            'warning': 'IR1/IR2 sensor timestamps checked within 2 ms' if args.source == 'realsense' else 'Same SBS host frame does NOT establish sensor exposure synchronization.'}
    save_json(run / 'settings.json', meta)
    save_json(run / 'calibration_snapshot.json', geometry.data)
    print('Model:', args.model, '| Classes:', model.names, flush=True)
    print('Camera NOT opened yet. Calibration:', args.calibration, flush=True)
    print('Per eye %dx%d | baseline %.3f mm | SGBM %s, %d disparities | device %s' %
          (*geometry.size, 1000*geometry.baseline, geometry.stereo_size, matcher.nd, device), flush=True)
    print('Input enhancement:', args.input_enhancement, '| CUDA convolution:', args.cuda_convolution,
          '| FP16:', predict['half'], flush=True)
    if not geometry.verified:
        print('WARNING: calibration distance accuracy is UNVERIFIED. Do not treat displayed Z as validated accuracy.', flush=True)
    if geometry.quality_passed is False:
        print('WARNING: the supplied water calibration has quality_passed=false. Check known distances before evaluating range accuracy.', flush=True)
    print('Output:', run, '| Q starts recording; W saves and exits.', flush=True)
    camera, pool = None, None
    recorder = VideoRecorder(run)
    show = args.live or args.show_offline
    rows, timestamps = [], deque(maxlen=60)
    frame_no, previous_seq, skipped = 0, -1, 0
    error = None
    timer_active = False
    began = time.perf_counter()
    try:
        offline = []
        rgb_projection = None
        if args.offline:
            paths = sorted((args.offline / 'original').glob('*_left.png'))[:12]
            if not paths:
                raise ValueError('No original/*_left.png offline pairs.')
            for lp in paths:
                rgb = read_image(lp.with_name(lp.name.replace('_left', '_color'))) if args.color_preview else None
                offline.append((read_image(lp), read_image(lp.with_name(lp.name.replace('_left', '_right'))), rgb))
        else:
            # DirectShow and Python 3.9 on Windows otherwise have coarse sleep
            # granularity. Balance the request on exit; no persistent OS change.
            if hasattr(ctypes, 'WinDLL'):
                timer_active = ctypes.WinDLL('winmm').timeBeginPeriod(1) == 0
            from capture_process import ProcessStereoSBS
            camera = ProcessStereoSBS(source=args.source, serial=args.serial, device=args.camera,
                               width=2*geometry.size[0], height=geometry.size[1], fps=30,
                               backend=args.backend, fourcc='MJPG', swap=args.swap)
            if camera.meta['eye_size'] != list(geometry.size):
                raise RuntimeError('Actual per-eye size differs from calibration. Refusing incorrect scaled depths.')
            save_json(run / 'camera.json', camera.meta)
        if args.color_preview:
            from rgb_depth_projection import ApproximateRgbProjection
            if camera:
                projection_meta = camera.meta
            elif args.camera_metadata:
                projection_meta = json.loads(args.camera_metadata.read_text(encoding='utf-8-sig'))
            else:
                raise ValueError('Offline RGB projection requires --camera-metadata pointing to a saved camera.json.')
            rgb_projection = ApproximateRgbProjection(projection_meta, geometry)
            save_json(run / 'rgb_projection.json', rgb_projection.metadata())
        if show:
            view_scale = min(1., 1613/(2*geometry.size[0]+128), 900/(geometry.size[1]+120))
            window_size = (round((2*geometry.size[0]+128)*view_scale), round((geometry.size[1]+120)*view_scale))
            cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(WINDOW, *window_size)
        pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix='stereo-cpu')
        limit = args.frames or (0 if args.live else 120)
        with (run / 'frame_metrics.csv').open('w', newline='', encoding='utf-8-sig') as log, \
                (run / 'detections.jsonl').open('w', encoding='utf-8') as detections_log:
            writer = csv.writer(log)
            writer.writerow(['processed_frame', 'capture_seq', 'capture_monotonic_s', 'source_age_ms',
                             'capture_fps', 'fresh_processing_fps', 'processing_ms', 'stereo_ms',
                             'yolo_ms', 'strict_valid_fraction', 'detections', 'skipped_camera_frames', 'enhancement_ms'])
            while not limit or frame_no < limit:
                if args.stop_file and args.stop_file.exists():
                    break
                if show and cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                    break
                key = cv2.pollKey() & 255 if show else -1
                action = key_action(key)
                if action == 'exit':
                    break
                if action == 'record' and frame_no:
                    recorder.start(canvas, time.perf_counter())
                if camera:
                    packet = camera.latest()
                    if packet.seq == previous_seq:
                        time.sleep(.001)
                        continue
                    left, right, stamp, seq = packet.left, packet.right, packet.timestamp, packet.seq
                    rgb_preview = packet.color if args.color_preview else None
                    sensor_metadata = packet.sensor_metadata
                    if args.color_preview and rgb_preview is None:
                        raise RuntimeError('No RGB color frame received from D435i.')
                    capture_fps = camera.health()['measured_arrival_fps']
                    if previous_seq >= 0:
                        skipped += max(0, seq-previous_seq-1)
                    previous_seq = seq
                else:
                    left, right, rgb_preview = offline[frame_no % len(offline)]
                    sensor_metadata = None
                    stamp, seq, capture_fps = time.monotonic(), frame_no, 0.
                start = time.perf_counter()
                color, stereo_images = geometry.rectify(left, right)
                future = pool.submit(matcher.compute, stereo_images, color)
                enhance_start = time.perf_counter()
                detector_input = enhancer.apply(rgb_preview if rgb_preview is not None else color)
                enhancement_ms = 1000*(time.perf_counter()-enhance_start)
                detect_start = time.perf_counter()
                with torch.inference_mode():
                    # Let the predictor resize once; its postprocessor returns
                    # boxes directly in the native calibrated input coordinates.
                    result = model.predict(detector_input, **predict)[0]
                    boxes = result.boxes.data.cpu().numpy() if result.boxes is not None else np.empty((0, 6))
                    boxes = boxes.copy()
                yolo_ms = 1000*(time.perf_counter()-detect_start)
                estimated, measured, stereo_ms = future.result()
                projected = rgb_projection.project(matcher.measurement_grid, matcher.measurement_geometry) if rgb_projection else None
                found = []
                for row in boxes:
                    box, score, cls = row[:4], float(row[4]), int(row[5])
                    if not np.isfinite(score) or score < max(.6, args.confidence):
                        continue
                    if projected is not None:
                        delta_ms = sensor_metadata.get('color_ir_delta_ms') if sensor_metadata else None
                        if camera and (delta_ms is None or delta_ms > 50.):
                            pos, count, status = None, 0, 'rgb_ir_time_mismatch'
                        else:
                            pos, count, status = projected.estimate(box, args.min_points)
                    else:
                        pos, count, status = estimate_target(matcher.measurement_grid, box, matcher.measurement_geometry, args.min_points)
                    found.append((box, cls, score, str(model.names[cls]), pos, count, status))
                now = time.perf_counter()
                # Fresh-frame throughput excludes repeated UI refreshes/encoded holds.
                fps = (len(timestamps)-1)/(timestamps[-1]-timestamps[0]) if len(timestamps) > 1 else 0.
                age_ms = max(0., 1000*(time.monotonic()-stamp))
                canvas, records = compose_view(detector_input if rgb_preview is None else color,
                                               estimated, found, geometry, matcher.roi, args,
                                               fps, capture_fps, age_ms, recorder.active, matcher.colors,
                                               rgb_preview=detector_input if rgb_preview is not None else None)
                if frame_no == 0 or frame_no == 29:
                    cv2.imencode('.png', canvas)[1].tofile(str(run / 'preview.png'))
                if frame_no == 29:
                    size_label = '%dx%d' % geometry.size
                    cv2.imencode('.png', color)[1].tofile(str(run/('left_rectified_'+size_label+'.png')))
                    cv2.imencode('.png', detector_input)[1].tofile(str(run/('detection_input_'+size_label+'.png')))
                    if rgb_preview is not None:
                        cv2.imencode('.png', rgb_preview)[1].tofile(str(run/'rgb_color_640x480.png'))
                    np.save(run/('depth_estimated_'+size_label+'_m.npy'), host_array(estimated))
                    np.save(run/('depth_measured_'+size_label+'_m.npy'), host_array(measured))
                    np.save(run/('depth_display_ONLY_completed_'+size_label+'_m.npy'), host_array(matcher.display_depth))
                    if projected is not None:
                        np.save(run/'depth_in_rgb_APPROXIMATE_m.npy', projected.depth_map)
                if show:
                    # Fit the physical desktop before HighGUI presentation;
                    # computation, saved arrays and video remain full resolution.
                    cv2.imshow(WINDOW, cv2.resize(canvas, window_size, interpolation=cv2.INTER_LINEAR))
                if args.record_offline and frame_no == 0:
                    recorder.start(canvas, time.perf_counter())
                elif recorder.active:
                    recorder.submit(canvas, time.perf_counter())
                frame_no += 1
                elapsed = 1000*(time.perf_counter()-start)
                row = [frame_no, seq, stamp, age_ms, capture_fps, fps, elapsed, stereo_ms, yolo_ms,
                       float(np.isfinite(matcher.measurement_grid).mean()), len(found), skipped, enhancement_ms]
                writer.writerow(row)
                detections_log.write(json.dumps({'frame': frame_no, 'capture_seq': seq, 'objects': records,
                                                'sensor_metadata': sensor_metadata}, ensure_ascii=False)+'\n')
                if frame_no % 30 == 0:
                    log.flush()
                    detections_log.flush()
                    print('Fresh %.1f FPS | processing %.1f ms | stereo %.1f ms | YOLO %.1f ms | detections %d' %
                          (fps, elapsed, stereo_ms, yolo_ms, len(found)), flush=True)
                timestamps.append(time.perf_counter())
                rows.append(row)
    except BaseException as exc:
        error = repr(exc)
        raise
    finally:
        try:
            recorder.close()
        finally:
            if camera:
                camera.close()
            if timer_active:
                ctypes.WinDLL('winmm').timeEndPeriod(1)
            if pool:
                pool.shutdown(wait=True)
            if show:
                cv2.destroyAllWindows()
            stable = rows[10:] if len(rows) > 10 else rows
            mean_ms = float(np.mean([r[6] for r in stable])) if stable else None
            summary = {'processed_frames': frame_no, 'mode': meta['mode'], 'error': error,
                       'run_wall_s': time.perf_counter()-began, 'skipped_camera_frames': skipped,
                       'mean_processing_ms_excluding_first10': mean_ms,
                       'processing_capacity_fps': 1000/mean_ms if mean_ms else None,
                       'last_window_fresh_fps': (len(timestamps)-1)/(timestamps[-1]-timestamps[0]) if len(timestamps)>1 else None,
                       'live_FPS_verified': bool(args.live and len(rows) >= 120),
                       'distance_accuracy_verified': geometry.verified,
                       'input_enhancement': enhancer.metadata(),
                       'mean_enhancement_ms': float(np.mean([r[12] for r in stable])) if stable else None,
                       'recording': str(recorder.path) if recorder.path else None,
                       'note': 'Offline capacity excludes camera/USB/GUI; encoded FPS is not fresh processing FPS.'}
            save_json(run / 'summary.json', summary)
            print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
