"""Approximate factory RGB registration of measured underwater IR stereo depth.

The IR geometry comes from the water calibration. RGB intrinsics and IR-to-RGB
extrinsics come from the device's factory AIR calibration, so this registration
is explicitly approximate and has no verified underwater distance accuracy.
Only finite, strictly valid measurement-grid samples are accepted. Display depth
completion, pixel splatting and refractive scale multipliers are not used here.
"""
from dataclasses import dataclass

import numpy as np


@dataclass
class ProjectedRgbDepth:
    """One visible RGB pixel per independent IR measurement, in RGB metres."""

    uv: np.ndarray
    points: np.ndarray
    source_indices: np.ndarray
    depth_map: np.ndarray
    expected_density: float

    def estimate(self, box, min_points=30):
        """Robust central-60% bbox estimate; not an instance segmentation mask."""
        bounds = np.asarray(box, dtype=np.float64).reshape(-1)
        if bounds.size != 4 or not np.isfinite(bounds).all():
            return None, 0, 'invalid_box'
        x1, y1, x2, y2 = bounds
        if x2 <= x1 or y2 <= y1:
            return None, 0, 'invalid_box'
        dx, dy = .2 * (x2 - x1), .2 * (y2 - y1)
        height, width = self.depth_map.shape
        x1, y1 = max(0., x1 + dx), max(0., y1 + dy)
        x2, y2 = min(float(width), x2 - dx), min(float(height), y2 - dy)
        if x2 <= x1 or y2 <= y1:
            return None, 0, 'insufficient_depth'
        inside = ((self.uv[:, 0] >= x1) & (self.uv[:, 0] < x2) &
                  (self.uv[:, 1] >= y1) & (self.uv[:, 1] < y2))
        points = self.points[inside]
        count = len(points)
        minimum = max(30, int(min_points))
        # Account for sparse projected matching-grid samples instead of treating
        # the RGB raster's unobserved pixels as failed depth measurements.
        expected = (x2 - x1) * (y2 - y1) * self.expected_density
        if count < minimum or count < .10 * expected:
            return None, count, 'insufficient_depth'
        z = points[:, 2]
        median = np.median(z)
        mad = np.median(np.abs(z - median))
        keep = np.abs(z - median) <= max(.003, 3. * 1.4826 * mad)
        points = points[keep]
        count = len(points)
        if count < minimum or count < .10 * expected:
            return None, count, 'insufficient_inliers'
        z = points[:, 2]
        if np.percentile(z, 90) - np.percentile(z, 10) > max(.03, .12 * np.median(z)):
            return None, count, 'mixed_depth'
        # A tiny isolated cluster should not acquire an unrelated large box.
        pixels = self.uv[inside][keep]
        span = np.ptp(pixels, axis=0)
        if span[0] < .15 * (x2 - x1) or span[1] < .15 * (y2 - y1):
            return None, count, 'insufficient_coverage'
        return np.median(points, axis=0), count, 'approximate_factory_rgb_region_estimate'


class ApproximateRgbProjection:
    """Project strict IR measurements using water IR and factory RGB geometry."""

    def __init__(self, meta, geometry):
        factory = meta['sdk_factory_air_parameters_not_water_calibration']
        color = factory['color']
        self.K = np.asarray(color['K'], dtype=np.float64)
        self.D = np.asarray(color['D'], dtype=np.float64).reshape(-1)
        self.model = str(color.get('model', 'distortion.brown_conrady')).split('.')[-1]
        self.R1 = np.asarray(geometry.R1, dtype=np.float64)
        self.R = np.asarray(factory['R_left_ir_to_color'], dtype=np.float64)
        self.T = np.asarray(factory['T_left_ir_to_color_m'], dtype=np.float64).reshape(3)
        self.color_size = tuple(int(v) for v in meta.get('color_reference_size', (640, 480)))
        if (len(self.color_size) != 2 or min(self.color_size) <= 0 or
                self.K.shape != (3, 3) or self.D.size != 5 or
                not np.isfinite(self.K).all() or not np.isfinite(self.D).all() or
                min(self.K[0, 0], self.K[1, 1]) <= 0 or not np.isfinite(self.T).all()):
            raise ValueError('Invalid factory RGB intrinsics or translation.')
        for rotation in (self.R1, self.R):
            if (rotation.shape != (3, 3) or not np.isfinite(rotation).all() or
                    not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-4) or
                    np.linalg.det(rotation) < .99):
                raise ValueError('Invalid IR rectification / factory RGB rotation.')
        if self.model == 'inverse_brown_conrady' and np.count_nonzero(self.D) == 0:
            self.model = 'none'
        if self.model not in ('none', 'brown_conrady', 'modified_brown_conrady'):
            raise ValueError('Unsupported factory RGB distortion model: ' + self.model)
        if self.model == 'none' and np.count_nonzero(self.D):
            raise ValueError('Nonzero RGB distortion coefficients with distortion.none.')

    def metadata(self):
        return {
            'method': 'approximate_factory_rgb_registration',
            'approximate_factory_rgb_registration': True,
            'verified': False,
            'distance_accuracy_verified': False,
            'rgb_calibration_medium': 'factory_air',
            'ir_calibration_medium': 'water',
            'coordinate_frame': 'original_rgb_optical',
            'units': 'metres',
            'color_size': list(self.color_size),
            'measurements': 'strict_unique_ir_samples_nearest_rgb_z_buffer',
            'roi': 'central_60_percent_bbox_with_MAD_and_density_gates',
        }

    def project(self, depth, measurement_geometry):
        """Return sparse RGB measurements, without filling any missing depth.

        GPU sampling already supplies fx=P1.fx/stride and cx=P1.cx/stride
        for samples at native pixels [0,stride,...]. CPU full-size grids keep
        the small matching-grid intrinsics, requiring pixel-centre scaling.
        """
        depth = np.asarray(depth)
        if depth.ndim != 2:
            raise ValueError('Strict measured depth must be a 2D array.')
        height, width = depth.shape
        stereo_width, stereo_height = measurement_geometry.stereo_size
        sx, sy = width / stereo_width, height / stereo_height
        if sx < 1. or sy < 1. or not np.isclose(sx, sy):
            raise ValueError('Measurement grid must match or uniformly upscale the stereo grid.')
        stride = int(round(sx))
        if not np.isclose(sx, stride):
            raise ValueError('Measurement grid scale must be an integer.')
        fx, fy = measurement_geometry.fx * sx, measurement_geometry.fy * sy
        cx = (measurement_geometry.cx + .5) * sx - .5
        cy = (measurement_geometry.cy + .5) * sy - .5
        if not np.isfinite((fx, fy, cx, cy)).all() or min(fx, fy) <= 0:
            raise ValueError('Invalid measurement-grid intrinsics.')
        # Exactly one sample per matching cell for an upsampled CPU grid; GPU
        # measurement grids are already sampled and therefore have stride 1.
        sampled = depth[::stride, ::stride]
        rows, cols = np.nonzero(np.isfinite(sampled) & (sampled > 0))
        yy, xx = rows * stride, cols * stride
        z = sampled[rows, cols].astype(np.float64)
        rectified = np.column_stack(((xx - cx) * z / fx, (yy - cy) * z / fy, z))
        # Row-vector inverse of X_rectified = R1 @ X_left.
        points = (rectified @ self.R1) @ self.R.T + self.T
        source_indices = yy * width + xx
        good = np.isfinite(points).all(axis=1) & (points[:, 2] > 0)
        points, source_indices = points[good], source_indices[good]
        x, y = points[:, 0] / points[:, 2], points[:, 1] / points[:, 2]
        if self.model in ('brown_conrady', 'modified_brown_conrady'):
            k1, k2, p1, p2, k3 = self.D
            r2 = x * x + y * y
            radial = 1. + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2
            if self.model == 'modified_brown_conrady':
                x, y = x * radial, y * radial
                xd = x + 2. * p1 * x * y + p2 * (r2 + 2. * x * x)
                yd = y + 2. * p2 * x * y + p1 * (r2 + 2. * y * y)
            else:
                xd = x * radial + 2. * p1 * x * y + p2 * (r2 + 2. * x * x)
                yd = y * radial + 2. * p2 * x * y + p1 * (r2 + 2. * y * y)
            x, y = xd, yd
        uv = np.column_stack((x * self.K[0, 0] + self.K[0, 2],
                              y * self.K[1, 1] + self.K[1, 2]))
        color_width, color_height = self.color_size
        # Bound floats before integer conversion; NaNs/infinities stay invalid.
        good = (np.isfinite(uv).all(axis=1) & (uv[:, 0] >= -.5) &
                (uv[:, 0] < color_width - .5) & (uv[:, 1] >= -.5) &
                (uv[:, 1] < color_height - .5))
        uv = np.floor(uv[good] + .5).astype(np.int32)
        points, source_indices = points[good], source_indices[good]
        # Competing IR samples may land in one RGB pixel. Keep the nearest
        # positive RGB Z only, without duplicating points into adjacent pixels.
        order = np.argsort(points[:, 2], kind='stable')
        pixel_ids = uv[order, 1] * color_width + uv[order, 0]
        _, first = np.unique(pixel_ids, return_index=True)
        keep = order[first]
        uv, points, source_indices = uv[keep], points[keep], source_indices[keep]
        depth_map = np.full((color_height, color_width), np.nan, dtype=np.float32)
        depth_map[uv[:, 1], uv[:, 0]] = points[:, 2]
        density = min(1., (fx / stride) * (fy / stride) /
                      (self.K[0, 0] * self.K[1, 1]))
        return ProjectedRgbDepth(uv, points, source_indices, depth_map, float(density))
