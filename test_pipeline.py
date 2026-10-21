"""Small CPU regression check against the saved pre-refactor Git baseline."""
import ast
import json
import math
from pathlib import Path
import subprocess
import tempfile
import time
from types import SimpleNamespace

import cv2
import numpy as np

from camera import Camera, FrameRateTracker
from output import detection_records
from run import parse_args
from stereo import WaterGeometry, StereoDepth, estimate_target


def legacy_geometry_and_depth():
    namespace = dict(cv2=cv2, np=np, json=json, math=math, time=time,
                     Path=Path, SimpleNamespace=SimpleNamespace)
    for file, names in (
        ('guided_depth_gpu.py', ('_nearest_support_indices', 'complete_disparity_for_display')),
        ('cpu_depth_display.py', ('complete_depth_for_display',)),
        ('coral_live.py', ('pixel_transform', 'WaterGeometry', 'StereoDepth', 'depth_colors', 'estimate_target')),
    ):
        source = subprocess.check_output(['git', 'show', '548ff84:'+file]).decode('utf-8')
        for node in ast.parse(source).body:
            if getattr(node, 'name', None) in names:
                code = ast.get_source_segment(source, node)
                code = code.replace('            from cpu_depth_display import complete_depth_for_display\n', '')
                exec(compile(code, file, 'exec'), namespace)
    return SimpleNamespace(**namespace)


def main():
    cv2.setNumThreads(2)
    old = legacy_geometry_and_depth()
    args = parse_args(['--offline', '.','--device', 'cpu'])
    left = np.random.default_rng(42).integers(0, 256, (720, 1280, 3), dtype=np.uint8)
    left = cv2.GaussianBlur(left, (3, 3), .7)
    right = np.roll(left, -96, axis=1)
    with tempfile.TemporaryDirectory() as temporary:
        yaml = Path(temporary)/'original.yaml'
        yaml.write_bytes(subprocess.check_output(['git', 'show', '548ff84:config/stereo_opencv.yaml']))
        old_geometry = old.WaterGeometry(yaml, 320)
        geometry = WaterGeometry(args.calibration, 320)
        for name in ('P1', 'P2', 'Q', 'P_left', 'P_right'):
            assert np.array_equal(getattr(old_geometry, name), getattr(geometry, name)), name
        color, pair = geometry.rectify(left, right)
        old_color, old_pair = old_geometry.rectify(left, right)
        assert np.array_equal(color, old_color)
        current, original = StereoDepth(geometry, args), old.StereoDepth(old_geometry, args)
        estimated, measured, _ = current.compute(pair, color)
        old_estimated, old_measured, _ = original.compute(old_pair, old_color)
        assert np.array_equal(estimated, old_estimated, equal_nan=True)
        assert np.array_equal(measured, old_measured, equal_nan=True)
        assert np.array_equal(current.colors, original.colors)
        box = np.array([400., 150., 1000., 650.])
        xyz, count, status = estimate_target(measured, box, geometry, 30)
        old_xyz, old_count, old_status = old.estimate_target(old_measured, box, old_geometry, 30)
        assert count > 30 and np.array_equal(xyz, old_xyz)
        assert (count, status) == (old_count, old_status)
        found = [(box, 0, .8, 'Healthy_Coral', xyz, count, status)]
        records = detection_records(found, geometry.size)
        assert records[0]['XYZ_rectified_left_m'] == xyz.tolist()
        assert records[0]['coordinate_frame'] == 'rectified_left'
        assert not detection_records([(box, 0, .59, 'Healthy_Coral', xyz, count, status)], geometry.size)
        try:
            geometry.rectify(left[:, :-1], right)
        except ValueError:
            pass
        else:
            raise AssertionError('Incorrect sensor dimensions were accepted')
    rate = FrameRateTracker()
    for stamp in np.arange(100)/30:
        health = rate.add(stamp)
    assert health['warmed_up'] and abs(health['measured_arrival_fps']-30) < 1e-6
    try:
        Camera(device=-1)
    except ValueError:
        pass
    else:
        raise AssertionError('Negative device index was accepted')
    print('PASS: original calibration/depth/XYZ, structured output, frame rate and input validation')


if __name__ == '__main__':
    main()
