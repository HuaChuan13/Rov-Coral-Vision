"""Small CPU regression check against the saved pre-refactor Git baseline."""

import ast
import contextlib
import io
import json
import math
from pathlib import Path
import subprocess
import tempfile
import threading
import time
from types import SimpleNamespace

import cv2
import numpy as np

from camera import Camera, FrameRateTracker
from output import detection_records
from run import parse_args
from stereo import WaterGeometry, StereoDepth, estimate_target


def legacy_geometry_and_depth():
    namespace = dict(
        cv2=cv2, np=np, json=json, math=math, time=time, Path=Path, SimpleNamespace=SimpleNamespace
    )
    for file, names in (
        ("guided_depth_gpu.py", ("_nearest_support_indices", "complete_disparity_for_display")),
        ("cpu_depth_display.py", ("complete_depth_for_display",)),
        (
            "coral_live.py",
            ("pixel_transform", "WaterGeometry", "StereoDepth", "depth_colors", "estimate_target"),
        ),
    ):
        source = subprocess.check_output(["git", "show", "548ff84:" + file]).decode("utf-8")
        for node in ast.parse(source).body:
            if getattr(node, "name", None) in names:
                code = ast.get_source_segment(source, node)
                code = code.replace(
                    "            from cpu_depth_display import complete_depth_for_display\n", ""
                )
                exec(compile(code, file, "exec"), namespace)
    return SimpleNamespace(**namespace)


def check_async(left, right):
    """A blocked depth job must not prevent six independent detection results."""
    import torch
    import ultralytics
    import run

    release = threading.Event()
    depth_calls = []
    original_yolo, original_compute = ultralytics.YOLO, StereoDepth.compute_frame

    class Detector:
        names = {0: "Healthy_Coral"}

        def __init__(self, model):
            self.calls = 0

        def predict(self, image, **options):
            self.calls += 1
            if self.calls == 11:  # Five warm-ups and six independently processed frames.
                release.set()
            return [
                SimpleNamespace(
                    boxes=SimpleNamespace(data=torch.tensor([[400.0, 150.0, 1000.0, 650.0, 0.9, 0.0]]))
                )
            ]

    def blocked_compute(self, color, right):
        depth_calls.append(True)
        assert release.wait(5), "Detection waited for unfinished depth"
        return original_compute(self, color, right)

    try:
        ultralytics.YOLO = Detector
        StereoDepth.compute_frame = blocked_compute
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "data" / "original"
            source.mkdir(parents=True)
            for side, image in (("left", left), ("right", right)):
                assert cv2.imwrite(str(source / ("sample_" + side + ".png")), image)
            with contextlib.redirect_stdout(io.StringIO()):
                assert (
                    run.main(
                        [
                            "--offline",
                            str(root / "data"),
                            "--frames",
                            "6",
                            "--device",
                            "cpu",
                            "--depth-fps",
                            "max",
                            "--output",
                            str(root / "out"),
                        ]
                    )
                    == 0
                )
            folder = next((root / "out").iterdir())
            summary = json.loads((folder / "summary.json").read_text())
            records = [json.loads(line) for line in (folder / "detections.jsonl").read_text().splitlines()]
            depths = [
                json.loads(line) for line in (folder / "depth_measurements.jsonl").read_text().splitlines()
            ]
            assert summary["processed_frames"] == 6 and summary["depth_updates"] == 1
            assert len(depth_calls) == 1, "Depth tasks accumulated in the executor queue"
            assert len(records) == 6 and len(depths) == 1 and depths[0]["capture_seq"] == 0
            assert depths[0]["objects"][0]["XYZ_rectified_left_m"] is not None
            assert all(r["objects"][0]["XYZ_rectified_left_m"] is None for r in records)
            assert all(r["objects"][0]["status"] == "depth_not_updated" for r in records)
            assert not summary["error"]
            vision = [json.loads(line) for line in (folder / "vision_observations.jsonl").read_text().splitlines()]
            assert [v["capture_seq"] for v in vision[:-1]] == list(range(6)), vision
            assert all(not v["valid"] for v in vision) and vision[-1]["status"] == "stopped"
            StereoDepth.compute_frame = original_compute
            with contextlib.redirect_stdout(io.StringIO()):
                assert (
                    run.main(
                        [
                            "--offline",
                            str(root / "data"),
                            "--frames",
                            "2",
                            "--device",
                            "cpu",
                            "--depth-fps",
                            "0",
                            "--depth-max-age",
                            "0.000001",
                            "--output",
                            str(root / "stale"),
                        ]
                    )
                    == 0
                )
            stale = next((root / "stale").iterdir())
            frames = [json.loads(line) for line in (stale / "detections.jsonl").read_text().splitlines()]
            assert len(frames) == 2 and all(not frame["depth_fresh"] for frame in frames)
            assert all(frame["objects"][0]["XYZ_rectified_left_m"] is None for frame in frames)
    finally:
        release.set()
        ultralytics.YOLO, StereoDepth.compute_frame = original_yolo, original_compute


def check_measurement_view():
    """Depth labels must refer to their own frame and disappear when stale."""
    import output

    args = parse_args([])
    assert args.depth_fps == "max"
    assert args.imgsz == 416 and args.stereo_width == 320 and not args.show_depth
    assert parse_args(["--depth-fps", "10"]).depth_fps == 10
    assert parse_args(["--depth-fps", "0"]).depth_fps == 0
    geometry = SimpleNamespace(color_size=(1280, 720), verified=False, quality_passed=False)
    color = np.full((720, 1280, 3), 50, np.uint8)
    depth = dict(seq=42, colors=color.copy(), objects=[
        dict(box_rectified_xyxy=[200, 150, 400, 450], **{
            "class": "Healthy_Coral", "XYZ_rectified_left_m": [0.123, -0.045, 0.862]}),
        dict(box_rectified_xyxy=[600, 150, 900, 450], **{
            "class": "Bleached_Coral", "XYZ_rectified_left_m": None}),
    ])
    original_text, labels = output.text, []

    def capture(image, label, *args, **kwargs):
        labels.append(label)
        return original_text(image, label, *args, **kwargs)

    try:
        output.text = capture
        fresh = output.compose_view(color, [], geometry, args, 18, False, depth, 12, 80, 35)
        assert fresh.shape == (928, 1280, 3)
        assert fresh.shape[1::-1] == output.view_size(geometry, args)
        assert np.array_equal(fresh[48:768, :1280], color), "Historical depth annotated current image"
        assert np.array_equal(depth["colors"], color), "Rendering modified the depth worker's image"
        assert any("D1" in line and "X: +12.3" in line and "Y: -4.5" in line and "Z: 86.2" in line for line in labels)
        assert any("D2" in line and "no valid depth" in line for line in labels)
        assert any("frame 42" in line and "80 ms" in line for line in labels)
        assert any("18.0 FPS" in line and "35 ms" in line for line in labels)
        assert not any("DEPTH DEBUG" in line for line in labels)
        labels.clear()
        stale = output.compose_view(color, [], geometry, args, 18, False, depth, 12, 1000)
        assert any("Depth expired" in line for line in labels)
        assert not any("86.2" in line or line == "D1" for line in labels)
        assert np.array_equal(stale[48:768], color)
        labels.clear()
        output.compose_view(color, [], geometry, args, 18, False)
        assert any("Waiting for depth" in line for line in labels)
        labels.clear()
        output.compose_view(color, [], geometry, args, 18, False, dict(depth, objects=[]), 12, 80)
        assert any("No target" in line for line in labels)
        args = parse_args(["--show-depth"])
        debug = output.compose_view(color, [], geometry, args, 18, False, depth, 12, 80)
        assert debug.shape == (928, 2688, 3)
        assert debug.shape[1::-1] == output.view_size(geometry, args)
        assert any("DEPTH DEBUG | 12.0 FPS" in line for line in labels)
        assert np.array_equal(debug[48:768, :1280], color)
        assert not np.array_equal(debug[48:768, 1280:2560], color), "Missing depth boxes"
        stale = output.compose_view(color, [], geometry, args, 18, False, depth, 12, 1000)
        assert np.all(stale[48:768, 1280:2560] == 25)
    finally:
        output.text = original_text


def main():
    cv2.setNumThreads(2)
    old = legacy_geometry_and_depth()
    args = parse_args(["--offline", ".", "--device", "cpu"])
    left = np.random.default_rng(42).integers(0, 256, (720, 1280, 3), dtype=np.uint8)
    left = cv2.GaussianBlur(left, (3, 3), 0.7)
    right = np.roll(left, -96, axis=1)
    with tempfile.TemporaryDirectory() as temporary:
        yaml = Path(temporary) / "original.yaml"
        yaml.write_bytes(subprocess.check_output(["git", "show", "548ff84:config/stereo_opencv.yaml"]))
        old_geometry = old.WaterGeometry(yaml, 320)
        geometry = WaterGeometry(args.calibration, 320)
        for name in ("P1", "P2", "Q", "P_left", "P_right"):
            assert np.array_equal(getattr(old_geometry, name), getattr(geometry, name)), name
        color = geometry.rectify_left(left)
        pair = geometry.matching_images(color, right)
        old_color, old_pair = old_geometry.rectify(left, right)
        assert np.array_equal(color, old_color)
        current, original = StereoDepth(geometry, args), old.StereoDepth(old_geometry, args)
        estimated, measured, _ = current.compute(pair, color)
        old_estimated, old_measured, _ = original.compute(old_pair, old_color)
        assert np.array_equal(estimated, old_estimated, equal_nan=True)
        assert np.array_equal(measured, old_measured, equal_nan=True)
        assert np.array_equal(current.colors, original.colors)
        box = np.array([400.0, 150.0, 1000.0, 650.0])
        xyz, count, status = estimate_target(measured, box, geometry, 30)
        old_xyz, old_count, old_status = old.estimate_target(old_measured, box, old_geometry, 30)
        assert count > 30 and np.array_equal(xyz, old_xyz)
        assert (count, status) == (old_count, old_status)
        found = [(box, 0, 0.8, "Healthy_Coral", xyz, count, status)]
        records = detection_records(found, geometry.size)
        assert records[0]["XYZ_rectified_left_m"] == xyz.tolist()
        assert records[0]["coordinate_frame"] == "rectified_left"
        assert not detection_records([(box, 0, 0.59, "Healthy_Coral", xyz, count, status)], geometry.size)
        try:
            geometry.rectify_left(left[:, :-1])
        except ValueError:
            pass
        else:
            raise AssertionError("Incorrect sensor dimensions were accepted")
    rate = FrameRateTracker()
    for stamp in np.arange(100) / 30:
        health = rate.add(stamp)
    assert health["warmed_up"] and abs(health["measured_arrival_fps"] - 30) < 1e-6
    try:
        Camera(device=-1)
    except ValueError:
        pass
    else:
        raise AssertionError("Negative device index was accepted")
    for options in (
        ["--depth-fps", "-1"],
        ["--depth-fps", "nan"],
        ["--depth-max-age", "0"],
        ["--capture-fps", "0"],
    ):
        with contextlib.redirect_stderr(io.StringIO()):
            try:
                parse_args(options)
            except SystemExit as exc:
                assert exc.code == 2
            else:
                raise AssertionError("Invalid options accepted: " + str(options))
    check_async(left, right)
    check_measurement_view()
    print(
        "PASS: original geometry/depth/XYZ, output, input validation, bounded asynchronous depth and frame association"
    )


if __name__ == "__main__":
    main()
