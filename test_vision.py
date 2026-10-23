"""Run with uv run --locked python test_vision.py; no hardware or serial access."""

import contextlib
import io
import json
import socket
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from output import compose_view
from preview_window import canvas_point
from run import parse_args
from vision import TargetObservation, ObservationOutput, yaw_slew_preview


def record(x=700, cls=0, score=0.9):
    return dict(class_id=cls, confidence=score, box_rectified_xyxy=[x, 200., x + 160., 400.],
                **{"class": "Healthy_Coral"})


def selected(args):
    tracker = TargetObservation((1280, 720), args, 797.55)
    tracker.update([record()], 0, 10., 10.02)
    tracker.select([record()], (750, 300), 10., 10.03)
    return tracker


def check_pipeline():
    """Real main loop with simulated camera/window: freeze cannot keep a valid lock."""
    import cv2
    import torch
    import ultralytics
    import preview_window
    import run

    published = []
    original_publish = ObservationOutput.publish
    def publish(self, packet):
        published.append(packet)
        original_publish(self, packet)

    class Detector:
        names = {0: "Healthy_Coral"}
        def __init__(self, model):
            pass
        def predict(self, image, **options):
            return [SimpleNamespace(boxes=SimpleNamespace(data=torch.tensor([[700., 200., 860., 400., .9, 0.]])))]

    class Camera:
        meta = {}
        def __init__(self, *a, **kw):
            self.seq, self.packet = -1, None
        def latest(self):
            if self.seq == 1 and time.monotonic() - self.packet.timestamp < .3:
                return self.packet
            self.seq += 1
            image = np.zeros((720, 1280, 3), np.uint8)
            self.packet = SimpleNamespace(seq=self.seq, timestamp=time.monotonic(), left=image,
                                          right=image, capture_fps=18.)
            return self.packet
        def close(self):
            pass

    class Window:
        def __init__(self, *a):
            self.shown, self.clicked = False, False
        def show(self, image):
            assert published, "Drawing happened before publishing"
            assert image.shape == (928, 1280, 3)
            self.shown = True
        def pop_click(self):
            if self.shown and not self.clicked:
                self.clicked = True
                return ("select", (750, 348))
        def close(self):
            pass

    replacements = [(ultralytics, "YOLO", Detector), (run, "Camera", Camera),
                    (preview_window, "PreviewWindow", Window), (cv2, "pollKey", lambda: -1),
                    (cv2, "getWindowProperty", lambda *a: 1), (ObservationOutput, "publish", publish)]
    originals = [(obj, key, getattr(obj, key)) for obj, key, _ in replacements]
    try:
        for obj, key, value in replacements:
            setattr(obj, key, value)
        with tempfile.TemporaryDirectory() as temporary, contextlib.redirect_stdout(io.StringIO()):
            assert run.main(["--frames", "3", "--device", "cpu", "--no-depth",
                             "--vision-port", "0", "--output", temporary]) == 0
            folder = next(Path(temporary).iterdir())
            summary = json.loads((folder / "summary.json").read_text())
            assert summary["depth_updates"] == 0 and summary["error"] is None
            assert json.loads((folder / "settings.json").read_text())["depth_policy"] == "disabled"
        assert any(p["valid"] and p["capture_seq"] == 1 for p in published)
        assert any(p["status"] == "stale" and p["capture_seq"] == 1 for p in published)
        assert all(not p["valid"] for p in published if p["capture_seq"] == 2)
        assert published[-1]["status"] == "stopped"
    finally:
        for obj, key, value in originals:
            setattr(obj, key, value)


def main():
    args = parse_args([])
    tracker = selected(args)
    tracker.update([record(710), record(100)], 1, 10.05, 10.08)
    packet = tracker.packet(10.08)
    assert packet["valid"] and packet["lock_id"] == 1
    assert packet["center_uv"] == [790, 300]
    assert packet["error_x"] == (790 - 640) / 640
    assert packet["principal_u"] != packet["reference_u"]
    assert packet["preview"]["yaw_target_slew_dps"] < 0
    assert packet["preview"]["sent_to_teensy"] is False
    assert "XYZ_rectified_left_m" not in packet["target"]
    original_stamp = packet["capture_monotonic_s"]
    assert tracker.expire(10.3)
    stale = tracker.packet(10.3)
    assert not stale["valid"] and stale["preview"]["yaw_target_slew_dps"] is None
    assert stale["capture_monotonic_s"] == original_stamp and stale["capture_seq"] == 1
    tracker.update([record(710)], 2, 10.31, 10.32)
    assert not tracker.packet(10.32)["valid"], "Stale lock automatically resumed"
    tracker.select([record(710)], (750, 300), 10.31, 10.32)
    assert tracker.packet(10.32)["lock_id"] == 2
    tracker.cancel()
    assert not tracker.packet(10.33)["valid"]

    for detections, reason in (([], "lost"), ([record(705), record(715)], "ambiguous"),
                               ([record(700, cls=1)], "lost"), ([record(700, score=.4)], "lost"),
                               ([record(float("nan"))], "lost"), ([record(1100)], "lost")):
        t = selected(args)
        t.update(detections, 1, 10.05, 10.08)
        p = t.packet(10.08)
        assert not p["valid"] and p["status"] == reason
        t.update([record()], 2, 10.1, 10.12)
        assert not t.packet(10.12)["valid"], "Lost/ambiguous target was silently reacquired"
    for seq, stamp, now in ((0, 10., 10.08), (-1, 9.9, 10.08), (1, 9.99, 10.08),
                            (1, 10.1, 10.4), (1, 10.4, 10.3), (1, 10.3, 10.31)):
        t = selected(args)
        t.update([record()], seq, stamp, now)
        assert not t.packet(now)["valid"]
    t = selected(args)
    t.update([record()], 1, 10., 10.08)  # Windows monotonic clock may give equal timestamps to distinct frames.
    assert t.packet(10.08)["valid"]
    t = selected(args)
    t.select([record(), record(710)], (750, 300), 10., 10.03)
    assert t.status == "ambiguous" and t.target is None
    t.select([record()], (750, 300), 9., 10.03)
    assert t.status == "stale"
    assert selected(args).session != selected(args).session

    assert yaw_slew_preview(.02, 10, .04, 5, -1) == 0
    assert abs(yaw_slew_preview(.040001, 10, .04, 5, -1)) < .00002
    assert yaw_slew_preview(1., 10, .04, 5, -1) == -5
    assert yaw_slew_preview(-1., 10, .04, 5, -1) == 5
    assert yaw_slew_preview(.25, 10, .04, 5, 1) == 2.1
    for options in (["--yaw-gain", "nan"], ["--yaw-max-slew", "-1"], ["--yaw-deadband", "1"],
                    ["--vision-max-age", "0"], ["--track-min-iou", "0"],
                    ["--vision-port", "65536"], ["--yaw-reference", "2"]):
        with contextlib.redirect_stderr(io.StringIO()):
            try:
                parse_args(options)
            except SystemExit as exc:
                assert exc.code == 2
            else:
                raise AssertionError(options)

    # Both layouts, resized or letterboxed: header, footer and right panel never select a left target.
    for size in ((1280, 928), (2688, 928)):
        for scale, offset in ((1., (0, 0)), (.5, (90, 40)), (1.5, (0, 170))):
            rect = (*offset, size[0] * scale, size[1] * scale)
            p = canvas_point(offset[0] + 750 * scale, offset[1] + 348 * scale, rect, size)
            assert np.allclose(p, (750, 348))
            assert canvas_point(offset[0] - 1, offset[1], rect, size) is None
            assert canvas_point(offset[0] + rect[2], offset[1], rect, size) is None

    with tempfile.TemporaryDirectory() as temporary, socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as receiver:
        receiver.bind(("127.0.0.1", 0))
        receiver.settimeout(1)
        out = ObservationOutput(Path(temporary), receiver.getsockname()[1])
        try:
            out.publish(packet)
            received = json.loads(receiver.recv(8192))
            assert received == packet and out.errors == 0
            out.publish(stale)
            received = json.loads(receiver.recv(8192))
            assert not received["valid"] and received["capture_monotonic_s"] == original_stamp
        finally:
            out.close()
        logs = [json.loads(s) for s in (Path(temporary) / "vision_observations.jsonl").read_text().splitlines()]
        assert logs == [packet, stale]

    import output
    labels, original = [], output.text
    def capture(image, label, *values, **kwargs):
        labels.append(label)
        return original(image, label, *values, **kwargs)
    try:
        output.text = capture
        geometry = SimpleNamespace(color_size=(1280, 720), verified=False)
        image = np.zeros((720, 1280, 3), np.uint8)
        compose_view(image, [], geometry, args, 18, False, observation=packet)
        assert any("Yaw target slew:" in s and "NOT SENT TO TEENSY" in s for s in labels)
        labels.clear()
        compose_view(image, [], geometry, args, 18, False, observation=stale)
        assert any("-- (INVALID)" in s for s in labels)
        assert not any("error X +" in s for s in labels)
    finally:
        output.text = original
    check_pipeline()
    print("PASS: selection, conservative association, stale/reordered frames, slew preview, click mapping, UDP and UI")


if __name__ == "__main__":
    main()
