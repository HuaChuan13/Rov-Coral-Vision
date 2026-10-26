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
from vision import TargetObservation, ObservationOutput, yaw_slew_preview, tracking_records


def record(x=700, cls=0, score=0.9):
    return dict(class_id=cls, confidence=score, box_rectified_xyxy=[x, 200., x + 160., 400.],
                **{"class": "Healthy_Coral"})


def selected(args):
    tracker = TargetObservation((1280, 720), args, 797.55)
    tracker.update([record()], 0, 10., 10.02)
    tracker.select([record()], (750, 300), 10., 10.03)
    return tracker


def check_pipeline(freeze=True, confidence=.6):
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
            self.calls = 0
        def predict(self, image, **options):
            assert options["conf"] == .35, "Weak detections discarded before tracking"
            self.calls += 1
            # Five warmups, then high -> low -> miss -> two high detections.
            score = confidence + .05 if freeze or self.calls != 7 else .4
            boxes = [] if not freeze and self.calls == 8 else [[700., 200., 860., 400., score, 0.]]
            return [SimpleNamespace(boxes=SimpleNamespace(data=torch.tensor(boxes)))]

    class Camera:
        meta = {}
        def __init__(self, *a, **kw):
            self.seq, self.packet = -1, None
        def latest(self):
            if freeze and self.seq == 1 and time.monotonic() - self.packet.timestamp < .3:
                return self.packet
            self.seq += 1
            image = np.zeros((720, 1280, 3), np.uint8)
            self.packet = SimpleNamespace(seq=self.seq, timestamp=time.monotonic(), left=image,
                                          right=image, capture_fps=18.)
            self.packet.received_timestamp = self.packet.timestamp
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

    encode = cv2.imencode
    def encode_after_stop(*a, **kw):
        assert published and published[-1]["status"] == "stopped", "PNG encoding blocked live observations"
        return encode(*a, **kw)
    replacements = [(ultralytics, "YOLO", Detector), (run, "Camera", Camera), (cv2, "imencode", encode_after_stop),
                    (preview_window, "PreviewWindow", Window), (cv2, "pollKey", lambda: -1),
                    (cv2, "getWindowProperty", lambda *a: 1), (ObservationOutput, "publish", publish)]
    originals = [(obj, key, getattr(obj, key)) for obj, key, _ in replacements]
    try:
        for obj, key, value in replacements:
            setattr(obj, key, value)
        with tempfile.TemporaryDirectory() as temporary, contextlib.redirect_stdout(io.StringIO()):
            assert run.main(["--frames", "3" if freeze else "5", "--device", "cpu", "--no-depth",
                             "--confidence", str(confidence),
                             "--vision-port", "0", "--output", temporary]) == 0
            folder = next(Path(temporary).iterdir())
            summary = json.loads((folder / "summary.json").read_text())
            assert summary["depth_updates"] == 0 and summary["error"] is None
            assert json.loads((folder / "settings.json").read_text())["depth_policy"] == "disabled"
        assert any(p["valid"] and p["capture_seq"] == 1 for p in published)
        if freeze:
            assert any(p["status"] == "stale" and p["capture_seq"] == 1 for p in published)
            assert all(not p["valid"] for p in published if p["capture_seq"] == 2)
        else:
            by_frame = {p["capture_seq"]: p for p in published if p["status"] != "stopped"}
            assert by_frame[1]["valid"] and by_frame[1]["target"]["confidence"] < .6
            assert by_frame[2]["status"] == by_frame[3]["status"] == "waiting"
            assert by_frame[4]["valid"] and by_frame[4]["lock_id"] == by_frame[1]["lock_id"]
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

    for detections, reason in (([], "waiting"), ([record(705), record(715)], "ambiguous"),
                               ([record(700, cls=1)], "waiting"), ([record(700, score=.2)], "waiting"),
                               ([record(float("nan"))], "waiting"), ([record(1100)], "waiting")):
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

    # Low-score detections continue an existing lock, but cannot create or recover one.
    t = selected(args)
    t.update([record(710, score=.4)], 1, 10.05, 10.08)
    low = t.packet(10.08)
    assert low["valid"] and low["lock_id"] == 1
    assert low["center_uv"] == [790, 300] and 780 < low["smoothed_center_uv"][0] < 790
    t.select([record(score=.4)], (750, 300), 10.05, 10.08)
    assert not t.packet(10.08)["valid"]
    t = selected(args)
    t.update([], 1, 10.05, 10.08)
    waiting = t.packet(10.08)
    assert waiting["status"] == "waiting" and not waiting["valid"]
    for key in ("target", "center_uv", "error_x", "smoothed_error_x", "smoothed_center_uv"):
        assert waiting[key] is None
    assert waiting["preview"]["yaw_target_slew_dps"] is None
    assert waiting["track_memory"]["display_only"] and waiting["track_memory"]["last_seen_monotonic_s"] == 10.
    t.update([record(710, score=.4)], 2, 10.1, 10.12)
    assert not t.packet(10.12)["valid"]
    t.update([record(710)], 3, 10.15, 10.18)
    assert t.packet(10.18)["track_memory"]["recovery_count"] == 1
    t.update([record(715)], 4, 10.2, 10.23)
    recovered = t.packet(10.23)
    assert recovered["valid"] and recovered["lock_id"] == waiting["lock_id"]
    assert recovered["center_uv"] == recovered["smoothed_center_uv"], "Recovery retained old filter history"
    t.cancel()
    assert t.packet(10.24)["track_memory"] is None

    # Misses / weak candidates / the first recovery match cannot extend the retention deadline.
    t = selected(args)
    for i, stamp in enumerate((10.05, 10.15, 10.25, 10.35), 1):
        t.update([] if i == 1 else [record(score=.4)], i, stamp, stamp + .01)
        assert t.packet(stamp + .01)["status"] == "waiting"
    t.update([record()], 5, 10.39, 10.395)
    assert not t.packet(10.395)["valid"]
    t.update([record()], 6, 10.41, 10.42)
    assert t.packet(10.42)["status"] == "lost"
    t.update([record()], 7, 10.46, 10.47)
    assert not t.packet(10.47)["valid"]
    t = selected(args)
    t.update([], 1, 10.05, 10.06)
    t.update([record()], 2, 10.1, 10.11)
    t.update([], 3, 10.15, 10.16)
    t.update([record()], 4, 10.2, 10.21)
    assert not t.packet(10.21)["valid"], "Recovery matches were not consecutive"
    t.update([record(), record(705)], 5, 10.25, 10.26)
    assert t.packet(10.26)["status"] == "ambiguous" and t.target is None

    from stereo import observations
    from output import detection_records
    boxes = np.array([[700, 200, 860, 400, .4, 0], [100, 200, 260, 400, .9, 0]])
    assert len(tracking_records(boxes, {0: "Coral"}, (1280, 720), .35)) == 2
    assert len(detection_records(observations(boxes, {0: "Coral"}), (1280, 720))) == 1
    # One CLI setting reaches display/logging, click selection and two-frame recovery.
    for threshold in (.5, .3):
        tuned = parse_args(["--confidence", str(threshold)])
        assert tuned.confidence == threshold and tuned.track_confidence == min(.35, threshold)
        candidates = np.array([[700, 200, 860, 400, threshold + .01, 0],
                               [100, 200, 260, 400, threshold - .01, 0]])
        found = observations(candidates, {0: "Coral"}, tuned.confidence)
        records = detection_records(found, (1280, 720), tuned.confidence)
        assert len(records) == 1 and records[0]["confidence"] < .6
        t = TargetObservation((1280, 720), tuned, 797.55)
        t.update(records, 0, 10., 10.01)
        t.select(records, (750, 300), 10., 10.02)
        assert t.packet(10.02)["valid"]
        assert t.packet(10.02)["selection_monotonic_s"] == 10.02
        t.update([], 1, 10.05, 10.06)
        t.update(records, 2, 10.1, 10.11)
        assert not t.packet(10.11)["valid"]
        t.update(records, 3, 10.15, 10.16)
        assert t.packet(10.16)["valid"]
        assert t.packet(10.16)["selection_monotonic_s"] == 10.02
    assert not tracking_records([[700, 200, 860, 400, .8, .5]], {0: "Coral"}, (1280, 720), .35)
    t = selected(parse_args(["--track-smooth-ms", "0"]))
    t.update([record(710)], 1, 10.05, 10.08)
    assert t.packet(10.08)["smoothed_center_uv"] == [790, 300]

    assert yaw_slew_preview(.02, 10, .04, 5, -1) == 0
    assert abs(yaw_slew_preview(.040001, 10, .04, 5, -1)) < .00002
    assert yaw_slew_preview(1., 10, .04, 5, -1) == -5
    assert yaw_slew_preview(-1., 10, .04, 5, -1) == 5
    assert yaw_slew_preview(.25, 10, .04, 5, 1) == 2.1
    for options in (["--yaw-gain", "nan"], ["--yaw-max-slew", "-1"], ["--yaw-deadband", "1"],
                    ["--confidence", "0"], ["--confidence", "60"], ["--confidence", "nan"],
                    ["--confidence", ".3", "--track-confidence", ".35"],
                    ["--vision-max-age", "0"], ["--track-min-iou", "0"],
                    ["--vision-port", "65536"], ["--yaw-reference", "2"],
                    ["--track-confidence", ".7"], ["--track-confidence", "0"],
                    ["--track-hold", "0"], ["--track-hold", "nan"], ["--track-smooth-ms", "-1"]):
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
            data, source = receiver.recvfrom(8192)
            received = json.loads(data)
            assert received == packet and out.errors == 0
            feedback = dict(schema="coral.control.v1", session=packet["session"], lock_id=packet["lock_id"],
                            status="ACTIVE", confirmed=True, requested_slew_dps=-2.1, accepted_slew_dps=-2.1,
                            result_monotonic_s=20., valid_until_monotonic_s=20.2)
            receiver.sendto(json.dumps(feedback).encode(), source)
            assert out.control_status(packet, 20.01)["confirmed"]
            for bad in (dict(feedback, lock_id=packet["lock_id"] + 1),
                        dict(feedback, session="b" * 32),
                        dict(feedback, accepted_slew_dps=float("nan")),
                        dict(feedback, accepted_slew_dps=10 ** 400)):
                bad["result_monotonic_s"] = 20.02
                receiver.sendto(json.dumps(bad).encode(), source)
                assert out.control_status(packet, 20.03)["result_monotonic_s"] == 20.
            assert not out.control_status(dict(packet, valid=False, status="cancelled"), 20.04)["confirmed"]
            assert out.control_status(packet, 20.21) is None
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
        assert any("CONTROL UNCONFIRMED" in s for s in labels)
        assert any("SENT: --" in s and "TEENSY: --" in s for s in labels)
        labels.clear()
        compose_view(image, [], geometry, args, 18, False, observation=stale)
        assert any("SENT: --" in s for s in labels)
        assert not any("smooth X +" in s for s in labels)
        labels.clear()
        rendered = compose_view(image, [], geometry, args, 18, False, observation=waiting)
        assert any("WAITING" in s for s in labels) and any("last position" in s for s in labels)
        assert any("SENT: --" in s for s in labels)
        assert np.any(np.all(rendered == (0, 165, 255), axis=2)), "Missing orange waiting outline"
        labels.clear()
        feedback = dict(status="ACTIVE", confirmed=True, mode="STABILIZE", requested_slew_dps=2.6,
                        accepted_slew_dps=2.5, yaw_target_deg=-176., yaw_actual_deg=169., lead_limit_deg=15.)
        compose_view(image, [], geometry, args, 18, False, observation=packet, control=feedback)
        assert any("SENT: +2.60" in s and "TEENSY: +2.50" in s for s in labels)
        assert any("LEAD LIMIT 15" in s for s in labels), "Wrapped heading limit missing"
    finally:
        output.text = original
    check_pipeline()
    check_pipeline(freeze=False)
    check_pipeline(freeze=False, confidence=.5)
    print("PASS: selection, conservative association, stale/reordered frames, slew preview, click mapping, UDP and UI")


if __name__ == "__main__":
    main()
