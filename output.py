"""Drawing, structured observations and bounded video recording."""

import json
import math
import queue
import threading
import time
from pathlib import Path

import cv2
import numpy as np


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    tmp.replace(path)


def read_image(path):
    image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Cannot decode image: " + str(path))
    return image


def host_array(value):
    return value.detach().cpu().numpy() if hasattr(value, "detach") else value


def text(image, label, xy, color=(235, 235, 235), scale=0.48):
    cv2.putText(image, label, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def visible_detections(detections, threshold=0.6):
    """Display/log safety gate, independent of YOLO's own confidence filter."""
    cutoff = float(threshold)
    return [d for d in detections if np.isfinite(d[2]) and float(d[2]) >= cutoff]


def draw_annotations(image, detections):
    """Current detections only; measurements belong to the depth source frame."""
    palette = ((70, 255, 80), (255, 90, 235), (255, 220, 30))  # BGR
    occupied = []
    h, w = image.shape[:2]
    font, scale, thickness = cv2.FONT_HERSHEY_SIMPLEX, 0.70, 2
    ordered = sorted(detections, key=lambda d: float(d[2]), reverse=True)
    for box, cls, score, name, pos, count, status in ordered:
        x1, y1, x2, y2 = np.rint(box).astype(int)
        x1, x2 = np.clip([x1, x2], 0, w - 1)
        y1, y2 = np.clip([y1, y2], 0, h - 1)
        accent = palette[int(cls) % len(palette)]
        cv2.rectangle(image, (x1, y1), (x2, y2), (10, 10, 10), 5, cv2.LINE_AA)
        cv2.rectangle(image, (x1, y1), (x2, y2), accent, 3, cv2.LINE_AA)
        center = ((x1 + x2) // 2, (y1 + y2) // 2)
        cv2.circle(image, center, 4, (10, 10, 10), -1, cv2.LINE_AA)
        cv2.circle(image, center, 2, (255, 255, 255), -1, cv2.LINE_AA)
    for box, cls, score, name, pos, count, status in ordered:
        x1, y1, x2, y2 = np.rint(box).astype(int)
        accent = palette[int(cls) % len(palette)]
        label = "%s  %.1f%%" % (name, 100 * score)
        cw = min(w - 8, cv2.getTextSize(label, font, scale, thickness)[0][0] + 16)
        ch = 34
        candidates = [
            (x1, y1 - ch - 5),
            (x1, y1 + 6),
            (x1, y2 - ch - 6),
            (x2 - cw, y1 - ch - 5),
            (x2 - cw, y2 - ch - 6),
            (x1, y2 + 6),
        ]
        rectangles = []
        for x, y in candidates:
            x, y = int(np.clip(x, 4, w - cw - 4)), int(np.clip(y, 4, h - ch - 4))
            rectangles.append((x, y, x + cw, y + ch))

        def overlap(rect):
            return sum(
                max(0, min(rect[2], r[2]) - max(rect[0], r[0]))
                * max(0, min(rect[3], r[3]) - max(rect[1], r[1]))
                for r in occupied
            )

        rect = min(rectangles, key=overlap)
        occupied.append(rect)
        x, y, _, _ = rect
        cv2.rectangle(image, (rect[0], rect[1]), (rect[2], rect[3]), (12, 12, 12), -1)
        cv2.rectangle(image, (rect[0], rect[1]), (rect[2], rect[3]), accent, 1, cv2.LINE_AA)
        cv2.putText(image, label, (x + 8, y + 23), font, scale, accent, thickness, cv2.LINE_AA)
    return image


def detection_records(detections, size, confidence=0.6):
    return [
        {
            "class_id": int(cls),
            "class": name,
            "confidence": float(score),
            "image_size": list(size),
            "status": status,
            "independent_samples": int(count),
            "box_rectified_xyxy": [float(v) for v in box],
            "coordinate_frame": "rectified_left",
            "XYZ_rectified_left_m": None if pos is None else pos.tolist(),
        }
        for box, cls, score, name, pos, count, status in visible_detections(detections, confidence)
    ]


def view_size(geometry, args):
    w, h = geometry.color_size
    return (2 * w + 128 if args.show_depth else w, h + 208)


def compose_view(
    color, detections, geometry, args, fps, recording, depth=None, depth_fps=0.0,
    depth_age_ms=None, detection_age_ms=None, observation=None, observation_output="UDP OFF",
):
    w, h = geometry.color_size
    if color.shape != (h, w, 3):
        raise ValueError("Visualization must match the calibrated sensor grid.")
    detections = visible_detections(detections, args.confidence)
    valid = observation is not None and observation["valid"]
    if valid:
        target = observation["target"]
        # Draw the selected target once, using its smoothed yellow box below.
        detections = [d for d in detections if not (int(d[1]) == target["class_id"] and np.allclose(
            np.clip(d[0], [0, 0, 0, 0], [w, h, w, h]), target["box_rectified_xyxy"], atol=1))]
    view = draw_annotations(color.copy(), detections) if detections else color
    stale = depth_age_ms is not None and depth_age_ms > args.depth_max_age * 1000
    if stale:
        depth = None
    width, height = view_size(geometry, args)
    canvas = np.full((height, width, 3), 25, np.uint8)
    canvas[48 : 48 + h, :w] = view

    def fitted(label, xy, max_width, color=(235, 235, 235), scale=0.8):
        pixels = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)[0][0]
        text(canvas, label, xy, color, min(scale, scale * max_width / max(1, pixels)))

    accent = (70, 220, 240)
    age = "--" if detection_age_ms is None else "%.0f" % detection_age_ms
    prefix = "LEFT" if args.live else "OFFLINE LEFT"
    fitted("%s | %.1f FPS | frame age %s ms" % (prefix, fps, age), (16, 32), w - 160)
    text(canvas, "REC" if recording else "REC OFF", (width - 120, 32),
         (30, 30, 255) if recording else (150, 150, 150), 0.7)

    state = observation["status"].upper() if observation else "IDLE"
    lock = observation["lock_id"] if observation else 0
    memory = observation.get("track_memory") if observation else None
    if memory and state == "WAITING":
        state += " %.0fms" % memory["age_ms"]
    error = "%+.1f%% half-width" % (100 * observation["smoothed_error_x"]) if valid else "--"
    slew = "%+.2f deg/s" % observation["preview"]["yaw_target_slew_dps"] if valid else "-- (INVALID)"
    fitted("TRACK %s | ID %d | smooth X %s | %s" % (state, lock, error, observation_output),
           (16, h + 72), width - 32, accent, 0.65)
    fitted("Yaw target slew: %s | PREVIEW ONLY - NOT SENT TO TEENSY" % slew,
           (16, h + 98), width - 32, accent, 0.7)
    if observation is not None:
        reference = observation["reference_u"]
        half_band = args.yaw_deadband * w / 2
        for x in (reference - half_band, reference, reference + half_band):
            if 0 <= x < w:
                for y in range(48, 48 + h, 24):
                    cv2.line(canvas, (round(x), y), (round(x), min(48 + h - 1, y + 10)), (180, 180, 180), 1)
        if memory:
            x1, y1, x2, y2 = np.rint(memory["box_xyxy"]).astype(int)
            x1, x2 = np.clip([x1, x2], 0, w - 1)
            y1, y2 = np.clip([y1, y2], 0, h - 1) + 48
            tint = accent if valid else (0, 165, 255)
            if valid:
                cv2.rectangle(canvas, (x1, y1), (x2, y2), tint, 3)
                label = "LOCK %d | %s %.1f%%" % (lock, target["class"], 100 * target["confidence"])
            else:
                for x in range(x1, x2, 18):
                    for y in (y1, y2):
                        cv2.line(canvas, (x, y), (min(x + 9, x2), y), tint, 2)
                for y in range(y1, y2, 18):
                    for x in (x1, x2):
                        cv2.line(canvas, (x, y), (x, min(y + 9, y2)), tint, 2)
                label = "WAIT ID %d | last position | resume %d/2" % (lock, memory["recovery_count"])
            label_y = max(72, y1 - 8)
            pixels = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, .6, 1)[0][0]
            label_x = max(4, min(x1, w - pixels - 8))
            cv2.rectangle(canvas, (label_x - 3, label_y - 21), (min(w - 1, label_x + pixels + 4), label_y + 5), (12, 12, 12), -1)
            text(canvas, label, (label_x, label_y), tint, .6)

    if args.show_depth:
        # Display completion never supplies numeric XYZ measurements.
        canvas[48 : 48 + h, w : 2 * w] = depth["colors"] if depth is not None else 25
        fitted("DEPTH DEBUG | %.1f FPS" % (depth_fps if depth is not None else 0),
               (w + 16, 32), w - 32)
        legend_x, bar_height = 2 * w + 12, h - 130
        text(canvas, "Z (cm)", (legend_x, 82), scale=0.65)
        gradient = np.rint(np.linspace(255, 10, bar_height)).astype(np.uint8).reshape(-1, 1)
        canvas[132 : 132 + bar_height, legend_x : legend_x + 28] = cv2.applyColorMap(
            np.repeat(gradient, 28, axis=1), cv2.COLORMAP_TURBO)
        for fraction in np.linspace(0, 1, 6):
            z = 1 / (1 / args.near - fraction * (1 / args.near - 1 / args.far))
            yy = 132 + int(fraction * (bar_height - 1))
            text(canvas, "%.0f" % (100 * z), (legend_x + 36, yy + 7), scale=0.65)
        canvas[h + 33 : h + 57, legend_x : legend_x + 28] = cv2.applyColorMap(
            np.zeros((24, 28), np.uint8), cv2.COLORMAP_TURBO)
        text(canvas, "N/A", (legend_x + 36, h + 53), scale=0.65)

    # Depth snapshots are independent of the current selected target.
    objects = depth["objects"] if depth is not None else []
    state = "OFF" if not args.depth else "STALE" if stale else "WAIT" if depth is None else "frame %s | age %.0f ms" % (
        depth["seq"], depth_age_ms)
    fitted("DEPTH SNAPSHOT | %s | historical measurements, not current boxes" % state,
           (16, h + 118), width - 32, (170, 170, 170), 0.6)
    for index, record in enumerate(objects[:2]):
        x1, y1, x2, y2 = np.rint(record["box_rectified_xyxy"]).astype(int)
        x1, x2 = np.clip([x1, x2], 0, w - 1)
        y1, y2 = np.clip([y1, y2], 0, h - 1)
        tag = "D%d" % (index + 1)
        if args.show_depth:
            cv2.rectangle(canvas, (w + x1, 48 + y1), (w + x2, 48 + y2), accent, 2, cv2.LINE_AA)
            tx, ty = w + min(x1 + 6, w - 64), 48 + max(24, y1 - 6)
            cv2.rectangle(canvas, (tx - 3, ty - 22), (tx + 48, ty + 5), (12, 12, 12), -1)
            text(canvas, tag, (tx, ty), accent, 0.75)
        xyz = record["XYZ_rectified_left_m"]
        values = "X: --    Y: --    Z: --  (no valid depth)"
        if xyz is not None and np.isfinite(xyz).all():
            values = "X: %+.1f    Y: %+.1f    Z: %.1f cm" % tuple(100 * v for v in xyz)
        xy = (index * w + 16, h + 144) if args.show_depth else (16, h + 144 + index * 26)
        fitted("%s  %s  |  %s" % (tag, record["class"], values), xy, w - 32, accent, 0.7)
    if not objects:
        message = (
            "XYZ: -- | Depth disabled" if not args.depth else "XYZ: -- | Depth expired" if stale else "XYZ: -- | Waiting for depth"
            if depth is None else "XYZ: -- | No target in depth frame"
        )
        fitted(message, (16, h + 144), width - 32, accent, 0.7)
    calibration = (
        "Calibration OK"
        if geometry.verified and getattr(geometry, "quality_passed", None) is not False
        else "Calibration unverified"
    )
    note = "X right / Y down / Z forward"
    if len(objects) > 2:
        note += " | +%d targets in log" % (len(objects) - 2)
    fitted("%s | %s | Click select / C cancel / Q record / Esc exit" % (calibration, note),
           (16, h + 196), width - 32, (170, 170, 170), 0.55)
    return canvas


class VideoRecorder:
    """Bounded asynchronous writer. CFR repeats preserve wall-time, not fresh FPS.

    Queue overload drops recording updates ONLY; never freezes camera/inference.
    Real acquisition frame rate is recorded independently in frame_metrics.csv.
    """

    def __init__(self, folder, fps=30.0):
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
        for suffix, codec in ((".mp4", "mp4v"), (".avi", "MJPG")):
            self.path = self.folder / ("visualization" + suffix)
            self.writer = cv2.VideoWriter(str(self.path), cv2.VideoWriter_fourcc(*codec), self.fps, self.size)
            if self.writer.isOpened():
                self.codec = codec
                break
            self.writer.release()
        else:
            raise RuntimeError("No working MP4/AVI video encoder; check output folder.")
        self.queue = queue.Queue(maxsize=12)
        self.active = True
        self.thread = threading.Thread(target=self._write, name="video-writer", daemon=True)
        self.thread.start()
        self.submit(image, stamp)

    def submit(self, image, stamp):
        if self.error:
            raise RuntimeError("Recording failed: " + self.error)
        if not self.active:
            return
        if image.shape[:2][::-1] != self.size:
            raise ValueError("Recording canvas dimensions changed.")
        try:
            self.queue.put_nowait((float(stamp), image.copy()))
        except queue.Full:
            self.dropped += 1

    def _write(self):
        last, next_index = None, 0
        try:
            while True:
                stamp, image = self.queue.get()
                target = max(0, int(math.ceil((stamp - self.start_time) * self.fps - 1e-8)))
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
                self.queue.put((float(stamp), None), timeout=0.1)
                break
            except queue.Full:
                continue
        self.thread.join()
        save_json(
            self.folder / "recording.json",
            {
                "video": str(self.path),
                "codec": self.codec,
                "size": self.size,
                "fps_container": self.fps,
                "wall_duration_s": stamp - self.start_time,
                "encoded_frames": self.written,
                "encoded_duration_s": self.written / self.fps,
                "extra_held_frames": self.repeated,
                "queue_dropped_updates": self.dropped,
                "writer_error": self.error,
                "timing_policy": "constant-rate encoding with previous-frame hold; NOT proof of fresh camera/inference FPS",
            },
        )
        print("Saved visualization:", self.path, flush=True)
