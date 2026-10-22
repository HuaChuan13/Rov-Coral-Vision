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
    cutoff = max(0.6, float(threshold))
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


def compose_view(
    color, detections, geometry, args, fps, recording, depth=None, depth_fps=0.0, depth_age_ms=None
):
    w, h = geometry.color_size
    if color.shape != (h, w, 3):
        raise ValueError("Visualization must match the calibrated sensor grid.")
    detections = visible_detections(detections, args.confidence)
    view = draw_annotations(color.copy(), detections) if detections else color
    stale = depth_age_ms is not None and depth_age_ms > args.depth_max_age * 1000
    if stale:
        depth = None
    # Pure depth colors only. Completed display pixels NEVER enter measurements.
    colors = depth["colors"] if depth is not None else np.full_like(color, 25)
    canvas = np.full((h + 120, 2 * w + 128, 3), 25, np.uint8)
    canvas[48 : 48 + h, :w] = view
    canvas[48 : 48 + h, w : 2 * w] = colors

    def fitted(label, xy, max_width, color=(235, 235, 235), scale=0.8):
        pixels = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)[0][0]
        text(canvas, label, xy, color, min(scale, scale * max_width / max(1, pixels)))

    accent = (70, 220, 240)
    prefix = "DETECTION" if args.live else "OFFLINE DETECTION"
    fitted("%s  %.1f FPS" % (prefix, fps), (16, 32), w - 32)
    state = "STALE" if stale else "WAIT" if depth is None else "sample age %.0f ms" % depth_age_ms
    fitted(
        "DEPTH  %.1f FPS | %s" % (depth_fps if depth is not None else 0.0, state),
        (w + 16, 32), w - 32,
    )
    text(
        canvas,
        "REC" if recording else "REC OFF",
        (canvas.shape[1] - 120, 32),
        (30, 30, 255) if recording else (150, 150, 150),
        0.7,
    )
    legend_x = 2 * w + 12
    text(canvas, "Z (cm)", (legend_x, 82), scale=0.65)
    bar_height = h - 130
    gradient = np.rint(np.linspace(255, 10, bar_height)).astype(np.uint8).reshape(-1, 1)
    bar = cv2.applyColorMap(np.repeat(gradient, 28, axis=1), cv2.COLORMAP_TURBO)
    canvas[132 : 132 + bar_height, legend_x : legend_x + 28] = bar
    for fraction in np.linspace(0, 1, 6):
        z = 1 / (1 / args.near - fraction * (1 / args.near - 1 / args.far))
        yy = 132 + int(fraction * (bar_height - 1))
        text(canvas, "%.0f" % (100 * z), (legend_x + 36, yy + 7), scale=0.65)
    canvas[h + 33 : h + 57, legend_x : legend_x + 28] = cv2.applyColorMap(
        np.zeros((24, 28), np.uint8), cv2.COLORMAP_TURBO
    )
    text(canvas, "N/A", (legend_x + 36, h + 53), scale=0.65)
    # D labels are local to this depth snapshot, NOT persistent tracking IDs.
    objects = depth["objects"] if depth is not None else []
    for index, record in enumerate(objects[:2]):
        x1, y1, x2, y2 = np.rint(record["box_rectified_xyxy"]).astype(int)
        x1, x2 = np.clip([x1, x2], 0, w - 1)
        y1, y2 = np.clip([y1, y2], 0, h - 1)
        cv2.rectangle(canvas, (w + x1, 48 + y1), (w + x2, 48 + y2), accent, 2, cv2.LINE_AA)
        tag = "D%d" % (index + 1)
        tx, ty = w + min(x1 + 6, w - 64), 48 + max(24, y1 - 6)
        cv2.rectangle(canvas, (tx - 3, ty - 22), (tx + 48, ty + 5), (12, 12, 12), -1)
        text(canvas, tag, (tx, ty), accent, 0.75)
        xyz = record["XYZ_rectified_left_m"]
        values = "X: --    Y: --    Z: --  (no valid depth)"
        if xyz is not None and np.isfinite(xyz).all():
            values = "X: %+.1f    Y: %+.1f    Z: %.1f cm" % tuple(100 * v for v in xyz)
        fitted(
            "%s  %s  |  %s" % (tag, record["class"], values),
            (index * w + 16, h + 80), w - 32, accent, 0.9,
        )
    if not objects:
        message = (
            "XYZ: -- | Depth expired" if stale else "XYZ: -- | Waiting for depth"
            if depth is None else "XYZ: -- | No target in depth frame"
        )
        fitted(message, (16, h + 80), 2 * w - 32, accent)
    calibration = (
        "Calibration OK"
        if geometry.verified and getattr(geometry, "quality_passed", None) is not False
        else "Calibration unverified"
    )
    note = "D1/D2: depth-frame targets | X right, Y down, Z forward"
    if len(objects) > 2:
        note += " | +%d targets in log" % (len(objects) - 2)
    fitted(
        "%s | %s | Q: record / W, Esc: exit" % (calibration, note),
        (16, h + 108),
        canvas.shape[1] - 30,
        color=(170, 170, 170), scale=0.6,
    )
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
