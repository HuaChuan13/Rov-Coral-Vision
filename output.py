"""Drawing, structured observations and bounded video recording."""
import json
import math
import queue
import threading
import time
from pathlib import Path

import cv2
import numpy as np
from stereo import depth_colors


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    tmp.replace(path)

def read_image(path):
    image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError('Cannot decode image: ' + str(path))
    return image

def host_array(value):
    return value.detach().cpu().numpy() if hasattr(value, 'detach') else value

def text(image, label, xy, color=(235, 235, 235), scale=.48):
    cv2.putText(image, label, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)

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

def detection_records(detections, size, confidence=.6):
    return [{'class_id': int(cls), 'class': name, 'confidence': float(score),
             'image_size': list(size), 'status': status, 'independent_samples': int(count),
             'box_rectified_xyxy': [float(v) for v in box], 'coordinate_frame': 'rectified_left',
             'XYZ_rectified_left_m': None if pos is None else pos.tolist()}
            for box, cls, score, name, pos, count, status in visible_detections(detections, confidence)]

def compose_view(color, estimated, detections, geometry, args, fps, capture_fps, age_ms, recording, rendered=None):
    w, h = geometry.color_size
    if color.shape != (h, w, 3) or estimated.shape != (h, w):
        raise ValueError('Visualization must match the calibrated sensor grid.')
    detections = visible_detections(detections, args.confidence)
    view = draw_annotations(color.copy(), detections) if detections else color
    # Pure depth colors only. Completed display pixels NEVER enter measurements.
    colors = depth_colors(estimated, args.near, args.far) if rendered is None else rendered.copy()
    canvas = np.full((h+120, 2*w+128, 3), 25, np.uint8)
    canvas[72:72+h, :w] = view
    canvas[72:72+h, w:2*w] = colors
    def fitted(label, xy, max_width, color=(235, 235, 235), scale=.8):
        pixels = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)[0][0]
        text(canvas, label, xy, color, min(scale, scale*max_width/max(1, pixels)))
    prefix = 'LEFT + YOLO'
    if getattr(args, 'input_enhancement', 'off') != 'off':
        prefix += ' + ENH'
    if not args.live:
        prefix = 'OFFLINE ' + prefix
    fitted('%s %dx%d | fresh %.1f FPS | capture %.1f FPS' % (prefix, w, h, fps, capture_fps), (15, 28), w-30)
    method = 'native' if geometry.stereo_size == geometry.color_size else 'guided reconstruction'
    if args.invalid_display == 'complete':
        method = 'COMPLETED / display estimate'
    depth_prefix = 'DEPTH'
    fitted('%s %dx%d | %s | SGBM %dx%d' % (depth_prefix, w, h, method, *geometry.stereo_size), (w+15, 28), w-30)
    verified = 'RANGE VALIDATED' if geometry.verified else 'CALIBRATION UNVERIFIED: estimated XYZ, check known distances'
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
    view_note = 'Rectified left XYZ (cm): +X right, +Y down, +Z forward'
    fitted('Q: record | W: save & exit | '+unavailable+' | '+view_note,
           (15, h+98), canvas.shape[1]-30, scale=.75)
    return canvas

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
