"""One USB/RTP reader process and one atomic latest LEFT|RIGHT frame pair."""

import ctypes
import multiprocessing as mp
import os
import queue
import time
from collections import deque
from dataclasses import dataclass

import cv2
import numpy as np


class FrameRateTracker:
    """Measure decoded camera arrivals, independently of preview/detection FPS.

    A host arrival timestamp does not establish hardware exposure synchronization.
    The camera reader supplies every frame, including ones the UI never displays.
    """

    def __init__(self, window_seconds=3.0):
        self.window_seconds = float(window_seconds)
        self.times = deque()

    def add(self, timestamp):
        timestamp = float(timestamp)
        if self.times and timestamp <= self.times[-1]:
            return self.snapshot(timestamp)
        self.times.append(timestamp)
        # Keep one sample just before the requested window boundary.
        while len(self.times) > 2 and self.times[1] < timestamp - self.window_seconds:
            self.times.popleft()
        return self.snapshot(timestamp)

    def snapshot(self, now):
        times = np.asarray(self.times, float)
        span = float(times[-1] - times[0]) if len(times) > 1 else 0.0
        intervals = np.diff(times)
        return {
            "measured_arrival_fps": float((len(times) - 1) / span) if span > 0 else 0.0,
            "window_span_s": span,
            "frame_count": len(times),
            "warmed_up": span >= self.window_seconds - 1e-6,
            "max_gap_s": float(intervals.max()) if len(intervals) else None,
            "latest_age_s": max(0.0, float(now - times[-1])) if len(times) else None,
            "measurement": "decoded_frame_arrivals_not_driver_report_or_gui_refresh",
        }


@dataclass
class Packet:
    seq: int
    timestamp: float
    left: np.ndarray
    right: np.ndarray
    capture_fps: float
    max_gap_s: float
    received_timestamp: float = 0.0
    stream_pts: float = float("nan")


def _network_worker(pixels, state, lock, stop, messages, options, shape):
    from network_video import network_frames

    buffer = np.frombuffer(pixels, np.uint8).reshape(shape)
    rate = FrameRateTracker()
    frames = network_frames(options, stop)
    try:
        for seq, (frame, stamp, received, pts) in enumerate(frames):
            health = rate.add(received)
            with lock:
                buffer[0] = frame[:, :shape[2]]
                buffer[1] = frame[:, shape[2]:]
                state[:] = (seq, stamp, health["measured_arrival_fps"], health["max_gap_s"] or 0., received, pts)
            if seq == 0:
                messages.put(("ready", dict(
                    source="rtp", host=options["stream_host"], port=options["stream_port"],
                    layout="horizontal_sbs", swap=False, eye_size=[shape[2], shape[1]],
                    actual_full_size=[frame.shape[1], frame.shape[0]], backend=options["stream_backend"] + " RTP/H264",
                    acquisition="one_process_atomic_latest_pair",
                    timing="PTS mapped to earliest decoded arrival; detects delay growth, NOT exposure age",
                )))
            if stop.is_set():
                break
    finally:
        frames.close()


def _capture_worker(pixels, state, lock, stop, messages, options, shape):
    cap, timer = None, False
    try:
        timer = hasattr(ctypes, "WinDLL") and ctypes.WinDLL("winmm").timeBeginPeriod(1) == 0
        if options["stream_host"]:
            _network_worker(pixels, state, lock, stop, messages, options, shape)
            return
        cv2.setNumThreads(1)
        backend = {"dshow": cv2.CAP_DSHOW, "msmf": cv2.CAP_MSMF, "any": cv2.CAP_ANY}[options["backend"]]
        cap = cv2.VideoCapture(options["device"], backend if os.name == "nt" else cv2.CAP_ANY)
        if not cap.isOpened():
            raise RuntimeError("Cannot open USB camera %s" % options["device"])
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        cap.set(cv2.CAP_PROP_FOURCC, fourcc)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, options["width"])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, options["height"])
        cap.set(cv2.CAP_PROP_FPS, options["fps"])
        # DirectShow size/FPS negotiation can reset MJPG; apply it last.
        cap.set(cv2.CAP_PROP_FOURCC, fourcc)
        if options["exposure"] is not None:
            cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
            cap.set(cv2.CAP_PROP_EXPOSURE, options["exposure"])
        buffer = np.frombuffer(pixels, np.uint8).reshape(shape)
        rate = FrameRateTracker()
        seq = 0
        while not stop.is_set():
            ok, frame = cap.read()
            stamp = time.monotonic()
            if not ok:
                raise RuntimeError("Camera opened but no decoded frame was received, or camera disconnected.")
            if frame.shape != (shape[1], shape[2] * 2, 3) or frame.dtype != np.uint8:
                raise ValueError(
                    "Expected %dx%d BGR SBS frame, received %s." % (shape[2] * 2, shape[1], frame.shape)
                )
            health = rate.add(stamp)
            # Keep the pair, frame number and host arrival time atomic.
            with lock:
                buffer[0] = frame[:, : shape[2]]
                buffer[1] = frame[:, shape[2] :]
                state[:] = (seq, stamp, health["measured_arrival_fps"], health["max_gap_s"] or 0.0, stamp, float("nan"))
            if seq == 0:
                code = int(cap.get(cv2.CAP_PROP_FOURCC))
                messages.put(
                    (
                        "ready",
                        {
                            "device_index": options["device"],
                            "backend": cap.getBackendName(),
                            "layout": "horizontal_sbs",
                            "swap": False,
                            "eye_size": [shape[2], shape[1]],
                            "requested_size": [options["width"], options["height"]],
                            "actual_full_size": [frame.shape[1], frame.shape[0]],
                            "requested_fps": options["fps"],
                            "reported_fps": cap.get(cv2.CAP_PROP_FPS),
                            "requested_fourcc": "MJPG",
                            "reported_fourcc": "".join(chr((code >> (8 * j)) & 255) for j in range(4)),
                            "reported_exposure_driver_units": cap.get(cv2.CAP_PROP_EXPOSURE),
                            "reported_auto_exposure": cap.get(cv2.CAP_PROP_AUTO_EXPOSURE),
                            "exposure_requested": options["exposure"],
                            "acquisition": "one_process_atomic_latest_pair",
                            "timing": "Host decoded arrivals; sensor exposure synchronization unverified",
                        },
                    )
                )
            seq += 1
    except BaseException as exc:
        messages.put(("error", repr(exc)))
    finally:
        if cap is not None:
            cap.release()
        if timer:
            ctypes.WinDLL("winmm").timeEndPeriod(1)


class Camera:
    def __init__(self, device=1, width=2560, height=720, fps=30, backend="dshow", exposure=None,
                 stream_host=None, stream_port=5600, ffmpeg="ffmpeg",
                 stream_backend="gstreamer", gstreamer="gst-launch-1.0"):
        if device < 0 or width <= 0 or width % 2 or height <= 0 or fps <= 0:
            raise ValueError("Expected nonnegative camera index, positive SBS dimensions and FPS.")
        if backend not in ("dshow", "msmf", "any") or exposure is not None and not np.isfinite(exposure):
            raise ValueError("Invalid capture backend/exposure.")
        if stream_host:
            from network_video import validate_network
            executable = validate_network(stream_host, stream_port,
                                          gstreamer if stream_backend == "gstreamer" else ffmpeg, stream_backend)
            if stream_backend == "gstreamer":
                gstreamer = executable
            else:
                ffmpeg = executable
        self.shape = (2, height, width // 2, 3)
        context = mp.get_context("spawn")
        self.pixels = context.RawArray(ctypes.c_ubyte, int(np.prod(self.shape)))
        self.state = context.RawArray(ctypes.c_double, [-1.0, 0.0, 0.0, 0.0, 0.0, float("nan")])
        self.lock, self.stop = context.Lock(), context.Event()
        self.messages = context.Queue(maxsize=4)
        self.cached, self.closed = None, False
        options = dict(device=device, width=width, height=height, fps=fps, backend=backend, exposure=exposure,
                       stream_host=stream_host, stream_port=stream_port, ffmpeg=ffmpeg,
                       stream_backend=stream_backend, gstreamer=gstreamer)
        self.process = context.Process(
            target=_capture_worker,
            args=(self.pixels, self.state, self.lock, self.stop, self.messages, options, self.shape),
            name="network-camera" if stream_host else "usb-camera",
            daemon=True,
        )
        self.process.start()
        try:
            kind, payload = self.messages.get(timeout=25)
            if kind != "ready":
                raise RuntimeError(payload)
            self.meta = payload
        except BaseException as exc:
            self.close()
            if isinstance(exc, queue.Empty):
                raise TimeoutError(
                    "Camera initialization timed out after 25 s; check sender, UDP firewall/port and decoder runtime"
                    if stream_host else "USB camera initialization timed out after 25 s; check device/backend"
                ) from exc
            raise

    def latest(self):
        try:
            kind, payload = self.messages.get_nowait()
            if kind == "error":
                raise RuntimeError(payload)
        except queue.Empty:
            pass
        if not self.process.is_alive():
            raise RuntimeError("Camera reader exited unexpectedly.")
        with self.lock:
            seq, stamp, fps, gap, received, pts = self.state[:]
            if self.cached is None or self.cached.seq != int(seq):
                pair = np.frombuffer(self.pixels, np.uint8).reshape(self.shape).copy()
                self.cached = Packet(int(seq), stamp, pair[0], pair[1], fps, gap, received, pts)
        if time.monotonic() - self.cached.received_timestamp > 5:
            raise RuntimeError("No fresh camera frame for 5 seconds.")
        return self.cached

    def close(self):
        if self.closed:
            return
        self.closed = True
        self.stop.set()
        self.process.join(timeout=3)
        if self.process.is_alive():
            self.process.terminate()  # Only our reader; it never owns a recording file.
            self.process.join(timeout=2)
        self.messages.close()
