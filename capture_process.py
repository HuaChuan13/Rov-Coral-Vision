"""Isolated USB acquisition, one atomic latest stereo pair in shared memory.

Separate Python/OpenCV runtime prevents inference from blocking acquisition.
The lock covers pair copy plus sequence/timestamp; mixed-eye frames are forbidden.
"""
import ctypes
import json
import multiprocessing as mp
import queue
import time

import numpy as np

COLOR_SHAPE = (480, 640, 3)
METADATA_CAPACITY = 8192


def _capture_worker(pixels, color_pixels, metadata_pixels, state, lock, stop, messages,
                    options, shape, source):
    import cv2
    from camera import RealSenseStereo, StereoSBS
    timer = False
    camera = None
    try:
        if hasattr(ctypes, 'WinDLL'):
            timer = ctypes.WinDLL('winmm').timeBeginPeriod(1) == 0
        cv2.setNumThreads(1)
        if source == 'realsense':
            # SDK IR streams do not accept the UVC device/format/swap options.
            sdk_options = {key: options[key] for key in ('width', 'height', 'fps', 'serial')
                           if key in options}
            camera = RealSenseStereo(**sdk_options)
        else:
            uvc_options = {key: value for key, value in options.items() if key != 'serial'}
            camera = StereoSBS(**uvc_options)
        expected_eye_size = [shape[2], shape[1]]
        if camera.meta['eye_size'] != expected_eye_size:
            raise ValueError('Requested per-eye %s, but camera returned %s.' %
                             (expected_eye_size, camera.meta['eye_size']))
        buffer = np.frombuffer(pixels, np.uint8).reshape(shape)
        color_buffer = (np.frombuffer(color_pixels, np.uint8).reshape(COLOR_SHAPE)
                        if source == 'realsense' else None)
        previous = -1
        ready = False
        while not stop.is_set():
            packet = camera.latest()
            if packet.seq == previous:
                time.sleep(.001)
                continue
            # Preserve the two IR views while satisfying the existing BGR
            # detection and stereo processing interface.
            pair = [cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR) if frame.ndim == 2 else frame
                    for frame in (packet.left, packet.right)]
            if any(frame.shape != shape[1:] for frame in pair):
                raise ValueError('Camera frame shape changed; expected %s, received %s.' %
                                 (shape[1:], [frame.shape for frame in pair]))
            color = packet.color if source == 'realsense' else None
            if color is not None and (color.shape != COLOR_SHAPE or color.dtype != np.uint8):
                raise ValueError('RealSense color must be 640x480 uint8 BGR; received %s %s.' %
                                 (color.shape, color.dtype))
            # A bounded shared payload keeps timing/frame numbers attached to
            # exactly the same IR and RGB packet without a growing frame queue.
            payload = json.dumps({'sensor_metadata': packet.sensor_metadata,
                                  'capture_health': packet.capture_health},
                                 separators=(',', ':')).encode('utf-8')
            if len(payload) > len(metadata_pixels):
                raise ValueError('Camera metadata exceeds shared memory capacity.')
            with lock:
                buffer[0] = pair[0]
                buffer[1] = pair[1]
                if color is not None:
                    color_buffer[:] = color
                metadata_pixels[:len(payload)] = payload
                state[0] = packet.seq
                state[1] = packet.timestamp
                state[2] = camera.health()['measured_arrival_fps']
                state[3] = float(color is not None)
                state[4] = len(payload)
            previous = packet.seq
            if not ready:
                meta = dict(camera.meta, source=source,
                            acquisition='isolated_process_atomic_latest_pair')
                messages.put(('ready', meta))
                ready = True
    except BaseException as exc:
        messages.put(('error', repr(exc)))
    finally:
        if camera:
            camera.close()
        if timer:
            ctypes.WinDLL('winmm').timeEndPeriod(1)


class ProcessStereoSBS:
    def __init__(self, source='sbs', **options):
        if source not in ('sbs', 'realsense'):
            raise ValueError('Unknown stereo source: %s' % source)
        options = dict(options)
        width = int(options.setdefault('width', 1280 if source == 'realsense' else 2560))
        height = int(options.setdefault('height', 480 if source == 'realsense' else 720))
        if width <= 0 or width % 2 or height <= 0:
            raise ValueError('Stereo width must be positive and even; height must be positive.')
        options.update(width=width, height=height)
        self.shape = (2, height, width // 2, 3)
        context = mp.get_context('spawn')
        self.pixels = context.RawArray(ctypes.c_ubyte, int(np.prod(self.shape)))
        self.color_pixels = context.RawArray(ctypes.c_ubyte,
            int(np.prod(COLOR_SHAPE)) if source == 'realsense' else 0)
        self.metadata_pixels = context.RawArray(ctypes.c_ubyte, METADATA_CAPACITY)
        self.state = context.RawArray(ctypes.c_double, [-1., 0., 0., 0., 0.])
        self.lock = context.Lock()
        self.stop = context.Event()
        self.messages = context.Queue(maxsize=4)
        self.cached = None
        self.fps = 0.
        self.closed = False
        self.process = context.Process(target=_capture_worker,
            args=(self.pixels, self.color_pixels, self.metadata_pixels, self.state, self.lock,
                  self.stop, self.messages, options, self.shape, source),
            name='isolated-stereo-camera', daemon=True)
        self.process.start()
        try:
            kind, payload = self.messages.get(timeout=25)
            if kind != 'ready':
                raise RuntimeError(payload)
            self.meta = payload
        except BaseException:
            self.close()
            raise

    def latest(self):
        from camera import Packet
        try:
            kind, payload = self.messages.get_nowait()
            if kind == 'error':
                raise RuntimeError(payload)
        except queue.Empty:
            pass
        if not self.process.is_alive():
            raise RuntimeError('Camera capture process exited unexpectedly.')
        with self.lock:
            seq, stamp, self.fps, has_color, metadata_length = self.state[:]
            if self.cached is None or self.cached.seq != int(seq):
                pair = np.frombuffer(self.pixels, np.uint8).reshape(self.shape).copy()
                color = (np.frombuffer(self.color_pixels, np.uint8).reshape(COLOR_SHAPE).copy()
                         if has_color else None)
                payload = json.loads(bytes(self.metadata_pixels[:int(metadata_length)]).decode('utf-8'))
                self.cached = Packet(int(seq), stamp, pair[0], pair[1],
                                     sensor_metadata=payload['sensor_metadata'], color=color,
                                     capture_health=payload['capture_health'])
        if time.monotonic()-self.cached.timestamp > 5:
            raise RuntimeError('No fresh camera frame for 5 seconds.')
        return self.cached

    def health(self):
        return {'measured_arrival_fps': self.fps}

    def close(self):
        if self.closed:
            return
        self.closed = True
        self.stop.set()
        self.process.join(timeout=6)
        if self.process.is_alive():
            # Only our acquisition child; it never owns a recording file.
            self.process.terminate()
            self.process.join(timeout=2)
        self.messages.close()
