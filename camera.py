"""Single UVC stream containing horizontal LEFT|RIGHT images."""
import os
import threading
import time
from dataclasses import dataclass

import cv2
import numpy as np

from capture_quality import FrameRateTracker


def split_sbs(frame, swap=False):
    if frame is None or frame.ndim not in (2, 3) or frame.shape[1] % 2:
        raise ValueError('SBS frame must have an even width; packed mono/10bit formats need another decoder')
    middle = frame.shape[1] // 2
    left, right = frame[:, :middle].copy(), frame[:, middle:].copy()
    return (right, left) if swap else (left, right)


def backend_code(name):
    return {'dshow': cv2.CAP_DSHOW, 'msmf': cv2.CAP_MSMF,
            'any': cv2.CAP_ANY}[name] if os.name == 'nt' else cv2.CAP_ANY


def probe_devices(max_devices=8, backend='dshow'):
    results = []
    for index in range(max_devices):
        cap = cv2.VideoCapture(index, backend_code(backend))
        try:
            if not cap.isOpened():
                continue
            ok, frame = cap.read()
            if ok:
                results.append({'index': index, 'frame_size': [frame.shape[1], frame.shape[0]],
                                'reported_fps': cap.get(cv2.CAP_PROP_FPS),
                                'backend': cap.getBackendName()})
        finally:
            cap.release()
    return results


@dataclass
class Packet:
    seq: int
    timestamp: float
    left: object
    right: object
    sensor_metadata: object = None
    color: object = None
    capture_health: object = None


class StereoSBS:
    def __init__(self, device='auto', width=2560, height=720, fps=30,
                 backend='dshow', fourcc='MJPG', swap=False, camera_id='new_sbs_camera',
                 exposure=None):
        if device == 'auto':
            candidates = probe_devices(backend=backend)
            if len(candidates) != 1:
                raise RuntimeError('Auto selection needs ONE available UVC camera. Candidates: %s. '
                                   'Use --device INDEX after previewing.' % candidates)
            device = candidates[0]['index']
        self.cap = cv2.VideoCapture(int(device), backend_code(backend))
        if not self.cap.isOpened():
            raise RuntimeError('Cannot open UVC device %s' % device)
        try:
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            self.cap.set(cv2.CAP_PROP_FPS, fps)
            # DirectShow mode negotiation may reset the pixel format when size
            # or FPS changes. Reapply the requested format LAST and read it back.
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
            if exposure is not None:
                self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, .25)
                self.cap.set(cv2.CAP_PROP_EXPOSURE, exposure)
            ok, frame = self.cap.read()
            if not ok:
                raise RuntimeError('Device opened but no decoded frame was received')
            left, right = split_sbs(frame, swap)
            reported_fourcc = int(self.cap.get(cv2.CAP_PROP_FOURCC))
            self.meta = {'camera_id': camera_id, 'device_index': int(device),
                         'backend': self.cap.getBackendName(), 'layout': 'horizontal_sbs',
                         'swap': bool(swap), 'requested_size': [width, height],
                         'actual_full_size': [frame.shape[1], frame.shape[0]],
                         'eye_size': [left.shape[1], left.shape[0]],
                         'requested_fps': fps, 'reported_fps': self.cap.get(cv2.CAP_PROP_FPS),
                         'requested_fourcc': fourcc,
                         'reported_fourcc': ''.join(chr((reported_fourcc >> (8*j)) & 255) for j in range(4)),
                         'reported_focus': self.cap.get(cv2.CAP_PROP_FOCUS),
                         'reported_autofocus': self.cap.get(cv2.CAP_PROP_AUTOFOCUS),
                         'reported_exposure_driver_units': self.cap.get(cv2.CAP_PROP_EXPOSURE),
                         'reported_auto_exposure': self.cap.get(cv2.CAP_PROP_AUTO_EXPOSURE),
                         'exposure_requested': exposure,
                         'timing': 'Both eyes in one host frame; sensor exposure synchronization unverified'}
            self.swap = swap
            self.lock = threading.Lock()
            self.stopped = threading.Event()
            self.rate_tracker = FrameRateTracker()
            stamp = time.monotonic()
            self.packet = Packet(0, stamp, left, right, capture_health=self.rate_tracker.add(stamp))
            self.error = None
            self.thread = threading.Thread(target=self._read, daemon=True)
            self.thread.start()
        except Exception:
            self.cap.release()
            raise

    def _read(self):
        seq = 0
        while not self.stopped.is_set():
            ok, frame = self.cap.read()
            if not ok:
                self.error = 'Camera read failed or disconnected'
                return
            try:
                left, right = split_sbs(frame, self.swap)
                if [left.shape[1], left.shape[0]] != self.meta['eye_size']:
                    raise ValueError('Camera changed resolution during capture')
                seq += 1
                with self.lock:
                    stamp = time.monotonic()
                    self.packet = Packet(seq, stamp, left, right,
                                         capture_health=self.rate_tracker.add(stamp))
            except Exception as exc:
                self.error = str(exc)
                return

    def latest(self):
        if self.error:
            raise RuntimeError(self.error)
        with self.lock:
            packet = self.packet
        if time.monotonic() - packet.timestamp > 5:
            raise RuntimeError('No fresh camera frame for 5 seconds')
        return packet

    def health(self):
        with self.lock:
            return self.rate_tracker.snapshot(time.monotonic())

    def close(self):
        self.stopped.set()
        self.thread.join(timeout=2)
        self.cap.release()
        self.thread.join(timeout=2)


class RealSenseStereo:
    """Read the two actual IR sensors, never split a single RGB view."""
    def __init__(self, width=1280, height=480, fps=30, serial=''):
        import pyrealsense2 as rs
        self.rs = rs
        self.pipeline = rs.pipeline()
        self.lock = threading.Lock()
        self.stopped = threading.Event()
        self.error = None
        self.rate_tracker = FrameRateTracker()
        self.changed = []
        self.thread = None
        config = rs.config()
        if serial:
            config.enable_device(serial)
        for index in (1, 2):
            config.enable_stream(rs.stream.infrared, index, width // 2, height, rs.format.y8, fps)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, fps)
        profile = self.pipeline.start(config)
        try:
            device = profile.get_device()
            for sensor in device.query_sensors():
                for option, value in [(rs.option.emitter_enabled, 0), (rs.option.global_time_enabled, 1)]:
                    if sensor.supports(option):
                        previous = sensor.get_option(option)
                        sensor.set_option(option, value)
                        self.changed.append((sensor, option, previous))
            lp = profile.get_stream(rs.stream.infrared, 1).as_video_stream_profile()
            rp = profile.get_stream(rs.stream.infrared, 2).as_video_stream_profile()
            cp = profile.get_stream(rs.stream.color).as_video_stream_profile()
            def intrinsics(p):
                i = p.get_intrinsics()
                return {'K': [[i.fx, 0., i.ppx], [0., i.fy, i.ppy], [0., 0., 1.]],
                        'D': list(i.coeffs), 'model': str(i.model)}
            extrinsics = lp.get_extrinsics_to(rp)
            color_extrinsics = lp.get_extrinsics_to(cp)
            self.meta = {'camera_id': 'realsense_' + device.get_info(rs.camera_info.serial_number),
                         'device_name': device.get_info(rs.camera_info.name),
                         'serial': device.get_info(rs.camera_info.serial_number),
                         'backend': 'REALSENSE_SDK', 'layout': 'two_ir_sensors', 'swap': False,
                         'requested_size': [width, height], 'actual_full_size': [lp.width()*2, lp.height()],
                         'eye_size': [lp.width(), lp.height()], 'requested_fps': fps,
                         'reported_fps': lp.fps(), 'emitter': 'off',
                         'color_reference_size': [cp.width(), cp.height()],
                         'sdk_factory_air_parameters_not_water_calibration': {
                             'left': intrinsics(lp), 'right': intrinsics(rp), 'color': intrinsics(cp),
                             'R_lr': np.asarray(extrinsics.rotation).reshape(3, 3, order='F').tolist(),
                             'T_lr_m': list(extrinsics.translation),
                             'R_left_ir_to_color': np.asarray(color_extrinsics.rotation).reshape(3, 3, order='F').tolist(),
                             'T_left_ir_to_color_m': list(color_extrinsics.translation)},
                         'timing': 'IR1/IR2 and RGB from one SDK frameset; RGB exposure is not synchronized to IR; sensor timestamps retained'}
            for _ in range(10):
                fs = self.pipeline.wait_for_frames(5000)
            self.packet = self._packet(fs, 0)
            self.thread = threading.Thread(target=self._read, daemon=True)
            self.thread.start()
        except Exception:
            self.close()
            raise

    def _packet(self, fs, seq):
        frames = [fs.get_infrared_frame(1), fs.get_infrared_frame(2)]
        if not all(frames):
            raise RuntimeError('Both IR frames must be present')
        timestamps = [f.get_timestamp() for f in frames]
        domains = [str(f.get_frame_timestamp_domain()) for f in frames]
        delta = abs(timestamps[0] - timestamps[1])
        if len(set(domains)) != 1 or delta > 2.:
            raise ValueError('IR timestamp synchronization not within 2 milliseconds')
        images = [np.asanyarray(f.get_data()).copy() for f in frames]
        metadata = {'frame_numbers': [f.get_frame_number() for f in frames],
                    'timestamps_ms': timestamps, 'timestamp_domains': domains, 'ir_sync_ms': delta}
        color_frame = fs.get_color_frame()
        color = np.asanyarray(color_frame.get_data()).copy() if color_frame else None
        if color_frame:
            metadata['color_frame_number'] = color_frame.get_frame_number()
            metadata['color_timestamp_ms'] = color_frame.get_timestamp()
            metadata['color_timestamp_domain'] = str(color_frame.get_frame_timestamp_domain())
            comparable = all(domain == metadata['color_timestamp_domain'] for domain in domains)
            metadata['color_ir_timestamp_comparable'] = comparable
            metadata['color_minus_left_ir_ms'] = (metadata['color_timestamp_ms'] - timestamps[0]
                                                  if comparable else None)
            metadata['color_ir_delta_ms'] = (max(abs(metadata['color_timestamp_ms'] - stamp)
                                                  for stamp in timestamps) if comparable else None)
        stamp = time.monotonic()
        # Caller holds the reader lock after startup.
        return Packet(seq, stamp, images[0], images[1], metadata, color,
                      self.rate_tracker.add(stamp))

    def _read(self):
        seq = 0
        while not self.stopped.is_set():
            try:
                fs = self.pipeline.wait_for_frames(2000)
                with self.lock:
                    packet = self._packet(fs, seq + 1)
                    seq += 1
                    self.packet = packet
            except ValueError:
                continue
            except RuntimeError as exc:
                if not self.stopped.is_set():
                    self.error = str(exc)
                return

    def latest(self):
        if self.error:
            raise RuntimeError(self.error)
        with self.lock:
            packet = self.packet
        if time.monotonic() - packet.timestamp > 5.:
            raise RuntimeError('No synchronized IR frames for 5 seconds')
        return packet

    def close(self):
        self.stopped.set()
        if self.thread:
            self.thread.join(timeout=3)
        for sensor, option, previous in reversed(self.changed):
            try:
                sensor.set_option(option, previous)
            except RuntimeError:
                pass
        self.pipeline.stop()

    def health(self):
        with self.lock:
            return self.rate_tracker.snapshot(time.monotonic())
