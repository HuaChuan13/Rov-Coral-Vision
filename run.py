"""USB stereo coral detection and calibrated XYZ. Q records; W saves/exits."""
import argparse
import ctypes
import csv
import hashlib
import json
import os
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from stereo import WaterGeometry, StereoDepth, estimate_target
from output import read_image, save_json, host_array, compose_view, detection_records, VideoRecorder

ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = ROOT / 'models/best.pt'
DEFAULT_CALIBRATION = ROOT / 'config/calibration.json'
WINDOW = 'Underwater coral detection and calibrated depth | Q record | W save and exit'


def key_action(key):
    if key in (ord('q'), ord('Q')):
        return 'record'
    if key in (ord('w'), ord('W'), 27):
        return 'exit'
    return None

def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    mode = p.add_mutually_exclusive_group()
    mode.add_argument('--live', action='store_true')
    mode.add_argument('--offline', type=Path, help='Dataset folder, reads original/*_left.png; NEVER opens a camera')
    p.add_argument('--calibration', type=Path, default=DEFAULT_CALIBRATION)
    p.add_argument('--model', type=Path, default=DEFAULT_MODEL)
    p.add_argument('--camera', type=int, default=1, help='UVC device index; verify identity before live run')
    p.add_argument('--backend', choices=('dshow', 'msmf', 'any'), default='dshow')
    p.add_argument('--device', default='auto', help='YOLO GPU index, cpu or auto')
    p.add_argument('--half', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--cuda-convolution', choices=('native', 'cudnn'), default='native',
                   help='Native CUDA avoids cuDNN allocation failures on this 4GB GPU; CUDA inference remains enabled')
    p.add_argument('--input-enhancement', choices=('mild', 'off'), default='mild',
                   help='Photometric enhancement for detection/display only, preserving stereo measurement inputs')
    p.add_argument('--imgsz', type=int, default=416)
    p.add_argument('--confidence', type=float, default=.6)
    p.add_argument('--stereo-width', type=int, choices=(320, 640, 1280), default=320)
    p.add_argument('--num-disparities', type=int, default=0)
    p.add_argument('--block-size', type=int, default=3)
    p.add_argument('--wls-lambda', type=float, default=1600.)
    p.add_argument('--wls-sigma', type=float, default=.8)
    p.add_argument('--wls-confidence', type=float, default=128.)
    p.add_argument('--near', type=float, default=.20)
    p.add_argument('--far', type=float, default=2.)
    p.add_argument('--min-points', type=int, default=30)
    p.add_argument('--cv-threads', type=int, default=2)
    p.add_argument('--enhance-stereo', action='store_true', help='Optional paired CLAHE for grayscale matching only')
    p.add_argument('--invalid-display', choices=('complete', 'purple'), default='complete',
                   help='Pure-color display completion or purple holes; never fills metric measurements')
    p.add_argument('--frames', type=int, default=0, help='0: live unlimited, offline 120 frames')
    p.add_argument('--show-offline', action='store_true')
    p.add_argument('--record-offline', action='store_true', help='Test encoding without camera')
    p.add_argument('--output', type=Path, default=ROOT / 'coral_runs')
    p.add_argument('--stop-file', type=Path, default=None, help='Graceful external stop, also finalizes a recording')
    args = p.parse_args(argv)
    args.live = args.offline is None
    if not 0 < args.near < args.far or not 0 < args.confidence <= 1:
        p.error('Invalid distance range/confidence.')
    args.confidence = max(.6, args.confidence)
    if args.frames < 0 or args.cv_threads < 1 or args.min_points < 1 or args.block_size < 1 or args.block_size % 2 != 1:
        p.error('Invalid frame count/thread count/block size/minimum points.')
    if args.num_disparities < 0 or args.num_disparities % 16 or args.imgsz < 160 or args.imgsz % 32:
        p.error('Invalid disparities or inference image size (multiple of 32).')
    if args.wls_lambda <= 0 or args.wls_sigma <= 0 or not 0 <= args.wls_confidence <= 255:
        p.error('Invalid WLS parameters.')
    if args.live and (args.show_offline or args.record_offline):
        p.error('Offline flags cannot be used with --live.')
    return args

def main(argv=None):
    args = parse_args(argv)
    if not args.model.is_file():
        raise FileNotFoundError('Trained coral model not found: ' + str(args.model))
    geometry = WaterGeometry(args.calibration, args.stereo_width)
    matcher = StereoDepth(geometry, args)
    # Do this before importing Torch. No persistent Windows/system setting is changed.
    os.environ.setdefault('CUDA_MODULE_LOADING', 'LAZY')
    import torch
    from ultralytics import YOLO
    from image_enhancement import UnderwaterEnhancer
    torch.set_num_threads(1)
    torch.backends.cudnn.enabled = args.cuda_convolution == 'cudnn'
    torch.backends.cudnn.benchmark = False
    enhancer = UnderwaterEnhancer(args.input_enhancement)
    device = ('0' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else args.device
    model = YOLO(str(args.model))
    predict = dict(conf=args.confidence, verbose=False, device=device, half=args.half and device != 'cpu', imgsz=args.imgsz)
    for _ in range(5):
        warmup = np.zeros((geometry.size[1], geometry.size[0], 3), np.uint8)
        model.predict(warmup, **predict)
    matcher.enable_gpu_reconstruction(device)
    # Ultralytics changes OpenCV global threading on import; set AFTER warm-up.
    cv2.setNumThreads(args.cv_threads)
    run = args.output / datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    run.mkdir(parents=True, exist_ok=False)
    meta = {'mode': 'live' if args.live else 'offline_benchmark_no_camera',
            'args': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
            'model_sha256': hashlib.sha256(args.model.read_bytes()).hexdigest(),
            'classes': model.names, 'device': device, 'torch_version': torch.__version__,
            'opencv_version': cv2.__version__, 'geometry': geometry.metadata(),
            'input_enhancement': enhancer.metadata(),
            'enhancement_scope': 'detector and left display only; stereo images and reconstruction guide remain original',
            'cuda_convolution': args.cuda_convolution, 'inference_half': predict['half'],
            'num_disparities': matcher.nd,
            'depth_display': '%dx%d full-FOV TURBO; unavailable=%s' % (*geometry.size, args.invalid_display),
            'border_policy': 'equal search padding; recovered border requires real-image correspondence + LR consistency',
            'measurement': 'strict WLS confidence + raw LR check; central bbox MAD-filtered median XYZ; rectified-left frame, meters (labels cm)',
            'image_source': 'color SBS',
            'warning': 'Same SBS host frame does NOT establish sensor exposure synchronization.'}
    save_json(run / 'settings.json', meta)
    save_json(run / 'calibration_snapshot.json', geometry.data)
    print('Model:', args.model, '| Classes:', model.names, flush=True)
    print('Camera NOT opened yet. Calibration:', args.calibration, flush=True)
    print('Per eye %dx%d | baseline %.3f mm | SGBM %s, %d disparities | device %s' %
          (*geometry.size, 1000*geometry.baseline, geometry.stereo_size, matcher.nd, device), flush=True)
    print('Input enhancement:', args.input_enhancement, '| CUDA convolution:', args.cuda_convolution,
          '| FP16:', predict['half'], flush=True)
    if not geometry.verified:
        print('WARNING: calibration distance accuracy is UNVERIFIED. Do not treat displayed Z as validated accuracy.', flush=True)
    if geometry.quality_passed is False:
        print('WARNING: the supplied water calibration has quality_passed=false. Check known distances before evaluating range accuracy.', flush=True)
    print('Output:', run, '| Q starts recording; W saves and exits.', flush=True)
    camera, pool, preview = None, None, None
    recorder = VideoRecorder(run)
    show = args.live or args.show_offline
    rows, timestamps = [], deque(maxlen=60)
    frame_no, previous_seq, skipped = 0, -1, 0
    error = None
    timer_active = False
    began = time.perf_counter()
    try:
        offline = []
        if args.offline:
            paths = sorted((args.offline / 'original').glob('*_left.png'))[:12]
            if not paths:
                raise ValueError('No original/*_left.png offline pairs.')
            for lp in paths:
                offline.append((read_image(lp), read_image(lp.with_name(lp.name.replace('_left', '_right')))))
        else:
            # DirectShow and Python 3.9 on Windows otherwise have coarse sleep
            # granularity. Balance the request on exit; no persistent OS change.
            if hasattr(ctypes, 'WinDLL'):
                timer_active = ctypes.WinDLL('winmm').timeBeginPeriod(1) == 0
            from camera import Camera
            camera = Camera(device=args.camera, width=2*geometry.size[0], height=geometry.size[1], backend=args.backend)
            save_json(run / 'camera.json', camera.meta)
        if show:
            from preview_window import PreviewWindow
            preview = PreviewWindow(WINDOW, (2*geometry.size[0]+128, geometry.size[1]+120))
        pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix='stereo-cpu')
        limit = args.frames or (0 if args.live else 120)
        with (run / 'frame_metrics.csv').open('w', newline='', encoding='utf-8-sig') as log, \
                (run / 'detections.jsonl').open('w', encoding='utf-8') as detections_log:
            writer = csv.writer(log)
            writer.writerow(['processed_frame', 'capture_seq', 'capture_monotonic_s', 'source_age_ms',
                             'capture_fps', 'fresh_processing_fps', 'processing_ms', 'stereo_ms',
                             'yolo_ms', 'strict_valid_fraction', 'detections', 'skipped_camera_frames', 'enhancement_ms'])
            while not limit or frame_no < limit:
                if args.stop_file and args.stop_file.exists():
                    break
                if show and cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                    break
                key = cv2.pollKey() & 255 if show else -1
                action = key_action(key)
                if action == 'exit':
                    break
                if action == 'record' and frame_no:
                    recorder.start(canvas, time.perf_counter())
                if camera:
                    packet = camera.latest()
                    if packet.seq == previous_seq:
                        time.sleep(.001)
                        continue
                    left, right, stamp, seq = packet.left, packet.right, packet.timestamp, packet.seq
                    capture_fps = packet.capture_fps
                    if previous_seq >= 0:
                        skipped += max(0, seq-previous_seq-1)
                    previous_seq = seq
                else:
                    left, right = offline[frame_no % len(offline)]
                    stamp, seq, capture_fps = time.monotonic(), frame_no, 0.
                start = time.perf_counter()
                color, stereo_images = geometry.rectify(left, right)
                future = pool.submit(matcher.compute, stereo_images, color)
                enhance_start = time.perf_counter()
                detector_input = enhancer.apply(color)
                enhancement_ms = 1000*(time.perf_counter()-enhance_start)
                detect_start = time.perf_counter()
                with torch.inference_mode():
                    # Let the predictor resize once; its postprocessor returns
                    # boxes directly in the native calibrated input coordinates.
                    result = model.predict(detector_input, **predict)[0]
                    boxes = result.boxes.data.cpu().numpy() if result.boxes is not None else np.empty((0, 6))
                    boxes = boxes.copy()
                yolo_ms = 1000*(time.perf_counter()-detect_start)
                estimated, measured, stereo_ms = future.result()
                found = []
                for row in boxes:
                    box, score, cls = row[:4], float(row[4]), int(row[5])
                    if not np.isfinite(score) or score < max(.6, args.confidence):
                        continue
                    pos, count, status = estimate_target(matcher.measurement_grid, box, matcher.measurement_geometry, args.min_points)
                    found.append((box, cls, score, str(model.names[cls]), pos, count, status))
                now = time.perf_counter()
                # Fresh-frame throughput excludes repeated UI refreshes/encoded holds.
                fps = (len(timestamps)-1)/(timestamps[-1]-timestamps[0]) if len(timestamps) > 1 else 0.
                age_ms = max(0., 1000*(time.monotonic()-stamp))
                records = detection_records(found, geometry.size, args.confidence)
                canvas = compose_view(detector_input, estimated, found, geometry, args, fps, capture_fps, age_ms, recorder.active, matcher.colors)
                if frame_no == 0 or frame_no == 29:
                    cv2.imencode('.png', canvas)[1].tofile(str(run / 'preview.png'))
                if frame_no == 29:
                    size_label = '%dx%d' % geometry.size
                    cv2.imencode('.png', color)[1].tofile(str(run/('left_rectified_'+size_label+'.png')))
                    cv2.imencode('.png', detector_input)[1].tofile(str(run/('detection_input_'+size_label+'.png')))
                    np.save(run/('depth_estimated_'+size_label+'_m.npy'), host_array(estimated))
                    np.save(run/('depth_measured_'+size_label+'_m.npy'), host_array(measured))
                    np.save(run/('depth_display_ONLY_completed_'+size_label+'_m.npy'), host_array(matcher.display_depth))
                if show:
                    preview.show(canvas)
                if args.record_offline and frame_no == 0:
                    recorder.start(canvas, time.perf_counter())
                elif recorder.active:
                    recorder.submit(canvas, time.perf_counter())
                frame_no += 1
                elapsed = 1000*(time.perf_counter()-start)
                row = [frame_no, seq, stamp, age_ms, capture_fps, fps, elapsed, stereo_ms, yolo_ms,
                       float(np.isfinite(matcher.measurement_grid).mean()), len(found), skipped, enhancement_ms]
                writer.writerow(row)
                detections_log.write(json.dumps({'frame': frame_no, 'capture_seq': seq, 'objects': records}, ensure_ascii=False)+'\n')
                if frame_no % 30 == 0:
                    log.flush()
                    detections_log.flush()
                    print('Fresh %.1f FPS | processing %.1f ms | stereo %.1f ms | YOLO %.1f ms | detections %d' %
                          (fps, elapsed, stereo_ms, yolo_ms, len(found)), flush=True)
                timestamps.append(time.perf_counter())
                rows.append(row)
    except BaseException as exc:
        error = repr(exc)
        raise
    finally:
        try:
            recorder.close()
        finally:
            if camera:
                camera.close()
            if timer_active:
                ctypes.WinDLL('winmm').timeEndPeriod(1)
            if pool:
                pool.shutdown(wait=True)
            if preview:
                preview.close()
            stable = rows[10:] if len(rows) > 10 else rows
            mean_ms = float(np.mean([r[6] for r in stable])) if stable else None
            summary = {'processed_frames': frame_no, 'mode': meta['mode'], 'error': error,
                       'run_wall_s': time.perf_counter()-began, 'skipped_camera_frames': skipped,
                       'mean_processing_ms_excluding_first10': mean_ms,
                       'processing_capacity_fps': 1000/mean_ms if mean_ms else None,
                       'last_window_fresh_fps': (len(timestamps)-1)/(timestamps[-1]-timestamps[0]) if len(timestamps)>1 else None,
                       'live_FPS_verified': bool(args.live and len(rows) >= 120),
                       'distance_accuracy_verified': geometry.verified,
                       'input_enhancement': enhancer.metadata(),
                       'mean_enhancement_ms': float(np.mean([r[12] for r in stable])) if stable else None,
                       'recording': str(recorder.path) if recorder.path else None,
                       'note': 'Offline capacity excludes camera/USB/GUI; encoded FPS is not fresh processing FPS.'}
            save_json(run / 'summary.json', summary)
            print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
