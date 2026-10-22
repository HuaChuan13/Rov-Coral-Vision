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
from stereo import WaterGeometry, StereoDepth, observations
from camera import Camera
from output import read_image, save_json, host_array, compose_view, view_size, detection_records, VideoRecorder

ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = ROOT / "models/best.pt"
DEFAULT_CALIBRATION = ROOT / "config/calibration.json"
WINDOW = "Underwater coral detection and calibrated depth | Q record | W save and exit"


def key_action(key):
    if key in (ord("q"), ord("Q")):
        return "record"
    if key in (ord("w"), ord("W"), 27):
        return "exit"
    return None


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--live", action="store_true")
    mode.add_argument(
        "--capture-only",
        action="store_true",
        help="Measure USB arrivals without YOLO, depth, GUI or video encoding",
    )
    mode.add_argument(
        "--offline", type=Path, help="Dataset folder, reads original/*_left.png; NEVER opens a camera"
    )
    p.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION)
    p.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    p.add_argument("--camera", type=int, default=1, help="UVC device index; verify identity before live run")
    p.add_argument("--capture-fps", type=float, default=30.0)
    p.add_argument(
        "--exposure",
        type=float,
        default=None,
        help="Optional driver exposure; default preserves current camera setting",
    )
    p.add_argument(
        "--depth-fps",
        type=lambda value: value if value == "max" else float(value),
        default="max",
        help="max: uncapped asynchronous depth (default); positive Hz: capped async; 0: same-frame synchronous",
    )
    p.add_argument(
        "--depth-max-age", type=float, default=0.75, help="Hide older depth results after this many seconds"
    )
    p.add_argument("--backend", choices=("dshow", "msmf", "any"), default="dshow")
    p.add_argument("--device", default="auto", help="YOLO GPU index, cpu or auto")
    p.add_argument("--half", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument(
        "--cuda-convolution",
        choices=("native", "cudnn"),
        default="native",
        help="CUDA convolution backend; native avoids cuDNN workspace allocation failures",
    )
    p.add_argument(
        "--input-enhancement",
        choices=("mild", "off"),
        default="mild",
        help="Photometric enhancement for detection/display only, preserving stereo measurement inputs",
    )
    p.add_argument("--imgsz", type=int, default=416)
    p.add_argument("--confidence", type=float, default=0.6)
    p.add_argument("--stereo-width", type=int, choices=(320, 640, 1280), default=320)
    p.add_argument("--num-disparities", type=int, default=0)
    p.add_argument("--block-size", type=int, default=3)
    p.add_argument("--wls-lambda", type=float, default=1600.0)
    p.add_argument("--wls-sigma", type=float, default=0.8)
    p.add_argument("--wls-confidence", type=float, default=128.0)
    p.add_argument("--near", type=float, default=0.20)
    p.add_argument("--far", type=float, default=2.0)
    p.add_argument("--min-points", type=int, default=30)
    p.add_argument("--cv-threads", type=int, default=2)
    p.add_argument(
        "--enhance-stereo", action="store_true", help="Optional paired CLAHE for grayscale matching only"
    )
    p.add_argument(
        "--invalid-display",
        choices=("complete", "purple"),
        default="complete",
        help="Pure-color display completion or purple holes; never fills metric measurements",
    )
    p.add_argument("--frames", type=int, default=0, help="0: live unlimited, offline 120, capture-only 150")
    p.add_argument("--show-offline", action="store_true")
    p.add_argument("--show-depth", action="store_true", help="Show the depth debug panel beside the left image")
    p.add_argument("--record-offline", action="store_true", help="Test encoding without camera")
    p.add_argument("--output", type=Path, default=ROOT / "coral_runs")
    p.add_argument(
        "--stop-file", type=Path, default=None, help="Graceful external stop, also finalizes a recording"
    )
    args = p.parse_args(argv)
    args.live = args.offline is None and not args.capture_only
    numeric = (
        args.near,
        args.far,
        args.confidence,
        args.wls_lambda,
        args.wls_sigma,
        args.wls_confidence,
        args.capture_fps,
        0 if args.depth_fps == "max" else args.depth_fps,
        args.depth_max_age,
    )
    if not np.isfinite(numeric).all() or args.exposure is not None and not np.isfinite(args.exposure):
        p.error("Expected finite numeric options.")
    if (
        args.camera < 0 or args.capture_fps <= 0
        or (args.depth_fps != "max" and args.depth_fps < 0) or args.depth_max_age <= 0
    ):
        p.error("Invalid camera index, capture/depth frequency or depth age limit.")
    if not 0 < args.near < args.far or not 0 < args.confidence <= 1:
        p.error("Invalid distance range/confidence.")
    args.confidence = max(0.6, args.confidence)
    if (
        args.frames < 0
        or args.cv_threads < 1
        or args.min_points < 1
        or args.block_size < 1
        or args.block_size % 2 != 1
    ):
        p.error("Invalid frame count/thread count/block size/minimum points.")
    if args.num_disparities < 0 or args.num_disparities % 16 or args.imgsz < 160 or args.imgsz % 32:
        p.error("Invalid disparities or inference image size (multiple of 32).")
    if args.wls_lambda <= 0 or args.wls_sigma <= 0 or not 0 <= args.wls_confidence <= 255:
        p.error("Invalid WLS parameters.")
    if (args.live or args.capture_only) and (args.show_offline or args.record_offline):
        p.error("Offline flags cannot be used with --live.")
    return args


def frame_rate(stamps):
    return (len(stamps) - 1) / (stamps[-1] - stamps[0]) if len(stamps) > 1 and stamps[-1] > stamps[0] else 0.0


def capture_benchmark(args):
    folder = args.output / ("capture_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    folder.mkdir(parents=True)
    camera, timer, error = None, False, None
    stamps, skipped, previous = [], 0, -1
    try:
        timer = hasattr(ctypes, "WinDLL") and ctypes.WinDLL("winmm").timeBeginPeriod(1) == 0
        camera = Camera(args.camera, fps=args.capture_fps, backend=args.backend, exposure=args.exposure)
        save_json(folder / "camera.json", camera.meta)
        with (folder / "capture_metrics.csv").open("w", newline="", encoding="utf-8-sig") as stream:
            writer = csv.writer(stream)
            writer.writerow(["capture_seq", "capture_monotonic_s", "capture_fps", "max_gap_s"])
            while len(stamps) < (args.frames or 150):
                if args.stop_file and args.stop_file.exists():
                    break
                packet = camera.latest()
                if packet.seq == previous:
                    time.sleep(0.001)
                    continue
                if previous >= 0:
                    skipped += max(0, packet.seq - previous - 1)
                previous = packet.seq
                stamps.append(packet.timestamp)
                writer.writerow([packet.seq, packet.timestamp, packet.capture_fps, packet.max_gap_s])
    except BaseException as exc:
        error = repr(exc)
        raise
    finally:
        if camera:
            camera.close()
        if timer:
            ctypes.WinDLL("winmm").timeEndPeriod(1)
        stable = stamps[10:] if len(stamps) > 11 else stamps
        gaps = np.diff(stable)
        summary = {
            "mode": "capture_only",
            "observed_frames": len(stamps),
            "observed_fps": frame_rate(stable),
            "skipped_reader_frames": skipped,
            "requested_fps": args.capture_fps,
            "p95_gap_ms": float(np.percentile(gaps, 95) * 1000) if len(gaps) else None,
            "error": error,
            "note": "Decoded host arrivals; no YOLO/depth/GUI/encoding. Not sensor exposure timestamps.",
        }
        save_json(folder / "summary.json", summary)
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
        print("Output:", folder, flush=True)
    return 0


def finish_depth(pending, names, args):
    result = pending["future"].result()
    found = observations(
        pending["boxes"], names, args.confidence, result["grid"], result["geometry"], args.min_points
    )
    result.update(
        seq=pending["seq"],
        stamp=pending["stamp"],
        found=found,
        objects=detection_records(found, result["geometry"].color_size, args.confidence),
        valid_fraction=float(np.isfinite(result["grid"]).mean()),
    )
    return result


def log_depth(stream, depth):
    stream.write(
        json.dumps(
            {
                "capture_seq": depth["seq"],
                "capture_monotonic_s": depth["stamp"],
                "result_age_ms": max(0.0, 1000 * (time.monotonic() - depth["stamp"])),
                "stereo_ms": depth["stereo_ms"],
                "objects": depth["objects"],
            },
            ensure_ascii=False,
        )
        + "\n"
    )


def main(argv=None):
    args = parse_args(argv)
    if args.capture_only:
        return capture_benchmark(args)
    if not args.model.is_file():
        raise FileNotFoundError("Trained coral model not found: " + str(args.model))
    geometry = WaterGeometry(args.calibration, args.stereo_width)
    matcher = StereoDepth(geometry, args)
    os.environ.setdefault("CUDA_MODULE_LOADING", "LAZY")
    import torch
    from ultralytics import YOLO
    from image_enhancement import UnderwaterEnhancer

    torch.set_num_threads(1)
    torch.backends.cudnn.enabled = args.cuda_convolution == "cudnn"
    torch.backends.cudnn.benchmark = False
    enhancer = UnderwaterEnhancer(args.input_enhancement)
    device = ("0" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    model = YOLO(str(args.model))
    predict = dict(
        conf=args.confidence,
        verbose=False,
        device=device,
        half=args.half and device != "cpu",
        imgsz=args.imgsz,
    )
    for _ in range(5):
        model.predict(np.zeros((geometry.size[1], geometry.size[0], 3), np.uint8), **predict)
    matcher.enable_gpu_reconstruction(device)
    cv2.setNumThreads(args.cv_threads)
    run = args.output / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run.mkdir(parents=True, exist_ok=False)
    meta = {
        "mode": "live" if args.live else "offline_benchmark_no_camera",
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "model_sha256": hashlib.sha256(args.model.read_bytes()).hexdigest(),
        "classes": model.names,
        "device": device,
        "torch_version": torch.__version__,
        "opencv_version": cv2.__version__,
        "geometry": geometry.metadata(),
        "input_enhancement": enhancer.metadata(),
        "inference_half": predict["half"],
        "num_disparities": matcher.nd,
        "depth_policy": "same-frame synchronous"
        if args.depth_fps == 0
        else "one bounded asynchronous job; separate frame IDs",
        "warning": "Host arrival times do not establish sensor exposure synchronization; range accuracy unverified.",
    }
    save_json(run / "settings.json", meta)
    save_json(run / "calibration_snapshot.json", geometry.data)
    print(
        "Model:",
        args.model,
        "| device:",
        device,
        "| depth:",
        args.depth_fps,
        "(max=uncapped async; positive=Hz; 0=synchronous)",
        flush=True,
    )
    print("Calibration:", args.calibration, "| baseline %.3f mm" % (1000 * geometry.baseline), flush=True)
    if not geometry.verified or geometry.quality_passed is False:
        print("WARNING: calibration distance accuracy is UNVERIFIED; check known distances.", flush=True)
    print("Output:", run, "| Q records; W saves/exits.", flush=True)
    camera, pool, preview, canvas = None, None, None, None
    recorder = VideoRecorder(run)
    show = args.live or args.show_offline
    stamps, depth_stamps = deque(maxlen=60), deque(maxlen=60)
    totals, warmup_totals, depth_totals = np.zeros(3), np.zeros(3), np.zeros(2)
    frame_no, previous_seq, skipped = 0, -1, 0
    pending, depth, next_depth = None, None, 0.0
    timer, error = False, None
    began = time.perf_counter()
    try:
        offline = []
        if args.offline:
            for lp in sorted((args.offline / "original").glob("*_left.png"))[:12]:
                offline.append((read_image(lp), read_image(lp.with_name(lp.name.replace("_left", "_right")))))
            if not offline:
                raise ValueError("No original/*_left.png offline pairs.")
        else:
            timer = hasattr(ctypes, "WinDLL") and ctypes.WinDLL("winmm").timeBeginPeriod(1) == 0
            camera = Camera(
                args.camera,
                width=2 * geometry.size[0],
                height=geometry.size[1],
                fps=args.capture_fps,
                backend=args.backend,
                exposure=args.exposure,
            )
            save_json(run / "camera.json", camera.meta)
        if show:
            from preview_window import PreviewWindow

            preview = PreviewWindow(WINDOW, view_size(geometry, args))
        pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="depth")
        limit = args.frames or (0 if args.live else 120)
        with (
            (run / "frame_metrics.csv").open("w", newline="", encoding="utf-8-sig") as log,
            (run / "detections.jsonl").open("w", encoding="utf-8") as detections_log,
            (run / "depth_measurements.jsonl").open("w", encoding="utf-8") as depth_log,
        ):
            writer = csv.writer(log)
            writer.writerow(
                [
                    "processed_frame",
                    "capture_seq",
                    "capture_monotonic_s",
                    "source_age_ms",
                    "capture_fps",
                    "fresh_processing_fps",
                    "processing_ms",
                    "stereo_ms",
                    "yolo_ms",
                    "strict_valid_fraction",
                    "detections",
                    "skipped_camera_frames",
                    "enhancement_ms",
                    "depth_frame_id",
                    "depth_age_ms",
                    "depth_updated",
                ]
            )
            while not limit or frame_no < limit:
                if args.stop_file and args.stop_file.exists():
                    break
                if show and cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                    break
                action = key_action(cv2.pollKey() & 255) if show else None
                if action == "exit":
                    break
                if action == "record" and canvas is not None:
                    recorder.start(canvas, time.perf_counter())
                updated = False
                if camera:
                    packet = camera.latest()
                    if packet.seq == previous_seq:
                        time.sleep(0.001)
                        continue
                    left, right, stamp, seq, capture_fps = (
                        packet.left,
                        packet.right,
                        packet.timestamp,
                        packet.seq,
                        packet.capture_fps,
                    )
                    if previous_seq >= 0:
                        skipped += max(0, seq - previous_seq - 1)
                    previous_seq = seq
                else:
                    left, right = offline[frame_no % len(offline)]
                    stamp, seq, capture_fps = time.monotonic(), frame_no, 0.0
                if pending is not None and pending["future"].done():
                    depth = finish_depth(pending, model.names, args)
                    pending, updated = None, True
                    depth_totals += (1, depth["stereo_ms"])
                    depth_stamps.append(depth["stamp"])
                    log_depth(depth_log, depth)
                start = time.perf_counter()
                color = geometry.rectify_left(left)
                scheduled = pending is None and (args.depth_fps == 0 or time.monotonic() >= next_depth)
                if scheduled:
                    pending = dict(
                        future=pool.submit(matcher.compute_frame, color, right),
                        seq=seq,
                        stamp=stamp,
                        boxes=None,
                    )
                    next_depth = (
                        time.monotonic() + 1 / args.depth_fps
                        if args.depth_fps not in ("max", 0)
                        else 0.0
                    )
                enhance_start = time.perf_counter()
                detector_input = enhancer.apply(color)
                enhancement_ms = 1000 * (time.perf_counter() - enhance_start)
                detect_start = time.perf_counter()
                with torch.inference_mode():
                    result = model.predict(detector_input, **predict)[0]
                    boxes = (
                        result.boxes.data.cpu().numpy().copy()
                        if result.boxes is not None
                        else np.empty((0, 6))
                    )
                yolo_ms = 1000 * (time.perf_counter() - detect_start)
                if scheduled:
                    pending["boxes"] = boxes
                if args.depth_fps == 0:
                    depth = finish_depth(pending, model.names, args)
                    pending, updated = None, True
                    depth_totals += (1, depth["stereo_ms"])
                    depth_stamps.append(depth["stamp"])
                    log_depth(depth_log, depth)
                depth_age_ms = max(0.0, 1000 * (time.monotonic() - depth["stamp"])) if depth else None
                usable = depth is not None and depth_age_ms <= args.depth_max_age * 1000
                # No target association is implied: historical XYZ stays with its original frame/boxes.
                found = (
                    depth["found"]
                    if usable and depth["seq"] == seq
                    else observations(boxes, model.names, args.confidence)
                )
                records = detection_records(found, geometry.size, args.confidence)
                age_ms = max(0.0, 1000 * (time.monotonic() - stamp))
                fps = frame_rate(stamps)
                canvas = compose_view(
                    detector_input,
                    found,
                    geometry,
                    args,
                    fps,
                    recorder.active,
                    depth=depth if usable else None,
                    depth_fps=frame_rate(depth_stamps),
                    depth_age_ms=depth_age_ms,
                    detection_age_ms=age_ms,
                )
                if frame_no in (0, 29):
                    cv2.imencode(".png", canvas)[1].tofile(str(run / "preview.png"))
                if frame_no == 29:
                    cv2.imencode(".png", color)[1].tofile(str(run / "left_rectified_1280x720.png"))
                    cv2.imencode(".png", detector_input)[1].tofile(str(run / "detection_input_1280x720.png"))
                    if depth:
                        for name, key in (
                            ("estimated", "estimated"),
                            ("measured", "measured"),
                            ("display_ONLY", "display_depth"),
                        ):
                            np.save(
                                run / ("depth_" + name + "_f%d_m.npy" % depth["seq"]), host_array(depth[key])
                            )
                if show:
                    preview.show(canvas)
                if args.record_offline and frame_no == 0:
                    recorder.start(canvas, time.perf_counter())
                elif recorder.active:
                    recorder.submit(canvas, time.perf_counter())
                frame_no += 1
                elapsed = 1000 * (time.perf_counter() - start)
                stereo_ms = depth["stereo_ms"] if updated else None
                row = [
                    frame_no,
                    seq,
                    stamp,
                    age_ms,
                    capture_fps,
                    fps,
                    elapsed,
                    stereo_ms,
                    yolo_ms,
                    depth["valid_fraction"] if depth else None,
                    len(found),
                    skipped,
                    enhancement_ms,
                    depth["seq"] if depth else None,
                    depth_age_ms,
                    updated,
                ]
                writer.writerow(row)
                detections_log.write(
                    json.dumps(
                        {
                            "frame": frame_no,
                            "capture_seq": seq,
                            "capture_monotonic_s": stamp,
                            "source_age_ms": age_ms,
                            "objects": records,
                            "depth_frame_id": depth["seq"] if depth else None,
                            "depth_age_ms": depth_age_ms,
                            "depth_fresh": usable,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                if frame_no % 30 == 0:
                    log.flush()
                    detections_log.flush()
                    depth_log.flush()
                    print(
                        "Detection %.1f FPS | capture %.1f FPS | depth %.1f Hz"
                        % (fps, capture_fps, frame_rate(depth_stamps)),
                        flush=True,
                    )
                stamps.append(time.perf_counter())
                totals += (1, elapsed, enhancement_ms)
                if frame_no <= 10:
                    warmup_totals += (1, elapsed, enhancement_ms)
            if pending is not None:
                depth = finish_depth(pending, model.names, args)
                depth_totals += (1, depth["stereo_ms"])
                depth_stamps.append(depth["stamp"])
                log_depth(depth_log, depth)
    except BaseException as exc:
        error = repr(exc)
        raise
    finally:
        try:
            recorder.close()
        finally:
            if camera:
                camera.close()
            if timer:
                ctypes.WinDLL("winmm").timeEndPeriod(1)
            if pool:
                pool.shutdown(wait=True)
            if preview:
                preview.close()
            stable = totals - warmup_totals if frame_no > 10 else totals
            mean_ms = float(stable[1] / stable[0]) if stable[0] else None
            summary = {
                "processed_frames": frame_no,
                "mode": meta["mode"],
                "error": error,
                "run_wall_s": time.perf_counter() - began,
                "skipped_camera_frames": skipped,
                "mean_processing_ms_excluding_first10": mean_ms,
                "processing_capacity_fps": 1000 / mean_ms if mean_ms else None,
                "last_window_fresh_fps": frame_rate(stamps),
                "depth_updates": int(depth_totals[0]),
                "depth_update_hz": frame_rate(depth_stamps),
                "mean_stereo_ms": float(depth_totals[1] / depth_totals[0]) if depth_totals[0] else None,
                "live_FPS_verified": bool(args.live and frame_no >= 120),
                "distance_accuracy_verified": geometry.verified,
                "input_enhancement": enhancer.metadata(),
                "mean_enhancement_ms": float(stable[2] / stable[0]) if stable[0] else None,
                "recording": str(recorder.path) if recorder.path else None,
                "note": "Detection FPS and depth update Hz are separate; encoded holds are not new observations.",
            }
            save_json(run / "summary.json", summary)
            print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
