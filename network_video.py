"""Bounded RTP/H.264 receiver using the installed FFmpeg executable."""

import ipaddress
import math
import multiprocessing as mp
import queue
import re
import shutil
import subprocess
import tempfile
import threading
import time
from collections import deque
from pathlib import Path

import numpy as np


class StreamClock:
    """Map PTS to the earliest observed arrival baseline, never hide added delay.

    This measures delay GROWTH, not camera exposure age. Constant upstream delay
    at startup remains unknown and must be measured on the actual bench.
    """

    def __init__(self):
        self.offset = None
        self.previous = None

    def stamp(self, pts, received):
        if not math.isfinite(pts) or not math.isfinite(received):
            raise ValueError("Missing/non-finite video timestamp")
        if self.previous is not None and pts <= self.previous:
            raise ValueError("Video timestamp restarted or went backwards; restart receiver and select again")
        if self.offset is not None and pts + self.offset > received + .2:
            raise ValueError("Video timestamp jumped forward; restart receiver and select again")
        self.previous = pts
        self.offset = received - pts if self.offset is None else min(self.offset, received - pts)
        return min(received, pts + self.offset)


def validate_network(host, port, ffmpeg):
    ipaddress.IPv4Address(host)
    if not 1024 <= port <= 65534:
        raise ValueError("RTP port must be 1024..65534 (RTP also reserves port+1 for RTCP)")
    executable = shutil.which(ffmpeg)
    if executable is None:
        raise ValueError("FFmpeg not found; install FFmpeg or specify --ffmpeg PATH")
    return executable


def sdp_text(host, port):
    ipaddress.IPv4Address(host)
    return (
        "v=0\no=- 0 0 IN IP4 127.0.0.1\ns=ROV stereo\n"
        "c=IN IP4 0.0.0.0\nt=0 0\n"
        "a=source-filter: incl IN IP4 0.0.0.0 %s\n"
        "m=video %d RTP/AVP 96\na=rtpmap:96 H264/90000\n"
        "a=fmtp:96 packetization-mode=1\na=recvonly\n"
    ) % (host, port)


def receiver_command(executable, sdp):
    return [
        executable, "-hide_banner", "-nostdin", "-nostats", "-loglevel", "info",
        "-protocol_whitelist", "file,udp,rtp", "-fflags", "+nobuffer+discardcorrupt",
        "-flags", "+low_delay", "-err_detect", "explode", "-threads", "1",
        "-max_delay", "30000", "-reorder_queue_size", "64", "-buffer_size", "262144",
        "-analyzeduration", "1000000", "-probesize", "1000000",
        "-i", str(sdp), "-map", "0:v:0", "-an", "-sn", "-dn",
        "-filter_threads", "1", "-vf", "format=bgr24,showinfo=checksum=0",
        "-fps_mode", "passthrough", "-c:v", "rawvideo", "-threads", "1",
        "-f", "rawvideo", "pipe:1",
    ]


def read_frame(stream, size):
    """Pipes may split frames across reads; an incomplete frame is never valid."""
    data = bytearray(size)
    view = memoryview(data)
    position = 0
    while position < size:
        count = stream.readinto(view[position:])
        if not count:
            return None
        position += count
    return data


def network_frames(options, stop):
    """Yield complete BGR frames and matching PTS; always reap our own decoder.

    FFmpeg drains the socket continuously. The caller publishes into the existing
    one-frame shared buffer, independently of detection/depth/rendering.
    """
    executable = validate_network(options["stream_host"], options["stream_port"], options["ffmpeg"])
    width, height = options["width"], options["height"]
    metadata, errors = queue.Queue(maxsize=4), deque(maxlen=8)
    finished = threading.Event()
    clock = StreamClock()
    info = re.compile(r"\bn:\s*(\d+).*?\bpts_time:\s*(\S+).*?\bs:(\d+)x(\d+)")
    with tempfile.TemporaryDirectory(prefix="coral-rtp-") as folder:
        sdp = Path(folder) / "camera.sdp"
        sdp.write_text(sdp_text(options["stream_host"], options["stream_port"]), encoding="ascii")
        process = subprocess.Popen(
            receiver_command(executable, sdp), stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )

        def read_metadata():
            try:
                for raw in process.stderr:
                    line = raw.decode("utf-8", errors="replace").strip()
                    match = info.search(line) if "showinfo" in line else None
                    if match:
                        n, pts, w, h = match.groups()
                        if (int(w), int(h)) != (width, height):
                            raise ValueError("Expected %dx%d SBS, received %sx%s; no automatic resizing" % (width, height, w, h))
                        metadata.put_nowait((int(n), float(pts)))
                    elif "showinfo" not in line and line:
                        errors.append(line)
            except (ValueError, queue.Full) as exc:
                errors.append("Invalid video metadata: " + str(exc))
                process.kill()

        def watch_stop():
            parent = mp.parent_process()
            while not finished.wait(.1):
                if stop.is_set() or parent is not None and not parent.is_alive():
                    process.kill()  # Decoder owns no output file; unblock a stalled pipe read.
                    return

        readers = [threading.Thread(target=read_metadata, daemon=True),
                   threading.Thread(target=watch_stop, daemon=True)]
        for thread in readers:
            thread.start()
        try:
            expected = 0
            while not stop.is_set():
                data = read_frame(process.stdout, width * height * 3)
                received = time.monotonic()
                if data is None:
                    if stop.is_set():
                        return
                    raise RuntimeError("Network decoder stopped: " + " | ".join(errors))
                try:
                    n, pts = metadata.get(timeout=1)
                except queue.Empty:
                    raise RuntimeError("No timestamp for decoded video frame") from None
                if n != expected:
                    raise RuntimeError("Video frame/timestamp sequence mismatch")
                expected += 1
                stamp = clock.stamp(pts, received)
                yield np.frombuffer(data, np.uint8).reshape(height, width, 3), stamp, received, pts
        finally:
            finished.set()
            if process.poll() is None:
                process.kill()
            process.wait(timeout=2)
            for thread in readers:
                thread.join(timeout=1)
            process.stdout.close()
            process.stderr.close()
