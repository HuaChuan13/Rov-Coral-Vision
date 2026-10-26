"""Bounded RTP/H.264 reception; GStreamer avoids FFmpeg's extra frame wait."""

import ipaddress
import io
import math
import multiprocessing as mp
import os
import queue
import re
import shutil
import socket
import struct
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
        self.started = None

    def stamp(self, pts, received):
        if not math.isfinite(pts) or not math.isfinite(received):
            raise ValueError("Missing/non-finite video timestamp")
        if self.previous is not None and pts <= self.previous:
            raise ValueError("Video timestamp restarted or went backwards; restart receiver and select again")
        if self.started is None:
            self.started = received
        if self.offset is not None and received - self.started >= 1 and pts + self.offset > received + .2:
            raise ValueError("Video timestamp jumped forward; restart receiver and select again")
        self.previous = pts
        self.offset = received - pts if self.offset is None else min(self.offset, received - pts)
        return min(received, pts + self.offset)


def validate_network(host, port, executable, backend="ffmpeg"):
    ipaddress.IPv4Address(host)
    if not 1024 <= port <= 65534:
        raise ValueError("RTP port must be 1024..65534 (RTP also reserves port+1 for RTCP)")
    if backend not in ("ffmpeg", "gstreamer"):
        raise ValueError("Unknown stream backend")
    executable = shutil.which(executable)
    if executable is None:
        raise ValueError("%s not found; install the runtime or specify its executable path" % backend)
    return executable


def gstreamer_command(executable, width, height):
    # RFC4571 preserves UDP packet boundaries across the stdin byte stream.
    # Python filters the peer IP; only this decoder owns the pipe and its frames.
    return [executable, "-q", "fdsrc", "fd=0", "do-timestamp=true", "!",
            "application/x-rtp-stream,media=video,encoding-name=H264,clock-rate=90000,payload=96", "!",
            "rtpstreamdepay", "!", "rtpjitterbuffer", "mode=none", "latency=10", "drop-on-latency=true", "do-lost=true", "!",
            "rtph264depay", "wait-for-keyframe=true", "request-keyframe=true", "!",
            "video/x-h264,alignment=au", "!", "h264parse", "!",
            "avdec_h264", "max-threads=1", "output-corrupt=false", "!", "videoconvert", "!",
            "video/x-raw,format=BGR,width=%d,height=%d" % (width, height), "!",
            "identity", "name=stamps", "silent=false", "!", "fdsink", "fd=1", "sync=false"]


def gst_metadata(line, frame_bytes):
    """The CLI trace supplies PTS for exactly the buffer sent to stdout."""
    if "<stamps> chain" not in line:
        return None
    match = re.search(r"\((\d+) bytes,.*?pts: (\d+):(\d+):(\d+\.\d+).*?flags: ([0-9a-fA-F]+)", line)
    if not match:
        raise ValueError("Missing GStreamer buffer timestamp")
    size, hours, minutes, seconds, flags = match.groups()
    if int(size) != frame_bytes or int(flags, 16) & 0x100:  # GST_BUFFER_FLAG_CORRUPTED
        raise ValueError("Wrong size or corrupted decoded video buffer")
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)


def write_packet(stream, packet):
    view = memoryview(struct.pack("!H", len(packet)) + packet)
    while view:
        count = stream.write(view)
        if not count:
            raise BrokenPipeError("Decoder input pipe closed")
        view = view[count:]


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

    The decoder drains the socket continuously. The caller publishes into the existing
    one-frame shared buffer, independently of detection/depth/rendering.
    """
    backend = options.get("stream_backend", "ffmpeg")
    executable = validate_network(options["stream_host"], options["stream_port"],
                                  options["gstreamer"] if backend == "gstreamer" else options["ffmpeg"], backend)
    width, height = options["width"], options["height"]
    metadata, errors = queue.Queue(maxsize=4), deque(maxlen=8)
    finished = threading.Event()
    clock = StreamClock()
    info = re.compile(r"\bn:\s*(\d+).*?\bpts_time:\s*(\S+).*?\bs:(\d+)x(\d+)")
    with tempfile.TemporaryDirectory(prefix="coral-rtp-") as folder:
        sdp = Path(folder) / "camera.sdp"
        sdp.write_text(sdp_text(options["stream_host"], options["stream_port"]), encoding="ascii")
        udp = None
        if backend == "gstreamer":
            udp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            try:
                udp.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 262144)
                udp.bind(("0.0.0.0", options["stream_port"]))
                udp.settimeout(.1)
            except BaseException:
                udp.close()
                raise
        env = os.environ.copy()
        if backend == "gstreamer":
            env.update(GST_DEBUG="identity:7", GST_DEBUG_NO_COLOR="1")
        try:
            process = subprocess.Popen(
                gstreamer_command(executable, width, height) if backend == "gstreamer" else receiver_command(executable, sdp),
                stdin=subprocess.PIPE if udp else subprocess.DEVNULL,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0, env=env,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        except BaseException:
            if udp:
                udp.close()
            raise
        process.stderr = io.BufferedReader(process.stderr)

        def read_metadata():
            try:
                gst_seq = 0
                for raw in process.stderr:
                    line = raw.decode("utf-8", errors="replace").strip()
                    if backend == "gstreamer":
                        pts = gst_metadata(line, width * height * 3)
                        if pts is not None:
                            metadata.put_nowait((gst_seq, pts))
                            gst_seq += 1
                        elif line:
                            errors.append(line)
                        continue
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

        def feed_rtp():
            ssrc = None
            try:
                while not finished.is_set() and not stop.is_set():
                    try:
                        data, peer = udp.recvfrom(65536)
                    except socket.timeout:
                        continue
                    if (peer[0] != options["stream_host"] or len(data) < 12
                            or data[0] >> 6 != 2 or data[1] & 127 != 96
                            or len(data) < 12 + 4 * (data[0] & 15)):
                        continue
                    if ssrc is not None and data[8:12] != ssrc:
                        raise ValueError("RTP sender restarted; restart receiver and select again")
                    ssrc = data[8:12]
                    write_packet(process.stdin, data)
            except (OSError, ValueError) as exc:
                if not finished.is_set() and not stop.is_set():
                    errors.append(str(exc))
                    process.kill()

        def watch_stop():
            parent = mp.parent_process()
            while not finished.wait(.1):
                if stop.is_set() or parent is not None and not parent.is_alive():
                    process.kill()  # Decoder owns no output file; unblock a stalled pipe read.
                    return

        readers = [threading.Thread(target=read_metadata, daemon=True),
                   threading.Thread(target=watch_stop, daemon=True)]
        if udp:
            readers.append(threading.Thread(target=feed_rtp, daemon=True))
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
                # Cold-start packets may arrive in a burst. Establish the clock
                # before publishing any of them as selectable camera frames.
                if received - clock.started < 1:
                    continue
                yield np.frombuffer(data, np.uint8).reshape(height, width, 3), stamp, received, pts
        finally:
            finished.set()
            if process.poll() is None:
                process.kill()
            process.wait(timeout=2)
            for thread in readers:
                thread.join(timeout=1)
            if udp:
                udp.close()
                process.stdin.close()
            process.stdout.close()
            process.stderr.close()
