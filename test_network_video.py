"""uv run --locked python test_network_video.py [--integration]; no serial/control IO."""

import argparse
import io
import socket
import subprocess
import time

import psutil

from camera import Camera
from network_video import StreamClock, read_frame, sdp_text, gst_metadata, write_packet
from run import parse_args
from test_vision import record
from vision import TargetObservation


def check_clock_and_lock():
    startup = StreamClock()
    startup.stamp(0., 10.)
    assert startup.stamp(.316, 10.004) <= 10.004  # Legitimate cold-start burst.
    clock = StreamClock()
    assert clock.stamp(0., 10.) == 10.
    assert abs(clock.stamp(.1, 10.12) - 10.1) < 1e-9
    # A frame arriving .5 seconds late must not acquire a new "fresh" timestamp.
    delayed = clock.stamp(.2, 10.7)
    assert abs(delayed - 10.2) < 1e-9
    tracker = TargetObservation((1280, 720), parse_args([]), 640.)
    tracker.update([record()], 0, 10., 10.01)
    tracker.select([record()], (750, 300), 10., 10.02)
    assert tracker.packet(10.03)["valid"]
    tracker.update([record()], 1, delayed, 10.7)
    assert not tracker.packet(10.7)["valid"]
    tracker.update([record()], 2, clock.stamp(.8, 10.8), 10.81)
    assert not tracker.packet(10.81)["valid"], "Recovery must require a new click"
    for bad in (.8, .7, float("nan"), float("inf")):
        try:
            clock.stamp(bad, 11.)
        except ValueError:
            pass
        else:
            raise AssertionError("Accepted duplicate/reversed/invalid PTS")
    try:
        clock.stamp(100., 11.)
    except ValueError:
        pass
    else:
        raise AssertionError("Accepted a forward discontinuity as a fresh frame")


def check_short_reads():
    class Pipe(io.BytesIO):
        def readinto(self, view):
            return super().readinto(view[:2])
    assert read_frame(Pipe(b"abcdef"), 6) == b"abcdef"
    assert read_frame(Pipe(b"abcde"), 6) is None
    class Writer(io.BytesIO):
        def write(self, value):
            return super().write(value[:3])
    writer = Writer()
    write_packet(writer, b"RTP DATA")
    assert writer.getvalue() == b"\x00\x08RTP DATA"
    line = '<stamps> chain (6144 bytes, dts: none, pts: 0:01:02.125000000, duration: none, flags: 00000040 discont)'
    assert gst_metadata(line, 6144) == 62.125
    for invalid in (line.replace('6144', '6143'), line.replace('00000040', '00000100'),
                    line.replace('0:01:02.125000000', 'none')):
        try:
            gst_metadata(invalid, 6144)
        except ValueError:
            pass
        else:
            raise AssertionError('Accepted wrong size, corrupted frame or missing PTS')
    assert "H264/90000" in sdp_text("192.168.10.2", 5600)
    try:
        sdp_text("192.168.10.2\nm=bad", 5600)
    except ValueError:
        pass
    else:
        raise AssertionError("Accepted SDP injection")


def free_port():
    for port in range(24560, 24660, 2):
        sockets = [socket.socket(socket.AF_INET, socket.SOCK_DGRAM) for _ in range(2)]
        try:
            for sock, offset in zip(sockets, (0, 1)):
                sock.bind(("0.0.0.0", port + offset))
            return port
        except OSError:
            pass
        finally:
            for sock in sockets:
                sock.close()
    raise RuntimeError("No free test RTP/RTCP ports")


def check_integration(backend):
    port = free_port()
    sender = subprocess.Popen([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-re",
        "-f", "lavfi", "-i", "testsrc2=size=64x32:rate=30",
        "-an", "-c:v", "libx264", "-preset", "ultrafast", "-tune", "zerolatency",
        "-g", "10", "-bf", "0", "-threads", "1", "-f", "rtp",
        "rtp://127.0.0.1:%d?pkt_size=1200" % port,
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
       creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    camera = None
    try:
        try:
            Camera(width=128, height=32, stream_host="127.0.0.1", stream_port=port, stream_backend=backend)
        except RuntimeError as exc:
            assert "Expected 128x32 SBS" in str(exc) or "not-negotiated" in str(exc), str(exc)
        else:
            raise AssertionError("Wrong stereo dimensions were silently accepted")
        camera = Camera(width=64, height=32, stream_host="127.0.0.1", stream_port=port, stream_backend=backend)
        first = camera.latest()
        assert first.left.shape == first.right.shape == (32, 32, 3)
        if backend == "gstreamer":
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as noise:
                for bad in (b'bad', b'\x00\x60' + bytes(22), b'\x80\x61' + bytes(22),
                            b'\x8f\x60' + bytes(22)):
                    noise.sendto(bad, ('127.0.0.1', port))
        # The slow consumer must skip, not queue or back-pressure the decoder.
        time.sleep(.5)
        latest = camera.latest()
        assert latest.seq >= first.seq + 8
        assert time.monotonic() - latest.timestamp < .2
        # A real transport gap cancels a selected lock, even if the image returns.
        tracker = TargetObservation((1280, 720), parse_args([]), 640.)
        tracker.update([record()], latest.seq, latest.timestamp, time.monotonic())
        tracker.select([record()], (750, 300), latest.timestamp, time.monotonic())
        assert tracker.packet(time.monotonic())["valid"]
        psutil.Process(sender.pid).suspend()
        time.sleep(.4)
        assert not tracker.packet(time.monotonic())["valid"]
        psutil.Process(sender.pid).resume()
        time.sleep(.5)
        recovered = camera.latest()
        assert recovered.seq > latest.seq
        tracker.update([record()], recovered.seq, recovered.timestamp, time.monotonic())
        assert not tracker.packet(time.monotonic())["valid"]
        sender.terminate()
        sender.wait(timeout=3)
        deadline = time.monotonic() + 7
        while time.monotonic() < deadline:
            try:
                camera.latest()
            except RuntimeError:
                break
            time.sleep(.05)
        else:
            raise AssertionError("A stopped stream never timed out")
    finally:
        if camera:
            camera.close()
            assert not camera.process.is_alive()
        if sender.poll() is None:
            sender.kill()
        sender.wait(timeout=3)
    # A new session can reuse the port after decoder shutdown.
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.bind(("0.0.0.0", port))
    finally:
        sock.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--integration", action="store_true")
    args = parser.parse_args()
    check_clock_and_lock()
    check_short_reads()
    if args.integration:
        for backend in ("gstreamer", "ffmpeg"):
            check_integration(backend)
    print("Network video checks passed" + (" (GStreamer/FFmpeg RTP, timeout, cleanup)" if args.integration else ""))
