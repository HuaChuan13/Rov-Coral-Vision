"""Real vision/Controller UDP + compiled firmware modules; simulated camera, IMU and USB only."""
import argparse
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

from run import parse_args
from vision import TargetObservation, ObservationOutput


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--firmware-sim", type=Path, required=True)
    cli.add_argument("--controller", type=Path, default=Path(__file__).resolve().parent.parent / "Controller")
    args = cli.parse_args()
    sys.path.insert(0, str(args.controller))
    from vision_bridge import VisionBridge

    config = SimpleNamespace(vision_port=0, vision_gain=10., vision_limit=5., vision_sign=-1,
                             vision_deadband=.04, vision_exit_deadband=.06)
    with patch("vision_bridge.time.monotonic", return_value=100.):
        bridge = VisionBridge(config)
    process = subprocess.Popen([str(args.firmware_sim.resolve()), "--vision-sim"],
                               stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
    observer = TargetObservation((1280, 720), parse_args([]), 797.55)
    record = dict(class_id=0, confidence=.9, box_rectified_xyxy=[700., 200., 860., 400.], **{"class": "Coral"})
    packets, headings = {}, {}
    frame, sequence = 0, 0
    try:
        with tempfile.TemporaryDirectory() as directory:
            output = ObservationOutput(Path(directory), bridge.socket.getsockname()[1])
            try:
                for tick in range(241):
                    now = 100. + tick * .005
                    # Camera at 20Hz; synthetic loss/recovery exercises the production observer.
                    if tick % 10 == 0 and tick < 200:
                        missing = tick in (20, 30)
                        observer.update([] if missing else [record], frame, now, now)
                        frame += 1
                        if tick in (10, 110, 150, 190):
                            observer.select([record], (750, 300), now, now)
                        if tick == 80:
                            observer.cancel()
                        p = observer.packet(now)
                        output.publish(p)
                    # Buttons only press Circle for one send; no implicit resume on release.
                    buttons = (1 << 1) if tick == 120 else 0
                    with patch("vision_bridge.time.monotonic", return_value=now):
                        bridge.receive(now, buttons)
                    line = "-"
                    if tick % 4 == 0:
                        sequence += 1
                        line = f"PKT {sequence} 0 0 0 0 0 0 {buttons}" + bridge.suffix(sequence, now)
                    mode = 1 if tick in (0, 184) else 0 if tick == 180 else -1
                    # Last section freezes image observations while IMU/manual packets keep arriving.
                    process.stdin.write(f"{round(now * 1000)} {mode} 0 {line}\n")
                    process.stdin.flush()
                    vis = process.stdout.readline().strip()
                    sim = process.stdout.readline().strip()
                    assert vis.startswith("VIS{") and sim.startswith("SIM{"), (vis, sim)
                    if tick % 4 == 0:
                        bridge.telemetry([vis], now)
                    fields = dict(item.split("=", 1) for item in vis[4:-1].split())
                    headings[tick] = float(dict(item.split("=", 1) for item in sim[4:-1].split())["yaw_sp"])
                    packets[tick] = int(fields["state"])
                    bridge.reply(now)
                    if tick == 64:
                        control = output.control_status(p, now)
                        assert control and control["confirmed"] and control["status"] == "ACTIVE", control
                assert packets[16] == 1 and headings[19] < 0, "Click did not drive negative heading slew through both loops"
                assert packets[24] == 2 and headings[29] == headings[24], "WAITING did not hold heading"
                assert packets[56] == 1, "Two-frame recovery did not resume original click"
                assert packets[84] == 0, "Cancel did not disable firmware vision"
                assert packets[116] == 1 and packets[124] in (0, 4)
                assert packets[140] not in (1, 2, 3), "Manual release silently rearmed"
                assert packets[156] == 1 and packets[184] not in (1, 2, 3), "Mode change silently rearmed"
                assert packets[196] == 1 and packets[240] not in (1, 2, 3), "Camera freeze did not expire assistance"
                print("PASS: vision selection -> real UDP -> Controller PKT -> compiled firmware -> accepted UI status;")
                print("      click, heading direction, waiting/recovery, cancel, manual override and mode re-entry")
            finally:
                output.close()
    finally:
        bridge.close()
        process.stdin.close()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
    assert process.returncode == 0


if __name__ == "__main__":
    main()
