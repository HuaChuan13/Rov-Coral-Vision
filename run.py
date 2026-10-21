"""Portable launcher; keep models/ and config/ next to this file.

Run with uv run python run.py. Q records; W saves/exits.
"""
import sys
from pathlib import Path
from coral_live import main

if __name__ == '__main__':
    folder = Path(__file__).resolve().parent
    raise SystemExit(main([
        '--live', '--source', 'sbs', '--camera', '1', '--depth-only-panel', '--confidence', '0.6', '--half',
        '--model', str(folder/'models/best.pt'),
        '--calibration', str(folder/'config/stereo_opencv.yaml'),
        *sys.argv[1:]]))
