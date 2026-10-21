"""Admission checks for formal stereo pairs; raw candidates are separate."""
from collections import deque

import numpy as np

from imaging import corner_distance


class FrameRateTracker:
    """Measure decoded camera arrivals, independently of preview/detection FPS.

    A host arrival timestamp does not establish hardware exposure synchronization.
    The camera reader supplies every frame, including ones the UI never displays.
    """
    def __init__(self, window_seconds=3.):
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
        span = float(times[-1] - times[0]) if len(times) > 1 else 0.
        intervals = np.diff(times)
        return {'measured_arrival_fps': float((len(times)-1)/span) if span > 0 else 0.,
                'window_span_s': span, 'frame_count': len(times),
                'warmed_up': span >= self.window_seconds - 1e-6,
                'max_gap_s': float(intervals.max()) if len(intervals) else None,
                'latest_age_s': max(0., float(now-times[-1])) if len(times) else None,
                'measurement': 'decoded_frame_arrivals_not_driver_report_or_gui_refresh'}


def rate_ok(health, minimum_fps=25., maximum_gap=.15, maximum_age=.25):
    if not health or not health.get('warmed_up'):
        return False, 'FPS warm-up: wait for 3 seconds of camera frames'
    fps = health.get('measured_arrival_fps', 0.)
    if not np.isfinite(fps) or fps < minimum_fps:
        return False, 'FPS %.1f < %.1f: official capture paused' % (fps, minimum_fps)
    gap, age = health.get('max_gap_s'), health.get('latest_age_s')
    if gap is None or not np.isfinite(gap) or gap > maximum_gap:
        return False, 'Camera frame delivery stalled; wait for stable FPS'
    if age is None or not np.isfinite(age) or age > maximum_age:
        return False, 'Camera frame is stale; official capture paused'
    return True, 'FPS ready'


class PairAdmission:
    """A manual request never bypasses these checks.

    Stable means multiple independently inspected pairs within a displacement
    tolerance of a fixed anchor; failed checks or long detection gaps reset it.
    """
    def __init__(self, size, minimum_fps=25., stable_seconds=2., motion_px=1.,
                 novelty=.025, minimum_checks=3, max_check_gap=1.5,
                 max_result_age=1.5, interval=1.2, policy='stable', pattern=(11, 8)):
        if policy not in ('stable', 'detected'):
            raise ValueError('Unknown capture policy')
        self.policy, self.pattern = policy, tuple(pattern)
        self.size = tuple(size)
        self.minimum_fps = float(minimum_fps)
        self.stable_seconds = float(stable_seconds)
        self.motion_px = float(motion_px)
        self.novelty = float(novelty)
        self.minimum_checks = int(minimum_checks)
        self.max_check_gap = float(max_check_gap)
        self.max_result_age = float(max_result_age)
        self.interval = float(interval)
        self.anchor = None
        self.since = None
        self.previous_timestamp = None
        self.previous_seq = None
        self.checks = 0
        self.last_accepted_time = -np.inf

    def reset_stability(self):
        self.anchor = None
        self.since = None
        self.checks = 0

    def evaluate(self, packet, results, current_health, accepted, now, mode_ok=True):
        decision = {'ok': False, 'reason': '', 'stable_s': 0., 'stable_checks': self.checks,
                    'camera_health_at_frame': getattr(packet, 'capture_health', None),
                    'camera_health_at_check': current_health,
                    'result_age_s': max(0., now-packet.timestamp),
                    'quality_policy': ('detected_corners_fps_v1' if self.policy == 'detected'
                                       else 'strict_fps_complete_clear_stable_diverse_v2')}

        def reject(reason, reset=True):
            decision['reason'] = reason
            if reset:
                self.reset_stability()
            return decision

        if not mode_ok:
            return reject('Resolution mismatch: use requested native mode before saving')
        for health in (getattr(packet, 'capture_health', None), current_health):
            ok, reason = rate_ok(health, self.minimum_fps)
            if not ok:
                return reject(reason)
        if now-packet.timestamp > self.max_result_age or now < packet.timestamp:
            return reject('Detection result too old; reduce processing load')
        if packet.seq == self.previous_seq:
            return reject('Repeated frame; waiting for a new pair', reset=False)
        if (self.previous_timestamp is not None and
                (packet.timestamp <= self.previous_timestamp or
                 packet.timestamp-self.previous_timestamp > self.max_check_gap)):
            self.reset_stability()
        self.previous_seq, self.previous_timestamp = packet.seq, packet.timestamp
        if self.policy == 'detected':
            for name in ('left', 'right'):
                points = np.asarray(results.get(name, {}).get('corners', []))
                if (points.shape != (self.pattern[0]*self.pattern[1], 1, 2) or
                        not np.isfinite(points).all()):
                    return reject(name + ': waiting for all %d corners' % (self.pattern[0]*self.pattern[1]))
                if ((points[..., 0] < 0).any() or (points[..., 0] >= self.size[0]).any() or
                        (points[..., 1] < 0).any() or (points[..., 1] >= self.size[1]).any()):
                    return reject(name + ': corners outside image')
            if now-self.last_accepted_time < self.interval:
                return reject('Both boards detected; next automatic save shortly', reset=False)
            decision.update(ok=True, reason='Both full corner sets detected: saving original pair',
                            strict_image_quality_passed=all(results[s].get('ok', False) for s in ('left', 'right')))
            return decision
        for name in ('left', 'right'):
            if name not in results or not results[name].get('ok') or 'corners' not in results[name]:
                return reject(name + ': ' + results.get(name, {}).get('reason', 'quality not checked'))
        current = {name: np.asarray(results[name]['corners']) for name in ('left', 'right')}
        if self.anchor is not None:
            movement = max(float(np.percentile(np.linalg.norm(
                current[name].reshape(-1, 2)-self.anchor[name].reshape(-1, 2), axis=1), 95))
                for name in current)
            if movement > self.motion_px:
                self.reset_stability()
        if self.anchor is None:
            self.anchor = {name: points.copy() for name, points in current.items()}
            self.since = packet.timestamp
        self.checks += 1
        held = packet.timestamp-self.since
        decision.update(stable_s=held, stable_checks=self.checks)
        if held < self.stable_seconds or self.checks < self.minimum_checks:
            return reject('Board clear: hold still %.1fs (%d quality checks)' %
                          (max(0., self.stable_seconds-held), self.checks), reset=False)
        diverse = all(max(corner_distance(current[name], sample['corners'][name], self.size)
                          for name in current) >= self.novelty for sample in accepted)
        if not diverse:
            return reject('Pose already saved: change position, distance or horizontal/vertical tilt', reset=False)
        if now-self.last_accepted_time < self.interval:
            return reject('Waiting between accepted poses', reset=False)
        decision.update(ok=True, reason='Qualified pair: FPS / complete / sharp / stable / new pose')
        return decision

    def accepted(self, now):
        self.last_accepted_time = now
        self.reset_stability()
