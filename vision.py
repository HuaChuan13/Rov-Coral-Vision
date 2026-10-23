"""One manually selected target; localhost observations, never serial motor commands."""

import json
import math
import socket
import uuid


def yaw_slew_preview(error, gain, deadband, limit, sign):
    """Continuous deadband; heading-setpoint slew, NOT the body yaw-rate target."""
    magnitude = min(limit, gain * max(0.0, abs(error) - deadband))
    return sign * math.copysign(magnitude, error) if magnitude else 0.0


def valid_targets(records, size, confidence):
    result = []
    for record in records:
        box = record.get("box_rectified_xyxy", [])
        score, cls = record.get("confidence", 0), record.get("class_id")
        if (len(box) != 4 or not all(math.isfinite(v) for v in (*box, score))
                or not confidence <= score <= 1 or type(cls) is not int or cls < 0):
            continue
        x1, y1, x2, y2 = box
        x1, x2 = max(0., x1), min(float(size[0]), x2)
        y1, y2 = max(0., y1), min(float(size[1]), y2)
        if x2 > x1 and y2 > y1:
            result.append(dict(record, box_rectified_xyxy=[x1, y1, x2, y2]))
    return result


def tracking_records(boxes, names, size, confidence):
    """Low-score candidates stay in tracking; display/XYZ keep their higher threshold."""
    records = []
    for row in boxes:
        if len(row) != 6 or not all(math.isfinite(v) for v in row) or row[5] != int(row[5]):
            continue
        cls = int(row[5])
        if cls not in names:
            continue
        records.append(dict(class_id=cls, confidence=float(row[4]),
                            box_rectified_xyxy=[float(v) for v in row[:4]], **{"class": names[cls]}))
    return valid_targets(records, size, confidence)


def match_metrics(a, b):
    ax, ay, ar, ab = a
    bx, by, br, bb = b
    overlap = max(0, min(ar, br) - max(ax, bx)) * max(0, min(ab, bb) - max(ay, by))
    union = (ar - ax) * (ab - ay) + (br - bx) * (bb - by) - overlap
    motion = math.hypot((ax + ar - bx - br) / 2, (ay + ab - by - bb) / 2)
    return overlap / union, motion / max(1., math.hypot(ar - ax, ab - ay))


class TargetObservation:
    def __init__(self, size, args, principal_x):
        self.size, self.args, self.principal_x = tuple(size), args, float(principal_x)
        self.session = uuid.uuid4().hex
        self.lock_id, self.observation_seq = 0, 0
        self.seq, self.stamp = -1, None
        self.target, self.status = None, "idle"
        self.selection_stamp = None
        self.last_seen, self.smooth_box = None, None
        self.recovery, self.recovery_count = None, 0

    def cancel(self, status="cancelled"):
        self.target, self.status = None, status
        self.selection_stamp = None
        self.last_seen, self.smooth_box = None, None
        self.recovery, self.recovery_count = None, 0

    def accept(self, target, stamp, reset=False):
        box = target["box_rectified_xyxy"]
        if reset or self.smooth_box is None or self.args.track_smooth_ms == 0:
            self.smooth_box = list(box)
        else:
            alpha = -math.expm1(-max(0., stamp - self.last_seen) / (self.args.track_smooth_ms / 1000))
            self.smooth_box = [old + alpha * (new - old) for old, new in zip(self.smooth_box, box)]
        self.target, self.last_seen, self.status = target, stamp, "tracking"
        self.recovery, self.recovery_count = None, 0

    def select(self, records, point, stamp, now):
        self.cancel()
        if not math.isfinite(stamp) or not 0 <= now - stamp <= self.args.vision_max_age:
            self.status = "stale"
            return
        x, y = point
        hits = [r for r in valid_targets(records, self.size, self.args.confidence)
                if r["box_rectified_xyxy"][0] <= x <= r["box_rectified_xyxy"][2]
                and r["box_rectified_xyxy"][1] <= y <= r["box_rectified_xyxy"][3]]
        if len(hits) == 1:
            self.lock_id += 1
            self.selection_stamp = now
            self.accept(hits[0], stamp, reset=True)
        else:
            self.status = "ambiguous" if hits else "no_target"

    def update(self, records, seq, stamp, now):
        self.expire(now)
        if (type(seq) is not int or seq <= self.seq or not math.isfinite(stamp)
                or not math.isfinite(now) or not 0 <= now - stamp <= self.args.vision_max_age
                or self.stamp is not None and stamp < self.stamp):
            self.cancel("stale")
            return
        if self.target is not None:
            if self.stamp is not None and stamp - self.stamp > self.args.vision_max_age:
                self.cancel("stale")
            else:
                # ponytail: conservative box association; add appearance tracking only if recordings justify it.
                matches = []
                for candidate in valid_targets(records, self.size, self.args.track_confidence):
                    if candidate["class_id"] != self.target["class_id"]:
                        continue
                    iou, motion = match_metrics(self.target["box_rectified_xyxy"], candidate["box_rectified_xyxy"])
                    if iou >= self.args.track_min_iou and motion <= self.args.track_max_motion:
                        matches.append(candidate)
                if len(matches) > 1:
                    self.cancel("ambiguous")
                elif not matches:
                    self.status = "waiting"
                    self.recovery, self.recovery_count = None, 0
                elif self.status == "waiting":
                    candidate = matches[0]
                    if candidate["confidence"] >= self.args.confidence:
                        coherent = False
                        if self.recovery is not None:
                            iou, motion = match_metrics(self.recovery["box_rectified_xyxy"], candidate["box_rectified_xyxy"])
                            coherent = iou >= self.args.track_min_iou and motion <= self.args.track_max_motion
                        self.recovery_count = self.recovery_count + 1 if coherent else 1
                        self.recovery = candidate
                        if self.recovery_count >= 2:
                            self.accept(candidate, stamp, reset=True)
                    else:
                        self.recovery, self.recovery_count = None, 0
                else:
                    self.accept(matches[0], stamp)
        self.seq, self.stamp = seq, stamp
        self.expire(now)

    def expire(self, now):
        if self.target is not None and not 0 <= now - self.stamp <= self.args.vision_max_age:
            self.cancel("stale")
            return True
        if self.status == "waiting" and now - self.last_seen >= self.args.track_hold:
            self.cancel("lost")
            return True
        return False

    def packet(self, now):
        self.expire(now)
        self.observation_seq += 1
        target, error, center, slew = None, None, None, None
        smooth_center, smooth_error, memory = None, None, None
        reference = self.args.yaw_reference * self.size[0]
        if self.target is not None:
            memory = dict(display_only=True, box_xyxy=list(self.smooth_box),
                          last_seen_monotonic_s=self.last_seen, age_ms=1000 * (now - self.last_seen),
                          remaining_ms=max(0., 1000 * (self.args.track_hold - (now - self.last_seen))),
                          recovery_count=self.recovery_count)
        if self.status == "tracking" and self.target is not None:
            target = {key: self.target[key] for key in ("class_id", "class", "confidence", "box_rectified_xyxy")}
            x1, y1, x2, y2 = target["box_rectified_xyxy"]
            center = [(x1 + x2) / 2, (y1 + y2) / 2]
            error = (center[0] - reference) / (self.size[0] / 2)
            x1, y1, x2, y2 = self.smooth_box
            smooth_center = [(x1 + x2) / 2, (y1 + y2) / 2]
            smooth_error = (smooth_center[0] - reference) / (self.size[0] / 2)
            slew = yaw_slew_preview(smooth_error, self.args.yaw_gain, self.args.yaw_deadband,
                                    self.args.yaw_max_slew, self.args.yaw_sign)
        return dict(
            schema="coral.observation.v1", session=self.session, observation_seq=self.observation_seq,
            capture_seq=self.seq, lock_id=self.lock_id, valid=target is not None, status=self.status,
            selection_monotonic_s=self.selection_stamp,
            capture_monotonic_s=self.stamp, result_monotonic_s=now,
            source_age_ms=None if self.stamp is None else max(0., 1000 * (now - self.stamp)),
            valid_for_ms=1000 * self.args.vision_max_age, coordinate_frame="rectified_left",
            image_size=list(self.size), reference_u=reference, principal_u=self.principal_x,
            target=target, center_uv=center, error_x=error,
            smoothed_center_uv=smooth_center, smoothed_error_x=smooth_error, track_memory=memory,
            preview=dict(yaw_target_slew_dps=slew, gain=self.args.yaw_gain,
                         deadband=self.args.yaw_deadband, limit_dps=self.args.yaw_max_slew,
                         sign=self.args.yaw_sign, smooth_ms=self.args.track_smooth_ms, sent_to_teensy=False),
        )


class ObservationOutput:
    def __init__(self, folder, port):
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM) if port else None
        if self.socket:
            self.socket.setblocking(False)
        self.address = ("127.0.0.1", port)
        self.log = (folder / "vision_observations.jsonl").open("w", encoding="utf-8")
        self.status = "UDP OFF"
        self.errors = 0
        self.feedback = None

    def control_status(self, observation, now):
        if not self.socket or not observation:
            return None
        for _ in range(16):
            try:
                data, source = self.socket.recvfrom(4097)
            except OSError:
                break
            if source != self.address or len(data) > 4096:
                continue
            try:
                value = json.loads(data)
                if (value["schema"] != "coral.control.v1" or value["session"] != observation["session"]
                        or type(value["lock_id"]) is not int or value["lock_id"] != observation["lock_id"]
                        or type(value["confirmed"]) is not bool
                        or value["status"] not in ("ACTIVE", "CENTERED", "WAITING", "AWAITING_TEENSY", "WAIT_LINK",
                            "OBSERVE_ONLY", "CLICK_AGAIN", "MANUAL_OVERRIDE", "EXPIRED", "RESTARTED",
                            "INVALID_DATA", "UDP_ERROR", "MODE_REJECTED", "NEW_VISION_SESSION")):
                    continue
                start, end = value["result_monotonic_s"], value["valid_until_monotonic_s"]
                if (type(start) not in (int, float) or type(end) not in (int, float)
                        or not math.isfinite(start) or not math.isfinite(end) or not start <= now < end <= start + .201
                        or self.feedback and start <= self.feedback["result_monotonic_s"]):
                    continue
                for key in ("requested_slew_dps", "accepted_slew_dps"):
                    v = value[key]
                    if v is not None and (type(v) not in (int, float) or not math.isfinite(v) or abs(v) > 5):
                        raise ValueError("invalid slew")
                if value["confirmed"] and value["accepted_slew_dps"] is None:
                    continue
                self.feedback = value
            except (ValueError, KeyError, TypeError, OverflowError):
                continue
        value = self.feedback
        if not value or now >= value["valid_until_monotonic_s"] or value["lock_id"] != observation["lock_id"]:
            return None
        if not observation["valid"]:
            return dict(value, status="WAITING" if observation["status"] == "waiting" else "CLICK_AGAIN",
                        confirmed=False, accepted_slew_dps=None)
        return value

    def publish(self, packet):
        payload = json.dumps(packet, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
        if self.socket:
            try:
                self.socket.sendto(payload.encode("utf-8"), self.address)
                self.status = "UDP SENT (no ACK)"
            except OSError:
                self.errors += 1
                self.status = "UDP ERROR"
        self.log.write(payload + "\n")

    def close(self):
        if self.socket:
            self.socket.close()
        self.log.close()
