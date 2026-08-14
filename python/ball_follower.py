"""
ball_follower.py  -  turn soccer-ball detections into one short, bounded action.

Pure decision logic: no Bridge calls, no camera access, no globals. It takes a
detection payload plus a sensor snapshot and returns a single short motion, or
None meaning "stop". That keeps it unit-testable without hardware and keeps the
physical safety boundary in the firmware where it belongs.

Safety posture (per the developer journey guide): missing model output, no
detections, ambiguous detections, and unavailable sensor readings all resolve to
a stop rather than to continued motion.
"""

# The trained impulse labels the ball; accept the usual spellings so a relabelled
# dataset does not silently stop matching.
BALL_LABELS = {"soccer_ball", "soccerball", "ball"}
GOAL_LABELS = {"goal"}
ROBOT_LABELS = {"robot"}

MIN_CONFIDENCE = 0.5

# Tuned aggressively: the firmware clamps speed at 255, and closing distance fast
# matters more than smooth tracking in a 5-minute match.

# Fraction of half-frame width the ball may be off-centre before we rotate
# instead of advancing. Wider means we charge rather than fussing over centring.
TURN_DEADZONE = 0.24

# Durations act as a dead-man timeout for drive_async: long enough to outlast one
# perception loop so motion is continuous, short enough that a stalled loop stops
# the robot promptly. Steering precision comes from the loop rate, not the pulse.
TURN_SPEED = 150
TURN_MS = 300
FORWARD_SPEED = 235
FORWARD_MS = 300

# Stop advancing once this close, so the robot noses up to the ball rather than
# driving through it. The ultrasonic is the only obstacle sense available.
ARRIVED_DISTANCE_MM = 110


def _best_of(detection: dict | None, labels: set) -> dict | None:
    """Highest-confidence detection whose label is in `labels`, or None."""
    if not detection or "detection" not in detection:
        return None

    candidates = []
    for item in detection["detection"] or []:
        label = str(item.get("class_name", "")).strip().lower()
        if label not in labels:
            continue
        confidence = item.get("confidence")
        if confidence is None:
            continue
        # Some runners report percentages, others 0..1. Normalise to 0..1.
        confidence = float(confidence)
        if confidence > 1.0:
            confidence /= 100.0
        if confidence >= MIN_CONFIDENCE:
            candidates.append((confidence, item))

    if not candidates:
        return None
    return max(candidates, key=lambda pair: pair[0])[1]


def _best_ball(detection: dict | None) -> dict | None:
    return _best_of(detection, BALL_LABELS)


def _best_goal(detection: dict | None) -> dict | None:
    return _best_of(detection, GOAL_LABELS)


def _best_robot(detection: dict | None) -> dict | None:
    """Highest-confidence *other* robot. Used for avoidance, never as a target."""
    return _best_of(detection, ROBOT_LABELS)


def _centre_offset(item: dict, frame_width: int) -> float | None:
    """Ball centre as a -1..1 offset from frame centre, or None if unusable.

    Negative means the ball is left of centre.
    """
    box = item.get("bounding_box_xyxy")
    if not box or len(box) < 4 or frame_width <= 0:
        return None

    x1, _, x2, _ = (float(v) for v in box[:4])
    centre_x = (x1 + x2) / 2.0

    # The runner reports boxes in source-frame pixels, verified against the live
    # 320x240 stream (values ran up to 293). A normalised 0..1 payload is still
    # tolerated in case a future model/runner reports that way instead.
    reference = 1.0 if max(x1, x2) <= 1.0 else float(frame_width)

    return (centre_x - reference / 2.0) / (reference / 2.0)


class BallFollower:
    """Decides one short action per observation."""

    def __init__(
        self,
        min_confidence: float = MIN_CONFIDENCE,
        turn_deadzone: float = TURN_DEADZONE,
        arrived_distance_mm: int = ARRIVED_DISTANCE_MM,
    ) -> None:
        self.min_confidence = min_confidence
        self.turn_deadzone = turn_deadzone
        self.arrived_distance_mm = arrived_distance_mm

    def decide(
        self,
        detection: dict | None,
        frame_width: int,
        sensors: dict | None,
    ) -> tuple[str, int, int] | None:
        """Return (command, speed, duration_ms), or None to stop.

        None is returned for: no model output, no ball, an unusable bounding box,
        arrival at the ball, and an unavailable distance reading when the next
        action would be forward motion.
        """
        item = _best_ball(detection)
        if item is None:
            return None

        offset = _centre_offset(item, frame_width)
        if offset is None:
            return None

        # Off-centre: rotate toward the ball. Rotation cannot close distance on an
        # obstacle, so it stays allowed even when the ultrasonic is unavailable.
        if offset < -self.turn_deadzone:
            return ("rotate_left", TURN_SPEED, TURN_MS)
        if offset > self.turn_deadzone:
            return ("rotate_right", TURN_SPEED, TURN_MS)

        # Centred: advance only on a distance reading we actually trust. The guide
        # treats -1 as "unavailable", never as "clear".
        distance_mm = (sensors or {}).get("ultrasonic_mm", -1)
        try:
            distance_mm = int(distance_mm)
        except (TypeError, ValueError):
            return None

        if distance_mm <= 0:
            return None
        if distance_mm <= self.arrived_distance_mm:
            return None

        return ("forward", FORWARD_SPEED, FORWARD_MS)


def describe_action(action: tuple[str, int, int] | None) -> str:
    if action is None:
        return "[POLICY] stop"
    command, speed, ms = action
    return f"[POLICY] {command} speed={speed} ms={ms}"
