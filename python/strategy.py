"""Soccer decision loop: find the ball, find which wall (team side) is ahead,
and drive the ball toward the opponent's goal.

Follows the uno-q-miniauto skill's "Add high-level robot behavior" pattern:
every tick reads sensors/vision, makes one bounded decision, and issues a
single short timed drive() call. Nothing here issues unbounded motion, and
vision failures (stale frame, no detection, ambiguous wall color) default to
stopping or holding position rather than guessing.

Detection backend (brick or eim) is selected by detector.make_detector() via
the DETECTOR_BACKEND environment variable. BallFollower from ball_follower.py
owns the turn/advance decision logic for APPROACH, keeping it unit-testable
independently of the full strategy.
"""
import cv2

from dataclasses import dataclass
from enum import Enum
from typing import Optional

from ball_follower import BallFollower, _best_ball, _centre_offset
from robot_client import MiniAutoRobot
from vision.camera_stream import CameraStream
from sensor_monitor import SensorMonitor
from vision.wall_detector import WallDetector

CENTER_DEADBAND = 0.12          # fraction of frame width considered "centered"
REALIGN_DEADBAND = 0.18         # wider deadband tolerated while already pushing
CLOSE_HEIGHT_THRESHOLD = 0.35   # bbox height fraction considered "ball is close"
PUSH_OBSTACLE_CM = 8            # ultrasonic distance treated as "hit something"
LOST_BALL_GRACE_TICKS = 3       # consecutive missed detections tolerated before re-searching

# Aggressive tuning: 255 is the firmware clamp.
SEARCH_SPEED, SEARCH_MS = 215, 200
PUSH_SPEED, PUSH_MS = 255, 400
RETREAT_SPEED, RETREAT_MS = 220, 350


class State(Enum):
    SEARCH = "search"
    APPROACH = "approach"
    PUSH = "push"
    RETREAT = "retreat"


@dataclass
class Config:
    stale_frame_s: float = 0.5
    ball_label: str = "soccerball"


def _ball_height_fraction(item: dict, frame_height: int) -> float:
    """Return the ball bbox height as a fraction of frame height, or 0."""
    box = item.get("bounding_box_xyxy")
    if not box or len(box) < 4 or frame_height <= 0:
        return 0.0
    _, y1, _, y2 = (float(v) for v in box[:4])
    reference = 1.0 if max(y1, y2) <= 1.0 else float(frame_height)
    return abs(y2 - y1) / reference


class SoccerStrategy:
    def __init__(
        self,
        robot: MiniAutoRobot,
        camera: CameraStream,
        detector,
        wall_detector: WallDetector,
        config: Optional[Config] = None,
        observer=None,
    ) -> None:
        self._robot = robot
        self._camera = camera
        self._detector = detector
        self._wall = wall_detector
        self._config = config or Config()
        self._follower = BallFollower()
        self._state = State.SEARCH
        self._sensors = SensorMonitor()
        self._misses = 0
        # Optional read-only telemetry sink (see dashboard.Dashboard.publish).
        # It never influences decisions; failures in it must not stop the robot.
        self._observer = observer

    def close(self) -> None:
        self._camera.close()
        if hasattr(self._detector, "close"):
            self._detector.close()

    def _team_color(self) -> str:
        return "BLUE" if self._robot.hold_toggle() else "RED"

    def _opponent_color(self) -> str:
        return "RED" if self._team_color() == "BLUE" else "BLUE"

    def tick(self) -> None:
        frame, age_s = self._camera.read()
        if frame is None or age_s > self._config.stale_frame_s:
            print(f"[STRATEGY] stale/missing camera frame ({age_s:.2f}s old) -> stop")
            self._robot.stop()
            return

        frame_h, frame_w = frame.shape[:2]

        # Encode OpenCV frame to JPEG for the detector backend
        _, jpeg_buf = cv2.imencode(".jpg", frame)
        detection = self._detector.detect(jpeg_buf.tobytes(), image_type="jpg")

        ball = _best_ball(detection)
        if ball is None or str(ball.get("class_name", "")).strip().lower() not in {
            self._config.ball_label, "soccer_ball", "ball"
        }:
            ball = None

        if ball is None:
            self._misses += 1
            if self._misses >= LOST_BALL_GRACE_TICKS:
                self._state = State.SEARCH
        else:
            self._misses = 0

        wall = self._wall.detect(frame)
        team = self._team_color()
        opponent = self._opponent_color()

        print(
            f"[STRATEGY] state={self._state.value} team={team} "
            f"wall={wall.side}(r={wall.red_pct:.1f}% b={wall.blue_pct:.1f}%) "
            f"ball={'seen' if ball else 'none'}"
        )

        sensors = self._sensors.update(self._robot.read_sensors())

        if self._observer is not None:
            self._observer(
                frame=frame,
                detection=detection,
                state=self._state.value,
                team=team,
                wall=wall,
                sensors=sensors,
                ball=ball,
                health=self._sensors.health(),
            )

        if self._state is State.SEARCH:
            self._do_search(ball)
        elif self._state is State.APPROACH:
            self._do_approach(detection, ball, frame_w, frame_h, sensors)
        elif self._state is State.PUSH:
            self._do_push(ball, frame_w, wall, opponent, sensors)
        else:
            self._do_retreat()

    def _do_search(self, ball: Optional[dict]) -> None:
        if ball is not None:
            self._state = State.APPROACH
            return
        self._robot.drive("rotate_left", SEARCH_SPEED, SEARCH_MS)

    def _do_approach(
        self,
        detection: dict,
        ball: Optional[dict],
        frame_w: int,
        frame_h: int,
        sensors: dict,
    ) -> None:
        if ball is None:
            return  # inside the lost-ball grace period; hold position this tick

        # Delegate turn/advance decision to BallFollower
        action = self._follower.decide(detection, frame_w, sensors)
        if action is None:
            # BallFollower stops when ball is at arrived distance -- transition to push
            self._state = State.PUSH
            return

        command, speed, ms = action

        # Override: if bbox height indicates the ball is very close, push regardless
        if _ball_height_fraction(ball, frame_h) >= CLOSE_HEIGHT_THRESHOLD:
            self._state = State.PUSH
            return

        self._robot.drive(command, speed, ms)

    def _do_push(
        self,
        ball: Optional[dict],
        frame_w: int,
        wall,
        opponent: str,
        sensors: dict,
    ) -> None:
        if ball is None:
            self._state = State.APPROACH
            return

        offset = _centre_offset(ball, frame_w)
        if offset is None or abs(offset) > REALIGN_DEADBAND:
            self._state = State.APPROACH
            return

        ultrasonic_cm = sensors.get("ultrasonic_cm", -1)

        if wall.side == opponent:
            self._robot.drive("forward", PUSH_SPEED, PUSH_MS)
        elif wall.side == "UNKNOWN":
            # Ambiguous wall color -- nudge forward cautiously and re-check
            # next tick instead of committing to a full push.
            self._robot.drive("forward", PUSH_SPEED, PUSH_MS // 2)
        else:
            # Own-side wall ahead -- pushing here risks an own goal.
            print("[STRATEGY] own-side wall ahead while pushing -> retreat")
            self._state = State.RETREAT
            return

        if 0 < ultrasonic_cm <= PUSH_OBSTACLE_CM and wall.side != opponent:
            print("[STRATEGY] unexpected close obstacle while pushing -> retreat")
            self._state = State.RETREAT

    def _do_retreat(self) -> None:
        self._robot.drive("backward", RETREAT_SPEED, RETREAT_MS)
        self._state = State.SEARCH
