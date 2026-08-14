"""
simple_strategy.py  -  the smallest end-to-end strategy that still plays.

A teaching companion to strategy.py. Same four states and the same interface,
so it is a drop-in replacement (ROBOCUP_STRATEGY=simple), but every rule is one
line and every state change goes through _go(), which logs *why* it happened:

    [SIMPLE] SEARCH -> APPROACH : ball seen

Read tick() top to bottom and you have the whole machine.

    SEARCH    spin until a ball appears                 -> APPROACH
    APPROACH  turn to centre it, then drive at it       -> PUSH (close) | SEARCH (lost)
    PUSH      shove it, but only toward the opponent    -> RETREAT (own wall) | SEARCH (lost)
    RETREAT   back off so we do not score an own goal   -> SEARCH

Safety is deliberately identical to the full strategy: a stale or missing frame
stops the robot, and every action is one short timed drive so the next tick
re-observes before doing anything else.
"""
import cv2

from dataclasses import dataclass
from enum import Enum
from typing import Optional

from ball_follower import _best_ball, _centre_offset
from robot_client import MiniAutoRobot
from vision.camera_stream import CameraStream
from sensor_monitor import SensorMonitor
from vision.wall_detector import WallDetector

# How far off-centre the ball may be before we bother turning (fraction of
# half-frame width). Bigger = lazier steering.
CENTRE_DEADBAND = 0.15

# Ball bounding-box height as a fraction of the frame. Bigger box = closer ball.
CLOSE_ENOUGH = 0.35

# Aggressive tuning: 255 is the firmware clamp. With drive_async the duration is
# a dead-man timeout, not a motion quantum: it must outlast one loop iteration so
# motion is continuous, but stay short enough to stop quickly if the loop dies.
TURN_SPEED, TURN_MS = 200, 300
FORWARD_SPEED, FORWARD_MS = 235, 300
PUSH_SPEED, PUSH_MS = 255, 300
BACK_SPEED, BACK_MS = 220, 300


class State(Enum):
    SEARCH = "search"
    APPROACH = "approach"
    PUSH = "push"
    RETREAT = "retreat"


@dataclass
class Config:
    stale_frame_s: float = 0.5
    ball_label: str = "soccer_ball"


def _ball_height_fraction(ball: dict, frame_height: int) -> float:
    """Ball box height as a fraction of the frame, used as a distance proxy."""
    box = ball.get("bounding_box_xyxy")
    if not box or len(box) < 4 or frame_height <= 0:
        return 0.0
    _, y1, _, y2 = (float(v) for v in box[:4])
    reference = 1.0 if max(y1, y2) <= 1.0 else float(frame_height)
    return abs(y2 - y1) / reference


class SimpleStrategy:
    """Four states, one rule each. Same interface as SoccerStrategy."""

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
        self._observer = observer
        self._state = State.SEARCH
        self._sensors = SensorMonitor()
        self._last_transition = ""

    def close(self) -> None:
        self._camera.close()
        if hasattr(self._detector, "close"):
            self._detector.close()

    # -- the only way the state ever changes ------------------------------

    def _go(self, new_state: State, reason: str) -> None:
        if new_state is self._state:
            return
        self._last_transition = f"{self._state.value} -> {new_state.value} : {reason}"
        print(f"[SIMPLE] {self._last_transition.upper()}")
        self._state = new_state

    def _team(self) -> str:
        return "BLUE" if self._robot.hold_toggle() else "RED"

    def _opponent(self) -> str:
        return "RED" if self._team() == "BLUE" else "BLUE"

    # -- one tick ---------------------------------------------------------

    def tick(self) -> None:
        frame, age_s = self._camera.read()
        if frame is None or age_s > self._config.stale_frame_s:
            # Never drive on a frame we cannot trust -- but do not call
            # robot.stop() here: it also clears program_enabled and would end the
            # run. Timed drives auto-stop, so simply issuing nothing is enough.
            print(f"[SIMPLE] stale/missing frame ({age_s:.2f}s) -> hold")
            return

        frame_h, frame_w = frame.shape[:2]

        _, jpeg = cv2.imencode(".jpg", frame)
        detection = self._detector.detect(jpeg.tobytes(), image_type="jpg")
        ball = _best_ball(detection)
        wall = self._wall.detect(frame)
        sensors = self._sensors.update(self._robot.read_sensors())

        if self._observer is not None:
            self._observer(
                frame=frame, detection=detection, state=self._state.value,
                team=self._team(), wall=wall, sensors=sensors, ball=ball, health=self._sensors.health(),
                transition=self._last_transition,
            )

        # Losing the ball always sends us back to searching, from any state.
        if ball is None:
            self._go(State.SEARCH, "ball lost")

        if self._state is State.SEARCH:
            if ball is not None:
                self._go(State.APPROACH, "ball seen")
            else:
                self._robot.drive_async("rotate_left", TURN_SPEED, TURN_MS)

        elif self._state is State.APPROACH:
            offset = _centre_offset(ball, frame_w)
            if offset is None:
                return  # unusable box; look again next tick
            if _ball_height_fraction(ball, frame_h) >= CLOSE_ENOUGH:
                self._go(State.PUSH, "ball is close")
            elif offset < -CENTRE_DEADBAND:
                self._robot.drive_async("rotate_left", TURN_SPEED, TURN_MS)
            elif offset > CENTRE_DEADBAND:
                self._robot.drive_async("rotate_right", TURN_SPEED, TURN_MS)
            else:
                self._robot.drive_async("forward", FORWARD_SPEED, FORWARD_MS)

        elif self._state is State.PUSH:
            if wall.side == self._team():
                # Our own wall is ahead: pushing now would be an own goal.
                self._go(State.RETREAT, "own wall ahead")
            else:
                # Opponent wall or unknown: shove, then re-check next tick.
                self._robot.drive_async("forward", PUSH_SPEED, PUSH_MS)

        elif self._state is State.RETREAT:
            self._robot.drive_async("backward", BACK_SPEED, BACK_MS)
            self._go(State.SEARCH, "backed off")
