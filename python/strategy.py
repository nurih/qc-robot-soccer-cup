"""Soccer decision loop: find the ball, find which wall (team side) is ahead,
and drive the ball toward the opponent's goal.

Follows the uno-q-miniauto skill's "Add high-level robot behavior" pattern:
every tick reads sensors/vision, makes one bounded decision, and issues a
single short timed drive() call. Nothing here issues unbounded motion, and
vision failures (stale frame, no detection, ambiguous wall color) default to
stopping or holding position rather than guessing.
"""
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from robot_client import MiniAutoRobot
from vision.ball_detector import BallDetector, Detection
from vision.camera_stream import CameraStream
from vision.wall_detector import WallDetector

CENTER_DEADBAND = 0.12          # fraction of frame width considered "centered"
REALIGN_DEADBAND = 0.18         # wider deadband tolerated while already pushing
CLOSE_HEIGHT_THRESHOLD = 0.35   # bbox height fraction considered "ball is close"
PUSH_OBSTACLE_CM = 8            # ultrasonic distance treated as "hit something"
LOST_BALL_GRACE_TICKS = 3       # consecutive missed detections tolerated before re-searching

SEARCH_SPEED, SEARCH_MS = 150, 260
TURN_SPEED, TURN_MS = 150, 220
APPROACH_SPEED, APPROACH_MS = 170, 350
PUSH_SPEED, PUSH_MS = 190, 400
RETREAT_SPEED, RETREAT_MS = 160, 400


class State(Enum):
    SEARCH = "search"
    APPROACH = "approach"
    PUSH = "push"
    RETREAT = "retreat"


@dataclass
class Config:
    stale_frame_s: float = 0.5
    ball_label: str = "soccerball"


class SoccerStrategy:
    def __init__(
        self,
        robot: MiniAutoRobot,
        camera: CameraStream,
        ball_detector: BallDetector,
        wall_detector: WallDetector,
        config: Optional[Config] = None,
    ) -> None:
        self._robot = robot
        self._camera = camera
        self._ball = ball_detector
        self._wall = wall_detector
        self._config = config or Config()
        self._state = State.SEARCH
        self._misses = 0

    def close(self) -> None:
        self._camera.close()
        self._ball.close()

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

        ball = self._ball.best(frame, self._config.ball_label)
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

        if self._state is State.SEARCH:
            self._do_search(ball)
        elif self._state is State.APPROACH:
            self._do_approach(ball)
        elif self._state is State.PUSH:
            self._do_push(ball, wall, opponent)
        else:
            self._do_retreat()

    def _do_search(self, ball: Optional[Detection]) -> None:
        if ball is not None:
            self._state = State.APPROACH
            return
        self._robot.drive("rotate_left", SEARCH_SPEED, SEARCH_MS)

    def _do_approach(self, ball: Optional[Detection]) -> None:
        if ball is None:
            return  # inside the lost-ball grace period; hold position this tick

        offset = ball.x_center - 0.5
        if abs(offset) > CENTER_DEADBAND:
            direction = "rotate_left" if offset < 0 else "rotate_right"
            self._robot.drive(direction, TURN_SPEED, TURN_MS)
            return

        if ball.height >= CLOSE_HEIGHT_THRESHOLD:
            self._state = State.PUSH
            return

        self._robot.drive("forward", APPROACH_SPEED, APPROACH_MS)

    def _do_push(self, ball: Optional[Detection], wall, opponent: str) -> None:
        if ball is None:
            self._state = State.APPROACH
            return

        offset = ball.x_center - 0.5
        if abs(offset) > REALIGN_DEADBAND:
            self._state = State.APPROACH
            return

        sensors = self._robot.read_sensors()
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
