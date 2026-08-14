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

from ball_follower import (
    BallFollower, _best_ball, _best_goal, _best_robot, _centre_offset,
)
from robot_client import MiniAutoRobot
from vision.camera_stream import CameraStream
from perception import PerceptionWorker
from sensor_monitor import SensorMonitor
from vision.wall_detector import WallDetector

CENTER_DEADBAND = 0.12          # fraction of frame width considered "centered"
REALIGN_DEADBAND = 0.18         # wider deadband tolerated while already pushing
CLOSE_HEIGHT_THRESHOLD = 0.35   # bbox height fraction considered "ball is close"
PUSH_OBSTACLE_CM = 8            # ultrasonic distance treated as "hit something"
LOST_BALL_GRACE_TICKS = 3       # consecutive missed detections tolerated before re-searching

# Opponent avoidance: cornering or tipping an opponent is a yellow card, so this
# check outranks whatever the robot was doing.
OPPONENT_AVOID_CM = 15
AVOID_TURN_SPEED, AVOID_TURN_MS = 200, 200
AVOID_FORWARD_SPEED, AVOID_FORWARD_MS = 220, 250

# Committed strike, fired only once ball and goal line up.
KICK_SPEED, KICK_MS = 255, 500
GOAL_ALIGN_DEADBAND = 0.20

# Aggressive tuning: 255 is the firmware clamp.
# With drive_async the duration is no longer a motion quantum - it is a dead-man
# timeout. It must outlast one loop iteration (~150 ms) so motion stays smooth
# between commands, while still stopping the robot quickly if the loop dies.
SEARCH_SPEED, SEARCH_MS = 150, 300
PUSH_SPEED, PUSH_MS = 255, 300
RETREAT_SPEED, RETREAT_MS = 220, 300


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
        perception=None,
    ) -> None:
        self._robot = robot
        self._camera = camera
        self._detector = detector
        self._wall = wall_detector
        self._config = config or Config()
        self._follower = BallFollower()
        self._state = State.SEARCH
        self._sensors = SensorMonitor()
        # Vision runs on its own thread so the control loop never waits for the
        # model; see perception.py.
        # Injectable so tests and demos can supply a synchronous stand-in and
        # get deterministic behaviour instead of racing a background thread.
        self._owns_perception = perception is None
        if perception is None:
            perception = PerceptionWorker(camera, detector, wall_detector)
            perception.start()
        self._perception = perception
        self._misses = 0
        self._last_transition = ""
        # Optional read-only telemetry sink (see dashboard.Dashboard.publish).
        # It never influences decisions; failures in it must not stop the robot.
        self._observer = observer

    def _go(self, new_state: "State", reason: str) -> None:
        """Change state, recording why so the dashboard can show the path."""
        if new_state is self._state:
            return
        self._last_transition = f"{self._state.value} -> {new_state.value} : {reason}"
        print(f"[STRATEGY] {self._last_transition}")
        self._state = new_state

    def close(self) -> None:
        if self._owns_perception:
            self._perception.close()
        self._camera.close()
        if hasattr(self._detector, "close"):
            self._detector.close()

    def _team_color(self) -> str:
        return "BLUE" if self._robot.hold_toggle() else "RED"

    def _opponent_color(self) -> str:
        return "RED" if self._team_color() == "BLUE" else "BLUE"

    def tick(self) -> None:
        snapshot = self._perception.latest()
        age_s = snapshot.age() if snapshot is not None else float("inf")
        if snapshot is None or age_s > self._config.stale_frame_s:
            # Issue no motion, but do NOT call robot.stop(): the firmware's stop
            # also clears program_enabled, which would end the run on a single
            # slow frame. Timed drives auto-stop, so skipping the tick is enough.
            print(f"[STRATEGY] stale/missing perception ({age_s:.2f}s old) -> hold")
            return

        frame = snapshot.frame
        detection = snapshot.detection
        frame_h, frame_w = frame.shape[:2]

        ball = _best_ball(detection)
        if ball is None or str(ball.get("class_name", "")).strip().lower() not in {
            self._config.ball_label, "soccer_ball", "ball"
        }:
            ball = None

        if ball is None:
            self._misses += 1
            if self._misses >= LOST_BALL_GRACE_TICKS:
                self._go(State.SEARCH, "ball lost")
        else:
            self._misses = 0

        goal = _best_goal(detection)
        opponent_robot = _best_robot(detection)

        wall = snapshot.wall
        team = self._team_color()
        opponent = self._opponent_color()

        sensors = self._sensors.update(self._robot.read_sensors())
        distance_cm = sensors.get("ultrasonic_cm", -1)

        print(
            f"[STRATEGY] state={self._state.value} team={team} "
            f"ball={'Y' if ball else '-'} goal={'Y' if goal else '-'} "
            f"opp={'Y' if opponent_robot else '-'} "
            f"wall={wall.side} us={distance_cm}cm"
        )

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
                transition=self._last_transition,
            )

        # Avoidance outranks every state, including PUSH: being shoved into a
        # wall mid-push is exactly the yellow-card scenario. It is a per-tick
        # override, not a state, so we resume wherever we were once clear.
        if self._avoid_opponent(opponent_robot, frame_w, distance_cm):
            return

        if self._state is State.SEARCH:
            self._do_search(ball)
        elif self._state is State.APPROACH:
            self._do_approach(detection, ball, frame_w, frame_h, sensors)
        elif self._state is State.PUSH:
            self._do_push(ball, goal, frame_w, wall, opponent, sensors)
        else:
            self._do_retreat()

    def _avoid_opponent(self, opponent_robot, frame_w: int, distance_cm) -> bool:
        """Steer clear of another robot. Returns True if it took the tick.

        Turn direction comes from which side of the frame the opponent is on -
        the bounding box already tells us, so there is no need to guess.
        """
        if opponent_robot is None:
            return False
        try:
            distance_cm = int(distance_cm)
        except (TypeError, ValueError):
            return False
        if distance_cm <= 0 or distance_cm > OPPONENT_AVOID_CM:
            return False

        offset = _centre_offset(opponent_robot, frame_w)
        if offset is None:
            return False

        if abs(offset) <= REALIGN_DEADBAND:
            # Dead ahead: turn away from the side it occupies.
            command = "rotate_right" if offset < 0 else "rotate_left"
            print(f"[STRATEGY] opponent at {distance_cm}cm dead ahead -> {command}")
            self._robot.drive_async(command, AVOID_TURN_SPEED, AVOID_TURN_MS)
        else:
            # Already off to one side: drive past it.
            print(f"[STRATEGY] slipping past opponent at {distance_cm}cm")
            self._robot.drive_async("forward", AVOID_FORWARD_SPEED, AVOID_FORWARD_MS)
        return True

    def _do_search(self, ball: Optional[dict]) -> None:
        if ball is not None:
            self._go(State.APPROACH, "ball seen")
            return
        self._robot.drive_async("rotate_left", SEARCH_SPEED, SEARCH_MS)

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
            self._go(State.PUSH, "arrived at ball")
            return

        command, speed, ms = action

        # Override: if bbox height indicates the ball is very close, push regardless
        if _ball_height_fraction(ball, frame_h) >= CLOSE_HEIGHT_THRESHOLD:
            self._go(State.PUSH, "ball is close")
            return

        self._robot.drive_async(command, speed, ms)

    def _do_push(
        self,
        ball: Optional[dict],
        goal: Optional[dict],
        frame_w: int,
        wall,
        opponent: str,
        sensors: dict,
    ) -> None:
        if ball is None:
            self._go(State.APPROACH, "lost ball while pushing")
            return

        offset = _centre_offset(ball, frame_w)
        if offset is None or abs(offset) > REALIGN_DEADBAND:
            self._go(State.APPROACH, "ball drifted off-centre")
            return

        ultrasonic_cm = sensors.get("ultrasonic_cm", -1)

        # Preferred: aim by actually seeing the goal behind the ball.
        goal_offset = _centre_offset(goal, frame_w) if goal is not None else None
        if goal_offset is not None:
            misalign = goal_offset - offset
            if abs(misalign) <= GOAL_ALIGN_DEADBAND:
                print(f"[STRATEGY] ball+goal aligned ({misalign:+.2f}) -> KICK")
                self._robot.drive_async("forward", KICK_SPEED, KICK_MS)
            else:
                # Line the goal up behind the ball before committing.
                command = "rotate_right" if misalign > 0 else "rotate_left"
                print(f"[STRATEGY] goal off by {misalign:+.2f} -> {command}")
                self._robot.drive_async(command, SEARCH_SPEED, 150)
        elif wall.side == opponent:
            # Fallback: no goal in frame, so trust wall colour as before. Goal
            # detection is unreliable, so this path stays the safety net.
            self._robot.drive_async("forward", PUSH_SPEED, PUSH_MS)
        elif wall.side == "UNKNOWN":
            # Ambiguous wall colour -- nudge forward cautiously and re-check
            # next tick instead of committing to a full push.
            self._robot.drive_async("forward", PUSH_SPEED, PUSH_MS // 2)
        else:
            # Own-side wall ahead -- pushing here risks an own goal.
            self._go(State.RETREAT, "own wall ahead")
            return

        if 0 < ultrasonic_cm <= PUSH_OBSTACLE_CM and wall.side != opponent:
            self._go(State.RETREAT, "unexpected obstacle while pushing")

    def _do_retreat(self) -> None:
        self._robot.drive_async("backward", RETREAT_SPEED, RETREAT_MS)
        self._go(State.SEARCH, "backed off")
