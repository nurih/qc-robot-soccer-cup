"""
state_machine_demo.py  -  watch the state machine run, with no robot attached.

Feeds SimpleStrategy a scripted sequence of fake camera/detector/wall inputs and
prints the state, the transition, and the motor command for every tick. Useful
for seeing how the states hand off to each other before trusting real hardware,
and it doubles as a regression test of the transition rules.

    python3 python/state_machine_demo.py

Runs anywhere: it stubs out arduino.app_utils so it works off the board too.
"""
import sys
import time
import types

from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

# robot_client imports the App Lab Bridge, which only exists on the board. Stub
# it before importing anything that pulls it in, so this demo runs on a laptop.
if "arduino.app_utils" not in sys.modules:
    arduino = types.ModuleType("arduino")
    app_utils = types.ModuleType("arduino.app_utils")

    class _Bridge:
        @staticmethod
        def call(*_args, **_kwargs):
            raise RuntimeError("demo never touches the Bridge")

    class _App:
        @staticmethod
        def run(*_args, **_kwargs):
            raise RuntimeError("demo never runs the App")

    app_utils.Bridge = _Bridge
    app_utils.App = _App
    arduino.app_utils = app_utils
    sys.modules["arduino"] = arduino
    sys.modules["arduino.app_utils"] = app_utils

import numpy as np  # noqa: E402

from perception import Snapshot  # noqa: E402
from simple_strategy import SimpleStrategy, State  # noqa: E402
from vision.wall_detector import WallReading  # noqa: E402

FRAME = np.zeros((240, 320, 3), dtype=np.uint8)


class FakeRobot:
    """Records commands instead of moving anything."""

    def __init__(self, team_is_blue: bool = False) -> None:
        self.team_is_blue = team_is_blue
        self.last_command = None

    def hold_toggle(self) -> bool:
        return self.team_is_blue

    def read_sensors(self) -> dict:
        return {"ultrasonic_mm": 600, "line_ok": True}

    def drive(self, command, speed, ms) -> None:
        self.last_command = f"{command} {speed}/{ms}ms"

    # The strategies issue motion without blocking; see robot_client.drive_async.
    drive_async = drive

    def stop(self) -> None:
        self.last_command = "stop"


class FakeCamera:
    def read(self):
        return FRAME, 0.0

    def close(self) -> None:
        pass


class FakeDetector:
    """Returns whatever detection the current step scripted."""

    def __init__(self) -> None:
        self.detection = {"detection": []}

    def detect(self, *_args, **_kwargs) -> dict:
        return self.detection


class SyncPerception:
    """Perception computed inline, so each tick sees exactly this step's inputs.

    The real PerceptionWorker runs on a thread, which is right for the robot but
    would make this walkthrough race the scheduler.
    """

    def __init__(self, camera, detector, wall_detector) -> None:
        self._camera = camera
        self._detector = detector
        self._wall = wall_detector

    def latest(self):
        frame, _age = self._camera.read()
        if frame is None:
            return None
        return Snapshot(
            frame=frame,
            detection=self._detector.detect(b"", image_type="jpg"),
            wall=self._wall.detect(frame),
            captured_at=time.monotonic(),
        )

    def close(self) -> None:
        pass


class FakeWall:
    def __init__(self) -> None:
        self.reading = WallReading(side="UNKNOWN", red_pct=0.0, blue_pct=0.0)

    def detect(self, _frame):
        return self.reading


def ball(cx: float, height_px: float = 20.0, confidence: float = 0.9) -> dict:
    """A soccer_ball detection centred at cx, height as a closeness proxy."""
    return {
        "detection": [{
            "class_name": "soccer_ball",
            "confidence": confidence,
            "bounding_box_xyxy": [cx - 13, 120, cx + 13, 120 + height_px],
        }]
    }


NO_BALL = {"detection": []}

# (label, detection, wall side) -- walked in order, one tick each.
SCRIPT = [
    ("nothing in view",            NO_BALL,            "UNKNOWN"),
    ("nothing in view",            NO_BALL,            "UNKNOWN"),
    ("ball appears, far left",     ball(60),           "UNKNOWN"),
    ("still left of centre",       ball(100),          "UNKNOWN"),
    ("roughly centred",            ball(160),          "UNKNOWN"),
    ("centred, drifting right",    ball(210),          "UNKNOWN"),
    ("centred and much closer",    ball(160, 90),      "UNKNOWN"),
    ("pushing, opponent wall",     ball(160, 90),      "BLUE"),
    ("pushing, our own wall",      ball(160, 90),      "RED"),
    ("after backing off",          ball(160, 90),      "RED"),
    ("ball vanishes",              NO_BALL,            "RED"),
]


def main() -> None:
    robot = FakeRobot(team_is_blue=False)  # our team is RED
    detector = FakeDetector()
    wall = FakeWall()
    camera = FakeCamera()
    strategy = SimpleStrategy(
        robot, camera, detector, wall,
        perception=SyncPerception(camera, detector, wall),
    )

    print(f"team = {'BLUE' if robot.team_is_blue else 'RED'}   "
          f"(pushing toward the {'RED' if robot.team_is_blue else 'BLUE'} wall)\n")
    print(f"{'#':>2}  {'situation':<26} {'wall':<8} {'state':<10} {'command':<20}")
    print("-" * 72)

    for index, (label, detection, wall_side) in enumerate(SCRIPT, start=1):
        detector.detection = detection
        wall.reading = WallReading(
            side=wall_side,
            red_pct=5.0 if wall_side == "RED" else 0.0,
            blue_pct=5.0 if wall_side == "BLUE" else 0.0,
        )
        robot.last_command = None

        before = strategy._state
        strategy.tick()
        after = strategy._state

        arrow = f"{before.value} -> {after.value}" if before is not after else after.value
        print(f"{index:>2}  {label:<26} {wall_side:<8} {arrow:<10} {robot.last_command or '-':<20}")

    print("\nFinal state:", strategy._state.value)


if __name__ == "__main__":
    main()
