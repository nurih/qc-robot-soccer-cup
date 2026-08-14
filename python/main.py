import os
import time

from arduino.app_utils import App
from robot_client import MiniAutoRobot, ProgramStopped
from strategy import Config, SoccerStrategy
from vision.ball_detector import BallDetector
from vision.camera_stream import CameraStream
from vision.wall_detector import WallDetector

robot = MiniAutoRobot()

CAMERA_URL = os.environ.get("CAMERA_STREAM_URL", "http://192.168.5.1:81/stream")
MODEL_PATH = os.environ.get("ROBOCUP_MODEL_PATH", "")
BALL_LABEL = os.environ.get("ROBOCUP_BALL_LABEL", "soccerball")
BALL_CONFIDENCE = float(os.environ.get("ROBOCUP_BALL_CONFIDENCE", "0.5"))
WALL_MIN_COVERAGE_PCT = float(os.environ.get("ROBOCUP_WALL_MIN_COVERAGE_PCT", "2.0"))
STALE_FRAME_S = int(os.environ.get("ROBOCUP_STALE_FRAME_MS", "500")) / 1000.0
MODE = os.environ.get("ROBOCUP_MODE", "match")  # "match" or "demo"

print(f"health   : {robot.health()}")
print(f"sensors  : {robot.read_sensors()}")

# Track the toggle so we only print when it actually changes.
# Hold the CAM boot button for 5 seconds to switch teams.
_last_toggle = robot.hold_toggle()
print(f"[TEAM] active team: {'BLUE' if _last_toggle else 'RED'}  (hold CAM button 5 s to switch)")


def demo_sequence() -> None:
    """Original canned motion/sensor smoke test. Set ROBOCUP_MODE=demo to run
    this instead of the match strategy -- useful for verifying drive/servo/
    sensor wiring before trusting the vision pipeline."""

    def drive(direction: str, speed: int = 150, ms: int = 500) -> None:
        print(f"  {direction} speed={speed} ms={ms}")
        robot.drive(direction, speed, ms)

    drive("forward",      speed=150, ms=500)
    drive("backward",     speed=150, ms=500)
    drive("left",         speed=150, ms=500)   # strafe left
    drive("right",        speed=150, ms=500)   # strafe right
    drive("rotate_left",  speed=255, ms=3250)  # spin in place, 255 firmware cap on speed
    time.sleep(0.5)
    drive("rotate_right", speed=255, ms=3250)

    robot.stop()
    time.sleep(0.5)

    sensors = robot.read_sensors()
    print(f"ultrasonic : {sensors.get('ultrasonic_cm')} cm")
    print(f"battery    : {sensors.get('battery_mv')} mV")
    print(f"line       : {sensors.get('line_digital')}")

    robot.led(True)
    robot.servo(90)
    robot.servo(150)
    robot.servo(30)
    robot.servo(90)
    robot.led(False)

    robot.stop()


def _build_strategy() -> SoccerStrategy:
    if not MODEL_PATH:
        raise SystemExit(
            "ROBOCUP_MODEL_PATH is not set. Point it at your externally supplied "
            ".eim object-detection model (see README.md Model Import); the "
            "model file itself is never committed to this repo."
        )

    camera = CameraStream(CAMERA_URL)
    ball_detector = BallDetector(MODEL_PATH, min_confidence=BALL_CONFIDENCE)
    wall_detector = WallDetector(min_coverage_pct=WALL_MIN_COVERAGE_PCT)

    camera.open()
    ball_detector.open()
    print(f"[STRATEGY] model labels: {ball_detector.labels}")
    if BALL_LABEL not in ball_detector.labels:
        print(
            f"[STRATEGY] WARNING: configured ball label '{BALL_LABEL}' is not "
            f"in the model's labels -- set ROBOCUP_BALL_LABEL to match exactly"
        )

    return SoccerStrategy(
        robot,
        camera,
        ball_detector,
        wall_detector,
        config=Config(stale_frame_s=STALE_FRAME_S, ball_label=BALL_LABEL),
    )


def run_match() -> None:
    strategy = _build_strategy()
    try:
        while True:
            if not robot.is_running():
                raise ProgramStopped
            strategy.tick()
            time.sleep(0.05)
    finally:
        strategy.close()


def loop() -> None:
    global _last_toggle
    current = robot.hold_toggle()
    if current != _last_toggle:
        _last_toggle = current
        team = "BLUE" if current else "RED"
        print(f"[TEAM] switched to: {team}")

    if MODE == "demo":
        demo_sequence()
    else:
        run_match()


print(f"[INFO] mode: {MODE}")
print("[INFO] waiting for BOOT button to start...")
try:
    App.run(user_loop=lambda: robot.run_program(loop))
finally:
    robot.stop()
