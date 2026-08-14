"""
main.py  -  soccer ball follower for the Hiwonder miniAuto on UNO Q.

Pipeline per the developer journey guide's application loop:
  acquire frame -> infer -> validate -> decide -> one short guarded action -> reobserve

Motion is gated three ways:
  * MOTION_ENABLED below (set False to observe detections without moving)
  * the BOOT button program-enable flow (robot.run_program)
  * robot.stop() in a finally block

The Edge Impulse .eim model is supplied at runtime, not stored in this repo.
See app.yaml for the path the object_detection brick loads it from.
"""
import time

from pathlib import Path

import cv2
import numpy as np
import requests

from arduino.app_utils import App

from ball_follower import BallFollower, describe_action
from detector import make_detector
from robot_client import MiniAutoRobot

CAMERA_URL = "http://192.168.5.1:81/stream"

PROJECT_FOLDER = Path(__file__).resolve().parent.parent
CAPTURES_FOLDER = PROJECT_FOLDER / "captures"

STREAM_TIMEOUT_SECONDS = 10
FRAME_READ_TIMEOUT_SECONDS = 15

# Set True only with the robot in a safe setup (wheels raised or clear floor space)
# and someone able to reach the power switch.
MOTION_ENABLED = False

# Log every detection payload for the first few frames so the label names and
# bounding-box coordinate space can be confirmed against the trained model.
DEBUG_FRAMES = 3


def grab_frame(url: str = CAMERA_URL, timeout: int = FRAME_READ_TIMEOUT_SECONDS) -> bytes | None:
    """Read the MJPEG stream until one complete JPEG (SOI..EOI) arrives."""
    deadline = time.monotonic() + timeout
    buffer = b""
    response = None
    try:
        response = requests.get(url, stream=True, timeout=(5, STREAM_TIMEOUT_SECONDS))
        response.raise_for_status()
        for chunk in response.iter_content(chunk_size=4096):
            if time.monotonic() > deadline:
                print("[WARN] timed out waiting for a complete frame")
                break
            buffer += chunk
            start = buffer.find(b"\xff\xd8")
            end = buffer.find(b"\xff\xd9", start + 2)
            if start != -1 and end != -1:
                return buffer[start:end + 2]
    except requests.exceptions.RequestException as err:
        print(f"[ERROR] camera stream unavailable: {err}")
    finally:
        if response is not None:
            response.close()
    return None


def frame_size(jpeg: bytes) -> tuple[int, int]:
    """Return (width, height) of a JPEG, or (0, 0) if it cannot be decoded."""
    image = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        return 0, 0
    return image.shape[1], image.shape[0]


robot = MiniAutoRobot()
print(f"health   : {robot.health()}")
print(f"sensors  : {robot.read_sensors()}")
print(f"capture folder: {CAPTURES_FOLDER}")
print(f"[INFO] motion {'ENABLED' if MOTION_ENABLED else 'DISABLED (observe only)'}")

detector = make_detector()
follower = BallFollower()

_frames_seen = 0


def observe_and_act() -> None:
    """One pass of the application loop."""
    global _frames_seen

    jpeg = grab_frame()
    if jpeg is None:
        print("[SAFE] no frame -> stop")
        robot.stop()
        return

    width, height = frame_size(jpeg)
    result = detector.detect(jpeg, image_type="jpg")

    _frames_seen += 1
    if _frames_seen <= DEBUG_FRAMES:
        print(f"[DEBUG] frame {width}x{height} raw detection payload: {result}")

    action = follower.decide(result, width, robot.read_sensors())
    print(describe_action(action))

    if action is None:
        robot.stop()
        return

    if not MOTION_ENABLED:
        return

    command, speed, ms = action
    robot.drive(command, speed, ms)


def loop() -> None:
    observe_and_act()
    time.sleep(0.05)


print("[INFO] waiting for BOOT button to start...")
try:
    App.run(user_loop=lambda: robot.run_program(loop))
finally:
    robot.stop()
