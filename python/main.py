"""
main.py  -  soccer ball follower with a live web dashboard.

Two loops:

  * A perception thread continuously grabs a frame, runs the model, and works out
    what the follower *would* do. It never touches the Bridge, so it is safe to
    run alongside the motion loop, and it keeps the dashboard live even while the
    robot is idle.

  * The follow routine, gated by the BOOT button, reads the latest decision and
    issues one short guarded motion at a time.

Dashboard: http://<board-ip>:7000 (join the camera's Wi-Fi AP to reach it).

Motion is gated three ways: MOTION_ENABLED below, the BOOT button program-enable
flow, and robot.stop() in a finally block.
"""
import base64
import io
import threading
import time

from pathlib import Path

import cv2
import numpy as np
import requests

from arduino.app_utils import App
from arduino.app_bricks.web_ui import WebUI

from ball_follower import BallFollower, describe_action
from detector import make_detector
from robot_client import MiniAutoRobot

CAMERA_URL = "http://192.168.5.1:81/stream"

PROJECT_FOLDER = Path(__file__).resolve().parent.parent
CAPTURES_FOLDER = PROJECT_FOLDER / "captures"

STREAM_TIMEOUT_SECONDS = 10
FRAME_READ_TIMEOUT_SECONDS = 15

# Set True only with the robot in a safe setup (wheels raised or clear floor
# space) and someone able to reach the power switch.
MOTION_ENABLED = True

PERCEPTION_INTERVAL_SECONDS = 0.05
PREVIEW_JPEG_QUALITY = 80

# Show low-confidence detections on the dashboard so near-misses are visible
# while tuning; the follower applies its own MIN_CONFIDENCE separately.
REPORT_CONFIDENCE = 0.01

robot = MiniAutoRobot()
follower = BallFollower()

_lock = threading.Lock()
_state = {
    "connected": False,
    "detections": [],
    "action": None,
    "sensors": {},
    "frames": 0,
    "ball_frames": 0,
    "fps": 0.0,
    "backend": "",
    "motion_enabled": MOTION_ENABLED,
    "error": "",
}
_preview_jpeg = b""


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
                return None
            buffer += chunk
            start = buffer.find(b"\xff\xd8")
            end = buffer.find(b"\xff\xd9", start + 2)
            if start != -1 and end != -1:
                return buffer[start:end + 2]
    except requests.exceptions.RequestException as err:
        with _lock:
            _state["error"] = str(err)[:200]
    finally:
        if response is not None:
            response.close()
    return None


def as_fraction(confidence) -> float:
    """The runner reports confidence as a percentage string; normalise to 0..1."""
    try:
        value = float(confidence)
    except (TypeError, ValueError):
        return 0.0
    return value / 100.0 if value > 1.0 else value


def annotate(jpeg: bytes, detections: list) -> bytes:
    """Draw detection boxes onto the frame, returning JPEG bytes."""
    image = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        return jpeg

    height, width = image.shape[:2]
    cv2.line(image, (width // 2, 0), (width // 2, height), (90, 90, 90), 1)

    for item in detections:
        label = str(item.get("class_name", "?"))
        score = as_fraction(item.get("confidence"))
        box = item.get("bounding_box_xyxy") or [0, 0, 0, 0]
        x1, y1, x2, y2 = (int(float(v)) for v in box[:4])
        # Ball in green, everything else muted so the target stands out.
        colour = (80, 220, 90) if label == "soccer_ball" else (170, 170, 170)
        cv2.rectangle(image, (x1, y1), (x2, y2), colour, 2)
        cv2.putText(
            image, f"{label} {score:.0%}", (x1, max(12, y1 - 4)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, colour, 1, cv2.LINE_AA,
        )

    ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, PREVIEW_JPEG_QUALITY])
    return encoded.tobytes() if ok else jpeg


def perception_loop() -> None:
    """Continuously perceive. Never calls the Bridge, so it is safe in a thread."""
    global _preview_jpeg

    detector = make_detector(confidence=REPORT_CONFIDENCE)
    with _lock:
        _state["backend"] = getattr(detector, "model_path", "brick")

    last = time.monotonic()
    while True:
        jpeg = grab_frame()
        if jpeg is None:
            with _lock:
                _state["connected"] = False
            time.sleep(1.0)
            continue

        try:
            result = detector.detect(jpeg, image_type="jpg")
        except Exception as err:  # keep perceiving even if one inference fails
            with _lock:
                _state["error"] = f"inference: {str(err)[:180]}"
            time.sleep(0.5)
            continue

        detections = (result or {}).get("detection") or []

        with _lock:
            sensors = dict(_state["sensors"])
        action = follower.decide(result, 320, sensors)

        preview = annotate(jpeg, detections)
        now = time.monotonic()
        elapsed = now - last
        last = now

        with _lock:
            _state["connected"] = True
            _state["error"] = ""
            _state["detections"] = detections
            _state["action"] = action
            _state["frames"] += 1
            if any(str(d.get("class_name")) == "soccer_ball" for d in detections):
                _state["ball_frames"] += 1
            if elapsed > 0:
                _state["fps"] = round(1.0 / elapsed, 1)
            _preview_jpeg = preview

        time.sleep(PERCEPTION_INTERVAL_SECONDS)


def follow_routine() -> None:
    """Act on the latest decision until the BOOT button disables the program.

    run_program() runs this once per button press, so the loop lives here.
    robot.stop() is never called from inside it: stop() also clears
    program_enabled and would end the session. Motion is already bounded by the
    firmware's timed auto-stop, so "stop" simply means issue no new motion.
    """
    while robot.is_running():
        with _lock:
            action = _state["action"]

        if action is not None and MOTION_ENABLED:
            command, speed, ms = action
            robot.drive(command, speed, ms)
        else:
            time.sleep(0.05)


# -- web dashboard ---------------------------------------------------------

def api_state() -> dict:
    with _lock:
        frames = _state["frames"]
        ball_frames = _state["ball_frames"]
        return {
            "connected": _state["connected"],
            "detections": [
                {
                    "label": d.get("class_name"),
                    "confidence": round(as_fraction(d.get("confidence")) * 100, 1),
                    "box": [round(float(v), 1) for v in (d.get("bounding_box_xyxy") or [])],
                }
                for d in _state["detections"]
            ],
            "action": describe_action(_state["action"]).replace("[POLICY] ", ""),
            "sensors": _state["sensors"],
            "frames": frames,
            "ball_rate": round(100.0 * ball_frames / frames, 1) if frames else 0.0,
            "fps": _state["fps"],
            "backend": _state["backend"],
            "motion_enabled": _state["motion_enabled"],
            "error": _state["error"],
        }


def api_preview() -> dict:
    with _lock:
        data = _preview_jpeg
    if not data:
        return {"image": ""}
    return {"image": base64.b64encode(data).decode("ascii")}


ui = WebUI(assets_dir_path=str(PROJECT_FOLDER / "assets"))
ui.expose_api("GET", "/state", api_state)
ui.expose_api("GET", "/preview", api_preview)

print(f"health   : {robot.health()}")
print(f"sensors  : {robot.read_sensors()}")
print(f"[INFO] motion {'ENABLED' if MOTION_ENABLED else 'DISABLED (observe only)'}")
print(f"[INFO] dashboard: {ui.url}")

threading.Thread(target=perception_loop, daemon=True).start()


def user_loop() -> None:
    """Refresh sensors for the dashboard and policy, then run the gated routine."""
    try:
        sensors = robot.read_sensors()
        with _lock:
            _state["sensors"] = sensors
    except Exception as err:
        with _lock:
            _state["error"] = f"sensors: {str(err)[:180]}"

    robot.run_program(follow_routine)


print("[INFO] waiting for BOOT button to start...")
try:
    App.run(user_loop=user_loop)
finally:
    robot.stop()
