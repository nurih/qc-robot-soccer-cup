"""
debug_detect.py  -  live detection viewer for tuning, with no robot motion.

Prints every detection and the action the follower policy *would* take, and
writes an annotated JPEG so you can see the boxes. Nothing here moves hardware
and it does not wait for the BOOT button, so it is safe to leave running while
you reposition the ball or the camera.

Run it inside the app container (the bricks and the model runner are only
reachable from there):

    adb shell 'docker exec -it miniautodriver-main-1 python3 /app/python/debug_detect.py'

or over ssh to the board:

    ssh arduino@<board>.local "docker exec -it miniautodriver-main-1 python3 /app/python/debug_detect.py"

Annotated frames land in captures/debug_latest.jpg; pull one with:

    adb pull /home/arduino/ArduinoApps/miniautodriver/captures/debug_latest.jpg .
"""
import sys
import time

from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ball_follower import BallFollower, describe_action  # noqa: E402
from detector import make_detector  # noqa: E402

CAMERA_URL = "http://192.168.5.1:81/stream"
CAPTURES_FOLDER = Path(__file__).resolve().parent.parent / "captures"
ANNOTATED_PATH = CAPTURES_FOLDER / "debug_latest.jpg"

INTERVAL_SECONDS = 0.5

# Show everything the model emits, not just confident detections, so you can see
# near-misses while tuning. The follower applies its own threshold separately.
REPORT_CONFIDENCE = 0.01


def grab_frame(url: str = CAMERA_URL) -> bytes | None:
    buffer = b""
    response = None
    try:
        response = requests.get(url, stream=True, timeout=(8, 15))
        response.raise_for_status()
        for chunk in response.iter_content(4096):
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


def as_fraction(confidence) -> float:
    """Runner reports confidence as a percentage string; normalise to 0..1."""
    try:
        value = float(confidence)
    except (TypeError, ValueError):
        return 0.0
    return value / 100.0 if value > 1.0 else value


def annotate(jpeg: bytes, detection: dict) -> None:
    try:
        from arduino.app_utils.image import draw_bounding_boxes
    except ImportError:
        return
    try:
        image = draw_bounding_boxes(image=jpeg, detection=detection)
        if image is not None:
            CAPTURES_FOLDER.mkdir(parents=True, exist_ok=True)
            image.convert("RGB").save(ANNOTATED_PATH, "JPEG", quality=85)
    except Exception as err:  # annotation is a debug aid, never fatal
        print(f"[WARN] could not annotate frame: {err}")


def main() -> None:
    detector = make_detector(confidence=REPORT_CONFIDENCE)
    follower = BallFollower()

    # A distance that is always "clear", so the printed decision reflects what the
    # camera alone is saying rather than the (currently flaky) ultrasonic.
    assumed_sensors = {"ultrasonic_mm": 600}

    frames = 0
    ball_frames = 0
    print(f"[INFO] streaming from {CAMERA_URL} - Ctrl-C to stop")

    while True:
        jpeg = grab_frame()
        if jpeg is None:
            time.sleep(1)
            continue

        result = detector.detect(jpeg, image_type="jpg")
        items = (result or {}).get("detection") or []
        frames += 1

        balls = [i for i in items if str(i.get("class_name", "")).lower() == "soccer_ball"]
        ball_frames += bool(balls)

        if items:
            parts = []
            for item in items:
                label = item.get("class_name")
                confidence = as_fraction(item.get("confidence"))
                box = item.get("bounding_box_xyxy") or [0, 0, 0, 0]
                centre_x = (float(box[0]) + float(box[2])) / 2.0
                parts.append(f"{label} {confidence:.0%} cx={centre_x:.0f}")
            print(f"[{frames:4}] " + " | ".join(parts))
        else:
            print(f"[{frames:4}] (nothing)")

        action = follower.decide(result, 320, assumed_sensors)
        print(f"        {describe_action(action)}   ball seen in {ball_frames}/{frames} frames")

        annotate(jpeg, result)
        time.sleep(INTERVAL_SECONDS)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[INFO] stopped")
