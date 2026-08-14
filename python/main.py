"""
main.py  -  grab a frame from the ESP32-S3 camera and save it on the robot.

Captures a single JPEG frame from the camera's MJPEG stream at startup and
writes it under captures/ in the app folder, which lives on the board's
filesystem at ~/ArduinoApps/<app>/captures/ so it can be pulled off with
`adb pull`.

The board must be joined to the camera's Wi-Fi access point for this to work.
"""
import time

from pathlib import Path

import cv2
import numpy as np
import requests

from arduino.app_utils import App
from robot_client import MiniAutoRobot

CAMERA_URL = "http://192.168.5.1:81/stream"

PROJECT_FOLDER = Path(__file__).resolve().parent.parent
CAPTURES_FOLDER = PROJECT_FOLDER / "captures"

STREAM_TIMEOUT_SECONDS = 10
FRAME_READ_TIMEOUT_SECONDS = 15

# The FOMO impulse expects 96x96 RGB with "squash" resize, i.e. the 4:3 frame is
# distorted to square rather than cropped. Training data must go through the same
# path so the model sees the same distortion at inference time.
MODEL_INPUT_SIZE = (96, 96)


def grab_frame(url: str = CAMERA_URL, timeout: int = FRAME_READ_TIMEOUT_SECONDS) -> bytes | None:
    """Read the MJPEG stream until one complete JPEG (SOI..EOI) arrives."""
    deadline = time.monotonic() + timeout
    buffer = b""
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
        try:
            response.close()
        except NameError:
            pass
    return None


def squash_to_model_input(jpeg: bytes, size: tuple[int, int] = MODEL_INPUT_SIZE):
    """Decode a JPEG and squash it to the model's input size (aspect ratio ignored)."""
    frame = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        return None
    # INTER_AREA is the right filter for downscaling; it averages rather than samples.
    return cv2.resize(frame, size, interpolation=cv2.INTER_AREA)


def save_frame(jpeg: bytes) -> tuple[Path, Path | None]:
    """Save the full-resolution frame plus a squashed model-input copy."""
    CAPTURES_FOLDER.mkdir(parents=True, exist_ok=True)
    stamp = int(time.time())

    full_path = CAPTURES_FOLDER / f"frame_{stamp}.jpg"
    full_path.write_bytes(jpeg)

    resized = squash_to_model_input(jpeg)
    if resized is None:
        print("[WARN] could not decode frame for resize")
        return full_path, None

    width, height = MODEL_INPUT_SIZE
    small_path = CAPTURES_FOLDER / f"frame_{stamp}_{width}x{height}.jpg"
    cv2.imwrite(str(small_path), resized)
    return full_path, small_path


robot = MiniAutoRobot()
print(f"health   : {robot.health()}")
print(f"sensors  : {robot.read_sensors()}")
print(f"capture folder: {CAPTURES_FOLDER}")

print(f"[INFO] connecting to camera: {CAMERA_URL}")
frame = grab_frame()

if frame is None:
    print("[ERROR] no frame captured - is the board joined to the camera Wi-Fi AP?")
else:
    full_path, small_path = save_frame(frame)
    print(f"[CAPTURE] saved {len(frame)} bytes -> {full_path}")
    if small_path is not None:
        width, height = MODEL_INPUT_SIZE
        print(f"[CAPTURE] {width}x{height} model input -> {small_path}")


def loop() -> None:
    """Nothing to do after the capture; idle so the app stays alive."""
    time.sleep(5)


App.run(user_loop=loop)
