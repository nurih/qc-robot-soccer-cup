"""
wall_detector.py  -  HSV-based wall / field-side detection.

The soccer field has coloured tape strips on the walls (red on one side, blue
on the other). This module detects which tape colour dominates the current
camera frame and compares it against the robot's active team
(robot.hold_toggle()) to report OWN SIDE or OPPONENT SIDE.

Pure HSV colour classification - no ML model or training data required.

Suggested use from main.py (per the developer journey guide's module layout):
camera setup (URL, VideoCapture, warmup) stays in main.py; pass the open
capture object's frames into WallDetector, which stays reusable/testable on
its own.
"""
import time

import cv2
import numpy as np

CAMERA_URL = "http://192.168.5.1:81/stream"

# OpenCV hue range is 0-179. Red wraps around 0/180, so it needs two ranges
# combined with bitwise_or. Saturation >= 120 and Value >= 70 filter out dark
# or washed-out pixels; tune both per your lighting conditions.
RED_HSV_RANGES = [
    ((0, 120, 70), (10, 255, 255)),
    ((160, 120, 70), (179, 255, 255)),
]
BLUE_HSV_RANGE = ((100, 120, 70), (130, 255, 255))

# ~2% of the frame is a good starting coverage threshold per the guide.
MIN_COVERAGE_FRACTION = 0.02

RED = "RED"
BLUE = "BLUE"
UNKNOWN = "UNKNOWN"


class WallDetector:
    """Classifies the dominant wall tape colour (RED/BLUE/UNKNOWN) from a BGR frame."""

    def __init__(self, min_coverage_fraction: float = MIN_COVERAGE_FRACTION) -> None:
        self.min_coverage_fraction = min_coverage_fraction

    def _coverage(self, hsv: np.ndarray, ranges: list[tuple[tuple[int, int, int], tuple[int, int, int]]]) -> float:
        mask = None
        for lower, upper in ranges:
            channel_mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
            mask = channel_mask if mask is None else cv2.bitwise_or(mask, channel_mask)
        total_pixels = hsv.shape[0] * hsv.shape[1]
        return float(cv2.countNonZero(mask)) / total_pixels if total_pixels else 0.0

    def coverage(self, frame: np.ndarray) -> tuple[float, float]:
        """Return (red_fraction, blue_fraction) of a BGR frame's pixels."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        red = self._coverage(hsv, RED_HSV_RANGES)
        blue = self._coverage(hsv, [BLUE_HSV_RANGE])
        return red, blue

    def detect(self, frame: np.ndarray) -> dict:
        """Classify the dominant wall colour in a single BGR frame."""
        red, blue = self.coverage(frame)

        side = UNKNOWN
        if red >= self.min_coverage_fraction and red >= blue:
            side = RED
        elif blue >= self.min_coverage_fraction and blue > red:
            side = BLUE

        return {"red": red, "blue": blue, "side": side}

    def classify_field_side(self, frame: np.ndarray, team_is_blue: bool) -> dict:
        """Compare the detected wall colour against the robot's active team.

        team_is_blue should come from robot.hold_toggle() (True = blue, False = red).
        """
        result = self.detect(frame)
        team = BLUE if team_is_blue else RED

        if result["side"] == UNKNOWN:
            field_side = UNKNOWN
        elif result["side"] == team:
            field_side = "OWN SIDE"
        else:
            field_side = "OPPONENT SIDE"

        result["team"] = team
        result["field_side"] = field_side
        return result


def open_stream(url: str = CAMERA_URL, warmup_timeout: float = 5.0) -> cv2.VideoCapture:
    """Open the MJPEG stream, discarding frames until a non-empty one arrives."""
    cap = cv2.VideoCapture(url)
    deadline = time.monotonic() + warmup_timeout
    while time.monotonic() < deadline:
        ok, frame = cap.read()
        if ok and frame is not None:
            break
        time.sleep(0.05)
    return cap


def format_diagnostic(result: dict) -> str:
    return f"[WallDetector] red={result['red'] * 100:.2f}%  blue={result['blue'] * 100:.2f}%  side={result['side']}"


def format_field_line(result: dict) -> str:
    return f"[FIELD] team={result['team']}  wall={result['side']} -> {result['field_side']}"


def main() -> None:
    """Standalone diagnostic loop: connects to the camera and prints classification."""
    detector = WallDetector()
    cap = open_stream()
    team_is_blue = False  # swap for robot.hold_toggle() once wired into main.py

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                time.sleep(0.05)
                continue
            result = detector.classify_field_side(frame, team_is_blue)
            print(format_diagnostic(result))
            print(format_field_line(result))
            time.sleep(0.2)
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()


if __name__ == "__main__":
    main()
