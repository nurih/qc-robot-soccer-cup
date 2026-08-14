"""HSV classification of the field's red/blue tape, per README.md's
"Wall / Field-Side Detection" section. No ML model or training data required.
"""
from dataclasses import dataclass

import cv2
import numpy as np

# OpenCV HSV ranges: H 0-179, S 0-255, V 0-255. Red wraps around 0/180.
RED_RANGE_1 = ((0, 120, 70), (10, 255, 255))
RED_RANGE_2 = ((160, 120, 70), (179, 255, 255))
BLUE_RANGE = ((100, 120, 70), (130, 255, 255))


@dataclass
class WallReading:
    side: str  # "RED", "BLUE", or "UNKNOWN"
    red_pct: float
    blue_pct: float


class WallDetector:
    def __init__(self, min_coverage_pct: float = 2.0) -> None:
        self._min_coverage_pct = min_coverage_pct

    def detect(self, frame_bgr: np.ndarray) -> WallReading:
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        total_px = hsv.shape[0] * hsv.shape[1]

        red_mask = cv2.bitwise_or(
            cv2.inRange(hsv, *RED_RANGE_1),
            cv2.inRange(hsv, *RED_RANGE_2),
        )
        blue_mask = cv2.inRange(hsv, *BLUE_RANGE)

        red_pct = 100.0 * cv2.countNonZero(red_mask) / total_px
        blue_pct = 100.0 * cv2.countNonZero(blue_mask) / total_px

        if red_pct < self._min_coverage_pct and blue_pct < self._min_coverage_pct:
            return WallReading("UNKNOWN", red_pct, blue_pct)
        side = "RED" if red_pct >= blue_pct else "BLUE"
        return WallReading(side, red_pct, blue_pct)
