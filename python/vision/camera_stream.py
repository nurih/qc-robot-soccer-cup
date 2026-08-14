"""Timestamped access to the ESP32-S3 MJPEG stream.

Keeps camera transport isolated from decision logic so the strategy layer
can treat "frame age" as a first-class safety signal instead of a stream
implementation detail.
"""
from typing import Optional, Tuple

import cv2
import numpy as np

import time


class StreamUnavailable(Exception):
    """Raised when the camera stream cannot be opened or never produces a frame."""


class CameraStream:
    def __init__(
        self,
        url: str,
        warmup_frames: int = 5,
        warmup_timeout_s: float = 5.0,
    ) -> None:
        self._url = url
        self._warmup_frames = warmup_frames
        self._warmup_timeout_s = warmup_timeout_s
        self._cap: Optional[cv2.VideoCapture] = None
        self._last_frame_ts: float = 0.0

    def open(self) -> None:
        cap = cv2.VideoCapture(self._url)
        if not cap.isOpened():
            raise StreamUnavailable(f"could not open camera stream at {self._url}")

        deadline = time.monotonic() + self._warmup_timeout_s
        good_frames = 0
        while good_frames < self._warmup_frames:
            ok, frame = cap.read()
            if ok and frame is not None and frame.size:
                good_frames += 1
            if time.monotonic() > deadline:
                cap.release()
                raise StreamUnavailable(
                    f"camera stream at {self._url} did not warm up in time"
                )

        self._cap = cap
        self._last_frame_ts = time.monotonic()

    def read(self) -> Tuple[Optional[np.ndarray], float]:
        """Return (frame, age_seconds). `frame` is None on a failed read;
        `age_seconds` is time since the last successfully decoded frame."""
        if self._cap is None:
            raise StreamUnavailable("camera stream is not open")

        ok, frame = self._cap.read()
        now = time.monotonic()
        if ok and frame is not None and frame.size:
            self._last_frame_ts = now
            return frame, 0.0
        return None, now - self._last_frame_ts

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
