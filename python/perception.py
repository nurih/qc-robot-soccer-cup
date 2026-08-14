"""
perception.py  -  run vision continuously, off the control loop's thread.

Inference costs ~75 ms and wall detection ~4 ms. Doing that inline means the
control loop can never react faster than one inference, even though deciding and
issuing a motion command costs almost nothing. This worker runs capture ->
inference -> wall detection in its own thread and publishes the newest result;
the control loop reads whatever is current and never waits for the model.

The snapshot carries the age of the *frame the model actually saw*, so the
staleness guard accounts for inference latency rather than pretending the
decision is based on the present moment.

Only ever runs inference on a frame it has not already seen, so a fast worker
does not burn CPU re-analysing a stalled camera.
"""
import threading
import time

from dataclasses import dataclass, field
from typing import Any, Optional

import cv2
import numpy as np


@dataclass
class Snapshot:
    """One consistent view of the world: frame, detections, and wall, all aligned."""

    frame: np.ndarray
    detection: dict = field(default_factory=dict)
    wall: Any = None
    captured_at: float = 0.0

    def age(self) -> float:
        return time.monotonic() - self.captured_at


class PerceptionWorker:
    """Continuously turns camera frames into detections on a background thread."""

    def __init__(self, camera, detector, wall_detector, jpeg_quality: int = 80) -> None:
        self._camera = camera
        self._detector = detector
        self._wall = wall_detector
        self._jpeg_quality = jpeg_quality

        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._snapshot: Optional[Snapshot] = None

        self._last_captured_at = 0.0
        self._inferences = 0
        self._skipped = 0
        self._errors = 0
        self._infer_ms = 0.0

    def start(self) -> None:
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                frame, age_s = self._camera.read()
                if frame is None:
                    time.sleep(0.01)
                    continue

                captured_at = time.monotonic() - age_s
                # Nothing new since the last pass: do not pay for inference twice
                # on the same picture.
                if captured_at <= self._last_captured_at:
                    time.sleep(0.005)
                    with self._lock:
                        self._skipped += 1
                    continue
                self._last_captured_at = captured_at

                started = time.monotonic()
                ok, buffer = cv2.imencode(
                    ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, self._jpeg_quality]
                )
                if not ok:
                    continue
                detection = self._detector.detect(buffer.tobytes(), image_type="jpg")
                wall = self._wall.detect(frame)
                elapsed_ms = (time.monotonic() - started) * 1000.0

                with self._lock:
                    self._snapshot = Snapshot(
                        frame=frame,
                        detection=detection or {},
                        wall=wall,
                        captured_at=captured_at,
                    )
                    self._inferences += 1
                    # Exponential average keeps the number current without a buffer.
                    self._infer_ms = (
                        elapsed_ms if not self._infer_ms
                        else 0.8 * self._infer_ms + 0.2 * elapsed_ms
                    )
            except Exception:
                with self._lock:
                    self._errors += 1
                time.sleep(0.05)

    def latest(self) -> Optional[Snapshot]:
        """Newest completed perception result, or None before the first one."""
        with self._lock:
            return self._snapshot

    def stats(self) -> dict:
        with self._lock:
            return {
                "inferences": self._inferences,
                "skipped": self._skipped,
                "errors": self._errors,
                "infer_ms": round(self._infer_ms, 1),
            }

    def close(self) -> None:
        self._stop.set()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=2.0)
            self._thread = None
