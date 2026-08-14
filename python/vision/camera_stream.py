"""Timestamped access to the ESP32-S3 MJPEG stream.

Keeps camera transport isolated from decision logic so the strategy layer
can treat "frame age" as a first-class safety signal instead of a stream
implementation detail.

A background thread drains the stream continuously and keeps only the newest
frame. Reading directly from cv2.VideoCapture on demand does not work here: the
camera pushes frames far faster than the perception loop consumes them, and
read() returns the *oldest* buffered frame, so latency grows without bound (tens
of seconds in practice). Draining in a thread keeps the buffer empty, so the
strategy always sees the present rather than the past.
"""
import threading
import time

from typing import Optional, Tuple

import cv2
import numpy as np


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

        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._frame: Optional[np.ndarray] = None
        self._frame_ts: float = 0.0
        self._frames_read = 0
        self._frames_dropped = 0
        self._consumed = True

    def open(self) -> None:
        cap = cv2.VideoCapture(self._url)
        if not cap.isOpened():
            raise StreamUnavailable(f"could not open camera stream at {self._url}")

        # Ask the backend to keep no backlog. Not honoured by every build, which
        # is why the draining thread below is the real fix rather than a tweak.
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass

        deadline = time.monotonic() + self._warmup_timeout_s
        good_frames = 0
        while good_frames < self._warmup_frames:
            ok, frame = cap.read()
            if ok and frame is not None and frame.size:
                good_frames += 1
                with self._lock:
                    self._frame = frame
                    self._frame_ts = time.monotonic()
            if time.monotonic() > deadline:
                cap.release()
                raise StreamUnavailable(
                    f"camera stream at {self._url} did not warm up in time"
                )

        self._cap = cap
        self._stop.clear()
        self._thread = threading.Thread(target=self._drain_loop, daemon=True)
        self._thread.start()

    def _drain_loop(self) -> None:
        """Consume frames as fast as they arrive, keeping only the latest."""
        while not self._stop.is_set():
            cap = self._cap
            if cap is None:
                break
            ok, frame = cap.read()
            if not ok or frame is None or not frame.size:
                # Transient decode hiccup; the age returned by read() will grow
                # and the strategy's staleness guard handles it.
                time.sleep(0.01)
                continue
            with self._lock:
                if not self._consumed:
                    # Superseded before the strategy got to it: this is the
                    # backlog we are deliberately throwing away.
                    self._frames_dropped += 1
                self._frame = frame
                self._frame_ts = time.monotonic()
                self._frames_read += 1
                self._consumed = False

    def read(self) -> Tuple[Optional[np.ndarray], float]:
        """Return (newest frame, age_seconds).

        age_seconds is measured from when the frame was actually received, so a
        stalled stream shows up as a growing age rather than a silent lag.
        """
        with self._lock:
            frame = self._frame
            ts = self._frame_ts
            self._consumed = True
        if frame is None:
            return None, 0.0
        # The same frame may be returned twice if the loop outruns the camera;
        # its age simply grows, which is what the staleness guard wants to see.
        return frame, time.monotonic() - ts

    def stats(self) -> dict:
        with self._lock:
            return {
                "frames_read": self._frames_read,
                "frames_dropped": self._frames_dropped,
            }

    def close(self) -> None:
        self._stop.set()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=2.0)
            self._thread = None
        if self._cap is not None:
            self._cap.release()
            self._cap = None
