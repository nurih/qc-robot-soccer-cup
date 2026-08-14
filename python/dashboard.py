"""
dashboard.py  -  live web view of what the strategy is seeing and deciding.

Read-only observer: it never touches the Bridge and never decides anything, so
attaching it cannot change robot behaviour. SoccerStrategy calls publish() once
per tick with the frame and telemetry it already has, which avoids opening a
second reader on the single CameraStream.

Serve it by constructing Dashboard() in the app entry point; browse to
http://<board-ip>:7000 (join the camera's Wi-Fi AP to reach it).
"""
import base64
import threading
import time

from pathlib import Path

import cv2

from arduino.app_bricks.web_ui import WebUI

ASSETS_DIR = Path(__file__).resolve().parent.parent / "assets"

# Encoding every frame is wasted work at strategy tick rate; the browser polls
# a few times a second.
PREVIEW_INTERVAL_SECONDS = 0.2
PREVIEW_JPEG_QUALITY = 80

# How many recent state transitions to keep for the dashboard.
TRANSITION_HISTORY = 12

BALL_COLOUR = (80, 220, 90)
OTHER_COLOUR = (170, 170, 170)


def as_fraction(confidence) -> float:
    """The detector reports confidence as a percentage string; normalise to 0..1."""
    try:
        value = float(confidence)
    except (TypeError, ValueError):
        return 0.0
    return value / 100.0 if value > 1.0 else value


def annotate(frame, detections: list):
    """Draw detection boxes and a centre line onto a BGR frame."""
    image = frame.copy()
    height, width = image.shape[:2]
    cv2.line(image, (width // 2, 0), (width // 2, height), (90, 90, 90), 1)

    for item in detections:
        label = str(item.get("class_name", "?"))
        score = as_fraction(item.get("confidence"))
        box = item.get("bounding_box_xyxy") or [0, 0, 0, 0]
        x1, y1, x2, y2 = (int(float(v)) for v in box[:4])
        colour = BALL_COLOUR if label in {"soccer_ball", "soccerball", "ball"} else OTHER_COLOUR
        cv2.rectangle(image, (x1, y1), (x2, y2), colour, 2)
        cv2.putText(
            image, f"{label} {score:.0%}", (x1, max(12, y1 - 4)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, colour, 1, cv2.LINE_AA,
        )
    return image


class Dashboard:
    """Collects strategy telemetry and serves it over HTTP."""

    def __init__(self, port: int = 7000, backend: str = "") -> None:
        self._lock = threading.Lock()
        self._preview = b""
        self._last_preview_at = 0.0
        self._started_at = time.monotonic()
        self._state = {
            "state": "idle",
            "team": "",
            "wall": {},
            "detections": [],
            "sensors": {},
            "ticks": 0,
            "ball_ticks": 0,
            "tick_rate": 0.0,
            "backend": backend,
            "error": "",
        }
        self._last_tick_at = 0.0
        self._last_transition = ""
        self._transitions: list[str] = []

        self._ui = WebUI(assets_dir_path=str(ASSETS_DIR))
        self._ui.expose_api("GET", "/state", self.api_state)
        self._ui.expose_api("GET", "/preview", self.api_preview)

    @property
    def url(self) -> str:
        return self._ui.url

    # -- called by the strategy ------------------------------------------

    def publish(
        self,
        frame=None,
        detection: dict | None = None,
        state: str = "",
        team: str = "",
        wall=None,
        sensors: dict | None = None,
        ball: dict | None = None,
        transition: str = "",
    ) -> None:
        """Record one tick of telemetry. Cheap, and never raises into the caller."""
        try:
            detections = (detection or {}).get("detection") or []
            now = time.monotonic()

            with self._lock:
                if self._last_tick_at:
                    elapsed = now - self._last_tick_at
                    if elapsed > 0:
                        self._state["tick_rate"] = round(1.0 / elapsed, 1)
                self._last_tick_at = now

                # Keep a short history so the state machine's path is visible,
                # not just where it happens to be right now.
                if transition and transition != self._last_transition:
                    self._last_transition = transition
                    self._transitions.append(transition)
                    del self._transitions[:-TRANSITION_HISTORY]

                self._state["state"] = state or self._state["state"]
                self._state["team"] = team or self._state["team"]
                self._state["detections"] = detections
                self._state["sensors"] = sensors or {}
                self._state["ticks"] += 1
                if ball is not None:
                    self._state["ball_ticks"] += 1
                if wall is not None:
                    self._state["wall"] = {
                        "side": getattr(wall, "side", ""),
                        "red_pct": round(getattr(wall, "red_pct", 0.0), 2),
                        "blue_pct": round(getattr(wall, "blue_pct", 0.0), 2),
                    }
                due = (now - self._last_preview_at) >= PREVIEW_INTERVAL_SECONDS

            if frame is not None and due:
                image = annotate(frame, detections)
                ok, encoded = cv2.imencode(
                    ".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, PREVIEW_JPEG_QUALITY]
                )
                if ok:
                    with self._lock:
                        self._preview = encoded.tobytes()
                        self._last_preview_at = now
        except Exception as err:  # telemetry must never break the robot loop
            with self._lock:
                self._state["error"] = f"dashboard: {str(err)[:180]}"

    def note_error(self, message: str) -> None:
        with self._lock:
            self._state["error"] = message[:200]

    # -- HTTP ------------------------------------------------------------

    def api_state(self) -> dict:
        with self._lock:
            ticks = self._state["ticks"]
            ball_ticks = self._state["ball_ticks"]
            return {
                "state": self._state["state"],
                "team": self._state["team"],
                "wall": dict(self._state["wall"]),
                "detections": [
                    {
                        "label": d.get("class_name"),
                        "confidence": round(as_fraction(d.get("confidence")) * 100, 1),
                        "box": [round(float(v), 1) for v in (d.get("bounding_box_xyxy") or [])],
                    }
                    for d in self._state["detections"]
                ],
                "sensors": dict(self._state["sensors"]),
                "ticks": ticks,
                "ball_rate": round(100.0 * ball_ticks / ticks, 1) if ticks else 0.0,
                "tick_rate": self._state["tick_rate"],
                "backend": self._state["backend"],
                "transitions": list(reversed(self._transitions)),
                "error": self._state["error"],
            }

    def api_preview(self) -> dict:
        with self._lock:
            data = self._preview
        return {"image": base64.b64encode(data).decode("ascii") if data else ""}
