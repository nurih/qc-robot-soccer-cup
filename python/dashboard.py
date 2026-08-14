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

# Treat the strategy as idle once no tick has arrived for this long, and only
# then let the dashboard read the camera itself.
IDLE_AFTER_SECONDS = 1.5
IDLE_POLL_SECONDS = 0.5

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

    def __init__(
        self,
        port: int = 7000,
        backend: str = "",
        on_stop=None,
        camera_url: str = "",
    ) -> None:
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
            "health": {},
            "camera_idle_feed": False,
            "error": "",
        }
        self._last_tick_at = 0.0
        self._last_transition = ""
        self._transitions: list[str] = []

        # The one action the dashboard can take. Stopping is always safe, and the
        # firmware's stop also clears program_enabled, so this doubles as a
        # software kill switch alongside the physical BOOT button.
        self._on_stop = on_stop

        self._ui = WebUI(assets_dir_path=str(ASSETS_DIR))
        self._ui.expose_api("GET", "/state", self.api_state)
        self._ui.expose_api("GET", "/preview", self.api_preview)
        self._ui.expose_api("POST", "/stop", self.api_stop)

        # The strategy only ticks while the program is enabled, so without this
        # the feed would be blank until someone presses BOOT. This thread reads
        # the stream itself, but only while the strategy is idle, so the two
        # never hold the camera at the same time.
        self._camera_url = camera_url
        if camera_url:
            threading.Thread(target=self._idle_preview_loop, daemon=True).start()

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
        health: dict | None = None,
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

                self._state["camera_idle_feed"] = False
                self._state["state"] = state or self._state["state"]
                self._state["team"] = team or self._state["team"]
                self._state["detections"] = detections
                self._state["sensors"] = sensors or {}
                if health is not None:
                    self._state["health"] = health
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

    # -- idle preview -----------------------------------------------------

    def _strategy_is_live(self) -> bool:
        with self._lock:
            last = self._last_tick_at
        return bool(last) and (time.monotonic() - last) < IDLE_AFTER_SECONDS

    def _idle_preview_loop(self) -> None:
        """Show the camera while the robot is idle, so the feed is never blank."""
        import requests  # local import: only needed when an idle feed is wanted

        while True:
            if self._strategy_is_live():
                time.sleep(IDLE_POLL_SECONDS)
                continue

            response = None
            try:
                response = requests.get(self._camera_url, stream=True, timeout=(5, 10))
                response.raise_for_status()
                buffer = b""
                for chunk in response.iter_content(4096):
                    if self._strategy_is_live():
                        break
                    buffer += chunk
                    start = buffer.find(b"\xff\xd8")
                    end = buffer.find(b"\xff\xd9", start + 2)
                    if start == -1 or end == -1:
                        continue
                    jpeg, buffer = buffer[start:end + 2], buffer[end + 2:]
                    with self._lock:
                        self._preview = jpeg
                        self._last_preview_at = time.monotonic()
                        self._state["camera_idle_feed"] = True
                    time.sleep(PREVIEW_INTERVAL_SECONDS)
            except Exception as err:
                with self._lock:
                    self._state["error"] = f"camera: {str(err)[:160]}"
                time.sleep(2.0)
            finally:
                if response is not None:
                    response.close()

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
                "health": dict(self._state["health"]),
                "camera_idle_feed": self._state["camera_idle_feed"],
                "transitions": list(reversed(self._transitions)),
                "error": self._state["error"],
            }

    def api_stop(self) -> dict:
        """Stop the robot. Also clears program_enabled, so the routine ends."""
        if self._on_stop is None:
            return {"ok": False, "error": "stop not wired up"}
        try:
            self._on_stop()
            return {"ok": True}
        except Exception as err:
            self.note_error(f"stop failed: {err}")
            return {"ok": False, "error": str(err)[:180]}

    def api_preview(self) -> dict:
        with self._lock:
            data = self._preview
        return {"image": base64.b64encode(data).decode("ascii") if data else ""}
