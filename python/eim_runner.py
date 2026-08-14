"""
eim_runner.py  -  drive an Edge Impulse .eim model directly, without the brick.

An .eim is a self-contained Linux executable that listens on a Unix socket and
speaks newline-delimited JSON. This module launches it and wraps that protocol
behind the same `detect(jpeg_bytes)` call the Arduino object_detection brick
exposes, so the two are interchangeable (see detector.py).

Why have this at all: the brick route requires registering the model in a
system-level registry on the board, which lives outside this repository and is
lost whenever arduino-app-cli updates. Running the .eim ourselves keeps the model
path in repo code and needs no board-level setup and no internet access.

Protocol, for reference (verified against this model's runner):
    -> {"id": 1, "hello": 1}
    <- {"id": 1, "model_parameters": {...}, "inferencing_engine": {...}}
    -> {"id": 2, "classify": [<features>]}
    <- {"id": 2, "success": true, "result": {"bounding_boxes": [...]}}

Every message needs a top-level "id". The runner serves a single client and
exits once that client disconnects, so the connection is held open for the
lifetime of this object.

For an RGB image impulse each feature is one pixel packed as 0xRRGGBB, in
row-major order, at the model's input resolution.
"""
import json
import os
import socket
import subprocess
import tempfile
import time

from pathlib import Path

import cv2
import numpy as np

DEFAULT_MODEL_PATH = "/app/models/soccer-fomo.eim"

STARTUP_TIMEOUT_SECONDS = 20
RESPONSE_TIMEOUT_SECONDS = 20


class EimError(RuntimeError):
    """Raised when the model process or its protocol misbehaves."""


class EimRunner:
    """Runs an .eim model and returns detections in the brick's output shape."""

    def __init__(self, model_path: str = DEFAULT_MODEL_PATH, socket_path: str | None = None) -> None:
        self.model_path = str(model_path)
        if not Path(self.model_path).exists():
            raise EimError(f"model not found: {self.model_path}")
        if not os.access(self.model_path, os.X_OK):
            raise EimError(f"model is not executable: {self.model_path}")

        self._socket_path = socket_path or os.path.join(
            tempfile.gettempdir(), f"eim-{os.getpid()}.sock"
        )
        self._process: subprocess.Popen | None = None
        self._sock: socket.socket | None = None
        self._buffer = b""
        self._next_id = 1

        self.input_width = 96
        self.input_height = 96
        self.labels: list[str] = []

        self._start()

    # -- lifecycle ---------------------------------------------------------

    def _start(self) -> None:
        if os.path.exists(self._socket_path):
            os.unlink(self._socket_path)

        self._process = subprocess.Popen(
            [self.model_path, self._socket_path],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )

        deadline = time.monotonic() + STARTUP_TIMEOUT_SECONDS
        while time.monotonic() < deadline:
            if self._process.poll() is not None:
                stderr = (self._process.stderr.read() or b"").decode(errors="replace")
                raise EimError(f"model exited during startup: {stderr.strip()[:400]}")
            if os.path.exists(self._socket_path):
                break
            time.sleep(0.05)
        else:
            raise EimError("timed out waiting for the model socket")

        self._sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._sock.settimeout(RESPONSE_TIMEOUT_SECONDS)
        self._sock.connect(self._socket_path)

        hello = self._request({"id": self._take_id(), "hello": 1})
        params = (hello or {}).get("model_parameters") or {}
        self.input_width = int(params.get("image_input_width") or self.input_width)
        self.input_height = int(params.get("image_input_height") or self.input_height)
        self.labels = list(params.get("labels") or [])

    def close(self) -> None:
        if self._sock is not None:
            try:
                self._sock.close()
            finally:
                self._sock = None
        if self._process is not None:
            self._process.terminate()
            try:
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._process.kill()
            self._process = None
        if os.path.exists(self._socket_path):
            try:
                os.unlink(self._socket_path)
            except OSError:
                pass

    def __enter__(self) -> "EimRunner":
        return self

    def __exit__(self, *_exc) -> None:
        self.close()

    # -- protocol ----------------------------------------------------------

    def _take_id(self) -> int:
        value = self._next_id
        self._next_id += 1
        return value

    def _request(self, message: dict) -> dict:
        if self._sock is None:
            raise EimError("not connected to the model")
        self._sock.sendall(json.dumps(message).encode() + b"\n")

        deadline = time.monotonic() + RESPONSE_TIMEOUT_SECONDS
        line = b""
        while True:
            if b"\n" in self._buffer:
                raw, self._buffer = self._buffer.split(b"\n", 1)
                # Responses are terminated with "\n\x00"; the trailing NUL is not
                # whitespace, so strip it explicitly or it parses as a blank line.
                line = raw.strip(b"\x00 \t\r\n")
                if line:
                    break
                continue
            if time.monotonic() > deadline:
                raise EimError("timed out waiting for a model response")
            chunk = self._sock.recv(65536)
            if not chunk:
                raise EimError("model closed the connection")
            self._buffer += chunk

        response = json.loads(line.decode())
        if response.get("success") is False:
            raise EimError(f"model reported failure: {response.get('error')}")
        return response

    # -- inference ---------------------------------------------------------

    def _features(self, jpeg: bytes) -> tuple[list[int], int, int]:
        image = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise EimError("could not decode frame")
        source_h, source_w = image.shape[:2]

        # "squash" resize: match the impulse, which ignores aspect ratio.
        resized = cv2.resize(
            image, (self.input_width, self.input_height), interpolation=cv2.INTER_AREA
        )
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.uint32)
        packed = (rgb[:, :, 0] << 16) | (rgb[:, :, 1] << 8) | rgb[:, :, 2]
        return packed.reshape(-1).tolist(), source_w, source_h

    def detect(self, image_bytes: bytes, image_type: str = "jpg", confidence: float | None = None) -> dict:
        """Detect objects in a JPEG, returning the brick's payload shape.

        Bounding boxes are scaled from model-input pixels back to source-frame
        pixels so both backends report in the same coordinate space.
        """
        features, source_w, source_h = self._features(image_bytes)
        response = self._request({"id": self._take_id(), "classify": features})

        boxes = ((response.get("result") or {}).get("bounding_boxes")) or []
        scale_x = source_w / float(self.input_width)
        scale_y = source_h / float(self.input_height)

        detections = []
        for box in boxes:
            score = float(box.get("value") or 0.0)
            if confidence is not None and score < confidence:
                continue
            x = float(box.get("x") or 0.0)
            y = float(box.get("y") or 0.0)
            width = float(box.get("width") or 0.0)
            height = float(box.get("height") or 0.0)
            detections.append(
                {
                    "class_name": box.get("label"),
                    # The brick reports a percentage string; match it so callers
                    # do not need to know which backend produced the result.
                    "confidence": f"{score * 100:.2f}",
                    "bounding_box_xyxy": [
                        x * scale_x,
                        y * scale_y,
                        (x + width) * scale_x,
                        (y + height) * scale_y,
                    ],
                }
            )

        return {"detection": detections}
