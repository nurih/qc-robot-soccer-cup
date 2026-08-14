"""Wraps an externally supplied Edge Impulse `.eim` object-detection model
(README.md "Model Import": labels trained as soccerball/robot/goal, exported
for the Linux aarch64 runtime).

The model binary is never committed to this repo (see .gitignore) -- point
`model_path` at wherever your team places the exported .eim file on the
robot's filesystem. Requires `pip install edge-impulse-linux` on the UNO Q's
Linux side.
"""
from dataclasses import dataclass
from typing import List, Optional

import cv2


@dataclass
class Detection:
    label: str
    confidence: float
    x_center: float  # 0..1, fraction of frame width
    y_center: float  # 0..1, fraction of frame height
    width: float      # 0..1, fraction of frame width
    height: float     # 0..1, fraction of frame height


class ModelUnavailable(Exception):
    """Raised when the .eim model cannot be imported, loaded, or run."""


class BallDetector:
    def __init__(self, model_path: str, min_confidence: float = 0.5) -> None:
        self._model_path = model_path
        self._min_confidence = min_confidence
        self._runner = None
        self.labels: List[str] = []

    def open(self) -> None:
        try:
            from edge_impulse_linux.image import ImageImpulseRunner
        except ImportError as exc:
            raise ModelUnavailable(
                "edge_impulse_linux is not installed; pip install edge-impulse-linux"
            ) from exc

        runner = ImageImpulseRunner(self._model_path)
        try:
            model_info = runner.init()
        except Exception as exc:
            raise ModelUnavailable(
                f"could not load model at {self._model_path}"
            ) from exc

        self._runner = runner
        self.labels = model_info.get("model_parameters", {}).get("labels", [])

    def close(self) -> None:
        if self._runner is not None:
            self._runner.stop()
            self._runner = None

    def detect(self, frame_bgr) -> List[Detection]:
        if self._runner is None:
            raise ModelUnavailable("model is not open")

        img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        features, cropped = self._runner.get_features_from_image_auto_studio_settings(
            img_rgb
        )
        result = self._runner.classify(features)
        boxes = result.get("result", {}).get("bounding_boxes", [])

        # Normalize against the model's actual crop size rather than a
        # hardcoded 96x96, so this keeps working if the model input changes.
        crop_h, crop_w = cropped.shape[0], cropped.shape[1]

        detections = []
        for box in boxes:
            if box["value"] < self._min_confidence:
                continue
            detections.append(
                Detection(
                    label=box["label"],
                    confidence=box["value"],
                    x_center=(box["x"] + box["width"] / 2) / crop_w,
                    y_center=(box["y"] + box["height"] / 2) / crop_h,
                    width=box["width"] / crop_w,
                    height=box["height"] / crop_h,
                )
            )
        return detections

    def best(self, frame_bgr, label: str) -> Optional[Detection]:
        candidates = [d for d in self.detect(frame_bgr) if d.label == label]
        if not candidates:
            return None
        return max(candidates, key=lambda d: d.confidence)
