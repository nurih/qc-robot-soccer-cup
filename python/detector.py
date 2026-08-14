"""
detector.py  -  pick how the Edge Impulse model is run.

Two interchangeable backends, both exposing `detect(jpeg_bytes, image_type=...)`
and returning `{"detection": [{"class_name", "confidence", "bounding_box_xyxy"}]}`
with boxes in source-frame pixels:

  "brick"  Arduino's arduino:object_detection brick. Needs the model registered
           in the board's model registry and declared in app.yaml. Proven path.

  "eim"    Run the .eim ourselves (see eim_runner.py). No board-level model
           registration, no internet, model path lives in repo code.

Select with the DETECTOR_BACKEND environment variable; defaults to the brick.

    DETECTOR_BACKEND=eim  python3 /app/python/debug_detect.py
"""
import os

DEFAULT_BACKEND = "brick"


def make_detector(backend: str | None = None, confidence: float | None = None):
    """Return a detector for the requested backend.

    `confidence` is the backend's own reporting threshold; the following policy
    applies its own threshold separately.
    """
    backend = (backend or os.environ.get("DETECTOR_BACKEND") or DEFAULT_BACKEND).strip().lower()

    if backend == "eim":
        from eim_runner import EimRunner

        model_path = os.environ.get("EIM_MODEL_PATH")
        runner = EimRunner(model_path) if model_path else EimRunner()
        print(
            f"[INFO] detector backend: eim ({runner.model_path}) "
            f"input={runner.input_width}x{runner.input_height} labels={runner.labels}"
        )
        return runner

    if backend == "brick":
        from arduino.app_bricks.object_detection import ObjectDetection

        detector = ObjectDetection(confidence=confidence) if confidence is not None else ObjectDetection()
        print("[INFO] detector backend: brick (arduino:object_detection)")
        return detector

    raise ValueError(f"unknown DETECTOR_BACKEND {backend!r}; use 'brick' or 'eim'")
