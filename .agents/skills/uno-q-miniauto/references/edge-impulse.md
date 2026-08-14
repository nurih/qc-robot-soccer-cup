# Edge Impulse workflow

Use this reference for dataset capture, labeling, impulse configuration, deployment, `.eim` files, or on-robot inference. Read the live `README.md` Model Import section and `python/capture.py` before making changes; they are the source of truth when details differ.

## Current repository pieces

- `python/capture.py` reads `http://192.168.5.1:81/stream`, extracts JPEG frames, publishes a WebUI preview at `GET /preview`, and saves labeled images under `captures/`.
- Its current label folders are `captures/soccer_ball/`, `captures/robot/`, and `captures/empty/`.
- Holding the mapped capture control saves at most one JPEG every `0.5` seconds. Capture quality is `92`; preview quality is `85`.
- The capture utility requires the App Lab runtime plus `requests` and Pillow. Verify how those packages are supplied before adding installation instructions.
- Its current camera request uses an unbounded read timeout and retains the last decoded frame without a freshness timestamp. For reliable collection, add a bounded read timeout plus frame timestamp/sequence checks before saving, or disclose the risk of stale captures.
- `docs/assets/edge-impulse/` contains screenshots used by the README tutorial.
- `.gitignore` excludes `.eim` and other common model formats. Do not force-add generated model binaries.
- The README links to public Edge Impulse project `1085406` as an optional source of pre-labeled capture data.

## Resolve the capture-control contract first

`python/capture.py` currently maps dynamic Bridge calls as follows:

| Bridge method | Capture folder |
| --- | --- |
| `object_a_count` | `soccer_ball` |
| `object_b_count` | `robot` |
| `object_c_count` | `empty` |

Do not assume these controls work merely because the Python file exists. At the time this reference was written, the checked-out `sketch/sketch.ino` registered the eight robot RPCs but did not register these three capture methods.

Before running capture:

1. Search the live sketch for each method and inspect every `Bridge.provide_safe(...)` registration.
2. If providers are absent, ask which physical buttons or UI controls should label captures. Do not invent Modulino wiring, pins, or dependencies.
3. If physical button counters are desired, implement their MCU reads, debounce/hold semantics, RPC functions, and registrations together. The current Python code captures only when a counter increases, so a held button must increment repeatedly at documented intervals or the Python control logic must change. Keep the implementation non-blocking and avoid pins already used by the robot or diagnostics.
4. If WebUI controls are preferred, refactor capture selection around explicit UI actions and remove the unavailable Bridge dependency.
5. Confirm App Lab's active Python entry point before switching between `python/main.py` and `python/capture.py`.
6. Keep capture mode motion-free and call `robot.stop()` first if it shares a running robot session.

## Capture and curate the dataset

1. Connect the UNO Q host to the camera access point and verify the MJPEG endpoint before collecting data. Read the live `CAMERA_SSID` and `CAMERA_PASS` constants from camera firmware; the checked-in README and firmware currently disagree on the password.
2. Choose one canonical class vocabulary and use it consistently in capture folders, Edge Impulse labels, exported-model metadata, and inference policy. The current documentation contains both `soccerball` and `soccer_ball`; resolve that mismatch rather than coding aliases implicitly.
3. Resolve the current class-set mismatch: capture provides `soccer_ball`, `robot`, and `empty`, while the documented detector outputs are `soccer_ball`, `robot`, and `goal`. Decide whether `empty` means unlabeled negative images and add a verified way to collect `goal` examples before training a three-object detector.
4. Capture varied distance, angle, lighting, occlusion, background, and opponent examples. Keep train, validation, and test scenes meaningfully distinct.
5. Include negative images with no target objects. Confirm whether Edge Impulse expects them as unlabeled images or as a project-specific `empty` class; do not draw artificial boxes around empty space.
6. Review images for blur, duplicates, corrupt frames, accidental labels, and data leakage before upload.
7. Label visible `soccer_ball`, `robot`, and `goal` objects with tight, consistent bounding boxes according to the chosen vocabulary.

Do not upload captures or modify an external Edge Impulse project unless the user explicitly authorizes that external write.

## Configure and train the impulse

The checked-in README currently documents this starting configuration:

- Input image: `96 x 96`, RGB, squash resize, 100% data subset.
- Processing block: Image.
- Learning block: Object Detection (Images).
- Intended outputs: `goal`, `robot`, and `soccer_ball`.
- Starting training settings: 150–180 cycles, learning rate `0.001`, CPU training.

Treat these as a baseline, not universal optimum. Verify the current Edge Impulse UI and available architecture in the user's project. Record the exact class list, split, preprocessing, architecture, thresholds, metrics, and deployment target used. Evaluate per-class precision/recall, confusion patterns, false positives on field elements, and performance on held-out robot-camera frames before deployment.

## Export and place the model

1. Confirm the actual UNO Q App Lab runtime architecture before export; use Linux aarch64 only when verified.
2. Select a deployment target that produces the runtime artifact required by the application. The README currently describes an Arduino UNO Q deployment and a Linux aarch64 `.eim`; reconcile those selections in the live Edge Impulse project rather than assuming they are interchangeable.
3. Keep the downloaded artifact untracked. The recommended local convention is:

   ```text
   models/
     <project>-linux-aarch64.eim
   ```

4. Pass the path through App Lab configuration or a task-specific environment variable instead of hard-coding a developer's absolute path.
5. Document project/version, export target, class order, input shape, and expected runtime alongside code without committing the binary.

## Integrate inference on the UNO Q

No inference runner is currently checked into this repository. Do not invent an Edge Impulse Python API from memory.

1. Inspect the UNO Q/App Lab environment for the installed Edge Impulse runtime and its exact invocation API. If it is absent, present the required dependency or runtime choice before implementation.
2. Add a focused inference module under `python/`; keep `MiniAutoRobot` as the only robot-control facade.
3. Reuse one camera connection where possible. Bound connection, frame-read, and inference timeouts and recover from stream interruption.
4. Validate model metadata at startup: model file exists, input shape is supported, and required labels are present.
5. Convert detections into a small typed result containing label, confidence, bounding box, frame timestamp, and inference latency.
6. Keep perception separate from motion policy. Require minimum confidence and persistence across frames; stop on stale frames, missing detections, runtime errors, disabled program state, or contradictory detections.
7. Start with a diagnostic entry point that performs camera plus inference only and never moves the robot. Add bounded, short-duration movement only after diagnostic results are reviewed.
8. Preserve `run_program(...)`, interruption checks, and `finally: robot.stop()` in any moving entry point.

## Validate progressively

1. Run `python3 -m py_compile python/*.py` and the bundled strict contract checker.
2. Test camera acquisition and preview without capture or motion.
3. Test each labeling control and confirm images land in the intended folder.
4. Inspect a sample from every class before upload.
5. Evaluate the trained model on held-out data and record per-class results.
6. Run exported-model inference against saved images, then the live stream, with motors disabled.
7. Measure end-to-end frame age and inference latency on the UNO Q.
8. Only with explicit authorization and a safe physical setup, test low-speed short pulses before autonomous motion.

Report separately what was verified in static checks, in Edge Impulse, on the UNO Q runtime, and on physical hardware.
