# Ball following with the Edge Impulse model

Vision-driven ball following for the miniAuto: grab a frame from the ESP32-S3
camera, run the FOMO object-detection model, and turn the result into one short,
guarded motion per observation.

## Modules

| File | Role |
| --- | --- |
| `python/main.py` | App entry point. Wires camera, detector, policy, and robot together; gated by the BOOT button. |
| `python/ball_follower.py` | Pure decision logic: detections + sensors in, one short action (or stop) out. No hardware access, unit-testable. |
| `python/detector.py` | Chooses the inference backend (`brick` or `eim`). |
| `python/eim_runner.py` | Runs the `.eim` directly over its Unix-socket JSON protocol. |
| `python/debug_detect.py` | Live detection viewer. No motion, no BOOT gating. |
| `python/wall_detector.py` | Separate HSV red/blue field-side detector (no ML). |

## The model file

The `.eim` is **not** in this repository — `.gitignore` excludes model binaries
on purpose. Each person supplies it at runtime. Export it from Edge Impulse as
**Linux (AARCH64)** for the UNO Q.

Copy it onto the board, into the app folder so it is visible inside the app
container as `/app/models/`:

```bash
adb push your-model.eim /home/arduino/ArduinoApps/<app>/models/soccer-fomo.eim
adb shell 'chmod +x /home/arduino/ArduinoApps/<app>/models/soccer-fomo.eim'
```

## Two inference backends

Both expose the same call and return the same payload shape, with bounding boxes
in **source-frame pixels**, so they are interchangeable. Select with the
`DETECTOR_BACKEND` environment variable (default `brick`).

### `eim` — run the model ourselves (no board setup)

Launches the `.eim` as a subprocess and speaks its socket protocol. Needs only
the file above: no model registration, no `app.yaml` entry, no internet.

```bash
adb shell 'docker exec -e DETECTOR_BACKEND=eim miniautodriver-main-1 \
  python3 -u /app/python/debug_detect.py'
```

Override the path with `EIM_MODEL_PATH` if you name the file differently.

### `brick` — Arduino's `arduino:object_detection`

Uses Arduino's model-runner sidecar. It only accepts a **registered model ID**,
not a path, so the model must be added to the board's registry. This is a
board-level, system-file change that lives outside this repo and is lost when
`arduino-app-cli` updates — repeat it per board:

1. Put the `.eim` where the runner container can see it:

   ```bash
   adb shell 'mkdir -p /var/lib/arduino-app-cli/models/edge-impulse'
   adb push your-model.eim /var/lib/arduino-app-cli/models/edge-impulse/soccer-fomo.eim
   ```

2. Append an entry to `/var/lib/arduino-app-cli/assets/<version>/models-list.yaml`
   (back it up first):

   ```yaml
    - soccer-fomo:
       name : "Robot Soccer Cup FOMO"
       description: "Custom Edge Impulse FOMO object detection."
       model_labels: [goal, robot, soccer_ball]
       supported_boards: ["unoq"]
       bricks:
         - id: "arduino:object_detection"
           model_configuration:
             "EI_OBJ_DETECTION_MODEL": "/var/lib/arduino-app-cli/models/edge-impulse/soccer-fomo.eim"
         - id: "arduino:video_object_detection"
           model_configuration:
             "EI_V_OBJ_DETECTION_MODEL": "/var/lib/arduino-app-cli/models/edge-impulse/soccer-fomo.eim"
       deployment:
         handler: "ei-handler"
         pre-loaded: true
   ```

3. Confirm with `arduino-app-cli model list | grep soccer-fomo`. `app.yaml`
   already references it by ID.

## The policy

`BallFollower.decide()` returns one short action, or `None` meaning stop:

- Ball left of centre beyond the deadzone → `rotate_left`
- Ball right of centre beyond the deadzone → `rotate_right`
- Ball centred and the path is measurably clear → `forward`
- Otherwise → stop

Every action is a short pulse so a fresh observation follows immediately.

**Stop is the default** for: no model output, no ball, unusable bounding box,
confidence below threshold, arrival at the ball, and an unavailable distance
reading when the next move would be forward. An `ultrasonic_mm` of `-1` means the
I2C read failed; it is treated as unknown, never as "clear". Rotation is still
allowed with a dead distance sensor, since turning cannot close on an obstacle.

Tunables live at the top of `ball_follower.py`: `MIN_CONFIDENCE`,
`TURN_DEADZONE`, speeds, pulse durations, `ARRIVED_DISTANCE_MM`.

## Debugging

```bash
adb shell 'docker exec -it miniautodriver-main-1 python3 -u /app/python/debug_detect.py'
```

Prints every detection with confidence and centre-x, the action the policy would
take, and a running ball-hit rate. Writes annotated frames to
`captures/debug_latest.jpg`. Safe to leave running while repositioning the ball.

Note that `arduino.app_utils` and the bricks exist **only inside the app
container** — plain `ssh` into the board cannot import them. Use `docker exec`.

## Running it

```bash
adb shell 'arduino-app-cli app start user:<app>'
adb shell 'arduino-app-cli app logs user:<app>'
```

Then press the camera BOOT button to enable the program.

Motion is off by default: set `MOTION_ENABLED = True` in `main.py` only with the
robot in a safe setup and someone able to reach the power switch. Start with the
wheels raised and confirm it turns the correct way before putting it on a floor.

## Gotchas found on real hardware

- **`app stop` halts the MCU, and a warm-cache `app start` does not re-flash it.**
  You then get `Bridge health timed out` or `method health not available`. Fix:
  `app stop` → `app clean-cache` → `app start` (rebuilds and re-uploads, 1–2 min).
  Pushing only Python and restarting is fine while the MCU is already running.
- **The board drifts off the camera AP onto venue Wi-Fi**, which silently kills
  the stream. Pin it:
  `nmcli connection modify <CAM_SSID> connection.autoconnect-priority 100`
  and disable autoconnect on the venue network.
- **The camera's HTTP server hangs** while still answering ping. Power-cycle the
  camera module.
- **Ultrasonic and line sensor drop out intermittently** (`-1` / `line_ok:false`),
  together, because they share one I2C bus with the camera button controller at
  `0x79`, which the sketch polls every loop with no retry or timeout. The policy
  tolerates this, but adding retries in `i2cReadData` would help a lot.
- **The model over-reports `robot`** on out-of-distribution scenes (desks,
  laptops, people) at high confidence. Detection is much better with the robot at
  floor level looking at the pitch, matching how the training data was captured.
