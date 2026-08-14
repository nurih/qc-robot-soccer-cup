# Robot Soccer — Behavior Architecture

Design document for the autonomous soccer behavior stack.
Read this before picking up a submodule.

---

## Context

**Robot:** Hiwonder miniAuto on Arduino UNO Q — 4 mecanum wheels (holonomic),
ultrasonic distance sensor, 4-channel line sensor, onboard RGB LED and buzzer.

**Vision:** Hiwonder ESP32-S3 camera → MJPEG stream at
`http://192.168.5.1:81/stream` (320×240 QVGA).
An Edge Impulse **object-detection** model (`.eim`, not committed — see
`.gitignore`) runs on the Python/Linux side and exposes three classes:
`soccerball`, `robot`, `goal`.

**Field-side detection:** The field walls are taped RED or BLUE.
`vision/wall_detector.py` identifies which color wall is in frame using HSV
thresholding — no ML model required. This is the primary signal for knowing
which goal the robot is facing.

**Team assignment:** Holding the CAM boot button for 5 seconds toggles the
team (RED/BLUE). Read via `robot.hold_toggle()` — `False` = RED, `True` = BLUE.
The strategy uses this to decide which wall color to drive toward.

---

## System Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    VISION  (python/vision/)                  │
│                                                             │
│  CameraStream ──► BallDetector (.eim) ──► Detection bbox    │
│                └► WallDetector  (HSV) ──► WallReading       │
└───────────────────────────┬─────────────────────────────────┘
                            │ frame, Detection, WallReading
┌───────────────────────────▼─────────────────────────────────┐
│                  STRATEGY  (python/strategy.py)              │
│                                                             │
│  SoccerStrategy.tick()  called every ~50 ms                 │
│                                                             │
│   ┌─────────┐  ball detected  ┌──────────┐                  │
│   │  SEARCH │───────────────► │ APPROACH │                  │
│   └─────────┘                 └────┬─────┘                  │
│        ▲   ◄── ball lost ─────────┘│ ball close (bbox tall) │
│        │                      ┌────▼─────┐                  │
│        │   ◄── ball lost ─────│   PUSH   │                  │
│        │                      └────┬─────┘                  │
│        │   ◄── own wall ahead ─────┘│ own wall = retreat    │
│        │                      ┌────▼─────┐                  │
│        └───────── done ───────│ RETREAT  │                  │
│                               └──────────┘                  │
└───────────────────────────┬─────────────────────────────────┘
                            │ robot.drive() / robot.stop()
┌───────────────────────────▼─────────────────────────────────┐
│                  HARDWARE  (python/robot_client.py)          │
│                  FIRMWARE  (sketch/sketch.ino)               │
└─────────────────────────────────────────────────────────────┘
```

---

## States

**Default / resting state is SEARCH.**
The robot scans by rotating until it sees the ball.

| State | Entry | Behavior | Exits to |
|---|---|---|---|
| `SEARCH` | startup; ball lost for ≥ `LOST_BALL_GRACE_TICKS` | Rotate left at `SEARCH_SPEED` | `APPROACH` when ball detected |
| `APPROACH` | Ball detected | If offset > `CENTER_DEADBAND` → rotate to center; else drive forward | `PUSH` when ball bbox height ≥ `CLOSE_HEIGHT_THRESHOLD`; `SEARCH` if ball lost |
| `PUSH` | Ball close (bbox height) | Check wall color → if opponent wall: push forward; if own wall: retreat | `RETREAT` if own wall; `APPROACH` if ball drifts off-center |
| `RETREAT` | Own-goal risk detected | Drive backward briefly | `SEARCH` |

---

## File Map

```
python/
├── main.py                   # App Lab entry point; builds strategy, runs loop
├── robot_client.py           # MiniAutoRobot Bridge wrapper (do not scatter raw Bridge.call)
├── strategy.py               # SoccerStrategy, State machine, Config
└── vision/
    ├── __init__.py
    ├── camera_stream.py      # CameraStream — timestamped MJPEG capture via OpenCV
    ├── ball_detector.py      # BallDetector — Edge Impulse ImageImpulseRunner wrapper
    └── wall_detector.py      # WallDetector — HSV red/blue wall classification
```

### `vision/camera_stream.py` — `CameraStream`

Opens the ESP32-S3 MJPEG stream via `cv2.VideoCapture`. Warms up for
`warmup_frames` good frames before returning. `read()` returns
`(frame_bgr | None, age_seconds)` — `age_seconds` is a safety signal;
strategy stops the robot if it exceeds `stale_frame_s`.

### `vision/ball_detector.py` — `BallDetector`, `Detection`

Wraps `edge_impulse_linux.image.ImageImpulseRunner`. Normalizes bounding
boxes against the model's actual crop size. `best(frame, label)` returns the
highest-confidence detection for a given label, or `None`.

```python
@dataclass
class Detection:
    label: str
    confidence: float
    x_center: float   # 0–1 fraction of frame width
    y_center: float   # 0–1 fraction of frame height
    width: float
    height: float
```

### `vision/wall_detector.py` — `WallDetector`, `WallReading`

HSV-based — no model required. Classifies the dominant wall color in a frame
as `"RED"`, `"BLUE"`, or `"UNKNOWN"`. Thresholds are tunable at construction
(`min_coverage_pct`).

### `strategy.py` — `SoccerStrategy`

Injected with `MiniAutoRobot`, `CameraStream`, `BallDetector`, `WallDetector`,
and a `Config`. Call `tick()` at ~20 Hz from the main loop. All robot motion
happens here via `robot.drive()` — no direct `Bridge.call` outside
`robot_client.py`.

**Key tunable constants** (top of `strategy.py`):

| Constant | Default | Description |
|---|---|---|
| `CENTER_DEADBAND` | `0.12` | Fraction of frame treated as centered on ball |
| `REALIGN_DEADBAND` | `0.18` | Looser centering tolerance while already pushing |
| `CLOSE_HEIGHT_THRESHOLD` | `0.35` | Ball bbox height fraction = "close enough to push" |
| `PUSH_OBSTACLE_CM` | `8` | Ultrasonic below this while pushing = unexpected obstacle |
| `LOST_BALL_GRACE_TICKS` | `3` | Missed detections before reverting to SEARCH |
| `SEARCH_SPEED/MS` | `150 / 260` | Rotation speed and duration per scan tick |
| `TURN_SPEED/MS` | `150 / 220` | Centering rotation on ball |
| `APPROACH_SPEED/MS` | `170 / 350` | Forward approach |
| `PUSH_SPEED/MS` | `190 / 400` | Forward push when at goal |
| `RETREAT_SPEED/MS` | `160 / 400` | Backward retreat to avoid own goal |

### `main.py` — Entry point

Reads configuration from environment variables. Supports two modes via
`ROBOCUP_MODE`:

| Mode | Behavior |
|---|---|
| `match` (default) | Builds strategy, runs `SoccerStrategy.tick()` loop |
| `demo` | Runs canned motion/sensor smoke test — use to verify wiring |

**Environment variables:**

| Variable | Default | Description |
|---|---|---|
| `ROBOCUP_MODEL_PATH` | *(required)* | Path to the `.eim` model on the robot's filesystem |
| `CAMERA_STREAM_URL` | `http://192.168.5.1:81/stream` | Camera MJPEG URL |
| `ROBOCUP_BALL_LABEL` | `soccerball` | Label string as exported from Edge Impulse |
| `ROBOCUP_BALL_CONFIDENCE` | `0.5` | Minimum confidence threshold |
| `ROBOCUP_WALL_MIN_COVERAGE_PCT` | `2.0` | Minimum % of frame with wall color to classify |
| `ROBOCUP_STALE_FRAME_MS` | `500` | Max frame age in ms before stopping robot |
| `ROBOCUP_MODE` | `match` | `match` or `demo` |

---

## Design Decisions

These were made deliberately — don't reverse them without discussion.

**Wall detection is HSV, not the EI model.**
The field walls have consistent, high-saturation red/blue tape. HSV
classification is fast, reliable, and requires no training data or model
inference overhead. The EI model is reserved for object detection (ball, robot,
goal).

**Default state is SEARCH (rotate and scan), not stop.**
A stationary robot loses without doing anything. Scanning maximizes the chance
of finding the ball quickly from any starting orientation.

**The PUSH state uses wall color to determine safe push direction.**
Rather than dead reckoning (which drifts without encoders), the robot reads the
wall color directly while in possession of the ball. Opponent color ahead = push.
Own color ahead = retreat to avoid own goal.

**Grace ticks before re-entering SEARCH.**
`LOST_BALL_GRACE_TICKS = 3` prevents the robot from aborting an approach on a
single bad frame (occlusion, motion blur). Increase if the robot jitters between
states on a real frame stream.

**Stale frame = stop.**
If the camera stream drops, the robot stops rather than continuing blind.
This is a safety decision — do not remove the stale-frame check.

---

## What Is Not Yet Implemented

These are open areas for teammates to contribute:

| Feature | Where | Notes |
|---|---|---|
| **Opponent robot detection** | `strategy.py` | `BallDetector.best(frame, "robot")` already works; use opponent position to block or evade |
| **Goal object detection** | `strategy.py` | `BallDetector.best(frame, "goal")` already works; could refine push direction vs. wall color alone |
| **Defend mode** | `strategy.py` | When ball is not visible and an opponent is detected, move to block rather than scan |
| **Strafe to center on ball** | `strategy.py` | Mecanum supports sideways motion — strafing while approaching keeps the ball centered without rotation |
| **Battery low warning** | `main.py` | `read_sensors()["battery_mv"]` < threshold → LED + buzz, slow speed |

---

## Validation

Run these before any hardware session:

```bash
python3 -m py_compile python/*.py python/vision/*.py
python3 .agents/skills/uno-q-miniauto/scripts/check_robot_contract.py --strict
```

First hardware session checklist:
1. Wheels off the ground
2. `ROBOCUP_MODE=demo` — verify all motors, servo, LED, buzzer
3. `ROBOCUP_MODE=match` — verify camera stream connects, model loads, labels print
4. Lower the robot; test SEARCH rotation, then APPROACH with a ball placed in view
5. Tune `CLOSE_HEIGHT_THRESHOLD` and `CENTER_DEADBAND` for your field conditions

---

## Safety Rules

- Every entry point that moves hardware must use `try/finally: robot.stop()`.
- Use `App.run(user_loop=lambda: robot.run_program(loop))` — the CAM button is the physical kill switch.
- `robot.drive()` checks `program_enabled` and raises `ProgramStopped` on button press.
- Never bypass `robot.stop()` cleanup in `run_program`.
- Strategy failures (stale frame, no model, exception) must stop the robot, not continue blind.
