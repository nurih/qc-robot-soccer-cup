# Hackathon Task Plan (5 people, 1 subsystem each)

Dev window per the event schedule: **11:00 AM - 4:00 PM** (5 hours), ending at Tournament Kickoff.

## Roles

| # | Owner | Subsystem | Repo scope |
|---|---|---|---|
| 1 | Firmware/Safety Lead | UNO Q sketch + hardware bring-up | `sketch/sketch.ino`, motor mixing, BOOT-button/team-toggle behavior, safety checklist (Stages 0-2) |
| 2 | Sensing Lead | Sensors + bounded policy | Line sensor, ultrasonic, battery divider, guarded `decide()`-style policy, serial diagnostics (Stage 3) |
| 3 | Vision/Camera Lead | ESP32-S3 camera + wall detection | `camera/HiwonderCamStream.ino` flashing, MJPEG stream, HSV wall/field-side detector (Stage 4 + no-ML challenge) |
| 4 | AI/ML Lead | Edge Impulse model | Clone prelabelled project, configure FOMO impulse, train, export `.eim`, own model iteration |
| 5 | Integration/Strategy Lead | Application loop + match logic | `python/main.py` app loop (Acquire -> Act), combines all 4 others' outputs, game-rule/redemption-cup strategy, owns final field testing |

Person 5 is the glue role: floats and unblocks others early, then owns integration once inputs exist.

## Edge Impulse status

Not set up yet as of this writing — only screenshot assets exist under `docs/assets/edge-impulse`; no project, trained model, or `.eim` file. This is a cloud/browser step (Edge Impulse Studio), not something installed locally, and can start as soon as the 10 AM workshop ends (doesn't need camera hardware).

Remaining steps:
1. Create an Edge Impulse account; clone the [prelabelled public project](https://studio.edgeimpulse.com/public/1085406/live) or start fresh and upload captured images.
2. Label bounding boxes: `soccerball`, `robot` (`goal` left unlabeled per the guide).
3. Create the impulse: 96x96 RGB image, squash resize, Object Detection (Images) learning block, 3 output classes (goal, robot, soccer_ball).
4. Generate image features, review class behavior.
5. Configure neural network: 150-180 training cycles, learning rate 0.001, CPU training processor. Train, check validation metrics.
6. Export as `.eim` for Linux aarch64 (target = Arduino UNO Q).
7. Wire it into `python/main.py`: load the `.eim`, run inference in the app loop, feed detections into the AI policy. None of this exists in the repo yet — the README explicitly notes no ML/inference files are included.

## Schedule (11 AM - 4 PM)

| Time | Firmware/Safety Lead | Sensing Lead | Vision/Camera Lead | AI/ML Lead | Integration/Strategy Lead |
|---|---|---|---|---|---|
| 11:00-11:20 | All-hands safety check: wiring, battery, wheels raised, confirm `health` payload | (joins all-hands) | (joins all-hands) | (joins all-hands) | Leads safety check; drafts `main.py` skeleton stub |
| 11:20-12:00 | Build/upload `sketch/sketch.ino`; verify Bridge health, serial | Validate sensor JSON (`line_ok`, `ultrasonic_mm`, `battery_mv`); log real values | Flash ESP32-S3 camera firmware, assign SSID/pass, verify MJPEG stream loads | Continue Edge Impulse: label check, create impulse, start first training run | Reviews interfaces (Bridge JSON shape, MJPEG URL) so integration won't guess later |
| 12:00-1:00 | Test named motion commands (forward/strafe/rotate), tune speed/duration | Write bounded `decide()` policy (distance/line thresholds, safe stop) | Get stream stable near AP, start HSV wall-tape tuning (red/blue ranges) | First model trained; review validation metrics; retrain if weak class | Start wiring app loop shell: Acquire -> Preprocess -> Infer stub (mocked) |
| 1:00-1:20 | Sync #1 — demo motion | Sync #1 — demo sensor decide() | Sync #1 — demo stream + wall color print | Sync #1 — demo trained model metrics | Sync #1 — confirm interfaces match what's demoed |
| 1:20-2:00 | Support integration: expose any needed serial tweaks | Support integration: hand off `decide()` for use in real loop | Export `CAMERA_URL`, hand off wall-detector module | Export `.eim` for Linux aarch64, hand to integration | Wire real camera capture + real sensor reads into loop |
| 2:00-2:45 | Available for hardware fixes | Available for sensor edge cases | Tune HSV thresholds against real field tape | Load `.eim`, run live inference, validate detections aren't stale/ambiguous | Combine model detections + wall side + sensors into one `decide()`/`act()` policy |
| 2:45-3:15 | Sync #2 — full dry run on field | Sync #2 | Sync #2 | Sync #2 | Sync #2 — drive the dry run |
| 3:15-3:50 | Fix any motor/servo issues found live | Fix sensor noise/false positives | Fix stream lag, retune HSV under real lighting | Retrain/adjust confidence thresholds if false detections | Tune full policy: speeds, stop conditions, team-toggle handling |
| 3:50-4:00 | Freeze code, final safety check, charge battery | — | — | — | Final freeze, confirm `robot.stop()` fail-safe in every path |
| 4:00 PM | Tournament Kickoff | | | | |
