#!/usr/bin/env bash
# Open the robot dashboard over USB, so you can stay on your normal Wi-Fi
# instead of joining the camera's access point.
#
#   scripts/dashboard.sh                    # first connected board, port 7000
#   scripts/dashboard.sh 2236859133         # a specific board by adb serial
#   scripts/dashboard.sh 2236859133 7001    # ...on a different local port
#
# The forward is per-connection: re-run it after unplugging or swapping boards.
# With two boards attached, give the second one a different local port so they
# do not collide.
set -euo pipefail

SERIAL="${1:-}"
PORT="${2:-7000}"

ADB=(adb)
if [ -n "$SERIAL" ]; then
  ADB=(adb -s "$SERIAL")
fi

"${ADB[@]}" forward "tcp:${PORT}" tcp:7000 >/dev/null

if curl -sf -o /dev/null -m 10 "http://localhost:${PORT}/"; then
  echo "dashboard: http://localhost:${PORT}"
else
  echo "port forwarded, but nothing is answering on it." >&2
  echo "the app is probably not running:" >&2
  echo "  ${ADB[*]} shell 'arduino-app-cli app start user:miniautodriver'" >&2
  exit 1
fi
