"""
sensor_monitor.py  -  track sensor health and bridge over brief I2C dropouts.

The ultrasonic (I2C 0x77) and line sensor (0x78) share a bus with the camera
button controller (0x79), which the sketch polls every loop with no retry. In
practice they fail intermittently, together, and a failed read surfaces as
ultrasonic_mm == -1 / line_ok == false.

Treating every single failure as "distance unknown" is correct but slow: it
blocks forward motion on 30-40% of ticks. This monitor instead reuses the last
*good* reading for a short window, so one dropped read no longer stalls the
robot, while a genuinely dead sensor still fails closed.

It is fed from the strategy tick rather than polling on its own thread, because
the Bridge is only exercised from the main loop.

    monitor = SensorMonitor()
    sensors = monitor.update(robot.read_sensors())   # repaired copy
    monitor.health()                                 # for the dashboard
"""
import time

# How long a good ultrasonic reading may be reused after a failed read. At
# ~250 mm/s this is a few centimetres of travel, so keep it short: too long and
# the robot acts on a distance it has already closed.
MAX_DISTANCE_AGE_S = 0.4

# Consecutive failures before we stop trusting the sensor at all.
FAILURE_STREAK_LIMIT = 8

# 3S 18650 pack: ~12.6 V full, ~11.1 V nominal, ~9.9 V empty. Warn early because
# voltage sags hard under full-speed current draw.
BATTERY_WARN_MV = 10800
BATTERY_CRITICAL_MV = 10200

# Rolling window for the health percentages.
WINDOW = 100


class SensorMonitor:
    """Repairs brief sensor dropouts and reports rolling health."""

    def __init__(
        self,
        max_distance_age_s: float = MAX_DISTANCE_AGE_S,
        failure_streak_limit: int = FAILURE_STREAK_LIMIT,
    ) -> None:
        self.max_distance_age_s = max_distance_age_s
        self.failure_streak_limit = failure_streak_limit

        self._last_good_mm: int | None = None
        self._last_good_at = 0.0
        self._distance_streak = 0
        self._line_streak = 0

        self._distance_results: list[bool] = []
        self._line_results: list[bool] = []
        self._battery_mv = 0
        self._battery_min_mv = 0
        self._reads = 0
        self._repairs = 0

    # -- per tick ---------------------------------------------------------

    def update(self, sensors: dict | None) -> dict:
        """Record a sensor snapshot and return a repaired copy.

        The copy carries two extra keys the callers can use without changing the
        firmware payload contract:
          ultrasonic_stale  - True when the distance came from cache
          ultrasonic_trusted - False when the sensor is considered dead
        """
        sensors = dict(sensors or {})
        now = time.monotonic()
        self._reads += 1

        raw_mm = sensors.get("ultrasonic_mm", -1)
        try:
            raw_mm = int(raw_mm)
        except (TypeError, ValueError):
            raw_mm = -1

        good = raw_mm > 0
        self._distance_results.append(good)
        del self._distance_results[:-WINDOW]

        if good:
            self._last_good_mm = raw_mm
            self._last_good_at = now
            self._distance_streak = 0
            sensors["ultrasonic_stale"] = False
            sensors["ultrasonic_trusted"] = True
        else:
            self._distance_streak += 1
            age = now - self._last_good_at
            dead = self._distance_streak >= self.failure_streak_limit
            usable = (
                self._last_good_mm is not None
                and age <= self.max_distance_age_s
                and not dead
            )
            if usable:
                # Bridge the dropout with the last good reading.
                sensors["ultrasonic_mm"] = self._last_good_mm
                sensors["ultrasonic_cm"] = self._last_good_mm // 10
                sensors["ultrasonic_stale"] = True
                sensors["ultrasonic_trusted"] = True
                self._repairs += 1
            else:
                # Fail closed: leave -1 so the policy refuses to drive forward.
                sensors["ultrasonic_stale"] = True
                sensors["ultrasonic_trusted"] = False

        line_ok = bool(sensors.get("line_ok"))
        self._line_results.append(line_ok)
        del self._line_results[:-WINDOW]
        self._line_streak = 0 if line_ok else self._line_streak + 1

        battery = sensors.get("battery_mv") or 0
        if battery:
            self._battery_mv = battery
            self._battery_min_mv = min(self._battery_min_mv or battery, battery)

        return sensors

    # -- reporting --------------------------------------------------------

    @staticmethod
    def _rate(results: list) -> float:
        return round(100.0 * sum(results) / len(results), 1) if results else 0.0

    def warnings(self) -> list:
        out = []
        if self._distance_streak >= self.failure_streak_limit:
            out.append(f"ultrasonic dead ({self._distance_streak} reads)")
        if self._line_streak >= self.failure_streak_limit:
            out.append(f"line sensor dead ({self._line_streak} reads)")
        if self._battery_mv and self._battery_mv <= BATTERY_CRITICAL_MV:
            out.append(f"battery critical {self._battery_mv/1000:.2f} V")
        elif self._battery_mv and self._battery_mv <= BATTERY_WARN_MV:
            out.append(f"battery low {self._battery_mv/1000:.2f} V")
        distance_rate = self._rate(self._distance_results)
        if len(self._distance_results) >= 20 and distance_rate < 60:
            out.append(f"ultrasonic flaky ({distance_rate:.0f}% good)")
        return out

    def health(self) -> dict:
        return {
            "ultrasonic_pct": self._rate(self._distance_results),
            "line_pct": self._rate(self._line_results),
            "distance_streak": self._distance_streak,
            "battery_mv": self._battery_mv,
            "battery_min_mv": self._battery_min_mv,
            "reads": self._reads,
            "repairs": self._repairs,
            "warnings": self.warnings(),
        }
