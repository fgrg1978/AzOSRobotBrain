# SPDX-License-Identifier: Apache-2.0 OR GPL-2.0-only
# SPDX-FileCopyrightText: 2026 Fernando Rodriguez
"""Battery monitor — advanced power management with history and prediction.

Tracks voltage, current, mAh consumed, and estimates remaining autonomy.
Integrates with INA219 sensor data from the kernel (Phase AP).

Usage:
    monitor = BatteryMonitor()
    monitor.update(voltage_mv=7200, current_ma=800, mah_used=1200)
    print(monitor.status_text())  # "72%, ~2.3h remaining"
"""

import logging
import time
from dataclasses import dataclass, field

logger = logging.getLogger("brain.battery")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
BATTERY_NOMINAL_MAH = 3600  # 2S3P pack (6× 18650 1200mAh)
BATTERY_NOMINAL_MV = 7400  # 2S nominal (3.7V × 2)
BATTERY_FULL_MV = 8400  # 2S full (4.2V × 2)
BATTERY_EMPTY_MV = 6000  # 2S empty (3.0V × 2)
BATTERY_CELLS = 2  # series cell count

# Voltage sag
VOLTAGE_SAG_THRESHOLD_MV = 500  # drop > this in 1s = sag
VOLTAGE_SAG_WINDOW_S = 1.0

# Failsafe levels (% remaining)
FAILSAFE_WARNING_PCT = 25
FAILSAFE_RTL_PCT = 15
FAILSAFE_LAND_PCT = 10
FAILSAFE_KILL_PCT = 5

# History
HISTORY_MAX_SAMPLES = 3600  # 1 hour at 1Hz


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------
@dataclass
class BatteryState:
    voltage_mv: int = 0
    current_ma: int = 0
    mah_used: int = 0
    capacity_pct: int = 100
    sag_detected: bool = False
    failsafe_level: int = 0  # 0=OK, 1=WARNING, 2=RTL, 3=LAND, 4=KILL
    autonomy_minutes: float = 0.0
    per_cell_mv: float = 0.0
    timestamp: float = field(default_factory=time.time)


# ---------------------------------------------------------------------------
# BatteryMonitor
# ---------------------------------------------------------------------------
class BatteryMonitor:
    """Tracks battery state with history and predictions."""

    def __init__(self, nominal_mah: int = BATTERY_NOMINAL_MAH):
        self._nominal_mah = nominal_mah
        self._state = BatteryState()
        self._history: list[tuple[float, int, int]] = []  # (time, mv, ma)
        self._sag_prev_mv: int = 0
        self._sag_prev_time: float = 0.0

    @property
    def state(self) -> BatteryState:
        return self._state

    def update(
        self,
        voltage_mv: int = 0,
        current_ma: int = 0,
        mah_used: int = 0,
        capacity_pct: int = -1,
        sag_flag: bool = False,
        failsafe: int = 0,
    ):
        """Update battery state from sensor data."""
        now = time.time()
        self._state.voltage_mv = voltage_mv
        self._state.current_ma = current_ma
        self._state.mah_used = mah_used
        self._state.sag_detected = sag_flag
        self._state.failsafe_level = failsafe
        self._state.timestamp = now
        self._state.per_cell_mv = voltage_mv / max(BATTERY_CELLS, 1)

        # Capacity: use kernel value if provided, else estimate from voltage
        if capacity_pct >= 0:
            self._state.capacity_pct = capacity_pct
        else:
            self._state.capacity_pct = self._estimate_pct_from_voltage(voltage_mv)

        # Autonomy prediction
        if current_ma > 0 and mah_used < self._nominal_mah:
            remaining_mah = self._nominal_mah - mah_used
            self._state.autonomy_minutes = (remaining_mah / current_ma) * 60
        else:
            self._state.autonomy_minutes = 0.0

        # Voltage sag detection (brain-side, complementary to kernel)
        if self._sag_prev_mv > 0:
            dt = now - self._sag_prev_time
            if dt <= VOLTAGE_SAG_WINDOW_S and dt > 0:
                drop = self._sag_prev_mv - voltage_mv
                if drop > VOLTAGE_SAG_THRESHOLD_MV:
                    self._state.sag_detected = True
                    logger.warning(
                        "[Battery] Voltage sag: %dmV → %dmV (-%dmV in %.1fs)",
                        self._sag_prev_mv,
                        voltage_mv,
                        drop,
                        dt,
                    )
        self._sag_prev_mv = voltage_mv
        self._sag_prev_time = now

        # History
        self._history.append((now, voltage_mv, current_ma))
        if len(self._history) > HISTORY_MAX_SAMPLES:
            self._history = self._history[-HISTORY_MAX_SAMPLES:]

    def status_text(self) -> str:
        """Format battery status for Telegram /battery command."""
        s = self._state
        lines = [f"Battery: {s.capacity_pct}%"]
        lines.append(f"  Voltage: {s.voltage_mv}mV " f"({s.per_cell_mv:.0f}mV/cell)")
        if s.current_ma > 0:
            lines.append(f"  Current: {s.current_ma}mA")
        if s.mah_used > 0:
            lines.append(f"  Used: {s.mah_used}/{self._nominal_mah} mAh")
        if s.autonomy_minutes > 0:
            hours = s.autonomy_minutes / 60
            if hours >= 1:
                lines.append(f"  Remaining: ~{hours:.1f}h")
            else:
                lines.append(f"  Remaining: ~{s.autonomy_minutes:.0f}min")
        if s.sag_detected:
            lines.append("  WARNING: Voltage sag detected!")
        fs_labels = {
            1: "WARNING",
            2: "RTL",
            3: "LAND",
            4: "KILL",
        }
        if s.failsafe_level > 0:
            lines.append(f"  FAILSAFE: {fs_labels.get(s.failsafe_level, '?')}")
        return "\n".join(lines)

    def get_history(
        self,
        last_n: int = 60,
    ) -> list[tuple[float, int, int]]:
        """Return recent history: [(timestamp, voltage_mv, current_ma), ...]."""
        return list(self._history[-last_n:])

    @staticmethod
    def _estimate_pct_from_voltage(voltage_mv: int) -> int:
        """Rough capacity estimate from voltage (LiPo 2S curve)."""
        if voltage_mv >= BATTERY_FULL_MV:
            return 100
        if voltage_mv <= BATTERY_EMPTY_MV:
            return 0
        # Linear approximation between empty and full
        v_range = BATTERY_FULL_MV - BATTERY_EMPTY_MV
        return int((voltage_mv - BATTERY_EMPTY_MV) * 100 / v_range)
