# SPDX-License-Identifier: Apache-2.0 OR GPL-2.0-only
# SPDX-FileCopyrightText: 2026 Fernando Rodriguez
"""Payload abstraction — control implements attached to the robot (Phase AB).

Supports: spray nozzle, gripper, PTO (power take-off), lights, custom GPIO.
Each payload is a named device with on/off/set commands sent via CONFIG_CMD.

Usage:
    pm = PayloadManager()
    pm.register("sprayer", PayloadType.SPRAY, gpio_pin=20)
    pm.activate("sprayer")
    pm.set_value("sprayer", 75)  # 75% duty
    pm.deactivate("sprayer")
"""

import enum
import logging
import time
from dataclasses import dataclass, field

logger = logging.getLogger("brain.payload")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MAX_PAYLOADS = 8
CONFIG_KEY_PAYLOAD_BASE = 0x20  # CONFIG_CMD keys 0x20-0x27 for payloads


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------


class PayloadType(enum.Enum):
    SPRAY = "spray"
    GRIPPER = "gripper"
    PTO = "pto"
    LIGHT = "light"
    GPIO = "gpio"
    CUSTOM = "custom"


class PayloadState(enum.Enum):
    IDLE = "idle"
    ACTIVE = "active"
    ERROR = "error"


@dataclass
class Payload:
    name: str
    payload_type: PayloadType
    gpio_pin: int = -1
    config_key: int = 0
    state: PayloadState = PayloadState.IDLE
    value: int = 0  # 0-100 (duty cycle / position / intensity)
    activated_at: float = 0.0
    total_active_s: float = 0.0


# ---------------------------------------------------------------------------
# PayloadManager
# ---------------------------------------------------------------------------


class PayloadManager:
    """Manages robot payloads (implements, tools, accessories)."""

    def __init__(self):
        self._payloads: dict[str, Payload] = {}
        self._next_config_key = CONFIG_KEY_PAYLOAD_BASE

    def register(
        self,
        name: str,
        payload_type: PayloadType,
        gpio_pin: int = -1,
    ) -> Payload:
        """Register a payload device."""
        if len(self._payloads) >= MAX_PAYLOADS:
            raise ValueError(f"Max {MAX_PAYLOADS} payloads")
        p = Payload(
            name=name,
            payload_type=payload_type,
            gpio_pin=gpio_pin,
            config_key=self._next_config_key,
        )
        self._next_config_key += 1
        self._payloads[name] = p
        logger.info("[Payload] Registered '%s' type=%s gpio=%d", name, payload_type.value, gpio_pin)
        return p

    def unregister(self, name: str):
        if name in self._payloads:
            del self._payloads[name]

    def activate(self, name: str) -> bool:
        p = self._payloads.get(name)
        if not p:
            return False
        p.state = PayloadState.ACTIVE
        p.activated_at = time.time()
        p.value = 100
        return True

    def deactivate(self, name: str) -> bool:
        p = self._payloads.get(name)
        if not p:
            return False
        if p.state == PayloadState.ACTIVE and p.activated_at > 0:
            p.total_active_s += time.time() - p.activated_at
        p.state = PayloadState.IDLE
        p.value = 0
        p.activated_at = 0.0
        return True

    def set_value(self, name: str, value: int) -> bool:
        p = self._payloads.get(name)
        if not p:
            return False
        p.value = max(0, min(100, value))
        return True

    def get(self, name: str) -> Payload | None:
        return self._payloads.get(name)

    @property
    def count(self) -> int:
        return len(self._payloads)

    def all_payloads(self) -> list[Payload]:
        return list(self._payloads.values())

    def active_payloads(self) -> list[Payload]:
        return [p for p in self._payloads.values() if p.state == PayloadState.ACTIVE]

    def deactivate_all(self):
        for name in list(self._payloads.keys()):
            self.deactivate(name)

    def status_text(self) -> str:
        if not self._payloads:
            return "No payloads registered."
        lines = []
        for p in self._payloads.values():
            state = p.state.value.upper()
            lines.append(f"  {p.name} ({p.payload_type.value}): {state} " f"value={p.value}%")
        return "\n".join(lines)
