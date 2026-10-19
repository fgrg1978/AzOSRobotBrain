# SPDX-License-Identifier: Apache-2.0 OR GPL-2.0-only
# SPDX-FileCopyrightText: 2026 Fernando Rodriguez
"""Tests for planner/payload.py — payload abstraction."""

import pytest
from planner.payload import (
    PayloadManager,
    Payload,
    PayloadType,
    PayloadState,
    MAX_PAYLOADS,
    CONFIG_KEY_PAYLOAD_BASE,
)


class TestConstants:
    def test_max_payloads(self):
        assert MAX_PAYLOADS > 0

    def test_config_key_base(self):
        assert CONFIG_KEY_PAYLOAD_BASE > 0


class TestPayloadManager:
    def test_empty(self):
        pm = PayloadManager()
        assert pm.count == 0

    def test_register(self):
        pm = PayloadManager()
        p = pm.register("sprayer", PayloadType.SPRAY, gpio_pin=20)
        assert p.name == "sprayer"
        assert p.payload_type == PayloadType.SPRAY
        assert pm.count == 1

    def test_register_multiple(self):
        pm = PayloadManager()
        pm.register("a", PayloadType.SPRAY)
        pm.register("b", PayloadType.GRIPPER)
        assert pm.count == 2

    def test_register_max_limit(self):
        pm = PayloadManager()
        for i in range(MAX_PAYLOADS):
            pm.register(f"p{i}", PayloadType.GPIO)
        with pytest.raises(ValueError):
            pm.register("overflow", PayloadType.GPIO)

    def test_activate(self):
        pm = PayloadManager()
        pm.register("s", PayloadType.SPRAY)
        assert pm.activate("s")
        assert pm.get("s").state == PayloadState.ACTIVE
        assert pm.get("s").value == 100

    def test_deactivate(self):
        pm = PayloadManager()
        pm.register("s", PayloadType.SPRAY)
        pm.activate("s")
        pm.deactivate("s")
        assert pm.get("s").state == PayloadState.IDLE
        assert pm.get("s").value == 0

    def test_set_value(self):
        pm = PayloadManager()
        pm.register("g", PayloadType.GRIPPER)
        pm.set_value("g", 50)
        assert pm.get("g").value == 50

    def test_set_value_clamped(self):
        pm = PayloadManager()
        pm.register("g", PayloadType.GRIPPER)
        pm.set_value("g", 200)
        assert pm.get("g").value == 100
        pm.set_value("g", -10)
        assert pm.get("g").value == 0

    def test_get_not_found(self):
        pm = PayloadManager()
        assert pm.get("nope") is None

    def test_activate_not_found(self):
        pm = PayloadManager()
        assert not pm.activate("nope")

    def test_active_payloads(self):
        pm = PayloadManager()
        pm.register("a", PayloadType.SPRAY)
        pm.register("b", PayloadType.GRIPPER)
        pm.activate("a")
        assert len(pm.active_payloads()) == 1

    def test_deactivate_all(self):
        pm = PayloadManager()
        pm.register("a", PayloadType.SPRAY)
        pm.register("b", PayloadType.GRIPPER)
        pm.activate("a")
        pm.activate("b")
        pm.deactivate_all()
        assert len(pm.active_payloads()) == 0

    def test_unregister(self):
        pm = PayloadManager()
        pm.register("x", PayloadType.GPIO)
        pm.unregister("x")
        assert pm.count == 0

    def test_status_text(self):
        pm = PayloadManager()
        pm.register("sprayer", PayloadType.SPRAY)
        pm.activate("sprayer")
        text = pm.status_text()
        assert "sprayer" in text
        assert "ACTIVE" in text

    def test_status_text_empty(self):
        pm = PayloadManager()
        assert "No payloads" in pm.status_text()

    def test_config_keys_unique(self):
        pm = PayloadManager()
        p1 = pm.register("a", PayloadType.SPRAY)
        p2 = pm.register("b", PayloadType.GRIPPER)
        assert p1.config_key != p2.config_key
