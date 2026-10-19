# SPDX-License-Identifier: Apache-2.0 OR GPL-2.0-only
# SPDX-FileCopyrightText: 2026 Fernando Rodriguez
"""Tests for guard-specific operating modes (F6)."""

from planner.modes import ModeManager, ModeConfig, load_modes

# ---------------------------------------------------------------------------
# Guard mode config fixtures
# ---------------------------------------------------------------------------

GUARD_CONFIG = {
    "modes": {
        "guardia_ausencia": {
            "skills": ["PATROL_PERIMETER"],
            "loop": True,
            "detect": ["person", "animal", "vehicle", "fire", "smoke", "open_door"],
            "on_detect": ["buzzer_alert", "notify_telegram_photo", "record_evidence", "deterrent"],
            "sensors": ["pir", "sound", "ir"],
            "track_intruder": True,
        },
        "guardia_presencia": {
            "skills": ["PATROL_PERIMETER"],
            "loop": True,
            "perimeter_only": True,
            "detect": ["person", "vehicle"],
            "on_detect": ["notify_telegram_photo", "record_evidence"],
            "sensors": ["pir"],
        },
        "guardia_nocturno": {
            "skills": ["PATROL_PERIMETER"],
            "loop": True,
            "detect": ["person", "animal", "vehicle", "fire"],
            "on_detect": ["buzzer_alert", "notify_telegram_photo", "record_evidence", "deterrent"],
            "sensors": ["pir", "sound", "ir"],
            "led_on_detect": True,
            "patrol_speed_pct": 50,
            "track_intruder": True,
        },
        "panico": {
            "skills": ["SCAN_360"],
            "loop": True,
            "detect": ["person", "animal", "vehicle"],
            "on_detect": ["buzzer_siren", "notify_telegram_photo", "record_evidence"],
            "sensors": ["pir", "sound", "ir"],
            "continuous_buzzer": True,
            "continuous_led": True,
        },
    },
    "tasks": {"default": "guardia_ausencia"},
}


# ---------------------------------------------------------------------------
# ModeConfig fields
# ---------------------------------------------------------------------------


class TestModeConfigFields:
    def test_sensors_field(self):
        mc = ModeConfig(name="test", skills=[], sensors=["pir", "sound"])
        assert mc.sensors == ["pir", "sound"]

    def test_perimeter_only_default_false(self):
        mc = ModeConfig(name="test", skills=[])
        assert mc.perimeter_only is False

    def test_patrol_speed_default_zero(self):
        mc = ModeConfig(name="test", skills=[])
        assert mc.patrol_speed_pct == 0

    def test_continuous_buzzer_default_false(self):
        mc = ModeConfig(name="test", skills=[])
        assert mc.continuous_buzzer is False

    def test_continuous_led_default_false(self):
        mc = ModeConfig(name="test", skills=[])
        assert mc.continuous_led is False

    def test_led_on_detect_default_false(self):
        mc = ModeConfig(name="test", skills=[])
        assert mc.led_on_detect is False

    def test_track_intruder_default_false(self):
        mc = ModeConfig(name="test", skills=[])
        assert mc.track_intruder is False


# ---------------------------------------------------------------------------
# load_modes — guard modes
# ---------------------------------------------------------------------------


class TestLoadGuardModes:
    def test_all_modes_loaded(self):
        modes = load_modes(GUARD_CONFIG)
        assert "guardia_ausencia" in modes
        assert "guardia_presencia" in modes
        assert "guardia_nocturno" in modes
        assert "panico" in modes

    def test_ausencia_sensors(self):
        modes = load_modes(GUARD_CONFIG)
        m = modes["guardia_ausencia"]
        assert m.sensors == ["pir", "sound", "ir"]

    def test_ausencia_track_intruder(self):
        modes = load_modes(GUARD_CONFIG)
        m = modes["guardia_ausencia"]
        assert m.track_intruder is True

    def test_ausencia_detects_all(self):
        modes = load_modes(GUARD_CONFIG)
        m = modes["guardia_ausencia"]
        assert "person" in m.detect
        assert "fire" in m.detect
        assert "smoke" in m.detect
        assert "open_door" in m.detect

    def test_ausencia_deterrent_in_actions(self):
        modes = load_modes(GUARD_CONFIG)
        m = modes["guardia_ausencia"]
        assert "deterrent" in m.on_detect

    def test_presencia_perimeter_only(self):
        modes = load_modes(GUARD_CONFIG)
        m = modes["guardia_presencia"]
        assert m.perimeter_only is True

    def test_presencia_fewer_detects(self):
        modes = load_modes(GUARD_CONFIG)
        m = modes["guardia_presencia"]
        assert len(m.detect) == 2
        assert "person" in m.detect
        assert "vehicle" in m.detect

    def test_presencia_pir_only(self):
        modes = load_modes(GUARD_CONFIG)
        m = modes["guardia_presencia"]
        assert m.sensors == ["pir"]

    def test_presencia_no_deterrent(self):
        modes = load_modes(GUARD_CONFIG)
        m = modes["guardia_presencia"]
        assert "deterrent" not in m.on_detect

    def test_nocturno_led_on_detect(self):
        modes = load_modes(GUARD_CONFIG)
        m = modes["guardia_nocturno"]
        assert m.led_on_detect is True

    def test_nocturno_patrol_speed(self):
        modes = load_modes(GUARD_CONFIG)
        m = modes["guardia_nocturno"]
        assert m.patrol_speed_pct == 50

    def test_nocturno_has_deterrent(self):
        modes = load_modes(GUARD_CONFIG)
        m = modes["guardia_nocturno"]
        assert "deterrent" in m.on_detect

    def test_panico_continuous_buzzer(self):
        modes = load_modes(GUARD_CONFIG)
        m = modes["panico"]
        assert m.continuous_buzzer is True

    def test_panico_continuous_led(self):
        modes = load_modes(GUARD_CONFIG)
        m = modes["panico"]
        assert m.continuous_led is True

    def test_panico_scan_360_skill(self):
        modes = load_modes(GUARD_CONFIG)
        m = modes["panico"]
        assert m.skills == ["SCAN_360"]


# ---------------------------------------------------------------------------
# ModeManager — guard mode switching
# ---------------------------------------------------------------------------


class TestModeManagerGuard:
    def test_default_mode(self):
        mm = ModeManager(GUARD_CONFIG)
        assert mm.current_name == "guardia_ausencia"

    def test_switch_to_presencia(self):
        mm = ModeManager(GUARD_CONFIG)
        assert mm.set_mode("guardia_presencia")
        assert mm.current_name == "guardia_presencia"
        assert mm.current.perimeter_only is True

    def test_switch_to_nocturno(self):
        mm = ModeManager(GUARD_CONFIG)
        assert mm.set_mode("guardia_nocturno")
        assert mm.current.led_on_detect is True

    def test_switch_to_panico(self):
        mm = ModeManager(GUARD_CONFIG)
        assert mm.set_mode("panico")
        assert mm.current.continuous_buzzer is True

    def test_switch_back_to_ausencia(self):
        mm = ModeManager(GUARD_CONFIG)
        mm.set_mode("panico")
        mm.set_mode("guardia_ausencia")
        assert mm.current.track_intruder is True

    def test_unknown_mode_returns_false(self):
        mm = ModeManager(GUARD_CONFIG)
        assert not mm.set_mode("nonexistent")
        assert mm.current_name == "guardia_ausencia"

    def test_should_detect_per_mode(self):
        mm = ModeManager(GUARD_CONFIG)
        assert mm.should_detect("person")
        assert mm.should_detect("fire")
        mm.set_mode("guardia_presencia")
        assert mm.should_detect("person")
        assert not mm.should_detect("fire")

    def test_on_detect_actions_per_mode(self):
        mm = ModeManager(GUARD_CONFIG)
        actions = mm.on_detect_actions()
        assert "deterrent" in actions
        mm.set_mode("guardia_presencia")
        actions = mm.on_detect_actions()
        assert "deterrent" not in actions
