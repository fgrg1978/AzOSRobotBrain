# SPDX-License-Identifier: Apache-2.0 OR GPL-2.0-only
# SPDX-FileCopyrightText: 2026 Fernando Rodriguez
"""Drift-detection smoke for the kernel↔brain const bridge.

`protocol_consts.py` is generated from the kernel-side `.config` by
`AzOS/tools/export_kernel_consts.py`.  This test loads it and
asserts invariants the brain side relies on.  If the kernel changes a
cap or flag without the brain being told, this test fails and surfaces
the drift at PR time (instead of at deployment).

If you see a failure here:
  1. Did the kernel `.config` change?  Re-run
       `python3 AzOS/tools/export_kernel_consts.py`
     to regenerate `AzOSRobotBrain/protocol_consts.py` and re-run tests.
  2. Did a brain assumption change?  Update the asserted invariant
     here and the relevant brain code together.
"""

from __future__ import annotations

import importlib
import pathlib

import pytest


# `protocol_consts.py` lives at the brain repo root, alongside this
# tests/ directory.  Importable as `protocol_consts` if pytest's
# rootdir is the brain repo, which is the default in CI.
@pytest.fixture(scope="module")
def kc():
    """Import the kernel-const bridge module, or skip if absent."""
    repo = pathlib.Path(__file__).resolve().parent.parent
    bridge = repo / "protocol_consts.py"
    if not bridge.exists():
        pytest.skip(
            "protocol_consts.py not present — run "
            "`python3 AzOS/tools/export_kernel_consts.py` to generate."
        )
    return importlib.import_module("protocol_consts")


# ── Sanity: required symbols are present and well-typed ──────────────────


def test_arch_flag_exactly_one_true(kc):
    """Exactly one ARCH_* must be true."""
    flags = [kc.KERNEL_ARCH_RISCV64, kc.KERNEL_ARCH_AARCH64, kc.KERNEL_ARCH_X86_64]
    assert sum(flags) == 1, f"Expected exactly one ARCH_* = True, got {flags}"


def test_profile_exactly_one_true(kc):
    """Exactly one PROFILE_* must be true."""
    flags = [
        kc.KERNEL_PROFILE_EMBEDDED,
        kc.KERNEL_PROFILE_EDGE,
        kc.KERNEL_PROFILE_FLEET,
    ]
    assert sum(flags) == 1, f"Expected exactly one PROFILE_* = True, got {flags}"


def test_board_at_most_one_specific(kc):
    """At most one specific BOARD_* may be true (Generic fills the gap)."""
    specific = [
        kc.KERNEL_BOARD_QEMU,
        kc.KERNEL_BOARD_VF2,
        kc.KERNEL_BOARD_K1,
        kc.KERNEL_BOARD_ESP32C3,
    ]
    assert sum(specific) <= 1, f"Multiple specific BOARD_* are true: {specific}"


# ── Brain capacity assumptions stay within kernel caps ────────────────────


def test_tcp_max_conns_is_set_and_positive(kc):
    """Brain's connection pool sizing depends on this."""
    assert kc.KERNEL_TCP_MAX_CONNS is not None
    assert kc.KERNEL_TCP_MAX_CONNS > 0
    assert (
        kc.KERNEL_TCP_MAX_CONNS <= kc.KERNEL_MAX_SOCKETS
    ), "TCP_MAX_CONNS must fit inside MAX_SOCKETS (kernel invariant)"


def test_max_sockets_above_tcp_with_udp_reserve(kc):
    """Kernel validator enforces MAX_SOCKETS ≥ TCP_MAX_CONNS + 4.

    Brain mirrors that assumption so its UDP-aware code paths have at
    least the same headroom the kernel reserved.
    """
    assert kc.KERNEL_MAX_SOCKETS >= kc.KERNEL_TCP_MAX_CONNS + 4


def test_eth_mtu_and_tcp_mss_consistent(kc):
    """MSS = MTU - 40 (IP + TCP headers, no options)."""
    assert (
        kc.KERNEL_TCP_MSS == kc.KERNEL_ETH_MTU - 40
    ), f"TCP_MSS ({kc.KERNEL_TCP_MSS}) should be ETH_MTU ({kc.KERNEL_ETH_MTU}) - 40"


def test_brain_server_port_in_valid_range(kc):
    """Default port the kernel will dial — must be a valid TCP port."""
    port = kc.KERNEL_BRAIN_SERVER_PORT_DEFAULT
    assert 1 <= port <= 65535, f"BRAIN_SERVER_PORT_DEFAULT out of range: {port}"


def test_ota_max_image_size_positive(kc):
    """Brain's fleet-OTA endpoint rejects images above this size."""
    assert kc.KERNEL_OTA_MAX_IMAGE_SIZE_MB is not None
    assert kc.KERNEL_OTA_MAX_IMAGE_SIZE_MB > 0


def test_max_tasks_sane(kc):
    """Brain's stub-fleet soak test caps its synthetic robot count by this."""
    assert kc.KERNEL_MAX_TASKS is not None
    # Sanity ceiling — if MAX_TASKS exceeds 32K we likely have a config bug.
    assert 16 <= kc.KERNEL_MAX_TASKS <= 32 * 1024


# ── Profile-vs-cap cross-checks ──────────────────────────────────────────


def test_embedded_profile_has_small_caps(kc):
    """If PROFILE_EMBEDDED, brain knows to use the small-fleet code paths."""
    if kc.KERNEL_PROFILE_EMBEDDED:
        # Embedded fleet handler caps its connection list.
        assert kc.KERNEL_TCP_MAX_CONNS <= 16


def test_fleet_profile_has_large_caps(kc):
    """If PROFILE_FLEET, brain knows the kernel has gateway-scale tables."""
    if kc.KERNEL_PROFILE_FLEET:
        assert kc.KERNEL_TCP_MAX_CONNS >= 128


# ── Helpers exposed by the bridge module ─────────────────────────────────


def test_selected_profile_consistent_with_flag(kc):
    """Helper string matches the boolean flag."""
    name = kc.selected_profile()
    if name == "fleet":
        assert kc.KERNEL_PROFILE_FLEET
    elif name == "embedded":
        assert kc.KERNEL_PROFILE_EMBEDDED
    else:
        assert kc.KERNEL_PROFILE_EDGE


def test_selected_board_consistent_with_flag(kc):
    name = kc.selected_board()
    mapping = {
        "qemu": kc.KERNEL_BOARD_QEMU,
        "vf2": kc.KERNEL_BOARD_VF2,
        "k1": kc.KERNEL_BOARD_K1,
        "esp32c3": kc.KERNEL_BOARD_ESP32C3,
        "generic": not (
            kc.KERNEL_BOARD_QEMU
            or kc.KERNEL_BOARD_VF2
            or kc.KERNEL_BOARD_K1
            or kc.KERNEL_BOARD_ESP32C3
        ),
    }
    assert mapping.get(name) is True, f"selected_board() = {name!r} but flag is False"
