# SPDX-License-Identifier: Apache-2.0 OR GPL-2.0-only
# SPDX-FileCopyrightText: 2026 Fernando Rodriguez
"""Robot OS CLI tool (AT4) — command-line interface for robot management.

Usage:
    python -m tools.robot_cli topic list
    python -m tools.robot_cli topic echo /sensors/imu
    python -m tools.robot_cli config get motor_max_speed
    python -m tools.robot_cli config set motor_max_speed 80
    python -m tools.robot_cli status
    python -m tools.robot_cli ping
    python -m tools.robot_cli flash firmware.elf  (placeholder)
    python -m tools.robot_cli monitor             (serial console)
    python -m tools.robot_cli describe             (show robot.yaml)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request

# ── Constants ────────────────────────────────────────────────────────────────

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8080
CLI_TIMEOUT_S = 5
SERIAL_DEFAULT_BAUD = 115200
TOPIC_ECHO_POLL_INTERVAL_S = 1.0
HTTP_SCHEME = "http"
API_CONTENT_TYPE = "application/json"
DESCRIBE_INDENT_SPACES = 2
STATUS_KEY_WIDTH = 20


# ── HTTP helpers (stdlib only) ───────────────────────────────────────────────


def _api_url(host: str, port: int, path: str) -> str:
    """Build a full API URL."""
    return f"{HTTP_SCHEME}://{host}:{port}{path}"


def _api_get(host: str, port: int, path: str) -> dict | str:
    """Perform a GET request and return parsed JSON or raw text."""
    url = _api_url(host, port, path)
    req = urllib.request.Request(url, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=CLI_TIMEOUT_S) as resp:
            body = resp.read().decode()
            try:
                return json.loads(body)
            except json.JSONDecodeError:
                return body
    except urllib.error.HTTPError as e:
        body = e.read().decode()
        try:
            return json.loads(body)
        except json.JSONDecodeError:
            return {"error": body, "status": e.code}
    except urllib.error.URLError as e:
        return {"error": f"connection failed: {e.reason}"}
    except Exception as e:
        return {"error": str(e)}


def _api_post(host: str, port: int, path: str, data: dict | None = None) -> dict | str:
    """Perform a POST request with JSON body."""
    url = _api_url(host, port, path)
    payload = json.dumps(data or {}).encode()
    req = urllib.request.Request(
        url,
        data=payload,
        method="POST",
        headers={"Content-Type": API_CONTENT_TYPE},
    )
    try:
        with urllib.request.urlopen(req, timeout=CLI_TIMEOUT_S) as resp:
            body = resp.read().decode()
            try:
                return json.loads(body)
            except json.JSONDecodeError:
                return body
    except urllib.error.HTTPError as e:
        body = e.read().decode()
        try:
            return json.loads(body)
        except json.JSONDecodeError:
            return {"error": body, "status": e.code}
    except urllib.error.URLError as e:
        return {"error": f"connection failed: {e.reason}"}
    except Exception as e:
        return {"error": str(e)}


def _print_json(data: dict | str | list) -> None:
    """Pretty-print a JSON-like response."""
    if isinstance(data, str):
        print(data)
    else:
        print(json.dumps(data, indent=DESCRIBE_INDENT_SPACES))


# ── Subcommands ──────────────────────────────────────────────────────────────


def cmd_ping(args: argparse.Namespace) -> int:
    """Check connectivity to the brain server."""
    result = _api_get(args.host, args.port, "/health")
    if isinstance(result, dict) and result.get("status") == "ok":
        uptime = result.get("uptime_s", "?")
        print(f"PONG — brain server is up (uptime: {uptime}s)")
        return 0
    print(f"FAIL — {result}")
    return 1


def cmd_status(args: argparse.Namespace) -> int:
    """Show full robot status."""
    result = _api_get(args.host, args.port, "/status")
    if isinstance(result, dict) and "error" in result:
        print(f"Error: {result['error']}")
        return 1
    if isinstance(result, dict):
        for key, value in result.items():
            print(f"  {key:<{STATUS_KEY_WIDTH}} {value}")
    else:
        print(result)
    return 0


def cmd_topic_list(args: argparse.Namespace) -> int:
    """List available topics."""
    result = _api_get(args.host, args.port, "/api/topics")
    if isinstance(result, dict) and "error" in result:
        # API may not have /api/topics — fall back to a helpful message
        print("Topics endpoint not available. Available status at /status.")
        return 1
    _print_json(result)
    return 0


def cmd_topic_echo(args: argparse.Namespace) -> int:
    """Poll and display a topic's latest message."""
    topic = args.name
    path = f"/api/topics/{topic.lstrip('/')}"
    print(f"Echoing {topic} (Ctrl+C to stop)...")
    try:
        while True:
            result = _api_get(args.host, args.port, path)
            _print_json(result)
            time.sleep(TOPIC_ECHO_POLL_INTERVAL_S)
    except KeyboardInterrupt:
        print("\nStopped.")
    return 0


def cmd_config_get(args: argparse.Namespace) -> int:
    """Get a configuration value."""
    result = _api_get(args.host, args.port, f"/api/config/{args.key}")
    _print_json(result)
    return 0


def cmd_config_set(args: argparse.Namespace) -> int:
    """Set a configuration value."""
    result = _api_post(args.host, args.port, f"/api/config/{args.key}", {"value": args.value})
    _print_json(result)
    return 0


def cmd_describe(args: argparse.Namespace) -> int:
    """Load and display the robot description YAML."""
    from planner.robot_description import RobotDescription, DEFAULT_ROBOT_YAML

    path = args.file if args.file else DEFAULT_ROBOT_YAML
    if not os.path.isfile(path):
        print(f"Robot description file not found: {path}")
        return 1

    desc = RobotDescription.from_yaml(path)
    print(f"Robot: {desc.name}")
    print(f"Type:  {desc.type}")
    print(
        f"Chassis: wheel_base={desc.chassis.wheel_base_mm}mm, "
        f"wheel_diam={desc.chassis.wheel_diameter_mm}mm, "
        f"max_speed={desc.chassis.max_speed_pct}%"
    )
    print(f"Sensors ({len(desc.sensors)}):")
    for s in desc.sensors:
        parts = [f"  - {s.type}"]
        if s.model:
            parts.append(f"({s.model})")
        if s.position:
            parts.append(f"@{s.position}")
        if s.bus:
            parts.append(f"[{s.bus}]")
        print(" ".join(parts))
    print(f"Actuators ({len(desc.actuators)}):")
    for a in desc.actuators:
        parts = [f"  - {a.type}"]
        if a.model:
            parts.append(f"({a.model})")
        if a.side:
            parts.append(f"@{a.side}")
        print(" ".join(parts))
    if desc.payloads:
        print(f"Payloads ({len(desc.payloads)}):")
        for p in desc.payloads:
            print(f"  - {p.type} ({p.watts}W)" if p.watts else f"  - {p.type}")
    return 0


def cmd_monitor(args: argparse.Namespace) -> int:
    """Open serial port and display data (requires pyserial)."""
    port_path = args.port_path
    baud = args.baud
    try:
        import serial  # type: ignore[import-untyped]

        ser = serial.Serial(port_path, baud, timeout=1)
        print(f"Monitoring {port_path} at {baud} baud (Ctrl+C to stop)...")
        try:
            while True:
                line = ser.readline()
                if line:
                    print(line.decode(errors="replace"), end="")
        except KeyboardInterrupt:
            print("\nStopped.")
        finally:
            ser.close()
    except ImportError:
        print("Serial monitoring requires pyserial (pip install pyserial).")
        print(f"Would monitor {port_path} at {baud} baud.")
        return 1
    return 0


def cmd_flash(args: argparse.Namespace) -> int:
    """Placeholder for firmware upload."""
    print(f"Flash not yet implemented. Target file: {args.firmware}")
    print("Use external tools (openocd, JLink, etc.) for now.")
    return 1


# ── Argument parser ──────────────────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser with all subcommands."""
    parser = argparse.ArgumentParser(
        prog="robot-cli",
        description="Robot OS CLI — manage and inspect the robot brain.",
    )
    parser.add_argument(
        "--host", default=DEFAULT_HOST, help=f"Brain server host (default: {DEFAULT_HOST})"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_PORT,
        help=f"Brain server port (default: {DEFAULT_PORT})",
    )

    sub = parser.add_subparsers(dest="command", help="Available commands")

    # ── ping ──────────────────────────────────────────────────────────────
    sub.add_parser("ping", help="Check brain server connectivity")

    # ── status ────────────────────────────────────────────────────────────
    sub.add_parser("status", help="Show full robot status")

    # ── topic ─────────────────────────────────────────────────────────────
    topic_parser = sub.add_parser("topic", help="Topic inspection")
    topic_sub = topic_parser.add_subparsers(dest="topic_cmd")
    topic_sub.add_parser("list", help="List available topics")
    echo_parser = topic_sub.add_parser("echo", help="Echo a topic")
    echo_parser.add_argument("name", help="Topic name (e.g., /sensors/imu)")

    # ── config ────────────────────────────────────────────────────────────
    config_parser = sub.add_parser("config", help="Configuration management")
    config_sub = config_parser.add_subparsers(dest="config_cmd")
    get_parser = config_sub.add_parser("get", help="Get a config value")
    get_parser.add_argument("key", help="Config key name")
    set_parser = config_sub.add_parser("set", help="Set a config value")
    set_parser.add_argument("key", help="Config key name")
    set_parser.add_argument("value", help="New value")

    # ── describe ──────────────────────────────────────────────────────────
    desc_parser = sub.add_parser("describe", help="Show robot description from YAML")
    desc_parser.add_argument("--file", default="", help="Path to robot.yaml (default: robot.yaml)")

    # ── monitor ───────────────────────────────────────────────────────────
    mon_parser = sub.add_parser("monitor", help="Serial console monitor")
    mon_parser.add_argument("port_path", nargs="?", default="/dev/ttyUSB0", help="Serial port path")
    mon_parser.add_argument(
        "--baud",
        type=int,
        default=SERIAL_DEFAULT_BAUD,
        help=f"Baud rate (default: {SERIAL_DEFAULT_BAUD})",
    )

    # ── flash ─────────────────────────────────────────────────────────────
    flash_parser = sub.add_parser("flash", help="Flash firmware (placeholder)")
    flash_parser.add_argument("firmware", help="Firmware ELF file path")

    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point for the CLI tool."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if not args.command:
        parser.print_help()
        return 0

    dispatch = {
        "ping": cmd_ping,
        "status": cmd_status,
        "describe": cmd_describe,
        "monitor": cmd_monitor,
        "flash": cmd_flash,
    }

    if args.command == "topic":
        if args.topic_cmd == "list":
            return cmd_topic_list(args)
        elif args.topic_cmd == "echo":
            return cmd_topic_echo(args)
        else:
            print("Usage: robot-cli topic {list|echo}")
            return 1

    if args.command == "config":
        if args.config_cmd == "get":
            return cmd_config_get(args)
        elif args.config_cmd == "set":
            return cmd_config_set(args)
        else:
            print("Usage: robot-cli config {get|set}")
            return 1

    handler = dispatch.get(args.command)
    if handler:
        return handler(args)

    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
