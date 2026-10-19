# SPDX-License-Identifier: Apache-2.0 OR GPL-2.0-only
# SPDX-FileCopyrightText: 2026 Fernando Rodriguez
"""Mission logging, replay, and analytics — Phase AD.

Records all robot events (sensor data, decisions, actions, alerts) to
a structured log file for post-mission analysis and replay.

Usage:
    ml = MissionLogger("data/logs")
    ml.start_mission("patrol_night")
    ml.log_event("sensor", {"battery_mv": 7200, "position": [100, 200]})
    ml.log_event("decision", {"action": "TURN_LEFT", "reason": "obstacle"})
    ml.end_mission()

    # Replay
    events = ml.load_mission("patrol_night_20260326_120000")
    analytics = ml.analyze(events)
"""

import json
import logging
import os
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone

logger = logging.getLogger("brain.logger")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
LOG_DIR_DEFAULT = "data/logs"
LOG_MAX_EVENTS = 100_000  # max events per mission file
LOG_FLUSH_INTERVAL = 50  # flush to disk every N events
LOG_RETENTION_DAYS = 90  # auto-cleanup old logs


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------


@dataclass
class LogEvent:
    timestamp: float
    event_type: str  # sensor, decision, action, alert, system
    data: dict = field(default_factory=dict)


@dataclass
class MissionSummary:
    mission_id: str
    start_time: float = 0.0
    end_time: float = 0.0
    duration_s: float = 0.0
    event_count: int = 0
    alert_count: int = 0
    distance_mm: int = 0
    laps: int = 0


@dataclass
class MissionAnalytics:
    total_events: int = 0
    events_by_type: dict = field(default_factory=dict)
    alert_count: int = 0
    duration_s: float = 0.0
    avg_battery_mv: float = 0.0
    min_battery_mv: int = 0
    distance_mm: int = 0


# ---------------------------------------------------------------------------
# MissionLogger
# ---------------------------------------------------------------------------


class MissionLogger:
    """Records and replays mission data."""

    def __init__(self, log_dir: str = LOG_DIR_DEFAULT):
        self._log_dir = log_dir
        self._mission_id: str = ""
        self._events: list[LogEvent] = []
        self._active = False
        self._start_time: float = 0.0
        self._file_path: str = ""
        self._flush_counter = 0

    @property
    def active(self) -> bool:
        return self._active

    @property
    def mission_id(self) -> str:
        return self._mission_id

    @property
    def event_count(self) -> int:
        return len(self._events)

    def start_mission(self, name: str = "mission"):
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        self._mission_id = f"{name}_{ts}"
        self._events = []
        self._start_time = time.time()
        self._active = True
        self._flush_counter = 0

        os.makedirs(self._log_dir, exist_ok=True)
        self._file_path = os.path.join(
            self._log_dir,
            f"{self._mission_id}.jsonl",
        )

        self.log_event("system", {"action": "mission_start", "name": name})
        logger.info("[Logger] Mission started: %s", self._mission_id)

    def end_mission(self) -> MissionSummary:
        if not self._active:
            return MissionSummary(mission_id="")

        self.log_event("system", {"action": "mission_end"})
        self._flush_all()
        self._active = False

        duration = time.time() - self._start_time
        alerts = sum(1 for e in self._events if e.event_type == "alert")
        summary = MissionSummary(
            mission_id=self._mission_id,
            start_time=self._start_time,
            end_time=time.time(),
            duration_s=duration,
            event_count=len(self._events),
            alert_count=alerts,
        )
        logger.info(
            "[Logger] Mission ended: %s (%d events, %.0fs)",
            self._mission_id,
            len(self._events),
            duration,
        )
        return summary

    def log_event(self, event_type: str, data: dict | None = None):
        if not self._active:
            return
        if len(self._events) >= LOG_MAX_EVENTS:
            return

        event = LogEvent(
            timestamp=time.time(),
            event_type=event_type,
            data=data or {},
        )
        self._events.append(event)

        self._flush_counter += 1
        if self._flush_counter >= LOG_FLUSH_INTERVAL:
            self._flush_batch()
            self._flush_counter = 0

    def load_mission(self, mission_id: str) -> list[LogEvent]:
        path = os.path.join(self._log_dir, f"{mission_id}.jsonl")
        if not os.path.exists(path):
            return []
        events = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                raw = json.loads(line)
                events.append(
                    LogEvent(
                        timestamp=raw.get("timestamp", 0),
                        event_type=raw.get("event_type", ""),
                        data=raw.get("data", {}),
                    )
                )
        return events

    def list_missions(self) -> list[str]:
        if not os.path.exists(self._log_dir):
            return []
        return sorted(
            [f.replace(".jsonl", "") for f in os.listdir(self._log_dir) if f.endswith(".jsonl")]
        )

    def analyze(self, events: list[LogEvent]) -> MissionAnalytics:
        if not events:
            return MissionAnalytics()

        by_type: dict[str, int] = {}
        battery_values = []
        alerts = 0

        for e in events:
            by_type[e.event_type] = by_type.get(e.event_type, 0) + 1
            if e.event_type == "alert":
                alerts += 1
            if "battery_mv" in e.data:
                battery_values.append(e.data["battery_mv"])

        duration = events[-1].timestamp - events[0].timestamp

        return MissionAnalytics(
            total_events=len(events),
            events_by_type=by_type,
            alert_count=alerts,
            duration_s=max(0, duration),
            avg_battery_mv=(sum(battery_values) / len(battery_values) if battery_values else 0),
            min_battery_mv=min(battery_values) if battery_values else 0,
        )

    def cleanup_old(self, retention_days: int = LOG_RETENTION_DAYS):
        if not os.path.exists(self._log_dir):
            return
        cutoff = time.time() - retention_days * 86400
        for f in os.listdir(self._log_dir):
            path = os.path.join(self._log_dir, f)
            if os.path.getmtime(path) < cutoff:
                os.remove(path)

    # ── Internal ──────────────────────────────────────────────────────────

    def _flush_batch(self):
        if not self._file_path or not self._events:
            return
        start = max(0, len(self._events) - LOG_FLUSH_INTERVAL)
        batch = self._events[start:]
        with open(self._file_path, "a") as f:
            for e in batch:
                f.write(json.dumps(asdict(e)) + "\n")

    def _flush_all(self):
        if not self._file_path:
            return
        with open(self._file_path, "w") as f:
            for e in self._events:
                f.write(json.dumps(asdict(e)) + "\n")
