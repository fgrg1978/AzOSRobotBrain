# SPDX-License-Identifier: Apache-2.0 OR GPL-2.0-only
# SPDX-FileCopyrightText: 2026 Fernando Rodriguez
"""Skill catalog — defines all available skills per robot type.

Each skill entry:
  description: human-readable
  args:        dict of arg_name -> (type_str, description)
  example:     example LLM output string

Used by task_planner.py to build the system prompt and validate outputs.
"""

# ── Universal skills (all robot types) ───────────────────────────────────────

UNIVERSAL_SKILLS: dict[str, dict] = {
    "STOP": {
        "description": "Stop all motion immediately.",
        "args": {},
        "example": "STOP",
    },
    "WAIT": {
        "description": "Pause for N seconds.",
        "args": {"seconds": ("float", "duration 0.1-60")},
        "example": "WAIT 3",
    },
    "ALERT": {
        "description": "Raise an alert flag and notify operator.",
        "args": {"message": ("str", "short description of the alert")},
        "example": "ALERT intruder detected",
    },
    "EMERGENCY": {
        "description": "Emergency stop — disables all actuators.",
        "args": {},
        "example": "EMERGENCY",
    },
    "SCAN_360": {
        "description": "Rotate 360° to survey surroundings.",
        "args": {"speed": ("int", "rotation speed 1-50 (default 20)")},
        "example": "SCAN_360 20",
    },
    "INVESTIGATE": {
        "description": "Move toward a direction of interest.",
        "args": {"direction": ("str", "left | right | forward | backward")},
        "example": "INVESTIGATE forward",
    },
    "NAVIGATE_TO": {
        "description": "Navigate to a named location.",
        "args": {"location": ("str", "name from locations config (home, A, B, C)")},
        "example": "NAVIGATE_TO home",
    },
    "REPORT": {
        "description": "Send a status report to the operator.",
        "args": {"message": ("str", "report content")},
        "example": "REPORT patrol complete",
    },
}

# ── Wheeled-specific skills ───────────────────────────────────────────────────

WHEELED_SKILLS: dict[str, dict] = {
    "FORWARD": {
        "description": "Move forward at given speed.",
        "args": {"speed": ("int", "motor speed 0-100")},
        "example": "FORWARD 60",
    },
    "BACKWARD": {
        "description": "Move backward at given speed.",
        "args": {"speed": ("int", "motor speed 0-100")},
        "example": "BACKWARD 30",
    },
    "TURN_LEFT": {
        "description": "Turn left by given degrees.",
        "args": {"degrees": ("int", "angle 1-360")},
        "example": "TURN_LEFT 45",
    },
    "TURN_RIGHT": {
        "description": "Turn right by given degrees.",
        "args": {"degrees": ("int", "angle 1-360")},
        "example": "TURN_RIGHT 90",
    },
    "FOLLOW_WALL": {
        "description": "Follow the wall on the given side.",
        "args": {"side": ("str", "left | right")},
        "example": "FOLLOW_WALL right",
    },
    "TRACK": {
        "description": "Track a moving target.",
        "args": {"target": ("str", "person | object description")},
        "example": "TRACK person",
    },
    "MAP_PERIMETER": {
        "description": "Explore and map the environment using LiDAR SLAM.",
        "args": {},
        "example": "MAP_PERIMETER",
    },
    "PATROL_PERIMETER": {
        "description": "Patrol recorded perimeter waypoints with LiDAR scanning.",
        "args": {"exclude_rtsp": ("bool", "skip RTSP-covered zones (default true)")},
        "example": "PATROL_PERIMETER",
    },
    "EXPLORE_FRONTIER": {
        "description": "Navigate to nearest unmapped frontier for exploration.",
        "args": {},
        "example": "EXPLORE_FRONTIER",
    },
    "DETERRENT": {
        "description": "Activate escalating deterrent sequence (LED, siren, spotlight, speaker).",
        "args": {},
        "example": "DETERRENT",
    },
    "AIM_AT": {
        "description": "Point turret (spotlight + laser) at pixel position in frame.",
        "args": {
            "px_x": ("int", "target pixel X coordinate"),
            "px_y": ("int", "target pixel Y coordinate"),
        },
        "example": "AIM_AT 320 240",
    },
    "STAND_DOWN": {
        "description": "De-escalate all deterrent systems, turret to neutral.",
        "args": {},
        "example": "STAND_DOWN",
    },
    "INVESTIGATE_ZONE": {
        "description": "Navigate to a zone flagged by RTSP camera, scan, and report.",
        "args": {
            "zone": ("str", "zone waypoint name from RTSP camera config"),
        },
        "example": "INVESTIGATE_ZONE garden_center",
    },
    "RETURN_TO_DOCK": {
        "description": "Navigate to charging dock, IR homing for final approach, back in.",
        "args": {},
        "example": "RETURN_TO_DOCK",
    },
    "UNDOCK": {
        "description": "Drive forward off charging dock, confirm undocked.",
        "args": {},
        "example": "UNDOCK",
    },
    "TRACK_INTRUDER": {
        "description": "Follow a detected person, keeping at safe distance.",
        "args": {
            "target": ("str", "target label (default: person)"),
        },
        "example": "TRACK_INTRUDER person",
    },
}

# ── Drone-specific skills ─────────────────────────────────────────────────────

DRONE_SKILLS: dict[str, dict] = {
    "HOVER": {
        "description": "Maintain current altitude and position.",
        "args": {"seconds": ("float", "hover duration (0 = indefinite)")},
        "example": "HOVER 5",
    },
    "TAKEOFF": {
        "description": "Arm motors and ascend to target altitude.",
        "args": {"altitude_m": ("float", "target altitude in meters")},
        "example": "TAKEOFF 2.0",
    },
    "LAND": {
        "description": "Descend and disarm motors.",
        "args": {},
        "example": "LAND",
    },
    "ASCEND": {
        "description": "Increase altitude by given meters.",
        "args": {"meters": ("float", "altitude change > 0")},
        "example": "ASCEND 1.5",
    },
    "DESCEND": {
        "description": "Decrease altitude by given meters.",
        "args": {"meters": ("float", "altitude change > 0")},
        "example": "DESCEND 1.0",
    },
    "YAW_LEFT": {
        "description": "Rotate counter-clockwise by given degrees.",
        "args": {"degrees": ("int", "angle 1-360")},
        "example": "YAW_LEFT 90",
    },
    "YAW_RIGHT": {
        "description": "Rotate clockwise by given degrees.",
        "args": {"degrees": ("int", "angle 1-360")},
        "example": "YAW_RIGHT 90",
    },
    "FLY_FORWARD": {
        "description": "Fly forward at given throttle offset.",
        "args": {"speed": ("int", "throttle offset 0-200 PWM")},
        "example": "FLY_FORWARD 100",
    },
    "RETURN_HOME": {
        "description": "Return to launch point and land.",
        "args": {},
        "example": "RETURN_HOME",
    },
}

# ── Humanoid-specific skills ──────────────────────────────────────────────────

HUMANOID_SKILLS: dict[str, dict] = {
    "STAND": {
        "description": "Stand upright in neutral pose.",
        "args": {},
        "example": "STAND",
    },
    "CROUCH": {
        "description": "Lower center of gravity — defensive pose.",
        "args": {},
        "example": "CROUCH",
    },
    "SIT": {
        "description": "Sit down.",
        "args": {},
        "example": "SIT",
    },
    "WALK_FORWARD": {
        "description": "Walk forward N steps.",
        "args": {"steps": ("int", "number of steps")},
        "example": "WALK_FORWARD 10",
    },
    "WALK_BACKWARD": {
        "description": "Walk backward N steps.",
        "args": {"steps": ("int", "number of steps")},
        "example": "WALK_BACKWARD 5",
    },
    "TURN_LEFT": {
        "description": "Rotate body left by given degrees.",
        "args": {"degrees": ("int", "angle 1-180")},
        "example": "TURN_LEFT 45",
    },
    "TURN_RIGHT": {
        "description": "Rotate body right by given degrees.",
        "args": {"degrees": ("int", "angle 1-180")},
        "example": "TURN_RIGHT 45",
    },
    "WAVE": {
        "description": "Wave right hand as greeting.",
        "args": {},
        "example": "WAVE",
    },
    "NOD": {
        "description": "Nod head yes.",
        "args": {},
        "example": "NOD",
    },
    "SHAKE_HEAD": {
        "description": "Shake head no.",
        "args": {},
        "example": "SHAKE_HEAD",
    },
}

# ── Robot type -> skill set ───────────────────────────────────────────────────

ROBOT_SKILLS: dict[str, dict] = {
    "wheeled": {**UNIVERSAL_SKILLS, **WHEELED_SKILLS},
    "drone": {**UNIVERSAL_SKILLS, **DRONE_SKILLS},
    "humanoid": {**UNIVERSAL_SKILLS, **HUMANOID_SKILLS},
    "ackermann": {**UNIVERSAL_SKILLS, **WHEELED_SKILLS},  # same shape as wheeled
}


def get_skills(robot_type: str) -> dict[str, dict]:
    """Return the skill catalog for the given robot type string."""
    return ROBOT_SKILLS.get(robot_type.lower(), ROBOT_SKILLS["wheeled"])


def skill_list_prompt(robot_type: str) -> str:
    """Return a compact skill list suitable for LLM system prompt injection."""
    skills = get_skills(robot_type)
    lines = []
    for name, info in skills.items():
        args = (
            ", ".join(f"<{a}: {t}>" for a, (t, _) in info["args"].items()) if info["args"] else ""
        )
        lines.append(f"- {name} {args}  # {info['description']}")
    return "\n".join(lines)
