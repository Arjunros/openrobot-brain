"""Robot and safety configuration.

The software limits here are deliberately *tighter* than the hard joint ranges
in the MuJoCo XML. A test asserts that every software range sits inside the
simulator range, so the two cannot silently drift apart.

All angles are in radians, all durations in seconds.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping


@dataclass(frozen=True)
class JointLimit:
    lower: float
    upper: float

    def contains(self, value: float) -> bool:
        return self.lower <= value <= self.upper


@dataclass(frozen=True)
class SafetyLimits:
    max_actions: int = 12
    max_step_delta_rad: float = 0.8  # largest change for one joint in one move
    max_joint_velocity_rad_s: float = 1.0  # delta / duration must not exceed this
    min_move_duration_s: float = 0.2
    max_move_duration_s: float = 5.0
    max_wait_s: float = 3.0
    max_total_duration_s: float = 30.0


@dataclass(frozen=True)
class RobotConfig:
    name: str
    joint_limits: Mapping[str, JointLimit]
    home_pose: Mapping[str, float]
    safety: SafetyLimits = field(default_factory=SafetyLimits)

    @property
    def joint_names(self) -> tuple[str, ...]:
        return tuple(self.joint_limits)


DEFAULT_ROBOT = RobotConfig(
    name="orb_arm_4dof",
    joint_limits=MappingProxyType(
        {
            "shoulder_yaw": JointLimit(-1.4, 1.4),
            "shoulder_pitch": JointLimit(-0.9, 1.2),
            "elbow_pitch": JointLimit(-1.6, 1.6),
            "wrist_pitch": JointLimit(-1.2, 1.2),
        }
    ),
    home_pose=MappingProxyType(
        {
            "shoulder_yaw": 0.0,
            "shoulder_pitch": 0.0,
            "elbow_pitch": 0.0,
            "wrist_pitch": 0.0,
        }
    ),
)
