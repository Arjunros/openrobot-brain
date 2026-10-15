"""Robot, workspace and safety configuration for the Franka Panda tabletop scene.

World frame: robot base at the origin, +x away from the robot, +y to the
robot's left, work surface at z = 0. Units: metres, radians, seconds.

Software joint limits are the Panda's hard limits shrunk by a margin; a test
checks they sit inside the MJCF ranges so the two cannot drift apart.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping

import numpy as np

JOINT_NAMES = tuple(f"joint{i}" for i in range(1, 8))

# Panda hard limits (from the Menagerie MJCF) -- used only to derive software limits.
_PANDA_HARD_LIMITS = {
    "joint1": (-2.8973, 2.8973),
    "joint2": (-1.7628, 1.7628),
    "joint3": (-2.8973, 2.8973),
    "joint4": (-3.0718, -0.0698),
    "joint5": (-2.8973, 2.8973),
    "joint6": (-0.0175, 3.7525),
    "joint7": (-2.8973, 2.8973),
}


@dataclass(frozen=True)
class Workspace:
    """Where the tool centre point (TCP) and objects are allowed to be."""
    x: tuple[float, float] = (0.32, 0.70)
    y: tuple[float, float] = (-0.30, 0.30)
    tcp_z_min: float = 0.015       # TCP at fingertip centre; cube centres sit at 0.02
    tcp_z_max: float = 0.60
    transit_z: float = 0.16        # all sideways motion happens at or above this height

    def contains_xy(self, x: float, y: float) -> bool:
        eps = 1e-6  # floating-point slack, far below any physical meaning
        return self.x[0] - eps <= x <= self.x[1] + eps and self.y[0] - eps <= y <= self.y[1] + eps


@dataclass(frozen=True)
class SafetyLimits:
    max_actions_per_round: int = 6
    max_rounds: int = 8
    max_consecutive_rejections: int = 2
    joint_margin_rad: float = 0.05
    max_joint_velocity_rad_s: float = 1.2      # Panda hardware allows ~2.1-2.6
    max_joint_step_rad: float = 0.15           # between dense samples; catches IK branch flips
    ik_pos_tol_m: float = 0.003
    ik_rot_tol_rad: float = 0.03
    max_wait_s: float = 3.0
    max_round_motion_s: float = 60.0
    place_clearance_m: float = 0.06            # free space required around a place_at target
    obstacle_place_clearance_m: float = 0.08   # place_at targets this far from fixed obstacles (room for the hand)
    max_stack_height: int = 3
    tracking_abort_rad: float = 0.12           # executor stops if the arm lags this far behind


@dataclass(frozen=True)
class RobotConfig:
    name: str
    joint_names: tuple[str, ...]
    joint_limits: Mapping[str, tuple[float, float]]
    rest_pose: np.ndarray        # arm tucked back: camera view of the table is clear
    ready_pose: np.ndarray       # elbow-up pose above the table; IK posture bias
    cube_size_m: float = 0.04
    gripper_open_ctrl: float = 255.0
    gripper_closed_ctrl: float = 0.0
    grasp_min_width_m: float = 0.01      # narrower than this after closing = nothing in the gripper
    grasp_width_tol_m: float = 0.006     # wider than cube + this = gripping the corners (misaligned)
    transit_speed_m_s: float = 0.25
    approach_speed_m_s: float = 0.10
    gripper_time_s: float = 0.7
    workspace: Workspace = field(default_factory=Workspace)
    safety: SafetyLimits = field(default_factory=SafetyLimits)

    def lower(self) -> np.ndarray:
        return np.array([self.joint_limits[j][0] for j in self.joint_names])

    def upper(self) -> np.ndarray:
        return np.array([self.joint_limits[j][1] for j in self.joint_names])


def _software_limits(margin: float) -> Mapping[str, tuple[float, float]]:
    return MappingProxyType({j: (lo + margin, hi - margin) for j, (lo, hi) in _PANDA_HARD_LIMITS.items()})


PANDA = RobotConfig(
    name="franka_panda_tabletop",
    joint_names=JOINT_NAMES,
    joint_limits=_software_limits(SafetyLimits().joint_margin_rad),
    rest_pose=np.array([0.0, -1.65, 0.0, -2.4, 0.0, 1.2, 0.785]),
    ready_pose=np.array([0.0, 0.0, 0.0, -1.57079, 0.0, 1.57079, -0.7853]),
)
