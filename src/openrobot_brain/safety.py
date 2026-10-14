"""Independent safety validation of one agent round.

The model's actions are expanded into the exact joint trajectory the robot
would follow (``skills.py``), then every sample of that trajectory is checked:

  * action count and logical preconditions (holding, stacking, clear space);
  * TCP inside the workspace box, sideways motion only at transit height;
  * IK reached each Cartesian target (position and orientation tolerance);
  * joint limits (with margin), joint step between samples, joint speed;
  * collisions against the *believed* world: the arm may not touch the floor
    or any object except the one it is grasping or placing;
  * total motion time.

Any violation rejects the whole round before anything moves. ``ValidatedPlan``
can only be created here, and the executor accepts nothing else.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import RobotConfig
from .schema import Action
from .simulation.kinematics import Kinematics
from .skills import Segment, SkillExpander
from .world import WorldState

_TOKEN = object()


class SafetyError(ValueError):
    def __init__(self, violations: list[str]):
        self.violations = violations
        super().__init__("round rejected by safety validator:\n  - " + "\n  - ".join(violations))


class ValidatedPlan:
    """Construct only via ``validate_round``."""

    def __init__(self, token, actions, robot, q_start, world_start, segments, world_end, finger_open=True):
        if token is not _TOKEN:
            raise TypeError("ValidatedPlan can only be created by safety.validate_round()")
        self.used = False            # single use: the world changes once a plan has run
        self.finger_open = finger_open
        self.actions = tuple(actions)
        self.robot = robot
        self.q_start = q_start.copy()
        self.world_start = world_start.copy()
        self.segments: tuple[Segment, ...] = tuple(segments)
        self.world_end = world_end

    @property
    def duration_s(self) -> float:
        return sum(s.duration for s in self.segments)


@dataclass
class _Check:
    robot: RobotConfig
    kin: Kinematics

    def segment(self, s: Segment, v: list[str]) -> None:
        r, ws, lim = self.robot, self.robot.workspace, self.robot.safety
        tag = f"action[{s.action_index}] {s.label}" if s.action_index >= 0 else s.label
        lo, hi = r.lower(), r.upper()

        # joint limits
        bad = np.where((s.q < lo - 1e-9) | (s.q > hi + 1e-9))
        if bad[0].size:
            j = r.joint_names[bad[1][0]]
            v.append(f"{tag}: {j} would leave its safe range")
        # continuity + speed
        dq = np.abs(np.diff(s.q, axis=0))
        if dq.size:
            if dq.max() > lim.max_joint_step_rad:
                v.append(f"{tag}: joint jump of {dq.max():.2f} rad between samples (IK branch flip)")
            dt = np.diff(s.t)[:, None]
            speed = (dq / np.maximum(dt, 1e-9)).max()
            if speed > lim.max_joint_velocity_rad_s + 1e-6:
                v.append(f"{tag}: joint speed {speed:.2f} rad/s exceeds {lim.max_joint_velocity_rad_s}")
        # Cartesian checks
        if s.kind == "cartesian":
            for (pe, re), p in zip(s.ik_errors, s.tcp_targets):
                if pe > lim.ik_pos_tol_m or re > lim.ik_rot_tol_rad:
                    v.append(f"{tag}: cannot reach ({p[0]:.3f}, {p[1]:.3f}, {p[2]:.3f}) "
                             f"(IK error {pe * 1000:.1f} mm / {re:.3f} rad)")
                    break
            p0, p1 = s.tcp_targets[0], s.tcp_targets[-1]
            for p in (p1,):
                if not ws.contains_xy(p[0], p[1]):
                    v.append(f"{tag}: TCP target ({p[0]:.3f}, {p[1]:.3f}) outside workspace x{ws.x} y{ws.y}")
                if not (ws.tcp_z_min - 1e-6 <= p[2] <= ws.tcp_z_max + 1e-6):
                    v.append(f"{tag}: TCP height {p[2]:.3f} outside [{ws.tcp_z_min}, {ws.tcp_z_max}]")
            if np.linalg.norm(p1[:2] - p0[:2]) > 0.01 and min(p0[2], p1[2]) < ws.transit_z - 2e-3:
                v.append(f"{tag}: sideways move below transit height {ws.transit_z} m")
        # collisions, every sample
        for q in s.q:
            for a, b in self.kin.contacts(q, s.finger_open, s.world):
                pair = {a, b}
                if "floor" in pair and "robot" in pair:
                    v.append(f"{tag}: arm would hit the table")
                    return
                if "robot" in pair:
                    other = (pair - {"robot"}).pop() if len(pair) > 1 else "robot"
                    if other not in s.may_touch and other != s.world.held:
                        v.append(f"{tag}: arm would hit {other!r}")
                        return
                    continue
                held = s.world.held
                other = (pair - {held}).pop() if len(pair) > 1 else held
                if other == "floor" and s.held_may_touch_floor:
                    continue
                if other in s.may_touch:
                    continue
                v.append(f"{tag}: held object {held!r} would hit {other!r}")
                return


def validate_round(actions: list[Action], robot: RobotConfig, kin: Kinematics, q_start: np.ndarray,
                   world: WorldState, finger_open: bool = True) -> ValidatedPlan:
    lim = robot.safety
    violations: list[str] = []
    if len(actions) > lim.max_actions_per_round:
        violations.append(f"{len(actions)} actions in one round; limit is {lim.max_actions_per_round}")
    lo, hi = robot.lower(), robot.upper()
    if np.any(q_start < lo - 1e-6) or np.any(q_start > hi + 1e-6):
        violations.append("start configuration is outside the safe joint range")
    if violations:
        raise SafetyError(violations)

    segments, world_end, problems = SkillExpander(kin, robot).expand(list(actions), q_start, world, finger_open)
    violations.extend(problems)
    check = _Check(robot, kin)
    for s in segments:
        check.segment(s, violations)
    total = sum(s.duration for s in segments)
    if total > lim.max_round_motion_s:
        violations.append(f"round motion time {total:.1f}s exceeds {lim.max_round_motion_s}s")
    # de-duplicate while keeping order
    violations = list(dict.fromkeys(violations))
    if violations:
        raise SafetyError(violations)
    return ValidatedPlan(_TOKEN, actions, robot, q_start, world, segments, world_end, finger_open)
