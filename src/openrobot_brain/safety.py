"""Independent safety validation.

The validator walks the plan in order and tracks the *predicted* joint state,
so each move is checked against where the arm will actually be at that point,
not against the starting pose. Every action is validated before anything is
returned; a plan with any violation is rejected as a whole.

``ValidatedPlan`` can only be produced here. The executor accepts nothing
else, so an unvalidated plan has no path to the simulator.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

from .config import RobotConfig
from .schema import ActionPlan, MoveJoints, Wait

_TOKEN = object()


class SafetyError(ValueError):
    def __init__(self, violations: list[str]):
        self.violations = violations
        super().__init__("plan rejected by safety validator:\n  - " + "\n  - ".join(violations))


@dataclass(frozen=True)
class ValidatedStep:
    index: int
    action: MoveJoints | Wait
    start_state: Mapping[str, float]
    end_state: Mapping[str, float]


class ValidatedPlan:
    """A plan that passed every check. Construct only via ``validate_plan``."""

    def __init__(self, token: object, plan: ActionPlan, robot: RobotConfig,
                 initial_state: Mapping[str, float], steps: tuple[ValidatedStep, ...]):
        if token is not _TOKEN:
            raise TypeError("ValidatedPlan can only be created by safety.validate_plan()")
        self.plan = plan
        self.robot = robot
        self.initial_state = MappingProxyType(dict(initial_state))
        self.steps = steps

    @property
    def total_duration_s(self) -> float:
        return sum(step.action.duration_s for step in self.steps)

    @property
    def final_state(self) -> Mapping[str, float]:
        return self.steps[-1].end_state if self.steps else self.initial_state


def validate_plan(plan: ActionPlan, robot: RobotConfig,
                  initial_state: Mapping[str, float] | None = None) -> ValidatedPlan:
    limits = robot.safety
    violations: list[str] = []

    state = dict(robot.home_pose if initial_state is None else initial_state)
    if set(state) != set(robot.joint_names):
        raise SafetyError([f"initial state must define exactly these joints: {list(robot.joint_names)}"])
    initial = dict(state)
    for name, value in state.items():
        if not math.isfinite(value) or not robot.joint_limits[name].contains(value):
            violations.append(f"initial state: {name}={value} is outside the safe range")

    if len(plan.actions) > limits.max_actions:
        violations.append(f"plan has {len(plan.actions)} actions; limit is {limits.max_actions}")

    total = 0.0
    steps: list[ValidatedStep] = []
    for i, action in enumerate(plan.actions):
        where = f"action[{i}] {action.action}"
        start = dict(state)
        d = action.duration_s
        total += d if math.isfinite(d) else 0.0

        if isinstance(action, Wait):
            if not (0.0 < d <= limits.max_wait_s):
                violations.append(f"{where}: duration {d}s must be in (0, {limits.max_wait_s}]")
        elif isinstance(action, MoveJoints):
            if not (limits.min_move_duration_s <= d <= limits.max_move_duration_s):
                violations.append(
                    f"{where}: duration {d}s must be in "
                    f"[{limits.min_move_duration_s}, {limits.max_move_duration_s}]"
                )
            seen: set[str] = set()
            for t in action.targets:
                if t.joint not in robot.joint_limits:
                    violations.append(f"{where}: unknown joint {t.joint!r}")
                    continue
                if t.joint in seen:
                    violations.append(f"{where}: joint {t.joint!r} targeted more than once")
                    continue
                seen.add(t.joint)
                lim = robot.joint_limits[t.joint]
                if not lim.contains(t.position_rad):
                    violations.append(
                        f"{where}: {t.joint}={t.position_rad:.3f} rad outside [{lim.lower}, {lim.upper}]"
                    )
                delta = abs(t.position_rad - state[t.joint])
                if delta > limits.max_step_delta_rad:
                    violations.append(
                        f"{where}: {t.joint} moves {delta:.3f} rad from predicted "
                        f"{state[t.joint]:.3f}; limit is {limits.max_step_delta_rad} per action"
                    )
                if d > 0 and delta / d > limits.max_joint_velocity_rad_s:
                    violations.append(
                        f"{where}: {t.joint} speed {delta / d:.2f} rad/s exceeds "
                        f"{limits.max_joint_velocity_rad_s} rad/s"
                    )
                state[t.joint] = t.position_rad
        else:  # pragma: no cover - schema already restricts this
            violations.append(f"{where}: unsupported action")

        steps.append(ValidatedStep(i, action, MappingProxyType(start), MappingProxyType(dict(state))))

    if total > limits.max_total_duration_s:
        violations.append(f"total duration {total:.1f}s exceeds {limits.max_total_duration_s}s")

    if violations:
        raise SafetyError(violations)
    return ValidatedPlan(_TOKEN, plan, robot, initial, tuple(steps))
