"""MuJoCo-only plan executor.

Accepts ``ValidatedPlan`` objects exclusively. Commands are streamed to
position actuators as smooth (minimum-jerk) interpolations between the
validated waypoints, so the controller never sees a step change larger than
one physics tick's worth of motion.

There is no hardware pathway in this module.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

import mujoco

from ..config import RobotConfig
from ..safety import ValidatedPlan
from . import DEFAULT_SCENE


class SimulationError(RuntimeError):
    pass


@dataclass
class StepResult:
    index: int
    action: str
    commanded: dict[str, float]
    reached: dict[str, float]
    max_tracking_error_rad: float


@dataclass
class ExecutionResult:
    success: bool
    steps: list[StepResult] = field(default_factory=list)
    final_state: dict[str, float] = field(default_factory=dict)
    sim_time_s: float = 0.0
    used_viewer: bool = False
    message: str = ""


def _min_jerk(s: float) -> float:
    s = min(max(s, 0.0), 1.0)
    return s * s * s * (10 - 15 * s + 6 * s * s)


@dataclass(frozen=True)
class JointBinding:
    joint_id: int
    qpos_adr: int
    actuator_id: int
    hard_range: tuple[float, float]


def load_model(xml_path: str | Path = DEFAULT_SCENE) -> mujoco.MjModel:
    path = Path(xml_path)
    if not path.is_file():
        raise SimulationError(f"scene file not found: {path}")
    try:
        return mujoco.MjModel.from_xml_path(str(path))
    except Exception as exc:  # mujoco raises ValueError with parser details
        raise SimulationError(f"failed to load MuJoCo scene {path.name}: {exc}") from exc


def bind_joints(model: mujoco.MjModel, robot: RobotConfig) -> dict[str, JointBinding]:
    """Map every configured joint to its qpos index and position actuator.

    Raises if a joint is missing, is not a hinge, has no actuator, or if the
    software limits are not contained within the simulator's hard range.
    """
    bindings: dict[str, JointBinding] = {}
    problems: list[str] = []
    for name, lim in robot.joint_limits.items():
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            problems.append(f"joint {name!r} not in scene")
            continue
        if model.jnt_type[jid] != mujoco.mjtJoint.mjJNT_HINGE:
            problems.append(f"joint {name!r} is not a hinge")
        lo, hi = (float(v) for v in model.jnt_range[jid])
        if not (lo <= lim.lower and lim.upper <= hi):
            problems.append(f"joint {name!r}: software range [{lim.lower}, {lim.upper}] exceeds scene range [{lo}, {hi}]")
        acts = [
            a for a in range(model.nu)
            if model.actuator_trntype[a] == mujoco.mjtTrn.mjTRN_JOINT and model.actuator_trnid[a, 0] == jid
        ]
        if len(acts) != 1:
            problems.append(f"joint {name!r} needs exactly one actuator, found {len(acts)}")
            continue
        aid = acts[0]
        clo, chi = (float(v) for v in model.actuator_ctrlrange[aid])
        if not (clo <= lim.lower and lim.upper <= chi):
            problems.append(f"actuator for {name!r}: ctrlrange [{clo}, {chi}] does not cover software range")
        bindings[name] = JointBinding(jid, int(model.jnt_qposadr[jid]), aid, (lo, hi))
    if problems:
        raise SimulationError("scene does not match robot config:\n  - " + "\n  - ".join(problems))
    return bindings


class MujocoExecutor:
    """Runs validated plans in MuJoCo. Use as a context manager."""

    def __init__(self, robot: RobotConfig, xml_path: str | Path = DEFAULT_SCENE,
                 headless: bool = True, realtime: bool | None = None,
                 tracking_tolerance_rad: float = 0.05):
        self.robot = robot
        self.model = load_model(xml_path)
        self.data = mujoco.MjData(self.model)
        self.bindings = bind_joints(self.model, robot)
        self.headless = headless
        self.realtime = (not headless) if realtime is None else realtime
        self.tolerance = tracking_tolerance_rad
        self._viewer = None
        self.viewer_error: str | None = None
        self.reset(robot.home_pose)

    # -- lifecycle -----------------------------------------------------------
    def __enter__(self) -> "MujocoExecutor":
        if not self.headless:
            self._open_viewer()
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def _open_viewer(self) -> None:
        try:
            import mujoco.viewer  # noqa: WPS433 - optional GUI dependency
            self._viewer = mujoco.viewer.launch_passive(self.model, self.data)
        except Exception as exc:  # no display, missing GL, macOS without mjpython...
            self._viewer = None
            self.viewer_error = f"viewer unavailable, running headless ({type(exc).__name__}: {exc})"
            self.realtime = False

    def close(self) -> None:
        if self._viewer is not None:
            try:
                self._viewer.close()
            finally:
                self._viewer = None

    # -- state ----------------------------------------------------------------
    def reset(self, pose: Mapping[str, float]) -> None:
        mujoco.mj_resetData(self.model, self.data)
        for name, b in self.bindings.items():
            self.data.qpos[b.qpos_adr] = pose[name]
            self.data.ctrl[b.actuator_id] = pose[name]
        mujoco.mj_forward(self.model, self.data)

    def joint_state(self) -> dict[str, float]:
        return {n: float(self.data.qpos[b.qpos_adr]) for n, b in self.bindings.items()}

    # -- execution ------------------------------------------------------------
    def _step(self) -> None:
        mujoco.mj_step(self.model, self.data)
        if self._viewer is not None:
            if not self._viewer.is_running():
                raise SimulationError("viewer was closed during execution")
            self._viewer.sync()
        if self.realtime:
            time.sleep(self.model.opt.timestep)

    def _settle(self, seconds: float) -> None:
        for _ in range(max(1, int(seconds / self.model.opt.timestep))):
            self._step()

    def execute(self, validated: ValidatedPlan) -> ExecutionResult:
        if not isinstance(validated, ValidatedPlan):
            raise TypeError("executor only accepts plans returned by safety.validate_plan()")
        if validated.robot is not self.robot:
            raise SimulationError("plan was validated against a different robot config")

        current = self.joint_state()
        drift = max(abs(current[n] - validated.initial_state[n]) for n in current)
        if drift > self.tolerance:
            raise SimulationError(
                f"simulator state differs from the plan's validated start state by {drift:.3f} rad; re-validate"
            )

        result = ExecutionResult(success=True, used_viewer=self._viewer is not None)
        dt = self.model.opt.timestep
        t0 = self.data.time
        for step in validated.steps:
            start, end = dict(step.start_state), dict(step.end_state)
            n_ticks = max(1, math.ceil(step.action.duration_s / dt))
            max_err = 0.0
            for k in range(1, n_ticks + 1):
                s = _min_jerk(k / n_ticks)
                for name, b in self.bindings.items():
                    self.data.ctrl[b.actuator_id] = start[name] + s * (end[name] - start[name])
                self._step()
                if step.action.action == "move_joints":
                    q = self.joint_state()
                    for name, b in self.bindings.items():
                        max_err = max(max_err, abs(q[name] - self.data.ctrl[b.actuator_id]))
            if step.action.action == "move_joints":
                self._settle(0.25)
            reached = self.joint_state()
            final_err = max(abs(reached[n] - end[n]) for n in reached)
            result.steps.append(StepResult(step.index, step.action.action, end, reached, round(max_err, 4)))
            if final_err > self.tolerance:
                result.success = False
                result.message = (
                    f"action[{step.index}] did not reach target (error {final_err:.3f} rad > {self.tolerance}); stopped"
                )
                break

        result.final_state = self.joint_state()
        result.sim_time_s = round(self.data.time - t0, 3)
        if result.success:
            result.message = "plan executed in simulation"
        return result


def smoke_check(robot: RobotConfig, xml_path: str | Path = DEFAULT_SCENE) -> dict[str, object]:
    """Load the scene, verify bindings, and confirm every joint responds to a command."""
    from .. import safety
    from ..schema import ActionPlan

    model = load_model(xml_path)
    bindings = bind_joints(model, robot)
    probe = {name: (0.3 if lim.upper >= 0.3 else lim.upper / 2) for name, lim in robot.joint_limits.items()}
    plan = ActionPlan.model_validate({
        "summary": "smoke test",
        "actions": [
            {"action": "move_joints", "duration_s": 1.0,
             "targets": [{"joint": n, "position_rad": v} for n, v in probe.items()]},
            {"action": "move_joints", "duration_s": 1.0,
             "targets": [{"joint": n, "position_rad": robot.home_pose[n]} for n in probe]},
        ],
    })
    validated = safety.validate_plan(plan, robot)
    with MujocoExecutor(robot, xml_path, headless=True) as ex:
        res = ex.execute(validated)
    return {
        "model": model.names.split(b"\x00", 1)[0].decode() if hasattr(model, "names") else "?",
        "nq": model.nq,
        "nu": model.nu,
        "joints": {n: {"qpos_adr": b.qpos_adr, "actuator_id": b.actuator_id, "scene_range": b.hard_range}
                   for n, b in bindings.items()},
        "responds_to_commands": res.success,
        "max_tracking_error_rad": max(s.max_tracking_error_rad for s in res.steps),
    }
