"""MuJoCo-only simulator wrapper: reset, observe (camera + state), execute.

``execute`` accepts only ``ValidatedPlan`` objects and plays their joint
trajectories exactly. There is no hardware pathway in this module.
"""

from __future__ import annotations

import io
import math
import time
from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import numpy as np

from ..config import RobotConfig
from ..safety import ValidatedPlan
from ..world import WorldObject, WorldState
from . import CUBE_BODIES, DEFAULT_SCENE
from .camera import CameraModel
from .kinematics import Kinematics, SceneError, load_model

DEFAULT_LAYOUT = {"red_cube": (0.45, 0.15), "green_cube": (0.60, 0.0), "blue_cube": (0.45, -0.15)}
IMAGE_W, IMAGE_H = 640, 480


class SimulationError(RuntimeError):
    pass


class VisionUnavailable(RuntimeError):
    pass


@dataclass
class ActionOutcome:
    index: int
    action: str
    ok: bool
    detail: str


@dataclass
class RoundResult:
    completed: bool
    outcomes: list[ActionOutcome] = field(default_factory=list)
    message: str = ""
    sim_time_s: float = 0.0


def random_layout(seed: int, robot: RobotConfig, min_gap: float = 0.09) -> dict[str, tuple[float, float]]:
    rng = np.random.default_rng(seed)
    ws = robot.workspace
    placed: dict[str, tuple[float, float]] = {}
    for name in CUBE_BODIES:
        for _ in range(500):
            p = (float(rng.uniform(ws.x[0] + 0.04, ws.x[1] - 0.04)), float(rng.uniform(ws.y[0] + 0.04, ws.y[1] - 0.04)))
            if all(math.dist(p, q) >= min_gap for q in placed.values()):
                placed[name] = (round(p[0], 3), round(p[1], 3))
                break
        else:  # pragma: no cover
            raise SimulationError("could not place cubes")
    return placed


class PandaSim:
    """Use as a context manager."""

    def __init__(self, robot: RobotConfig, xml_path: str | Path = DEFAULT_SCENE, headless: bool = True,
                 speed: float = 1.0):
        if not (0.05 <= speed <= 10):
            raise ValueError("speed must be between 0.05 and 10")
        self.robot = robot
        try:
            self.model = load_model(xml_path)
        except SceneError as exc:
            raise SimulationError(str(exc)) from exc
        self.data = mujoco.MjData(self.model)
        self.kin = Kinematics(self.model, robot)
        self.headless = headless
        self.speed = speed
        self._viewer = None
        self._renderer = None
        self.viewer_error: str | None = None
        self.camera: CameraModel | None = None
        self.recorder = None
        self._frame_every = 1
        self._step_count = 0
        self.reset()

    # -- recording ---------------------------------------------------------------
    def start_recording(self, path: str | Path, camera: str = "front") -> None:
        from .recorder import Recorder
        self.recorder = Recorder(self.model, path, camera=camera)
        # video plays at --speed: sample one frame every (speed / fps) simulated seconds
        self._frame_every = max(1, round(self.speed / (self.recorder.fps * self.model.opt.timestep)))

    def annotate(self, title: str | None = None, subtitle: str | None = None, badge=False) -> None:
        if self.recorder is not None:
            self.recorder.annotate(title, subtitle, badge)

    def record_still(self, seconds: float) -> None:
        if self.recorder is not None:
            self.recorder.still(self.data, seconds)

    def stop_recording(self) -> Path | None:
        if self.recorder is None:
            return None
        rec, self.recorder = self.recorder, None
        return rec.close()

    # -- lifecycle --------------------------------------------------------------
    def __enter__(self) -> "PandaSim":
        if not self.headless:
            self._open_viewer()
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def _open_viewer(self) -> None:
        try:
            import mujoco.viewer
            self._viewer = mujoco.viewer.launch_passive(self.model, self.data, show_left_ui=False,
                                                        show_right_ui=False)
            cam = self._viewer.cam
            cam.lookat[:] = (0.30, 0.0, 0.32)     # same view as the 'front' video camera
            cam.distance, cam.azimuth, cam.elevation = 1.85, 137.0, -23.0
            self._viewer.sync()
        except Exception as exc:  # no display, missing OpenGL, macOS without mjpython ...
            self._viewer = None
            self.viewer_error = f"viewer unavailable, running headless ({type(exc).__name__}: {exc})"

    @property
    def has_viewer(self) -> bool:
        return self._viewer is not None

    def hold(self) -> None:
        """Keep the viewer window open until the user closes it."""
        while self._viewer is not None and self._viewer.is_running():
            self._viewer.sync()
            time.sleep(0.03)

    def close(self) -> None:
        if self.recorder is not None:
            self.stop_recording()
        if self._viewer is not None:
            try:
                self._viewer.close()
            finally:
                self._viewer = None
        if self._renderer is not None:
            try:
                self._renderer.close()
            finally:
                self._renderer = None

    # -- state ------------------------------------------------------------------
    def reset(self, layout: dict[str, tuple[float, float]] | None = None) -> None:
        layout = layout or DEFAULT_LAYOUT
        m, d, k = self.model, self.data, self.kin
        mujoco.mj_resetData(m, d)
        d.qpos[k.qadr] = self.robot.rest_pose
        d.qpos[k.finger_adr] = 0.04
        d.ctrl[:7] = self.robot.rest_pose
        d.ctrl[7] = self.robot.gripper_open_ctrl
        for name in CUBE_BODIES:
            x, y = layout[name]
            a = k.cube_adr[name]
            d.qpos[a:a + 3] = (x, y, self.robot.cube_size_m / 2)
            d.qpos[a + 3:a + 7] = (1, 0, 0, 0)
        mujoco.mj_forward(m, d)
        self._settle(0.3)
        self.camera = CameraModel.from_sim(m, d, "overhead", IMAGE_W, IMAGE_H)

    def q(self) -> np.ndarray:
        return self.data.qpos[self.kin.qadr].copy()

    def gripper_width(self) -> float:
        return float(sum(self.data.qpos[a] for a in self.kin.finger_adr))

    def gripper_open(self) -> bool:
        return self.data.ctrl[7] > (self.robot.gripper_open_ctrl + self.robot.gripper_closed_ctrl) / 2

    def holding_something(self) -> bool:
        return (not self.gripper_open()) and self.gripper_width() > self.robot.grasp_min_width_m

    def tcp(self) -> np.ndarray:
        return self.data.site_xpos[self.kin.tcp].copy()

    def ground_truth(self) -> WorldState:
        w = WorldState(cube_size=self.robot.cube_size_m)
        tcp = self.tcp()
        for name in CUBE_BODIES:
            a = self.kin.cube_adr[name]
            pos = self.data.qpos[a:a + 3].copy()
            w.objects[name] = WorldObject(name, name.replace("_", " "), pos)
            if self.holding_something() and np.linalg.norm(pos - tcp) < 0.03:
                w.held = name
        return w

    # -- camera -----------------------------------------------------------------
    def render(self, camera: str = "overhead") -> np.ndarray:
        if self._renderer is None:
            try:
                self._renderer = mujoco.Renderer(self.model, IMAGE_H, IMAGE_W)
            except Exception as exc:
                raise VisionUnavailable(
                    f"offscreen rendering is not available ({type(exc).__name__}: {exc}). "
                    "Use --no-vision, or on Linux set MUJOCO_GL=egl or osmesa.") from exc
        self._renderer.update_scene(self.data, camera=camera)
        return self._renderer.render().copy()

    @staticmethod
    def to_png(img: np.ndarray) -> bytes:
        from PIL import Image
        buf = io.BytesIO()
        Image.fromarray(img).save(buf, format="PNG")
        return buf.getvalue()

    # -- stepping ---------------------------------------------------------------
    def _settle(self, seconds: float) -> None:
        for _ in range(int(seconds / self.model.opt.timestep)):
            mujoco.mj_step(self.model, self.data)

    def _paced_step(self, t0_wall: float, t0_sim: float) -> None:
        mujoco.mj_step(self.model, self.data)
        self._step_count += 1
        if self.recorder is not None and self._step_count % self._frame_every == 0:
            self.recorder.capture(self.data)
        if self._viewer is None:
            return
        if not self._viewer.is_running():
            raise SimulationError("viewer was closed during execution")
        steps = round(self.data.time / self.model.opt.timestep)
        if steps % 8 == 0:  # ~60 Hz display
            self._viewer.sync()
            ahead = (self.data.time - t0_sim) / self.speed - (time.perf_counter() - t0_wall)
            if ahead > 0:
                time.sleep(ahead)

    def execute(self, vp: ValidatedPlan) -> RoundResult:
        if not isinstance(vp, ValidatedPlan):
            raise TypeError("PandaSim.execute only accepts plans returned by safety.validate_round()")
        if vp.robot is not self.robot:
            raise SimulationError("plan was validated against a different robot config")
        if vp.used:
            raise SimulationError("this plan was already executed; the world has changed, re-plan")
        if vp.finger_open != self.gripper_open():
            raise SimulationError("gripper state differs from the plan's validated start; re-plan")
        drift = float(np.max(np.abs(self.q() - vp.q_start)))
        if drift > 0.05:
            raise SimulationError(f"arm is {drift:.3f} rad from the plan's validated start; re-plan")

        vp.used = True
        m, d, r = self.model, self.data, self.robot
        dt = m.opt.timestep
        res = RoundResult(completed=True)
        t0_wall, t0_sim = time.perf_counter(), d.time
        done_actions: dict[int, ActionOutcome] = {}

        for seg in vp.segments:
            if seg.action_index >= 0:
                self.annotate(subtitle=f"action[{seg.action_index}] {vp.actions[seg.action_index].action}: {seg.label}")
            else:
                self.annotate(subtitle=seg.label)
            n = max(1, math.ceil(seg.duration / dt))
            for k in range(1, n + 1):
                t = min(seg.duration, k * dt)
                q_cmd = np.array([np.interp(t, seg.t, seg.q[:, j]) for j in range(7)])
                d.ctrl[:7] = q_cmd
                d.ctrl[7] = seg.gripper
                self._paced_step(t0_wall, t0_sim)
                if seg.kind != "gripper":
                    err = float(np.max(np.abs(self.q() - q_cmd)))
                    if err > r.safety.tracking_abort_rad:
                        res.completed = False
                        res.message = f"stopped: arm blocked during '{seg.label}' (tracking error {err:.2f} rad)"
                        break
            if not res.completed:
                break

            i = seg.action_index
            if seg.label == "close gripper":
                w = self.gripper_width()
                ok = w > r.grasp_min_width_m
                done_actions[i] = ActionOutcome(i, "pick", ok, f"gripper closed at {w * 100:.1f} cm "
                                                + ("(holding an object)" if ok else "(closed on nothing: grasp missed)"))
                if not ok:
                    res.completed = False
                    res.message = f"stopped: grasp missed in action[{i}]"
                    break
            elif seg.label.startswith("lift"):
                if not self.holding_something():
                    done_actions[i] = ActionOutcome(i, "pick", False, "object slipped out during lift")
                    res.completed = False
                    res.message = f"stopped: object dropped in action[{i}]"
                    break
            elif seg.label == "retreat":
                act = vp.actions[i].action
                done_actions[i] = ActionOutcome(i, act, True, "released object")

        for i, a in enumerate(vp.actions):
            if i not in done_actions:
                started = any(s.action_index == i for s in vp.segments[:])
                if res.completed:
                    done_actions[i] = ActionOutcome(i, a.action, True, "done")
                elif started:
                    done_actions.setdefault(i, ActionOutcome(i, a.action, False, "not completed"))
        res.outcomes = [done_actions[i] for i in sorted(done_actions)]
        res.sim_time_s = round(d.time - t0_sim, 2)
        if res.completed:
            res.message = "round executed"
        return res


def smoke_check(robot: RobotConfig, xml_path: str | Path = DEFAULT_SCENE) -> dict[str, object]:
    """Load the scene, check the config matches it, and run one real pick-and-place headless."""
    import json

    from ..safety import validate_round
    from ..schema import parse_agent_json

    with PandaSim(robot, xml_path, headless=True) as sim:
        m = sim.model
        problems = []
        for j, (lo, hi) in robot.joint_limits.items():
            jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j)
            hlo, hhi = m.jnt_range[jid]
            if not (hlo <= lo and hi <= hhi):
                problems.append(f"{j}: software range exceeds scene range")
        if problems:
            raise SimulationError("; ".join(problems))
        resp = parse_agent_json(json.dumps({"status": "act", "message": "smoke", "observations": [], "actions": [
            {"action": "pick", "object": "red_cube"}, {"action": "place_on", "object": "blue_cube"}]}))
        vp = validate_round(resp.actions, robot, sim.kin, sim.q(), sim.ground_truth(), sim.gripper_open())
        result = sim.execute(vp)
        gt = sim.ground_truth()
        red, blue = gt.objects["red_cube"].pos, gt.objects["blue_cube"].pos
        stacked = bool(np.linalg.norm(red[:2] - blue[:2]) < 0.015 and abs(red[2] - blue[2] - 0.04) < 0.006)
        return {
            "scene": "franka_panda + 3 cubes",
            "nq": m.nq, "nu": m.nu,
            "pick_and_place_executed": result.completed,
            "red_on_blue": stacked,
            "red_cube_final": [round(float(v), 3) for v in red],
            "sim_time_s": result.sim_time_s,
        }
