"""Kinematics on a private scratch copy of the scene: FK, IK, and contact checks.

Nothing here steps physics or touches the simulation that the executor runs.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

from ..config import RobotConfig
from ..world import WorldState
from . import CUBE_BODIES, DEFAULT_SCENE, ROBOT_BODIES

# TCP orientation used for every manipulation pose: fingers pointing straight down,
# fingers closing along world x. (Measured at the ready pose.)
R_DOWN = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]])
_PARK = np.array([5.0, 5.0, 0.02])  # where unused cube bodies go in collision checks


def rot_z(a: float) -> np.ndarray:
    """Rotation by angle a (radians) about the world's vertical axis."""
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def grasp_rotation(yaw: float) -> np.ndarray:
    """Gripper pointing straight down, turned by `yaw` about the vertical.
    At yaw = 0 the fingers close along world x; at yaw = a they close along the
    x-axis turned by a, i.e. square onto a cube that is itself turned by a."""
    return rot_z(yaw) @ R_DOWN


def yaw_quat(yaw: float) -> tuple[float, float, float, float]:
    """MuJoCo quaternion (w, x, y, z) for a turn of `yaw` about the vertical."""
    return (float(np.cos(yaw / 2)), 0.0, 0.0, float(np.sin(yaw / 2)))


class SceneError(RuntimeError):
    pass


def load_model(xml_path: str | Path = DEFAULT_SCENE) -> mujoco.MjModel:
    path = Path(xml_path)
    if not path.is_file():
        raise SceneError(f"scene file not found: {path}")
    try:
        return mujoco.MjModel.from_xml_path(str(path))
    except Exception as exc:
        raise SceneError(f"failed to load MuJoCo scene {path.name}: {exc}") from exc


@dataclass(frozen=True)
class IKResult:
    q: np.ndarray
    pos_err: float
    rot_err: float


class Kinematics:
    def __init__(self, model: mujoco.MjModel, robot: RobotConfig):
        self.m = model
        self.d = mujoco.MjData(model)
        self.robot = robot
        jid = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, j) for j in robot.joint_names]
        if min(jid) < 0:
            raise SceneError("scene is missing Panda arm joints")
        self.qadr = np.array([model.jnt_qposadr[j] for j in jid])
        self.dadr = np.array([model.jnt_dofadr[j] for j in jid])
        self.finger_adr = [model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"finger_joint{i}")]
                           for i in (1, 2)]
        self.tcp = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "tcp")
        if self.tcp < 0:
            raise SceneError("scene is missing the 'tcp' site")
        self.cube_adr = {}
        self.cube_geom = {}
        for name in CUBE_BODIES:
            j = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            self.cube_adr[name] = model.jnt_qposadr[j]
            b = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
            self.cube_geom[name] = b
        self.robot_bodies = {mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, b) for b in ROBOT_BODIES}
        self.floor_geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        self.lo, self.hi = robot.lower(), robot.upper()
        self._jp = np.zeros((3, model.nv))
        self._jr = np.zeros((3, model.nv))

    # -- forward ---------------------------------------------------------------
    def _set_arm(self, q: np.ndarray) -> None:
        self.d.qpos[self.qadr] = q

    def fk(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        self._set_arm(q)
        mujoco.mj_kinematics(self.m, self.d)
        return self.d.site_xpos[self.tcp].copy(), self.d.site_xmat[self.tcp].reshape(3, 3).copy()

    # -- inverse ---------------------------------------------------------------
    def ik(self, p_target: np.ndarray, q_seed: np.ndarray, R_target: np.ndarray = R_DOWN,
           iters: int = 200, posture_gain: float = 0.05) -> IKResult:
        """Damped least squares with null-space pull toward the ready pose. Joint limits are clamped."""
        q = np.clip(q_seed.astype(float).copy(), self.lo, self.hi)
        ready = self.robot.ready_pose
        ep = er = np.zeros(3)
        for _ in range(iters):
            self._set_arm(q)
            mujoco.mj_kinematics(self.m, self.d)
            mujoco.mj_comPos(self.m, self.d)
            p = self.d.site_xpos[self.tcp]
            R = self.d.site_xmat[self.tcp].reshape(3, 3)
            ep = p_target - p
            er = 0.5 * sum(np.cross(R[:, i], R_target[:, i]) for i in range(3))
            if np.linalg.norm(ep) < 1e-4 and np.linalg.norm(er) < 1e-3:
                break
            mujoco.mj_jacSite(self.m, self.d, self._jp, self._jr, self.tcp)
            J = np.vstack([self._jp[:, self.dadr], self._jr[:, self.dadr]])
            e = np.concatenate([ep, er])
            JJt = J @ J.T + 1e-4 * np.eye(6)
            dq = J.T @ np.linalg.solve(JJt, e)
            null = np.eye(7) - J.T @ np.linalg.solve(JJt, J)
            dq += null @ (posture_gain * (ready - q))
            q = np.clip(q + np.clip(dq, -0.2, 0.2), self.lo, self.hi)
        return IKResult(q, float(np.linalg.norm(ep)), float(np.linalg.norm(er)))

    # -- contacts --------------------------------------------------------------
    def contacts(self, q: np.ndarray, finger_open: bool, world: WorldState) -> list[tuple[str, str]]:
        """Contacts for arm config ``q`` with objects placed at their *believed*
        positions (a held object rides at the TCP). Returns (a, b) name pairs where
        a/b are 'robot', 'floor', or an object id. Robot-robot and resting
        object-object contacts are not reported."""
        self._set_arm(q)
        w = 0.04 if finger_open else 0.0
        if world.held is not None:
            w = min(w, world.cube_size / 2) if not finger_open else w
        for a in self.finger_adr:
            self.d.qpos[a] = w
        mujoco.mj_kinematics(self.m, self.d)
        tcp = self.d.site_xpos[self.tcp].copy()

        body_to_obj: dict[int, str] = {}
        objs = list(world.objects.values())
        for i, name in enumerate(CUBE_BODIES):
            adr = self.cube_adr[name]
            if i < len(objs):
                o = objs[i]
                pos = tcp if o.id == world.held else o.pos
                body_to_obj[self.cube_geom[name]] = o.id
            else:
                pos = _PARK
            self.d.qpos[adr:adr + 3] = pos
            # each object at its believed rotation: a turned cube reaches further toward its neighbours
            self.d.qpos[adr + 3:adr + 7] = yaw_quat(objs[i].yaw) if i < len(objs) else (1, 0, 0, 0)
        mujoco.mj_fwdPosition(self.m, self.d)

        out: list[tuple[str, str]] = []
        for c in self.d.contact[: self.d.ncon]:
            if c.dist > 0:
                continue
            names = []
            for g in (c.geom1, c.geom2):
                b = self.m.geom_bodyid[g]
                if g == self.floor_geom:
                    names.append("floor")
                elif b in self.robot_bodies:
                    names.append("robot")
                elif b in body_to_obj:
                    names.append(body_to_obj[b])
                else:
                    names.append("other")
            a, b = names
            if a == b == "robot":
                continue
            if "robot" not in (a, b) and world.held not in (a, b):
                continue  # resting objects touching each other or the floor
            out.append((a, b))
        return out