"""Pinhole model of a fixed MuJoCo camera: world <-> pixel.

Pixel convention (matches ``mujoco.Renderer`` output): origin at the top-left,
u to the right, v down, pixel centres at integer + 0.5.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np


@dataclass(frozen=True)
class CameraModel:
    pos: np.ndarray       # camera origin, world
    R: np.ndarray         # camera axes as columns (x right, y up, looks along -z)
    f: float              # focal length in pixels
    width: int
    height: int

    @classmethod
    def from_sim(cls, m: mujoco.MjModel, d: mujoco.MjData, name: str, width: int, height: int) -> "CameraModel":
        cid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, name)
        if cid < 0:
            raise ValueError(f"camera {name!r} not in scene")
        fovy = np.deg2rad(m.cam_fovy[cid])
        f = 0.5 * height / np.tan(0.5 * fovy)
        return cls(d.cam_xpos[cid].copy(), d.cam_xmat[cid].reshape(3, 3).copy(), float(f), width, height)

    def project(self, p_world: np.ndarray) -> tuple[float, float]:
        pc = self.R.T @ (np.asarray(p_world, float) - self.pos)
        if pc[2] >= 0:
            raise ValueError("point is behind the camera")
        u = self.width / 2 + self.f * pc[0] / -pc[2]
        v = self.height / 2 - self.f * pc[1] / -pc[2]
        return float(u), float(v)

    def unproject_to_plane(self, u: float, v: float, z: float) -> np.ndarray:
        """Intersect the pixel's viewing ray with the horizontal plane at height z."""
        d_cam = np.array([(u - self.width / 2) / self.f, -(v - self.height / 2) / self.f, -1.0])
        d = self.R @ d_cam
        if abs(d[2]) < 1e-9:
            raise ValueError("ray parallel to plane")
        t = (z - self.pos[2]) / d[2]
        if t <= 0:
            raise ValueError("plane is behind the camera")
        return self.pos + t * d

    def in_image(self, u: float, v: float) -> bool:
        return 0 <= u <= self.width and 0 <= v <= self.height
