"""Expand high-level actions into dense, timed joint trajectories.

Each skill checks its own preconditions against the predicted world (e.g. you
can't pick while holding something) and produces ``Segment``s. The segments are
exactly what the executor will play, so the safety validator checks the real
motion, not a description of it.

Rules built into every skill:
  * the TCP always points straight down;
  * sideways motion only happens at or above ``workspace.transit_z``;
  * vertical approach/retreat runs at ``approach_speed``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .config import RobotConfig
from .schema import Action, GoHome, MoveTo, Pick, PlaceAt, PlaceOn, Wait
from .simulation.kinematics import Kinematics
from .world import WorldState

CART_STEP_M = 0.005
JOINT_STEP_RAD = 0.02
MIN_SEG_S = 0.3


def min_jerk(s: np.ndarray | float) -> np.ndarray | float:
    s = np.clip(s, 0.0, 1.0)
    return s ** 3 * (10 - 15 * s + 6 * s ** 2)


@dataclass
class Segment:
    action_index: int        # -1 for automatic segments (e.g. return to camera pose)
    label: str
    kind: str                # cartesian | joint | gripper | wait
    t: np.ndarray            # (N,) seconds, relative to segment start, t[0] = 0
    q: np.ndarray            # (N, 7) joint targets
    gripper: float           # gripper ctrl held during the segment
    finger_open: bool        # predicted finger state, for collision checks
    world: WorldState        # predicted world during the segment
    may_touch: frozenset[str] = frozenset()   # object ids the robot / held object may contact
    held_may_touch_floor: bool = False
    tcp_targets: list[np.ndarray] = field(default_factory=list)
    ik_errors: list[tuple[float, float]] = field(default_factory=list)

    @property
    def duration(self) -> float:
        return float(self.t[-1]) if len(self.t) else 0.0


class SkillError(ValueError):
    pass


class SkillExpander:
    def __init__(self, kin: Kinematics, robot: RobotConfig):
        self.kin = kin
        self.robot = robot
        self.ws = robot.workspace

    # -- helpers ------------------------------------------------------------------
    def _cartesian(self, idx, label, q0, p1, world, gripper, finger_open, speed, **kw) -> Segment:
        p0, _ = self.kin.fk(q0)
        dist = float(np.linalg.norm(p1 - p0))
        n = max(4, math.ceil(dist / CART_STEP_M))
        T = max(MIN_SEG_S, 1.875 * dist / speed)  # peak min-jerk speed = 1.875 * dist / T
        s = np.linspace(0, 1, n + 1)
        qs, targets, errs = [q0.copy()], [p0], [(0.0, 0.0)]
        q = q0
        for si in min_jerk(s[1:]):
            p = p0 + si * (p1 - p0)
            res = self.kin.ik(p, q)
            q = res.q
            qs.append(q)
            targets.append(p)
            errs.append((res.pos_err, res.rot_err))
        return Segment(idx, label, "cartesian", s * T, np.array(qs), gripper, finger_open, world.copy(),
                       tcp_targets=targets, ik_errors=errs, **kw)

    def _joint(self, idx, label, q0, q1, world, gripper, finger_open) -> Segment:
        D = float(np.max(np.abs(q1 - q0)))
        n = max(4, math.ceil(D / JOINT_STEP_RAD))
        vmax = 0.8 * self.robot.safety.max_joint_velocity_rad_s
        T = max(0.5, 1.875 * D / vmax)
        s = np.linspace(0, 1, n + 1)
        qs = q0 + np.outer(min_jerk(s), q1 - q0)
        return Segment(idx, label, "joint", s * T, qs, gripper, finger_open, world.copy())

    def _hold(self, idx, label, kind, q, duration, world, gripper, finger_open, **kw) -> Segment:
        return Segment(idx, label, kind, np.array([0.0, duration]), np.array([q, q]), gripper, finger_open,
                       world.copy(), **kw)

    def _at_rest(self, q) -> bool:
        return float(np.max(np.abs(q - self.robot.rest_pose))) < 0.05

    def _tcp(self, q) -> np.ndarray:
        return self.kin.fk(q)[0]

    def _lookup(self, world: WorldState, oid: str) -> np.ndarray:
        if oid not in world.objects:
            known = ", ".join(world.objects) or "none"
            raise SkillError(f"unknown object {oid!r} (known: {known})")
        return world.objects[oid].pos

    # -- expansion -------------------------------------------------------------
    def expand(self, actions: list[Action], q_start: np.ndarray, world_start: WorldState,
               finger_open_start: bool = True) -> tuple[list[Segment], WorldState, list[str]]:
        """Returns (segments, predicted final world, precondition violations).
        Expansion stops at the first violated precondition."""
        r, ws = self.robot, self.ws
        world = world_start.copy()
        q = q_start.copy()
        open_ = finger_open_start
        segs: list[Segment] = []
        problems: list[str] = []
        grip = lambda: r.gripper_open_ctrl if open_ else r.gripper_closed_ctrl  # noqa: E731

        def leave_rest(i):
            nonlocal q
            if self._at_rest(q):
                segs.append(self._joint(i, "leave camera pose", q, r.ready_pose, world, grip(), open_))
                q = r.ready_pose.copy()

        def rise(i, label):
            nonlocal q
            p = self._tcp(q)
            if p[2] < ws.transit_z - 1e-3:
                segs.append(self._cartesian(i, label, q, np.array([p[0], p[1], ws.transit_z]), world, grip(), open_,
                                            r.approach_speed_m_s, may_touch=frozenset({world.held} - {None})))
                q = segs[-1].q[-1]

        def travel(i, label, x, y):
            nonlocal q
            segs.append(self._cartesian(i, label, q, np.array([x, y, ws.transit_z]), world, grip(), open_,
                                        r.transit_speed_m_s))
            q = segs[-1].q[-1]

        for i, a in enumerate(actions):
            where = f"action[{i}] {a.action}"
            try:
                if isinstance(a, Pick):
                    if world.held is not None:
                        raise SkillError(f"already holding {world.held!r}; place it first")
                    p = self._lookup(world, a.object)
                    above = world.object_above(a.object)
                    if above:
                        raise SkillError(f"{a.object!r} has {above!r} on top of it")
                    if not ws.contains_xy(p[0], p[1]):
                        raise SkillError(f"{a.object!r} at ({p[0]:.3f}, {p[1]:.3f}) is outside the reachable workspace")
                    leave_rest(i)
                    if not open_:
                        segs.append(self._hold(i, "open gripper", "gripper", q, r.gripper_time_s, world,
                                               r.gripper_open_ctrl, True))
                        open_ = True
                    rise(i, "rise")
                    travel(i, f"move above {a.object}", p[0], p[1])
                    segs.append(self._cartesian(i, f"descend to {a.object}", q, p.copy(), world, grip(), open_,
                                                r.approach_speed_m_s, may_touch=frozenset({a.object})))
                    q = segs[-1].q[-1]
                    segs.append(self._hold(i, "close gripper", "gripper", q, r.gripper_time_s, world,
                                           r.gripper_closed_ctrl, False, may_touch=frozenset({a.object}),
                                           held_may_touch_floor=True))
                    open_ = False
                    world.held = a.object
                    segs.append(self._cartesian(i, f"lift {a.object}", q, np.array([p[0], p[1], ws.transit_z]),
                                                world, grip(), open_, r.approach_speed_m_s,
                                                may_touch=frozenset({a.object}), held_may_touch_floor=True))
                    q = segs[-1].q[-1]

                elif isinstance(a, (PlaceOn, PlaceAt)):
                    if world.held is None:
                        raise SkillError("not holding anything")
                    c = world.cube_size
                    if isinstance(a, PlaceOn):
                        if a.object == world.held:
                            raise SkillError("cannot place an object on itself")
                        base = self._lookup(world, a.object)
                        above = world.object_above(a.object)
                        if above:
                            raise SkillError(f"{a.object!r} already has {above!r} on top of it")
                        if world.stack_level(a.object) + 2 > r.safety.max_stack_height:
                            raise SkillError(f"stack would exceed {r.safety.max_stack_height} objects")
                        dest = base + np.array([0, 0, c])
                        touch = frozenset({world.held, a.object})
                        label = f"lower onto {a.object}"
                    else:
                        if not ws.contains_xy(a.x, a.y):
                            raise SkillError(f"({a.x:.3f}, {a.y:.3f}) is outside the workspace "
                                             f"x{ws.x} y{ws.y}")
                        for o in world.resting():
                            if np.linalg.norm(o.pos[:2] - np.array([a.x, a.y])) < r.safety.place_clearance_m:
                                raise SkillError(f"({a.x:.3f}, {a.y:.3f}) is too close to {o.id!r}; need "
                                                 f"{r.safety.place_clearance_m} m clearance")
                        dest = np.array([a.x, a.y, c / 2])
                        touch = frozenset({world.held})
                        label = f"lower to ({a.x:.2f}, {a.y:.2f})"
                    leave_rest(i)
                    rise(i, "rise")
                    travel(i, "carry", dest[0], dest[1])
                    release = dest + np.array([0, 0, 0.003])
                    segs.append(self._cartesian(i, label, q, release, world, grip(), open_, r.approach_speed_m_s,
                                                may_touch=touch, held_may_touch_floor=isinstance(a, PlaceAt)))
                    q = segs[-1].q[-1]
                    held = world.held
                    segs.append(self._hold(i, "open gripper", "gripper", q, r.gripper_time_s, world,
                                           r.gripper_open_ctrl, True, may_touch=touch,
                                           held_may_touch_floor=isinstance(a, PlaceAt)))
                    open_ = True
                    world.objects[held].pos = dest.copy()
                    world.held = None
                    segs.append(self._cartesian(i, "retreat", q, np.array([dest[0], dest[1], ws.transit_z]), world,
                                                grip(), open_, r.approach_speed_m_s, may_touch=touch - {None}))
                    q = segs[-1].q[-1]

                elif isinstance(a, MoveTo):
                    if not ws.contains_xy(a.x, a.y):
                        raise SkillError(f"({a.x:.3f}, {a.y:.3f}) is outside the workspace x{ws.x} y{ws.y}")
                    if not (ws.tcp_z_min <= a.z <= ws.tcp_z_max):
                        raise SkillError(f"z={a.z:.3f} outside [{ws.tcp_z_min}, {ws.tcp_z_max}]")
                    leave_rest(i)
                    rise(i, "rise")
                    travel(i, "move", a.x, a.y)
                    if abs(a.z - ws.transit_z) > 1e-4:
                        segs.append(self._cartesian(i, "move vertically", q, np.array([a.x, a.y, a.z]), world,
                                                    grip(), open_, r.approach_speed_m_s))
                        q = segs[-1].q[-1]

                elif isinstance(a, GoHome):
                    if not self._at_rest(q):
                        rise(i, "rise")
                        segs.append(self._joint(i, "return to camera pose", q, r.rest_pose, world, grip(), open_))
                        q = r.rest_pose.copy()

                elif isinstance(a, Wait):
                    if not (0 < a.duration_s <= r.safety.max_wait_s):
                        raise SkillError(f"duration {a.duration_s}s must be in (0, {r.safety.max_wait_s}]")
                    segs.append(self._hold(i, "wait", "wait", q, a.duration_s, world, grip(), open_))
            except SkillError as exc:
                problems.append(f"{where}: {exc}")
                return segs, world, problems

        # Always finish in the camera pose so the next observation sees the table.
        if not self._at_rest(q):
            rise(-1, "rise")
            segs.append(self._joint(-1, "return to camera pose", q, r.rest_pose, world, grip(), open_))
        return segs, world, problems
