"""The robot's belief about the table: which objects exist, where, and what it holds.

The same structure is used for ground truth (text mode, offline mode, evaluation)
and for what the model perceived from the camera image (vision mode). Skills and
safety checks always plan against the *belief*, which is what a real robot has.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

STACK_XY_TOL = 0.025  # objects closer than this in xy are treated as one stack


@dataclass
class WorldObject:
    id: str
    label: str
    pos: np.ndarray  # centre, metres

    def copy(self) -> "WorldObject":
        return WorldObject(self.id, self.label, self.pos.copy())


@dataclass
class WorldState:
    objects: dict[str, WorldObject] = field(default_factory=dict)
    held: str | None = None
    cube_size: float = 0.04

    def copy(self) -> "WorldState":
        return WorldState({k: o.copy() for k, o in self.objects.items()}, self.held, self.cube_size)

    def resting(self) -> list[WorldObject]:
        return [o for o in self.objects.values() if o.id != self.held]

    def object_above(self, oid: str) -> str | None:
        """Id of an object sitting directly on ``oid``, if any."""
        base = self.objects[oid]
        for o in self.resting():
            if o.id == oid:
                continue
            dz = o.pos[2] - base.pos[2]
            if np.linalg.norm(o.pos[:2] - base.pos[:2]) < STACK_XY_TOL and 0.5 * self.cube_size < dz < 1.5 * self.cube_size:
                return o.id
        return None

    def stack_level(self, oid: str) -> int:
        """0 = on the table, 1 = on one object, ..."""
        return max(0, int(round((self.objects[oid].pos[2] - self.cube_size / 2) / self.cube_size)))

    def describe(self) -> str:
        if not self.objects:
            return "  (no objects)"
        lines = []
        for o in self.objects.values():
            where = "held in gripper" if o.id == self.held else f"x={o.pos[0]:.3f} y={o.pos[1]:+.3f} z={o.pos[2]:.3f}"
            lines.append(f"  - {o.id} ({o.label}): {where}")
        return "\n".join(lines)
