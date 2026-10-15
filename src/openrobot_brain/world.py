"""The robot's belief about the table: which objects exist, where, and what it holds.

The same structure is used for ground truth (text mode, offline mode, evaluation)
and for what the model perceived from the camera image (vision mode). Skills and
safety checks always plan against the *belief*, which is what a real robot has.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

STACK_XY_TOL = 0.025  # objects closer than this in xy are treated as one stack


def wrap_square(a: float) -> float:
    """A square looks identical every 90°, so fold any angle into [-45°, +45°)."""
    return (a + np.pi / 4) % (np.pi / 2) - np.pi / 4


@dataclass
class WorldObject:
    id: str
    label: str
    pos: np.ndarray  # centre, metres
    yaw: float = 0.0  # rotation about the vertical axis, radians

    def copy(self) -> "WorldObject":
        return WorldObject(self.id, self.label, self.pos.copy(), self.yaw)


@dataclass(frozen=True)
class Obstacle:
    """A fixed obstacle from the cell's map (not perceived, never moved).
    Footprint is a rectangle (a post is treated as its bounding square), turned by `yaw`."""
    id: str
    kind: str                         # "wall" | "post"
    x: float
    y: float
    yaw: float
    half: tuple[float, float, float]  # half-sizes along its own x, y, and z; it stands on the table

    @property
    def height(self) -> float:
        return 2 * self.half[2]

    def footprint_distance(self, px: float, py: float) -> float:
        """Distance in the table plane from a point to this obstacle's footprint (0 if inside)."""
        c, s = np.cos(-self.yaw), np.sin(-self.yaw)
        dx, dy = px - self.x, py - self.y
        lx, ly = c * dx - s * dy, s * dx + c * dy
        ox, oy = max(abs(lx) - self.half[0], 0.0), max(abs(ly) - self.half[1], 0.0)
        return float(np.hypot(ox, oy))

    def corners(self, margin: float) -> list[tuple[float, float]]:
        """Footprint corners pushed out by `margin`: candidate waypoints for going around."""
        c, s = np.cos(self.yaw), np.sin(self.yaw)
        hx, hy = self.half[0] + margin, self.half[1] + margin
        return [(self.x + c * a - s * b, self.y + s * a + c * b) for a, b in ((hx, hy), (hx, -hy), (-hx, hy), (-hx, -hy))]

    def describe(self) -> str:
        dims = (f"{2 * self.half[0] * 100:.0f} cm long x {2 * self.half[1] * 100:.0f} cm thick" if self.kind == "wall"
                else f"{2 * self.half[0] * 100:.0f} cm wide")
        return (f"  - {self.id} ({self.kind}, {dims}, {self.height * 100:.0f} cm tall): centre x={self.x:.3f} "
                f"y={self.y:+.3f}, turned {np.degrees(self.yaw):+.0f} deg")


@dataclass(frozen=True)
class Zone:
    """A marked area of the table from the cell's map (a square, flat; nothing collides with it)."""
    id: str
    label: str
    x: float
    y: float
    half: float                       # half the side length, metres
    slot_offset: float = 0.032        # place_in puts cubes at the centres of 4 quadrants

    def contains(self, px: float, py: float) -> bool:
        return abs(px - self.x) <= self.half and abs(py - self.y) <= self.half

    def slots(self) -> list[tuple[float, float]]:
        d = self.slot_offset
        return [(self.x + a, self.y + b) for a, b in ((-d, d), (-d, -d), (d, d), (d, -d))]

    def describe(self) -> str:
        return (f"  - {self.id} ({self.label}, {2 * self.half * 100:.0f} cm square): centre x={self.x:.3f} "
                f"y={self.y:+.3f}")


@dataclass
class WorldState:
    objects: dict[str, WorldObject] = field(default_factory=dict)
    held: str | None = None
    cube_size: float = 0.04
    obstacles: dict[str, Obstacle] = field(default_factory=dict)   # from the map; frozen, safe to share
    zones: dict[str, Zone] = field(default_factory=dict)           # from the map; frozen, safe to share

    def copy(self) -> "WorldState":
        return WorldState({k: o.copy() for k, o in self.objects.items()}, self.held, self.cube_size,
                          dict(self.obstacles), dict(self.zones))

    def zone_of(self, oid: str) -> str | None:
        """The zone an object rests in (by its centre), if any."""
        o = self.objects[oid]
        if oid == self.held:
            return None
        return next((z.id for z in self.zones.values() if z.contains(o.pos[0], o.pos[1])), None)

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

    def describe_obstacles(self) -> str:
        return "\n".join(o.describe() for o in self.obstacles.values()) or "  (none)"

    def describe_zones(self) -> str:
        return "\n".join(z.describe() for z in self.zones.values()) or "  (none)"

    def describe(self) -> str:
        if not self.objects:
            return "  (no objects)"
        lines = []
        for o in self.objects.values():
            where = ("held in gripper" if o.id == self.held else
                     f"x={o.pos[0]:.3f} y={o.pos[1]:+.3f} z={o.pos[2]:.3f} turned {np.degrees(o.yaw):+.0f} deg")
            zone = self.zone_of(o.id) if o.id != self.held else None
            lines.append(f"  - {o.id} ({o.label}): {where}" + (f", in {zone}" if zone else ""))
        return "\n".join(lines)
