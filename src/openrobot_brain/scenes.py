"""Named scenes: which fixed obstacles stand on the table, and where the cubes start.

Obstacles come from this map, the way a real robot cell knows where its fixtures are.
Only the cubes are perceived (by the model, in vision mode).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .world import Obstacle, Zone

# Geometry of each obstacle body in orb_tabletop.xml (half-sizes in metres; a test checks they match).
OBSTACLE_SPECS = {
    "wall": {"body": "obstacle_wall", "kind": "wall", "half": (0.10, 0.02, 0.08)},
    "post_a": {"body": "obstacle_post_a", "kind": "post", "half": (0.025, 0.025, 0.08)},
    "post_b": {"body": "obstacle_post_b", "kind": "post", "half": (0.025, 0.025, 0.08)},
}

# Marked areas on the mat (visual only). Body names in orb_tabletop.xml; half side 7.5 cm.
ZONE_SPECS = {
    "red_zone": {"body": "zone_red", "label": "red zone"},
    "green_zone": {"body": "zone_green", "label": "green zone"},
    "blue_zone": {"body": "zone_blue", "label": "blue zone"},
}
ZONE_HALF_M = 0.075

Layout = dict[str, tuple[float, float, float]]  # cube -> (x, y, yaw)


def make_zone(zid: str, x: float, y: float) -> Zone:
    return Zone(zid, ZONE_SPECS[zid]["label"], x, y, ZONE_HALF_M)


def make_obstacle(oid: str, x: float, y: float, yaw: float = 0.0) -> Obstacle:
    spec = OBSTACLE_SPECS[oid]
    return Obstacle(oid, spec["kind"], x, y, yaw, spec["half"])


@dataclass(frozen=True)
class Scene:
    name: str
    description: str
    layout: Layout
    obstacles: dict[str, Obstacle] = field(default_factory=dict)
    zones: dict[str, Zone] = field(default_factory=dict)


SCENES: dict[str, Scene] = {
    "default": Scene(
        "default", "three cubes, no obstacles",
        {"red_cube": (0.45, 0.15, 0.0), "green_cube": (0.60, 0.0, 0.0), "blue_cube": (0.45, -0.15, 0.0)},
    ),
    "wall": Scene(
        "wall", "a 16 cm tall wall splits the table; red and blue are on opposite sides",
        {"red_cube": (0.45, 0.15, 0.0), "green_cube": (0.63, 0.15, 0.0), "blue_cube": (0.45, -0.15, 0.0)},
        {"wall": make_obstacle("wall", 0.52, 0.0)},
    ),
    "posts": Scene(
        "posts", "two 16 cm posts stand between the cubes",
        {"red_cube": (0.45, 0.15, 0.0), "green_cube": (0.63, 0.15, 0.0), "blue_cube": (0.45, -0.15, 0.0)},
        {"post_a": make_obstacle("post_a", 0.45, 0.0), "post_b": make_obstacle("post_b", 0.58, -0.04)},
    ),
    "sorting": Scene(
        "sorting", "three coloured zones; the cubes start outside them, each away from its own colour",
        {"red_cube": (0.60, -0.18, 0.3), "green_cube": (0.47, 0.0, -0.2), "blue_cube": (0.60, 0.12, 0.1)},
        zones={"red_zone": make_zone("red_zone", 0.40, 0.21), "green_zone": make_zone("green_zone", 0.63, 0.23),
               "blue_zone": make_zone("blue_zone", 0.40, -0.21)},
    ),
}


def get_scene(name: str) -> Scene:
    if name not in SCENES:
        raise ValueError(f"unknown scene {name!r}; choose from {', '.join(SCENES)}")
    return SCENES[name]
