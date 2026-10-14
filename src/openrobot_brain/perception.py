"""Turn the model's image observations into a world belief, and score them.

The model reports each object's pixel centre (of its visible top face) and what
it is stacked on. We convert that to metres ourselves with the calibrated camera
model, so the model never invents raw coordinates in vision mode.

``score_perception`` compares the belief with simulator ground truth. It is for
reporting only; control never uses ground truth in vision mode.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .schema import Observation
from .simulation.camera import CameraModel
from .world import WorldObject, WorldState

COLORS = ("red", "green", "blue", "yellow", "orange", "purple", "black", "white")


def perceive(observations: list[Observation], cam: CameraModel, cube_size: float,
             held: WorldObject | None = None) -> tuple[WorldState, list[str]]:
    warnings: list[str] = []
    by_id = {o.id: o for o in observations}

    def level(o: Observation, seen: tuple[str, ...] = ()) -> int:
        if not o.on_top_of:
            return 0
        if o.on_top_of in seen or o.on_top_of == o.id:
            warnings.append(f"{o.id}: circular 'on_top_of'; assumed on the table")
            return 0
        base = by_id.get(o.on_top_of)
        if base is None:
            warnings.append(f"{o.id}: on_top_of {o.on_top_of!r} is not an observed object; assumed on the table")
            return 0
        return 1 + level(base, seen + (o.id,))

    world = WorldState(cube_size=cube_size)
    for o in observations:
        if held is not None and o.id == held.id:
            continue
        if not cam.in_image(o.u, o.v):
            warnings.append(f"{o.id}: pixel ({o.u:.0f}, {o.v:.0f}) is outside the {cam.width}x{cam.height} image; ignored")
            continue
        lv = min(level(o), 5)
        top_z = (lv + 1) * cube_size
        p = cam.unproject_to_plane(o.u, o.v, top_z)
        world.objects[o.id] = WorldObject(o.id, o.label, np.array([p[0], p[1], top_z - cube_size / 2]))
    if held is not None:
        world.objects[held.id] = held.copy()
        world.held = held.id
    return world, warnings


@dataclass
class PerceptionScore:
    id: str
    label: str
    matched: str | None
    error_cm: float | None
    label_ok: bool


def score_perception(belief: WorldState, truth: WorldState) -> list[PerceptionScore]:
    out = []
    for o in belief.resting():
        best, dist = None, float("inf")
        for t in truth.resting():
            dd = float(np.linalg.norm(o.pos - t.pos))
            if dd < dist:
                best, dist = t, dd
        if best is None:
            out.append(PerceptionScore(o.id, o.label, None, None, False))
            continue
        said = {c for c in COLORS if c in o.label.lower()}
        actual = {c for c in COLORS if c in best.id}
        out.append(PerceptionScore(o.id, o.label, best.id, round(dist * 100, 1), said == actual))
    return out
