"""Independent goal checks: did the task actually succeed?

The model decides when to say "done". These checks look at the simulator's ground truth and
say whether the goal is really met, so a confident but wrong "done" is visible. They recognise
a few task families from the task text; anything else gets no check (None), never a guess.

  * sort      "sort the cubes into their zones", "put each cube in its colour's zone"
  * tower     "build a tower", "stack all the cubes" (optionally "blue at the bottom, then red, ...")
  * stack     "stack/put the red cube on the blue cube"
  * row       "arrange the cubes in a row: red, green, blue" ("left to right" = robot's left to right)
  * pick      "pick up the green cube"
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

from .world import WorldState

COLORS = ("red", "green", "blue")
XY_TOL = 0.015          # cubes in one stack: centres within 1.5 cm
Z_TOL = 0.008           # stack levels within 8 mm of 2, 6, 10 cm ...
ROW_DEV = 0.02          # row: every cube within 2 cm of the best-fit line
ROW_GAP = 0.05          # row: neighbours at least 5 cm apart


@dataclass(frozen=True)
class GoalCheck:
    goal: str
    passed: bool
    detail: str


def _colors_in_order(text: str) -> list[str]:
    return re.findall(r"\b(red|green|blue)\b", text)


def _cube(world: WorldState, color: str) -> str | None:
    cid = f"{color}_cube"
    return cid if cid in world.objects else None


def _on_top(world: WorldState, top: str, base: str) -> bool:
    a, b = world.objects[top].pos, world.objects[base].pos
    return (top != world.held and base != world.held and np.linalg.norm(a[:2] - b[:2]) < XY_TOL
            and abs(a[2] - b[2] - world.cube_size) < Z_TOL)


def check_goal(task: str, world: WorldState) -> GoalCheck | None:
    t = task.lower()
    colors = [c for c in COLORS if _cube(world, c)]

    if world.zones and ("sort" in t or ("zone" in t and re.search(r"\b(each|own|its|their|matching)\b", t))):
        wrong = []
        for c in colors:
            zone = f"{c}_zone"
            if zone in world.zones and world.zone_of(f"{c}_cube") != zone:
                wrong.append(f"{c}_cube is in {world.zone_of(f'{c}_cube') or 'no zone'}")
        return GoalCheck("sort into colour zones", not wrong,
                         "; ".join(wrong) or "every cube is in its own colour's zone")

    if "tower" in t or re.search(r"stack (all|every)", t):
        cubes = [f"{c}_cube" for c in colors]
        if any(c == world.held for c in cubes):
            return GoalCheck("tower", False, f"{world.held} is still in the gripper")
        by_z = sorted(cubes, key=lambda c: world.objects[c].pos[2])
        ok = all(_on_top(world, top, base) for base, top in zip(by_z, by_z[1:]))
        order = _colors_in_order(t)
        if ok and "bottom" in t and order:
            wanted = [f"{c}_cube" for c in dict.fromkeys(order)]
            if by_z[:len(wanted)] != wanted:
                return GoalCheck("tower", False, f"stacked bottom-up as {', '.join(by_z)}; asked for {', '.join(wanted)}")
        return GoalCheck("tower", ok, ("all cubes in one aligned stack, bottom-up: " + ", ".join(by_z)) if ok
                         else "the cubes are not one aligned stack")

    m = re.search(r"\b(red|green|blue)\b(?: cube)? (?:on|onto|on top of) (?:the )?\b(red|green|blue)\b", t)
    if m and ("stack" in t or "put" in t or "place" in t):
        top, base = _cube(world, m.group(1)), _cube(world, m.group(2))
        if top and base:
            ok = _on_top(world, top, base)
            return GoalCheck("stack", ok, f"{top} {'is' if ok else 'is not'} resting on {base}")

    if "row" in t or "line" in t:
        order = [c for c in dict.fromkeys(_colors_in_order(t)) if _cube(world, c)] or colors
        cubes = [f"{c}_cube" for c in order]
        if any(c == world.held for c in cubes):
            return GoalCheck("row", False, f"{world.held} is still in the gripper")
        pts = np.array([world.objects[c].pos for c in cubes])
        if np.any(np.abs(pts[:, 2] - world.cube_size / 2) > Z_TOL):
            return GoalCheck("row", False, "not every cube is standing on the table")
        xy = pts[:, :2]
        centre = xy.mean(axis=0)
        direction = np.linalg.svd(xy - centre)[2][0]
        along = (xy - centre) @ direction
        off = np.abs((xy - centre) @ np.array([-direction[1], direction[0]]))
        if off.max() > ROW_DEV:
            return GoalCheck("row", False, f"not in a straight line (a cube is {off.max() * 100:.1f} cm off it)")
        if np.any(np.abs(np.diff(np.sort(along))) < ROW_GAP):
            return GoalCheck("row", False, "neighbours are closer than 5 cm")
        monotonic = np.all(np.diff(along) > 0) or np.all(np.diff(along) < 0)
        if not monotonic:
            return GoalCheck("row", False, f"in a line, but not in the order {', '.join(order)}")
        if "left to right" in t and not (xy[0, 1] > xy[-1, 1]):
            return GoalCheck("row", False, f"in order, but {cubes[0]} is not on the robot's left")
        return GoalCheck("row", True, f"in a straight line in the order {', '.join(order)}")

    m = re.search(r"\b(pick up|lift|grab)\b.*\b(red|green|blue)\b", t)
    if m and _cube(world, m.group(2)):
        cid = f"{m.group(2)}_cube"
        return GoalCheck("pick", world.held == cid, f"{cid} {'is' if world.held == cid else 'is not'} in the gripper")
    return None
