"""Milestone 4: zones, place_in, independent goal checks, and ordered tasks."""

import io
import json

import mujoco
import numpy as np
import pytest

from openrobot_brain.agent import run_agent
from openrobot_brain.goals import check_goal
from openrobot_brain.perception import perceive
from openrobot_brain.planner import AstraPlanner, OfflinePlanner
from openrobot_brain.safety import SafetyError, validate_round
from openrobot_brain.scenes import SCENES, ZONE_SPECS, get_scene, make_zone
from openrobot_brain.schema import Observation, PlanParseError, parse_agent_json
from openrobot_brain.simulation.executor import PandaSim, random_layout
from openrobot_brain.world import WorldObject, WorldState
from conftest import pick, respond
from fakes import ScriptedClient


def acts(*a):
    return list(parse_agent_json(json.dumps({"status": "act", "message": "", "observations": [],
                                             "actions": list(a)})).actions)


def place_in(z):
    return {"action": "place_in", "zone": z}


def sorting_world(**moved):
    sc = get_scene("sorting")
    w = WorldState({n: WorldObject(n, n, np.array([x, y, 0.02]), yaw) for n, (x, y, yaw) in sc.layout.items()},
                   zones=dict(sc.zones))
    for k, xy in moved.items():
        w.objects[k].pos = np.array([*xy, 0.02])
    return w


def cubes(**pos):
    return WorldState({k: WorldObject(k, k, np.array(v, float)) for k, v in pos.items()})


# --- zones -------------------------------------------------------------------------------

def test_zone_geometry():
    z = make_zone("red_zone", 0.40, 0.20)
    assert z.contains(0.40, 0.20) and z.contains(0.47, 0.13) and not z.contains(0.48, 0.20)
    slots = z.slots()
    assert len(slots) == 4 and all(z.contains(*s) for s in slots)
    gaps = [np.hypot(a[0] - b[0], a[1] - b[1]) for i, a in enumerate(slots) for b in slots[i + 1:]]
    assert min(gaps) >= 0.06                      # cubes in one zone stay 6 cm apart


def test_zone_of_and_describe():
    w = sorting_world(red_cube=(0.40, 0.21))
    assert w.zone_of("red_cube") == "red_zone" and w.zone_of("blue_cube") is None
    assert "in red_zone" in w.describe() and "red_zone (red zone, 15 cm square)" in w.describe_zones()
    w.held = "red_cube"
    assert w.zone_of("red_cube") is None
    assert w.copy().zones == w.zones


def test_zone_bodies_match_the_map(robot):
    with PandaSim(robot, scene="sorting") as sim:
        for z in sim.scene.zones.values():
            bid = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_BODY, ZONE_SPECS[z.id]["body"])
            assert np.allclose(sim.data.xpos[bid][:2], [z.x, z.y], atol=1e-6)
            g = next(g for g in range(sim.model.ngeom) if sim.model.geom_bodyid[g] == bid)
            assert np.isclose(sim.model.geom_size[g][0], z.half)
            assert sim.model.geom_contype[g] == 0 and sim.model.geom_conaffinity[g] == 0
        assert sim.ground_truth().zones.keys() == sim.scene.zones.keys()


def test_unused_zones_are_parked(robot):
    with PandaSim(robot) as sim:
        for spec in ZONE_SPECS.values():
            bid = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_BODY, spec["body"])
            assert abs(sim.data.xpos[bid][1]) > 5


def test_sorting_scene_starts_unsorted():
    w = sorting_world()
    assert all(w.zone_of(c) is None for c in w.objects)
    assert not check_goal("sort the cubes into their zones", w).passed


def test_random_layouts_avoid_zones(robot):
    sc = get_scene("sorting")
    for seed in range(10):
        for x, y, _ in random_layout(seed, robot, scene=sc).values():
            assert not any(z.contains(x, y) for z in sc.zones.values())


# --- place_in -----------------------------------------------------------------------------

def test_place_in_schema():
    a = acts(place_in("red_zone"))[0]
    assert a.action == "place_in" and a.zone == "red_zone"
    with pytest.raises(PlanParseError):
        parse_agent_json(respond("act", [{"action": "place_in", "zone": "red_zone", "x": 0.5}]))


def test_place_in_validates_and_lands_in_a_slot(robot, kin):
    w = sorting_world()
    vp = validate_round(acts(pick("red_cube"), place_in("red_zone")), robot, kin, robot.rest_pose, w)
    end = vp.segments[-1].world
    assert end.held is None and end.zone_of("red_cube") == "red_zone"
    assert any(np.allclose(end.objects["red_cube"].pos[:2], s) for s in w.zones["red_zone"].slots())


def test_place_in_unknown_zone(robot, kin):
    with pytest.raises(SafetyError, match="unknown zone 'purple_zone'"):
        validate_round(acts(pick("red_cube"), place_in("purple_zone")), robot, kin, robot.rest_pose, sorting_world())


def test_place_in_without_zones(robot, kin):
    w = cubes(red_cube=(0.45, 0.15, 0.02), blue_cube=(0.45, -0.15, 0.02))
    with pytest.raises(SafetyError, match="none in this scene"):
        validate_round(acts(pick("red_cube"), place_in("red_zone")), robot, kin, robot.rest_pose, w)


def test_place_in_empty_handed(robot, kin):
    with pytest.raises(SafetyError):
        validate_round(acts(place_in("red_zone")), robot, kin, robot.rest_pose, sorting_world())


def test_place_in_skips_occupied_slots(robot, kin):
    zone = get_scene("sorting").zones["red_zone"]
    first = zone.slots()[0]
    w = sorting_world(green_cube=first)
    vp = validate_round(acts(pick("red_cube"), place_in("red_zone")), robot, kin, robot.rest_pose, w)
    landed = vp.segments[-1].world.objects["red_cube"].pos[:2]
    assert not np.allclose(landed, first) and zone.contains(*landed)


def test_full_zone_is_refused(robot, kin):
    zone = make_zone("red_zone", 0.45, 0.0)
    w = WorldState({"red_cube": WorldObject("red_cube", "r", np.array([0.60, 0.20, 0.02])),
                    "a": WorldObject("a", "a", np.array([0.45, 0.0, 0.02]))}, zones={"red_zone": zone})
    with pytest.raises(SafetyError, match="no free spot"):
        validate_round(acts(pick("red_cube"), place_in("red_zone")), robot, kin, robot.rest_pose, w)


# --- goal checks --------------------------------------------------------------------------

def test_goal_sort():
    w = sorting_world(red_cube=(0.40, 0.21), green_cube=(0.63, 0.23), blue_cube=(0.40, -0.21))
    assert check_goal("sort the cubes into their zones", w).passed
    assert check_goal("put each cube in its own zone", w).passed
    w.objects["blue_cube"].pos = np.array([0.40, 0.21, 0.02])
    g = check_goal("sort the cubes", w)
    assert not g.passed and "blue_cube is in red_zone" in g.detail


def test_goal_tower_and_order():
    w = cubes(blue_cube=(0.5, 0, 0.02), red_cube=(0.5, 0, 0.06), green_cube=(0.5, 0, 0.10))
    assert check_goal("build a tower", w).passed
    assert check_goal("build a tower with blue at the bottom", w).passed
    assert not check_goal("build a tower with green at the bottom", w).passed
    w.objects["green_cube"].pos = np.array([0.53, 0, 0.10])
    assert not check_goal("build a tower", w).passed
    w2 = cubes(blue_cube=(0.5, 0, 0.02), red_cube=(0.5, 0, 0.06), green_cube=(0.5, 0, 0.10))
    w2.held = "green_cube"
    assert "gripper" in check_goal("build a tower", w2).detail


def test_goal_stack():
    w = cubes(blue_cube=(0.5, 0, 0.02), red_cube=(0.5, 0.005, 0.06), green_cube=(0.6, 0.1, 0.02))
    assert check_goal("stack the red cube on the blue cube", w).passed
    assert not check_goal("put the green cube on the blue cube", w).passed


def test_goal_row():
    w = cubes(blue_cube=(0.5, 0.12, 0.02), red_cube=(0.5, 0.0, 0.02), green_cube=(0.5, -0.12, 0.02))
    t = "arrange the cubes in a row: blue, red, green, left to right"
    assert check_goal(t, w).passed
    assert not check_goal("arrange the cubes in a row: green, red, blue, left to right", w).passed
    assert not check_goal("arrange the cubes in a row: red, blue, green", w).passed     # wrong order
    w.objects["red_cube"].pos = np.array([0.55, 0.0, 0.02])
    assert "straight line" in check_goal(t, w).detail
    w.objects["red_cube"].pos = np.array([0.5, 0.10, 0.02])
    assert "5 cm" in check_goal(t, w).detail
    w.objects["red_cube"].pos = np.array([0.5, 0.0, 0.06])
    assert "table" in check_goal(t, w).detail


def test_goal_pick_and_unknown():
    w = cubes(green_cube=(0.5, 0, 0.1))
    assert not check_goal("pick up the green cube", w).passed
    w.held = "green_cube"
    assert check_goal("pick up the green cube", w).passed
    assert check_goal("juggle", w) is None


# --- the model's view ---------------------------------------------------------------------

def test_prompt_lists_zones(robot):
    with PandaSim(robot, scene="sorting") as sim:
        client = ScriptedClient(respond("done"))
        run_agent("sort the cubes into their zones", AstraPlanner(robot, vision=False, client=client), sim, robot,
                  execute=True, vision=False, out=io.StringIO())
    text = client.calls[0]["input"][0]["content"][0]["text"]
    assert "ZONES" in text and "red_zone (red zone, 15 cm square)" in text
    assert "place_in" in client.calls[0]["instructions"]


def test_done_but_goal_not_met_is_flagged(robot):
    with PandaSim(robot, scene="sorting") as sim:
        out = io.StringIO()
        outcome = run_agent("sort the cubes into their zones", AstraPlanner(robot, vision=False,
                            client=ScriptedClient(respond("done"))), sim, robot, execute=True, vision=False, out=out)
    assert outcome.status == "done" and outcome.goal is not None and not outcome.goal.passed
    assert "NOT MET" in out.getvalue() and "goal is not met" in out.getvalue()


def test_perception_ignores_zone_markings(robot):
    with PandaSim(robot, scene="sorting") as sim:
        u, v = sim.camera.project(np.array([0.47, 0.0, 0.04]))
        obs = [Observation(id="green_cube", label="green cube", u=u, v=v),
               Observation(id="red_zone", label="red square", u=100, v=100),
               Observation(id="patch", label="blue zone marking", u=200, v=300)]
        w, warns = perceive(obs, sim.camera, 0.04, zones=sim.scene.zones)
    assert set(w.objects) == {"green_cube"} and set(w.zones) == set(sim.scene.zones)
    assert sum("zone marking" in x for x in warns) == 2


def test_zone_tints_are_not_seen_as_cubes(robot):
    from fakes import find_blobs_with_yaw
    with PandaSim(robot, scene="sorting") as sim:
        blobs = find_blobs_with_yaw(sim.to_png(sim.render("overhead")))
    assert len(blobs) == 3


# --- physics, offline planner -------------------------------------------------------------

@pytest.mark.parametrize("task,scene", [
    ("sort the cubes into their zones", "sorting"),
    ("arrange the cubes in a row: blue, red, green, left to right", "default"),
    ("build a tower with green at the bottom", "default"),
])
def test_smart_tasks_in_physics(robot, task, scene):
    with PandaSim(robot, scene=scene) as sim:
        out = io.StringIO()
        outcome = run_agent(task, OfflinePlanner(), sim, robot, execute=True, vision=False, out=out)
        assert outcome.status == "done", out.getvalue()
        assert outcome.goal is not None and outcome.goal.passed, outcome.goal
        assert "MET" in out.getvalue()


def test_row_blocked_by_wall_is_infeasible(robot):
    with PandaSim(robot, scene="wall") as sim:
        outcome = run_agent("arrange the cubes in a row: red, green, blue", OfflinePlanner(), sim, robot,
                            execute=True, vision=False, out=io.StringIO())
    assert outcome.status == "infeasible" and not outcome.goal.passed
