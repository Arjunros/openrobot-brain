"""Milestone 3: fixed obstacles, clearance, and routing around / over them."""

import io
import json

import mujoco
import numpy as np
import pytest

from openrobot_brain.agent import run_agent
from openrobot_brain.planner import AstraPlanner, OfflinePlanner
from openrobot_brain.safety import SafetyError, validate_round
from openrobot_brain.scenes import OBSTACLE_SPECS, SCENES, get_scene, make_obstacle
from openrobot_brain.schema import Observation, parse_agent_json
from openrobot_brain.simulation.executor import PandaSim, random_layout
from openrobot_brain.simulation.kinematics import OBSTACLE_CLEARANCE_M, Kinematics, load_model
from openrobot_brain.world import WorldObject, WorldState
from conftest import place_at, respond
from fakes import ScriptedClient


def acts(*a):
    return list(parse_agent_json(json.dumps({"status": "act", "message": "", "observations": [],
                                             "actions": list(a)})).actions)


STACK = acts({"action": "pick", "object": "red_cube"}, {"action": "place_on", "object": "blue_cube"})


def world_from(scene_name, layout=None, **extra):
    sc = get_scene(scene_name)
    lay = layout or sc.layout
    w = WorldState({n: WorldObject(n, n, np.array([x, y, 0.02]), yaw) for n, (x, y, yaw) in lay.items()},
                   obstacles=dict(sc.obstacles))
    for k, v in extra.items():
        w.objects[k] = WorldObject(k, k, np.array(v, float))
    return w


def route_labels(vp):
    return {s.label.split("(")[-1].rstrip(")") for s in vp.segments if "(" in s.label}


# --- scene and geometry -----------------------------------------------------------------

def test_obstacle_sizes_match_the_scene_file():
    m = load_model()
    for oid, spec in OBSTACLE_SPECS.items():
        bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, spec["body"])
        assert bid >= 0 and m.body_mocapid[bid] >= 0, oid
        g = next(g for g in range(m.ngeom) if m.geom_bodyid[g] == bid)
        hx, hy, hz = spec["half"]
        if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX:
            assert np.allclose(m.geom_size[g], [hx, hy, hz]), oid
        else:  # cylinder: radius, half-height
            assert np.allclose(m.geom_size[g][:2], [hx, hz]) and hx == hy, oid


@pytest.mark.parametrize("name", [n for n in SCENES if SCENES[n].obstacles])
def test_obstacles_stand_where_the_map_says(robot, name):
    with PandaSim(robot, scene=name) as sim:
        for o in sim.scene.obstacles.values():
            bid = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_BODY, OBSTACLE_SPECS[o.id]["body"])
            assert np.allclose(sim.data.xpos[bid], [o.x, o.y, o.half[2]], atol=1e-6)
        assert sim.ground_truth().obstacles.keys() == sim.scene.obstacles.keys()


def test_unused_obstacles_are_parked_off_the_table(robot):
    with PandaSim(robot) as sim:
        for spec in OBSTACLE_SPECS.values():
            bid = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_BODY, spec["body"])
            assert sim.data.xpos[bid][1] > 5


def test_footprint_distance_and_corners():
    wall = make_obstacle("wall", 0.5, 0.0)          # 20 cm along x, 4 cm thick
    assert wall.footprint_distance(0.5, 0.0) == 0
    assert np.isclose(wall.footprint_distance(0.5, 0.10), 0.08)
    assert np.isclose(wall.footprint_distance(0.70, 0.0), 0.10)
    turned = make_obstacle("wall", 0.5, 0.0, np.pi / 2)  # now along y
    assert np.isclose(turned.footprint_distance(0.5, 0.10), 0.0)
    for cx, cy in wall.corners(0.05):
        assert np.isclose(wall.footprint_distance(cx, cy), np.hypot(0.05, 0.05))


@pytest.mark.parametrize("name", [n for n in SCENES if SCENES[n].obstacles])
def test_random_layouts_keep_clear_of_obstacles(robot, name):
    sc = get_scene(name)
    for seed in range(10):
        for x, y, _ in random_layout(seed, robot, scene=sc).values():
            assert all(o.footprint_distance(x, y) >= 0.10 for o in sc.obstacles.values())


def test_default_scene_layouts_are_unchanged_by_obstacle_support(robot):
    assert random_layout(5, robot) == random_layout(5, robot, scene=get_scene("default"))


# --- clearance ----------------------------------------------------------------------------

def test_checker_keeps_its_own_model(robot):
    m = load_model()
    Kinematics(m, robot)
    assert m.geom_margin.max() == 0.0          # physics never sees the clearance margins


def test_clearance_from_obstacles(robot, kin):
    """A carried cube whose face is 1 cm from the wall counts as a hit (inside the 1.5 cm clearance);
    3 cm away does not. Measured at a height where only the cube is beside the wall."""
    wall = get_scene("wall").obstacles["wall"]
    face = wall.y + wall.half[1]
    half = robot.cube_size_m / 2

    def hits_wall(gap):
        w = WorldState({"c": WorldObject("c", "c", np.zeros(3))}, held="c", obstacles={"wall": wall})
        q = kin.ik(np.array([wall.x, face + half + gap, 0.145]), robot.ready_pose).q
        return any("wall" in pair for pair in kin.contacts(q, False, w))
    assert OBSTACLE_CLEARANCE_M == 0.015
    assert hits_wall(0.010)
    assert not hits_wall(0.030)


def test_arm_through_the_wall_is_named(robot, kin):
    w = world_from("wall")
    q = kin.ik(np.array([0.52, 0.0, 0.10]), robot.ready_pose).q
    assert ("robot", "wall") in kin.contacts(q, True, w) or ("wall", "robot") in kin.contacts(q, True, w)


# --- routing ------------------------------------------------------------------------------

def test_carry_routes_around_the_wall(robot, kin):
    vp = validate_round(STACK, robot, kin, robot.rest_pose, world_from("wall"))
    assert route_labels(vp) & {"around wall", "over at 30 cm"}, [s.label for s in vp.segments]


def test_no_obstacles_means_straight_moves(robot, kin):
    vp = validate_round(STACK, robot, kin, robot.rest_pose, world_from("default"))
    assert not route_labels(vp)


def test_goes_over_when_around_is_impossible(robot, kin):
    """A wall across the table with posts closing both ends: the only way is over the top."""
    obstacles = {"wall": make_obstacle("wall", 0.52, 0.0, np.pi / 2),
                 "post_a": make_obstacle("post_a", 0.52, 0.20), "post_b": make_obstacle("post_b", 0.52, -0.20)}
    w = WorldState({"red_cube": WorldObject("red_cube", "r", np.array([0.38, 0.0, 0.02])),
                    "blue_cube": WorldObject("blue_cube", "b", np.array([0.66, 0.0, 0.02]))}, obstacles=obstacles)
    vp = validate_round(STACK, robot, kin, robot.rest_pose, w)
    assert any(l.startswith("over at") for l in route_labels(vp))
    carry_z = [kin.fk(q)[0][2] for s in vp.segments if s.label.startswith("carry") for q in s.q]
    assert max(carry_z) > 0.16 + 0.10


@pytest.mark.parametrize("name", [n for n in SCENES if SCENES[n].obstacles])
def test_random_obstacle_layouts_validate(robot, kin, name):
    sc = get_scene(name)
    for seed in range(6):
        validate_round(STACK, robot, kin, robot.rest_pose, world_from(name, random_layout(seed, robot, scene=sc)))


def test_cube_right_against_the_wall_is_rejected(robot, kin):
    w = world_from("wall")
    w.objects["red_cube"].pos = np.array([0.52, 0.045, 0.02])          # 2.5 cm from the wall's face
    with pytest.raises(SafetyError, match="wall"):
        validate_round(STACK, robot, kin, robot.rest_pose, w)


def test_place_at_next_to_an_obstacle_is_refused(robot, kin):
    w = world_from("wall")
    with pytest.raises(SafetyError, match="too close to obstacle 'wall'"):
        validate_round(acts({"action": "pick", "object": "red_cube"}, place_at(0.52, -0.06)), robot, kin,
                       robot.rest_pose, w)


# --- the model's view ---------------------------------------------------------------------

def test_prompt_lists_obstacles_from_the_map(robot):
    with PandaSim(robot, scene="wall") as sim:
        client = ScriptedClient(respond("done"))
        run_agent("stack red on blue", AstraPlanner(robot, vision=False, client=client), sim, robot,
                  execute=True, vision=False, out=io.StringIO())
    text = client.calls[0]["input"][0]["content"][0]["text"]
    assert "FIXED OBSTACLES" in text and "wall (wall, 20 cm long x 4 cm thick, 16 cm tall)" in text
    assert "FIXED OBSTACLES" in client.calls[0]["instructions"]


def test_perception_ignores_obstacles_the_model_lists(sim_wall):
    from openrobot_brain.perception import perceive
    u, v = sim_wall.camera.project(np.array([0.45, 0.15, 0.04]))
    obs = [Observation(id="red_cube", label="red cube", u=u, v=v),
           Observation(id="grey_wall", label="grey wall", u=320, v=240)]
    w, warns = perceive(obs, sim_wall.camera, 0.04, obstacles=sim_wall.scene.obstacles)
    assert set(w.objects) == {"red_cube"} and "wall" in w.obstacles
    assert any("fixed obstacle" in x for x in warns)


@pytest.fixture
def sim_wall(robot):
    with PandaSim(robot, scene="wall") as s:
        yield s


# --- physics ------------------------------------------------------------------------------

@pytest.mark.parametrize("name", ["wall", "posts"])
def test_tower_is_built_around_obstacles_in_physics(robot, name):
    with PandaSim(robot, scene=name) as sim:
        out = io.StringIO()
        outcome = run_agent("build a tower", OfflinePlanner(), sim, robot, execute=True, vision=False, out=out)
        assert outcome.status == "done", out.getvalue()
        gt = sim.ground_truth().objects
        b, r, g = gt["blue_cube"].pos, gt["red_cube"].pos, gt["green_cube"].pos
        assert np.linalg.norm(r[:2] - b[:2]) < 0.015 and np.linalg.norm(g[:2] - r[:2]) < 0.015
        assert "arm blocked" not in out.getvalue()
