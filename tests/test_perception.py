import numpy as np
import pytest

from openrobot_brain.perception import perceive, score_perception
from openrobot_brain.schema import Observation
from openrobot_brain.simulation.executor import PandaSim, VisionUnavailable, random_layout
from conftest import make_world
from fakes import find_blobs


@pytest.fixture(scope="module")
def sim(robot):
    with PandaSim(robot) as s:
        yield s


def obs(id_, u, v, on=""):
    return Observation(id=id_, label=id_.replace("_", " "), u=float(u), v=float(v), on_top_of=on)


def test_projection_round_trip(sim):
    cam = sim.camera
    for p in ([0.45, 0.15, 0.04], [0.6, 0.0, 0.08], [0.33, -0.28, 0.04]):
        u, v = cam.project(np.array(p))
        assert np.allclose(cam.unproject_to_plane(u, v, p[2]), p, atol=1e-9)


def test_image_conventions(sim):
    cam = sim.camera
    u0, v0 = cam.project(np.array([0.5, 0.0, 0.04]))
    u_far, v_far = cam.project(np.array([0.6, 0.0, 0.04]))
    u_left, _ = cam.project(np.array([0.5, 0.1, 0.04]))
    assert v_far < v0           # +x (away from robot) is UP in the image
    assert u_left < u0          # +y (robot's left) is LEFT in the image


def test_rendered_cubes_are_where_the_camera_model_says(sim):
    try:
        png = sim.to_png(sim.render())
    except VisionUnavailable as exc:
        pytest.skip(str(exc))
    blobs = find_blobs(png)
    truth = sim.ground_truth()
    for color in ("red", "green", "blue"):
        p = truth.objects[f"{color}_cube"].pos + [0, 0, 0.02]   # top face
        u, v = sim.camera.project(p)
        assert np.hypot(blobs[color][0] - u, blobs[color][1] - v) < 4.0, color


def test_perceive_converts_pixels_to_metres(sim, robot):
    cam = sim.camera
    u, v = cam.project(np.array([0.5, 0.1, 0.04]))
    w, warns = perceive([obs("red_cube", u, v)], cam, robot.cube_size_m)
    assert not warns
    assert np.allclose(w.objects["red_cube"].pos, [0.5, 0.1, 0.02], atol=1e-6)


def test_perceive_stack_levels(sim, robot):
    cam = sim.camera
    ub, vb = cam.project(np.array([0.5, -0.1, 0.04]))
    ur, vr = cam.project(np.array([0.5, -0.1, 0.08]))
    w, _ = perceive([obs("blue_cube", ub, vb), obs("red_cube", ur, vr, "blue_cube")], cam, robot.cube_size_m)
    assert np.allclose(w.objects["red_cube"].pos, [0.5, -0.1, 0.06], atol=1e-6)
    assert w.object_above("blue_cube") == "red_cube"


def test_perceive_warnings(sim, robot):
    cam = sim.camera
    w, warns = perceive([obs("a_cube", 9999, 10), obs("b_cube", 300, 200, "ghost"),
                         obs("c_cube", 200, 200, "d_cube"), obs("d_cube", 210, 200, "c_cube")],
                        cam, robot.cube_size_m)
    assert "a_cube" not in w.objects
    assert any("outside" in x for x in warns)
    assert any("ghost" in x for x in warns)
    assert any("circular" in x for x in warns)


def test_held_object_is_kept_and_not_reobserved(sim, robot):
    held = make_world(red_cube=(0.5, 0.0, 0.3)).objects["red_cube"]
    w, _ = perceive([obs("red_cube", 100, 100), obs("blue_cube", 300, 300)], sim.camera, robot.cube_size_m, held)
    assert w.held == "red_cube"
    assert np.allclose(w.objects["red_cube"].pos, [0.5, 0.0, 0.3])


def test_score_perception():
    belief = make_world(red_cube=(0.452, 0.15, 0.02), mystery=(0.6, 0.0, 0.02))
    belief.objects["mystery"].label = "yellow cube"
    truth = make_world()
    scores = {s.id: s for s in score_perception(belief, truth)}
    assert scores["red_cube"].matched == "red_cube" and scores["red_cube"].error_cm == 0.2
    assert scores["red_cube"].label_ok
    assert scores["mystery"].matched == "green_cube" and not scores["mystery"].label_ok


def test_random_layout_is_deterministic_and_spaced(robot):
    a, b = random_layout(7, robot), random_layout(7, robot)
    assert a == b and a != random_layout(8, robot)
    pts = list(a.values())
    assert min(np.hypot(p[0] - q[0], p[1] - q[1]) for i, p in enumerate(pts) for q in pts[i + 1:]) >= 0.09
    assert all(robot.workspace.contains_xy(p[0], p[1]) for p in pts)
    assert all(-np.pi / 4 <= p[2] <= np.pi / 4 for p in pts)
    assert any(abs(p[2]) > 0.05 for p in pts)          # actually rotated


def test_rotated_cubes_round_trip_through_the_simulator(robot):
    layout = random_layout(3, robot)
    with PandaSim(robot) as sim:
        sim.reset(layout)
        truth = sim.ground_truth()
    for name, (x, y, yaw) in layout.items():
        o = truth.objects[name]
        assert np.allclose(o.pos[:2], [x, y], atol=0.003)
        assert abs(o.yaw - yaw) < np.radians(1.0), name


def test_wrap_square():
    from openrobot_brain.world import wrap_square
    for given, expected in ((0, 0), (30, 30), (50, -40), (-46, 44), (90, 0), (135, -45)):
        assert np.isclose(np.degrees(wrap_square(np.radians(given))), expected)