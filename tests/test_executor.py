import mujoco
import numpy as np
import pytest

from openrobot_brain.safety import validate_round
from openrobot_brain.schema import parse_agent_json
from openrobot_brain.simulation import CUBE_BODIES, DEFAULT_SCENE
from openrobot_brain.simulation.executor import PandaSim, SimulationError, smoke_check
from openrobot_brain.simulation.kinematics import SceneError, load_model
from conftest import pick, place_on, respond


def plan(sim, robot, actions, world=None):
    r = parse_agent_json(respond("act", actions))
    return validate_round(list(r.actions), robot, sim.kin, sim.q(), world or sim.ground_truth(), sim.gripper_open())


def test_scene_matches_config(robot):
    m = load_model(DEFAULT_SCENE)
    for j, (lo, hi) in robot.joint_limits.items():
        jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j)
        assert m.jnt_range[jid][0] <= lo < hi <= m.jnt_range[jid][1]
    for name in CUBE_BODIES:
        assert mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, name) >= 0
    assert mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "tcp") >= 0
    assert mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, "overhead") >= 0
    assert m.nu == 8


def test_missing_scene():
    with pytest.raises(SceneError, match="not found"):
        load_model("nope.xml")


def test_smoke_check_stacks_a_cube(robot):
    report = smoke_check(robot)
    assert report["pick_and_place_executed"] and report["red_on_blue"]


def test_physical_pick_and_place(robot):
    with PandaSim(robot) as sim:
        res = sim.execute(plan(sim, robot, [pick("green_cube"), {"action": "place_at", "x": 0.4, "y": 0.0}]))
        assert res.completed, res.message
        assert [o.ok for o in res.outcomes] == [True, True]
        assert "holding an object" in res.outcomes[0].detail
        g = sim.ground_truth().objects["green_cube"].pos
        assert np.allclose(g[:2], [0.4, 0.0], atol=0.01) and abs(g[2] - 0.02) < 0.004
        assert np.max(np.abs(sim.q() - robot.rest_pose)) < 0.05  # parked for the next photo


def test_missed_grasp_is_detected_and_stops_the_round(robot):
    """The belief says red is 10 cm from where it really is: validation passes
    (it checks the belief), physics shows an empty gripper, the round stops."""
    with PandaSim(robot) as sim:
        belief = sim.ground_truth()
        belief.objects["red_cube"].pos = belief.objects["red_cube"].pos + [0, -0.10, 0]
        res = sim.execute(plan(sim, robot, [pick("red_cube"), place_on("blue_cube")], belief))
        assert not res.completed
        assert "grasp missed" in res.message
        assert res.outcomes[0].ok is False and "closed on nothing" in res.outcomes[0].detail
        assert np.allclose(sim.ground_truth().objects["red_cube"].pos[:2], [0.45, 0.15], atol=0.01)


def test_executor_rejects_unvalidated_plans(robot):
    with PandaSim(robot) as sim:
        with pytest.raises(TypeError):
            sim.execute(parse_agent_json(respond("act", [pick("red_cube")])))


def test_executor_rejects_reused_or_stale_plans(robot):
    with PandaSim(robot) as sim:
        vp = plan(sim, robot, [pick("red_cube")])
        stale = plan(sim, robot, [pick("blue_cube")])     # validated with an open, empty gripper
        sim.execute(vp)                                    # now holding red, gripper closed
        with pytest.raises(SimulationError, match="already executed"):
            sim.execute(vp)
        with pytest.raises(SimulationError, match="gripper state"):
            sim.execute(stale)


def test_viewer_failure_falls_back_to_headless(robot, monkeypatch):
    import mujoco.viewer

    def boom(*a, **k):
        raise RuntimeError("no display")
    monkeypatch.setattr(mujoco.viewer, "launch_passive", boom)
    with PandaSim(robot, headless=False) as sim:
        assert "no display" in sim.viewer_error and not sim.has_viewer
        sim.hold()  # returns immediately without a viewer


def test_speed_bounds(robot):
    with pytest.raises(ValueError):
        PandaSim(robot, speed=0)


def test_record_gif(robot, tmp_path):
    from openrobot_brain.simulation.executor import VisionUnavailable  # noqa: F401
    from openrobot_brain.simulation.recorder import RecorderError
    with PandaSim(robot, speed=4) as sim:
        try:
            sim.start_recording(tmp_path / "run.gif")
        except RecorderError as exc:
            pytest.skip(str(exc))
        sim.annotate(title="test", subtitle="pick", badge=("VALIDATED", (30, 150, 70)))
        sim.record_still(0.2)
        sim.execute(plan(sim, robot, [pick("red_cube")]))
        path = sim.stop_recording()
    assert path and path.read_bytes()[:3] == b"GIF"


def test_record_rejects_bad_extension(robot, tmp_path):
    from openrobot_brain.simulation.recorder import RecorderError
    with PandaSim(robot) as sim:
        with pytest.raises(RecorderError, match=".mp4 or .gif"):
            sim.start_recording(tmp_path / "run.avi")


def _turn(sim, name, deg):
    a = sim.kin.cube_adr[name]
    yaw = np.radians(deg)
    sim.data.qpos[a + 3:a + 7] = (np.cos(yaw / 2), 0, 0, np.sin(yaw / 2))
    mujoco.mj_forward(sim.model, sim.data)


def test_rotated_cube_is_gripped_on_its_faces(robot):
    with PandaSim(robot) as sim:
        _turn(sim, "red_cube", 30)
        res = sim.execute(plan(sim, robot, [pick("red_cube"), place_on("blue_cube")]))
        assert res.completed, res.message
        assert "closed at 4.0 cm" in res.outcomes[0].detail
        gt = sim.ground_truth().objects
        assert abs(np.degrees(gt["red_cube"].yaw - gt["blue_cube"].yaw)) < 1.5   # stacked square on


def test_misaligned_grasp_is_detected(robot):
    """The belief says the cube is straight, but it is really turned 30 degrees: the plan is valid
    against the belief, the fingers land on the corners, and the executor reports it."""
    with PandaSim(robot) as sim:
        belief = sim.ground_truth()
        _turn(sim, "red_cube", 30)
        res = sim.execute(plan(sim, robot, [pick("red_cube"), place_on("blue_cube")], belief))
        assert not res.completed
        assert "grasp misaligned" in res.message
        assert "gripping the corners" in res.outcomes[0].detail