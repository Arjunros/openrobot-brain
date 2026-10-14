from dataclasses import replace
from types import MappingProxyType

import mujoco
import pytest

from openrobot_brain.config import JointLimit
from openrobot_brain.safety import validate_plan
from openrobot_brain.schema import parse_plan_json
from openrobot_brain.simulation import DEFAULT_SCENE
from openrobot_brain.simulation.executor import (
    MujocoExecutor, SimulationError, bind_joints, load_model, smoke_check,
)
from conftest import move, plan_json


def test_scene_loads():
    model = load_model(DEFAULT_SCENE)
    assert model.nq == 4 and model.nu == 4


def test_actuators_bound_to_matching_joints(robot):
    model = load_model(DEFAULT_SCENE)
    b = bind_joints(model, robot)
    assert set(b) == set(robot.joint_names)
    for name, binding in b.items():
        act_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, binding.actuator_id)
        assert act_name == f"act_{name}"


def test_software_limits_inside_scene_limits(robot):
    b = bind_joints(load_model(DEFAULT_SCENE), robot)
    for name, lim in robot.joint_limits.items():
        lo, hi = b[name].hard_range
        assert lo <= lim.lower < lim.upper <= hi


def test_bind_rejects_mismatched_config(robot):
    limits = dict(robot.joint_limits)
    limits["shoulder_yaw"] = JointLimit(-3.0, 3.0)  # wider than the scene allows
    bad = replace(robot, joint_limits=MappingProxyType(limits))
    with pytest.raises(SimulationError, match="exceeds scene range"):
        bind_joints(load_model(DEFAULT_SCENE), bad)


def test_bind_rejects_unknown_joint(robot):
    limits = {**robot.joint_limits, "gripper": JointLimit(0, 0.1)}
    bad = replace(robot, joint_limits=MappingProxyType(limits), home_pose={**robot.home_pose, "gripper": 0.0})
    with pytest.raises(SimulationError, match="gripper"):
        bind_joints(load_model(DEFAULT_SCENE), bad)


def test_missing_scene_file():
    with pytest.raises(SimulationError, match="not found"):
        load_model("does_not_exist.xml")


def test_arm_tracks_position_commands(robot):
    vp = validate_plan(parse_plan_json(plan_json(
        move(1.0, shoulder_yaw=0.6, elbow_pitch=-0.5), {"action": "wait", "duration_s": 0.5})), robot)
    with MujocoExecutor(robot, headless=True) as ex:
        res = ex.execute(vp)
        q = ex.joint_state()
    assert res.success, res.message
    assert abs(q["shoulder_yaw"] - 0.6) < 0.02
    assert abs(q["elbow_pitch"] + 0.5) < 0.02
    assert res.used_viewer is False


def test_executor_rejects_unvalidated_plan(robot):
    raw = parse_plan_json(plan_json(move(1.0, shoulder_yaw=0.1)))
    with MujocoExecutor(robot, headless=True) as ex:
        with pytest.raises(TypeError):
            ex.execute(raw)


def test_executor_rejects_stale_start_state(robot):
    vp = validate_plan(parse_plan_json(plan_json(move(1.0, shoulder_yaw=0.5))), robot)
    with MujocoExecutor(robot, headless=True) as ex:
        ex.execute(vp)  # arm now at 0.5; the same plan assumed it started at home
        with pytest.raises(SimulationError, match="re-validate"):
            ex.execute(vp)


def test_viewer_failure_falls_back_to_headless(robot, monkeypatch):
    import mujoco.viewer

    def boom(*a, **k):
        raise RuntimeError("no display")
    monkeypatch.setattr(mujoco.viewer, "launch_passive", boom)
    vp = validate_plan(parse_plan_json(plan_json(move(0.5, wrist_pitch=0.2))), robot)
    with MujocoExecutor(robot, headless=False) as ex:
        assert "no display" in ex.viewer_error
        assert ex.realtime is False
        assert ex.execute(vp).success


def test_smoke_check(robot):
    report = smoke_check(robot)
    assert report["responds_to_commands"] is True
    assert report["nu"] == 4
