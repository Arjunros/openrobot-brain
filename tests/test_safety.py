import json
from dataclasses import replace

import pytest

from openrobot_brain.config import SafetyLimits
from openrobot_brain.planner import OFFLINE_PLANS
from openrobot_brain.safety import SafetyError, ValidatedPlan, validate_plan
from openrobot_brain.schema import parse_plan_json
from conftest import move, plan_json


def check(robot, *actions, initial=None):
    return validate_plan(parse_plan_json(plan_json(*actions)), robot, initial)


def test_valid_plan_tracks_predicted_state(robot):
    vp = check(robot, move(1.0, shoulder_yaw=0.5), move(1.0, shoulder_yaw=1.0), {"action": "wait", "duration_s": 1})
    assert isinstance(vp, ValidatedPlan)
    assert vp.steps[1].start_state["shoulder_yaw"] == 0.5
    assert vp.final_state["shoulder_yaw"] == 1.0
    assert vp.final_state["elbow_pitch"] == 0.0  # untouched joints hold
    assert vp.total_duration_s == 3.0


def test_unknown_joint_rejected(robot):
    with pytest.raises(SafetyError, match="unknown joint 'gripper'"):
        check(robot, move(1.0, gripper=0.1))


def test_duplicate_joint_in_one_move_rejected(robot):
    with pytest.raises(SafetyError, match="more than once"):
        check(robot, {"action": "move_joints", "duration_s": 1.0, "targets": [
            {"joint": "shoulder_yaw", "position_rad": 0.1}, {"joint": "shoulder_yaw", "position_rad": 0.2}]})


@pytest.mark.parametrize("joint,value", [("shoulder_yaw", 1.5), ("shoulder_pitch", -1.0), ("wrist_pitch", 1.3)])
def test_joint_limit_violation(robot, joint, value):
    initial = dict(robot.home_pose)
    # start close to the limit so only the limit check (not the delta check) fires
    initial[joint] = max(min(value, robot.joint_limits[joint].upper), robot.joint_limits[joint].lower)
    with pytest.raises(SafetyError, match="outside"):
        check(robot, move(1.0, **{joint: value}), initial=initial)


def test_excessive_single_move(robot):
    with pytest.raises(SafetyError, match="limit is 0.8 per action"):
        check(robot, move(3.0, shoulder_yaw=1.0))


def test_excessive_velocity(robot):
    with pytest.raises(SafetyError, match="speed"):
        check(robot, move(0.2, shoulder_yaw=0.5))


def test_unsafe_sequence_each_step_fine_from_home_but_not_in_sequence(robot):
    """Every move is within 0.8 rad of HOME, but the third is 1.4 rad from where
    the arm will actually be. A validator that compared against the initial
    pose would wrongly accept this."""
    actions = [move(1.0, shoulder_yaw=0.7), move(1.0, shoulder_yaw=0.7), move(2.0, shoulder_yaw=-0.7)]
    for a in actions:  # sanity: each passes on its own
        check(robot, a)
    with pytest.raises(SafetyError, match=r"action\[2\].*predicted 0.700"):
        check(robot, *actions)


def test_sequence_can_drift_into_limit(robot):
    with pytest.raises(SafetyError, match=r"action\[2\].*outside"):
        check(robot, move(1.0, shoulder_yaw=0.7), move(1.0, shoulder_yaw=1.3), move(1.0, shoulder_yaw=1.5))


@pytest.mark.parametrize("action,match", [
    ({"action": "wait", "duration_s": 0.0}, "wait"),
    ({"action": "wait", "duration_s": -1.0}, "wait"),
    ({"action": "wait", "duration_s": 10.0}, "wait"),
    (move(0.05, shoulder_yaw=0.01), "duration"),
    (move(60.0, shoulder_yaw=0.01), "duration"),
])
def test_duration_bounds(robot, action, match):
    with pytest.raises(SafetyError, match=match):
        check(robot, action)


def test_plan_length_limit(robot):
    with pytest.raises(SafetyError, match="limit is 12"):
        check(robot, *[{"action": "wait", "duration_s": 0.1}] * 13)


def test_total_duration_limit(robot):
    tight = replace(robot, safety=SafetyLimits(max_total_duration_s=2.0))
    with pytest.raises(SafetyError, match="total duration"):
        check(tight, *[{"action": "wait", "duration_s": 1.0}] * 3)


def test_all_violations_reported_together(robot):
    with pytest.raises(SafetyError) as exc:
        check(robot, move(1.0, gripper=1.0), {"action": "wait", "duration_s": 99})
    assert len(exc.value.violations) >= 2  # unknown joint, long wait (+ total duration)


def test_model_safety_claims_are_ignored(robot):
    with pytest.raises(SafetyError):
        check(robot, move(0.2, shoulder_yaw=0.79))  # summary/claims can't help; nothing in schema to claim with


def test_validated_plan_cannot_be_forged(robot):
    plan = parse_plan_json(plan_json(move(1.0, shoulder_yaw=0.1)))
    with pytest.raises(TypeError):
        ValidatedPlan(object(), plan, robot, robot.home_pose, ())


def test_bad_initial_state_rejected(robot):
    with pytest.raises(SafetyError):
        check(robot, {"action": "wait", "duration_s": 1}, initial={"shoulder_yaw": 0.0})


@pytest.mark.parametrize("name", [k for k in OFFLINE_PLANS if k != "unsafe demo"])
def test_offline_demo_plans_are_safe(robot, name):
    check(robot, *OFFLINE_PLANS[name]["actions"])


def test_offline_unsafe_demo_is_rejected(robot):
    with pytest.raises(SafetyError):
        check(robot, *OFFLINE_PLANS["unsafe demo"]["actions"])
