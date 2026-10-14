import json

import pytest

from openrobot_brain.schema import ActionPlan, MoveJoints, Wait, parse_plan_json, PlanParseError
from conftest import move, plan_json


def test_valid_plan_parses():
    plan = parse_plan_json(plan_json(move(1.0, shoulder_yaw=0.3), {"action": "wait", "duration_s": 0.5}))
    assert isinstance(plan, ActionPlan)
    assert isinstance(plan.actions[0], MoveJoints)
    assert isinstance(plan.actions[1], Wait)


def test_integer_angles_are_accepted_as_numbers():
    plan = parse_plan_json(plan_json(move(1, shoulder_yaw=0)))
    assert plan.actions[0].targets[0].position_rad == 0


@pytest.mark.parametrize("text", [
    "", "   ", "not json", '{"actions": [', "```json\n{}\n```", "[1, 2]", '"just a string"', "null",
])
def test_malformed_json_rejected(text):
    with pytest.raises(PlanParseError):
        parse_plan_json(text)


@pytest.mark.parametrize("payload", [
    {"actions": []},  # empty plan
    {"summary": "x"},  # missing actions
    {"actions": [{"action": "move_joints", "duration_s": 1.0}]},  # missing targets
    {"actions": [{"action": "move_joints", "targets": [], "duration_s": 1.0}]},  # empty targets
    {"actions": [{"action": "wait"}]},  # missing duration
    {"actions": [move(1.0, shoulder_yaw="0.5")]},  # string number
    {"actions": [move(1.0, shoulder_yaw=True)]},  # bool
    {"actions": [{"action": "wait", "duration_s": None}]},
])
def test_schema_violations_rejected(payload):
    with pytest.raises(PlanParseError, match="schema violation"):
        parse_plan_json(json.dumps(payload))


@pytest.mark.parametrize("action", ["grasp", "run_python", "shell", "MOVE_JOINTS", ""])
def test_unknown_action_rejected(action):
    with pytest.raises(PlanParseError):
        parse_plan_json(json.dumps({"actions": [{"action": action, "duration_s": 1.0}]}))


@pytest.mark.parametrize("payload", [
    {"actions": [{**move(1.0, shoulder_yaw=0.1), "speed": 9}]},
    {"actions": [{"action": "wait", "duration_s": 1.0, "command": "rm -rf /"}]},
    {"actions": [move(1.0, shoulder_yaw=0.1)], "safety_verified": True},
    {"actions": [{"action": "move_joints", "duration_s": 1.0,
                  "targets": [{"joint": "shoulder_yaw", "position_rad": 0.1, "force": 100}]}]},
])
def test_extra_fields_rejected(payload):
    with pytest.raises(PlanParseError, match="(?i)extra"):
        parse_plan_json(json.dumps(payload))


@pytest.mark.parametrize("bad", ["NaN", "Infinity", "-Infinity", "1e999", "-1e999"])
def test_non_finite_numbers_rejected(bad):
    text = '{"actions": [{"action": "move_joints", "duration_s": 1.0, ' \
           '"targets": [{"joint": "shoulder_yaw", "position_rad": %s}]}]}' % bad
    with pytest.raises(PlanParseError, match="non-finite"):
        parse_plan_json(text)


def test_non_finite_rejected_even_when_constructed_in_python():
    with pytest.raises(Exception):
        ActionPlan.model_validate({"actions": [{"action": "wait", "duration_s": float("nan")}]})
