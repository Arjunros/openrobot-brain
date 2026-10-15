import json

import pytest

from openrobot_brain.schema import AgentResponse, PlanParseError, parse_agent_json, response_json_schema
from conftest import pick, place_at, respond


def test_valid_act_response():
    r = parse_agent_json(respond("act", [pick("red_cube"), {"action": "place_on", "object": "blue_cube"}],
                                 observations=[{"id": "red_cube", "label": "red cube", "u": 231.0, "v": 299,
                                                "on_top_of": ""}]))
    assert r.status == "act" and len(r.actions) == 2 and r.observations[0].u == 231.0


@pytest.mark.parametrize("status", ["done", "infeasible"])
def test_terminal_status_without_actions(status):
    assert parse_agent_json(respond(status)).status == status


@pytest.mark.parametrize("text,match", [
    (respond("act", []), "requires at least one action"),
    (respond("done", [pick("red_cube")]), "must not include actions"),
    (respond("infeasible", [pick("red_cube")]), "must not include actions"),
    (respond("act", [pick("red_cube")], observations=[
        {"id": "a", "label": "x", "u": 1, "v": 1, "on_top_of": ""},
        {"id": "a", "label": "y", "u": 2, "v": 2, "on_top_of": ""}]), "unique"),
])
def test_status_rules(text, match):
    with pytest.raises(PlanParseError, match=match):
        parse_agent_json(text)


@pytest.mark.parametrize("text", ["", "  ", "nope", "[1,2]", "null", '{"status": "act"', "```json\n{}\n```"])
def test_malformed_json(text):
    with pytest.raises(PlanParseError):
        parse_agent_json(text)


@pytest.mark.parametrize("action", [
    {"action": "grasp", "object": "red_cube"},
    {"action": "run_python", "code": "import os"},
    {"action": "move_joints", "targets": []},
    {"action": "pick"},
    {"action": "pick", "object": "Red Cube"},           # bad id format
    {"action": "pick", "object": "x" * 30},
    {"action": "place_at", "x": "0.5", "y": 0.0},       # string number
    {"action": "move_to", "x": 0.5, "y": 0.0, "z": True},
])
def test_bad_actions(action):
    with pytest.raises(PlanParseError):
        parse_agent_json(respond("act", [action]))


@pytest.mark.parametrize("payload", [
    {"status": "act", "message": "", "observations": [], "actions": [{**pick("red_cube"), "force": 99}]},
    {"status": "act", "message": "", "observations": [], "actions": [pick("red_cube")], "safety_ok": True},
    {"status": "act", "message": "", "observations": [{"id": "a", "label": "x", "u": 1, "v": 1,
                                                        "on_top_of": "", "x_m": 0.4}], "actions": [pick("a")]},
])
def test_extra_fields_rejected(payload):
    with pytest.raises(PlanParseError, match="(?i)extra"):
        parse_agent_json(json.dumps(payload))


@pytest.mark.parametrize("bad", ["NaN", "Infinity", "-Infinity", "1e999"])
def test_non_finite_rejected(bad):
    text = '{"status":"act","message":"","observations":[],"actions":[{"action":"place_at","x":%s,"y":0}]}' % bad
    with pytest.raises(PlanParseError, match="non-finite"):
        parse_agent_json(text)


def test_too_many_actions_in_schema():
    with pytest.raises(PlanParseError):
        parse_agent_json(respond("act", [place_at(0.5, 0)] * 17))


def test_strict_schema_is_closed_and_fully_required():
    def walk(node):
        if isinstance(node, dict):
            if node.get("type") == "object":
                assert node["additionalProperties"] is False
                assert set(node["required"]) == set(node["properties"])
            assert "minItems" not in node and "maxItems" not in node
            for v in node.values():
                walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)
    schema = response_json_schema()
    walk(schema)
    names = {a["properties"]["action"]["enum"][0] for a in schema["properties"]["actions"]["items"]["anyOf"]}
    assert names == {"pick", "place_on", "place_at", "place_in", "move_to", "go_home", "wait"}


def test_schema_has_no_code_or_safety_override_fields():
    flat = json.dumps(response_json_schema())
    for word in ("code", "command", "safe", "override", "joint"):
        assert f'"{word}"' not in flat
    assert AgentResponse.model_config["extra"] == "forbid"


def test_label_length_limit_matches_instructions():
    from openrobot_brain.config import PANDA
    from openrobot_brain.planner import system_instructions
    long_ok = "red cube partially hidden under the green one"   # 45 chars, seen from a real Astra run
    parse_agent_json(respond("done", observations=[{"id": "red_cube", "label": long_ok, "u": 1, "v": 1,
                                                    "on_top_of": ""}]))
    with pytest.raises(PlanParseError):
        parse_agent_json(respond("done", observations=[{"id": "red_cube", "label": "x" * 81, "u": 1, "v": 1,
                                                        "on_top_of": ""}]))
    text = system_instructions(PANDA, True, 640, 480)
    assert "at most 80 characters" in text and "max 24 chars" in text
