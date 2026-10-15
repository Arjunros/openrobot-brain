import json

import pytest

from openrobot_brain.planner import (
    DEFAULT_MODEL, AstraPlanner, MissingCredentialsError, OfflinePlanner, PlannerError, RoundInput, redact,
    system_instructions,
)
from conftest import make_world, respond
from fakes import ScriptedClient


def rin(prompt="TASK: x", image=b"\x89PNG fake", truth=None, task="stack the red cube on the blue cube"):
    return RoundInput(task, 1, 8, prompt, image, truth or make_world())


def test_missing_key(robot):
    with pytest.raises(MissingCredentialsError, match="OPENAI_API_KEY"):
        AstraPlanner(robot, vision=True)


def test_default_model_is_astra_and_env_override(robot, monkeypatch):
    assert AstraPlanner(robot, vision=True, client=ScriptedClient()).model == DEFAULT_MODEL == "gpt-6-astra"
    monkeypatch.setenv("OPENAI_MODEL", "explicit-choice")
    assert AstraPlanner(robot, vision=True, client=ScriptedClient()).model == "explicit-choice"


def test_vision_request_shape(robot):
    client = ScriptedClient(respond("done"))
    p = AstraPlanner(robot, vision=True, client=client, effort="medium")
    p.respond(rin())
    call = client.calls[0]
    assert call["model"] == "gpt-6-astra" and call["store"] is False
    assert call["reasoning"] == {"effort": "medium"}
    fmt = call["text"]["format"]
    assert fmt["type"] == "json_schema" and fmt["strict"] is True
    parts = call["input"][0]["content"]
    assert parts[0] == {"type": "input_text", "text": "TASK: x"}
    assert parts[1]["type"] == "input_image" and parts[1]["image_url"].startswith("data:image/png;base64,")
    assert parts[1]["detail"] == "high"
    for banned in ("temperature", "top_p", "tools", "tool_choice"):
        assert banned not in call
    assert "PERCEPTION (vision mode)" in call["instructions"] and "UP is +x" in call["instructions"]
    assert "yaw_deg" in call["instructions"] and "counter-clockwise" in call["instructions"]


def test_state_mode_sends_no_image(robot):
    client = ScriptedClient(respond("done"))
    AstraPlanner(robot, vision=False, client=client).respond(rin(image=None))
    parts = client.calls[0]["input"][0]["content"]
    assert len(parts) == 1 and "PERCEPTION (state mode)" in client.calls[0]["instructions"]


def test_vision_without_image_is_an_error(robot):
    with pytest.raises(PlannerError, match="image"):
        AstraPlanner(robot, vision=True, client=ScriptedClient()).respond(rin(image=None))


def test_instructions_contain_real_limits(robot):
    text = system_instructions(robot, True, 640, 480)
    ws = robot.workspace
    assert f"x {ws.x[0]}..{ws.x[1]}" in text and "640x480" in text
    assert "infeasible" in text and "never claim success" in text.lower()


@pytest.mark.parametrize("err,hint", [
    ("Error code: 429 - insufficient_quota", "add credit"),
    ("Error code: 429 - rate_limit_exceeded", "Rate limited"),
    ("APIConnectionError: Connection error.", "Could not reach"),
    ("Error code: 404 - model_not_found", "not substituted"),
    ("Error code: 401 - invalid_api_key", "rejected"),
])
def test_error_hints_and_key_redaction(robot, err, hint):
    key = "sk-proj-SECRETsecretSECRET0123456789"

    class Boom:
        class responses:
            @staticmethod
            def create(**kw):
                raise RuntimeError(f"{err} (key {key})")
    with pytest.raises(PlannerError) as exc:
        AstraPlanner(robot, vision=False, api_key=key, client=Boom()).respond(rin(image=None))
    assert hint in str(exc.value) and key not in str(exc.value)


def test_incomplete_and_empty(robot):
    from types import SimpleNamespace

    class C:
        def __init__(self, r):
            self.responses = SimpleNamespace(create=lambda **k: r)
    inc = SimpleNamespace(status="incomplete", incomplete_details=SimpleNamespace(reason="max_output_tokens"),
                          output_text="{")
    with pytest.raises(PlannerError, match="incomplete"):
        AstraPlanner(robot, vision=False, client=C(inc)).respond(rin(image=None))
    empty = SimpleNamespace(status="completed", incomplete_details=None, output_text="")
    with pytest.raises(PlannerError, match="no text"):
        AstraPlanner(robot, vision=False, client=C(empty)).respond(rin(image=None))


def test_invalid_effort(robot):
    with pytest.raises(PlannerError):
        AstraPlanner(robot, vision=True, client=ScriptedClient(), effort="minimal")


def test_real_client_constructs_offline(robot):
    assert AstraPlanner(robot, vision=True, api_key="sk-test-not-real-000000000000")._client is not None


def test_redact():
    assert "sk-" not in redact("boom sk-abcdefghijklmnopqrstuvwxyz", [])


@pytest.mark.parametrize("task,status,first", [
    ("stack the red cube on the blue cube", "act", "pick"),
    ("pick up the green cube", "act", "pick"),
    ("build a tower", "act", "pick"),
    ("unsafe demo", "act", "pick"),
    ("make coffee", "infeasible", None),
])
def test_offline_planner(task, status, first):
    out = json.loads(OfflinePlanner().respond(rin(task=task)))
    assert out["status"] == status
    if first:
        assert out["actions"][0]["action"] == first
