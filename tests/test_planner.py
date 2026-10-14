import json
from types import SimpleNamespace

import pytest

from openrobot_brain.planner import (
    DEFAULT_MODEL, GeminiPlanner, MissingCredentialsError, OfflinePlanner, PlannerError, plan, redact,
)
from openrobot_brain.schema import PlanParseError
from conftest import move, plan_json


class FakeInteractions:
    def __init__(self, text=None, exc=None):
        self.text, self.exc, self.calls = text, exc, []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.exc:
            raise self.exc
        return SimpleNamespace(output_text=self.text)


def fake_client(**kw):
    return SimpleNamespace(interactions=FakeInteractions(**kw))


def test_missing_credentials(robot):
    with pytest.raises(MissingCredentialsError, match="GEMINI_API_KEY"):
        GeminiPlanner(robot)


def test_default_model_is_argon_and_not_substituted(robot, monkeypatch):
    assert GeminiPlanner(robot, client=fake_client(text="{}")).model == DEFAULT_MODEL == "gemini-4-argon"
    monkeypatch.setenv("GEMINI_MODEL", "my-explicit-choice")
    assert GeminiPlanner(robot, client=fake_client(text="{}")).model == "my-explicit-choice"


def test_request_uses_interactions_api_with_structured_output(robot):
    client = fake_client(text=plan_json(move(1.0, shoulder_yaw=0.2)))
    p = GeminiPlanner(robot, client=client, thinking_level="medium")
    result = plan(p, "turn a little left")
    call = client.interactions.calls[0]
    assert call["model"] == "gemini-4-argon"
    assert call["input"] == "turn a little left"
    assert call["response_format"]["mime_type"] == "application/json"
    assert call["response_format"]["schema"]["additionalProperties"] is False
    assert call["generation_config"] == {"thinking_level": "medium"}
    assert call["store"] is False
    assert "temperature" not in call and "temperature" not in call["generation_config"]
    assert "shoulder_yaw" in call["system_instruction"]
    assert result.actions[0].targets[0].position_rad == 0.2


def test_malformed_model_output_raises_parse_error(robot):
    p = GeminiPlanner(robot, client=fake_client(text="Sure! Here's your plan: {"))
    with pytest.raises(PlanParseError):
        plan(p, "wave")


def test_empty_model_output(robot):
    p = GeminiPlanner(robot, client=fake_client(text=""))
    with pytest.raises(PlannerError, match="no text"):
        p.plan_text("wave")


def test_api_errors_are_wrapped_and_key_redacted(robot):
    key = "AIzaSySECRETsecretSECRETsecret0000"
    p = GeminiPlanner(robot, api_key=key,
                      client=fake_client(exc=RuntimeError(f"404 model not found (key={key})")))
    with pytest.raises(PlannerError) as exc:
        p.plan_text("wave")
    assert key not in str(exc.value)
    assert "not substituted" in str(exc.value)


@pytest.mark.parametrize("err,expected", [
    ("429 RESOURCE_EXHAUSTED. quota_metric: generate_content_free_tier_requests, limit: 0", "zero quota"),
    ("429 Too Many Requests", "free tier"),
])
def test_quota_errors_explain_billing(robot, err, expected):
    p = GeminiPlanner(robot, client=fake_client(exc=RuntimeError(err)))
    with pytest.raises(PlannerError, match=expected):
        p.plan_text("wave")


def test_real_client_is_built_with_limited_retries(robot):
    p = GeminiPlanner(robot, api_key="test-key-not-real-123456", max_retries=1)
    assert p._client is not None  # constructing must not touch the network


def test_redact_masks_key_shaped_strings():
    assert "AIza" not in redact("bad key AIzaSyABCDEFGHIJKLMNOPQRSTUVWXYZ012", [])


@pytest.mark.parametrize("task", ["", "   ", "x" * 501])
def test_task_length_checked(robot, task):
    with pytest.raises(PlannerError):
        GeminiPlanner(robot, client=fake_client(text="{}")).plan_text(task)


def test_invalid_thinking_level(robot):
    with pytest.raises(PlannerError):
        GeminiPlanner(robot, client=fake_client(text="{}"), thinking_level="max")


def test_offline_planner_matches_keywords():
    data = json.loads(OfflinePlanner().plan_text("Could you WAVE at me?"))
    assert data["actions"]
    with pytest.raises(PlannerError, match="demo tasks"):
        OfflinePlanner().plan_text("make me coffee")


@pytest.mark.live
def test_live_gemini_roundtrip(robot):  # pragma: no cover - needs a paid key
    import os
    if not (os.environ.get("ORB_LIVE_KEY")):
        pytest.skip("set ORB_LIVE_KEY to a Gemini API key to run the live test")
    p = GeminiPlanner(robot, api_key=os.environ["ORB_LIVE_KEY"])
    assert plan(p, "turn slightly to the left, then return home").actions
