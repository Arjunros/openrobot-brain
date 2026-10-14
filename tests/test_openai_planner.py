import json
from types import SimpleNamespace

import pytest

from openrobot_brain import cli
from openrobot_brain.planner import (
    OPENAI_DEFAULT_MODEL, GeminiPlanner, MissingCredentialsError, OpenAIPlanner, PlannerError,
    default_provider, make_planner, plan,
)
from openrobot_brain.schema import PlanParseError, openai_response_schema
from conftest import move, plan_json


class FakeResponses:
    def __init__(self, text=None, exc=None, status="completed"):
        self.text, self.exc, self.status, self.calls = text, exc, status, []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.exc:
            raise self.exc
        return SimpleNamespace(output_text=self.text, status=self.status,
                               incomplete_details=SimpleNamespace(reason="max_output_tokens"))


def fake_client(**kw):
    return SimpleNamespace(responses=FakeResponses(**kw))


def test_missing_openai_key(robot):
    with pytest.raises(MissingCredentialsError, match="OPENAI_API_KEY"):
        OpenAIPlanner(robot)


def test_default_model_is_astra_and_not_substituted(robot, monkeypatch):
    assert OpenAIPlanner(robot, client=fake_client(text="{}")).model == OPENAI_DEFAULT_MODEL == "gpt-6-astra"
    monkeypatch.setenv("OPENAI_MODEL", "explicit-choice")
    assert OpenAIPlanner(robot, client=fake_client(text="{}")).model == "explicit-choice"


def test_request_uses_responses_api_strict_structured_output(robot):
    client = fake_client(text=plan_json(move(1.0, elbow_pitch=0.3)))
    p = OpenAIPlanner(robot, client=client, thinking_level="xhigh")
    result = plan(p, "bend the elbow a bit")
    call = client.responses.calls[0]
    assert call["model"] == "gpt-6-astra"
    assert call["input"] == "bend the elbow a bit"
    fmt = call["text"]["format"]
    assert fmt["type"] == "json_schema" and fmt["strict"] is True and fmt["name"] == "action_plan"
    assert call["reasoning"] == {"effort": "xhigh"}
    assert call["store"] is False
    for banned in ("temperature", "top_p", "logprobs", "top_logprobs"):
        assert banned not in call
    assert "elbow_pitch" in call["instructions"]
    assert result.actions[0].targets[0].joint == "elbow_pitch"


def test_strict_schema_has_no_array_size_keywords_and_is_closed(robot):
    s = json.dumps(openai_response_schema(robot.joint_names))
    assert "minItems" not in s and "maxItems" not in s
    assert '"additionalProperties": true' not in s


def test_model_output_is_still_untrusted(robot):
    # Even with strict structured outputs, a bad plan must fail locally.
    p = OpenAIPlanner(robot, client=fake_client(text='{"summary":"x","actions":[]}'))
    with pytest.raises(PlanParseError):
        plan(p, "wave")


def test_incomplete_response(robot):
    p = OpenAIPlanner(robot, client=fake_client(text="{", status="incomplete"))
    with pytest.raises(PlannerError, match="incomplete"):
        p.plan_text("wave")


def test_insufficient_quota_explains_billing_and_redacts_key(robot):
    key = "sk-proj-SECRETsecretSECRETsecret1234"
    p = OpenAIPlanner(robot, api_key=key, client=fake_client(
        exc=RuntimeError(f"Error code: 429 - insufficient_quota (key {key})")))
    with pytest.raises(PlannerError) as exc:
        p.plan_text("wave")
    assert key not in str(exc.value)
    assert "platform.openai.com" in str(exc.value)


def test_invalid_effort(robot):
    with pytest.raises(PlannerError):
        OpenAIPlanner(robot, client=fake_client(text="{}"), thinking_level="minimal")


def test_gemini_rejects_openai_only_effort(robot):
    with pytest.raises(PlannerError):
        GeminiPlanner(robot, client=SimpleNamespace(), thinking_level="xhigh")


def test_real_openai_client_builds_without_network(robot):
    assert OpenAIPlanner(robot, api_key="sk-test-not-real-0000000000")._client is not None


def test_provider_selection(monkeypatch):
    assert default_provider() == "gemini"
    monkeypatch.setenv("OPENAI_API_KEY", "sk-x")
    assert default_provider() == "openai"  # only an OpenAI key present
    monkeypatch.setenv("GEMINI_API_KEY", "g-x")
    assert default_provider() == "gemini"  # both present: Gemini unless told otherwise
    monkeypatch.setenv("ORB_PROVIDER", "openai")
    assert default_provider() == "openai"
    monkeypatch.setenv("ORB_PROVIDER", "claude")
    with pytest.raises(PlannerError):
        default_provider()


def test_make_planner_unknown(robot):
    with pytest.raises(PlannerError):
        make_planner("other", robot)


def test_cli_openai_without_key(capsys, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    assert cli.main(["plan", "wave", "--provider", "openai"]) == cli.EXIT_PLANNER
    assert "OPENAI_API_KEY" in capsys.readouterr().out


def test_cli_openai_end_to_end_with_mock(capsys, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    client = fake_client(text=plan_json(move(1.0, shoulder_yaw=0.4)))
    monkeypatch.setattr(cli, "make_planner",
                        lambda provider, robot, model=None, thinking_level="low":
                        OpenAIPlanner(robot, client=client, thinking_level=thinking_level))
    assert cli.main(["plan", "turn left", "--provider", "openai"]) == cli.EXIT_OK
    out = capsys.readouterr().out
    assert "OpenAI model gpt-6-astra" in out and "Dry run" in out


@pytest.mark.live
def test_live_openai_roundtrip(robot):  # pragma: no cover - needs a funded key
    import os
    if not os.environ.get("ORB_LIVE_OPENAI_KEY"):
        pytest.skip("set ORB_LIVE_OPENAI_KEY to run the live OpenAI test")
    p = OpenAIPlanner(robot, api_key=os.environ["ORB_LIVE_OPENAI_KEY"])
    assert plan(p, "turn slightly to the left, then return home").actions
