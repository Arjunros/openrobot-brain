import json
from typing import Any

import pytest

from openrobot_brain.config import DEFAULT_ROBOT


@pytest.fixture
def robot():
    return DEFAULT_ROBOT


@pytest.fixture(autouse=True)
def no_real_keys(monkeypatch):
    """Unit tests must never see a real API key."""
    for var in ("GEMINI_API_KEY", "GOOGLE_API_KEY", "GEMINI_MODEL", "OPENAI_API_KEY", "OPENAI_MODEL", "ORB_PROVIDER"):
        monkeypatch.delenv(var, raising=False)


def move(duration: float = 1.0, **targets: float) -> dict[str, Any]:
    return {"action": "move_joints", "duration_s": duration,
            "targets": [{"joint": j, "position_rad": v} for j, v in targets.items()]}


def plan_json(*actions: dict[str, Any], summary: str = "test") -> str:
    return json.dumps({"summary": summary, "actions": list(actions)})
