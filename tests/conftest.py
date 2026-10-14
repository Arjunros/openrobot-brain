import json
import os
import sys
from typing import Any

# Headless Linux CI: use OSMesa for offscreen rendering. (Windows/macOS use their defaults.)
if sys.platform.startswith("linux") and not os.environ.get("DISPLAY"):
    os.environ.setdefault("MUJOCO_GL", "osmesa")
    os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from openrobot_brain.config import PANDA  # noqa: E402
from openrobot_brain.simulation.kinematics import Kinematics, load_model  # noqa: E402
from openrobot_brain.world import WorldObject, WorldState  # noqa: E402


@pytest.fixture(autouse=True)
def no_real_keys(monkeypatch):
    """Unit tests must never see a real API key."""
    for var in ("OPENAI_API_KEY", "OPENAI_MODEL", "GEMINI_API_KEY", "GOOGLE_API_KEY", "ORB_PROVIDER"):
        monkeypatch.delenv(var, raising=False)


@pytest.fixture(scope="session")
def robot():
    return PANDA


@pytest.fixture(scope="session")
def kin():
    return Kinematics(load_model(), PANDA)


def make_world(**positions: tuple[float, float, float]) -> WorldState:
    if not positions:
        positions = {"red_cube": (0.45, 0.15, 0.02), "green_cube": (0.60, 0.0, 0.02),
                     "blue_cube": (0.45, -0.15, 0.02)}
    return WorldState({k: WorldObject(k, k.replace("_", " "), np.array(v, float)) for k, v in positions.items()})


def respond(status: str = "act", actions: list[dict[str, Any]] | None = None, message: str = "test",
            observations: list[dict[str, Any]] | None = None) -> str:
    return json.dumps({"status": status, "message": message, "observations": observations or [],
                       "actions": actions or []})


def pick(o: str) -> dict:
    return {"action": "pick", "object": o}


def place_on(o: str) -> dict:
    return {"action": "place_on", "object": o}


def place_at(x: float, y: float) -> dict:
    return {"action": "place_at", "x": x, "y": y}


def move_to(x: float, y: float, z: float) -> dict:
    return {"action": "move_to", "x": x, "y": y, "z": z}
