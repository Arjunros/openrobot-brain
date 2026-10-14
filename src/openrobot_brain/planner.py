"""Planners: Gemini 4 Argon (runtime AI) and an offline, rule-based demo planner.

Both return raw JSON *text*. Turning that text into a plan and deciding
whether it is safe happens elsewhere (``schema.parse_plan_json`` and
``safety.validate_plan``), so the model output is always treated as untrusted.

Gemini integration notes (verified 2026-10-14 against google-genai 2.30.1
and ai.google.dev docs):
  * ``gemini-4-argon`` is served through the Interactions API
    (``client.interactions.create``), paid tier only.
  * Structured output uses ``response_format={"type": "text",
    "mime_type": "application/json", "schema": ...}``; the text is read from
    ``interaction.output_text``.
  * Thinking is set with ``generation_config={"thinking_level": ...}``
    (low | medium | high). temperature/top_p/top_k are ignored by Argon, so
    they are not sent.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Protocol

from .config import RobotConfig
from .schema import ActionPlan, gemini_response_schema, parse_plan_json

DEFAULT_MODEL = "gemini-4-argon"
THINKING_LEVELS = ("low", "medium", "high")
MAX_TASK_CHARS = 500
KEY_ENV_VARS = ("GEMINI_API_KEY", "GOOGLE_API_KEY")


class PlannerError(RuntimeError):
    pass


class MissingCredentialsError(PlannerError):
    pass


class Planner(Protocol):
    name: str

    def plan_text(self, task: str) -> str: ...


def _check_task(task: str) -> str:
    task = (task or "").strip()
    if not task:
        raise PlannerError("task is empty")
    if len(task) > MAX_TASK_CHARS:
        raise PlannerError(f"task is longer than {MAX_TASK_CHARS} characters")
    return task


def redact(text: str, secrets: list[str | None]) -> str:
    for s in secrets:
        if s and len(s) >= 6:
            text = text.replace(s, "***")
    # Also mask anything shaped like a Google API key that slipped through.
    return re.sub(r"AIza[0-9A-Za-z_\-]{20,}", "***", text)


def build_system_instruction(robot: RobotConfig) -> str:
    lim = robot.safety
    joints = "\n".join(
        f"  - {n}: [{l.lower}, {l.upper}] rad (home {robot.home_pose[n]})"
        for n, l in robot.joint_limits.items()
    )
    return f"""You are the motion planner for a SIMULATED 4-joint robot arm ({robot.name}) in MuJoCo.
Convert the user's instruction into a short joint-space action plan.

Robot joints (name: allowed range, home position):
{joints}

Conventions: shoulder_yaw rotates the arm left (+) / right (-) about the vertical axis.
shoulder_pitch, elbow_pitch and wrist_pitch tilt forward (+) / backward (-).
The arm has no gripper, no camera and no inverse kinematics. It can only move joints.
The arm starts at the home position unless told otherwise.

Allowed actions (nothing else):
  - move_joints: targets = list of {{joint, position_rad}}, duration_s
  - wait: duration_s

Hard limits that a separate validator will enforce (plans that break any are rejected):
  - at most {lim.max_actions} actions
  - each joint may change by at most {lim.max_step_delta_rad} rad per move_joints, measured from its position after the previous action
  - joint speed (change / duration_s) at most {lim.max_joint_velocity_rad_s} rad/s
  - move_joints duration_s between {lim.min_move_duration_s} and {lim.max_move_duration_s}
  - wait duration_s greater than 0 and at most {lim.max_wait_s}
  - total duration at most {lim.max_total_duration_s} s
Split large motions into several smaller moves. Joints not listed in a move keep their position.

If the instruction needs a capability this arm does not have (grasping, seeing, walking, speech,
anything outside the simulator), return a single wait of 0.5 s and explain in `summary` that it
cannot be done. Never claim the arm did something it cannot do.
Return only the JSON object."""


class GeminiPlanner:
    name = "gemini"

    def __init__(self, robot: RobotConfig, model: str | None = None, api_key: str | None = None,
                 thinking_level: str = "low", client: Any | None = None, timeout_s: float = 120.0):
        if thinking_level not in THINKING_LEVELS:
            raise PlannerError(f"thinking_level must be one of {THINKING_LEVELS}")
        self.robot = robot
        self.model = model or os.environ.get("GEMINI_MODEL") or DEFAULT_MODEL
        self.thinking_level = thinking_level
        self.timeout_s = timeout_s
        self._api_key = api_key or next((os.environ[v] for v in KEY_ENV_VARS if os.environ.get(v)), None)

        if client is not None:
            self._client = client
        else:
            if not self._api_key:
                raise MissingCredentialsError(
                    "no Gemini API key found. Set GEMINI_API_KEY in your environment or .env file, "
                    "or run with --offline for the rule-based demo planner."
                )
            from google import genai  # imported lazily so offline mode needs no SDK init

            self._client = genai.Client(api_key=self._api_key)

    def request_kwargs(self, task: str) -> dict[str, Any]:
        return {
            "model": self.model,
            "system_instruction": build_system_instruction(self.robot),
            "input": task,
            "response_format": {
                "type": "text",
                "mime_type": "application/json",
                "schema": gemini_response_schema(self.robot.joint_names),
            },
            "generation_config": {"thinking_level": self.thinking_level},
            "store": False,
        }

    def plan_text(self, task: str) -> str:
        task = _check_task(task)
        try:
            interaction = self._client.interactions.create(**self.request_kwargs(task), timeout=self.timeout_s)
        except Exception as exc:
            msg = redact(f"{type(exc).__name__}: {exc}", [self._api_key])
            hint = ""
            low = msg.lower()
            if "connection" in low or "timeout" in low or "timed out" in low:
                hint = "\nCould not reach the Gemini API (network, proxy or firewall). Nothing was planned."
            elif "404" in msg or "not found" in low:
                hint = (f"\nModel {self.model!r} was not found for this key. Gemini 4 Argon is paid-tier only; "
                        "check billing on your AI Studio project. The model is not substituted automatically.")
            elif "permission" in low or "403" in msg or "401" in msg or "api key" in low:
                hint = "\nThe API key was rejected or lacks access."
            elif "429" in msg or "quota" in low:
                hint = "\nRate limit or quota exceeded."
            raise PlannerError(f"Gemini request failed: {msg}{hint}") from None
        text = getattr(interaction, "output_text", None)
        if not text:
            raise PlannerError("Gemini returned no text output")
        return text


# --------------------------------------------------------------------------
# Offline demo planner: deterministic, no network, no AI.
# --------------------------------------------------------------------------

def _move(duration: float, **targets: float) -> dict[str, Any]:
    return {"action": "move_joints", "duration_s": duration,
            "targets": [{"joint": j, "position_rad": v} for j, v in targets.items()]}


def _wait(duration: float) -> dict[str, Any]:
    return {"action": "wait", "duration_s": duration}


OFFLINE_PLANS: dict[str, dict[str, Any]] = {
    "wave": {
        "summary": "Raise the forearm and wave by swinging the shoulder yaw.",
        "actions": [
            _move(1.0, shoulder_pitch=-0.3, elbow_pitch=-0.8),
            _move(0.8, shoulder_yaw=0.35),
            _move(1.0, shoulder_yaw=-0.35),
            _move(1.0, shoulder_yaw=0.35),
            _move(0.8, shoulder_yaw=0.0),
            _move(1.0, shoulder_pitch=0.0, elbow_pitch=0.0),
        ],
    },
    "nod": {
        "summary": "Nod by tilting the wrist down and up twice.",
        "actions": [_move(0.8, wrist_pitch=0.6), _move(0.8, wrist_pitch=-0.2),
                    _move(0.8, wrist_pitch=0.6), _move(0.8, wrist_pitch=0.0)],
    },
    "look left": {
        "summary": "Rotate the arm to the left, pause, and return.",
        "actions": [_move(1.0, shoulder_yaw=0.8), _move(1.0, shoulder_yaw=1.2), _wait(1.0),
                    _move(1.5, shoulder_yaw=0.4), _move(1.0, shoulder_yaw=0.0)],
    },
    "reach forward": {
        "summary": "Lean forward in two stages, hold, then return home.",
        "actions": [_move(1.0, shoulder_pitch=0.5, elbow_pitch=0.4),
                    _move(1.0, shoulder_pitch=0.9, elbow_pitch=0.7, wrist_pitch=0.3),
                    _wait(1.0),
                    _move(1.0, shoulder_pitch=0.4, elbow_pitch=0.3, wrist_pitch=0.0),
                    _move(1.0, shoulder_pitch=0.0, elbow_pitch=0.0)],
    },
    "home": {"summary": "Move every joint to the home position.",
             "actions": [_move(1.5, shoulder_yaw=0.0, shoulder_pitch=0.0, elbow_pitch=0.0, wrist_pitch=0.0)]},
    # Deliberately unsafe: each move is in range, but the sequence jumps too far.
    # Exists to demonstrate that the validator rejects it.
    "unsafe demo": {
        "summary": "UNSAFE on purpose: swings shoulder_yaw from +1.2 to -1.2 in one move.",
        "actions": [_move(1.0, shoulder_yaw=0.6), _move(1.0, shoulder_yaw=1.2), _move(1.0, shoulder_yaw=-1.2)],
    },
}


class OfflinePlanner:
    """Keyword-matched canned plans for demos and CI. Not an AI planner."""

    name = "offline"

    def plan_text(self, task: str) -> str:
        task = _check_task(task).lower()
        for key in sorted(OFFLINE_PLANS, key=len, reverse=True):
            if key in task:
                return json.dumps(OFFLINE_PLANS[key])
        raise PlannerError(
            "offline planner only knows these demo tasks: " + ", ".join(repr(k) for k in OFFLINE_PLANS)
            + ". Use Gemini for free-form instructions."
        )


def plan(planner: Planner, task: str) -> ActionPlan:
    return parse_plan_json(planner.plan_text(task))
