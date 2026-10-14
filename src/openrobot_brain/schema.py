"""Strict, typed action-plan schema.

This module only checks *shape*: required fields, types, finite numbers, no
extra fields, and the action allowlist. Robot-specific checks (joint names,
limits, motion deltas) live in ``safety.py`` so they are independent of
whatever the model claims.
"""

from __future__ import annotations

import json
import math
from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, ValidationError

FiniteFloat = Annotated[float, Field(strict=True, allow_inf_nan=False)]


class PlanParseError(ValueError):
    """Raised when model output cannot be turned into a valid ActionPlan."""


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class JointTarget(_Strict):
    joint: Annotated[str, Field(min_length=1, max_length=64)]
    position_rad: FiniteFloat


class MoveJoints(_Strict):
    action: Literal["move_joints"]
    targets: Annotated[list[JointTarget], Field(min_length=1, max_length=8)]
    duration_s: FiniteFloat


class Wait(_Strict):
    action: Literal["wait"]
    duration_s: FiniteFloat


Action = Annotated[Union[MoveJoints, Wait], Field(discriminator="action")]


class ActionPlan(_Strict):
    summary: Annotated[str, Field(max_length=300)] = ""
    actions: Annotated[list[Action], Field(min_length=1, max_length=64)]


def _reject_constant(name: str) -> Any:
    raise PlanParseError(f"non-finite number {name!r} is not allowed")


def _finite_float(text: str) -> float:
    value = float(text)
    if not math.isfinite(value):
        raise PlanParseError(f"non-finite number {text!r} is not allowed")
    return value


def parse_plan_json(text: str) -> ActionPlan:
    """Parse untrusted JSON text into an ActionPlan, or raise PlanParseError."""
    if not isinstance(text, str) or not text.strip():
        raise PlanParseError("model returned an empty response")
    try:
        data = json.loads(
            text,
            parse_constant=_reject_constant,  # NaN, Infinity, -Infinity
            parse_float=_finite_float,  # e.g. 1e999 overflows to inf
        )
    except PlanParseError:
        raise
    except json.JSONDecodeError as exc:
        raise PlanParseError(f"malformed JSON: {exc.msg} (line {exc.lineno}, col {exc.colno})") from exc
    if not isinstance(data, dict):
        raise PlanParseError("plan must be a JSON object")
    try:
        return ActionPlan.model_validate(data)
    except ValidationError as exc:
        problems = "; ".join(
            f"{'.'.join(str(p) for p in err['loc']) or '<root>'}: {err['msg']}" for err in exc.errors()
        )
        raise PlanParseError(f"schema violation: {problems}") from exc


def gemini_response_schema(joint_names: tuple[str, ...]) -> dict[str, Any]:
    """JSON schema sent to Gemini's structured-output mode.

    Constraining joint names with an enum steers the model, but the local
    validators still re-check everything.
    """
    target = {
        "type": "object",
        "properties": {
            "joint": {"type": "string", "enum": list(joint_names)},
            "position_rad": {"type": "number"},
        },
        "required": ["joint", "position_rad"],
        "additionalProperties": False,
    }
    move = {
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": ["move_joints"]},
            "targets": {"type": "array", "items": target, "minItems": 1},
            "duration_s": {"type": "number"},
        },
        "required": ["action", "targets", "duration_s"],
        "additionalProperties": False,
    }
    wait = {
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": ["wait"]},
            "duration_s": {"type": "number"},
        },
        "required": ["action", "duration_s"],
        "additionalProperties": False,
    }
    return {
        "type": "object",
        "properties": {
            "summary": {"type": "string"},
            "actions": {"type": "array", "items": {"anyOf": [move, wait]}, "minItems": 1},
        },
        "required": ["summary", "actions"],
        "additionalProperties": False,
    }
