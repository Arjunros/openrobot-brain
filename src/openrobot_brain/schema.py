"""Strict schema for one round of the agent loop.

Each round the model returns:

    {
      "status": "act" | "done" | "infeasible",
      "message": "...",
      "observations": [{"id", "label", "u", "v", "on_top_of"}],   # vision mode
      "actions": [ pick | place_on | place_at | move_to | go_home | wait ]
    }

This module checks *shape* only. Whether an action is possible and safe for
the robot and the current world is decided in ``skills.py`` / ``safety.py``.
"""

from __future__ import annotations

import json
import math
from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

FiniteFloat = Annotated[float, Field(strict=True, allow_inf_nan=False)]
ObjectId = Annotated[str, Field(pattern=r"^[a-z][a-z0-9_]{0,23}$")]


class PlanParseError(ValueError):
    """Model output could not be turned into a valid AgentResponse."""


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class Pick(_Strict):
    action: Literal["pick"]
    object: ObjectId


class PlaceOn(_Strict):
    action: Literal["place_on"]
    object: ObjectId


class PlaceAt(_Strict):
    action: Literal["place_at"]
    x: FiniteFloat
    y: FiniteFloat


class MoveTo(_Strict):
    action: Literal["move_to"]
    x: FiniteFloat
    y: FiniteFloat
    z: FiniteFloat


class GoHome(_Strict):
    action: Literal["go_home"]


class Wait(_Strict):
    action: Literal["wait"]
    duration_s: FiniteFloat


Action = Annotated[Union[Pick, PlaceOn, PlaceAt, MoveTo, GoHome, Wait], Field(discriminator="action")]


class Observation(_Strict):
    id: ObjectId
    label: Annotated[str, Field(min_length=1, max_length=40)]
    u: FiniteFloat
    v: FiniteFloat
    on_top_of: Annotated[str, Field(max_length=24)] = ""


class AgentResponse(_Strict):
    status: Literal["act", "done", "infeasible"]
    message: Annotated[str, Field(max_length=500)]
    observations: Annotated[list[Observation], Field(max_length=12)] = []
    actions: Annotated[list[Action], Field(max_length=16)] = []

    @model_validator(mode="after")
    def _status_matches_actions(self) -> "AgentResponse":
        if self.status == "act" and not self.actions:
            raise ValueError("status 'act' requires at least one action")
        if self.status != "act" and self.actions:
            raise ValueError(f"status '{self.status}' must not include actions")
        ids = [o.id for o in self.observations]
        if len(ids) != len(set(ids)):
            raise ValueError("observation ids must be unique")
        return self


def _reject_constant(name: str) -> Any:
    raise PlanParseError(f"non-finite number {name!r} is not allowed")


def _finite_float(text: str) -> float:
    value = float(text)
    if not math.isfinite(value):
        raise PlanParseError(f"non-finite number {text!r} is not allowed")
    return value


def parse_agent_json(text: str) -> AgentResponse:
    """Parse untrusted model text into an AgentResponse, or raise PlanParseError."""
    if not isinstance(text, str) or not text.strip():
        raise PlanParseError("model returned an empty response")
    try:
        data = json.loads(text, parse_constant=_reject_constant, parse_float=_finite_float)
    except PlanParseError:
        raise
    except json.JSONDecodeError as exc:
        raise PlanParseError(f"malformed JSON: {exc.msg} (line {exc.lineno}, col {exc.colno})") from exc
    if not isinstance(data, dict):
        raise PlanParseError("response must be a JSON object")
    try:
        return AgentResponse.model_validate(data)
    except ValidationError as exc:
        problems = "; ".join(
            f"{'.'.join(str(p) for p in err['loc']) or '<root>'}: {err['msg']}" for err in exc.errors()
        )
        raise PlanParseError(f"schema violation: {problems}") from exc


def _obj(props: dict[str, Any]) -> dict[str, Any]:
    return {"type": "object", "properties": props, "required": list(props), "additionalProperties": False}


def response_json_schema() -> dict[str, Any]:
    """JSON schema for OpenAI strict structured outputs (a JSON-Schema subset:
    every property required, no additionalProperties). It only steers the
    model; ``parse_agent_json`` and the safety validator still check everything."""
    s, n = {"type": "string"}, {"type": "number"}

    def act(name: str, **fields: Any) -> dict[str, Any]:
        return _obj({"action": {"type": "string", "enum": [name]}, **fields})

    return _obj({
        "status": {"type": "string", "enum": ["act", "done", "infeasible"]},
        "message": s,
        "observations": {"type": "array", "items": _obj({"id": s, "label": s, "u": n, "v": n, "on_top_of": s})},
        "actions": {"type": "array", "items": {"anyOf": [
            act("pick", object=s),
            act("place_on", object=s),
            act("place_at", x=n, y=n),
            act("move_to", x=n, y=n, z=n),
            act("go_home"),
            act("wait", duration_s=n),
        ]}},
    })
