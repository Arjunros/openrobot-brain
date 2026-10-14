"""Planners for the agent loop.

* ``AstraPlanner`` -- OpenAI GPT-6 Astra via the Responses API (the only AI planner).
* ``OfflinePlanner`` -- deterministic scripted policy for demos/CI. Not AI.

Both return raw JSON text for one round. Parsing, perception and safety checks
happen elsewhere; model output is always untrusted.

OpenAI integration (verified 2026-10-14 against openai 3.31.0):
  * ``client.responses.create(model="gpt-6-astra", instructions=..., input=[...])``
  * image input: ``{"type": "input_image", "image_url": "data:image/png;base64,...", "detail": "high"}``
  * strict structured output: ``text={"format": {"type": "json_schema", "strict": True, ...}}``
  * ``reasoning={"effort": low|medium|high|xhigh|max}``; temperature/top_p are rejected by Astra.
"""

from __future__ import annotations

import base64
import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Protocol

from .config import RobotConfig
from .schema import response_json_schema
from .world import WorldState

DEFAULT_MODEL = "gpt-6-astra"
EFFORTS = ("low", "medium", "high", "xhigh", "max")
MAX_TASK_CHARS = 500


class PlannerError(RuntimeError):
    pass


class MissingCredentialsError(PlannerError):
    pass


@dataclass
class RoundInput:
    task: str
    round: int
    max_rounds: int
    prompt: str                    # full text the AI sees this round
    image_png: bytes | None        # overhead camera image (vision mode)
    truth: WorldState              # ground truth: ONLY the offline scripted planner may use it
    history: list[dict[str, Any]] = field(default_factory=list)


class Planner(Protocol):
    name: str
    model: str

    def respond(self, rin: RoundInput) -> str: ...


def check_task(task: str) -> str:
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
    return re.sub(r"sk-[0-9A-Za-z_\-]{16,}", "***", text)


def system_instructions(robot: RobotConfig, vision: bool, image_w: int, image_h: int) -> str:
    ws, lim, c = robot.workspace, robot.safety, robot.cube_size_m
    perception = (
        f"""PERCEPTION (vision mode)
Each round you get one top-down image ({image_w}x{image_h} px) from a fixed overhead camera.
Pixel origin is the top-left corner; u grows to the right, v grows downward.
In the image, UP is +x (away from the robot) and RIGHT is -y (the robot's right).
List every object you can see in `observations`:
  id: lowercase letters, digits, underscores, max 24 chars, starting with a letter (e.g. "red_cube");
      keep ids stable across rounds
  label: 2-4 words, colour + shape, at most 80 characters (e.g. "red cube")
  u, v: pixel centre of the object's visible TOP face, as precisely as you can
  on_top_of: id of the object it rests on, or "" if it is on the table
The robot converts your pixel coordinates to metres itself. Refer to objects by id in actions.
The arm is parked out of view while the photo is taken; an object held in the gripper is not visible."""
        if vision else
        """PERCEPTION (state mode)
Each round you get the list of objects with ids and positions in metres. Leave `observations` empty."""
    )
    return f"""You control a SIMULATED Franka Emika Panda arm with a parallel gripper in MuJoCo, working on a
tabletop with {c * 100:.0f} cm cubes. You plan high-level actions; a separate safety system turns them
into motions, checks every one, and rejects the whole round if anything is unsafe.

WORLD FRAME (metres): robot base at the origin, +x away from the robot, +y to the robot's left,
table surface at z = 0. Reachable area: x {ws.x[0]}..{ws.x[1]}, y {ws.y[0]}..{ws.y[1]}.

{perception}

ACTIONS (the only ones that exist):
  pick {{object}}            grasp an object from above. Gripper must be empty; nothing may be on top of it.
  place_on {{object}}        put the held object on top of another object (stacks up to {lim.max_stack_height} high).
  place_at {{x, y}}          put the held object on the table at (x, y); needs {lim.place_clearance_m} m clear of other objects.
  move_to {{x, y, z}}        move the gripper tip to a point (e.g. to point at something). z >= {ws.tcp_z_min}.
  go_home {{}}               return to the parked camera pose.
  wait {{duration_s}}        pause, at most {lim.max_wait_s} s.
At most {lim.max_actions_per_round} actions per round. After each round the arm parks, you get feedback
on what happened, and (in vision mode) a fresh image.

STATUS
  "act"         you are sending actions now.
  "done"        the task is complete AS SHOWN BY THE CURRENT OBSERVATION. Send no actions.
  "infeasible"  the task cannot be done with this robot (no such object, needs abilities it lacks,
                outside reach...). Explain why in `message`. Send no actions.

RULES
- Prefer a few actions per round, then check the result in the next observation.
- If feedback says a grasp missed or the safety system rejected your plan, adapt; don't repeat the same thing.
- Never claim success you cannot see. `message` is one or two short sentences for the human operator.
"""


class AstraPlanner:
    name = "openai"

    def __init__(self, robot: RobotConfig, *, vision: bool, image_size: tuple[int, int] = (640, 480),
                 model: str | None = None, api_key: str | None = None, effort: str = "low",
                 client: Any | None = None, timeout_s: float = 180.0, max_retries: int = 1,
                 image_detail: str = "high"):
        if effort not in EFFORTS:
            raise PlannerError(f"reasoning effort must be one of {EFFORTS}")
        self.robot = robot
        self.vision = vision
        self.image_size = image_size
        self.model = model or os.environ.get("OPENAI_MODEL") or DEFAULT_MODEL
        self.effort = effort
        self.image_detail = image_detail
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY") or None
        if client is not None:
            self._client = client
        else:
            if not self._api_key:
                raise MissingCredentialsError(
                    "no OpenAI API key found. Set OPENAI_API_KEY in your environment or .env file, "
                    "or run with --offline for the scripted demo planner.")
            from openai import OpenAI
            self._client = OpenAI(api_key=self._api_key, max_retries=max_retries, timeout=timeout_s)

    def request_kwargs(self, rin: RoundInput) -> dict[str, Any]:
        content: list[dict[str, Any]] = [{"type": "input_text", "text": rin.prompt}]
        if self.vision:
            if not rin.image_png:
                raise PlannerError("vision mode needs a camera image")
            b64 = base64.b64encode(rin.image_png).decode("ascii")
            content.append({"type": "input_image", "image_url": f"data:image/png;base64,{b64}",
                            "detail": self.image_detail})
        return {
            "model": self.model,
            "instructions": system_instructions(self.robot, self.vision, *self.image_size),
            "input": [{"role": "user", "content": content}],
            "text": {"format": {"type": "json_schema", "name": "robot_round", "strict": True,
                                "schema": response_json_schema()}},
            "reasoning": {"effort": self.effort},
            "store": False,
        }

    def respond(self, rin: RoundInput) -> str:
        try:
            response = self._client.responses.create(**self.request_kwargs(rin))
        except PlannerError:
            raise
        except Exception as exc:
            msg = redact(f"{type(exc).__name__}: {exc}", [self._api_key])[:800]
            low = msg.lower()
            hint = ""
            if "insufficient_quota" in low or "billing" in low:
                hint = ("\nYour OpenAI account has no API credit. API usage is billed separately from ChatGPT: "
                        "add credit at platform.openai.com -> Settings -> Billing.")
            elif "429" in msg or "rate limit" in low:
                hint = "\nRate limited (HTTP 429). Wait a minute and retry."
            elif "connection" in low or "timeout" in low or "timed out" in low:
                hint = "\nCould not reach the OpenAI API (network, proxy or firewall)."
            elif "404" in msg or "model_not_found" in low or "does not exist" in low:
                hint = (f"\nModel {self.model!r} is not available to this key/organization. "
                        "It is not substituted automatically.")
            elif "401" in msg or "invalid_api_key" in low or "incorrect api key" in low:
                hint = "\nThe API key was rejected. Create a new one at platform.openai.com/api-keys."
            elif "image" in low and ("unsupported" in low or "invalid" in low):
                hint = "\nThe model rejected the image input. Try --no-vision."
            raise PlannerError(f"OpenAI request failed: {msg}{hint}") from None
        if getattr(response, "status", "completed") == "incomplete":
            reason = getattr(getattr(response, "incomplete_details", None), "reason", "unknown")
            raise PlannerError(f"OpenAI response was incomplete ({reason})")
        text = getattr(response, "output_text", None)
        if not text:
            raise PlannerError("OpenAI returned no text output (the model may have refused)")
        return text


# ------------------------------------------------------------------------------
# Offline scripted planner: deterministic, uses simulator ground truth. Not AI.
# ------------------------------------------------------------------------------

_COLORS = ("red", "green", "blue")


def _cube(color: str) -> str:
    return f"{color}_cube"


def _resp(status: str, message: str, actions: list[dict] | None = None) -> str:
    return json.dumps({"status": status, "message": message, "observations": [], "actions": actions or []})


class OfflinePlanner:
    """Handles a few demo tasks: 'stack red on blue', 'pick up green', 'build a tower',
    and 'unsafe demo' (deliberately rejected by the safety validator)."""

    name = "offline"
    model = "scripted"

    def respond(self, rin: RoundInput) -> str:
        task = rin.task.lower()
        w = rin.truth
        colors = [c for c in re.findall(r"\b(red|green|blue)\b", task)]

        def on(a: str, b: str) -> bool:
            pa, pb = w.objects[a].pos, w.objects[b].pos
            return w.held != a and abs(pa[0] - pb[0]) < 0.02 and abs(pa[1] - pb[1]) < 0.02 and 0.03 < pa[2] - pb[2] < 0.05

        if "unsafe" in task:
            return _resp("act", "Deliberately unsafe demo: carry the red cube off the table.",
                         [{"action": "pick", "object": "red_cube"}, {"action": "place_at", "x": 1.2, "y": 0.0}])

        if "tower" in task or "all" in task:
            order = colors if len(colors) == 3 else ["blue", "red", "green"]
            for below, above in zip(order, order[1:]):
                if not on(_cube(above), _cube(below)):
                    return self._stack(w, _cube(above), _cube(below))
            return _resp("done", f"Tower built: {' / '.join(reversed(order))}.")

        if len(colors) >= 2 and any(k in task for k in ("stack", " on ", "onto", "put", "place")):
            a, b = _cube(colors[0]), _cube(colors[1])
            if on(a, b):
                return _resp("done", f"The {colors[0]} cube is on the {colors[1]} cube.")
            return self._stack(w, a, b)

        if len(colors) == 1 and any(k in task for k in ("pick", "grab", "lift")):
            a = _cube(colors[0])
            if w.held == a:
                return _resp("done", f"Holding the {colors[0]} cube.")
            return _resp("act", f"Picking up the {colors[0]} cube.", [{"action": "pick", "object": a}])

        return _resp("infeasible", "The offline demo planner only knows: 'stack <colour> on <colour>', "
                                   "'pick up <colour>', 'build a tower', 'unsafe demo'.")

    @staticmethod
    def _stack(w: WorldState, a: str, b: str) -> str:
        acts = [] if w.held == a else [{"action": "pick", "object": a}]
        acts.append({"action": "place_on", "object": b})
        return _resp("act", f"Stacking {a.replace('_', ' ')} on {b.replace('_', ' ')}.", acts)
