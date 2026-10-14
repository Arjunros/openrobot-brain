"""Closed-loop agent: observe -> plan (AI) -> validate -> execute -> feed back -> repeat.

Every round:
  1. Park pose, take an overhead image (vision) or read object states (state mode).
  2. Ask the planner for {status, message, observations, actions}.
  3. Vision: convert the model's pixel observations to a metric world belief.
  4. Expand + validate the actions against that belief. Rejections are fed back
     to the model (capped), nothing moves.
  5. Execute in MuJoCo, then report per-action outcomes to the model next round.

Stops on "done", "infeasible", too many consecutive rejections, a planner error,
or the round limit. In dry-run mode it stops after validating round 1.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TextIO

import numpy as np

from .config import RobotConfig
from .perception import perceive, score_perception
from .planner import Planner, PlannerError, RoundInput, check_task
from .safety import SafetyError, ValidatedPlan, validate_round
from .schema import PlanParseError, parse_agent_json
from .simulation.executor import PandaSim
from .world import WorldState


@dataclass
class AgentOutcome:
    status: str                       # done | infeasible | rejected | error | max_rounds | dry_run | aborted
    message: str
    rounds: int
    history: list[dict[str, Any]] = field(default_factory=list)


def _fmt_action(a) -> str:
    d = a.model_dump()
    name = d.pop("action")
    args = ", ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}" for k, v in d.items())
    return f"{name}({args})"


def build_prompt(task: str, rnd: int, max_rounds: int, vision: bool, belief_text: str | None,
                 held_note: str, history: list[dict[str, Any]]) -> str:
    lines = [f"TASK: {task}", f"ROUND {rnd} of at most {max_rounds}.", f"GRIPPER: {held_note}"]
    if history:
        lines.append("\nWHAT HAPPENED SO FAR:")
        for h in history[-4:]:
            lines.append(f"- round {h['round']}: you said status={h['status']}: {h['message']}")
            if h.get("actions"):
                lines.append(f"  actions: {'; '.join(h['actions'])}")
            if h.get("rejected"):
                lines.append("  SAFETY SYSTEM REJECTED THE ROUND (nothing moved):")
                lines += [f"    * {v}" for v in h["rejected"]]
            if h.get("parse_error"):
                lines.append(f"  YOUR RESPONSE WAS INVALID: {h['parse_error']}")
            if h.get("outcomes"):
                lines += [f"    * {o}" for o in h["outcomes"]]
    if vision:
        lines.append("\nThe current overhead image is attached. Report observations, then decide.")
    else:
        lines.append("\nOBJECTS NOW:\n" + (belief_text or "  (none)"))
    return "\n".join(lines)


def run_agent(task: str, planner: Planner, sim: PandaSim, robot: RobotConfig, *, execute: bool,
              vision: bool, max_rounds: int | None = None, out: TextIO | None = None,
              save_dir: Path | None = None) -> AgentOutcome:
    out = out or sys.stdout
    say = lambda *a: print(*a, file=out, flush=True)  # noqa: E731
    task = check_task(task)
    max_rounds = min(max_rounds or robot.safety.max_rounds, robot.safety.max_rounds)
    history: list[dict[str, Any]] = []
    rejections = 0
    held_belief = None  # vision mode: the object we believe is in the gripper (not visible)

    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)
    who = "Scripted demo (not AI)" if planner.name == "offline" else "GPT-6 Astra" if "astra" in planner.model \
        else planner.model

    for rnd in range(1, max_rounds + 1):
        say(f"\n━━ Round {rnd} " + "━" * 40)
        sim.annotate(title=f"Round {rnd} · {who}: \"{task}\"", subtitle="observing the table…",
                     badge=("PLANNING", (40, 90, 180)))
        sim.record_still(0.8)
        truth = sim.ground_truth()
        img_png = None
        if vision:
            img = sim.render("overhead")
            img_png = sim.to_png(img)
            if save_dir:
                (save_dir / f"round{rnd}_overhead.png").write_bytes(img_png)
        if not sim.holding_something():
            held_belief = None
        held_note = (f"holding {held_belief.id} (out of view)" if (vision and held_belief) else
                     f"holding {truth.held}" if (not vision and truth.held) else "empty")
        prompt = build_prompt(task, rnd, max_rounds, vision, None if vision else truth.describe(), held_note, history)
        rin = RoundInput(task, rnd, max_rounds, prompt, img_png, truth, history)

        t0 = time.perf_counter()
        try:
            raw = planner.respond(rin)
        except PlannerError as exc:
            say(f"Planner error: {exc}")
            return AgentOutcome("error", str(exc), rnd, history)
        dt = time.perf_counter() - t0
        if planner.name != "offline":
            say(f"{planner.model} replied in {dt:.1f} s")

        entry: dict[str, Any] = {"round": rnd, "status": "invalid", "message": ""}
        history.append(entry)
        try:
            resp = parse_agent_json(raw)
        except PlanParseError as exc:
            say(f"Invalid model output, nothing executed: {exc}")
            entry["parse_error"] = str(exc)
            rejections += 1
            if rejections > robot.safety.max_consecutive_rejections:
                return AgentOutcome("rejected", "too many invalid responses", rnd, history)
            continue
        entry.update(status=resp.status, message=resp.message,
                     actions=[_fmt_action(a) for a in resp.actions])
        say(f"Model: [{resp.status}] {resp.message}")
        sim.annotate(subtitle=resp.message, badge=None)

        # Belief about the world
        if vision:
            belief, warns = perceive(list(resp.observations), sim.camera, robot.cube_size_m, held_belief)
            for w in warns:
                say(f"  perception warning: {w}")
            scores = score_perception(belief, truth)
            if scores:
                say("Perception (model's pixel estimate → metres, checked against simulator truth):")
                for s in scores:
                    ok = "✓" if s.label_ok and s.error_cm is not None and s.error_cm < 2.5 else "✗"
                    o = belief.objects[s.id].pos
                    say(f"  {ok} {s.id:<12} '{s.label}' at ({o[0]:.3f}, {o[1]:+.3f}, {o[2]:.3f})"
                        f"  error {s.error_cm} cm vs {s.matched}")
            entry["perception"] = [s.__dict__ for s in scores]
        else:
            belief = truth

        if resp.status == "done":
            say("\n✅ DONE (model reports the task is complete)")
            sim.annotate(badge=("DONE", (30, 150, 70)))
            sim.record_still(2.5)
            _final_state(say, sim)
            return AgentOutcome("done", resp.message, rnd, history)
        if resp.status == "infeasible":
            say(f"\n⛔ NOT FEASIBLE: {resp.message}\nNothing was executed.")
            sim.annotate(badge=("NOT FEASIBLE - nothing executed", (200, 120, 20)))
            sim.record_still(3.0)
            return AgentOutcome("infeasible", resp.message, rnd, history)

        try:
            vp = validate_round(list(resp.actions), robot, sim.kin, sim.q(), belief, sim.gripper_open())
        except SafetyError as exc:
            say("🛑 REJECTED by safety validator, nothing moved:")
            for v in exc.violations:
                say(f"   - {v}")
            entry["rejected"] = exc.violations
            sim.annotate(subtitle=exc.violations[0], badge=("REJECTED BY SAFETY CHECK - nothing moved", (190, 40, 40)))
            sim.record_still(2.5)
            rejections += 1
            if rejections > robot.safety.max_consecutive_rejections:
                say(f"\nStopping: {rejections} rejected rounds in a row.")
                return AgentOutcome("rejected", "safety validator rejected repeated plans", rnd, history)
            continue
        rejections = 0
        _preview(say, vp)
        sim.annotate(badge=("VALIDATED: limits + collisions OK", (30, 150, 70)))
        sim.record_still(1.2)

        if not execute:
            say("\nDry run: validated but not executed. Add --execute to run the closed loop in MuJoCo.")
            return AgentOutcome("dry_run", resp.message, rnd, history)

        result = sim.execute(vp)
        held_belief = (vp.world_end.objects.get(vp.world_end.held) if vp.world_end.held else None)
        if held_belief is not None and not sim.holding_something():
            held_belief = None
        entry["outcomes"] = [f"action[{o.index}] {o.action}: {'ok' if o.ok else 'FAILED'} - {o.detail}"
                             for o in result.outcomes] + ([result.message] if not result.completed else [])
        say(f"Executed ({result.sim_time_s:.1f} s sim): {result.message}")
        if not result.completed:
            sim.annotate(subtitle=result.message, badge=("FAILED - reported back to the model", (200, 120, 20)))
            sim.record_still(2.0)
        for o in result.outcomes:
            say(f"   {'✓' if o.ok else '✗'} action[{o.index}] {o.action}: {o.detail}")

    say(f"\nStopped after {max_rounds} rounds without the model reporting 'done'.")
    _final_state(say, sim)
    return AgentOutcome("max_rounds", "round limit reached", max_rounds, history)


def _preview(say, vp: ValidatedPlan) -> None:
    say(f"✓ Validated: {len(vp.actions)} action(s), {len(vp.segments)} motion segments, "
        f"{vp.duration_s:.1f} s of motion — all collision/limit checks passed")
    for i, a in enumerate(vp.actions):
        say(f"   [{i}] {_fmt_action(a)}")


def _final_state(say, sim: PandaSim) -> None:
    gt = sim.ground_truth()
    say("Simulator state (ground truth):")
    for o in gt.objects.values():
        where = "in gripper" if o.id == gt.held else f"({o.pos[0]:.3f}, {o.pos[1]:+.3f}, {o.pos[2]:.3f})"
        say(f"   {o.id:<11} {where}")
