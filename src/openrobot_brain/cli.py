"""Command-line interface.

    orb plan "wave hello"              # Gemini plan, dry run (default)
    orb plan "wave" --offline          # rule-based demo planner, no API key
    orb plan "wave" --offline --execute --headless
    orb validate plan.json [--execute]
    orb sim-check
    orb demos
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Callable, Sequence, TextIO

from . import __version__
from .config import DEFAULT_ROBOT, RobotConfig
from .planner import OFFLINE_PLANS, THINKING_LEVELS, GeminiPlanner, OfflinePlanner, Planner, PlannerError
from .safety import SafetyError, ValidatedPlan, validate_plan
from .schema import PlanParseError, parse_plan_json

EXIT_OK, EXIT_REJECTED, EXIT_PLANNER, EXIT_SIM = 0, 2, 3, 4

ExecutorFactory = Callable[[RobotConfig, bool], object]


def _default_executor_factory(robot: RobotConfig, headless: bool):
    from .simulation.executor import MujocoExecutor  # lazy: dry runs never import MuJoCo
    return MujocoExecutor(robot, headless=headless)


def format_plan(vp: ValidatedPlan) -> str:
    lines = [f"Plan: {vp.plan.summary or '(no summary)'}",
             f"Actions: {len(vp.steps)}   total duration: {vp.total_duration_s:.1f}s", ""]
    for step in vp.steps:
        a = step.action
        if a.action == "wait":
            lines.append(f"  [{step.index}] wait {a.duration_s:.2f}s")
        else:
            moves = ", ".join(f"{t.joint} {step.start_state[t.joint]:+.2f}→{t.position_rad:+.2f}" for t in a.targets)
            lines.append(f"  [{step.index}] move_joints {a.duration_s:.2f}s  {moves}")
    lines.append("")
    lines.append("Final pose: " + ", ".join(f"{k}={v:+.2f}" for k, v in vp.final_state.items()))
    return "\n".join(lines)


def run_pipeline(plan_text: str, *, robot: RobotConfig = DEFAULT_ROBOT, execute: bool = False,
                 headless: bool = True, out: TextIO | None = None,
                 executor_factory: ExecutorFactory = _default_executor_factory) -> int:
    """Parse → validate → preview → (optionally) execute. Returns an exit code."""
    out = out or sys.stdout  # resolved at call time so redirection works
    try:
        plan = parse_plan_json(plan_text)
        validated = validate_plan(plan, robot)
    except (PlanParseError, SafetyError) as exc:
        print(f"REJECTED — nothing was executed.\n{exc}", file=out)
        return EXIT_REJECTED

    print("Validated plan (passed schema + safety checks):\n", file=out)
    print(format_plan(validated), file=out)

    if not execute:
        print("\nDry run: no simulation started. Add --execute to run this plan in MuJoCo.", file=out)
        return EXIT_OK

    print(f"\nExecuting in MuJoCo ({'headless' if headless else 'viewer'})...", file=out)
    try:
        with executor_factory(robot, headless) as ex:
            if getattr(ex, "viewer_error", None):
                print(f"Note: {ex.viewer_error}", file=out)
            result = ex.execute(validated)
    except Exception as exc:
        print(f"Simulation error: {exc}", file=out)
        return EXIT_SIM

    for s in result.steps:
        print(f"  [{s.index}] {s.action:<11} peak tracking error {s.max_tracking_error_rad:.3f} rad", file=out)
    print(f"Result: {result.message} (sim time {result.sim_time_s:.2f}s)", file=out)
    print("Final joint state: " + ", ".join(f"{k}={v:+.3f}" for k, v in result.final_state.items()), file=out)
    return EXIT_OK if result.success else EXIT_SIM


def _make_planner(args) -> Planner:
    if args.offline:
        return OfflinePlanner()
    return GeminiPlanner(DEFAULT_ROBOT, model=args.model, thinking_level=args.thinking)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="orb", description="OpenRobot-Brain: language → validated plan → MuJoCo.")
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    def add_exec_flags(sp):
        sp.add_argument("--execute", action="store_true", help="run the validated plan in MuJoCo (default: dry run)")
        sp.add_argument("--headless", action="store_true", help="no viewer window")

    sp = sub.add_parser("plan", help="plan a natural-language task")
    sp.add_argument("task")
    sp.add_argument("--offline", action="store_true", help="use the rule-based demo planner (no API key)")
    sp.add_argument("--model", default=None, help="Gemini model ID (default: $GEMINI_MODEL or gemini-4-argon)")
    sp.add_argument("--thinking", choices=THINKING_LEVELS, default="low")
    sp.add_argument("--save", type=Path, help="write the raw plan JSON to this file")
    add_exec_flags(sp)

    sp = sub.add_parser("validate", help="validate (and optionally run) a plan JSON file")
    sp.add_argument("file", type=Path)
    add_exec_flags(sp)

    sub.add_parser("sim-check", help="load the MuJoCo scene and run a headless smoke test")
    sub.add_parser("demos", help="list offline demo tasks")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        from dotenv import load_dotenv
        load_dotenv()  # reads .env from the current directory; values never printed
    except ImportError:  # pragma: no cover
        pass

    if args.cmd == "demos":
        for k, v in OFFLINE_PLANS.items():
            print(f"  {k:<14} {v['summary']}")
        return EXIT_OK

    if args.cmd == "sim-check":
        from .simulation.executor import SimulationError, smoke_check
        try:
            report = smoke_check(DEFAULT_ROBOT)
        except SimulationError as exc:
            print(f"sim-check FAILED: {exc}")
            return EXIT_SIM
        print(json.dumps(report, indent=2, default=str))
        print("sim-check OK" if report["responds_to_commands"] else "sim-check FAILED: arm did not track commands")
        return EXIT_OK if report["responds_to_commands"] else EXIT_SIM

    if args.cmd == "validate":
        try:
            text = args.file.read_text(encoding="utf-8")
        except OSError as exc:
            print(f"cannot read {args.file}: {exc}")
            return EXIT_PLANNER
        return run_pipeline(text, execute=args.execute, headless=args.headless)

    # plan
    try:
        planner = _make_planner(args)
        label = "offline demo planner (rule-based, no AI)" if args.offline else f"Gemini model {planner.model}"
        print(f"Planning with {label}...")
        text = planner.plan_text(args.task)
    except PlannerError as exc:
        print(f"Planner error: {exc}")
        return EXIT_PLANNER
    if args.save:
        args.save.write_text(text, encoding="utf-8")
        print(f"Raw plan saved to {args.save}")
    return run_pipeline(text, execute=args.execute, headless=args.headless)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
