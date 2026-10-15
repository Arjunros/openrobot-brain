"""Command-line interface.

    orb run "stack the red cube on the blue cube"              # GPT-6 Astra + vision, dry run
    orb run "build a tower" --execute                          # closed loop in the MuJoCo viewer
    orb run "build a tower" --execute --hold --speed 0.5       # keep window open, half speed (for recording)
    orb run "stack red on blue" --no-vision --execute          # model gets object positions, no image
    orb run "build a tower" --execute --headless --record tower.mp4   # captioned video for sharing
    orb run "build a tower" --offline --execute                # scripted demo, no API key
    orb snapshot --seed 3                                      # save camera images of a layout
    orb sim-check
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

from . import __version__
from .config import PANDA

EXIT_OK, EXIT_REJECTED, EXIT_PLANNER, EXIT_SIM, EXIT_NOT_DONE = 0, 2, 3, 4, 5
DEMOS = {
    "stack the red cube on the blue cube": "pick + place_on",
    "pick up the green cube": "single pick",
    "build a tower": "three-cube tower over several closed-loop rounds",
    "unsafe demo": "carries a cube off the table; the validator rejects it every round",
}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="orb", description="OpenRobot-Brain: GPT-6 Astra + vision → validated "
                                                        "pick-and-place on a simulated Franka Panda.")
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("run", aliases=["plan"], help="run a natural-language task")
    sp.add_argument("task")
    sp.add_argument("--execute", action="store_true", help="execute in MuJoCo (default: validate round 1 only)")
    sp.add_argument("--headless", action="store_true", help="no viewer window")
    sp.add_argument("--hold", action="store_true", help="keep the viewer open after the task ends")
    sp.add_argument("--speed", type=float, default=1.0, help="viewer playback speed, e.g. 0.5 = slow motion")
    sp.add_argument("--offline", action="store_true", help="scripted demo planner (no API key, not AI)")
    sp.add_argument("--no-vision", action="store_true", help="give the model object positions instead of an image")
    sp.add_argument("--effort", choices=("low", "medium", "high", "xhigh", "max"), default="low",
                    help="Astra reasoning effort")
    sp.add_argument("--model", default=None, help="model ID (default: $OPENAI_MODEL or gpt-6-astra)")
    sp.add_argument("--rounds", type=int, default=None, help=f"max rounds (cap {PANDA.safety.max_rounds})")
    sp.add_argument("--seed", type=int, default=None, help="random cube layout (default: fixed layout)")
    sp.add_argument("--save-images", type=Path, default=None, help="save each round's camera image here")
    sp.add_argument("--record", type=Path, default=None,
                    help="render a captioned video of the run (.mp4 needs the [video] extra, .gif works anywhere)")

    sp = sub.add_parser("snapshot", help="save overhead and front camera images of the scene")
    sp.add_argument("--seed", type=int, default=None)
    sp.add_argument("--out", type=Path, default=Path("snapshots"))

    sub.add_parser("sim-check", help="load the scene and run a headless pick-and-place smoke test")
    sub.add_parser("demos", help="list offline demo tasks")
    return p


def _cmd_run(args) -> int:
    from .agent import run_agent
    from .planner import AstraPlanner, OfflinePlanner, PlannerError
    from .simulation.executor import IMAGE_H, IMAGE_W, PandaSim, SimulationError, VisionUnavailable, random_layout

    vision = not args.no_vision and not args.offline
    try:
        if args.offline:
            planner = OfflinePlanner()
            print("Planner: offline scripted demo (not AI; uses simulator state)")
        else:
            planner = AstraPlanner(PANDA, vision=vision, image_size=(IMAGE_W, IMAGE_H), model=args.model,
                                   effort=args.effort)
            print(f"Planner: OpenAI {planner.model} (effort={args.effort}, "
                  f"{'vision: overhead camera image' if vision else 'state: object positions'})")
    except PlannerError as exc:
        print(f"Planner error: {exc}")
        return EXIT_PLANNER

    headless = args.headless or not args.execute
    try:
        with PandaSim(PANDA, headless=headless, speed=args.speed) as sim:
            if args.seed is not None:
                sim.reset(random_layout(args.seed, PANDA))
            if sim.viewer_error:
                print(f"Note: {sim.viewer_error}")
            if args.record:
                try:
                    sim.start_recording(args.record)
                except Exception as exc:
                    print(f"Recording error: {exc}")
                    return EXIT_SIM
            try:
                outcome = run_agent(args.task, planner, sim, PANDA, execute=args.execute, vision=vision,
                                    max_rounds=args.rounds, save_dir=args.save_images)
            except KeyboardInterrupt:
                print("\nCancelled.")
                return EXIT_PLANNER
            if args.record:
                path = sim.stop_recording()
                if path:
                    print(f"\nVideo saved: {path}")
            if args.hold and sim.has_viewer:
                print("\nViewer held open — close the window to exit.")
                sim.hold()
    except VisionUnavailable as exc:
        print(f"Vision error: {exc}")
        return EXIT_SIM
    except (SimulationError, PlannerError) as exc:
        print(f"Error: {exc}")
        return EXIT_SIM
    return {"done": EXIT_OK, "dry_run": EXIT_OK, "rejected": EXIT_REJECTED, "error": EXIT_PLANNER}.get(
        outcome.status, EXIT_NOT_DONE)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        from dotenv import find_dotenv, load_dotenv
        load_dotenv(find_dotenv(usecwd=True))  # .env in the current directory; values are never printed
    except ImportError:  # pragma: no cover
        pass

    if args.cmd == "demos":
        for k, v in DEMOS.items():
            print(f'  orb run "{k}" --offline --execute    # {v}')
        return EXIT_OK

    if args.cmd == "sim-check":
        import json

        from .simulation.executor import SimulationError, smoke_check
        try:
            report = smoke_check(PANDA)
        except SimulationError as exc:
            print(f"sim-check FAILED: {exc}")
            return EXIT_SIM
        print(json.dumps(report, indent=2))
        ok = report["pick_and_place_executed"] and report["red_on_blue"]
        print("sim-check OK" if ok else "sim-check FAILED")
        return EXIT_OK if ok else EXIT_SIM

    if args.cmd == "snapshot":
        from .simulation.executor import PandaSim, VisionUnavailable, random_layout
        try:
            with PandaSim(PANDA) as sim:
                if args.seed is not None:
                    sim.reset(random_layout(args.seed, PANDA))
                args.out.mkdir(parents=True, exist_ok=True)
                for cam in ("overhead", "front"):
                    path = args.out / f"{cam}.png"
                    path.write_bytes(sim.to_png(sim.render(cam)))
                    print(f"saved {path}")
        except VisionUnavailable as exc:
            print(f"Vision error: {exc}")
            return EXIT_SIM
        return EXIT_OK

    return _cmd_run(args)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
