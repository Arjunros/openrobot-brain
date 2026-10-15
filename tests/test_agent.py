import io

import numpy as np
import pytest

from openrobot_brain.agent import run_agent
from openrobot_brain.planner import AstraPlanner, OfflinePlanner
from openrobot_brain.simulation.executor import PandaSim, VisionUnavailable
from conftest import pick, place_at, respond
from fakes import FakeVisionClient, ScriptedClient


def stacked(sim, a, b):
    gt = sim.ground_truth()
    pa, pb = gt.objects[a].pos, gt.objects[b].pos
    return np.linalg.norm(pa[:2] - pb[:2]) < 0.015 and abs(pa[2] - pb[2] - 0.04) < 0.006


@pytest.fixture
def sim(robot):
    with PandaSim(robot) as s:
        try:
            s.render()
        except VisionUnavailable as exc:
            pytest.skip(str(exc))
        yield s


def run(sim, robot, planner, task, **kw):
    out = io.StringIO()
    outcome = run_agent(task, planner, sim, robot, out=out, **{"execute": True, "vision": True, **kw})
    return outcome, out.getvalue()


def test_vision_closed_loop_stacks_cube(sim, robot):
    client = FakeVisionClient()
    planner = AstraPlanner(robot, vision=True, client=client)
    outcome, log = run(sim, robot, planner, "stack the red cube on the blue cube")
    assert outcome.status == "done", log
    assert outcome.rounds == 2
    assert stacked(sim, "red_cube", "blue_cube")
    # perception came from the image and was accurate
    errors = [p["error_cm"] for p in outcome.history[0]["perception"]]
    assert len(errors) == 3 and max(errors) < 1.0


def test_vision_reads_cube_angles_and_grips_rotated_cubes(robot):
    """Rotated layout, seen only through the camera: the 'model' reports each cube's angle from
    the image, the gripper turns to match, and the stack lines up."""
    from openrobot_brain.simulation.executor import random_layout
    with PandaSim(robot) as sim:
        try:
            sim.render()
        except VisionUnavailable as exc:
            pytest.skip(str(exc))
        sim.reset(random_layout(3, robot))
        outcome, log = run(sim, robot, AstraPlanner(robot, vision=True, client=FakeVisionClient()),
                           "stack the red cube on the blue cube")
        assert outcome.status == "done", log
        assert stacked(sim, "red_cube", "blue_cube")
        assert "closed at 4.0 cm" in log
        yaw_errors = [p["yaw_error_deg"] for p in outcome.history[0]["perception"]]
        assert max(yaw_errors) < 2.0, yaw_errors


def test_ignoring_cube_angles_is_caught_and_reported(robot):
    """If the model reports every cube as straight, the grip lands on the corners: the executor
    flags 'grasp misaligned' and that failure is fed back to the model in the next round."""
    from openrobot_brain.simulation.executor import random_layout
    with PandaSim(robot) as sim:
        try:
            sim.render()
        except VisionUnavailable as exc:
            pytest.skip(str(exc))
        layout = random_layout(3, robot)
        layout["red_cube"] = (*layout["red_cube"][:2], np.radians(35))
        sim.reset(layout)
        client = FakeVisionClient(report_yaw=False)
        outcome, log = run(sim, robot, AstraPlanner(robot, vision=True, client=client),
                           "stack the red cube on the blue cube", max_rounds=2)
        assert "grasp misaligned" in log
        assert "gripping the corners" in client.calls[1]["input"][0]["content"][0]["text"]


def test_closed_loop_recovers_from_bad_perception(sim, robot):
    """Round 1: the 'model' misreads red's position by 60 px -> grasp misses.
    The failure is reported back, round 2 retries from a fresh image and succeeds."""
    client = FakeVisionClient(bias_px=60)
    outcome, log = run(sim, robot, AstraPlanner(robot, vision=True, client=client),
                       "stack the red cube on the blue cube")
    assert outcome.status == "done", log
    assert "grasp missed" in log
    round2_prompt = client.calls[1]["input"][0]["content"][0]["text"]
    assert "closed on nothing" in round2_prompt
    assert stacked(sim, "red_cube", "blue_cube")


def test_validator_rejection_is_fed_back(sim, robot):
    client = ScriptedClient(
        respond("act", [pick("red_cube"), place_at(1.2, 0.0)]),
        respond("infeasible", message="I can't put it there; it's outside my reach."),
    )
    outcome, log = run(sim, robot, AstraPlanner(robot, vision=False, client=client), "put the red cube far away",
                       vision=False)
    assert outcome.status == "infeasible"
    assert "REJECTED" in log
    second = client.calls[1]["input"][0]["content"][0]["text"]
    assert "SAFETY SYSTEM REJECTED" in second and "outside the workspace" in second
    assert np.allclose(sim.ground_truth().objects["red_cube"].pos[:2], [0.45, 0.15], atol=0.005)  # never moved


def test_invalid_json_is_fed_back(sim, robot):
    client = ScriptedClient("Sure! Here's the plan: {", respond("done", message="Nothing to do."))
    outcome, _ = run(sim, robot, AstraPlanner(robot, vision=False, client=client), "do nothing", vision=False)
    assert outcome.status == "done"
    assert "YOUR RESPONSE WAS INVALID" in client.calls[1]["input"][0]["content"][0]["text"]


def test_repeated_rejections_stop_the_loop(sim, robot):
    outcome, log = run(sim, robot, OfflinePlanner(), "unsafe demo", vision=False)
    assert outcome.status == "rejected" and outcome.rounds == 3
    assert log.count("REJECTED") == 3


def test_dry_run_moves_nothing(sim, robot):
    before = sim.ground_truth()
    q0 = sim.q()
    client = FakeVisionClient()
    outcome, log = run(sim, robot, AstraPlanner(robot, vision=True, client=client),
                       "stack the red cube on the blue cube", execute=False)
    assert outcome.status == "dry_run" and len(client.calls) == 1
    assert "Dry run" in log
    assert np.allclose(sim.q(), q0)
    for k, o in sim.ground_truth().objects.items():
        assert np.allclose(o.pos, before.objects[k].pos)


def test_infeasible_moves_nothing(sim, robot):
    client = ScriptedClient(respond("infeasible", message="No gripper can juggle."))
    outcome, log = run(sim, robot, AstraPlanner(robot, vision=True, client=client), "juggle the cubes")
    assert outcome.status == "infeasible" and "NOT FEASIBLE" in log and "Nothing was executed" in log


def test_offline_tower_in_state_mode(robot):
    with PandaSim(robot) as sim:
        outcome, log = run(sim, robot, OfflinePlanner(), "build a tower", vision=False)
        assert outcome.status == "done", log
        assert stacked(sim, "red_cube", "blue_cube") and stacked(sim, "green_cube", "red_cube")


def test_round_limit(robot):
    with PandaSim(robot) as sim:
        client = ScriptedClient(*[respond("act", [{"action": "wait", "duration_s": 0.1}])] * 2)
        outcome, log = run(sim, robot, AstraPlanner(robot, vision=False, client=client), "wait", vision=False,
                           max_rounds=2)
        assert outcome.status == "max_rounds" and "Stopped after 2 rounds" in log


def test_images_saved(sim, robot, tmp_path):
    client = ScriptedClient(respond("done"))
    run(sim, robot, AstraPlanner(robot, vision=True, client=client), "look", save_dir=tmp_path)
    assert (tmp_path / "round1_overhead.png").read_bytes()[:4] == b"\x89PNG"


@pytest.mark.live
def test_live_astra_vision(robot):  # pragma: no cover - needs a funded key
    import os
    key = os.environ.get("ORB_LIVE_OPENAI_KEY")
    if not key:
        pytest.skip("set ORB_LIVE_OPENAI_KEY to run the live test")
    with PandaSim(robot) as s:
        outcome, log = run(s, robot, AstraPlanner(robot, vision=True, api_key=key),
                           "stack the red cube on the blue cube")
        print(log)
        assert outcome.status == "done"
