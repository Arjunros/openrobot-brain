import io
import json

import pytest

from openrobot_brain import cli
from openrobot_brain.planner import OFFLINE_PLANS
from conftest import move, plan_json


class SpyExecutor:
    instances = []

    def __init__(self, robot, headless):
        self.executed = []
        self.viewer_error = None
        SpyExecutor.instances.append(self)

    def __enter__(self):
        return self

    def __exit__(self, *a):
        pass

    def execute(self, vp):
        from openrobot_brain.simulation.executor import ExecutionResult
        self.executed.append(vp)
        return ExecutionResult(success=True, final_state=dict(vp.final_state), message="ok")


@pytest.fixture(autouse=True)
def reset_spy():
    SpyExecutor.instances.clear()


def run(text, **kw):
    out = io.StringIO()
    code = cli.run_pipeline(text, out=out, executor_factory=SpyExecutor, **kw)
    return code, out.getvalue()


def test_dry_run_is_default_and_never_creates_executor():
    code, out = run(plan_json(move(1.0, shoulder_yaw=0.3)))
    assert code == cli.EXIT_OK
    assert "Dry run" in out and "shoulder_yaw +0.00→+0.30" in out
    assert SpyExecutor.instances == []


def test_execute_flag_runs_validated_plan():
    code, out = run(plan_json(move(1.0, shoulder_yaw=0.3)), execute=True)
    assert code == cli.EXIT_OK
    assert len(SpyExecutor.instances) == 1 and len(SpyExecutor.instances[0].executed) == 1


@pytest.mark.parametrize("text", [
    "garbage",
    plan_json({"action": "teleport", "duration_s": 1}),
    plan_json(move(1.0, gripper=0.2)),
    plan_json(move(1.0, shoulder_yaw=0.7), move(1.0, shoulder_yaw=0.7), move(2.0, shoulder_yaw=-0.7)),
    json.dumps(OFFLINE_PLANS["unsafe demo"]),
    '{"actions":[{"action":"wait","duration_s":NaN}]}',
])
def test_invalid_plans_never_reach_executor(text):
    code, out = run(text, execute=True)
    assert code == cli.EXIT_REJECTED
    assert "nothing was executed" in out
    assert SpyExecutor.instances == []


def test_main_offline_dry_run(capsys):
    assert cli.main(["plan", "wave", "--offline"]) == cli.EXIT_OK
    out = capsys.readouterr().out
    assert "offline demo planner" in out and "Dry run" in out


def test_main_without_key_fails_cleanly(capsys, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)  # make sure no stray .env is picked up
    assert cli.main(["plan", "wave"]) == cli.EXIT_PLANNER
    assert "GEMINI_API_KEY" in capsys.readouterr().out


def test_main_offline_execute_headless_real_mujoco(capsys):
    assert cli.main(["plan", "nod", "--offline", "--execute", "--headless"]) == cli.EXIT_OK
    assert "plan executed in simulation" in capsys.readouterr().out


def test_validate_file(tmp_path, capsys):
    f = tmp_path / "p.json"
    f.write_text(plan_json(move(1.0, wrist_pitch=0.4)))
    assert cli.main(["validate", str(f)]) == cli.EXIT_OK


def test_sim_check_command(capsys):
    assert cli.main(["sim-check"]) == cli.EXIT_OK
    assert "sim-check OK" in capsys.readouterr().out
