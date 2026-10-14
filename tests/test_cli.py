import pytest

from openrobot_brain import cli


def test_demos(capsys):
    assert cli.main(["demos"]) == 0
    assert "build a tower" in capsys.readouterr().out


def test_offline_dry_run(capsys):
    assert cli.main(["run", "stack the red cube on the blue cube", "--offline"]) == cli.EXIT_OK
    out = capsys.readouterr().out
    assert "offline scripted demo" in out and "Validated" in out and "Dry run" in out


def test_plan_alias_and_offline_execute_headless(capsys):
    assert cli.main(["plan", "stack the red cube on the blue cube", "--offline", "--execute", "--headless"]) == 0
    assert "DONE" in capsys.readouterr().out


def test_offline_unsafe_exit_code(capsys):
    assert cli.main(["run", "unsafe demo", "--offline", "--execute", "--headless"]) == cli.EXIT_REJECTED


def test_offline_infeasible_exit_code(capsys):
    assert cli.main(["run", "juggle", "--offline", "--execute", "--headless"]) == cli.EXIT_NOT_DONE
    assert "NOT FEASIBLE" in capsys.readouterr().out


def test_missing_key(capsys, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)  # no stray .env
    assert cli.main(["run", "stack red on blue"]) == cli.EXIT_PLANNER
    assert "OPENAI_API_KEY" in capsys.readouterr().out


def test_sim_check(capsys):
    assert cli.main(["sim-check"]) == 0
    assert "sim-check OK" in capsys.readouterr().out


def test_snapshot(tmp_path, capsys):
    code = cli.main(["snapshot", "--seed", "3", "--out", str(tmp_path)])
    if code == cli.EXIT_SIM:
        pytest.skip("offscreen rendering unavailable")
    assert (tmp_path / "overhead.png").exists() and (tmp_path / "front.png").exists()


def test_seeded_layout_runs(capsys):
    assert cli.main(["run", "build a tower", "--offline", "--execute", "--headless", "--seed", "4"]) == 0
