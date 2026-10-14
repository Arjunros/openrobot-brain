# OpenRobot-Brain

**Natural language → Gemini 4 Argon → validated action plan → MuJoCo simulation.**

OpenRobot-Brain turns an instruction like *"wave hello"* into a joint-space plan for a simulated 4-DOF arm. The language model only *proposes* a plan. Local code parses it strictly, validates it against the robot's limits as a sequence, shows it to you, and runs it in MuJoCo only when you ask.

```
"wave hello"
   │
   ▼
Gemini 4 Argon (Interactions API, structured JSON output)      ← untrusted
   │  raw JSON text
   ▼
schema.parse_plan_json   strict Pydantic: allowlisted actions, no extra fields, finite numbers
   │  ActionPlan
   ▼
safety.validate_plan     joint names, limits, per-step delta, speed, durations — tracked step by step
   │  ValidatedPlan (only this type can reach the executor)
   ▼
dry-run preview  ── default, stops here
   │  --execute
   ▼
MujocoExecutor           minimum-jerk interpolation, headless or viewer
```

> **Scope (Milestone 1).** Simulation only. The arm has no gripper, camera or inverse kinematics, and the project does not claim grasping, perception or IK. There is no code path to real hardware.

## Requirements

- Windows 10/11 (also runs on Linux/macOS), Python 3.11+
- For live planning: a Gemini API key on the **paid tier** (Gemini 4 Argon isn't on the free tier). Everything else, including the offline demo and the full test suite, works without a key.

## Install (PowerShell)

```powershell
git clone https://github.com/<your-username>/openrobot-brain.git
cd openrobot-brain
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e ".[dev]"
```

If activation is blocked: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`.

Set up your key (optional):

```powershell
Copy-Item .env.example .env
notepad .env        # set GEMINI_API_KEY=...
```

## Run

```powershell
orb sim-check                                        # load scene, verify joints/actuators, headless motion test
orb demos                                            # list offline demo tasks
orb plan "wave" --offline                            # offline planner, dry run (default)
orb plan "wave" --offline --execute                  # run in the MuJoCo viewer
orb plan "wave" --offline --execute --headless       # run without a window
orb plan "unsafe demo" --offline --execute           # watch the validator reject a plan

orb plan "turn left, pause, then come back" --execute          # Gemini 4 Argon
orb plan "nod twice slowly" --thinking medium --save plan.json # keep the raw model output
orb validate examples\turn_left.json --execute
orb validate examples\unsafe_sequence.json                     # rejected
```

Exit codes: `0` ok, `2` plan rejected, `3` planner/credential error, `4` simulation error.

## Test

```powershell
pytest                       # unit + simulation tests, no API key needed
$env:ORB_LIVE_KEY = "your-key"; pytest -m live   # optional live Gemini round-trip
```

## Project layout

```
src/openrobot_brain/
  config.py            robot joints, software limits, safety limits
  schema.py            strict Pydantic action schema + JSON parsing
  safety.py            independent, sequence-aware validator → ValidatedPlan
  planner.py           Gemini 4 Argon adapter + offline demo planner
  cli.py               `orb` command
  simulation/
    orb_arm.xml        4-DOF arm, position actuators
    executor.py        MuJoCo-only executor, smoke check
tests/                 pytest suite (mocked Gemini, real MuJoCo)
docs/architecture.md   design and data flow
docs/safety.md         threat model and every check
examples/              hand-written plans for `orb validate`
```

## How the Gemini call works

`gemini-4-argon` is served through Google's Interactions API. The planner calls `client.interactions.create` with a system instruction describing the arm and its limits, a JSON schema in `response_format`, `generation_config={"thinking_level": ...}`, and `store=False`. Sampling parameters are not sent because Argon ignores them. The model ID defaults to `gemini-4-argon`; set `GEMINI_MODEL` or `--model` to change it. It is never substituted automatically. If the model isn't available to your key, the CLI says so and stops.

## License

Apache-2.0. See [LICENSE](LICENSE).
