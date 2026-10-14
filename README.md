# OpenRobot-Brain

**Natural language → Gemini 4 Argon or GPT-6 Astra → validated action plan → MuJoCo simulation.**

OpenRobot-Brain turns an instruction like *"wave hello"* into a joint-space plan for a simulated 4-DOF arm. The language model only *proposes* a plan. Local code parses it strictly, validates it against the robot's limits as a sequence, shows it to you, and runs it in MuJoCo only when you ask.

```
"wave hello"
   │
   ▼
Gemini 4 Argon  or  GPT-6 Astra   (structured JSON output)     ← untrusted
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
- For live planning, **one** of:
  - a Gemini API key on the **paid tier** (Gemini 4 Argon isn't on the free tier), or
  - an OpenAI API key with credit on the account (GPT-6 Astra; API billing is separate from ChatGPT Plus/Pro).

  Everything else, including the offline demo and the full test suite, works without a key.

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
notepad .env        # set GEMINI_API_KEY=... and/or OPENAI_API_KEY=...
```

## Run

```powershell
orb sim-check                                        # load scene, verify joints/actuators, headless motion test
orb demos                                            # list offline demo tasks
orb plan "wave" --offline                            # offline planner, dry run (default)
orb plan "wave" --offline --execute                  # run in the MuJoCo viewer
orb plan "wave" --offline --execute --headless       # run without a window
orb plan "unsafe demo" --offline --execute           # watch the validator reject a plan

orb plan "turn left, pause, then come back" --execute                     # Gemini 4 Argon
orb plan "turn left, pause, then come back" --provider openai --execute   # GPT-6 Astra
orb plan "nod twice slowly" --thinking medium --save plan.json # keep the raw model output
orb validate examples\turn_left.json --execute
orb validate examples\unsafe_sequence.json                     # rejected
```

Exit codes: `0` ok, `2` plan rejected, `3` planner/credential error, `4` simulation error.

## Test

```powershell
pytest                       # unit + simulation tests, no API key needed
$env:ORB_LIVE_KEY = "your-gemini-key"; pytest -m live          # optional live Gemini round-trip
$env:ORB_LIVE_OPENAI_KEY = "your-openai-key"; pytest -m live   # optional live OpenAI round-trip
```

## Project layout

```
src/openrobot_brain/
  config.py            robot joints, software limits, safety limits
  schema.py            strict Pydantic action schema + JSON parsing
  safety.py            independent, sequence-aware validator → ValidatedPlan
  planner.py           Gemini 4 Argon + GPT-6 Astra adapters, offline demo planner
  cli.py               `orb` command
  simulation/
    orb_arm.xml        4-DOF arm, position actuators
    executor.py        MuJoCo-only executor, smoke check
tests/                 pytest suite (mocked Gemini, real MuJoCo)
docs/architecture.md   design and data flow
docs/safety.md         threat model and every check
examples/              hand-written plans for `orb validate`
```

## Planners

| Provider | Default model | API | Structured output | Effort flag |
|---|---|---|---|---|
| `gemini` | `gemini-4-argon` | Interactions API (`client.interactions.create`) | `response_format` JSON schema | `--thinking low/medium/high` |
| `openai` | `gpt-6-astra` | Responses API (`client.responses.create`) | `text.format` json_schema, `strict: true` | `--thinking low/medium/high/xhigh/max` |

Pick one with `--provider`, or set `ORB_PROVIDER` in `.env`. Without either, the CLI uses Gemini if `GEMINI_API_KEY` is set, otherwise OpenAI. Both get the same system prompt and schema, and both outputs go through the same parser and safety validator, so neither model is trusted more than the other. Sampling parameters (`temperature`, `top_p`) aren't sent; both models ignore or reject them. Requests use `store=False`. Model IDs are never substituted automatically; override with `--model` or `GEMINI_MODEL` / `OPENAI_MODEL`.

## License

Apache-2.0. See [LICENSE](LICENSE).
