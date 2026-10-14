# Architecture

## Design principle

The language model is a *proposal generator*, not a controller. Everything it returns is untrusted text. Authority over what moves sits entirely in local, deterministic, tested code.

## Data flow

| Stage | Module | Input → Output | Fails with |
|---|---|---|---|
| Plan | `planner.py` | task string → raw JSON text | `PlannerError`, `MissingCredentialsError` |
| Parse | `schema.py` | text → `ActionPlan` | `PlanParseError` |
| Validate | `safety.py` | `ActionPlan` + start state → `ValidatedPlan` | `SafetyError` (all violations listed) |
| Preview | `cli.py` | `ValidatedPlan` → human-readable table | — |
| Execute | `simulation/executor.py` | `ValidatedPlan` → `ExecutionResult` | `SimulationError` |

Each stage only accepts the previous stage's output type. `ValidatedPlan` can only be constructed inside `safety.validate_plan` (a private token guards the constructor), and `MujocoExecutor.execute` type-checks for it. An unvalidated plan has no path to the simulator.

## Planners

All planners implement `plan_text(task) -> str`, so they all go through the same parser and validator.

- **`GeminiPlanner`** — `google-genai` ≥ 2.3, Interactions API (`client.interactions.create`). Sends a system instruction with joint names, ranges, home pose and every safety limit; requests structured JSON via `response_format` with a schema whose joint names are an enum; `thinking_level` low/medium/high; `store=False`. The client is injectable, which is how tests mock it.
- **`OpenAIPlanner`** — `openai` SDK, Responses API (`client.responses.create`) with `gpt-6-astra`. Same system instruction as Gemini, sent as `instructions`; strict structured output via `text.format` (`json_schema`, `strict: true`). Strict mode accepts only a JSON-schema subset, so `openai_response_schema` drops array-size keywords; local validation still enforces them. `reasoning.effort` low…max; `store=False`; SDK retries capped at 1 so quota errors show up quickly. Incomplete responses are treated as failures.
- **Provider selection** — `make_planner(provider, ...)`; `--provider` > `ORB_PROVIDER` > whichever key is set (Gemini first).
- **`OfflinePlanner`** — keyword lookup over a few hand-written plans. It is not an AI planner; it exists so the demo and CI work without a key. One entry (`unsafe demo`) is deliberately unsafe to show rejection.

## Schema

`ActionPlan = {summary: str, actions: [MoveJoints | Wait]}` with a discriminator on `action`.

- `MoveJoints`: `targets: [{joint, position_rad}]`, `duration_s`
- `Wait`: `duration_s`

All models use `extra="forbid"`, `strict=True`, and finite floats. Joint targets are a list of objects rather than a dict, because that maps cleanly onto JSON-schema structured output. Units are radians and seconds throughout.

## Simulation

`orb_arm.xml` defines a 4-hinge arm (`shoulder_yaw`, `shoulder_pitch`, `elbow_pitch`, `wrist_pitch`) with one position actuator per joint and gravity compensation on the links. `bind_joints` maps each configured joint to its `qpos` index and actuator by inspecting the compiled model (not by assuming order), and fails if a joint is missing, has zero or several actuators, or if the software range in `config.py` isn't inside both the joint range and the actuator `ctrlrange`.

The executor streams setpoints along a minimum-jerk curve between validated waypoints at the physics rate (500 Hz), lets each move settle for 0.25 s, and stops if the arm ends a step more than 0.05 rad from its target. Before running, it checks that the simulator's current state matches the start state the plan was validated against; otherwise it refuses and asks for re-validation.

The viewer is optional. If `mujoco.viewer.launch_passive` fails (no display, missing OpenGL), the executor records the reason and continues headless. Resources are released through the context manager.

## Extending

- **New robot:** add a `RobotConfig` and an MJCF file; `bind_joints` and the tests will catch mismatches.
- **New action:** add a Pydantic model to the `Action` union, a branch in `validate_plan`, a branch in the executor, and the schema entry in `gemini_response_schema`. Do all four or the action stays unreachable.
- **Real hardware:** intentionally out of scope. It would need its own executor behind the same `ValidatedPlan` boundary, plus hardware-level limits, e-stop and watchdogs that don't depend on this code.
