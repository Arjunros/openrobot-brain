# Architecture

## Principle

The language model is a proposal generator. Its output is untrusted text. Everything that decides whether and how the robot moves is local, deterministic, and tested.

## One round of the loop (`agent.py`)

| Step | Module | In → Out | On failure |
|---|---|---|---|
| Observe | `simulation/executor.py` | sim → overhead PNG (vision) or object list (state) | `VisionUnavailable` → suggest `--no-vision` |
| Plan | `planner.py` | prompt + image → JSON text | `PlannerError`, round stops |
| Parse | `schema.py` | text → `AgentResponse` | `PlanParseError`, fed back to the model |
| Ground | `perception.py` | observations (pixels) → `WorldState` (metres) | warnings; bad objects dropped |
| Expand | `skills.py` | actions + belief → timed joint `Segment`s | precondition violations |
| Validate | `safety.py` | segments → `ValidatedPlan` | `SafetyError`, fed back, nothing moves |
| Execute | `simulation/executor.py` | `ValidatedPlan` → `RoundResult` | stops on missed grasp, drop, or blocked arm |

After 3 consecutive invalid or rejected rounds, or the round limit (8), the loop stops.

## Planner (`planner.py`)

`AstraPlanner` calls `client.responses.create` with:

- `model="gpt-6-astra"` (override with `--model` or `OPENAI_MODEL`; never substituted automatically);
- `instructions`: robot, frame conventions, image conventions, the action list with its limits, and the status protocol;
- `input`: the round prompt (task, round number, gripper state, last 4 rounds of actions, outcomes and rejections) plus the overhead image as a base64 PNG (`detail: high`);
- `text.format`: strict `json_schema` (`schema.response_json_schema()`);
- `reasoning.effort` low…max, `store=False`, at most 1 SDK retry.

`OfflinePlanner` is a short deterministic script that reads simulator ground truth. It exists for demos and CI and is labelled "not AI" in the CLI and videos.

## Perception (`perception.py`, `simulation/camera.py`)

The overhead camera is a fixed pinhole camera (640×480, 38° vertical FOV, 1.2 m above the table, looking straight down). The model reports each object's pixel centre of its visible top face and what it is stacked on. `perceive()` intersects that pixel's viewing ray with the plane at the object's top-face height to get metres. Stack levels come from the `on_top_of` chain; unknown or circular references are warned about and treated as "on the table".

An object held in the gripper is not visible (the arm parks out of view), so the belief carries it over from the previous round.

`score_perception()` compares the belief with simulator ground truth (nearest object, distance, colour word match). This is reporting only and is never used for control in vision mode.

## Skills (`skills.py`)

Each action becomes a list of `Segment`s: dense joint waypoints with timestamps, a gripper command, and the predicted world during the segment.

- `pick`: leave the park pose → rise to transit height (16 cm) → move above the object → descend → close the gripper → lift.
- `place_on` / `place_at`: carry at transit height → descend to 3 mm above the release height → open → retreat.
- `move_to`: rise → move above → descend.
- `go_home` and the automatic end of every round: rise → joint-space move to the park pose.

Cartesian segments follow straight lines with a minimum-jerk profile (peak 0.25 m/s in transit, 0.10 m/s vertically), sampled every ~5 mm, with IK chained from the previous sample. IK is damped least squares on position and orientation (gripper straight down) with a null-space pull toward a comfortable posture, clamped to the software joint limits.

## Safety (`safety.py`)

See [safety.md](safety.md). `ValidatedPlan` can only be constructed by `validate_round` (token-guarded), it is single-use, and it records the gripper state it was validated with.

## Simulation (`simulation/`)

- `orb_tabletop.xml`: the Menagerie Panda, a work mat, three free-floating 4 cm cubes, and two cameras (`overhead` for the model, `front` for videos). Physics timestep 2 ms, elliptic friction cones.
- `PandaSim.execute` streams the trajectory to the position actuators at the physics rate. After "close gripper" it reads the finger width: below 1 cm means the gripper closed on nothing ("grasp missed"). It also checks the object is still held after lifting, and aborts if the arm lags its command by more than 0.12 rad (something unexpected is blocking it).
- Viewer is optional (`launch_passive`); if it fails, the run continues headless. `--speed` paces the viewer and video; physics is unaffected.
- `Recorder` renders the `front` camera with captions (round, model message, current motion, validation badge) to MP4 (imageio/ffmpeg) or GIF (Pillow).

## Extending

- **New object types:** extend `WorldState` with sizes and shapes, the scene XML, and the grasp heights in `skills.py`.
- **Grasp orientation:** add a yaw parameter to `pick` and rotate `R_DOWN`; the validator needs no change.
- **Real hardware:** out of scope by design. It would need its own executor behind the same `ValidatedPlan` boundary, plus hardware-level limits, e-stop, and perception that does not rely on the simulator.
