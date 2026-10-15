# Safety

This is a simulation project. These checks keep plans physically sensible and keep model output from doing anything it shouldn't. They are **not** a certified safety system and are not sufficient for real hardware.

## Threat model

Model output is untrusted. It may be malformed, add fields, invent objects, misread the image, ask for unreachable or colliding motions, repeat a rejected plan, or claim success it can't see.

The model never supplies:

- code, shell commands, or tool calls (the schema has no such fields; the API request has no tools);
- joint angles or raw trajectories (only seven high-level actions);
- object coordinates in vision mode (only pixels, converted locally);
- any flag that skips or relaxes a check (unknown fields are rejected).

## Layer 1: schema (`schema.py`)

| Check | Result |
|---|---|
| Not JSON, not an object, empty | rejected |
| `NaN`, `Infinity`, overflow (`1e999`) | rejected during decoding |
| Unknown action, missing or extra field (at any level) | rejected |
| Wrong type (string number, bool) | rejected (strict mode) |
| `act` without actions, `done`/`infeasible` with actions | rejected |
| Duplicate observation ids, malformed ids, > 16 actions | rejected |

## Layer 2: skill preconditions (`skills.py`)

| Check | Rule |
|---|---|
| Unknown object id | rejected, known ids listed |
| `pick` while holding | rejected |
| `pick` an object with something on top | rejected |
| `place_*` while empty-handed | rejected |
| `place_on` itself, or on an object that already has something on top | rejected |
| Stack height | at most 3 |
| `place_at` | inside workspace, at least 6 cm from other objects and 8 cm from fixed obstacles |
| `place_in` | known zone; first of its 4 spots that passes the `place_at` rules; a full zone is rejected |
| `move_to` | inside workspace, 1.5 cm ≤ z ≤ 60 cm |
| `wait` | 0 < duration ≤ 3 s |

## Layer 3: trajectory validation (`safety.py`)

Every sample of the expanded trajectory is checked against the world **belief**:

| Check | Limit |
|---|---|
| Actions per round | ≤ 6 |
| Start configuration | inside software joint limits |
| Joint limits | Panda hard limits shrunk by 0.05 rad |
| Joint step between samples | ≤ 0.15 rad (catches IK branch flips) |
| Joint speed | ≤ 1.2 rad/s |
| IK accuracy | ≤ 3 mm and ≤ 0.03 rad at every Cartesian sample |
| TCP workspace | x 0.32–0.70 m, y −0.30–0.30 m, z 0.015–0.60 m |
| Sideways motion | only at or above 16 cm |
| Arm vs table | no contact |
| Arm or held object vs fixed obstacles | at least 1.5 cm clearance (contact margin on a private copy of the model; physics is unaffected) |
| Arm vs objects | contact only with the object being picked or placed (each object at its believed angle) |
| Held object vs world | contact only with its place target (or the table for `place_at`) |
| Round motion time | ≤ 60 s |

All violations are collected and reported together. Any violation rejects the whole round, nothing moves, and the reasons are sent back to the model.

## Layer 4: execution (`simulation/executor.py`)

- Accepts only `ValidatedPlan` (token-guarded constructor).
- Single use: a plan cannot be replayed after the world has changed.
- Refuses if the arm's joints or the gripper state differ from what the plan was validated with.
- Stops if the gripper closes on nothing (< 1 cm), closes wider than the cube + 6 mm (gripping the corners: "grasp misaligned"), the object slips during the lift, or the arm lags its command by more than 0.12 rad.
- MuJoCo only. No hardware drivers exist in this codebase.

## Loop limits (`agent.py`)

- At most 8 rounds.
- Stops after 3 consecutive invalid or rejected rounds.
- Dry run (no `--execute`) plans and validates round 1 only.
- "Done" is the model's claim. An independent goal check (`goals.py`) reads simulator ground truth and prints GOAL MET / NOT MET, with a warning when the model said "done" but the goal is not met. It reports; it never moves the robot.

## Known limitations

- Validation trusts the **belief**. If perception is wrong, a plan can be valid against the belief and still fail physically. Missed grasps and drops are detected and fed back, but a misplaced object could be pushed. On real hardware this would need force limits and independent sensing.
- Robot self-collision is not checked (the Menagerie collision meshes overlap at adjacent links); the posture bias keeps the arm in a self-collision-free region in practice.
- Collision checks use at most three cube bodies; extra perceived objects beyond three are not represented in the collision scene.

## Secrets

- The key is read from `OPENAI_API_KEY` or a git-ignored `.env`.
- Error messages pass through `redact()`, which removes the key and anything shaped like `sk-…`.
- Requests use `store=False`. Tests remove key variables and use fake clients.
