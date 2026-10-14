# Safety

This is a simulation project. These checks keep plans physically sensible and keep model output from doing anything it shouldn't. They are **not** a certified safety system and are not sufficient for real hardware.

## Threat model

Model output is treated as untrusted input. It may be malformed, may include fields we didn't ask for, may name joints that don't exist, may contain NaN or huge numbers, may describe motions that are individually fine but dangerous in sequence, and may assert that it is safe. None of that is trusted.

What the system never does with model output:

- No `eval`/`exec`, no generated code, no shell commands, no tool calls. The only effect a plan can have is joint setpoints in MuJoCo.
- No model-provided field can switch off or relax a check. The schema has no "safe" or "verified" field, and unknown fields are rejected.

## Layer 1: parsing (`schema.py`)

| Check | Behaviour |
|---|---|
| Empty / non-JSON / markdown-wrapped / non-object | rejected |
| `NaN`, `Infinity`, `-Infinity`, overflow like `1e999` | rejected during JSON decoding |
| Unknown `action` (only `move_joints`, `wait`) | rejected |
| Missing required field | rejected |
| Extra field at any level | rejected |
| Wrong type (string number, bool, null) | rejected (strict mode) |
| Plan with 0 actions, move with 0 targets | rejected |

## Layer 2: safety validation (`safety.py`)

The validator walks the plan in order and keeps a **predicted joint state**. Each move is checked against where the arm will be after the previous action, not against the start pose. Joints not mentioned in a move keep their predicted value.

| Check | Default |
|---|---|
| Joint name exists in config | 4 joints |
| Same joint twice in one move | rejected |
| Target inside software range | per joint, tighter than the MJCF range |
| Change per joint per move | ≤ 0.8 rad |
| Speed (change / duration) | ≤ 1.0 rad/s |
| Move duration | 0.2 – 5.0 s |
| Wait duration | > 0, ≤ 3.0 s |
| Number of actions | ≤ 12 |
| Total duration | ≤ 30 s |
| Start state | must define every joint and be in range |

All violations are collected and reported together. Any violation rejects the whole plan before anything runs.

Example (`examples/unsafe_sequence.json`): `shoulder_yaw → 0.7`, then `→ -0.7`. Each target is 0.7 rad from home, so a validator that compared to the start pose would pass it. Ours sees a 1.4 rad jump and rejects it.

## Layer 3: execution (`simulation/executor.py`)

- Accepts only `ValidatedPlan` (constructor is token-guarded).
- Refuses if the simulator's current state differs from the plan's validated start state by more than 0.05 rad.
- Smooth minimum-jerk interpolation; no setpoint jumps.
- Software limits are verified to sit inside the MJCF joint ranges and actuator `ctrlrange` at load time.
- Stops if a step ends more than 0.05 rad from its target.
- MuJoCo only. No hardware drivers exist in this codebase.

## Defaults

- Dry run is the default; `--execute` is required to start the simulator.
- The validated plan is printed before anything runs.

## Secrets

- The key is read from `GEMINI_API_KEY` / `GOOGLE_API_KEY` or a git-ignored `.env`.
- Error messages pass through `redact()`, which strips the configured key and anything shaped like a Google API key.
- Requests are sent with `store=False`.
- Unit tests remove key variables from the environment and use a mocked client.
