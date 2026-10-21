# step.py

## Purpose

The core step of spec §4: the per-physics-step code (camera yaw, shell drag, containment, the tier's drive) around `mj_step`, the `Tier` protocol of spec §6, and `Core`, which holds a compiled model and its data.

## Interface

- `Core(tier=None, level=FLAT, camera_action=False)`: compiles the model (default tier: `AbstractTier`) and resets.
- `Core.reset(spawn=(0, 0, 0.5), yaw=0.0)`, `Core.step(action, n=FRAME_SKIP)` (one action held over `n` physics steps), `Core.state()` and `Core.set_state(s)` (the `mjSTATE_INTEGRATION` vector), `Core.yaw`, `Core.action_space` (the tier's, plus `cam_yaw` under the camera flag).
- `physics_step(model, data, tier, action, cam)` and its parts `camera_yaw`, `drag`, `contain`, `wrap`.
- `Tier`: `fragment`, `nuserdata`, `resting_radius`, `action_space()`, `reset()`, `drive()`, `obs()`. The core touches a tier only through these.

## How it works

Each physics step: clear `xfrc_applied`; refresh positions (`mj_kinematics`, `mj_comPos`); camera yaw into `userdata[0]`; drag `−k|v|v` on the shell; containment on the hamster root's free joint (an inelastic impulse from 5 mm past the resting radius, a position stop and `mj_forward` past 20 mm); `tier.drive()`; `mj_step`.

The position refresh is what makes the core state complete. `mj_step` leaves `xpos` and friends computed for the state before it, so code that read them directly would depend on the previous step, and a restored state would step differently. Refreshing costs two cheap kinematics passes and means every step reads only `qpos`, `qvel`, `userdata`, and `time`.

Kernel shape: ids and constants are module constants, small-vector maths is written out in Python floats, and no function keeps Python-side state. `Core` holds only the model, the data, the tier, and the camera flag.

## Invariants and checks

- `uv run pytest tests/test_core_state.py`: setting a stored core state and stepping reproduces the run bit for bit.
- `uv run pytest tests/test_feel.py`: the spec §5 feel reference.

## Known quirks

- Containment is a step-level state edit, not a MuJoCo constraint, so one step can still carry the hamster past the 20 mm stop by that step's travel. Measured worst sinks for throws at 0 / 4 / 8 / 12 m/s are 14 / 17 / 20 / 23 mm, under spec §5's 16 / 21 / 30 / 36 mm, because this core reads fresh positions (see History, 2026-10-22).
- The chase yaw follows any shell motion above 0.8 m/s, backwards included. Driving up a 20° slope from rest, the shell first rolls back at up to 1.4 m/s, and the yaw swings about 56° toward the downhill heading, which turns a held "forward" off the slope: the climb drops from 5.17 m to 4.34 m in 3 s. The feel test turns the chase off to measure the drive alone.
- The Fibonacci tiles aren't isotropic: the uncapped 1–3 s acceleration is 4.05 m/s² along +x and 4.16 along +y. The feel test runs along +x, as the prototype did.

## Decisions

- 2026-10-21, issue 13 verdict (play-tested 2026-10-20), recorded in `prototypes/honest_drive.md` on `prototype/honest-physics`: drag `k = 0.059` on the shell, not gated on ground contact; `field` containment with a 5 mm start and a 20 mm hard stop.
- 2026-10-21, ticket hamsterball-v1/01: the per-step code refreshes positions with `mj_kinematics` and `mj_comPos` before reading them, so the core state alone determines the next step.

## Tried and rejected

See the prototype's doc for containment by solimp, a spring-damper field, and a contact-gated drag.

## History

- 2026-10-21, ticket hamsterball-v1/01: ported from the prototype's `control()`, with the camera yaw rewritten to spec §4 (the prototype chased the free camera's azimuth in its UI loop).
- 2026-10-22, ticket hamsterball-v1/01, agent measurement (`uv run pytest -s tests/test_feel.py`, M2, MuJoCo 3.16.0, Python 3.11 and 3.14 alike), against spec §5:

  | Measurement | Spec §5 | Core | Drift |
  |---|---|---|---|
  | 1–3 s acceleration, full effort | 4.06 m/s² | 4.05 | −0.3% |
  | Speed at 3 s, half / full effort | 5.6 / 9.6 m/s | 5.64 / 9.72 | +0.7% / +1.3% |
  | Mean lean, 2–3 s at full effort | 54° | 53.9° | −0.2% |
  | Top speed, flat, full effort | 14.3 m/s | 14.31 | +0.1% |
  | Coast loss from top speed | 2.49 m/s² | 2.49 | 0.0% |
  | Distance climbed up 20° in 3 s | 5.1 m | 5.17 | +1.3% |
  | Worst sink, throw at 0 / 4 / 8 / 12 m/s | 16 / 21 / 30 / 36 mm | 14 / 17 / 20 / 23 | see below |

  The prototype re-run at 1/600 s on the same machine gives 4.06, 5.64 / 9.66, 53.9°, 14.29, 2.49, 5.15, and 16 / 21 / 30 / 36 mm. The feel test asserts the drive rows within 2% of the spec and the sinks within 1.5 mm of the measured values.

  A first run measured along +y with the chase yaw on gave 4.25 m/s² and a 4.34 m climb. The causes were the tile anisotropy and the chase yaw on the climb (Known quirks), not the drive.

  The sink drift has one cause: the prototype read `xpos` and `xipos` one step stale, so containment saw the hamster where it was before the last step. With the position refresh turned off, this core sinks 16 / 21 / 31 / 40 mm, the prototype's numbers. The fresh read is what makes the core state complete (Decisions, 2026-10-21), so it stays, and the sink row of spec §5 is the one to update.
