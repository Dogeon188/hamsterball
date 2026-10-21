# abstract_tier.py

## Purpose

The abstract control tier of spec §5: the hamster is one 2.4 kg sphere, and a run direction and effort set a target lean that a foot force on the shell's inner wall chases.

## Interface

`AbstractTier` implements the `Tier` protocol in [step.md](step.md):

- `fragment`: a free body `hamster` with one frictionless sphere (`condim 1`, `priority 1`, solref `0.04 2`, 2 mm margin, group 2) and a site `hamster_eye` at its centre.
- `nuserdata = 1`: `userdata[1]` holds the last wall-touch time.
- `resting_radius = 0.35` m.
- `action_space()`: `Box(-1, 1, (2,), float32)` = `[run_x, run_y]` in the camera yaw frame.
- `reset(model, data, spawn)`: the hamster 1 cm above the bottom of a shell centred at `spawn`, with the last touch set to now.
- `drive(model, data, action, yaw)`: the airborne cut, then the foot-force pair and grip couple into `xfrc_applied`.
- `obs(model, data, yaw)`: `{"wall_contact": bool}`, true inside the 45 ms grace.

## How it works

`drive()` follows spec §5 step by step, in hand-written scalar maths: the wall gap over the 320 tile normals (one NumPy mat-vec), the target lean `sin(70°·e)·d + cos(70°·e)·down`, the PD foot force with its normal part removed and its size capped, the pair applied at the foot as forces plus torques about each centre of mass, and the grip couple on the world-frame angular velocities. Velocities come straight from the free joints' `qvel`; positions from `xpos`, `xipos`, and `xmat`, which the core refreshes before calling it.

## Invariants and checks

- The pair adds no net force and no net torque about any point.
- `uv run pytest tests/test_feel.py` checks the §5 feel reference.

## Known quirks

- The PD alone holds about 54° against the 70° ceiling: there is no gravity feed-forward, by decision.
- With the hamster exactly at the shell centre, `n` is the zero vector (the `+1e-9` guard), as in the prototype.

## Decisions

- 2026-10-21, issue 13 verdict (play-tested 2026-10-20), recorded in `prototypes/honest_drive.md` on `prototype/honest-physics`: `foot+grip` pair, 2.4 / 0.8 kg, 70° lean ceiling, no feed-forward, gains and cap scaled by hamster mass.
- 2026-10-21, issue 16: the constants hold at 1/600 s unchanged.

## Tried and rejected

See the prototype's doc (`prototypes/honest_drive.md`, branch `prototype/honest-physics`) for the offset and centre pairs, hamster–wall friction, other masses and lean ceilings, and the fade cap.

## History

- 2026-10-21, ticket hamsterball-v1/01: ported from the prototype's `control()` (the `foot+grip` branch).
- 2026-10-22, ticket hamsterball-v1/01: the feel test reproduces spec §5's drive rows within 1.3%; the measured table is in [step.md](step.md), History.
