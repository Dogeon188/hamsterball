# articulated_tier.py

## Purpose

The articulated control tier ([spec](../../../.scratch/articulated-tier/spec.md), ticket 02): the hamster is a 2.4 kg jointed quadruped inside the shell, driven by one normalised torque per joint. It implements the `Tier` protocol of [step.md](step.md) beside [abstract_tier.md](abstract_tier.md).

## Interface

`ArticulatedTier` implements `Tier`:

- `fragment`: the free torso `hamster` (capsule), a `head` body (sphere, holds the site `hamster_eye`), and four legs `fl fr rl rr`, each a `<leg>_thigh` body with hinge `<leg>_hip` and a `<leg>_shin` body with hinge `<leg>_knee` and a sphere geom `<leg>_foot`. All hinges are about the torso's y axis. The torso's forward axis is +x.
- `sections`: a top-level `<actuator>` (one `motor` per joint, `ctrlrange -1 1`, gear = peak torque, in the order fl hip, fl knee, fr hip, ... rr knee) and a `<keyframe>` `stand` (shell at (0, 0, 0.5), torso standing at the height where every foot just touches the inner wall, the standing joint angles). `model.build` appends it after the bodies (see [model.md](model.md)).
- `nuserdata = 0`, `resting_radius = 0.40` m.
- `action_space()`: `Box(-1, 1, (8,), float32)`, the torques, normalised by the gears.
- `reset(model, data, spawn)`: copies the keyframe's pose (relative to the shell) to `spawn`, torso yawed so its forward axis points along the facing (`userdata[FACING]`, which `Core.reset` now sets before calling the tier), joint velocities and `ctrl` zero.
- `drive(...)`: `ctrl[:8] = action`. Touches no `xfrc_applied`.
- `obs(...)`: `{"wall_contact": bool}`, true when any hamster geom touches a tile. It runs `mj_kinematics` and `mj_collision` first, since after `mj_step` the contact list belongs to the previous state; neither writes the core state.

Play-test viewer: `uv run --extra game python -m hamsterball.game.articulated_playtest` (`game/articulated_playtest.py`, no doc of its own). Keys: W/S all hips, A/D all knees (posture offsets over a PD stand), T canned trot, F limp, Space shove-and-roll, R reset, Q/E orbit, Z/X zoom, Esc quit.

## How it works

Contacts: torso, head, thighs and shins have `condim 1`, `priority 1`, so they are frictionless against the tiles. Feet are `condim 3` spheres with a 2 mm margin; their friction mixes with the tiles' `1 .005 .0001` by max, so the feet's own friction cannot go under 1. Every hamster geom has `contype 2`, `conaffinity 0`, so they hit the tiles (`conaffinity 2`) but never each other (with `conaffinity 2` the legs collide with one another). Geoms are only capsules and spheres: MJX has sphere-box and capsule-box but not cylinder-box or ellipsoid-box. Solref is `0.04 2` on every geom, as the tiles.

Masses are set per geom so the subtree is exactly 2.4 kg: torso 1.38, head 0.30, and per leg thigh 0.10, shin 0.06, foot 0.02.

## Constants (first cut, picked to pass the tests; play-test verdict: **pending**)

| | value |
|---|---|
| torso capsule | radius 0.05, half length 0.08 m |
| head | sphere r 0.045 at (0.17, 0, 0.025) |
| hips | (±0.08, ±0.085, −0.02) from the torso centre |
| thigh, shin, foot | 0.09 m, 0.09 m, r 0.02 m (capsule radii 0.015 and 0.012) |
| hip, knee range | −0.9..0.9 rad, −1.8..0.05 rad |
| gear hip, knee | 5.0, 4.0 N m |
| joint damping, armature | 0.05, 0.002 |
| standing pose | hip 0.4, knee −0.8 on every leg (positive hip swings the foot back, negative knee bends the shin forward) |
| standing torso height | −0.27 m, i.e. the COM stands 0.27 m from the shell centre |
| resting radius | 0.40 m |

Resting radius: limp hamsters dropped in random orientations settle with the COM 0.31 to 0.38 m from the shell centre, and swing through at most 0.395 m while falling; 0.40 sits above all of them and under the 0.47 m inner wall less the torso's 0.05 m radius (0.42).

## Invariants and checks

- `uv run pytest tests/test_articulated_tier.py`: the fragment contract (free root `hamster`, 2.4 kg, group 2, `hamster_eye`, only MJX-supported joint, geom and actuator types), reset stands with its feet on the inner wall and the torso along the facing, a stored core state reproduces the run bit for bit with `ctrl` included, thrown hamsters sink no further than the hard stop plus a step (measured 0 / 20 / 20 / 20 mm for 0 / 4 / 8 / 12 m/s), a fallen hamster gets no containment impulse, and the play-test's posture loop holds the stand until a shove topples it.

## Known quirks

- The tier has no gait or balance: with zero torque it still stands on its legs for a while (the jointed legs hold the load), and a shove topples it onto its back or side where it lies at about 0.377 m.
- First-cut push measurement (PD stand plus a trot, 15 s, shell displacement): gears 3 / 2.5 N m move the shell under 0.2 m; 6 / 5 N m push it 1.7 m but the hamster tumbles; 5 / 4 N m with a hip swing of 0.5 rad or less barely moves it and above 0.7 rad tumbles. The feet push the wall at the shell's ground contact, which gives the shell almost no torque: the shell rolls only once the hamster's weight sits ahead of the contact (the rolling friction of 0.04 m takes about 1.3 N m). A real gait has to climb the wall as the abstract tier's lean does.
- `wall_contact` costs a collision pass per env step.

## Decisions

- 2026-10-27, ticket articulated-tier/02: skeleton constants above, chosen to pass the tests with no user sign-off yet.
- 2026-10-27, ticket articulated-tier/02: `Tier` gains `sections` (top-level MJCF after the bodies), since a fragment inside `<worldbody>` cannot hold an `<actuator>` or a `<keyframe>`; the abstract tier's is `""`, so its model is unchanged.
- 2026-10-27, ticket articulated-tier/02: `contype 2`, `conaffinity 0` on hamster geoms in place of the ticket's "bit 2 on both", so the legs never collide with one another.
- 2026-10-27, ticket articulated-tier/02: `Core.reset` sets the facing and pitch before `tier.reset`, so a tier can point along the facing without a new protocol argument. Abstract-tier trajectories are unchanged.

## Tried and rejected

- 2026-10-27, ticket articulated-tier/02: gears 3 / 2.5 N m, the first cut: standing is fine, but the trot moved the shell under 0.2 m in 15 s.

## History

- 2026-10-27, ticket articulated-tier/02: built with the core-seam tests of the spec. The play-test (stand, push the shell, fall naturally) awaits the user.
