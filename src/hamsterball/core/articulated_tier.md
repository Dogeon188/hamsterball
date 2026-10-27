# articulated_tier.py

## Purpose

The articulated control tier ([spec](../../../.scratch/articulated-tier/spec.md), ticket 02): the hamster is a 2.4 kg jointed quadruped inside the shell, driven by one normalised torque per joint. It implements the `Tier` protocol of [step.md](step.md) beside [abstract_tier.md](abstract_tier.md).

## Interface

`ArticulatedTier` implements `Tier`:

- `fragment`: the free torso `hamster` (a spine capsule plus a shoulder bar and a pelvis bar, capsules along y through the hips, so every leg attaches to solid body), a `head` body (sphere, holds the site `hamster_eye`), and four legs `fl fr rl rr`, each a `<leg>_thigh` body with hinge `<leg>_hip` and a `<leg>_shin` body with hinge `<leg>_knee` and a sphere geom `<leg>_foot`. All hinges are about the torso's y axis. The torso's forward axis is +x.
- `features = False`: no tail or ears (the user's play-test verdict; the head and torso show the facing).
- Look: torso orange, head cream with a dark nose and two black eyes on its +x side (massless, contactless, group 2 sphere geoms), legs brown, feet dark.
- `sections`: a top-level `<actuator>` (one `motor` per joint, `ctrlrange -1 1`, gear = peak torque, in the order fl hip, fl knee, fr hip, ... rr knee) and a `<keyframe>` `stand` (shell at (0, 0, 0.5), torso standing at the height where every foot just touches the inner wall, the standing joint angles). `model.build` appends it after the bodies (see [model.md](model.md)).
- `nuserdata = 0`, `resting_radius = 0.44` m.
- `action_space()`: `Box(-1, 1, (8,), float32)`, the torques, normalised by the gears.
- `reset(model, data, spawn)`: copies the keyframe's pose (relative to the shell) to `spawn`, torso yawed so its forward axis points along the facing (`userdata[FACING]`, which `Core.reset` now sets before calling the tier), joint velocities and `ctrl` zero.
- `drive(...)`: `ctrl[:8] = action`. Touches no `xfrc_applied`.
- `obs(...)`: `{"wall_contact": bool, "proprio": float32 (26,)}`. `wall_contact` is true when any hamster geom touches a tile. `proprio` is 8 joint positions, 8 joint velocities, 4 foot-contact flags (fl fr rl rr, that foot on a tile), the torso's forward and up axes in the facing frame (a standing hamster along the facing reads forward (0, 1, 0), up (0, 0, 1)); the env exposes it (see [env.md](../env.md)). It runs `mj_kinematics` and `mj_collision` first, since after `mj_step` the contact list belongs to the previous state; neither writes the core state.

Play-test viewer: `uv run --extra game python -m hamsterball.game.articulated_playtest [--gait NAME] [--level hill]` (`game/articulated_playtest.py`, no doc of its own). Keys: G gait on/off ([lowlevel.md](../lowlevel.md)), W/A/S/D or the arrows the run vector (camera-relative), K 20 degree slope toggle, F limp, Space shove-and-roll, R reset, Q/E orbit, Z/X zoom, Esc quit.

## How it works

Contacts: torso, head, thighs and shins have `condim 1`, `priority 1`, so they are frictionless against the tiles. Feet are `condim 3` spheres with a 2 mm margin; their friction mixes with the tiles' `1 .005 .0001` by max, so the feet's own friction cannot go under 1. Every hamster geom has `contype 2`, `conaffinity 0`, so they hit the tiles (`conaffinity 2`) but never each other (with `conaffinity 2` the legs collide with one another). Geoms are only capsules and spheres: MJX has sphere-box and capsule-box but not cylinder-box or ellipsoid-box. Solref is `0.04 2` on every geom, as the tiles.

Masses are set per geom so the subtree is exactly 2.4 kg: head 0.80, per leg thigh 0.0599, shin 0.0742, foot 0.02, and the spine takes the rest (0.984). The two bars are massless (they only widen the collision shape and the look), so the masses stay where the prototype put them. Torso, bars and legs are frictionless against the tiles; feet have friction 2.075 (the tiles' 1 is the lower, the larger applies). The front legs are short (0.06 + 0.066 m) and the rear long (0.148 + 0.06 m), so standing, the rear feet touch the wall and the front ones hang about 74 mm off it until the body settles.

## Constants (the prototype's `fast` skeleton with the wide torso)

| | value |
|---|---|
| spine capsule | radius 0.05, half length 0.0746 m (mass 0.984 kg) |
| bars | capsules radius 0.04 along y, from y = -0.1137 to 0.1137, at x = +0.0544 (shoulders) and -0.1177 (pelvis), massless |
| head | sphere r 0.045 at (0.1925, 0, 0.025), 0.80 kg |
| hips | (+0.0544, -0.1177 for rear, ±0.1137, -0.0005) from the torso centre |
| front thigh, shin | 0.06 m, 0.066 m |
| rear thigh, shin | 0.1477 m, 0.06 m |
| foot | sphere r 0.0246, friction 2.075; capsule radii 0.015 (thigh) and 0.012 (shin) |
| hip, knee range | -0.508..1.675 rad, -2.199..0.0194 rad (written in degrees in the MJCF, see Decisions) |
| gear hip, knee | 17.15, 21.09 N m |
| joint damping, armature | 0.23, 0.005 |
| standing pose | front legs hip 0.7128, knee -1.7116; rear legs hip 0.5091, knee -1.7597 (positive hip swings the foot back, negative knee bends the shin forward) |
| resting radius | 0.44 m |

Resting radius: limp hamsters dropped in 24 random orientations and run for 1.5 s with containment off swing through 0.415 m at most and rest at 0.365 to 0.399 m from the shell centre; 0.44 sits above both and about equals the prototype's measured 0.434. The thrown-hamster test measures 0 / 11 / 20 / 20 mm of sinking for 0 / 4 / 8 / 12 m/s (the 20 mm hard stop).

## Invariants and checks

- Pushing the shell, climbing and steering are tested with the scripted gait (ticket 04, [lowlevel.md](../lowlevel.md)); the play-test no longer has a canned trot.
- Torso tests: the hip joints lie inside the torso capsules' skins, and every torso geom is a frictionless capsule.
- `uv run pytest tests/test_articulated_tier.py`: the fragment contract (free root `hamster`, 2.4 kg, group 2, `hamster_eye`, only MJX-supported joint, geom and actuator types), reset stands with its feet on the inner wall and the torso along the facing, a stored core state reproduces the run bit for bit with `ctrl` included, thrown hamsters sink no further than the hard stop plus a step (measured 0 / 20 / 20 / 20 mm for 0 / 4 / 8 / 12 m/s), a fallen hamster gets no containment impulse, the compiled joint ranges equal the constants in radians, and a 60 Hz PD hold of the standing pose (kp 80, kd 1) keeps the joints within 0.2 rad of it and the torso upright (kp 15, kd 0.1, the gait's gains); once the torques go to zero after a shove the hamster collapses onto the wall (COM over 0.36 m from the shell centre, belly down) with no containment impulse.

## Known quirks

- The tier has no gait or balance: with zero torque it still stands on its legs for a while (the jointed legs hold the load), and a shove drops it belly down on the wall at about 0.38 m; the wide torso rarely rolls it onto its side.
- Pushing speeds below were measured on the broken skeleton (near-rigid legs) and are void; the rest of this item is kept as history. Pushing is limited by the gait, not the skeleton's torque. Half the abstract tier's top speed (7.15 m/s of 14.3) is out of reach: the best non-tumbling gait found reaches about 1 m/s sustained (peaks 2.3 m/s). To hold still against a shell rolling at `v` the stance feet must travel back at `v` relative to the wall, so `v` is bounded by stride length times stride rate (about 0.3 m of foot travel at 7.5 Hz, under the 60 Hz control, is 8 samples per stride), and the open-loop trot has no attitude control: stronger trots pitch the torso nose-down and roll it over within a second. The abstract tier's feet are a force, with no such bound. The result is sensitive: rounding the gait constants in the 4th digit turned a 15 s non-tumbler into one that tumbles later, so the play-test's constants are one of the few of 40 roundings that held 30 s.
- `wall_contact` costs a collision pass per env step.

## Decisions

- 2026-10-27, ticket articulated-tier/02 redo: the skeleton is the prototype's `fast` (user pick), constants rounded to 3 or 4 digits, with the anatomy fix: the thin torso capsule left the hips outside the body, so a shoulder and a pelvis bar (massless capsules along y through the hips) join the spine; capsules rather than a box, since MJX's collision pairs against tiles are only certain for sphere and capsule. Resting radius 0.44 m measured lying flat. Final sign-off on the gait's play-test is pending.

- 2026-10-27, ticket articulated-tier/02: skeleton constants above, chosen to pass the tests with no user sign-off yet.
- 2026-10-27, ticket articulated-tier/02: `Tier` gains `sections` (top-level MJCF after the bodies), since a fragment inside `<worldbody>` cannot hold an `<actuator>` or a `<keyframe>`; the abstract tier's is `""`, so its model is unchanged.
- 2026-10-27, ticket articulated-tier/02: `contype 2`, `conaffinity 0` on hamster geoms in place of the ticket's "bit 2 on both", so the legs never collide with one another.
- 2026-10-27, ticket articulated-tier/02: `Core.reset` sets the facing and pitch before `tier.reset`, so a tier can point along the facing without a new protocol argument. Abstract-tier trajectories are unchanged.
- 2026-10-27, ticket articulated-tier/02 play-test (user): no features for this tier, and a checker ground with a gradient sky for both tiers (see [views.md](views.md), [model.md](model.md)); head, legs and feet get their own colours and the head a nose and eyes.
- 2026-10-27, ticket articulated-tier/02, second play-test (user: "Stronger push"): gears 16.0 / 12.8 N m, hip range ±1.5 rad, standing pose (0.28, −0.87), and a canned trot with nose-down feedback, per the spec's Further Notes (gears and layout change, never the shell constants). The gait is part of the play-test script, not the tier; the scripted gait of the later ticket replaces it.

## Tried and rejected

- 2026-10-27, ticket articulated-tier/02: gears 3 / 2.5 N m, the first cut: standing is fine, but the trot moved the shell under 0.2 m in 15 s.
- 2026-10-27, second play-test retune (sweeps in a headless script, 15 s runs of the canned trot on flat ground, a run counts as tumbled when the torso's up axis drops under 0.3): (1) 300 random trots over gears 5 to 24 N m, hip swing, stride 1 to 6 Hz, offset and PD gain on the first skeleton: 176 never tumbled but moved the shell at most 0.18 m/s; every trot with a larger speed tumbled within 3 s (2.1 m/s peak). (2) 480 random skeletons (legs 0.09/0.12/0.15 m, hip spacing 0.085 or 0.12, hip range ±0.9/1.2/1.5, standing pose, gears to 24 N m): 47 non-tumblers, best 0.26 m/s. Longer legs and higher gears did not help without attitude feedback. (3) adding a feedback of the torso's nose-down and roll to the knees and hips, and hill-climbing the 15 gait and skeleton numbers over 7 seeds of 400 runs: 1.14 m/s sustained (best 0.82 m/s over 40 s, so the speed is a limit cycle, not steady); a second climb from it found nothing better. Roll feedback went to zero in the best runs, and the short legs (0.09 m) and narrow track (0.085 m) won.

## History

- 2026-10-27, ticket articulated-tier/02 redo: a prototype (branch `prototype/articulated-skeleton`, throwaway, 4 evolution rounds of skeleton and gait on a CPU cluster, aiming at a 20 degree climb and about 2 m/s on flat ground) produced candidates `best`, `fast` and `sprint`. Play-test verdict (user): `fast` is "slightly better than best", but its "anatomy is odd (leg disjoint from body)": the hips sat outside the thin torso capsule. Chosen fix: a wider torso covering all four hips, masses unchanged. Ported here; the final sign-off is pending on the ticket 04 play-test.

- 2026-10-27, ticket articulated-tier/02: built with the core-seam tests of the spec. The play-test (stand, push the shell, fall naturally) awaits the user.
- 2026-10-27, ticket articulated-tier/02 play-test (user): "model is off (three floating spheres with unknown purpose), and the uni-color ground makes it hard to observe hamster behavior"; the constants are not yet judged.
- 2026-10-27, second play-test (user): "Stronger push". It stands and falls fine, but barely moves the shell. Retuned the gears (5.0/4.0 to 16.0/12.8 N m), hip range (±0.9 to ±1.5 rad), standing pose and the canned trot (see Tried and rejected); the shell constants are untouched. Measured about 1 m/s of shell speed, tumble-free over 15 s; the new play-test verdict is pending.
- 2026-10-27, third play-test (user): signed off at about 1 m/s. Half the abstract top speed is out of reach with this skeleton and an open-loop trot; the user chose to accept it and leave push speed to the scripted gait (ticket 04), whose posture loop closes the balance the canned trot lacks. Matching the abstract feel stays out of scope.
- 2026-10-27, ticket articulated-tier/03: `obs()` gains `proprio`; the env builds `tier="articulated"` from it.
- 2026-10-27, bug found during ticket 04: the joint `range` attributes were radians but MJCF's compiler angle is degrees, so the compiled ranges were ±0.026 rad (hips) and −0.031..0.0009 rad (knees): the legs sat on near-zero stops and the standing keyframe lay outside them. Every earlier play-test (including the "about 1 m/s" sign-off) ran on that near-rigid skeleton, so the sign-off is void and is redone on the scripted gait's play-test. Fix: write the ranges with `math.degrees`; the model's compiler angle stays degrees, so levels are untouched. Real joints made the 60 Hz PD (kp 80, kd 1, ctrl held across 10 physics steps) unstable with damping 0.05 and armature 0.002 (the legs tumble in 2 s): damping 0.5 and armature 0.01 stabilise kd from 0.05 to 1 and kp 20 to 120. Peak torques, standing pose, masses, lengths and the resting radius check are unchanged. A test now compares the compiled ranges to the constants.
- 2026-10-27, final play-test (user): the `fast` skeleton with the wide torso and the `scripted_trot_v1` gait are signed off as is. This closes ticket 02.
