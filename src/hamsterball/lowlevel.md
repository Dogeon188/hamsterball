# lowlevel.py

## Purpose

Low-level policies for the articulated tier ([spec](../../.scratch/articulated-tier/spec.md), ticket 04): an object with a `name` and `act(obs, run_vector) -> torques`, looked up by name in a registry. It works outside the game, as a baseline or a reference controller. It reads the tier's `proprio` ([articulated_tier.md](core/articulated_tier.md)).

## Interface

- `act(obs, run_vector)`: `obs` is the env's observation dict (its `proprio` key is used) or the 26-float `proprio` itself; `run_vector` is `[run_x, run_y]` in the facing frame (the chase view's frame), length 0 to 1. Returns the 8 normalised torques (float32), no camera dims. Call it once per env step (60 Hz).
- `reset()`: zeroes the policy's state (stride phase, effort).
- `get(name, **overrides)`: builds a registered policy; an unknown name raises `KeyError` listing the known names. `REGISTRY`, `DEFAULT`.
- Registered: `scripted_trot_v1` (`ScriptedTrot`; the name is free again, its first version was never merged). A change that alters a gait's torques registers a new name; old names stay.
- Play-test: `uv run --extra game python -m hamsterball.game.articulated_playtest [--gait NAME] [--level hill]`. G switches the gait on, W/A/S/D or the arrows give the run vector relative to the camera, K toggles a 20 degree slope (uphill is the start facing), F limp, Space shove, R reset, Q/E orbit, Z/X zoom, Esc quit.

## How it works

`ScriptedTrot` is a closed loop on joint targets with a PD hold (`kp 15`, `kd 0.1`, in joint torque before the gear). It is the prototype's gait for the `fast` skeleton ([articulated_tier.md](core/articulated_tier.md)), found by evolution on a CPU cluster, and ran unchanged on the production skeleton with its wide torso:

- Effort is the run vector's length, eased (2 per second). It sets the stride rate (0.54 to 7.7 Hz) and scales the hip stroke.
- Front and rear legs have their own stroke amplitude (0.62 and 0.41 rad), knee lift (0.84 and 1.07 rad), hip offset (+0.46 and -0.17 rad) and knee offset; each leg has its own stride phase (fl 0, fr 0.48, rl 0.77, rr 0.10). The stroke is a triangle wave: the hip sweeps the foot back during the stance (60% of the stride) and forward in the swing, where the knee lifts (sine).
- Steering: the heading error between the torso's forward axis and the run vector (both in the facing frame) sets a turn in [-1, 1] (`-err / 0.58`); the left and right strokes scale by `1 ± 0.33 turn`, and the whole stroke shrinks toward 0.39 as the error grows to 90 degrees or more.
- Posture: the torso's forward-axis height (nose up or down) goes to the front and rear knees and hips, and its roll (the up axis's lean toward the torso's left) to the left and right knees and hips, as target offsets.

## Invariants and checks

`uv run pytest tests/test_lowlevel.py`: the registry, a zero run vector standing with all feet on the wall (and holding a 3 degree slope), a full forward run vector above 2 m/s for 20 s without tumbling, the 20 degree climb above 1 m/s, sideways run vectors turning the hamster within 35 degrees of them in 10 s, and the frozen feel reference.

Feel reference of `scripted_trot_v1` (run vector (0, 1), no noise, frozen in the test within 5%): 5.63 m travelled in the first 3 s on flat ground, 2.78 m/s mean over 10 to 20 s, and 1.47 m/s mean over 10 to 20 s on a 20 degree slope. For comparison the abstract tier tops out at 14.3 m/s.

Target check, measured on a CPU cluster on fresh seeds (5001 to 5004, joint noise 0.03 rad, torso heading to the run vector in the facing frame): flat 2.82 m/s (all alive at 20 s), 10 degree climb 2.26 m/s, 20 degree climb 1.49 m/s (all alive at 20 s), and 48 runs of 30 s at 12 headings 30 degrees apart tumbled 0 times, with a mean 2.83 m/s along the run vector and a final heading error of 4.4 degrees.

## Known quirks

- Only the extremes of effort are stable: a run vector of length 0.25 creeps (0.01 m/s), 0.5 tumbles after 2.3 s and 0.75 after 5.7 s (no-noise starts). The search only scored full-length run vectors, and the ease-in at 2 per second passes through the middle quickly enough to be fine. A later gait version should be tested at part effort (a thumbstick).
- Standing (zero run vector) the up axis dips to 0.7 while the shell rolls out from under the hamster, then rests at 0.875 with the front feet down (the body leans nose-down); a 3 degree slope holds.
- The gait's constants came from a search and carry the full digits; rounding them was not tried on this skeleton (the first gait was sensitive to it).

## Decisions

- 2026-10-27, ticket articulated-tier/04: the first entry is a trot, kept with duty above 0.5 so three or four feet are down most of the time. A pure trot (two feet down) rolled the torso over within 2 s.
- 2026-10-27, ticket articulated-tier/04: stable running and turning over top speed (the user's brief); no attempt to match the abstract tier's feel.

## Tried and rejected

- 2026-10-27, ticket articulated-tier/04, superseded: the first gait (one amplitude, lift and offset for all four legs, `kp 40`, `kd 0.5`, diagonal phases) on the first skeleton ran 0.49 m/s flat and tumbled on any slope above about 1 degree (holding a ball with that skeleton's 0.28 m COM depth on 20 degrees needed the COM about 54 degrees up the wall). The prototype's skeleton search is the answer: lower, longer body, short front legs, long rear legs, strong knees, foot friction 2.1.
- 2026-10-27, ticket articulated-tier/04: a retune after the torso change was not needed: the prototype's constants ran the target on the wide torso unchanged.

## History

- 2026-10-27, ticket articulated-tier/02 fix: the joint ranges were degrees in disguise (see articulated_tier.md); the first gait of ticket 04 had been tuned on that broken skeleton and was rewritten.
- 2026-10-27, ticket articulated-tier/04 redo: the first gait did not climb, so the user chose to rethink the skeleton (target: 20 degree climb, about 2 m/s on flat ground, no tumbling). The prototype found `fast`; the gait here is its gait, ported without a retune (the user's anatomy fix, a wide torso covering the hips, left it unchanged). `scripted_trot_v1` now means this gait.
- 2026-10-27, ticket articulated-tier/04: **Play-test verdict: pending** (the user judges that it runs, climbs, steers and looks right).
