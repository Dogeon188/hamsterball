# lowlevel.py

## Purpose

Low-level policies for the articulated tier ([spec](../../.scratch/articulated-tier/spec.md), ticket 04): an object with a `name` and `act(obs, run_vector) -> torques`, looked up by name in a registry. It works outside the game, as a baseline or a reference controller. It reads the tier's `proprio` ([articulated_tier.md](core/articulated_tier.md)).

## Interface

- `act(obs, run_vector)`: `obs` is the env's observation dict (its `proprio` key is used) or the 26-float `proprio` itself; `run_vector` is `[run_x, run_y]` in the facing frame (the chase view's frame), length 0 to 1. Returns the 8 normalised torques (float32), no camera dims. Call it once per env step (60 Hz).
- `reset()`: zeroes the policy's state (stride phase, effort, yaw-rate filter).
- `get(name, **overrides)`: builds a registered policy; an unknown name raises `KeyError` listing the known names. `REGISTRY`, `DEFAULT`.
- Registered: `scripted_trot_v1` (`ScriptedTrot`). A change that alters a gait's torques registers a new name; old names stay.
- Play-test: `uv run --extra game python -m hamsterball.game.articulated_playtest [--gait NAME]`. G switches the gait on, W/A/S/D or the arrows give the run vector relative to the camera, Q/E orbit, R resets, F limp, Space shove, T the old canned trot.

## How it works

`ScriptedTrot` is a closed loop on joint targets with a PD hold (`kp 40`, `kd 0.5`, ceiling set by the 60 Hz control; see articulated_tier.md):

- Effort is the run vector's length, eased (2 per second). It sets the stride rate (0.97 to 1.95 Hz) and scales the hip stroke (amplitude 0.49 rad at full effort).
- Each leg's stride phase is offset by half a stride between the diagonals (fl rr against fr rl). The stroke is a triangle wave: the hip sweeps the foot back during the stance (75% of the stride, so three or four feet are down most of the time) and forward in the swing, where the knee lifts (0.8 rad, sine).
- Steering: the heading error between the torso's forward axis and the run vector (both in the facing frame) sets a turn in [-1, 1] (`-err / 1.16`); the left and right strokes scale by `1 ± 0.16 turn`, and the whole stroke shrinks toward 0.74 as the error grows to 90 degrees or more.
- Posture: the torso's forward-axis height (nose up or down) goes to the front and rear knees and hips, and its roll (the up axis's lean toward the torso's left) to the left and right knees and hips, as target offsets.

## Invariants and checks

`uv run pytest tests/test_lowlevel.py`: the registry, a zero run vector standing upright with all feet on the wall (and holding a 3 degree slope), a full forward run vector pushing the shell above 0.4 m/s for 20 s without tumbling, sideways run vectors turning the hamster within 35 degrees of them in 10 s, and the frozen feel reference.

Feel reference of `scripted_trot_v1` (flat ground, run vector (0, 1), frozen in the test within 5%, half effort within 10%): 1.24 m travelled in the first 3 s, 0.487 m/s mean over 10 to 20 s, 0.093 m/s at half effort. For comparison the abstract tier tops out at 14.3 m/s.

## Known quirks

- Top speed is about 0.5 m/s. The stance feet must sweep back as fast as the shell rolls, and a longer stroke or faster stride tumbled the torso in every search.
- The 20 degree climb of the abstract tier's feel test is out of reach, and so is any climb above about 0 to 1 degrees under a run vector: holding a ball with the hamster's 0.28 m COM depth on a slope `a` needs the COM to sit `(m_shell + m_ham) R sin a / m_ham` forward of the shell centre, which at 20 degrees is 0.23 m, about 54 degrees up the wall, where the legs cannot stand. With the run vector at zero it holds 3 degrees and tumbled at 6.
- The gait is sensitive to its constants: rounding them to 2 decimals changed turning from passing to failing, so `P` keeps the full values.
- Robustness, measured once on a CPU cluster: held for 30 s at 24 run directions 15 degrees apart (full effort, from rest), 21 stayed up and ended within 9 degrees of the run direction; 3 tumbled (run vector at 90 degrees right of forward at 25 s, 210 at 18 s, 300 at 4 s). Perturbing amp, duty, lift, hz and kp by 3% tumbled about 1 run in 3. The gait is not tumble-proof; a later gait version should start from this failure list.

## Decisions

- 2026-10-27, ticket articulated-tier/04: the first entry is a trot, kept with duty above 0.5 so three or four feet are down most of the time. A pure trot (two feet down) rolled the torso over within 2 s.
- 2026-10-27, ticket articulated-tier/04: stable running and turning over top speed (the user's brief); no attempt to match the abstract tier's feel.

## Tried and rejected

- 2026-10-27, ticket articulated-tier/04: the first controller (sine strokes, open duty 50%, nose-down feedback on the knees and hips, many steering knobs), tuned on the broken-range skeleton: on the real joints 6 of 1500 random draws survived 12 s and none ran forward (it ran backward at 0.3 m/s mean magnitude); 150 generations of hill-climbing on 3 headings stayed at 0.1 (no run survived all three). The triangle-stroke, duty 0.75 controller above: 116 of 3000 random draws survived 12 s going forward, the best 0.43 m/s; hill-climbing 100 generations on three headings gave 0.47 m/s, then 40 on six turn headings, then 30 with a 2 degree slope added.
- 2026-10-27, ticket articulated-tier/04: a 20 degree climb (see Known quirks).

## History

- 2026-10-27, ticket articulated-tier/02 fix: the joint ranges were degrees in disguise (see articulated_tier.md); the first gait of ticket 04 had been tuned on that broken skeleton and was rewritten.
- 2026-10-27, ticket articulated-tier/04: the gait runs, turns and holds the stand in the tests above. **Play-test verdict: pending** (the user judges that it runs and steers).
