# step.py

## Purpose

The core step of spec §4: the per-physics-step code (facing, shell drag, containment, the tier's drive) around `mj_step`, the `Tier` protocol of spec §6, and `Core`, which holds a compiled model and its data.

## Interface

- `Core(tier=None, level=FLAT, camera_action=False, view_size=GAME_SIZE)`: compiles the model (`view_size` sizes its offscreen buffer, see [model.md](model.md)) (default tier: `AbstractTier`) and resets. `level` is a `Level` (its kinematic bodies compiled and driven) or MJCF as `model.build` takes it, such as `Level.mjcf()` (no bodies). `Core.bodies` holds a `Body(qpos, dof, pos, motion)` NamedTuple per body, constants; a seesaw's motion is `None` and its addresses are its joint's. `Core.tiles` is `None`, or `Tiles(slot, body, delay, tiles)` for a level's collapsing tiles (the first trigger-time slot, the first tile's body id, and a `Tile(qpos, dof, pos, bottom)` per tile), constants.
- `Core.reset(spawn=(0, 0, 0.5), facing=0.0)`, `Core.step(action, n=FRAME_SKIP)` (one action held over `n` physics steps; under the camera flag `turn` = 1 turns the facing `TURN_STEP` = 30° and `cam_pitch` = 1 pitches `PITCH_STEP` = 15°, spread evenly over the `n` steps), `Core.state()` and `Core.set_state(s)` (the `mjSTATE_INTEGRATION` vector), `Core.facing`, `Core.pitch`, `Core.action_space` (the tier's, plus `turn` and `cam_pitch` under the camera flag).
- `physics_step(model, data, tier, action, d_facing, d_pitch, bodies=(), tiles=None)` (`d_facing` and `d_pitch`: this physics step's change of facing and pitch, rad) and its parts `drive_bodies`, `drive_tiles`, `touch_tiles`, `turn_facing`, `camera_pitch`, `drag`, `contain`, `wrap`.
- `Tier`: `fragment`, `nuserdata`, `resting_radius`, `action_space()`, `reset()`, `drive()`, `obs()`. The core touches a tier only through these.

## How it works

Each physics step: clear `xfrc_applied`; pose the kinematic bodies (`drive_bodies`) and the collapsing tiles (`drive_tiles`); refresh positions (`mj_kinematics`, `mj_comPos`); facing into `userdata[FACING]` (`wrap(facing + TURN_STEP·turn/n)`; nothing but the controller turns it); camera pitch into `userdata[PITCH]` (`clip(pitch + PITCH_STEP·cam_pitch/n, PITCH_MIN, PITCH_MAX)`, read only by the views, so it moves no physics); drag `−k|v|v` on the shell; containment on the hamster root's free joint (an inelastic impulse from 5 mm past the resting radius, a position stop and `mj_forward` past 20 mm); `tier.drive()`; `mj_step`; record the tiles the shell touched (`touch_tiles`).

The position refresh is what makes the core state complete. `mj_step` leaves `xpos` and friends computed for the state before it, so code that read them directly would depend on the previous step, and a restored state would step differently. Refreshing costs two cheap kinematics passes and means every step reads only `qpos`, `qvel`, `userdata`, and `time`.

Kinematic bodies: `drive_bodies` writes each body's free joint from `data.time` alone: qpos = `pos` + the motion's offset and the yaw quaternion at `t`, qvel = the secant to the pose at `t + DT` (linear part `(p1 − p0)/DT`, angular part `mju_subQuat(q1, q0)/DT`, which is in the body frame as a free joint's is). The velocity is what makes friction see the motion; `mj_step` then integrates the body to the next pose plus a tiny contact perturbation, which the next step's write discards. `Core.reset` drives them too, so a body is posed at time 0. The write is a pure function of time, which is in the core state, so the core state stays complete. It does change the core state's size per level: each body adds 7 qpos and 6 qvel (and their warmstart), so `survive_pushers` has 186 numbers to the 124 of a level without bodies.

Collapsing tiles: tile `i`'s userdata slot (`CORE_SLOTS + tier.nuserdata + i`) holds the sim time the shell first touched it, `+inf` until then (`Core.reset` sets it, since `mj_resetData` zeroes userdata). `drive_tiles` writes each tile's free joint like `drive_bodies`: at its authored pose until `delay` after the trigger, then falling at `G` = 9.81 m/s² (MuJoCo's gravity, so a resting hamsterball falls with it and loses contact), held from `TILE_DROP` = 1 m under the kill plane on, with the secant velocity. `touch_tiles` runs right after `mj_step` and reads `data.contact`: `mj_step` collides on the state the step started from and nothing recomputes contacts after the integration, so the contacts are a pure function of the core state (and the trigger, `data.time` after the step, too). A tile pair with the shell is matched by `geom_bodyid`, since tile bodies are contiguous after the kinematic bodies. Reading the contacts anywhere else would break restoring: after `Core.set_state` (which runs `mj_forward`) they belong to the current state, after a live `mj_step` to the previous one. A fallen tile leaves no debris: it is kinematic and excluded from everything but the shell, and it sits under the kill plane. Each tile adds 7 qpos, 6 qvel and their warmstart, and one userdata slot, to the core state.

Kernel shape: ids and constants are module constants, small-vector maths is written out in Python floats, and no function keeps Python-side state. `Core` holds only the model, the data, the tier, and the camera flag.
Seesaws (ticket survive-movers/05): a seesaw's hinge or ball joint is passive, so `drive_bodies` skips it (motion `None`) and MuJoCo integrates it like the shell: its state is ordinary qpos and qvel, and the core state stays complete with no write. A hinge adds 1 qpos and 1 qvel, a ball 4 and 3.

Tilt (ticket survive-movers/04): `drive_bodies` writes `motion.quat` (yaw then tilt), and the angular secant `mju_subQuat(q1, q0)/DT` covers the tilt as well, in the body frame. A motion without a tilt writes the same quaternion as before, bit for bit.

## Invariants and checks

- `uv run pytest tests/test_core_state.py`: setting a stored core state and stepping reproduces the run bit for bit (facing and pitch included); `turn` = 1 turns the facing 30° in one env step and `cam_pitch` = 1 pitches 15° within −10° to 60°, whatever `n`; the pitch moves no physics; without a turn input the facing holds at its reset value however fast the shell runs; without the camera flag the pitch holds its default.
- `uv run pytest tests/test_bodies.py`: bodies compile after the view mocaps and touch only the shell, follow their motion, restore bit for bit from a stored core state, shove a resting hamsterball, and carry one by friction.
- `uv run pytest tests/test_collapse.py`: a touched tile falls after its delay with the hamsterball on it and an untouched one never does, tiles touch only the shell, and a stored core state on a 49-tile level restores bit for bit with tiles triggering before and after the restore.
- `uv run pytest tests/test_seesaw.py`: a seesaw compiles after the view mocaps with the fixed ids held, restores bit for bit (hinge and ball), tips toward an off-pivot hamsterball and stays level at the pivot (numbers in [level.md](level.md)).
- `uv run pytest tests/test_feel.py`: the spec §5 feel reference.

## Known quirks

- Containment is a step-level state edit, not a MuJoCo constraint, so one step can still carry the hamster past the 20 mm stop by that step's travel. Measured worst sinks for throws at 0 / 4 / 8 / 12 m/s are 14 / 17 / 20 / 23 mm, under spec §5's 16 / 21 / 30 / 36 mm, because this core reads fresh positions (see History, 2026-10-22).
- Camera pitch constants, **needing play-test**: `PITCH_DEFAULT` 15° below the horizon (the old chase pitch, defined in [model.py](model.md)), `PITCH_MIN`/`PITCH_MAX` −10°/60°, `PITCH_STEP` 15° per env step at `cam_pitch` = 1; and `TURN_STEP` 30° per env step at `turn` = 1. The angles are per env step whatever the frame skip, so a rate in °/s scales with it (frame skip 5 doubles it). The pitch is in the core state and the observation, so changing them bumps N.
- The Fibonacci tiles aren't isotropic: the uncapped 1–3 s acceleration is 4.05 m/s² along +x and 4.16 along +y. The feel test runs along +x, as the prototype did.

## Decisions

- 2026-10-21, issue 13 verdict (play-tested 2026-10-20), recorded in `prototypes/honest_drive.md` on `prototype/honest-physics`: drag `k = 0.059` on the shell, not gated on ground contact; `field` containment with a 5 mm start and a 20 mm hard stop.
- 2026-10-21, ticket hamsterball-v1/01: the per-step code refreshes positions with `mj_kinematics` and `mj_comPos` before reading them, so the core state alone determines the next step.
- 2026-10-22, ticket survive-movers/03: a kinematic body is a free-jointed body whose qpos and qvel the core writes every physics step from sim time, gravity compensated (`gravcomp=1`) and heavy (1000 kg), excluded from the world's contacts. Spec §4's "levels add no joints, so it has one shape for every level of a tier" no longer holds; the Collector stores the core state as a variable-length list (see [collector.md](../data/collector.md)).
- 2026-10-22, ticket survive-movers/06: a collapsing tile is a kinematic body (as ticket 03's) whose pose is a function of time and its trigger time in userdata, and touch detection reads `data.contact` right after `mj_step` (exploration agent: the post-step contact set from a run and from `mj_setState` of the stored pre-step state matched over 300 steps).
- 2026-10-22, ticket mouse-camera/04: only the controller turns the facing; the chase auto-turn toward the shell's travel (`CHASE_ALPHA`, `CHASE_MIN_SPEED2`) is deleted, so without the camera flag the facing stays at the spawn's. Under the camera flag `turn` and `cam_pitch` are angles per env step (30° and 15° at 1), not rates, so a policy can turn a lot in one step and a game frame's mouse move lands at once.

## Tried and rejected

See the prototype's doc for containment by solimp, a spring-damper field, and a contact-gated drag.

- 2026-10-22, ticket survive-movers/03 (exploration agent's measurements, MuJoCo 3.16.0, a 3×3×0.2 m platform moving +x at 1 m/s under a resting inputless hamsterball, shell x after 2 s): a **mocap** body with `mocap_pos` written every step reached −0.134 m and the ball fell off (a mocap body has no dofs, so its contact velocity is 0 and friction never sees the motion); a jointed body with **qpos only** written gave −0.13 m, the same; qpos **and** qvel at 1000 kg with gravcomp gave 1.353 m (1.351 at 1e6 kg, 1.460 at 10 kg, where the contact pushes a light body within the step; 1.483 without gravcomp, the body free-falling g·dt each step and the shell sinking 4 mm). Also rejected without a run: **position/velocity actuators** on the joints (the body lags and overshoots by an amount that depends on load and gains, so its pose stops being a function of time) and a **weld to a mocap target** (soft, lags, and the velocity still comes only from dofs).
- 2026-10-22, ticket survive-movers/06 (exploration agent, not built): releasing a passive tile by turning off a weld (`eq_active`, in the core state) sags under the ball (the weld is soft) and leaves dynamic debris; writing `geom_contype` per step from userdata mutates a model the env LRU and a batched backend share; `mjcb_contactfilter` is process-global and slow; a second `mj_collision` pass after the position refresh would be pure too but costs a collision pass per step.

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
- 2026-10-22, code review of hamsterball-v1: the offscreen buffer was a fixed 1280×720, so a `view_size` wider or taller raised in `mujoco.Renderer`; spec §7 asks for at least `view_size`. `Core` passes a `view_size` through to `build`.
- 2026-10-22, ticket survive-movers/03, agent measurement (`tests/test_bodies.py`, M2): a 3 m bar oscillating along −y (amplitude 5 m, period 8 s) from 4 m away shoves a resting shell to y = −4.35 m after 5 s; a 4×4×0.2 m platform moving +x at 0.5 m/s under a resting shell (no floor) carries it to x = 1.772 m, z 0.499, after 4 s, when the platform is at 2 m (the ball rolls back about 0.23 m at the instant start, then keeps pace).
- 2026-10-22, ticket survive-movers/04, agent measurement (`tests/test_platforms.py`, M2): a 4×4×0.2 m platform at a steady 0.25 m/s along +x carries an inputless hamsterball for the whole 8 s, its attached region with it (success); the ball's offset from the platform centre at 2, 4, 6, 8 s is −0.093, −0.090, −0.085, −0.076 m (it rolls back at the instant start, then keeps pace, creeping forward 1.7 cm in 6 s). A scratch run at 0.5 m/s (`path` 0→8 m, period 32 s) settles the offset at −0.227 m from 3 s on.
- 2026-10-22, ticket survive-movers/05: `drive_bodies` skips a body without a motion (a seesaw); `Core.bodies` keeps an entry for it, so body indices still match `Level.bodies`.
- 2026-10-22, code review of survive-movers: `Core.bodies` and `Core.tiles` hold NamedTuples (`Body`, `Tiles`, `Tile`) in place of plain tuples read by index; the same values, so every bundled level's core states are byte-identical.
- 2026-10-22, ticket mouse-camera/02 (prefactor, no behaviour change): the core's userdata slots are named in `model.py` (`YAW`, `CORE_SLOTS`); the per-step code reads the yaw at `YAW`, and the tiles' first slot is `CORE_SLOTS + tier.nuserdata`.
- 2026-10-22, ticket mouse-camera/02: `CORE_SLOTS` = 2 with `PITCH` = 1; under the camera flag `Core.step` splits `[cam_yaw, cam_pitch]` off the action, and `camera_pitch` integrates and clamps the pitch each physics step after the yaw. `Core.reset` sets it to `PITCH_DEFAULT`. The core state grows by one number (110 → 111 on a level without bodies). Two runs that differ only in `cam_pitch` have identical qpos, qvel and time.
- 2026-10-22, code review of mouse-camera: `physics_step` takes `cam_yaw, cam_pitch` side by side (was `cam, bodies=(), tiles=None, cam_pitch=0.0`); `PITCH_DEFAULT` moved to `model.py`, the other pitch constants stay here. No behaviour change.
- 2026-10-22, ticket mouse-camera/04 (rename, no behaviour change): "camera yaw" is now **facing** (GLOSSARY.md): the userdata slot `YAW` is `FACING`, `Core.yaw` is `Core.facing`, `Core.reset` takes `facing=`, `camera_yaw` is `turn_facing`, and the action dim `cam_yaw` is `turn`.
- 2026-10-22, ticket mouse-camera/04: removed the chase auto-turn, test-first (the facing holds at the spawn's at 2+ m/s, with and without the camera flag); this retires the Known quirk where the chase swung a held "forward" off a 20° climb. `ORBIT_RATE` 120°/s and `PITCH_RATE` 90°/s became `TURN_STEP` 30° and `PITCH_STEP` 15° per env step, divided over `Core.step`'s `n`; `physics_step` takes the per-step angles. `test_feel.py` no longer patches the chase off.
- 2026-10-23, ticket mouse-camera/05: the tail mocap ([views.md](views.md)) adds 13 numbers to the core state (7 of mocap pose and 6 of `xfrc_applied`, which `mjSTATE_INTEGRATION` holds for every body: 111 → 124 on a level without bodies, 173 → 186 on `survive_pushers`); no trajectory changes.
- 2026-10-23, code review of mouse-camera tickets 04–05 (no behaviour change): `physics_step` takes `d_facing, d_pitch` (was `turn, pitch`), and `Core.step` names its per-physics-step radians the same, so `turn` is only the unitless action.
- 2026-10-27, ticket gpu-backend/02 (no behaviour change): the batched core ([gpu/batch_core.md](../gpu/batch_core.md)) ports these per-step functions to Warp kernels of the same names; it reads this module's constants, and past the hard stop it refreshes positions where this module calls `mj_forward`.
