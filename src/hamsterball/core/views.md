# views.py

## Purpose

The three views of spec §11 (`chase`, `follow`, `first_person`): a model `<camera>` per view on a mocap body, posed from the core state once per env step, and the `MjvOption` each view renders with. Also the hamster's **features** (GLOSSARY.md): a tail and two ears, spheres on their own mocap body, posed the same way, that show the facing in the chase and follow views.

## Interface

- `VIEWS`: the view names, in that order.
- `FRAGMENT`: MJCF for the three mocap bodies (`view_chase`, `view_follow`, `view_first_person`), each holding a camera named after its view, then the `features` mocap body holding the `tail`, `ear_left`, and `ear_right` sphere geoms. `model.build` appends it after the tier's fragment.
- `pose(model, data, follow_yaw)`: writes the views' and the features' `mocap_pos` and `mocap_quat` from fresh positions (after `mj_kinematics`). The cameras move only on the next `mj_kinematics` plus `mj_camlight`.
- `option(view, hitboxes=False)`: a fresh `MjvOption`. Groups 0 to 2 shown and 3 to 5 hidden for `chase` and `follow`; only group 0 for `first_person`. `hitboxes=True` also shows group 4, the **hitbox view**'s geoms (GLOSSARY.md): the shell-centre dot ([model.md](model.md)); the game window adds the kill plane to its scene ([window.md](../game/window.md)). The env never passes it, so pixel observations never show hitboxes.

The env ([env.md](../env.md)) calls `pose` every env step and owns the renderer.

## How it works

- `chase`: `CHASE_DIST` back from the shell centre along the facing, pitched down by the camera pitch (`userdata[PITCH]`, see [step.md](step.md)), looking at the shell centre.
- `follow`: the same construction at `FOLLOW_DIST` and `FOLLOW_PITCH`, with the level's `follow_yaw_deg` in place of the facing.
- `first_person`: at the `hamster_eye` site, looking along the facing, pitched by the camera pitch minus `PITCH_DEFAULT`, so level with the horizon at the default.
- The features: three spheres placed in their body's frame (x right, y forward, z up): the tail behind the centre, the ears on top, a little forward, one each side. `pose` puts the body at the hamster centre and turns it by the facing about z, so the features follow the facing whatever the hamster's roll (the hamster is a free sphere that rolls, so they can't ride its body). They are in group 2, the hamster's, so `first_person` hides them with no change to `option`.
- A MuJoCo camera looks along its −z with +y up, so the mocap's rotation matrix has columns right, up, −forward, with forward `(−sin yaw·cos pitch, cos yaw·cos pitch, −sin pitch)` (yaw 0 faces +y, as the facing frame).

## Invariants and checks

- The view mocaps carry no geoms, and the features' geoms have `mass="0"` and `contype`/`conaffinity` 0, so none adds mass or contacts and none changes a trajectory (a mocap body is welded to the world anyway). They do add 13 numbers each to the core state (3 of `mocap_pos`, 4 of `mocap_quat`, and 6 of `xfrc_applied`, which `mjSTATE_INTEGRATION` holds for every body), so the env poses them every env step, views or not, and `core_state` doesn't depend on the `views` kwarg.
- `uv run pytest tests/test_views.py`: shapes and dtype, chase and first person centred on the facing and pitched by the camera pitch (the follow view not), follow on the level's yaw, the geom groups, `obs_state=False`, rendering on a worker thread, the `MUJOCO_GL` default, the first-person near ground, rendering across `model_cache` eviction, a `view_size` larger than 1280×720, the tail straight behind the hamster and the ears on top, one each side, after reset and after turning, and the features' group and physics-free flags. `tests/test_rerender.py` checks a re-render draws them where they were.

## Known quirks

- Look constants, picked by eye from sample frames and **needing play-test**. They change pixels, not trajectories, so changing them doesn't bump N:

  | Constant | Value |
  |---|---|
  | `CHASE_DIST` | 3.0 m (its pitch is the camera pitch, default 15° down: see [step.md](step.md)) |
  | `FOLLOW_DIST`, `FOLLOW_PITCH` (the follow offset) | 6.0 m, 35° down |
  | `FOVY` (vertical) | chase 60°, follow 50°, first person 75° |
  | Marker colours (`MARKER_RGBA` in [level.py](level.md)) | goal `0.2 0.9 0.3 0.4` green, waypoints `0.2 0.5 1.0 0.4` blue, items `1.0 0.85 0.1 0.4` yellow, keys `1.0 0.4 0.1 0.4` orange, survive region `0.6 0.3 0.9 0.25` purple |
  | `HAZARD_RGBA`, `LOCKED_RGBA` | `0.9 0.1 0.1 0.5` red; `0.5 0.5 0.5 0.15` faint grey |
  | `PASSED_RGBA` (a passed waypoint) | `0.2 0.9 0.3 0.2` dim green |
  | `TAIL` | 0.13 m behind the hamster centre (its radius is 0.12), radius 0.04 m |
  | `EARS` | ±0.07 m right, 0.04 m forward, 0.10 m up, radius 0.035 m |
  | `FEATURE_RGBA` | `1 .6 .2 1`, the hamster's orange |
  | `HITBOX_DOT` (in [model.py](model.md)) | 0.04 m, opaque white |
  | `KILL_PLANE_RGBA`, `KILL_PLANE_HALF` (in [window.py](../game/window.md)) | `0.9 0.1 0.1 0.2` red; 100 m |

- The chase and follow cameras aim at the shell centre, so at 84×84 the shell covers the near part of the goal when the goal is straight ahead.
- Pixels can't show the `first_person` group hiding: from inside, back-face culling already drops the shell sphere and the hamster. The test checks the option itself.
- Keys (orange) and hazards (red) are close in hue, and the locked goal (faint grey) almost vanishes against the black sky; worth a look in play-test.
- The features may poke through the shell when the hamster swings inside it; that changes pixels only.
- The background is the clear colour, black: a skybox needs a texture, and spec §11 rules textures out.

## Decisions

- 2026-10-22, spec §11 and issue 17 comments: views are model cameras on mocap bodies drawn through `mjCAMERA_FIXED`, not free `MjvCamera`s, so a later GPU renderer (which draws only model cameras) shares them.
- 2026-10-22, ticket hamsterball-v1/06: `pose` lives in the core package but the env calls it, so `Core` keeps its constructor; the level's `follow_yaw_deg` reaches it as an argument.
- 2026-10-23, ticket hitbox-view/03: the hitbox geoms get their own group, 4, which every default option already hid, so one flag shows them and no default changes.

## Tried and rejected

## History

- 2026-10-22, ticket hamsterball-v1/06: added. Agent finding: `mj_kinematics` alone leaves `cam_xpos` at the pose `mj_step` last saw (it's `mj_camlight` that computes cameras), so the first frame after reset rendered from the origin, all black. The env calls `mj_camlight` after posing.
- 2026-10-22, ticket hamsterball-v1/06: first picks were chase 2.5 m at 20° and follow 6 m at 40°. In sample frames the shell filled the chase view and hid the goal, and the goal sat on the follow view's top edge, so they moved to 3.0 m at 15° and 35°.
- 2026-10-22, merge of tickets 03 to 05: the marker colours they added (`MARKER_RGBA`, `HAZARD_RGBA`, `LOCKED_GOAL_RGBA` in `level.py`) join the look constants above, unchanged and still needing play-test. The env sets marker looks before rendering, so a collected item or key vanishes in the same env step's frame.
- 2026-10-22, ticket hamsterball-v1/06, agent finding from sample frames after merging tickets 03 to 05: on the 30 m `locked_goal` and `waypoint_loop` levels the first-person view's bottom rows were black, because MuJoCo's near clip plane (`znear` 0.01 × the model's extent) passed the ground in front of the eye. `model.build` now sets `znear` 0.001.
- 2026-10-22, code review of hamsterball-v1: the offscreen buffer was a fixed 1280×720, so a `view_size` wider or taller raised in `mujoco.Renderer`; spec §7 asks for at least `view_size`. Views now render at any `view_size` (checked at 800×1600).
- 2026-10-22, user play-test request: passed waypoints turn dim green; `PASSED_RGBA` (in [level.py](level.md)) joins the look constants, needing play-test like the rest.
- 2026-10-22, user play-test request: waypoints after the next render in the locked grey, renamed `LOCKED_RGBA` (was `LOCKED_GOAL_RGBA`).
- 2026-10-22, ticket mouse-camera/02: `CHASE_PITCH` is gone; the chase view reads the camera pitch from the core state, and the first-person view tilts by it minus its 15° default (the old chase pitch), so neither view changes at the default. The follow view keeps `FOLLOW_PITCH`.
- 2026-10-22, code review of mouse-camera: `PITCH_DEFAULT` comes from `model.py` (was `step.py`), so this module imports only from `model.py`. No behaviour change.
- 2026-10-22, ticket mouse-camera/04 (rename, no behaviour change): "camera yaw" is now **facing** (GLOSSARY.md): `pose` reads the facing from `userdata[FACING]`.
- 2026-10-23, ticket mouse-camera/05: added the tail. First pick 0.15 m long, 0.05 m base, `.55 .3 .1` brown; in a sample chase frame it read as a faint nub against the orange hamster, so it went to 0.17 m, 0.06 m, and a darker brown. A test caught that `geom_xpos` lags `mocap_pos` until the next `mj_kinematics`, which the env runs only when rendering; the core state holds the mocap pose, so a re-render still draws the tail where it was.
- 2026-10-23, code review of mouse-camera tickets 04–05 (no behaviour change): the comment over `FRAGMENT` says the view mocaps hold only their cameras (it said "no geoms" over a fragment that also holds the tail). Reusing `motion.quat` for the tail's quaternion was considered and left: it takes a motion dict and a time, so the call would be longer than the one-line quaternion. The tail's re-render test moved to `test_rerender.py` and goes through `rerender` (see [rerender.md](../data/rerender.md)).
- 2026-10-23, user request: the tail cone became the hamster's **features**, a tail ball and two ears, all spheres in the hamster's orange on one `features` mocap body at the hamster centre (was the `tail` body behind it). The mesh asset (`ASSET`) is gone. Same mocap count, so the core state's length is unchanged. In a sample chase frame the ears read clearly from behind, where the cone had shown only end-on as a dark nub.
- 2026-10-23, ticket hitbox-view/03: added `option`'s `hitboxes` flag. The geomgroup test checks group 4 is off in every default and on with the flag; the hitbox look constants join the table, needing play-test.
- 2026-10-23, code review of hitbox-view: group 4 holds only the centre dot; the kill plane moved from the level's MJCF to the game window's scene, so the model's extent (and env pixels) match `main`.
