# views.py

## Purpose

The three views of spec §11 (`chase`, `follow`, `first_person`): a model `<camera>` per view on a mocap body, posed from the core state once per env step, and the `MjvOption` each view renders with.

## Interface

- `VIEWS`: the view names, in that order.
- `FRAGMENT`: MJCF for the three mocap bodies (`view_chase`, `view_follow`, `view_first_person`), each holding a camera named after its view. `model.build` appends it after the tier's fragment.
- `pose(model, data, follow_yaw)`: writes `mocap_pos` and `mocap_quat` from fresh positions (after `mj_kinematics`). The cameras move only on the next `mj_kinematics` plus `mj_camlight`.
- `option(view)`: a fresh `MjvOption`. Groups 0 to 2 shown and 3 to 5 hidden for `chase` and `follow`; only group 0 for `first_person`.

The env ([env.md](../env.md)) calls `pose` every env step and owns the renderer.

## How it works

- `chase`: `CHASE_DIST` back from the shell centre along the camera yaw, pitched `CHASE_PITCH` down, looking at the shell centre.
- `follow`: the same construction at `FOLLOW_DIST` and `FOLLOW_PITCH`, with the level's `follow_yaw_deg` in place of the camera yaw.
- `first_person`: at the `hamster_eye` site, facing the camera yaw, level with the horizon.
- A MuJoCo camera looks along its −z with +y up, so the mocap's rotation matrix has columns right, up, −forward, with forward `(−sin yaw·cos pitch, cos yaw·cos pitch, −sin pitch)` (yaw 0 faces +y, as the camera yaw frame).

## Invariants and checks

- The mocaps carry no geoms, so they add no mass or contacts and never change a trajectory. They do add 21 numbers (`mocap_pos`, `mocap_quat`) to the core state, so the env poses them every env step, views or not, and `core_state` doesn't depend on the `views` kwarg.
- `uv run pytest tests/test_views.py`: shapes and dtype, chase and first person centred on the camera yaw, follow on the level's yaw, the geom groups, `obs_state=False`, rendering on a worker thread, the `MUJOCO_GL` default.

## Known quirks

- Look constants, picked by eye from sample frames and **needing play-test**. They change pixels, not trajectories, so changing them doesn't bump N:

  | Constant | Value |
  |---|---|
  | `CHASE_DIST`, `CHASE_PITCH` | 3.0 m, 15° down |
  | `FOLLOW_DIST`, `FOLLOW_PITCH` (the follow offset) | 6.0 m, 35° down |
  | `FOVY` (vertical) | chase 60°, follow 50°, first person 75° |
  | Marker colours (`MARKER_RGBA` in [level.py](level.md)) | goal `0.2 0.9 0.3 0.4` green, waypoints `0.2 0.5 1.0 0.4` blue, items `1.0 0.85 0.1 0.4` yellow, keys `1.0 0.4 0.1 0.4` orange, survive region `0.6 0.3 0.9 0.25` purple |
  | `HAZARD_RGBA`, `LOCKED_GOAL_RGBA` | `0.9 0.1 0.1 0.5` red; `0.5 0.5 0.5 0.15` faint grey |

- The chase and follow cameras aim at the shell centre, so at 84×84 the shell covers the near part of the goal when the goal is straight ahead.
- Pixels can't show the `first_person` group hiding: from inside, back-face culling already drops the shell sphere and the hamster. The test checks the option itself.
- Keys (orange) and hazards (red) are close in hue, and the locked goal (faint grey) almost vanishes against the black sky; worth a look in play-test.
- The background is the clear colour, black: a skybox needs a texture, and spec §11 rules textures out.

## Decisions

- 2026-10-22, spec §11 and issue 17 comments: views are model cameras on mocap bodies drawn through `mjCAMERA_FIXED`, not free `MjvCamera`s, so a later GPU renderer (which draws only model cameras) shares them.
- 2026-10-22, ticket hamsterball-v1/06: `pose` lives in the core package but the env calls it, so `Core` keeps its constructor; the level's `follow_yaw_deg` reaches it as an argument.

## Tried and rejected

## History

- 2026-10-22, ticket hamsterball-v1/06: added. Agent finding: `mj_kinematics` alone leaves `cam_xpos` at the pose `mj_step` last saw (it's `mj_camlight` that computes cameras), so the first frame after reset rendered from the origin, all black. The env calls `mj_camlight` after posing.
- 2026-10-22, ticket hamsterball-v1/06: first picks were chase 2.5 m at 20° and follow 6 m at 40°. In sample frames the shell filled the chase view and hid the goal, and the goal sat on the follow view's top edge, so they moved to 3.0 m at 15° and 35°.
- 2026-10-22, merge of tickets 03 to 05: the marker colours they added (`MARKER_RGBA`, `HAZARD_RGBA`, `LOCKED_GOAL_RGBA` in `level.py`) join the look constants above, unchanged and still needing play-test. The env sets marker looks before rendering, so a collected item or key vanishes in the same env step's frame.
- 2026-10-22, ticket hamsterball-v1/06, agent finding from sample frames after merging tickets 03 to 05: on the 30 m `locked_goal` and `waypoint_loop` levels the first-person view's bottom rows were black, because MuJoCo's near clip plane (`znear` 0.01 × the model's extent) passed the ground in front of the eye. `model.build` now sets `znear` 0.001.
