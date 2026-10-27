# views.py (gpu)

## Purpose

The Warp vector env's pixel views ([vector_env.md](vector_env.md)): `core.views.pose` ([views.md](../core/views.md)) as a kernel over every world, and Warp's ray tracer drawing `chase`, `follow` and `first_person` as uint8 `(N, H, W, 3)` arrays. The CPU views stay the reference; pixels are not promised to match the CPU renderer's.

## Interface

- `Views(core, views, size, follow_yaw)`: `core` a `BatchCore`, `views` a subset of `core.views.VIEWS`, `size` `(height, width)`, `follow_yaw` a float Warp array of each bank entry's `follow_yaw_deg` in radians. With `views=()` it builds no render context and only poses.
- `Views.pose()`: launches the `pose` kernel, which writes `mocap_pos` and `mocap_quat` of the three view bodies and the `features` body (the hamster's tail and ears) from `xpos`, `site_xpos` (the `hamster_eye` site), `userdata[FACING]`, `userdata[PITCH]` and the world's follow yaw. It reads positions, so run it after the kinematics refresh. Poses are in the core state (mocap), so the env poses every step whether or not views are on, as the CPU env does.
- `Views.render(out=None)`: runs `mjw.kinematics` and `mjw.camlight` (mocap poses to bodies, bodies to cameras), then for each context `refit_bvh`, `render`, `get_rgb`, and a kernel that rounds the floats to bytes into `out[view]` (default `Views.pixels`, a dict of uint8 Warp arrays `(N, H, W, 3)`).

## How it works

- Two render contexts, both without textures or shadows: `chase` and `follow` share one (geom groups 0 and 2), `first_person` has its own (group 0 only, so the hamster is hidden as on the CPU; the shell sphere, group 1, is hidden there too). Inner tiles (group 3) and hitbox geoms (group 4) are in neither list. A context exists only if one of its views was asked for.
- The model, lights and headlight are the CPU env's (`core.model.build` compiles them into the one batch model), so the look constants carry over: the one directional light, MuJoCo's default headlight, no textures, shadows off, a black background.
- The pose is the CPU's: the rotation is `Rz(yaw) Rx(pi/2 - pitch)` (the CPU builds the same matrix and converts it), the camera `dist` behind its target along the view direction; first person sits at the eye site with the pitch minus `PITCH_DEFAULT`. The constants (`CHASE_DIST`, `FOLLOW_DIST`, `FOLLOW_PITCH`) are imported from `core.views`.
- Marker looks need no work here. Each world's marker colours are per-world `geom_rgba` ([vector_env.md](vector_env.md): `look_markers`), alpha 0 hides a geom, and `refit_bvh` before each render keeps the tracer's BVH in step with geoms that moved (ticket 01).
- `Views.render` is called outside the captured step graph: the vector env's `step` captures the physics and the env-step kernels, and draws after it.

## Invariants and checks

- `uv run --extra gpu pytest tests/test_warp_views.py`: shapes, dtype, device and observation space per view subset and `view_size`; `obs_state=False`; an unknown view; the final observation's views are the ended step's; the view and feature mocap poses equal the CPU's after replaying recorded CPU states through `step_to` (position within 2e-5, quaternion up to sign; the replay uses the camera flag so facing and pitch vary, and a follow yaw); and, per world, a collected item vanishes and the unlock goal changes colour in that world's pixels only (counted by colour in the chase and follow views); first person sees the goal ahead.
- No test compares pixels with the CPU renderer.

## Known quirks

- **Translucency.** The ray tracer draws any alpha above 0 fully opaque. The translucent markers (alpha under 1: the goal, items, keys, hazards, the locked grey and passed-waypoint green, the survive region) therefore render solid and occlude what is behind them; a tall hazard hides what is behind it.
- **The shell is not drawn in chase and follow.** Its outer sphere (alpha 0.25) would render opaque and hide the hamster and its features, which are what show the facing. The views show the hamster (orange) and its features, not the shell's outline. The CPU renderer draws the shell translucent over the hamster.
- `get_rgb` takes the camera's position in the context's active list, not its model index (index 2, `first_person`, is 0 in its own context). A wrong index reads out of bounds and segfaults.
- The context's `cam_res` is `(width, height)`; `view_size` here and on the CPU is `(height, width)`.
- Marker colours and every look constant are the CPU's, which await play-test ([views.md](../core/views.md)).

## Decisions

- 2026-10-27, spec views section and ticket gpu-backend/10: two render contexts, one for `chase` and `follow` and one for `first_person` with groups 1 and 2 off, because groups are per context, not per world.
- 2026-10-27, ticket gpu-backend/10: chase and follow also hide group 1, the shell. Spec: they show groups 0 to 2 as the CPU's `option` does. The tracer's opaque shell hid the hamster in every frame, so the facing features could not be seen; hiding it keeps the hamster visible, and the shell's outline is lost instead.
- 2026-10-27, ticket gpu-backend/10: the final observation holds the ended step's views (drawn before the autoreset), as the CPU env's does, at the cost of a second render on steps where a world ended.

## Tried and rejected

- 2026-10-27, ticket gpu-backend/10: all three views in one context with `first_person` hiding groups per camera. Warp's `enabled_geom_groups` belongs to the context, so there is no per-camera mask.

## History

- 2026-10-27, ticket gpu-backend/10: first version. Agent finding while building: `get_rgb`'s camera argument is an index into the context's active cameras; passing the model camera id for `first_person` (2) in its own one-camera context segfaulted the process.
