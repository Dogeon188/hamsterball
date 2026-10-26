"""The batched core: N worlds of the CPU core on one Warp device. See batch_core.md."""

try:
    import mujoco_warp as mjw
    import warp as wp
except ImportError as e:  # a plain install has no warp
    raise ImportError("the GPU backend needs the `gpu` extra: pip install 'hamsterball[gpu]'") from e

import mujoco
import numpy as np

from ..core.abstract_tier import AbstractTier
from ..core.level import Level
from ..core.model import (FACING, FLAT, GAME_SIZE, HAM, HAM_DOF, HAM_QPOS, PITCH, PITCH_DEFAULT, R, SHELL, SHELL_DOF,
                          SHELL_QPOS, build)
from ..core.step import (DRAG_K, FIELD_HARD, FIELD_START, FRAME_SKIP, M_HAM, M_SHELL, MU, PITCH_MAX, PITCH_MIN, PITCH_STEP,
                         STATE, TAU, TURN_STEP)
from .abstract_tier import WarpAbstractTier

# core.step's constants, as kernel constants
_SHELL, _HAM, _SHELL_DOF, _HAM_DOF, _SHELL_QPOS, _HAM_QPOS = (wp.constant(x) for x in (SHELL, HAM, SHELL_DOF, HAM_DOF, SHELL_QPOS, HAM_QPOS))
_FACING, _PITCH = wp.constant(FACING), wp.constant(PITCH)
_DRAG_K, _FIELD_START, _FIELD_HARD, _MU = (wp.constant(x) for x in (DRAG_K, FIELD_START, FIELD_HARD, MU))
_M_HAM, _M_SHELL = wp.constant(M_HAM), wp.constant(M_SHELL)
_PITCH_MIN, _PITCH_MAX, _PITCH_DEFAULT = (wp.constant(x) for x in (PITCH_MIN, PITCH_MAX, PITCH_DEFAULT))
_TURN_STEP, _PITCH_STEP, _TAU, _PI = (wp.constant(x) for x in (TURN_STEP, PITCH_STEP, TAU, TAU / 2))

TIERS = {AbstractTier: WarpAbstractTier}  # a tier's CPU class -> its batched twin; the articulated tier joins in effort 1


@wp.func
def wrap(a: float):
    """Angle into (-pi, pi]."""
    a = a - _TAU * wp.floor(a / _TAU + 0.5)
    if a == -_PI:
        a = _PI
    return a


@wp.kernel
def turn_facing(action: wp.array2d(dtype=float), turn_col: int, n: int, userdata: wp.array2d(dtype=float)):
    """Turn the facing by this physics step's share of the action's turn (turn_col < 0: no camera flag, no turn)."""
    w = wp.tid()
    d = float(0.0)
    if turn_col >= 0:
        d = _TURN_STEP * action[w, turn_col] / float(n)
    userdata[w, _FACING] = wrap(userdata[w, _FACING] + d)


@wp.kernel
def camera_pitch(action: wp.array2d(dtype=float), pitch_col: int, n: int, userdata: wp.array2d(dtype=float)):
    """Pitch by this physics step's share of the action's cam_pitch and clamp. Moves no physics."""
    w = wp.tid()
    d = float(0.0)
    if pitch_col >= 0:
        d = _PITCH_STEP * action[w, pitch_col] / float(n)
    userdata[w, _PITCH] = wp.min(wp.max(userdata[w, _PITCH] + d, _PITCH_MIN), _PITCH_MAX)


@wp.kernel
def drag(qvel: wp.array2d(dtype=float), xfrc_applied: wp.array2d(dtype=wp.spatial_vector)):
    """-k |v| v on the shell, grounded or not."""
    w = wp.tid()
    v = wp.vec3(qvel[w, _SHELL_DOF], qvel[w, _SHELL_DOF + 1], qvel[w, _SHELL_DOF + 2])
    xfrc_applied[w, _SHELL] = xfrc_applied[w, _SHELL] + wp.spatial_vector(-_DRAG_K * wp.length(v) * v, wp.vec3())


@wp.kernel
def contain(
    resting_radius: float,
    xpos: wp.array2d(dtype=wp.vec3),
    subtree_com: wp.array2d(dtype=wp.vec3),
    qpos: wp.array2d(dtype=float),
    qvel: wp.array2d(dtype=float),
):
    """Keep the hamster inside: an inelastic impulse past FIELD_START, a position stop at FIELD_HARD. Needs fresh xpos
    and subtree_com; after a position stop the caller refreshes them."""
    w = wp.tid()
    r = subtree_com[w, _HAM] - xpos[w, _SHELL]
    dist = wp.length(r)
    depth = dist - resting_radius - _FIELD_START
    if depth <= 0.0:
        return
    u = r / dist
    v = wp.vec3(
        qvel[w, _HAM_DOF] - qvel[w, _SHELL_DOF],
        qvel[w, _HAM_DOF + 1] - qvel[w, _SHELL_DOF + 1],
        qvel[w, _HAM_DOF + 2] - qvel[w, _SHELL_DOF + 2],
    )
    closing = wp.dot(v, u)
    if closing > 0.0:
        j = wp.min(1.0, depth / (_FIELD_HARD - _FIELD_START)) * _MU * closing
        for i in range(3):
            qvel[w, _HAM_DOF + i] -= j / _M_HAM * u[i]
            qvel[w, _SHELL_DOF + i] += j / _M_SHELL * u[i]
    over = depth - (_FIELD_HARD - _FIELD_START)
    if over > 0.0:  # a sustained load still creeps past the impulse, so stop it in position
        for i in range(3):
            qpos[w, _HAM_QPOS + i] -= over * _M_SHELL / (_M_HAM + _M_SHELL) * u[i]
            qpos[w, _SHELL_QPOS + i] += over * _M_HAM / (_M_HAM + _M_SHELL) * u[i]


@wp.kernel
def _shell_spawn(mask: wp.array(dtype=wp.bool), spawn: wp.array(dtype=wp.vec3), facing: wp.array(dtype=float),
                 qpos: wp.array2d(dtype=float), userdata: wp.array2d(dtype=float)):
    w = wp.tid()
    if not mask[w]:
        return
    for i in range(3):
        qpos[w, _SHELL_QPOS + i] = spawn[w][i]
    userdata[w, _FACING] = facing[w]
    userdata[w, _PITCH] = _PITCH_DEFAULT


class BatchCore:
    """`num_envs` worlds of `core.Core`, stepped together. Like `Core` it holds constants (the model) plus the state in
    `data`; a world's state is read and written in the CPU core-state layout of its level."""

    def __init__(self, tier=None, levels=(FLAT,), num_envs=1, camera_action=False, device="cpu", graph=True):
        """`levels`: the bank, each a `Level` or MJCF sections as `core.model.build` takes them. This slice batches only
        levels that compile to one model (a plane, no bodies). `device`: "cpu" or "cuda:N"."""
        tier = tier or AbstractTier()
        if type(tier) not in TIERS:
            raise NotImplementedError(f"{type(tier).__name__} has no batched drive/obs yet")
        sections = [lv if isinstance(lv, str) else lv.mjcf() for lv in levels]
        if any(isinstance(lv, Level) and (lv.bodies or lv.collapse) for lv in levels) or len(set(sections)) != 1:
            raise NotImplementedError("levels that need padded slots (bodies, tiles, other terrain) come in a later slice")
        wp.config.quiet = True
        self.device = wp.get_device(device)  # raises RuntimeError for a CUDA device that doesn't exist
        self.tier, self.num_envs, self.camera_action = tier, num_envs, camera_action
        self.graph = graph and self.device.is_cuda
        self.nlevels = len(sections)
        self.level_index = np.zeros(num_envs, np.int32)
        self.cpu_model = build(tier, sections[0], GAME_SIZE)
        self.adim = tier.action_space().shape[0] + 2 * camera_action
        with wp.ScopedDevice(self.device):
            self.model = mjw.put_model(self.cpu_model)
            self.data = mjw.make_data(self.cpu_model, nworld=num_envs)
            self.warp_tier = TIERS[type(tier)](tier, self.device)
            self.action = wp.zeros((num_envs, self.adim), dtype=float)
            self.obs_out = self.warp_tier.obs_arrays(num_envs, self.device)
            self.nstate = mujoco.mj_stateSize(self.cpu_model, STATE)
            self._state = wp.zeros((num_envs, self.nstate), dtype=float)
        self._graphs = {}
        self.reset()

    def reset(self, worlds=None, bank_index=None, spawn=(0.0, 0.0, R), facing=0.0):
        """Reset the given worlds (indices; default all) onto bank levels `bank_index` (default 0), as `Core.reset`."""
        n = self.num_envs
        worlds = np.arange(n) if worlds is None else np.atleast_1d(worlds)
        mask = np.zeros(n, bool)
        mask[worlds] = True
        spawns, facings = np.zeros((n, 3), np.float32), np.zeros(n, np.float32)
        spawns[worlds], facings[worlds] = spawn, facing
        if bank_index is not None:
            self.level_index[worlds] = bank_index
        with wp.ScopedDevice(self.device):
            mask, spawns, facings = wp.array(mask, dtype=wp.bool), wp.array(spawns, dtype=wp.vec3), wp.array(facings)
            mjw.reset_data(self.model, self.data, mask)
            wp.launch(_shell_spawn, n, [mask, spawns, facings, self.data.qpos, self.data.userdata])
            self.warp_tier.reset(n, mask, spawns, self.data)
            self._refresh()

    def _refresh(self):
        """Positions (xpos, xipos, xmat, subtree_com) from qpos: what `mj_kinematics` and `mj_comPos` leave."""
        mjw.kinematics(self.model, self.data)
        mjw.com_pos(self.model, self.data)

    def _physics_step(self, n):
        """One physics step, in `core.step.physics_step`'s order."""
        m, d, nw = self.model, self.data, self.num_envs
        turn_col = self.adim - 2 if self.camera_action else -1
        d.xfrc_applied.zero_()
        self._refresh()
        wp.launch(turn_facing, nw, [self.action, turn_col, n, d.userdata])
        wp.launch(camera_pitch, nw, [self.action, turn_col + 1 if turn_col >= 0 else -1, n, d.userdata])
        wp.launch(drag, nw, [d.qvel, d.xfrc_applied])
        wp.launch(contain, nw, [self.tier.resting_radius, d.xpos, d.subtree_com, d.qpos, d.qvel])
        # contain's hard stop moved qpos: refresh again where the CPU calls mj_forward. mj_step recomputes everything
        # else, and a world that didn't move gets the same positions back
        self._refresh()
        self.warp_tier.drive(nw, self.action, d)
        mjw.step(m, d)

    def step(self, action, n=FRAME_SKIP):
        """Hold one action per world, shape (num_envs, action dim), over n physics steps: one graph launch on CUDA."""
        torch_tensor = type(action).__module__.split(".")[0] == "torch"
        with wp.ScopedDevice(self.device):
            wp.copy(self.action, wp.from_torch(action.detach()) if torch_tensor else wp.array(action, dtype=float))
            if self.graph:
                if n not in self._graphs:
                    with wp.ScopedCapture() as capture:
                        self._launch(n)
                    self._graphs[n] = capture.graph
                wp.capture_launch(self._graphs[n])
            else:
                self._launch(n)

    def _launch(self, n):
        for _ in range(n):
            self._physics_step(n)
        self.warp_tier.obs(self.num_envs, self.data, self.obs_out)

    def obs(self):
        """The tier's observation after the last step, one numpy array per entry, over the worlds."""
        return {k: v.numpy() for k, v in self.obs_out.items()}

    def state(self, world):
        """One world's state in the CPU core-state layout (`Core.state()`), float32 precision."""
        with wp.ScopedDevice(self.device):
            mjw.get_state(self.model, self.data, self._state, int(STATE))
        return self._state.numpy()[world].astype(np.float64)

    def set_state(self, world, s):
        """Load a CPU core state into one world, as `Core.set_state` (float32 precision; the vector carries the
        solver's warm start, as on the CPU). Positions refresh at the next step."""
        row = self._state.numpy()
        row[world] = s
        with wp.ScopedDevice(self.device):
            self._state.assign(row)
            active = np.zeros(self.num_envs, bool)
            active[world] = True
            mjw.set_state(self.model, self.data, self._state, int(STATE), wp.array(active, dtype=wp.bool))
            self._refresh()
