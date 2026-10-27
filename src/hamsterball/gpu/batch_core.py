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
from ..core.model import (FACING, GAME_SIZE, HAM, HAM_DOF, HAM_QPOS, PITCH, PITCH_DEFAULT, R, SHELL, SHELL_DOF,
                          SHELL_QPOS, build)
from ..core.step import (DRAG_K, FIELD_HARD, FIELD_START, FRAME_SKIP, M_HAM, M_SHELL, MU, PITCH_MAX, PITCH_MIN, PITCH_STEP,
                         STATE, TAU, TURN_STEP)
from .abstract_tier import WarpAbstractTier
from .bank import BATCHED, TASK_NAMES, Bank, DeviceTables, body_kind
from .bodies import BodyTables
from .seesaws import SEESAW_FIELDS, SeesawTables

# core.step's constants, as kernel constants
_SHELL, _HAM, _SHELL_DOF, _HAM_DOF, _SHELL_QPOS, _HAM_QPOS = (wp.constant(x) for x in (SHELL, HAM, SHELL_DOF, HAM_DOF, SHELL_QPOS, HAM_QPOS))
_FACING, _PITCH = wp.constant(FACING), wp.constant(PITCH)
_DRAG_K, _FIELD_START, _FIELD_HARD, _MU = (wp.constant(x) for x in (DRAG_K, FIELD_START, FIELD_HARD, MU))
_M_HAM, _M_SHELL = wp.constant(M_HAM), wp.constant(M_SHELL)
_PITCH_MIN, _PITCH_MAX, _PITCH_DEFAULT = (wp.constant(x) for x in (PITCH_MIN, PITCH_MAX, PITCH_DEFAULT))
_TURN_STEP, _PITCH_STEP, _TAU, _PI = (wp.constant(x) for x in (TURN_STEP, PITCH_STEP, TAU, TAU / 2))

# the bank of a core made with no levels: an infinite plane (the CPU core's FLAT)
FLAT_LEVEL = Level.from_dict({
    "kill_plane_z": -10, "geoms": [{"type": "plane", "size": [0, 0, 0.1], "rgba": [0.35, 0.4, 0.35, 1]}],
    "tasks": {"reach_goal": {"spawn": {"pos": [0, 0, R]}, "goal": {"type": "sphere", "pos": [0, 8, R], "radius": 1}, "time_limit_s": 30}}})

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

    def __init__(self, tier=None, levels=(FLAT_LEVEL,), num_envs=1, camera_action=False, device="cpu", graph=True,
                 tasks=TASK_NAMES, keep=()):
        """`levels`: the bank, a list of `Level`s, padded into one model (see bank.py); `tasks` limits the (level, task)
        entries a world can be on, and `keep` appends entries a world still needs (see `bank.Bank`). `device`: "cpu" or
        "cuda:N". Collapsing tiles raise `NotImplementedError`."""
        tier = tier or AbstractTier()
        if type(tier) not in TIERS:
            raise NotImplementedError(f"{type(tier).__name__} has no batched drive/obs yet")
        self.bank = Bank(levels, tasks, keep)
        wp.config.quiet = True
        try:
            self.device = wp.get_device(device)
        except ValueError as e:  # Warp's error for a device it doesn't have
            raise RuntimeError(f"no Warp device {device!r}: {e}") from e
        self.tier, self.num_envs, self.camera_action = tier, num_envs, camera_action
        self.graph = graph and self.device.is_cuda
        valid = np.flatnonzero(self.bank.first_entry >= 0)
        if not len(valid):
            raise ValueError("no level in the bank has an entry for the tasks")
        self.level_index = np.full(num_envs, valid[0], np.int32)
        self.cpu_model = build(tier, self.bank.mjcf(), GAME_SIZE, self.bank.body_mjcf())
        self.adim = tier.action_space().shape[0] + 2 * camera_action
        with wp.ScopedDevice(self.device):
            seesaws = self.bank.nhinge + self.bank.nball > 0
            self.model = mjw.put_model(self.cpu_model, batch_sizes={k: num_envs for k in BATCHED + (SEESAW_FIELDS if seesaws else ())})
            if seesaws:  # the solver's tolerance scales with the level's mean inertia: one per world
                self.model.stat.meaninertia = wp.zeros(num_envs, dtype=float)
            self.data = mjw.make_data(self.cpu_model, nworld=num_envs)
            self.tables = DeviceTables(self.bank, self.cpu_model, self.device)
            self.bodies = BodyTables(self.bank, self.cpu_model, self.device)
            self.seesaws = SeesawTables(self.bank, self.cpu_model, self.device)
            self.on = wp.ones(num_envs, dtype=wp.bool)
            self.entry = wp.zeros(num_envs, dtype=int)  # each world's (level, task) entry in the bank's tables
            self.warp_tier = TIERS[type(tier)](tier, self.device)
            self.action = wp.zeros((num_envs, self.adim), dtype=float)
            self.obs_out = self.warp_tier.obs_arrays(num_envs, self.device)
            self.nstate = mujoco.mj_stateSize(self.cpu_model, STATE)
            self._state = wp.zeros((num_envs, self.nstate), dtype=float)
        self._layouts = self._state_layouts()
        self._graphs = {}
        self.reset()

    def reset(self, worlds=None, bank_index=None, spawn=(0.0, 0.0, R), facing=0.0):
        """Reset the given worlds (indices; default all) onto bank levels `bank_index` (default 0, or the level they are
        on), as `Core.reset`. A world takes its level's first task's markers."""
        n = self.num_envs
        worlds = np.arange(n) if worlds is None else np.atleast_1d(worlds)
        mask = np.zeros(n, bool)
        mask[worlds] = True
        spawns, facings = np.zeros((n, 3), np.float32), np.zeros(n, np.float32)
        spawns[worlds], facings[worlds] = spawn, facing
        if bank_index is not None:
            self.level_index[worlds] = bank_index
        entry = self.entry.numpy()
        entry[worlds] = self.bank.first_entry[self.level_index[worlds]]
        if (entry[worlds] < 0).any():
            raise ValueError(f"bank level {self.level_index[worlds][entry[worlds] < 0][0]} has no entry for the tasks")
        with wp.ScopedDevice(self.device):
            self.entry.assign(entry)
            self.reset_masked(wp.array(mask, dtype=wp.bool), wp.array(spawns, dtype=wp.vec3), wp.array(facings))

    def reset_masked(self, mask, spawns, facings):
        """`reset` on the device: `mask` (bool), `spawns` (vec3) and `facings` over the worlds, all Warp arrays, so it
        can run inside a captured graph."""
        mjw.reset_data(self.model, self.data, mask)
        self.load_levels(mask)
        self.bodies.drive(mask, self.entry, self.data.time, self.data.qpos, self.data.qvel)
        wp.launch(_shell_spawn, self.num_envs, [mask, spawns, facings, self.data.qpos, self.data.userdata])
        self.warp_tier.reset(self.num_envs, mask, spawns, self.data)
        self._refresh()

    def load_levels(self, mask):
        """Write the masked worlds' levels into the model: each world's slots take its `entry`'s values, and the unused
        ones park. Positions refresh at the next step."""
        self.tables.load(mask, self.entry, self.model, self.data)
        self.seesaws.load(mask, self.entry, self.model)

    def _refresh(self):
        """Positions (xpos, xipos, xmat, subtree_com) from qpos: what `mj_kinematics` and `mj_comPos` leave."""
        mjw.kinematics(self.model, self.data)
        mjw.com_pos(self.model, self.data)

    def _physics_step(self, n):
        """One physics step, in `core.step.physics_step`'s order."""
        m, d, nw = self.model, self.data, self.num_envs
        turn_col = self.adim - 2 if self.camera_action else -1
        d.xfrc_applied.zero_()
        self.bodies.drive(self.on, self.entry, d.time, d.qpos, d.qvel)
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

    def step(self, action, n=FRAME_SKIP, post=None):
        """Hold one action per world, shape (num_envs, action dim), over n physics steps: one graph launch on CUDA.
        `post` (a callable that launches kernels) runs after them, inside the graph; a graph is kept per `n`, so pass
        the same `post` every call."""
        torch_tensor = type(action).__module__.split(".")[0] == "torch"
        with wp.ScopedDevice(self.device):
            wp.copy(self.action, wp.from_torch(action.detach()) if torch_tensor else wp.array(action, dtype=float))
            if self.graph:
                if n not in self._graphs:
                    with wp.ScopedCapture() as capture:
                        self._launch(n, post)
                    self._graphs[n] = capture.graph
                wp.capture_launch(self._graphs[n])
            else:
                self._launch(n, post)

    def launch(self, key, fn):
        """Run `fn` (a callable that launches kernels) as one graph kept under `key` on CUDA, eagerly on the CPU device."""
        with wp.ScopedDevice(self.device):
            if not self.graph:
                return fn()
            if key not in self._graphs:
                with wp.ScopedCapture() as capture:
                    fn()
                self._graphs[key] = capture.graph
            wp.capture_launch(self._graphs[key])

    def _launch(self, n, post):
        for _ in range(n):
            self._physics_step(n)
        self.warp_tier.obs(self.num_envs, self.data, self.obs_out)
        if post:
            post()

    def obs(self):
        """The tier's observation after the last step, one numpy array per entry, over the worlds."""
        return {k: v.numpy() for k, v in self.obs_out.items()}

    def _state_layouts(self):
        """The padded state vector's sizes per segment (time, qpos, qvel, warm start, `qfrc_applied`, `xfrc_applied`, mocap,
        userdata), the address of each body slot in the segments that hold bodies, and the template of a world with none
        of its slots in use. The slots are the last bodies, joints and dofs of the model, in `bank.body_names` order; a
        level's own bodies take the slots `bank.slot_of` gives (see `_layout`)."""
        m = self.cpu_model
        sizes = [1, m.nq, m.nv, m.nv, m.nv, 6 * m.nbody, 3 * m.nmocap, 4 * m.nmocap, m.nuserdata]
        assert sum(sizes) == self.nstate, (sizes, self.nstate)
        ids = [m.body(n) for n in self.bank.body_names]
        slots = [(int(m.jnt_qposadr[b.jntadr[0]]), int(m.jnt_dofadr[b.jntadr[0]]), int(b.id)) for b in ids]
        template = np.zeros(self.nstate, np.float32)
        template[1:1 + m.nq] = m.qpos0  # an unused slot rests at the model's qpos0 (a free joint's unit quaternion)
        self._layout_cache = {}
        return sizes, slots, template

    def _layout(self, level):
        """The indices into the padded state vector that make up the CPU core-state vector of `level`: each segment's
        prefix (the shell, the hamster, the views), then the level's bodies in its own order, each at its slot's address.
        Cached per sequence of body kinds."""
        sizes, slots, _ = self._layouts
        kinds = tuple(body_kind(b) for b in level.bodies)
        if kinds not in self._layout_cache:
            m = self.cpu_model
            nq0, nv0, nb0 = (slots[0][0], slots[0][1], slots[0][2]) if slots else (m.nq, m.nv, m.nbody)
            size = {"b": (7, 6), "h": (1, 1), "n": (4, 3)}
            off = np.cumsum([0] + sizes)
            qa, da, ba = [np.arange(nq0)], [np.arange(nv0)], [np.arange(nb0)]
            for si, k in zip(self.bank.slot_of(level), kinds):
                nq, nv = size[k]
                qa.append(slots[si][0] + np.arange(nq)), da.append(slots[si][1] + np.arange(nv)), ba.append([slots[si][2]])
            qa, da, ba = (np.concatenate(x).astype(int) for x in (qa, da, ba))
            xf = (6 * ba[:, None] + np.arange(6)).ravel()
            parts = [np.arange(1), off[1] + qa, off[2] + da, off[3] + da, off[4] + da, off[5] + xf]
            parts += [np.arange(off[j], off[j + 1]) for j in (6, 7, 8)]
            self._layout_cache[kinds] = np.concatenate(parts)
        return self._layout_cache[kinds]

    def _level_of(self, world_or_entry, entry=False):
        """The `Level` a world (or an entry) is on."""
        e = world_or_entry if entry else int(self.entry.numpy()[world_or_entry])
        return self.bank.levels[self.bank.entries[e][0]]

    def state(self, world):
        """One world's state in the CPU core-state layout (`Core.state()`) of its level, float32 precision."""
        with wp.ScopedDevice(self.device):
            mjw.get_state(self.model, self.data, self._state, int(STATE))
        return self._state.numpy()[world][self._layout(self._level_of(world))].astype(np.float64)

    def set_state(self, world, s):
        """Load a CPU core state into one world, as `Core.set_state` (float32 precision; the vector carries the
        solver's warm start, as on the CPU). `s` is in the layout of the world's level. Positions refresh at the next
        step."""
        row = self._state.numpy()
        row[world] = self._pad(world, s, self._level_of(world))
        active = np.zeros(self.num_envs, bool)
        active[world] = True
        self._load(row, active)

    def _pad(self, world, s, level):
        idx = self._layout(level)
        s = np.asarray(s)
        if len(s) != len(idx):
            raise ValueError(f"world {world}: a state of {len(s)} numbers, its level's layout has {len(idx)}")
        row = self._layouts[2].copy()
        row[idx] = s
        return row

    def states(self):
        """Every world's state in the padded layout, `(num_envs, padded state size)`, float32: equal to the CPU layout
        when no level in the bank has bodies. `cpu_states()` gives each world's own."""
        with wp.ScopedDevice(self.device):
            mjw.get_state(self.model, self.data, self._state, int(STATE))
        return self._state.numpy().copy()

    def cpu_states(self):
        """Every world's state in the CPU layout of its level: a list of float32 vectors, which differ in length when the
        levels' bodies do."""
        rows, entry = self.states(), self.entry.numpy()
        return [rows[w][self._layout(self._level_of(e, True))] for w, e in enumerate(entry)]

    def set_states(self, states):
        """`set_state` for every world at once: a `(num_envs, size)` array, or a list of one vector per world, in each
        world's level's CPU layout."""
        entry = self.entry.numpy()
        self._load(np.stack([self._pad(w, states[w], self._level_of(e, True)) for w, e in enumerate(entry)]),
                   np.ones(self.num_envs, bool))

    def _load(self, rows, active):
        with wp.ScopedDevice(self.device):
            self._state.assign(rows)
            mjw.set_state(self.model, self.data, self._state, int(STATE), wp.array(active, dtype=wp.bool))
            self._refresh()
