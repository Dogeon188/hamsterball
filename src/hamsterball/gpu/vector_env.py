"""The Warp vector env: `make_vec(..., backend="warp")`. N worlds on one device, each on a (level, task) of a level bank,
with the env-step logic as kernels in the same graph as the physics. See vector_env.md."""

import math

try:
    import mujoco_warp  # noqa: F401
    import torch
    import warp as wp
except ImportError as e:  # a plain install has none of them
    raise ImportError("the GPU backend needs the `gpu` extra: pip install 'hamsterball[gpu]'") from e

import gymnasium as gym
import numpy as np

from ..core.abstract_tier import AbstractTier
from ..core.generator import LevelGenerator
from ..core.model import DT
from ..core.step import FRAME_SKIP, Core
from ..env import DIFFICULTY, LEVELS_DIR, TASKS, _file_pool_difficulty, _levels
from . import bank as B
from . import env_step as E
from .batch_core import BatchCore

class WarpVectorEnv(gym.vector.VectorEnv):
    """`num_envs` worlds of `HamsterballEnv`, each on a (level, task) it draws from a level bank at every reset. `step`, `reset` and
    `step_to` take and return float32 torch tensors on the device; a finished world resets inside the same `step`
    (`AutoresetMode.SAME_STEP`). The tensors are views of buffers the next call overwrites: clone what you keep."""

    metadata = {"autoreset_mode": gym.vector.AutoresetMode.SAME_STEP}

    def __init__(
        self,
        num_envs,
        tasks=TASKS,
        levels=None,
        tier="abstract",
        camera_action=False,
        views=(),
        view_size=(84, 84),
        obs_state=True,
        frame_skip=FRAME_SKIP,
        time_limit_s=None,
        truncate=True,
        shaped=False,
        shaping_gamma=0.99,
        model_cache=16,  # meaningless here (one compiled model), accepted because registered ids pass it
        difficulty=DIFFICULTY,
        bank_size=1024,  # a generator pool's bank size; accepted, used when generator pools arrive
        device="cuda:0",
    ):
        if tier != "abstract":
            raise NotImplementedError(f"tier {tier!r}: v1 builds only the abstract tier")
        if tuple(views):
            raise NotImplementedError("views: pixel views are not built on the GPU yet")
        if not obs_state:
            raise ValueError("obs_state=False needs at least one view")
        self.num_envs, self.frame_skip, n = num_envs, frame_skip, num_envs
        self.tasks, self.camera_action, self.bank_size, self.device_name = list(tasks), camera_action, bank_size, device
        self.time_limit_s = time_limit_s
        self.truncate, self.shaped, self.shaping_gamma = truncate, shaped, shaping_gamma
        self.difficulty = DIFFICULTY
        pool, difficulty = self._pool([LEVELS_DIR] if levels is None else levels, difficulty)

        self.single_action_space = Core(AbstractTier(), camera_action=camera_action).action_space
        inf = np.float32(np.inf)
        self.single_observation_space = gym.spaces.Dict({
            "state": gym.spaces.Box(-inf, inf, (21,), np.float32), "goal": gym.spaces.Box(-inf, inf, (10,), np.float32)})
        self.observation_space = gym.vector.utils.batch_space(self.single_observation_space, n)
        self.action_space = gym.vector.utils.batch_space(self.single_action_space, n)

        self.core = self._build_core(pool, difficulty)
        self.torch_device = torch.device(self.core.device.alias if self.core.device.is_cuda else "cpu")
        with wp.ScopedDevice(self.core.device):
            z = lambda shape, dt=float: wp.zeros(shape, dtype=dt)  # noqa: E731
            self.todo = wp.zeros((n, self.markers.max_targets), dtype=wp.bool)  # the targets left, padded with False
            self.steps, self.phi, self.prev_pos = z(n, int), z(n), z(n, wp.vec3)
            self.reward, self.event, self.time, self.shaping = z(n), z(n), z(n), z(n)
            self.terminated, self.truncated, self.success, self.done = (z(n, wp.bool) for _ in range(4))
            self.failure = z(n, int)
            self.state, self.goal, self.final_state, self.final_goal = z((n, 21)), z((n, 10)), z((n, 21)), z((n, 10))
            self.all, self.none = wp.ones(n, dtype=wp.bool), z(n, wp.bool)
            self.spawn, self.facing = z(n, wp.vec3), z(n)
            self.level_index, self.task_idx = z(n, int), z(n, int)
            self.rng = z(n, wp.uint32)
            self.force_level, self.force_task = wp.full(n, -1, dtype=int), wp.full(n, -1, dtype=int)
        t = wp.to_torch
        self._obs = {"state": t(self.state), "goal": t(self.goal)}
        self._final_obs = {"state": t(self.final_state), "goal": t(self.final_goal)}
        terms = {"event": t(self.event), "time": t(self.time), "shaping": t(self.shaping)}
        self._step_info = {"success": t(self.success), "failure_reason": t(self.failure), "reward_terms": terms,
                           "level_index": t(self.level_index), "task_index": t(self.task_idx)}
        self._done = t(self.done)
        self._seed_rng()

    def _pool(self, levels, difficulty):
        """Validate a pool (`HamsterballEnv.set_levels`'s forms) and make it a list of levels: a file pool as given, a
        generator's `bank_size` levels drawn now. Returns it and the difficulty the generator was called with."""
        if isinstance(levels, LevelGenerator):
            self.generator = levels
            allowed = [t for t in self.tasks if t in levels.tasks]
            if not allowed:
                raise ValueError(f"the generator supports none of {self.tasks}")
            if difficulty is not None:
                self.difficulty = difficulty
            seeds = self.np_random.integers(2**63, size=self.bank_size)
            picks = self.np_random.integers(len(allowed), size=self.bank_size)
            return [levels(allowed[p], int(seed), self.difficulty) for seed, p in zip(seeds, picks)], self.difficulty
        _file_pool_difficulty(difficulty)
        pool = [lv for x in levels for lv in _levels(x)]
        if not any(t in lv.tasks for lv in pool for t in self.tasks):
            raise ValueError(f"no level in the pool supports any of {self.tasks}")
        self.generator, self.difficulty = None, DIFFICULTY
        return pool, DIFFICULTY

    def _build_core(self, pool, difficulty, keep=()):
        """A batched core on the pool and the tables the env-step kernels read per (level, task) entry."""
        core = BatchCore(AbstractTier(), pool, self.num_envs, self.camera_action, self.device_name, tasks=self.tasks, keep=keep)
        bank = core.bank
        rows = [(t, bank.levels[i].tasks[t], bank.levels[i].hazards) for i, t in bank.entries]
        for task, block, hazards in rows:
            if any("motion" in m or "attach" in m for m in [*hazards, *(m for _, m in E.targets(task, block))]):
                raise NotImplementedError("moving and attached markers are not built on the GPU yet")
        self.levels = pool
        limit = lambda b: self.time_limit_s if self.time_limit_s is not None else b["time_limit_s"]  # noqa: E731
        # the epsilon keeps a whole number of env steps from rounding up to one more, as in HamsterballEnv
        steps = [math.ceil(limit(b) / (self.frame_skip * DT) - 1e-9) for _, b, _ in rows]
        with wp.ScopedDevice(core.device):
            self.markers = E.Markers(rows, core.device)
            f = lambda x, dt: wp.array(x, dtype=dt)  # noqa: E731
            blocks = [b for _, b, _ in rows]
            self.entry_spawn = f(np.array([b["spawn"]["pos"] for b in blocks], np.float32), wp.vec3)
            self.entry_yaw = f(np.radians([b["spawn"]["yaw_deg"] for b in blocks]).astype(np.float32), float)
            self.entry_kill_z = f(np.array([bank.levels[i].kill_plane_z for i, _ in bank.entries], np.float32), float)
            self.entry_max_steps = f(np.array(steps, np.int32), int)
            # each marker row's geom in the padded model: its slot, found when the bank's tables were built
            geom = np.zeros((len(rows), self.markers.stride), np.int32)
            for e, slots in enumerate(core.bank.row_slots):
                geom[e, :len(slots)] = [core.bank.slot_ids[s] for s in slots]
            self.row_geom = f(geom.reshape(-1), int)
        return core

    def _seed_rng(self):
        with wp.ScopedDevice(self.core.device):
            wp.launch(B.seed_rng, self.num_envs, [int(self.np_random.integers(2**31)), self.rng])

    def set_levels(self, levels, difficulty=None):
        """Replace the level pool: a `LevelGenerator` (a bank of `bank_size` new levels is drawn), or a list of bundled
        names, paths, directories or `Level`s. This rebuilds the bank: it recompiles the padded model, reuploads every
        table and recaptures the step graph, which takes seconds to minutes for a large bank, so call it between
        rollouts, not per episode. A world mid-episode finishes it on the level it started on, and draws from the new
        pool at its next reset (its `level_index` is -1 until then, unless the new pool holds that very `Level`)."""
        pool, difficulty = self._pool(levels, difficulty)
        old = self.core
        states, entry, old_todo = old.states(), old.entry.numpy(), self.todo.numpy()
        new_level = np.full(self.num_envs, -1, np.int32)
        direct, keep = {}, []  # old entry -> (pool index) or kept entry index
        for e in sorted(set(entry.tolist())):
            lv, task = old.bank.levels[old.bank.entries[e][0]], old.bank.entries[e][1]
            same = next((i for i, p in enumerate(pool) if p is lv), None)
            if same is not None:
                direct[e] = same
            else:
                direct[e] = -1 - len(keep)
                keep.append((lv, task))
        self.core = self._build_core(pool, difficulty, keep)
        new = self.core
        with wp.ScopedDevice(new.device):
            bank = new.bank
            task = self.task_idx.numpy()
            for w in range(self.num_envs):
                d = direct[int(entry[w])]
                new_level[w] = d if d >= 0 else -1
                entry[w] = bank.entry_of[d, task[w]] if d >= 0 else bank.keep_entry[-1 - d]
            new.entry.assign(entry)
            self.level_index.assign(new_level)
            new.set_states(states)
            new.load_levels(self.all)
            todo = np.zeros((self.num_envs, self.markers.max_targets), bool)
            k = min(todo.shape[1], old_todo.shape[1])
            todo[:, :k] = old_todo[:, :k]
            self.todo = wp.array(todo, dtype=wp.bool)
            self._look()
        self._seed_rng()

    def reset(self, *, seed=None, options=None):
        """Reset every world onto a drawn (level, task). `options`: `level_index` and `task_index`, each an int or a
        `(num_envs,)` tensor, force a world's draw (-1 leaves it to the draw)."""
        super().reset(seed=seed)
        if seed is not None:
            self._seed_rng()
        self._force(options or {})
        with wp.ScopedDevice(self.core.device):
            self._draw(self.all)
            self.core.reset_masked(self.all, self.spawn, self.facing)
            wp.launch(E.reset_episode, self.num_envs, self._episode_args(self.all))
            self._observe(self.all, self.none)
            self._look()
            self.force_level.fill_(-1)  # the autoreset draws freely
            self.force_task.fill_(-1)
        for a in (self.reward, self.event, self.time, self.shaping, self.terminated, self.truncated, self.success, self.done):
            a.zero_()
        self.failure.zero_()
        return self._obs, {k: v for k, v in self._step_info.items()}

    def _force(self, options):
        """Load the reset options into the force arrays, checking each forced draw has an entry."""
        if unknown := set(options) - {"level_index", "task_index"}:
            raise ValueError(f"unknown reset options {sorted(unknown)}; choose from level_index, task_index")
        bank, tb = self.core.bank, self.core.tables
        want = {}
        for k in ("level_index", "task_index"):
            v = options.get(k, -1)
            v = v.cpu().numpy() if isinstance(v, torch.Tensor) else np.asarray(v)
            want[k] = np.broadcast_to(v, (self.num_envs,)).astype(np.int32)
        lv, tk = want["level_index"], want["task_index"]
        if ((lv < -1) | (lv >= bank.n_pool)).any() or ((tk < -1) | (tk >= len(TASKS))).any():
            raise ValueError(f"level_index must be in [0, {bank.n_pool}) and task_index in [0, {len(TASKS)}), or -1")
        row = np.where(tk >= 0, tk, len(TASKS))  # the last row of the draw's candidate table: any task
        bad = np.where(lv >= 0, np.where(tk >= 0, bank.entry_of[np.maximum(lv, 0), np.maximum(tk, 0)], bank.first_entry[np.maximum(lv, 0)]) < 0,
                       tb.count_host[row] == 0)
        if bad.any():
            w = int(np.flatnonzero(bad)[0])
            raise ValueError(f"world {w}: no pool level supports level_index {lv[w]} with task_index {tk[w]} (tasks {self.tasks})")
        self.force_level.assign(lv)
        self.force_task.assign(tk)

    def _draw(self, mask):
        """The masked worlds' (level, task) from the force arrays or the device RNG, and the spawn it gives."""
        c, tb = self.core, self.core.tables
        wp.launch(B.draw, self.num_envs, [mask, self.force_level, self.force_task, tb.cand, tb.count, tb.entry_of, self.rng,
                                          self.level_index, self.task_idx, c.entry])
        wp.launch(E.set_spawn, self.num_envs, [mask, c.entry, self.entry_spawn, self.entry_yaw, self.spawn, self.facing])

    def step(self, action):
        """One env step: `action` is (num_envs, action dim), a torch tensor or array; returns (obs, reward, terminated,
        truncated, info) as tensors. `info`: `success`, `failure_reason` (an index into `env_step.FAILURE_CODES`),
        `reward_terms`, `level_index`, `task_index`, and for worlds that ended, `final_obs` and `final_info` with their
        `_final_obs` and `_final_info` masks."""
        a = torch.as_tensor(action, dtype=torch.float32, device=self.torch_device).contiguous()
        self.core.step(a, self.frame_skip, post=self._env_step)
        return self._outputs()

    def step_to(self, core_states):
        """The env step after the physics, to recorded core states in the CPU layout, (num_envs, state size): loads them
        and runs the env-step logic with no physics. Returns what `step` returns."""
        if isinstance(core_states, torch.Tensor):
            core_states = core_states.cpu().numpy()
        self.core.set_states(core_states)
        with wp.ScopedDevice(self.core.device):
            self._env_step()
        return self._outputs()

    def _outputs(self):
        done = self._done
        info = {**self._step_info, "final_obs": self._final_obs, "_final_obs": done,
                "final_info": {k: self._step_info[k] for k in ("success", "failure_reason", "reward_terms")},
                "_final_info": done}
        return (self._obs, wp.to_torch(self.reward), wp.to_torch(self.terminated), wp.to_torch(self.truncated), info)

    def _episode_args(self, mask):
        m = self.markers
        return [mask, self.core.data.qpos, m.centre, self.core.entry, m.stride, m.ntargets, m.mode, self.todo, self.steps,
                self.prev_pos, self.phi]

    def _env_step(self):
        """The kernels after the physics (HamsterballEnv._advance and _observe), then the same-step autoreset."""
        n, m, d = self.num_envs, self.markers, self.core.data
        wp.launch(E.advance, n, [d.qpos, m.kind, m.a, m.b, m.rad, m.rot, m.half, m.centre, self.core.entry, m.stride,
                                 m.ntargets, m.n, m.mode, m.goal_row, m.survive, m.time_penalty, self.entry_kill_z,
                                 self.entry_max_steps, int(self.truncate), int(self.shaped), self.shaping_gamma,
                                 self.todo, self.steps, self.prev_pos, self.phi,
                                 self.reward, self.terminated, self.truncated, self.success, self.failure, self.event,
                                 self.time, self.shaping])
        self._observe(self.all, self.success)
        wp.launch(E.either, n, [self.terminated, self.truncated, self.done])
        wp.launch(E.stash_final, n, [self.done, self.state, self.goal, self.final_state, self.final_goal])
        self._draw(self.done)
        self.core.reset_masked(self.done, self.spawn, self.facing)
        wp.launch(E.reset_episode, n, self._episode_args(self.done))
        self._observe(self.done, self.none)
        self._look()

    def _look(self):
        """The episode task's marker colours into each world's `geom_rgba`, from the targets left."""
        m = self.markers
        wp.launch(E.look_markers, self.num_envs, [self.todo, self.core.entry, m.stride, m.ntargets, m.mode, self.row_geom,
                                                  m.role, m.base, self.core.model.geom_rgba])

    def _observe(self, mask, success):
        """`state` and `goal` for the worlds in `mask`, after the derived quantities `mj_step` leaves behind are refreshed
        from the state (_observe does the same on the CPU)."""
        import mujoco_warp as mjw

        c, d = self.core, self.core.data
        c._refresh()
        mjw.com_vel(c.model, d)
        mjw.subtree_vel(c.model, d)
        c.warp_tier.obs(self.num_envs, d, c.obs_out)
        wp.launch(E.observe, self.num_envs, [mask, d.xpos, d.xmat, d.qvel, d.subtree_com, d.subtree_linvel, d.userdata,
                                             c.obs_out["wall_contact"], self.markers.centre, self.todo, self.steps, c.entry,
                                             self.markers.stride, self.markers.ntargets, self.markers.mode,
                                             self.markers.survive, self.entry_max_steps, self.entry_kill_z, self.task_idx,
                                             success, self.state, self.goal])
