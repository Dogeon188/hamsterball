"""The Warp vector env: `make_vec(..., backend="warp")`. N worlds of one level on one device, with the env-step logic
as kernels in the same graph as the physics. See vector_env.md."""

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
from ..env import DIFFICULTY, LEVELS_DIR, TASKS, TIME_PENALTY, _file_pool_difficulty, _levels
from . import env_step as E
from .batch_core import BatchCore


class WarpVectorEnv(gym.vector.VectorEnv):
    """`num_envs` worlds of `HamsterballEnv`, all on one level and one task so far. `step`, `reset` and
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
        _file_pool_difficulty(difficulty)
        pool = [LEVELS_DIR] if levels is None else levels
        if isinstance(pool, LevelGenerator):
            raise NotImplementedError("a generator pool: the GPU backend takes one level so far")
        pool = [lv for x in pool for lv in _levels(x)]
        allowed = [t for t in tasks if any(t in lv.tasks for lv in pool)]
        if len(pool) != 1 or len(allowed) != 1:
            raise NotImplementedError(
                f"the GPU backend runs one task on one level so far; got {len(pool)} levels and tasks {allowed}")
        self.level, self.levels, self.task = pool[0], pool, allowed[0]
        block = self.level.tasks[self.task]
        if any("motion" in m or "attach" in m for m in [*self.level.hazards, *(m for _, m in E.targets(self.task, block))]):
            raise NotImplementedError("moving and attached markers are not built on the GPU yet")

        self.num_envs, self.frame_skip, n = num_envs, frame_skip, num_envs
        self.core = BatchCore(AbstractTier(), pool, n, camera_action, device)
        self.torch_device = torch.device(self.core.device.alias if self.core.device.is_cuda else "cpu")
        limit = time_limit_s if time_limit_s is not None else block["time_limit_s"]
        # the epsilon keeps a whole number of env steps from rounding up to one more, as in HamsterballEnv
        self.max_steps = math.ceil(limit / (frame_skip * DT) - 1e-9)
        self.truncate, self.shaped, self.shaping_gamma = truncate, shaped, shaping_gamma
        self.task_index = TASKS.index(self.task)

        self.single_action_space = Core(AbstractTier(), camera_action=camera_action).action_space
        inf = np.float32(np.inf)
        self.single_observation_space = gym.spaces.Dict({
            "state": gym.spaces.Box(-inf, inf, (21,), np.float32), "goal": gym.spaces.Box(-inf, inf, (10,), np.float32)})
        self.observation_space = gym.vector.utils.batch_space(self.single_observation_space, n)
        self.action_space = gym.vector.utils.batch_space(self.single_action_space, n)

        with wp.ScopedDevice(self.core.device):
            self.markers = E.Markers(self.task, block, self.level.hazards, self.core.device)
            self.geoms = wp.array([self.core.cpu_model.geom(g).id for g in self.markers.geom_names], dtype=int)
            self.todo = wp.ones((n, self.markers.ntargets), dtype=wp.bool)
            self.kill_z = float(self.level.kill_plane_z)
            z = lambda shape, dt=float: wp.zeros(shape, dtype=dt)  # noqa: E731
            self.steps, self.phi, self.prev_pos = z(n, int), z(n), z(n, wp.vec3)
            self.reward, self.event, self.time, self.shaping = z(n), z(n), z(n), z(n)
            self.terminated, self.truncated, self.success, self.done = (z(n, wp.bool) for _ in range(4))
            self.failure = z(n, int)
            self.state, self.goal, self.final_state, self.final_goal = z((n, 21)), z((n, 10)), z((n, 21)), z((n, 10))
            self.all, self.none = wp.ones(n, dtype=wp.bool), z(n, wp.bool)
            spawn = np.tile(block["spawn"]["pos"], (n, 1)).astype(np.float32)
            self.spawn = wp.array(spawn, dtype=wp.vec3)
            self.facing = wp.array(np.full(n, math.radians(block["spawn"]["yaw_deg"])), dtype=float)
            self.level_index, self.task_idx = z(n, int), wp.array(np.full(n, self.task_index), dtype=int)
        m = self.core.cpu_model  # other tasks' markers stay hidden for good; the episode task's are looked each env step
        rgba = np.tile(m.geom_rgba, (n, 1, 1))
        for g in range(m.ngeom):
            task, _, rest = m.geom(g).name.partition("/")
            if rest and task != self.task:
                rgba[:, g, 3] = 0
        self.core.model.geom_rgba.assign(rgba)
        t = wp.to_torch
        self._obs = {"state": t(self.state), "goal": t(self.goal)}
        self._final_obs = {"state": t(self.final_state), "goal": t(self.final_goal)}
        terms = {"event": t(self.event), "time": t(self.time), "shaping": t(self.shaping)}
        self._step_info = {"success": t(self.success), "failure_reason": t(self.failure), "reward_terms": terms,
                           "level_index": t(self.level_index), "task_index": t(self.task_idx)}
        self._done = t(self.done)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if options:
            raise NotImplementedError("reset options come with the level bank")
        with wp.ScopedDevice(self.core.device):
            self.core.reset_masked(self.all, self.spawn, self.facing)
            wp.launch(E.reset_episode, self.num_envs, self._episode_args(self.all))
            self._observe(self.all, self.none)
            self._look()
        for a in (self.reward, self.event, self.time, self.shaping, self.terminated, self.truncated, self.success, self.done):
            a.zero_()
        self.failure.zero_()
        return self._obs, {k: v for k, v in self._step_info.items()}

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
        return [mask, self.core.data.qpos, m.centre, m.ntargets, m.mode, self.todo, self.steps, self.prev_pos, self.phi]

    def _env_step(self):
        """The kernels after the physics (HamsterballEnv._advance and _observe), then the same-step autoreset."""
        n, m, d = self.num_envs, self.markers, self.core.data
        wp.launch(E.advance, n, [d.qpos, m.kind, m.a, m.b, m.rad, m.rot, m.half, m.centre, m.ntargets, m.n, m.mode,
                                 m.goal_row, int(self.task == "survive"), 0.0 if self.task == "survive" else TIME_PENALTY,
                                 self.kill_z, self.max_steps, int(self.truncate), int(self.shaped), self.shaping_gamma,
                                 self.todo, self.steps, self.prev_pos, self.phi,
                                 self.reward, self.terminated, self.truncated, self.success, self.failure, self.event,
                                 self.time, self.shaping])
        self._observe(self.all, self.success)
        wp.launch(E.either, n, [self.terminated, self.truncated, self.done])
        wp.launch(E.stash_final, n, [self.done, self.state, self.goal, self.final_state, self.final_goal])
        self.core.reset_masked(self.done, self.spawn, self.facing)
        wp.launch(E.reset_episode, n, self._episode_args(self.done))
        self._observe(self.done, self.none)
        self._look()

    def _look(self):
        """The episode task's marker colours into each world's `geom_rgba`, from the targets left."""
        m = self.markers
        wp.launch(E.look_markers, self.num_envs, [self.todo, m.ntargets, m.mode, self.geoms, m.role, m.base,
                                                  self.core.model.geom_rgba])

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
                                             c.obs_out["wall_contact"], self.markers.centre, self.todo, self.steps,
                                             self.markers.ntargets, self.markers.mode, int(self.task == "survive"),
                                             self.max_steps, self.kill_z, self.task_index, success, self.state, self.goal])
