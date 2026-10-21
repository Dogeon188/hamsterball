"""The Gymnasium env over the core (spec §9, §10): level and task per episode, observations, markers, reward, info."""

import math
from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np

from .core.abstract_tier import AbstractTier
from .core.level import Level
from .core.marker import centre, crosses
from .core.model import DT, HAM, SHELL
from .core.step import FRAME_SKIP, Core

TASKS = ("reach_goal", "waypoint_race", "collect_items", "unlock_goal", "survive")  # the goal one-hot's order
LEVELS_DIR = Path(__file__).parent / "levels"
TIME_PENALTY = -0.001  # per env step


def _level(x):
    """A Level, a bundled level's name, or a path to a level file."""
    if isinstance(x, Level):
        return x
    p = Path(x)
    return Level.load(p if p.suffix else LEVELS_DIR / f"{x}.yaml")


class HamsterballEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, tasks, levels=None, tier="abstract", camera_action=False, frame_skip=FRAME_SKIP, time_limit_s=None):
        if tier != "abstract":
            raise NotImplementedError(f"tier {tier!r}: v1 builds only the abstract tier")
        self.tasks = list(tasks)
        pool = [_level(x) for x in (sorted(LEVELS_DIR.glob("*.yaml")) if levels is None else levels)]
        self.levels = [lv for lv in pool if any(t in lv.tasks for t in self.tasks)]
        if not self.levels:
            raise ValueError(f"no level in the pool supports any of {self.tasks}")
        self.camera_action = camera_action
        self.frame_skip = frame_skip
        self.time_limit_s = time_limit_s
        self._cores = {}  # ponytail: unbounded, keyed by level id; spec §7's model_cache LRU when pools grow
        self.action_space = self._core(self.levels[0]).action_space
        inf = np.float32(np.inf)
        self.observation_space = gym.spaces.Dict(
            state=gym.spaces.Box(-inf, inf, (19,), np.float32),
            goal=gym.spaces.Box(-inf, inf, (10,), np.float32),
        )

    def _core(self, level):
        if level.id not in self._cores:
            self._cores[level.id] = Core(AbstractTier(), level.mjcf(), self.camera_action)
        return self._cores[level.id]

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.level = self.levels[self.np_random.integers(len(self.levels))]
        allowed = [t for t in self.tasks if t in self.level.tasks]
        self.task = allowed[self.np_random.integers(len(allowed))]
        block = self.level.tasks[self.task]
        self.core = self._core(self.level)
        self.core.reset(block["spawn"]["pos"], math.radians(block["spawn"]["yaw_deg"]))
        limit = self.time_limit_s if self.time_limit_s is not None else block["time_limit_s"]
        # the epsilon keeps a whole number of env steps (30 s at 60 Hz) from rounding up to one more
        self._max_steps = math.ceil(limit / (self.frame_skip * DT) - 1e-9)
        self._steps = 0
        self._success, self._failure = False, None
        obs = self._observe()
        self._prev = obs["state"][:3].astype(float)  # at reset the swept segment is a point
        return obs, {"core_state": self.core.state(), **obs}

    def step(self, action):
        self.core.step(np.asarray(action), n=self.frame_skip)
        self._steps += 1
        pos = self.core.data.qpos[0:3].copy()
        event = 0.0
        if crosses(self.level.tasks[self.task]["goal"], self._prev, pos):  # success wins over failure
            self._success, event = True, 1.0
        elif pos[2] < self.level.kill_plane_z:
            self._failure, event = "kill_plane", -1.0
        self._prev = pos
        terminated = self._success or self._failure is not None
        truncated = not terminated and self._steps >= self._max_steps
        obs = self._observe()
        info = {
            "core_state": self.core.state(),
            **obs,
            "success": self._success,
            "failure_reason": self._failure,
            "reward_terms": {"event": event, "time": TIME_PENALTY, "shaping": 0.0},
        }
        return obs, event + TIME_PENALTY, terminated, truncated, info

    def _observe(self):
        m, d = self.core.model, self.core.data
        # mj_step leaves derived quantities at the state before it; recompute them (derived only: the state is untouched)
        mujoco.mj_kinematics(m, d)
        mujoco.mj_comPos(m, d)
        mujoco.mj_comVel(m, d)
        mujoco.mj_subtreeVel(m, d)
        yaw = float(d.userdata[0])
        cy, sy = math.cos(yaw), math.sin(yaw)

        def cam(v):  # world -> camera yaw frame: rotate by -yaw about z
            return v[0] * cy + v[1] * sy, -v[0] * sy + v[1] * cy, v[2]

        c = d.xpos[SHELL]
        v_shell = d.qvel[0:3]
        w_shell = d.xmat[SHELL].reshape(3, 3) @ d.qvel[3:6]  # free-joint angular velocity is in the body frame
        state = np.array(
            [
                *c,
                c[2] - self.level.kill_plane_z,
                sy,
                cy,
                *cam(v_shell),
                *cam(w_shell),
                *cam(d.subtree_com[HAM] - c),
                *cam(d.subtree_linvel[HAM] - v_shell),
                float(self.core.tier.obs(m, d, yaw)["wall_contact"]),
            ],
            np.float32,
        )
        goal = np.zeros(10, np.float32)
        goal[5 + TASKS.index(self.task)] = 1
        if not self._success:
            goal[:3] = cam(centre(self.level.tasks[self.task]["goal"]) - c)
            goal[3:5] = 1  # active target; fraction remaining
        return {"state": state, "goal": goal}
