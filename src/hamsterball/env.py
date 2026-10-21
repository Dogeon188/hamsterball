"""The Gymnasium env over the core (spec §9, §10): level and task per episode, observations, markers, reward, info."""

import math
from collections import OrderedDict
from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np

from .core.abstract_tier import AbstractTier
from .core.generator import LevelGenerator
from .core.level import LOCKED_GOAL_RGBA, MARKER_LISTS, MARKER_RGBA, TASK_MARKERS, Level
from .core.marker import centre, crosses
from .core.model import DT, HAM, SHELL
from .core.step import FRAME_SKIP, Core

TASKS = ("reach_goal", "waypoint_race", "collect_items", "unlock_goal", "survive")  # the goal one-hot's order
LEVELS_DIR = Path(__file__).parent / "levels"
TIME_PENALTY = -0.001  # per env step
PHI_SCALE = -0.01  # shaping potential per metre to the current target
DIFFICULTY = 0.5  # the default difficulty passed to a generator


def _levels(x):
    """The levels one pool entry names: a Level, a bundled level's name, a level file, or a directory of them."""
    if isinstance(x, Level):
        return [x]
    p = Path(x)
    if p.is_dir():
        return [Level.load(f) for f in sorted(p.glob("*.yaml"))]
    return [Level.load(p if p.suffix else LEVELS_DIR / f"{x}.yaml")]


class HamsterballEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self, tasks, levels=None, tier="abstract", camera_action=False, frame_skip=FRAME_SKIP, time_limit_s=None,
        shaped=False, shaping_gamma=0.99, model_cache=16, difficulty=DIFFICULTY,
    ):
        if tier != "abstract":
            raise NotImplementedError(f"tier {tier!r}: v1 builds only the abstract tier")
        self.tasks = list(tasks)
        self.set_levels([LEVELS_DIR] if levels is None else levels, difficulty)
        self.camera_action = camera_action
        self.frame_skip = frame_skip
        self.time_limit_s = time_limit_s
        self.shaped, self.shaping_gamma = shaped, shaping_gamma
        self._cores, self.model_cache = OrderedDict(), model_cache  # LRU of compiled cores, keyed by level id
        # a generator pool has no level yet: the action space is the tier's, whatever the level
        core = self._core(self.levels[0]) if self.levels else Core(AbstractTier(), camera_action=camera_action)
        self.action_space = core.action_space
        inf = np.float32(np.inf)
        self.observation_space = gym.spaces.Dict(
            state=gym.spaces.Box(-inf, inf, (19,), np.float32),
            goal=gym.spaces.Box(-inf, inf, (10,), np.float32),
        )

    def _core(self, level):
        """The level's compiled core, from the LRU or compiled now (evicting the least recent)."""
        if level.id in self._cores:
            self._cores.move_to_end(level.id)
        else:
            self._cores[level.id] = Core(AbstractTier(), level.mjcf(), self.camera_action)
            if len(self._cores) > self.model_cache:
                self._cores.popitem(last=False)
        return self._cores[level.id]

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        options = options or {}
        self.level, self.task, difficulty = self._draw(*(options.get(k) for k in ("level", "task", "difficulty")))
        block = self.level.tasks[self.task]
        self.core = self._core(self.level)
        self.core.reset(block["spawn"]["pos"], math.radians(block["spawn"]["yaw_deg"]))
        limit = self.time_limit_s if self.time_limit_s is not None else block["time_limit_s"]
        # the epsilon keeps a whole number of env steps (30 s at 60 Hz) from rounding up to one more
        self._max_steps = math.ceil(limit / (self.frame_skip * DT) - 1e-9)
        self._steps = 0
        self._success, self._failure = False, None
        # targets left, in task order, each (marker key, index or None): keys before the goal, waypoint i before i + 1
        self._todo = [
            (k, i) for k in TASK_MARKERS[self.task] for i in (range(len(block[k])) if k in MARKER_LISTS else [None])
        ]
        self._n_targets = len(self._todo)
        # marker looks: the episode task's markers in their kind's colour, the unlock goal locked, other tasks' hidden
        m = self.core.model
        for g in range(m.ngeom):
            task, _, rest = m.geom(g).name.partition("/")
            if rest:
                k = rest.split("/")[0]
                m.geom_rgba[g] = MARKER_RGBA[k] if task == self.task else (*MARKER_RGBA[k][:3], 0)
        if self.task == "unlock_goal":
            self._look(("goal", None), LOCKED_GOAL_RGBA)
        obs = self._observe()
        self._prev = obs["state"][:3].astype(float)  # at reset the swept segment is a point
        self._phi = self._potential(self._prev)
        reset_options = {"level": self.level.canonical(), "level_id": self.level.id, "task": self.task}
        if self.level.generator is not None:  # a generated level's difficulty is in its provenance
            difficulty = self.level.generator["difficulty"]
        if difficulty is not None:
            reset_options["difficulty"] = difficulty
        return obs, {"reset_options": reset_options, "core_state": self.core.state(), **obs}

    def _draw(self, level, task, difficulty):
        """Spec §10's option cases: the given fields as given, the missing ones from np_random, level first (a pool
        index, or a 63-bit generator seed). Returns (level, task, the difficulty the generator was called with)."""
        pick = lambda xs: xs[self.np_random.integers(len(xs))]  # noqa: E731
        if level is None and self.generator is not None:
            seed = int(self.np_random.integers(2**63))
            task = task or pick(self._allowed(self.generator))
            if task not in self.generator.tasks:
                raise ValueError(f"the generator doesn't support task {task!r}")
            difficulty = self.difficulty if difficulty is None else difficulty
            return self.generator(task, seed, difficulty), task, difficulty
        if level is None and difficulty not in (None, DIFFICULTY):
            raise ValueError("a difficulty other than the default needs a generator pool")
        if level is None:
            pool = [lv for lv in self.levels if (task in lv.tasks if task else self._allowed(lv))]
            if not pool:
                raise ValueError(f"no pool level supports {task or self.tasks}")
            level = pick(pool)
        elif not isinstance(level, Level):
            level = Level.from_yaml(level)  # a canonical YAML string
        if task is None:
            task = pick(self._allowed(level) or list(level.tasks))
        if task not in level.tasks:
            raise ValueError(f"level {level.id[:12]} doesn't support task {task!r}")
        return level, task, None

    def _allowed(self, level):
        """The env's tasks that a level (or a generator) supports."""
        return [t for t in self.tasks if t in level.tasks]

    def set_levels(self, levels, difficulty=None):
        """Replace the level pool from the next reset: a LevelGenerator, or a list of bundled names, paths,
        directories, or Level objects. `difficulty` goes to a generator (None keeps the current one); a file pool
        takes only the default."""
        if isinstance(levels, LevelGenerator):
            if not self._allowed(levels):
                raise ValueError(f"the generator supports none of {self.tasks}")
            self.generator, self.levels = levels, []
        else:
            if difficulty not in (None, DIFFICULTY):
                raise ValueError("a difficulty other than the default needs a generator pool")
            pool = [lv for x in levels for lv in _levels(x)]
            if not any(self._allowed(lv) for lv in pool):
                raise ValueError(f"no level in the pool supports any of {self.tasks}")
            self.generator, self.levels, difficulty = None, pool, DIFFICULTY
        if difficulty is not None:
            self.difficulty = difficulty

    def step(self, action):
        self.core.step(np.asarray(action), n=self.frame_skip)
        self._steps += 1
        pos = self.core.data.qpos[0:3].copy()
        event = 0.0
        survive = self.task == "survive"
        if survive:  # a point test on the current centre: the region means "stay inside"
            inside = crosses(self._marker(("region", None)), pos, pos)
            if inside and self._steps >= self._max_steps:
                self._todo.clear()
        # rerun until nothing new triggers, so one step can complete several targets in task order
        while not survive and (hit := [t for t in self._live() if crosses(self._marker(t), self._prev, pos)]):
            for t in hit:
                self._todo.remove(t)
                event += t[0] != "goal"  # +1 per item, key, or waypoint
                if t[0] in ("items", "keys"):
                    self._look(t, (*MARKER_RGBA[t[0]][:3], 0))
            if self.task == "unlock_goal" and self._todo == [("goal", None)]:
                self._look(("goal", None), MARKER_RGBA["goal"])
        if not self._todo:  # success wins over failure
            self._success, event = True, event + 1.0
        elif any(crosses(h, self._prev, pos) for h in self.level.hazards):
            self._failure, event = "hazard", event - 1.0
        elif pos[2] < self.level.kill_plane_z:
            self._failure, event = "kill_plane", event - 1.0
        elif survive and not inside:
            self._failure, event = "left_region", event - 1.0
        time = 0.0 if survive else TIME_PENALTY  # survive has no time penalty
        self._prev = pos
        terminated = self._success or self._failure is not None
        truncated = not terminated and self._steps >= self._max_steps
        obs = self._observe()
        phi = 0.0 if terminated else self._potential(pos)  # Φ = 0 at terminal states
        shaping = self.shaping_gamma * phi - self._phi if self.shaped else 0.0
        self._phi = phi
        info = {
            "core_state": self.core.state(),
            **obs,
            "success": self._success,
            "failure_reason": self._failure,
            "reward_terms": {"event": event, "time": time, "shaping": shaping},
        }
        return obs, event + time + shaping, terminated, truncated, info

    def _marker(self, target):
        k, i = target
        m = self.level.tasks[self.task][k]
        return m if i is None else m[i]

    def _look(self, target, rgba):
        k, i = target
        m = self.core.model
        m.geom_rgba[m.geom(f"{self.task}/{k}" + ("" if i is None else f"/{i}")).id] = rgba

    def _live(self):
        """The targets that can trigger now: only the next waypoint; the unlock goal only once every key is in."""
        if self.task == "waypoint_race":
            return self._todo[:1]
        if self.task == "unlock_goal" and len(self._todo) > 1:
            return self._todo[:-1]
        return self._todo

    def _target(self, pos):
        """The current target's centre: the nearest live target (spec §9)."""
        return min((centre(self._marker(t)) for t in self._live()), key=lambda c: np.linalg.norm(c - pos))

    def _potential(self, pos):
        return PHI_SCALE * float(np.linalg.norm(self._target(pos) - pos))

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
            goal[:3] = cam(self._target(c) - c)
            goal[3] = 1  # active target
            # fraction remaining; survive's is time left over the time limit, in env steps
            goal[4] = 1 - self._steps / self._max_steps if self.task == "survive" else len(self._todo) / self._n_targets
        return {"state": state, "goal": goal}


class DiscreteAction(gym.ActionWrapper):
    """Spec §5: action 0 is a no-op; 1 + 8·e + k runs toward k·45° clockwise from camera forward at effort EFFORTS[e].

    Under the camera flag, cam_yaw stays 0.
    """

    EFFORTS = (0.25, 0.5, 1.0)

    def __init__(self, env):
        super().__init__(env)
        self._table = np.zeros((25, *env.action_space.shape), np.float32)
        for j in range(24):
            e, th = self.EFFORTS[j // 8], math.radians(45 * (j % 8))
            self._table[1 + j, :2] = e * math.sin(th), e * math.cos(th)
        self.action_space = gym.spaces.Discrete(25)

    def action(self, action):
        return self._table[action]
