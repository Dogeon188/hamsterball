"""The Gymnasium env over the core (spec §9, §10): level and task per episode, observations, markers, reward, info."""

import math
import os
from collections import OrderedDict
from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np

from .core.abstract_tier import AbstractTier
from .core.level import LOCKED_GOAL_RGBA, MARKER_LISTS, MARKER_RGBA, TASK_MARKERS, Level
from .core.marker import centre, crosses
from .core.model import DT, HAM, SHELL
from .core.step import FRAME_SKIP, Core
from .core import views as V

TASKS = ("reach_goal", "waypoint_race", "collect_items", "unlock_goal", "survive")  # the goal one-hot's order
LEVELS_DIR = Path(__file__).parent / "levels"
TIME_PENALTY = -0.001  # per env step
PHI_SCALE = -0.01  # shaping potential per metre to the current target


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
        self,
        tasks,
        levels=None,
        tier="abstract",
        camera_action=False,
        views=(),
        view_size=(84, 84),
        obs_state=True,
        frame_skip=FRAME_SKIP,
        time_limit_s=None,
        shaped=False,
        shaping_gamma=0.99,
        model_cache=16,
    ):
        if tier != "abstract":
            raise NotImplementedError(f"tier {tier!r}: v1 builds only the abstract tier")
        self.views = tuple(views)
        if unknown := set(self.views) - set(V.VIEWS):
            raise ValueError(f"unknown views {sorted(unknown)}; choose from {V.VIEWS}")
        if not self.views and not obs_state:
            raise ValueError("obs_state=False needs at least one view")
        if self.views and os.environ.get("MUJOCO_GL") == "glfw":
            raise RuntimeError("MUJOCO_GL=glfw: env views render offscreen (egl or osmesa on Linux, unset on macOS)")
        self.view_size = tuple(view_size)
        self.obs_state = obs_state
        self.tasks = list(tasks)
        self.set_levels([LEVELS_DIR] if levels is None else levels)
        self.camera_action = camera_action
        self.frame_skip = frame_skip
        self.time_limit_s = time_limit_s
        self.shaped, self.shaping_gamma = shaped, shaping_gamma
        # LRU keyed by level id: [compiled core, its Renderer at view_size or None], evicted together (spec §7, §11)
        self._cores, self.model_cache = OrderedDict(), model_cache
        self.action_space = self._core(self.levels[0]).action_space
        inf = np.float32(np.inf)
        spaces = {}
        if obs_state:
            spaces["state"] = gym.spaces.Box(-inf, inf, (19,), np.float32)
            spaces["goal"] = gym.spaces.Box(-inf, inf, (10,), np.float32)
        for v in self.views:
            spaces[v] = gym.spaces.Box(0, 255, (*self.view_size, 3), np.uint8)
        self.observation_space = gym.spaces.Dict(spaces)

    def _core(self, level):
        """The level's compiled core, from the LRU or compiled now (evicting the least recent)."""
        if level.id in self._cores:
            self._cores.move_to_end(level.id)
        else:
            self._cores[level.id] = [Core(AbstractTier(), level.mjcf(), self.camera_action), None]
            if len(self._cores) > self.model_cache:
                _, renderer = self._cores.popitem(last=False)[1]
                if renderer is not None:
                    renderer.close()
        return self._cores[level.id][0]

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        options = options or {}
        self.level, self.task = self._draw(options.get("level"), options.get("task"))
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
        obs, vec = self._observe()
        self._prev = vec["state"][:3].astype(float)  # at reset the swept segment is a point
        self._phi = self._potential(self._prev)
        reset_options = {"level": self.level.canonical(), "level_id": self.level.id, "task": self.task}
        return obs, {"reset_options": reset_options, "core_state": self.core.state(), **vec}

    def _draw(self, level, task):
        """Spec §10's option cases: the given fields as given, the missing ones from np_random, level first."""
        pick = lambda xs: xs[self.np_random.integers(len(xs))]  # noqa: E731
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
        return level, task

    def _allowed(self, level):
        return [t for t in self.tasks if t in level.tasks]

    def set_levels(self, levels):
        """Replace the level pool from the next reset: bundled names, paths, directories, or Level objects."""
        pool = [lv for x in levels for lv in _levels(x)]
        if not any(self._allowed(lv) for lv in pool):
            raise ValueError(f"no level in the pool supports any of {self.tasks}")
        self.levels = pool

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
        obs, vec = self._observe()
        phi = 0.0 if terminated else self._potential(pos)  # Φ = 0 at terminal states
        shaping = self.shaping_gamma * phi - self._phi if self.shaped else 0.0
        self._phi = phi
        info = {
            "core_state": self.core.state(),
            **vec,
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
        # posed even without views, so core_state (which holds the mocap poses) doesn't depend on the views kwarg
        V.pose(m, d, math.radians(self.level.follow_yaw_deg))
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
        vec = {"state": state, "goal": goal}
        obs = dict(vec) if self.obs_state else {}
        if self.views:
            mujoco.mj_kinematics(m, d)  # carry the new mocap poses to their bodies,
            mujoco.mj_camlight(m, d)  # then to the cameras: mj_step left them at the previous env step's poses
            r = self._renderer()
            for v in self.views:
                r.update_scene(d, camera=v, scene_option=V.option(v))
                obs[v] = r.render(out=np.empty((*self.view_size, 3), np.uint8))
        return obs, vec

    def _renderer(self):
        """The current level's renderer, made lazily, so on the thread and in the process that steps the env."""
        entry = self._cores[self.level.id]  # the current level's entry, most recent, so never evicted here
        if entry[1] is None:
            entry[1] = mujoco.Renderer(entry[0].model, *self.view_size)
            entry[1].scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0  # env views: shadows off (spec §11)
        return entry[1]

    def close(self):
        for entry in self._cores.values():
            if entry[1] is not None:
                entry[1].close()
                entry[1] = None


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
