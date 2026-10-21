"""The game's headless logic (spec §13): input to action, real-time pacing, the pause/result flow, and the HUD text.

The window shell (window.py) feeds it key states, stick axes, and a clock, so all of it runs without a display.
"""

import math
import time

import numpy as np

from ..core.model import DT

DEADZONE = 0.15  # radial, both sticks
WALK, RUN, SPRINT = 0.25, 0.5, 1.0


def _stick(m):
    """Effort for a stick deflection m beyond the deadzone, rescaled to run 0 to 1."""
    return min((m - DEADZONE) / (1 - DEADZONE), 1.0)


def action(keys, left=(0.0, 0.0), right_x=0.0):
    """The 3-D action [run_x, run_y, cam_yaw] (camera frame: x right, y forward) from held keys and stick axes.

    keys: names held, from {"w", "a", "s", "d", "shift", "ctrl", "q", "e"}. Sticks are GLFW axes (y down).
    A stick outside its deadzone overrides the keys it stands for.
    """
    m = math.hypot(*left)
    if m > DEADZONE:
        run = np.array([left[0], -left[1]]) / m * _stick(m)
    else:
        d = np.array([("d" in keys) - ("a" in keys), ("w" in keys) - ("s" in keys)], float)
        n = np.linalg.norm(d)
        effort = WALK if "shift" in keys else SPRINT if "ctrl" in keys else RUN
        run = d / n * effort if n else d
    if abs(right_x) > DEADZONE:
        cam = -math.copysign(_stick(abs(right_x)), right_x)
    else:
        cam = ("q" in keys) - ("e" in keys)
    return np.array([*run, cam], np.float32)


class Pacer:
    """One env step per frame at real-time pace: tick() sleeps until the frame's deadline.

    A frame that ends more than `slack` past its deadline is an overrun: it's counted, and the schedule restarts from
    now, so the game slows down rather than skipping or bursting steps (spec §13).
    """

    def __init__(self, period, clock=time.perf_counter, sleep=time.sleep, slack=0.002):
        self.period, self.clock, self.sleep, self.slack = period, clock, sleep, slack
        self.overruns = 0
        self.reset()

    def reset(self):
        """Start a fresh schedule at the next tick (after a pause, or a new episode)."""
        self._deadline = None

    def tick(self):
        now = self.clock()
        if self._deadline is None:
            self._deadline = now
        elif now > self._deadline + self.slack:
            self.overruns += 1
            self._deadline = now
        elif now < self._deadline:
            self.sleep(self._deadline - now)
        self._deadline += self.period


class Flow:
    """The play screen's states: "play" (physics runs), "paused" (Esc), "result" (the episode ended).

    press(name) takes a key ("esc", "enter", "r"), a pause button ("resume", "retry", "menu", "quit"), or "close" (the
    window), and returns the command for the shell to run, if any: "retry" (same level and seed), "next" (same
    settings), "menu", or "quit". Leaving with retry, menu, quit, or close before the result is an abandoned run.
    """

    # state -> {input: (next state, command)}; "close" works everywhere
    TABLE = {
        "play": {"esc": ("paused", None)},
        "paused": {
            "esc": ("play", None),
            "resume": ("play", None),
            "retry": ("play", "retry"),
            "menu": ("menu", "menu"),
            "quit": ("quit", "quit"),
        },
        "result": {"enter": ("play", "next"), "r": ("play", "retry"), "esc": ("menu", "menu")},
    }

    def __init__(self):
        self.state, self.outcome = "play", None

    def end(self, outcome):
        """The episode ended: show the result card with this outcome."""
        self.state, self.outcome = "result", outcome

    def press(self, name):
        if name == "close":
            self.state = "quit"
            return "quit"
        if name not in self.TABLE.get(self.state, {}):
            return None
        self.state, command = self.TABLE[self.state][name]
        if self.state == "play":
            self.outcome = None
        return command


FAILURES = {"hazard": "hazard", "kill_plane": "fell off", "left_region": "left the region"}


def outcome(terminated, truncated, info):
    """The result card's headline for a step that ended the episode, or None while it runs."""
    if terminated:
        return "Success" if info["success"] else f"Failed: {FAILURES[info['failure_reason']]}"
    return "Time up" if truncated else None


def sim_time(env):
    """(elapsed, limit) in sim seconds."""
    e = env.unwrapped
    dt = e.frame_skip * DT
    return e._steps * dt, e._max_steps * dt


def progress(env):
    """Task progress: waypoint 2/5, items 3/7, keys 1/2, or survive time left."""
    e = env.unwrapped  # ponytail: reads the env's private target list; a public progress accessor if a second reader needs it
    block = e.level.tasks[e.task]
    left = [k for k, _ in e._todo]
    if e.task == "survive":
        t, limit = sim_time(env)
        return f"survive: {limit - t:.1f} s left"
    if e.task == "waypoint_race":
        n = len(block["waypoints"])
        return f"waypoint {n - len(left) + 1}/{n}"
    if e.task == "collect_items":
        n = len(block["items"])
        return f"items {n - len(left)}/{n}"
    if e.task == "unlock_goal":
        n = len(block["keys"])
        got = n - left.count("keys")
        return f"keys {got}/{n}" + (": reach the goal" if got == n else "")
    return "reach the goal"


def hud(env, act, recording):
    """The HUD's lines: sim time against the limit, task progress, recording state, effort, and shell speed."""
    t, limit = sim_time(env)
    return [
        f"{t:.1f} / {limit:.1f} s",
        progress(env),
        "REC" if recording else "not recording",
        f"effort {min(math.hypot(act[0], act[1]), 1.0):.2f}",
        f"speed {np.linalg.norm(env.unwrapped.core.data.qvel[0:3]):.1f} m/s",
    ]
