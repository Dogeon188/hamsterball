"""The game's headless logic (spec §13): input to action, real-time pacing, the pause/result flow, and the HUD text.

The window shell (window.py) feeds it key states, stick axes, and a clock, so all of it runs without a display.
"""

import math
import time

import numpy as np

from ..core.model import DT

DEADZONE = 0.15  # radial, both sticks
WALK, RUN, SPRINT = 0.25, 0.5, 1.0
MOUSE_SENS = 0.05  # turn per pixel: 0.1 deg/px, since turn 1 turns 2 deg in one env step


def _stick(m):
    """Effort for a stick deflection m beyond the deadzone, rescaled to run 0 to 1."""
    return min((m - DEADZONE) / (1 - DEADZONE), 1.0)


class Mouse:
    """Mouse motion to (turn, cam_pitch): right turns right and up looks up, each at most ±1 per frame; the rest
    carries to later frames, each axis on its own."""

    def __init__(self, sens=MOUSE_SENS):
        self.sens = sens
        self.clear()

    def clear(self):
        """Drop carried motion (the cursor was released, or an episode starts)."""
        self.carry = np.zeros(2)

    def move(self, dx, dy):
        """This frame's (turn, cam_pitch) from (dx, dy) pixels of motion plus what's carried. Screen y is down and
        the pitch is below the horizon, so dy adds to it as is."""
        self.carry += np.array([-dx, dy]) * self.sens
        cam = np.clip(self.carry, -1.0, 1.0)
        self.carry -= cam
        return tuple(cam.tolist())


def _axis(v):
    """A right stick axis rescaled with its sign, or 0 inside the deadzone."""
    return math.copysign(_stick(abs(v)), v) if abs(v) > DEADZONE else 0.0


def action(keys, left=(0.0, 0.0), right=(0.0, 0.0), mouse=(0.0, 0.0)):
    """The 4-D action [run_x, run_y, turn, cam_pitch] (camera frame: x right, y forward) from held keys, stick
    axes, and this frame's Mouse.move value.

    keys: names held, from {"w", "a", "s", "d", "shift", "ctrl", "q", "e"}. Sticks are GLFW axes (y down).
    A stick outside its deadzone overrides the keys it stands for; non-zero mouse motion on an axis overrides that
    axis of the right stick (and Q/E for the turn). No keys drive the pitch.
    """
    m = math.hypot(*left)
    if m > DEADZONE:
        run = np.array([left[0], -left[1]]) / m * _stick(m)
    else:
        d = np.array([("d" in keys) - ("a" in keys), ("w" in keys) - ("s" in keys)], float)
        n = np.linalg.norm(d)
        effort = WALK if "shift" in keys else SPRINT if "ctrl" in keys else RUN
        run = d / n * effort if n else d
    cam = mouse[0] or -_axis(right[0]) or ("q" in keys) - ("e" in keys)  # right stick right turns right
    pitch = mouse[1] or _axis(right[1])  # GLFW y down: pushing up lowers the pitch, looking up
    return np.array([*run, cam, pitch], np.float32)


class Pacer:
    """One env step per frame at real-time pace: wait() sleeps until the frame's deadline.

    A frame that ends more than `slack` past its deadline is an overrun: it's counted, and the schedule restarts from
    now, so the game slows down rather than skipping or bursting steps (spec §13).
    """

    def __init__(self, period, clock=time.perf_counter, sleep=time.sleep, slack=0.002):
        self.period, self.clock, self.sleep, self.slack = period, clock, sleep, slack
        self.overruns = 0
        self.reset()

    def reset(self):
        """Start a fresh schedule at the next wait (after a pause, or a new episode)."""
        self._deadline = None

    def wait(self):
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
        return f"waypoint {min(n - len(left) + 1, n)}/{n}"  # n/n after a win
    if e.task == "collect_items":
        n = len(block["items"])
        return f"items {n - len(left)}/{n}"
    if e.task == "unlock_goal":
        n = len(block["keys"])
        got = n - left.count("keys")
        return f"keys {got}/{n}" + (": reach the goal" if got == n else "")
    return "reach the goal"


def hud(env, act, recording):
    """The HUD's lines: sim time (against the limit for survive only), task progress, recording state, effort, and
    shell speed. A human's non-survive episode has no time limit (the session's env doesn't truncate)."""
    t, limit = sim_time(env)
    return [
        f"{t:.1f} / {limit:.1f} s" if env.unwrapped.task == "survive" else f"{t:.1f} s",
        progress(env),
        "REC" if recording else "not recording",
        f"effort {min(math.hypot(act[0], act[1]), 1.0):.2f}",
        f"speed {np.linalg.norm(env.unwrapped.core.data.qvel[0:3]):.1f} m/s",
    ]
