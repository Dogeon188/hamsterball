"""Low-level policies for the articulated tier: `act(obs, run_vector) -> torques`, chosen by name from a registry.

See lowlevel.md.
"""

import math

import numpy as np

from .core.articulated_tier import GEAR_HIP, GEAR_KNEE, STAND

DT = 1 / 60  # one env step
GEAR = np.array([GEAR_HIP, GEAR_KNEE] * 4)
SIDE = np.array([1.0, -1.0, 1.0, -1.0])  # per leg fl fr rl rr: +1 on the torso's left
FRONT = np.array([1.0, 1.0, -1.0, -1.0])
DIAG = np.array([0.0, 0.5, 0.5, 0.0])  # trot: fl and rr together, half a stride from fr and rl


class ScriptedTrot:
    """A trot of triangle-wave hip strokes and knee lifts, its feet down `duty` of each stride so three or four are
    usually on the wall. Effort (the run vector's length) eases in and sets the stride rate and stroke. The heading
    error to the run vector steers by a stronger stroke on the outside legs, shrinking the stroke when it is large.
    A PD loop holds the joints on the targets, which carry pitch and roll feedback from `proprio`. Stateful in its
    stride phase, effort and the torso's yaw rate."""

    name = "scripted_trot_v1"
    P = dict(
        duty=0.7529031903053366,
        amp=0.4885060429308078,
        lift=0.8,
        hz=1.9450376745844056,
        hz_min=0.9725188372922028,
        bias=0.10382090594486951,
        pitch_knee=0.10106161059460633,
        pitch_hip=-0.12955948402671752,
        roll_knee=-0.861143096165033,
        roll_hip=-0.10745318540671121,
        kp=40.0,
        kd=0.5,
        tg=0.1615051752417482,
        steer=1.1596252894489498,
        yaw_damp=0.0,
        turn_amp=0.7412017038622045,
        ramp=2.0,
    )

    def __init__(self, **overrides):
        self.p = {**self.P, **overrides}
        self.reset()

    def reset(self):
        self.phase = self.effort = self.rate = 0.0
        self.theta = None

    def act(self, obs, run_vector):
        p, pr = self.p, np.asarray(obs["proprio"] if isinstance(obs, dict) else obs, float)
        q, qd, fwd, up = pr[:8], pr[8:16], pr[20:23], pr[23:26]
        run = np.asarray(run_vector, float)
        step = p["ramp"] * DT
        self.effort += max(-step, min(step, min(float(np.hypot(*run)), 1.0) - self.effort))
        theta = math.atan2(fwd[1], fwd[0])
        if self.theta is not None:  # yaw rate by difference, lightly filtered
            self.rate += 0.5 * (((theta - self.theta + math.pi) % (2 * math.pi) - math.pi) / DT - self.rate)
        self.theta = theta
        target = np.array(STAND, float)
        if self.effort > 1e-3:
            err = math.atan2(fwd[0] * run[1] - fwd[1] * run[0], fwd[1] * run[1] + fwd[0] * run[0])  # + : run is to the left
            turn = max(-1.0, min(1.0, -err / p["steer"] + p["yaw_damp"] * self.rate))
            self.phase = (self.phase + (p["hz_min"] + (p["hz"] - p["hz_min"]) * self.effort) * DT) % 1.0
            ph = (self.phase + DIAG) % 1.0
            stance = ph < p["duty"]
            s = np.where(stance, ph / p["duty"], (ph - p["duty"]) / (1 - p["duty"]))
            stroke = np.where(stance, 2 * s - 1, 1 - 2 * s)  # the foot sweeps back while it stands, forward in the swing
            amp = p["amp"] * self.effort * (1 + p["tg"] * turn * SIDE) * (p["turn_amp"] + (1 - p["turn_amp"]) * max(0.0, math.cos(err)))
            target[0::2] += p["bias"] * self.effort + amp * stroke
            target[1::2] -= p["lift"] * self.effort * np.where(stance, 0.0, np.sin(math.pi * s))
        roll = -up[0] * fwd[1] + up[1] * fwd[0]  # the up axis's lean toward the torso's left
        target[1::2] -= FRONT * p["pitch_knee"] * fwd[2] + SIDE * p["roll_knee"] * roll
        target[0::2] -= FRONT * p["pitch_hip"] * fwd[2] + SIDE * p["roll_hip"] * roll
        return np.clip((p["kp"] * (target - q) - p["kd"] * qd) / GEAR, -1, 1).astype(np.float32)


REGISTRY = {ScriptedTrot.name: ScriptedTrot}
DEFAULT = ScriptedTrot.name


def get(name, **kw):
    try:
        return REGISTRY[name](**kw)
    except KeyError:
        raise KeyError(f"unknown low-level policy {name!r}; known: {sorted(REGISTRY)}") from None
