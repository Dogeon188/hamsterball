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
FRONT_B = FRONT > 0


class ScriptedTrot:
    """A trot of triangle-wave hip strokes and knee lifts, its feet down `duty` of each stride so three or four are
    usually on the wall; front and rear legs have their own stroke, lift and offsets, and each leg its own phase.
    Effort (the run vector's length) eases in and sets the stride rate and stroke. The heading error to the run vector
    steers by a stronger stroke on the outside legs, shrinking the stroke when it is large. A PD loop holds the joints
    on the targets, which carry pitch and roll feedback from `proprio`. Stateful in its stride phase and effort."""

    name = "scripted_trot_v1"
    P = dict(
        duty=0.6041490368718223,
        amp_f=0.6206269594797135,
        amp_r=0.4091433124218371,
        lift_f=0.8358649436867193,
        lift_r=1.0718596360678831,
        hz=7.725850181182517,
        hz_min=0.5422755472665778,
        bias_f=0.4575531295546219,
        bias_r=-0.17428732399906333,
        kbias_f=-0.042431900946042655,
        kbias_r=-0.3112523015212403,
        ph_fr=0.48307034890432,
        ph_rl=0.7744000356035667,
        ph_rr=0.10323179187126902,
        pitch_knee=0.20018121491531254,
        pitch_hip=-0.5898462847223856,
        roll_knee=-0.38061220604329826,
        roll_hip=0.21538792082887834,
        kp=15.0,
        kd=0.1,
        tg=0.3295182354768352,
        steer=0.5833501512243375,
        turn_amp=0.39386392340482407,
        ramp=2.0,
    )

    def __init__(self, **overrides):
        self.p = {**self.P, **overrides}
        self.reset()

    def reset(self):
        self.phase = self.effort = 0.0

    def act(self, obs, run_vector):
        p, pr = self.p, np.asarray(obs["proprio"] if isinstance(obs, dict) else obs, float)
        q, qd, fwd, up = pr[:8], pr[8:16], pr[20:23], pr[23:26]
        run = np.asarray(run_vector, float)
        step = p["ramp"] * DT
        self.effort += max(-step, min(step, min(float(np.hypot(*run)), 1.0) - self.effort))
        target = np.array(STAND, float)
        if self.effort > 1e-3:
            err = math.atan2(fwd[0] * run[1] - fwd[1] * run[0], fwd[1] * run[1] + fwd[0] * run[0])  # + : run is to the left
            turn = max(-1.0, min(1.0, -err / p["steer"]))
            self.phase = (self.phase + (p["hz_min"] + (p["hz"] - p["hz_min"]) * self.effort) * DT) % 1.0
            ph = (self.phase + np.array([0.0, p["ph_fr"], p["ph_rl"], p["ph_rr"]])) % 1.0
            stance = ph < p["duty"]
            s = np.where(stance, ph / p["duty"], (ph - p["duty"]) / (1 - p["duty"]))
            stroke = np.where(stance, 2 * s - 1, 1 - 2 * s)  # the foot sweeps back while it stands, forward in the swing
            amp = np.where(FRONT_B, p["amp_f"], p["amp_r"]) * self.effort * (1 + p["tg"] * turn * SIDE) * (
                p["turn_amp"] + (1 - p["turn_amp"]) * max(0.0, math.cos(err)))
            target[0::2] += np.where(FRONT_B, p["bias_f"], p["bias_r"]) * self.effort + amp * stroke
            lift = np.where(FRONT_B, p["lift_f"], p["lift_r"]) * self.effort
            target[1::2] += np.where(FRONT_B, p["kbias_f"], p["kbias_r"]) * self.effort - lift * np.where(stance, 0.0, np.sin(math.pi * s))
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
