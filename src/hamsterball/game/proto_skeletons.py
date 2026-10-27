"""PROTOTYPE (throwaway): parametric articulated skeleton + parametric gait + evaluation, for the skeleton search.

Question: is there a skeleton plus gait that climbs 20 degrees and runs about 2 m/s on flat ground, steering, no tumble?
Nothing here is production. See the prototype branch's commit message for the verdict.
"""

import json
import math
import sys

import gymnasium as gym
import mujoco
import numpy as np

from ..core import Core
from ..core import articulated_tier as AT
from ..core.model import HAM, HAM_DOF, HAM_QPOS, M_HAM, R, SOLREF, T

DT = 1 / 60
INNER = R - T
LEGS = AT.LEGS
DEG = math.degrees

# ---------------------------------------------------------------- skeleton
BASE = dict(
    torso_r=0.05, torso_half=0.08, head_r=0.045, head_x=0.17, head_z=0.025,
    hip_xf=0.08, hip_xr=0.08, hip_y=0.085, hip_z=-0.02,
    thigh_f=0.09, shin_f=0.09, thigh_r=0.09, shin_r=0.09, foot_r=0.02,
    m_head=0.30, m_thigh=0.10, m_shin=0.06, m_foot=0.02,
    gear_hip=16.0, gear_knee=12.8, damping=0.5, armature=0.01,
    hip_lo=-1.5, hip_hi=1.5, knee_lo=-1.8, knee_hi=0.05,
    sf_h=0.28, sf_k=-0.87, sr_h=0.28, sr_k=-0.87, mu=1.0,
)


def _leg(s, name, sx, sy):
    front = sx > 0
    th, sh = (s["thigh_f"], s["shin_f"]) if front else (s["thigh_r"], s["shin_r"])
    hx = s["hip_xf"] if front else s["hip_xr"]
    fr = AT._FROZEN
    return (
        f'<body name="{name}_thigh" pos="{sx * hx} {sy * s["hip_y"]} {s["hip_z"]}">'
        f'<joint name="{name}_hip" axis="0 1 0" range="{DEG(s["hip_lo"])} {DEG(s["hip_hi"])}" damping="{s["damping"]}" armature="{s["armature"]}"/>'
        f'<geom type="capsule" fromto="0 0 0 0 0 {-th}" size="0.015" mass="{s["m_thigh"]}" {fr} {AT._LEG}/>'
        f'<body name="{name}_shin" pos="0 0 {-th}">'
        f'<joint name="{name}_knee" axis="0 1 0" range="{DEG(s["knee_lo"])} {DEG(s["knee_hi"])}" damping="{s["damping"]}" armature="{s["armature"]}"/>'
        f'<geom type="capsule" fromto="0 0 0 0 0 {-sh}" size="0.012" mass="{s["m_shin"]}" {fr} {AT._LEG}/>'
        f'<geom name="{name}_foot" type="sphere" pos="0 0 {-sh}" size="{s["foot_r"]}" mass="{s["m_foot"]}" {AT._GEOM} '
        f'margin="0.002" condim="3" friction="{s["mu"]} .005 .0001" rgba=".25 .15 .1 1"/>'
        f'</body></body>'
    )


def stand_vec(s):
    return np.array([s["sf_h"], s["sf_k"], s["sf_h"], s["sf_k"], s["sr_h"], s["sr_k"], s["sr_h"], s["sr_k"]])


def stand_height(s):
    z = -1e9
    st = stand_vec(s)
    for i, (_, sx, sy) in enumerate(LEGS):
        front = sx > 0
        th, sh = (s["thigh_f"], s["shin_f"]) if front else (s["thigh_r"], s["shin_r"])
        hx = s["hip_xf"] if front else s["hip_xr"]
        h, k = st[2 * i], st[2 * i + 1]
        fx = sx * hx + th * math.sin(h) + sh * math.sin(h + k)
        fz = s["hip_z"] - th * math.cos(h) - sh * math.cos(h + k)
        z = max(z, -math.sqrt(max((INNER - s["foot_r"]) ** 2 - fx * fx - s["hip_y"] ** 2, 0)) - fz)
    return z


def make_tier(s0=None, **over):
    """A tier instance for skeleton parameters s (BASE overridden by s0, then over)."""
    s = {**BASE, **(s0 or {}), **over}
    m_torso = M_HAM - s["m_head"] - 4 * (s["m_thigh"] + s["m_shin"] + s["m_foot"])
    assert m_torso > 0.3, m_torso
    h = stand_height(s)
    st = stand_vec(s)
    head = (
        f'<body name="head" pos="{s["head_x"]} 0 {s["head_z"]}">'
        f'<geom type="sphere" size="{s["head_r"]}" mass="{s["m_head"]}" {AT._FROZEN} {AT._HEAD}/>'
        f'<geom type="sphere" pos="{s["head_r"] * 0.95} 0 -0.005" size="0.012" {AT._VISUAL} rgba=".15 .08 .08 1"/>'
        f'<geom type="sphere" pos="{s["head_r"] * 0.7} 0.025 0.025" size="0.01" {AT._VISUAL} rgba="0 0 0 1"/>'
        f'<geom type="sphere" pos="{s["head_r"] * 0.7} -0.025 0.025" size="0.01" {AT._VISUAL} rgba="0 0 0 1"/>'
        f'<site name="hamster_eye" pos="0.03 0 0.01"/></body>'
    )
    frag = (
        f'<body name="hamster" pos="0 0 {R + h!r}"><freejoint/>'
        f'<geom type="capsule" fromto="{-s["torso_half"]} 0 0 {s["torso_half"]} 0 0" size="{s["torso_r"]}" mass="{m_torso}" '
        f'{AT._FROZEN} {AT._RGBA}/>' + head + "".join(_leg(s, *leg) for leg in LEGS) + "</body>"
    )
    gears = (s["gear_hip"], s["gear_knee"])
    motors = "".join(
        f'<motor name="{n}_{j}" joint="{n}_{j}" gear="{g}" ctrlrange="-1 1"/>'
        for n, _, _ in LEGS for j, g in (("hip", gears[0]), ("knee", gears[1]))
    )
    key = " ".join(repr(float(v)) for v in (0, 0, R, 1, 0, 0, 0, 0, 0, R + h, 1, 0, 0, 0, *st))
    sections = f'<actuator>{motors}</actuator><keyframe><key name="stand" qpos="{key}"/></keyframe>'

    class Tier(AT.ArticulatedTier):
        pass

    Tier.fragment, Tier.sections, Tier.resting_radius = frag, sections, 0.42
    t = Tier()
    t.skel = s
    t.stand = st
    t.gear = np.array([gears[0], gears[1]] * 4)
    return t


def measure_rest(tier, drops=4, seconds=1.5, seed=0):
    """Max subtree-COM distance from the shell centre, limp, dropped in random orientations (rest and swing)."""
    keep, tier.resting_radius = tier.resting_radius, 10.0  # no containment: measure the unclipped radius
    core = Core(tier)
    m, d = core.model, core.data
    rng = np.random.default_rng(seed)
    best = 0.0
    for _ in range(drops):
        core.reset()
        q = rng.normal(size=4)
        d.qpos[HAM_QPOS + 3:HAM_QPOS + 7] = q / np.linalg.norm(q)
        d.qpos[HAM_QPOS + 2] = R + 0.05
        mujoco.mj_forward(m, d)
        for _ in range(round(seconds / DT)):
            core.step(np.zeros(8, np.float32))
            best = max(best, float(np.linalg.norm(d.subtree_com[HAM] - d.xpos[1])))
    tier.resting_radius = keep
    return best


# ---------------------------------------------------------------- gait
SIDE = np.array([1.0, -1.0, 1.0, -1.0])
FRONT = np.array([1.0, 1.0, -1.0, -1.0])

GAIT = dict(
    duty=0.753, amp_f=0.4885, amp_r=0.4885, lift_f=0.8, lift_r=0.8, hz=1.945, hz_min=0.9725,
    bias_f=0.1038, bias_r=0.1038, kbias_f=0.0, kbias_r=0.0,
    ph_fr=0.5, ph_rl=0.5, ph_rr=0.0,
    pitch_knee=0.101, pitch_hip=-0.1296, roll_knee=-0.861, roll_hip=-0.107,
    kp=40.0, kd=0.5, tg=0.1615, steer=1.1596, turn_amp=0.7412, ramp=2.0,
)


class Gait:
    """The ticket-04 trot, generalised: front and rear amplitudes, lifts and offsets, free leg phases."""

    def __init__(self, tier, **g):
        self.p = {**GAIT, **g}
        self.stand, self.gear = tier.stand, tier.gear
        self.reset()

    def reset(self):
        self.phase = self.effort = 0.0

    def act(self, proprio, run):
        p = self.p
        pr = np.asarray(proprio, float)
        q, qd, fwd, up = pr[:8], pr[8:16], pr[20:23], pr[23:26]
        run = np.asarray(run, float)
        step = p["ramp"] * DT
        self.effort += max(-step, min(step, min(float(np.hypot(*run)), 1.0) - self.effort))
        target = self.stand.copy()
        if self.effort > 1e-3:
            err = math.atan2(fwd[0] * run[1] - fwd[1] * run[0], fwd[1] * run[1] + fwd[0] * run[0])
            turn = max(-1.0, min(1.0, -err / p["steer"]))
            self.phase = (self.phase + (p["hz_min"] + (p["hz"] - p["hz_min"]) * self.effort) * DT) % 1.0
            ph = (self.phase + np.array([0.0, p["ph_fr"], p["ph_rl"], p["ph_rr"]])) % 1.0
            stance = ph < p["duty"]
            s = np.where(stance, ph / p["duty"], (ph - p["duty"]) / (1 - p["duty"]))
            stroke = np.where(stance, 2 * s - 1, 1 - 2 * s)
            amp = np.where(FRONT > 0, p["amp_f"], p["amp_r"]) * self.effort * (1 + p["tg"] * turn * SIDE) * (
                p["turn_amp"] + (1 - p["turn_amp"]) * max(0.0, math.cos(err)))
            target[0::2] += np.where(FRONT > 0, p["bias_f"], p["bias_r"]) * self.effort + amp * stroke
            lift = np.where(FRONT > 0, p["lift_f"], p["lift_r"]) * self.effort
            target[1::2] += np.where(FRONT > 0, p["kbias_f"], p["kbias_r"]) * self.effort - lift * np.where(stance, 0.0, np.sin(math.pi * s))
        roll = -up[0] * fwd[1] + up[1] * fwd[0]
        target[1::2] -= FRONT * p["pitch_knee"] * fwd[2] + SIDE * p["roll_knee"] * roll
        target[0::2] -= FRONT * p["pitch_hip"] * fwd[2] + SIDE * p["roll_hip"] * roll
        return np.clip((p["kp"] * (target - q) - p["kd"] * qd) / self.gear, -1, 1).astype(np.float32)


# ---------------------------------------------------------------- evaluation
def tilt(core, deg):
    g = math.radians(deg)
    core.model.opt.gravity[:] = [0, -9.81 * math.sin(g), -9.81 * math.cos(g)]  # uphill is +y


def run_sim(core, gait, run, seconds, deg=0.0, seed=0, noise=0.03):
    """Returns dict: alive seconds, mean shell velocity (world x, y) over the last half, torso heading error at end."""
    rng = np.random.default_rng(seed)
    tilt(core, deg)
    core.reset(facing=0.0)
    d = core.data
    d.qpos[HAM_QPOS + 7:HAM_QPOS + 15] += rng.normal(0, noise, 8)
    d.qvel[:2] = rng.normal(0, 0.02, 2)
    gait.reset()
    n = round(seconds / DT)
    vel = []
    alive = seconds
    pitch_min = 1.0
    for i in range(n):
        obs = core.tier.obs(core.model, core.data, core.facing)
        if obs["proprio"][25] < 0.3:
            alive = i * DT
            break
        core.step(gait.act(obs["proprio"], run))
        vel.append(d.qvel[:2].copy())
    if not np.isfinite(d.qpos).all():
        alive = 0.0
    vel = np.array(vel) if vel else np.zeros((1, 2))
    tail = vel[len(vel) // 2:]
    mv = tail.mean(0)
    fwd = obs["proprio"][20:23]
    head = math.degrees(math.atan2(-fwd[0], fwd[1]))  # + left of +y in the facing frame
    return dict(alive=alive, vx=float(mv[0]), vy=float(mv[1]), head=head,
                com=float(np.linalg.norm(d.subtree_com[HAM] - d.xpos[1])))


def score(core, gait, seed=0, T=10.0):
    """Search fitness: flat 2 m/s forward, a 90-degree turn then run, climbs of 10 and 20 degrees (each 0..1+)."""
    out = {}
    r = run_sim(core, gait, (0.0, 1.0), T, 0.0, seed)
    out["flat"] = r
    f = r["alive"] / T * np.clip(r["vy"] / 2.0, -0.3, 1.1)
    r = run_sim(core, gait, (-1.0, 0.0), T, 0.0, seed + 1)
    out["turn"] = r
    f += 0.5 * r["alive"] / T * np.clip(-r["vx"] / 1.0, -0.3, 1.1)
    for deg, tgt, w in ((10, 1.0, 1.0), (20, 0.5, 2.0)):
        r = run_sim(core, gait, (0.0, 1.0), T, deg, seed + 2 + deg)
        out[f"c{deg}"] = r
        f += w * r["alive"] / T * np.clip(r["vy"] / tgt, -0.3, 1.1)
    return float(f), out


def score2(core, gait, seed=0):
    """Robust fitness: flat 15 s, two random headings 20 s, climbs 10 (10 s) and 20 (15 s, two seeds). A dead run
    counts half its alive fraction. Targets: flat 2.2, heading 1.5 along, climb10 1.0, climb20 0.7 m/s."""
    rng = np.random.default_rng(seed)
    out, f = {}, 0.0

    def term(r, along, tgt, T, w, key):
        nonlocal f
        k = 1.0 if r["alive"] >= T - 1e-6 else 0.5 * r["alive"] / T
        f += w * k * float(np.clip(along / tgt, -0.3, 1.15))
        out[key] = r

    r = run_sim(core, gait, (0.0, 1.0), 15.0, 0.0, seed)
    term(r, r["vy"], 2.2, 15.0, 1.5, "flat")
    for i in range(2):
        a = math.radians(rng.uniform(0, 360))
        run = (-math.sin(a), math.cos(a))
        r = run_sim(core, gait, run, 20.0, 0.0, seed + 10 + i)
        term(r, r["vx"] * run[0] + r["vy"] * run[1], 1.5, 20.0, 0.5, f"head{i}")
    r = run_sim(core, gait, (0.0, 1.0), 10.0, 10, seed + 20)
    term(r, r["vy"], 1.0, 10.0, 1.0, "c10")
    for i in range(2):
        r = run_sim(core, gait, (0.0, 1.0), 15.0, 20, seed + 30 + i)
        term(r, r["vy"], 0.7, 15.0, 1.5, f"c20_{i}")
    return float(f), out


def score3(core, gait, seed=0):
    """Harder fitness (round 3): flat to 2.6, three headings 30 s to 2.2, climbs 10 to 1.5 and 20 to 1.2 (3 seeds)."""
    rng = np.random.default_rng(seed)
    out, f = {}, 0.0

    def term(r, along, tgt, T, w, key):
        nonlocal f
        k = 1.0 if r["alive"] >= T - 1e-6 else 0.4 * r["alive"] / T
        f += w * k * float(np.clip(along / tgt, -0.3, 1.1))
        out[key] = r

    r = run_sim(core, gait, (0.0, 1.0), 15.0, 0.0, seed)
    term(r, r["vy"], 2.6, 15.0, 2.0, "flat")
    for i in range(3):
        a = math.radians(rng.uniform(0, 360))
        run = (-math.sin(a), math.cos(a))
        r = run_sim(core, gait, run, 30.0, 0.0, seed + 10 + i)
        term(r, r["vx"] * run[0] + r["vy"] * run[1], 2.2, 30.0, 0.7, f"head{i}")
    r = run_sim(core, gait, (0.0, 1.0), 10.0, 10, seed + 20)
    term(r, r["vy"], 1.5, 10.0, 1.0, "c10")
    for i in range(3):
        r = run_sim(core, gait, (0.0, 1.0), 15.0, 20, seed + 30 + i)
        term(r, r["vy"], 1.2, 15.0, 1.0, f"c20_{i}")
    return float(f), out


def score4(core, gait, seed=0):
    """Round 4, robustness: flat 15 s, two headings 30 s, four 20-degree climbs of 25 s (alive counts strongly)."""
    rng = np.random.default_rng(seed)
    out, f = {}, 0.0

    def term(r, along, tgt, T, w, key):
        nonlocal f
        k = 1.0 if r["alive"] >= T - 1e-6 else 0.2 * r["alive"] / T
        f += w * k * float(np.clip(along / tgt, -0.3, 1.1))
        out[key] = r

    r = run_sim(core, gait, (0.0, 1.0), 15.0, 0.0, seed)
    term(r, r["vy"], 2.6, 15.0, 1.5, "flat")
    for i in range(2):
        a = math.radians(rng.uniform(0, 360))
        run = (-math.sin(a), math.cos(a))
        r = run_sim(core, gait, run, 30.0, 0.0, seed + 10 + i)
        term(r, r["vx"] * run[0] + r["vy"] * run[1], 2.2, 30.0, 0.7, f"head{i}")
    for i in range(4):
        r = run_sim(core, gait, (0.0, 1.0), 25.0, 20, seed + 30 + i)
        term(r, r["vy"], 1.5, 25.0, 1.0, f"c20_{i}")
    return float(f), out


def build_core(skel):
    tier = make_tier(skel)
    rest = measure_rest(tier)
    tier.resting_radius = rest + 0.01
    core = Core(tier)
    return core, rest


if __name__ == "__main__":
    import time
    t0 = time.time()
    core, rest = build_core({})
    print("rest", rest, "build+rest s", time.time() - t0)
    g = Gait(core.tier)
    t0 = time.time()
    f, out = score(core, g)
    print("score", f, "s", time.time() - t0)
    print(json.dumps(out, indent=1))
