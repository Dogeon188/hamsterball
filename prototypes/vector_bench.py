# /// script
# requires-python = ">=3.11"
# dependencies = ["mujoco", "glfw", "numpy", "gymnasium"]
# ///
"""TASK, throwaway: measure state-only env throughput on the CPU, single process and vectorized.

Answers the "Measure CPU vector throughput" ticket. Not production code. The core step is honest_drive.control() with
PICK at 1/600 s (the "Re-check the feel at 1/600 s" setup), timed as-is and as a tidied copy (cached ids, no name
lookups), plus the chase yaw per physics step, a swept marker test per env step, and the 19-D state / 10-D goal obs.

Run: uv run prototypes/vector_bench.py [check|raw|single|vector|all] [--secs 5] [--reps 3] [--cores asis,fast]
"""

import functools
import math
import os
import platform
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")

import gymnasium as gym
import mujoco
import numpy as np

import honest_drive as hd

hd.TIMESTEP = 1 / 600
PICK = hd.PICK
FRAME_SKIP = 10
EP_STEPS = 600
YAW_TAU, YAW_MIN_SPEED = 0.55, 0.8
GOAL_R = 0.5
HF_N, HF_HALF, HF_Z = 256, 60.0, 0.6  # hfield grid, half extent (m), max height (m)
DRAG_K = 0.059  # frozen at the "Re-check the feel at 1/600 s" value


def build(level):
    m_ham, m_shell = PICK.masses
    slip = ' priority="1" condim="1"'
    ham = (f'<body name="ham" pos="0 0 {hd.T + hd.R_HAM + .01}"><freejoint/>'
           f'<geom type="sphere" size="{hd.R_HAM}" mass="{m_ham}" solref="{hd.SOLREF}" margin="{hd.MARGIN}"{slip} '
           f'contype="2" conaffinity="2"/></body>')
    if level == "flat":
        asset, ground, z0 = "", '<geom type="plane" size="80 80 .1"/>', 0.0
    else:
        asset = f'<asset><hfield name="terrain" nrow="{HF_N}" ncol="{HF_N}" size="{HF_HALF} {HF_HALF} {HF_Z} 0.1"/></asset>'
        ground, z0 = '<geom type="hfield" hfield="terrain"/>', HF_Z
    xml = f"""<mujoco><option timestep="{hd.TIMESTEP}" integrator="implicitfast"/><size nuserdata="1"/>{asset}
<worldbody>{ground}
<body name="shell" pos="0 0 {z0 + hd.R + .05}"><freejoint/>
{hd.shell_wall(m_shell, True)}
</body>
{ham}
</worldbody></mujoco>"""
    m = mujoco.MjModel.from_xml_string(xml)
    if level == "hfield":  # rolling sines, 0..1 scaled by HF_Z; centre lifted so the spawn sits on a crest
        x = np.linspace(-1, 1, HF_N) * HF_HALF
        X, Y = np.meshgrid(x, x)
        h = 0.5 + 0.25 * np.sin(X / 4.0) * np.cos(Y / 5.0) + 0.25 * np.sin((X + Y) / 9.0)
        m.hfield_data[:] = np.clip(h, 0, 1).ravel()
    d = mujoco.MjData(m)
    d.qpos[7:10] = d.qpos[0:3] - [0, 0, hd.L_ARM - 0.01]
    return m, d


def cross(a, b):  # np.cross spends ~20 us per call on axis handling for 3-vectors
    return np.array([a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]])


class FastCore:
    """control() for PICK (foot+grip, drag, field) with ids, normals, and constants hoisted out of the step, and a
    hand-written cross product."""

    def __init__(self, m):
        self.s, self.h = m.body("shell").id, m.body("ham").id
        a = m.body_geomadr[self.s]
        self.normals = m.geom_pos[a:a + hd.N_TILES] / (hd.R - hd.T / 2)
        self.m_ham, self.m_shell = PICK.masses
        self.mu = self.m_ham * self.m_shell / (self.m_ham + self.m_shell)
        self.scale = self.m_ham / 2.0
        self.lean = math.radians(PICK.lean)
        self.res = np.zeros(6)
        self.down = np.array([0, 0, -1.0])

    def vel(self, m, d, b):
        mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_BODY, b, self.res, 0)
        return self.res[3:].copy()

    def control(self, m, d, cal, v, e):
        s, h = self.s, self.h
        xf = d.xfrc_applied
        xf[:] = 0
        u = d.qvel[:3]
        xf[s, :3] = -cal * math.sqrt(u @ u) * u
        r = d.xipos[h] - d.xpos[s]
        dist = math.sqrt(r @ r)
        depth = dist - hd.L_ARM - hd.FIELD_START
        if depth > 0:
            u = r / dist
            out = (d.qvel[6:9] - d.qvel[0:3]) @ u
            if out > 0:
                j = min(1.0, depth / (hd.FIELD_HARD - hd.FIELD_START)) * self.mu * out
                d.qvel[6:9] -= j / self.m_ham * u
                d.qvel[0:3] += j / self.m_shell * u
            over = depth - (hd.FIELD_HARD - hd.FIELD_START)
            if over > 0:
                tot = self.m_ham + self.m_shell
                d.qpos[7:10] -= over * self.m_shell / tot * u
                d.qpos[0:3] += over * self.m_ham / tot * u
                mujoco.mj_forward(m, d)
        rl = d.xmat[s].reshape(3, 3).T @ (d.xipos[h] - d.xpos[s])
        if hd.R - hd.T - hd.R_HAM - (self.normals @ rl).max() <= hd.MARGIN + hd.C_TOUCH:
            d.userdata[0] = d.time
        if d.time - d.userdata[0] > hd.C_GRACE:
            return
        c, hp = d.xpos[s], d.xipos[h]
        n = hp - c
        n /= math.sqrt(n @ n) + 1e-9
        target = math.sin(e * self.lean) * v + math.cos(e * self.lean) * self.down
        rel_v = self.vel(m, d, h) - self.vel(m, d, s)
        f = self.scale * (hd.C_KP * hd.L_ARM * (target - n) - hd.C_KD * rel_v)
        f -= (f @ n) * n
        norm = math.sqrt(f @ f)
        if norm > hd.C_F_MAX * self.scale:
            f *= hd.C_F_MAX * self.scale / norm
        p = c + n * (hd.R - hd.T)
        xf[h, :3] += f
        xf[s, :3] -= f
        xf[h, 3:] += cross(p - hp, f)
        xf[s, 3:] += cross(p - d.xipos[s], -f)
        grip = -hd.K_GRIP * self.scale * (d.cvel[h, :3] - d.cvel[s, :3])
        xf[h, 3:] += grip
        xf[s, 3:] -= grip


def wall_contact(m, d, core):
    if core is not None:
        return float(d.time - d.userdata[0] <= hd.C_GRACE)
    return float(hd.wall_gap(m, d) <= hd.MARGIN + hd.C_TOUCH)


def seg_hits_sphere(a, b, c, r):
    ab = b - a
    t = 0.0 if (L := ab @ ab) == 0 else min(1.0, max(0.0, (c - a) @ ab / L))
    q = a + t * ab - c
    return q @ q <= r * r


class BenchEnv(gym.Env):
    """Minimal abstract-tier reach-goal env. core: "asis" (honest_drive.control) or "fast" (FastCore)."""

    def __init__(self, level="flat", core="asis", profile=False):
        self.m, self.d = build(level)
        self.level = level
        self.fast = FastCore(self.m) if core == "fast" else None
        self.s, self.h = self.m.body("shell").id, self.m.body("ham").id
        self.profile = profile
        self.t = dict(control=0.0, mj_step=0.0, yaw=0.0, marker=0.0, obs=0.0, total=0.0)
        self.action_space = gym.spaces.Box(-1, 1, (2,), np.float32)
        self.observation_space = gym.spaces.Dict(
            state=gym.spaces.Box(-np.inf, np.inf, (19,), np.float32),
            goal=gym.spaces.Box(-np.inf, np.inf, (10,), np.float32))
        self.qpos0 = self.d.qpos.copy()

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.m, self.d)
        self.d.qpos[:] = self.qpos0
        mujoco.mj_forward(self.m, self.d)
        self.yaw, self.n = 0.0, 0
        self.goal = np.array([*self.np_random.uniform(-20, 20, 2), self.d.qpos[2]])
        self.prev = self.d.qpos[0:3].copy()
        return self._obs(), {}

    def _step_yaw(self):
        vx, vy = self.d.qvel[0], self.d.qvel[1]
        if vx * vx + vy * vy > YAW_MIN_SPEED ** 2:
            err = (math.atan2(vy, vx) - self.yaw + math.pi) % (2 * math.pi) - math.pi
            self.yaw += err * hd.TIMESTEP / YAW_TAU

    def _obs(self):
        d, s, h = self.d, self.s, self.h
        cy, sy = math.cos(self.yaw), math.sin(self.yaw)
        rot = np.array([[cy, sy, 0], [-sy, cy, 0], [0, 0, 1.0]])  # world -> camera yaw frame
        state = np.empty(19, np.float32)
        state[0:3] = d.qpos[0:3]
        state[3] = d.qpos[2] + 1.0
        state[4:6] = sy, cy
        state[6:9] = rot @ d.qvel[0:3]
        state[9:12] = rot @ d.qvel[3:6]
        state[12:15] = rot @ (d.xipos[h] - d.xpos[s])
        state[15:18] = rot @ (d.qvel[6:9] - d.qvel[0:3])
        state[18] = wall_contact(self.m, d, self.fast)
        goal = np.zeros(10, np.float32)
        goal[0:3] = rot @ (self.goal - d.qpos[0:3])
        goal[3], goal[4], goal[5] = 1.0, 1.0, 1.0
        return dict(state=state, goal=goal)

    def step(self, action):
        pc = time.perf_counter
        t0 = pc()
        m, d = self.m, self.d
        a = np.asarray(action, np.float64)
        e = min(1.0, math.hypot(a[0], a[1]))
        cy, sy = math.cos(self.yaw), math.sin(self.yaw)
        v = np.array([cy * a[0] - sy * a[1], sy * a[0] + cy * a[1], 0.0])
        v /= np.linalg.norm(v) + 1e-9
        prof = self.profile
        for _ in range(FRAME_SKIP):
            t1 = pc() if prof else 0
            if self.fast:
                self.fast.control(m, d, DRAG_K, v, e)
            else:
                hd.control(m, d, PICK, DRAG_K, v, e)
            if prof:
                t2 = pc(); self.t["control"] += t2 - t1
            mujoco.mj_step(m, d)
            if prof:
                t3 = pc(); self.t["mj_step"] += t3 - t2
            self._step_yaw()
            if prof:
                self.t["yaw"] += pc() - t3
        t4 = pc() if prof else 0
        cur = d.qpos[0:3].copy()
        hit = seg_hits_sphere(self.prev, cur, self.goal, GOAL_R)
        fell = cur[2] < -1.0 or not np.isfinite(d.qpos).all()
        self.prev = cur
        if prof:
            t5 = pc(); self.t["marker"] += t5 - t4
        obs = self._obs()
        if prof:
            self.t["obs"] += pc() - t5
            self.t["total"] += pc() - t0
        self.n += 1
        off = abs(cur[0]) > HF_HALF - 5 or abs(cur[1]) > HF_HALF - 5
        return obs, float(hit) - 0.01, bool(hit or fell), self.n >= EP_STEPS or off, {}


def make_env(level, core):
    return BenchEnv(level, core)


class Actions:
    """Random run vectors held for 60 env steps: steers like a wandering player, keeps the ball moving."""

    def __init__(self, n, seed=0):
        self.rng, self.n, self.k = np.random.default_rng(seed), n, 0
        self.a = None

    def __call__(self):
        if self.k % 60 == 0:
            ang = self.rng.uniform(-math.pi, math.pi, self.n)
            eff = self.rng.uniform(0.2, 1.0, self.n)
            self.a = np.stack([np.cos(ang) * eff, np.sin(ang) * eff], 1).astype(np.float32)
        self.k += 1
        return self.a


def bench_raw(level, secs):
    m, d = build(level)
    mujoco.mj_step(m, d, 600)
    n, t0 = 0, time.perf_counter()
    while time.perf_counter() - t0 < secs:
        mujoco.mj_step(m, d, 600)
        n += 600
    return n / (time.perf_counter() - t0)


def bench_single(level, core, secs):
    env = BenchEnv(level, core, profile=True)
    env.reset(seed=0)
    act = Actions(1)
    n, t0 = 0, time.perf_counter()
    while time.perf_counter() - t0 < secs:
        _, _, term, trunc, _ = env.step(act()[0])
        n += 1
        if term or trunc:
            env.reset()
    wall = time.perf_counter() - t0
    per = {k: v / n * 1e6 for k, v in env.t.items()}  # us per env step
    return n / wall, per


def bench_vector(kind, level, core, nenv, secs):
    fns = [functools.partial(make_env, level, core) for _ in range(nenv)]
    if kind == "sync":
        venv = gym.vector.SyncVectorEnv(fns)
    else:
        venv = gym.vector.AsyncVectorEnv(fns, context="spawn")
    venv.reset(seed=0)
    act = Actions(nenv)
    for _ in range(50):
        venv.step(act())
    n, t0 = 0, time.perf_counter()
    while time.perf_counter() - t0 < secs:
        venv.step(act())
        n += nenv
    rate = n / (time.perf_counter() - t0)
    venv.close()
    return rate


def med(xs):
    return float(np.median(xs))


def check():
    """The tidied core must match control() exactly: same actions, 1200 env steps, bitwise-equal qpos on both levels."""
    for lv in ["flat", "hfield"]:
        a, b = BenchEnv(lv, "asis"), BenchEnv(lv, "fast")
        a.reset(seed=1); b.reset(seed=1)
        act = Actions(1, seed=1)
        top = 0.0
        for _ in range(1200):
            x = act()[0]
            a.step(x); b.step(x)
            top = max(top, float(np.linalg.norm(a.d.qvel[:2])))
        assert np.array_equal(a.d.qpos, b.d.qpos), (lv, np.abs(a.d.qpos - b.d.qpos).max())
        print(f"check {lv}: fast == asis over 1200 env steps, top speed {top:.1f} m/s, final xy {a.d.qpos[:2].round(1)}")


def main():
    args = sys.argv[1:]
    mode = args[0] if args and not args[0].startswith("--") else "all"
    secs = float(args[args.index("--secs") + 1]) if "--secs" in args else 5.0
    reps = int(args[args.index("--reps") + 1]) if "--reps" in args else 3
    print(f"# {platform.platform()} {platform.processor()} cpus={os.cpu_count()} python={platform.python_version()} "
          f"mujoco={mujoco.__version__} gymnasium={gym.__version__} numpy={np.__version__} secs={secs} reps={reps}", flush=True)
    levels = ["flat", "hfield"]
    cores = args[args.index("--cores") + 1].split(",") if "--cores" in args else ["asis", "fast"]
    if mode in ("check", "all"):
        check()
    if mode in ("raw", "all"):
        for lv in levels:
            r = med([bench_raw(lv, secs) for _ in range(reps)])
            print(f"raw  {lv:6s} mj_step/s {r:9.0f}  env-equiv/s {r / FRAME_SKIP:7.0f}", flush=True)
    if mode in ("single", "all"):
        for lv in levels:
            for c in cores:
                runs = [bench_single(lv, c, secs) for _ in range(reps)]
                rate = med([r for r, _ in runs])
                per = {k: med([p[k] for _, p in runs]) for k in runs[0][1]}
                parts = " ".join(f"{k}={v:6.1f}" for k, v in per.items())
                print(f"single {lv:6s} {c:4s} env/s {rate:7.0f}  us/env-step {parts}", flush=True)
    if mode in ("vector", "all"):
        for lv in levels:
            for c in cores:
                for kind, ns in [("sync", [1, 4, 16]), ("async", [1, 2, 4, 8, 16])]:
                    for nenv in ns:
                        r = med([bench_vector(kind, lv, c, nenv, secs) for _ in range(reps)])
                        print(f"vector {kind:5s} {lv:6s} {c:4s} n={nenv:2d} env/s {r:7.0f}  per-copy {r / nenv:6.0f}", flush=True)


if __name__ == "__main__":
    main()
