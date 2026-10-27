"""Feel on the batched core (GPU effort, ticket 11): spec §5's drive measurements, taken as tests/test_feel.py takes them
on `Core` (same runs at 1/600 s, along +x), but through `BatchCore`, with the spec's values at 3% and the containment sink
rows within 3 mm of the CPU core's own, measured here. Worlds run side by side: one core for the free-running rows
(half and full effort, twelve thrown hamsters), one with drag off for the acceleration, one tilted for the climb."""

import math
import os

import numpy as np
import pytest

from test_feel import FULL, HALF, IDLE, STEPS_PER_S, ham_offset, new_core, run

pytestmark = pytest.mark.gpu

DEVICE = os.environ.get("HAMSTERBALL_WARP_DEVICE", "cpu")
THROWS = [(vr, d) for vr in (0, 4, 8, 12) for d in [(0, 0, -1), (0, 0, 1), (1, 0, 0)]]


def make(n):
    from hamsterball.gpu.batch_core import BatchCore

    core = BatchCore(num_envs=n, device=DEVICE)
    core.reset(facing=-math.pi / 2)  # camera forward is +x
    return core


def advance(core, actions, steps, each=None):
    """`steps` physics steps (one per call, as the CPU test), one action row per world; `each(i)` reads after each."""
    a = np.asarray(actions, np.float32)
    for i in range(steps):
        core.step(a, n=1)
        if each:
            each(i)


def view(core):
    return core.data.qpos.numpy(), core.data.qvel.numpy()


def speeds(core):
    return np.hypot(*view(core)[1][:, :2].T)


def lean_deg(core):
    q, _ = view(core)
    u = q[:, 7:10] - q[:, 0:3]
    return np.degrees(np.arccos(-u[:, 2] / np.linalg.norm(u, axis=1)))


@pytest.fixture(scope="module")
def feel():
    out = {}

    core = make(1)  # acceleration is measured uncapped, as the prototype's accel() did
    core.drag_k = 0.0
    advance(core, [FULL], STEPS_PER_S)
    v1 = speeds(core)[0]
    advance(core, [FULL], 2 * STEPS_PER_S)
    out["accel"] = (speeds(core)[0] - v1) / 2

    core = make(1)
    g = math.radians(20)
    core.model.opt.gravity.assign(np.array([[-9.81 * math.sin(g), 0, -9.81 * math.cos(g)]], np.float32))  # uphill is +x
    advance(core, [FULL], 3 * STEPS_PER_S)
    out["climb"] = view(core)[0][0, 0]

    core = make(2 + len(THROWS))
    nq = core.cpu_model.nq
    for i, (vr, direction) in enumerate(THROWS):
        s = core.state(2 + i)
        s[1 + 7:1 + 10] = s[1:4]  # the hamster starts at the shell's centre
        s[1 + nq + 6:1 + nq + 9] = np.array(direction) * vr
        core.set_state(2 + i, s)
    actions = np.zeros((core.num_envs, 2), np.float32)
    actions[0], actions[1] = HALF, FULL
    leans, worst = [], np.zeros(len(THROWS))

    def watch(i):
        t = (i + 1) / STEPS_PER_S
        if t > 2.0 and t <= 3.0:
            leans.append(lean_deg(core)[1])
        if t <= 1.0:
            q, _ = view(core)
            np.maximum(worst, np.linalg.norm(q[2:, 7:10] - q[2:, 0:3], axis=1) - 0.35, out=worst)

    advance(core, actions, 3 * STEPS_PER_S, watch)
    out["v3_half"], out["v3_full"], out["lean"] = speeds(core)[0], speeds(core)[1], float(np.mean(leans))
    advance(core, actions, 17 * STEPS_PER_S)
    out["top"] = speeds(core)[1]
    actions[1] = IDLE
    advance(core, actions, STEPS_PER_S)  # the hamster swings back to the bottom in the first second
    v1 = speeds(core)[1]
    advance(core, actions, 2 * STEPS_PER_S)
    out["coast"] = (v1 - speeds(core)[1]) / 2
    assert np.isfinite(view(core)[0][:2]).all()
    for vr in (0, 4, 8, 12):
        out[f"sink{vr}"] = max(worst[i] for i, (v, _) in enumerate(THROWS) if v == vr)
    print({k: round(float(v), 4) for k, v in out.items()})
    return out


@pytest.fixture(scope="module")
def cpu_sink():
    """The CPU core's sink rows, measured as tests/test_feel.py does."""
    out = {}
    for vr in (0, 4, 8, 12):
        worst = 0.0
        for direction in [(0, 0, -1), (0, 0, 1), (1, 0, 0)]:
            core = new_core()
            d = core.data
            d.qpos[7:10] = d.qpos[0:3]
            d.qvel[6:9] = np.array(direction) * vr

            def track():
                nonlocal worst
                worst = max(worst, np.linalg.norm(ham_offset(core)) - 0.35)

            run(core, 1.0, IDLE, track)
        out[vr] = worst
    return out


# Spec §5 values, at the spec's tolerance for the GPU (the CPU test holds 2%).
@pytest.mark.parametrize(
    "key, expected",
    [("accel", 4.06), ("v3_half", 5.6), ("v3_full", 9.6), ("lean", 54), ("top", 14.3), ("coast", 2.49), ("climb", 5.1)],
)
def test_the_batched_core_drives_like_the_play_tested_prototype(feel, key, expected):
    assert feel[key] == pytest.approx(expected, rel=0.03)


@pytest.mark.parametrize("vr", [0, 4, 8, 12])
def test_the_batched_core_contains_a_thrown_hamster_like_the_cpu_core(feel, cpu_sink, vr):
    assert feel[f"sink{vr}"] * 1000 == pytest.approx(cpu_sink[vr] * 1000, abs=3.0)


def test_the_drift_measurement_runs_on_a_few_drops():
    from measure_hfield_drift import levels, measure, summary

    cpu, gpu = measure(levels()["hill (bundled)"], 4, 0.3, DEVICE)
    assert cpu.shape == gpu.shape == (4, 3) and np.isfinite(gpu).all() and (gpu[:, 2] > 0).all()
    assert summary(cpu, gpu)["max_mm"] < 50  # a 0.3 s drop has not diverged
