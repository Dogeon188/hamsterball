"""PROTOTYPE (throwaway): (1+lambda) evolution of skeleton + gait constants on the CPU cluster.

Usage: python -m hamsterball.game.proto_sweep NAME --start base|low --gens 60 --pop 24 --workers 10 --seed 0 [--init best.json]
Writes $PVQ_RUN_DIR/NAME.jsonl (one line per generation) and NAME_best.json.
"""

import json
import os
import sys
import time
from multiprocessing import Pool

import click
import numpy as np

from . import proto_skeletons as P

# (key, lo, hi, kind); kind s skeleton, g gait
SPEC = [
    ("torso_half", 0.06, 0.15, "s"), ("head_x", 0.10, 0.30, "s"), ("m_head", 0.10, 0.80, "s"),
    ("hip_xf", 0.03, 0.18, "s"), ("hip_xr", 0.03, 0.18, "s"), ("hip_y", 0.06, 0.20, "s"), ("hip_z", -0.05, 0.0, "s"),
    ("thigh_f", 0.06, 0.18, "s"), ("shin_f", 0.06, 0.18, "s"), ("thigh_r", 0.06, 0.18, "s"), ("shin_r", 0.06, 0.18, "s"),
    ("foot_r", 0.015, 0.03, "s"), ("m_thigh", 0.04, 0.20, "s"), ("m_shin", 0.02, 0.10, "s"),
    ("gear_hip", 8, 40, "s"), ("gear_knee", 8, 40, "s"), ("damping", 0.2, 1.5, "s"), ("armature", 0.005, 0.03, "s"),
    ("hip_lo", -1.8, -0.5, "s"), ("hip_hi", 0.5, 1.8, "s"), ("knee_lo", -2.4, -1.0, "s"), ("knee_hi", 0.0, 0.5, "s"),
    ("sf_h", -0.2, 1.0, "s"), ("sf_k", -1.8, -0.2, "s"), ("sr_h", -0.2, 1.0, "s"), ("sr_k", -1.8, -0.2, "s"),
    ("mu", 1.0, 2.5, "s"),
    ("duty", 0.55, 0.9, "g"), ("amp_f", 0.1, 1.3, "g"), ("amp_r", 0.1, 1.3, "g"), ("lift_f", 0.2, 1.4, "g"),
    ("lift_r", 0.2, 1.4, "g"), ("hz", 1.0, 8.0, "g"), ("hz_min", 0.5, 4.0, "g"),
    ("bias_f", -0.3, 0.6, "g"), ("bias_r", -0.3, 0.6, "g"), ("kbias_f", -0.5, 0.5, "g"), ("kbias_r", -0.5, 0.5, "g"),
    ("ph_fr", 0.0, 1.0, "g"), ("ph_rl", 0.0, 1.0, "g"), ("ph_rr", 0.0, 1.0, "g"),
    ("pitch_knee", -1.0, 1.0, "g"), ("pitch_hip", -1.0, 1.0, "g"), ("roll_knee", -1.5, 1.5, "g"), ("roll_hip", -1.0, 1.0, "g"),
    ("kp", 15, 150, "g"), ("kd", 0.1, 2.0, "g"), ("tg", 0.0, 0.6, "g"), ("steer", 0.5, 2.0, "g"), ("turn_amp", 0.3, 1.0, "g"),
]
KEYS = [k for k, *_ in SPEC]
LO = np.array([s[1] for s in SPEC])
HI = np.array([s[2] for s in SPEC])

STARTS = {
    "base": {},
    # low long body: long torso, wide splayed hips, short legs, head and mass forward
    "low": dict(torso_half=0.13, hip_xf=0.12, hip_xr=0.12, hip_y=0.13, hip_z=-0.02, thigh_f=0.08, shin_f=0.09,
                thigh_r=0.08, shin_r=0.09, head_x=0.22, m_head=0.4, sf_h=0.5, sf_k=-1.1, sr_h=0.5, sr_k=-1.1, mu=1.6),
}


def vec(params):
    return np.clip(np.array([params[k] for k in KEYS], float), LO, HI)


def split(x):
    s, g = {}, {}
    for (k, _, _, kind), v in zip(SPEC, x):
        (s if kind == "s" else g)[k] = float(v)
    return s, g


def evaluate(args):
    x, seed, fit = args
    s, g = split(x)
    try:
        core, rest = P.build_core(s)
        if core.tier.skel["knee_lo"] >= core.tier.skel["knee_hi"]:
            return -9.0, {}
        gait = P.Gait(core.tier, **g)
        f, out = {1: P.score, 2: P.score2, 3: P.score3, 4: P.score4}[fit](core, gait, seed)
        stand_r = float(np.linalg.norm(core.data.subtree_com[2] - core.data.xpos[1]))
        out["rest"] = rest
        return f, {k: (v if k == "rest" else {a: round(b, 3) for a, b in v.items()}) for k, v in out.items()}
    except Exception as e:  # a skeleton the compiler rejects
        return -9.0, {"err": repr(e)}


@click.command()
@click.argument("name")
@click.option("--start", default="base")
@click.option("--init", default=None, help="json of a best vector's params to start from")
@click.option("--gens", default=60)
@click.option("--pop", default=24)
@click.option("--workers", default=10)
@click.option("--seed", default=0)
@click.option("--sigma", default=0.12)
@click.option("--fit", default=1)
def main(name, start, init, gens, pop, workers, seed, sigma, fit):
    out = os.environ.get("PVQ_RUN_DIR", ".")
    rng = np.random.default_rng(seed)
    params = {**P.BASE, **P.GAIT, **STARTS[start]}
    if init:
        params.update(json.load(open(init))["params"])
    best = vec(params)
    log = open(f"{out}/{name}.jsonl", "a")
    with Pool(workers) as pool:
        best_f, info = pool.map(evaluate, [(best, 0, fit)])[0]
        print("start", best_f, flush=True)
        t0 = time.time()
        for gen in range(gens):
            sig = sigma * (0.5 + 0.5 * (1 - gen / gens))
            kids = [np.clip(best + rng.normal(0, sig, len(best)) * (rng.random(len(best)) < 0.3) * (HI - LO), LO, HI)
                    for _ in range(pop)]
            cseed = int(rng.integers(1_000_000))  # fresh scenario seed each generation, same for the family
            res = pool.map(evaluate, [(k, cseed, fit) for k in kids] + [(best, cseed, fit)])
            bf = res[-1][0]  # re-score the parent on this generation's seed so the comparison is fair
            i = int(np.argmax([r[0] for r in res[:-1]]))
            if res[i][0] > bf:
                best, best_f, info = kids[i], res[i][0], res[i][1]
            else:
                best_f = bf
            rec = dict(gen=gen, f=best_f, parent=bf, kid=res[i][0], t=round(time.time() - t0), info=info)
            log.write(json.dumps(rec) + "\n")
            log.flush()
            json.dump(dict(f=best_f, params=dict(zip(KEYS, best.tolist())), info=info), open(f"{out}/{name}_best.json", "w"))
            print(gen, round(best_f, 3), round(res[i][0], 3), round(bf, 3), round(time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
