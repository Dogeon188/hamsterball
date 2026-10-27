"""PROTOTYPE (throwaway): the final measurement of a candidate: flat speed, 20 degree climb, tumbles over headings.

Usage: python -m hamsterball.game.proto_final CANDIDATES.json [NAME ...] -> prints one JSON per candidate (and writes
$PVQ_RUN_DIR/final_<name>.json).
"""

import json
import math
import os
import sys
from multiprocessing import Pool

import numpy as np

from . import proto_skeletons as P

HEADINGS = range(0, 360, 30)
SEEDS = (4001, 4002, 4003, 4004)  # fresh seeds, never used in a search


def one(args):
    c, kind, a, seed = args
    core, rest = P.build_core(c["skel"])
    core.tier.resting_radius = c.get("rest", rest + 0.01)
    gait = P.Gait(core.tier, **c["gait"])
    if kind == "flat":
        r = P.run_sim(core, gait, (0.0, 1.0), 20.0, 0.0, seed)
    elif kind == "climb":
        r = P.run_sim(core, gait, (0.0, 1.0), 20.0, a, seed)
    else:  # heading a degrees left of forward, 30 s
        th = math.radians(a)
        r = P.run_sim(core, gait, (-math.sin(th), math.cos(th)), 30.0, 0.0, seed)
        r["speed_along"] = r["vx"] * -math.sin(th) + r["vy"] * math.cos(th)
    r["kind"], r["a"], r["seed"], r["rest"] = kind, a, seed, rest
    return r


def measure(name, c, pool):
    jobs = [(c, "flat", 0, s) for s in SEEDS] + [(c, "climb", d, s) for d in (10, 20) for s in SEEDS]
    jobs += [(c, "head", a, s) for a in HEADINGS for s in SEEDS]
    res = pool.map(one, jobs)
    f = [r for r in res if r["kind"] == "flat"]
    out = dict(name=name, rest=res[0]["rest"], com=float(np.mean([r["com"] for r in f])))
    out["flat_speed"] = round(float(np.mean([r["vy"] for r in f])), 3)
    out["flat_alive"] = round(float(np.mean([r["alive"] >= 20 for r in f])), 2)
    for d in (10, 20):
        k = [r for r in res if r["kind"] == "climb" and r["a"] == d]
        out[f"climb{d}_vy"] = round(float(np.mean([r["vy"] for r in k])), 3)
        out[f"climb{d}_alive"] = round(float(np.mean([r["alive"] >= 20 for r in k])), 2)
    h = [r for r in res if r["kind"] == "head"]
    out["tumble_rate"] = round(float(np.mean([r["alive"] < 30 for r in h])), 3)
    out["head_speed_along"] = round(float(np.mean([r["speed_along"] for r in h if r["alive"] >= 30] or [0])), 3)
    out["head_err_deg"] = round(float(np.mean([abs((r["head"] - r["a"] + 180) % 360 - 180) for r in h if r["alive"] >= 30] or [0])), 1)
    return out, res


if __name__ == "__main__":
    cands = json.load(open(sys.argv[1]))
    names = sys.argv[2:] or list(cands)
    od = os.environ.get("PVQ_RUN_DIR", ".")
    with Pool(int(os.environ.get("WORKERS", 10))) as pool:
        for n in names:
            out, res = measure(n, cands[n], pool)
            print(json.dumps(out), flush=True)
            json.dump(dict(summary=out, runs=res), open(f"{od}/final_{n}.json", "w"))
