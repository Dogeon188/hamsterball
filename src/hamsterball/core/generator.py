"""Procedural levels (spec §8): the LevelGenerator protocol and ProceduralGenerator, which builds a solvable hfield
level for any task at any difficulty, and the same level on every platform for the same (task, seed, difficulty,
version)."""

import base64
import heapq
import math
from typing import Protocol, runtime_checkable

import numpy as np

from .level import Level

TASKS = ("reach_goal", "waypoint_race", "collect_items", "unlock_goal", "survive")
# Spec §8's endpoint table: each row is (value at d = 0, value at d = 1), and a constructor override.
# Angles are degrees. cliff_height interpolates over d in [0.5, 1] and is 0 below.
TABLE = {
    "route_length": (5.0, 40.0),
    "corridor_width": (12.0, 3.0),
    "noise_amplitude": (0.0, 0.3),
    "climb_grade": (5.0, 15.0),
    "descent_grade": (5.0, 30.0),
    "hill_rise": (0.5, 4.0),
    "cliff_height": (0.0, 3.0),
    "hazards_per_m": (0.0, 1 / 8),
    "waypoints": (3, 8),
    "items": (2, 8),
    "keys": (1, 4),
    "time_slack": (3.0, 1.5),
    "survive_disc_radius": (4.0, 1.5),
    "survive_ridge_width": (3.0, 1.5),
    "survive_falloff_grade": (10.0, 30.0),
    "survive_duration": (10.0, 60.0),
}
COUNTS = ("waypoints", "items", "keys")  # round to whole markers
CLIFFS_FROM = 0.5  # difficulty at which cliffs start

# Spec §8's fixed values
CELL = 0.25  # hfield cell (m)
MARGIN = 4.0  # grid margin around the route's bounding box (m)
MAX_GRID = 256
KILL_DEPTH = 2.0  # kill plane below the lowest cell (m)
TARGET_RADIUS = 1.0
HAZARD_SIZE = (0.5, 0.5, 0.25)  # half-sizes of the 1 x 1 x 0.5 m hazard boxes
SPEED = 5.0  # m/s, for the time limit
ATTEMPTS = 32
MAX_CLIMB, MAX_DESCENT = math.radians(15), math.radians(30)  # the flood fill's step limits
NOISE_FEATURE = 2.0  # noise lattice spacing (m)
SHELL_R = 0.5

# ponytail: unspecified shape constants, picked to keep levels readable; promote to the table if a curriculum needs them
PIT = 1.0  # off-corridor ground sits this far below the kill plane
MAX_TURN = math.radians(45)  # largest bend between straights
STRAIGHT = 8.0  # longest straight between bends (m)
SEGMENT = 5.0  # target length of one hill or valley segment (m)
LANDING = 0.75  # flat radius around the spawn and each target (m)
RIDGE_LENGTH = 8.0  # survive ridge, end to end (m)
HAZARD_TRIES = 20
HAZARD_CLEAR = 2.0  # hazard centre to spawn or target, at least (m)


@runtime_checkable
class LevelGenerator(Protocol):
    """Anything `levels=` accepts as a generator: the tasks it supports, a version, and a call that builds a Level."""

    tasks: tuple
    version: int

    def __call__(self, task: str, seed: int, difficulty: float) -> Level: ...


class GenerationError(RuntimeError):
    """Every one of the 32 attempts failed placement or the solvability check."""


class _Retry(Exception):
    pass


class ProceduralGenerator:
    """Spec §8's generator. `ProceduralGenerator(climb_grade=(5, 10))` overrides a row of the endpoint table."""

    name = "procedural"
    version = 1
    tasks = TASKS

    def __init__(self, **overrides):
        unknown = overrides.keys() - TABLE.keys()
        if unknown:
            raise TypeError(f"unknown generator parameters {sorted(unknown)}; the table has {sorted(TABLE)}")
        self.table = TABLE | {k: (float(v[0]), float(v[1])) for k, v in overrides.items()}

    def params(self, difficulty):
        """Every table row at this difficulty: linear between its endpoints, counts rounded to whole markers."""
        out = {}
        for k, (lo, hi) in self.table.items():
            t = max(0.0, (difficulty - CLIFFS_FROM) / (1 - CLIFFS_FROM)) if k == "cliff_height" else difficulty
            v = lo + (hi - lo) * t
            out[k] = math.floor(v + 0.5) if k in COUNTS else v
        return out

    def __call__(self, task, seed, difficulty):
        return self.generate(task, seed, difficulty)[0]

    def generate(self, task, seed, difficulty):
        """(level, attempts used). Raises GenerationError after 32 failed attempts."""
        if task not in TASKS:
            raise ValueError(f"task {task!r} is not one of {TASKS}")
        if not 0 <= difficulty <= 1:
            raise ValueError(f"difficulty {difficulty} is not in [0, 1]")
        rng = np.random.Generator(np.random.PCG64(seed))  # retries take the next draws from the same stream
        p = self.params(difficulty)
        for attempt in range(1, ATTEMPTS + 1):
            try:
                d = _survive(p, rng) if task == "survive" else _route(task, p, rng)
            except _Retry:
                continue
            d["generator"] = {"name": self.name, "version": self.version, "task": task, "seed": int(seed),
                              "difficulty": float(difficulty)}
            return Level.from_dict(d), attempt
        raise GenerationError(f"{task} seed {seed} difficulty {difficulty}: no solvable level in {ATTEMPTS} attempts")


# --- construction -------------------------------------------------------------------------------------------------


def _route(task, p, rng):
    """Spec §8 steps 1-5 for every task but survive; raises _Retry when placement or the check fails."""
    n = {"reach_goal": 1, "waypoint_race": p["waypoints"], "collect_items": p["items"], "unlock_goal": p["keys"] + 1}[
        task
    ]
    length, hw = p["route_length"], p["corridor_width"] / 2
    # 1. the spine: straights from the spawn through the targets, a bend between straights
    per_leg = math.ceil(length / n / STRAIGHT)
    sub = length / (n * per_leg)
    # a bend's landing (flat, so the corridor's inner side has no seam) must fit in 30% of a straight
    lim = min(MAX_TURN, 2 * math.atan(max(0.0, 0.3 * sub - CELL) / hw))
    heading, pts, turns = rng.uniform(0, 2 * math.pi), [(0.0, 0.0)], [0.0]
    for i in range(n * per_leg):
        if i:
            turns.append(rng.uniform(-lim, lim))
            heading += turns[-1]
        x, y = pts[-1]
        pts.append((round(x + sub * math.cos(heading), 3), round(y + sub * math.sin(heading), 3)))
    turns.append(0.0)
    pts = np.array(pts)
    seg = np.diff(pts, axis=0)
    seg_len = np.sqrt((seg**2).sum(1))
    s_vert = np.concatenate([[0.0], np.cumsum(seg_len)])
    total = float(s_vert[-1])
    targets = list(range(per_leg, n * per_leg + 1, per_leg))  # vertex index of each target, in spine order

    # 2. elevation along the spine: flat landings at the spawn, targets, and bends; hills and valleys between
    s_k, u_k, u, prev_b = [], [], 0.0, 0.0
    for j, s in enumerate(s_vert):
        r = hw * math.tan(abs(turns[j]) / 2) + CELL if turns[j] else 0.0
        if j == 0 or j in targets:
            r = max(r, LANDING)
        r = min(r, 0.3 * sub)
        a, b = max(0.0, s - r), min(total, s + r)
        u += max(0.0, a - prev_b)
        s_k += [a, b]
        u_k += [u, u]
        prev_b = b
    profile_u, profile_h = _profile(u, p, rng)

    # 3. the corridor and the grid
    x, y, shape, centre = _grid(pts, 0.0)
    dist, s_cell = _nearest(x, y, pts, seg, seg_len, s_vert)
    corridor = dist <= hw
    h = np.interp(np.interp(s_cell, s_k, u_k), profile_u, profile_h)
    if _seams(s_cell, h, corridor):
        raise _Retry  # the corridor folded onto another part of itself at another height
    h += _noise(x, y, p["noise_amplitude"], rng)
    kill = round(float(h[corridor].min()) - KILL_DEPTH, 3)
    h[~corridor] = kill - PIT
    hfield, h = _hfield(h, shape, centre)

    spawn_cell = _cell(pts[0], shape, centre)
    target_cells = [_cell(pts[j], shape, centre) for j in targets]

    # 4-5. hazards off the path the flood fill found, then the check over the hazards too
    cliff = p["cliff_height"]
    found = _reach(h, corridor, spawn_cell, target_cells, cliff)
    if found is None:
        raise _Retry
    hazards = []
    n_haz = math.floor(p["hazards_per_m"] * length + 0.5)
    if n_haz:
        on_path = np.zeros(shape, bool)
        on_path[tuple(np.array(found[1]).T)] = True
        blocked = np.zeros(shape, bool)
        for _ in range(n_haz):
            box = _hazard(x, y, h, corridor, on_path, pts, seg, seg_len, s_vert, hw, total, targets, rng)
            hazards.append(box)
            blocked |= _footprint(x, y, box["pos"])
        found = _reach(h, corridor & ~blocked, spawn_cell, target_cells, cliff)
        if found is None:
            raise _Retry

    near = corridor & ((x - pts[0, 0]) ** 2 + (y - pts[0, 1]) ** 2 <= SHELL_R**2)
    spawn = [float(pts[0, 0]), float(pts[0, 1]), round(float(h[near].max()) + SHELL_R, 3)]
    marks = [{"type": "sphere", "pos": [float(pts[j, 0]), float(pts[j, 1]), round(float(h[c]) + SHELL_R, 3)],
              "radius": TARGET_RADIUS} for j, c in zip(targets, target_cells)]
    block = {"spawn": {"pos": spawn, "yaw_deg": _yaw(pts[1] - pts[0])},
             "time_limit_s": round(found[0] / SPEED * p["time_slack"], 3)}
    if task == "reach_goal":
        block["goal"] = marks[0]
    elif task == "unlock_goal":
        block["keys"], block["goal"] = marks[:-1], marks[-1]
    else:
        block["waypoints" if task == "waypoint_race" else "items"] = marks
    return {"kill_plane_z": kill, "geoms": [hfield], "hazards": hazards, "tasks": {task: block},
            "follow_yaw_deg": _yaw(pts[targets[-1]] - pts[0])}


def _profile(length, p, rng):
    """Knots (u, h) of alternating climbs and descents over `length`, at least one of each; from d = 0.5 a descent
    may hold a cliff, a drop between two knots 1 µm apart."""
    n = max(2, math.ceil(length / SEGMENT))
    ell = length / n
    up = rng.random() < 0.5
    us, hs = [0.0], [0.0]
    for _ in range(n):
        u0, h0 = us[-1], hs[-1]
        if up:
            hs.append(h0 + rng.uniform(0.5, 1) * min(p["hill_rise"], ell * math.tan(math.radians(p["climb_grade"]))))
            us.append(u0 + ell)
        else:
            fall = rng.uniform(0.5, 1) * min(p["hill_rise"], ell * math.tan(math.radians(p["descent_grade"])))
            if p["cliff_height"] > 0 and rng.random() < 0.5:
                t, c = rng.uniform(0.2, 0.8), rng.uniform(0.5, 0.9) * p["cliff_height"]
                us += [u0 + t * ell, u0 + t * ell + 1e-6]
                hs += [h0 - t * fall, h0 - t * fall - c]
                h0 -= c
            us.append(u0 + ell)
            hs.append(h0 - fall)
        up = not up
    return np.array(us), np.array(hs)


def _survive(p, rng):
    """A disc region on a plateau or a capsule region along a ridge, 50/50; each falls off at the falloff grade."""
    if rng.random() < 0.5:
        a = b = np.zeros(2)
        flat = p["survive_disc_radius"]
        region = {"type": "capsule", "fromto": [0.0, 0.0, SHELL_R, 0.0, 0.0, SHELL_R + 1.0], "radius": flat}
        yaw = 0.0
    else:
        phi = rng.uniform(0, math.pi)
        b = np.array([round(RIDGE_LENGTH / 2 * math.cos(phi), 3), round(RIDGE_LENGTH / 2 * math.sin(phi), 3)])
        a = -b
        flat = p["survive_ridge_width"] / 2
        region = {"type": "capsule", "fromto": [*map(float, a), SHELL_R, *map(float, b), SHELL_R], "radius": flat}
        yaw = _yaw(b - a)
    pts = np.array([a, b])
    x, y, shape, centre = _grid(pts, flat)
    seg = (pts[1] - pts[0])[None]
    seg_len = np.sqrt((seg**2).sum(1))
    dist, _ = _nearest(x, y, pts, seg, seg_len, np.array([0.0, float(seg_len[0])]))
    h = -np.maximum(0.0, dist - flat) * math.tan(math.radians(p["survive_falloff_grade"]))
    hfield, h = _hfield(h, shape, centre)
    return {
        "kill_plane_z": round(float(h.min()) - KILL_DEPTH, 3),
        "geoms": [hfield],
        "tasks": {"survive": {"spawn": {"pos": [0.0, 0.0, SHELL_R], "yaw_deg": yaw}, "region": region,
                              "time_limit_s": round(p["survive_duration"], 3)}},
        "follow_yaw_deg": yaw,
    }


# --- terrain --------------------------------------------------------------------------------------------------------


def _grid(pts, reach):
    """Node coordinates (x, y) of a CELL grid over the points' bounding box plus `reach` and MARGIN."""
    pad = MARGIN + reach
    lo, hi = pts.min(0) - pad, pts.max(0) + pad
    ncol, nrow = (int(math.ceil((hi[i] - lo[i]) / CELL)) + 1 for i in (0, 1))
    if max(nrow, ncol) > MAX_GRID:
        raise _Retry
    centre = (round(float(lo[0]) + (ncol - 1) * CELL / 2, 3), round(float(lo[1]) + (nrow - 1) * CELL / 2, 3))
    xs = centre[0] - (ncol - 1) * CELL / 2 + CELL * np.arange(ncol)
    ys = centre[1] + (nrow - 1) * CELL / 2 - CELL * np.arange(nrow)  # row 0 is the +y edge (level.md)
    x, y = np.meshgrid(xs, ys)
    return x, y, (nrow, ncol), centre


def _nearest(x, y, pts, seg, seg_len, s_vert):
    """Each node's distance to the spine and the arc length of its nearest spine point."""
    best, s_best = np.full(x.shape, np.inf), np.zeros(x.shape)
    for j in range(len(seg)):
        dx, dy = x - pts[j, 0], y - pts[j, 1]
        t = np.clip((dx * seg[j, 0] + dy * seg[j, 1]) / max(seg_len[j] ** 2, 1e-12), 0, 1)
        d = np.sqrt((dx - t * seg[j, 0]) ** 2 + (dy - t * seg[j, 1]) ** 2)
        closer = d < best
        best[closer], s_best[closer] = d[closer], s_vert[j] + t[closer] * seg_len[j]
    return best, s_best


def _seams(s, h, corridor):
    """Whether neighbouring corridor nodes map to distant spine points at heights a climb couldn't join."""
    for a, b in ((np.s_[1:, :], np.s_[:-1, :]), (np.s_[:, 1:], np.s_[:, :-1])):
        both = corridor[a] & corridor[b]
        if np.any(both & (np.abs(s[a] - s[b]) > 1.0) & (np.abs(h[a] - h[b]) > CELL * math.tan(MAX_CLIMB))):
            return True
    return False


def _noise(x, y, amp, rng):
    """Value noise: uniform lattice values in [-amp, amp] every NOISE_FEATURE metres, smoothstep between them."""
    gx, gy = (x - x.min()) / NOISE_FEATURE, (y.max() - y) / NOISE_FEATURE
    lattice = rng.uniform(-amp, amp, (int(gy.max()) + 2, int(gx.max()) + 2))
    ix, iy = gx.astype(int), gy.astype(int)
    fx, fy = gx - ix, gy - iy
    fx, fy = fx * fx * (3 - 2 * fx), fy * fy * (3 - 2 * fy)
    top = lattice[iy, ix] * (1 - fx) + lattice[iy, ix + 1] * fx
    bottom = lattice[iy + 1, ix] * (1 - fx) + lattice[iy + 1, ix + 1] * fx
    return top * (1 - fy) + bottom * fy


def _hfield(h, shape, centre):
    """The hfield geom of node heights h, and the heights MuJoCo will see after uint16 quantization."""
    lo, hi = float(h.min()), float(h.max())
    z0, top = round(lo, 3), round(hi - lo, 3)
    # samples span 0..65535 exactly, as MuJoCo scales the lowest to 0 and the highest to the max height
    q = np.round((h - lo) / (hi - lo) * 65535).astype("<u2")
    nrow, ncol = shape
    size = [(ncol - 1) * CELL / 2, (nrow - 1) * CELL / 2, top]
    geom = {"type": "hfield", "pos": [centre[0], centre[1], z0], "size": size,
            "nrow": nrow, "ncol": ncol, "samples": base64.b64encode(q.tobytes()).decode()}
    return geom, z0 + q / 65535 * top


def _cell(q, shape, centre):
    """The (row, column) of the node nearest point q."""
    nrow, ncol = shape
    return (int(round((centre[1] + (nrow - 1) * CELL / 2 - q[1]) / CELL)),
            int(round((q[0] - centre[0] + (ncol - 1) * CELL / 2) / CELL)))


def _yaw(v):
    """Yaw (degrees, 0 faces +y) facing direction v."""
    return round(math.degrees(math.atan2(-float(v[0]), float(v[1]))), 3)


# --- solvability ----------------------------------------------------------------------------------------------------


def _reach(h, ok, src, targets, cliff):
    """The directed flood fill: (path length, path nodes) from src through every target in order, or None."""
    nrow, ncol = h.shape
    hl, okl = h.ravel().tolist(), ok.ravel().tolist()
    total, path, at = 0.0, [src], src[0] * ncol + src[1]
    for t in targets:
        leg = _astar(hl, okl, nrow, ncol, at, t[0] * ncol + t[1], cliff)
        if leg is None:
            return None
        total += leg[0]
        path += [divmod(i, ncol) for i in leg[1]]
        at = t[0] * ncol + t[1]
    return total, path


def _astar(h, ok, nrow, ncol, src, dst, cliff):
    """Shortest directed path over 8-neighbour steps. A step passes if it climbs at most 15 degrees, or drops at most
    30 degrees plus `cliff` metres (a cliff sits on a descent, so its cell carries the descent too)."""
    if not (ok[src] and ok[dst]):
        return None
    tr, tc = divmod(dst, ncol)
    heur = lambda i: CELL * math.hypot(i // ncol - tr, i % ncol - tc)  # noqa: E731
    dist, prev, heap = {src: 0.0}, {}, [(heur(src), src)]
    up, down = math.tan(MAX_CLIMB), math.tan(MAX_DESCENT)
    while heap:
        _, i = heapq.heappop(heap)
        if i == dst:
            nodes = []
            while i != src:
                nodes.append(i)
                i = prev[i]
            return dist[dst], nodes[::-1]
        r, c = divmod(i, ncol)
        d0, h0 = dist[i], h[i]
        for dr in (-1, 0, 1):
            rr = r + dr
            if not 0 <= rr < nrow:
                continue
            for dc in (-1, 0, 1):
                cc = c + dc
                if (dr or dc) and 0 <= cc < ncol:
                    j = rr * ncol + cc
                    if not ok[j]:
                        continue
                    w = CELL * 1.4142135623730951 if dr and dc else CELL
                    dh = h[j] - h0
                    if dh > up * w or -dh > down * w + cliff:
                        continue
                    nd = d0 + w
                    if nd < dist.get(j, math.inf):
                        dist[j], prev[j] = nd, i
                        heapq.heappush(heap, (nd + heur(j), j))
    return None


def _footprint(x, y, pos):
    """Nodes a hazard blocks: its box plus one cell, as the marker test runs on the swept shell centre."""
    return (np.abs(x - pos[0]) <= HAZARD_SIZE[0] + CELL) & (np.abs(y - pos[1]) <= HAZARD_SIZE[1] + CELL)


def _hazard(x, y, h, corridor, on_path, pts, seg, seg_len, s_vert, hw, total, targets, rng):
    """A hazard box beside the route: on the corridor, clear of the spawn and targets, off the flood fill's path."""
    for _ in range(HAZARD_TRIES):
        s, off = rng.uniform(0, total), rng.uniform(-hw, hw)
        j = min(int(np.searchsorted(s_vert, s, side="right")) - 1, len(seg) - 1)
        t = (s - s_vert[j]) / seg_len[j]
        nx, ny = -seg[j, 1] / seg_len[j], seg[j, 0] / seg_len[j]
        px = round(float(pts[j, 0] + t * seg[j, 0] + off * nx), 3)
        py = round(float(pts[j, 1] + t * seg[j, 1] + off * ny), 3)
        k = np.unravel_index(np.argmin((x - px) ** 2 + (y - py) ** 2), x.shape)
        clear = all(math.hypot(px - pts[v, 0], py - pts[v, 1]) >= HAZARD_CLEAR for v in [0, *targets])
        if corridor[k] and clear and not on_path[_footprint(x, y, (px, py))].any():
            return {"type": "box", "pos": [px, py, round(float(h[k]) + SHELL_R, 3)], "size": list(HAZARD_SIZE)}
    raise _Retry
