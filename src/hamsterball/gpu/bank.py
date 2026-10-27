"""The level bank: a pool of levels compiled into one padded batch model, and the per-(level, task) tables the worlds
read at each reset. See bank.md."""

try:
    import warp as wp
except ImportError as e:  # a plain install has no warp
    raise ImportError("the GPU backend needs the `gpu` extra: pip install 'hamsterball[gpu]'") from e

import base64
import math

import mujoco
import numpy as np

from ..core.level import CONTACT, HAZARD_RGBA, MARKER_LISTS, MARKER_RGBA, TASK_MARKERS

TASK_NAMES = tuple(TASK_MARKERS)  # the goal one-hot's order, as env.TASKS
STATIC_TYPES = ("plane", "hfield", "box", "sphere", "capsule")  # a level's solids, in slot order
MARKER_TYPES = ("sphere", "capsule", "box")
GEOM_RGBA = (0.45, 0.5, 0.45, 1.0)  # a level solid's colour when it names none (level._geom_mjcf)
PARK_DEPTH = 1000.0  # m below the kill plane an unused slot sits
PARK_MARGIN = -1000.0  # an unused slot's geom_margin: nothing touches it (Warp shares contype across worlds)
BATCHED = ("geom_pos", "geom_quat", "geom_size", "geom_rgba", "geom_margin", "geom_dataid", "geom_aabb", "geom_rbound")


def marker_rows(level, task):
    """The markers a world on (level, task) draws, as (marker, rgba): the task's markers in task order, then the level's
    hazards."""
    block = level.tasks[task]
    rows = []
    for k in TASK_MARKERS[task]:
        rows += [(m, MARKER_RGBA[k]) for m in (block[k] if k in MARKER_LISTS else [block[k]])]
    return [r for r in rows if "attach" not in r[0]] + [(h, HAZARD_RGBA) for h in level.hazards]


def _quat_mat(euler_deg):
    q, mat = np.empty(4), np.empty(9)
    mujoco.mju_euler2Quat(q, np.radians(euler_deg), "xyz")
    mujoco.mju_quat2Mat(mat, q)
    return q, mat


def _hfield_key(g):
    return g["nrow"], g["ncol"], tuple(g["size"]), g["samples"]


class Bank:
    """The host side of a pool of levels. `levels` (`Level`s) is the pool, indexed by bank index; `keep` is a list of
    (Level, task) pairs appended after it that a world still on the last pool's levels needs until it resets (they can't
    be drawn). Entry e is one (level, task) pair with its own slot contents and tables; `tasks` limits the pairs to those
    tasks."""

    def __init__(self, levels, tasks=TASK_NAMES, keep=()):
        self.n_pool = len(levels)
        pairs = [(lv, None) for lv in levels] + [(lv, t) for lv, t in keep]
        for lv, _ in pairs:
            if lv.bodies or lv.collapse:
                raise NotImplementedError("levels with bodies, seesaws or collapsing tiles come with later tickets")
            if any(CONTACT.keys() & g.keys() for g in lv.geoms):
                raise NotImplementedError("a geom with friction, solref or solimp is not batched yet")
        self.levels = [lv for lv, _ in pairs]
        self.tasks = tuple(tasks)
        self.entries = []  # (bank index, task name)
        self.entry_of = np.full((len(self.levels), len(TASK_NAMES)), -1, np.int32)  # [bank index, task index] -> entry
        for i, (lv, only) in enumerate(pairs):
            for t in TASK_NAMES:
                if t in lv.tasks and (t == only if only else t in self.tasks):
                    self.entry_of[i, TASK_NAMES.index(t)] = len(self.entries)
                    self.entries.append((i, t))
        self.first_entry = np.array([next((e for e in row if e >= 0), -1) for row in self.entry_of], np.int32)  # -1: none
        self.keep_entry = [int(self.entry_of[self.n_pool + j, TASK_NAMES.index(t)]) for j, (_, t) in enumerate(keep)]
        self.hfields = {}  # deduplicated hfield key -> asset index
        for i, _ in self.entries:
            for g in self.levels[i].geoms:
                if g["type"] == "hfield":
                    self.hfields.setdefault(_hfield_key(g), len(self.hfields))
        # the caps: per type, the most any entry holds
        self.static_counts = {k: 0 for k in STATIC_TYPES}
        self.marker_counts = {k: 0 for k in MARKER_TYPES}
        for i, t in self.entries:
            n = {k: 0 for k in STATIC_TYPES}
            for g in self.levels[i].geoms:
                n[g["type"]] += 1
            m = {k: 0 for k in MARKER_TYPES}
            for mk, _ in marker_rows(self.levels[i], t):
                m[mk["type"]] += 1
            self.static_counts = {k: max(self.static_counts[k], n[k]) for k in STATIC_TYPES}
            self.marker_counts = {k: max(self.marker_counts[k], m[k]) for k in MARKER_TYPES}
        self.slots = [("s", k, j) for k in STATIC_TYPES for j in range(self.static_counts[k])] + [
            ("m", k, j) for k in MARKER_TYPES for j in range(self.marker_counts[k])]

    def name(self, slot):
        return "/".join(map(str, slot))

    def mjcf(self):
        """MJCF sections for `core.model.build`: every hfield as an asset, then one geom per slot, each at the origin."""
        assets = []
        for (nrow, ncol, size, samples), i in self.hfields.items():
            elev = np.frombuffer(base64.b64decode(samples), "<u2")
            assets.append(f'<hfield name="hfield{i}" nrow="{nrow}" ncol="{ncol}" size="{" ".join(map(repr, size))} 1.0" '
                          f'elevation="{" ".join(map(str, elev.tolist()))}"/>')
        out = []
        for slot in self.slots:
            kind, typ, _ = slot
            if kind == "s":  # contact on, like a level solid (level._geom_mjcf)
                shape = 'hfield="hfield0"' if typ == "hfield" else 'size="1 1 1"'
                out.append(f'<geom name="{self.name(slot)}" type="{typ}" {shape} contype="1" conaffinity="1" group="0"/>')
            else:  # contact off, like a marker (level._marker_mjcf)
                out.append(f'<geom name="{self.name(slot)}" type="{typ}" size="1 1 1" contype="0" conaffinity="0" group="0"/>')
        return "<asset>{}</asset>\n<worldbody>\n{}\n</worldbody>".format("\n".join(assets), "\n".join(out))

    def tables(self, mjm):
        """The slot tables as numpy, one row per entry: what each world's model fields take for that (level, task).
        `mjm` is the compiled padded model, which gives the slot geom ids and the values a parked slot keeps."""
        ids = self.slot_ids = [mjm.geom(self.name(s)).id for s in self.slots]
        self.row_slots = []  # per entry, the slot of each marker row
        E, S = len(self.entries), len(self.slots)
        t = {"pos": np.zeros((E, S, 3), np.float32), "quat": np.zeros((E, S, 4), np.float32),
             "size": np.zeros((E, S, 3), np.float32), "aabb": np.zeros((E, S, 2, 3), np.float32),
             "rbound": np.zeros((E, S), np.float32), "xmat": np.zeros((E, S, 3, 3), np.float32),
             "rgba": np.zeros((E, S, 4), np.float32), "margin": np.zeros((E, S), np.float32),
             "dataid": np.zeros((E, S), np.int32)}
        for e, (i, task) in enumerate(self.entries):
            lv = self.levels[i]
            solids = [("s", g["type"], _geom_row(g, self.hfields)) for g in lv.geoms]
            marks = [("m", m["type"], _marker_row(m, rgba)) for m, rgba in marker_rows(lv, task)]
            used, filled = {}, {}
            self.row_slots.append([])
            for kind, typ, row in solids + marks:
                j = used[kind, typ] = used.get((kind, typ), -1) + 1
                filled[kind, typ, j] = row
                if kind == "m":
                    self.row_slots[-1].append(self.slots.index((kind, typ, j)))
            park = lv.kill_plane_z - PARK_DEPTH
            for s, slot in enumerate(self.slots):
                g = ids[s]
                if slot in filled:
                    row = filled[slot]
                    pos, quat, size, aabb, rbound, rgba, dataid = row
                    mat = np.empty(9)
                    mujoco.mju_quat2Mat(mat, quat)
                    margin = 0.0
                else:  # parked: far under the kill plane, no contacts, invisible; its compiled size and bounds stay
                    pos, quat, mat, margin = (0.0, 0.0, park), (1.0, 0, 0, 0), np.eye(3), PARK_MARGIN
                    size, aabb, rbound = mjm.geom_size[g], mjm.geom_aabb[g].reshape(2, 3), mjm.geom_rbound[g]
                    rgba, dataid = (*mjm.geom_rgba[g][:3], 0.0), int(mjm.geom_dataid[g])
                if slot[1] == "plane":  # the plane's bounds are infinite, so they are the compiled ones
                    aabb, rbound = mjm.geom_aabb[g].reshape(2, 3), mjm.geom_rbound[g]
                t["pos"][e, s], t["quat"][e, s], t["size"][e, s] = pos, quat, size
                t["aabb"][e, s], t["rbound"][e, s], t["xmat"][e, s] = aabb, rbound, np.reshape(mat, (3, 3))
                t["rgba"][e, s], t["margin"][e, s], t["dataid"][e, s] = rgba, margin, dataid
        return ids, t


def _geom_row(g, hfields):
    """(pos, quat, size, aabb, rbound, rgba, dataid) of a level solid, as MuJoCo compiles it."""
    quat, _ = _quat_mat(g["euler"])
    size, typ, dataid = np.zeros(3), g["type"], -1
    size[:len(g["size"])] = g["size"]
    if typ == "box":
        aabb, rbound = [[0, 0, 0], size], np.linalg.norm(size)
    elif typ == "sphere":
        aabb, rbound = [[0, 0, 0], [size[0]] * 3], size[0]
    elif typ == "capsule":
        aabb, rbound = [[0, 0, 0], [size[0], size[0], size[0] + size[1]]], size[0] + size[1]
    elif typ == "hfield":
        hx, hy, top = size
        base = 1.0  # level.HFIELD_BASE
        aabb, rbound = [[0, 0, (top - base) / 2], [hx, hy, (top + base) / 2]], math.sqrt(hx * hx + hy * hy + max(top, base) ** 2)
        dataid = hfields[_hfield_key(g)]
    else:  # a plane: its bounds are the compiled ones
        aabb, rbound = [[0, 0, 0], [0, 0, 0]], 0.0
    return g["pos"], quat, size, aabb, rbound, g.get("rgba", GEOM_RGBA), dataid


def _marker_row(m, rgba):
    """The same for a marker geom (a capsule's fromto becomes a centre, half-length and axis, as the compiler does)."""
    typ = m["type"]
    if typ == "sphere":
        r = m["radius"]
        return m["pos"], (1.0, 0, 0, 0), [r, 0, 0], [[0, 0, 0], [r] * 3], r, rgba, -1
    if typ == "box":
        size = np.array(m["size"], float)
        return m["pos"], _quat_mat(m["euler"])[0], size, [[0, 0, 0], size], np.linalg.norm(size), rgba, -1
    a, b = np.array(m["fromto"][:3]), np.array(m["fromto"][3:])
    quat, r, half = np.empty(4), m["radius"], np.linalg.norm(b - a) / 2
    mujoco.mju_quatZ2Vec(quat, b - a)
    return (a + b) / 2, quat, [r, half, 0], [[0, 0, 0], [r, r, r + half]], r + half, rgba, -1


class DeviceTables:
    """A bank's tables on the device, and the kernels that load a world's slots and draw its (level, task)."""

    def __init__(self, bank, mjm, device):
        self.bank = bank
        ids, t = bank.tables(mjm)
        f = lambda x, dt: wp.array(x, dtype=dt, device=device)  # noqa: E731
        self.slot_geom = f(np.array(ids, np.int32), int)
        self.pos, self.size = f(t["pos"], wp.vec3), f(t["size"], wp.vec3)
        self.quat, self.rgba = f(t["quat"], wp.quat), f(t["rgba"], wp.vec4)
        self.aabb, self.xmat = f(t["aabb"], wp.vec3), f(t["xmat"], wp.mat33)
        self.rbound, self.margin, self.dataid = f(t["rbound"], float), f(t["margin"], float), f(t["dataid"], int)
        # the draw: pool levels that hold an entry for task t (row t), or any entry (the last row)
        nt, n = len(TASK_NAMES), bank.n_pool
        cand = np.zeros((nt + 1, max(n, 1)), np.int32)
        count = np.zeros(nt + 1, np.int32)
        for lv in range(n):
            for k in np.flatnonzero(bank.entry_of[lv] >= 0):
                cand[k, count[k]] = lv
                count[k] += 1
            if (bank.entry_of[lv] >= 0).any():
                cand[nt, count[nt]] = lv
                count[nt] += 1
        self.cand, self.count = f(cand, int), f(count, int)
        self.count_host = count
        self.entry_of = f(bank.entry_of, int)
        self.entry_level = f(np.array([i for i, _ in bank.entries], np.int32), int)
        self.entry_task = f(np.array([TASK_NAMES.index(t) for _, t in bank.entries], np.int32), int)

    def load(self, mask, entry, model, data):
        """Write each masked world's slots from its entry: the model fields, and the world pose of a static geom."""
        n, s = entry.shape[0], self.slot_geom.shape[0]
        wp.launch(load_slots, (n, s), [mask, entry, self.slot_geom, self.pos, self.quat, self.size, self.aabb, self.rbound,
                                       self.xmat, self.rgba, self.margin, self.dataid, model.geom_pos, model.geom_quat,
                                       model.geom_size, model.geom_aabb, model.geom_rbound, model.geom_rgba,
                                       model.geom_margin, model.geom_dataid, data.geom_xpos, data.geom_xmat])


@wp.kernel
def load_slots(
    mask: wp.array(dtype=wp.bool),
    entry: wp.array(dtype=int),
    slot_geom: wp.array(dtype=int),
    t_pos: wp.array2d(dtype=wp.vec3),
    t_quat: wp.array2d(dtype=wp.quat),
    t_size: wp.array2d(dtype=wp.vec3),
    t_aabb: wp.array3d(dtype=wp.vec3),
    t_rbound: wp.array2d(dtype=float),
    t_xmat: wp.array2d(dtype=wp.mat33),
    t_rgba: wp.array2d(dtype=wp.vec4),
    t_margin: wp.array2d(dtype=float),
    t_dataid: wp.array2d(dtype=int),
    geom_pos: wp.array2d(dtype=wp.vec3),
    geom_quat: wp.array2d(dtype=wp.quat),
    geom_size: wp.array2d(dtype=wp.vec3),
    geom_aabb: wp.array3d(dtype=wp.vec3),
    geom_rbound: wp.array2d(dtype=float),
    geom_rgba: wp.array2d(dtype=wp.vec4),
    geom_margin: wp.array2d(dtype=float),
    geom_dataid: wp.array2d(dtype=int),
    geom_xpos: wp.array2d(dtype=wp.vec3),
    geom_xmat: wp.array2d(dtype=wp.mat33),
):
    w, s = wp.tid()
    if not mask[w]:
        return
    e = entry[w]
    g = slot_geom[s]
    geom_pos[w, g] = t_pos[e, s]
    geom_quat[w, g] = t_quat[e, s]
    geom_size[w, g] = t_size[e, s]
    geom_aabb[w, g, 0] = t_aabb[e, s, 0]
    geom_aabb[w, g, 1] = t_aabb[e, s, 1]
    geom_rbound[w, g] = t_rbound[e, s]
    geom_rgba[w, g] = t_rgba[e, s]
    geom_margin[w, g] = t_margin[e, s]
    geom_dataid[w, g] = t_dataid[e, s]
    geom_xpos[w, g] = t_pos[e, s]  # a static geom's world pose is its local one, and forward never recomputes it
    geom_xmat[w, g] = t_xmat[e, s]


@wp.kernel
def seed_rng(seed: int, state: wp.array(dtype=wp.uint32)):
    w = wp.tid()
    state[w] = wp.rand_init(seed, w)


@wp.kernel
def draw(
    mask: wp.array(dtype=wp.bool),
    force_level: wp.array(dtype=int),
    force_task: wp.array(dtype=int),
    cand: wp.array2d(dtype=int),
    count: wp.array(dtype=int),
    entry_of: wp.array2d(dtype=int),
    rng: wp.array(dtype=wp.uint32),
    level_out: wp.array(dtype=int),
    task_out: wp.array(dtype=int),
    entry_out: wp.array(dtype=int),
):
    """Each masked world's (bank index, task) and so its entry: a forced value as given, the others uniformly at random
    among the pool levels that hold an entry for the forced task (any task: any entry), then among that level's tasks."""
    w = wp.tid()
    if not mask[w]:
        return
    st = rng[w]
    nt = entry_of.shape[1]
    t = int(force_task[w])
    lv = int(force_level[w])
    if lv < 0:
        row = int(nt)
        if t >= 0:
            row = t
        lv = cand[row, wp.randi(st, 0, count[row])]
    if t < 0:
        k = int(0)
        for j in range(nt):
            if entry_of[lv, j] >= 0:
                k += 1
        pick = wp.randi(st, 0, k)
        for j in range(nt):
            if entry_of[lv, j] >= 0:
                if pick == 0:
                    t = j
                pick -= 1
    rng[w] = st
    level_out[w] = lv
    task_out[w] = t
    entry_out[w] = entry_of[lv, t]
