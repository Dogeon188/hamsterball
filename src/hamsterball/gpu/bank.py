"""The level bank: a pool of levels compiled into one padded batch model, and the per-(level, task) tables the worlds
read at each reset. See bank.md."""

import base64
import itertools
import math
from typing import NamedTuple

import mujoco
import numpy as np
import warp as wp

from ..core.level import HAZARD_RGBA, MARKER_LISTS, MARKER_RGBA, TASK_MARKERS, free_mjcf, sections
from ..core.model import ROLL, TORSION
from .env_step import BALL, FREE, HINGE

TASK_NAMES = tuple(TASK_MARKERS)  # the goal one-hot's order, as env.TASKS
STATIC_TYPES = ("plane", "hfield", "box", "sphere", "capsule")  # a level's solids, in slot order
MARKER_TYPES = ("sphere", "capsule", "box")
BODY_TYPES = ("box", "sphere", "capsule")  # a body's geoms (level.BODY_GEOMS), in slot order
SEESAW_RANGE = 170.0  # deg: a parked seesaw slot's joint range, wide because jnt_limited is shared across worlds
vec5 = wp.types.vector(length=5, dtype=float)  # solimp (mujoco_warp's own `vec5f`)


class Kind(NamedTuple):
    """What one kind of body slot is, in every place that switches on it."""
    name: str  # the slot's model body name stem: body0, hinge0, ball0
    nq: int  # the joint's qpos and dof counts, which size the slot in the state vector
    nv: int
    attach: int  # how an attached marker reads the body's pose (env_step's FREE, BALL, HINGE)
    seesaw: bool  # a passive joint with a pivot and per-world fields (seesaws.py), not a driven free body
    joint: str  # the slot's MJCF joint attributes ("" for a free joint)
    com: str  # the slot's inertial centre


# the body kinds, in slot order: b driven (free joint), h hinge seesaw, n ball seesaw. A ball slot's centre of mass is off its
# joint: MuJoCo marks a ball body with its centre at the joint "simple" (a diagonal block in the mass matrix) and Warp builds
# that sparsity once, but a level's seesaw may not be
KINDS = {
    "b": Kind("body", 7, 6, FREE, False, "", "0 0 0"),
    "h": Kind("hinge", 1, 1, HINGE, True, f'type="hinge" axis="1 0 0" range="-{SEESAW_RANGE!r} {SEESAW_RANGE!r}"', "0 0 0"),
    "n": Kind("ball", 4, 3, BALL, True, f'type="ball" range="0 {SEESAW_RANGE!r}"', "0.1 0 0"),
}
GEOM_RGBA = (0.45, 0.5, 0.45, 1.0)  # a level solid's colour when it names none (level._geom_mjcf)
PARK_DEPTH = 1000.0  # m below the kill plane an unused slot sits
PARK_MARGIN = -1000.0  # an unused slot's geom_margin: nothing touches it (Warp shares contype across worlds)
BATCHED = ("geom_pos", "geom_quat", "geom_size", "geom_rgba", "geom_margin", "geom_dataid", "geom_aabb", "geom_rbound",
           "geom_friction", "geom_solref", "geom_solimp")
# a solid slot's contact values compile as the shell's outer sphere mixes with a geom that sets none (friction max, solref and
# solimp MuJoCo's defaults), at priority 1 like a level geom that sets some, so a world's slot takes its level's values whole
SOLID_CONTACT = f'priority="1" condim="6" friction="1 {TORSION!r} {ROLL!r}"'
SOLREF, SOLIMP = (0.02, 1.0), (0.9, 0.95, 0.001, 0.5, 2.0)  # MuJoCo's defaults


def body_kind(body):
    """A level body's slot kind: `b` for a driven (free-joint) body, `h` a hinge seesaw, `n` a ball seesaw."""
    return "b" if "motion" in body else "h" if body["joint"]["type"] == "hinge" else "n"


def is_body(slot_kind):
    """Whether a slot's kind names a body's geom (`b0`, `h0`, `n0`) and not a level solid (`s`) or marker (`m`)."""
    return slot_kind not in ("s", "m")


def marker_rows(level, task):
    """The markers a world on (level, task) draws, as (marker, rgba): the task's markers in task order, then the level's
    hazards. A marker attached to a body is a row too: it is authored in the body's frame, and the env-step kernels
    pose its slot on the body each step."""
    block = level.tasks[task]
    rows = []
    for k in TASK_MARKERS[task]:
        rows += [(m, MARKER_RGBA[k]) for m in (block[k] if k in MARKER_LISTS else [block[k]])]
    return rows + [(h, HAZARD_RGBA) for h in level.hazards]


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
        # the body pool: as many free-joint slots as the most driven bodies a level has, then the most hinge seesaws, then
        # the most ball seesaws, each with its geom slots per type at the most any entry's body in that slot holds
        # ponytail: every world pays for the bank's largest level (the most bodies, seesaws, tiles and geoms of any entry): one
        # outlier level pads them all, in model size and per-world memory; bucket the levels by size into several banks to lift it
        most = {k: max([sum(body_kind(b) == k for b in self.levels[i].bodies) for i, _ in self.entries], default=0)
                for k in KINDS}
        self.nbodies, self.nhinge, self.nball = most["b"], most["h"], most["n"]
        self.body_kinds = [k for k in KINDS for j in range(most[k])]  # a body slot's kind
        self.body_prefix = [f"{k}{j}" for k in KINDS for j in range(most[k])]  # its geom-slot prefix
        self.body_names = [f"{KINDS[k].name}{j}" for k in KINDS for j in range(most[k])]  # and its model body name
        self.body_counts = [{k: 0 for k in BODY_TYPES} for _ in self.body_prefix]
        for i, _ in self.entries:
            for si, b in zip(self.slot_of(self.levels[i]), self.levels[i].bodies):
                for k in BODY_TYPES:
                    self.body_counts[si][k] = max(self.body_counts[si][k], sum(g["type"] == k for g in b["geoms"]))
        # the tile pool (ticket 08): as many free-joint slots as the most tiles a level has, each with one geom slot per type
        # a tile of that slot has in any entry; a level's collapse tiles are its first slots
        self.ntiles = max([len(self.levels[i].collapse["tiles"]) if self.levels[i].collapse else 0 for i, _ in self.entries], default=0)
        self.tile_counts = [{k: 0 for k in BODY_TYPES} for _ in range(self.ntiles)]
        for i, _ in self.entries:
            for slot, g in zip(self.tile_counts, self.levels[i].collapse["tiles"] if self.levels[i].collapse else []):
                slot[g["type"]] = 1
        self.slots = [("s", k, j) for k in STATIC_TYPES for j in range(self.static_counts[k])] + [
            ("m", k, j) for k in MARKER_TYPES for j in range(self.marker_counts[k])] + [
            (p, k, j) for p, c in zip(self.body_prefix, self.body_counts) for k in BODY_TYPES for j in range(c[k])] + [
            (f"t{b}", k, j) for b in range(self.ntiles) for k in BODY_TYPES for j in range(self.tile_counts[b][k])]

    def slot_of(self, level):
        """For each of a level's bodies, the index in `body_names` of the slot it takes: the n-th driven body takes the
        n-th free-joint slot, the n-th hinge seesaw the n-th hinge slot, and so on."""
        seen, out = {}, []
        for b in level.bodies:
            k = body_kind(b)
            out.append(self.body_prefix.index(f"{k}{seen.get(k, 0)}"))
            seen[k] = seen.get(k, 0) + 1
        return out

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
            if kind != "m":  # contact on, like a level solid (level._geom_mjcf)
                if is_body(kind):
                    continue  # a body's geoms are in its body (body_mjcf)
                shape = 'hfield="hfield0"' if typ == "hfield" else 'size="1 1 1"'
                out.append(f'<geom name="{self.name(slot)}" type="{typ}" {shape} contype="1" conaffinity="1" group="0" {SOLID_CONTACT}/>')
            else:  # contact off, like a marker (level._marker_mjcf)
                out.append(f'<geom name="{self.name(slot)}" type="{typ}" size="1 1 1" contype="0" conaffinity="0" group="0"/>')
        return "<asset>{}</asset>\n<worldbody>\n{}\n</worldbody>".format("\n".join(assets), "\n".join(out))

    def body_mjcf(self):
        """MJCF sections for `core.model.build`'s `bodies`: the body slots, each a body at the origin holding its geom
        slots, and a contact exclude from the world and from each other. A driven slot is a kinematic body as the CPU
        compiles one (`level._free_mjcf`); a seesaw slot is a gravity-compensated hinge or ball joint with a wide range
        and an inertial of its own, which a world's level overwrites ([seesaws.md](seesaws.md)). "" for a bank without
        bodies."""
        out = []
        for kind, prefix, name, counts in zip(self.body_kinds, self.body_prefix, self.body_names, self.body_counts):
            geoms = "".join(f'<geom name="{prefix}/{typ}/{j}" type="{typ}" size="1 1 1" contype="1" conaffinity="1" group="0" '
                            f'{SOLID_CONTACT}/>' for typ in BODY_TYPES for j in range(counts[typ]))
            if not KINDS[kind].seesaw:
                out.append(free_mjcf(name, [0.0, 0.0, 0.0], geoms))
                continue
            out.append(f'<body name="{name}" pos="0 0 0" gravcomp="1"><joint {KINDS[kind].joint} limited="true" stiffness="0" '
                       f'damping="0"/><inertial pos="{KINDS[kind].com}" mass="1" diaginertia="1 1 1"/>{geoms}</body>')
        if not out:
            return ""
        return sections(out, [("world", n) for n in self.body_names] + list(itertools.combinations(self.body_names, 2)))

    def tile_mjcf(self):
        """MJCF sections for `core.model.build`'s `bodies`, after `body_mjcf()`'s: the tile slots `tile<k>`, each a kinematic
        body as `Level.tile_mjcf` compiles a tile, at the origin holding its geom slots, excluded from the world, the body
        slots and each other. "" for a bank without tiles."""
        if not self.ntiles:
            return ""
        names = [f"tile{k}" for k in range(self.ntiles)]
        out = []
        for k, n in enumerate(names):
            geoms = "".join(f'<geom name="t{k}/{typ}/{j}" type="{typ}" size="1 1 1" contype="1" conaffinity="1" group="0" '
                            f'{SOLID_CONTACT}/>' for typ in BODY_TYPES for j in range(self.tile_counts[k][typ]))
            out.append(free_mjcf(n, [0.0, 0.0, 0.0], geoms))
        return sections(out, [(a, t) for t in names for a in ["world", *self.body_names]] + list(itertools.combinations(names, 2)))

    def tables(self, mjm):
        """The slot tables as numpy, one row per entry: what each world's model fields take for that (level, task).
        `mjm` is the compiled padded model, which gives the slot geom ids and the values a parked slot keeps."""
        ids = self.slot_ids = [mjm.geom(self.name(s)).id for s in self.slots]
        self.row_slots = []  # per entry, the slot of each marker row
        self.tile_slots = []  # per entry, the slot of each of the level's tiles (its one geom)
        E, S = len(self.entries), len(self.slots)
        t = {"pos": np.zeros((E, S, 3), np.float32), "quat": np.zeros((E, S, 4), np.float32),
             "size": np.zeros((E, S, 3), np.float32), "aabb": np.zeros((E, S, 2, 3), np.float32),
             "rbound": np.zeros((E, S), np.float32), "xmat": np.zeros((E, S, 3, 3), np.float32),
             "rgba": np.zeros((E, S, 4), np.float32), "margin": np.zeros((E, S), np.float32),
             "dataid": np.zeros((E, S), np.int32), "friction": np.zeros((E, S, 3), np.float32),
             "solref": np.zeros((E, S, 2), np.float32), "solimp": np.zeros((E, S, 5), np.float32)}
        for e, (i, task) in enumerate(self.entries):
            lv = self.levels[i]
            solids = [("s", g["type"], _geom_row(g, self.hfields)) for g in lv.geoms]
            solids += [(self.body_prefix[si], g["type"], _geom_row(g, self.hfields))
                       for si, body in zip(self.slot_of(lv), lv.bodies) for g in body["geoms"]]
            tiles = [(f"t{k}", g["type"], _geom_row({**g, "pos": [0.0, 0.0, 0.0]}, self.hfields))  # a tile geom is at its body's origin
                     for k, g in enumerate(lv.collapse["tiles"] if lv.collapse else [])]
            marks = [("m", m["type"], _marker_row(m, rgba)) for m, rgba in marker_rows(lv, task)]
            used, filled = {}, {}
            self.row_slots.append([])
            self.tile_slots.append([])
            for kind, typ, row in solids + tiles + marks:
                j = used[kind, typ] = used.get((kind, typ), -1) + 1
                filled[kind, typ, j] = row
                if kind == "m":
                    self.row_slots[-1].append(self.slots.index((kind, typ, j)))
                elif kind[0] == "t":
                    self.tile_slots[-1].append(self.slots.index((kind, typ, j)))
            park = lv.kill_plane_z - PARK_DEPTH
            for s, slot in enumerate(self.slots):
                g = ids[s]
                if slot in filled:
                    row = filled[slot]
                    pos, quat, size, aabb, rbound, rgba, dataid, contact = row
                    mat = np.empty(9)
                    mujoco.mju_quat2Mat(mat, quat)
                    margin = 0.0
                else:  # parked: far under the kill plane (a body's geom, far under its body), no contacts, invisible; its
                    # compiled size and bounds stay
                    pos, quat, mat, margin = (0.0, 0.0, -PARK_DEPTH if is_body(slot[0]) else park), (1.0, 0, 0, 0), np.eye(3), PARK_MARGIN
                    size, aabb, rbound = mjm.geom_size[g], mjm.geom_aabb[g].reshape(2, 3), mjm.geom_rbound[g]
                    rgba, dataid, contact = (*mjm.geom_rgba[g][:3], 0.0), int(mjm.geom_dataid[g]), None
                if slot[1] == "plane":  # the plane's bounds are infinite, so they are the compiled ones
                    aabb, rbound = mjm.geom_aabb[g].reshape(2, 3), mjm.geom_rbound[g]
                t["pos"][e, s], t["quat"][e, s], t["size"][e, s] = pos, quat, size
                t["aabb"][e, s], t["rbound"][e, s], t["xmat"][e, s] = aabb, rbound, np.reshape(mat, (3, 3))
                t["rgba"][e, s], t["margin"][e, s], t["dataid"][e, s] = rgba, margin, dataid
                if contact is None:  # a marker or a parked slot: the compiled values (nothing touches it)
                    contact = mjm.geom_friction[g], mjm.geom_solref[g], mjm.geom_solimp[g]
                t["friction"][e, s], t["solref"][e, s], t["solimp"][e, s] = contact
        return ids, t


def _geom_row(g, hfields):
    """(pos, quat, size, aabb, rbound, rgba, dataid, contact) of a level solid, as MuJoCo compiles it; `contact` is its
    (friction, solref, solimp), the level's values (`level._geom_mjcf`) or what the shell's outer sphere mixes to when it sets none."""
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
    slide, roll = g.get("friction", [1.0, ROLL])
    contact = (slide, TORSION, roll), g.get("solref", SOLREF), g.get("solimp", SOLIMP)
    return g["pos"], quat, size, aabb, rbound, g.get("rgba", GEOM_RGBA), dataid, contact


def _marker_row(m, rgba):
    """The same for a marker geom (a capsule's fromto becomes a centre, half-length and axis, as the compiler does)."""
    typ = m["type"]
    if typ == "sphere":
        r = m["radius"]
        return m["pos"], (1.0, 0, 0, 0), [r, 0, 0], [[0, 0, 0], [r] * 3], r, rgba, -1, None
    if typ == "box":
        size = np.array(m["size"], float)
        return m["pos"], _quat_mat(m["euler"])[0], size, [[0, 0, 0], size], np.linalg.norm(size), rgba, -1, None
    a, b = np.array(m["fromto"][:3]), np.array(m["fromto"][3:])
    quat, r, half = np.empty(4), m["radius"], np.linalg.norm(b - a) / 2
    mujoco.mju_quatZ2Vec(quat, b - a)
    return (a + b) / 2, quat, [r, half, 0], [[0, 0, 0], [r, r, r + half]], r + half, rgba, -1, None


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
        self.friction, self.solref, self.solimp = f(t["friction"], wp.vec3), f(t["solref"], wp.vec2), f(t["solimp"], vec5)
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
                                       self.xmat, self.rgba, self.margin, self.dataid, self.friction, self.solref, self.solimp,
                                       model.geom_pos, model.geom_quat, model.geom_size, model.geom_aabb, model.geom_rbound,
                                       model.geom_rgba, model.geom_margin, model.geom_dataid, model.geom_friction,
                                       model.geom_solref, model.geom_solimp, data.geom_xpos, data.geom_xmat])


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
    t_friction: wp.array2d(dtype=wp.vec3),
    t_solref: wp.array2d(dtype=wp.vec2),
    t_solimp: wp.array2d(dtype=vec5),
    geom_pos: wp.array2d(dtype=wp.vec3),
    geom_quat: wp.array2d(dtype=wp.quat),
    geom_size: wp.array2d(dtype=wp.vec3),
    geom_aabb: wp.array3d(dtype=wp.vec3),
    geom_rbound: wp.array2d(dtype=float),
    geom_rgba: wp.array2d(dtype=wp.vec4),
    geom_margin: wp.array2d(dtype=float),
    geom_dataid: wp.array2d(dtype=int),
    geom_friction: wp.array2d(dtype=wp.vec3),
    geom_solref: wp.array2d(dtype=wp.vec2),
    geom_solimp: wp.array2d(dtype=vec5),
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
    geom_friction[w, g] = t_friction[e, s]
    geom_solref[w, g] = t_solref[e, s]
    geom_solimp[w, g] = t_solimp[e, s]
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
