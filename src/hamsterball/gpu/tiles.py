"""Collapsing tiles on the device: each tile slot's fall from its trigger time, the contact test that sets the trigger, and
the warning colour. See tiles.md."""

import numpy as np
import warp as wp

from ..core.level import TILE_WARN_RGBA
from ..core.model import CORE_SLOTS
from ..core.step import G, TILE_DROP
from .bodies import _DT

_G = wp.constant(wp.float64(G))
_WARN = wp.constant(wp.vec4(*TILE_WARN_RGBA))
_SHELL = wp.constant(1)  # core.model.SHELL


@wp.func
def fall(s: wp.float64, bottom: wp.float64):
    """core.step._fall: a tile's z offset s seconds into its fall, free fall at g held at `bottom`."""
    z = wp.float64(0.0)
    if s > wp.float64(0.0):
        z = wp.max(-wp.float64(0.5) * _G * s * s, bottom)
    return z


class TileTables:
    """The bank's tiles on the device: per entry the tile count and delay, per (entry, slot) the authored position and the
    z offset the tile stops falling at, and the addresses of the tile slots' qpos and dofs, the first tile body's id and
    the first trigger-time userdata slot in the compiled model."""

    def __init__(self, bank, cpu_model, tier, device):
        T = self.T = bank.ntiles
        self.first_slot = CORE_SLOTS + tier.nuserdata  # the tile slots' userdata follow the tier's
        j = [cpu_model.body_jntadr[cpu_model.body(f"tile{k}").id] for k in range(T)]
        self.qadr = [int(cpu_model.jnt_qposadr[x]) for x in j]
        self.slots = [(q, int(cpu_model.jnt_dofadr[x]), int(cpu_model.body(f"tile{k}").id)) for k, (q, x) in enumerate(zip(self.qadr, j))]
        self.body0 = int(cpu_model.body("tile0").id) if T else 0
        pos, bottom, count, delay = [], [], [], []
        for i, _ in bank.entries:
            lv = bank.levels[i]
            c = lv.collapse
            tiles = c["tiles"] if c else []
            count.append(len(tiles))
            delay.append(c["delay"] if c else 0.0)
            for k in range(T):
                p = tiles[k]["pos"] if k < len(tiles) else [0.0, 0.0, 0.0]
                pos.append(p)
                bottom.append(lv.kill_plane_z - TILE_DROP - p[2])
        f = lambda x, t, dt: wp.array(np.array(x, t), dtype=dt, device=device)  # noqa: E731
        self.count = f(count or [0], np.int32, int)
        self.delay = f(delay or [0.0], np.float64, wp.float64)
        self.pos = f(pos or [[0.0] * 3], np.float64, wp.vec3d)
        self.bottom = f(bottom or [0.0], np.float64, wp.float64)
        self.dev_qadr = f(self.qadr or [0], np.int32, int)
        self.dev_dadr = f([int(cpu_model.jnt_dofadr[x]) for x in j] or [0], np.int32, int)
        slots = np.full((len(bank.entries), max(T, 1)), -1, np.int32)  # each tile's geom slot, -1 when unused
        for e, row in enumerate(bank.tile_slots):
            slots[e, :len(row)] = row
        self.tile_slot = f(slots.reshape(-1), np.int32, int)

    def init(self, mask, entry, userdata):
        """Every tile of a masked world's level untouched (+inf), at a reset."""
        if self.T:
            wp.launch(init_tiles, (entry.shape[0], self.T), [mask, entry, self.count, self.first_slot, userdata])

    def drive(self, mask, entry, time, userdata, qpos, qvel):
        """`drive_tiles` for the masked worlds."""
        if self.T:
            wp.launch(drive_tiles, (entry.shape[0], self.T), [mask, entry, time, self.T, self.count, self.delay, self.pos,
                                                              self.bottom, self.first_slot, userdata, self.dev_qadr,
                                                              self.dev_dadr, qpos, qvel])

    def touch(self, model, data, entry):
        """`touch_tiles` for every world, after `mjw.step`."""
        if self.T:
            wp.launch(touch_tiles, data.contact.geom.shape[0], [data.nacon, data.contact.worldid, data.contact.geom, model.geom_bodyid,
                                                                entry, self.count, self.body0, self.first_slot, data.time, data.userdata])

    def look(self, entry, time, userdata, slot_geom, t_rgba, geom_rgba):
        """The warning colours, as `HamsterballEnv._look_tiles`, for every world."""
        if self.T:
            wp.launch(look_tiles, (entry.shape[0], self.T), [entry, self.T, self.count, self.delay, self.tile_slot, slot_geom,
                                                             t_rgba, self.first_slot, time, userdata, geom_rgba])


@wp.kernel
def init_tiles(mask: wp.array(dtype=wp.bool), entry: wp.array(dtype=int), count: wp.array(dtype=int), first: int,
               userdata: wp.array2d(dtype=float)):
    w, k = wp.tid()
    if mask[w] and k < count[entry[w]]:
        userdata[w, first + k] = wp.inf


@wp.kernel
def drive_tiles(
    mask: wp.array(dtype=wp.bool),
    entry: wp.array(dtype=int),
    time: wp.array(dtype=float),
    T: int,
    count: wp.array(dtype=int),
    delay: wp.array(dtype=wp.float64),
    pos: wp.array(dtype=wp.vec3d),
    bottom: wp.array(dtype=wp.float64),
    first: int,
    userdata: wp.array2d(dtype=float),
    qadr: wp.array(dtype=int),
    dadr: wp.array(dtype=int),
    qpos: wp.array2d(dtype=float),
    qvel: wp.array2d(dtype=float),
):
    """core.step.drive_tiles: each tile slot a masked world's level uses is at rest until `delay` after its trigger time
    (+inf: untouched), then falling at g, held at `bottom`, with the secant velocity to one physics step later. Float64, as
    `bodies.drive_bodies`."""
    w, k = wp.tid()
    e = entry[w]
    if not mask[w] or k >= count[e]:
        return
    r = e * T + k
    t = wp.float64(time[w])
    start = wp.float64(userdata[w, first + k]) + delay[e]
    z0 = fall(t - start, bottom[r])
    z1 = fall(t + _DT - start, bottom[r])
    p = pos[r]
    qa = qadr[k]
    da = dadr[k]
    qpos[w, qa] = float(p[0])
    qpos[w, qa + 1] = float(p[1])
    qpos[w, qa + 2] = float(p[2] + z0)
    qpos[w, qa + 3] = 1.0
    for i in range(3):
        qpos[w, qa + 4 + i] = 0.0
        qvel[w, da + i] = 0.0
        qvel[w, da + 3 + i] = 0.0
    qvel[w, da + 2] = float((z1 - z0) / _DT)


@wp.kernel
def touch_tiles(
    nacon: wp.array(dtype=int),
    worldid: wp.array(dtype=int),
    geom: wp.array(dtype=wp.vec2i),
    geom_bodyid: wp.array(dtype=int),
    entry: wp.array(dtype=int),
    count: wp.array(dtype=int),
    body0: int,
    first: int,
    time: wp.array(dtype=float),
    userdata: wp.array2d(dtype=float),
):
    """core.step.touch_tiles: store the world's sim time in the slot of every untouched tile the shell touches, from the
    contacts `mjw.step` leaves (the ones of the state the step started from). One thread per contact."""
    c = wp.tid()
    if c >= nacon[0]:
        return
    b1 = geom_bodyid[geom[c][0]]
    b2 = geom_bodyid[geom[c][1]]
    other = int(-1)
    if b1 == _SHELL:
        other = b2
    elif b2 == _SHELL:
        other = b1
    if other < 0:
        return
    w = worldid[c]
    i = other - body0
    if i >= 0 and i < count[entry[w]] and userdata[w, first + i] == wp.inf:
        userdata[w, first + i] = time[w]  # the same value from every contact of the tile, so the race is benign


@wp.kernel
def look_tiles(
    entry: wp.array(dtype=int),
    T: int,
    count: wp.array(dtype=int),
    delay: wp.array(dtype=wp.float64),
    tile_slot: wp.array(dtype=int),
    slot_geom: wp.array(dtype=int),
    t_rgba: wp.array2d(dtype=wp.vec4),
    first: int,
    time: wp.array(dtype=float),
    userdata: wp.array2d(dtype=float),
    geom_rgba: wp.array2d(dtype=wp.vec4),
):
    """HamsterballEnv._look_tiles: each tile's colour blended from its authored one toward TILE_WARN_RGBA over `delay` from
    its trigger (a delay of 0 is the warning colour throughout, as on the CPU; an untouched tile is its own)."""
    w, k = wp.tid()
    e = entry[w]
    if k >= count[e]:
        return
    s = tile_slot[e * T + k]
    f = float(1.0)
    if delay[e] != wp.float64(0.0):
        f = float((wp.float64(time[w]) - wp.float64(userdata[w, first + k])) / delay[e])
        f = wp.min(f, 1.0)
    f = wp.max(0.0, f)
    base = t_rgba[e, s]
    geom_rgba[w, slot_geom[s]] = base + f * (_WARN - base)
