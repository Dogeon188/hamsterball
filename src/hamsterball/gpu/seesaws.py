"""Seesaws on the device: the per-world joint and mass fields of the bank's hinge and ball slots, and each world's solver
scale (`stat.meaninertia`), which every bank needs. See seesaws.md."""

import mujoco
import numpy as np
import warp as wp

from ..core.level import _seesaw_mjcf
from .bank import KINDS, body_kind

# the model fields a seesaw slot's values live in; each needs a world axis (the bank adds them to `put_model`'s batch_sizes)
SEESAW_FIELDS = ("jnt_axis", "jnt_stiffness", "jnt_range", "dof_damping", "body_pos", "body_mass", "body_inertia",
                 "body_ipos", "body_iquat", "body_subtreemass", "body_invweight0", "dof_invweight0")


class SeesawTables:
    """The bank's seesaw slots' values per entry, on the device, and `load`, which writes them into the model's batched
    fields for the masked worlds. A slot an entry's level does not use holds the padded model's own values, which is
    the wide-range, unit-mass slot the compile made."""

    def __init__(self, bank, cpu_model, device):
        names = bank.body_names[bank.nbodies:]  # the hinge slots, then the ball slots
        self.S = S = len(names)
        m = cpu_model
        bodies = [m.body(n) for n in names]
        self.body = [int(b.id) for b in bodies]
        self.joint = [int(b.jntadr[0]) for b in bodies]
        self.dof = [int(m.jnt_dofadr[j]) for j in self.joint]
        self.ndof = [KINDS[k].nv for k in bank.body_kinds[bank.nbodies:]]
        E = len(bank.entries)
        t = {"pos": np.zeros((E, S, 3)), "axis": np.zeros((E, S, 3)), "stiffness": np.zeros((E, S)),
             "damping": np.zeros((E, S)), "range": np.zeros((E, S, 2)), "mass": np.zeros((E, S)),
             "inertia": np.zeros((E, S, 3)), "ipos": np.zeros((E, S, 3)), "iquat": np.zeros((E, S, 4)),
             "subtree": np.zeros((E, S)), "bw": np.zeros((E, S, 2)), "dw": np.zeros((E, S, 3))}
        cache = {}
        # the solver scales its tolerance by `stat.meaninertia * nv` with the padded nv (mujoco_warp's solver.py), so a world's
        # value is the trace of its own level's mass matrix at qpos0 (the prefix, its driven bodies' slots, its seesaws, its
        # tiles) over the padded nv: the product is the CPU model's, whatever the bank pads to
        d = mujoco.MjData(m)
        mujoco.mj_forward(m, d)
        diag = _mass_diag(m, d)
        dof = lambda n: int(m.jnt_dofadr[m.body_jntadr[m.body(n).id]])  # noqa: E731
        nv0 = dof(bank.body_names[0]) if bank.body_names else dof("tile0") if bank.ntiles else m.nv  # the prefix's dofs
        free_dof = {k: dof(n) for k, n in enumerate(bank.body_names[:bank.nbodies])}
        tile_dof = [dof(f"tile{k}") for k in range(bank.ntiles)]
        mean = np.zeros(E)
        for e, (i, _) in enumerate(bank.entries):
            lv = bank.levels[i]
            used = {}
            total = diag[:nv0].sum()
            for si, b in zip(bank.slot_of(lv), lv.bodies):
                if KINDS[body_kind(b)].seesaw:
                    used[si - bank.nbodies] = b
                else:
                    total += diag[free_dof[si]:free_dof[si] + KINDS["b"].nv].sum()
            for s in range(S):
                if s in used:
                    key = id(used[s])
                    if key not in cache:
                        cache[key] = _compile(used[s])
                    row = cache[key]
                    total += row["mdiag"].sum()
                else:
                    row = _parked(m, self.body[s], self.joint[s], self.dof[s], self.ndof[s])
                for k, v in row.items():
                    if k != "mdiag":
                        t[k][e, s, ...] = v
            if lv.collapse:
                total += sum(diag[tile_dof[k]:tile_dof[k] + KINDS["b"].nv].sum() for k in range(len(lv.collapse["tiles"])))
            mean[e] = total / m.nv
        f = lambda x, dt: wp.array(np.array(x, np.float32), dtype=dt, device=device)  # noqa: E731
        self.meaninertia = f(mean, float)
        i = lambda x: wp.array(x, dtype=int, device=device)  # noqa: E731
        self.dev_body, self.dev_joint, self.dev_dof, self.dev_ndof = i(self.body), i(self.joint), i(self.dof), i(self.ndof)
        self.pos, self.axis, self.inertia, self.ipos = (f(t[k], wp.vec3) for k in ("pos", "axis", "inertia", "ipos"))
        self.dw, self.range, self.bw, self.iquat = f(t["dw"], wp.vec3), f(t["range"], wp.vec2), f(t["bw"], wp.vec2), f(t["iquat"], wp.quat)
        self.stiffness, self.damping, self.mass, self.subtree = (f(t[k], float) for k in ("stiffness", "damping", "mass", "subtree"))

    def load(self, mask, entry, model):
        """Write each masked world's solver scale and, in a bank with seesaws, its seesaw slots from its entry."""
        wp.launch(load_meaninertia, entry.shape[0], [mask, entry, self.meaninertia, model.stat.meaninertia])
        if self.S:
            wp.launch(load_seesaws, (entry.shape[0], self.S), [
                mask, entry, self.dev_body, self.dev_joint, self.dev_dof, self.dev_ndof, self.pos, self.axis, self.stiffness,
                self.damping, self.range, self.mass, self.inertia, self.ipos, self.iquat, self.subtree, self.bw, self.dw,
                model.body_pos, model.jnt_axis, model.jnt_stiffness, model.dof_damping, model.jnt_range, model.body_mass,
                model.body_inertia, model.body_ipos, model.body_iquat, model.body_subtreemass, model.body_invweight0,
                model.dof_invweight0])


def attach_frames(bank, slot_qadr):
    """For each entry, each of its level's bodies as an attached marker sees it: `(qadr, frame, pivot, axis)`, the qpos
    address of the body's slot (`slot_qadr`, indexed as `bank.body_names`) and how to read its pose (see
    `env_step.Motion`): a driven body's free joint, a ball seesaw's quaternion about its pivot, a hinge seesaw's angle
    about its axis through its pivot."""
    out = []
    for i, _ in bank.entries:
        lv = bank.levels[i]
        row = []
        for si, b in zip(bank.slot_of(lv), lv.bodies):
            kind = KINDS[body_kind(b)]
            axis = b["joint"].get("axis", [0, 0, 0]) if kind.seesaw else [0, 0, 0]
            row.append((slot_qadr[si], kind.attach, list(b["pos"]) if kind.seesaw else [0, 0, 0], list(axis)))
        out.append(row)
    return out


def _compile(body):
    """A seesaw's values as MuJoCo compiles them for the CPU: its body alone in a scratch model (a seesaw's tree is its
    own, so its mass, inertia and invweight0 don't depend on the rest of the level), and a quick compile, not the level's.
    The attached markers weigh nothing, so they are left out."""
    m = mujoco.MjModel.from_xml_string(f"<mujoco><worldbody>{_seesaw_mjcf('s', body, '')}</worldbody></mujoco>")
    nd = m.nv
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    dw = np.zeros(3)
    dw[:nd] = m.dof_invweight0[:nd]
    return {"pos": m.body_pos[1], "axis": m.jnt_axis[0], "stiffness": m.jnt_stiffness[0], "damping": m.dof_damping[0],
            "range": m.jnt_range[0], "mass": m.body_mass[1], "inertia": m.body_inertia[1], "ipos": m.body_ipos[1],
            "iquat": m.body_iquat[1], "subtree": m.body_subtreemass[1], "bw": m.body_invweight0[1], "dw": dw, "mdiag": _mass_diag(m, d)}


def _mass_diag(m, d):
    """The mass matrix's diagonal at qpos0 (`d` after mj_forward)."""
    out, col = np.zeros(m.nv), np.zeros(m.nv)
    for i in range(m.nv):
        e = np.eye(m.nv)[i]
        mujoco.mj_mulM(m, d, col, e)
        out[i] = col[i]
    return out


def _parked(m, body, joint, dof, nd):
    """An unused slot's values: the padded model's own."""
    dw = np.zeros(3)
    dw[:nd] = m.dof_invweight0[dof:dof + nd]
    return {"pos": m.body_pos[body], "axis": m.jnt_axis[joint], "stiffness": m.jnt_stiffness[joint],
            "damping": m.dof_damping[dof], "range": m.jnt_range[joint], "mass": m.body_mass[body],
            "inertia": m.body_inertia[body], "ipos": m.body_ipos[body], "iquat": m.body_iquat[body],
            "subtree": m.body_subtreemass[body], "bw": m.body_invweight0[body], "dw": dw}


@wp.kernel
def load_meaninertia(mask: wp.array(dtype=wp.bool), entry: wp.array(dtype=int), t_mean: wp.array(dtype=float),
                     meaninertia: wp.array(dtype=float)):
    """One masked world's solver scale from its entry."""
    w = wp.tid()
    if mask[w]:
        meaninertia[w] = t_mean[entry[w]]


@wp.kernel
def load_seesaws(
    mask: wp.array(dtype=wp.bool),
    entry: wp.array(dtype=int),
    body: wp.array(dtype=int),
    joint: wp.array(dtype=int),
    dof: wp.array(dtype=int),
    ndof: wp.array(dtype=int),
    t_pos: wp.array2d(dtype=wp.vec3),
    t_axis: wp.array2d(dtype=wp.vec3),
    t_stiffness: wp.array2d(dtype=float),
    t_damping: wp.array2d(dtype=float),
    t_range: wp.array2d(dtype=wp.vec2),
    t_mass: wp.array2d(dtype=float),
    t_inertia: wp.array2d(dtype=wp.vec3),
    t_ipos: wp.array2d(dtype=wp.vec3),
    t_iquat: wp.array2d(dtype=wp.quat),
    t_subtree: wp.array2d(dtype=float),
    t_bw: wp.array2d(dtype=wp.vec2),
    t_dw: wp.array2d(dtype=wp.vec3),
    body_pos: wp.array2d(dtype=wp.vec3),
    jnt_axis: wp.array2d(dtype=wp.vec3),
    jnt_stiffness: wp.array2d(dtype=float),
    dof_damping: wp.array2d(dtype=float),
    jnt_range: wp.array2d(dtype=wp.vec2),
    body_mass: wp.array2d(dtype=float),
    body_inertia: wp.array2d(dtype=wp.vec3),
    body_ipos: wp.array2d(dtype=wp.vec3),
    body_iquat: wp.array2d(dtype=wp.quat),
    body_subtreemass: wp.array2d(dtype=float),
    body_invweight0: wp.array2d(dtype=wp.vec2),
    dof_invweight0: wp.array2d(dtype=float),
):
    """One masked world's seesaw slot s: the joint's axis, stiffness and range, the dofs' damping, and the body's pivot,
    mass, inertia (with its centre and principal axes) and the derived subtree mass and invweight0 that MuJoCo's
    set_const would refresh, from the world's entry."""
    w, s = wp.tid()
    if not mask[w]:
        return
    e = entry[w]
    b = body[s]
    j = joint[s]
    body_pos[w, b] = t_pos[e, s]
    body_mass[w, b] = t_mass[e, s]
    body_inertia[w, b] = t_inertia[e, s]
    body_ipos[w, b] = t_ipos[e, s]
    body_iquat[w, b] = t_iquat[e, s]
    body_subtreemass[w, b] = t_subtree[e, s]
    body_invweight0[w, b] = t_bw[e, s]
    jnt_axis[w, j] = t_axis[e, s]
    jnt_stiffness[w, j] = t_stiffness[e, s]
    jnt_range[w, j] = t_range[e, s]
    for d in range(ndof[s]):
        dof_damping[w, dof[s] + d] = t_damping[e, s]
        dof_invweight0[w, dof[s] + d] = t_dw[e, s][d]
