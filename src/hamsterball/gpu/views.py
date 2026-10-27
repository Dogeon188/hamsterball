"""Pixel views on the device: `core.views.pose` as a kernel, and the Warp ray tracer's two render contexts. See views.md."""

import math

import mujoco_warp as mjw
import warp as wp

from ..core.model import FACING, HAM, PITCH, PITCH_DEFAULT, SHELL
from ..core.views import CHASE_DIST, FOLLOW_DIST, FOLLOW_PITCH, VIEWS

_FACING, _PITCH, _SHELL, _HAM = (wp.constant(x) for x in (FACING, PITCH, SHELL, HAM))
_CHASE_DIST, _FOLLOW_DIST, _FOLLOW_PITCH, _PITCH_DEFAULT = (
    wp.constant(x) for x in (CHASE_DIST, FOLLOW_DIST, FOLLOW_PITCH, PITCH_DEFAULT))
_HALF_PI = wp.constant(math.pi / 2)


@wp.func
def _aim(yaw: float, pitch: float):
    """Orientation, w first as MuJoCo holds it, of a camera facing `yaw` (0 = +y) tilted `pitch` below the horizon:
    Rz(yaw) Rx(pi/2 - pitch), since a MuJoCo camera looks along its -z with +y up."""
    cz, sz = wp.cos(yaw / 2.0), wp.sin(yaw / 2.0)
    a = _HALF_PI - pitch
    cx, sx = wp.cos(a / 2.0), wp.sin(a / 2.0)
    return wp.quat(cz * cx, cz * sx, sz * sx, cx * sz)


@wp.func
def _back(at: wp.vec3, yaw: float, pitch: float, dist: float):
    """The point `dist` behind `at` along the view direction."""
    cp = wp.cos(pitch)
    return at + dist * wp.vec3(wp.sin(yaw) * cp, -wp.cos(yaw) * cp, wp.sin(pitch))


@wp.kernel
def pose(
    xpos: wp.array2d(dtype=wp.vec3),
    site_xpos: wp.array2d(dtype=wp.vec3),
    userdata: wp.array2d(dtype=float),
    entry: wp.array(dtype=int),
    follow_yaw: wp.array(dtype=float),
    eye: int,
    chase: int,
    follow: int,
    first_person: int,
    features: int,
    mocap_pos: wp.array2d(dtype=wp.vec3),
    mocap_quat: wp.array2d(dtype=wp.quat),
):
    """`core.views.pose` for every world: the view and feature mocaps from the facing, the pitch and the level's follow yaw,
    after the positions are fresh."""
    w = wp.tid()
    facing, pitch = userdata[w, _FACING], userdata[w, _PITCH]
    c = xpos[w, _SHELL]
    q = _aim(facing, pitch)
    mocap_pos[w, chase] = _back(c, facing, pitch, _CHASE_DIST)
    mocap_quat[w, chase] = q
    yaw = follow_yaw[entry[w]]
    q = _aim(yaw, _FOLLOW_PITCH)
    mocap_pos[w, follow] = _back(c, yaw, _FOLLOW_PITCH, _FOLLOW_DIST)
    mocap_quat[w, follow] = q
    p = pitch - _PITCH_DEFAULT  # level at the default
    q = _aim(facing, p)
    mocap_pos[w, first_person] = _back(site_xpos[w, eye], facing, p, 0.0)
    mocap_quat[w, first_person] = q
    mocap_pos[w, features] = xpos[w, _HAM]
    mocap_quat[w, features] = wp.quat(wp.cos(facing / 2.0), 0.0, 0.0, wp.sin(facing / 2.0))


@wp.kernel
def to_uint8(rgb: wp.array3d(dtype=wp.vec3), out: wp.array4d(dtype=wp.uint8)):
    """Float colours in [0, 1] to bytes, rounded."""
    w, y, x = wp.tid()
    c = rgb[w, y, x]
    for k in range(3):
        out[w, y, x, k] = wp.uint8(wp.int32(wp.clamp(c[k], 0.0, 1.0) * 255.0 + 0.5))


class Views:
    """The renderer of a `BatchCore`: poses the view mocaps each env step and renders `views` for every world through
    two ray-tracer contexts (`chase` and `follow` share one; `first_person` has its own with the shell's and hamster's groups off).
    `size` is (height, width). Each rendered view is a uint8 `(num_envs, height, width, 3)` Warp array in `pixels[view]`."""

    def __init__(self, core, views, size, follow_yaw):
        m = core.cpu_model
        self.core, self.views, (self.h, self.w) = core, tuple(views), size
        self.follow_yaw = follow_yaw  # per bank entry, radians
        mocap = lambda body: int(m.body_mocapid[m.body(body).id])  # noqa: E731
        self.ids = [mocap(f"view_{v}") for v in VIEWS] + [mocap("features")]
        self.eye = m.site("hamster_eye").id
        self.cam = {v: m.camera(v).id for v in VIEWS}
        n = core.num_envs
        with wp.ScopedDevice(core.device):
            self.rgb = wp.zeros((n, self.h, self.w), dtype=wp.vec3)
            self.pixels = {v: wp.zeros((n, self.h, self.w, 3), dtype=wp.uint8) for v in self.views}
            # Inner tiles (group 3) and hitbox geoms (group 4) are in no list; first person also hides the shell (1) and the
            # hamster (2). The ray tracer draws the shell's translucent outer sphere opaque, which would hide the hamster
            # and its features from chase and follow, so those hide the shell too (views.md, Decisions).
            self.contexts = []
            for names, groups in ((("chase", "follow"), [0, 2]), (("first_person",), [0])):
                names = [v for v in names if v in self.views]
                if names:
                    rc = mjw.create_render_context(
                        m, nworld=n, cam_res=(self.w, self.h), render_rgb=True, use_textures=False, use_shadows=False,
                        enabled_geom_groups=groups, cam_active=[self.cam[v] for v in names])
                    self.contexts.append((rc, names))

    def pose(self):
        """Pose the view and feature mocaps from the current positions (`core.views.pose`)."""
        c, d = self.core, self.core.data
        wp.launch(pose, c.num_envs, [d.xpos, d.site_xpos, d.userdata, c.entry, self.follow_yaw, self.eye, *self.ids,
                                     d.mocap_pos, d.mocap_quat], device=c.device)

    def render(self, out=None):
        """Draw every view into `out` (a dict of uint8 Warp arrays; default `pixels`) from the posed mocaps."""
        c, d = self.core, self.core.data
        out = self.pixels if out is None else out
        with wp.ScopedDevice(c.device):
            mjw.kinematics(c.model, d)  # the mocap poses to their bodies,
            mjw.camlight(c.model, d)  # then to the cameras
            for rc, names in self.contexts:
                mjw.refit_bvh(c.model, d, rc)  # geoms moved since the last render
                mjw.render(c.model, d, rc)
                for i, v in enumerate(names):  # get_rgb indexes the context's active cameras, not the model's
                    mjw.get_rgb(rc, i, self.rgb)
                    wp.launch(to_uint8, (c.num_envs, self.h, self.w), [self.rgb, out[v]])
