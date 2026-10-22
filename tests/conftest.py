"""Skip tests marked `gl` where no OpenGL context can be made (GitHub's macOS runners have no GPU)."""

import pytest

import hamsterball  # noqa: F401  sets the MUJOCO_GL default before mujoco loads
import mujoco


def pytest_configure(config):
    config.addinivalue_line("markers", "gl: renders through MuJoCo's offscreen GL context")


def _gl_error():
    try:
        mujoco.GLContext(1, 1).free()
    except Exception as e:  # CGLError, EGL or OSMesa errors: each backend raises its own type
        return f"{type(e).__name__}: {e}"


def pytest_collection_modifyitems(config, items):
    gl = [item for item in items if item.get_closest_marker("gl")]
    if gl and (err := _gl_error()):
        skip = pytest.mark.skip(reason=f"no OpenGL context ({err})")
        for item in gl:
            item.add_marker(skip)
