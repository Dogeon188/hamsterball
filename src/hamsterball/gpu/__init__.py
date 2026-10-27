"""The GPU-batched backend. Needs the `gpu` extra: importing any of its modules without it raises the ImportError below, so a
plain install never imports `warp` or `torch` (nothing outside this package imports it at load time)."""

try:
    import mujoco_warp  # noqa: F401
    import torch  # noqa: F401
    import warp  # noqa: F401
except ImportError as e:  # a plain install has none of them
    raise ImportError("the GPU backend needs the `gpu` extra: pip install 'hamsterball[gpu]'") from e
