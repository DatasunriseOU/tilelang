"""where_broadcast: broadcasted ``tl.where`` over a 2D tile."""
from __future__ import annotations

from typing import Any, Callable

import numpy as np

try:
    import triton  # type: ignore
    import triton.language as tl  # type: ignore
except ImportError:  # pragma: no cover -- triton optional
    triton = None  # type: ignore
    tl = None  # type: ignore


N_ROWS = 8
N_COLS = 32

LAUNCH_GRID: tuple[int, ...] = (1,)
META_ARGS: dict = {"N_ROWS": N_ROWS, "N_COLS": N_COLS}
TTIR_SIGNATURE: dict = {
    "x_ptr": "*fp32",
    "bias_ptr": "*fp32",
    "out_ptr": "*fp32",
    "N_ROWS": "constexpr",
    "N_COLS": "constexpr",
}

ATOL = 1e-4
RTOL = 1e-4


if triton is not None:

    @triton.jit
    def _where_broadcast_kernel(
        x_ptr,
        bias_ptr,
        out_ptr,
        N_ROWS: tl.constexpr,
        N_COLS: tl.constexpr,
    ):
        row_offsets = tl.arange(0, N_ROWS)[:, None]
        col_offsets = tl.arange(0, N_COLS)[None, :]
        offsets = row_offsets * N_COLS + col_offsets
        x = tl.load(x_ptr + offsets)
        bias = tl.load(bias_ptr + col_offsets)
        y = tl.where(x > 0.0, x, bias)
        tl.store(out_ptr + offsets, y)

    TRITON_KERNEL: Callable[..., Any] = _where_broadcast_kernel
else:  # pragma: no cover -- triton missing
    TRITON_KERNEL = None  # type: ignore


def make_inputs(
    *, dtype: str = "float32", seed: int = 0
) -> tuple[list[np.ndarray], np.ndarray]:
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((N_ROWS, N_COLS)).astype(dtype)
    bias = rng.standard_normal((N_COLS,)).astype(dtype)
    out = np.zeros_like(x)
    expected = numpy_reference([x, bias, out])
    return [x, bias, out], expected


def numpy_reference(args: list[np.ndarray]) -> np.ndarray:
    x, bias, _out = args
    return np.where(x > 0.0, x, bias).astype(x.dtype)
