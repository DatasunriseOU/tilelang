"""softmax: row-wise softmax on a (4, 128) tile.

Exercises ``tt.reduce`` (max + sum) and ``tt.exp`` on top of the
elementwise surface. One launch per row (4 launches total).

Source: synthetic, modeled on Triton's tutorial
``02-fused-softmax.py``. We use a single BLOCK_SIZE = N_COLS so the
inner reduce is a one-shot warp reduce.
"""
from __future__ import annotations

from typing import Any, Callable, List, Tuple

import numpy as np

try:
    import triton  # type: ignore
    import triton.language as tl  # type: ignore
except ImportError:  # pragma: no cover -- triton optional
    triton = None  # type: ignore
    tl = None  # type: ignore


N_ROWS = 4
N_COLS = 128
BLOCK_SIZE = 128  # >= N_COLS so the reduce is one-shot

LAUNCH_GRID: Tuple[int, ...] = (N_ROWS,)
META_ARGS: dict = {"BLOCK_SIZE": BLOCK_SIZE}

# Mapping ``argN`` field-name -> int value for TileLang's Metal args struct.
# Triton signature for ``_softmax_kernel`` is
#   (x_ptr, out_ptr, x_row_stride, out_row_stride, n_cols, BLOCK_SIZE)
# with BLOCK_SIZE constexpr-folded out. PrimFunc params are the two
# buffers (arg0, arg1) followed by the three i32 scalars, so the args
# struct holds arg2/arg3/arg4 = x_row_stride/out_row_stride/n_cols. All
# three are N_COLS for our (N_ROWS, N_COLS) tile.
KERNEL_SCALAR_ARGS: dict = {
    "arg2": N_COLS,  # x_row_stride
    "arg3": N_COLS,  # out_row_stride
    "arg4": N_COLS,  # n_cols
}

ATOL = 1e-4
RTOL = 1e-3


if triton is not None:

    @triton.jit
    def _softmax_kernel(
        x_ptr,
        out_ptr,
        x_row_stride,
        out_row_stride,
        n_cols,
        BLOCK_SIZE: "tl.constexpr",
    ):
        row = tl.program_id(axis=0)
        col_offsets = tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < n_cols
        x_row_ptrs = x_ptr + row * x_row_stride + col_offsets
        x = tl.load(x_row_ptrs, mask=mask, other=-float("inf"))
        x_max = tl.max(x, axis=0)
        x_centered = x - x_max
        numer = tl.exp(x_centered)
        denom = tl.sum(numer, axis=0)
        y = numer / denom
        out_row_ptrs = out_ptr + row * out_row_stride + col_offsets
        tl.store(out_row_ptrs, y, mask=mask)

    TRITON_KERNEL: Callable[..., Any] = _softmax_kernel
else:  # pragma: no cover -- triton missing
    TRITON_KERNEL = None  # type: ignore


def make_inputs(
    *, dtype: str = "float32", seed: int = 0
) -> Tuple[List[np.ndarray], np.ndarray]:
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((N_ROWS, N_COLS)).astype(dtype)
    out = np.zeros_like(x)
    expected = _softmax_np(x)
    return [x, out], expected


def _softmax_np(x: np.ndarray) -> np.ndarray:
    x_max = x.max(axis=1, keepdims=True)
    e = np.exp(x - x_max)
    return (e / e.sum(axis=1, keepdims=True)).astype(x.dtype)


def numpy_reference(args: List[np.ndarray]) -> np.ndarray:
    x, _out = args
    return _softmax_np(x)
