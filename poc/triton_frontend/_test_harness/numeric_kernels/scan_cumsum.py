"""scan_cumsum: row-wise inclusive cumulative sum via ``tl.associative_scan``.

Exercises ``tt.scan`` (associative scan with arith.addf combiner) in the
live TTIR capture path, parallel to ``row_sum.py`` which exercises
``tt.reduce``. The cumulative sum is a canonical scan workload: trivial
to verify against a numpy reference, but it forces the frontend to lower
``tt.scan`` (not ``tt.reduce``), which is a distinct OP_TABLE entry.

Source: synthetic; mirrors Triton's ``tl.associative_scan`` recipe used
in flash-attention v3 and the FLA gated-delta-rule kernel.
"""
from __future__ import annotations

from typing import Any, Callable

import numpy as np

try:
    import triton  # type: ignore
    import triton.language as tl  # type: ignore
except ImportError:  # pragma: no cover -- triton optional
    triton = None  # type: ignore
    tl = None  # type: ignore


N_ROWS = 4
N_COLS = 128
BLOCK_SIZE = 128

LAUNCH_GRID: tuple[int, ...] = (N_ROWS,)
META_ARGS: dict = {"BLOCK_SIZE": BLOCK_SIZE}
TTIR_SIGNATURE: dict = {
    "x_ptr": "*fp32",
    "out_ptr": "*fp32",
    "BLOCK_SIZE": "constexpr",
}

ATOL = 1e-4
RTOL = 1e-4


if triton is not None:

    @triton.jit
    def _add_combine(a, b):
        return a + b

    @triton.jit
    def _scan_cumsum_kernel(
        x_ptr,
        out_ptr,
        BLOCK_SIZE: tl.constexpr,
    ):
        row = tl.program_id(axis=0)
        offsets = tl.arange(0, BLOCK_SIZE)
        x = tl.load(x_ptr + row * BLOCK_SIZE + offsets)
        # Inclusive prefix sum along axis=0 via tl.associative_scan.
        # Lowers to tt.scan with arith.addf combiner.
        prefix = tl.associative_scan(x, axis=0, combine_fn=_add_combine)
        tl.store(out_ptr + row * BLOCK_SIZE + offsets, prefix)

    TRITON_KERNEL: Callable[..., Any] = _scan_cumsum_kernel
else:  # pragma: no cover -- triton missing
    TRITON_KERNEL = None  # type: ignore


def make_inputs(*, dtype: str = "float32", seed: int = 0) -> tuple[list[np.ndarray], np.ndarray]:
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((N_ROWS, N_COLS)).astype(dtype)
    out = np.zeros_like(x)
    expected = numpy_reference([x, out])
    return [x, out], expected


def numpy_reference(args: list[np.ndarray]) -> np.ndarray:
    x, _out = args
    return np.cumsum(x, axis=-1).astype(x.dtype)
