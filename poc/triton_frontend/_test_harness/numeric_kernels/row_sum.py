"""row_sum: row-wise ``tl.sum`` over a (4, 256) tile.

This is the canonical single-axis reduction from the reducer corpus, but
kept as a live Triton kernel so the conformance ladder exercises real TTIR
capture instead of only a hand-written text fixture.
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
N_COLS = 256
BLOCK_N = 256

LAUNCH_GRID: tuple[int, ...] = (N_ROWS,)
META_ARGS: dict = {"BLOCK_N": BLOCK_N}
TTIR_SIGNATURE: dict = {
    "x_ptr": "*fp32",
    "out_ptr": "*fp32",
    "BLOCK_N": "constexpr",
}

ATOL = 1e-4
RTOL = 1e-4


if triton is not None:

    @triton.jit
    def _row_sum_kernel(
        x_ptr,
        out_ptr,
        BLOCK_N: tl.constexpr,
    ):
        row = tl.program_id(axis=0)
        offsets = tl.arange(0, BLOCK_N)
        x = tl.load(x_ptr + row * BLOCK_N + offsets)
        summed = tl.sum(x, axis=0)
        tl.store(out_ptr + row, summed)

    TRITON_KERNEL: Callable[..., Any] = _row_sum_kernel
else:  # pragma: no cover -- triton missing
    TRITON_KERNEL = None  # type: ignore


def make_inputs(*, dtype: str = "float32", seed: int = 0) -> tuple[list[np.ndarray], np.ndarray]:
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((N_ROWS, N_COLS)).astype(dtype)
    out = np.zeros((N_ROWS,), dtype=dtype)
    expected = numpy_reference([x, out])
    return [x, out], expected


def numpy_reference(args: list[np.ndarray]) -> np.ndarray:
    x, _out = args
    return x.sum(axis=1).astype(x.dtype)
