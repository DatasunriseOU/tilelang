"""gather_rows_3d: batched row gather over a 3D tensor.

This mirrors the reducer corpus' nanochat-style gather/scatter pattern with
a live Triton kernel: each program copies one selected row for one batch from
``src[batch, idx[gather], :]`` to ``out[batch, gather, :]``.
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


N_BATCH = 2
N_ROWS = 5
N_GATHER = 4
N_COLS = 128
BLOCK_N = 128

LAUNCH_GRID: tuple[int, ...] = (N_GATHER, N_BATCH)
META_ARGS: dict = {
    "N_ROWS": N_ROWS,
    "N_GATHER": N_GATHER,
    "BLOCK_N": BLOCK_N,
}
TTIR_SIGNATURE: dict = {
    "src_ptr": "*fp32",
    "idx_ptr": "*i32",
    "out_ptr": "*fp32",
    "N_ROWS": "constexpr",
    "N_GATHER": "constexpr",
    "BLOCK_N": "constexpr",
}

ATOL = 1e-4
RTOL = 1e-4


if triton is not None:

    @triton.jit
    def _gather_rows_3d_kernel(
        src_ptr,
        idx_ptr,
        out_ptr,
        N_ROWS: tl.constexpr,
        N_GATHER: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        gather = tl.program_id(axis=0)
        batch = tl.program_id(axis=1)
        cols = tl.arange(0, BLOCK_N)
        row = tl.load(idx_ptr + gather)

        src_offsets = (batch * N_ROWS + row) * BLOCK_N + cols
        out_offsets = (batch * N_GATHER + gather) * BLOCK_N + cols
        values = tl.load(src_ptr + src_offsets)
        tl.store(out_ptr + out_offsets, values)

    TRITON_KERNEL: Callable[..., Any] = _gather_rows_3d_kernel
else:  # pragma: no cover -- triton missing
    TRITON_KERNEL = None  # type: ignore


def make_inputs(*, dtype: str = "float32", seed: int = 0) -> tuple[list[np.ndarray], np.ndarray]:
    rng = np.random.default_rng(seed)
    src = rng.standard_normal((N_BATCH, N_ROWS, N_COLS)).astype(dtype)
    idx = np.array([3, 1, 4, 0], dtype=np.int32)
    out = np.zeros((N_BATCH, N_GATHER, N_COLS), dtype=dtype)
    expected = numpy_reference([src, idx, out])
    return [src, idx, out], expected


def numpy_reference(args: list[np.ndarray]) -> np.ndarray:
    src, idx, _out = args
    return src[:, idx.astype(np.int64), :].astype(src.dtype)
