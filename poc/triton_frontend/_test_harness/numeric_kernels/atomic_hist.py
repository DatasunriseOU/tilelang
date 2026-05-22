"""atomic_hist: integer histogram via ``tl.atomic_add``.

This is the live numeric counterpart for the reducer corpus' atomic histogram
row. The kernel uses a single program so the in-kernel zero fill and atomic
updates have deterministic ordering.
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


N = 128
NUM_BINS = 16
BLOCK_N = 128

LAUNCH_GRID: tuple[int, ...] = (1,)
META_ARGS: dict = {
    "N": N,
    "NUM_BINS": NUM_BINS,
    "BLOCK_N": BLOCK_N,
}
TTIR_SIGNATURE: dict = {
    "src_ptr": "*i32",
    "hist_ptr": "*i32",
    "N": "constexpr",
    "NUM_BINS": "constexpr",
    "BLOCK_N": "constexpr",
}

ATOL = 0.0
RTOL = 0.0


if triton is not None:

    @triton.jit
    def _atomic_hist_kernel(
        src_ptr,
        hist_ptr,
        N: tl.constexpr,
        NUM_BINS: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        bin_offsets = tl.arange(0, NUM_BINS)
        tl.store(hist_ptr + bin_offsets, tl.zeros((NUM_BINS,), dtype=tl.int32))

        pid = tl.program_id(axis=0)
        offsets = pid * BLOCK_N + tl.arange(0, BLOCK_N)
        mask = offsets < N
        values = tl.load(src_ptr + offsets, mask=mask, other=0)
        ones = tl.full((BLOCK_N,), 1, dtype=tl.int32)
        tl.atomic_add(hist_ptr + values, ones, sem="relaxed", mask=mask)

    TRITON_KERNEL: Callable[..., Any] = _atomic_hist_kernel
else:  # pragma: no cover -- triton missing
    TRITON_KERNEL = None  # type: ignore


def make_inputs(*, dtype: str = "int32", seed: int = 0) -> tuple[list[np.ndarray], np.ndarray]:
    rng = np.random.default_rng(seed)
    src = rng.integers(0, NUM_BINS, size=(N,), dtype=np.int32)
    hist = np.zeros((NUM_BINS,), dtype=np.int32)
    expected = numpy_reference([src, hist])
    return [src.astype(dtype, copy=False), hist], expected


def numpy_reference(args: list[np.ndarray]) -> np.ndarray:
    src, _hist = args
    return np.bincount(src.astype(np.int64), minlength=NUM_BINS)[:NUM_BINS].astype(np.int32)
