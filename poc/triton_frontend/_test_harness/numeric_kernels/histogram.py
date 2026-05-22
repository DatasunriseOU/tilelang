"""histogram: Triton ``tl.histogram`` numeric smoke.

This extends the RFC section 5.5 ladder with an op that was previously
covered only by structural emitter tests.  The kernel uses a single program
and writes one count per bin so the result can be compared directly against
``numpy.bincount``.
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


N = 64
NUM_BINS = 16

LAUNCH_GRID: tuple[int, ...] = (1,)
META_ARGS: dict = {
    "N": N,
    "NUM_BINS": NUM_BINS,
}
TTIR_SIGNATURE: dict = {
    "src_ptr": "*i32",
    "out_ptr": "*i32",
    "N": "constexpr",
    "NUM_BINS": "constexpr",
}

ATOL = 0.0
RTOL = 0.0


if triton is not None:

    @triton.jit
    def _histogram_kernel(
        src_ptr,
        out_ptr,
        N: tl.constexpr,
        NUM_BINS: tl.constexpr,
    ):
        offsets = tl.arange(0, N)
        values = tl.load(src_ptr + offsets)
        counts = tl.histogram(values, NUM_BINS)
        bins = tl.arange(0, NUM_BINS)
        tl.store(out_ptr + bins, counts)

    TRITON_KERNEL: Callable[..., Any] = _histogram_kernel
else:  # pragma: no cover -- triton missing
    TRITON_KERNEL = None  # type: ignore


def make_inputs(
    *, dtype: str = "int32", seed: int = 0
) -> tuple[list[np.ndarray], np.ndarray]:
    rng = np.random.default_rng(seed)
    src = rng.integers(0, NUM_BINS, size=(N,), dtype=np.int32)
    out = np.zeros((NUM_BINS,), dtype=np.int32)
    expected = numpy_reference([src, out])
    return [src.astype(dtype, copy=False), out], expected


def numpy_reference(args: list[np.ndarray]) -> np.ndarray:
    src, _out = args
    return np.bincount(src.astype(np.int64), minlength=NUM_BINS)[:NUM_BINS].astype(
        np.int32
    )
