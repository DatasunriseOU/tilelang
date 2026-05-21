"""dot_reduce_atomic: dot + reduce_sum + atomic_add numeric smoke.

This locks the RFC section 5.5 Wave-2 surface into the live numeric ladder.
The launch intentionally uses a single program so the output can be zeroed
before the atomic add without relying on cross-program ordering.
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


M = N = K = 32
BLOCK_M = BLOCK_N = BLOCK_K = 32

LAUNCH_GRID: tuple[int, ...] = (1,)
META_ARGS: dict = {
    "BLOCK_M": BLOCK_M,
    "BLOCK_N": BLOCK_N,
    "BLOCK_K": BLOCK_K,
}
TTIR_SIGNATURE: dict = {
    "a_ptr": "*fp16",
    "b_ptr": "*fp16",
    "acc_ptr": "*fp32",
    "BLOCK_M": "constexpr",
    "BLOCK_N": "constexpr",
    "BLOCK_K": "constexpr",
}

ATOL = 2e-1
RTOL = 2e-2


if triton is not None:

    @triton.jit
    def _dot_reduce_atomic_kernel(
        a_ptr,
        b_ptr,
        acc_ptr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        offs_m = tl.arange(0, BLOCK_M)
        offs_n = tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)

        a = tl.load(a_ptr + offs_m[:, None] * BLOCK_K + offs_k[None, :])
        b = tl.load(b_ptr + offs_k[:, None] * BLOCK_N + offs_n[None, :])
        c = tl.dot(a, b)
        row_sum = tl.sum(c, axis=0)

        tl.store(acc_ptr + offs_m, tl.zeros([BLOCK_M], dtype=tl.float32))
        tl.atomic_add(acc_ptr + offs_m, row_sum)

    TRITON_KERNEL: Callable[..., Any] = _dot_reduce_atomic_kernel
else:  # pragma: no cover -- triton missing
    TRITON_KERNEL = None  # type: ignore


def make_inputs(
    *, dtype: str = "float16", seed: int = 0
) -> tuple[list[np.ndarray], np.ndarray]:
    rng = np.random.default_rng(seed)
    a = rng.standard_normal((M, K)).astype(dtype)
    b = rng.standard_normal((K, N)).astype(dtype)
    acc = np.zeros((M,), dtype=np.float32)
    expected = numpy_reference([a, b, acc])
    return [a, b, acc], expected


def numpy_reference(args: list[np.ndarray]) -> np.ndarray:
    a, b, _acc = args
    c = a.astype(np.float32) @ b.astype(np.float32)
    return c.sum(axis=1).astype(np.float32)
