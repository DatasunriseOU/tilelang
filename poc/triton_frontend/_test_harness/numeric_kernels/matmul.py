"""matmul: ``c = a @ b`` for square fp32 with M = N = K = 64.

Smallest matmul tile that still exercises ``tt.dot`` end-to-end. We use
BLOCK_M = BLOCK_N = BLOCK_K = 64 so the kernel runs a SINGLE block tile
on a single program -- no inter-program reduction, no boundary masking
on the K axis.

Source: synthetic, modeled on Triton's tutorial ``03-matrix-multiplication.py``
collapsed to the single-tile case.
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


M = N = K = 64
BLOCK_M = BLOCK_N = BLOCK_K = 64

LAUNCH_GRID: Tuple[int, ...] = (1, 1)
META_ARGS: dict = {
    "BLOCK_M": BLOCK_M,
    "BLOCK_N": BLOCK_N,
    "BLOCK_K": BLOCK_K,
}

ATOL = 1e-3   # matmul accumulation needs slightly looser tol
RTOL = 1e-2


if triton is not None:

    @triton.jit
    def _matmul_kernel(
        a_ptr,
        b_ptr,
        c_ptr,
        M,
        N,
        K,
        stride_am,
        stride_ak,
        stride_bk,
        stride_bn,
        stride_cm,
        stride_cn,
        BLOCK_M: "tl.constexpr",
        BLOCK_N: "tl.constexpr",
        BLOCK_K: "tl.constexpr",
    ):
        pid_m = tl.program_id(axis=0)
        pid_n = tl.program_id(axis=1)
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)
        a_ptrs = a_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
        b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)
        accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for _ in range(0, K, BLOCK_K):
            a = tl.load(a_ptrs)
            b = tl.load(b_ptrs)
            accumulator += tl.dot(a, b)
            a_ptrs += BLOCK_K * stride_ak
            b_ptrs += BLOCK_K * stride_bk
        c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
        tl.store(c_ptrs, accumulator)

    TRITON_KERNEL: Callable[..., Any] = _matmul_kernel
else:  # pragma: no cover -- triton missing
    TRITON_KERNEL = None  # type: ignore


def make_inputs(
    *, dtype: str = "float32", seed: int = 0
) -> Tuple[List[np.ndarray], np.ndarray]:
    rng = np.random.default_rng(seed)
    a = rng.standard_normal((M, K)).astype(dtype)
    b = rng.standard_normal((K, N)).astype(dtype)
    c = np.zeros((M, N), dtype=dtype)
    expected = (a @ b).astype(dtype)
    return [a, b, c], expected


def numpy_reference(args: List[np.ndarray]) -> np.ndarray:
    a, b, _c = args
    return (a @ b).astype(a.dtype)
