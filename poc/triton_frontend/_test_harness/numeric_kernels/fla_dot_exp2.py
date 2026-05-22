"""fla_dot_exp2: dot accumulator followed by an elementwise ``tl.exp2``.

This is the live numeric counterpart to the reducer corpus' FLA
``tt.dot`` + ``math.exp2`` motif. It keeps the matrix small and
deterministic while preserving the headline TTIR op cohort used by the
flash-linear-attention gated-delta kernels.
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


M = N = K = 16
BLOCK_M = M
BLOCK_N = N
BLOCK_K = K
SCALE = 0.03125

LAUNCH_GRID: tuple[int, ...] = (1,)
META_ARGS: dict = {
    "BLOCK_M": BLOCK_M,
    "BLOCK_N": BLOCK_N,
    "BLOCK_K": BLOCK_K,
}
TTIR_SIGNATURE: dict = {
    "a_ptr": "*fp32",
    "b_ptr": "*fp32",
    "out_ptr": "*fp32",
    "BLOCK_M": "constexpr",
    "BLOCK_N": "constexpr",
    "BLOCK_K": "constexpr",
}

ATOL = 1e-3
RTOL = 1e-3


if triton is not None:

    @triton.jit
    def _fla_dot_exp2_kernel(
        a_ptr,
        b_ptr,
        out_ptr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        offs_m = tl.arange(0, BLOCK_M)
        offs_n = tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)
        a = tl.load(a_ptr + offs_m[:, None] * BLOCK_K + offs_k[None, :])
        b = tl.load(b_ptr + offs_k[:, None] * BLOCK_N + offs_n[None, :])
        acc = tl.dot(a, b)
        out = tl.exp2(acc * 0.03125)
        tl.store(out_ptr + offs_m[:, None] * BLOCK_N + offs_n[None, :], out)

    TRITON_KERNEL: Callable[..., Any] = _fla_dot_exp2_kernel
else:  # pragma: no cover -- triton missing
    TRITON_KERNEL = None  # type: ignore


def make_inputs(*, dtype: str = "float32", seed: int = 0) -> tuple[list[np.ndarray], np.ndarray]:
    rng = np.random.default_rng(seed)
    a = (rng.standard_normal((M, K)) * 0.25).astype(dtype)
    b = (rng.standard_normal((K, N)) * 0.25).astype(dtype)
    out = np.zeros((M, N), dtype=dtype)
    expected = numpy_reference([a, b, out])
    return [a, b, out], expected


def numpy_reference(args: list[np.ndarray]) -> np.ndarray:
    a, b, _out = args
    return np.exp2((a @ b) * SCALE).astype(a.dtype)
