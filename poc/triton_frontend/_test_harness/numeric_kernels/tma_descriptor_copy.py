"""tma_descriptor_copy: live descriptor-load fallback for RFC 5.4.

The RFC maps Hopper TMA descriptor transfers to native TMA on NVIDIA and to
plain pointer-arith ``T.copy`` elsewhere. This kernel exercises Triton's
portable tensor-descriptor TTIR surface without requiring Hopper hardware.
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


M = N = 16
BLOCK_M = M
BLOCK_N = N

LAUNCH_GRID: tuple[int, ...] = (1,)
META_ARGS: dict = {
    "M": M,
    "N": N,
    "BLOCK_M": BLOCK_M,
    "BLOCK_N": BLOCK_N,
}
TTIR_SIGNATURE: dict = {
    "base_ptr": "*fp32",
    "out_ptr": "*fp32",
    "M": "constexpr",
    "N": "constexpr",
    "BLOCK_M": "constexpr",
    "BLOCK_N": "constexpr",
}

ATOL = 0.0
RTOL = 0.0


if triton is not None:

    @triton.jit
    def _tma_descriptor_copy_kernel(
        base_ptr,
        out_ptr,
        M: tl.constexpr,
        N: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        desc = tl.make_tensor_descriptor(
            base_ptr,
            [M, N],
            [N, 1],
            [BLOCK_M, BLOCK_N],
        )
        tile = desc.load([0, 0])
        offs_m = tl.arange(0, BLOCK_M)
        offs_n = tl.arange(0, BLOCK_N)
        tl.store(out_ptr + offs_m[:, None] * BLOCK_N + offs_n[None, :], tile)

    TRITON_KERNEL: Callable[..., Any] = _tma_descriptor_copy_kernel
else:  # pragma: no cover -- triton missing
    TRITON_KERNEL = None  # type: ignore


def make_inputs(*, dtype: str = "float32", seed: int = 0) -> tuple[list[np.ndarray], np.ndarray]:
    rng = np.random.default_rng(seed)
    src = rng.standard_normal((M, N)).astype(dtype)
    out = np.zeros((M, N), dtype=dtype)
    expected = numpy_reference([src, out])
    return [src, out], expected


def numpy_reference(args: list[np.ndarray]) -> np.ndarray:
    src, _out = args
    return src.copy()
