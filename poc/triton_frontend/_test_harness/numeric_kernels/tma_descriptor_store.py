"""tma_descriptor_store: copy via TMA descriptor LOAD + TMA descriptor STORE.

Pairs ``tma_descriptor_copy.py`` (which only exercises ``descriptor.load``)
with the matching ``descriptor.store`` op so the frontend's
``tt.descriptor_store`` emitter is hit by a live capture rather than only
the hand-written canned TTIR fixture.

On NVIDIA hardware this round-trips through real TMA on both legs; on
Apple Metal / AMD it falls back to pointer-arith ``T.copy`` per RFC §5.4.
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
    def _tma_descriptor_store_roundtrip_kernel(
        base_ptr,
        out_ptr,
        M: tl.constexpr,
        N: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        src_desc = tl.make_tensor_descriptor(
            base_ptr,
            [M, N],
            [N, 1],
            [BLOCK_M, BLOCK_N],
        )
        dst_desc = tl.make_tensor_descriptor(
            out_ptr,
            [M, N],
            [N, 1],
            [BLOCK_M, BLOCK_N],
        )
        tile = src_desc.load([0, 0])
        dst_desc.store([0, 0], tile)

    TRITON_KERNEL: Callable[..., Any] = _tma_descriptor_store_roundtrip_kernel
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
