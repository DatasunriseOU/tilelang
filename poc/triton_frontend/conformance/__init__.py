"""Conformance suite for the Triton -> TileLang TIR frontend.

Implements the kernel list in ``RFC_unified_fused_kernel.md`` section 5.5
in ascending difficulty:

    1. vector_add         -- mask + program_id
    2. softmax            -- reduce + broadcast (TODO)
    3. matmul             -- dot + multi-stage load (TODO)
    4. layer_norm         -- Triton tutorial 05 Welford (TODO)
    5. fa_v2              -- Triton tutorial 06 (TODO)
    6. fa_v3              -- Hopper TMA + WGMMA (TODO)
    7. paged_attn         -- vLLM port (TODO)
    8. dot_reduce_atomic  -- dot + reduce + atomic_add (Wave-2 add)

Each ``kernel_<name>`` returns a ``@T.prim_func`` (or ``None`` if TileLang
isn't importable in the runtime environment, so callers can ``skipif``).
"""
from __future__ import annotations

from typing import Any, Callable, Dict, Optional

__all__ = [
    "KERNELS",
    "kernel_vector_add",
    "kernel_softmax",
    "kernel_matmul",
    "kernel_layer_norm",
    "kernel_fa_v2",
    "kernel_fa_v3",
    "kernel_paged_attn",
    "kernel_dot_reduce_atomic",
]


def _try_tilelang() -> Optional[Any]:
    """Return the ``tilelang.language`` module or None if unavailable."""
    try:
        import tilelang.language as T  # type: ignore
        return T
    except Exception:
        return None


def kernel_vector_add(N: int = 1024, BLOCK: int = 128) -> Optional[Any]:
    """RFC 5.5 #1: masked elementwise add over a 1D grid.

    Mirrors the canonical Triton tutorial 01 kernel: each program loads a
    BLOCK-sized slice of A and B, adds them, writes to Y, with a mask
    guarding the tail block when N % BLOCK != 0.
    """
    T = _try_tilelang()
    if T is None:
        return None
    import tilelang
    grid = (N + BLOCK - 1) // BLOCK

    @tilelang.jit
    @T.prim_func
    def vector_add(
        A: T.Tensor((N,), "float32"),
        B: T.Tensor((N,), "float32"),
        Y: T.Tensor((N,), "float32"),
    ):
        with T.Kernel(grid, threads=BLOCK) as bx:
            A_tile = T.alloc_fragment((BLOCK,), "float32")
            B_tile = T.alloc_fragment((BLOCK,), "float32")
            Y_tile = T.alloc_fragment((BLOCK,), "float32")
            base = bx * BLOCK
            for i in T.Parallel(BLOCK):
                idx = base + i
                # Mask the tail; out-of-range loads are zero by convention.
                A_tile[i] = T.if_then_else(idx < N, A[idx], T.float32(0))
                B_tile[i] = T.if_then_else(idx < N, B[idx], T.float32(0))
            for i in T.Parallel(BLOCK):
                Y_tile[i] = A_tile[i] + B_tile[i]
            for i in T.Parallel(BLOCK):
                idx = base + i
                if idx < N:
                    Y[idx] = Y_tile[i]

    return vector_add


def kernel_softmax() -> None:
    """RFC 5.5 #2: row-wise softmax. TODO: port from triton-shared examples."""
    pass


def kernel_matmul() -> None:
    """RFC 5.5 #3: tiled matmul with multi-stage software pipelining."""
    pass


def kernel_layer_norm() -> None:
    """RFC 5.5 #4: two-pass Welford layer-norm (Triton tutorial 05)."""
    pass


def kernel_fa_v2() -> None:
    """RFC 5.5 #5: FlashAttention-2 (pipelined dot + online softmax)."""
    pass


def kernel_fa_v3() -> None:
    """RFC 5.5 #6: FlashAttention-3 (Hopper TMA + WGMMA + WS)."""
    pass


def kernel_paged_attn() -> None:
    """RFC 5.5 #7: paged-attention, ported from vLLM."""
    pass


def kernel_dot_reduce_atomic(
    M: int = 64, N: int = 64, K: int = 64, BLOCK: int = 32
) -> Optional[Any]:
    """Wave-2 conformance: tile-grained dot + reduce_sum + atomic_add.

    Each block computes a partial M_tile x N_tile tile of A @ B, takes a
    column-wise reduce_sum into a vector, and atomically adds it into a
    shared accumulator buffer ``Acc`` (shape (N,)). This exercises three
    primitives that the frontend lowers via ``map_tt_dot`` /
    ``map_tt_reduce`` / ``map_tt_atomic_rmw``.
    """
    T = _try_tilelang()
    if T is None:
        return None
    import tilelang

    @tilelang.jit
    @T.prim_func
    def dot_reduce_atomic(
        A: T.Tensor((M, K), "float16"),
        B: T.Tensor((K, N), "float16"),
        Acc: T.Tensor((N,), "float32"),
    ):
        with T.Kernel(T.ceildiv(M, BLOCK), T.ceildiv(N, BLOCK), threads=128) as (bx, by):
            A_s = T.alloc_shared((BLOCK, K), "float16")
            B_s = T.alloc_shared((K, BLOCK), "float16")
            C_f = T.alloc_fragment((BLOCK, BLOCK), "float32")
            T.clear(C_f)
            T.copy(A[bx * BLOCK, 0], A_s)
            T.copy(B[0, by * BLOCK], B_s)
            T.gemm(A_s, B_s, C_f)

            col_sum = T.alloc_fragment((BLOCK,), "float32")
            T.reduce_sum(C_f, col_sum, dim=0, clear=True)

            for j in T.Parallel(BLOCK):
                T.atomic_add(Acc[by * BLOCK + j], col_sum[j])

    return dot_reduce_atomic


KERNELS: Dict[str, Callable[..., Optional[Any]]] = {
    "vector_add": kernel_vector_add,
    "softmax": kernel_softmax,
    "matmul": kernel_matmul,
    "layer_norm": kernel_layer_norm,
    "fa_v2": kernel_fa_v2,
    "fa_v3": kernel_fa_v3,
    "paged_attn": kernel_paged_attn,
    "dot_reduce_atomic": kernel_dot_reduce_atomic,
}
