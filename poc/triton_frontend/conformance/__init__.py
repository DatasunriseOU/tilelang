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


def kernel_softmax(M: int = 128, N: int = 256, BLOCK_N: int = 256) -> Optional[Any]:
    """RFC 5.5 #2: row-wise softmax (one program per row).

    Loads a row, subtracts the row-max for numerical stability, takes
    exp, divides by the row-sum. Mirrors the Triton tutorial 02 layout
    but uses TileLang's high-level ``T.reduce_max`` / ``T.reduce_sum``.
    """
    T = _try_tilelang()
    if T is None:
        return None
    import tilelang

    @tilelang.jit
    @T.prim_func
    def softmax(
        X: T.Tensor((M, N), "float32"),
        Y: T.Tensor((M, N), "float32"),
    ):
        with T.Kernel(M, threads=BLOCK_N) as bx:
            row = T.alloc_fragment((BLOCK_N,), "float32")
            row_max = T.alloc_fragment((1,), "float32")
            row_sum = T.alloc_fragment((1,), "float32")
            for j in T.Parallel(BLOCK_N):
                row[j] = T.if_then_else(j < N, X[bx, j], T.float32(-3.4e38))
            T.reduce_max(row, row_max, dim=0, clear=True)
            for j in T.Parallel(BLOCK_N):
                row[j] = T.exp(row[j] - row_max[0])
            T.reduce_sum(row, row_sum, dim=0, clear=True)
            for j in T.Parallel(BLOCK_N):
                if j < N:
                    Y[bx, j] = row[j] / row_sum[0]

    return softmax


def kernel_matmul(
    M: int = 128, N: int = 128, K: int = 64,
    BLOCK_M: int = 64, BLOCK_N: int = 64, BLOCK_K: int = 32,
) -> Optional[Any]:
    """RFC 5.5 #3: tiled fp16 matmul -> fp32 accumulator.

    Each program owns a ``(BLOCK_M, BLOCK_N)`` output tile and walks the
    K dimension in ``BLOCK_K`` chunks via ``T.Pipelined`` for software
    pipelining. Mirrors the Triton tutorial 03 layout.
    """
    T = _try_tilelang()
    if T is None:
        return None
    import tilelang

    @tilelang.jit
    @T.prim_func
    def matmul(
        A: T.Tensor((M, K), "float16"),
        B: T.Tensor((K, N), "float16"),
        C: T.Tensor((M, N), "float32"),
    ):
        with T.Kernel(T.ceildiv(M, BLOCK_M), T.ceildiv(N, BLOCK_N), threads=128) as (bx, by):
            A_s = T.alloc_shared((BLOCK_M, BLOCK_K), "float16")
            B_s = T.alloc_shared((BLOCK_K, BLOCK_N), "float16")
            C_f = T.alloc_fragment((BLOCK_M, BLOCK_N), "float32")
            T.clear(C_f)
            for k in T.Pipelined(T.ceildiv(K, BLOCK_K), num_stages=2):
                T.copy(A[bx * BLOCK_M, k * BLOCK_K], A_s)
                T.copy(B[k * BLOCK_K, by * BLOCK_N], B_s)
                T.gemm(A_s, B_s, C_f)
            T.copy(C_f, C[bx * BLOCK_M, by * BLOCK_N])

    return matmul


def kernel_layer_norm(
    M: int = 128, N: int = 256, BLOCK_N: int = 256, eps: float = 1e-5,
) -> Optional[Any]:
    """RFC 5.5 #4: layer-norm — two-pass mean/variance (Triton tutorial 05).

    One program per row. Computes mean, then variance, then normalizes
    and applies optional gamma/beta. We use the simple two-pass form
    (not Welford) — sufficient for fp32 conformance.
    """
    T = _try_tilelang()
    if T is None:
        return None
    import tilelang

    @tilelang.jit
    @T.prim_func
    def layer_norm(
        X: T.Tensor((M, N), "float32"),
        Gamma: T.Tensor((N,), "float32"),
        Beta: T.Tensor((N,), "float32"),
        Y: T.Tensor((M, N), "float32"),
    ):
        with T.Kernel(M, threads=BLOCK_N) as bx:
            row = T.alloc_fragment((BLOCK_N,), "float32")
            mean = T.alloc_fragment((1,), "float32")
            var = T.alloc_fragment((1,), "float32")
            inv_n = T.float32(1.0) / T.float32(N)
            for j in T.Parallel(BLOCK_N):
                row[j] = T.if_then_else(j < N, X[bx, j], T.float32(0))
            T.reduce_sum(row, mean, dim=0, clear=True)
            mean[0] = mean[0] * inv_n
            sq = T.alloc_fragment((BLOCK_N,), "float32")
            for j in T.Parallel(BLOCK_N):
                d = row[j] - mean[0]
                sq[j] = T.if_then_else(j < N, d * d, T.float32(0))
            T.reduce_sum(sq, var, dim=0, clear=True)
            var[0] = var[0] * inv_n
            inv_std = T.float32(1.0) / T.sqrt(var[0] + T.float32(eps))
            for j in T.Parallel(BLOCK_N):
                if j < N:
                    Y[bx, j] = (row[j] - mean[0]) * inv_std * Gamma[j] + Beta[j]

    return layer_norm


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
