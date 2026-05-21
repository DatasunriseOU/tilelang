"""Conformance suite for the Triton -> TileLang TIR frontend.

Implements the kernel list in ``RFC_unified_fused_kernel.md`` section 5.5
in ascending difficulty:

    1. vector_add         -- mask + program_id
    2. softmax            -- reduce + broadcast
    3. matmul             -- dot + multi-stage load
    4. layer_norm         -- Triton tutorial 05 Welford
    5. fa_v2              -- Triton tutorial 06 (wired up to numeric_kernels)
    6. fa_v3              -- Hopper TMA + WGMMA, with TileLang fallback
    7. paged_attn         -- vLLM-style block-table KV indirection
    8. dot_reduce_atomic  -- dot + reduce + atomic_add (Wave-2 add)

Each implemented ``kernel_<name>`` returns a compiled TileLang kernel (or
``None`` if TileLang isn't importable in the runtime environment, so callers
can ``skipif``).
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
    "kernel_dot_reduce_atomic_trans_b",
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

    return tilelang.compile(vector_add)


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

    return tilelang.compile(softmax)


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

    return tilelang.compile(matmul)


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

    return tilelang.compile(layer_norm)


def _build_fa_v2_prim(
    SEQ_Q: int = 128,
    SEQ_K: int = 128,
    HEAD_DIM: int = 32,
    BLOCK_M: int = 32,
    BLOCK_N: int = 32,
    BLOCK_DMODEL: int = 32,
    SM_SCALE: float = 0.17677669529663687,
    dtype: str = "float16",
    accum_dtype: str = "float32",
) -> Optional[Any]:
    """Build RFC 5.5 #5 TileLang IR for dense FA-v2 forward conformance.

    This mirrors the dense Triton target in
    ``_test_harness/numeric_kernels/flash_attention.py``: one query block
    streams over K/V blocks and applies the online softmax recurrence without
    materializing the full attention matrix.
    """
    T = _try_tilelang()
    if T is None:
        return None

    NUM_KV_BLOCKS = SEQ_K // BLOCK_N

    @T.prim_func
    def fa_v2(
        Q: T.Tensor((SEQ_Q, HEAD_DIM), dtype),
        K: T.Tensor((SEQ_K, HEAD_DIM), dtype),
        V: T.Tensor((SEQ_K, HEAD_DIM), dtype),
        O: T.Tensor((SEQ_Q, HEAD_DIM), accum_dtype),
    ):
        if False:  # noqa: SIM103
            _ = (
                SEQ_Q,
                SEQ_K,
                HEAD_DIM,
                NUM_KV_BLOCKS,
                dtype,
                accum_dtype,
            )
        with T.Kernel(T.ceildiv(SEQ_Q, BLOCK_M), threads=128) as bx:
            Q_shared = T.alloc_shared((BLOCK_M, BLOCK_DMODEL), dtype)
            K_shared = T.alloc_shared((BLOCK_N, BLOCK_DMODEL), dtype)
            V_shared = T.alloc_shared((BLOCK_N, BLOCK_DMODEL), dtype)
            Probs_shared = T.alloc_shared((BLOCK_M, BLOCK_N), dtype)
            Scores = T.alloc_shared((BLOCK_M, BLOCK_N), accum_dtype)
            Probs = T.alloc_fragment((BLOCK_M, BLOCK_N), accum_dtype)
            Probs_cast = T.alloc_fragment((BLOCK_M, BLOCK_N), dtype)
            Acc = T.alloc_shared((BLOCK_M, BLOCK_DMODEL), accum_dtype)
            row_max = T.alloc_fragment((BLOCK_M,), accum_dtype)
            row_max_prev = T.alloc_fragment((BLOCK_M,), accum_dtype)
            page_max = T.alloc_fragment((BLOCK_M,), accum_dtype)
            row_sum = T.alloc_fragment((BLOCK_M,), accum_dtype)
            page_sum = T.alloc_fragment((BLOCK_M,), accum_dtype)
            row_scale = T.alloc_fragment((BLOCK_M,), accum_dtype)

            T.copy(Q[bx * BLOCK_M, 0], Q_shared)
            T.clear(Acc)
            neg_max = T.cast(-3.4028234663852886e38, accum_dtype)
            T.fill(row_max, neg_max)
            T.clear(row_sum)

            for kv_block in T.serial(NUM_KV_BLOCKS):
                T.copy(K[kv_block * BLOCK_N, 0], K_shared)
                T.copy(V[kv_block * BLOCK_N, 0], V_shared)

                T.clear(Scores)
                T.gemm(
                    Q_shared,
                    K_shared,
                    Scores,
                    transpose_B=True,
                    policy=T.GemmWarpPolicy.FullRow,
                )
                for i, j in T.Parallel(BLOCK_M, BLOCK_N):
                    Scores[i, j] = Scores[i, j] * T.cast(SM_SCALE, accum_dtype)

                T.copy(row_max, row_max_prev)
                T.fill(page_max, neg_max)
                for i in T.Parallel(BLOCK_M):
                    for j in T.serial(BLOCK_N):
                        page_max[i] = T.max(page_max[i], Scores[i, j])

                for i in T.Parallel(BLOCK_M):
                    row_max[i] = T.max(row_max_prev[i], page_max[i])
                    row_scale[i] = T.exp(row_max_prev[i] - row_max[i])

                for i, j in T.Parallel(BLOCK_M, BLOCK_N):
                    Probs[i, j] = T.exp(Scores[i, j] - row_max[i])
                    Probs_cast[i, j] = T.cast(Probs[i, j], dtype)

                T.clear(page_sum)
                for i in T.Parallel(BLOCK_M):
                    for j in T.serial(BLOCK_N):
                        page_sum[i] = page_sum[i] + Probs[i, j]
                for i in T.Parallel(BLOCK_M):
                    row_sum[i] = row_sum[i] * row_scale[i] + page_sum[i]

                for i, j in T.Parallel(BLOCK_M, BLOCK_DMODEL):
                    Acc[i, j] = Acc[i, j] * row_scale[i]
                T.copy(Probs_cast, Probs_shared)
                T.gemm(Probs_shared, V_shared, Acc, policy=T.GemmWarpPolicy.FullRow)

            for i, j in T.Parallel(BLOCK_M, BLOCK_DMODEL):
                Acc[i, j] = Acc[i, j] / row_sum[i]
            T.copy(Acc, O[bx * BLOCK_M, 0])

    return fa_v2


def kernel_fa_v2() -> Optional[Any]:
    """RFC 5.5 #5: FlashAttention-2 (pipelined dot + online softmax)."""
    prim = _build_fa_v2_prim()
    if prim is None:
        return None
    import tilelang

    return tilelang.compile(prim)


def kernel_fa_v3(num_stages: int = 3) -> Optional[Any]:
    """RFC 5.5 #6: FlashAttention-3 (Hopper TMA + WGMMA + WS)."""
    prim = _build_fa_v3_tma_fallback_prim(NUM_STAGES=num_stages)
    if prim is None:
        return None
    import tilelang

    return tilelang.compile(prim)


def _build_fa_v3_tma_fallback_prim(
    SEQ_Q: int = 128,
    SEQ_K: int = 128,
    HEAD_DIM: int = 32,
    BLOCK_M: int = 32,
    BLOCK_N: int = 32,
    BLOCK_DMODEL: int = 32,
    NUM_STAGES: int = 3,
    SM_SCALE: float = 0.17677669529663687,
    dtype: str = "float16",
    accum_dtype: str = "float32",
) -> Optional[Any]:
    """Build RFC 5.5 #6 TileLang IR for the FA-v3 TMA fallback path.

    Hopper targets may lower the shared-memory copies and GEMMs to TMA/WGMMA.
    Non-NV targets keep the same fused attention contract with pipelined
    ``T.copy`` stages and target-native ``T.gemm`` lowering.
    """
    T = _try_tilelang()
    if T is None:
        return None

    NUM_KV_BLOCKS = SEQ_K // BLOCK_N

    @T.prim_func
    def fa_v3_tma_fallback(
        Q: T.Tensor((SEQ_Q, HEAD_DIM), dtype),
        K: T.Tensor((SEQ_K, HEAD_DIM), dtype),
        V: T.Tensor((SEQ_K, HEAD_DIM), dtype),
        O: T.Tensor((SEQ_Q, HEAD_DIM), accum_dtype),
    ):
        if False:  # noqa: SIM103
            _ = (
                SEQ_Q,
                SEQ_K,
                HEAD_DIM,
                NUM_KV_BLOCKS,
                NUM_STAGES,
                dtype,
                accum_dtype,
            )
        with T.Kernel(T.ceildiv(SEQ_Q, BLOCK_M), threads=128) as bx:
            Q_shared = T.alloc_shared((BLOCK_M, BLOCK_DMODEL), dtype)
            K_shared = T.alloc_shared((BLOCK_N, BLOCK_DMODEL), dtype)
            V_shared = T.alloc_shared((BLOCK_N, BLOCK_DMODEL), dtype)
            Probs_shared = T.alloc_shared((BLOCK_M, BLOCK_N), dtype)
            Scores = T.alloc_shared((BLOCK_M, BLOCK_N), accum_dtype)
            Probs = T.alloc_fragment((BLOCK_M, BLOCK_N), accum_dtype)
            Probs_cast = T.alloc_fragment((BLOCK_M, BLOCK_N), dtype)
            Acc = T.alloc_shared((BLOCK_M, BLOCK_DMODEL), accum_dtype)
            row_max = T.alloc_fragment((BLOCK_M,), accum_dtype)
            row_max_prev = T.alloc_fragment((BLOCK_M,), accum_dtype)
            page_max = T.alloc_fragment((BLOCK_M,), accum_dtype)
            row_sum = T.alloc_fragment((BLOCK_M,), accum_dtype)
            page_sum = T.alloc_fragment((BLOCK_M,), accum_dtype)
            row_scale = T.alloc_fragment((BLOCK_M,), accum_dtype)

            T.copy(Q[bx * BLOCK_M, 0], Q_shared)
            T.clear(Acc)
            neg_max = T.cast(-3.4028234663852886e38, accum_dtype)
            T.fill(row_max, neg_max)
            T.clear(row_sum)

            for kv_block in T.Pipelined(NUM_KV_BLOCKS, num_stages=NUM_STAGES):
                T.copy(K[kv_block * BLOCK_N, 0], K_shared)
                T.copy(V[kv_block * BLOCK_N, 0], V_shared)

                T.clear(Scores)
                T.gemm(
                    Q_shared,
                    K_shared,
                    Scores,
                    transpose_B=True,
                    policy=T.GemmWarpPolicy.FullRow,
                )
                for i, j in T.Parallel(BLOCK_M, BLOCK_N):
                    Scores[i, j] = Scores[i, j] * T.cast(SM_SCALE, accum_dtype)

                T.copy(row_max, row_max_prev)
                T.fill(page_max, neg_max)
                for i in T.Parallel(BLOCK_M):
                    for j in T.serial(BLOCK_N):
                        page_max[i] = T.max(page_max[i], Scores[i, j])

                for i in T.Parallel(BLOCK_M):
                    row_max[i] = T.max(row_max_prev[i], page_max[i])
                    row_scale[i] = T.exp(row_max_prev[i] - row_max[i])

                for i, j in T.Parallel(BLOCK_M, BLOCK_N):
                    Probs[i, j] = T.exp(Scores[i, j] - row_max[i])
                    Probs_cast[i, j] = T.cast(Probs[i, j], dtype)

                T.clear(page_sum)
                for i in T.Parallel(BLOCK_M):
                    for j in T.serial(BLOCK_N):
                        page_sum[i] = page_sum[i] + Probs[i, j]
                for i in T.Parallel(BLOCK_M):
                    row_sum[i] = row_sum[i] * row_scale[i] + page_sum[i]

                for i, j in T.Parallel(BLOCK_M, BLOCK_DMODEL):
                    Acc[i, j] = Acc[i, j] * row_scale[i]
                T.copy(Probs_cast, Probs_shared)
                T.gemm(Probs_shared, V_shared, Acc, policy=T.GemmWarpPolicy.FullRow)

            for i, j in T.Parallel(BLOCK_M, BLOCK_DMODEL):
                Acc[i, j] = Acc[i, j] / row_sum[i]
            T.copy(Acc, O[bx * BLOCK_M, 0])

    return fa_v3_tma_fallback


def _build_paged_attention_prim(
    SEQ_Q: int = 16,
    NUM_PAGES: int = 4,
    PAGE_SIZE: int = 16,
    HEAD_DIM: int = 32,
    NUM_PHYS_BLOCKS: int = 6,
    BLOCK_M: int = 16,
    BLOCK_N: int = 16,
    BLOCK_DMODEL: int = 32,
    SM_SCALE: float = 0.17677669529663687,
    dtype: str = "float16",
    accum_dtype: str = "float32",
) -> Optional[Any]:
    """Build RFC 5.5 #7 TileLang IR for paged-attention conformance.

    The kernel mirrors the Triton numeric target in
    ``_test_harness/numeric_kernels/paged_attention.py``: one query block
    gathers KV pages through ``BlockTable`` and runs the streaming softmax
    recurrence without materializing the full logical KV sequence.
    """
    T = _try_tilelang()
    if T is None:
        return None

    @T.prim_func
    def paged_attention(
        Q: T.Tensor((SEQ_Q, HEAD_DIM), dtype),
        KCache: T.Tensor((NUM_PHYS_BLOCKS, PAGE_SIZE, HEAD_DIM), dtype),
        VCache: T.Tensor((NUM_PHYS_BLOCKS, PAGE_SIZE, HEAD_DIM), dtype),
        BlockTable: T.Tensor((NUM_PAGES,), "int32"),
        O: T.Tensor((SEQ_Q, HEAD_DIM), accum_dtype),
    ):
        if False:  # noqa: SIM103
            _ = (
                SEQ_Q,
                NUM_PAGES,
                PAGE_SIZE,
                HEAD_DIM,
                NUM_PHYS_BLOCKS,
                dtype,
                accum_dtype,
            )
        with T.Kernel(T.ceildiv(SEQ_Q, BLOCK_M), threads=128) as bx:
            Q_shared = T.alloc_shared((BLOCK_M, BLOCK_DMODEL), dtype)
            K_shared = T.alloc_shared((BLOCK_N, BLOCK_DMODEL), dtype)
            V_shared = T.alloc_shared((BLOCK_N, BLOCK_DMODEL), dtype)
            Probs_shared = T.alloc_shared((BLOCK_M, BLOCK_N), dtype)
            Scores = T.alloc_shared((BLOCK_M, BLOCK_N), accum_dtype)
            Probs = T.alloc_fragment((BLOCK_M, BLOCK_N), accum_dtype)
            Probs_cast = T.alloc_fragment((BLOCK_M, BLOCK_N), dtype)
            Acc = T.alloc_shared((BLOCK_M, BLOCK_DMODEL), accum_dtype)
            row_max = T.alloc_fragment((BLOCK_M,), accum_dtype)
            row_max_prev = T.alloc_fragment((BLOCK_M,), accum_dtype)
            page_max = T.alloc_fragment((BLOCK_M,), accum_dtype)
            row_sum = T.alloc_fragment((BLOCK_M,), accum_dtype)
            page_sum = T.alloc_fragment((BLOCK_M,), accum_dtype)
            row_scale = T.alloc_fragment((BLOCK_M,), accum_dtype)

            T.copy(Q[bx * BLOCK_M, 0], Q_shared)
            T.clear(Acc)
            neg_max = T.cast(-3.4028234663852886e38, accum_dtype)
            T.fill(row_max, neg_max)
            T.clear(row_sum)

            for logical_page in T.serial(NUM_PAGES):
                physical_page = BlockTable[logical_page]
                T.copy(KCache[physical_page, 0, 0], K_shared)
                T.copy(VCache[physical_page, 0, 0], V_shared)

                T.clear(Scores)
                T.gemm(
                    Q_shared,
                    K_shared,
                    Scores,
                    transpose_B=True,
                    policy=T.GemmWarpPolicy.FullRow,
                )
                for i, j in T.Parallel(BLOCK_M, BLOCK_N):
                    Scores[i, j] = Scores[i, j] * T.cast(SM_SCALE, accum_dtype)

                T.copy(row_max, row_max_prev)
                T.fill(page_max, neg_max)
                for i in T.Parallel(BLOCK_M):
                    for j in T.serial(BLOCK_N):
                        page_max[i] = T.max(page_max[i], Scores[i, j])

                for i in T.Parallel(BLOCK_M):
                    row_max[i] = T.max(row_max_prev[i], page_max[i])
                    row_scale[i] = T.exp(row_max_prev[i] - row_max[i])

                for i, j in T.Parallel(BLOCK_M, BLOCK_N):
                    Probs[i, j] = T.exp(Scores[i, j] - row_max[i])
                    Probs_cast[i, j] = T.cast(Probs[i, j], dtype)

                T.clear(page_sum)
                for i in T.Parallel(BLOCK_M):
                    for j in T.serial(BLOCK_N):
                        page_sum[i] = page_sum[i] + Probs[i, j]
                for i in T.Parallel(BLOCK_M):
                    row_sum[i] = row_sum[i] * row_scale[i] + page_sum[i]

                for i, j in T.Parallel(BLOCK_M, BLOCK_DMODEL):
                    Acc[i, j] = Acc[i, j] * row_scale[i]
                T.copy(Probs_cast, Probs_shared)
                T.gemm(Probs_shared, V_shared, Acc, policy=T.GemmWarpPolicy.FullRow)

            for i, j in T.Parallel(BLOCK_M, BLOCK_DMODEL):
                Acc[i, j] = Acc[i, j] / row_sum[i]
            T.copy(Acc, O[bx * BLOCK_M, 0])

    return paged_attention


def kernel_paged_attn() -> Optional[Any]:
    """RFC 5.5 #7: paged-attention, ported from vLLM-style KV caches."""
    prim = _build_paged_attention_prim()
    if prim is None:
        return None
    import tilelang

    return tilelang.compile(prim)


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

    return tilelang.compile(dot_reduce_atomic)


def kernel_dot_reduce_atomic_trans_b(
    M: int = 64, N: int = 64, K: int = 64, BLOCK: int = 32
) -> Optional[Any]:
    """Phase-1 migration: ``tl.dot(A, B, trans_b=True)`` + reduce_sum + atomic_add.

    Mirrors ``kernel_dot_reduce_atomic`` but takes B in ``(N, K)`` layout
    and asks ``T.gemm`` to transpose B at MMA time. Locks the
    ``map_tt_dot`` + ``map_tt_trans`` fold path used by cppmega.mlx
    kernels (``dsa_splitk_indexer_loss``, ``sparse_mla_path_c``).
    """
    T = _try_tilelang()
    if T is None:
        return None
    import tilelang

    @T.prim_func
    def dot_reduce_atomic_trans_b(
        A: T.Tensor((M, K), "float16"),
        B: T.Tensor((N, K), "float16"),  # transposed-B layout
        Acc: T.Tensor((N,), "float32"),
    ):
        with T.Kernel(T.ceildiv(M, BLOCK), T.ceildiv(N, BLOCK), threads=128) as (bx, by):
            A_s = T.alloc_shared((BLOCK, K), "float16")
            B_s = T.alloc_shared((BLOCK, K), "float16")
            C_f = T.alloc_fragment((BLOCK, BLOCK), "float32")
            T.clear(C_f)
            T.copy(A[bx * BLOCK, 0], A_s)
            T.copy(B[by * BLOCK, 0], B_s)
            T.gemm(A_s, B_s, C_f, transpose_B=True)

            col_sum = T.alloc_fragment((BLOCK,), "float32")
            T.reduce_sum(C_f, col_sum, dim=0, clear=True)

            for j in T.Parallel(BLOCK):
                T.atomic_add(Acc[by * BLOCK + j], col_sum[j])

    return tilelang.compile(dot_reduce_atomic_trans_b)


KERNELS: Dict[str, Callable[..., Optional[Any]]] = {
    "vector_add": kernel_vector_add,
    "softmax": kernel_softmax,
    "matmul": kernel_matmul,
    "layer_norm": kernel_layer_norm,
    "fa_v2": kernel_fa_v2,
    "fa_v3": kernel_fa_v3,
    "paged_attn": kernel_paged_attn,
    "dot_reduce_atomic": kernel_dot_reduce_atomic,
    "dot_reduce_atomic_trans_b": kernel_dot_reduce_atomic_trans_b,
}
