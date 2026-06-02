# ruff: noqa
import os
import torch
import tilelang
from tilelang import language as T
from utils import assert_tensors_similar


# ---------------------------------------------------------------------------
# Arch-aware shared-memory budgeting (GB10 / sm_121 re-tile).
#
# Consumer Blackwell (sm_120 / sm_121, e.g. GB10) caps *dynamic* shared memory
# at 99 KiB per block (101376 B), vs Hopper sm_90 (227 KiB) and B200 sm_100
# (228 KiB).  The deepseek_v32 sparse-MLA tile authored for Hopper allocates
# ~216 KiB of smem (Q/KV/O each 64 KiB at H_per_block=64,D=512) and, when
# double-buffered by num_stages=2, lowers to ~180-296 KiB -- which ptxas
# rejects on sm_121 ("too much shared data" / cuFuncSetAttribute reject).
#
# `_select_gb10` returns True when we should emit the GB10-fitting variant.
# It is driven by an explicit `gb10=` kwarg (preferred) or, when unset, auto-
# detected from the live device compute-capability.  The Hopper path is fully
# preserved: pass gb10=False (the default on non-sm_12x devices) to get the
# byte-identical original kernel.
# ---------------------------------------------------------------------------

# Compute-capability (major, minor) -> max dynamic smem (KiB).
_SMEM_CAP_KIB = {
    (7, 0): 96, (7, 5): 64, (8, 0): 163, (8, 6): 99, (8, 9): 99,
    (9, 0): 228, (10, 0): 228, (10, 1): 228, (12, 0): 99, (12, 1): 99,
}
# Caps where the 99 KiB-fitting (aggressive-merge + drop-O_shared) variant is required.
_GB10_CAPS = frozenset({(12, 0), (12, 1)})


def _detect_cap():
    """Return the live CUDA compute-capability (major, minor), or None."""
    env = os.getenv("TILELANG_COMPUTE_CAP")  # e.g. "12.1" for offline builds
    if env:
        try:
            major, minor = env.split(".")
            return (int(major), int(minor))
        except (ValueError, TypeError):
            pass
    try:
        if torch.cuda.is_available():
            return torch.cuda.get_device_capability()
    except Exception:
        pass
    return None


def _select_gb10(gb10):
    """Resolve the gb10 flag: explicit kwarg wins, else auto-detect sm_12x."""
    if gb10 is not None:
        return bool(gb10)
    cap = _detect_cap()
    return cap in _GB10_CAPS


def smem_cap_kib(cap=None):
    """Max dynamic smem (KiB) for a compute-cap; pessimistic 99 default."""
    if cap is None:
        cap = _detect_cap()
    return _SMEM_CAP_KIB.get(cap, 99)


def sparse_mla_fwd(
    heads,
    dim,
    tail_dim,
    topk,
    kv_group=1,
    sm_scale=None,
    is_causal=True,
    CP0=True,
    block_I=64,
    num_stages=2,
    threads=256,
    gb10=None,
    static_shape=None,
):
    """Build the sparse-MLA forward kernel.

    Arch-aware: when ``gb10`` resolves True (sm_120/sm_121, or explicit), emit
    the 99 KiB-fitting variant -- aggressive smem merge + TMA-lower disabled +
    ``O_shared`` dropped (store ``acc_o`` straight to ``Output``).  Otherwise
    emit the original Hopper kernel byte-for-byte.

    ``static_shape``: optional ``(batch, seq_len, seq_len_kv)`` tuple.  On the
    merge/upstream-codegen-reorg branch the tir->tirx migration regresses
    symbolic dynamic shapes (stmt_functor.cc:694 -> "is not a callable
    object"), so static shapes are required to lower there.  Pass concrete
    ints to bake them into the prim_func; leave None for the dynamic build.
    """
    use_gb10 = _select_gb10(gb10)

    if use_gb10:
        # MEASURED 99 KiB fit on real sm_121a ptxas (CUDA 13.3, GB10).
        # The aggressive-merge pass does NOT overlay the always-live Q_shared
        # (64 KiB at H_per_block=64,D=512) with anything, so the peak smem is
        # essentially the naive single-stage buffer sum.  Measured ptxas budget
        # by block_I at num_stages=1, drop-O, aggressive-merge:
        #   block_I=64 -> 152.0 KiB (overflow)   block_I=32 -> 112.0 KiB (overflow)
        #   block_I=16 ->  93.0 KiB (FITS, 0x173f0 <= 0x18c00)
        # and num_stages=2 double-buffers the KV/S tiles -> 225 KiB (overflow).
        # So the GB10 path forces block_I=16, num_stages=1 unless the caller
        # passed explicit non-default values (which we honor and let fail loud).
        if block_I == 64:
            block_I = 16
        if num_stages == 2:
            num_stages = 1

    pass_configs = {
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    }
    if use_gb10:
        # GB10 sm_121 99 KiB cap: overlay non-overlapping smem buffers and
        # disable the Hopper TMA lowering (sm_12x has no Hopper TMA path).
        pass_configs[tilelang.PassConfigKey.TL_ENABLE_AGGRESSIVE_SHARED_MEMORY_MERGE] = True
        pass_configs[tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER] = True

    # Apply the dynamically-built jit decorator to the (uncalled) lazy builder
    # -- it must RETURN a PrimFunc so tilelang.jit infers "lazy" mode.  Calling
    # the wrapped builder triggers compilation and returns the kernel object.
    jit_decorator = tilelang.jit(out_idx=[-2, -1], pass_configs=pass_configs)
    wrapped = jit_decorator(_build_sparse_mla_fwd)
    return wrapped(
        heads, dim, tail_dim, topk, kv_group, sm_scale, is_causal, CP0,
        block_I, num_stages, threads, use_gb10, static_shape,
    )


def _build_sparse_mla_fwd(
    heads,
    dim,
    tail_dim,
    topk,
    kv_group,
    sm_scale,
    is_causal,
    CP0,
    block_I,
    num_stages,
    threads,
    use_gb10,
    static_shape=None,
):
    assert dim == tilelang.math.next_power_of_2(dim), f"haven't check padding correctness yet, dim={dim}"
    assert tail_dim == tilelang.math.next_power_of_2(tail_dim), f"haven't check padding correctness yet, dim={tail_dim}"
    assert is_causal == True, "non-casual is not supported"
    assert topk % block_I == 0, "otherwise will load some index=0 thus causing wrong kv to be loaded"
    if sm_scale is None:
        sm_scale = (1.0 / (dim + tail_dim)) ** 0.5 * 1.44269504  # log2(e)
    else:
        sm_scale = sm_scale * 1.44269504  # log2(e)

    if static_shape is not None:
        batch, seq_len, seq_len_kv = static_shape
    else:
        batch = T.dynamic("batch")
        seq_len = T.dynamic("seq_len")
        seq_len_kv = T.dynamic("seq_len_kv")

    head_kv = heads // kv_group
    q_shape = [batch, seq_len, heads, dim + tail_dim]
    kv_shape = [batch, seq_len_kv, kv_group, dim + tail_dim]
    o_shape = [batch, seq_len, heads, dim]
    indices_shape = [batch, seq_len, kv_group, topk]
    lse_shape = [batch, seq_len, heads]
    indices_dtype = T.int32
    dtype = T.bfloat16
    accum_dtype = T.float32

    G = kv_group
    H = head_kv
    padded_H = max(tilelang.math.next_power_of_2(head_kv), 16)
    if padded_H != H:
        assert kv_group == 1, (
            "here we solve the H padding automatically, other wise you should handle Q copy and Output copy with your mask (when kv_group == 1, use g_i * padded_H:(g_i+1) * padded_H would be handled automatically)"
        )
    BI = block_I
    NI = tilelang.cdiv(topk, block_I)
    D = dim
    D_tail = tail_dim

    if head_kv > 64:
        assert head_kv % 64 == 0, "head_kv should be a multiple of 64"
        REPLICATE_H = head_kv // 64
    else:
        REPLICATE_H = 1

    H_per_block = padded_H if REPLICATE_H == 1 else 64

    @T.prim_func
    def main(
        Q: T.Tensor(q_shape, dtype),  # type: ignore
        KV: T.Tensor(kv_shape, dtype),  # type: ignore
        Indices: T.Tensor(indices_shape, indices_dtype),  # type: ignore
        Output: T.Tensor(o_shape, dtype),  # type: ignore
        Lse: T.Tensor(lse_shape, accum_dtype),  # type: ignore
    ):
        with T.Kernel(seq_len * REPLICATE_H, batch, kv_group, threads=threads) as (
            bx,
            by,
            bz,
        ):
            Q_shared = T.alloc_shared([H_per_block, D], dtype)
            Q_tail_shared = T.alloc_shared([H_per_block, D_tail], dtype)
            KV_shared = T.alloc_shared([BI, D], dtype)
            K_tail_shared = T.alloc_shared([BI, D_tail], dtype)
            # GB10: drop O_shared (64 KiB at H=64,D=512); store acc_o directly
            # to Output.  Hopper keeps the staged O_shared for vectorized store.
            if not use_gb10:
                O_shared = T.alloc_shared([H_per_block, D], dtype)
            Lse_shared = T.alloc_shared([H_per_block], accum_dtype)
            mask = T.alloc_fragment([BI], "bool")

            acc_o = T.alloc_fragment([H_per_block, D], accum_dtype)
            acc_s = T.alloc_fragment([H_per_block, BI], accum_dtype)
            S_shared = T.alloc_shared([H_per_block, BI], dtype)
            sumexp = T.alloc_fragment([H_per_block], accum_dtype)
            sumexp_i = T.alloc_fragment([H_per_block], accum_dtype)
            alpha = T.alloc_fragment([H_per_block], accum_dtype)
            m_i = T.alloc_fragment([H_per_block], accum_dtype)
            m_i_prev = T.alloc_fragment([H_per_block], accum_dtype)

            T.fill(acc_o, 0)
            T.fill(sumexp, 0)
            T.fill(m_i, -(2**30))  # avoid -inf - inf to cause nan

            b_i, g_i = by, bz
            s_i = bx if REPLICATE_H == 1 else (bx // REPLICATE_H)
            q_i = s_i
            max_kv_i = q_i

            H0 = g_i * padded_H + (0 if REPLICATE_H == 1 else (bx % REPLICATE_H) * 64)
            H1 = H0 + H_per_block

            T.copy(Q[b_i, s_i, H0:H1, :D], Q_shared)
            T.copy(Q[b_i, s_i, H0:H1, D:], Q_tail_shared)

            for i_i in T.Pipelined(NI, num_stages=num_stages):
                for bi_i in T.Parallel(BI):
                    mask[bi_i] = Indices[b_i, s_i, g_i, i_i * BI + bi_i] <= max_kv_i

                for bi_i, d_i in T.Parallel(BI, D):
                    KV_shared[bi_i, d_i] = KV[b_i, Indices[b_i, s_i, g_i, i_i * BI + bi_i], g_i, d_i]
                for bi_i, d_i in T.Parallel(BI, D_tail):
                    K_tail_shared[bi_i, d_i] = KV[b_i, Indices[b_i, s_i, g_i, i_i * BI + bi_i], g_i, D + d_i]

                for h_i, bi_i in T.Parallel(H_per_block, BI):
                    acc_s[h_i, bi_i] = T.if_then_else(mask[bi_i], 0, -T.infinity(acc_s.dtype))
                T.gemm(
                    Q_shared,
                    KV_shared,
                    acc_s,
                    transpose_B=True,
                    policy=T.GemmWarpPolicy.FullRow,
                )
                T.gemm(
                    Q_tail_shared,
                    K_tail_shared,
                    acc_s,
                    transpose_B=True,
                    policy=T.GemmWarpPolicy.FullRow,
                )
                T.copy(m_i, m_i_prev)
                T.reduce_max(acc_s, m_i, dim=1, clear=False)
                for h_i in T.Parallel(H_per_block):
                    m_i[h_i] = T.max(m_i[h_i], m_i_prev[h_i])
                for h_i in T.Parallel(H_per_block):
                    alpha[h_i] = T.exp2((m_i_prev[h_i] - m_i[h_i]) * sm_scale)
                for h_i, bi_i in T.Parallel(H_per_block, BI):
                    acc_s[h_i, bi_i] = T.exp2(acc_s[h_i, bi_i] * sm_scale - m_i[h_i] * sm_scale)
                T.reduce_sum(acc_s, sumexp_i, dim=1)
                for h_i in T.Parallel(H_per_block):
                    sumexp[h_i] = sumexp[h_i] * alpha[h_i] + sumexp_i[h_i]
                for h_i, d_i in T.Parallel(H_per_block, D):
                    acc_o[h_i, d_i] = acc_o[h_i, d_i] * alpha[h_i]

                T.copy(acc_s, S_shared)
                T.gemm(S_shared, KV_shared, acc_o, policy=T.GemmWarpPolicy.FullRow)

            # Rescale
            for h_i, d_i in T.Parallel(H_per_block, D):
                acc_o[h_i, d_i] /= sumexp[h_i]
            for h_i in T.Parallel(H_per_block):
                sumexp[h_i] = T.log2(sumexp[h_i]) + m_i[h_i] * sm_scale

            if use_gb10:
                # Direct store: acc_o -> Output (no O_shared round-trip).
                T.copy(acc_o, Output[b_i, s_i, H0:H1, :])
            else:
                T.copy(acc_o, O_shared)
                T.copy(O_shared, Output[b_i, s_i, H0:H1, :])
            T.copy(sumexp, Lse[b_i, s_i, H0:H1])

    return main


def sparse_mla_fwd_interface(q, kv, indices, sm_scale=None, return_p_sum: bool = False, d_v=512, block_I=64, num_stages=2, threads=256, gb10=None, static_shape=False):
    is_casual = True
    assert return_p_sum == False, "This kernel file is for fwd only"
    assert q.is_contiguous() and kv.is_contiguous() and indices.is_contiguous()
    batch, seq_len, heads, dim_plus_tail_dim = q.shape
    _, seq_len_kv, kv_group, _ = kv.shape

    assert dim_plus_tail_dim == 576, "you should assign dim otherwise"
    dim = d_v

    assert kv.shape[-1] == dim_plus_tail_dim
    tail_dim = dim_plus_tail_dim - dim
    assert kv.shape[0] == batch
    _, _, _, topk = indices.shape
    assert indices.shape == (batch, seq_len, kv_group, topk)

    # static_shape=True bakes concrete (batch, seq_len, seq_len_kv) into the
    # prim_func -- required on branches where tirx dynamic shapes regress.
    # On the GB10 (sm_12x) path the dynamic-symbolic build CRASHES the compiler
    # (LLVM APInt assertion "Value is not an N-bit signed value" -> core dump
    # during layout inference on merge/upstream-codegen-reorg), so we force
    # static shapes there. Callers that explicitly pass static_shape=True keep
    # it; the Hopper path (gb10 resolves False) keeps the dynamic build.
    use_static = bool(static_shape) or _select_gb10(gb10)
    ss = (batch, seq_len, seq_len_kv) if use_static else None
    kernel = sparse_mla_fwd(
        heads, dim, tail_dim, topk, kv_group, sm_scale, is_casual, block_I=block_I, num_stages=num_stages, threads=threads, gb10=gb10, static_shape=ss
    )
    out, lse = kernel(q, kv, indices)
    return out, lse


def ref_sparse_mla_fwd_interface(q, kv, indices, sm_scale=None, is_casual=True):
    q = q.float()
    kv = kv.float()
    indices = indices.transpose(1, 2)
    b, sq, h, dim_q = q.shape
    b, sk, g, _ = kv.shape

    assert kv.shape[-1] == 576, "you should assign dim otherwise"
    dim = 512
    k = kv
    v = kv[..., :dim]

    b, _, _, dim_v = v.shape
    g_index = g
    h_index = h // g
    compressed_casual_mask = torch.arange(0, sq, dtype=torch.int32, device="cuda").view(-1, 1) >= torch.arange(
        1 - 1, sk * 1, 1, dtype=torch.int32, device="cuda"
    ).view(1, -1)

    mask = q.new_zeros(b, g_index, sq, sk + 1, dtype=torch.bool).scatter(3, indices.long(), 1)
    mask = mask[..., :-1]
    mask = mask & compressed_casual_mask.view(1, 1, sq, sk)
    mask[:, :, : 1 - 1, 0] = True
    mask = mask.view(b, g_index, 1, sq, sk)

    q = q.view(b, sq, g, -1, dim_q)
    score = torch.einsum("bmghd,bngd->bghmn", q, k)
    sm_scale = dim_q**-0.5 if sm_scale is None else sm_scale
    score = score.masked_fill(~mask, float("-inf")).mul(sm_scale)
    p = score.softmax(dim=-1)
    p = p.view(b, g_index, h_index, -1, sq, sk)
    p = p.view(b, g, -1, sq, sk)
    o = torch.einsum("bghmn,bngd->bmghd", p.type(v.dtype), v)
    o = o.reshape(b, sq, h, dim_v)
    return o.to(torch.bfloat16)


def test_sparse_mla_fwd(
    B=1,
    S=4096,
    SKV=8192,
    H=128,
    HKV=1,
    DQK=576,
    DV=512,
    topk=2048,
    dtype=torch.bfloat16,
    check_correctness=True,
    block_I=64,
    num_stages=2,
    threads=256,
    gb10=None,
):
    torch.random.manual_seed(0)
    q = torch.randn((B, S, H, DQK), dtype=dtype, device="cuda").requires_grad_(True)
    kv = torch.randn((B, SKV, HKV, DQK), dtype=dtype, device="cuda").requires_grad_(True)

    indices = torch.full((B, S, HKV, topk), SKV, dtype=torch.int32, device="cuda")
    for b in range(B):
        for t in range(S):
            for h in range(HKV):
                i_i = torch.randperm(max(1, t))[:topk]
                indices[b, t, h, : len(i_i)] = i_i

    tl_out, tl_lse = sparse_mla_fwd_interface(q, kv, indices, block_I=block_I, num_stages=num_stages, threads=threads, gb10=gb10)

    if check_correctness:
        # otherwise may cause out of memory
        ref_out = ref_sparse_mla_fwd_interface(q, kv, indices)
        assert_tensors_similar(tl_out, ref_out, eps=1e-2, name="out")
        print("assert_tensors_similar passed")

    def fn():
        return sparse_mla_fwd_interface(q, kv, indices, block_I=block_I, num_stages=num_stages, threads=threads, gb10=gb10)

    from tilelang.profiler import do_bench

    ms = do_bench(
        fn,
        rep=100,
        warmup=250,
    )
    print(f"Average time: {ms:.3f} ms")
    print("fwd io bandwidth = ", (B * S * DQK * topk * 2) / (ms * 1e-3) / 1e12)
    print("fwd tflops = ", (B * S * (DQK + DV) * topk * 2 * H) / (ms * 1e-3) / 1e12)


def run_regression_perf(
    B=1, S=4096, SKV=8192, H=128, HKV=1, DQK=576, DV=512, topk=2048, dtype=torch.bfloat16, block_I=64, num_stages=2, threads=256, gb10=None
):
    torch.random.manual_seed(0)
    q = torch.randn((B, S, H, DQK), dtype=dtype, device="cuda").requires_grad_(True)
    kv = torch.randn((B, SKV, HKV, DQK), dtype=dtype, device="cuda").requires_grad_(True)

    indices = torch.full((B, S, HKV, topk), SKV, dtype=torch.int32, device="cuda")
    for b in range(B):
        for t in range(S):
            for h in range(HKV):
                i_i = torch.randperm(max(1, t))[:topk]
                indices[b, t, h, : len(i_i)] = i_i

    is_casual = True
    _, _, heads, dim_plus_tail_dim = q.shape
    _, _, kv_group, _ = kv.shape
    dim = 512
    tail_dim = dim_plus_tail_dim - dim
    _, _, _, topk = indices.shape
    kernel = sparse_mla_fwd(heads, dim, tail_dim, topk, kv_group, None, is_casual, block_I=block_I, num_stages=num_stages, threads=threads, gb10=gb10)

    def run_kernel_only():
        kernel(q, kv, indices)

    from tilelang.profiler import do_bench

    return do_bench(run_kernel_only, backend="cupti")


if __name__ == "__main__":
    test_sparse_mla_fwd(
        B=1,
        S=4096,
        SKV=4096,
        H=128,
        HKV=1,
        DQK=576,
        DV=512,
        topk=2048,
        dtype=torch.bfloat16,
        check_correctness=True,
        block_I=64,
        num_stages=2,
        threads=256,
    )
