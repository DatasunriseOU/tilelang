# ruff: noqa
# Minimal repro: gather + online-softmax rescale interaction on sm_121.
# Tests the hypothesis that the indexed KV gather is miscompiled ONLY when
# combined with the multi-iteration online-rescale loop.
#
# Two kernels, identical EXCEPT the KV load:
#   - "copy"   : contiguous T.copy(KV[b, i0:i1, ...], KV_shared)
#   - "gather" : KV_shared[bi,d] = KV[b, Idx[bi], ...]  (identity indices)
# With identity indices the two MUST produce identical output.
import os, sys, math
import torch
import tilelang
from tilelang import language as T

torch.manual_seed(0)

def build(mode, B, S, SKV, H, D, BI, NI, num_stages, threads, barrier=False, D_tail=0, masked_init=False):
    dtype = T.bfloat16
    accum = T.float32
    DA = D + D_tail
    q_shape = [B, S, H, DA]
    kv_shape = [B, SKV, 1, DA]
    idx_shape = [B, S, 1, BI * NI]
    o_shape = [B, S, H, D]
    sm_scale = (1.0 / DA) ** 0.5 * 1.44269504

    @T.prim_func
    def main(
        Q: T.Tensor(q_shape, dtype),
        KV: T.Tensor(kv_shape, dtype),
        Idx: T.Tensor(idx_shape, T.int32),
        Output: T.Tensor(o_shape, dtype),
    ):
        with T.Kernel(S, B, 1, threads=threads) as (bx, by, bz):
            Q_shared = T.alloc_shared([H, D], dtype)
            KV_shared = T.alloc_shared([BI, D], dtype)
            S_shared = T.alloc_shared([H, BI], dtype)
            if D_tail:
                Q_tail_shared = T.alloc_shared([H, D_tail], dtype)
                K_tail_shared = T.alloc_shared([BI, D_tail], dtype)
            if masked_init == 5:
                mask = T.alloc_shared([BI], "bool", scope="shared")
            else:
                mask = T.alloc_fragment([BI], "bool")
            mask2 = T.alloc_fragment([H, BI], "bool")  # 2D mask for candidate B
            acc_o = T.alloc_fragment([H, D], accum)
            acc_s = T.alloc_fragment([H, BI], accum)
            sumexp = T.alloc_fragment([H], accum)
            sumexp_i = T.alloc_fragment([H], accum)
            alpha = T.alloc_fragment([H], accum)
            m_i = T.alloc_fragment([H], accum)
            m_i_prev = T.alloc_fragment([H], accum)

            T.fill(acc_o, 0)
            T.fill(sumexp, 0)
            T.fill(m_i, -(2**30))

            b_i, s_i = by, bx
            T.copy(Q[b_i, s_i, :, :D], Q_shared)
            if D_tail:
                T.copy(Q[b_i, s_i, :, D:], Q_tail_shared)

            for i_i in T.Pipelined(NI, num_stages=num_stages):
                for bi_i in T.Parallel(BI):
                    mask[bi_i] = Idx[b_i, s_i, 0, i_i * BI + bi_i] >= 0
                if masked_init == 6:
                    # FIX candidate B: build a [H,BI] mask matching acc_s's rank/layout
                    for h_i, bi_i in T.Parallel(H, BI):
                        mask2[h_i, bi_i] = Idx[b_i, s_i, 0, i_i * BI + bi_i] >= 0
                if mode == "copy":
                    T.copy(KV[b_i, i_i * BI:(i_i + 1) * BI, 0, :D], KV_shared)
                    if D_tail:
                        T.copy(KV[b_i, i_i * BI:(i_i + 1) * BI, 0, D:], K_tail_shared)
                else:
                    for bi_i, d_i in T.Parallel(BI, D):
                        KV_shared[bi_i, d_i] = KV[b_i, Idx[b_i, s_i, 0, i_i * BI + bi_i], 0, d_i]
                    if D_tail:
                        for bi_i, d_i in T.Parallel(BI, D_tail):
                            K_tail_shared[bi_i, d_i] = KV[b_i, Idx[b_i, s_i, 0, i_i * BI + bi_i], 0, D + d_i]
                if barrier:
                    T.tvm_storage_sync("shared")

                if masked_init == 1:
                    # real pattern: read [BI] mask fragment inside [H,BI] parallel init
                    for h_i, bi_i in T.Parallel(H, BI):
                        acc_s[h_i, bi_i] = T.if_then_else(mask[bi_i], 0, -T.infinity(acc_s.dtype))
                elif masked_init == 2:
                    # const if_then_else, NO mask fragment read (isolates the mask layout)
                    for h_i, bi_i in T.Parallel(H, BI):
                        acc_s[h_i, bi_i] = T.if_then_else(bi_i >= 0, 0.0, -T.infinity(acc_s.dtype))
                elif masked_init == 3:
                    # plain 0 init via Parallel (isolates Parallel-init vs T.clear)
                    for h_i, bi_i in T.Parallel(H, BI):
                        acc_s[h_i, bi_i] = T.Cast(accum, 0)
                elif masked_init == 4:
                    # FIX candidate A: clear, GEMM, THEN apply mask.
                    T.clear(acc_s)
                elif masked_init == 5:
                    # FIX candidate C: mask in SHARED memory (no fragment layout to infer)
                    for h_i, bi_i in T.Parallel(H, BI):
                        acc_s[h_i, bi_i] = T.if_then_else(mask[bi_i], 0, -T.infinity(acc_s.dtype))
                elif masked_init == 6:
                    # FIX candidate B: read the [H,BI] mask2 (matches acc_s rank)
                    for h_i, bi_i in T.Parallel(H, BI):
                        acc_s[h_i, bi_i] = T.if_then_else(mask2[h_i, bi_i], 0, -T.infinity(acc_s.dtype))
                else:
                    T.clear(acc_s)
                T.gemm(Q_shared, KV_shared, acc_s, transpose_B=True,
                       policy=T.GemmWarpPolicy.FullRow)
                if D_tail:
                    T.gemm(Q_tail_shared, K_tail_shared, acc_s, transpose_B=True,
                           policy=T.GemmWarpPolicy.FullRow)
                if masked_init == 4:
                    for h_i, bi_i in T.Parallel(H, BI):
                        acc_s[h_i, bi_i] = T.if_then_else(mask[bi_i], acc_s[h_i, bi_i], -T.infinity(acc_s.dtype))

                T.copy(m_i, m_i_prev)
                T.reduce_max(acc_s, m_i, dim=1, clear=False)
                for h_i in T.Parallel(H):
                    m_i[h_i] = T.max(m_i[h_i], m_i_prev[h_i])
                for h_i in T.Parallel(H):
                    alpha[h_i] = T.exp2((m_i_prev[h_i] - m_i[h_i]) * sm_scale)
                for h_i, bi_i in T.Parallel(H, BI):
                    acc_s[h_i, bi_i] = T.exp2(acc_s[h_i, bi_i] * sm_scale - m_i[h_i] * sm_scale)
                T.reduce_sum(acc_s, sumexp_i, dim=1)
                for h_i in T.Parallel(H):
                    sumexp[h_i] = sumexp[h_i] * alpha[h_i] + sumexp_i[h_i]
                for h_i, d_i in T.Parallel(H, D):
                    acc_o[h_i, d_i] = acc_o[h_i, d_i] * alpha[h_i]

                T.copy(acc_s, S_shared)
                T.gemm(S_shared, KV_shared, acc_o, policy=T.GemmWarpPolicy.FullRow)

            for h_i, d_i in T.Parallel(H, D):
                acc_o[h_i, d_i] /= sumexp[h_i]
            T.copy(acc_o, Output[b_i, s_i, :, :])

    return main


def ref(q, kv, idx, D):
    # idx: [B,S,1,T] identity over first T kv rows. q/kv last dim = D+D_tail.
    q = q.float(); kv = kv.float()
    B, S, H, DA = q.shape
    T_ = idx.shape[-1]
    k = kv[:, :T_, 0, :]   # [B,T,DA]
    v = k[..., :D]         # value uses first D dims
    score = torch.einsum("bshd,btd->bsht", q, k) * ((1.0/DA)**0.5)
    p = score.softmax(dim=-1)
    o = torch.einsum("bsht,btd->bshd", p, v)
    return o.to(torch.bfloat16)


def cos(a, b):
    a = a.float().flatten(); b = b.float().flatten()
    return (a @ b / (a.norm() * b.norm() + 1e-12)).item()


def run(mode, B=1, S=8, SKV=128, H=16, D=64, BI=16, NI=4, num_stages=1, threads=64,
        barrier=False, D_tail=0, masked_init=False, merge=False):
    T_ = BI * NI
    DA = D + D_tail
    q = torch.randn(B, S, H, DA, dtype=torch.bfloat16, device="cuda")
    kv = torch.randn(B, SKV, 1, DA, dtype=torch.bfloat16, device="cuda")
    idx = torch.arange(T_, dtype=torch.int32, device="cuda").view(1,1,1,T_).expand(B,S,1,T_).contiguous()
    pc = {
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
    }
    if merge:
        pc[tilelang.PassConfigKey.TL_ENABLE_AGGRESSIVE_SHARED_MEMORY_MERGE] = True
    kern = tilelang.compile(build(mode, B,S,SKV,H,D,BI,NI,num_stages,threads,barrier,D_tail,masked_init),
                            out_idx=[-1], pass_configs=pc)
    out = kern(q, kv, idx)
    rf = ref(q, kv, idx, D)
    c = cos(out, rf)
    print(f"mode={mode:7s} ns={num_stages} bar={int(barrier)} NI={NI} BI={BI} H={H} D={D} Dt={D_tail} thr={threads} mask={int(masked_init)} mrg={int(merge)} -> cos={c:.6f}")
    return c, kern


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("copy", "all"):
        run("copy")
    if which in ("gather", "all"):
        run("gather")
    if which == "real":
        # Mirror the real sparse-MLA tile: D=512, D_tail=64, H=64, threads=256,
        # masked-init, tail-split second GEMM, BI=16 num_stages=1.
        for md in ("copy", "gather"):
            run(md, B=1, S=8, SKV=128, H=64, D=512, D_tail=64, BI=16, NI=4,
                num_stages=1, threads=256, masked_init=True)
    if which == "real2":
        # Same but num_stages=2 and with aggressive merge (the GB10 recipe minus drop-O).
        for md in ("copy", "gather"):
            run(md, B=1, S=8, SKV=128, H=64, D=512, D_tail=64, BI=16, NI=4,
                num_stages=2, threads=256, masked_init=True, merge=True)
    if which == "ablate":
        # Bisect what triggers it: toggle each real-kernel feature on top of the clean repro.
        base = dict(B=1, S=8, SKV=128, H=64, D=512, BI=16, NI=4, num_stages=1, threads=256)
        print("# baseline (no tail, no mask):")
        run("gather", **base)
        print("# +D_tail=64:")
        run("gather", D_tail=64, **base)
        print("# +masked_init:")
        run("gather", masked_init=True, **base)
        print("# +D_tail +masked_init (== real):")
        run("gather", D_tail=64, masked_init=True, **base)
        print("# +D_tail +masked_init +merge:")
        run("gather", D_tail=64, masked_init=True, merge=True, **base)
        print("# masked_init=2 (const if_then_else, NO mask read):")
        run("gather", masked_init=2, **base)
        print("# masked_init=3 (Parallel zero-init, no if_then_else):")
        run("gather", masked_init=3, **base)
    if which == "fix":
        base = dict(B=1, S=8, SKV=128, H=64, D=512, D_tail=64, BI=16, NI=4, num_stages=1, threads=256)
        print("# baseline broken (masked_init=1):")
        run("gather", masked_init=1, **base)
        print("# candidate A=4 (mask applied post-GEMM):")
        run("gather", masked_init=4, **base)
        print("# candidate C=5 (mask in shared):")
        run("gather", masked_init=5, **base)
        print("# candidate B=6 (2D [H,BI] mask fragment):")
        run("gather", masked_init=6, **base)
    if which == "matrix":
        for ns in (1, 2):
            for ni in (1, 2, 4):
                run("copy", num_stages=ns, NI=ni)
                run("gather", num_stages=ns, NI=ni)
        print("--- with barrier ---")
        run("gather", num_stages=1, NI=4, barrier=True)
        run("gather", num_stages=2, NI=4, barrier=True)
    if which == "dump":
        _, k = run("gather", num_stages=1, NI=4)
        src = k.get_kernel_source()
        open("/tmp/gather_src.cu","w").write(src)
        _, k2 = run("copy", num_stages=1, NI=4)
        open("/tmp/copy_src.cu","w").write(k2.get_kernel_source())
        print("dumped /tmp/gather_src.cu /tmp/copy_src.cu")
