# ruff: noqa
# Kernel-level benchmark: fused sparse-MLA (fwd+bwd) vs O(n^2) dense MLA reference
# at local_gb10_quarter model shapes (seq=4096, H=28, head_dim=128 -> MLA latent
# D=512 + tail=64 = 576). gb10 / sm_121a. Real measured numbers, fail-loud.
import sys, os, json, gc, argparse
sys.path.insert(0, "/home/dave/source/tilelang/examples/deepseek_v32")
import torch
from sparse_mla_fwd import (sparse_mla_fwd_interface, ref_sparse_mla_fwd_interface,
                            sparse_mla_fwd)
from sparse_mla_bwd import sparse_mla_bwd, preprocess, bwd as build_bwd, postprocess
from tilelang.profiler import do_bench

DEV = "cuda"
DQK = 576      # MLA Q/K latent = dim(512) + tail(64)
DV = 512       # MLA value/out dim


def make_indices(B, S, SKV, topk, kv_group=1, in_bounds=True):
    """topk causal sparse indices [B,S,kv_group,topk]. in_bounds avoids the OOB
    sentinel so memory/latency aren't distorted by degenerate rows."""
    idx = torch.full((B, S, kv_group, topk), SKV, dtype=torch.int32, device=DEV)
    for b in range(B):
        for t in range(S):
            for g in range(kv_group):
                hi = max(1, min(t + 1, SKV))
                n = min(topk, hi)
                perm = torch.randperm(hi, device=DEV)[:n].to(torch.int32)
                idx[b, t, g, :n] = perm
                if in_bounds and n < topk:
                    # pad remaining slots with a valid in-bounds index (slot 0)
                    idx[b, t, g, n:] = perm[0] if n > 0 else 0
    return idx


def mem_reset():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()


def bench_fwd(B, S, SKV, H, topk, threads, reps, warmup):
    mem_reset()
    torch.manual_seed(0)
    q = torch.randn((B, S, H, DQK), dtype=torch.bfloat16, device=DEV)
    kv = torch.randn((B, SKV, 1, DQK), dtype=torch.bfloat16, device=DEV)
    idx = make_indices(B, S, SKV, topk)
    base_mem = torch.cuda.memory_allocated()

    # Build the kernel ONCE (static shapes baked) so the timed loop is kernel-only,
    # not recompilation. Mirrors run_regression_perf.
    dim, tail_dim = DV, DQK - DV
    kernel = sparse_mla_fwd(H, dim, tail_dim, topk, 1, None, True,
                            threads=threads, gb10=True,
                            static_shape=(B, S, SKV))

    def fn():
        return kernel(q, kv, idx)

    out, lse = fn()
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated()
    ms = do_bench(fn, rep=reps, warmup=warmup)
    # FLOPs: 2 GEMMs (QK^T over DQK, P@V over DV) per (query, head, topk-key)
    flops = B * S * H * topk * (DQK + DV) * 2
    tflops = flops / (ms * 1e-3) / 1e12
    tok_s = (B * S) / (ms * 1e-3)
    return dict(ms=ms, tflops=tflops, tok_s=tok_s,
                peak_gb=peak / 1e9, input_gb=base_mem / 1e9,
                out_nan=int(torch.isnan(out).sum()))


def bench_bwd(B, S, SKV, H, topk, reps, warmup):
    mem_reset()
    torch.manual_seed(0)
    q = torch.randn((B, S, H, DQK), dtype=torch.bfloat16, device=DEV)
    kv = torch.randn((B, SKV, 1, DQK), dtype=torch.bfloat16, device=DEV)
    do = torch.randn((B, S, H, DV), dtype=torch.bfloat16, device=DEV)
    idx = make_indices(B, S, SKV, topk)
    # need fwd out + lse for bwd
    fwd_kernel = sparse_mla_fwd(H, DV, DQK - DV, topk, 1, None, True,
                                threads=128 if H <= 32 else 256, gb10=True,
                                static_shape=(B, S, SKV))
    out, lse = fwd_kernel(q, kv, idx)
    torch.cuda.synchronize()

    # Build the 3 bwd kernels ONCE.
    D, D_tail = DV, DQK - DV
    pre_k = preprocess(B, S, H, D)
    bwd_k = build_bwd(B, S, SKV, H, D, D_tail, topk, 1, None, True, gb10=True)
    post_k = postprocess(B, SKV, D, D_tail, 1)
    base_mem = torch.cuda.memory_allocated()

    def fn():
        delta = pre_k(out, do)
        dkv = torch.zeros_like(kv, dtype=torch.float32)
        dq = bwd_k(q, kv, do, idx, lse, delta, dkv)
        dkv = post_k(dkv)
        return dq, dkv

    dq, dkv = fn()
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated()
    ms = do_bench(fn, rep=reps, warmup=warmup)
    # bwd ~ 5 matmul-equivalents per (query,head,topk) (dP, dQ, dKV, dP@V, P@dO)
    flops = B * S * H * topk * (3 * DQK + 2 * DV) * 2
    tflops = flops / (ms * 1e-3) / 1e12
    tok_s = (B * S) / (ms * 1e-3)
    return dict(ms=ms, tflops=tflops, tok_s=tok_s,
                peak_gb=peak / 1e9, input_gb=base_mem / 1e9,
                dq_nan=int(torch.isnan(dq).sum()), dkv_nan=int(torch.isnan(dkv).sum()))


def bench_ref_dense(B, S, SKV, H, topk, reps, warmup):
    """O(n^2) dense MLA reference: full S x SKV scores then sparse-mask+softmax+PV,
    the baseline the fused kernel replaces (materializes the full attention matrix)."""
    mem_reset()
    torch.manual_seed(0)
    q = torch.randn((B, S, H, DQK), dtype=torch.bfloat16, device=DEV).requires_grad_(True)
    kv = torch.randn((B, SKV, 1, DQK), dtype=torch.bfloat16, device=DEV).requires_grad_(True)
    idx = make_indices(B, S, SKV, topk)
    base_mem = torch.cuda.memory_allocated()

    def fn():
        return ref_sparse_mla_fwd_interface(q, kv, idx)

    try:
        o = fn()
        torch.cuda.synchronize()
    except torch.cuda.OutOfMemoryError as e:
        peak = torch.cuda.max_memory_allocated()
        return dict(oom=True, err=str(e)[:200], peak_gb_at_oom=peak / 1e9,
                    input_gb=base_mem / 1e9)
    peak = torch.cuda.max_memory_allocated()
    ms = do_bench(fn, rep=reps, warmup=warmup)
    flops = B * S * H * SKV * (DQK + DV) * 2  # dense: over ALL SKV keys
    tflops = flops / (ms * 1e-3) / 1e12
    tok_s = (B * S) / (ms * 1e-3)
    return dict(ms=ms, tflops=tflops, tok_s=tok_s, peak_gb=peak / 1e9,
                input_gb=base_mem / 1e9, out_nan=int(torch.isnan(o).sum()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--S", type=int, default=4096)
    ap.add_argument("--SKV", type=int, default=4096)
    ap.add_argument("--H", type=int, default=28)
    ap.add_argument("--threads", type=int, default=128)
    ap.add_argument("--topks", type=str, default="256,512,1024,2048")
    ap.add_argument("--reps", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--do_ref", action="store_true")
    ap.add_argument("--do_bwd", action="store_true")
    ap.add_argument("--out", type=str, default="/tmp/sparse_mla_bench_result.json")
    args = ap.parse_args()

    B = 1
    topks = [int(x) for x in args.topks.split(",")]
    cap = torch.cuda.get_device_capability()
    results = {"meta": dict(device=torch.cuda.get_device_name(0), cap=list(cap),
                            cuda=torch.version.cuda, S=args.S, SKV=args.SKV,
                            H=args.H, DQK=DQK, DV=DV, threads=args.threads,
                            reps=args.reps, warmup=args.warmup), "rows": []}
    print(f"# device={results['meta']['device']} cap={cap} cuda={torch.version.cuda}")
    print(f"# S={args.S} SKV={args.SKV} H={args.H} DQK={DQK} DV={DV} threads={args.threads}")

    for topk in topks:
        row = {"topk": topk}
        print(f"\n=== topk={topk} (S={args.S}, SKV={args.SKV}, H={args.H}) ===")
        f = bench_fwd(B, args.S, args.SKV, args.H, topk, args.threads, args.reps, args.warmup)
        row["fwd"] = f
        print(f"  FUSED FWD : {f['ms']:.3f} ms | {f['tflops']:.1f} TFLOP/s | "
              f"{f['tok_s']:.0f} tok/s | peak {f['peak_gb']:.3f} GB | nan={f['out_nan']}")
        if args.do_bwd:
            bwd_threads = 128 if args.H <= 32 else 256
            b = bench_bwd(B, args.S, args.SKV, args.H, topk, args.reps, args.warmup)
            row["bwd"] = b
            print(f"  FUSED BWD : {b['ms']:.3f} ms | {b['tflops']:.1f} TFLOP/s | "
                  f"{b['tok_s']:.0f} tok/s | peak {b['peak_gb']:.3f} GB | "
                  f"dq_nan={b['dq_nan']} dkv_nan={b['dkv_nan']}")
            comb = dict(ms=f["ms"] + b["ms"],
                        peak_gb=max(f["peak_gb"], b["peak_gb"]),
                        tok_s=(B * args.S) / ((f["ms"] + b["ms"]) * 1e-3))
            row["fwd_plus_bwd"] = comb
            print(f"  FWD+BWD   : {comb['ms']:.3f} ms | {comb['tok_s']:.0f} tok/s | "
                  f"peak {comb['peak_gb']:.3f} GB")
        if args.do_ref:
            r = bench_ref_dense(B, args.S, args.SKV, args.H, topk, max(5, args.reps // 5), 10)
            row["ref_dense"] = r
            if r.get("oom"):
                print(f"  REF DENSE : OOM at {r['peak_gb_at_oom']:.1f} GB ({r['err'][:80]})")
            else:
                print(f"  REF DENSE : {r['ms']:.3f} ms | {r['tflops']:.1f} TFLOP/s | "
                      f"{r['tok_s']:.0f} tok/s | peak {r['peak_gb']:.3f} GB | nan={r['out_nan']}")
                speedup = r["ms"] / f["ms"]
                memred = r["peak_gb"] / f["peak_gb"]
                row["speedup_fwd"] = speedup
                row["mem_reduction"] = memred
                print(f"  >>> SPEEDUP fwd = {speedup:.2f}x | MEM reduction = {memred:.2f}x")
        results["rows"].append(row)

    with open(args.out, "w") as fp:
        json.dump(results, fp, indent=2)
    print(f"\n# wrote {args.out}")


if __name__ == "__main__":
    main()
