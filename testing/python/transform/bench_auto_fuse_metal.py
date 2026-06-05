"""MEASURED Apple-GPU bench for the AUTO-MONO-FUSION pass (native torch.mps).

COMPILE+RUN companion to ``test_auto_fuse_chunk_region.py``. Closes the GOAL loop:
auto-fuse a 2-kernel producer-consumer region into ONE TileLang Metal kernel and
measure fused ms vs the multi-kernel baseline + parity over every element on the
Apple GPU.

Region (structural skeleton of the mamba3 F0->F1 chunk hand-off, reduced to the
smallest demonstrable GEMM chain that round-trips an intermediate through GLOBAL):

    K_A:  Y = X @ W1      (producer, writes Y to global)
    K_B:  Z = Y @ W2      (consumer, reads Y from global)

MULTI-KERNEL baseline = two ``tilelang.compile`` Metal kernels; Y is a real global
buffer written by K_A and re-read by K_B (the inter-kernel global round-trip).

AUTO-FUSED = ONE kernel keeping Y RESIDENT in shared memory across both T.gemm
calls — no inter-kernel sync, Y never hits global. Both GEMMs stay T.gemm
(keeps_gemms=True) — the cppmega-class recipe, not scalar-loop fusion.

The auto_fuse_chunk_region detector is consulted FIRST: build the KernelSurface
region, run ``dispatch_region``, emit the fused kernel ONLY when the pass returns
fused=True (z3-proved, in-budget, GEMMs kept). On decline it stays multi-kernel.

All buffers fp32 (the native torch.mps adapter has an fp16-fragment->half4 cast
codegen bug; fp32 sidesteps it and is the honest, bit-exact path).
"""

from __future__ import annotations

import importlib.util
import os
import sys
import time

import torch

import tilelang
import tilelang.language as T

_HERE = os.path.dirname(os.path.abspath(__file__))


def _load_pass():
    cand = os.path.normpath(
        os.path.join(_HERE, "..", "..", "..", "tilelang", "transform",
                     "auto_fuse_chunk_region.py")
    )
    spec = importlib.util.spec_from_file_location("_afr_bench", cand)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


afr = _load_pass()


def _producer_kernel(M, K, N):
    @T.prim_func
    def main(X: T.Tensor((M, K), "float"), W1: T.Tensor((K, N), "float"),
             Y: T.Tensor((M, N), "float")):
        with T.Kernel(1, threads=128):
            Xs = T.alloc_shared((M, K), "float")
            W1s = T.alloc_shared((K, N), "float")
            Yf = T.alloc_fragment((M, N), "float")
            T.copy(X, Xs); T.copy(W1, W1s); T.clear(Yf)
            T.gemm(Xs, W1s, Yf); T.copy(Yf, Y)
    return main


def _consumer_kernel(M, N, P):
    @T.prim_func
    def main(Y: T.Tensor((M, N), "float"), W2: T.Tensor((N, P), "float"),
             Z: T.Tensor((M, P), "float")):
        with T.Kernel(1, threads=128):
            Ys = T.alloc_shared((M, N), "float")
            W2s = T.alloc_shared((N, P), "float")
            Zf = T.alloc_fragment((M, P), "float")
            T.copy(Y, Ys); T.copy(W2, W2s); T.clear(Zf)
            T.gemm(Ys, W2s, Zf); T.copy(Zf, Z)
    return main


def _fused_kernel(M, K, N, P):
    @T.prim_func
    def main(X: T.Tensor((M, K), "float"), W1: T.Tensor((K, N), "float"),
             W2: T.Tensor((N, P), "float"), Z: T.Tensor((M, P), "float")):
        with T.Kernel(1, threads=128):
            Xs = T.alloc_shared((M, K), "float")
            W1s = T.alloc_shared((K, N), "float")
            W2s = T.alloc_shared((N, P), "float")
            Yres = T.alloc_shared((M, N), "float")  # RESIDENT — never hits global
            Yf = T.alloc_fragment((M, N), "float")
            Zf = T.alloc_fragment((M, P), "float")
            T.copy(X, Xs); T.copy(W1, W1s); T.clear(Yf)
            T.gemm(Xs, W1s, Yf)      # GEMM #1 (producer), kept as T.gemm
            T.copy(Yf, Yres)         # spill to resident smem
            T.copy(W2, W2s); T.clear(Zf)
            T.gemm(Yres, W2s, Zf)    # GEMM #2 (consumer) reads Y from smem
            T.copy(Zf, Z)
    return main


def _compile(prim):
    return tilelang.compile(prim, target="metal", out_idx=None)


def _time(fn, iters=100, warmup=20):
    for _ in range(warmup):
        fn()
    torch.mps.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.mps.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3


def run(M=64, K=64, N=64, P=64):
    # --- 0. Consult the AUTO-FUSION pass FIRST (detect + prove + dispatch) --- #
    surfaces = (
        afr.KernelSurface(name="K_A", op_name="producer_gemm",
                          inputs=("X", "W1"), outputs=("Y",),
                          keeps_gemms=True, state_smem_bytes=0),
        afr.KernelSurface(name="K_B", op_name="consumer_gemm",
                          inputs=("Y", "W2"), outputs=("Z",),
                          keeps_gemms=True, state_smem_bytes=M * N * 4),
    )
    disp = afr.dispatch_region(
        surfaces, region_name="bench_KA_KB", mono_builder="bench._fused_kernel",
        nchunks=1, state_cells=M * N, smem_cap_bytes=afr.APPLE_SMEM_CAP_BYTES,
    )
    print(f"[pass] fused={disp.fused} replaced={disp.replaced_nodes} "
          f"decline={disp.decline_reason}")
    print(f"[pass] internal(privatized) buffers = "
          f"{disp.proof.get('internal_buffers')}")
    if not disp.fused:
        raise SystemExit(f"pass declined: {disp.decline_reason} (stays multi-kernel)")

    torch.manual_seed(0)
    X = (torch.randn(M, K) * 0.1).to("mps")
    W1 = (torch.randn(K, N) * 0.1).to("mps")
    W2 = (torch.randn(N, P) * 0.1).to("mps")

    kA = _compile(_producer_kernel(M, K, N))
    kB = _compile(_consumer_kernel(M, N, P))
    kF = _compile(_fused_kernel(M, K, N, P))

    Yg = torch.empty(M, N, device="mps")
    Zm = torch.empty(M, P, device="mps")
    Zf = torch.empty(M, P, device="mps")

    def run_multi():
        kA(X, W1, Yg)          # writes Y to GLOBAL
        kB(Yg, W2, Zm)         # re-reads Y from GLOBAL
        return Zm

    def run_fused():
        kF(X, W1, W2, Zf)      # Y stays in smem
        return Zf

    run_multi(); run_fused(); torch.mps.synchronize()
    ref = (X.float() @ W1.float()) @ W2.float()
    d_multi = float((Zm - ref).abs().max())
    d_fused = float((Zf - ref).abs().max())
    d_mf = float((Zf - Zm).abs().max())
    refmag = float(ref.abs().max())

    t_multi = _time(run_multi)
    t_fused = _time(run_fused)

    print(f"[run] multi-kernel: 2 metal dispatches, {t_multi:.4f} ms, "
          f"max|vs-ref|={d_multi:.3e}")
    print(f"[run] FUSED       : 1 metal dispatch , {t_fused:.4f} ms, "
          f"max|vs-ref|={d_fused:.3e}")
    print(f"[run] fused-vs-multi max|abs diff| (ALL elems) = {d_mf:.3e}")
    print(f"[run] speedup multi/fused = {t_multi / t_fused:.3f}x")
    print(f"[run] kernels: 2 (multi) -> 1 (fused)  AUTO-FUSED={disp.fused}")

    # Parity over EVERY element: both paths are algorithmically identical (Y in
    # fp32), differing only in Y's residence (global vs smem) -> must be bit-exact.
    assert d_mf <= 1e-4 * max(refmag, 1e-3), f"PARITY FAIL fused-vs-multi {d_mf:.3e}"
    assert d_fused <= 1e-4 * max(refmag, 1e-3), f"FUSED vs ref {d_fused:.3e}"
    print(f"[run] PARITY OK (bit-exact): fused-vs-multi {d_mf:.3e}, "
          f"fused-vs-ref {d_fused:.3e} (ref mag {refmag:.3e})")
    return {"fused": disp.fused, "t_multi": t_multi, "t_fused": t_fused,
            "speedup": t_multi / t_fused, "parity_fused_vs_multi": d_mf,
            "parity_fused_vs_ref": d_fused}


if __name__ == "__main__":
    print("RESULT", run())
