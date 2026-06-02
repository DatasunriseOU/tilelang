# ruff: noqa
# Regression guard for the fwd inf bug (merge-pass reduce-scratch aliasing onto
# S_shared -> inf in Output cols 256-511 on partial-softmax rows).
# Runs the OFFICIAL assert_tensors_similar over ALL rows with the gb10 fwd kernel
# built WITHOUT the aggressive-shared-memory-merge pass (merge=False, the shipped
# gb10 config) vs WITH it (merge=True, the old broken control). Confirms the
# merge=False build: (a) launches on real ptxas (fits 99 KiB), (b) out has no
# inf/nan, (c) full-tensor parity passes; and that merge=True reproduces the bug.
#   python test_mergeoff_official.py small   # S=512/topk=256: merge OFF pass, ON fail
#   python test_mergeoff_official.py big     # S=4096/topk=2048: merge OFF pass
#   python test_mergeoff_official.py both     # merge OFF at both shapes
import os, sys
import torch
import tilelang

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sparse_mla_fwd as M
from utils import assert_tensors_similar


def build_kernel(merge, S, SKV, H, topk):
    # Build the REAL gb10 builder (block_I=16, num_stages=1, drop-O, mask in shared)
    # but toggle the merge pass.
    func = M._build_sparse_mla_fwd(
        heads=H, dim=512, tail_dim=64, topk=topk, kv_group=1, sm_scale=None,
        is_causal=True, CP0=True, block_I=16, num_stages=1, threads=256,
        use_gb10=True, static_shape=(1, S, SKV),
    )
    pcfg = {
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
    }
    if merge:
        pcfg[tilelang.PassConfigKey.TL_ENABLE_AGGRESSIVE_SHARED_MEMORY_MERGE] = True
    return tilelang.compile(func, out_idx=[-2, -1], pass_configs=pcfg)


def make_inputs(B, S, SKV, H, HKV, DQK, topk, seed=0):
    torch.random.manual_seed(seed)
    q = torch.randn((B, S, H, DQK), dtype=torch.bfloat16, device="cuda")
    kv = torch.randn((B, SKV, HKV, DQK), dtype=torch.bfloat16, device="cuda")
    indices = torch.full((B, S, HKV, topk), SKV, dtype=torch.int32, device="cuda")
    for t in range(S):
        i_i = torch.randperm(max(1, t))[:topk]
        indices[0, t, 0, : len(i_i)] = i_i
    return q, kv, indices


def cos(a, b):
    a = a.double().flatten(); b = b.double().flatten()
    dn = (a * a + b * b).sum()
    return (2 * (a * b).sum() / dn).item() if dn > 0 else float('nan')


def run(merge, S, SKV, H, topk):
    print(f"\n===== merge={merge} S={S} SKV={SKV} H={H} topk={topk} =====")
    try:
        kernel = build_kernel(merge, S, SKV, H, topk)
    except Exception as e:
        print(f"  COMPILE EXCEPTION: {type(e).__name__}: {str(e)[:300]}")
        return
    q, kv, indices = make_inputs(1, S, SKV, H, 1, 576, topk)
    try:
        out, lse = kernel(q, kv, indices)
    except Exception as e:
        print(f"  LAUNCH EXCEPTION: {type(e).__name__}: {str(e)[:300]}")
        return
    out = out.float(); lse = lse.float()
    n_inf = torch.isinf(out).sum().item(); n_nan = torch.isnan(out).sum().item()
    ref = M.ref_sparse_mla_fwd_interface(q, kv, indices).float()
    fullc = cos(out, ref)
    relerr = ((out - ref).norm() / ref.norm()).item()
    print(f"  out inf={n_inf} nan={n_nan}  lse inf={torch.isinf(lse).sum().item()}")
    print(f"  FULL-tensor cos={fullc}  FULL relerr={relerr}")
    # official assert (eps=1e-2)
    try:
        assert_tensors_similar(out.to(torch.bfloat16), ref.to(torch.bfloat16), eps=1e-2, name="out")
        print("  OFFICIAL assert_tensors_similar(eps=1e-2): PASS")
    except AssertionError:
        print("  OFFICIAL assert_tensors_similar(eps=1e-2): FAIL")


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "small"
    if which == "small":
        run(False, 512, 1024, 128, 256)
        run(True, 512, 1024, 128, 256)  # control: shows the bug
    elif which == "big":
        run(False, 4096, 8192, 128, 2048)
    elif which == "both":
        run(False, 512, 1024, 128, 256)
        run(False, 4096, 8192, 128, 2048)
