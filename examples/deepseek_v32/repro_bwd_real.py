# ruff: noqa
# Real end-to-end bwd parity using the production pipeline:
#   tl fwd -> (o, lse) ; tl preprocess -> delta ; tl bwd -> dq, dkv
# compared against the autograd reference.  In-bounds distinct indices.
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
torch.manual_seed(0)
import sparse_mla_bwd as Bwd
from sparse_mla_fwd import sparse_mla_fwd_interface, ref_sparse_mla_fwd_interface

def cos(a, b):
    a = a.float().flatten(); b = b.float().flatten()
    return (torch.dot(a, b) / (a.norm() * b.norm() + 1e-20)).item()
def relerr(a, b):
    a = a.float(); b = b.float()
    return ((a - b).norm() / (b.norm() + 1e-20)).item()

def ref_bwd(q, kv, do, indices):
    q = q.detach().clone().float().requires_grad_(True)
    kv = kv.detach().clone().float().requires_grad_(True)
    o = ref_sparse_mla_fwd_interface(q, kv, indices)
    o.backward(do.float())
    return q.grad.detach().clone(), kv.grad.detach().clone()

def main(S=256, SKV=512, H=64, topk=128):
    B_, HKV, DQK, DV = 1, 1, 576, 512
    dtype = torch.bfloat16; dev = "cuda"
    torch.manual_seed(0)
    q = torch.randn((B_, S, H, DQK), dtype=dtype, device=dev)
    kv = torch.randn((B_, SKV, HKV, DQK), dtype=dtype, device=dev)
    do = torch.randn((B_, S, H, DV), dtype=dtype, device=dev)
    # Only query rows with t >= topk have `topk` DISTINCT in-bounds keys (a
    # fully-filled softmax).  Rows t < topk are degenerate (fewer valid keys ->
    # the harness would need duplicate padding, which the masked reference and
    # the summing kernel treat differently).  Zero `do` on the degenerate rows
    # so they contribute nothing to dkv, and compare dq only on the filled rows.
    indices = torch.zeros((B_, S, HKV, topk), dtype=torch.int32, device=dev)
    for t in range(S):
        hi = max(1, min(t + 1, SKV))
        perm = torch.randperm(hi)[:topk]
        if perm.numel() < topk:  # degenerate early row: pad (will be excluded)
            pad = perm[-1:].repeat(topk - perm.numel()) if perm.numel() else torch.zeros(topk, dtype=torch.long)
            perm = torch.cat([perm, pad])
        indices[0, t, 0, :] = perm.to(torch.int32)
    do[:, :topk] = 0  # degenerate rows do not back-propagate into dkv

    tl_out, tl_lse = sparse_mla_fwd_interface(q, kv, indices, gb10=True)
    ref_out = ref_sparse_mla_fwd_interface(q.float(), kv.float(), indices)
    fcos = cos(tl_out[:, topk:], ref_out[:, topk:])
    print(f"[S={S} SKV={SKV} H={H} topk={topk}] fwd cos(t>=topk)={fcos:.5f}")

    tl_dq, tl_dkv = Bwd.sparse_mla_bwd(q, kv, tl_out, do, indices, tl_lse, gb10=True)
    ref_dq, ref_dkv = ref_bwd(q, kv, do, indices)
    # dq: compare only fully-filled query rows
    dq_f, ref_dq_f = tl_dq[:, topk:], ref_dq[:, topk:]
    print(f"  dq  type={type(tl_dq).__name__} nan={torch.isnan(tl_dq).sum().item()}  "
          f"dkv nan={torch.isnan(tl_dkv).sum().item()}")
    print(f"  DQ (t>=topk) cos={cos(dq_f, ref_dq_f):.5f} relerr={relerr(dq_f, ref_dq_f):.5f}")
    print(f"  DKV          cos={cos(tl_dkv, ref_dkv):.5f} relerr={relerr(tl_dkv, ref_dkv):.5f}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        main(int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]))
    else:
        main()
