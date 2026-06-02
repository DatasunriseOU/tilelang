# ruff: noqa
# Diagnostic: characterize the forward LSE/online-softmax inf bug on gb10 sm_121.
# Runs the re-tiled gb10 sparse_mla_fwd, locates inf rows, correlates with
# per-row valid-key counts, and compares against the reference over ALL rows.
import os, sys
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sparse_mla_fwd import sparse_mla_fwd_interface, ref_sparse_mla_fwd_interface


def make_inputs(B, S, SKV, H, HKV, DQK, topk, seed=0):
    torch.random.manual_seed(seed)
    q = torch.randn((B, S, H, DQK), dtype=torch.bfloat16, device="cuda")
    kv = torch.randn((B, SKV, HKV, DQK), dtype=torch.bfloat16, device="cuda")
    indices = torch.full((B, S, HKV, topk), SKV, dtype=torch.int32, device="cuda")
    for b in range(B):
        for t in range(S):
            for h in range(HKV):
                i_i = torch.randperm(max(1, t))[:topk]
                indices[b, t, h, : len(i_i)] = i_i
    return q, kv, indices


def valid_counts(indices, S, SKV, topk):
    # number of in-causal-range (index <= q) AND in-bounds (index < SKV) slots per row
    # indices: [B, S, HKV, topk]; HKV=1, group g=0
    idx = indices[0, :, 0, :].long()  # [S, topk]
    qpos = torch.arange(S, device=idx.device).view(-1, 1)
    causal = idx <= qpos
    inbound = idx < SKV
    valid = (causal & inbound)
    return valid.sum(dim=1)  # [S]


def run(B, S, SKV, H, HKV, DQK, DV, topk, gb10):
    q, kv, indices = make_inputs(B, S, SKV, H, HKV, DQK, topk)
    out, lse = sparse_mla_fwd_interface(q, kv, indices, gb10=gb10)
    out = out.float()
    lse = lse.float()
    vc = valid_counts(indices, S, SKV, topk)  # [S]

    out_nan = torch.isnan(out).sum().item()
    out_inf = torch.isinf(out).sum().item()
    lse_nan = torch.isnan(lse).sum().item()
    lse_inf = torch.isinf(lse).sum().item()

    # rows (b=0, over S, any head, any dim) that have inf/nan in out
    out_bad = (torch.isinf(out) | torch.isnan(out)).any(dim=(2, 3))[0]  # [S]
    bad_rows = torch.nonzero(out_bad).flatten().tolist()
    print(f"=== gb10={gb10} S={S} SKV={SKV} H={H} topk={topk} ===")
    print(f"out: nan={out_nan} inf={out_inf} | lse: nan={lse_nan} inf={lse_inf}")
    print(f"num bad rows (out inf/nan): {len(bad_rows)} of {S}")
    if bad_rows:
        # correlate bad rows with valid-key count
        bad_vc = vc[bad_rows]
        print(f"  bad-row valid-key counts: min={bad_vc.min().item()} max={bad_vc.max().item()} "
              f"mean={bad_vc.float().mean().item():.1f}")
        print(f"  first 15 bad rows: {bad_rows[:15]}")
        print(f"  their valid counts: {bad_vc[:15].tolist()}")
        # how many rows total have valid_count == 0 / < topk
        n_zero = (vc == 0).sum().item()
        n_lt = (vc < topk).sum().item()
        print(f"  rows with valid_count==0: {n_zero}; valid_count<topk: {n_lt}")
        # are ALL bad rows ones with valid_count < topk? or specifically small?
        print(f"  bad rows with vc==0: {(bad_vc==0).sum().item()}; "
              f"vc<topk: {(bad_vc<topk).sum().item()}; vc>=topk: {(bad_vc>=topk).sum().item()}")
        # show lse for a few bad rows (head 0)
        for r in bad_rows[:5]:
            l = lse[0, r, :]
            print(f"  row {r}: vc={vc[r].item()} lse[h0..h3]={l[:4].tolist()} "
                  f"lse_inf={torch.isinf(l).sum().item()} lse_nan={torch.isnan(l).sum().item()}")

    # reference + ALL-ROWS parity
    ref = ref_sparse_mla_fwd_interface(q, kv, indices).float()
    ref_nan = torch.isnan(ref).sum().item()
    ref_inf = torch.isinf(ref).sum().item()
    print(f"ref: nan={ref_nan} inf={ref_inf}")

    # full-tensor cos (nan/inf included -> will be nan if any bad)
    def cos(a, b):
        a = a.double().flatten(); b = b.double().flatten()
        denom = (a*a + b*b).sum()
        if denom == 0:
            return float('nan')
        return (2*(a*b).sum()/denom).item()
    full = cos(out, ref)
    # finite-only cos
    finite = torch.isfinite(out) & torch.isfinite(ref)
    fin_cos = cos(out[finite], ref[finite])
    print(f"FULL-tensor cos (all rows): {full}")
    print(f"finite-only cos: {fin_cos}  (finite frac={finite.float().mean().item():.4f})")
    return bad_rows, vc


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--S", type=int, default=512)
    p.add_argument("--SKV", type=int, default=1024)
    p.add_argument("--H", type=int, default=128)
    p.add_argument("--topk", type=int, default=256)
    p.add_argument("--gb10", type=int, default=1)
    a = p.parse_args()
    run(1, a.S, a.SKV, a.H, 1, 576, 512, a.topk, bool(a.gb10))
