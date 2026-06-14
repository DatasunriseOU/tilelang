"""Partial-mask OOB-safety parity: the EmptyRegionCollapse change ONLY touches the
masked-OOB zero-fill partition loop in _emit_oob_zero_partition. Validate that for
EVERY partial-mask shape (hd<BLK and/or ds<BLK and/or cs<BLK forcing non-empty OOB
zero regions) the ROUTED (patched) kernel is BIT-EXACT vs native mamba_ssm triton.

A non-empty OOB region is exactly where the reorder could (in principle) change the
written set; RULE #1 demands MAXDIFF=0 (OOB lanes still read 0) on all of them.

usage: partmask_parity.py <pf_json>
"""
import sys, os
sys.path.insert(0, "/home/dave/source/tilelang")
os.environ["TILELANG_DISABLE_CACHE"] = "1"
import torch, triton
import tilelang, tvm

jp = sys.argv[1] if len(sys.argv) > 1 else "/tmp/pf__chunk_scan_bwd_dstates.json"
pf = tvm.ir.load_json(open(jp).read())
kernel = tilelang.compile(pf, target="cuda")
print("compiled nparams=%d" % len(kernel.params), flush=True)

dev = "cuda"
from mamba_ssm.ops.triton.ssd_chunk_scan import _chunk_scan_bwd_dstates_kernel as _autok
_jit = getattr(_autok, "fn", _autok)
BM, BN, BK = 64, 64, 32

# (hd, ds, cs) partial-mask shapes. The §P1 routed PrimFunc was baked BM=BN=64,BK=32.
#  hd<64 -> M-axis OOB; ds<64 -> N-axis OOB; cs not mult of BK -> K-tail partial.
SHAPES = [
    # name,    hd,  ds,  cs   (b,nh,ng,seq derived)
    ("hd100_ds72",      100, 72, 64),   # both M & N OOB (hd>64 spans 2 M-tiles, partial 2nd)
    ("hd64_ds48",        64, 48, 64),   # N-axis OOB only
    ("hd48_ds40_cs48",   48, 40, 48),   # M & N OOB + K-tail partial (cs=48, BK=32 -> 16-tail)
    ("hd40_ds64",        40, 64, 64),   # M-axis OOB only
    ("hd128_ds128",     128,128, 64),   # multi-tile (2x2) all full -> regression control
    ("hd100_ds72_cs48", 100, 72, 48),   # all three axes partial
]

def run_shape(name, hd, ds, cs):
    torch.manual_seed(0)
    b, nh, ng = 1, 8, 2
    seq = 256
    nc = seq // cs if seq % cs == 0 else (seq // cs)
    # keep seq a multiple of cs so chunking is clean
    seq = nc * cs
    dout = torch.randn(b, seq, nh, hd, device=dev, dtype=torch.float32)
    C = torch.randn(b, seq, ng, ds, device=dev, dtype=torch.float32)
    dA = torch.randn(b, nh, nc, cs, device=dev, dtype=torch.float32)
    native = torch.empty(b, nc, nh, hd, ds, device=dev, dtype=torch.float32)
    gd0 = triton.cdiv(hd, BM) * triton.cdiv(ds, BN)
    grid = (gd0, b * nc, nh)
    _jit[grid](dout, C, native, dA, None, hd, ds, cs, b, seq, nc, nh // ng,
        dout.stride(0), dout.stride(1), dout.stride(2), dout.stride(3),
        C.stride(0), C.stride(1), C.stride(2), C.stride(3),
        native.stride(0), native.stride(1), native.stride(2), native.stride(3), native.stride(4),
        dA.stride(0), dA.stride(2), dA.stride(1), dA.stride(3), 0, 0,
        HAS_SEQ_IDX=False, BLOCK_SIZE_M=BM, BLOCK_SIZE_N=BN, BLOCK_SIZE_K=BK)
    torch.cuda.synchronize()

    gd1 = b * nc; gd2 = nh
    dprev = torch.zeros(b, nc, nh, hd, ds, device=dev, dtype=torch.float32)
    sd = [int(x) for x in dout.stride()]; sc = [int(x) for x in C.stride()]
    sp = [int(x) for x in dprev.stride()]; sa = [int(x) for x in dA.stride()]
    args = [dout.reshape(-1), C.reshape(-1), dprev.reshape(-1), dA.reshape(-1),
            torch.zeros(1, device=dev, dtype=torch.float32),
            hd, ds, cs, b, seq, nc, nh // ng,
            sd[0], sd[1], sd[2], sd[3], sc[0], sc[1], sc[2], sc[3],
            sp[0], sp[1], sp[2], sp[3], sp[4], sa[0], sa[2], sa[1], sa[3], 0, 0,
            gd2, gd1, gd0]
    try:
        kernel(*args); torch.cuda.synchronize()
    except ValueError as e:
        if "expected 37" in str(e):
            args = args[:-3] + [gd2, gd2, gd1, gd1, gd0, gd0]
            kernel(*args); torch.cuda.synchronize()
        else:
            raise
    md = float((dprev - native).abs().max())
    ac = torch.allclose(dprev, native, atol=1e-3, rtol=1e-3)
    nz_r = int((dprev.abs() > 0).sum()); nz_n = int((native.abs() > 0).sum())
    bitexact = (md == 0.0) and (nz_r == nz_n)
    # OOB structure note: how many M/N tiles, whether 2nd tile is partial
    mtiles = triton.cdiv(hd, BM); ntiles = triton.cdiv(ds, BN)
    m_oob = (hd % BM != 0); n_oob = (ds % BN != 0); k_part = (cs % BK != 0)
    print("[%-16s] hd=%-3d ds=%-3d cs=%-2d Mtiles=%d Ntiles=%d OOB(M=%s N=%s Ktail=%s) "
          "nz_r=%d nz_n=%d MAXDIFF=%.3e BITEXACT=%s ALLCLOSE=%s -> %s"
          % (name, hd, ds, cs, mtiles, ntiles, m_oob, n_oob, k_part,
             nz_r, nz_n, md, bitexact, ac, "PASS" if bitexact else "FAIL"), flush=True)
    return bitexact, md

results = {}
for name, hd, ds, cs in SHAPES:
    try:
        be, md = run_shape(name, hd, ds, cs)
    except Exception as e:
        import traceback; traceback.print_exc()
        print("[%s] EXCEPTION %s" % (name, str(e)[:200]), flush=True)
        be, md = False, float("nan")
    results[name] = (be, md)

npass = sum(1 for be, _ in results.values() if be)
print("\n===== PARTIAL-MASK PARITY SUMMARY: BITEXACT %d/%d =====" % (npass, len(results)))
for n, (be, md) in results.items():
    print("%-18s %s  MAXDIFF=%.3e" % (n, "BITEXACT" if be else "FAIL", md))
sys.exit(0 if npass == len(results) else 1)
