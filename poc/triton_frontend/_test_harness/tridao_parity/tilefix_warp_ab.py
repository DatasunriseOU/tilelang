"""FINAL HONEST A/B for the TileFix MEASURE deliverable.

Reproduces the documented §P1 baseline (the 745ms "OPT" from em_dstates_cpasync:
default config = num_warps=4, num_stages=2, prologue_opt=True, cp.async) and
compares it to the TileFix autotune config (num_warps=8, num_stages=3 -- the
config the native _chunk_scan_bwd_dstates_kernel autotunes to). Same raw TTIR,
same prologue_opt + cp.async for both. Interleaved CUDA-event timing + parity.

This isolates the TileFix: honoring the Triton autotune warp/stage config.
"""
import os, sys, re
sys.path.insert(0, "/home/dave/source/tilelang")
import torch, triton
import tilelang, tvm
from poc.triton_frontend import from_ttir

ttir = open("/tmp/ttir7/_chunk_scan_bwd_dstates.ttir").read()

def build(nw, ns):
    os.environ["TL_FORCE_CP_ASYNC"] = "1"
    kw = {}
    if nw is not None: kw["num_warps"] = nw
    if ns is not None: kw["num_stages"] = ns
    pf = from_ttir(ttir, name="_chunk_scan_bwd_dstates_kernel", target="cuda",
                   _allow_text_ttir=True, prologue_opt=True, **kw)
    return tilelang.compile(pf, target="cuda")

def census(k, tag):
    try:
        p = "/tmp/tfa_%s.sass" % tag; k.export_sass(p); sass = open(p).read()
    except Exception:
        sass = k._get_sass()
    c = lambda pat: len(re.findall(pat, sass))
    d = dict(HMMA=c(r"\bHMMA"), LDGSTS=c(r"\bLDGSTS"), IMAD=c(r"\bIMAD"),
             ISETP=c(r"\bISETP"), spill=c(r"\bSTL\b") + c(r"\bLDL\b"))
    print("CENSUS_%s HMMA=%d LDGSTS=%d IMAD=%d ISETP=%d spill=%d" % (
        tag, d["HMMA"], d["LDGSTS"], d["IMAD"], d["ISETP"], d["spill"]), flush=True)
    return d

# BASE = documented 745ms config (defaults: num_warps=4, num_stages=2)
# FIX  = TileFix autotune config (num_warps=8, num_stages=3)
print("building BASE (defaults nw=4 ns=2 == the 745ms config)...", flush=True)
k_base = build(None, None)
print("building FIX (TileFix autotune nw=8 ns=3)...", flush=True)
k_fix = build(8, 3)
cb = census(k_base, "BASE")
cf = census(k_fix, "FIX")

torch.manual_seed(0); dev = "cuda"
b, nh, hd, ds = 1, 112, 64, 64; cs, ng = 64, 8; s = 4096; nc = s // cs
BM, BN, BK = 64, 64, 32
dout = torch.randn(b, s, nh, hd, device=dev, dtype=torch.float32)
C = torch.randn(b, s, ng, ds, device=dev, dtype=torch.float32)
dA = torch.randn(b, nh, nc, cs, device=dev, dtype=torch.float32)
from mamba_ssm.ops.triton.ssd_chunk_scan import _chunk_scan_bwd_dstates_kernel as _autok
_jit = getattr(_autok, "fn", _autok)
native = torch.empty(b, nc, nh, hd, ds, device=dev, dtype=torch.float32)
grid = (triton.cdiv(hd, BM) * triton.cdiv(ds, BN), b * nc, nh)
def run_native(out):
    _jit[grid](dout, C, out, dA, None, hd, ds, cs, b, s, nc, nh // ng,
        dout.stride(0), dout.stride(1), dout.stride(2), dout.stride(3),
        C.stride(0), C.stride(1), C.stride(2), C.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3), out.stride(4),
        dA.stride(0), dA.stride(2), dA.stride(1), dA.stride(3), 0, 0,
        HAS_SEQ_IDX=False, BLOCK_SIZE_M=BM, BLOCK_SIZE_N=BN, BLOCK_SIZE_K=BK)
run_native(native); torch.cuda.synchronize()
gd0 = triton.cdiv(hd, BM) * triton.cdiv(ds, BN); gd1 = b * nc; gd2 = nh
sd = [int(x) for x in dout.stride()]; sc = [int(x) for x in C.stride()]; sa = [int(x) for x in dA.stride()]
def make_args(dprev):
    sp = [int(x) for x in dprev.stride()]
    return [dout.reshape(-1), C.reshape(-1), dprev.reshape(-1), dA.reshape(-1),
        torch.zeros(1, device=dev, dtype=torch.float32), hd, ds, cs, b, s, nc, nh // ng,
        sd[0], sd[1], sd[2], sd[3], sc[0], sc[1], sc[2], sc[3],
        sp[0], sp[1], sp[2], sp[3], sp[4], sa[0], sa[2], sa[1], sa[3], 0, 0, gd2, gd1, gd0]
def run(k):
    dprev = torch.zeros(b, nc, nh, hd, ds, device=dev, dtype=torch.float32)
    k(*make_args(dprev)); torch.cuda.synchronize(); return dprev
parity = {}
for lab, k in [("BASE", k_base), ("FIX", k_fix)]:
    r = run(k); md = float((r - native).abs().max())
    ac = bool(torch.allclose(r, native, atol=1e-3, rtol=1e-3))
    parity[lab] = md
    print("PARITY_%s MAXDIFF=%.6e %s" % (lab, md, "PASS" if ac else "FAIL"), flush=True)
dprev = torch.zeros(b, nc, nh, hd, ds, device=dev, dtype=torch.float32)
args = make_args(dprev)
def timed(k, n):
    for _ in range(10): k(*args)
    torch.cuda.synchronize(); ts = []
    for _ in range(n):
        st = torch.cuda.Event(enable_timing=True); en = torch.cuda.Event(enable_timing=True)
        st.record(); k(*args); en.record(); torch.cuda.synchronize(); ts.append(st.elapsed_time(en))
    ts.sort(); return ts[len(ts) // 2]
def timed_nat(n):
    out = torch.empty(b, nc, nh, hd, ds, device=dev, dtype=torch.float32)
    for _ in range(10): run_native(out)
    torch.cuda.synchronize(); ts = []
    for _ in range(n):
        st = torch.cuda.Event(enable_timing=True); en = torch.cuda.Event(enable_timing=True)
        st.record(); run_native(out); en.record(); torch.cuda.synchronize(); ts.append(st.elapsed_time(en))
    ts.sort(); return ts[len(ts) // 2]
N = 50; mb = []; mf = []
for rep in range(4):
    a = timed(k_base, N); mb.append(a); bb = timed(k_fix, N); mf.append(bb)
    print("rep%d BASE=%.4f FIX=%.4f" % (rep, a, bb), flush=True)
base = sorted(mb)[len(mb) // 2]; fix = sorted(mf)[len(mf) // 2]; nat = timed_nat(N)
print("=== TILEFIX A/B (N=%d x4, median-of-medians) ===" % N, flush=True)
print("BASE_MS=%.4f FIX_MS=%.4f NATIVE_MS=%.4f" % (base, fix, nat), flush=True)
print("FIX_vs_BASE=%.4fx FIX_GAP_TO_NATIVE=%.2fx BASE_GAP_TO_NATIVE=%.2fx" % (
    base / fix, fix / nat, base / nat), flush=True)
print("TFARESULT BASE_MS=%.4f FIX_MS=%.4f NATIVE_MS=%.4f BASE_spill=%d FIX_spill=%d "
      "BASE_HMMA=%d FIX_HMMA=%d BASE_LDGSTS=%d FIX_LDGSTS=%d BASE_IMAD=%d FIX_IMAD=%d "
      "BASE_ISETP=%d FIX_ISETP=%d BASE_MAXDIFF=%.6e FIX_MAXDIFF=%.6e" % (
      base, fix, nat, cb["spill"], cf["spill"], cb["HMMA"], cf["HMMA"],
      cb["LDGSTS"], cf["LDGSTS"], cb["IMAD"], cf["IMAD"], cb["ISETP"], cf["ISETP"],
      parity["BASE"], parity["FIX"]), flush=True)
print("DONE", flush=True)
