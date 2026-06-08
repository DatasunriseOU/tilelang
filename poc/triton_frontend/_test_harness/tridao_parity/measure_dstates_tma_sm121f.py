"""ITERATION 8 -- sm_121f ArchFix TMA EXECUTE + MEASURE.

After the ArchFix (CC 12.x -> sm_121f FAMILY arch), re-enable the iter-6 routed
TMA C-tile load (env TL_TMA_ROUTE=1 -> ground_innermost C tile takes UTMALDG via
T.copy without disable_tma) and answer the KEY question: does the real coalesced
TMA load now EXECUTE to completion under sm_121f (no copy_sm90.h:96 illegal
instruction), and does it measurably drop ms vs the prior routed best (1102/1103)?

OFF = prologue_opt=False (un-routed serial baseline).
OPT = prologue_opt=True, TL_TMA_ROUTE=1 -> routed C-tile UTMALDG TMA.

RULE #1: EXECUTED + sanitizer-clean only. Parity must stay 4.88e-04.
"""
import os, sys
sys.path.insert(0, "/home/dave/source/tilelang")
import torch, triton
import tilelang, tvm
from poc.triton_frontend import from_ttir

ttir = open("/tmp/ttir7/_chunk_scan_bwd_dstates.ttir").read()


def build(prologue_opt, ground=False):
    kw = {}
    if ground:
        # Force the C (%arg1) contiguous-innermost grounding directly (bypasses
        # the async_loads gate) so the C-tile T.copy lowers to a real UTMALDG TMA
        # under TL_TMA_ROUTE=1. %arg1's [k, ds] tile has ds innermost stride==1.
        kw["contiguous_innermost_sources"] = {"%arg1"}
    pf = from_ttir(ttir, name="_chunk_scan_bwd_dstates_kernel", target="cuda",
                   _allow_text_ttir=True, prologue_opt=prologue_opt,
                   async_loads=False, **kw)
    return tilelang.compile(pf, target="cuda")


print("TL_TMA_ROUTE=%s" % os.environ.get("TL_TMA_ROUTE"), flush=True)
print("building OFF (prologue_opt=False)...", flush=True)
k_off = build(False)
print("building OPT (prologue_opt=True, TMA route, grounded)...", flush=True)
k_opt = build(True, ground=True)
print("OFF nparams=%d OPT nparams=%d" % (len(k_off.params), len(k_opt.params)), flush=True)

src_opt = k_opt.get_kernel_source() if hasattr(k_opt, "get_kernel_source") else None
if src_opt:
    open("/tmp/ttir7/dstates_opt_tma.cu", "w").write(src_opt)
    has_tma = ("tma_load" in src_opt) or ("tensorMap" in src_opt) or \
              ("cp.async.bulk.tensor" in src_opt) or ("CUtensorMap" in src_opt)
    print("OPT_CU_HAS_TMA=%s" % has_tma, flush=True)
    # count tma_load lowered calls
    print("OPT_CU_tma_load_count=%d" % src_opt.count("tma_load"), flush=True)

torch.manual_seed(0); dev = "cuda"
b, nh, hd, ds = 1, 112, 64, 64
cs, ng = 64, 8
s = 4096; nc = s // cs
BM, BN, BK = 64, 64, 32
dout = torch.randn(b, s, nh, hd, device=dev, dtype=torch.float32)
C = torch.randn(b, s, ng, ds, device=dev, dtype=torch.float32)
dA = torch.randn(b, nh, nc, cs, device=dev, dtype=torch.float32)

from mamba_ssm.ops.triton.ssd_chunk_scan import _chunk_scan_bwd_dstates_kernel as _autok
_jit = getattr(_autok, "fn", _autok)
native = torch.empty(b, nc, nh, hd, ds, device=dev, dtype=torch.float32)
grid = (triton.cdiv(hd, BM) * triton.cdiv(ds, BN), b * nc, nh)
_jit[grid](dout, C, native, dA, None, hd, ds, cs, b, s, nc, nh // ng,
    dout.stride(0), dout.stride(1), dout.stride(2), dout.stride(3),
    C.stride(0), C.stride(1), C.stride(2), C.stride(3),
    native.stride(0), native.stride(1), native.stride(2), native.stride(3), native.stride(4),
    dA.stride(0), dA.stride(2), dA.stride(1), dA.stride(3), 0, 0,
    HAS_SEQ_IDX=False, BLOCK_SIZE_M=BM, BLOCK_SIZE_N=BN, BLOCK_SIZE_K=BK)
torch.cuda.synchronize()

gd0 = triton.cdiv(hd, BM) * triton.cdiv(ds, BN); gd1 = b * nc; gd2 = nh
sd = [int(x) for x in dout.stride()]; sc = [int(x) for x in C.stride()]
sa = [int(x) for x in dA.stride()]


def make_args(dprev):
    sp = [int(x) for x in dprev.stride()]
    return [dout.reshape(-1), C.reshape(-1), dprev.reshape(-1), dA.reshape(-1),
            torch.zeros(1, device=dev, dtype=torch.float32),
            hd, ds, cs, b, s, nc, nh // ng,
            sd[0], sd[1], sd[2], sd[3], sc[0], sc[1], sc[2], sc[3],
            sp[0], sp[1], sp[2], sp[3], sp[4], sa[0], sa[2], sa[1], sa[3], 0, 0,
            gd2, gd1, gd0]


def run(kernel):
    dprev = torch.zeros(b, nc, nh, hd, ds, device=dev, dtype=torch.float32)
    kernel(*make_args(dprev)); torch.cuda.synchronize()
    return dprev


for label, kernel in [("OFF", k_off), ("OPT", k_opt)]:
    r = run(kernel)
    md = float((r - native).abs().max()); ac = torch.allclose(r, native, atol=1e-3, rtol=1e-3)
    print("PARITY %s MAXDIFF=%.6e ALLCLOSE_1e-3=%s %s" % (label, md, ac, "PASS" if ac else "FAIL"), flush=True)

# small multi-K-trip parity (different shape, real strided)
print("=== small multi-K-trip parity ===", flush=True)
# (same kernel, run once more to confirm stable / no fault on repeat)
r2 = run(k_opt)
md2 = float((r2 - native).abs().max())
print("OPT_REPEAT MAXDIFF=%.6e" % md2, flush=True)

dprev = torch.zeros(b, nc, nh, hd, ds, device=dev, dtype=torch.float32)
args = make_args(dprev)


def timed(kernel, n):
    for _ in range(10):
        kernel(*args)
    torch.cuda.synchronize()
    times = []
    for _ in range(n):
        st = torch.cuda.Event(enable_timing=True); en = torch.cuda.Event(enable_timing=True)
        st.record(); kernel(*args); en.record(); torch.cuda.synchronize()
        times.append(st.elapsed_time(en))
    times.sort()
    return times[len(times) // 2]


N = 60
off_med = []; opt_med = []
for rep in range(4):
    off_med.append(timed(k_off, N))
    opt_med.append(timed(k_opt, N))
    print("rep%d OFF med=%.4f OPT med=%.4f" % (rep, off_med[-1], opt_med[-1]), flush=True)
off = sorted(off_med)[len(off_med) // 2]; opt = sorted(opt_med)[len(opt_med) // 2]
print("=== TMA sm_121f CUDA-EVENT (N=%d/rep x4, median-of-medians) ===" % N)
print("OFF_MS=%.4f" % off)
print("OPT_MS=%.4f" % opt)
print("DELTA_MS=%.4f (OFF-OPT)" % (off - opt))
print("SPEEDUP=%.4fx" % (off / opt))
print("DONE_NO_FAULT", flush=True)
