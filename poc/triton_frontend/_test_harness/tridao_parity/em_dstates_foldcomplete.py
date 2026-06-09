"""COMPLETE-FOLD §P1 dstates measurement.

Tests whether feeding the *fully canonicalized* TTIR (Triton make_ttir passes
applied -> guard chain extsi/cmpi/INT_MAX/andi folded to native targets) through
OUR walker drops register spills below the emitter-level prologue_opt fold (272)
toward ~0, with the 64 serial prologue loops gone and parity == 4.88e-04.

Variants (all on the EXECUTED cubin, torch+triton in sys.modules):
  OFF   : raw 895-line TTIR, prologue_opt=False  (serial-prologue baseline)
  OPT   : raw 895-line TTIR, prologue_opt=True + cp.async  (emitter fold, 272 spills)
  FOLD  : folded 362-line TTIR (guard chain gone) + prologue_opt=True + cp.async
          (the COMPLETE backend-agnostic fold -- this is what we want at ~0 spills)

Each variant is parity-checked against the native Triton kernel and CUDA-event
timed (N>=50 interleaved). RULE#1: bit-correct EXECUTED numbers only.
"""
import os, sys, re
sys.path.insert(0, "/home/dave/source/tilelang")
import torch, triton  # libtriton loaded -> executed §P1 path
print("TORCH_IN_MODULES=%s TRITON_IN_MODULES=%s" % (
    "torch" in sys.modules, "triton" in sys.modules), flush=True)
import tilelang, tvm
from poc.triton_frontend import from_ttir

RAW = "/tmp/ttir7/_chunk_scan_bwd_dstates.ttir"
FOLDED = "/tmp/dstates_folded.ttir"
ttir_raw = open(RAW).read()
ttir_folded = open(FOLDED).read()
print("RAW lines=%d FOLDED lines=%d" % (
    ttir_raw.count(chr(10)), ttir_folded.count(chr(10))), flush=True)
# census of the guard chain in each input
def census(s, tag):
    print("CENSUS %s extsi=%d cmpi=%d INT_MAX=%d andi=%d" % (
        tag, s.count("arith.extsi"), s.count("arith.cmpi"),
        s.count("2147483647"), s.count("arith.andi")), flush=True)
census(ttir_raw, "RAW")
census(ttir_folded, "FOLDED")


def build(ttir, prologue_opt, force_cp_async):
    if force_cp_async:
        os.environ["TL_FORCE_CP_ASYNC"] = "1"
    else:
        os.environ.pop("TL_FORCE_CP_ASYNC", None)
    pf = from_ttir(ttir, name="_chunk_scan_bwd_dstates_kernel", target="cuda",
                   _allow_text_ttir=True, prologue_opt=prologue_opt)
    return tilelang.compile(pf, target="cuda")


print("building OFF (raw, prologue_opt=False, plain LDG)...", flush=True)
k_off = build(ttir_raw, False, force_cp_async=False)
print("building OPT (raw, prologue_opt=True, cp.async)...", flush=True)
k_opt = build(ttir_raw, True, force_cp_async=True)
print("building FOLD (folded TTIR, prologue_opt=True, cp.async)...", flush=True)
k_fold = build(ttir_folded, True, force_cp_async=True)
print("OFF nparams=%d OPT nparams=%d FOLD nparams=%d" % (
    len(k_off.params), len(k_opt.params), len(k_fold.params)), flush=True)


def sass_counts(kernel, label):
    path = "/tmp/emfc_sass_%s.sass" % label
    try:
        kernel.export_sass(path)
        sass = open(path).read()
    except Exception:
        sass = kernel._get_sass()
        open(path, "w").write(sass)

    def cnt(pat):
        return len(re.findall(pat, sass))
    d = dict(
        UTMALDG=cnt(r'\bUTMALDG'), LDG=cnt(r'\bLDG\b'), LDGSTS=cnt(r'\bLDGSTS'),
        HMMA=cnt(r'\bHMMA'), STL=cnt(r'\bSTL\b'), LDL=cnt(r'\bLDL\b'),
        IMAD=cnt(r'\bIMAD'), ISETP=cnt(r'\bISETP'),
    )
    print("SASS_%s UTMALDG=%d LDG=%d LDGSTS=%d HMMA=%d STL=%d LDL=%d spill=%d IMAD=%d ISETP=%d len=%d" % (
        label, d["UTMALDG"], d["LDG"], d["LDGSTS"], d["HMMA"],
        d["STL"], d["LDL"], d["STL"] + d["LDL"], d["IMAD"], d["ISETP"], len(sass)), flush=True)
    return d


sass_off = sass_counts(k_off, "OFF")
sass_opt = sass_counts(k_opt, "OPT")
sass_fold = sass_counts(k_fold, "FOLD")

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


parity = {}
for label, kernel in [("OFF", k_off), ("OPT", k_opt), ("FOLD", k_fold)]:
    r = run(kernel)
    md = float((r - native).abs().max())
    ac = bool(torch.allclose(r, native, atol=1e-3, rtol=1e-3))
    nz = int((r != 0).sum()); tot = r.numel()
    parity[label] = (md, ac, nz, tot)
    print("PARITY %s MAXDIFF=%.6e ALLCLOSE_1e-3=%s nonzero=%d/%d %s" % (
        label, md, ac, nz, tot, "PASS" if ac else "FAIL"), flush=True)


def time_native(n):
    out = torch.empty(b, nc, nh, hd, ds, device=dev, dtype=torch.float32)
    def once():
        _jit[grid](dout, C, out, dA, None, hd, ds, cs, b, s, nc, nh // ng,
            dout.stride(0), dout.stride(1), dout.stride(2), dout.stride(3),
            C.stride(0), C.stride(1), C.stride(2), C.stride(3),
            out.stride(0), out.stride(1), out.stride(2), out.stride(3), out.stride(4),
            dA.stride(0), dA.stride(2), dA.stride(1), dA.stride(3), 0, 0,
            HAS_SEQ_IDX=False, BLOCK_SIZE_M=BM, BLOCK_SIZE_N=BN, BLOCK_SIZE_K=BK)
    for _ in range(10):
        once()
    torch.cuda.synchronize()
    times = []
    for _ in range(n):
        st = torch.cuda.Event(enable_timing=True); en = torch.cuda.Event(enable_timing=True)
        st.record(); once(); en.record(); torch.cuda.synchronize()
        times.append(st.elapsed_time(en))
    times.sort()
    return times[len(times) // 2]


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


N = 50
opt_med = []; fold_med = []
# Skip re-timing OFF here (already measured at ~3140ms in em_dstates run); time
# the two fast routed variants interleaved + native.
for rep in range(4):
    m2 = timed(k_opt, N); opt_med.append(m2)
    m3 = timed(k_fold, N); fold_med.append(m3)
    print("rep%d OPT med=%.4f FOLD med=%.4f" % (rep, m2, m3), flush=True)
opt = sorted(opt_med)[len(opt_med) // 2]
fold = sorted(fold_med)[len(fold_med) // 2]
nat = time_native(N)

print("=== INTERLEAVED CUDA-EVENT (N=%d/rep x4 reps, median-of-medians) ===" % N)
print("OPT_MS=%.4f" % opt)
print("FOLD_MS=%.4f" % fold)
print("NATIVE_MS=%.4f" % nat)
print("FOLD_vs_OPT=%.4fx" % (opt / fold))
print("FOLD_GAP_TO_NATIVE=%.2fx" % (fold / nat))
print("EMFCRESULT OPT_SPILL=%d FOLD_SPILL=%d OPT_LDGSTS=%d FOLD_LDGSTS=%d "
      "FOLD_IMAD=%d FOLD_ISETP=%d OPT_MAXDIFF=%.6e FOLD_MAXDIFF=%.6e "
      "OPT_MS=%.4f FOLD_MS=%.4f NATIVE_MS=%.4f" % (
      sass_opt["STL"] + sass_opt["LDL"], sass_fold["STL"] + sass_fold["LDL"],
      sass_opt["LDGSTS"], sass_fold["LDGSTS"], sass_fold["IMAD"], sass_fold["ISETP"],
      parity["OPT"][0], parity["FOLD"][0], opt, fold, nat), flush=True)
print("DONE", flush=True)
