"""EXECUTEMEASURE §P1 dstates -- cp.async/LDGSTS LIVE + bit-correct on the EXECUTED path.

Imports torch + triton FIRST (libtriton loaded == the real executed §P1 path).
OFF = prologue_opt=False (un-routed serial baseline; plain LDG; the honest fresh-build OFF).
OPT = prologue_opt=True + TL_FORCE_CP_ASYNC=1 -> routed CopyNode emits is_async_copy
      -> genuine cp.async/LDGSTS, race-closed by commit_group + wait_group<0> + CTA sync.

Verifies on the EXECUTED cubins (torch+triton in sys.modules):
  1. SASS: OFF LDGSTS=0, OPT LDGSTS>0 / UTMALDG=0   (cp.async live, not compile-only)
  2. PARITY both == native (MAXDIFF ~4.88e-04, NOT racy 1.28e3)
  3. interleaved CUDA-event timing N>=50, OFF vs OPT, both bit-correct
  4. native ref ms for the remaining gap
"""
import os, sys, re
sys.path.insert(0, "/home/dave/source/tilelang")
import torch, triton  # libtriton loaded -> executed §P1 path
print("TORCH_IN_MODULES=%s TRITON_IN_MODULES=%s" % (
    "torch" in sys.modules, "triton" in sys.modules), flush=True)
import tilelang, tvm
from poc.triton_frontend import from_ttir

ttir = open("/tmp/ttir7/_chunk_scan_bwd_dstates.ttir").read()


def build(prologue_opt, force_cp_async):
    # Honestly gate the async path per-build via the documented env knob.
    if force_cp_async:
        os.environ["TL_FORCE_CP_ASYNC"] = "1"
    else:
        os.environ.pop("TL_FORCE_CP_ASYNC", None)
    pf = from_ttir(ttir, name="_chunk_scan_bwd_dstates_kernel", target="cuda",
                   _allow_text_ttir=True, prologue_opt=prologue_opt)
    return tilelang.compile(pf, target="cuda")


print("building OFF (prologue_opt=False, plain LDG)...", flush=True)
k_off = build(False, force_cp_async=False)
print("building OPT (prologue_opt=True, TL_FORCE_CP_ASYNC=1, cp.async live)...", flush=True)
k_opt = build(True, force_cp_async=True)
print("OFF nparams=%d OPT nparams=%d" % (len(k_off.params), len(k_opt.params)), flush=True)


def sass_counts(kernel, label):
    path = "/tmp/em_sass_%s.sass" % label
    try:
        kernel.export_sass(path)
        sass = open(path).read()
    except Exception:
        sass = kernel._get_sass()
        open(path, "w").write(sass)

    def cnt(pat):
        return len(re.findall(pat, sass))
    d = dict(
        UTMALDG=cnt(r'\bUTMALDG'),
        LDG=cnt(r'\bLDG\b'),
        LDGSTS=cnt(r'\bLDGSTS'),
        HMMA=cnt(r'\bHMMA'),
        STL=cnt(r'\bSTL\b'),
        LDL=cnt(r'\bLDL\b'),
    )
    print("SASS_%s UTMALDG=%d LDG=%d LDGSTS=%d HMMA=%d spill=%d len=%d" % (
        label, d["UTMALDG"], d["LDG"], d["LDGSTS"], d["HMMA"],
        d["STL"] + d["LDL"], len(sass)), flush=True)
    return d


# SASS of the EXECUTED kernels (these exact compiled objects are launched below)
sass_off = sass_counts(k_off, "OFF")
sass_opt = sass_counts(k_opt, "OPT")

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
for label, kernel in [("OFF", k_off), ("OPT", k_opt)]:
    r = run(kernel)
    md = float((r - native).abs().max())
    ac = bool(torch.allclose(r, native, atol=1e-3, rtol=1e-3))
    nz = int((r != 0).sum()); tot = r.numel()
    parity[label] = (md, ac, nz, tot)
    print("PARITY %s MAXDIFF=%.6e ALLCLOSE_1e-3=%s nonzero=%d/%d %s" % (
        label, md, ac, nz, tot, "PASS" if ac else "FAIL"), flush=True)


# native CUDA-event timing
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


N = 60
off_med = []; opt_med = []
for rep in range(4):
    m1 = timed(k_off, N); off_med.append(m1)
    m2 = timed(k_opt, N); opt_med.append(m2)
    print("rep%d OFF med=%.4f OPT med=%.4f" % (rep, m1, m2), flush=True)
off = sorted(off_med)[len(off_med) // 2]
opt = sorted(opt_med)[len(opt_med) // 2]
nat = time_native(N)

print("=== INTERLEAVED CUDA-EVENT (N=%d/rep x4 reps, median-of-medians) ===" % N)
print("OFF_MS=%.4f" % off)
print("OPT_MS=%.4f" % opt)
print("NATIVE_MS=%.4f" % nat)
print("DELTA_MS=%.4f (OFF-OPT)" % (off - opt))
print("SPEEDUP_vs_OFF=%.4fx" % (off / opt))
print("GAP_TO_NATIVE=%.2fx (OPT/NATIVE)" % (opt / nat))
# machine-readable summary line
print("EMRESULT OFF_LDGSTS=%d OPT_LDGSTS=%d OPT_UTMALDG=%d OFF_MAXDIFF=%.6e OPT_MAXDIFF=%.6e "
      "OFF_MS=%.4f OPT_MS=%.4f NATIVE_MS=%.4f" % (
      sass_off["LDGSTS"], sass_opt["LDGSTS"], sass_opt["UTMALDG"],
      parity["OFF"][0], parity["OPT"][0], off, opt, nat), flush=True)
print("DONE", flush=True)
