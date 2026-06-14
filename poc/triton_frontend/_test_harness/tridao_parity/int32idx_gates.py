"""LEVER (int32 element-index static-shape specialization) -- full gate suite.

Uses the production module ``poc.triton_frontend.specialize_static_shape`` +
``int32_index_safe``. For each shape it:
  - computes the per-tensor element counts and the dispatch decision,
  - if int32-safe: bakes concrete strides/dims/gridDim + numels into the
    PrimFunc (ABI-preserving) and validates BIT-EXACT (MAXDIFF=0) vs native,
  - reports SASS IMAD.WIDE/LEA.HI + int64_t cast collapse + ptxas regs/spill,
  - reports routed wall-clock A/B (symbolic int64 baseline vs int32-spec).

Gates run:
  prod P1 (b1 nh112 hd64 ds64 nc64 cs64), 6 partial-mask, int64 2.82GB,
  int64 4.35GB, and a SYNTHETIC >2^31-element shape that MUST route int64.

usage: int32idx_gates.py <pf_json> [--gate prod|partmask|int64|safety|all]
"""
import sys, os, re, time, subprocess
sys.path.insert(0, "/home/dave/source/tilelang")
os.environ["TILELANG_DISABLE_CACHE"] = "1"
import torch, triton
import tilelang, tvm
from poc.triton_frontend import specialize_static_shape, int32_index_safe, INT32_ELEM_LIMIT

PF_JSON = sys.argv[1] if len(sys.argv) > 1 else "/tmp/pf__chunk_scan_bwd_dstates.json"
GATE = "all"
if "--gate" in sys.argv:
    GATE = sys.argv[sys.argv.index("--gate") + 1]

_pf_base_json = open(PF_JSON).read()


def load_base():
    return tvm.ir.load_json(_pf_base_json)


from mamba_ssm.ops.triton.ssd_chunk_scan import _chunk_scan_bwd_dstates_kernel as _autok
_jit = getattr(_autok, "fn", _autok)
BM, BN, BK = 64, 64, 32
dev = "cuda"


def make_tensors(b, nh, hd, ds, ng, s, cs):
    nc = s // cs
    dout = torch.randn(b, s, nh, hd, device=dev, dtype=torch.float32)
    C = torch.randn(b, s, ng, ds, device=dev, dtype=torch.float32)
    dA = torch.randn(b, nh, nc, cs, device=dev, dtype=torch.float32)
    dprev = torch.zeros(b, nc, nh, hd, ds, device=dev, dtype=torch.float32)
    return dout, C, dA, dprev, nc


def native_ref(dout, C, dA, b, nh, hd, ds, ng, s, cs):
    nc = s // cs
    native = torch.empty(b, nc, nh, hd, ds, device=dev, dtype=torch.float32)
    grid = (triton.cdiv(hd, BM) * triton.cdiv(ds, BN), b * nc, nh)
    _jit[grid](dout, C, native, dA, None, hd, ds, cs, b, s, nc, nh // ng,
        dout.stride(0), dout.stride(1), dout.stride(2), dout.stride(3),
        C.stride(0), C.stride(1), C.stride(2), C.stride(3),
        native.stride(0), native.stride(1), native.stride(2), native.stride(3), native.stride(4),
        dA.stride(0), dA.stride(2), dA.stride(1), dA.stride(3), 0, 0,
        HAS_SEQ_IDX=False, BLOCK_SIZE_M=BM, BLOCK_SIZE_N=BN, BLOCK_SIZE_K=BK)
    torch.cuda.synchronize()
    return native


def build_args(dout, C, dA, dprev, b, nh, hd, ds, ng, s, cs):
    nc = s // cs
    gd0 = triton.cdiv(hd, BM) * triton.cdiv(ds, BN); gd1 = b * nc; gd2 = nh
    sd = [int(x) for x in dout.stride()]; sc = [int(x) for x in C.stride()]
    sp = [int(x) for x in dprev.stride()]; sa = [int(x) for x in dA.stride()]
    scalar_values = {
        "arg5": hd, "arg6": ds, "arg7": cs, "arg8": b, "arg9": s,
        "arg10": nc, "arg11": nh // ng,
        "arg12": sd[0], "arg13": sd[1], "arg14": sd[2], "arg15": sd[3],
        "arg16": sc[0], "arg17": sc[1], "arg18": sc[2], "arg19": sc[3],
        "arg20": sp[0], "arg21": sp[1], "arg22": sp[2], "arg23": sp[3], "arg24": sp[4],
        "arg25": sa[0], "arg26": sa[2], "arg27": sa[1], "arg28": sa[3],
        "arg29": 0, "arg30": 0,
        "gridDim_2": gd2, "gridDim_1": gd1, "gridDim_0": gd0,
    }
    numels = {"arg0": dout.numel(), "arg1": C.numel(),
              "arg2": dprev.numel(), "arg3": dA.numel()}
    runtime_args = [dout.reshape(-1), C.reshape(-1), dprev.reshape(-1), dA.reshape(-1),
        torch.zeros(1, device=dev, dtype=torch.float32),
        hd, ds, cs, b, s, nc, nh // ng,
        sd[0], sd[1], sd[2], sd[3], sc[0], sc[1], sc[2], sc[3],
        sp[0], sp[1], sp[2], sp[3], sp[4], sa[0], sa[2], sa[1], sa[3], 0, 0,
        gd2, gd2, gd1, gd1, gd0, gd0]
    return scalar_values, numels, runtime_args


def call_kernel(kernel, runtime_args, gd_tail):
    try:
        kernel(*runtime_args); torch.cuda.synchronize()
    except ValueError as e:
        if "expected 34" in str(e) or "expected 3" in str(e):
            a = runtime_args[:-6] + gd_tail
            kernel(*a); torch.cuda.synchronize()
        else:
            raise


def sass_and_regs(kernel, tag):
    src = kernel.get_kernel_source()
    n64 = src.count("int64_t")
    cu = "/tmp/i32g_%s.cu" % tag
    open(cu, "w").write(src)
    # SASS
    try:
        sp_path = "/tmp/i32g_%s.sass" % tag
        kernel.export_sass(sp_path); sass = open(sp_path).read()
    except Exception:
        sass = ""
    iw = len(re.findall(r"IMAD\.WIDE", sass)); lh = len(re.findall(r"LEA\.HI", sass))
    hmma = len(re.findall(r"\bHMMA", sass))
    # ptxas regs
    regs = spill = stack = "?"
    try:
        root = "/home/dave/source/tilelang"
        out = subprocess.run(
            ["/usr/local/cuda/bin/nvcc", "-arch=sm_121a", "--ptxas-options=-v",
             "-cubin", "-std=c++17", "--expt-relaxed-constexpr",
             "-I", root + "/src", "-I", root + "/3rdparty/cutlass/include",
             "-DENABLE_BF16", "-o", "/tmp/i32g_%s.cubin" % tag, cu],
            capture_output=True, text=True, timeout=300)
        v = out.stderr + out.stdout
        r = re.findall(r"Used (\d+) registers", v); regs = r[0] if r else "?"
        sp_ = re.findall(r"(\d+) bytes spill stores", v); spill = sp_[0] if sp_ else "0"
        st = re.findall(r"(\d+) bytes stack frame", v); stack = st[0] if st else "0"
    except Exception as e:
        regs = "ERR:%s" % e
    return dict(int64_t=n64, imad_wide=iw, lea_hi=lh, hmma=hmma,
                regs=regs, spill=spill, stack=stack)


def run_prod():
    print("\n########## GATE: prod P1 ##########")
    b, nh, hd, ds, cs, ng, s = 1, 112, 64, 64, 64, 8, 4096
    torch.manual_seed(0)
    dout, C, dA, dprev, nc = make_tensors(b, nh, hd, ds, ng, s, cs)
    native = native_ref(dout, C, dA, b, nh, hd, ds, ng, s, cs)
    sv, numels, rargs = build_args(dout, C, dA, dprev, b, nh, hd, ds, ng, s, cs)
    counts = list(numels.values())
    safe = int32_index_safe(counts)
    print("element_counts=%s int32_safe=%s (limit=%d)" % (counts, safe, INT32_ELEM_LIMIT))
    assert safe, "prod P1 must be int32-safe"
    gd2 = nh; gd1 = b * nc; gd0 = triton.cdiv(hd, BM) * triton.cdiv(ds, BN)
    gd_tail = [gd2, gd1, gd0]

    pf_base = load_base()
    k_base = tilelang.compile(pf_base, target="cuda")
    pf_i32 = specialize_static_shape(pf_base, scalar_values=sv, buffer_element_counts=numels)
    k_i32 = tilelang.compile(pf_i32, target="cuda")

    mb = sass_and_regs(k_base, "prod_base")
    mi = sass_and_regs(k_i32, "prod_i32")
    print("BASE : int64_t=%(int64_t)d IMAD.WIDE=%(imad_wide)d LEA.HI=%(lea_hi)d HMMA=%(hmma)d regs=%(regs)s spill=%(spill)s stack=%(stack)s" % mb)
    print("INT32: int64_t=%(int64_t)d IMAD.WIDE=%(imad_wide)d LEA.HI=%(lea_hi)d HMMA=%(hmma)d regs=%(regs)s spill=%(spill)s stack=%(stack)s" % mi)

    # parity int32
    dprev.zero_(); call_kernel(k_i32, rargs, [gd2, gd2, gd1, gd1, gd0, gd0])
    md_i32 = float((dprev - native).abs().max())
    print("INT32 PARITY MAXDIFF=%.6e BITEXACT=%s" % (md_i32, md_i32 == 0.0))
    # parity base
    dprev.zero_(); call_kernel(k_base, rargs, [gd2, gd2, gd1, gd1, gd0, gd0])
    md_b = float((dprev - native).abs().max())
    print("BASE  PARITY MAXDIFF=%.6e BITEXACT=%s" % (md_b, md_b == 0.0))

    def timed(kernel, n):
        for _ in range(10):
            call_kernel(kernel, rargs, [gd2, gd2, gd1, gd1, gd0, gd0])
        torch.cuda.synchronize(); t0 = time.time()
        for _ in range(n):
            call_kernel(kernel, rargs, [gd2, gd2, gd1, gd1, gd0, gd0])
        torch.cuda.synchronize()
        return (time.time() - t0) / n * 1000
    for r in range(3):
        tb = timed(k_base, 50); ti = timed(k_i32, 50)
        print("rep%d ROUTED base=%.4fms int32=%.4fms speedup=%.4fx" % (r, tb, ti, tb / ti))
    return md_i32 == 0.0


def run_partmask():
    print("\n########## GATE: 6 partial-mask ##########")
    SHAPES = [
        ("hd100_ds72", 100, 72, 64), ("hd64_ds48", 64, 48, 64),
        ("hd48_ds40_cs48", 48, 40, 48), ("hd40_ds64", 40, 64, 64),
        ("hd128_ds128", 128, 128, 64), ("hd100_ds72_cs48", 100, 72, 48),
    ]
    allpass = True
    for name, hd, ds, cs in SHAPES:
        torch.manual_seed(0)
        b, nh, ng = 1, 8, 2
        seq = 256; nc = seq // cs; seq = nc * cs
        dout, C, dA, dprev, _ = make_tensors(b, nh, hd, ds, ng, seq, cs)
        native = native_ref(dout, C, dA, b, nh, hd, ds, ng, seq, cs)
        sv, numels, rargs = build_args(dout, C, dA, dprev, b, nh, hd, ds, ng, seq, cs)
        safe = int32_index_safe(list(numels.values()))
        gd0 = triton.cdiv(hd, BM) * triton.cdiv(ds, BN); gd1 = b * nc; gd2 = nh
        pf_i32 = specialize_static_shape(load_base(), scalar_values=sv, buffer_element_counts=numels)
        k_i32 = tilelang.compile(pf_i32, target="cuda")
        dprev.zero_(); call_kernel(k_i32, rargs, [gd2, gd2, gd1, gd1, gd0, gd0])
        md = float((dprev - native).abs().max())
        nz_r = int((dprev.abs() > 0).sum()); nz_n = int((native.abs() > 0).sum())
        be = (md == 0.0) and (nz_r == nz_n)
        allpass = allpass and be
        print("[%-16s] int32_safe=%s nz_r=%d nz_n=%d MAXDIFF=%.3e BITEXACT=%s -> %s"
              % (name, safe, nz_r, nz_n, md, be, "PASS" if be else "FAIL"))
    print("PARTMASK SUMMARY: %s" % ("ALL PASS" if allpass else "FAIL"))
    return allpass


def run_int64_large(label, s):
    b, nh, hd, ds, cs, ng = 1, 112, 64, 64, 64, 8
    nc = s // cs
    out_elems = b * nc * nh * hd * ds
    dout_elems = b * s * nh * hd
    print("\n########## GATE: %s (s=%d nc=%d) ##########" % (label, s, nc))
    torch.manual_seed(0)
    dout, C, dA, dprev, _ = make_tensors(b, nh, hd, ds, ng, s, cs)
    native = native_ref(dout, C, dA, b, nh, hd, ds, ng, s, cs)
    sv, numels, rargs = build_args(dout, C, dA, dprev, b, nh, hd, ds, ng, s, cs)
    counts = list(numels.values())
    safe = int32_index_safe(counts)
    print("element_counts max=%d int32_safe=%s dstates_GB=%.2f dout_GB=%.2f"
          % (max(counts), safe, out_elems * 4 / 1e9, dout_elems * 4 / 1e9))
    gd0 = triton.cdiv(hd, BM) * triton.cdiv(ds, BN); gd1 = b * nc; gd2 = nh
    if safe:
        pf = specialize_static_shape(load_base(), scalar_values=sv, buffer_element_counts=numels)
        route = "INT32"
    else:
        pf = load_base()
        route = "INT64(symbolic)"
    k = tilelang.compile(pf, target="cuda")
    n64 = k.get_kernel_source().count("int64_t")
    dprev.zero_(); call_kernel(k, rargs, [gd2, gd2, gd1, gd1, gd0, gd0])
    md = float((dprev - native).abs().max())
    print("ROUTE=%s int64_t_in_cu=%d MAXDIFF=%.6e BITEXACT=%s" % (route, n64, md, md == 0.0))
    return md == 0.0, route


def run_safety_dispatch():
    # Synthetic >2^31-element shape: confirm the GUARD routes to int64 (no compile/run,
    # just prove the dispatch decision; running 8.6GB+ may OOM the device).
    print("\n########## GATE: >2^31-element safety dispatch ##########")
    # 8.59GB f32 boundary: choose element counts straddling 2^31.
    below = [2**31 - 1, 100, 100, 100]
    at = [2**31, 100, 100, 100]
    above = [3 * 10**9, 100, 100, 100]
    print("counts=%r -> int32_safe=%s (expect True)" % (below, int32_index_safe(below)))
    print("counts=%r -> int32_safe=%s (expect False)" % (at, int32_index_safe(at)))
    print("counts=%r -> int32_safe=%s (expect False)" % (above, int32_index_safe(above)))
    ok = int32_index_safe(below) and (not int32_index_safe(at)) and (not int32_index_safe(above))
    # And prove: when NOT safe, the dispatch leaves the symbolic int64 kernel (compile it,
    # confirm it still carries int64 addressing).
    k = tilelang.compile(load_base(), target="cuda")
    n64 = k.get_kernel_source().count("int64_t")
    print("symbolic-int64 kernel int64_t_in_cu=%d (must be >0, addressing stays int64)" % n64)
    ok = ok and (n64 > 0)
    print("SAFETY DISPATCH: %s" % ("PASS" if ok else "FAIL"))
    return ok


if __name__ == "__main__":
    results = {}
    if GATE in ("prod", "all"):
        results["prod"] = run_prod()
    if GATE in ("partmask", "all"):
        results["partmask"] = run_partmask()
    if GATE in ("int64", "all"):
        r1, _ = run_int64_large("int64 2.82GB", 98304)   # nc=1536, dstates 2.82GB
        results["int64_2.82GB"] = r1
        r2, _ = run_int64_large("int64 4.35GB", 151552)  # nc=2368, dstates ~4.35GB
        results["int64_4.35GB"] = r2
    if GATE in ("safety", "all"):
        results["safety"] = run_safety_dispatch()
    print("\n===== GATE SUMMARY =====")
    for k_, v in results.items():
        print("%-18s %s" % (k_, "PASS" if v else "FAIL"))
    sys.exit(0 if all(results.values()) else 1)
