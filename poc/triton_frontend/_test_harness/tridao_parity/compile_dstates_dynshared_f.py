"""DYNSHARED COMPILE PROBE -- does the routed dstates C-tile TMA kernel
ptxas-COMPILE under sm_121f when its >48KB shared is routed to shared.dyn?

Gates:
  TL_ARCH_SUFFIX=f          -> compile compute_120f FAMILY target (48KB static cap)
  TL_FORCE_DYN_SHARED=1     -> route shared tiles to shared.dyn (DYNAMIC)
  TL_TMA_ROUTE=1            -> ground C-tile to real UTMALDG TMA

This is COMPILE-ONLY (no GPU exec). Prints:
  - static vs dynamic shared bytes in the emitted .cu
  - whether ptxas accepted (tilelang.compile returns a kernel w/ a cubin)
  - the 0xc000 / 48KB reject if it fails
"""
import os, sys, re, traceback
sys.path.insert(0, "/home/dave/source/tilelang")
import tilelang
from poc.triton_frontend import from_ttir

ttir = open("/tmp/ttir7/_chunk_scan_bwd_dstates.ttir").read()

print("TL_ARCH_SUFFIX=%s TL_FORCE_DYN_SHARED=%s TL_TMA_ROUTE=%s" % (
    os.environ.get("TL_ARCH_SUFFIX"), os.environ.get("TL_FORCE_DYN_SHARED"),
    os.environ.get("TL_TMA_ROUTE")), flush=True)


def build(prologue_opt, ground=False):
    kw = {}
    if ground:
        # Grounded C-tile TMA recipe (matches measure_dstates_tma_exec.py which
        # produced tl::tma_load x2): contiguous-innermost %arg1 + async_loads ON.
        kw["contiguous_innermost_sources"] = {"%arg1"}
        kw["async_loads"] = True
    else:
        kw["async_loads"] = False
    pf = from_ttir(ttir, name="_chunk_scan_bwd_dstates_kernel", target="cuda",
                   _allow_text_ttir=True, prologue_opt=prologue_opt, **kw)
    return tilelang.compile(pf, target="cuda")


try:
    print("building OPT (grounded TMA, dyn-shared route)...", flush=True)
    k_opt = build(True, ground=True)
    print("COMPILE_OK=True", flush=True)
    src = k_opt.get_kernel_source() if hasattr(k_opt, "get_kernel_source") else ""
    open("/tmp/ttir7/dstates_dynshared_f.cu", "w").write(src or "")
    # count tma + extern shared
    print("tma_load_count=%d" % (src.count("tma_load") if src else -1), flush=True)
    print("tl_tma_load_count=%d" % (src.count("tl::tma_load") if src else -1), flush=True)
    print("HAS_prefetch_tma=%s" % ("prefetch_tma" in (src or "")), flush=True)
    print("HAS_extern_shared=%s" % ("extern __shared__" in src), flush=True)
    print("HAS_static_shared=%s" % bool(re.search(r"__shared__ [^\n]*\[\d+\]", src)), flush=True)
    # static shared decls with explicit sizes (these count against 48KB)
    for m in re.finditer(r"(__shared__[^\n;]*\[\d+\][^\n;]*);", src or ""):
        print("  STATIC_SHARED_DECL:", m.group(1).strip()[:120], flush=True)
    print("DONE_COMPILE_PROBE", flush=True)
except Exception as e:
    print("COMPILE_OK=False", flush=True)
    print("EXC:", str(e)[:2000], flush=True)
    traceback.print_exc()
    # surface ptxas 48KB / 0xc000 reject explicitly
    txt = str(e)
    if "0xc000" in txt or "48" in txt or "shared" in txt.lower():
        print("PTXAS_SHARED_REJECT_DETECTED", flush=True)
    sys.exit(1)
