"""METAL CODEGEN-ONLY check for the backend-agnostic guard fold.

Proves the SAME folded tilelang/tvm prim (emitter-level prologue_opt guard fold,
NO cp.async / CUDA-only op) lowers to a Metal (MSL) target. CODEGEN-ONLY: we run
the tvm metal codegen pass to emit MSL source text; we DO NOT dispatch on any
Apple GPU (this host is Linux/CUDA -- it physically cannot launch an Apple GPU,
so the check is inherently watchdog-safe).

The point: the guard fold lives in OUR walker (op_emitters/arith.py), is
target-agnostic, and the resulting prim contains no CUDA-only op, so it lowers to
Metal exactly as it lowers to CUDA. If lowering raises, we surface it (no silent
fallback).
"""
import os, sys
sys.path.insert(0, "/home/dave/source/tilelang")
# Build the prim WITHOUT torch/triton in modules and WITHOUT cp.async so the
# walker emits a pure target-agnostic prim. (cp.async is CUDA-only and gated by
# TL_FORCE_CP_ASYNC, which we leave unset.)
os.environ.pop("TL_FORCE_CP_ASYNC", None)
import tilelang, tvm
from tvm.target import Target
from poc.triton_frontend import from_ttir

ttir = open("/tmp/ttir7/_chunk_scan_bwd_dstates.ttir").read()

print("metal codegen registered:",
      tvm.get_global_func("target.build.metal", allow_missing=True) is not None, flush=True)


def build_prim(prologue_opt, target):
    pf = from_ttir(ttir, name="_chunk_scan_bwd_dstates_kernel", target=target,
                   _allow_text_ttir=True, prologue_opt=prologue_opt)
    return pf


# 1. Build the folded prim with target='cuda' (the walker's emit is target-agnostic
#    at the TIR level; the guard fold fires regardless).
pf_fold = build_prim(prologue_opt=True, target="cuda")
print("FOLDED prim built (prologue_opt=True, no cp.async)", flush=True)

# 2. Lower the SAME folded prim to a Metal target -- CODEGEN ONLY (emit MSL text).
metal_target = Target("metal")
try:
    # tilelang.lower produces the lowered IRModule for the requested target;
    # tvm.build with the metal target then runs target.build.metal codegen,
    # which emits MSL source. No device dispatch.
    lowered = tilelang.lower(pf_fold, target=metal_target)
    print("tilelang.lower(metal) OK; type=%s" % type(lowered).__name__, flush=True)
    # Pull the MSL source out of the lowered/built module.
    msl = None
    try:
        rt_mod = lowered if hasattr(lowered, "imported_modules") else getattr(lowered, "mod", None)
        if rt_mod is not None and hasattr(rt_mod, "imported_modules"):
            for m in [rt_mod] + list(rt_mod.imported_modules):
                try:
                    src = m.get_source()
                except Exception:
                    src = None
                if src and ("kernel" in src or "[[" in src or "metal" in src.lower()):
                    msl = src
                    break
    except Exception as e:
        print("MSL extract note: %r" % e, flush=True)
    if msl is None:
        # Fall back to building explicitly to force codegen and capture source.
        built = tvm.build(lowered if hasattr(lowered, "functions") else pf_fold, target=metal_target)
        for m in [built] + list(getattr(built, "imported_modules", [])):
            try:
                src = m.get_source()
            except Exception:
                src = None
            if src:
                msl = src
                break
    if msl:
        print("=== MSL SOURCE (first 60 lines) ===", flush=True)
        print("\n".join(msl.splitlines()[:60]), flush=True)
        print("=== MSL stats: lines=%d, has_kernel_attr=%s, has_2147483647=%s ===" % (
            msl.count("\n"),
            ("kernel" in msl or "[[kernel]]" in msl),
            ("2147483647" in msl)), flush=True)
        print("METALRESULT MSL_EMITTED=1 MSL_LINES=%d MSL_HAS_GUARD_CONST=%d" % (
            msl.count("\n"), int("2147483647" in msl)), flush=True)
    else:
        print("METALRESULT MSL_EMITTED=0 (lowered to metal but no MSL source extracted)", flush=True)
except Exception as exc:
    import traceback
    traceback.print_exc()
    print("METALRESULT MSL_EMITTED=ERROR cause=%r" % exc, flush=True)
print("DONE", flush=True)
