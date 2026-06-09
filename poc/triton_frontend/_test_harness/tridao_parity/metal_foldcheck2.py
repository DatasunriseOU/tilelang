"""METAL CODEGEN-ONLY check #2 -- isolate the guard fold from pipeline planning.

The dstates kernel's multi-stage copy loop trips TileLang's Metal `PipelinePlanning`
pass (an orthogonal software-pipelining limitation, present with or without the
fold). To prove the *fold itself* is Metal-portable we lower a folded prim built
from a non-pipelined guarded kernel (`_chunk_state_bwd_ddAcs_stable`, scf.for=0,
80 overflow guards) to a Metal target and confirm MSL source is emitted with the
guard chain folded out. CODEGEN ONLY -- no Apple GPU dispatch (Linux/CUDA host).
"""
import os, sys
sys.path.insert(0, "/home/dave/source/tilelang")
os.environ.pop("TL_FORCE_CP_ASYNC", None)
import tilelang, tvm
from tvm.target import Target
from poc.triton_frontend import from_ttir

TTIR = "/tmp/ttir7/_chunk_state_bwd_ddAcs_stable.ttir"
ttir = open(TTIR).read()
print("input guards(2147483647)=%d cmpi=%d andi=%d extsi=%d" % (
    ttir.count("2147483647"), ttir.count("arith.cmpi"),
    ttir.count("arith.andi"), ttir.count("arith.extsi")), flush=True)
print("metal codegen registered:",
      tvm.get_global_func("target.build.metal", allow_missing=True) is not None, flush=True)

name = "_chunk_state_bwd_ddAcs_stable_kernel"


def lower_to_metal(prologue_opt):
    pf = from_ttir(ttir, name=name, target="cuda",
                   _allow_text_ttir=True, prologue_opt=prologue_opt)
    metal_target = Target("metal")
    with metal_target:
        mod = tilelang.lower(pf, target=metal_target)
    return mod


for tag, popt in [("FOLD(prologue_opt=True)", True)]:
    try:
        mod = lower_to_metal(popt)
        print("%s tilelang.lower(metal) OK type=%s" % (tag, type(mod).__name__), flush=True)
        # Extract MSL source from the device module.
        msl = None
        candidates = []
        rt = getattr(mod, "mod", mod)
        if hasattr(rt, "imported_modules"):
            candidates = [rt] + list(rt.imported_modules)
        elif hasattr(mod, "imported_modules"):
            candidates = [mod] + list(mod.imported_modules)
        for m in candidates:
            try:
                src = m.get_source()
            except Exception:
                src = None
            if src and len(src) > 50:
                msl = src
                break
        if msl is None:
            # Force codegen explicitly via the metal target build func.
            try:
                built = tvm.build(mod, target=Target("metal"))
                for m in [built] + list(getattr(built, "imported_modules", [])):
                    try:
                        src = m.get_source()
                    except Exception:
                        src = None
                    if src and len(src) > 50:
                        msl = src
                        break
            except Exception as e2:
                print("explicit metal build note: %r" % e2, flush=True)
        if msl:
            is_msl = ("kernel" in msl) or ("[[" in msl) or ("metal_stdlib" in msl) or ("device " in msl)
            print("=== MSL SOURCE head (40 lines) for %s ===" % tag, flush=True)
            print("\n".join(msl.splitlines()[:40]), flush=True)
            print("METAL2RESULT MSL_EMITTED=1 MSL_LINES=%d LOOKS_LIKE_MSL=%d HAS_GUARD_CONST_2147483647=%d" % (
                msl.count("\n"), int(is_msl), int("2147483647" in msl)), flush=True)
        else:
            print("METAL2RESULT MSL_EMITTED=0 (lowered ok but no source extracted)", flush=True)
    except Exception as exc:
        import traceback
        traceback.print_exc()
        print("METAL2RESULT MSL_EMITTED=ERROR cause=%r" % exc, flush=True)
print("DONE", flush=True)
