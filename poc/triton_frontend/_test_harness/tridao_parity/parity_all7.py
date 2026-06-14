"""
PROVE PARITY: all 7 Tri-Dao bwd kernels, routed (tilelang->tvm) vs NATIVE triton.

Honest, no fabrication. For each kernel:
  1. Patch Autotuner.run to CAPTURE the exact materialised positional args + the
     autotune-selected constexpr config + the launch grid.
  2. Drive native via its real mamba_ssm wrapper -> NATIVE reference output (the
     captured OUTPUT tensor, the one the native kernel wrote into).
  3. Reconstruct ROUTED args from the SAME captured input tensors (so inputs are
     identical), with a FRESH zero output buffer of identical shape; drop constexprs
     (baked into the routed PrimFunc); append gridDim params (walker convention
     [gd1, gd2, gd0, 1]).
  4. Compare routed-out vs native-out: MAXDIFF + torch.allclose(1e-3).

REAL strided: cs != BLOCK_K so multi-K-trip; never strides=0 degenerate.
RULE#1: a failing kernel RAISES its exact defect; no fabricated PASS.
"""
import sys, os, time
sys.path.insert(0, "/home/dave/source/tilelang")
import torch, triton
import tilelang, tvm
from mamba_ssm.ops.triton import ssd_chunk_scan as MSCAN
from mamba_ssm.ops.triton import ssd_chunk_state as MSTATE

MODE = sys.argv[1] if len(sys.argv) > 1 else "small"
ONLY = sys.argv[2] if len(sys.argv) > 2 else None
dev = "cuda"
torch.manual_seed(0)

if MODE == "prod":
    batch, nheads, headdim, dstate = 1, 112, 64, 64
    chunk_size, ngroups = 64, 8
    seqlen = 4096
else:
    batch, nheads, headdim, dstate = 1, 8, 64, 64
    chunk_size, ngroups = 64, 2
    seqlen = 256
nchunks = seqlen // chunk_size

import triton as _triton
from triton.runtime.autotuner import Autotuner
_orig_run = Autotuner.run
CAP = {}

# FORCE the routed-JSON block config (ttir7 was baked BM=64,BN=64,BK=32). The native
# autotuner must use the SAME tile sizes or buffer extents/grid mismatch the routed
# PrimFunc -> segfault. Pin each kernel's autotune to one matching config.
PIN = {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32}

def pin_kernel(k):
    keys = list(k.configs[0].kwargs.keys())
    nw = getattr(k.configs[0], "num_warps", 4)
    ns = getattr(k.configs[0], "num_stages", 2)
    kw = {kk: PIN.get(kk, 64) for kk in keys}  # default 64 for any block key not in PIN (e.g. ddAcs BLOCK_SIZE_M absent)
    k.configs = [_triton.Config(kw, num_warps=nw, num_stages=ns)]
    return kw

def _patched_run(self, *args, **kwargs):
    # capture grid BEFORE native launch (KernelInterface stores self.grid via __getitem__)
    out = _orig_run(self, *args, **kwargs)
    try:
        bc = self.best_config
        meta = dict(bc.kwargs) if bc is not None else {}
    except Exception:
        meta = {}
    if "args" not in CAP:  # capture FIRST (outermost) launch only
        CAP["args"] = args
        CAP["kwargs"] = dict(kwargs)
        CAP["meta"] = meta
        CAP["arg_names"] = self.fn.arg_names
    return out

Autotuner.run = _patched_run

# capture grid: patch KernelInterface.__getitem__
from triton.runtime.jit import KernelInterface
_orig_getitem = KernelInterface.__getitem__
def _patched_getitem(self, grid):
    runner = _orig_getitem(self, grid)
    CAP["grid_obj"] = grid
    return runner
KernelInterface.__getitem__ = _patched_getitem

def mk(*shape):
    return torch.randn(*shape, device=dev, dtype=torch.float32)

def native_dstates():
    C = mk(batch, seqlen, ngroups, dstate); dA = mk(batch, nheads, nchunks, chunk_size)
    dout = mk(batch, seqlen, nheads, headdim)
    return MSCAN._chunk_scan_bwd_dstates(C, dA, dout)

def native_dc():
    prev = mk(batch, nchunks, nheads, headdim, dstate); dA = mk(batch, nheads, nchunks, chunk_size)
    dout = mk(batch, seqlen, nheads, headdim); C = mk(batch, seqlen, ngroups, dstate)
    return MSCAN._chunk_scan_bwd_dC(prev, dA, dout, C=C, ngroups=ngroups)

def native_dcb():
    # _chunk_scan_bwd_dcb(x, dt, dA_cumsum, dout, seq_idx=None, CB=None, ngroups=1)
    x = mk(batch, seqlen, nheads, headdim); dout = mk(batch, seqlen, nheads, headdim)
    cb = mk(batch, nchunks, ngroups, chunk_size, chunk_size); dt = mk(batch, nheads, nchunks, chunk_size)
    dA = mk(batch, nheads, nchunks, chunk_size)
    return MSCAN._chunk_scan_bwd_dcb(x, dt, dA, dout, CB=cb, ngroups=ngroups)

def native_dx():
    # _chunk_scan_bwd_dx(cb, x, dt, dA_cumsum, dout, D=None)
    x = mk(batch, seqlen, nheads, headdim); cb = mk(batch, nchunks, ngroups, chunk_size, chunk_size)
    dout = mk(batch, seqlen, nheads, headdim); dt = mk(batch, nheads, nchunks, chunk_size)
    dA = mk(batch, nheads, nchunks, chunk_size)
    return MSCAN._chunk_scan_bwd_dx(cb, x, dt, dA, dout)

def native_state_db():
    x = mk(batch, seqlen, nheads, headdim); dstates = mk(batch, nchunks, nheads, headdim, dstate)
    dt = mk(batch, nheads, nchunks, chunk_size); dA = mk(batch, nheads, nchunks, chunk_size)
    b = mk(batch, seqlen, ngroups, dstate)
    return MSTATE._chunk_state_bwd_db(x, dt, dA, dstates, B=b, ngroups=ngroups)

def native_state_dx():
    x = mk(batch, seqlen, nheads, headdim); b = mk(batch, seqlen, ngroups, dstate)
    dstates = mk(batch, nchunks, nheads, headdim, dstate); dt = mk(batch, nheads, nchunks, chunk_size)
    dA = mk(batch, nheads, nchunks, chunk_size)
    return MSTATE._chunk_state_bwd_dx(b, x, dt, dA, dstates)

def native_state_ddAcs():
    x = mk(batch, seqlen, nheads, headdim); b = mk(batch, seqlen, ngroups, dstate)
    dstates = mk(batch, nchunks, nheads, headdim, dstate); dt = mk(batch, nheads, nchunks, chunk_size)
    dA = mk(batch, nheads, nchunks, chunk_size)
    return MSTATE._chunk_state_bwd_ddAcs_stable(b, x, dt, dA, dstates)

# output ptr param name per kernel
OUT_PTR = {
    "_chunk_scan_bwd_dstates": "dprev_states_ptr",
    "_chunk_scan_bwd_dc": "dc_ptr",
    "_chunk_scan_bwd_dcb": "dcb_ptr",
    "_chunk_scan_bwd_dx": "dx_ptr",
    "_chunk_state_bwd_db": "db_ptr",
    "_chunk_state_bwd_dx": "dx_ptr",
    "_chunk_state_bwd_ddAcs_stable": None,  # writes ddA via atomic; output is return tensor
}
KERNELS = {
    "_chunk_scan_bwd_dstates": native_dstates,
    "_chunk_scan_bwd_dc": native_dc,
    "_chunk_scan_bwd_dcb": native_dcb,
    "_chunk_scan_bwd_dx": native_dx,
    "_chunk_state_bwd_db": native_state_db,
    "_chunk_state_bwd_dx": native_state_dx,
    "_chunk_state_bwd_ddAcs_stable": native_state_ddAcs,
}
KOBJ = {
    "_chunk_scan_bwd_dstates": MSCAN._chunk_scan_bwd_dstates_kernel,
    "_chunk_scan_bwd_dc": MSCAN._chunk_scan_bwd_dc_kernel,
    "_chunk_scan_bwd_dcb": MSCAN._chunk_scan_bwd_dcb_kernel,
    "_chunk_scan_bwd_dx": MSCAN._chunk_scan_bwd_dx_kernel,
    "_chunk_state_bwd_db": MSTATE._chunk_state_bwd_db_kernel,
    "_chunk_state_bwd_dx": MSTATE._chunk_state_bwd_dx_kernel,
    "_chunk_state_bwd_ddAcs_stable": MSTATE._chunk_state_bwd_ddAcs_stable_kernel,
}

def is_constexpr(nm):
    return nm.isupper() or nm.startswith("HAS_") or nm.startswith("BLOCK_") or nm.startswith("IS_") or nm.startswith("USE_")

def run_one(name, native_fn):
    print("=========================================================", flush=True)
    print("KERNEL %s MODE=%s cfg b%d nh%d hd%d ds%d nc%d cs%d ng%d s%d" % (
        name, MODE, batch, nheads, headdim, dstate, nchunks, chunk_size, ngroups, seqlen), flush=True)
    for k in list(CAP): CAP.pop(k)
    pinned = pin_kernel(KOBJ[name])
    print("PINNED native config=%s" % pinned, flush=True)
    try:
        native_out = native_fn()
    except Exception as e:
        import traceback; traceback.print_exc()
        print("NATIVE_FAIL %s: %s" % (name, str(e)[:200]), flush=True)
        return ("NATIVE_FAIL", float("nan"), False)
    torch.cuda.synchronize()
    if "args" not in CAP:
        print("NO_CAPTURE %s" % name, flush=True)
        return ("NO_CAPTURE", float("nan"), False)
    arg_names = CAP["arg_names"]; meta = CAP["meta"]
    pos = list(CAP["args"]); named = {}
    for i, v in enumerate(pos):
        if i < len(arg_names): named[arg_names[i]] = v
    named.update(CAP["kwargs"])
    for nm in arg_names:
        if nm not in named: named[nm] = meta.get(nm, 0)
    # native output tensor (captured input buffer the kernel wrote into)
    opn = OUT_PTR[name]
    if opn is not None and isinstance(named.get(opn), torch.Tensor):
        native_ref = named[opn].clone()
        out_buf = torch.zeros_like(named[opn]).reshape(-1)
        named_routed = dict(named); named_routed[opn] = None  # placeholder, replaced below
    else:
        native_ref = native_out if isinstance(native_out, torch.Tensor) else native_out[0]
        out_buf = None
    # grid
    grid_obj = CAP.get("grid_obj")
    if callable(grid_obj): grid = grid_obj(meta)
    else: grid = grid_obj
    g0 = grid[0]; g1 = grid[1] if len(grid) > 1 else 1; g2 = grid[2] if len(grid) > 2 else 1
    # routed compile
    js = open("/tmp/pf_%s.json" % name).read()
    pf = tvm.ir.load_json(js)
    kernel = tilelang.compile(pf, target="cuda")
    nparams = len(kernel.params)
    # Determine declared buffer extents from the routed PrimFunc, substituting grid dims.
    # Symbolic extents look like '4096 * gridDim_1 * gridDim_2 * gridDim_0 * gridDim_0_1'
    # or 'carry_index_NNN_numel_MMM' (index/gather tensors). Allocate each buffer at the
    # declared extent and PACK real native data so the routed gather reads valid memory.
    bm = pf.buffer_map
    subs = {"gridDim_0": g0, "gridDim_1": g1, "gridDim_2": g2, "gridDim_0_1": 1}
    def declared_numel(param):
        if param not in bm:
            return None
        b = bm[param]
        n = 1
        for s in b.shape:
            try:
                n *= int(s)
            except Exception:
                # symbolic: evaluate by string substitution of known grid dims; unknown
                # carry_index_*_numel -> use a generous pad (4096*G)
                expr = str(s)
                val = None
                # multiplicative term parse
                import re
                toks = [t.strip() for t in expr.split("*")]
                acc = 1; ok = True
                for t in toks:
                    if t.isdigit(): acc *= int(t)
                    elif t in subs: acc *= subs[t]
                    else: ok = False
                if ok: val = acc
                if val is None:
                    val = 4096 * max(1, g0) * max(1, g1) * max(1, g2)
                n *= val
        return n
    # build routed runtime args, packing each tensor into a buffer of declared extent
    routed = []
    pidx = 0
    out_param = None
    for nm in arg_names:
        if is_constexpr(nm): continue
        param = pf.params[pidx] if pidx < nparams else None
        dn = declared_numel(param) if param is not None else None
        v = named[nm]
        if nm == opn and out_buf is not None:
            sz = dn if dn else out_buf.numel()
            buf = torch.zeros(max(sz, out_buf.numel()), device=dev, dtype=torch.float32)
            routed.append(buf)
            out_param = (pidx, buf, named[opn].shape, named[opn].numel())
        elif isinstance(v, torch.Tensor):
            flat = v.reshape(-1)
            sz = dn if (dn and dn >= flat.numel()) else flat.numel()
            buf = torch.zeros(sz, device=dev, dtype=torch.float32)
            buf[:flat.numel()] = flat
            routed.append(buf)
        elif v is None:
            routed.append(torch.zeros(max(1, dn or 1), device=dev, dtype=torch.float32))
        else:
            routed.append(int(v))
        pidx += 1
    ngd = nparams - len(routed)
    print("ROUTED nparams=%d routed_runtime=%d gridDims_expected=%d meta=%s grid=%s" % (
        nparams, len(routed), ngd, {k: meta[k] for k in meta if is_constexpr(k)}, (g0, g1, g2)), flush=True)
    if ngd < 0:
        print("MARSHAL_FAIL %s ngd=%d -- routed wants FEWER params than native non-constexpr" % (name, ngd), flush=True)
        return ("MARSHAL_FAIL ngd<0", float("nan"), False)
    # Canonical PrimFunc grid-param TAIL order is (gridDim_2, gridDim_1,
    # gridDim_0): axis2<-g2, axis1<-g1, axis0<-g0. Passing [g1,g2,g0] SWAPS
    # gridDim_2/gridDim_1 -> the strided per-block base for chunk/head axes is
    # computed from the wrong block count, so only a subset of blocks runs and
    # half the output stays zero (the nz=65536/131072 symptom). Align with the
    # §P1 prod harness (which passes gd2,gd1,gd0).
    # GRID-DIM TAIL (robust to repeated gridDim params): read the actual tail
    # param NAMES from the PrimFunc and map each gridDim_N -> its value. The
    # fresh from_ttir emit can declare a grid axis param more than once (e.g.
    # gridDim_2 twice); a fixed [g2,g1,g0,1] list then under/over-fills. Build
    # the trailing values positionally from the real param names instead.
    _gd_val = {"gridDim_2": g2, "gridDim_1": g1, "gridDim_0": g0, "gridDim_0_1": 1}
    tail_params = pf.params[len(routed):]
    trailing = []
    for tp in tail_params:
        tn = getattr(tp, "name", "") or str(tp)
        if tn not in _gd_val:
            raise RuntimeError("unexpected trailing param %r (not a known gridDim)" % tn)
        trailing.append(_gd_val[tn])
    args = routed + trailing
    try:
        kernel(*args); torch.cuda.synchronize()
    except Exception as e:
        print("ROUTED_LAUNCH_FAIL %s: %s: %s" % (name, type(e).__name__, str(e)[:260]), flush=True)
        return ("LAUNCH_FAIL", float("nan"), False)
    if out_buf is None or out_param is None:
        print("NO_OUTBUF %s (atomic/return-only output) -- cannot direct-compare" % name, flush=True)
        return ("NO_OUTBUF", float("nan"), False)
    _, obuf, oshape, onumel = out_param
    routed_out = obuf[:onumel].view(oshape)
    nz_r = int((routed_out.abs() > 0).sum()); nz_n = int((native_ref.abs() > 0).sum())
    md = float((routed_out - native_ref).abs().max())
    ac = torch.allclose(routed_out, native_ref, atol=1e-3, rtol=1e-3)
    print("NATIVE nz=%d/%d sum=%.4f" % (nz_n, native_ref.numel(), float(native_ref.sum())), flush=True)
    print("ROUTED nz=%d/%d sum=%.4f" % (nz_r, routed_out.numel(), float(routed_out.sum())), flush=True)
    print("MAXDIFF=%.6e ALLCLOSE_1e-3=%s -> %s" % (md, ac, "PASS" if ac else "FAIL"), flush=True)
    return ("PASS" if ac else "FAIL", md, ac)

results = {}
items = [(ONLY, KERNELS[ONLY])] if ONLY else list(KERNELS.items())
for name, fn in items:
    results[name] = run_one(name, fn)

npass = sum(1 for s, m, a in results.values() if a)
print("\n===== SUMMARY MODE=%s  PASS=%d/%d =====" % (MODE, npass, len(results)))
for n, (status, md, ac) in results.items():
    print("%-34s %-14s MAXDIFF=%.6e" % (n, status, md))
