"""RegPressure A/B: baseline (min_blocks=1) vs patched (min_blocks=5), same
TTIR, interleaved timing to cancel drift. Cache OFF. Both parity-checked.
"""
import sys, os, re
sys.path.insert(0, "/home/dave/source/tilelang")
os.environ["TILELANG_DISABLE_CACHE"] = "1"
import torch, triton
import tilelang, tvm

base_json = "/tmp/pf_dstates_baseline.json"
patch_json = "/tmp/pf_dstates_patched.json"

def build(jp):
    pf = tvm.ir.load_json(open(jp).read())
    k = tilelang.compile(pf, target="cuda")
    src = k.get_kernel_source()
    lb = re.findall(r"__launch_bounds__\(([^)]*)\)", src)
    return k, (lb[0] if lb else "NONE")

kb, lbb = build(base_json)
kp, lbp = build(patch_json)
print("BASE launch_bounds=%s" % lbb)
print("PATCH launch_bounds=%s" % lbp)

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

def make_args():
    dprev = torch.zeros(b, nc, nh, hd, ds, device=dev, dtype=torch.float32)
    sd = [int(x) for x in dout.stride()]; sc = [int(x) for x in C.stride()]
    sp = [int(x) for x in dprev.stride()]; sa = [int(x) for x in dA.stride()]
    args = [dout.reshape(-1), C.reshape(-1), dprev.reshape(-1), dA.reshape(-1),
            torch.zeros(1, device=dev, dtype=torch.float32),
            hd, ds, cs, b, s, nc, nh // ng,
            sd[0], sd[1], sd[2], sd[3], sc[0], sc[1], sc[2], sc[3],
            sp[0], sp[1], sp[2], sp[3], sp[4], sa[0], sa[2], sa[1], sa[3], 0, 0,
            gd2, gd1, gd0]
    return dprev, args

def call(k, args):
    try:
        k(*args); torch.cuda.synchronize()
    except ValueError as e:
        if "expected 37" in str(e):
            args2 = args[:-3] + [gd2, gd2, gd1, gd1, gd0, gd0]
            k(*args2); torch.cuda.synchronize()
            return args2
        raise
    return args

# parity both
for tag, k in (("BASE", kb), ("PATCH", kp)):
    dprev, args = make_args()
    args = call(k, args)
    md = float((dprev - native).abs().max())
    print("%s MAXDIFF=%.6e PARITY=%s" % (tag, md, "PASS" if md == 0.0 else "FAIL"))

def timed(k, args, n):
    for _ in range(15): k(*args)
    torch.cuda.synchronize(); ts = []
    for _ in range(n):
        st = torch.cuda.Event(enable_timing=True); en = torch.cuda.Event(enable_timing=True)
        st.record(); k(*args); en.record(); torch.cuda.synchronize()
        ts.append(st.elapsed_time(en))
    ts.sort(); return ts[len(ts) // 2]

_, ab = make_args(); ab = call(kb, ab)
_, ap = make_args(); ap = call(kp, ap)
N = 80
print("=== A/B interleaved (median of %d, 4 reps) ===" % N)
bvals = []; pvals = []
for r in range(4):
    tb = timed(kb, ab, N); tp = timed(kp, ap, N)
    bvals.append(tb); pvals.append(tp)
    print("rep%d BASE=%.5f PATCH=%.5f delta=%+.2f%%" % (r, tb, tp, (tp - tb) / tb * 100))
bm = sorted(bvals)[len(bvals) // 2]; pm = sorted(pvals)[len(pvals) // 2]
print("BASE_MED_MS=%.5f" % bm)
print("PATCH_MED_MS=%.5f" % pm)
print("DELTA_PCT=%+.2f" % ((pm - bm) / bm * 100))
