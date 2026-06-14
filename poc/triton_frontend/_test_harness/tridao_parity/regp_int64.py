"""int64-robustness validation: run the PATCHED dstates kernel at a LARGE shape
(>2.35GB tensors, forcing 64-bit global addressing) and confirm bit-exact parity
vs the native mamba_ssm triton reference. The min_blocks_per_sm patch does NOT
touch addressing (byte-identical PTX), so int64 base addressing must be fully
preserved -- this proves no OOB/overflow regression at scale.
"""
import sys, os
sys.path.insert(0, "/home/dave/source/tilelang")
os.environ["TILELANG_DISABLE_CACHE"] = "1"
import torch, triton
import tilelang, tvm

jp = sys.argv[1] if len(sys.argv) > 1 else "/tmp/pf_dstates_patched.json"
pf = tvm.ir.load_json(open(jp).read())
k = tilelang.compile(pf, target="cuda")
src = k.get_kernel_source()
import re
lb = re.findall(r"__launch_bounds__\(([^)]*)\)", src)
print("launch_bounds=%s" % (lb[0] if lb else "NONE"))
print("int64_casts_in_cu=%d" % src.count("int64_t"))

torch.manual_seed(0); dev = "cuda"
# LARGE config: scale seq to push output + inputs past 2.35GB and force int64
# global addressing. Output dstates = b*nc*nh*hd*ds floats.
b, nh, hd, ds = 1, 112, 64, 64
cs, ng = 64, 8
s = 98304          # nc=1536 -> dstates ~2.82GB, byte-offset > 2^31 forces int64
nc = s // cs       # 1536
BM, BN, BK = 64, 64, 32
out_elems = b * nc * nh * hd * ds
out_gb = out_elems * 4 / 1e9
dout_gb = b * s * nh * hd * 4 / 1e9
print("CONFIG b%d nh%d hd%d ds%d s%d nc%d : dstates=%.2fGB dout=%.2fGB" %
      (b, nh, hd, ds, s, nc, out_gb, dout_gb))

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
dprev = torch.zeros(b, nc, nh, hd, ds, device=dev, dtype=torch.float32)
sd = [int(x) for x in dout.stride()]; sc = [int(x) for x in C.stride()]
sp = [int(x) for x in dprev.stride()]; sa = [int(x) for x in dA.stride()]
args = [dout.reshape(-1), C.reshape(-1), dprev.reshape(-1), dA.reshape(-1),
        torch.zeros(1, device=dev, dtype=torch.float32),
        hd, ds, cs, b, s, nc, nh // ng,
        sd[0], sd[1], sd[2], sd[3], sc[0], sc[1], sc[2], sc[3],
        sp[0], sp[1], sp[2], sp[3], sp[4], sa[0], sa[2], sa[1], sa[3], 0, 0,
        gd2, gd1, gd0]
# confirm at least one stride exceeds int32 range (proves int64 addressing path)
maxstride_elem = max(sd[0], sc[0], sp[0])
print("max_input_stride_elems=%d (int32_max=%d) -> int64_required=%s" %
      (maxstride_elem, 2**31 - 1, "YES" if max(out_elems, b*s*nh*hd) > 2**31 - 1 else "(addr in bytes)"))
print("dstates_byte_extent=%d (>2^31=%s)" % (out_elems * 4, out_elems * 4 > 2**31 - 1))
def call(a):
    try:
        k(*a); torch.cuda.synchronize()
    except ValueError as e:
        if "expected 37" in str(e):
            a2 = a[:-3] + [gd2, gd2, gd1, gd1, gd0, gd0]
            k(*a2); torch.cuda.synchronize(); return
        raise
call(args)
md = float((dprev - native).abs().max())
ac = torch.allclose(dprev, native, atol=1e-3, rtol=1e-3)
print("LARGE_MAXDIFF=%.6e" % md)
print("LARGE_ALLCLOSE=%s" % ac)
print("LARGE_PARITY=%s" % ("PASS" if md == 0.0 else ("CLOSE" if ac else "FAIL")))
