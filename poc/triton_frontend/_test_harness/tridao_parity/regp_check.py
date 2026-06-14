"""RegPressure A/B: compile dstates PrimFunc JSON, dump .cu, extract launch_bounds,
ptxas reg count, single-tile parity MAXDIFF, routed ms. Cache OFF (env).

usage: regp_check.py <json_path> <tag>
"""
import sys, os, time, subprocess, re
sys.path.insert(0, "/home/dave/source/tilelang")
os.environ["TILELANG_DISABLE_CACHE"] = "1"
import torch, triton
import tilelang, tvm

json_path = sys.argv[1]
tag = sys.argv[2]

js = open(json_path).read()
pf = tvm.ir.load_json(js)
kernel = tilelang.compile(pf, target="cuda")
src = kernel.get_kernel_source()
cu_path = "/tmp/regp_%s.cu" % tag
open(cu_path, "w").write(src)

# launch_bounds extraction
lb = re.findall(r"__launch_bounds__\(([^)]*)\)", src)
print("TAG=%s" % tag)
print("LAUNCH_BOUNDS=%s" % (lb if lb else "NONE"))

# ptxas reg count: compile the .cu to cubin with ptxas verbose
# find the kernel's global function name
gname = re.findall(r"__global__ void[^(]*?(\w+)\s*\(", src)
print("GLOBAL_FUNCS=%s" % gname)

# nvcc -> ptxas -v to get regs. sm_121a.
try:
    _root = "/home/dave/source/tilelang"
    out = subprocess.run(
        ["/usr/local/cuda/bin/nvcc", "-arch=sm_121a", "--ptxas-options=-v", "-cubin",
         "-std=c++17", "--expt-relaxed-constexpr",
         "-I", _root + "/src",
         "-I", _root + "/3rdparty/cutlass/include",
         "-DENABLE_BF16",
         "-o", "/tmp/regp_%s.cubin" % tag, cu_path],
        capture_output=True, text=True, timeout=300)
    verbose = out.stderr + out.stdout
    regs = re.findall(r"Used (\d+) registers", verbose)
    spills = re.findall(r"(\d+) bytes spill stores", verbose)
    stack = re.findall(r"(\d+) bytes stack frame", verbose)
    print("PTXAS_REGS=%s" % regs)
    print("PTXAS_SPILL_STORES=%s" % spills)
    print("PTXAS_STACK=%s" % stack)
    if out.returncode != 0:
        print("NVCC_RC=%d" % out.returncode)
        print("NVCC_STDERR_TAIL=%s" % verbose[-1500:])
except Exception as e:
    print("PTXAS_EXTRACT_ERR=%r" % e)

# single-tile parity (grid 1,1,1) MAXDIFF vs native, plus routed ms at full grid
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
dprev = torch.zeros(b, nc, nh, hd, ds, device=dev, dtype=torch.float32)
sd = [int(x) for x in dout.stride()]; sc = [int(x) for x in C.stride()]
sp = [int(x) for x in dprev.stride()]; sa = [int(x) for x in dA.stride()]
args = [dout.reshape(-1), C.reshape(-1), dprev.reshape(-1), dA.reshape(-1),
        torch.zeros(1, device=dev, dtype=torch.float32),
        hd, ds, cs, b, s, nc, nh // ng,
        sd[0], sd[1], sd[2], sd[3], sc[0], sc[1], sc[2], sc[3],
        sp[0], sp[1], sp[2], sp[3], sp[4], sa[0], sa[2], sa[1], sa[3], 0, 0,
        gd2, gd1, gd0]
# kernel ABI may expect duplicated gridDim params (37) or deduped (34).
# Probe the adapter's expected_inputs and pad the gridDim tail to match.
try:
    expected = kernel.torch_function.__self__  # adapter
except Exception:
    expected = None
n_have = len(args)
# brute: try 34 first, on mismatch pad with extra gridDim copies (37)
import inspect
def _try_call(a):
    kernel(*a); torch.cuda.synchronize()
try:
    _try_call(list(args))
except ValueError as e:
    msg = str(e)
    m = re.search(r"expected (\d+) inputs", msg)
    if m and int(m.group(1)) == 37:
        # duplicated-gridDim ABI: tail becomes gd2,gd2,gd1,gd1,gd0,gd0
        args = args[:-3] + [gd2, gd2, gd1, gd1, gd0, gd0]
        _try_call(list(args))
    else:
        raise
routed = dprev
md = float((routed - native).abs().max())
ac = torch.allclose(routed, native, atol=1e-3, rtol=1e-3)
print("MAXDIFF=%.6e" % md)
print("ALLCLOSE_1e-3=%s" % ac)
print("PARITY=%s" % ("PASS" if ac else "FAIL"))
# routed ms (median of repeats, cuda events)
def timed(n):
    for _ in range(10): kernel(*args)
    torch.cuda.synchronize(); ts = []
    for _ in range(n):
        st = torch.cuda.Event(enable_timing=True); en = torch.cuda.Event(enable_timing=True)
        st.record(); kernel(*args); en.record(); torch.cuda.synchronize()
        ts.append(st.elapsed_time(en))
    ts.sort(); return ts[len(ts) // 2]
m = timed(60)
print("ROUTED_MS=%.5f" % m)
