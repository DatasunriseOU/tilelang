import sys, traceback, re
sys.path.insert(0, "/home/dave/source/cppmega_mlx")
import tilelang
import tvm
from cppmega_mlx.nn._tilelang.mamba3_chunked_precompute_core import chunk_precompute_fwd_cuda_prim
prim = chunk_precompute_fwd_cuda_prim(batch=1, seqlen=256, chunk_size=64, ngroups=1, nheads=4, headdim=64, dstate=64)
from tilelang.engine.lower import lower_to_host_device_ir
tgt=tvm.target.Target({"kind":"cuda","arch":"sm_121"})
with tgt:
    _,device_mod,_,target,_=lower_to_host_device_ir(prim,target=tgt)
print("F0 device fns:", list(device_mod.functions.keys()))
print("=== NVPTX on path_c F0 ===")
try:
    nt=tvm.target.Target({"kind":"nvptx","mcpu":"sm_121"}, host=tvm.target.Target({"kind":"llvm"}))
    f=tvm.ffi.get_global_func("target.build.nvptx")
    m=f(device_mod, nt)
    print("NVPTX OK (unexpected)", len(m.inspect_source()))
except Exception as e:
    tb=traceback.format_exc()
    mm=re.search(r'unknown intrinsic.*', tb)
    print("NVPTX FAILED:", mm.group(0)[:200] if mm else tb[-400:])
