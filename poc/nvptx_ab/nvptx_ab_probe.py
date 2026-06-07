import sys, traceback
import tilelang
import tilelang.language as T
import tvm
from tvm.target import Target

def make_matmul(M,N,K,bM,bN,bK,dtype="float16",acc="float32"):
    @T.prim_func
    def gemm(A: T.Tensor((M,K),dtype), B: T.Tensor((K,N),dtype), C: T.Tensor((M,N),dtype)):
        with T.Kernel(T.ceildiv(N,bN), T.ceildiv(M,bM), threads=128) as (bx,by):
            As=T.alloc_shared((bM,bK),dtype); Bs=T.alloc_shared((bK,bN),dtype)
            Cl=T.alloc_fragment((bM,bN),acc)
            T.clear(Cl)
            for k in T.Pipelined(T.ceildiv(K,bK),num_stages=3):
                T.copy(A[by*bM,k*bK],As); T.copy(B[k*bK,bx*bN],Bs); T.gemm(As,Bs,Cl)
            T.copy(Cl,C[by*bM,bx*bN])
    return gemm

prim = make_matmul(1024,1024,1024,128,128,32)
from tilelang.engine.lower import lower_to_host_device_ir
tgt = tvm.target.Target({"kind":"cuda","arch":"sm_121"})
with tgt:
    host_mod, device_mod, params, target, target_host = lower_to_host_device_ir(prim, target=tgt)
print("=== TARGET ===", target)
print("=== device fns ===", list(device_mod.functions.keys()))
src = str(device_mod)
for tok in ["ptx_ldmatrix","ptx_cp_async","gemm","ptx_mma","mma","ldmatrix","cp_async"]:
    n = src.count(tok)
    if n: print(f"  TIR has '{tok}': {n}x")

print("\n=== ATTEMPT 1: tilelang_cuda (DEFAULT source codegen) ===")
try:
    f = tvm.ffi.get_global_func("target.build.tilelang_cuda")
    cuda_mod = f(device_mod, target)
    cuda_src = cuda_mod.inspect_source()
    print("SOURCE CODEGEN OK len:", len(cuda_src))
    for m in ["#include","tl::","wmma","__global__","ldmatrix","cp.async","mma"]:
        if m in cuda_src: print(f"   src has '{m}'")
except Exception as e:
    print("SOURCE CODEGEN FAILED:", repr(e))

print("\n=== ATTEMPT 2: target.build.nvptx (LLVM NVPTX) on SAME device_mod ===")
try:
    nvptx_target = tvm.target.Target({"kind":"nvptx","mcpu":"sm_121"}, host=tvm.target.Target({"kind":"llvm"}))
    f = tvm.ffi.get_global_func("target.build.nvptx")
    nvptx_mod = f(device_mod, nvptx_target)
    print("NVPTX CODEGEN OK:", nvptx_mod)
    print(nvptx_mod.inspect_source()[:400])
except Exception as e:
    print("NVPTX CODEGEN FAILED (verbatim):")
    tb = traceback.format_exc()
    print(tb[-1500:])
