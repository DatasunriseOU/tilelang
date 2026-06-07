import tilelang, tilelang.language as T, tvm, re, traceback
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
prim=make_matmul(1024,1024,1024,128,128,32)
from tilelang.engine.lower import lower_to_host_device_ir
tgt=tvm.target.Target({"kind":"cuda","arch":"sm_121"})
with tgt:
    _,device_mod,_,target,_=lower_to_host_device_ir(prim,target=tgt)
src=str(device_mod)
ops=sorted(set(re.findall(r'\btl\.[a-zA-Z0-9_]+', src)))
print("tl.* ops in device TIR that NVPTX must handle (and does NOT):")
for o in ops: print("   ", o, "x"+str(src.count(o)))
ptxops=sorted(set(re.findall(r'\bptx_[a-zA-Z0-9_]+|tvm_mma_sync|tvm_load_matrix_sync|mma_sync', src)))
print("PTX/mma builtins:", ptxops)

# === provenance: confirm DEFAULT jit path produces a working CUDA module ===
print("\n=== DEFAULT jit (source codegen) provenance ===")
try:
    k=tilelang.jit(prim, out_idx=[-1])
    s=k.get_kernel_source()
    print("jit OK, source len", len(s), "has tl::", "tl::" in s, "has __global__", "__global__" in s, "has ldmatrix", "ldmatrix" in s)
except Exception as e:
    print("jit path err:", repr(e)[:200])
