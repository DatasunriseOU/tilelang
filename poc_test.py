import tilelang as tl
import tilelang.language as T
import tvm
from tilelang.utils.target import determine_target

auto_target = tvm.target.Target(determine_target("auto"))

def make_mod(dtype):
    M, N, K = 64, 64, 64
    block_M, block_N, block_K = 64, 64, 32
    @T.prim_func
    def main(
        A: T.Tensor((M, K), dtype),
        B_t: T.Tensor((N, K), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(1, threads=128) as _:
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_N, block_K), dtype)
            C_local = T.alloc_fragment((block_M, block_N), dtype)
            
            T.clear(C_local)
            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=3):
                T.copy(A[0:block_M, k * block_K : (k+1)*block_K], A_shared)
                T.copy(B_t[0:block_N, k * block_K : (k+1)*block_K], B_shared)
                T.gemm(A_shared, B_shared, C_local, transpose_B=True)
                
            T.copy(C_local, C[0:block_M, 0:block_N])
    return tvm.IRModule({"main": main})

print("Testing float16")
mod_f16 = make_mod(T.float16)
with tvm.target.Target(auto_target):
    mod_f16 = tvm.tir.transform.BindTarget(auto_target)(mod_f16)
    mod_f16 = tl.transform.LayoutInference()(mod_f16)
print("float16 passed")

print("Testing int8")
def make_mod_int8():
    M, N, K = 64, 64, 64
    block_M, block_N, block_K = 64, 64, 32
    @T.prim_func
    def main(
        A: T.Tensor((M, K), T.int8),
        B_t: T.Tensor((N, K), T.int8),
        C: T.Tensor((M, N), T.int32),
    ):
        with T.Kernel(1, threads=128) as _:
            A_shared = T.alloc_shared((block_M, block_K), T.int8)
            B_shared = T.alloc_shared((block_N, block_K), T.int8)
            C_local = T.alloc_fragment((block_M, block_N), T.int32)
            
            T.clear(C_local)
            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=3):
                T.copy(A[0:block_M, k * block_K : (k+1)*block_K], A_shared)
                T.copy(B_t[0:block_N, k * block_K : (k+1)*block_K], B_shared)
                T.gemm(A_shared, B_shared, C_local, transpose_B=True)
                
            T.copy(C_local, C[0:block_M, 0:block_N])
    return tvm.IRModule({"main": main})

mod_i8 = make_mod_int8()
with tvm.target.Target(auto_target):
    mod_i8 = tvm.tir.transform.BindTarget(auto_target)(mod_i8)
    mod_i8 = tl.transform.LayoutInference()(mod_i8)
print("int8 passed")
