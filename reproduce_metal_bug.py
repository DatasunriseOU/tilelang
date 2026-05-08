import tilelang
import tilelang.language as T

@tilelang.jit(out_idx=[-1], target="metal")
def test_simdgroup_add(M, N, K, block_M, block_N, block_K, dtype=T.float16, accum_dtype=T.float32):
    @T.prim_func
    def gemm(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((K, N), dtype),
        C: T.Tensor((M, N), accum_dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_K, block_N), dtype)
            
            c_frag = T.alloc_fragment((block_M, block_N), accum_dtype, scope="local.fragment")
            T.clear(c_frag)

            for k in T.serial(T.ceildiv(K, block_K)):
                T.copy(A[by * block_M, k * block_K], A_shared)
                T.copy(B[k * block_K, bx * block_N], B_shared)
                T.gemm(A_shared, B_shared, c_frag, clear_accum=False)

            # Scalar addition after GEMM (this is what arith.addf would emit)
            c_local = T.alloc_fragment((block_M, block_N), accum_dtype, scope="local")
            for i, j in T.grid(block_M, block_N):
                c_local[i, j] = c_frag[i, j] + 1.0

            # Copy back to global
            T.copy(c_local, C[by * block_M, bx * block_N])

    return gemm

def main():
    kernel = test_simdgroup_add(128, 128, 128, 32, 32, 32)
    print("Metal Source:")
    print(kernel.get_kernel_source())

if __name__ == "__main__":
    main()
