import tilelang as tl
import tilelang.language as T
import tvm

def test_dynamic_parallel():
    @T.prim_func
    def kernel(A: T.Buffer((128,), "float32"), B: T.Buffer((128,), "float32"), valid_len: T.int32):
        with T.Kernel(T.ceildiv(128, 64), threads=64) as bx:
            A_shared = T.alloc_shared((128,), "float32")
            for i in T.Parallel(valid_len):
                A_shared[i] = A[i]

    mod = tl.lower(kernel)
    print(mod.get_source())

test_dynamic_parallel()
