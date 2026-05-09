import tilelang
import tilelang.language as T
import torch

def parallel_dynamic_frag(max_len=512, threads=256, dtype=T.float32):
    @T.prim_func
    def main(
        A: T.Tensor((max_len,), dtype),
        B: T.Tensor((max_len,), dtype),
        valid_len: T.int32,
    ):
        with T.Kernel(1, threads=threads) as _:
            frag = T.alloc_fragment((max_len,), dtype)
            span = T.min(valid_len, max_len)
            for i in T.Parallel(span):
                frag[i] = A[i] - 1.0
            for i in T.Parallel(span):
                B[i] = frag[i]

    return main

func = parallel_dynamic_frag()
try:
    tilelang.lower(func)
    print("Success")
except Exception as e:
    import traceback
    traceback.print_exc()
