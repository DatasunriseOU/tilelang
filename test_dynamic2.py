import tilelang
import tilelang.language as T
import torch
import tvm
from tilelang import transform

def parallel_elementwise_dynamic(max_len=512, threads=256, dtype=T.float32):
    @T.prim_func
    def main(
        A: T.Tensor((max_len,), dtype),
        B: T.Tensor((max_len,), dtype),
        valid_len: T.int32,
    ):
        with T.Kernel(1, threads=threads) as _:
            for i in T.Parallel(max_len):
                B[i] = 0.0
            span = T.min(valid_len, max_len)
            for i in T.Parallel(span):
                B[i] = A[i] - 1.0

    return main

func = parallel_elementwise_dynamic()
mod = tvm.IRModule.from_expr(func)
mod = transform.FrontendLegalize()(mod)
mod = transform.LayoutInference()(mod)
print(mod.script())
