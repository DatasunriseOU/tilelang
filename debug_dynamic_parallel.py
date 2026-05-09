import tilelang
import tilelang.language as T
import tvm

def get_prim_func(max_len=512, threads=256, dtype=T.float32):
    @T.prim_func
    def main(
        A: T.Tensor((max_len,), dtype),
        B: T.Tensor((max_len,), dtype),
        valid_len: T.int32,
    ):
        with T.Kernel(1, threads=threads) as _:
            F = T.alloc_fragment((max_len,), dtype)
            span = T.min(valid_len, max_len)
            for i in T.Parallel(span):
                F[i] = A[i] - 1.0
            for i in T.Parallel(span):
                B[i] = F[i]
    return main

func = get_prim_func()
func = func.with_attr("target", tvm.target.Target("cuda"))
mod = tvm.IRModule.from_expr(func)

from tilelang.transform import LayoutInference, PipelinePlanning, Simplify, LowerTileOp

try:
    with tvm.target.Target("cuda"):
        mod = LayoutInference()(mod)
        mod = LowerTileOp()(mod)
        mod = PipelinePlanning()(mod)
        mod = Simplify()(mod)
    print("Lowered TIR for dynamic bound with fragment:")
    print(mod)
except Exception as e:
    import traceback
    traceback.print_exc()
