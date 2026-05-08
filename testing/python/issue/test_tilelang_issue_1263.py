import tilelang
import tilelang.testing
import tilelang.language as T
from tilelang import tvm
from tilelang.engine.lower import LowerAndLegalize, OptimizeForTarget, PreLowerSemanticCheck, canon_target_host
from tilelang.env import env
from tilelang.utils.target import determine_target


def _test_kernel(M, N):
    dtype = "bfloat16"

    @T.prim_func
    def fwd_main(
        KV: T.Tensor((M, N), dtype),
        ids: T.Tensor((4,), "int32"),
        ids2: T.Tensor((4,), "int32"),
    ):
        with T.Kernel(4, threads=1):
            A = T.alloc_shared([N], dtype)
            B = T.alloc_shared([N], dtype)

            for i in T.Pipelined(4, num_stages=1):
                id = ids[i]
                id2 = ids2[id]
                T.copy(KV[id2, :], A)
                T.clear(B)

    return fwd_main


def _test_kernel_if_cond(M, N):
    dtype = "bfloat16"

    @T.prim_func
    def fwd_main(
        KV: T.Tensor((M, N), dtype),
        ids: T.Tensor((4,), "int32"),
        ids2: T.Tensor((4,), "int32"),
    ):
        with T.Kernel(4, threads=1):
            A = T.alloc_shared([N], dtype)
            B = T.alloc_shared([N], dtype)

            for i in T.Pipelined(4, num_stages=1):
                id = ids[i]
                id2 = ids2[id]
                if id2 > 1:
                    T.copy(KV[id2, :], A)
                    T.clear(B)

    return fwd_main


def _run_optimize_pipeline(func, pass_configs=None):
    target = determine_target(env.get_default_target())
    target_host = tvm.target.Target.canon_target(canon_target_host(target, None))
    target = tvm.target.Target(target, target_host)
    mod = tvm.IRModule({func.attrs["global_symbol"]: func})
    PreLowerSemanticCheck(mod)
    with tvm.transform.PassContext(opt_level=3, config=pass_configs or {}), target:
        mod = LowerAndLegalize(mod, target)
        OptimizeForTarget(mod, target)


def test_issue_1263_pipeline_no_consumer():
    _run_optimize_pipeline(_test_kernel(1024, 1024))
    _run_optimize_pipeline(
        _test_kernel(1024, 1024),
        pass_configs={tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True},
    )
    _run_optimize_pipeline(_test_kernel_if_cond(1024, 1024))
    _run_optimize_pipeline(
        _test_kernel_if_cond(1024, 1024),
        pass_configs={tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True},
    )


if __name__ == "__main__":
    tilelang.testing.main()
