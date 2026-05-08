# ruff: noqa
import tilelang
import tilelang.testing
import tilelang.language as T
from tilelang import tvm
from tilelang.engine.lower import LowerAndLegalize, OptimizeForTarget, PreLowerSemanticCheck, canon_target_host
from tilelang.env import env
from tilelang.utils.target import determine_target


def _run_optimize_pipeline(func):
    target = determine_target(env.get_default_target())
    target_host = tvm.target.Target.canon_target(canon_target_host(target, None))
    target = tvm.target.Target(target, target_host)
    mod = tvm.IRModule({func.attrs["global_symbol"]: func})
    PreLowerSemanticCheck(mod)
    with tvm.transform.PassContext(opt_level=3), target:
        mod = LowerAndLegalize(mod, target)
        OptimizeForTarget(mod, target)


def test_issue_1678():
    @tilelang.jit
    def qwq():
        @T.prim_func
        def qwq_kernel():
            with T.Kernel(4096, 1, threads=1) as (pid_y, pid_x):
                i = T.alloc_var("int32")
                i = 1
                tmp_row = T.alloc_local((4,), "float32")
                amax_local = T.alloc_var("float32")
                j = 0
                amax_local = T.max(amax_local, tmp_row[j])

        return qwq_kernel

    _run_optimize_pipeline(qwq.get_tir())


if __name__ == "__main__":
    tilelang.testing.main()
