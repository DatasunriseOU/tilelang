import tilelang
import tilelang.testing
from tilelang import language as T
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


def test_issue_1728():
    @tilelang.jit()
    def get_qwq(hidden: int):
        num_tokens = T.dynamic("num_tokens")
        num_sms = num_tokens

        @T.prim_func
        def qwq(A: T.Tensor[(num_tokens,)]):
            with T.Kernel(num_sms) as sm_id:
                stop = sm_id + 1
                for block_idx in T.serial(sm_id, stop):
                    _pid_x, _pid_y = (block_idx, hidden)

        return qwq

    _run_optimize_pipeline(get_qwq.get_tir(1))


if __name__ == "__main__":
    tilelang.testing.main()
