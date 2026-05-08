import tilelang
import tilelang.language as T
import tilelang.testing
from tilelang import tvm
from tilelang.engine.lower import LowerAndLegalize, OptimizeForTarget, PreLowerSemanticCheck, canon_target_host
from tilelang.env import env
from tilelang.utils.target import determine_target


def _make_kernel(M, N):
    dtype = T.bfloat16

    @T.prim_func
    def fwd_main(KV: T.Tensor((M, N), dtype), ids: T.Tensor((4,), T.int32)):
        with T.Kernel(4, threads=1):
            A = T.alloc_shared([N], dtype)
            B = T.alloc_shared([N], dtype)

            # Regression for a bug where InjectSoftwarePipeline left the loop
            # variable as a free var, causing MakePackedAPI to fail
            for i in T.Pipelined(4, num_stages=1):
                _id = ids[i]
                T.copy(KV[_id, :], A)
                T.clear(B)

    return fwd_main


def _make_kernel_if_cond(M, N):
    dtype = T.bfloat16

    @T.prim_func
    def fwd_main(KV: T.Tensor((M, N), dtype), ids: T.Tensor((4,), T.int32)):
        with T.Kernel(4, threads=1):
            A = T.alloc_shared([N], dtype)
            B = T.alloc_shared([N], dtype)

            # Regression for a bug where InjectSoftwarePipeline left the loop
            # variable as a free var, causing MakePackedAPI to fail
            for i in T.Pipelined(4, num_stages=1):
                if i > 1:
                    _id = ids[i]
                    T.copy(KV[_id, :], A)
                    T.clear(B)

    return fwd_main


def _run_make_packed_api_pipeline(func, pass_configs):
    target = determine_target(env.get_default_target())
    target_host = tvm.target.Target.canon_target(canon_target_host(target, None))
    target = tvm.target.Target(target, target_host)
    mod = tvm.IRModule({func.attrs["global_symbol"]: func})
    PreLowerSemanticCheck(mod)
    with tvm.transform.PassContext(opt_level=3, config=pass_configs), target:
        mod = LowerAndLegalize(mod, target)
        OptimizeForTarget(mod, target)


def test_make_packed_api_no_free_loop_var():
    func, func_if_cond = _make_kernel(4, 4), _make_kernel_if_cond(4, 4)
    # Keep warp-specialization/TMA disabled to match the original repro
    cfg = {tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True}
    _run_make_packed_api_pipeline(func, cfg)
    _run_make_packed_api_pipeline(func_if_cond, cfg)


if __name__ == "__main__":
    tilelang.testing.main()
